"""C07 4-stage repair state machine, reactive edge watchdog, sign-off dossier builder."""

from __future__ import annotations

import base64
import io
import json
import re
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from fieldscan.cad_exporter import build_escalation_package, export_autocad_dxf, export_parametric_stl
from fieldscan.modbus_rtu import DIP_WEIGHTS, address_to_dip_bits, describe_dip_pins

_ROOT = Path(__file__).resolve().parents[2]
_EXPORTS = _ROOT / "exports"
_OUTBOX = _ROOT / "outbox"

REPAIR_STAGES = (
    "IDLE_MONITORING",
    "STEP1_OPEN_FAULT_DETECTED",
    "STEP2_GUIDING_DIP_MODBUS",
    "STEP3_SCADA_ONLINE_VERIFIED",
    "STEP4_SIGNOFF_PACKAGE_READY",
)

_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


class LiveRepairCoordinator:
    """Closed-loop repair orchestrator with 350 ms edge watchdog."""

    def __init__(self) -> None:
        self.stage = "IDLE_MONITORING"
        self.andon_state = "GREEN"
        self.highlighted_zone: str | None = None
        self.manual_page: int | None = None
        self.explode_pcb = False
        self.focus_dip = False
        self.target_slave_id = 1
        self.target_dip_bits: list[int] = address_to_dip_bits(1)
        self.operator_notes = ""
        self.proactive_event: str | None = None
        self.voice_prompt: str | None = None
        self.snapshots: dict[str, str] = {}
        self._last_modbus_healthy = False
        self._last_modbus_7fff = False
        self._last_pcb_open = False
        self._pcb_cooldown_until = 0.0
        self._last_transition_ts = 0.0
        self.signoff_package: dict[str, Any] | None = None

    def reset(self) -> None:
        self.__init__()

    def _parse_requested_modbus_target(self, text: str) -> dict[str, Any] | None:
        q = (text or "").strip().lower()
        if not q:
            return None
        result: dict[str, Any] = {}

        m_code = re.search(r"\bcode\s+(\d{1,3})\b", q)
        if m_code:
            addr = int(m_code.group(1))
            result["slave_id"] = addr
            result["dip_bits"] = address_to_dip_bits(addr)
            return {**describe_dip_pins(addr), **result}

        m_slave = re.search(r"\bslave\s+(\d{1,3})\b", q)
        if not m_slave:
            m_slave = re.search(r"\baddress\s+(" + "|".join(_NUMBER_WORDS) + r"|\d{1,3})\b", q)
        if m_slave:
            token = m_slave.group(1)
            addr = int(token) if token.isdigit() else _NUMBER_WORDS.get(token, 1)
            result["slave_id"] = addr
            result["dip_bits"] = address_to_dip_bits(addr)
            return {**describe_dip_pins(addr), **result}

        pin_hits: list[str] = []
        m_pins = re.search(r"\bpins?\s+([\d,\sand]+)", q)
        if m_pins:
            pin_hits = re.findall(r"\d+", m_pins.group(1))
        if not pin_hits:
            pin_hits = re.findall(r"\bpin\s*(\d+)\b", q)
        if pin_hits:
            bits = [0] * 8
            for p in pin_hits:
                idx = int(p) - 1
                if 0 <= idx < 8:
                    bits[idx] = 1
            addr = sum(w for b, w in zip(bits, DIP_WEIGHTS) if b)
            if addr < 1:
                addr = 1
            result["slave_id"] = addr
            result["dip_bits"] = bits
            return {**describe_dip_pins(addr), **result}
        return None

    def handle_operator_intent(self, text: str) -> dict[str, Any]:
        parsed = self._parse_requested_modbus_target(text)
        reply: dict[str, Any] = {"handled": False, "stage": self.stage}
        q = (text or "").lower()

        if parsed:
            self.target_slave_id = int(parsed["slave_id"])
            self.target_dip_bits = list(parsed["dip_bits"])
            if self.stage == "IDLE_MONITORING":
                self.stage = "STEP2_GUIDING_DIP_MODBUS"
            elif self.stage == "STEP1_OPEN_FAULT_DETECTED":
                self.stage = "STEP2_GUIDING_DIP_MODBUS"
            self.andon_state = "YELLOW"
            self.focus_dip = True
            self.explode_pcb = True
            self.highlighted_zone = "upper_barrel_dip"
            self.manual_page = 5
            on_pins = parsed.get("on_pins", [])
            off_pins = parsed.get("off_pins", [])
            reply = {
                "handled": True,
                "stage": self.stage,
                "slave_id": self.target_slave_id,
                "dip_bits": self.target_dip_bits,
                "on_pins": on_pins,
                "off_pins": off_pins,
                "message": (
                    f"Set SW1 to slave {self.target_slave_id}: pins ON {on_pins}, OFF {off_pins}. "
                    "Power-cycle DC V+ so the MCU latches the DIP pattern."
                ),
            }
            self.voice_prompt = reply["message"]
            return reply

        if any(k in q for k in ("sign off", "signoff", "complete repair", "finish")):
            self.stage = "STEP4_SIGNOFF_PACKAGE_READY"
            self.andon_state = "GREEN"
            reply = {
                "handled": True,
                "stage": self.stage,
                "message": "Sign-off stage ready. Generate the manufacturer dossier when SCADA is verified.",
            }
            self.voice_prompt = reply["message"]
            return reply

        if any(k in q for k in ("connect modbus", "verify scada", "check bus")):
            self.stage = "STEP2_GUIDING_DIP_MODBUS"
            self.andon_state = "YELLOW"
            reply = {
                "handled": True,
                "stage": self.stage,
                "message": "Confirm DIP-8 addressing, common RS-485 ground, then poll holding registers 0x0000-0x0007.",
            }
            self.voice_prompt = reply["message"]
            return reply

        if any(k in q for k in ("open", "inspect", "fault", "no signal")):
            self.stage = "STEP1_OPEN_FAULT_DETECTED"
            self.andon_state = "RED"
            self.explode_pcb = True
            self.focus_dip = True
            self.highlighted_zone = "upper_barrel_dip"
            self.manual_page = 5
            reply = {
                "handled": True,
                "stage": self.stage,
                "message": "Open-fault workflow started. Expose SW1 PCB and tell me the target slave address.",
            }
            self.voice_prompt = reply["message"]
            return reply

        return reply

    def _record_snapshot(self, label: str, vision_tel: dict[str, Any] | None) -> None:
        vis = vision_tel or {}
        b64 = vis.get("last_frame_jpeg_b64") or vis.get("snapshot_b64")
        if b64:
            self.snapshots[label] = str(b64)

    def check_reactive_transitions(
        self,
        vision_tel: dict[str, Any] | None = None,
        modbus_tel: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        vis = vision_tel or {}
        tel = modbus_tel or {}
        event: str | None = None
        now = time.time()

        healthy = bool(tel.get("healthy"))
        online = bool(tel.get("online"))
        seven = bool(tel.get("sensor_abnormal_7fff"))
        pcb_open = bool(vis.get("pcb_open_detected"))

        if healthy and not self._last_modbus_healthy:
            event = "MODBUS_VALIDATED_ONLINE"
            self.stage = "STEP3_SCADA_ONLINE_VERIFIED"
            self.andon_state = "GREEN"
            self.explode_pcb = False
            self.focus_dip = False
            self.highlighted_zone = None
            self.manual_page = 6
            temp = tel.get("temp_c")
            hum = tel.get("humidity_rh")
            self.voice_prompt = (
                f"Modbus validated online. Live reading {temp} degrees Celsius and {hum} percent relative humidity."
            )
            self._record_snapshot("after_online", vis)

        elif seven and not self._last_modbus_7fff:
            event = "MODBUS_7FFF_ALARM"
            self.andon_state = "RED"
            self.highlighted_zone = "strainer_head"
            self.manual_page = 6
            self.voice_prompt = (
                "Alert: Modbus register sentinel 0x7FFF indicates an abnormal or disconnected SHT30 probe."
            )

        elif self._last_modbus_healthy and not healthy and online is False:
            event = "MODBUS_SIGNAL_LOST"
            self.stage = "STEP1_OPEN_FAULT_DETECTED"
            self.andon_state = "RED"
            self.explode_pcb = True
            self.focus_dip = True
            self.highlighted_zone = "upper_barrel_dip"
            self.manual_page = 5
            self.voice_prompt = (
                "Modbus signal lost after verified online state. Open the upper barrel, verify DIP-8 and RS-485 wiring."
            )
            self._record_snapshot("before_fault", vis)

        elif pcb_open and not self._last_pcb_open and not healthy:
            if now >= self._pcb_cooldown_until:
                event = "CAMERA_OPEN_PCB_DETECTED"
                self.stage = "STEP1_OPEN_FAULT_DETECTED"
                self.explode_pcb = True
                self.focus_dip = True
                self.andon_state = "RED"
                self.highlighted_zone = "upper_barrel_dip"
                self.manual_page = 5
                self._pcb_cooldown_until = now + 12.0
                self.voice_prompt = (
                    "DIP PCB detected on camera. Tell me the target Modbus slave ID, for example slave 1 or code 176."
                )
                self._record_snapshot("before_fault", vis)

        self._last_modbus_healthy = healthy
        self._last_modbus_7fff = seven
        self._last_pcb_open = pcb_open
        if event:
            self.proactive_event = event
            self._last_transition_ts = now
        return {"event": event, "stage": self.stage, "andon_state": self.andon_state}

    def get_state(
        self,
        vision_tel: dict[str, Any] | None = None,
        modbus_tel: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "andon_state": self.andon_state,
            "highlighted_zone": self.highlighted_zone,
            "manual_page": self.manual_page,
            "explode_pcb": self.explode_pcb,
            "focus_dip": self.focus_dip,
            "target_slave_id": self.target_slave_id,
            "target_dip_bits": self.target_dip_bits,
            "target_dip_pins": describe_dip_pins(self.target_slave_id),
            "proactive_event": self.proactive_event,
            "voice_prompt": self.voice_prompt,
            "snapshots": dict(self.snapshots),
            "signoff_package": self.signoff_package,
            "vision": {
                "pcb_open_detected": bool((vision_tel or {}).get("pcb_open_detected")),
                "anomaly_detected": bool((vision_tel or {}).get("anomaly_detected")),
            },
            "modbus": {
                "online": bool((modbus_tel or {}).get("online")),
                "healthy": bool((modbus_tel or {}).get("healthy")),
                "sensor_abnormal_7fff": bool((modbus_tel or {}).get("sensor_abnormal_7fff")),
            },
        }

    def _decode_data_url(self, data_url: str | None) -> Image.Image | None:
        if not data_url:
            return None
        try:
            raw = base64.b64decode(data_url.split(",", 1)[-1])
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            return None

    def _render_headless_quad(self) -> Image.Image:
        canvas = Image.new("RGB", (1280, 800), (10, 14, 20))
        draw = ImageDraw.Draw(canvas)
        quadrants = [
            (0, 0, 640, 400, "Isometric"),
            (640, 0, 1280, 400, "Front Dim"),
            (0, 400, 640, 800, "Exploded DIP"),
            (640, 400, 1280, 800, "Side Gland"),
        ]
        for x0, y0, x1, y1, title in quadrants:
            draw.rectangle((x0 + 8, y0 + 8, x1 - 8, y1 - 8), outline=(56, 189, 248))
            draw.text((x0 + 20, y0 + 20), title, fill=(56, 189, 248))
            draw.line((x0 + 40, y1 - 60, x1 - 40, y1 - 60), fill=(100, 120, 140))
        draw.text((24, 760), "TZ THT-02 FieldScan headless CAD quad", fill=(180, 200, 220))
        return canvas

    def _render_strip(self, cad_views: dict[str, Any] | None) -> Image.Image:
        strip = Image.new("RGB", (1260, 380), (12, 17, 24))
        draw = ImageDraw.Draw(strip)
        labels = ["before_fault", "during_repair", "after_online"]
        for i, label in enumerate(labels):
            x0 = i * 420
            draw.rectangle((x0 + 6, 6, x0 + 414, 374), outline=(51, 65, 85))
            draw.text((x0 + 16, 16), label.replace("_", " ").title(), fill=(56, 189, 248))
            snap = self.snapshots.get(label) or (cad_views or {}).get(label)
            img = self._decode_data_url(snap) if isinstance(snap, str) else None
            if img:
                thumb = img.resize((390, 300))
                strip.paste(thumb, (x0 + 12, 50))
        return strip

    def generate_signoff_dossier(
        self,
        operator_notes: str = "",
        cad_views: dict[str, Any] | None = None,
        modbus_tel: dict[str, Any] | None = None,
        fault_zone: str | None = None,
    ) -> dict[str, Any]:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        package_id = f"SIGNOFF-THT02-{stamp}"
        work = _EXPORTS / package_id
        work.mkdir(parents=True, exist_ok=True)

        zone = fault_zone or self.highlighted_zone
        dxf = export_autocad_dxf(work / f"{package_id}.dxf", fault_zone=zone)
        stl = export_parametric_stl(work / f"{package_id}.stl")

        quad_keys = ("isometric", "front_dim", "exploded_dip", "side_gland")
        quad = Image.new("RGB", (1280, 800), (10, 14, 20))
        views = cad_views or {}
        for i, key in enumerate(quad_keys):
            x0 = (i % 2) * 640
            y0 = (i // 2) * 400
            tile = self._decode_data_url(views.get(key))
            if tile is None:
                tile = self._render_headless_quad().crop((x0, y0, x0 + 640, y0 + 400))
            tile = tile.resize((640, 400))
            quad.paste(tile, (x0, y0))
        quad_path = work / f"{package_id}_3d_4angles.png"
        quad.save(quad_path)

        strip = self._render_strip(views)
        strip_path = work / f"{package_id}_before_during_after.png"
        strip.save(strip_path)

        md_path = work / f"{package_id}_signoff_report.md"
        md_path.write_text(
            "\n".join(
                [
                    "# TZ THT-02 Manufacturer Sign-Off Dossier",
                    "",
                    f"- Package: `{package_id}`",
                    f"- Stage: `{self.stage}`",
                    f"- Target slave: `{self.target_slave_id}`",
                    f"- Operator notes: {operator_notes or self.operator_notes or '(none)'}",
                    "",
                    "## Modbus snapshot",
                    "",
                    f"```json\n{json.dumps(modbus_tel or {}, indent=2)}\n```",
                ]
            ),
            encoding="utf-8",
        )

        zip_path = _EXPORTS / f"{package_id}_bundle.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in (dxf, stl, quad_path, strip_path, md_path):
                zf.write(p, arcname=p.name)

        esc = build_escalation_package(
            _EXPORTS,
            _OUTBOX,
            fault_zone=zone,
            diagnosis_text=f"Sign-off dossier for stage {self.stage}",
            operator_notes=operator_notes or self.operator_notes,
            modbus_tel=modbus_tel,
        )

        self.stage = "STEP4_SIGNOFF_PACKAGE_READY"
        self.signoff_package = {
            "package_id": package_id,
            "paths": {
                "dxf": str(dxf),
                "stl": str(stl),
                "quad_png": str(quad_path),
                "strip_png": str(strip_path),
                "md": str(md_path),
                "zip": str(zip_path),
                "eml": esc["paths"]["eml"],
            },
        }
        return self.signoff_package


_coordinator = LiveRepairCoordinator()


def get_coordinator() -> LiveRepairCoordinator:
    return _coordinator


def check_reactive_transitions(
    vision_tel: dict[str, Any] | None = None,
    modbus_tel: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _coordinator.check_reactive_transitions(vision_tel, modbus_tel)


def handle_operator_intent(text: str) -> dict[str, Any]:
    return _coordinator.handle_operator_intent(text)


def get_state(
    vision_tel: dict[str, Any] | None = None,
    modbus_tel: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _coordinator.get_state(vision_tel, modbus_tel)


def generate_signoff_dossier(
    operator_notes: str = "",
    cad_views: dict[str, Any] | None = None,
    modbus_tel: dict[str, Any] | None = None,
    fault_zone: str | None = None,
) -> dict[str, Any]:
    return _coordinator.generate_signoff_dossier(
        operator_notes=operator_notes,
        cad_views=cad_views,
        modbus_tel=modbus_tel,
        fault_zone=fault_zone,
    )


def _parse_requested_modbus_target(text: str) -> dict[str, Any] | None:
    return _coordinator._parse_requested_modbus_target(text)


__all__ = [
    "REPAIR_STAGES",
    "LiveRepairCoordinator",
    "check_reactive_transitions",
    "handle_operator_intent",
    "get_state",
    "generate_signoff_dossier",
    "_parse_requested_modbus_target",
    "get_coordinator",
]