"""C09 FastAPI orchestrator: 26 REST routes, MJPEG stream, offline lifecycle glue for TZ THT-02."""

from __future__ import annotations

import email
import json
import platform
import re
import time
from contextlib import asynccontextmanager
from email import policy
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from fieldscan.cad_exporter import (
    _ZONE_ALIAS_MAP,
    build_escalation_package,
    export_autocad_dxf,
    export_parametric_stl,
    resolve_zone_id,
)
from fieldscan.lfm_engine import LFMEngine
from fieldscan.live_repair import (
    LiveRepairCoordinator,
    check_reactive_transitions,
    generate_signoff_dossier,
    get_coordinator,
    get_state,
    handle_operator_intent,
)
from fieldscan.modbus_rtu import ModbusRTUPoller
from fieldscan.offline_guard import (
    get_blocked_external_attempts,
    install_offline_socket_guard,
    verify_offline_enforcement,
)
from fieldscan.pdf_ingest import ingest_manual, normalize_asr_equipment_terms
from fieldscan.vision_scanner import VisionScanner

install_offline_socket_guard()

ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = ROOT / "static"
EXPORTS_DIR = ROOT / "exports"
OUTBOX_DIR = ROOT / "outbox"
MANUAL_PAGES_DIR = ROOT / "assets" / "manual_pages"
KNOWLEDGE_SPEC = ROOT / "knowledge" / "THT02_spec.json"

BOUNDARY = b"frame"
MJPEG_SLEEP_S = 0.036

_REPAIR_INTENT_RE = re.compile(
    r"(configure\s+dip|slave\s+\d|code\s+\d{1,3}|9600\s*bps|which\s+pins?|"
    r"open\s+and\s+no\s+signal|pins?\s+[\d,\sand]+|address\s+(one|two|three|\d))",
    re.I,
)
_DIP_QUERY_RE = re.compile(r"\b(dip|switch|pin|pcb|sw1)\b", re.I)

vision_scanner = VisionScanner()
modbus_poller = ModbusRTUPoller()
lfm_engine = LFMEngine()
live_repair: LiveRepairCoordinator = get_coordinator()

_manual_cache: dict[str, Any] | None = None
_last_andon_driven: str | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    vision_scanner.start()
    modbus_poller.start()
    try:
        yield
    finally:
        vision_scanner.stop()
        modbus_poller.stop()
        lfm_engine.close()


app = FastAPI(
    title="FieldScan TZ THT-02",
    version="1.0.0",
    lifespan=lifespan,
)

for directory, mount in (
    (STATIC_DIR, "/static"),
    (EXPORTS_DIR, "/exports"),
    (OUTBOX_DIR, "/outbox"),
    (MANUAL_PAGES_DIR, "/manual_pages"),
):
    directory.mkdir(parents=True, exist_ok=True)
    app.mount(mount, StaticFiles(directory=str(directory)), name=mount.replace("/", ""))


def _system_info() -> dict[str, Any]:
    try:
        import psutil

        vm = psutil.virtual_memory()
        return {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": psutil.cpu_count(logical=True),
            "ram_total_gb": round(vm.total / (1024**3), 2),
            "ram_available_gb": round(vm.available / (1024**3), 2),
            "ram_percent": vm.percent,
            "uptime_s": round(time.time() - psutil.boot_time(), 1),
        }
    except Exception:
        return {
            "platform": platform.platform(),
            "python": platform.python_version(),
        }


def _cached_manual() -> dict[str, Any]:
    global _manual_cache
    if _manual_cache is None:
        _manual_cache = ingest_manual()
    return _manual_cache


def _vision_tel() -> dict[str, Any]:
    return vision_scanner.get_telemetry()


def _modbus_tel() -> dict[str, Any]:
    return modbus_poller.get_telemetry()


def _drive_andon_if_changed(andon_state: str) -> None:
    global _last_andon_driven
    state = (andon_state or "RED").upper()
    if state == _last_andon_driven:
        return
    modbus_poller.drive_andon_relay(state)
    _last_andon_driven = state


def _run_reactive_edges() -> dict[str, Any]:
    vis = _vision_tel()
    tel = _modbus_tel()
    edge = check_reactive_transitions(vis, tel)
    repair_state = get_state(vis, tel)
    _drive_andon_if_changed(repair_state.get("andon_state", "RED"))
    return edge


def _detect_conversational_repair_intent(query: str, vision_tel: dict[str, Any]) -> bool:
    q = (query or "").strip()
    if not q or not _REPAIR_INTENT_RE.search(q):
        return False
    pcb_open = bool(vision_tel.get("pcb_open_detected"))
    if live_repair.stage != "IDLE_MONITORING" or pcb_open:
        return True
    return False


def _dip_query(text: str) -> bool:
    return bool(_DIP_QUERY_RE.search(text or ""))


def _apply_healthy_modbus_guard(
    result: dict[str, Any],
    modbus_tel: dict[str, Any],
    vision_tel: dict[str, Any],
) -> dict[str, Any]:
    healthy = bool(modbus_tel.get("healthy"))
    anomaly = bool(vision_tel.get("anomaly_detected"))
    seven = bool(modbus_tel.get("sensor_abnormal_7fff"))
    if healthy and not anomaly and not seven:
        result["highlighted_zone"] = None
    return result


def _select_diagnose_image(query: str, vision_tel: dict[str, Any], include_camera: bool) -> bytes | None:
    if not include_camera:
        return None
    if _dip_query(query) and vision_tel.get("pcb_open_detected") and vision_tel.get("dip_crop_available"):
        crop = vision_scanner.get_dip_focus_jpeg()
        if crop:
            return crop
    return vision_scanner.get_jpeg(raw=True)


def _run_diagnose_pipeline(
    query: str,
    *,
    include_camera: bool = True,
    zone: str | None = None,
    dip_address: int | None = None,
    dip_bits: list[int] | None = None,
) -> dict[str, Any]:
    q = normalize_asr_equipment_terms(query or "")
    vis = _vision_tel()
    tel = _modbus_tel()

    zone_hint = resolve_zone_id(zone) if zone else None
    if dip_address is not None:
        live_repair.target_slave_id = int(dip_address)
    if dip_bits is not None:
        from fieldscan.modbus_rtu import address_to_dip_bits, dip_bits_to_address

        bits = [int(bool(b)) for b in (dip_bits + [0] * 8)[:8]]
        live_repair.target_dip_bits = bits
        live_repair.target_slave_id = dip_bits_to_address(bits) or live_repair.target_slave_id
        if live_repair.target_slave_id < 1:
            live_repair.target_slave_id = 1

    if _detect_conversational_repair_intent(q, vis):
        repair_reply = handle_operator_intent(q)
        if repair_reply.get("handled"):
            message = repair_reply.get("message", "")
            return {
                "mode": "repair_intent",
                "reply": message,
                "diagnosis": message,
                "highlighted_zone": live_repair.highlighted_zone,
                "probable_causes": [],
                "citations": [],
                "repair": repair_reply,
                "audio_wav_b64": lfm_engine.synthesize_b64(message),
                "vl_available": False,
            }

    image_jpeg = _select_diagnose_image(q, vis, include_camera)
    result = lfm_engine.diagnose(
        q,
        image_jpeg=image_jpeg,
        zone_hint=zone_hint,
        modbus_tel=tel,
        vision_tel=vis,
    )
    result = _apply_healthy_modbus_guard(result, tel, vis)
    hz = result.get("highlighted_zone")
    if hz:
        resolved = resolve_zone_id(str(hz))
        if resolved:
            result["highlighted_zone"] = resolved
            live_repair.highlighted_zone = resolved
    return result


def _execute_repair_step(step: str, body: dict[str, Any]) -> dict[str, Any]:
    step = (step or "").strip().lower()
    tel = _modbus_tel()
    hw = bool(tel.get("usb_adapter_present"))
    vis = _vision_tel()

    if step == "reset":
        live_repair.reset()
        if not hw:
            vision_scanner.set_mode("sim_healthy")
            modbus_poller.set_mode("sim_offline_open")
        return {"ok": True, "step": step, "repair": get_state(_vision_tel(), _modbus_tel())}

    if step == "inspect_open":
        live_repair.stage = "STEP1_OPEN_FAULT_DETECTED"
        live_repair.andon_state = "RED"
        live_repair.explode_pcb = True
        live_repair.focus_dip = True
        live_repair.highlighted_zone = "upper_barrel_dip"
        live_repair.manual_page = 5
        if not hw:
            vision_scanner.set_mode("sim_open_dip")
            modbus_poller.set_mode("sim_offline_open")
        return {"ok": True, "step": step, "repair": get_state(_vision_tel(), _modbus_tel())}

    if step == "guide_dip":
        live_repair.stage = "STEP2_GUIDING_DIP_MODBUS"
        live_repair.andon_state = "YELLOW"
        live_repair.explode_pcb = True
        live_repair.focus_dip = True
        live_repair.highlighted_zone = "upper_barrel_dip"
        live_repair.manual_page = 5
        slave = body.get("slave_id") or live_repair.target_slave_id
        if slave is not None:
            live_repair.target_slave_id = int(slave)
        if not hw:
            vision_scanner.set_mode("sim_open_dip")
            modbus_poller.set_mode("sim_offline_open", slave_id=live_repair.target_slave_id)
        return {"ok": True, "step": step, "repair": get_state(_vision_tel(), _modbus_tel())}

    if step == "connect_modbus":
        live_repair.stage = "STEP2_GUIDING_DIP_MODBUS"
        live_repair.andon_state = "YELLOW"
        if not hw:
            modbus_poller.set_mode("sim_healthy", slave_id=live_repair.target_slave_id)
        _run_reactive_edges()
        return {"ok": True, "step": step, "repair": get_state(_vision_tel(), _modbus_tel())}

    if step == "signoff":
        live_repair.stage = "STEP4_SIGNOFF_PACKAGE_READY"
        live_repair.andon_state = "GREEN"
        return {"ok": True, "step": step, "repair": get_state(_vision_tel(), _modbus_tel())}

    return {"ok": False, "error": f"unknown step: {step}"}


async def _mjpeg_stream(raw: int = 0) -> AsyncIterator[bytes]:
    while True:
        jpeg = vision_scanner.get_jpeg(raw=bool(raw))
        if jpeg:
            header = (
                b"--" + BOUNDARY + b"\r\n"
                b"Content-Type: image/jpeg\r\n"
                + f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
            )
            yield header + jpeg + b"\r\n"
        await _async_sleep(MJPEG_SLEEP_S)


async def _async_sleep(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)


def _register_alias(path: str, endpoint, methods: list[str]) -> None:
    app.add_api_route(path, endpoint, methods=methods, include_in_schema=True)


# --- Models ---


class VideoModeBody(BaseModel):
    mode: str = "auto"
    dip_bits: list[int] | None = None


class ModbusModeBody(BaseModel):
    mode: str = "auto"
    slave_id: int | None = 1
    baud_bps: int | None = 9600
    port: str | None = None


class RepairStepBody(BaseModel):
    step: str
    slave_id: int | None = None
    operator_notes: str | None = None


class SignoffBody(BaseModel):
    operator_notes: str = ""
    cad_views: dict[str, Any] | None = None


class DiagnoseBody(BaseModel):
    query: str = ""
    include_camera: bool = True
    zone: str | None = None
    dip_address: int | None = None
    dip_bits: list[int] | None = None


class TtsBody(BaseModel):
    text: str = ""


class EscalateBody(BaseModel):
    fault_zone: str | None = None
    diagnosis_text: str = ""
    operator_notes: str = ""
    snapshot_b64: str | None = None


# --- Routes ---


@app.get("/")
async def root_index() -> FileResponse:
    index = STATIC_DIR / "index.html"
    if not index.exists():
        index.parent.mkdir(parents=True, exist_ok=True)
        index.write_text(
            "<!DOCTYPE html><html><head><title>FieldScan TZ THT-02</title></head>"
            "<body><h1>FieldScan TZ THT-02</h1><p>UI assets loading.</p></body></html>",
            encoding="utf-8",
        )
    return FileResponse(index)


async def api_status() -> dict[str, Any]:
    edge = _run_reactive_edges()
    offline = verify_offline_enforcement()
    vis = _vision_tel()
    tel = _modbus_tel()
    repair = get_state(vis, tel)
    return {
        "status": "ok",
        "sensor_id": "TZ-THT-02",
        "offline_verified": bool(offline.get("offline_verified")),
        "blocked_external_attempts": get_blocked_external_attempts(),
        "fps": vis.get("fps", 0.0),
        "vision": vis,
        "modbus": tel,
        "repair": repair,
        "system": _system_info(),
        "offline": offline,
        "reactive_edge": edge,
        "proactive_event": repair.get("proactive_event"),
    }


_register_alias("/api/status", api_status, ["GET"])
_register_alias("/api/health", api_status, ["GET"])


async def api_spec() -> dict[str, Any]:
    if KNOWLEDGE_SPEC.exists():
        return json.loads(KNOWLEDGE_SPEC.read_text(encoding="utf-8"))
    return {}


_register_alias("/api/spec", api_spec, ["GET"])
_register_alias("/api/knowledge", api_spec, ["GET"])


async def api_manual_index() -> dict[str, Any]:
    return _cached_manual()


_register_alias("/api/manual", api_manual_index, ["GET"])
_register_alias("/api/manual/index", api_manual_index, ["GET"])


@app.get("/api/manual/page/{page_n}")
async def api_manual_page(page_n: int) -> FileResponse:
    if page_n < 1 or page_n > 9:
        return JSONResponse({"error": "page out of range"}, status_code=404)  # type: ignore[return-value]
    png = MANUAL_PAGES_DIR / f"page_{page_n}.png"
    if not png.exists():
        _cached_manual()
    if not png.exists():
        return JSONResponse({"error": "page not found"}, status_code=404)  # type: ignore[return-value]
    return FileResponse(png, media_type="image/png")


async def api_video_stream(raw: int = Query(0)) -> StreamingResponse:
    return StreamingResponse(
        _mjpeg_stream(raw),
        media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY.decode()}",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


_register_alias("/api/video/stream", api_video_stream, ["GET"])
_register_alias("/api/camera/stream", api_video_stream, ["GET"])


async def api_dip_crop() -> Response:
    data = vision_scanner.get_dip_focus_jpeg()
    if not data:
        return Response(status_code=404)
    return Response(
        content=data,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


_register_alias("/api/video/dip_crop", api_dip_crop, ["GET"])
_register_alias("/api/camera/dip_crop", api_dip_crop, ["GET"])


async def api_video_telemetry() -> dict[str, Any]:
    return _vision_tel()


_register_alias("/api/video/telemetry", api_video_telemetry, ["GET"])
_register_alias("/api/scanner/telemetry", api_video_telemetry, ["GET"])


async def api_video_mode(body: VideoModeBody) -> dict[str, Any]:
    vision_scanner.set_mode(body.mode, dip_bits=body.dip_bits)
    return vision_scanner.get_telemetry()


_register_alias("/api/video/mode", api_video_mode, ["POST"])
_register_alias("/api/camera/mode", api_video_mode, ["POST"])


async def api_video_upload(file: UploadFile = File(...)) -> dict[str, Any]:
    raw = await file.read()
    vision_scanner.load_uploaded_frame(raw)
    return vision_scanner.get_telemetry()


_register_alias("/api/video/upload", api_video_upload, ["POST"])
_register_alias("/api/camera/upload", api_video_upload, ["POST"])
_register_alias("/api/scanner/upload", api_video_upload, ["POST"])


async def api_modbus_telemetry() -> dict[str, Any]:
    return _modbus_tel()


_register_alias("/api/modbus/telemetry", api_modbus_telemetry, ["GET"])
_register_alias("/api/scada/telemetry", api_modbus_telemetry, ["GET"])


@app.post("/api/modbus/mode")
async def api_modbus_mode(body: ModbusModeBody) -> dict[str, Any]:
    tel = modbus_poller.set_mode(
        body.mode,
        slave_id=body.slave_id,
        baud_bps=body.baud_bps,
        port=body.port,
    )
    _run_reactive_edges()
    return tel


@app.get("/api/repair/state")
async def api_repair_state() -> dict[str, Any]:
    vis = _vision_tel()
    tel = _modbus_tel()
    state = get_state(vis, tel)
    return {**state, "proactive_event": live_repair.proactive_event}


@app.post("/api/repair/step")
async def api_repair_step(body: RepairStepBody) -> dict[str, Any]:
    payload = body.model_dump()
    result = _execute_repair_step(body.step, payload)
    _drive_andon_if_changed(live_repair.andon_state)
    return result


@app.post("/api/repair/signoff")
async def api_repair_signoff(body: SignoffBody) -> dict[str, Any]:
    if body.operator_notes:
        live_repair.operator_notes = body.operator_notes
    package = generate_signoff_dossier(
        operator_notes=body.operator_notes,
        cad_views=body.cad_views,
        modbus_tel=_modbus_tel(),
        fault_zone=live_repair.highlighted_zone,
    )
    return package


@app.post("/api/diagnose")
async def api_diagnose(body: DiagnoseBody) -> dict[str, Any]:
    return _run_diagnose_pipeline(
        body.query,
        include_camera=body.include_camera,
        zone=body.zone,
        dip_address=body.dip_address,
        dip_bits=body.dip_bits,
    )


async def api_voice(
    audio: UploadFile | None = File(None),
    file: UploadFile | None = File(None),
    zone: str | None = Form(None),
    dip_address: int | None = Form(None),
) -> dict[str, Any]:
    upload = audio or file
    if upload is None:
        return {"empty_speech": True, "transcript": "", "error": "no audio file"}
    audio_bytes = await upload.read()
    transcript = normalize_asr_equipment_terms(lfm_engine.transcribe_audio(audio_bytes))
    alnum_only = re.sub(r"[^a-zA-Z0-9]+", "", transcript)
    if len(alnum_only) < 2:
        return {"empty_speech": True, "transcript": ""}

    result = _run_diagnose_pipeline(
        transcript,
        include_camera=True,
        zone=zone,
        dip_address=dip_address,
    )
    result["transcript"] = transcript
    result["empty_speech"] = False
    return result


_register_alias("/api/voice", api_voice, ["POST"])
_register_alias("/api/audio/ask", api_voice, ["POST"])


@app.post("/api/tts")
async def api_tts(body: TtsBody) -> dict[str, str]:
    return {"audio_wav_b64": lfm_engine.synthesize_b64(body.text)}


async def api_cad_dxf(zone: str | None = Query(None)) -> FileResponse:
    resolved = resolve_zone_id(zone) if zone else None
    suffix = resolved or "general"
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = EXPORTS_DIR / f"TZ_THT02_{suffix}.dxf"
    export_autocad_dxf(path, fault_zone=resolved)
    return FileResponse(path, media_type="application/dxf", filename=path.name)


_register_alias("/api/cad/dxf", api_cad_dxf, ["GET"])
_register_alias("/api/export/dxf", api_cad_dxf, ["GET"])


async def api_cad_stl() -> FileResponse:
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = EXPORTS_DIR / "TZ_THT02_140x16mm.stl"
    export_parametric_stl(path)
    return FileResponse(path, media_type="model/stl", filename=path.name)


_register_alias("/api/cad/stl", api_cad_stl, ["GET"])
_register_alias("/api/export/stl", api_cad_stl, ["GET"])


async def api_escalate(body: EscalateBody) -> dict[str, Any]:
    zone = resolve_zone_id(body.fault_zone)
    return build_escalation_package(
        EXPORTS_DIR,
        OUTBOX_DIR,
        fault_zone=zone,
        diagnosis_text=body.diagnosis_text,
        operator_notes=body.operator_notes,
        snapshot_b64=body.snapshot_b64,
        modbus_tel=_modbus_tel(),
    )


_register_alias("/api/escalate", api_escalate, ["POST"])
_register_alias("/api/escalation", api_escalate, ["POST"])


def _parse_outbox_eml(eml_path: Path) -> dict[str, Any]:
    try:
        raw = eml_path.read_bytes()
        msg = email.message_from_bytes(raw, policy=policy.default)
        subject = str(msg.get("Subject", ""))
        to_addr = str(msg.get("To", ""))
        fault = ""
        for line in (msg.get_payload(decode=False) or "").splitlines() if isinstance(msg.get_payload(), str) else []:
            if "Fault zone:" in line:
                fault = line.split(":", 1)[-1].strip()
                break
        body_text = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        body_text = payload.decode("utf-8", errors="replace")
                        break
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                body_text = payload.decode("utf-8", errors="replace")
        if not fault and "Fault zone:" in body_text:
            for line in body_text.splitlines():
                if "Fault zone:" in line:
                    fault = line.split(":", 1)[-1].strip()
                    break
        bundle_name = eml_path.stem + "_bundle.zip"
        bundle_path = EXPORTS_DIR / bundle_name
        if not bundle_path.exists():
            alt = list(EXPORTS_DIR.glob(f"{eml_path.stem}*.zip"))
            bundle_path = alt[0] if alt else bundle_path
        return {
            "id": eml_path.stem,
            "filename": eml_path.name,
            "subject": subject,
            "to": to_addr,
            "fault_zone": fault or None,
            "eml_url": f"/outbox/{eml_path.name}",
            "bundle_url": f"/exports/{bundle_path.name}" if bundle_path.exists() else None,
            "mtime": eml_path.stat().st_mtime,
        }
    except Exception:
        return {
            "id": eml_path.stem,
            "filename": eml_path.name,
            "subject": "",
            "to": "",
            "fault_zone": None,
            "eml_url": f"/outbox/{eml_path.name}",
            "bundle_url": None,
            "mtime": eml_path.stat().st_mtime,
        }


@app.get("/api/outbox")
async def api_outbox() -> dict[str, Any]:
    OUTBOX_DIR.mkdir(parents=True, exist_ok=True)
    items = [_parse_outbox_eml(p) for p in OUTBOX_DIR.glob("*.eml")]
    items.sort(key=lambda x: float(x.get("mtime", 0)), reverse=True)
    for item in items:
        item.pop("mtime", None)
    return {"count": len(items), "items": items}


__all__ = [
    "app",
    "_ZONE_ALIAS_MAP",
    "ROOT",
    "vision_scanner",
    "modbus_poller",
    "lfm_engine",
    "live_repair",
]