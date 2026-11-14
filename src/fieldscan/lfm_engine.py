"""C02/C03 LFM VL/Audio client: 3-mode query router, 6-turn memory, ffmpeg PCM pipe, Piper TTS."""

from __future__ import annotations

import base64
import io
import json
import math
import re
import struct
import subprocess
import wave
from collections import deque
from pathlib import Path
from typing import Any

import httpx

from fieldscan.pdf_ingest import normalize_asr_equipment_terms, retrieve_relevant_pages

_ROOT = Path(__file__).resolve().parents[2]
_PIPER_ONNX = _ROOT / "vendor" / "piper" / "en_US-lessac-medium.onnx"

VL_CHAT_URL = "http://127.0.0.1:8091/v1/chat/completions"
AUDIO_CHAT_URL = "http://127.0.0.1:8092/v1/chat/completions"
VL_MODEL = "LFM2.5-VL-1.6B"
AUDIO_MODEL = "LFM2.5-Audio-1.5B"

SYSTEM_CONVERSATIONAL_EN = (
    "You are FieldScan AI Engineer, a friendly, concise offline industrial field assistant "
    "running locally on edge hardware for the Tzone TZ THT-02 RS-485 Temperature & Humidity Sensor.\n"
    "Reply naturally and briefly in 1 to 3 sentences in 100% English. Do NOT invent hardware faults, "
    "alarms, or red warnings when the operator is simply greeting you or chatting casually. Invite them "
    "to ask any question about TZ THT-02 operation, DIP-8 addressing, RS-485 wiring, or live camera inspection."
)

SYSTEM_TECH_QA_EN = (
    "You are FieldScan AI Engineer, an authoritative offline industrial instrumentation specialist for "
    "the Tzone TZ THT-02 Digital Temperature & Humidity Sensor (User's Manual V1.1).\n"
    "Answer the operator's technical question accurately and concisely in 2 to 4 sentences in 100% English "
    "using ONLY verified TZ THT-02 facts:\n"
    "- Dimensions: 140.0 mm total length x 16.0 mm barrel diameter; snap-on mounting bracket 42.0 x 20.0 mm "
    "(32.0 mm hole spacing, 4.2 mm holes).\n"
    "- Sensing element: Sensirion SHT30 CMOSens (-40 to +125 deg C, 0 to 100 %RH).\n"
    "- Supply & Wiring (Page 4): DC 5V to 24V (max 5 mA). Red = V+, Black = GND (common RS-485 ground "
    "required), Yellow = RS-485 A+, Green/Blue = RS-485 B-.\n"
    "- SW1 8-Pin DIP Switch (Page 5): Binary weights left-to-right [128, 64, 32, 16, 8, 4, 2, 1] "
    "(Pin 1=128 .. Pin 8=1). Slave ID 1 = Pin 8 ON, Pins 1..7 OFF. Power-cycle required after changing switches.\n"
    "- Modbus-RTU (Pages 6-7): 9600 bps, 8N1, Function 0x03. Reg 0x0000 = Temp (signed int16 x 0.1 C), "
    "Reg 0x0001 = Humidity (uint16 x 0.1 %RH), Sentinel 0x7FFF = abnormal/disconnected SHT30 probe, "
    "Reg 0x0004 = Slave ID, Reg 0x0005 = Baud (0x2580 = 9600).\n"
    "Do NOT claim a fault exists unless telemetry explicitly shows an error."
)

SYSTEM_PROMPT_EN = (
    "You are FieldScan AI Engineer inspecting a live workbench camera frame and telemetry for a "
    "Tzone TZ THT-02 RS-485 Temperature & Humidity Sensor (140mm x 16mm).\n"
    'Respond in 100% English as a compact JSON object with keys:\n'
    "{\n"
    '  "diagnosis": "2-4 sentence technical field diagnosis grounded in camera + Modbus telemetry + Manual V1.1",\n'
    '  "highlighted_zone": "strainer_head | upper_seal_ring | upper_barrel_dip | mounting_flange | '
    'lower_barrel_rs485 | waterproof_gland_leads | null",\n'
    '  "probable_causes": ["actionable cause/step 1 with Manual Page ref", "actionable cause/step 2"],\n'
    '  "citations": [5, 6]\n'
    "}\n"
    'Rules: If Modbus telemetry is HEALTHY ONLINE and no surface anomaly is present, set "highlighted_zone": null. '
    "If SW1 DIP-8 PCB is exposed, use weights [128,64,32,16,8,4,2,1] (Pin 8 ON = Slave 1, Page 5)."
)

_CONVERSATIONAL_RE = re.compile(
    r"^(hi|hello|hey|greetings|good\s+(morning|afternoon|evening)|"
    r"thank\s+you|thanks|who\s+are\s+you|can\s+you\s+hear\s+me)\b",
    re.I,
)
_TECH_KEYWORDS_RE = re.compile(
    r"\b(how\s+does|what\s+voltage|supply|voltage|current|baud|register|pinout|manual|wiring|"
    r"slave\s+\d|dip|switch|rs-?485|modbus|sht30|humidity|temperature|"
    r"0x[0-9a-f]+|crc|probe|spec|datasheet)\b",
    re.I,
)
_VISUAL_KEYWORDS_RE = re.compile(
    r"\b(inspect|look\s+at|camera|fault|broken|crack|stain|damage|"
    r"0x7fff|7fff|anomaly|red\s+stain|visual|see\s+the|photo)\b",
    re.I,
)

_VALID_ZONES = frozenset(
    {
        "strainer_head",
        "upper_seal_ring",
        "upper_barrel_dip",
        "mounting_flange",
        "lower_barrel_rs485",
        "waterproof_gland_leads",
    }
)


def _classify_query_mode(query: str, zone_hint: str | None = None) -> str:
    q = normalize_asr_equipment_terms(query or "").strip()
    if zone_hint and str(zone_hint).strip().lower() not in ("", "none", "null"):
        return "visual_diagnosis"
    if _VISUAL_KEYWORDS_RE.search(q):
        return "visual_diagnosis"
    if _TECH_KEYWORDS_RE.search(q):
        return "technical_qa"
    if _CONVERSATIONAL_RE.search(q) or (len(q) < 48 and not _TECH_KEYWORDS_RE.search(q)):
        return "conversational"
    if any(g in q.lower() for g in ("hello", "hi ", "hey", "thank")):
        return "conversational"
    return "technical_qa"


def _extract_json_dict(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    start = text.find("{")
    while start >= 0:
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    chunk = text[start : i + 1]
                    try:
                        obj = json.loads(chunk)
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def _normalize_zone(zone: Any) -> str | None:
    if zone is None:
        return None
    s = str(zone).strip().lower()
    if s in ("", "none", "null", "undefined"):
        return None
    if s in _VALID_ZONES:
        return s
    return None


def _clean_for_tts(text: str) -> str:
    out = text or ""
    out = re.sub(r"[*_#`>]", "", out)
    out = out.replace("0x7FFF", "hex 7 F F F").replace("0x7fff", "hex 7 F F F")
    out = out.replace("RS-485", "R S 485").replace("°C", " degrees Celsius")
    out = out.replace("%RH", " percent relative humidity")
    if len(out) > 460:
        out = out[:457] + "..."
    return out.strip()


def _generate_placeholder_tone_wav(duration_s: float = 0.35, sample_rate: int = 22050) -> bytes:
    n = int(sample_rate * duration_s)
    frames = bytearray()
    for i in range(n):
        t = i / sample_rate
        f0, f1 = 440.0, 660.0
        freq = f0 + (f1 - f0) * min(1.0, t / max(duration_s, 1e-6))
        amp = 0.22 * (1.0 - 0.15 * math.sin(2 * math.pi * 3 * t))
        sample = int(amp * 32767 * math.sin(2 * math.pi * freq * t))
        frames.extend(struct.pack("<h", sample))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(bytes(frames))
    return buf.getvalue()


def _normalize_to_16k_wav(audio_bytes: bytes) -> bytes:
    if not audio_bytes:
        return _generate_placeholder_tone_wav(0.1, 16000)
    if audio_bytes[:4] == b"RIFF":
        try:
            with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
                if (
                    wf.getnchannels() == 1
                    and wf.getsampwidth() == 2
                    and wf.getframerate() == 16000
                ):
                    return audio_bytes
        except wave.Error:
            pass
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            "-f",
            "wav",
            "pipe:1",
        ],
        input=audio_bytes,
        capture_output=True,
        check=False,
    )
    if proc.returncode == 0 and proc.stdout and proc.stdout[:4] == b"RIFF":
        return proc.stdout
    return _generate_placeholder_tone_wav(0.1, 16000)


def _conversational_fallback(query: str) -> dict[str, Any]:
    q = (query or "").strip().lower()
    if re.search(r"\b(thank|thanks)\b", q):
        reply = (
            "You are welcome. I am ready whenever you need TZ THT-02 wiring, DIP-8, or live camera help."
        )
    elif re.search(r"\b(who\s+are\s+you)\b", q):
        reply = (
            "I am FieldScan AI Engineer, your offline copilot for the TZ THT-02 sensor on this edge node."
        )
    elif re.search(r"\b(can\s+you\s+hear)\b", q):
        reply = "Yes, audio path is live. Ask me anything about the TZ THT-02 in English."
    else:
        reply = (
            "Hello. FieldScan is online and fully offline. Ask about DIP-8 addressing, RS-485 wiring, "
            "or point the camera for a visual inspection."
        )
    return {
        "mode": "conversational",
        "reply": reply,
        "diagnosis": reply,
        "highlighted_zone": None,
        "probable_causes": [],
        "citations": [],
        "vl_available": False,
    }


def _technical_qa_fallback(query: str, modbus_tel: dict[str, Any] | None) -> dict[str, Any]:
    pages = retrieve_relevant_pages(query, top_k=3)
    snippets = " ".join(p.get("snippet", "")[:120] for p in pages)
    tel = modbus_tel or {}
    status = ""
    if tel.get("online") and tel.get("healthy"):
        status = (
            f" Live bus reads {tel.get('temp_c')} degrees C and {tel.get('humidity_rh')} percent RH "
            f"on slave {tel.get('slave_id')}."
        )
    elif tel.get("sensor_abnormal_7fff"):
        status = " Telemetry shows register sentinel 0x7FFF (abnormal SHT30 probe)."
    reply = (
        "TZ THT-02 is a 140.0 mm by 16.0 mm RS-485 probe with Sensirion SHT30 sensing, 8-position SW1 "
        f"DIP addressing, and 9600 bps Modbus-RTU. Manual excerpts: {snippets[:220]}.{status}"
    )
    citations = [int(p["page"]) for p in pages[:3] if "page" in p]
    return {
        "mode": "technical_qa",
        "reply": reply,
        "diagnosis": reply,
        "highlighted_zone": None,
        "probable_causes": [],
        "citations": citations,
        "vl_available": False,
    }


def _visual_diagnosis_fallback(
    query: str,
    modbus_tel: dict[str, Any] | None,
    vision_tel: dict[str, Any] | None,
    zone_hint: str | None,
) -> dict[str, Any]:
    tel = modbus_tel or {}
    vis = vision_tel or {}
    healthy = bool(tel.get("healthy") and tel.get("online"))
    anomaly = bool(vis.get("anomaly_detected"))
    zone: str | None = _normalize_zone(zone_hint)
    causes: list[str] = []
    citations: list[int] = []
    if tel.get("sensor_abnormal_7fff"):
        zone = zone or "strainer_head"
        causes = [
            "SHT30 probe sentinel 0x7FFF — inspect strainer and sensor carrier (Manual Page 6).",
            "Re-seat internal probe harness after power cycle.",
        ]
        citations = [6]
        diagnosis = (
            "Modbus reports 0x7FFF on temperature or humidity registers with valid CRC. "
            "Treat the internal SHT30 element as abnormal until replaced or reseated."
        )
    elif healthy and not anomaly:
        zone = None
        diagnosis = (
            "Modbus telemetry is healthy and no optical anomaly flag is set. "
            "No hardware fault is indicated from bus data; continue monitoring."
        )
        citations = [6]
    elif vis.get("pcb_open_detected") or re.search(r"\b(dip|switch|pcb)\b", query or "", re.I):
        zone = zone or "upper_barrel_dip"
        causes = [
            "Verify SW1 DIP-8 weights [128..1]; Slave 1 = Pin 8 ON only (Manual Page 5).",
            "Cold-boot power cycle after changing DIP rockers.",
        ]
        citations = [5, 8]
        diagnosis = (
            "DIP switch PCB appears exposed. Configure slave address on SW1, then power-cycle so "
            "the MCU latches the new DIP pattern."
        )
    else:
        zone = zone or "strainer_head"
        causes = [
            "Inspect strainer mesh for blockage or mechanical damage (Manual Page 1).",
            "Confirm 12 V bench supply, common ground, and RS-485 A/B polarity (Manual Page 4).",
        ]
        citations = [1, 4]
        diagnosis = (
            "Vision language model is offline; using deterministic FieldScan rules. "
            "Review the highlighted mechanical zone and RS-485 wiring before declaring the sensor healthy."
        )
    return {
        "mode": "visual_diagnosis",
        "reply": diagnosis,
        "diagnosis": diagnosis,
        "highlighted_zone": zone,
        "probable_causes": causes,
        "citations": citations,
        "vl_available": False,
    }


class LFMEngine:
    """Offline LFM3-VL + LFM-Audio router with Piper TTS."""

    def __init__(
        self,
        vl_url: str = VL_CHAT_URL,
        audio_url: str = AUDIO_CHAT_URL,
    ) -> None:
        self.vl_url = vl_url
        self.audio_url = audio_url
        self._chat_history: deque[dict[str, str]] = deque(maxlen=6)
        self._http = httpx.Client(trust_env=False, timeout=httpx.Timeout(90.0))
        self._piper_voice: Any = None
        self._piper_tried = False

    def close(self) -> None:
        self._http.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _classify_query_mode(self, query: str, zone_hint: str | None = None) -> str:
        """Instance wrapper around the module-level 3-mode intent classifier."""
        return _classify_query_mode(query, zone_hint)

    def _vl_reachable(self) -> bool:
        try:
            base = self.vl_url.split("/v1/")[0]
            r = self._http.get(f"{base}/health", timeout=0.35)
            return r.status_code < 500
        except Exception:
            try:
                r = self._http.get(self.vl_url.replace("/v1/chat/completions", "/health"), timeout=0.35)
                return r.status_code < 500
            except Exception:
                return False

    def _load_piper(self) -> Any:
        if self._piper_tried:
            return self._piper_voice
        self._piper_tried = True
        if not _PIPER_ONNX.exists():
            return None
        try:
            from piper import PiperVoice  # type: ignore

            self._piper_voice = PiperVoice.load(str(_PIPER_ONNX))
        except Exception:
            self._piper_voice = None
        return self._piper_voice

    def synthesize_wav(self, text: str) -> bytes:
        clean = _clean_for_tts(text)
        if not clean:
            return _generate_placeholder_tone_wav()
        voice = self._load_piper()
        if voice is not None:
            try:
                buf = io.BytesIO()
                synthesize = getattr(voice, "synthesize_wav", None)
                if callable(synthesize):
                    synthesize(clean, buf)
                    data = buf.getvalue()
                    if data[:4] == b"RIFF":
                        return data
                chunks = list(voice.synthesize(clean))
                if chunks:
                    raw_pcm = b"".join(getattr(c, "audio_int16_bytes", b"") for c in chunks)
                    if raw_pcm:
                        out = io.BytesIO()
                        with wave.open(out, "wb") as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(22050)
                            wf.writeframes(raw_pcm)
                        return out.getvalue()
            except Exception:
                pass
        return _generate_placeholder_tone_wav()

    def synthesize_b64(self, text: str) -> str:
        return base64.b64encode(self.synthesize_wav(text)).decode("ascii")

    def transcribe_audio(self, audio_bytes: bytes) -> str:
        wav = _normalize_to_16k_wav(audio_bytes)
        payload = {
            "model": AUDIO_MODEL,
            "messages": [
                {"role": "system", "content": "Perform ASR."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": base64.b64encode(wav).decode("ascii"),
                                "format": "wav",
                            },
                        }
                    ],
                },
            ],
            "temperature": 0.0,
            "max_tokens": 96,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        try:
            r = self._http.post(self.audio_url, json=payload, timeout=45.0)
            if r.status_code == 200:
                data = r.json()
                content = (
                    data.get("choices", [{}])[0].get("message", {}).get("content") or ""
                ).strip()
                return normalize_asr_equipment_terms(content)
        except Exception:
            pass
        return ""

    def _chat_completion(
        self,
        system_prompt: str,
        user_content: list[dict[str, Any]] | str,
        max_tokens: int,
        attach_meta: bool = False,
    ) -> str:
        if isinstance(user_content, str):
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
        for turn in self._chat_history:
            messages.insert(-1, turn)
        payload: dict[str, Any] = {
            "model": VL_MODEL,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        r = self._http.post(self.vl_url, json=payload, timeout=90.0)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        if attach_meta:
            self._chat_history.append({"role": "user", "content": str(user_content)[:400]})
            self._chat_history.append({"role": "assistant", "content": text[:600]})
        return str(text).strip()

    def diagnose(
        self,
        query: str,
        image_jpeg: bytes | None = None,
        zone_hint: str | None = None,
        modbus_tel: dict[str, Any] | None = None,
        vision_tel: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        q = normalize_asr_equipment_terms(query or "")
        mode = _classify_query_mode(q, zone_hint)
        tel = modbus_tel or {}
        vis = vision_tel or {}
        vl_ok = self._vl_reachable()

        if mode == "conversational":
            if not vl_ok:
                result = _conversational_fallback(q)
            else:
                try:
                    reply = self._chat_completion(SYSTEM_CONVERSATIONAL_EN, q, 140, attach_meta=True)
                    result = {
                        "mode": mode,
                        "reply": reply,
                        "diagnosis": reply,
                        "highlighted_zone": None,
                        "probable_causes": [],
                        "citations": [],
                        "vl_available": True,
                    }
                except Exception:
                    result = _conversational_fallback(q)
            result["audio_wav_b64"] = self.synthesize_b64(result["reply"])
            return result

        if mode == "technical_qa":
            pages = retrieve_relevant_pages(q, top_k=3)
            ctx = "\n".join(
                f"Manual Page {p['page']}: {p.get('snippet', '')[:240]}" for p in pages
            )
            if tel:
                ctx += (
                    f"\nModbus: online={tel.get('online')} healthy={tel.get('healthy')} "
                    f"temp_c={tel.get('temp_c')} humidity_rh={tel.get('humidity_rh')} "
                    f"slave_id={tel.get('slave_id')} abnormal_7fff={tel.get('sensor_abnormal_7fff')}"
                )
            if not vl_ok:
                result = _technical_qa_fallback(q, tel)
            else:
                try:
                    reply = self._chat_completion(
                        SYSTEM_TECH_QA_EN,
                        f"Operator question:\n{q}\n\nContext:\n{ctx}",
                        200,
                        attach_meta=True,
                    )
                    citations = [int(p["page"]) for p in pages[:3]]
                    result = {
                        "mode": mode,
                        "reply": reply,
                        "diagnosis": reply,
                        "highlighted_zone": None,
                        "probable_causes": [],
                        "citations": citations,
                        "vl_available": True,
                    }
                except Exception:
                    result = _technical_qa_fallback(q, tel)
            result["audio_wav_b64"] = self.synthesize_b64(result["reply"])
            return result

        # visual_diagnosis
        healthy_online = bool(tel.get("online") and tel.get("healthy"))
        anomaly = bool(vis.get("anomaly_detected"))
        if healthy_online and not anomaly and not zone_hint:
            # Server may override zone; still allow explicit hints
            pass

        if not vl_ok or image_jpeg is None:
            result = _visual_diagnosis_fallback(q, tel, vis, zone_hint)
            if healthy_online and not anomaly and not tel.get("sensor_abnormal_7fff"):
                result["highlighted_zone"] = None
            result["audio_wav_b64"] = self.synthesize_b64(result["diagnosis"])
            return result

        b64_img = base64.b64encode(image_jpeg).decode("ascii")
        user_parts: list[dict[str, Any]] = [
            {"type": "text", "text": f"Operator query: {q}\nTelemetry: {json.dumps(tel)[:800]}"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
            },
        ]
        try:
            raw = self._chat_completion(SYSTEM_PROMPT_EN, user_parts, 320, attach_meta=True)
            parsed = _extract_json_dict(raw) or {}
            zone = _normalize_zone(parsed.get("highlighted_zone"))
            if healthy_online and not anomaly:
                zone = None
            causes = parsed.get("probable_causes") or []
            if not isinstance(causes, list):
                causes = [str(causes)]
            cites = parsed.get("citations") or []
            if not isinstance(cites, list):
                cites = []
            diagnosis = str(parsed.get("diagnosis") or raw).strip()
            result = {
                "mode": mode,
                "reply": diagnosis,
                "diagnosis": diagnosis,
                "highlighted_zone": zone,
                "probable_causes": [str(c) for c in causes],
                "citations": [int(c) for c in cites if str(c).isdigit()],
                "vl_available": True,
                "raw_vl_text": raw,
            }
        except Exception:
            result = _visual_diagnosis_fallback(q, tel, vis, zone_hint)
            if healthy_online and not anomaly and not tel.get("sensor_abnormal_7fff"):
                result["highlighted_zone"] = None
        result["audio_wav_b64"] = self.synthesize_b64(result["diagnosis"])
        return result


__all__ = [
    "LFMEngine",
    "SYSTEM_CONVERSATIONAL_EN",
    "SYSTEM_TECH_QA_EN",
    "SYSTEM_PROMPT_EN",
    "_classify_query_mode",
    "_normalize_to_16k_wav",
    "_generate_placeholder_tone_wav",
]