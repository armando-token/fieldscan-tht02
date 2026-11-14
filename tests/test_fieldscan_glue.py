"""12-gate offline pytest verification suite for FieldScan TZ THT-02 (T01..T12)."""

from __future__ import annotations

import io
import re
import socket
import struct
import wave
from email import policy
from email.parser import BytesParser
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "static"


@pytest.fixture(scope="module")
def app_client():
    from fastapi.testclient import TestClient
    from fieldscan.server import app

    with TestClient(app) as client:
        yield client


def test_t01_offline_socket_guard_blocks_external_and_allows_loopback():
    from fieldscan.offline_guard import OfflineViolationError, install_offline_socket_guard, verify_offline_enforcement

    install_offline_socket_guard()
    proof = verify_offline_enforcement()
    assert proof["blocked_external_probe"] is True
    assert proof["offline_verified"] is True

    with pytest.raises(OfflineViolationError):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.settimeout(0.3)
            s.connect(("8.8.8.8", 53))
        finally:
            s.close()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    port = srv.getsockname()[1]
    srv.listen(1)
    try:
        c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        c.settimeout(1.0)
        c.connect(("127.0.0.1", port))
        c.close()
    finally:
        srv.close()


def test_t02_pdf_manual_ingestion_and_asr_term_normalization():
    from fieldscan.pdf_ingest import ingest_manual, normalize_asr_equipment_terms, retrieve_relevant_pages

    manual = ingest_manual()
    assert manual["page_count"] == 9
    pages_dir = ROOT / "assets" / "manual_pages"
    for i in range(1, 10):
        p = pages_dir / f"page_{i}.png"
        assert p.exists(), p
        assert p.stat().st_size > 5 * 1024
    assert "TZ-THT-02" in normalize_asr_equipment_terms("THS 00 probe")
    hits = retrieve_relevant_pages("configure DIP switch slave address", manual)
    assert hits and hits[0]["page"] == 5


def test_t03_dip_switch_msb_to_lsb_math_and_pin_descriptions():
    from fieldscan.modbus_rtu import address_to_dip_bits, describe_dip_pins, dip_bits_to_address

    assert address_to_dip_bits(1) == [0, 0, 0, 0, 0, 0, 0, 1]
    assert dip_bits_to_address([1, 0, 1, 1, 0, 0, 0, 0]) == 176
    desc = describe_dip_pins(1)
    assert desc["on_pins"] == [8]
    assert desc["dip_bits"] == [0, 0, 0, 0, 0, 0, 0, 1]


def test_t04_modbus_crc16_and_tht02_frame_decoder_including_negative_and_7fff():
    from fieldscan.modbus_rtu import (
        build_read_holding_request,
        compute_modbus_crc16,
        decode_tht02_holding_response,
    )

    req = build_read_holding_request(1)
    assert req == bytes.fromhex("010300000008440C")

    def pack(slave: int, regs: list[int]) -> bytes:
        body = bytes([slave, 0x03, 16]) + struct.pack(">8H", *regs)
        return body + struct.pack("<H", compute_modbus_crc16(body))

    healthy = pack(1, [256, 582, 0, 0, 1, 0x2580, 0, 0])  # 25.6C, 58.2%RH
    d = decode_tht02_holding_response(healthy, 1)
    assert d["healthy"] is True
    assert d["temp_c"] == 25.6
    assert d["humidity_rh"] == 58.2

    neg = pack(1, [0xFF9B, 400, 0, 0, 1, 0x2580, 0, 0])  # -10.1C
    d2 = decode_tht02_holding_response(neg, 1)
    assert d2["temp_c"] == -10.1

    bad = pack(1, [0x7FFF, 0x7FFF, 0, 0, 1, 0x2580, 0, 0])
    d3 = decode_tht02_holding_response(bad, 1)
    assert d3["sensor_abnormal_7fff"] is True
    assert d3["healthy"] is False


def test_t05_vision_scanner_tracks_cylinder_and_masks_hand_screwdriver_occlusion():
    import time

    from fieldscan.vision_scanner import VisionScanner

    vs = VisionScanner()
    try:
        vs.set_mode("sim_healthy")
        time.sleep(0.25)
        if hasattr(vs, "_refresh_once"):
            vs._refresh_once()
        tel = vs.get_telemetry()
        assert tel["detected"] is True
        assert 110 <= float(tel["estimated_length_mm"]) <= 170

        vs.set_mode("sim_hand_screwdriver")
        time.sleep(0.35)
        if hasattr(vs, "_refresh_once"):
            vs._refresh_once()
        tel2 = vs.get_telemetry()
        assert tel2.get("hand_present") or tel2.get("tool_present")
        assert 110 <= float(tel2["estimated_length_mm"]) <= 170

        vs.set_mode("sim_open_dip", dip_bits=[1, 0, 1, 1, 0, 0, 0, 0])
        time.sleep(0.35)
        if hasattr(vs, "_refresh_once"):
            vs._refresh_once()
        tel3 = vs.get_telemetry()
        assert tel3["pcb_open_detected"] is True
        assert tel3.get("dip_bits_optical") == [1, 0, 1, 1, 0, 0, 0, 0] or tel3.get("dip_slave_optical") == 176
        crop = vs.get_dip_focus_jpeg()
        assert crop and crop[:2] == b"\xff\xd8"
    finally:
        vs.stop()


def test_t06_cad_exporter_generates_valid_r2010_dxf_binary_stl_and_offline_eml(tmp_path):
    import ezdxf

    from fieldscan.cad_exporter import build_escalation_package, export_autocad_dxf, export_parametric_stl

    dxf_path = tmp_path / "t.dxf"
    export_autocad_dxf(dxf_path, fault_zone="strainer_head")
    doc = ezdxf.readfile(dxf_path)
    assert "FAULT_CALLOUT" in doc.layers

    stl_path = tmp_path / "t.stl"
    export_parametric_stl(stl_path)
    raw = stl_path.read_bytes()
    tri_count = struct.unpack_from("<I", raw, 80)[0]
    assert tri_count > 500
    assert len(raw) == 84 + 50 * tri_count

    pkg = build_escalation_package(
        fault_zone="strainer_head",
        diagnosis_text="Crimson stain on strainer head.",
        operator_notes="Bench unit test",
        exports_dir=ROOT / "exports",
        outbox_dir=ROOT / "outbox",
    )
    eml_path = Path(pkg.get("eml_path") or pkg.get("eml") or next(iter((ROOT / "outbox").glob("*.eml"))))
    msg = BytesParser(policy=policy.default).parsebytes(eml_path.read_bytes())
    assert msg.get("X-FieldScan-Outbox-Mode") == "STORE_AND_FORWARD_OFFLINE"


def test_t07_lfm_query_classifier_keeps_greetings_out_of_fault_mode():
    from fieldscan.lfm_engine import LFMEngine

    eng = LFMEngine()
    mode = eng._classify_query_mode("Hello! Can you hear me?", None)
    assert mode == "conversational"
    r = eng.diagnose("Hello! Can you hear me?")
    assert r.get("mode") == "conversational" or r.get("highlighted_zone") in (None, "null")
    assert r.get("highlighted_zone") in (None, "null", "")

    assert eng._classify_query_mode("What supply voltage does the sensor need?", None) == "technical_qa"
    assert eng._classify_query_mode("Look at the camera and inspect for a red stain fault", None) == "visual_diagnosis"


def test_t08_live_repair_4step_closed_loop_and_proactive_scada_transition(tmp_path):
    from fieldscan.live_repair import LiveRepairCoordinator

    lr = LiveRepairCoordinator()
    vision = {"pcb_open_detected": False, "anomaly_detected": False}
    modbus = {"healthy": False, "online": False, "sensor_abnormal_7fff": False, "temp_c": None, "humidity_rh": None}

    lr.handle_operator_intent("open fault inspect")
    state = lr.get_state(vision, modbus)
    assert "STEP1" in str(state.get("stage", ""))
    assert state.get("andon_state") == "RED"

    lr.handle_operator_intent("slave 1 configure dip")
    state = lr.get_state(vision, modbus)
    assert "STEP2" in str(state.get("stage", ""))

    modbus_online = {
        "healthy": True,
        "online": True,
        "sensor_abnormal_7fff": False,
        "temp_c": 24.6,
        "humidity_rh": 56.2,
    }
    evt = lr.check_reactive_transitions(vision, modbus_online)
    assert evt.get("event") == "MODBUS_VALIDATED_ONLINE"
    state = lr.get_state(vision, modbus_online)
    assert state.get("andon_state") == "GREEN"
    assert "STEP3" in str(state.get("stage", ""))

    modbus_lost = {"healthy": False, "online": False, "sensor_abnormal_7fff": False, "temp_c": None, "humidity_rh": None}
    evt2 = lr.check_reactive_transitions(vision, modbus_lost)
    assert evt2.get("event") == "MODBUS_SIGNAL_LOST"
    state = lr.get_state(vision, modbus_lost)
    assert state.get("andon_state") == "RED"
    assert state.get("explode_pcb") is True

    dossier = lr.generate_signoff_dossier(operator_notes="unit test", cad_views={})
    assert dossier
    zip_path = Path((dossier.get("paths") or {}).get("zip") or dossier.get("zip_path") or "")
    assert zip_path.exists() and zip_path.suffix == ".zip"


def test_t09_fastapi_endpoints_and_zero_purple_ui_compliance(app_client):
    c = app_client
    assert c.get("/").status_code == 200
    st = c.get("/api/status")
    assert st.status_code == 200
    body = st.json()
    assert body.get("status") == "ok" or body.get("sensor_id") == "TZ-THT-02"
    assert c.get("/api/spec").status_code == 200
    assert c.get("/api/manual").status_code == 200
    assert c.get("/api/video/telemetry").status_code == 200
    assert c.post("/api/modbus/mode", json={"mode": "sim_healthy", "slave_id": 1}).status_code == 200
    diag = c.post("/api/diagnose", json={"query": "Hello!", "include_camera": False})
    assert diag.status_code == 200
    assert c.post("/api/escalate", json={"fault_zone": "strainer_head", "diagnosis_text": "test"}).status_code == 200
    assert c.get("/api/outbox").status_code == 200

    forbidden = ("#8b5cf6", "#a855f7", "#6366f1", "purple", "violet")
    for name in ("index.html", "styles.css", "app.js"):
        text = (STATIC / name).read_text(encoding="utf-8").lower()
        for tok in forbidden:
            assert tok not in text, f"{name} contains forbidden token {tok}"


def test_t10_strict_100_percent_english_codebase_and_ui_audit():
    diacritics = re.compile(r"[\u00e1\u00e9\u00ed\u00f3\u00fa\u00f1\u00bf\u00a1]")
    # Build banned tokens without embedding non-English literals in this file.
    banned = re.compile(
        r"\b("
        + "|".join(
            [
                "configuraci" + chr(0xF3) + "n",
                "reparaci" + chr(0xF3) + "n",
                "sensor" + " " + "de",
                "hola" + " " + "mundo",
                "gra" + "cias",
            ]
        )
        + r")\b",
        re.I,
    )
    exts = {".py", ".html", ".css", ".js", ".json", ".sh"}
    skip_dirs = {".venv", ".git", "vendor", "models", "bin", "__pycache__", "node_modules"}
    offenders: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in exts:
            continue
        if any(part in skip_dirs for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        if diacritics.search(text) or banned.search(text):
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, f"Non-English content in: {offenders}"


def test_t11_andon_modbus_coil_command_frame_builder():
    from fieldscan.modbus_rtu import build_write_single_coil_request, compute_modbus_crc16

    frame = build_write_single_coil_request(2, 2, True)  # GREEN coil 0x0002 ON
    # Correct Modbus CRC-16 for 02 05 00 02 FF 00 is 0xC92D (bytes 2D C9).
    # Mega-plan typo listed 2D F9; algorithm is validated by T04 known vector 44 0C.
    assert frame == bytes.fromhex("02050002FF002DC9")
    assert compute_modbus_crc16(frame[:6]) == 0xC92D


def test_t12_voice_endpoint_filters_empty_ambient_noise_for_handsfree_vad(app_client, monkeypatch):
    # 16 kHz silent WAV
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 1600)
    wav_bytes = buf.getvalue()

    import fieldscan.server as server_mod

    class FakeLFM:
        def transcribe_audio(self, *_a, **_k):
            return ""

        def diagnose(self, *_a, **_k):
            return {"diagnosis": "should not run", "highlighted_zone": None}

        def synthesize_b64(self, text: str):
            return ""

    monkeypatch.setattr(server_mod, "lfm_engine", FakeLFM())

    resp = app_client.post(
        "/api/voice",
        files={"audio": ("silence.wav", wav_bytes, "audio/wav")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("empty_speech") is True
    assert data.get("transcript", "") == ""
