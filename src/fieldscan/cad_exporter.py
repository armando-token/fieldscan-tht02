"""C04 AutoCAD R2010 DXF, vectorized binary STL, HUD PNG, offline EML/ZIP escalation."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import struct
import zipfile
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import ezdxf
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_ROOT = Path(__file__).resolve().parents[2]
_SPEC_PATH = _ROOT / "knowledge" / "THT02_spec.json"

_STL_HEADER = b"TZ THT-02 Parametric Twin | 140.0x16.0mm | FieldScan Offline CAD"
_STL_DTYPE = np.dtype(
    [
        ("normal", "<f4", (3,)),
        ("v1", "<f4", (3,)),
        ("v2", "<f4", (3,)),
        ("v3", "<f4", (3,)),
        ("attr", "<u2"),
    ]
)

_ZONE_ALIAS_MAP: dict[str, str] = {
    "seal_ring": "upper_seal_ring",
    "middle_barrel": "lower_barrel_rs485",
    "rs485_barrel": "lower_barrel_rs485",
    "dip_section": "upper_barrel_dip",
    "strainer": "strainer_head",
    "gland": "waterproof_gland_leads",
    "flange": "mounting_flange",
}

_ZONE_SPANS_MM: dict[str, tuple[float, float]] = {
    "strainer_head": (0.0, 32.0),
    "upper_seal_ring": (32.0, 38.0),
    "upper_barrel_dip": (38.0, 68.0),
    "mounting_flange": (68.0, 88.0),
    "lower_barrel_rs485": (88.0, 122.0),
    "waterproof_gland_leads": (122.0, 140.0),
}

_FUZZY_ZONE_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(strainer|sht|filter)\b", re.I), "strainer_head"),
    (re.compile(r"\b(seal|oring|o-ring)\b", re.I), "upper_seal_ring"),
    (re.compile(r"\b(dip|pcb|switch)\b", re.I), "upper_barrel_dip"),
    (re.compile(r"\b(flange|bracket)\b", re.I), "mounting_flange"),
    (re.compile(r"\b(rs485|transceiver|max485)\b", re.I), "lower_barrel_rs485"),
    (re.compile(r"\b(gland|cable|wire|lead)\b", re.I), "waterproof_gland_leads"),
]


def _load_spec() -> dict[str, Any]:
    if _SPEC_PATH.exists():
        return json.loads(_SPEC_PATH.read_text(encoding="utf-8"))
    return {}


def resolve_zone_id(name: str | None) -> str | None:
    if not name:
        return None
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    if key in _ZONE_SPANS_MM:
        return key
    if key in _ZONE_ALIAS_MAP:
        return _ZONE_ALIAS_MAP[key]
    for pattern, zone in _FUZZY_ZONE_RULES:
        if pattern.search(name):
            return zone
    return None


def _layer_setup(doc: ezdxf.document.Drawing) -> None:
    doc.header["$INSUNITS"] = 4
    layers = [
        ("OUTLINE", 7, 0.35),
        ("STRAINER_VENTS", 4, 0.25),
        ("SEAL_ORINGS", 8, 0.25),
        ("INTERNAL_PCB", 5, 0.25),
        ("GLAND_RIBS", 9, 0.25),
        ("CENTERLINE", 1, 0.18),
        ("DIMENSIONS", 3, 0.25),
        ("TITLE_BLOCK", 7, 0.35),
        ("LEAD_RED_VP", 1, 0.35),
        ("LEAD_BLACK_GND", 250, 0.35),
        ("LEAD_YELLOW_AP", 2, 0.35),
        ("LEAD_GREEN_BM", 3, 0.35),
        ("FAULT_CALLOUT", 1, 0.50),
    ]
    for lname, aci, width in layers:
        layer = doc.layers.add(lname, color=aci)
        layer.dxf.lineweight = int(width * 100)
    fault = doc.layers.get("FAULT_CALLOUT")
    fault.rgb = (255, 23, 68)


def export_autocad_dxf(path: Path | str, fault_zone: str | None = None) -> Path:
    """Write AutoCAD R2010 side elevation + front section with optional FAULT_CALLOUT."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    zone = resolve_zone_id(fault_zone) if fault_zone else None

    doc = ezdxf.new("R2010", setup=True)
    _layer_setup(doc)
    msp = doc.modelspace()

    half = 8.0
    outline = msp.add_lwpolyline(
        [(0, -half), (32, -7.5), (38, -half), (68, -half), (88, -half), (122, -half), (140, -7.2), (140, half), (122, half), (88, half), (68, half), (38, half), (32, 7.5), (0, half), (0, -half)],
        dxfattribs={"layer": "OUTLINE"},
        close=True,
    )
    outline.close(True)

    for i, x0 in enumerate([6, 12, 18, 24, 30]):
        msp.add_line((x0, -5.5), (x0 + 4, 5.5), dxfattribs={"layer": "STRAINER_VENTS"})
    msp.add_circle((35, 0), 8.2, dxfattribs={"layer": "SEAL_ORINGS"})
    msp.add_lwpolyline([(40, -4), (66, -4), (66, 4), (40, 4)], dxfattribs={"layer": "INTERNAL_PCB"}, close=True)
    sw1 = msp.add_text("SW1 DIP-8", dxfattribs={"layer": "INTERNAL_PCB", "height": 2.5})
    sw1.set_placement((42, 5))
    msp.add_line((0, 0), (140, 0), dxfattribs={"layer": "CENTERLINE", "linetype": "CENTER"})

    msp.add_lwpolyline(
        [(68, -21), (88, -21), (88, 21), (68, 21)],
        dxfattribs={"layer": "OUTLINE"},
        close=True,
    )
    for y in (-16, 16):
        msp.add_circle((78, y), 2.1, dxfattribs={"layer": "OUTLINE"})
    for i in range(6):
        z = 124 + i * 2.4
        msp.add_line((z, -7.8), (z, 7.8), dxfattribs={"layer": "GLAND_RIBS"})

    lead_specs = [
        ("LEAD_RED_VP", 140, 5.0, 176, 5.0),
        ("LEAD_BLACK_GND", 140, 1.5, 176, 1.5),
        ("LEAD_YELLOW_AP", 140, -1.5, 176, -1.5),
        ("LEAD_GREEN_BM", 140, -5.0, 176, -5.0),
    ]
    for layer, x0, y0, x1, y1 in lead_specs:
        msp.add_line((x0, y0), (x1, y1), dxfattribs={"layer": layer})

    cx, cy = 232.0, 0.0
    msp.add_circle((cx, cy), 8.0, dxfattribs={"layer": "OUTLINE"})
    msp.add_circle((cx, cy), 6.4, dxfattribs={"layer": "INTERNAL_PCB"})
    msp.add_lwpolyline(
        [(cx - 21, cy - 8), (cx + 21, cy - 8), (cx + 21, cy + 8), (cx - 21, cy + 8)],
        dxfattribs={"layer": "OUTLINE"},
        close=True,
    )
    for ox in (-16, 16):
        msp.add_circle((cx + ox, cy), 2.1, dxfattribs={"layer": "OUTLINE"})

    dim = msp.add_linear_dim(
        base=(70, -30),
        p1=(0, -12),
        p2=(140, -12),
        dxfattribs={"layer": "DIMENSIONS"},
    )
    dim.render()

    msp.add_lwpolyline(
        [(-22, -58), (288, -58), (288, 56), (-22, 56)],
        dxfattribs={"layer": "TITLE_BLOCK"},
        close=True,
    )
    title = msp.add_text(
        "TZ THT-02 FieldScan Offline CAD",
        dxfattribs={"layer": "TITLE_BLOCK", "height": 4.5},
    )
    title.set_placement((-18, 44))

    if zone and zone in _ZONE_SPANS_MM:
        z0, z1 = _ZONE_SPANS_MM[zone]
        z_mid = (z0 + z1) / 2.0
        env = 9.5 if zone == "mounting_flange" else 8.6
        msp.add_lwpolyline(
            [(z0 - 1.2, -env), (z1 + 1.2, -env), (z1 + 1.2, env), (z0 - 1.2, env)],
            dxfattribs={"layer": "FAULT_CALLOUT"},
            close=True,
        )
        msp.add_line((z0 - 1.2, -env), (z1 + 1.2, env), dxfattribs={"layer": "FAULT_CALLOUT"})
        msp.add_line((z0 - 1.2, env), (z1 + 1.2, -env), dxfattribs={"layer": "FAULT_CALLOUT"})
        msp.add_line((z_mid, env), (z_mid + 8, 39), dxfattribs={"layer": "FAULT_CALLOUT"})
        msp.add_circle((cx, cy), 8.4, dxfattribs={"layer": "FAULT_CALLOUT"})

    doc.saveas(out)
    return out


def _radius_at_z(z: float, theta: np.ndarray) -> np.ndarray:
    r_base = 8.0
    if z < 4:
        taper = 0.55 + 0.45 * (z / 4.0)
        r = np.full_like(theta, r_base * taper, dtype=np.float64)
    elif z < 32:
        slot = np.cos(6 * theta)
        r = np.where(slot > 0.45, r_base - 1.25, r_base)
    elif z < 38:
        r = np.full_like(theta, r_base + 0.2, dtype=np.float64)
    elif z < 122:
        r = np.full_like(theta, r_base, dtype=np.float64)
    else:
        taper = 1.0 - 0.08 * ((z - 122) / 18.0)
        rib = 0.15 * np.sin(12 * theta)
        r = r_base * taper + rib
    if 78 <= z <= 88:
        r = np.maximum(r, r_base + 0.05)
    return r


def _revolve_profile_triangles(n_theta: int = 48) -> np.ndarray:
    z_stations = np.linspace(0.0, 140.0, 22)
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    tris: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    prev_ring: list[tuple[float, float, float]] | None = None
    for z in z_stations:
        r = _radius_at_z(float(z), theta)
        ring = [(float(z), float(r[i] * np.cos(theta[i])), float(r[i] * np.sin(theta[i]))) for i in range(n_theta)]
        if prev_ring is not None:
            for i in range(n_theta):
                j = (i + 1) % n_theta
                a, b, c, d = prev_ring[i], prev_ring[j], ring[j], ring[i]
                tris.append(
                    (
                        np.array(a, dtype=np.float32),
                        np.array(b, dtype=np.float32),
                        np.array(c, dtype=np.float32),
                    )
                )
                tris.append(
                    (
                        np.array(a, dtype=np.float32),
                        np.array(c, dtype=np.float32),
                        np.array(d, dtype=np.float32),
                    )
                )
        prev_ring = ring
    return np.array(tris, dtype=object)


def _perforated_flange_ear_triangles() -> np.ndarray:
    z = 78.0
    span = 21.0
    hole_r = 2.1
    extra: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for side in (-1.0, 1.0):
        cx = side * 16.0
        outer = [
            (z, cx - span, -6.0),
            (z + 10, cx - span, -6.0),
            (z + 10, cx + span, -6.0),
            (z, cx + span, -6.0),
        ]
        o0, o1, o2, o3 = [np.array(p, dtype=np.float32) for p in outer]
        extra.append((o0, o1, o2))
        extra.append((o0, o2, o3))
        for ang in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            hx = cx + hole_r * np.cos(ang)
            hz = z + 5 + hole_r * np.sin(ang)
            extra.append(
                (
                    np.array([z, hx, -6.0], dtype=np.float32),
                    np.array([z + 10, hx, -6.0], dtype=np.float32),
                    np.array([z + 5, cx, -4.5], dtype=np.float32),
                )
            )
    return np.array(extra, dtype=object)


def export_parametric_stl(path: Path | str) -> Path:
    """Binary STL with standard 80-byte header and numpy-packed triangles."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rev = _revolve_profile_triangles(48)
    ears = _perforated_flange_ear_triangles()
    all_tris = list(rev) + list(ears)

    records = []
    for v1, v2, v3 in all_tris:
        n = np.cross(v2 - v1, v3 - v1)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            n_hat = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            n_hat = (n / norm).astype(np.float32)
        records.append((n_hat, v1.astype(np.float32), v2.astype(np.float32), v3.astype(np.float32), 0))

    packed = np.array(records, dtype=_STL_DTYPE)
    header = _STL_HEADER[:80].ljust(80, b"\x00")
    with out.open("wb") as f:
        f.write(header)
        f.write(struct.pack("<I", len(packed)))
        f.write(packed.tobytes())
    return out


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _render_annotated_snapshot(
    out_path: Path,
    fault_zone: str | None,
    diagnosis_text: str,
    snapshot_b64: str | None = None,
) -> Path:
    if snapshot_b64:
        try:
            raw = base64.b64decode(snapshot_b64.split(",", 1)[-1])
            img = Image.open(io.BytesIO(raw)).convert("RGB")  # type: ignore[name-defined]
            img = img.resize((960, 540))
        except Exception:
            img = None
    else:
        img = None
    if img is None:
        img = Image.new("RGB", (960, 540), (12, 17, 24))
        draw = ImageDraw.Draw(img)
        draw.rectangle((40, 80, 920, 460), outline=(56, 189, 248), width=2)
        draw.text((60, 40), "TZ THT-02 Offline Escalation Snapshot", fill=(56, 189, 248))
        z = resolve_zone_id(fault_zone)
        if z:
            z0, z1 = _ZONE_SPANS_MM.get(z, (0, 32))
            x0 = 40 + int(880 * (z0 / 140.0))
            x1 = 40 + int(880 * (z1 / 140.0))
            draw.rectangle((x0, 120, x1, 420), outline=(255, 23, 68), width=3)
    draw = ImageDraw.Draw(img)
    zone = resolve_zone_id(fault_zone) or "none"
    draw.text((20, 500), f"Zone: {zone} | {diagnosis_text[:96]}", fill=(200, 220, 235))
    img.save(out_path, format="PNG")
    return out_path


def _write_report_md(path: Path, manifest: list[tuple[str, str]], body: dict[str, Any]) -> Path:
    lines = [
        "# FieldScan TZ THT-02 Escalation Report",
        "",
        f"- Package ID: `{body.get('package_id')}`",
        f"- Fault zone: `{body.get('fault_zone')}`",
        f"- Generated UTC: `{body.get('generated_utc')}`",
        "",
        "## SHA-256 manifest",
        "",
        "| File | SHA-256 |",
        "| --- | --- |",
    ]
    for name, digest in manifest:
        lines.append(f"| {name} | `{digest}` |")
    lines.extend(
        [
            "",
            "## Diagnosis",
            "",
            body.get("diagnosis_text") or "",
            "",
            "## Operator notes",
            "",
            body.get("operator_notes") or "(none)",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _build_offline_eml(
    path: Path,
    subject: str,
    body_text: str,
    attachments: list[tuple[str, bytes, str]],
) -> Path:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = "fieldscan@localhost"
    msg["To"] = "manufacturer-escalation@localhost"
    msg["X-FieldScan-Outbox-Mode"] = "STORE_AND_FORWARD_OFFLINE"
    msg.set_content(body_text)
    for fname, data, mime in attachments:
        maintype, subtype = mime.split("/", 1)
        msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=fname)
    path.write_bytes(msg.as_bytes())
    return path


def build_escalation_package(
    exports_dir: Path | str,
    outbox_dir: Path | str,
    fault_zone: str | None = None,
    diagnosis_text: str = "",
    operator_notes: str = "",
    snapshot_b64: str | None = None,
    modbus_tel: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile ESC-THT02 package: dxf, stl, png, md, eml, zip."""
    exports = Path(exports_dir)
    outbox = Path(outbox_dir)
    exports.mkdir(parents=True, exist_ok=True)
    outbox.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    seq = len(list(exports.glob("ESC-THT02-*"))) + 1
    package_id = f"ESC-THT02-{stamp}-{seq:03d}"
    work = exports / package_id
    work.mkdir(parents=True, exist_ok=True)

    zone = resolve_zone_id(fault_zone)
    dxf_path = export_autocad_dxf(work / f"{package_id}.dxf", fault_zone=zone)
    stl_path = export_parametric_stl(work / f"{package_id}.stl")
    png_path = _render_annotated_snapshot(
        work / f"{package_id}_snapshot.png", zone, diagnosis_text, snapshot_b64
    )

    meta = {
        "package_id": package_id,
        "fault_zone": zone,
        "generated_utc": stamp,
        "diagnosis_text": diagnosis_text,
        "operator_notes": operator_notes,
        "modbus_tel": modbus_tel or {},
    }
    manifest = [
        (dxf_path.name, _sha256_file(dxf_path)),
        (stl_path.name, _sha256_file(stl_path)),
        (png_path.name, _sha256_file(png_path)),
    ]
    md_path = _write_report_md(work / f"{package_id}_report.md", manifest, meta)
    manifest.append((md_path.name, _sha256_file(md_path)))

    zip_path = exports / f"{package_id}_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in (dxf_path, stl_path, png_path, md_path):
            zf.write(p, arcname=p.name)

    attach_bytes: list[tuple[str, bytes, str]] = []
    for p, mime in (
        (dxf_path, "application/dxf"),
        (stl_path, "model/stl"),
        (png_path, "image/png"),
        (md_path, "text/markdown"),
    ):
        attach_bytes.append((p.name, p.read_bytes(), mime))

    eml_path = _build_offline_eml(
        outbox / f"{package_id}.eml",
        subject=f"[FieldScan Offline] TZ THT-02 Escalation {zone or 'general'}",
        body_text=(
            f"FieldScan offline escalation package {package_id}.\n\n"
            f"Fault zone: {zone}\n"
            f"Diagnosis: {diagnosis_text}\n\n"
            f"Operator notes: {operator_notes or '(none)'}\n"
        ),
        attachments=attach_bytes,
    )

    return {
        "package_id": package_id,
        "fault_zone": zone,
        "paths": {
            "dxf": str(dxf_path),
            "stl": str(stl_path),
            "png": str(png_path),
            "md": str(md_path),
            "eml": str(eml_path),
            "zip": str(zip_path),
        },
        "manifest_sha256": manifest,
    }


__all__ = [
    "_ZONE_ALIAS_MAP",
    "resolve_zone_id",
    "export_autocad_dxf",
    "export_parametric_stl",
    "build_escalation_package",
]