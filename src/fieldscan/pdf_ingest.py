"""C02 PDF manual ingestion: fast CTM regex extractor + ASR phonetic normalizer."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PDF = _ROOT / "assets" / "TZ_THT02_Manual_V1.1.pdf"
_DEFAULT_PAGES = _ROOT / "assets" / "manual_pages"
_SPEC_PATH = _ROOT / "knowledge" / "THT02_spec.json"

_PDF_IMAGE_CTM_RE = re.compile(
    r"([+-]?\d*\.?\d+)\s+([+-]?\d*\.?\d+)\s+([+-]?\d*\.?\d+)\s+([+-]?\d*\.?\d+)\s+"
    r"([+-]?\d*\.?\d+)\s+([+-]?\d*\.?\d+)\s+cm\s*/(Im\d+|img\d+|X\d+)\s+Do"
)

_ASR_REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\btee[\s\-]?aitch[\s\-]?tee\b", re.I), "tht"),
    (re.compile(r"\bTHS[\s\-]?0{0,2}\b", re.I), "TZ-THT-02"),
    (re.compile(r"\bTHP[\s\-]?0?2\b", re.I), "TZ-THT-02"),
    (re.compile(r"\bTZ[\s\-]?THT[\s\-]?0?2\b", re.I), "TZ-THT-02"),
    (re.compile(r"\bour[\s\-]?s[\s\-]?four[\s\-]?eighty[\s\-]?five\b", re.I), "RS-485"),
    (re.compile(r"\bR[\s\-]?S[\s\-]?485\b", re.I), "RS-485"),
    (re.compile(r"\bs[\s\-]?h[\s\-]?t[\s\-]?thirty\b", re.I), "SHT30"),
    (re.compile(r"\bseven[\s\-]?triple[\s\-]?f\b", re.I), "0x7FFF"),
    (re.compile(r"\bhex[\s\-]?7[\s\-]?f[\s\-]?f[\s\-]?f\b", re.I), "0x7FFF"),
]

_PAGE_TOPIC_BOOST: dict[int, tuple[str, ...]] = {
    1: ("overview", "strainer", "dimensions", "sensor"),
    2: ("supply", "voltage", "current", "rating"),
    3: ("housing", "barrel", "polycarbonate"),
    4: ("wire", "wiring", "yellow", "green", "red", "black", "gland", "lead"),
    5: ("dip", "switch", "sw1", "slave", "address", "pin"),
    6: ("register", "modbus", "0x7fff", "7fff", "crc", "temperature"),
    7: ("baud", "protocol", "humidity", "dew"),
    8: ("pcb", "dip", "installation"),
    9: ("cable", "gland", "lead", "harness"),
}


def normalize_asr_equipment_terms(text: str) -> str:
    """Rewrite spoken ASR misrecognitions into canonical equipment terms."""
    out = text or ""
    for pattern, repl in _ASR_REPLACEMENTS:
        out = pattern.sub(repl, out)
    out = re.sub(r"\bTZ\s+THT\s*0?2\b", "TZ-THT-02", out, flags=re.I)
    out = re.sub(r"\bTHS\s*00\b", "TZ-THT-02", out, flags=re.I)
    return out


def _load_spec() -> dict[str, Any]:
    if _SPEC_PATH.exists():
        return json.loads(_SPEC_PATH.read_text(encoding="utf-8"))
    return {}


def ensure_synthetic_manual_pdf(pdf_path: Path | None = None) -> Path:
    """Create a minimal 9-page English manual PDF if the official asset is absent."""
    target = Path(pdf_path) if pdf_path else _DEFAULT_PDF
    if target.exists() and target.stat().st_size > 1024:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    import fitz  # PyMuPDF

    doc = fitz.open()
    page_bodies = [
        "TZ THT-02 Digital Temperature & Humidity Sensor Overview. Length 140.0 mm x Dia 16.0 mm. Strainer head with SHT30.",
        "Electrical ratings: DC 5V to 24V supply, max operating current 5 mA. Accuracy +/-0.2 C at 25 C.",
        "Lower polycarbonate barrel houses MAX485 transceiver and DC LDO regulator.",
        "Wiring: Red = V+, Black = GND common RS-485 ground, Yellow = RS-485 A+, Green = RS-485 B-. Waterproof gland leads.",
        "SW1 8-Pin DIP Switch addressing. Binary weights left-to-right 128 64 32 16 8 4 2 1. Slave ID 1 = Pin 8 ON. Power-cycle required.",
        "Modbus-RTU registers. Reg 0x0000 Temperature. Reg 0x0001 Humidity. Sentinel 0x7FFF abnormal probe. CRC-16 required.",
        "Baud rate Reg 0x0005 default 0x2580 = 9600 bps 8N1. Protocol and reporting mode registers.",
        "Installation notes for DIP PCB and upper barrel housing. Switch configuration checklist.",
        "Cable harness, compression gland, and four color-coded leads Red Black Yellow Green.",
    ]
    for i, body in enumerate(page_bodies, start=1):
        page = doc.new_page(width=595, height=842)
        page.insert_text((48, 72), f"TZ THT-02 User Manual V1.1 - Page {i}", fontsize=14)
        page.insert_text((48, 120), body, fontsize=11)
        # Small embedded image placeholder for CTM extraction paths
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 64, 32), 0)
        pix.set_rect(pix.irect, (180, 180, 190))
        page.insert_image(fitz.Rect(48, 200, 200, 260), pixmap=pix)
    doc.save(target)
    doc.close()
    return target


def _extract_ctm_diagrams(page: Any) -> list[dict[str, Any]]:
    diagrams: list[dict[str, Any]] = []
    page_h = float(page.rect.height)
    try:
        xref = page.get_contents()[0] if page.get_contents() else None
        if xref is None:
            return diagrams
        import fitz

        raw = page.parent.xref_stream(xref)
        text = raw.decode("latin1", errors="ignore")
    except Exception:
        return diagrams
    for m in _PDF_IMAGE_CTM_RE.finditer(text):
        a, _b, _c, d, e, f, name = m.groups()
        a_f, d_f, e_f, f_f = float(a), float(d), float(e), float(f)
        x0 = e_f
        y0 = page_h - (f_f + d_f)
        x1 = e_f + a_f
        y1 = page_h - f_f
        diagrams.append(
            {
                "name": name,
                "bbox": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
            }
        )
    return diagrams


def ingest_manual(
    pdf_path: Path | str | None = None,
    pages_dir: Path | str | None = None,
    force_render: bool = False,
) -> dict[str, Any]:
    """Render page_1..page_9 PNGs idempotently and extract text blocks + CTM bboxes."""
    import fitz

    pdf = ensure_synthetic_manual_pdf(Path(pdf_path) if pdf_path else None)
    out_dir = Path(pages_dir) if pages_dir else _DEFAULT_PAGES
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    doc = fitz.open(pdf)
    pages: list[dict[str, Any]] = []
    for i in range(min(9, doc.page_count)):
        page = doc.load_page(i)
        png_path = out_dir / f"page_{i + 1}.png"
        if force_render or not png_path.exists() or png_path.stat().st_size < 5120:
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
            pix.save(png_path)
        blocks_raw = page.get_text("blocks")
        blocks = []
        for b in sorted(blocks_raw, key=lambda x: (x[1], x[0])):
            blocks.append(
                {
                    "bbox": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                    "text": str(b[4]).strip(),
                }
            )
        diagrams = _extract_ctm_diagrams(page)
        pages.append(
            {
                "page": i + 1,
                "image_url": f"/api/manual/page/{i + 1}",
                "image_path": str(png_path),
                "blocks": blocks,
                "diagrams": diagrams,
                "text": "\n".join(bl["text"] for bl in blocks if bl["text"]),
            }
        )
    doc.close()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "sensor_id": "TZ-THT-02",
        "pdf_path": str(pdf),
        "page_count": len(pages),
        "pages": pages,
        "elapsed_ms": elapsed_ms,
        "spec": _load_spec(),
    }


def retrieve_relevant_pages(query: str, manual: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
    """Score manual pages by keyword overlap + topic boost for DIP/register/wiring queries."""
    manual = manual or ingest_manual()
    q = normalize_asr_equipment_terms(query or "").lower()
    tokens = set(re.findall(r"[a-z0-9%/+.-]+", q))
    scored: list[tuple[float, dict[str, Any]]] = []
    for page in manual.get("pages", []):
        text = (page.get("text") or "").lower()
        page_tokens = set(re.findall(r"[a-z0-9%/+.-]+", text))
        overlap = len(tokens & page_tokens)
        score = float(overlap)
        n = int(page["page"])
        for kw in _PAGE_TOPIC_BOOST.get(n, ()):
            if kw in q:
                score += 3.0
        if any(k in q for k in ("dip", "switch", "slave", "pin")) and n in (5, 8):
            score += 5.0
        if any(k in q for k in ("7fff", "register", "crc", "modbus")) and n in (6, 7):
            score += 5.0
        if any(k in q for k in ("wire", "yellow", "gland", "cable", "lead")) and n in (4, 9):
            score += 4.0
        scored.append((score, page))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {
            "page": p["page"],
            "score": s,
            "snippet": (p.get("text") or "")[:280],
            "image_url": p.get("image_url"),
        }
        for s, p in scored[:top_k]
        if s > 0 or True
    ]


__all__ = [
    "normalize_asr_equipment_terms",
    "ingest_manual",
    "retrieve_relevant_pages",
    "ensure_synthetic_manual_pdf",
]
