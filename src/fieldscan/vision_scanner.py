"""C01 30 FPS OpenCV spatial CV, DIP-8 optical reader, and bench synthesizer for TZ THT-02."""

from __future__ import annotations

import glob
import io
import os
import platform
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from fieldscan.mac_uvc_worker import FRAME_MAGIC, HEADER_STRUCT
from fieldscan.modbus_rtu import dip_bits_to_address

SENSOR_ZONES: list[dict[str, Any]] = [
    {
        "id": "waterproof_gland_leads",
        "label": "Waterproof Gland & 4-Wire RS-485",
        "z_range_mm": [-70.0, -48.0],
        "manual_pages": [4, 7, 9],
    },
    {
        "id": "lower_barrel_rs485",
        "label": "Lower Barrel (MAX485 & LDO)",
        "z_range_mm": [-48.0, -18.0],
        "manual_pages": [3, 4, 6, 7],
    },
    {
        "id": "mounting_flange",
        "label": "Mounting Flange Bracket",
        "z_range_mm": [-18.0, -2.0],
        "manual_pages": [1, 4],
    },
    {
        "id": "upper_barrel_dip",
        "label": "Upper Barrel & SW1 DIP-8 PCB",
        "z_range_mm": [-2.0, 27.0],
        "manual_pages": [5, 6, 8],
    },
    {
        "id": "upper_seal_ring",
        "label": "Upper Seal Ring Collar",
        "z_range_mm": [27.0, 38.0],
        "manual_pages": [1, 4],
    },
    {
        "id": "strainer_head",
        "label": "SHT30 Slotted Strainer Head",
        "z_range_mm": [38.0, 70.0],
        "manual_pages": [1, 2, 4, 6],
    },
]

DIP_PIN8_NOTE = "MSB..LSB weights [128,64,32,16,8,4,2,1]"
PROC_W = 800
SYNTH_W, SYNTH_H = 800, 480
OUT_W, OUT_H = 1920, 1080
_VALID_MODES = frozenset(
    {
        "auto",
        "usb",
        "sim_healthy",
        "sim_fault_strainer",
        "sim_open_dip",
        "sim_hand_screwdriver",
    }
)
_ASSETS = Path(__file__).resolve().parents[2] / "assets" / "manual_pages"


def _read_exact(pipe: Any, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = pipe.read(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _z_cv_to_u_norm(z_cv: float) -> float:
    return max(0.0, min(1.0, (z_cv + 70.0) / 140.0))


def _project_zone_pixels(
    zone: dict[str, Any],
    p_start: list[float],
    p_end: list[float],
    major_px: float,
) -> dict[str, Any]:
    z0, z1 = zone["z_range_mm"]
    u0 = _z_cv_to_u_norm(z0)
    u1 = _z_cv_to_u_norm(z1)
    sx, sy = p_start[0], p_start[1]
    ex, ey = p_end[0], p_end[1]
    px0 = [sx + (ex - sx) * u0, sy + (ey - sy) * u0]
    px1 = [sx + (ex - sx) * u1, sy + (ey - sy) * u1]
    cx = (px0[0] + px1[0]) / 2.0
    cy = (px0[1] + px1[1]) / 2.0
    return {
        "id": zone["id"],
        "label": zone["label"],
        "z_range_mm": [z0, z1],
        "center": [round(cx, 1), round(cy, 1)],
        "p0": [round(px0[0], 1), round(px0[1], 1)],
        "p1": [round(px1[0], 1), round(px1[1], 1)],
        "manual_pages": list(zone["manual_pages"]),
    }


class _WorkbenchSynthesizer:
    """Deterministic 800x480 industrial bench simulator (upscaled to 1920x1080 downstream)."""

    def __init__(self, mode: str = "sim_healthy", dip_bits: list[int] | None = None) -> None:
        self.mode = mode
        self.dip_bits = list(dip_bits or [1, 0, 1, 1, 0, 0, 0, 0])
        self._t0 = time.time()
        self._page5 = self._load_crop("page_5.png", (372, 418, 56, 622))
        self._page9 = self._load_crop("page_9.png", (826, 900, 42, 588))

    def _load_crop(self, name: str, bounds: tuple[int, int, int, int]) -> np.ndarray | None:
        path = _ASSETS / name
        if not path.is_file():
            return None
        img = cv2.imread(str(path))
        if img is None:
            return None
        y0, y1, x0, x1 = bounds
        crop = img[y0:y1, x0:x1].copy()
        if name == "page_5.png":
            b, g, r = cv2.split(crop)
            mask = (r.astype(np.int16) - g.astype(np.int16) > 50) & (r > 150)
            crop[mask] = (140, 140, 140)
        return crop

    def _base_bench(self) -> np.ndarray:
        frame = np.full((SYNTH_H, SYNTH_W, 3), (38, 36, 34), dtype=np.uint8)
        cv2.rectangle(frame, (0, int(SYNTH_H * 0.72)), (SYNTH_W, SYNTH_H), (52, 50, 48), -1)
        return frame

    def _draw_cylinder(
        self,
        frame: np.ndarray,
        p_start: tuple[int, int],
        p_end: tuple[int, int],
        minor: int,
        *,
        open_pcb: bool = False,
        dip_bits: list[int] | None = None,
        stain: bool = False,
    ) -> tuple[float, float, list[float], list[float], float, float, float]:
        sx, sy = p_start
        ex, ey = p_end
        dx, dy = ex - sx, ey - sy
        major = float(np.hypot(dx, dy)) or 1.0
        angle = float(np.degrees(np.arctan2(-dy, dx)))
        thickness = max(8, minor)
        cv2.line(frame, p_start, p_end, (210, 208, 205), thickness, cv2.LINE_AA)
        cv2.circle(frame, p_start, thickness // 2 + 2, (180, 178, 175), -1, cv2.LINE_AA)
        cv2.circle(frame, p_end, thickness // 2 + 2, (200, 198, 195), -1, cv2.LINE_AA)
        if stain:
            mid = (int((sx + ex) * 0.85), int((sy + ey) * 0.85))
            cv2.circle(frame, mid, thickness, (40, 40, 200), -1, cv2.LINE_AA)
        if open_pcb:
            mx = int((sx + ex) * 0.42)
            my = int((sy + ey) * 0.42)
            bw, bh = int(major * 0.22), max(28, thickness + 12)
            rect = (mx - bw // 2, my - bh // 2, bw, bh)
            cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (200, 120, 40), -1)
            bits = dip_bits or self.dip_bits
            slot_w = max(4, rect[2] // 10)
            for i, bit in enumerate(bits):
                px = rect[0] + 8 + i * slot_w
                color = (240, 240, 240) if bit else (80, 80, 90)
                cv2.rectangle(
                    frame,
                    (px, rect[1] + 6),
                    (px + slot_w - 2, rect[1] + bh // 2 - 2),
                    color,
                    -1,
                )
        return major, float(thickness), [float(sx), float(sy)], [float(ex), float(ey)], angle, major / 140.0, thickness / 16.0

    def _draw_hand_tool(self, frame: np.ndarray) -> tuple[bool, bool, list[float], list[float], list[float] | None]:
        hand_color = (124, 156, 206)
        hx, hy, hw, hh = 0, 180, 160, 220
        cv2.rectangle(frame, (hx, hy), (hx + hw, hy + hh), hand_color, -1)
        cv2.ellipse(frame, (hx + 40, hy + 40), (36, 28), 0, 0, 360, hand_color, -1)
        cv2.ellipse(frame, (hx + 90, hy + 30), (32, 24), 0, 0, 360, hand_color, -1)
        tool_color = (50, 50, 220)
        tip = [float(hx + hw - 20), float(hy + hh // 2)]
        base = [float(hx + hw + 140), float(hy + hh // 2 - 6)]
        cv2.line(frame, (int(base[0]), int(base[1])), (int(tip[0]), int(tip[1])), (200, 200, 200), 4, cv2.LINE_AA)
        cv2.line(frame, (int(base[0]) - 40, int(base[1])), (int(base[0]), int(base[1])), tool_color, 10, cv2.LINE_AA)
        hand_box = [float(hx), float(hy), float(hw), float(hh)]
        tool_box = [base[0] - 50, base[1] - 12, 190.0, 24.0]
        return True, True, hand_box, tool_box, tip

    def render(self, frame_idx: int = 0) -> tuple[np.ndarray, dict[str, Any]]:
        t = time.time() - self._t0
        wobble = 3.8 * np.sin(t * 1.7 + frame_idx * 0.04)
        frame = self._base_bench()
        cy = int(SYNTH_H * 0.52 + wobble)
        p_start = (100, cy + 8)
        p_end = (700, cy - 12)
        meta: dict[str, Any] = {
            "hand_present": False,
            "tool_present": False,
            "pcb_open_detected": False,
            "anomaly_detected": False,
            "anomaly_zone": None,
            "dip_bits_optical": None,
        }
        open_pcb = self.mode == "sim_open_dip"
        stain = self.mode == "sim_fault_strainer"
        major, minor, ps, pe, angle, px_per_mm, diam_mm = self._draw_cylinder(
            frame,
            p_start,
            p_end,
            46,
            open_pcb=open_pcb,
            dip_bits=self.dip_bits,
            stain=stain,
        )
        if stain:
            meta["anomaly_detected"] = True
            meta["anomaly_zone"] = "strainer_head"
        if open_pcb:
            meta["pcb_open_detected"] = True
            meta["dip_bits_optical"] = list(self.dip_bits)
        hand_box = tool_box = None
        tool_tip = None
        if self.mode == "sim_hand_screwdriver":
            hp, tp, hand_box, tool_box, tool_tip = self._draw_hand_tool(frame)
            meta["hand_present"] = hp
            meta["tool_present"] = tp
        if self._page9 is not None and open_pcb:
            thumb = cv2.resize(self._page9, (120, 36))
            frame[SYNTH_H - 50 : SYNTH_H - 14, 20:140] = thumb
        noise = np.random.default_rng(frame_idx).normal(0, 2.4, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        meta.update(
            {
                "detected": True,
                "p_start": ps,
                "p_end": pe,
                "major_px": major,
                "minor_px": minor,
                "angle_deg": angle,
                "px_per_mm": px_per_mm,
                "estimated_length_mm": 140.0,
                "estimated_diameter_mm": max(16.0, diam_mm * 16.0),
                "hand_box": hand_box,
                "tool_box": tool_box,
                "tool_tip": tool_tip,
                "pose_locked_by_occlusion": bool(meta["hand_present"] or meta["tool_present"]),
            }
        )
        return frame, meta


class VisionScanner:
    """USB / V4L2 / FSCN macOS pipe reader with bench simulator fallback."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._mode = "auto"
        self._sim_mode = "sim_healthy"
        self._dip_bits: list[int] = [1, 0, 1, 1, 0, 0, 0, 0]
        self._synth = _WorkbenchSynthesizer(self._sim_mode, self._dip_bits)
        self._frame_bgr: np.ndarray | None = None
        self._dip_crop_jpeg: bytes | None = None
        self._uploaded: np.ndarray | None = None
        self._usb_proc: subprocess.Popen[bytes] | None = None
        self._cap: cv2.VideoCapture | None = None
        self._usb_device = ""
        self._usb_available = False
        self._active_source = "sim"
        self._camera_mode = "sim_healthy"
        self._frame_idx = 0
        self._last_ts = time.time()
        self._fps = 30.0
        self._locked_pose: dict[str, Any] | None = None
        self._telemetry = self._empty_telemetry()

    def _empty_telemetry(self) -> dict[str, Any]:
        return {
            "camera_mode": "sim_healthy",
            "active_source": "sim",
            "usb_available": False,
            "usb_device": "",
            "fps": 0.0,
            "frame_size": [OUT_W, OUT_H],
            "detected": False,
            "tracking_locked": False,
            "pose_locked_by_occlusion": False,
            "angle_deg": 0.0,
            "center_norm": [0.5, 0.5],
            "bbox": None,
            "corners": None,
            "p_start": None,
            "p_end": None,
            "major_px": 0.0,
            "minor_px": 0.0,
            "px_per_mm": 0.0,
            "estimated_length_mm": 0.0,
            "estimated_diameter_mm": 0.0,
            "pcb_open_detected": False,
            "dip_box": None,
            "dip_corners": None,
            "dip_crop_available": False,
            "dip_bits_optical": None,
            "dip_slave_optical": None,
            "dip_optical_Pin8_note": DIP_PIN8_NOTE,
            "hand_present": False,
            "hand_box": None,
            "hand_boxes": [],
            "tool_present": False,
            "tool_box": None,
            "tool_boxes": [],
            "tool_tip": None,
            "tool_shaft": None,
            "tool_target_zone": None,
            "anomaly_detected": False,
            "anomaly_zone": None,
            "anomaly_zone_label": None,
            "anomaly_score": 0.0,
            "anomaly_z_mm": None,
            "zones_2d": [],
        }

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="vision-scanner", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._close_usb()

    def set_mode(self, mode: str, dip_bits: list[int] | None = None) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(f"unsupported camera mode: {mode}")
        with self._lock:
            self._mode = mode
            if dip_bits is not None:
                self._dip_bits = [int(bool(b)) for b in dip_bits[:8]]
                while len(self._dip_bits) < 8:
                    self._dip_bits.append(0)
            if mode.startswith("sim_"):
                self._sim_mode = mode
                self._synth = _WorkbenchSynthesizer(mode, self._dip_bits)
                self._camera_mode = mode
                self._active_source = "sim"
            elif mode == "usb":
                self._camera_mode = "usb"
            else:
                self._camera_mode = mode
        if not (self._thread and self._thread.is_alive()):
            self._refresh_once()

    def load_uploaded_frame(self, data: bytes) -> None:
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            with self._lock:
                self._uploaded = img
                self._active_source = "upload"

    def get_telemetry(self) -> dict[str, Any]:
        if self._frame_bgr is None and not (self._thread and self._thread.is_alive()):
            self._refresh_once()
        with self._lock:
            return dict(self._telemetry)

    def get_jpeg(self, raw: bool = False) -> bytes | None:
        with self._lock:
            frame = self._frame_bgr
        if frame is None:
            return None
        quality = 95 if raw else 82
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return bytes(buf) if ok else None

    def get_dip_focus_jpeg(self) -> bytes | None:
        with self._lock:
            return self._dip_crop_jpeg

    def _close_usb(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._usb_proc is not None:
            try:
                self._usb_proc.terminate()
                self._usb_proc.wait(timeout=1.0)
            except Exception:
                pass
            self._usb_proc = None

    def _discover_v4l2(self) -> tuple[int, str] | None:
        for dev in sorted(glob.glob("/dev/video*")):
            name_path = f"/sys/class/video4linux/{os.path.basename(dev)}/name"
            label = ""
            try:
                with open(name_path, encoding="utf-8") as fh:
                    label = fh.read().strip()
            except OSError:
                label = dev
            cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
            if cap.isOpened():
                cap.release()
                return 0, f"v4l2:{dev} ({label})"
        return None

    def _ensure_usb(self) -> bool:
        if platform.system() == "Darwin":
            if self._usb_proc is None or self._usb_proc.poll() is not None:
                self._usb_proc = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "fieldscan.mac_uvc_worker",
                        "--width",
                        str(OUT_W),
                        "--height",
                        str(OUT_H),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    bufsize=0,
                )
            return self._usb_proc.stdout is not None
        if self._cap is None or not self._cap.isOpened():
            found = self._discover_v4l2()
            if found is None:
                return False
            idx, label = found
            dev = label.split("(")[0].replace("v4l2:", "").strip()
            self._cap = cv2.VideoCapture(dev if dev.startswith("/dev") else idx, cv2.CAP_V4L2)
            if not self._cap.isOpened():
                return False
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(OUT_W))
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(OUT_H))
            self._cap.set(cv2.CAP_PROP_AUTOFOCUS, 1.0)
            self._usb_device = label
        return True

    def _read_mac_frame(self) -> tuple[np.ndarray | None, str]:
        if self._usb_proc is None or self._usb_proc.stdout is None:
            return None, ""
        pipe = self._usb_proc.stdout
        hdr = _read_exact(pipe, HEADER_STRUCT.size)
        if hdr is None:
            return None, ""
        magic, w, h, src_len, payload_len = HEADER_STRUCT.unpack(hdr)
        if magic != FRAME_MAGIC:
            return None, ""
        src_b = _read_exact(pipe, src_len) or b""
        raw = _read_exact(pipe, payload_len)
        if raw is None or len(raw) != payload_len:
            return None, ""
        if payload_len != w * h * 3:
            return None, src_b.decode("utf-8", errors="replace")
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3)).copy()
        return frame, src_b.decode("utf-8", errors="replace")

    def _acquire_frame(self) -> tuple[np.ndarray | None, dict[str, Any] | None]:
        want_usb = self._mode in ("auto", "usb")
        synth_meta: dict[str, Any] | None = None
        if self._uploaded is not None:
            with self._lock:
                up = self._uploaded.copy()
            frame = cv2.resize(up, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)
            return frame, None
        if want_usb and self._ensure_usb():
            frame: np.ndarray | None = None
            label = self._usb_device
            if platform.system() == "Darwin":
                frame, label = self._read_mac_frame()
            elif self._cap is not None:
                ok, raw = self._cap.read()
                if ok and raw is not None:
                    frame = raw
            if frame is not None:
                self._usb_available = True
                self._usb_device = label
                self._active_source = "usb"
                self._camera_mode = "usb"
                if frame.shape[1] != OUT_W or frame.shape[0] != OUT_H:
                    frame = cv2.resize(frame, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)
                return frame, None
        sim = self._sim_mode if self._mode == "auto" else self._mode
        if sim not in _VALID_MODES or sim in ("auto", "usb"):
            sim = "sim_healthy"
        if sim != self._synth.mode:
            self._synth = _WorkbenchSynthesizer(sim, self._dip_bits)
        small, synth_meta = self._synth.render(self._frame_idx)
        frame = cv2.resize(small, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)
        self._active_source = "sim"
        self._camera_mode = sim
        self._usb_available = False
        return frame, synth_meta

    def _scale_synth_pose(self, meta: dict[str, Any]) -> dict[str, Any]:
        sx = OUT_W / SYNTH_W
        sy = OUT_H / SYNTH_H
        ps = meta["p_start"]
        pe = meta["p_end"]
        out = dict(meta)
        out["p_start"] = [ps[0] * sx, ps[1] * sy]
        out["p_end"] = [pe[0] * sx, pe[1] * sy]
        out["major_px"] = meta["major_px"] * sx
        out["minor_px"] = meta["minor_px"] * sy
        out["px_per_mm"] = out["major_px"] / 140.0
        if meta.get("hand_box"):
            hb = meta["hand_box"]
            out["hand_box"] = [hb[0] * sx, hb[1] * sy, hb[2] * sx, hb[3] * sy]
        if meta.get("tool_box"):
            tb = meta["tool_box"]
            out["tool_box"] = [tb[0] * sx, tb[1] * sy, tb[2] * sx, tb[3] * sy]
        if meta.get("tool_tip"):
            tt = meta["tool_tip"]
            out["tool_tip"] = [tt[0] * sx, tt[1] * sy]
        return out

    def _detect_hand_tool_masks(self, proc_bgr: np.ndarray) -> dict[str, Any]:
        ph, pw = proc_bgr.shape[:2]
        sc = PROC_W / max(pw, 1)
        hsv = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2YCrCb)
        skin_ycrcb = cv2.inRange(ycrcb, (55, 138, 77), (255, 174, 121))
        h, s, v = cv2.split(hsv)
        skin_hsv = (((h <= 25) | (h >= 166)) & (s >= 38) & (s <= 168) & (v >= 60)).astype(np.uint8) * 255
        skin_raw = cv2.bitwise_and(skin_ycrcb, skin_hsv)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(skin_raw, connectivity=8)
        hand_mask = np.zeros_like(skin_raw)
        hand_boxes: list[list[float]] = []
        for i in range(1, num):
            x, y, w, h, area = stats[i]
            if area < 950 * sc * sc:
                continue
            if area > 0.36 * pw * ph:
                continue
            border = (
                x <= 0.08 * pw or y <= 0.08 * ph or (x + w) >= 0.92 * pw or (y + h) >= 0.92 * ph
            )
            if not border:
                continue
            hand_mask[labels == i] = 255
            hand_boxes.append([float(x), float(y), float(w), float(h)])
        hand_present = bool(hand_boxes)
        gray = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2GRAY)
        bright = ((v > 130) & (s < 72)).astype(np.uint8) * 255
        bright = cv2.bitwise_and(bright, cv2.bitwise_not(skin_raw))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        thick = cv2.dilate(cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel), kernel)
        shaft = cv2.bitwise_and(bright, cv2.bitwise_not(thick))
        grip = ((s > 118) & (v > 75) & ~((h >= 88) & (h <= 138))).astype(np.uint8) * 255
        hand_d = cv2.dilate(hand_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)))
        tool_mask = cv2.bitwise_or(shaft, grip)
        tool_mask = cv2.bitwise_and(tool_mask, hand_d)
        tool_present = bool(cv2.countNonZero(tool_mask) > 200)
        tool_boxes: list[list[float]] = []
        if tool_present:
            ys, xs = np.where(tool_mask > 0)
            if xs.size:
                tool_boxes.append(
                    [
                        float(xs.min()),
                        float(ys.min()),
                        float(xs.max() - xs.min()),
                        float(ys.max() - ys.min()),
                    ]
                )
        return {
            "hand_present": hand_present,
            "hand_boxes": hand_boxes,
            "tool_present": tool_present or hand_present,
            "tool_boxes": tool_boxes,
            "exclusion_mask": cv2.dilate(
                cv2.bitwise_or(hand_mask, tool_mask),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            ),
        }

    def _detect_cylinder(self, proc_bgr: np.ndarray, exclusion: np.ndarray | None) -> dict[str, Any] | None:
        blur = cv2.medianBlur(proc_bgr, 7)
        lab = cv2.cvtColor(blur, cv2.COLOR_BGR2Lab)
        l_ch = lab[:, :, 0]
        bg = cv2.boxFilter(l_ch, -1, (51, 51))
        mask = ((l_ch.astype(np.int16) - bg.astype(np.int16)) > 9).astype(np.uint8) * 255
        if exclusion is not None:
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(exclusion))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = 0.0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1100:
                continue
            rect = cv2.minAreaRect(cnt)
            (_, _), (rw, rh), _ = rect
            major, minor = (max(rw, rh), min(rw, rh)) if rw and rh else (0.0, 0.0)
            if minor < 10 or major < 40:
                continue
            aspect = major / minor if minor else 0.0
            if aspect < 2.2 or aspect > 22.0:
                continue
            score = area * aspect
            if score > best_score:
                best_score = score
                best = rect
        if best is None:
            return None
        box = cv2.boxPoints(best)
        box = np.array(box, dtype=np.float32)
        idx = int(np.argmax(box[:, 0]) - np.argmin(box[:, 0]) if True else 0)
        ps = box[np.argmin(box[:, 0])]
        pe = box[np.argmax(box[:, 0])]
        major = float(np.linalg.norm(pe - ps)) or 1.0
        _, (rw, rh), _ = best
        minor = float(min(rw, rh))
        angle = float(np.degrees(np.arctan2(-(pe[1] - ps[1]), pe[0] - ps[0])))
        return {
            "p_start": [float(ps[0]), float(ps[1])],
            "p_end": [float(pe[0]), float(pe[1])],
            "major_px": major,
            "minor_px": minor,
            "angle_deg": angle,
            "corners": box.tolist(),
            "px_per_mm": major / 140.0,
            "estimated_length_mm": 140.0,
            "estimated_diameter_mm": minor / (major / 140.0) if major else 16.0,
        }

    def _build_dip_crop(self, frame: np.ndarray, pose: dict[str, Any], bits: list[int]) -> bytes | None:
        ps = pose["p_start"]
        pe = pose["p_end"]
        mx = (ps[0] + pe[0]) / 2.0
        my = (ps[1] + pe[1]) / 2.0
        angle = pose.get("angle_deg", 0.0)
        crop = frame[max(0, int(my) - 80) : int(my) + 80, max(0, int(mx) - 120) : int(mx) + 120]
        if crop.size == 0:
            crop = frame.copy()
        h, w = crop.shape[:2]
        mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 3.3)
        warped = cv2.warpAffine(
            crop,
            mat,
            (528, 286),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REPLICATE,
        )
        blur = cv2.GaussianBlur(warped, (0, 0), 1.2)
        sharp = cv2.addWeighted(warped, 1.38, blur, -0.38, 0)
        slave = dip_bits_to_address(bits)
        cv2.putText(
            sharp,
            f"SLAVE #{slave}",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        weights = "128 64 32 16 8 4 2 1"
        cv2.putText(
            sharp,
            weights,
            (12, 270),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        ok, buf = cv2.imencode(".jpg", sharp, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        return bytes(buf) if ok else None

    def _update_from_frame(self, frame: np.ndarray, synth_meta: dict[str, Any] | None) -> None:
        now = time.time()
        dt = max(now - self._last_ts, 1e-6)
        self._fps = 0.85 * self._fps + 0.15 * (1.0 / dt)
        self._last_ts = now
        self._frame_idx += 1
        scale_x = OUT_W / PROC_W
        proc = cv2.resize(frame, (PROC_W, int(PROC_W * frame.shape[0] / frame.shape[1])))
        ht = self._empty_telemetry()
        ht["camera_mode"] = self._camera_mode
        ht["active_source"] = self._active_source
        ht["usb_available"] = self._usb_available
        ht["usb_device"] = self._usb_device
        ht["fps"] = round(self._fps, 1)
        ht["frame_size"] = [OUT_W, OUT_H]

        if synth_meta is not None:
            pose = self._scale_synth_pose(synth_meta)
            ht["detected"] = bool(pose.get("detected"))
            ht["tracking_locked"] = ht["detected"]
            ht["pose_locked_by_occlusion"] = bool(pose.get("pose_locked_by_occlusion"))
            for key in (
                "p_start",
                "p_end",
                "major_px",
                "minor_px",
                "angle_deg",
                "px_per_mm",
                "estimated_length_mm",
                "estimated_diameter_mm",
                "hand_present",
                "tool_present",
                "hand_box",
                "tool_box",
                "tool_tip",
                "pcb_open_detected",
                "anomaly_detected",
                "anomaly_zone",
            ):
                if key in pose:
                    ht[key] = pose[key]
            if pose.get("hand_box"):
                ht["hand_boxes"] = [pose["hand_box"]]
            if pose.get("tool_box"):
                ht["tool_boxes"] = [pose["tool_box"]]
            if pose.get("dip_bits_optical"):
                ht["dip_bits_optical"] = list(pose["dip_bits_optical"])
                ht["dip_slave_optical"] = dip_bits_to_address(ht["dip_bits_optical"])
            if ht["hand_present"] or ht["tool_present"]:
                if self._locked_pose is None:
                    self._locked_pose = {
                        "p_start": ht["p_start"],
                        "p_end": ht["p_end"],
                        "major_px": ht["major_px"],
                        "minor_px": ht["minor_px"],
                        "angle_deg": ht["angle_deg"],
                        "estimated_length_mm": ht["estimated_length_mm"],
                    }
                else:
                    ht["p_start"] = self._locked_pose["p_start"]
                    ht["p_end"] = self._locked_pose["p_end"]
                    ht["major_px"] = self._locked_pose["major_px"]
                    ht["minor_px"] = self._locked_pose["minor_px"]
                    ht["angle_deg"] = self._locked_pose["angle_deg"]
                    ht["estimated_length_mm"] = self._locked_pose["estimated_length_mm"]
                    ht["pose_locked_by_occlusion"] = True
            else:
                self._locked_pose = None
        else:
            masks = self._detect_hand_tool_masks(proc)
            det = self._detect_cylinder(proc, masks.get("exclusion_mask"))
            sx = frame.shape[1] / proc.shape[1]
            sy = frame.shape[0] / proc.shape[0]
            if det:
                ht["detected"] = True
                ht["tracking_locked"] = True
                for k in ("p_start", "p_end"):
                    det[k] = [det[k][0] * sx, det[k][1] * sy]
                det["major_px"] *= sx
                det["minor_px"] *= sx
                ht.update(det)
            ht["hand_present"] = masks["hand_present"]
            ht["tool_present"] = masks["tool_present"]
            ht["hand_boxes"] = masks["hand_boxes"]
            ht["tool_boxes"] = masks["tool_boxes"]

        if ht["p_start"] and ht["p_end"]:
            cx = (ht["p_start"][0] + ht["p_end"][0]) / 2.0
            cy = (ht["p_start"][1] + ht["p_end"][1]) / 2.0
            ht["center_norm"] = [round(cx / OUT_W, 3), round(cy / OUT_H, 3)]
            mx = ht["major_px"]
            my = ht["minor_px"]
            ht["bbox"] = [
                int(cx - mx / 2),
                int(cy - my / 2),
                int(mx),
                int(my),
            ]
            ht["zones_2d"] = [
                _project_zone_pixels(z, ht["p_start"], ht["p_end"], ht["major_px"])
                for z in SENSOR_ZONES
            ]

        if ht.get("anomaly_zone"):
            zid = ht["anomaly_zone"]
            for z in SENSOR_ZONES:
                if z["id"] == zid:
                    ht["anomaly_zone_label"] = z["label"]
                    ht["anomaly_score"] = 0.92
                    ht["anomaly_z_mm"] = (z["z_range_mm"][0] + z["z_range_mm"][1]) / 2.0
                    break

        dip_bits = ht.get("dip_bits_optical") or (
            self._dip_bits if ht.get("pcb_open_detected") else None
        )
        if ht.get("pcb_open_detected") and dip_bits:
            ht["dip_bits_optical"] = list(dip_bits)
            ht["dip_slave_optical"] = dip_bits_to_address(dip_bits)
            ht["dip_crop_available"] = True
            pose = {
                "p_start": ht["p_start"],
                "p_end": ht["p_end"],
                "angle_deg": ht["angle_deg"],
            }
            self._dip_crop_jpeg = self._build_dip_crop(frame, pose, dip_bits)
        else:
            ht["dip_crop_available"] = False
            self._dip_crop_jpeg = None

        if ht.get("tool_tip") and ht.get("p_start"):
            ht["tool_target_zone"] = "upper_barrel_dip"

        with self._lock:
            self._frame_bgr = frame
            self._telemetry = ht

    def _refresh_once(self) -> None:
        try:
            frame, synth_meta = self._acquire_frame()
            if frame is not None:
                self._update_from_frame(frame, synth_meta)
        except Exception:
            pass

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._refresh_once()
            time.sleep(1.0 / 30.0)


def main() -> None:
    scanner = VisionScanner()
    scanner.set_mode("sim_healthy")
    scanner.start()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        scanner.stop()


if __name__ == "__main__":
    main()