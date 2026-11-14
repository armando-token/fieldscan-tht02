"""C01 macOS AVFoundation UVC worker: lossless raw BGR frames on a binary FSCN pipe."""

from __future__ import annotations

import argparse
import os
import platform
import re
import struct
import subprocess
import sys
import time
from typing import Any

import numpy as np

FRAME_MAGIC = b"FSCN"
HEADER_STRUCT = struct.Struct("<4sHHHI")

_SKIP_DEVICE_TOKENS = (
    "iphone",
    "ipad",
    "continuity",
    "capture screen",
    "desk view",
    "virtual",
)

_USB_RANK_TOKENS = (
    "usb",
    "uvc",
    "acer",
    "4k",
    "uhd",
    "webcam",
    "logitech",
    "elgato",
    "cam",
    "hd",
)


def _load_corefoundation() -> Any | None:
    if platform.system() != "Darwin":
        return None
    try:
        import ctypes

        cf = ctypes.CDLL(
            "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
        )
        cf.CFRunLoopRunInMode.argtypes = [
            ctypes.c_void_p,
            ctypes.c_double,
            ctypes.c_bool,
        ]
        cf.CFRunLoopRunInMode.restype = ctypes.c_int
        cf.kCFRunLoopDefaultMode = ctypes.c_void_p.in_dll(cf, "kCFRunLoopDefaultMode")
        return cf
    except Exception:
        return None


def _pump_runloop(cf: Any | None, seconds: float) -> None:
    if cf is None:
        time.sleep(seconds)
        return
    cf.CFRunLoopRunInMode(cf.kCFRunLoopDefaultMode, seconds, False)


def _list_avfoundation_devices() -> list[tuple[int, str, int]]:
    """Return (index, label, score) for avfoundation video devices."""
    devices: list[tuple[int, str, int]] = []
    label = ""
    try:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-f",
                "avfoundation",
                "-list_devices",
                "true",
                "-i",
                "",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        text = (proc.stderr or "") + (proc.stdout or "")
        for line in text.splitlines():
            m = re.search(r"\[(\d+)\]\s+(.+)", line)
            if m and "AVFoundation video devices" in text:
                idx = int(m.group(1))
                label = m.group(2).strip()
                lower = label.lower()
                if any(tok in lower for tok in _SKIP_DEVICE_TOKENS):
                    continue
                score = 10
                if any(tok in lower for tok in _USB_RANK_TOKENS):
                    score += 450
                elif "facetime" not in lower and "built-in" not in lower:
                    score += 100
                devices.append((idx, label, score))
    except Exception:
        pass

    try:
        proc = subprocess.run(
            ["/usr/sbin/system_profiler", "SPCameraDataType", "-json"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if proc.returncode == 0:
            import json

            data = json.loads(proc.stdout)
            for cam in data.get("SPCameraDataType", []):
                name = str(cam.get("_name", ""))
                lower = name.lower()
                if any(tok in lower for tok in _SKIP_DEVICE_TOKENS):
                    continue
                score = 10
                if any(tok in lower for tok in _USB_RANK_TOKENS):
                    score += 450
                elif "facetime" not in lower:
                    score += 100
                devices.append((0, name, score))
    except Exception:
        pass

    if not devices:
        devices.append((0, "avfoundation:0 (default)", 10))
    devices.sort(key=lambda t: -t[2])
    return devices


def _open_camera(width: int, height: int) -> tuple[Any, str]:
    import cv2

    ranked = _list_avfoundation_devices()
    last_err: str | None = None
    for idx, label, _ in ranked:
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            last_err = f"open_failed:{label}"
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1.0)
        src = f"avfoundation:{idx} ({label})"
        return cap, src
    raise RuntimeError(last_err or "no_camera")


def _write_frame(frame_pipe: Any, frame: np.ndarray, src_label: str) -> None:
    contig = np.ascontiguousarray(frame, dtype=np.uint8)
    h, w = contig.shape[:2]
    raw_bytes = contig.tobytes()
    src_bytes = src_label.encode("utf-8")[:512]
    hdr = HEADER_STRUCT.pack(FRAME_MAGIC, w, h, len(src_bytes), len(raw_bytes))
    frame_pipe.write(hdr)
    frame_pipe.write(src_bytes)
    frame_pipe.write(raw_bytes)
    frame_pipe.flush()


def run_worker(width: int, height: int) -> int:
    if platform.system() != "Darwin":
        print(
            "mac_uvc_worker supports Darwin only; use vision_scanner V4L2 on Linux.",
            file=sys.stderr,
        )
        return 1

    frame_fd = os.dup(1)
    os.dup2(2, 1)
    frame_pipe = os.fdopen(frame_fd, "wb", buffering=0)

    cf = _load_corefoundation()
    _pump_runloop(cf, 0.025)

    import cv2

    cap, src_label = _open_camera(width, height)
    _pump_runloop(cf, 0.025)

    try:
        while True:
            _pump_runloop(cf, 0.002)
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            _write_frame(frame_pipe, frame, src_label)
    except BrokenPipeError:
        return 0
    finally:
        cap.release()
        try:
            frame_pipe.close()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="macOS FSCN raw BGR UVC pipe worker")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args()
    sys.exit(run_worker(args.width, args.height))


if __name__ == "__main__":
    main()