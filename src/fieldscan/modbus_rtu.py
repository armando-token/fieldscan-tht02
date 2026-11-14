"""C05 RS-485 Modbus-RTU master for TZ THT-02 + Slave-2 Andon coil driver."""

from __future__ import annotations

import glob
import os
import platform
import struct
import threading
import time
from collections import deque
from typing import Any

try:
    import serial  # type: ignore
    from serial.tools import list_ports  # type: ignore
except Exception:  # pragma: no cover
    serial = None  # type: ignore
    list_ports = None  # type: ignore

DIP_WEIGHTS = (128, 64, 32, 16, 8, 4, 2, 1)
BAUD_MAP = {0x12C0: 4800, 0x2580: 9600, 0x4B00: 19200}
ANDON_COILS = {"RED": 0x0000, "YELLOW": 0x0001, "GREEN": 0x0002}


def address_to_dip_bits(slave_id: int) -> list[int]:
    """Manual Page 5 MSB->LSB: Pin 1=128 .. Pin 8=1."""
    clamped = max(0, min(255, int(slave_id)))
    return [(clamped >> (7 - i)) & 1 for i in range(8)]


def dip_bits_to_address(bits: list[int]) -> int:
    return sum(w for b, w in zip((bits + [0] * 8)[:8], DIP_WEIGHTS) if int(bool(b)))


def describe_dip_pins(slave_id: int) -> dict[str, Any]:
    bits = address_to_dip_bits(slave_id)
    on_pins = [i + 1 for i, b in enumerate(bits) if b]
    off_pins = [i + 1 for i, b in enumerate(bits) if not b]
    return {
        "slave_id": int(slave_id),
        "dip_bits": bits,
        "on_pins": on_pins,
        "off_pins": off_pins,
        "weights": list(DIP_WEIGHTS),
        "power_cycle_required": True,
        "note": "MCU latches SW1 only at cold boot or when Reg 0x0004 == 0",
    }


def compute_modbus_crc16(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc >> 1) ^ 0xA001) if (crc & 0x0001) else (crc >> 1)
    return crc & 0xFFFF


def build_read_holding_request(slave_id: int = 1, start_reg: int = 0x0000, count: int = 8) -> bytes:
    body = struct.pack(">BBHH", slave_id & 0xFF, 0x03, start_reg & 0xFFFF, count & 0xFFFF)
    return body + struct.pack("<H", compute_modbus_crc16(body))


def build_write_single_coil_request(slave_id: int, coil_addr: int, turn_on: bool) -> bytes:
    val = 0xFF00 if turn_on else 0x0000
    body = struct.pack(">BBHH", slave_id & 0xFF, 0x05, coil_addr & 0xFFFF, val)
    return body + struct.pack("<H", compute_modbus_crc16(body))


def _u16_to_signed(u16: int) -> int:
    return u16 - 0x10000 if u16 >= 0x8000 else u16


def decode_tht02_holding_response(rx: bytes, expected_slave_id: int = 1) -> dict[str, Any]:
    """Decode exact 21-byte Function 0x03 response for 8 holding registers."""
    result: dict[str, Any] = {
        "ok": False,
        "healthy": False,
        "sensor_abnormal_7fff": False,
        "temp_c": None,
        "humidity_rh": None,
        "dew_point_c": None,
        "slave_id": expected_slave_id,
        "dip_bits": address_to_dip_bits(expected_slave_id),
        "baud_bps": 9600,
        "raw_regs": [],
        "error": None,
    }
    if len(rx) != 21:
        result["error"] = f"bad_length:{len(rx)}"
        return result
    if rx[0] != (expected_slave_id & 0xFF) or rx[1] != 0x03 or rx[2] != 16:
        result["error"] = "header_mismatch"
        return result
    if compute_modbus_crc16(rx[:19]) != struct.unpack("<H", rx[19:21])[0]:
        result["error"] = "crc_fail"
        return result
    regs = list(struct.unpack(">8H", rx[3:19]))
    result["raw_regs"] = regs
    raw_temp, raw_hum, raw_dew = regs[0], regs[1], regs[2]
    soft_slave = regs[4]
    baud_code = regs[5]
    if raw_temp == 0x7FFF or raw_hum == 0x7FFF:
        result["sensor_abnormal_7fff"] = True
        result["ok"] = True
        result["healthy"] = False
        result["error"] = "SENSOR_ABNORMAL_7FFF"
        return result
    temp_c = round(_u16_to_signed(raw_temp) / 10.0, 1)
    humidity_rh = round(raw_hum / 10.0, 1)
    dew = None
    if raw_dew not in (0, 0x7FFF):
        dew = round(_u16_to_signed(raw_dew) / 10.0, 1)
    slave = soft_slave if 1 <= soft_slave <= 247 else expected_slave_id
    result.update(
        {
            "ok": True,
            "temp_c": temp_c,
            "humidity_rh": humidity_rh,
            "dew_point_c": dew,
            "slave_id": slave,
            "dip_bits": address_to_dip_bits(slave),
            "baud_bps": BAUD_MAP.get(baud_code, 9600),
            "healthy": (-45.0 <= temp_c <= 130.0) and (0.0 <= humidity_rh <= 100.0),
        }
    )
    return result


def discover_rs485_ports() -> list[str]:
    """Prefer FTDI FT2232HL/FT232R Channel A; rewrite macOS tty->cu (non-blocking)."""
    ports: list[str] = []
    if list_ports is not None:
        for p in list_ports.comports():
            dev = p.device or ""
            desc = f"{p.description} {p.manufacturer} {p.hwid}".lower()
            if any(tok in desc for tok in ("bluetooth", "debug", "console")):
                continue
            if platform.system() == "Darwin" and dev.startswith("/dev/tty."):
                dev = "/dev/cu." + dev[len("/dev/tty.") :]
            score = 0
            if getattr(p, "vid", None) == 0x0403:
                score += 100
            if any(tok in desc for tok in ("ftdi", "ft2232", "ft232", "usbserial", "waveshare")):
                score += 50
            if dev.endswith("0") or "USB0" in dev or "usbserial" in dev:
                score += 10
            ports.append((score, dev))
        ports_sorted = [d for s, d in sorted(ports, key=lambda x: -x[0]) if d]
        if ports_sorted:
            return ports_sorted
    # Fallback glob
    patterns = ["/dev/cu.usbserial*", "/dev/ttyUSB*", "/dev/ttyACM*"]
    found: list[str] = []
    for pat in patterns:
        for dev in sorted(glob.glob(pat)):
            if platform.system() == "Darwin" and dev.startswith("/dev/tty."):
                dev = "/dev/cu." + dev[len("/dev/tty.") :]
            found.append(dev)
    return found


def _pack_sim_frame(slave_id: int, temp_c: float, humidity_rh: float, abnormal: bool = False) -> bytes:
    if abnormal:
        regs = [0x7FFF, 0x7FFF, 0, 0, slave_id, 0x2580, 0, 0]
    else:
        temp_u = int(round(temp_c * 10)) & 0xFFFF
        hum_u = int(round(humidity_rh * 10)) & 0xFFFF
        regs = [temp_u, hum_u, 0, 0, slave_id, 0x2580, 0, 0]
    body = bytes([slave_id & 0xFF, 0x03, 16]) + struct.pack(">8H", *regs)
    return body + struct.pack("<H", compute_modbus_crc16(body))


class ModbusRTUPoller:
    """Half-duplex Modbus-RTU master with 3-cycle miss debounce and Andon coils."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.mode = "auto"
        self.slave_id = 1
        self.baud_bps = 9600
        self.port: str | None = None
        self._ser = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._hw_miss_streak = 0
        self._miss_cycles = 0
        self._last_hw_success: dict[str, Any] | None = None
        self._andon_state = "RED"
        self._rolling: deque[dict[str, Any]] = deque(maxlen=120)
        self._telemetry: dict[str, Any] = self._idle_telemetry("BOOT")
        self._sim_t0 = time.time()

    def _idle_telemetry(self, reason: str) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "online": False,
            "healthy": False,
            "sensor_abnormal_7fff": False,
            "temp_c": None,
            "humidity_rh": None,
            "dew_point_c": None,
            "slave_id": self.slave_id,
            "dip_bits": address_to_dip_bits(self.slave_id),
            "baud_bps": self.baud_bps,
            "port": self.port,
            "andon_state": self._andon_state,
            "debounce_hold": False,
            "last_tx_hex": "",
            "last_rx_hex": "",
            "last_error": reason,
            "rolling_history": list(self._rolling),
            "usb_adapter_present": bool(discover_rs485_ports()),
        }

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="modbus-rtu", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._close_serial()

    def set_mode(
        self,
        mode: str,
        slave_id: int | None = None,
        baud_bps: int | None = None,
        port: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            self.mode = mode
            if slave_id is not None:
                self.slave_id = int(slave_id)
            if baud_bps is not None:
                self.baud_bps = int(baud_bps)
            if port is not None:
                self.port = port
                self._close_serial()
            self._hw_miss_streak = 0
            self._miss_cycles = 0
        self._poll_once()
        return self.get_telemetry()

    def get_telemetry(self) -> dict[str, Any]:
        with self._lock:
            tel = dict(self._telemetry)
            tel["rolling_history"] = list(self._rolling)
            return tel

    def drive_andon_relay(self, state: str) -> bytes | None:
        state = state.upper()
        with self._lock:
            self._andon_state = state
            if self._ser is None and self.mode in ("hardware", "auto"):
                # Still build frame for unit tests / offline queue
                pass
            frames = []
            for color, coil in ANDON_COILS.items():
                frames.append(build_write_single_coil_request(2, coil, color == state))
            if self._ser is not None:
                for fr in frames:
                    try:
                        self._ser.reset_input_buffer()
                        self._ser.write(fr)
                        self._ser.flush()
                        time.sleep(0.03)
                        self._ser.read(8)
                    except Exception:
                        pass
            return frames[list(ANDON_COILS).index(state)] if state in ANDON_COILS else frames[-1]

    def _close_serial(self) -> None:
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._poll_once()
            except Exception as exc:  # pragma: no cover
                with self._lock:
                    self._telemetry["last_error"] = str(exc)
            self._stop.wait(0.50)

    def _poll_once(self) -> None:
        with self._lock:
            mode = self.mode
        if mode == "sim_healthy":
            self._apply_sim(healthy=True)
        elif mode == "sim_offline_open":
            self._apply_offline("NO_RESPONSE_TIMEOUT")
        elif mode == "sim_7fff_fault":
            self._apply_sim(healthy=False, abnormal=True)
        elif mode == "hardware":
            self._poll_hardware(strict=True)
        else:  # auto
            ports = discover_rs485_ports()
            if ports:
                self._poll_hardware(strict=False)
            else:
                self._apply_offline("NO_ADAPTER")

    def _apply_sim(self, healthy: bool, abnormal: bool = False) -> None:
        t = time.time() - self._sim_t0
        temp = 24.6 + 0.35 * __import__("math").sin(t / 7.0)
        hum = 56.2 + 1.1 * __import__("math").sin(t / 11.0)
        slave = self.slave_id
        frame = _pack_sim_frame(slave, temp, hum, abnormal=abnormal)
        decoded = decode_tht02_holding_response(frame, slave)
        tx = build_read_holding_request(slave)
        with self._lock:
            self._andon_state = "GREEN" if decoded.get("healthy") else "RED"
            sample = {
                "t": time.time(),
                "temp_c": decoded.get("temp_c"),
                "humidity_rh": decoded.get("humidity_rh"),
            }
            if decoded.get("temp_c") is not None:
                self._rolling.append(sample)
            self._telemetry = {
                "mode": self.mode,
                "online": True,
                "healthy": bool(decoded.get("healthy")),
                "sensor_abnormal_7fff": bool(decoded.get("sensor_abnormal_7fff")),
                "temp_c": decoded.get("temp_c"),
                "humidity_rh": decoded.get("humidity_rh"),
                "dew_point_c": decoded.get("dew_point_c"),
                "slave_id": decoded.get("slave_id", slave),
                "dip_bits": decoded.get("dip_bits", address_to_dip_bits(slave)),
                "baud_bps": decoded.get("baud_bps", 9600),
                "port": "sim",
                "andon_state": self._andon_state,
                "debounce_hold": False,
                "last_tx_hex": tx.hex(" ").upper(),
                "last_rx_hex": frame.hex(" ").upper(),
                "last_error": decoded.get("error"),
                "rolling_history": list(self._rolling),
                "usb_adapter_present": bool(discover_rs485_ports()),
            }
            self._last_hw_success = dict(self._telemetry)

    def _apply_offline(self, reason: str) -> None:
        with self._lock:
            if self._last_hw_success and self._hw_miss_streak < 3 and self._last_hw_success.get("healthy"):
                tel = dict(self._last_hw_success)
                tel["debounce_hold"] = True
                tel["andon_state"] = "GREEN"
                tel["last_error"] = f"debounce_hold:{reason}"
                self._telemetry = tel
                self._hw_miss_streak += 1
                self._miss_cycles += 1
                return
            self._andon_state = "RED"
            self._telemetry = self._idle_telemetry(reason)
            self._telemetry["andon_state"] = "RED"
            self._hw_miss_streak += 1
            self._miss_cycles += 1

    def _open_serial(self) -> bool:
        if serial is None:
            return False
        ports = [self.port] if self.port else discover_rs485_ports()
        ports = [p for p in ports if p]
        for port in ports:
            try:
                self._ser = serial.Serial(
                    port,
                    baudrate=self.baud_bps,
                    bytesize=8,
                    parity="N",
                    stopbits=1,
                    timeout=0.22,
                    write_timeout=0.22,
                )
                self.port = port
                return True
            except Exception:
                self._ser = None
        return False

    def _poll_hardware(self, strict: bool) -> None:
        if self._ser is None and not self._open_serial():
            if strict:
                self._apply_offline("PORT_OPEN_FAIL")
            else:
                self._apply_offline("NO_ADAPTER")
            return
        assert self._ser is not None
        candidates = [self.slave_id, 1, 2, 3]
        # Deduplicate preserving order
        seen: set[int] = set()
        scan = []
        for s in candidates:
            if s not in seen:
                seen.add(s)
                scan.append(s)
        if self._miss_cycles < 6:
            scan = [self.slave_id]

        for sid in scan:
            tx = build_read_holding_request(sid)
            rx = self._transact(tx)
            if rx is None or len(rx) < 21:
                time.sleep(0.025)
                rx = self._transact(tx)
            if rx and len(rx) >= 21:
                decoded = decode_tht02_holding_response(rx[:21], sid)
                if decoded.get("ok"):
                    with self._lock:
                        self.slave_id = sid
                        self._hw_miss_streak = 0
                        self._miss_cycles = 0
                        self._andon_state = "GREEN" if decoded.get("healthy") else "RED"
                        if decoded.get("temp_c") is not None:
                            self._rolling.append(
                                {
                                    "t": time.time(),
                                    "temp_c": decoded.get("temp_c"),
                                    "humidity_rh": decoded.get("humidity_rh"),
                                }
                            )
                        self._telemetry = {
                            "mode": self.mode,
                            "online": True,
                            "healthy": bool(decoded.get("healthy")),
                            "sensor_abnormal_7fff": bool(decoded.get("sensor_abnormal_7fff")),
                            "temp_c": decoded.get("temp_c"),
                            "humidity_rh": decoded.get("humidity_rh"),
                            "dew_point_c": decoded.get("dew_point_c"),
                            "slave_id": decoded.get("slave_id", sid),
                            "dip_bits": decoded.get("dip_bits"),
                            "baud_bps": decoded.get("baud_bps", self.baud_bps),
                            "port": self.port,
                            "andon_state": self._andon_state,
                            "debounce_hold": False,
                            "last_tx_hex": tx.hex(" ").upper(),
                            "last_rx_hex": rx[:21].hex(" ").upper(),
                            "last_error": decoded.get("error"),
                            "rolling_history": list(self._rolling),
                            "usb_adapter_present": True,
                        }
                        self._last_hw_success = dict(self._telemetry)
                    return
        self._apply_offline("NO_RESPONSE_TIMEOUT")
        if self._miss_cycles >= 6:
            self._miss_cycles = 0  # trigger scan next cycles

    def _transact(self, tx: bytes) -> bytes | None:
        if self._ser is None:
            return None
        try:
            self._ser.reset_input_buffer()
            self._ser.reset_output_buffer()
            self._ser.write(tx)
            self._ser.flush()
            time.sleep(0.035)
            return self._ser.read(21)
        except Exception:
            return None


__all__ = [
    "address_to_dip_bits",
    "dip_bits_to_address",
    "describe_dip_pins",
    "compute_modbus_crc16",
    "build_read_holding_request",
    "build_write_single_coil_request",
    "decode_tht02_holding_response",
    "discover_rs485_ports",
    "ModbusRTUPoller",
]
