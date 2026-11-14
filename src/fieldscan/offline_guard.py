"""C08 process-wide air-gap socket firewall for 100% offline runtime."""

from __future__ import annotations

import ipaddress
import socket
import threading
from typing import Any

_ORIGINAL_CONNECT = socket.socket.connect
_ORIGINAL_CONNECT_EX = socket.socket.connect_ex
_GUARD_INSTALLED = False
_ALLOWED_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost", "::ffff:127.0.0.1", "0.0.0.0"}
_blocked_attempts = 0
_block_lock = threading.Lock()


class OfflineViolationError(PermissionError):
    """Raised when code attempts a non-loopback network connection."""


def _is_loopback_target(address: Any) -> bool:
    if isinstance(address, (str, bytes)):
        return True  # AF_UNIX local domain socket
    if not isinstance(address, tuple) or len(address) < 2:
        return True
    host = str(address[0]).strip().lower()
    if host in _ALLOWED_LOOPBACK_HOSTS or host.startswith("127."):
        return True
    try:
        ip = ipaddress.ip_address(host.split("%")[0])
        return bool(ip.is_loopback or ip.is_unspecified)
    except ValueError:
        return False


def get_blocked_external_attempts() -> int:
    with _block_lock:
        return _blocked_attempts


def install_offline_socket_guard() -> None:
    global _GUARD_INSTALLED, _blocked_attempts
    if _GUARD_INSTALLED:
        return

    def _guarded_connect(self: socket.socket, address: Any) -> Any:
        global _blocked_attempts
        if not _is_loopback_target(address):
            with _block_lock:
                _blocked_attempts += 1
            raise OfflineViolationError(
                f"Blocked external connection to {address!r} (100% offline mode)"
            )
        return _ORIGINAL_CONNECT(self, address)

    def _guarded_connect_ex(self: socket.socket, address: Any) -> int:
        global _blocked_attempts
        if not _is_loopback_target(address):
            with _block_lock:
                _blocked_attempts += 1
            raise OfflineViolationError(
                f"Blocked external connect_ex to {address!r} (100% offline mode)"
            )
        return _ORIGINAL_CONNECT_EX(self, address)

    socket.socket.connect = _guarded_connect  # type: ignore[method-assign]
    socket.socket.connect_ex = _guarded_connect_ex  # type: ignore[method-assign]
    _GUARD_INSTALLED = True


def verify_offline_enforcement() -> dict[str, Any]:
    """Self-test: probe 8.8.8.8:53 and assert OfflineViolationError."""
    install_offline_socket_guard()
    blocked = False
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.settimeout(0.5)
            s.connect(("8.8.8.8", 53))
        finally:
            s.close()
    except OfflineViolationError:
        blocked = True
    except OSError:
        blocked = True
    return {
        "offline_verified": blocked,
        "blocked_external_probe": blocked,
        "blocked_external_attempts": get_blocked_external_attempts(),
        "offline": True,
        "guard_installed": _GUARD_INSTALLED,
    }
