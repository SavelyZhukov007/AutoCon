# -*- coding: utf-8 -*-
"""Single-instance guard for the compiled desktop app."""

from __future__ import annotations

import ctypes
import os
import socket
from dataclasses import dataclass

LOCK_PORT = 41637
ERROR_ALREADY_EXISTS = 183


@dataclass
class InstanceGuard:
    socket: socket.socket | None = None
    mutex: int | None = None

    def release(self) -> None:
        if self.socket:
            try:
                self.socket.close()
            except OSError:
                pass
            self.socket = None
        if self.mutex and os.name == "nt":
            try:
                ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(self.mutex)
            except Exception:
                pass
            self.mutex = None


def ensure_single_instance(app_name: str = "AutoCon") -> InstanceGuard:
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p]
        kernel32.CreateMutexW.restype = ctypes.c_void_p
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle.restype = ctypes.c_bool
        handle = kernel32.CreateMutexW(None, False, f"Global\\{app_name}-single-instance")
        if not handle:
            raise SystemExit(0)
        if ctypes.get_last_error() == ERROR_ALREADY_EXISTS:
            kernel32.CloseHandle(handle)
            raise SystemExit(0)
        return InstanceGuard(mutex=handle)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", LOCK_PORT))
        sock.listen(1)
        return InstanceGuard(sock)
    except OSError:
        # Quiet exit: the requirement is to prevent a second app and avoid popups.
        raise SystemExit(0)
