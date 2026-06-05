# -*- coding: utf-8 -*-
"""Helpers for subprocesses that must never open a terminal window."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterable, Optional


def startup_kwargs() -> dict:
    if os.name != "nt":
        return {}
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = 0
    return {
        "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0),
        "startupinfo": startupinfo,
    }


def run_hidden(
    cmd: Iterable[str | Path],
    *,
    timeout: Optional[int] = None,
    cwd: str | Path | None = None,
    env: Optional[dict] = None,
    check: bool = False,
) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        [str(part) for part in cmd],
        cwd=str(cwd) if cwd else None,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        **startup_kwargs(),
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            proc.stdout.strip() or f"Command failed with {proc.returncode}"
        )
    return proc


def popen_hidden(
    cmd: Iterable[str | Path],
    *,
    cwd: str | Path | None = None,
    env: Optional[dict] = None,
) -> subprocess.Popen:
    return subprocess.Popen(
        [str(part) for part in cmd],
        cwd=str(cwd) if cwd else None,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        **startup_kwargs(),
    )
