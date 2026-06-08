# -*- coding: utf-8 -*-
"""Helpers for subprocesses that must never open a terminal window."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterable, Optional

from .. import config


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
    cmd_list = [str(part) for part in cmd]
    merged_env = config.central_environment(env)
    config.log_event("$ " + " ".join(cmd_list))
    proc = subprocess.run(
        cmd_list,
        cwd=str(cwd) if cwd else None,
        env=merged_env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        **startup_kwargs(),
    )
    if proc.stdout:
        config.log_event(proc.stdout.rstrip())
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
    cmd_list = [str(part) for part in cmd]
    config.log_event("$ " + " ".join(cmd_list))
    return subprocess.Popen(
        cmd_list,
        cwd=str(cwd) if cwd else None,
        env=config.central_environment(env),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        **startup_kwargs(),
    )
