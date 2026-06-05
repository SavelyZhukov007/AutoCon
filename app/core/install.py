# -*- coding: utf-8 -*-
"""First-run optional dependency installation."""

from __future__ import annotations

from typing import Callable, Optional

from . import runtime


def check() -> list[dict]:
    return runtime.check_features()


def install(
    keys: list[str],
    on_progress: Optional[Callable[[dict], None]] = None,
    gpu: bool = False,
) -> dict:
    selected = list(keys or [])
    if gpu and "gpu" not in selected:
        selected.append("gpu")
    return runtime.RuntimeInstaller(on_progress=on_progress).install(selected)
