# -*- coding: utf-8 -*-
"""Hardware discovery and device policy."""

from __future__ import annotations

import json
import multiprocessing
import platform
import shutil
from typing import Optional

from .. import config
from .hidden import run_hidden


def _probe_nvidia_smi() -> list[dict]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = run_hidden(
            [
                exe,
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            timeout=5,
        ).stdout
    except Exception:
        return []
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0])
            vram = round(float(parts[2]) / 1024, 1)
        except ValueError:
            continue
        gpus.append({"index": idx, "name": parts[1] or "NVIDIA GPU", "vram_gb": vram})
    return gpus


def _probe_torch() -> dict:
    info = {"torch": None, "cuda_ready": False, "gpus": []}
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda_ready"] = True
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                info["gpus"].append(
                    {
                        "index": index,
                        "name": props.name,
                        "vram_gb": round(props.total_memory / (1024**3), 1),
                    }
                )
    except Exception:
        pass
    return info


def detect(force: bool = False) -> dict:
    if not force and config.DEVICE_PATH.exists():
        try:
            return json.loads(config.DEVICE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass

    smi_gpus = _probe_nvidia_smi()
    torch_info = _probe_torch()
    merged = {g["index"]: g for g in smi_gpus}
    for gpu in torch_info["gpus"]:
        merged[gpu["index"]] = gpu

    info = {
        "os": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": multiprocessing.cpu_count(),
        "gpu_available": bool(merged),
        "cuda_ready": bool(torch_info["cuda_ready"]),
        "has_cuda": bool(torch_info["cuda_ready"]),
        "gpus": [merged[k] for k in sorted(merged)],
        "torch": torch_info["torch"],
    }
    try:
        config.DEVICE_PATH.write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass
    return info


def resolve_device(settings: Optional[dict] = None) -> str:
    settings = settings or config.load_settings()
    pref = str(settings.get("device", "auto"))
    dev = detect()
    if pref == "cpu":
        return "cpu"
    if pref.startswith("cuda"):
        return pref if dev.get("cuda_ready") else "cpu"
    if dev.get("cuda_ready"):
        return f"cuda:{int(settings.get('gpu_index') or 0)}"
    return "cpu"


def summary(settings: Optional[dict] = None) -> dict:
    settings = settings or config.load_settings()
    dev = detect()
    selected = resolve_device(settings)
    return {
        **dev,
        "selected_device": selected,
        "gpu_setup_needed": bool(
            dev.get("gpu_available") and not dev.get("cuda_ready")
        ),
    }
