# -*- coding: utf-8 -*-
"""Managed runtime installer for optional ML dependencies."""

from __future__ import annotations

import importlib
import json
import os
import platform
import queue
import shutil
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional
from urllib.request import urlretrieve

from .. import config
from .hidden import popen_hidden, run_hidden

PYTHON_VERSION = "3.10"
UV_INSTALL_PS1 = "https://astral.sh/uv/install.ps1"
UV_INSTALL_SH = "https://astral.sh/uv/install.sh"


FEATURES = {
    "vision": {
        "title": "YOLO11 и обработка видео",
        "desc": "Ultralytics YOLO11, OpenCV, NumPy и Pillow.",
        "packages": [
            "ultralytics>=8.3",
            "opencv-python>=4.9",
            "numpy>=1.24",
            "pillow>=10",
        ],
        "modules": ["ultralytics", "cv2", "numpy", "PIL"],
    },
    "tracking": {
        "title": "Трекинг объектов",
        "desc": "Зависимости для стабильного сопровождения объектов между кадрами.",
        "packages": ["lap>=0.5.12"],
        "modules": ["lap"],
    },
    "plates": {
        "title": "Номера и OCR",
        "desc": "fast-alpr / fast-plate-ocr для чтения номерных знаков.",
        "packages": ["fast-alpr>=0.1.3", "fast-plate-ocr>=0.3.0", "onnxruntime>=1.17"],
        "modules": ["fast_alpr", "fast_plate_ocr", "onnxruntime"],
    },
    "hf": {
        "title": "Загрузка моделей",
        "desc": "Hugging Face Hub для curated community weights.",
        "packages": ["huggingface-hub>=0.23"],
        "modules": ["huggingface_hub"],
    },
    "vehicle_dino": {
        "title": "Марки/модели авто",
        "desc": "ONNX runtime для VehicleDINO и похожих моделей.",
        "packages": ["onnxruntime>=1.17"],
        "modules": ["onnxruntime"],
    },
    "gpu": {
        "title": "CUDA ускорение",
        "desc": "Torch/ONNX runtime с GPU-провайдерами, если доступна NVIDIA GPU.",
        "packages": ["torch>=2.2", "torchvision>=0.17", "onnxruntime-gpu>=1.17"],
        "modules": ["torch", "onnxruntime"],
    },
    "voice": {
        "title": "Голосовые комментарии",
        "desc": "Опциональная локальная озвучка через системный TTS.",
        "packages": ["pyttsx3>=2.90"],
        "modules": ["pyttsx3"],
    },
}

ALL_FEATURE_KEYS = tuple(FEATURES.keys())


@dataclass
class RuntimeStats:
    start: float
    last_time: float
    last_size: int


class RuntimeInstaller:
    def __init__(self, on_progress: Optional[Callable[[dict], None]] = None) -> None:
        self.on_progress = on_progress
        self.runtime = config.runtime_dir()
        self.venv = self.runtime / ".venv"
        self.tools = self.runtime / "tools"
        self.log_path = (
            config.logs_dir() / f"runtime-install-{time.strftime('%Y%m%d-%H%M%S')}.log"
        )
        self.stats = RuntimeStats(time.time(), time.time(), self._runtime_size())

    def install(self, keys: list[str]) -> dict:
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.tools.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        ordered = packages_for(keys)
        if not ordered:
            self._progress(1.0, "done", "Нечего устанавливать")
            return self._result(True, [], [])

        archive = config.runtime_archive_path()
        if archive.exists():
            self._progress(0.03, "archive", f"Найден offline-cache {archive.name}")
            ok, error = self._restore_archive(archive)
            if ok:
                config.bootstrap_runtime_packages()
                health = health_check(keys)
                return self._result(
                    health["ok"],
                    ordered if health["ok"] else [],
                    health.get("failed", []),
                    health,
                )
            self._log(f"Archive restore failed: {error}")

        uv = self._ensure_uv()
        self._ensure_python(uv)
        self._ensure_venv(uv)

        installed, failed = [], []
        total = len(ordered)
        for index, package in enumerate(ordered):
            base = 0.22 + (index / max(1, total)) * 0.62
            self._progress(base, "install", f"Установка {package}", package=package)
            code, output = self._run(
                [
                    str(uv),
                    "pip",
                    "install",
                    "--python",
                    str(config.runtime_python()),
                    "--upgrade",
                    package,
                ],
                timeout=3600,
            )
            if code == 0:
                installed.append(package)
            else:
                failed.append({"package": package, "error": short_error(output)})

        config.bootstrap_runtime_packages()
        health = health_check(keys)
        if not health["ok"]:
            failed.extend(health["failed"])

        if not failed:
            self._progress(0.92, "archive", "Создание offline-cache runtime")
            try:
                self.create_archive(archive)
            except Exception as exc:
                self._log(f"Archive create failed: {exc}")

        ok = not failed
        self._progress(
            1.0, "done" if ok else "failed", "Готово" if ok else "Готово с ошибками"
        )
        return self._result(ok, installed, failed, health)

    def create_archive(self, path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        if tmp.exists():
            tmp.unlink()
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
            for item in self.runtime.rglob("*"):
                if item in (tmp, path) or not item.is_file():
                    continue
                zf.write(item, item.relative_to(self.runtime))
        tmp.replace(path)

    def _ensure_uv(self) -> Path:
        found = shutil.which("uv")
        if found:
            return Path(found)
        local = self.tools / ("uv.exe" if os.name == "nt" else "uv")
        if local.exists():
            return local
        self._progress(0.06, "uv", "Установка uv")
        if os.name == "nt":
            cmd = [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                f"$env:UV_INSTALL_DIR='{self.tools}'; irm {UV_INSTALL_PS1} | iex",
            ]
        else:
            cmd = ["sh", "-c", f"curl -LsSf {UV_INSTALL_SH} | sh"]
        code, output = self._run(
            cmd, timeout=300, extra_env={"UV_INSTALL_DIR": str(self.tools)}
        )
        if code != 0 or not local.exists():
            raise RuntimeError("Не удалось установить uv: " + short_error(output))
        return local

    def _ensure_python(self, uv: Path) -> None:
        self._progress(0.11, "python", f"Проверка Python {PYTHON_VERSION}")
        code, output = self._run(
            [str(uv), "python", "install", PYTHON_VERSION], timeout=1800
        )
        if code != 0:
            self._log("uv python install returned non-zero: " + short_error(output))

    def _ensure_venv(self, uv: Path) -> None:
        if config.runtime_python().exists():
            return
        self._progress(0.17, "venv", "Создание runtime-окружения")
        code, output = self._run(
            [str(uv), "venv", "--python", PYTHON_VERSION, str(self.venv)], timeout=900
        )
        if code != 0:
            raise RuntimeError("Не удалось создать runtime: " + short_error(output))

    def _restore_archive(self, archive: Path) -> tuple[bool, str]:
        try:
            if self.runtime.exists():
                for child in self.runtime.iterdir():
                    if child.name == "tools":
                        continue
                    if child.is_dir():
                        shutil.rmtree(child, ignore_errors=True)
                    else:
                        child.unlink(missing_ok=True)
            with zipfile.ZipFile(archive, "r") as zf:
                zf.extractall(self.runtime)
            self._progress(0.9, "archive", "Runtime восстановлен из offline-cache")
            return True, ""
        except Exception as exc:
            return False, str(exc)

    def _run(
        self, cmd: list[str], timeout: int, extra_env: Optional[dict] = None
    ) -> tuple[int, str]:
        self._log("$ " + " ".join(cmd))
        env = os.environ.copy()
        env.update(extra_env or {})
        proc = popen_hidden(cmd, env=env)
        lines: list[str] = []
        q: queue.Queue[str | None] = queue.Queue()

        def reader() -> None:
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    q.put(line)
            finally:
                q.put(None)

        import threading

        threading.Thread(target=reader, daemon=True).start()
        deadline = time.time() + timeout
        while True:
            if time.time() > deadline:
                proc.kill()
                lines.append("timeout")
                break
            try:
                line = q.get(timeout=0.2)
            except queue.Empty:
                if proc.poll() is not None:
                    break
                continue
            if line is None:
                if proc.poll() is not None:
                    break
                continue
            text = line.rstrip()
            lines.append(text)
            self._log(text)
            if len(lines) % 15 == 0:
                self._progress(None, "install", text[-180:])
        code = proc.wait(timeout=5)
        return code, "\n".join(lines)

    def _runtime_size(self) -> int:
        if not self.runtime.exists():
            return 0
        total = 0
        for item in self.runtime.rglob("*"):
            if item.is_file():
                try:
                    total += item.stat().st_size
                except OSError:
                    pass
        return total

    def _progress(
        self, progress: Optional[float], stage: str, text: str, **extra
    ) -> None:
        if not self.on_progress:
            return
        now = time.time()
        size = self._runtime_size()
        elapsed = now - self.stats.start
        payload = {
            "progress": progress,
            "stage": stage,
            "text": text,
            "elapsed": int(elapsed),
            "eta": estimate_eta(progress, elapsed),
            "runtime_bytes": size,
            "log_path": str(self.log_path),
        }
        payload.update(extra)
        self.on_progress(payload)

    def _log(self, text: str) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(text + "\n")
        config.log_event(text)

    def _result(
        self, ok: bool, installed: list, failed: list, health: Optional[dict] = None
    ) -> dict:
        return {
            "ok": ok,
            "installed": installed,
            "failed": failed,
            "health": health or health_check(),
            "target": str(self.runtime),
            "python": str(config.runtime_python()),
            "archive": str(config.runtime_archive_path()),
            "archive_exists": config.runtime_archive_path().exists(),
            "log_path": str(self.log_path),
        }


def packages_for(keys: Iterable[str]) -> list[str]:
    selected = list(keys or [])
    packages: list[str] = []
    for key in selected:
        if key in FEATURES:
            packages.extend(FEATURES[key]["packages"])
    if "gpu" in set(selected):
        packages = [pkg for pkg in packages if not pkg.startswith("onnxruntime>=")]
    out, seen = [], set()
    for package in packages:
        if package not in seen:
            seen.add(package)
            out.append(package)
    return out


def module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def health_check(keys: Optional[Iterable[str]] = None) -> dict:
    config.bootstrap_runtime_packages()
    key_set = set(keys or [])
    modules: list[str] = []
    if keys:
        for key in keys:
            modules.extend(FEATURES.get(key, {}).get("modules", []))
    else:
        modules = ["ultralytics", "cv2", "numpy"]
    failed = []
    for module in dict.fromkeys(modules):
        try:
            importlib.import_module(module)
        except Exception as exc:
            failed.append({"module": module, "error": str(exc)})
    cuda = False
    warnings = []
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
    except Exception as exc:
        if "gpu" in key_set and not any(item.get("module") == "torch" for item in failed):
            failed.append({"module": "torch", "error": str(exc)})
    if "gpu" in key_set and not cuda and not any(item.get("module") == "torch" for item in failed):
        warnings.append({"module": "torch", "warning": "CUDA is unavailable; AutoCon will use CPU until CUDA-ready torch is installed."})
    return {"ok": not failed, "failed": failed, "warnings": warnings, "cuda": cuda}


def check_features() -> list[dict]:
    config.bootstrap_runtime_packages()
    from . import device

    gpu_available = bool(device.detect().get("gpu_available"))
    out = []
    for key, meta in FEATURES.items():
        installed = all(module_available(module) for module in meta["modules"])
        if key == "gpu":
            installed = module_available("torch") and module_available("onnxruntime")
        out.append(
            {
                "key": key,
                "title": meta["title"],
                "desc": meta["desc"],
                "packages": meta["packages"],
                "installed": installed,
                "recommended": not installed
                and key in {"vision", "tracking", "plates", "hf"}
                or (key == "gpu" and gpu_available and not installed),
                "available": key != "gpu" or gpu_available,
            }
        )
    return out


def estimate_eta(progress: Optional[float], elapsed: float) -> int:
    if not progress or progress <= 0.02:
        return 0
    return int(max(0, elapsed * (1 - progress) / progress))


def short_error(text: str) -> str:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return lines[-1] if lines else "неизвестная ошибка"
