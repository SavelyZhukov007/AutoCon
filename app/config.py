# -*- coding: utf-8 -*-
"""Configuration, paths and persistent settings for AutoCon."""

from __future__ import annotations

import json
import logging
import os
import platform
import sys
import threading
from pathlib import Path

APP_NAME = "AutoCon"
_LOGGING_HOOKS_INSTALLED = False


def app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent.parent


def web_dir() -> Path:
    return app_root() / "web"


def user_data_dir() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home()))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    path = base / APP_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sub(name: str) -> Path:
    path = user_data_dir() / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def projects_dir() -> Path:
    return _sub("projects")


def runtime_dir() -> Path:
    return _sub("runtime")


def logs_dir() -> Path:
    return _sub("logs")


def models_dir() -> Path:
    return _sub("models")


def cache_dir() -> Path:
    return _sub("cache")


def exports_dir() -> Path:
    return _sub("exports")


def ollama_dir() -> Path:
    return _sub("ollama")


def ollama_models_dir() -> Path:
    path = ollama_dir() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def full_log_path() -> Path:
    path = user_data_dir() / "ful_log_app.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def cache_subdir(name: str) -> Path:
    path = cache_dir() / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def central_environment(base: dict | None = None) -> dict:
    env = dict(base or os.environ)
    env.update(
        {
            "OLLAMA_MODELS": str(ollama_models_dir()),
            "HF_HOME": str(cache_subdir("huggingface")),
            "HUGGINGFACE_HUB_CACHE": str(cache_subdir("huggingface") / "hub"),
            "TORCH_HOME": str(cache_subdir("torch")),
            "YOLO_CONFIG_DIR": str(cache_subdir("ultralytics")),
            "ULTRALYTICS_CONFIG_DIR": str(cache_subdir("ultralytics")),
            "MPLCONFIGDIR": str(cache_subdir("matplotlib")),
        }
    )
    if os.name != "nt":
        env.setdefault("XDG_CACHE_HOME", str(cache_dir()))
    return env


def configure_central_environment() -> None:
    os.environ.update(central_environment())


def bootstrap_logging() -> Path:
    global _LOGGING_HOOKS_INSTALLED
    configure_central_environment()
    path = full_log_path()
    root = logging.getLogger()
    if not any(
        isinstance(handler, logging.FileHandler)
        and Path(getattr(handler, "baseFilename", "")) == path
        for handler in root.handlers
    ):
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s [%(name)s] %(threadName)s: %(message)s"
            )
        )
        root.addHandler(handler)
        root.setLevel(logging.INFO)

    if not _LOGGING_HOOKS_INSTALLED:
        def excepthook(exc_type, exc, tb) -> None:
            logging.getLogger(APP_NAME).critical(
                "Uncaught exception", exc_info=(exc_type, exc, tb)
            )
            if getattr(sys, "__excepthook__", None):
                sys.__excepthook__(exc_type, exc, tb)

        sys.excepthook = excepthook

        if hasattr(threading, "excepthook"):
            old_thread_hook = threading.excepthook

            def thread_hook(args) -> None:
                logging.getLogger(APP_NAME).critical(
                    "Uncaught thread exception",
                    exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
                )
                old_thread_hook(args)

            threading.excepthook = thread_hook
        _LOGGING_HOOKS_INSTALLED = True
    return path


def log_event(message: str, *, level: int = logging.INFO, exc_info=None) -> None:
    try:
        bootstrap_logging()
        logging.getLogger(APP_NAME).log(level, message, exc_info=exc_info)
    except Exception:
        try:
            with full_log_path().open("a", encoding="utf-8") as file:
                file.write(message + "\n")
        except Exception:
            pass


def executable_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return app_root()


def runtime_archive_name() -> str:
    system = platform.system().lower() or "unknown"
    machine = platform.machine().lower().replace("amd64", "x86_64")
    return f"autocon-runtime-{system}-{machine}.zip"


def runtime_archive_path() -> Path:
    return executable_dir() / runtime_archive_name()


def runtime_python() -> Path:
    base = runtime_dir() / ".venv"
    if os.name == "nt":
        return base / "Scripts" / "python.exe"
    return base / "bin" / "python"


def _runtime_site_packages() -> list[Path]:
    base = runtime_dir() / ".venv"
    candidates = [base / "Lib" / "site-packages"]
    lib = base / "lib"
    if lib.exists():
        candidates.extend(lib.glob("python*/site-packages"))
    return candidates


def bootstrap_runtime_packages() -> Path:
    """Make packages installed by the first-run wizard importable."""
    configure_central_environment()
    candidates = _runtime_site_packages()
    for path in reversed(candidates):
        if path.exists():
            text = str(path)
            if text not in sys.path:
                sys.path.insert(0, text)

    if os.name == "nt":
        dll_dirs: list[Path] = []
        for site in candidates:
            dll_dirs.extend(
                [
                    site / "onnxruntime" / "capi",
                    site / "torch" / "lib",
                    site / "nvidia" / "cublas" / "bin",
                    site / "nvidia" / "cudnn" / "bin",
                ]
            )
        path_parts = []
        for path in dll_dirs:
            if path.exists():
                path_parts.append(str(path))
                if hasattr(os, "add_dll_directory"):
                    try:
                        os.add_dll_directory(str(path))
                    except OSError:
                        pass
        if path_parts:
            os.environ["PATH"] = os.pathsep.join(
                path_parts + [os.environ.get("PATH", "")]
            )
    return runtime_dir()


def current_build_id() -> str:
    meta = app_root() / "build-info.json"
    if meta.exists():
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
            return str(data.get("build_id") or "")
        except Exception:
            pass
    return "source"


DEVICE_PATH = user_data_dir() / "device.json"
SETTINGS_PATH = user_data_dir() / "settings.json"


DEFAULTS = {
    "region": "ru_cis",
    "first_run_done": False,
    "build_id": "",
    "device": "auto",
    "gpu_index": 0,
    "target_fps": 8,
    "imgsz": 960,
    "conf": 0.35,
    "iou": 0.65,
    "tracker": "bytetrack.yaml",
    "frame_stride": 1,
    "yolo_vehicle_model": "yolo11s.pt",
    "traffic_sign_model": "",
    "plate_model": "",
    "vehicle_dino_model": "",
    "ocr_enabled": True,
    "commentary_enabled": True,
    "voice_enabled": False,
    "commentary_interval_sec": 4,
    "ollama_host": "http://127.0.0.1:11434",
    "ollama_model": "",
    "default_model": "qwen2.5:3b",
    "vision_model": "qwen2.5vl:3b",
}


def load_settings() -> dict:
    data = dict(DEFAULTS)
    if SETTINGS_PATH.exists():
        try:
            data.update(json.loads(SETTINGS_PATH.read_text(encoding="utf-8")))
        except Exception:
            pass
    return data


def save_settings(patch: dict) -> dict:
    data = load_settings()
    data.update(patch or {})
    SETTINGS_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return data
