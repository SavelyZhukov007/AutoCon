#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AutoCon build helper."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent
APP_NAME = "AutoCon"
ENTRY = ROOT / "app" / "main.py"
WEB_DIR = ROOT / "web"
REQUIREMENTS = ROOT / "requirements.txt"
BUILD_META_DIR = ROOT / ".build-meta"
BUILD_INFO = BUILD_META_DIR / "build-info.json"

C_RESET = "\033[0m"
C_DIM = "\033[2m"
C_OK = "\033[92m"
C_WARN = "\033[93m"
C_HEAD = "\033[96m"


def say(text: str, color: str = C_RESET) -> None:
    sys.stdout.write(f"{color}{text}{C_RESET}\n")
    sys.stdout.flush()


def head(text: str) -> None:
    say("\n" + "=" * 70, C_DIM)
    say("  " + text, C_HEAD)
    say("=" * 70, C_DIM)


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


def run(cmd: list[str | Path], check: bool = True, hidden: bool = False) -> int:
    say("  $ " + " ".join(str(part) for part in cmd), C_DIM)
    kwargs = startup_kwargs() if hidden else {}
    proc = subprocess.run([str(part) for part in cmd], **kwargs)
    if check and proc.returncode != 0:
        raise SystemExit(f"Command failed with code {proc.returncode}")
    return proc.returncode


def which(name: str) -> str | None:
    return shutil.which(name)


def appdata_dir() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / APP_NAME


def clean_targets() -> list[Path]:
    targets = [ROOT / "build", ROOT / "dist", BUILD_META_DIR, appdata_dir()]
    targets.extend(ROOT.glob("autocon-runtime-*.zip"))
    return targets


def remove_clean_target(target: Path, allowed: set[Path]) -> None:
    resolved = target.resolve(strict=False)
    if resolved not in allowed:
        raise SystemExit(f"Refusing to remove unexpected path: {resolved}")
    if target.is_dir():
        shutil.rmtree(target, ignore_errors=True)
        say(f"  removed dir {target}", C_DIM)
    elif target.exists():
        target.unlink()
        say(f"  removed file {target}", C_DIM)


def clean_artifacts() -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/IM", f"{APP_NAME}.exe", "/T"],
            capture_output=True,
            text=True,
            **startup_kwargs(),
        )
    targets = clean_targets()
    targets.extend(ROOT.glob("*.spec"))
    allowed = {target.resolve(strict=False) for target in targets}
    for target in targets:
        remove_clean_target(target, allowed)
    for cache in ROOT.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)


def write_build_info() -> str:
    BUILD_META_DIR.mkdir(parents=True, exist_ok=True)
    build_id = time.strftime("%Y%m%d-%H%M%S")
    BUILD_INFO.write_text(
        json.dumps({"app": APP_NAME, "build_id": build_id}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return build_id


def _console_progress(prefix: str) -> Callable[[dict], None]:
    def progress(payload: dict) -> None:
        text = str(payload.get("text") or payload.get("stage") or "")
        pct = payload.get("progress")
        if isinstance(pct, (int, float)):
            say(f"  [{prefix}] {int(pct * 100):3d}% {text}", C_DIM)
        elif text:
            say(f"  [{prefix}] {text}", C_DIM)

    return progress


def _wait_for_ollama(host: str, timeout: int = 45) -> bool:
    import requests

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = requests.get(host.rstrip("/") + "/api/tags", timeout=2)
            if response.ok:
                return True
        except Exception:
            time.sleep(1)
    return False


def _start_central_ollama(config_module) -> bool:
    ollama = which("ollama")
    if not ollama:
        say("  ollama not found; skipping Ollama pull", C_WARN)
        return False
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/IM", "ollama.exe", "/T"],
            capture_output=True,
            text=True,
            **startup_kwargs(),
        )
    env = config_module.central_environment()
    env["OLLAMA_MODELS"] = str(config_module.ollama_models_dir())
    say(f"  OLLAMA_MODELS={env['OLLAMA_MODELS']}", C_DIM)
    subprocess.Popen(
        [ollama, "serve"],
        cwd=str(ROOT),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
        **startup_kwargs(),
    )
    return _wait_for_ollama("http://127.0.0.1:11434")


def cmd_recheck_runtime(build_id: str) -> None:
    head("Recheck AppData runtime, models and settings")
    from app import config
    from app.core import install, llm, model_registry, runtime

    config.bootstrap_logging()
    config.configure_central_environment()
    archive = config.runtime_archive_path()
    if archive.exists():
        archive.unlink()
        say(f"  removed old runtime archive {archive}", C_DIM)

    features = list(runtime.ALL_FEATURE_KEYS)
    say("  installing all runtime feature blocks", C_HEAD)
    runtime_res = install.install(
        features, on_progress=_console_progress("runtime"), gpu=True
    )
    if not runtime_res.get("ok"):
        raise SystemExit(
            "Runtime recheck failed: "
            + json.dumps(runtime_res.get("failed") or runtime_res, ensure_ascii=False)
        )
    if archive.exists():
        archive.unlink()
        say(f"  removed generated runtime archive {archive}", C_DIM)

    runtime_python = config.runtime_python()
    if not runtime_python.exists():
        raise SystemExit(f"Runtime Python was not created: {runtime_python}")

    settings_patch = {}
    say("  installing all curated CV model packs", C_HEAD)
    for key in model_registry.ALL_PACK_KEYS:
        res = model_registry.install_pack(
            key, runtime_python, on_progress=_console_progress(key)
        )
        if not res.get("ok"):
            raise SystemExit(
                f"Model pack {key} failed: "
                + json.dumps(res, ensure_ascii=False)
            )
        meta = model_registry.PACKS.get(key, {})
        settings_key = meta.get("settings_key")
        if settings_key and res.get("path"):
            settings_patch[settings_key] = str(Path(res["path"]))

    default_model = "qwen2.5:3b"
    vision_model = "qwen2.5vl:3b"
    migration = llm.migrate_legacy_store_to_central([default_model, vision_model])
    if migration.get("ok"):
        say(
            "  migrated existing Ollama models to "
            + str(config.ollama_models_dir()),
            C_DIM,
        )
    if not _start_central_ollama(config):
        raise SystemExit(
            "Ollama did not start with central OLLAMA_MODELS. "
            f"Target: {config.ollama_models_dir()}"
        )
    cli = llm.OllamaClient("http://127.0.0.1:11434", default_model=default_model)
    for model in (default_model, vision_model):
        say(f"  pulling Ollama model {model}", C_HEAD)
        res = cli.pull(
            model,
            on_progress=lambda pct, text, model=model: say(
                f"  [ollama:{model}] {int((pct or 0) * 100):3d}% {text}", C_DIM
            ),
        )
        if not res.get("ok"):
            raise SystemExit(
                f"Ollama model {model} failed: " + json.dumps(res, ensure_ascii=False)
            )

    settings_patch.update(
        {
            "first_run_done": True,
            "build_id": build_id,
            "default_model": default_model,
            "vision_model": vision_model,
        }
    )
    config.save_settings(settings_patch)
    say(f"  prepared: {config.user_data_dir()}", C_OK)
    say(f"  log     : {config.full_log_path()}", C_OK)


def cmd_install(_args=None) -> None:
    head("Install lightweight build dependencies")
    uv = which("uv")
    if uv:
        run([uv, "pip", "install", "--python", sys.executable, "-r", REQUIREMENTS])
    else:
        say("uv not found; falling back to pip", C_WARN)
        run([sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS])


def cmd_build(args) -> Path:
    head("Build AutoCon executable")
    clean_artifacts()
    build_id = write_build_info()
    say(f"  build-id: {build_id}", C_DIM)

    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        cmd_install(None)

    sep = ";" if os.name == "nt" else ":"
    icon = ROOT / "web" / "assets" / "icon.ico"
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--clean",
        "--name", APP_NAME,
        "--windowed",
        "--collect-all", "webview",
        "--collect-all", "qrcode",
        "--add-data", f"{WEB_DIR}{sep}web",
        "--add-data", f"{BUILD_INFO}{sep}.",
        "--hidden-import", "qrcode.image.svg",
    ]
    for hidden in (
        # Runtime ML packages are installed outside the exe, but when they are
        # imported by the frozen app they still need these stdlib modules in
        # PyInstaller's embedded Python archive.
        "colorsys",
        "pickletools",
        "lzma",
        "bz2",
        "sqlite3",
        "xml.etree.ElementTree",
    ):
        cmd += ["--hidden-import", hidden]
    if not args.onedir:
        cmd.append("--onefile")
    if icon.exists():
        cmd += ["--icon", icon]
    for mod in (
        "ultralytics", "cv2", "numpy", "PIL", "torch", "torchvision", "onnxruntime",
        "fast_alpr", "fast_plate_ocr", "huggingface_hub", "lap", "pyttsx3",
    ):
        cmd += ["--exclude-module", mod]
    cmd.append(ENTRY)
    run(cmd)
    exe = ROOT / "dist" / (APP_NAME + (".exe" if os.name == "nt" else ""))
    say(f"\nBuilt: {exe}", C_OK)
    if getattr(args, "recheck", False):
        cmd_recheck_runtime(build_id)
    return exe


def cmd_run(_args) -> None:
    head("Run AutoCon from source")
    run([sys.executable, ENTRY])


def cmd_clean(_args) -> None:
    head("Clean build artifacts")
    clean_artifacts()
    say("Done.", C_OK)


def cmd_doctor(_args) -> None:
    head("Doctor")
    say(f"Python : {sys.version.split()[0]} ({sys.executable})")
    say(f"uv     : {which('uv') or 'not found'}")
    say(f"ffmpeg : {which('ffmpeg') or 'not found'}")
    say(f"ollama : {which('ollama') or 'not found'}")
    say(f"gh     : {which('gh') or 'not found'}")
    try:
        from app.core import device

        info = device.detect(force=True)
        say("GPU    : " + (", ".join(g["name"] for g in info.get("gpus", [])) or "not found"))
        say("CUDA   : " + ("yes" if info.get("cuda_ready") else "no"))
    except Exception as exc:
        say(f"Device check failed: {exc}", C_WARN)


def cmd_force(args) -> None:
    head("--force: install, build, launch")
    cmd_install(None)
    exe = cmd_build(args)
    if exe.exists():
        say("Launching executable...", C_HEAD)
        subprocess.Popen([str(exe)], cwd=str(exe.parent), **startup_kwargs())
    else:
        raise SystemExit("Executable was not created")


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoCon build helper")
    parser.add_argument("--force", action="store_true", help="install via uv, build, then launch dist/AutoCon.exe")
    parser.add_argument("--onedir", action="store_true", help="build as folder instead of onefile")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("install").set_defaults(func=cmd_install)
    build_parser = sub.add_parser("build")
    build_parser.add_argument(
        "--recheck",
        action="store_true",
        help="clean, build, then fully prepare %%APPDATA%%\\AutoCon runtime/models/settings",
    )
    build_parser.set_defaults(func=cmd_build)
    sub.add_parser("run").set_defaults(func=cmd_run)
    sub.add_parser("clean").set_defaults(func=cmd_clean)
    sub.add_parser("doctor").set_defaults(func=cmd_doctor)
    args = parser.parse_args()
    if args.force:
        cmd_force(args)
        return
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
