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


def clean_artifacts() -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/IM", f"{APP_NAME}.exe", "/T"],
            capture_output=True,
            text=True,
            **startup_kwargs(),
        )
    targets = [ROOT / "build", ROOT / "dist", BUILD_META_DIR]
    targets.extend(ROOT.glob("*.spec"))
    for target in targets:
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
            say(f"  removed dir {target}", C_DIM)
        elif target.exists():
            target.unlink()
            say(f"  removed file {target}", C_DIM)
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
    sub.add_parser("build").set_defaults(func=cmd_build)
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
