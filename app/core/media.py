# -*- coding: utf-8 -*-
"""Media helpers based on ffprobe and OpenCV."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .hidden import run_hidden

VIDEO_TYPES = (
    "*.mp4;*.mkv;*.mov;*.avi;*.webm",
    "Видео (*.mp4;*.mkv;*.mov;*.avi;*.webm)",
)


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def probe_duration(path: str) -> float:
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        try:
            out = run_hidden(
                [ffprobe, "-v", "quiet", "-print_format", "json", "-show_format", path],
                timeout=30,
            ).stdout
            return float(json.loads(out)["format"]["duration"])
        except Exception:
            pass
    try:
        import cv2

        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()
        return frames / fps if fps else 0.0
    except Exception:
        return 0.0


def media_info(path: str) -> dict:
    data = {"duration": probe_duration(path), "width": 0, "height": 0, "fps": 0.0}
    try:
        import cv2

        cap = cv2.VideoCapture(path)
        data.update(
            {
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
                "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0),
            }
        )
        cap.release()
    except Exception:
        pass
    return data


def list_cameras(max_index: int = 8) -> list[dict]:
    cameras = []
    try:
        import cv2

        for index in range(max_index):
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            ok = cap.isOpened()
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            cap.release()
            if ok:
                cameras.append(
                    {
                        "index": index,
                        "name": f"Камера {index}",
                        "width": width,
                        "height": height,
                    }
                )
    except Exception:
        pass
    return cameras


def safe_title(path: str) -> str:
    return Path(path).stem or "video"
