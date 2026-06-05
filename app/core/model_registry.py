# -*- coding: utf-8 -*-
"""Curated model pack registry and downloads."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Optional

from .. import config
from .hidden import run_hidden

PACKS = {
    "yolo11s": {
        "title": "YOLO11s транспорт/люди",
        "desc": "Официальная Ultralytics-модель COCO для транспорта, людей и базовых объектов.",
        "recommended": True,
        "settings_key": "yolo_vehicle_model",
        "target": "yolo11s.pt",
        "kind": "ultralytics",
    },
    "traffic_signs_100": {
        "title": "Дорожные знаки 100 классов",
        "desc": "Community YOLO-веса для стартового распознавания дорожных знаков; можно заменить своими.",
        "recommended": True,
        "settings_key": "traffic_sign_model",
        "target": "traffic-signs-100.pt",
        "kind": "hf",
        "repo": "RZhukotynskyi/sign-detection-yolov8s",
        "filenames": ["best.pt", "model.pt", "sign-detection-yolov8s.pt"],
    },
    "license_plate": {
        "title": "Детектор номерных знаков",
        "desc": "YOLO-веса для поиска номерных знаков перед OCR.",
        "recommended": True,
        "settings_key": "plate_model",
        "target": "license-plate-yolo11.pt",
        "kind": "hf",
        "repo": "morsetechlab/yolov11-license-plate-detection",
        "filenames": ["best.pt", "license_plate_detector.pt", "model.pt"],
    },
    "vehicle_dino": {
        "title": "VehicleDINO INT8 ONNX",
        "desc": "Опциональная модель для типа, марки/модели и re-id транспорта.",
        "recommended": False,
        "settings_key": "vehicle_dino_model",
        "target": "vehicledino-int8.onnx",
        "kind": "hf",
        "repo": "wms2537/VehicleDINO",
        "filenames": ["vehicle-dino-int8.onnx", "model_int8.onnx", "model.onnx"],
    },
}


def model_path(filename: str) -> Path:
    return config.models_dir() / filename


def list_packs(settings: Optional[dict] = None) -> list[dict]:
    settings = settings or config.load_settings()
    items = []
    for key, meta in PACKS.items():
        target = model_path(meta["target"])
        active = settings.get(meta["settings_key"]) or (
            meta["target"] if key == "yolo11s" else ""
        )
        items.append(
            {
                "key": key,
                "title": meta["title"],
                "desc": meta["desc"],
                "recommended": meta["recommended"],
                "target": str(target),
                "installed": target.exists() or (key == "yolo11s" and bool(active)),
                "active": active,
                "kind": meta["kind"],
            }
        )
    return items


def install_pack(
    key: str, runtime_python: Path, on_progress: Optional[Callable[[dict], None]] = None
) -> dict:
    if key not in PACKS:
        return {"ok": False, "error": "Unknown model pack"}
    meta = PACKS[key]
    target = model_path(meta["target"])
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return {"ok": True, "path": str(target), "already": True}

    def progress(value: float, text: str) -> None:
        if on_progress:
            on_progress({"progress": value, "text": text, "pack": key})

    try:
        if meta["kind"] == "ultralytics":
            progress(0.1, "Запуск Ultralytics download")
            script = "from ultralytics import YOLO\n" f"YOLO({meta['target']!r})\n"
            result = run_hidden([str(runtime_python), "-c", script], timeout=900)
            if result.returncode != 0:
                return {"ok": False, "error": result.stdout[-1000:]}
            cache = find_file(Path.home(), meta["target"])
            if cache and cache != target:
                shutil.copy2(cache, target)
            elif not target.exists():
                # Ultralytics can load by model name from cache; store the model name as active.
                progress(1.0, "Модель будет загружаться кэшем Ultralytics")
                return {"ok": True, "path": meta["target"], "virtual": True}
            progress(1.0, "Модель установлена")
            return {"ok": True, "path": str(target)}

        progress(0.1, "Загрузка через Hugging Face Hub")
        filenames = meta.get("filenames") or [meta["target"]]
        last_error = ""
        for filename in filenames:
            script = (
                "from huggingface_hub import hf_hub_download\n"
                "import shutil\n"
                f"p=hf_hub_download(repo_id={meta['repo']!r}, filename={filename!r})\n"
                f"shutil.copy2(p, {str(target)!r})\n"
            )
            result = run_hidden([str(runtime_python), "-c", script], timeout=1800)
            if result.returncode == 0 and target.exists():
                progress(1.0, "Модель установлена")
                return {"ok": True, "path": str(target), "filename": filename}
            last_error = result.stdout[-1000:]
        return {
            "ok": False,
            "error": last_error or "Не удалось найти файл модели в репозитории",
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def find_file(root: Path, name: str) -> Path | None:
    try:
        for item in root.rglob(name):
            if item.is_file():
                return item
    except Exception:
        return None
    return None
