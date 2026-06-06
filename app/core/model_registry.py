# -*- coding: utf-8 -*-
"""Curated model pack registry and downloads."""

from __future__ import annotations

import shutil
import json
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
        "desc": "Community YOLO-веса для стартового распознавания дорожных знаков; если Hub недоступен, импортируйте свои .pt/.onnx.",
        "recommended": True,
        "settings_key": "traffic_sign_model",
        "target": "traffic-signs-100.pt",
        "kind": "hf",
        "repo": "RZhukotynskyi/sign-detection-yolov8s",
        "filenames": ["sdv4.pt", "sdv3.pt", "sdv4.onnx"],
        "repos": [
            {
                "repo": "RZhukotynskyi/sign-detection-yolov8s",
                "filenames": ["sdv4.pt", "sdv3.pt", "sdv4.onnx"],
            },
            {
                "repo": "Phearith/Traffic_Sign_Detection_Using_YOLOv8",
                "filenames": ["best_yolov8m.pt"],
            },
            {
                "repo": "cvtechniques/JC-Traffic-Sign-Detection",
                "filenames": [
                    "trainv11/weights/best.pt",
                    "trainv26/weights/best.pt",
                    "trainv8/weights/best.pt",
                ],
            },
        ],
        "manual_import": True,
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
                "manual_import": bool(meta.get("manual_import")),
            }
        )
    return items


def candidate_repos(meta: dict) -> list[dict]:
    if meta.get("repos"):
        return list(meta["repos"])
    return [{"repo": meta.get("repo", ""), "filenames": meta.get("filenames") or [meta["target"]]}]


def pick_existing_hf_file(runtime_python: Path, repo: str, filenames: list[str]) -> dict:
    script = (
        "import json\n"
        "from huggingface_hub import HfApi\n"
        f"repo={repo!r}\n"
        f"wanted={filenames!r}\n"
        "files=set(HfApi().list_repo_files(repo_id=repo))\n"
        "match=next((name for name in wanted if name in files), '')\n"
        "print(json.dumps({'match': match, 'files': sorted(files)}))\n"
    )
    result = run_hidden([str(runtime_python), "-c", script], timeout=120)
    if result.returncode != 0:
        return {"ok": False, "error": result.stdout[-1000:]}
    try:
        data = json.loads(result.stdout.strip().splitlines()[-1])
    except Exception:
        return {"ok": False, "error": result.stdout[-1000:]}
    if not data.get("match"):
        visible = ", ".join(data.get("files", [])[:8])
        return {
            "ok": False,
            "error": f"В репозитории {repo} не найдены ожидаемые веса. Есть: {visible}",
        }
    return {"ok": True, "filename": data["match"]}


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

        progress(0.1, "Проверка файлов Hugging Face Hub")
        last_error = ""
        for repo_index, candidate in enumerate(candidate_repos(meta)):
            repo = candidate["repo"]
            filenames = candidate.get("filenames") or [meta["target"]]
            picked = pick_existing_hf_file(runtime_python, repo, filenames)
            if not picked.get("ok"):
                last_error = picked.get("error", "")
                continue
            filename = picked["filename"]
            progress(
                0.25 + repo_index * 0.1,
                f"Загрузка {repo}/{filename}",
            )
            script = (
                "from huggingface_hub import hf_hub_download\n"
                "import shutil\n"
                f"p=hf_hub_download(repo_id={repo!r}, filename={filename!r})\n"
                f"shutil.copy2(p, {str(target)!r})\n"
            )
            result = run_hidden([str(runtime_python), "-c", script], timeout=1800)
            if result.returncode == 0 and target.exists():
                progress(1.0, "Модель установлена")
                return {"ok": True, "path": str(target), "repo": repo, "filename": filename}
            last_error = result.stdout[-1000:]
        return {
            "ok": False,
            "error": (
                last_error
                or "Не удалось найти файл модели. Можно импортировать свои .pt/.onnx в настройках."
            ),
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
