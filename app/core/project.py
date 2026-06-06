# -*- coding: utf-8 -*-
"""Project JSON persistence."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from .. import config
from . import media

SCHEMA = {
    "duration": 0.0,
    "media": {},
    "settings": {},
    "tracks": [],
    "detections": [],
    "sign_sequences": [],
    "vehicles": [],
    "plates": [],
    "comments": [],
    "contexts": [],
    "exam_cases": [],
    "summary": "",
    "exports": [],
}


def new_project(video_path: str, settings: dict | None = None) -> dict:
    data = {
        "id": uuid.uuid4().hex[:12],
        "created": time.time(),
        "updated": time.time(),
        "source_path": video_path,
        "title": media.safe_title(video_path),
    }
    data.update(
        {k: (v.copy() if isinstance(v, (list, dict)) else v) for k, v in SCHEMA.items()}
    )
    data["duration"] = media.probe_duration(video_path)
    data["media"] = media.media_info(video_path)
    data["settings"] = dict(settings or {})
    return data


def _backfill(data: dict) -> dict:
    for key, value in SCHEMA.items():
        data.setdefault(key, value.copy() if isinstance(value, (list, dict)) else value)
    return data


def save(data: dict) -> str:
    data["updated"] = time.time()
    path = config.projects_dir() / f"{data['id']}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def load(project_id: str) -> dict:
    path = config.projects_dir() / f"{project_id}.json"
    return _backfill(json.loads(path.read_text(encoding="utf-8")))


def list_projects() -> list[dict]:
    items = []
    for path in config.projects_dir().glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        items.append(
            {
                "id": data.get("id"),
                "title": data.get("title", "video"),
                "updated": data.get("updated", 0),
                "duration": data.get("duration", 0),
                "source_path": data.get("source_path", ""),
                "events": len(data.get("sign_sequences", []))
                + len(data.get("plates", [])),
            }
        )
    return sorted(items, key=lambda x: -x["updated"])


def delete(project_id: str) -> bool:
    path = config.projects_dir() / f"{project_id}.json"
    if path.exists():
        path.unlink()
        return True
    return False
