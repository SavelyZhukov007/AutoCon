# -*- coding: utf-8 -*-
"""Report exporters."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .. import config


def ts(seconds: float) -> str:
    seconds = max(0, float(seconds or 0))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def to_markdown(project: dict) -> str:
    lines = [f"# AutoCon report: {project.get('title', 'video')}", ""]
    if project.get("summary"):
        lines += ["## Краткая сводка", "", project["summary"], ""]
    lines += ["## Последовательности знаков", ""]
    if not project.get("sign_sequences"):
        lines.append("_Нет распознанных последовательностей._")
    for seq in project.get("sign_sequences", []):
        lines.append(
            f"- `{ts(seq['start'])}-{ts(seq['end'])}` {seq['label']} x{seq.get('count', 1)} · {seq.get('position', '')}"
        )
    lines += ["", "## Авто и номера", ""]
    if not project.get("vehicles") and not project.get("plates"):
        lines.append("_Нет данных._")
    for vehicle in project.get("vehicles", []):
        lines.append(
            f"- `{ts(vehicle.get('first_t', 0))}` {vehicle.get('label', 'vehicle')} · id={vehicle.get('track_id', '-')}"
        )
    for plate in project.get("plates", []):
        lines.append(
            f"- `{ts(plate.get('first_t', 0))}` номер `{plate.get('text', '')}` · conf={plate.get('confidence', 0):.2f}"
        )
    lines += ["", "## Комментарии", ""]
    for item in project.get("comments", []):
        lines.append(f"- `{ts(item.get('t', 0))}` {item.get('text', '')}")
    return "\n".join(lines)


def export_project(project: dict, fmt: str = "md") -> str:
    fmt = (fmt or "md").lower()
    out_dir = config.exports_dir() / project["id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        path = out_dir / "autocon-report.json"
        path.write_text(
            json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    elif fmt == "csv":
        path = out_dir / "autocon-events.csv"
        with path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(
                file, fieldnames=["kind", "time", "label", "text", "confidence"]
            )
            writer.writeheader()
            for seq in project.get("sign_sequences", []):
                writer.writerow(
                    {
                        "kind": "sign",
                        "time": ts(seq["start"]),
                        "label": seq["label"],
                        "text": "",
                        "confidence": seq.get("confidence", ""),
                    }
                )
            for plate in project.get("plates", []):
                writer.writerow(
                    {
                        "kind": "plate",
                        "time": ts(plate.get("first_t", 0)),
                        "label": "plate",
                        "text": plate.get("text", ""),
                        "confidence": plate.get("confidence", ""),
                    }
                )
    else:
        path = out_dir / "autocon-report.md"
        path.write_text(to_markdown(project), encoding="utf-8")
    project.setdefault("exports", []).append(str(path))
    return str(path)
