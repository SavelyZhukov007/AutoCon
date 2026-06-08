# -*- coding: utf-8 -*-
"""Ollama client and prompts for road-scene commentary."""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Callable, Optional

import requests

from .. import config


LOG = logging.getLogger(__name__)


def legacy_models_dir() -> Path:
    return Path.home() / ".ollama" / "models"


def _manifest_path(root: Path, name: str) -> Path:
    model, _, tag = name.partition(":")
    tag = tag or "latest"
    return root / "manifests" / "registry.ollama.ai" / "library" / model / tag


def model_in_store(root: Path, name: str) -> bool:
    return bool(name and _manifest_path(root, name).exists())


def migrate_legacy_store_to_central(names: Optional[list[str]] = None) -> dict:
    legacy = legacy_models_dir()
    target = config.ollama_models_dir()
    names = list(dict.fromkeys(names or []))
    missing = [name for name in names if not model_in_store(target, name)]
    if names and not missing:
        return {"ok": True, "already": True, "target": str(target)}
    if not legacy.exists():
        return {
            "ok": False,
            "skipped": True,
            "error": "Legacy Ollama model store was not found.",
            "legacy": str(legacy),
            "target": str(target),
        }
    if names and not any(model_in_store(legacy, name) for name in missing):
        return {
            "ok": False,
            "skipped": True,
            "error": "Required models are not present in the legacy Ollama store.",
            "missing": missing,
            "legacy": str(legacy),
            "target": str(target),
        }

    target.mkdir(parents=True, exist_ok=True)
    linked = 0
    copied = 0
    skipped = 0
    bytes_total = 0
    for item in legacy.rglob("*"):
        dest = target / item.relative_to(legacy)
        if item.is_dir():
            dest.mkdir(parents=True, exist_ok=True)
            continue
        if not item.is_file():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        size = item.stat().st_size
        if dest.exists() and dest.stat().st_size == size:
            skipped += 1
            continue
        if dest.exists():
            dest.unlink()
        try:
            os.link(item, dest)
            linked += 1
        except OSError:
            shutil.copy2(item, dest)
            copied += 1
        bytes_total += size

    still_missing = [name for name in names if not model_in_store(target, name)]
    result = {
        "ok": not still_missing,
        "legacy": str(legacy),
        "target": str(target),
        "linked": linked,
        "copied": copied,
        "skipped": skipped,
        "bytes": bytes_total,
        "missing": still_missing,
    }
    config.log_event("Ollama legacy migration: " + json.dumps(result, ensure_ascii=False))
    return result


class OllamaClient:
    def __init__(
        self, host: str, model: str = "", default_model: str = "qwen2.5:3b"
    ) -> None:
        config.configure_central_environment()
        self.host = host.rstrip("/")
        self.model = model
        self.default_model = default_model

    def available(self) -> bool:
        try:
            requests.get(self.host + "/api/tags", timeout=2)
            return True
        except Exception:
            return False

    def list_models(self) -> list[str]:
        try:
            response = requests.get(self.host + "/api/tags", timeout=5)
            response.raise_for_status()
            return [item["name"] for item in response.json().get("models", [])]
        except Exception:
            return []

    def model_in_central_store(self, name: str) -> bool:
        return model_in_store(config.ollama_models_dir(), name)

    def central_store_status(self, names: list[str]) -> dict:
        missing = [name for name in names if not self.model_in_central_store(name)]
        legacy = legacy_models_dir()
        return {
            "ok": not missing,
            "missing": missing,
            "target": str(config.ollama_models_dir()),
            "legacy_exists": legacy.exists(),
            "legacy_path": str(legacy),
            "env": os.environ.get("OLLAMA_MODELS", ""),
        }

    def resolve_model(self) -> str:
        if self.model:
            return self.model
        models = self.list_models()
        qwen = [name for name in models if "qwen" in name.lower()]
        self.model = (qwen or models or [self.default_model])[0]
        return self.model

    def pull(
        self, name: str, on_progress: Optional[Callable[[float, str], None]] = None
    ) -> dict:
        try:
            migration = migrate_legacy_store_to_central([name])
            if self.model_in_central_store(name) and name in self.list_models():
                config.log_event(f"Ollama model ready from central store: {name}")
                return {"ok": True, "model": name, "already": True, "migration": migration}
            with requests.post(
                self.host + "/api/pull",
                json={"name": name, "stream": True},
                stream=True,
                timeout=3600,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line.decode("utf-8"))
                    total = chunk.get("total") or 0
                    completed = chunk.get("completed") or 0
                    progress = (completed / total) if total else 0.0
                    if on_progress:
                        on_progress(min(progress, 0.999), chunk.get("status", ""))
                    if chunk.get("error"):
                        return {"ok": False, "error": chunk["error"]}
            if on_progress:
                on_progress(1.0, "Модель загружена")
            if not self.model_in_central_store(name):
                status = self.central_store_status([name])
                message = (
                    "Ollama pulled the model, but it was not found in AutoCon's "
                    f"central store: {status['target']}. Restart Ollama with "
                    "OLLAMA_MODELS set to that directory, then run first setup again."
                )
                config.log_event(message, level=logging.ERROR)
                return {"ok": False, "error": message, "model": name, "central": status}
            config.log_event(f"Ollama model ready: {name}")
            return {"ok": True, "model": name}
        except Exception as exc:
            LOG.exception("Ollama pull failed for %s", name)
            return {"ok": False, "error": str(exc)}

    def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        on_token: Optional[Callable[[str], None]] = None,
        temperature: float = 0.25,
        model: str | None = None,
        images: Optional[list[str]] = None,
    ) -> str:
        payload = {
            "model": model or self.resolve_model(),
            "prompt": prompt,
            "system": system,
            "stream": bool(on_token),
            "options": {"temperature": temperature, "num_ctx": 8192},
        }
        if images:
            payload["images"] = images
        if not on_token:
            try:
                response = requests.post(
                    self.host + "/api/generate", json=payload, timeout=120
                )
                response.raise_for_status()
                return response.json().get("response", "")
            except Exception:
                LOG.exception("Ollama generate failed")
                raise
        out = []
        try:
            with requests.post(
                self.host + "/api/generate", json=payload, stream=True, timeout=180
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line.decode("utf-8"))
                    token = chunk.get("response", "")
                    if token:
                        out.append(token)
                        on_token(token)
                    if chunk.get("done"):
                        break
        except Exception:
            LOG.exception("Ollama streaming generate failed")
            raise
        return "".join(out)


SYS_AUTOCON = (
    "Ты локальный ассистент AutoCon для анализа дорожной обстановки. "
    "Отвечай по-русски, кратко, осторожно, без гарантий безопасности. "
    "Не выдумывай объекты: используй только события, которые пришли во входных данных."
)


def prompt_scene_commentary(snapshot: dict) -> str:
    return (
        "Сформулируй короткий комментарий для водителя/оператора по текущей сцене. "
        "Формат: 1-2 предложения, сначала важное. Если данных мало, скажи нейтрально. "
        "Нельзя утверждать юридические выводы или точные нарушения без уверенности.\n\n"
        f"События JSON:\n{json.dumps(snapshot, ensure_ascii=False, indent=2)}"
    )


def prompt_report_summary(project: dict) -> str:
    brief = {
        "title": project.get("title"),
        "sign_sequences": project.get("sign_sequences", [])[:80],
        "vehicles": project.get("vehicles", [])[:80],
        "plates": project.get("plates", [])[:80],
        "comments": project.get("comments", [])[:40],
    }
    return (
        "Сделай краткую сводку дорожной сцены по данным AutoCon: основные знаки, "
        "заметные транспортные объекты, номера и важные моменты. До 8 пунктов.\n\n"
        f"Данные JSON:\n{json.dumps(brief, ensure_ascii=False, indent=2)}"
    )


def prompt_video_chat(question: str, project: dict) -> str:
    context = {
        "title": project.get("title"),
        "duration": project.get("duration"),
        "summary": project.get("summary", ""),
        "sign_sequences": project.get("sign_sequences", [])[:120],
        "vehicles": project.get("vehicles", [])[:120],
        "plates": project.get("plates", [])[:80],
        "comments": project.get("comments", [])[:80],
    }
    return (
        "Ответь на вопрос пользователя по уже обработанному видео AutoCon. "
        "Опирайся только на JSON-контекст: знаки, последовательности, транспорт, номера, комментарии и сводку. "
        "Если данных недостаточно или модель могла ошибиться, прямо скажи, что нужно перепроверить кадры/веса/пороги. "
        "Отвечай по-русски, кратко, но полезно.\n\n"
        f"Вопрос: {question}\n\n"
        f"Контекст JSON:\n{json.dumps(context, ensure_ascii=False, indent=2)}"
    )


def prompt_sign_explanation(
    sign: dict, context: dict | None, project: dict | None = None
) -> str:
    payload = {
        "sign": sign,
        "context": context or {},
        "project": {
            "title": (project or {}).get("title", ""),
            "duration": (project or {}).get("duration", 0),
            "nearby_vehicles": (project or {}).get("vehicles", [])[-12:],
            "known_plates": (project or {}).get("plates", [])[-8:],
        },
    }
    return (
        "Explain in Russian how the selected road sign affects driver behavior "
        "in this exact road situation. Use only the provided JSON. Mention lane "
        "applicability if it is known. Keep the answer compact: 2-4 sentences, "
        "practical, cautious, no invented legal certainty.\n\n"
        f"JSON:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


PDD_RU_CONTEXT = {
    "version": "ПДД РФ, Постановление Правительства РФ N 1090, локальная краткая база",
    "principles": [
        "Водитель обязан соблюдать требования дорожных знаков, разметки и сигналов.",
        "Запрещающие и предписывающие знаки применяются к направлению или полосе, где они установлены, если таблички или разметка не уточняют иное.",
        "Знаки приоритета, светофоры, временные знаки и регулировщик имеют приоритет над обычной разметкой в типичных конфликтных ситуациях.",
        "При сомнении нельзя делать категоричный вывод: нужно указать, какие элементы кадра следует проверить.",
    ],
    "common_signs": {
        "stop": "движение без остановки запрещено",
        "give way": "уступить дорогу",
        "speed limit": "ограничение максимальной скорости",
        "no entry": "въезд запрещён",
        "no stopping": "остановка запрещена",
        "no parking": "стоянка запрещена",
        "pedestrian crossing": "пешеходный переход",
        "traffic light": "регулируемый участок",
        "lane direction": "направления движения по полосам",
    },
}


def prompt_exam_photo(question: str, findings: dict) -> str:
    context = {
        "question": question,
        "visual_findings": findings,
        "pdd_context": PDD_RU_CONTEXT,
    }
    return (
        "Проанализируй экзаменационную дорожную ситуацию по изображению и вопросу. "
        "Используй видимые объекты, знаки, разметку, светофоры и краткую локальную базу ПДД РФ. "
        "Если данных недостаточно, прямо укажи неопределённость. Не выдумывай элементы, которых нет в данных. "
        "Формат ответа: 1) короткий ответ; 2) почему; 3) какие знаки/разметка/правила повлияли; 4) уверенность.\n\n"
        f"Данные JSON:\n{json.dumps(context, ensure_ascii=False, indent=2)}"
    )
