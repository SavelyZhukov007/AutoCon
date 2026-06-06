# -*- coding: utf-8 -*-
"""Ollama client and prompts for road-scene commentary."""

from __future__ import annotations

import json
from typing import Callable, Optional

import requests


class OllamaClient:
    def __init__(
        self, host: str, model: str = "", default_model: str = "qwen2.5:3b"
    ) -> None:
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
            return {"ok": True, "model": name}
        except Exception as exc:
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
            response = requests.post(
                self.host + "/api/generate", json=payload, timeout=120
            )
            response.raise_for_status()
            return response.json().get("response", "")
        out = []
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
