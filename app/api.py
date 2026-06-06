# -*- coding: utf-8 -*-
"""pywebview API bridge used by the AutoCon UI."""

from __future__ import annotations

import json
import threading
import traceback
from pathlib import Path

try:
    import webview
except Exception:  # pragma: no cover - tests can run without pywebview
    webview = None

from . import config
from .core import device, export, install, llm, media, model_registry, project, server
from .core.vision import CameraWorker, EventAggregator, VisionEngine


def _json(data) -> str:
    return json.dumps(data, ensure_ascii=False)


class Api:
    def __init__(self) -> None:
        self.settings = config.load_settings()
        self.project: dict | None = None
        self._server = server.MediaServer()
        self._server.set_data_root(config.user_data_dir())
        self._llm: llm.OllamaClient | None = None
        self._camera: CameraWorker | None = None
        self._busy = False

    # ------------------------------------------------------------------ events
    def _window(self):
        if not webview or not getattr(webview, "windows", None):
            return None
        return webview.windows[0] if webview.windows else None

    def _emit(self, event: str, payload: dict) -> None:
        window = self._window()
        if not window:
            return
        try:
            window.evaluate_js(
                f"window.AutoConBus && window.AutoConBus.emit({_json(event)}, {_json(payload)})"
            )
        except Exception:
            pass

    def _bg(self, fn, *args, **kwargs) -> None:
        threading.Thread(
            target=self._guard, args=(fn, args, kwargs), daemon=True
        ).start()

    def _guard(self, fn, args, kwargs) -> None:
        try:
            fn(*args, **kwargs)
        except Exception as exc:
            self._emit("error", {"message": str(exc), "trace": traceback.format_exc()})

    # --------------------------------------------------------------- settings/env
    def get_settings(self) -> dict:
        return self.settings

    def update_settings(self, patch: dict) -> dict:
        self.settings = config.save_settings(patch or {})
        self._llm = None
        return self.settings

    def environment(self) -> dict:
        build_id = config.current_build_id()
        first_run_done = bool(
            self.settings.get("first_run_done")
            and (build_id == "source" or self.settings.get("build_id") == build_id)
        )
        if not first_run_done:
            device.detect(force=True)
        cli = self._get_llm()
        return {
            "settings": self.settings,
            "first_run_done": first_run_done,
            "build_id": build_id,
            "ffmpeg": media.has_ffmpeg(),
            "device": device.summary(self.settings),
            "packages": install.check(),
            "model_packs": model_registry.list_packs(self.settings),
            "ollama": cli.available(),
            "ollama_models": cli.list_models() if cli.available() else [],
            "ollama_model": (
                cli.resolve_model()
                if cli.available()
                else self.settings.get("default_model")
            ),
            "server_base": self._server.base_url(),
        }

    def finish_first_run(self) -> bool:
        self.settings = config.save_settings(
            {"first_run_done": True, "build_id": config.current_build_id()}
        )
        return True

    # ------------------------------------------------------------- dependencies
    def install_packages(self, keys: list[str]) -> dict:
        self._emit("install:start", {"keys": keys or []})

        def progress(payload: dict) -> None:
            self._emit("install:progress", payload)

        def run() -> None:
            res = install.install(
                keys or [], on_progress=progress, gpu="gpu" in set(keys or [])
            )
            device.detect(force=True)
            self._emit("install:done", res)

        self._bg(run)
        return {"ok": True}

    def list_model_packs(self) -> list[dict]:
        return model_registry.list_packs(self.settings)

    def install_model_pack(self, key: str) -> dict:
        self._emit("model:start", {"key": key})

        def progress(payload: dict) -> None:
            self._emit("model:progress", payload)

        def run() -> None:
            runtime_python = config.runtime_python()
            if not runtime_python.exists():
                self._emit(
                    "model:done",
                    {
                        "ok": False,
                        "error": "Сначала установите runtime-зависимости YOLO.",
                    },
                )
                return
            res = model_registry.install_pack(key, runtime_python, on_progress=progress)
            if res.get("ok"):
                meta = model_registry.PACKS.get(key, {})
                settings_key = meta.get("settings_key")
                if settings_key:
                    value = res.get("path")
                    if value and not res.get("virtual"):
                        value = str(Path(value))
                    self.update_settings(
                        {settings_key: value or meta.get("target", "")}
                    )
            self._emit("model:done", res)

        self._bg(run)
        return {"ok": True}

    def import_user_model(self, kind: str) -> dict | None:
        window = self._window()
        if not window:
            return None
        file_types = ("Модели (*.pt;*.onnx)", "Все файлы (*.*)")
        picked = window.create_file_dialog(
            webview.OPEN_DIALOG, allow_multiple=False, file_types=file_types
        )
        if not picked:
            return None
        path = str(picked[0])
        mapping = {
            "traffic": "traffic_sign_model",
            "plate": "plate_model",
            "vehicle_dino": "vehicle_dino_model",
        }
        if kind in mapping:
            self.update_settings({mapping[kind]: path})
        return {"ok": True, "path": path}

    # ------------------------------------------------------------------- Ollama
    def _get_llm(self) -> llm.OllamaClient:
        if self._llm is None:
            self._llm = llm.OllamaClient(
                self.settings.get("ollama_host", "http://127.0.0.1:11434"),
                self.settings.get("ollama_model", ""),
                self.settings.get("default_model", "qwen2.5:3b"),
            )
        return self._llm

    def list_ollama_models(self) -> dict:
        cli = self._get_llm()
        return {
            "models": cli.list_models(),
            "current": cli.resolve_model(),
            "default": self.settings.get("default_model"),
        }

    def set_ollama_model(self, name: str) -> str:
        self.update_settings({"ollama_model": name or ""})
        return self._get_llm().resolve_model()

    def pull_ollama_model(self, name: str) -> dict:
        self._emit("pull:start", {"model": name})

        def progress(f: float, text: str) -> None:
            self._emit("pull:progress", {"progress": f, "text": text, "model": name})

        def run() -> None:
            res = self._get_llm().pull(name, on_progress=progress)
            self._emit("pull:done", res)

        self._bg(run)
        return {"ok": True}

    # -------------------------------------------------------------------- media
    def list_projects(self) -> list[dict]:
        return project.list_projects()

    def list_chat_projects(self) -> list[dict]:
        items = []
        for item in project.list_projects():
            try:
                data = project.load(item["id"])
            except Exception:
                continue
            has_context = bool(
                data.get("summary")
                or data.get("sign_sequences")
                or data.get("vehicles")
                or data.get("plates")
                or data.get("comments")
            )
            if has_context:
                items.append({**item, "has_context": True})
        return items

    def pick_video(self) -> dict | None:
        window = self._window()
        if not window:
            return None
        file_types = ("Видео (*.mp4;*.mkv;*.mov;*.avi;*.webm)", "Все файлы (*.*)")
        picked = window.create_file_dialog(
            webview.OPEN_DIALOG, allow_multiple=False, file_types=file_types
        )
        if not picked:
            return None
        self.project = project.new_project(str(picked[0]), self.settings)
        project.save(self.project)
        return self._project_payload(self.project)

    def open_project(self, project_id: str) -> dict:
        self.project = project.load(project_id)
        return self._project_payload(self.project)

    def media_uri(self, path: str) -> str:
        try:
            return self._server.serve(path)
        except Exception:
            return Path(path).as_uri()

    def _project_payload(self, data: dict) -> dict:
        return {**data, "media_url": self.media_uri(data["source_path"])}

    # ------------------------------------------------------------------ vision
    def process_video(self) -> dict:
        if not self.project:
            return {"ok": False, "error": "Нет открытого видео"}
        if self._busy:
            return {"ok": False, "error": "Анализ уже идёт"}
        self._busy = True
        self._bg(self._process_video_bg)
        return {"ok": True}

    def _process_video_bg(self) -> None:
        assert self.project is not None
        path = self.project["source_path"]
        aggregator = EventAggregator()
        self._emit("process:start", {"title": self.project.get("title", "")})

        def progress(payload: dict) -> None:
            self._emit("process:progress", payload)

        def event_router(event: str, payload: dict) -> None:
            self._emit(event, payload)
            if event == "vision:scene" and self.settings.get(
                "commentary_enabled", True
            ):
                self._bg(self._comment_scene, payload)

        try:
            engine = VisionEngine(
                self.settings, on_event=event_router, on_progress=progress
            )
            result = engine.process_video(path, aggregator)
            self.project.update(result)
            self._summarize_project()
            project.save(self.project)
            self._emit("process:done", self._project_payload(self.project))
        finally:
            self._busy = False

    def _comment_scene(self, snapshot: dict) -> None:
        text = self._scene_text(snapshot)
        self._emit("vision:commentary", {"time": snapshot.get("time", 0), "text": text})
        if self.project is not None:
            self.project.setdefault("comments", []).append(
                {"t": snapshot.get("time", 0), "text": text}
            )
        if self.settings.get("voice_enabled"):
            self._speak(text)

    def _scene_text(self, snapshot: dict) -> str:
        cli = self._get_llm()
        if cli.available():
            try:
                return cli.generate(
                    llm.prompt_scene_commentary(snapshot), system=llm.SYS_AUTOCON
                ).strip()
            except Exception:
                pass
        signs = [d.get("label", "знак") for d in snapshot.get("signs", [])]
        vehicles = [d.get("label", "авто") for d in snapshot.get("vehicles", [])]
        parts = []
        if signs:
            parts.append("В зоне внимания: " + ", ".join(signs[:3]) + ".")
        if vehicles:
            parts.append(f"Транспортных объектов в кадре: {len(vehicles)}.")
        return " ".join(parts) or "Сцена анализируется, значимых объектов пока мало."

    def _summarize_project(self) -> None:
        if not self.project:
            return
        cli = self._get_llm()
        if not cli.available():
            return
        try:
            self.project["summary"] = cli.generate(
                llm.prompt_report_summary(self.project), system=llm.SYS_AUTOCON
            ).strip()
        except Exception:
            pass

    def _speak(self, text: str) -> None:
        def run() -> None:
            try:
                import pyttsx3

                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
            except Exception:
                pass

        self._bg(run)

    def chat_about_project(self, project_id: str, question: str) -> dict:
        if not project_id:
            return {"ok": False, "error": "Выберите обработанное видео"}
        if not (question or "").strip():
            return {"ok": False, "error": "Введите вопрос"}
        self._emit("chat:start", {"project_id": project_id, "question": question})
        self._bg(self._chat_about_project_bg, project_id, question.strip())
        return {"ok": True}

    def _chat_about_project_bg(self, project_id: str, question: str) -> None:
        try:
            data = project.load(project_id)
            cli = self._get_llm()
            if not cli.available():
                self._emit("chat:error", {"message": "Ollama не запущена или недоступна"})
                return
            answer = cli.generate(
                llm.prompt_video_chat(question, data),
                system=llm.SYS_AUTOCON,
                on_token=lambda token: self._emit("chat:token", {"token": token}),
            )
            self._emit("chat:done", {"answer": answer, "project_id": project_id})
        except Exception as exc:
            self._emit("chat:error", {"message": str(exc)})

    def list_camera_devices(self) -> list[dict]:
        return media.list_cameras()

    def start_camera(self, index: int = 0) -> dict:
        self.stop_camera()
        self._camera = CameraWorker(int(index), self.settings, self._camera_event)
        self._camera.start()
        return {"ok": True}

    def _camera_event(self, event: str, payload: dict) -> None:
        self._emit(event, payload)
        if event == "vision:frame" and self.settings.get("commentary_enabled", True):
            snapshot = {
                "time": payload.get("time", 0),
                "signs": [
                    d for d in payload.get("detections", []) if d.get("kind") == "sign"
                ],
                "vehicles": [
                    d
                    for d in payload.get("detections", [])
                    if d.get("kind") == "vehicle"
                ],
                "plates": [],
            }
            if (
                int(payload.get("time", 0))
                % max(1, int(self.settings.get("commentary_interval_sec") or 4))
                == 0
            ):
                self._bg(self._comment_scene, snapshot)

    def stop_camera(self) -> dict:
        if self._camera:
            self._camera.stop()
            self._camera = None
        return {"ok": True}

    # ------------------------------------------------------------------ export
    def export_report(self, fmt: str = "md") -> dict:
        if not self.project:
            return {"ok": False, "error": "Нет открытого проекта"}
        path = export.export_project(self.project, fmt)
        project.save(self.project)
        return {"ok": True, "path": path}
