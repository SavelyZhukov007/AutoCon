# -*- coding: utf-8 -*-
"""pywebview API bridge used by the AutoCon UI."""

from __future__ import annotations

import json
import base64
import logging
import shutil
import threading
import traceback
from pathlib import Path

try:
    import webview
except Exception:  # pragma: no cover - tests can run without pywebview
    webview = None

from . import config
from .core import device, export, install, llm, media, model_registry, project, runtime, server
from .core.vision import (
    CameraWorker,
    EventAggregator,
    VideoAnalysisSession,
    VisionEngine,
    sign_description,
)


LOG = logging.getLogger(__name__)


def _json(data) -> str:
    return json.dumps(data, ensure_ascii=False)


class Api:
    def __init__(self) -> None:
        config.bootstrap_logging()
        self.settings = config.load_settings()
        self.project: dict | None = None
        self._server = server.MediaServer()
        self._server.set_data_root(config.user_data_dir())
        self._llm: llm.OllamaClient | None = None
        self._camera: CameraWorker | None = None
        self._video_session: VideoAnalysisSession | None = None
        self._exam_image: str = ""
        self._busy = False
        self._installing = False

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
            LOG.exception("Background task failed: %s", getattr(fn, "__name__", fn))
            self._emit("error", {"message": str(exc), "trace": traceback.format_exc()})

    # --------------------------------------------------------------- settings/env
    def get_settings(self) -> dict:
        return self.settings

    def update_settings(self, patch: dict) -> dict:
        self.settings = config.save_settings(patch or {})
        self._llm = None
        return self.settings

    def _required_ollama_models(self) -> list[str]:
        names = [
            self.settings.get("default_model") or "qwen2.5:3b",
            self.settings.get("vision_model") or "qwen2.5vl:3b",
        ]
        return list(dict.fromkeys(name for name in names if name))

    def first_run_readiness(self) -> dict:
        cli = self._get_llm()
        packages = install.check()
        runtime_health = runtime.health_check(runtime.ALL_FEATURE_KEYS)
        model_packs = model_registry.list_packs(self.settings)
        missing_packs = [pack for pack in model_packs if not pack.get("installed")]
        ollama_available = cli.available()
        ollama_models = cli.list_models() if ollama_available else []
        required_ollama = self._required_ollama_models()
        missing_ollama = [
            name
            for name in required_ollama
            if name not in ollama_models or not cli.model_in_central_store(name)
        ]
        central_ollama = cli.central_store_status(required_ollama)
        full_log = config.full_log_path()
        ok = bool(
            runtime_health.get("ok")
            and not missing_packs
            and ollama_available
            and not missing_ollama
            and central_ollama.get("ok")
            and full_log.exists()
        )
        return {
            "ok": ok,
            "packages": packages,
            "runtime_health": runtime_health,
            "model_packs": model_packs,
            "missing_model_packs": missing_packs,
            "ollama": ollama_available,
            "ollama_models": ollama_models,
            "required_ollama_models": required_ollama,
            "missing_ollama_models": missing_ollama,
            "central_ollama": central_ollama,
            "full_log": str(full_log),
            "full_log_exists": full_log.exists(),
        }

    def environment(self) -> dict:
        build_id = config.current_build_id()
        readiness = self.first_run_readiness()
        build_matches = build_id == "source" or self.settings.get("build_id") == build_id
        if readiness.get("ok") and (
            not self.settings.get("first_run_done") or not build_matches
        ):
            self.settings = config.save_settings(
                {"first_run_done": True, "build_id": build_id}
            )
            build_matches = True
        first_run_done = bool(
            self.settings.get("first_run_done")
            and build_matches
            and readiness.get("ok")
        )
        if not first_run_done:
            device.detect(force=True)
        cli = self._get_llm()
        return {
            "settings": self.settings,
            "first_run_done": first_run_done,
            "first_run_readiness": readiness,
            "build_id": build_id,
            "paths": {
                "user_data": str(config.user_data_dir()),
                "runtime": str(config.runtime_dir()),
                "models": str(config.models_dir()),
                "ollama_models": str(config.ollama_models_dir()),
                "full_log": str(config.full_log_path()),
            },
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
        readiness = self.first_run_readiness()
        if not readiness.get("ok"):
            config.log_event(
                "finish_first_run refused: " + json.dumps(readiness, ensure_ascii=False),
                level=logging.WARNING,
            )
            return False
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

    def _apply_model_result(self, key: str, res: dict) -> None:
        if not res.get("ok"):
            return
        meta = model_registry.PACKS.get(key, {})
        settings_key = meta.get("settings_key")
        value = res.get("path")
        if settings_key and value:
            self.update_settings({settings_key: str(Path(value))})

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
            self._apply_model_result(key, res)
            self._emit("model:done", res)

        self._bg(run)
        return {"ok": True}

    def install_everything(self) -> dict:
        if self._installing:
            return {"ok": False, "error": "setup already running"}
        self._installing = True
        self._emit(
            "setup:start",
            {
                "features": list(runtime.ALL_FEATURE_KEYS),
                "model_packs": list(model_registry.ALL_PACK_KEYS),
                "ollama_models": self._required_ollama_models(),
            },
        )
        self._bg(self._install_everything_bg)
        return {"ok": True}

    def _setup_progress(self, progress: float, text: str, **extra) -> None:
        payload = {"progress": max(0.0, min(1.0, progress)), "text": text}
        payload.update(extra)
        self._emit("setup:progress", payload)

    def _install_everything_bg(self) -> None:
        failures: list[dict] = []
        try:
            config.bootstrap_logging()
            features = list(runtime.ALL_FEATURE_KEYS)
            packs = list(model_registry.ALL_PACK_KEYS)
            ollama_models = self._required_ollama_models()
            total_steps = 1 + len(packs) + len(ollama_models)

            def runtime_progress(payload: dict) -> None:
                self._emit("install:progress", payload)
                inner = float(payload.get("progress") or 0)
                self._setup_progress(
                    inner / total_steps,
                    payload.get("text") or "Installing runtime",
                    phase="runtime",
                    log_path=payload.get("log_path"),
                )

            self._setup_progress(0.0, "Installing runtime", phase="runtime")
            runtime_res = install.install(
                features, on_progress=runtime_progress, gpu=True
            )
            self._emit("install:done", runtime_res)
            if not runtime_res.get("ok"):
                failures.append({"phase": "runtime", "result": runtime_res})
            device.detect(force=True)

            runtime_python = config.runtime_python()
            if not runtime_python.exists():
                failures.append(
                    {"phase": "models", "error": "Runtime Python was not created."}
                )
            else:
                for index, key in enumerate(packs, start=1):
                    step_offset = index

                    def model_progress(payload: dict, *, step_offset=step_offset) -> None:
                        self._emit("model:progress", payload)
                        inner = float(payload.get("progress") or 0)
                        self._setup_progress(
                            (step_offset + inner) / total_steps,
                            payload.get("text") or f"Installing {key}",
                            phase="model",
                            pack=key,
                        )

                    self._emit("model:start", {"key": key})
                    self._setup_progress(
                        step_offset / total_steps, f"Installing model pack {key}", phase="model", pack=key
                    )
                    res = model_registry.install_pack(
                        key, runtime_python, on_progress=model_progress
                    )
                    self._apply_model_result(key, res)
                    self._emit("model:done", res)
                    if not res.get("ok"):
                        failures.append({"phase": "model", "key": key, "result": res})

            cli = self._get_llm()
            migration = llm.migrate_legacy_store_to_central(ollama_models)
            if migration.get("ok"):
                self._setup_progress(
                    (1 + len(packs)) / total_steps,
                    "Migrated existing Ollama models to AutoCon",
                    phase="ollama",
                    log_path=str(config.full_log_path()),
                )
            if not cli.available():
                failures.append(
                    {
                        "phase": "ollama",
                        "error": (
                            "Ollama is not available. Start Ollama with OLLAMA_MODELS="
                            + str(config.ollama_models_dir())
                        ),
                    }
                )
            else:
                for offset, name in enumerate(ollama_models, start=1 + len(packs)):
                    self._emit("pull:start", {"model": name})

                    def pull_progress(f: float, text: str, *, name=name, offset=offset) -> None:
                        self._emit(
                            "pull:progress",
                            {"progress": f, "text": text, "model": name},
                        )
                        self._setup_progress(
                            (offset + float(f or 0)) / total_steps,
                            text or f"Pulling {name}",
                            phase="ollama",
                            model=name,
                        )

                    if name in cli.list_models() and cli.model_in_central_store(name):
                        res = {"ok": True, "model": name, "already": True}
                    else:
                        res = cli.pull(name, on_progress=pull_progress)
                    self._emit("pull:done", res)
                    if not res.get("ok"):
                        failures.append({"phase": "ollama", "model": name, "result": res})

            readiness = self.first_run_readiness()
            ok = not failures and readiness.get("ok")
            if ok:
                self.settings = config.save_settings(
                    {"first_run_done": True, "build_id": config.current_build_id()}
                )
            result = {
                "ok": ok,
                "failed": failures,
                "readiness": readiness,
                "target": str(config.user_data_dir()),
                "log_path": str(config.full_log_path()),
            }
            self._setup_progress(1.0, "Setup complete" if ok else "Setup failed", phase="done")
            config.log_event("install_everything result: " + json.dumps(result, ensure_ascii=False))
            self._emit("setup:done", result)
        except Exception as exc:
            LOG.exception("install_everything failed")
            self._emit(
                "setup:done",
                {
                    "ok": False,
                    "error": str(exc),
                    "trace": traceback.format_exc(),
                    "log_path": str(config.full_log_path()),
                },
            )
        finally:
            self._installing = False

    def _centralize_user_model(self, path: str | Path) -> Path:
        source = Path(path)
        target_dir = config.models_dir() / "user-imports"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / source.name
        if target.exists() and source.resolve() != target.resolve():
            stem = source.stem
            suffix = source.suffix
            idx = 1
            while target.exists():
                target = target_dir / f"{stem}-{idx}{suffix}"
                idx += 1
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        config.log_event(f"User model imported: {source} -> {target}")
        return target

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
        path = self._centralize_user_model(picked[0])
        mapping = {
            "traffic": "traffic_sign_model",
            "traffic_signs_100": "traffic_sign_model",
            "sign": "traffic_sign_model",
            "plate": "plate_model",
            "license_plate": "plate_model",
            "vehicle_dino": "vehicle_dino_model",
        }
        if kind in mapping:
            self.update_settings({mapping[kind]: str(path)})
        return {"ok": True, "path": str(path)}

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

    def pull_vision_model(self, name: str = "") -> dict:
        model = name or self.settings.get("vision_model", "qwen2.5vl:3b")
        self._emit("pull:start", {"model": model})

        def progress(f: float, text: str) -> None:
            self._emit("pull:progress", {"progress": f, "text": text, "model": model})

        def run() -> None:
            res = self._get_llm().pull(model, on_progress=progress)
            self._emit("pull:done", res)

        self._bg(run)
        return {"ok": True}

    def set_vision_model(self, name: str) -> str:
        model = name or "qwen2.5vl:3b"
        self.update_settings({"vision_model": model})
        return model

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
        self.stop_video_realtime()
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
        self.stop_video_realtime()
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
    def start_video_realtime(self, project_id: str = "") -> dict:
        if project_id:
            if not self.project or self.project.get("id") != project_id:
                self.project = project.load(project_id)
        if not self.project:
            return {"ok": False, "error": "Нет открытого видео"}
        if self._video_session and self._video_session.status().get("running"):
            if self._video_session.project_id == self.project.get("id"):
                return {"ok": True, "status": self._video_session.status()}
            self._video_session.stop()

        session = VideoAnalysisSession(
            self.project["id"],
            self.project["source_path"],
            dict(self.settings),
            self._video_event,
            self._video_analysis_done,
        )
        self._video_session = session
        session.start()
        return {"ok": True, "status": session.status()}

    def stop_video_realtime(self) -> dict:
        if self._video_session:
            self._video_session.stop()
            status = self._video_session.status()
            self._video_session = None
            return {"ok": True, "status": status}
        return {"ok": True, "status": {"running": False}}

    def video_realtime_status(self) -> dict:
        if not self._video_session:
            return {"running": False}
        return self._video_session.status()

    def _video_event(self, event: str, payload: dict) -> None:
        self._emit(event, payload)
        if event == "vision:scene" and self.settings.get("commentary_enabled", True):
            self._bg(self._comment_scene, payload)

    def _video_analysis_done(self, result: dict) -> None:
        if not self.project or result.get("project_id") != self.project.get("id"):
            return
        comments = list(self.project.get("comments", []))
        for key, value in result.items():
            if key in {"project_id", "status", "comments"}:
                continue
            self.project[key] = value
        if result.get("comments"):
            comments.extend(result["comments"])
        self.project["comments"] = comments
        if not result.get("cancelled"):
            self._summarize_project()
        project.save(self.project)
        self._emit("process:done", self._project_payload(self.project))

    def process_video(self) -> dict:
        if not self.project:
            return {"ok": False, "error": "Нет открытого видео"}
        self.stop_video_realtime()
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
        except Exception as exc:
            LOG.exception("Video processing failed")
            self._emit(
                "process:error",
                {"message": str(exc), "trace": traceback.format_exc()},
            )
        finally:
            self._busy = False

    def pick_exam_image(self) -> dict | None:
        window = self._window()
        if not window:
            return None
        file_types = ("Изображения (*.jpg;*.jpeg;*.png;*.webp)", "Все файлы (*.*)")
        picked = window.create_file_dialog(
            webview.OPEN_DIALOG, allow_multiple=False, file_types=file_types
        )
        if not picked:
            return None
        self._exam_image = str(picked[0])
        payload = {"ok": True, "path": self._exam_image, "image_url": self.media_uri(self._exam_image)}
        self._emit("exam:image_loaded", payload)
        return payload

    def analyze_exam_image(self, question: str) -> dict:
        if not self._exam_image:
            return {"ok": False, "error": "Сначала загрузите изображение"}
        if not (question or "").strip():
            return {"ok": False, "error": "Введите вопрос по картинке"}
        self._emit("exam:analysis_start", {"path": self._exam_image, "question": question})
        self._bg(self._analyze_exam_image_bg, self._exam_image, question.strip())
        return {"ok": True}

    def _analyze_exam_image_bg(self, image_path: str, question: str) -> None:
        try:
            findings = {
                "detections": [],
                "context": {},
                "pdd_version": llm.PDD_RU_CONTEXT["version"],
            }
            try:
                engine = VisionEngine(self.settings, on_event=lambda _e, _p: None)
                findings.update(engine.analyze_image(image_path))
            except Exception as exc:
                findings["vision_error"] = str(exc)

            cli = self._get_llm()
            if not cli.available():
                raise RuntimeError("Ollama недоступна. Запустите Ollama или скачайте vision-модель.")
            image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
            model = self.settings.get("vision_model") or "qwen2.5vl:3b"
            answer = cli.generate(
                llm.prompt_exam_photo(question, findings),
                system=llm.SYS_AUTOCON,
                model=model,
                images=[image_b64],
                temperature=0.15,
            ).strip()
            case = {
                "image_path": image_path,
                "question": question,
                "answer": answer,
                "findings": findings,
                "vision_model": model,
            }
            if self.project is not None:
                self.project.setdefault("exam_cases", []).append(case)
                project.save(self.project)
            self._emit("exam:analysis_done", case)
        except Exception as exc:
            self._emit("exam:error", {"message": str(exc)})

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

    def _context_near_time(self, data: dict | None, t: float) -> dict | None:
        contexts = (data or {}).get("contexts", [])
        best = None
        best_delta = 999999.0
        for ctx in contexts:
            delta = abs(float(ctx.get("time") or 0) - float(t or 0))
            if delta < best_delta:
                best = ctx
                best_delta = delta
        return best if best_delta <= 1.5 else None

    def explain_sign(
        self,
        project_id: str = "",
        detection: dict | None = None,
        context: dict | None = None,
        time: float = 0.0,
    ) -> dict:
        detection = detection or {}
        label = str(detection.get("label") or detection.get("kind") or "sign")
        data = self.project
        if project_id and (not data or data.get("id") != project_id):
            try:
                data = project.load(project_id)
            except Exception:
                data = self.project
        ctx = context or self._context_near_time(data, float(time or detection.get("time") or 0))
        fallback = f"Знак {label}: {sign_description(label)}."
        if ctx and ctx.get("explanation"):
            fallback += " " + str(ctx.get("explanation"))
        cli = self._get_llm()
        if not cli.available():
            return {"ok": True, "answer": fallback, "fallback": True}
        try:
            answer = cli.generate(
                llm.prompt_sign_explanation(detection, ctx, data),
                system=llm.SYS_AUTOCON,
                temperature=0.15,
            ).strip()
            return {"ok": True, "answer": answer or fallback, "fallback": not bool(answer)}
        except Exception as exc:
            LOG.exception("Sign explanation failed")
            return {
                "ok": True,
                "answer": fallback,
                "fallback": True,
                "error": str(exc),
            }

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

    def import_model_pack(self, kind: str, path: str = "") -> dict | None:
        if not path:
            return self.import_user_model(kind)
        model_path = Path(path)
        if not model_path.exists() or model_path.suffix.lower() not in {".pt", ".onnx"}:
            return {"ok": False, "error": "Выберите существующий .pt или .onnx файл"}
        mapping = {
            "traffic": "traffic_sign_model",
            "traffic_signs_100": "traffic_sign_model",
            "sign": "traffic_sign_model",
            "plate": "plate_model",
            "license_plate": "plate_model",
            "vehicle_dino": "vehicle_dino_model",
        }
        key = mapping.get(kind)
        if not key:
            return {"ok": False, "error": "Неизвестный тип модели"}
        central_path = self._centralize_user_model(model_path)
        self.update_settings({key: str(central_path)})
        return {"ok": True, "path": str(central_path), "settings_key": key}

    # ------------------------------------------------------------------ export
    def export_report(self, fmt: str = "md") -> dict:
        if not self.project:
            return {"ok": False, "error": "Нет открытого проекта"}
        path = export.export_project(self.project, fmt)
        project.save(self.project)
        return {"ok": True, "path": path}
