# -*- coding: utf-8 -*-
"""YOLO11 frame processing and event aggregation."""

from __future__ import annotations

import base64
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .. import config
from . import device, media, model_registry

VEHICLE_LABELS = {"car", "truck", "bus", "motorcycle", "bicycle", "train"}
SIGN_OBJECT_LABELS = {"stop sign", "traffic light"}

SIGN_DESCRIPTIONS = {
    "stop": "нужно остановиться перед продолжением движения",
    "give way": "нужно уступить дорогу",
    "yield": "нужно уступить дорогу",
    "speed limit": "ограничение максимальной скорости",
    "no entry": "въезд запрещён",
    "no stopping": "остановка запрещена",
    "no parking": "стоянка запрещена",
    "pedestrian crossing": "рядом пешеходный переход",
    "traffic light": "участок регулируется светофором",
    "lane": "знак связан с направлением или полосой движения",
}


@dataclass
class FrameShape:
    width: int
    height: int


def normalize_label(label: str) -> str:
    return (label or "").strip().lower().replace("_", " ")


def position_for_bbox(bbox: list[float], shape: FrameShape | None) -> str:
    if not shape or not shape.width or not shape.height:
        return "center"
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) / 2) / shape.width
    cy = ((y1 + y2) / 2) / shape.height
    horiz = "left" if cx < 0.33 else "right" if cx > 0.66 else "center"
    vert = "top" if cy < 0.33 else "bottom" if cy > 0.66 else "middle"
    return f"{vert}-{horiz}"


def sign_description(label: str) -> str:
    norm = normalize_label(label)
    for key, desc in SIGN_DESCRIPTIONS.items():
        if key in norm:
            return desc
    return "требование знака нужно уточнить по классу модели"


def sign_lane_hint(det: dict, shape: FrameShape | None = None) -> str:
    bbox = det.get("bbox") or [0, 0, 0, 0]
    if shape and shape.width:
        cx = ((float(bbox[0]) + float(bbox[2])) / 2) / shape.width
        if cx < 0.38:
            return "left"
        if cx > 0.62:
            return "right"
    position = det.get("position", "")
    if "left" in position:
        return "left"
    if "right" in position:
        return "right"
    return "center"


class RoadContextAnalyzer:
    """Estimate coarse ego-lane context from a single frame."""

    def analyze(self, frame, detections: list[dict]) -> dict:
        shape = FrameShape(width=int(frame.shape[1]), height=int(frame.shape[0]))
        lane = "unknown"
        confidence = 0.15
        evidence: list[str] = []
        try:
            import cv2

            h, w = frame.shape[:2]
            roi = frame[int(h * 0.55) : h, :]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(gray, 60, 160)
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=math.pi / 180,
                threshold=45,
                minLineLength=max(35, w // 12),
                maxLineGap=35,
            )
            left = 0
            right = 0
            if lines is not None:
                for line in lines[:, 0]:
                    x1, y1, x2, y2 = [float(v) for v in line]
                    if abs(x2 - x1) < 1:
                        continue
                    slope = (y2 - y1) / (x2 - x1)
                    if abs(slope) < 0.35:
                        continue
                    mid_x = (x1 + x2) / 2
                    if slope < 0 and mid_x < w * 0.62:
                        left += 1
                    elif slope > 0 and mid_x > w * 0.38:
                        right += 1
            if left and right:
                lane = "center"
                confidence = min(0.85, 0.45 + 0.04 * min(left + right, 8))
                evidence.append(f"разметка слева/справа: {left}/{right}")
            elif left:
                lane = "right"
                confidence = min(0.6, 0.35 + 0.04 * min(left, 5))
                evidence.append(f"видна левая граница полосы: {left}")
            elif right:
                lane = "left"
                confidence = min(0.6, 0.35 + 0.04 * min(right, 5))
                evidence.append(f"видна правая граница полосы: {right}")
        except Exception:
            evidence.append("разметку не удалось оценить")

        if any(d.get("kind") == "vehicle" for d in detections):
            evidence.append("в кадре есть транспорт для ориентира")

        return {
            "lane": lane,
            "confidence": round(confidence, 2),
            "evidence": evidence,
            "shape": {"width": shape.width, "height": shape.height},
        }


class SignSetInterpreter:
    """Convert visible signs and lane context into a compact Russian hint."""

    def interpret(self, t: float, detections: list[dict], road: dict) -> dict:
        shape_data = road.get("shape") or {}
        shape = FrameShape(
            width=int(shape_data.get("width") or 0),
            height=int(shape_data.get("height") or 0),
        )
        signs = [
            d
            for d in detections
            if d.get("kind") == "sign"
            or normalize_label(d.get("label", "")) in SIGN_OBJECT_LABELS
        ]
        visible = []
        applicability = []
        for det in signs[:10]:
            applies = self._applies_to_ego_lane(det, road, shape)
            item = {
                "label": det.get("label", "sign"),
                "description": sign_description(det.get("label", "")),
                "position": det.get("position", ""),
                "confidence": det.get("confidence", 0),
                "applies_to_ego_lane": applies,
            }
            visible.append(item)
            applicability.append(applies)

        if not visible:
            explanation = "В текущем кадре дорожные знаки не подтверждены."
            overall = "unknown"
            confidence = road.get("confidence", 0.15)
        else:
            names = ", ".join(item["label"] for item in visible[:4])
            details = "; ".join(
                f"{item['label']}: {item['description']}" for item in visible[:3]
            )
            lane_text = {
                "left": "похоже, авто в левой полосе",
                "center": "похоже, авто в своей центральной полосе",
                "right": "похоже, авто в правой полосе",
            }.get(road.get("lane"), "полоса движения не уверенно определена")
            explanation = f"Вижу знаки: {names}. {details}. {lane_text}."
            if "yes" in applicability:
                overall = "yes"
            elif applicability and all(value == "no" for value in applicability):
                overall = "no"
            else:
                overall = "unknown"
            sign_conf = sum(float(s.get("confidence") or 0) for s in visible) / max(
                1, len(visible)
            )
            confidence = round((sign_conf * 0.65) + (road.get("confidence", 0) * 0.35), 2)

        warnings = []
        if road.get("lane") == "unknown":
            warnings.append("применимость к полосе оценена с низкой уверенностью")
        if visible and overall == "unknown":
            warnings.append("для части знаков неясно, относятся ли они к текущей полосе")

        return {
            "time": round(t, 2),
            "visible_signs": visible,
            "lane": road,
            "applies_to_ego_lane": overall,
            "explanation": explanation,
            "confidence": round(float(confidence or 0), 2),
            "warnings": warnings,
        }

    def _applies_to_ego_lane(self, det: dict, road: dict, shape: FrameShape) -> str:
        lane = road.get("lane", "unknown")
        if lane == "unknown":
            return "unknown"
        label = normalize_label(det.get("label", ""))
        hint = sign_lane_hint(det, shape)
        if any(word in label for word in ("lane", "полос", "left", "right", "turn")):
            if hint == "center" or hint == lane:
                return "yes"
            return "no"
        if hint in {"center", "right"}:
            return "yes"
        return "unknown"


class EventAggregator:
    """Collect noisy per-frame detections into stable scene events."""

    def __init__(self, gap_sec: float = 2.0) -> None:
        self.gap_sec = gap_sec
        self.sign_sequences: list[dict] = []
        self.vehicles: dict[str, dict] = {}
        self.plates: dict[str, dict] = {}
        self.comments: list[dict] = []
        self.contexts: list[dict] = []

    def update(self, t: float, detections: list[dict]) -> list[dict]:
        events = []
        for det in detections:
            kind = det.get("kind")
            if kind == "sign":
                event = self._update_sign(t, det)
                if event:
                    events.append(event)
            elif kind == "vehicle":
                self._update_vehicle(t, det)
            elif kind == "plate" and det.get("text"):
                self._update_plate(t, det)
        return events

    def _update_sign(self, t: float, det: dict) -> Optional[dict]:
        label = det.get("label", "sign")
        position = det.get("position", "")
        last = self.sign_sequences[-1] if self.sign_sequences else None
        if last and last["label"] == label and t - last["end"] <= self.gap_sec:
            count = last.get("count", 1) + 1
            last.update(
                {
                    "end": t,
                    "count": count,
                    "confidence": round(
                        (
                            (last.get("confidence", 0) * (count - 1))
                            + det.get("confidence", 0)
                        )
                        / count,
                        3,
                    ),
                    "position": position or last.get("position", ""),
                }
            )
            return None
        seq = {
            "id": f"sign-{len(self.sign_sequences) + 1}",
            "label": label,
            "start": t,
            "end": t,
            "count": 1,
            "confidence": round(float(det.get("confidence", 0)), 3),
            "position": position,
        }
        self.sign_sequences.append(seq)
        return {"kind": "sign_sequence", **seq}

    def _update_vehicle(self, t: float, det: dict) -> None:
        track_id = det.get("track_id")
        key = (
            str(track_id)
            if track_id is not None
            else f"{det.get('label', 'vehicle')}:{det.get('position', '')}"
        )
        item = self.vehicles.setdefault(
            key,
            {
                "track_id": track_id,
                "label": det.get("label", "vehicle"),
                "first_t": t,
                "last_t": t,
                "count": 0,
                "confidence": 0.0,
                "position": det.get("position", ""),
            },
        )
        item["last_t"] = t
        item["count"] += 1
        n = item["count"]
        item["confidence"] = round(
            ((item["confidence"] * (n - 1)) + det.get("confidence", 0)) / n, 3
        )
        item["position"] = det.get("position", item["position"])

    def _update_plate(self, t: float, det: dict) -> None:
        text = "".join(ch for ch in det.get("text", "").upper() if ch.isalnum())
        if not text:
            return
        item = self.plates.setdefault(
            text,
            {
                "text": text,
                "first_t": t,
                "last_t": t,
                "count": 0,
                "confidence": 0.0,
                "position": det.get("position", ""),
            },
        )
        item["last_t"] = t
        item["count"] += 1
        n = item["count"]
        item["confidence"] = round(
            ((item["confidence"] * (n - 1)) + det.get("confidence", 0)) / n, 3
        )

    def add_comment(self, t: float, text: str) -> None:
        self.comments.append({"t": t, "text": text})

    def add_context(self, context: dict) -> None:
        self.contexts.append(context)

    def snapshot(self, t: float, detections: list[dict], context: dict | None = None) -> dict:
        return {
            "time": round(t, 2),
            "signs": [d for d in detections if d.get("kind") == "sign"][:12],
            "vehicles": [d for d in detections if d.get("kind") == "vehicle"][:12],
            "plates": list(self.plates.values())[-8:],
            "recent_sign_sequences": self.sign_sequences[-8:],
            "context": context or {},
        }

    def result(self) -> dict:
        return {
            "sign_sequences": self.sign_sequences,
            "vehicles": list(self.vehicles.values()),
            "plates": list(self.plates.values()),
            "comments": self.comments,
            "contexts": self.contexts,
        }


class PlateReader:
    def __init__(self) -> None:
        self.reader = None
        try:
            from fast_alpr import ALPR

            self.reader = ALPR()
            self.kind = "fast_alpr"
        except Exception:
            try:
                from fast_plate_ocr import LicensePlateRecognizer

                self.reader = LicensePlateRecognizer()
                self.kind = "fast_plate_ocr"
            except Exception:
                self.reader = None
                self.kind = ""

    def read(self, frame, bbox: list[float]) -> tuple[str, float]:
        if self.reader is None:
            return "", 0.0
        try:
            import cv2
            import numpy as np

            x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return "", 0.0
            if self.kind == "fast_alpr":
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                results = self.reader.predict(rgb)
                if results:
                    plate = results[0]
                    text = getattr(plate, "ocr", None) or getattr(plate, "plate", "")
                    confidence = float(getattr(plate, "confidence", 0.0) or 0.0)
                    return str(text), confidence
            pred = self.reader.run(np.asarray(crop))
            if isinstance(pred, list) and pred:
                item = pred[0]
                if isinstance(item, dict):
                    return str(item.get("text") or item.get("plate") or ""), float(
                        item.get("confidence") or 0.0
                    )
                return str(item), 0.5
        except Exception:
            pass
        return "", 0.0


class VisionEngine:
    def __init__(
        self,
        settings: dict,
        *,
        on_event: Optional[Callable[[str, dict], None]] = None,
        on_progress: Optional[Callable[[dict], None]] = None,
    ) -> None:
        config.bootstrap_runtime_packages()
        self.settings = settings
        self.on_event = on_event or (lambda _e, _p: None)
        self.on_progress = on_progress or (lambda _p: None)
        self.device = device.resolve_device(settings)
        self.vehicle_model = None
        self.sign_model = None
        self.plate_model = None
        self.plate_reader = None
        self.names: dict[str, dict] = {}
        self.road_context = RoadContextAnalyzer()
        self.sign_interpreter = SignSetInterpreter()
        self._load_models()

    def _load_models(self) -> None:
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError(
                "YOLO runtime не установлен. Откройте первый запуск и установите блок 'YOLO11 и обработка видео'."
            ) from exc

        vehicle_ref = self.settings.get("yolo_vehicle_model") or "yolo11s.pt"
        self.vehicle_model = YOLO(vehicle_ref)
        self.names["vehicle"] = getattr(self.vehicle_model, "names", {}) or {}

        sign_ref = self.settings.get("traffic_sign_model") or ""
        if sign_ref and Path(sign_ref).exists():
            self.sign_model = YOLO(sign_ref)
            self.names["sign"] = getattr(self.sign_model, "names", {}) or {}

        plate_ref = self.settings.get("plate_model") or ""
        if plate_ref and Path(plate_ref).exists():
            self.plate_model = YOLO(plate_ref)
            self.names["plate"] = getattr(self.plate_model, "names", {}) or {}
        if self.settings.get("ocr_enabled", True):
            self.plate_reader = PlateReader()

    def process_video(
        self,
        path: str,
        aggregator: EventAggregator,
        stop_event: threading.Event | None = None,
    ) -> dict:
        import cv2

        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        target_fps = max(1, float(self.settings.get("target_fps") or 8))
        step = max(1, int(round(fps / target_fps)))
        detections_all = []
        frame_idx = 0
        processed = 0
        last_comment_t = -999.0
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            ok = cap.grab()
            if not ok:
                break
            if frame_idx % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                t = frame_idx / fps if fps else 0
                detections = self.detect_frame(frame, t)
                events = aggregator.update(t, detections)
                context = self.analyze_context(frame, t, detections)
                aggregator.add_context(context)
                detections_all.extend(detections)
                self.on_event(
                    "vision:detections", {"time": t, "detections": detections}
                )
                self.on_event("vision:context", context)
                for event in events:
                    self.on_event("vision:event", event)
                if total:
                    self.on_progress(
                        {
                            "progress": min(0.999, frame_idx / total),
                            "label": "Анализ видео",
                            "time": t,
                        }
                    )
                if self.settings.get(
                    "commentary_enabled", True
                ) and t - last_comment_t >= float(
                    self.settings.get("commentary_interval_sec") or 4
                ):
                    last_comment_t = t
                    self.on_event(
                        "vision:scene", aggregator.snapshot(t, detections, context)
                    )
                processed += 1
            frame_idx += 1
        cap.release()
        self.on_progress({"progress": 1.0, "label": "Готово"})
        return {
            "detections": detections_all,
            **aggregator.result(),
            "processed_frames": processed,
            "cancelled": bool(stop_event and stop_event.is_set()),
        }

    def detect_frame(self, frame, t: float) -> list[dict]:
        shape = FrameShape(width=int(frame.shape[1]), height=int(frame.shape[0]))
        detections: list[dict] = []
        detections.extend(
            self._run_model(
                self.vehicle_model, frame, t, "vehicle", shape, tracking=True
            )
        )
        if self.sign_model is not None:
            detections.extend(self._run_model(self.sign_model, frame, t, "sign", shape))
        if self.plate_model is not None:
            plates = self._run_model(self.plate_model, frame, t, "plate", shape)
            for plate in plates:
                if self.plate_reader:
                    text, conf = self.plate_reader.read(frame, plate["bbox"])
                    plate["text"] = text
                    if conf:
                        plate["confidence"] = max(plate["confidence"], conf)
            detections.extend(plates)
        return detections

    def analyze_context(self, frame, t: float, detections: list[dict]) -> dict:
        road = self.road_context.analyze(frame, detections)
        return self.sign_interpreter.interpret(t, detections, road)

    def analyze_image(self, path: str) -> dict:
        import cv2

        frame = cv2.imread(path)
        if frame is None:
            raise RuntimeError("Не удалось прочитать изображение")
        detections = self.detect_frame(frame, 0.0)
        context = self.analyze_context(frame, 0.0, detections)
        return {"detections": detections, "context": context}

    def _run_model(
        self,
        model,
        frame,
        t: float,
        kind: str,
        shape: FrameShape,
        tracking: bool = False,
    ) -> list[dict]:
        if model is None:
            return []
        kwargs = {
            "imgsz": int(self.settings.get("imgsz") or 960),
            "conf": float(self.settings.get("conf") or 0.35),
            "iou": float(self.settings.get("iou") or 0.65),
            "device": self.device,
            "verbose": False,
        }
        try:
            if tracking:
                results = model.track(
                    frame,
                    persist=True,
                    tracker=self.settings.get("tracker") or "bytetrack.yaml",
                    **kwargs,
                )
            else:
                results = model.predict(frame, **kwargs)
        except TypeError:
            kwargs.pop("device", None)
            results = model.predict(frame, **kwargs)
        out = []
        for result in results or []:
            names = getattr(result, "names", None) or getattr(model, "names", {}) or {}
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            xyxy = (
                boxes.xyxy.cpu().tolist()
                if hasattr(boxes.xyxy, "cpu")
                else boxes.xyxy.tolist()
            )
            confs = (
                boxes.conf.cpu().tolist()
                if hasattr(boxes.conf, "cpu")
                else boxes.conf.tolist()
            )
            clss = (
                boxes.cls.cpu().tolist()
                if hasattr(boxes.cls, "cpu")
                else boxes.cls.tolist()
            )
            ids = None
            if getattr(boxes, "id", None) is not None:
                ids = (
                    boxes.id.cpu().tolist()
                    if hasattr(boxes.id, "cpu")
                    else boxes.id.tolist()
                )
            for index, bbox in enumerate(xyxy):
                cls_id = int(clss[index])
                label = str(names.get(cls_id, cls_id))
                norm = normalize_label(label)
                actual_kind = kind
                if kind == "vehicle" and norm not in VEHICLE_LABELS:
                    if norm not in {"person", "traffic light", "stop sign"}:
                        continue
                    actual_kind = "road_object"
                out.append(
                    {
                        "time": round(t, 3),
                        "kind": actual_kind,
                        "label": label,
                        "confidence": round(float(confs[index]), 3),
                        "bbox": [round(float(v), 1) for v in bbox],
                        "position": position_for_bbox([float(v) for v in bbox], shape),
                        "track_id": (
                            int(ids[index])
                            if ids is not None and not math.isnan(float(ids[index]))
                            else None
                        ),
                        "source": kind,
                    }
                )
        return out


class VideoAnalysisSession:
    def __init__(
        self,
        project_id: str,
        path: str,
        settings: dict,
        emit: Callable[[str, dict], None],
        done: Callable[[dict], None],
    ) -> None:
        self.project_id = project_id
        self.path = path
        self.settings = settings
        self.emit = emit
        self.done = done
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.aggregator = EventAggregator()
        self._status = {
            "project_id": project_id,
            "running": False,
            "progress": 0.0,
            "analyzed_time": 0.0,
            "processed_frames": 0,
            "label": "ожидание",
        }

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def status(self) -> dict:
        return dict(self._status)

    def _run(self) -> None:
        self._status.update({"running": True, "label": "анализ видео"})
        self.emit("video:analysis_start", self.status())

        def progress(payload: dict) -> None:
            self._status.update(
                {
                    "progress": float(payload.get("progress") or 0),
                    "analyzed_time": float(payload.get("time") or 0),
                    "label": payload.get("label") or "анализ видео",
                }
            )
            self.emit("video:analysis_status", self.status())

        try:
            engine = VisionEngine(self.settings, on_event=self.emit, on_progress=progress)
            result = engine.process_video(self.path, self.aggregator, self.stop_event)
            result["project_id"] = self.project_id
            result["status"] = self.status()
            self.done(result)
            self._status.update(
                {
                    "running": False,
                    "progress": 1.0 if not result.get("cancelled") else self._status["progress"],
                    "processed_frames": result.get("processed_frames", 0),
                    "label": "остановлено" if result.get("cancelled") else "готово",
                }
            )
            self.emit("video:analysis_done", self.status())
        except Exception as exc:
            self._status.update({"running": False, "label": str(exc)})
            self.emit(
                "video:analysis_done",
                {**self.status(), "ok": False, "error": str(exc)},
            )


class CameraWorker:
    def __init__(
        self, camera_index: int, settings: dict, emit: Callable[[str, dict], None]
    ) -> None:
        self.camera_index = camera_index
        self.settings = settings
        self.emit = emit
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.engine: VisionEngine | None = None
        self.aggregator = EventAggregator()

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def _run(self) -> None:
        try:
            import cv2
        except Exception:
            self.emit("camera:error", {"message": "OpenCV runtime не установлен"})
            return
        try:
            self.engine = VisionEngine(self.settings, on_event=self.emit)
            cap, backend = media.open_camera(int(self.camera_index))
            if not cap.isOpened():
                self.emit(
                    "camera:error",
                    {"message": f"Камера {self.camera_index} недоступна"},
                )
                return
            self.emit("camera:started", {"index": self.camera_index, "backend": backend})
            target_fps = max(1, float(self.settings.get("target_fps") or 8))
            delay = 1.0 / target_fps
            start = time.time()
            last_sent = 0.0
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.1)
                    continue
                now = time.time()
                if now - last_sent < delay:
                    continue
                last_sent = now
                t = now - start
                detections = self.engine.detect_frame(frame, t)
                events = self.aggregator.update(t, detections)
                context = self.engine.analyze_context(frame, t, detections)
                self.aggregator.add_context(context)
                ok, jpeg = cv2.imencode(
                    ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 72]
                )
                image = (
                    "data:image/jpeg;base64,"
                    + base64.b64encode(jpeg.tobytes()).decode("ascii")
                    if ok
                    else ""
                )
                self.emit(
                    "vision:frame",
                    {"time": t, "image": image, "detections": detections},
                )
                self.emit("vision:context", context)
                for event in events:
                    self.emit("vision:event", event)
            cap.release()
        except Exception as exc:
            config.log_event(f"camera error: {exc}")
            self.emit("camera:error", {"message": str(exc)})
