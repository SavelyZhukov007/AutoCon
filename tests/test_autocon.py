import os
import subprocess
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from app import config
from app.api import Api
from app.core import device, llm, model_registry, runtime
from app.core.hidden import startup_kwargs
from app.core.vision import (
    EventAggregator,
    FrameShape,
    SignSetInterpreter,
    position_for_bbox,
)


class RuntimeRegistryTests(unittest.TestCase):
    def test_gpu_replaces_cpu_onnxruntime(self):
        pkgs = runtime.packages_for(["plates", "gpu"])
        self.assertIn("onnxruntime-gpu>=1.17", pkgs)
        self.assertNotIn("onnxruntime>=1.17", pkgs)

    def test_packages_are_deduplicated_in_order(self):
        pkgs = runtime.packages_for(["vision", "vision", "tracking"])
        self.assertEqual(pkgs.count("ultralytics>=8.3"), 1)
        self.assertEqual(pkgs[-1], "lap>=0.5.12")

    def test_cuda_warning_does_not_fail_if_torch_imports(self):
        health = runtime.health_check(["gpu"])
        torch_failed = any(item.get("module") == "torch" for item in health["failed"])
        if not torch_failed:
            self.assertTrue(health["ok"])

    def test_all_features_are_deduplicated_and_gpu_replaces_onnxruntime(self):
        pkgs = runtime.packages_for(runtime.ALL_FEATURE_KEYS)
        self.assertEqual(len(pkgs), len(set(pkgs)))
        self.assertIn("onnxruntime-gpu>=1.17", pkgs)
        self.assertNotIn("onnxruntime>=1.17", pkgs)


class HiddenProcessTests(unittest.TestCase):
    def test_windows_startup_kwargs_hide_window(self):
        kwargs = startup_kwargs()
        if os.name == "nt":
            self.assertIn("creationflags", kwargs)
            self.assertIn("startupinfo", kwargs)
        else:
            self.assertEqual(kwargs, {})

    def test_build_includes_runtime_stdlib_hidden_imports(self):
        import build

        commands = []

        def fake_run(cmd, check=True, hidden=False):
            commands.append([str(x) for x in cmd])
            return 0

        old_run = build.run
        old_clean = build.clean_artifacts
        old_write = build.write_build_info
        try:
            build.run = fake_run
            build.clean_artifacts = lambda: None
            build.write_build_info = lambda: "test"
            build.cmd_build(Namespace(onedir=False))
        finally:
            build.run = old_run
            build.clean_artifacts = old_clean
            build.write_build_info = old_write
        joined = " ".join(commands[-1])
        self.assertIn("pickletools", joined)
        self.assertIn("colorsys", joined)

    def test_build_clean_targets_include_appdata_autocon(self):
        import build

        targets = [str(path) for path in build.clean_targets()]
        self.assertIn(str(build.ROOT / "build"), targets)
        self.assertIn(str(build.ROOT / "dist"), targets)
        self.assertIn(str(build.BUILD_META_DIR), targets)
        self.assertTrue(any(path.endswith("AutoCon") for path in targets))


class DevicePolicyTests(unittest.TestCase):
    def test_cpu_preference_wins(self):
        self.assertEqual(
            device.resolve_device({"device": "cpu", "gpu_index": 0}), "cpu"
        )


class ModelRegistryTests(unittest.TestCase):
    def test_model_packs_have_required_fields(self):
        packs = model_registry.list_packs(config.DEFAULTS)
        keys = {p["key"] for p in packs}
        self.assertIn("yolo11s", keys)
        self.assertIn("traffic_signs_100", keys)
        for pack in packs:
            self.assertTrue(pack["title"])
            self.assertTrue(pack["target"])

    def test_yolo11s_requires_real_model_file(self):
        with tempfile.TemporaryDirectory() as td:
            old_model_path = model_registry.model_path
            try:
                model_registry.model_path = lambda name: Path(td) / name
                packs = {
                    pack["key"]: pack
                    for pack in model_registry.list_packs(
                        {**config.DEFAULTS, "yolo_vehicle_model": "yolo11s.pt"}
                    )
                }
            finally:
                model_registry.model_path = old_model_path
        self.assertFalse(packs["yolo11s"]["installed"])

    def test_first_run_readiness_requires_full_log_and_models(self):
        api = Api.__new__(Api)
        api.settings = dict(config.DEFAULTS)

        class FakeCli:
            def available(self):
                return True

            def list_models(self):
                return ["qwen2.5:3b", "qwen2.5vl:3b"]

            def model_in_central_store(self, _name):
                return True

            def central_store_status(self, _names):
                return {"ok": True, "missing": []}

        old_get_llm = Api._get_llm
        import app.core.install as install_module

        old_install_check = install_module.check
        old_health = runtime.health_check
        old_list_packs = model_registry.list_packs
        old_full_log = config.full_log_path
        try:
            Api._get_llm = lambda self: FakeCli()
            install_module.check = lambda: []
            runtime.health_check = lambda _keys=None: {"ok": True, "failed": []}
            model_registry.list_packs = lambda _settings=None: [
                {"key": "yolo11s", "installed": True},
                {"key": "license_plate", "installed": False},
                {"key": "vehicle_dino", "installed": False},
            ]
            with tempfile.TemporaryDirectory() as td:
                missing_log = Path(td) / "ful_log_app.log"
                config.full_log_path = lambda: missing_log
                readiness = api.first_run_readiness()
        finally:
            Api._get_llm = old_get_llm
            install_module.check = old_install_check
            runtime.health_check = old_health
            model_registry.list_packs = old_list_packs
            config.full_log_path = old_full_log
        self.assertFalse(readiness["ok"])
        self.assertFalse(readiness["full_log_exists"])
        self.assertEqual(
            {pack["key"] for pack in readiness["missing_model_packs"]},
            {"license_plate", "vehicle_dino"},
        )

    def test_traffic_sign_pack_prefers_existing_hub_files(self):
        meta = model_registry.PACKS["traffic_signs_100"]
        repos = model_registry.candidate_repos(meta)
        self.assertEqual(repos[0]["repo"], "RZhukotynskyi/sign-detection-yolov8s")
        self.assertIn("sdv4.pt", repos[0]["filenames"])
        self.assertNotIn("sign-detection-yolov8s.pt", repos[0]["filenames"])

    def test_install_pack_downloads_picked_hub_file(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "traffic.pt"
            old_model_path = model_registry.model_path
            old_pick = model_registry.pick_existing_hf_file
            old_run = model_registry.run_hidden
            commands = []

            def fake_model_path(_name):
                return target

            def fake_pick(_python, repo, filenames):
                self.assertIn("sdv4.pt", filenames)
                return {"ok": True, "filename": "sdv4.pt"} if "RZhukotynskyi" in repo else {"ok": False}

            def fake_run(cmd, timeout=None):
                commands.append(" ".join(str(x) for x in cmd))
                target.write_text("weights", encoding="utf-8")
                return subprocess.CompletedProcess(cmd, 0, "")

            try:
                model_registry.model_path = fake_model_path
                model_registry.pick_existing_hf_file = fake_pick
                model_registry.run_hidden = fake_run
                res = model_registry.install_pack("traffic_signs_100", Path("python.exe"))
            finally:
                model_registry.model_path = old_model_path
                model_registry.pick_existing_hf_file = old_pick
                model_registry.run_hidden = old_run

        self.assertTrue(res["ok"])
        self.assertEqual(res["filename"], "sdv4.pt")
        self.assertIn("sdv4.pt", commands[-1])


class EventAggregatorTests(unittest.TestCase):
    def test_signs_are_grouped_by_gap(self):
        agg = EventAggregator(gap_sec=2)
        first = agg.update(
            1.0,
            [
                {
                    "kind": "sign",
                    "label": "speed limit 60",
                    "confidence": 0.8,
                    "position": "top-right",
                }
            ],
        )
        second = agg.update(
            2.0,
            [
                {
                    "kind": "sign",
                    "label": "speed limit 60",
                    "confidence": 0.6,
                    "position": "top-right",
                }
            ],
        )
        third = agg.update(
            5.5,
            [
                {
                    "kind": "sign",
                    "label": "speed limit 60",
                    "confidence": 0.9,
                    "position": "top-right",
                }
            ],
        )
        self.assertEqual(len(first), 1)
        self.assertEqual(second, [])
        self.assertEqual(len(third), 1)
        self.assertEqual(len(agg.sign_sequences), 2)
        self.assertEqual(agg.sign_sequences[0]["count"], 2)

    def test_plate_voting(self):
        agg = EventAggregator()
        agg.update(
            1.0,
            [
                {
                    "kind": "plate",
                    "text": "a123bc77",
                    "confidence": 0.6,
                    "position": "middle-center",
                }
            ],
        )
        agg.update(
            2.0,
            [
                {
                    "kind": "plate",
                    "text": "A123BC77",
                    "confidence": 0.8,
                    "position": "middle-center",
                }
            ],
        )
        plate = agg.result()["plates"][0]
        self.assertEqual(plate["text"], "A123BC77")
        self.assertEqual(plate["count"], 2)

    def test_position_labels(self):
        self.assertEqual(
            position_for_bbox([900, 10, 1000, 100], FrameShape(1000, 500)), "top-right"
        )

    def test_contexts_are_saved_in_result(self):
        agg = EventAggregator()
        agg.add_context({"time": 1, "explanation": "ok"})
        self.assertEqual(agg.result()["contexts"][0]["explanation"], "ok")


class RoadContextTests(unittest.TestCase):
    def test_sign_set_interpreter_marks_clear_lane_applicability(self):
        interp = SignSetInterpreter()
        ctx = interp.interpret(
            3.0,
            [
                {
                    "kind": "sign",
                    "label": "speed limit 60",
                    "confidence": 0.9,
                    "bbox": [760, 80, 860, 180],
                    "position": "top-right",
                }
            ],
            {
                "lane": "center",
                "confidence": 0.6,
                "shape": {"width": 1000, "height": 600},
            },
        )
        self.assertEqual(ctx["applies_to_ego_lane"], "yes")
        self.assertIn("speed limit 60", ctx["explanation"])


class PromptTests(unittest.TestCase):
    def test_scene_prompt_contains_structured_data(self):
        prompt = llm.prompt_scene_commentary({"signs": [{"label": "stop"}]})
        self.assertIn("JSON", prompt)
        self.assertIn("stop", prompt)

    def test_video_chat_prompt_contains_question_and_context(self):
        prompt = llm.prompt_video_chat("что распознано неверно?", {"title": "road", "plates": [{"text": "A123BC77"}]})
        self.assertIn("что распознано неверно?", prompt)
        self.assertIn("A123BC77", prompt)

    def test_exam_photo_prompt_contains_question_findings_and_pdd(self):
        prompt = llm.prompt_exam_photo(
            "можно ли повернуть?",
            {"detections": [{"label": "no left turn"}]},
        )
        self.assertIn("можно ли повернуть?", prompt)
        self.assertIn("no left turn", prompt)
        self.assertIn("ПДД РФ", prompt)

    def test_ollama_generate_sends_images_for_vision_model(self):
        seen = {}

        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {"response": "ok"}

        old_post = llm.requests.post
        try:
            llm.requests.post = lambda _url, json, timeout: seen.update(json) or Response()
            cli = llm.OllamaClient("http://127.0.0.1:11434", model="qwen2.5:3b")
            self.assertEqual(
                cli.generate("prompt", model="qwen2.5vl:3b", images=["abc"]),
                "ok",
            )
        finally:
            llm.requests.post = old_post
        self.assertEqual(seen["model"], "qwen2.5vl:3b")
        self.assertEqual(seen["images"], ["abc"])


if __name__ == "__main__":
    unittest.main()
