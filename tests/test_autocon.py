import os
import unittest

from app import config
from app.core import device, llm, model_registry, runtime
from app.core.hidden import startup_kwargs
from app.core.vision import EventAggregator, position_for_bbox, FrameShape


class RuntimeRegistryTests(unittest.TestCase):
    def test_gpu_replaces_cpu_onnxruntime(self):
        pkgs = runtime.packages_for(["plates", "gpu"])
        self.assertIn("onnxruntime-gpu>=1.17", pkgs)
        self.assertNotIn("onnxruntime>=1.17", pkgs)

    def test_packages_are_deduplicated_in_order(self):
        pkgs = runtime.packages_for(["vision", "vision", "tracking"])
        self.assertEqual(pkgs.count("ultralytics>=8.3"), 1)
        self.assertEqual(pkgs[-1], "lap>=0.5.12")


class HiddenProcessTests(unittest.TestCase):
    def test_windows_startup_kwargs_hide_window(self):
        kwargs = startup_kwargs()
        if os.name == "nt":
            self.assertIn("creationflags", kwargs)
            self.assertIn("startupinfo", kwargs)
        else:
            self.assertEqual(kwargs, {})


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


class PromptTests(unittest.TestCase):
    def test_scene_prompt_contains_structured_data(self):
        prompt = llm.prompt_scene_commentary({"signs": [{"label": "stop"}]})
        self.assertIn("JSON", prompt)
        self.assertIn("stop", prompt)


if __name__ == "__main__":
    unittest.main()
