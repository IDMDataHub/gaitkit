"""Regression and compatibility tests for built-in example data."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, "/home/ffer/gaitkit/python/src")

import gaitkit


class TestLegacyCompatibility(unittest.TestCase):
    def test_build_angle_frames_unit_scaling(self):
        frames = [
            {
                "frame_index": 0,
                "left_knee_angle": 1.0,
                "landmark_positions": {"left_ankle": (1.0, 0.0, 0.0)},
            }
        ]
        out = gaitkit.build_angle_frames(
            frames,
            units={"position": "m", "angles": "rad"},
        )
        self.assertEqual(len(out), 1)
        f = out[0]
        self.assertAlmostEqual(f.left_knee_angle, 57.295779513, places=6)
        self.assertAlmostEqual(f.landmark_positions["left_ankle"][0], 1000.0, places=6)

    def test_detect_events_structured_and_export(self):
        trial = gaitkit.load_example("healthy")
        payload = gaitkit.detect_events_structured(
            "bike",
            trial["angle_frames"],
            trial["fps"],
            units={"position": "mm", "angles": "deg"},
        )
        self.assertIn("meta", payload)
        self.assertIn("heel_strikes", payload)
        self.assertIn("toe_offs", payload)
        self.assertIn("cycles", payload)
        self.assertEqual(payload["meta"]["detector"], "bayesian_bis")

        with tempfile.TemporaryDirectory(prefix="gaitkit_test_") as tmp:
            prefix = Path(tmp) / "trial01"
            paths = gaitkit.export_detection(payload, prefix, formats=("json", "csv"))
            self.assertTrue(Path(paths["json"]).exists())
            self.assertTrue(Path(paths["csv_events"]).exists())
            self.assertTrue(Path(paths["csv_cycles"]).exists())
            data = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
            self.assertEqual(data["meta"]["detector"], "bayesian_bis")


class TestExampleBikeRegression(unittest.TestCase):
    def test_bike_outputs_are_stable_on_bundled_examples(self):
        expected_path = Path(__file__).parent / "expected_example_bike_regression.json"
        expected = json.loads(expected_path.read_text(encoding="utf-8"))

        for example_name, exp in expected.items():
            with self.subTest(example=example_name):
                trial = gaitkit.load_example(example_name)
                result = gaitkit.detect(trial, method="bike")
                got = {
                    "fps": trial.get("fps"),
                    "n_frames": trial.get("n_frames", len(trial["angle_frames"])),
                    "left_hs": [e["frame"] for e in result.left_hs],
                    "right_hs": [e["frame"] for e in result.right_hs],
                    "left_to": [e["frame"] for e in result.left_to],
                    "right_to": [e["frame"] for e in result.right_to],
                }
                self.assertEqual(got, exp)


if __name__ == "__main__":
    unittest.main()
