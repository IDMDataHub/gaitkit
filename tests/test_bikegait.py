from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_ROOT))

import gaitkit
from src.extractors import FukuchiExtractor, NatureC3DExtractor


def _load_one_real_extraction():
    candidates = [
        (
            "nature",
            NatureC3DExtractor,
            PROJECT_ROOT / "data/nature/extracted/A multimodal dataset of human gait at different walking speeds",
        ),
        ("fukuchi", FukuchiExtractor, PROJECT_ROOT / "data/fukuchi"),
    ]
    for name, extractor_cls, data_path in candidates:
        if not data_path.exists():
            continue
        try:
            extractor = extractor_cls(str(data_path))
            files = extractor.list_files()
            if not files:
                continue
            extraction = extractor.extract_file(files[0])
            if extraction.angle_frames and extraction.fps > 0:
                return name, extraction
        except Exception:
            continue
    return None, None


class TestgaitkitUnit(unittest.TestCase):
    def test_list_methods_contract(self):
        methods = gaitkit.list_methods()
        self.assertIsInstance(methods, list)
        self.assertNotIn("intellevent", methods)
        for required in [
            "bayesian_bis",
            "zeni",
            "oconnor",
            "hreljac",
            "mickelborough",
            "ghoussayni",
            "dgei",
            "vancanneyt",
        ]:
            self.assertIn(required, methods)

    def test_invalid_fps_raises(self):
        with self.assertRaises(ValueError):
            gaitkit.detect_events_structured("bayesian_bis", [], fps=0.0)

    def test_units_scaling_m_and_rad(self):
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


class TestgaitkitIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset_name, cls.extraction = _load_one_real_extraction()

    def _require_real_extraction(self):
        if self.extraction is None:
            self.skipTest("No real dataset available locally for integration test")

    def test_detect_all_methods_on_real_file(self):
        self._require_real_extraction()
        frames = self.extraction.angle_frames
        fps = float(self.extraction.fps)
        n = len(frames)
        executed = 0
        for method in gaitkit.list_methods():
            try:
                payload = gaitkit.detect_events_structured(method, frames, fps)
            except Exception as exc:
                msg = str(exc).lower()
                if "no module named" in msg or "keras" in msg or "onnxruntime" in msg:
                    continue
                raise
            executed += 1
            self.assertIn("meta", payload)
            self.assertIn("heel_strikes", payload)
            self.assertIn("toe_offs", payload)
            self.assertIn("cycles", payload)
            events = list(payload["heel_strikes"]) + list(payload["toe_offs"])
            for ev in events:
                fi = int(ev["frame_index"])
                self.assertGreaterEqual(fi, 0)
                self.assertLess(fi, n)
        self.assertGreater(executed, 0)

    def test_output_json_and_csv_on_real_file(self):
        self._require_real_extraction()
        payload = gaitkit.detect_events_structured(
            "bayesian_bis",
            self.extraction.angle_frames,
            float(self.extraction.fps),
        )
        with tempfile.TemporaryDirectory(prefix="bikegait_test_") as tmp:
            prefix = Path(tmp) / "trial01"
            paths = gaitkit.export_detection(payload, prefix, formats=("json", "csv"))
            self.assertIn("json", paths)
            self.assertIn("csv_events", paths)
            self.assertIn("csv_cycles", paths)
            self.assertTrue(Path(paths["json"]).exists())
            self.assertTrue(Path(paths["csv_events"]).exists())
            self.assertTrue(Path(paths["csv_cycles"]).exists())
            data = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
            self.assertEqual(data["meta"]["detector"], "bayesian_bis")
            self.assertGreater(float(data["meta"]["fps_hz"]), 0.0)

    def test_output_xlsx_dependency_behavior(self):
        self._require_real_extraction()
        payload = gaitkit.detect_events_structured(
            "bayesian_bis",
            self.extraction.angle_frames,
            float(self.extraction.fps),
        )
        with tempfile.TemporaryDirectory(prefix="bikegait_test_") as tmp:
            prefix = Path(tmp) / "trial01"
            try:
                paths = gaitkit.export_detection(payload, prefix, formats=("xlsx",))
                self.assertTrue(Path(paths["xlsx"]).exists())
            except RuntimeError as exc:
                self.assertIn("openpyxl", str(exc).lower())


if __name__ == "__main__":
    unittest.main()
