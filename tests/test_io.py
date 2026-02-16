"""Unit tests for gaitkit I/O helpers."""

from __future__ import annotations

import sys
import types
import unittest
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python" / "src"))

from gaitkit import _io


class TestLoadC3D(unittest.TestCase):
    def test_load_example_name_must_be_non_empty_string(self):
        with self.assertRaises(ValueError):
            _io.load_example("")
        with self.assertRaises(ValueError):
            _io.load_example(None)  # type: ignore[arg-type]

    def test_load_example_accepts_human_friendly_alias_spelling(self):
        base = _io.load_example("parkinson")
        self.assertEqual(_io.load_example("ParkinSon")["population"], base["population"])
        self.assertEqual(_io.load_example("par kin son")["population"], base["population"])
        self.assertEqual(_io.load_example("par-kin-son")["population"], base["population"])
        self.assertEqual(_io.load_example("par\tkin\nson")["population"], base["population"])

    def test_load_c3d_rejects_missing_or_invalid_path_before_import(self):
        with self.assertRaises(ValueError):
            _io.load_c3d("not_a_c3d.txt")
        with self.assertRaises(ValueError):
            _io.load_c3d("")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            _io.load_c3d(None)  # type: ignore[arg-type]
        with self.assertRaises(FileNotFoundError):
            _io.load_c3d("missing_file.c3d")

    def test_invalid_marker_set_raises(self):
        fake_ezc3d = types.SimpleNamespace(
            c3d=lambda _path: {
                "header": {"points": {"frame_rate": 100.0}},
                "data": {"points": np.zeros((4, 1, 1))},
                "parameters": {"POINT": {"LABELS": {"value": ["LHEE"]}}},
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            c3d_path = Path(tmp) / "dummy.c3d"
            c3d_path.write_bytes(b"")
            with mock.patch.dict(sys.modules, {"ezc3d": fake_ezc3d}):
                with self.assertRaises(ValueError):
                    _io.load_c3d(str(c3d_path), marker_set="unknown")
                with self.assertRaises(ValueError):
                    _io.load_c3d(str(c3d_path), marker_set="")

    def test_auto_detects_imy_marker_set(self):
        labels = ["LCAL", "RCAL", "LFMH1", "RFMH1", "LMM", "LLM", "RMM", "RLM", "LASIS", "RASIS"]
        points = np.zeros((4, len(labels), 3), dtype=float)
        points[0, labels.index("LCAL"), :] = [1.0, 1.1, 1.2]
        points[0, labels.index("RCAL"), :] = [2.0, 2.1, 2.2]
        points[0, labels.index("LFMH1"), :] = [3.0, 3.1, 3.2]
        points[0, labels.index("RFMH1"), :] = [4.0, 4.1, 4.2]

        fake_ezc3d = types.SimpleNamespace(
            c3d=lambda _path: {
                "header": {"points": {"frame_rate": 200.0}},
                "data": {"points": points},
                "parameters": {"POINT": {"LABELS": {"value": labels}}},
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            c3d_path = Path(tmp) / "imy.c3d"
            c3d_path.write_bytes(b"")
            with mock.patch.dict(sys.modules, {"ezc3d": fake_ezc3d}):
                out = _io.load_c3d(str(c3d_path), marker_set="auto")
        lp0 = out["angle_frames"][0]["landmark_positions"]
        self.assertIsNotNone(lp0)
        self.assertIn("left_heel", lp0)
        self.assertIn("right_heel", lp0)
        self.assertIn("left_toe", lp0)
        self.assertIn("right_toe", lp0)

    def test_custom_marker_map_is_supported(self):
        labels = ["XH", "YH", "XT", "YT", "XA", "YA"]
        points = np.zeros((4, len(labels), 2), dtype=float)
        points[:, labels.index("XH"), :] = 1.0
        points[:, labels.index("YH"), :] = 2.0
        points[:, labels.index("XT"), :] = 3.0
        points[:, labels.index("YT"), :] = 4.0
        points[:, labels.index("XA"), :] = 5.0
        points[:, labels.index("YA"), :] = 6.0

        fake_ezc3d = types.SimpleNamespace(
            c3d=lambda _path: {
                "header": {"points": {"frame_rate": 100.0}},
                "data": {"points": points},
                "parameters": {"POINT": {"LABELS": {"value": labels}}},
            }
        )
        marker_map = {
            "left_heel": "XH",
            "right_heel": "YH",
            "left_toe": "XT",
            "right_toe": "YT",
            "left_ankle": "XA",
            "right_ankle": "YA",
        }
        with tempfile.TemporaryDirectory() as tmp:
            c3d_path = Path(tmp) / "custom.c3d"
            c3d_path.write_bytes(b"")
            with mock.patch.dict(sys.modules, {"ezc3d": fake_ezc3d}):
                out = _io.load_c3d(str(c3d_path), marker_map=marker_map)
        lp0 = out["angle_frames"][0]["landmark_positions"]
        self.assertIn("left_heel", lp0)
        self.assertIn("right_toe", lp0)

    def test_no_supported_marker_error_lists_detected_labels(self):
        labels = ["A", "B", "C"]
        points = np.zeros((4, len(labels), 2), dtype=float)
        fake_ezc3d = types.SimpleNamespace(
            c3d=lambda _path: {
                "header": {"points": {"frame_rate": 100.0}},
                "data": {"points": points},
                "parameters": {"POINT": {"LABELS": {"value": labels}}},
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            c3d_path = Path(tmp) / "none.c3d"
            c3d_path.write_bytes(b"")
            with mock.patch.dict(sys.modules, {"ezc3d": fake_ezc3d}):
                with self.assertRaises(ValueError) as ctx:
                    _io.load_c3d(str(c3d_path), marker_set="auto")
        self.assertIn("Detected labels", str(ctx.exception))

    def test_angle_extraction_failure_is_non_blocking(self):
        # Include an angle label that does not exist in points data to force IndexError
        labels = ["LHEE", "RHEE", "LTOE", "RTOE", "LKneeAngles"]
        points = np.zeros((4, 4, 2), dtype=float)  # only 4 markers available

        fake_ezc3d = types.SimpleNamespace(
            c3d=lambda _path: {
                "header": {"points": {"frame_rate": 100.0}},
                "data": {"points": points},
                "parameters": {"POINT": {"LABELS": {"value": labels}}},
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            c3d_path = Path(tmp) / "dummy.c3d"
            c3d_path.write_bytes(b"")
            with mock.patch.dict(sys.modules, {"ezc3d": fake_ezc3d}):
                with mock.patch.object(_io.logger, "debug") as debug_mock:
                    out = _io.load_c3d(str(c3d_path), marker_set="pig")
        self.assertEqual(out["n_frames"], 2)
        self.assertEqual(len(out["angle_frames"]), 2)
        self.assertTrue(debug_mock.called)


if __name__ == "__main__":
    unittest.main()
