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
