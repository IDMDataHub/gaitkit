"""Unit tests for gaitkit I/O helpers."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python" / "src"))

from gaitkit import _io


class TestLoadC3D(unittest.TestCase):
    def test_invalid_marker_set_raises(self):
        fake_ezc3d = types.SimpleNamespace(
            c3d=lambda _path: {
                "header": {"points": {"frame_rate": 100.0}},
                "data": {"points": np.zeros((4, 1, 1))},
                "parameters": {"POINT": {"LABELS": {"value": ["LHEE"]}}},
            }
        )
        with mock.patch.dict(sys.modules, {"ezc3d": fake_ezc3d}):
            with self.assertRaises(ValueError):
                _io.load_c3d("dummy.c3d", marker_set="unknown")

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

        with mock.patch.dict(sys.modules, {"ezc3d": fake_ezc3d}):
            with mock.patch.object(_io.logger, "debug") as debug_mock:
                out = _io.load_c3d("dummy.c3d", marker_set="pig")
        self.assertEqual(out["n_frames"], 2)
        self.assertEqual(len(out["angle_frames"]), 2)
        self.assertTrue(debug_mock.called)


if __name__ == "__main__":
    unittest.main()
