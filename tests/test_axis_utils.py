"""Unit tests for axis auto-detection helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python" / "src"))

from gaitkit.detectors import axis_utils as au


class _Frame:
    def __init__(self, lp):
        self.landmark_positions = lp


class TestAxisUtils(unittest.TestCase):
    def test_detect_axes_defaults_on_empty_input(self):
        self.assertEqual(au.detect_axes([]), (1, 2))

    def test_detect_axes_supports_dict_frames(self):
        frames = []
        for i in range(20):
            frames.append({
                "landmark_positions": {
                    "left_ankle": (0.0, float(i), 10.0 + 0.1 * i),
                }
            })
        ap, vert = au.detect_axes(frames)
        self.assertEqual(ap, 1)
        self.assertIn(vert, (0, 2))

    def test_detect_walking_direction_rejects_invalid_ap_axis(self):
        with self.assertRaises(ValueError):
            au.detect_walking_direction([], ap_axis=3)

    def test_direction_score_handles_non_positive_fps(self):
        frames = [_Frame({"left_ankle": (0.0, float(i), 0.0),
                          "right_ankle": (0.0, float(i), 0.0),
                          "left_hip": (0.0, float(i), 0.0),
                          "right_hip": (0.0, float(i), 0.0)}) for i in range(25)]
        score = au._direction_score_acceleration(frames, ap_axis=1, direction=1, fps=0)
        self.assertIsInstance(score, float)


if __name__ == "__main__":
    unittest.main()
