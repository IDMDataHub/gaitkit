"""Unit tests for IntellEvent detector guards independent from ONNX runtime."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python" / "src"))

from gaitkit.detectors.intellevent_detector import IntellEventDetector


class TestIntellEventDetector(unittest.TestCase):
    def test_constructor_validates_fps_before_onnx_dependency(self):
        with self.assertRaises(ValueError):
            IntellEventDetector(fps=0)

    def test_side_helpers_clip_index_on_short_signal(self):
        det = IntellEventDetector.__new__(IntellEventDetector)
        l = [0.1, 0.2]
        r = [0.3, 0.0]
        self.assertEqual(det._determine_side_ic(99, l, r), "right")
        self.assertEqual(det._determine_side_fo(99, l, r), "left")


if __name__ == "__main__":
    unittest.main()
