"""Unit tests for DGEI detector parameter and short-input guards."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python" / "src"))

from gaitkit.detectors.dgei_detector import DGEIDetector


class TestDGEIDetector(unittest.TestCase):
    def test_constructor_validates_parameters(self):
        with self.assertRaises(ValueError):
            DGEIDetector(fps=0)
        with self.assertRaises(ValueError):
            DGEIDetector(sleep_frames=0)
        with self.assertRaises(ValueError):
            DGEIDetector(bar_threshold=-0.1)
        with self.assertRaises(ValueError):
            DGEIDetector(peak_ratio=0)
        with self.assertRaises(ValueError):
            DGEIDetector(queue_length=0)

    def test_short_sequence_returns_empty_events_and_error_debug(self):
        det = DGEIDetector()
        hs, to, dbg = det.detect_gait_events([])
        self.assertEqual(hs, [])
        self.assertEqual(to, [])
        self.assertEqual(dbg.get("error"), "Too few frames")


if __name__ == "__main__":
    unittest.main()
