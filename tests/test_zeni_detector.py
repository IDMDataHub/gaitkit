"""Unit tests for Zeni detector parameter guards."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python" / "src"))

from gaitkit.detectors.zeni_detector import ZeniDetector


class TestZeniDetector(unittest.TestCase):
    def test_constructor_validates_parameters(self):
        with self.assertRaises(ValueError):
            ZeniDetector(fps=0)
        with self.assertRaises(ValueError):
            ZeniDetector(min_cycle_duration=0)
        with self.assertRaises(ValueError):
            ZeniDetector(max_cycle_duration=0)
        with self.assertRaises(ValueError):
            ZeniDetector(min_cycle_duration=1.0, max_cycle_duration=0.5)

    def test_detect_returns_empty_on_short_sequences(self):
        det = ZeniDetector()
        hs, to, cycles = det.detect([])
        self.assertEqual(hs, [])
        self.assertEqual(to, [])
        self.assertEqual(cycles, [])


if __name__ == "__main__":
    unittest.main()
