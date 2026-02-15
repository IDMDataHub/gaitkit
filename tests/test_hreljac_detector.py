"""Unit tests for Hreljac detector parameter guards."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python" / "src"))

from gaitkit.detectors.hreljac_detector import HreljacDetector


class TestHreljacDetector(unittest.TestCase):
    def test_constructor_validates_parameters(self):
        with self.assertRaises(ValueError):
            HreljacDetector(fps=0)
        with self.assertRaises(ValueError):
            HreljacDetector(min_cycle_duration=0)

    def test_detect_returns_empty_on_short_sequences(self):
        det = HreljacDetector()
        hs, to, cycles = det.detect([])
        self.assertEqual(hs, [])
        self.assertEqual(to, [])
        self.assertEqual(cycles, [])


if __name__ == "__main__":
    unittest.main()
