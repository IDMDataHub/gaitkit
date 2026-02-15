"""Unit tests for Mickelborough detector parameter guards."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python" / "src"))

from gaitkit.detectors.mickelborough_detector import MickelboroughDetector


class TestMickelboroughDetector(unittest.TestCase):
    def test_constructor_validates_parameters(self):
        with self.assertRaises(ValueError):
            MickelboroughDetector(fps=0)
        with self.assertRaises(ValueError):
            MickelboroughDetector(smooth_window=3)
        with self.assertRaises(ValueError):
            MickelboroughDetector(min_cycle_duration=0)
        with self.assertRaises(ValueError):
            MickelboroughDetector(threshold_fraction=-0.1)

    def test_detect_returns_empty_on_short_sequences(self):
        det = MickelboroughDetector()
        hs, to, cycles = det.detect([])
        self.assertEqual(hs, [])
        self.assertEqual(to, [])
        self.assertEqual(cycles, [])


if __name__ == "__main__":
    unittest.main()
