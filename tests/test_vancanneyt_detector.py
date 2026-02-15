"""Unit tests for Vancanneyt detector parameter and short-input guards."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python" / "src"))

from gaitkit.detectors.vancanneyt_detector import VancanneytDetector


class TestVancanneytDetector(unittest.TestCase):
    def test_constructor_validates_parameters(self):
        with self.assertRaises(ValueError):
            VancanneytDetector(fps=0)
        with self.assertRaises(ValueError):
            VancanneytDetector(filter_cutoff=0)
        with self.assertRaises(ValueError):
            VancanneytDetector(vx_threshold=-0.1)
        with self.assertRaises(ValueError):
            VancanneytDetector(vy_threshold=-0.1)
        with self.assertRaises(ValueError):
            VancanneytDetector(vz_threshold=-0.1)
        with self.assertRaises(ValueError):
            VancanneytDetector(windowing_mph_coeff=-0.1)
        with self.assertRaises(ValueError):
            VancanneytDetector(windowing_mpd=0)

    def test_detect_returns_empty_on_short_sequences(self):
        det = VancanneytDetector()
        hs, to, cycles = det.detect([])
        self.assertEqual(hs, [])
        self.assertEqual(to, [])
        self.assertEqual(cycles, [])


if __name__ == "__main__":
    unittest.main()
