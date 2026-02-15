"""Unit tests for Bayesian BIS detector parameter and short-input guards."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python" / "src"))

from gaitkit.detectors.bayesian_bis import BayesianBisGaitDetector


class TestBayesianBisDetector(unittest.TestCase):
    def test_constructor_validates_parameters(self):
        with self.assertRaises(ValueError):
            BayesianBisGaitDetector(fps=0)
        with self.assertRaises(ValueError):
            BayesianBisGaitDetector(smoothing_window=3)
        with self.assertRaises(ValueError):
            BayesianBisGaitDetector(min_crossing_distance=0)
        with self.assertRaises(ValueError):
            BayesianBisGaitDetector(rhythm_sigma_ratio=0)

    def test_detect_returns_empty_on_short_sequences(self):
        det = BayesianBisGaitDetector()
        hs, to, cycles = det.detect_gait_events([])
        self.assertEqual(hs, [])
        self.assertEqual(to, [])
        self.assertEqual(cycles, [])


if __name__ == "__main__":
    unittest.main()
