"""Unit tests for Bayesian BIS V2 detector parameter and short-input guards."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python" / "src"))

from gaitkit.detectors.bayesian_bis_v2 import BayesianBisV2GaitDetector


class TestBayesianBisV2Detector(unittest.TestCase):
    def test_constructor_validates_parameters(self):
        with self.assertRaises(ValueError):
            BayesianBisV2GaitDetector(fps=0)
        with self.assertRaises(ValueError):
            BayesianBisV2GaitDetector(smoothing_window=3)
        with self.assertRaises(ValueError):
            BayesianBisV2GaitDetector(min_crossing_distance=0)
        with self.assertRaises(ValueError):
            BayesianBisV2GaitDetector(rhythm_sigma_ratio=0)

    def test_detect_returns_empty_on_short_sequences(self):
        det = BayesianBisV2GaitDetector()
        hs, to, cycles = det.detect_gait_events([])
        self.assertEqual(hs, [])
        self.assertEqual(to, [])
        self.assertEqual(cycles, [])


if __name__ == "__main__":
    unittest.main()
