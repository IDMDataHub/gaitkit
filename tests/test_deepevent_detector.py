"""Unit tests for DeepEvent detector constructor and short-input guards."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python" / "src"))

from gaitkit.detectors.deepevent_detector import DeepEventDetector


class TestDeepEventDetector(unittest.TestCase):
    def test_constructor_validates_parameters(self):
        with self.assertRaises(ValueError):
            DeepEventDetector(fps=0)
        with self.assertRaises(ValueError):
            DeepEventDetector(event_threshold=-0.1)
        with self.assertRaises(ValueError):
            DeepEventDetector(event_threshold=1.1)
        with self.assertRaises(ValueError):
            DeepEventDetector(filter_cutoff=0)

    def test_detect_returns_empty_on_short_sequences(self):
        det = DeepEventDetector()
        hs, to, cycles = det.detect([])
        self.assertEqual(hs, [])
        self.assertEqual(to, [])
        self.assertEqual(cycles, [])


if __name__ == "__main__":
    unittest.main()
