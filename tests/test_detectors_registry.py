"""Tests for detector registry helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python" / "src"))

from gaitkit.detectors import get_detector, list_detectors


class TestDetectorRegistry(unittest.TestCase):
    def test_get_detector_validates_name(self):
        with self.assertRaises(ValueError):
            get_detector("")
        with self.assertRaises(ValueError):
            get_detector(None)  # type: ignore[arg-type]

    def test_get_detector_normalizes_whitespace(self):
        det = get_detector(" zeni ", fps=100.0)
        self.assertIsNotNone(det)

    def test_list_detectors_contains_bayesian_bis(self):
        self.assertIn("bayesian_bis", list_detectors())


if __name__ == "__main__":
    unittest.main()
