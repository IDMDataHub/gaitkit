"""Tests for ensemble API behavior and method alias support."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python" / "src"))

import gaitkit


class TestEnsembleAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trial = gaitkit.load_example("healthy")

    def test_accepts_bike_alias_in_methods(self):
        result = gaitkit.detect_ensemble(
            self.trial,
            methods=["bike", "zeni", "oconnor"],
            min_votes=2,
        )
        self.assertEqual(result.method, "ensemble")
        self.assertGreaterEqual(len(result.heel_strikes) + len(result.toe_offs), 0)

    def test_invalid_method_name_raises(self):
        with self.assertRaises(ValueError):
            gaitkit.detect_ensemble(self.trial, methods=["unknown", "zeni"])


if __name__ == "__main__":
    unittest.main()
