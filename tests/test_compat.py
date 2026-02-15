"""Unit tests for legacy compatibility helpers."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python" / "src"))

from gaitkit import _compat


class TestCompatHelpers(unittest.TestCase):
    def test_normalize_units_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            _compat._normalize_units({"position": "cm"})
        with self.assertRaises(ValueError):
            _compat._normalize_units({"angles": "grad"})
        with self.assertRaises(ValueError):
            _compat._normalize_units(["mm", "deg"])

    def test_build_angle_frames_rejects_string_input(self):
        with self.assertRaises(ValueError):
            _compat.build_angle_frames("not-a-sequence-of-frames")

    def test_export_detection_rejects_empty_and_unknown_formats(self):
        payload = {"heel_strikes": [], "toe_offs": [], "cycles": []}
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "trial"
            with self.assertRaises(ValueError):
                _compat.export_detection(payload, prefix, formats=())
            with self.assertRaises(ValueError):
                _compat.export_detection(payload, prefix, formats=("json", "xml"))


if __name__ == "__main__":
    unittest.main()
