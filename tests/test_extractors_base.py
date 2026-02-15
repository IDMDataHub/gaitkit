"""Unit tests for shared extractor utilities and base-class guards."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python" / "src"))

import gaitkit.extractors as extractors
from gaitkit.extractors.base_extractor import BaseExtractor, compute_signed_angle_2d


class _DummyExtractor(BaseExtractor):
    @property
    def name(self) -> str:
        return "dummy"

    @property
    def description(self) -> str:
        return "dummy extractor"

    def list_files(self):
        return [self.data_dir / "a", self.data_dir / "b"]

    def extract_file(self, filepath):
        return filepath


class TestBaseExtractor(unittest.TestCase):
    def test_constructor_validates_data_dir(self):
        with self.assertRaises(ValueError):
            _DummyExtractor("")
        with self.assertRaises(ValueError):
            _DummyExtractor("/definitely/missing/path")

    def test_extract_all_validates_max_files(self):
        with tempfile.TemporaryDirectory() as td:
            ex = _DummyExtractor(td)
            with self.assertRaises(ValueError):
                ex.extract_all(max_files=0)
            with self.assertRaises(ValueError):
                ex.extract_all(max_files=-1)
            with self.assertRaises(ValueError):
                ex.extract_all(max_files=1.5)
            out = ex.extract_all(max_files=1)
            self.assertEqual(len(out), 1)


class TestAngleHelpers(unittest.TestCase):
    def test_compute_signed_angle_is_wrapped_to_documented_range(self):
        v1 = np.array([-1.0, 0.0])  # 180 degrees
        v2 = np.array([0.0, -1.0])  # -90 degrees
        angle = compute_signed_angle_2d(v1, v2)
        self.assertGreaterEqual(angle, -180.0)
        self.assertLessEqual(angle, 180.0)
        self.assertAlmostEqual(angle, 90.0, places=7)


class TestExtractorModuleImport(unittest.TestCase):
    def test_extractors_module_exposes_base_without_optional_dependencies(self):
        self.assertTrue(hasattr(extractors, "BaseExtractor"))
        self.assertIn("BaseExtractor", extractors.__all__)


if __name__ == "__main__":
    unittest.main()
