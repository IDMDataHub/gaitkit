"""Unit tests for DeepEvent weight management helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python" / "src"))

from gaitkit.detectors import deepevent_detector as dd


class TestDeepEventWeightsHelpers(unittest.TestCase):
    def test_hdf5_signature_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp) / "good.h5"
            bad = Path(tmp) / "bad.h5"

            good.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 16)
            bad.write_bytes(b"NOT_HDF5_CONTENT")

            self.assertTrue(dd._is_hdf5_file(good))
            self.assertFalse(dd._is_hdf5_file(bad))

    def test_weights_urls_prioritize_project_repo(self):
        self.assertGreaterEqual(len(dd._WEIGHTS_URLS), 1)
        self.assertIn("IDMDataHub/gaitkit", dd._WEIGHTS_URLS[0])
        self.assertIn("DeepEventWeight.h5", dd._WEIGHTS_URLS[0])

    def test_default_cache_path_location(self):
        self.assertEqual(dd._DEFAULT_WEIGHT_PATH.name, "DeepEventWeight.h5")
        self.assertIn(".cache", str(dd._DEFAULT_WEIGHT_PATH))


if __name__ == "__main__":
    unittest.main()
