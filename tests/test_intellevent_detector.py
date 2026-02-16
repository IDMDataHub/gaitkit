"""Unit tests for IntellEvent detector guards independent from ONNX runtime."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python" / "src"))

from gaitkit.detectors.intellevent_detector import IntellEventDetector, _looks_like_lfs_pointer


class TestIntellEventDetector(unittest.TestCase):
    def test_constructor_validates_fps_before_onnx_dependency(self):
        with self.assertRaises(ValueError):
            IntellEventDetector(fps=0)

    def test_side_helpers_clip_index_on_short_signal(self):
        det = IntellEventDetector.__new__(IntellEventDetector)
        l = [0.1, 0.2]
        r = [0.3, 0.0]
        self.assertEqual(det._determine_side_ic(99, l, r), "right")
        self.assertEqual(det._determine_side_fo(99, l, r), "left")

    def test_lfs_pointer_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "model.onnx"
            p.write_text("version https://git-lfs.github.com/spec/v1\n")
            self.assertTrue(_looks_like_lfs_pointer(p))

    def test_binary_model_is_not_lfs_pointer(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "model.onnx"
            p.write_bytes(b"\x08\x08\x12\x07tf2onnx")
            self.assertFalse(_looks_like_lfs_pointer(p))


if __name__ == "__main__":
    unittest.main()
