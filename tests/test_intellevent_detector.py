"""Unit tests for IntellEvent detector guards independent from ONNX runtime."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
import tempfile
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python" / "src"))

from gaitkit.detectors.intellevent_detector import (
    IntellEventDetector,
    _looks_like_lfs_pointer,
    _download_intellevent_model,
)


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

    def test_download_intellevent_model_success_with_mocked_response(self):
        payload = b"\x08\x08\x12\x07tf2onnx" + b"\x00" * 64

        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def __init__(self, data: bytes):
                self._data = data
                self._idx = 0

            def read(self, n: int = -1):
                if self._idx >= len(self._data):
                    return b""
                if n < 0:
                    n = len(self._data) - self._idx
                out = self._data[self._idx:self._idx + n]
                self._idx += len(out)
                return out

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "ic_intellevent.onnx"
            with mock.patch(
                "gaitkit.detectors.intellevent_detector._INTELLEVENT_MODEL_URLS",
                {"ic_intellevent.onnx": ["https://example.test/ic.onnx"]},
            ):
                with mock.patch(
                    "gaitkit.detectors.intellevent_detector.urllib.request.urlopen",
                    return_value=_Resp(payload),
                ):
                    out = _download_intellevent_model(target, "ic_intellevent.onnx")
            self.assertEqual(out, target)
            self.assertTrue(target.exists())
            self.assertFalse(_looks_like_lfs_pointer(target))


if __name__ == "__main__":
    unittest.main()
