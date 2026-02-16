"""Unit tests for DeepEvent weight management helpers."""

from __future__ import annotations

import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock
import os

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
        self.assertTrue(
            (".cache" in str(dd._DEFAULT_WEIGHT_PATH)) or ("/tmp/" in str(dd._DEFAULT_WEIGHT_PATH))
        )

    def test_resolve_cache_dir_prefers_env_var(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"GAITKIT_CACHE_DIR": tmp}):
                cache_dir = dd._resolve_cache_dir()
        self.assertEqual(str(cache_dir), tmp)

    def test_hdf5_signature_missing_file_returns_false(self):
        missing = Path(tempfile.gettempdir()) / "does_not_exist_deepevent.h5"
        self.assertFalse(dd._is_hdf5_file(missing))

    def test_download_weights_success_with_mocked_response(self):
        content = b"\x89HDF\r\n\x1a\n" + b"\x00" * 64

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
            target = Path(tmp) / "DeepEventWeight.h5"
            with mock.patch.object(dd, "_WEIGHTS_URLS", ["https://example.test/w.h5"]):
                with mock.patch.object(dd.urllib.request, "urlopen", return_value=_Resp(content)):
                    out = dd._download_deepevent_weights(target)
            self.assertEqual(out, target)
            self.assertTrue(target.exists())
            self.assertTrue(dd._is_hdf5_file(target))

    def test_download_weights_returns_none_on_invalid_payload(self):
        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self, n: int = -1):
                # return bad bytes once, then EOF
                if getattr(self, "_done", False):
                    return b""
                self._done = True
                return b"not_hdf5"

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "DeepEventWeight.h5"
            with mock.patch.object(dd, "_WEIGHTS_URLS", ["https://example.test/w_bad.h5"]):
                with mock.patch.object(dd.urllib.request, "urlopen", return_value=_Resp()):
                    out = dd._download_deepevent_weights(target)
            self.assertIsNone(out)
            self.assertFalse(target.exists())

    def test_download_weights_returns_none_on_url_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "DeepEventWeight.h5"
            with mock.patch.object(dd, "_WEIGHTS_URLS", ["https://example.test/w_missing.h5"]):
                with mock.patch.object(
                    dd.urllib.request,
                    "urlopen",
                    side_effect=urllib.error.URLError("network down"),
                ):
                    out = dd._download_deepevent_weights(target)
            self.assertIsNone(out)
            self.assertFalse(target.exists())

    def test_download_weights_returns_none_if_cache_dir_creation_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "nested" / "DeepEventWeight.h5"
            with mock.patch("pathlib.Path.mkdir", side_effect=OSError("permission denied")):
                out = dd._download_deepevent_weights(target)
            self.assertIsNone(out)


if __name__ == "__main__":
    unittest.main()
