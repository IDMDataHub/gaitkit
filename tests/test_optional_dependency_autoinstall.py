"""Tests for optional dependency auto-install behavior in core detector factory."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python" / "src"))

from gaitkit import _core


class _FakeIntellEventDetector:
    def __init__(self, fps: float = 100.0):
        self.fps = fps


class TestAutoInstallSwitch(unittest.TestCase):
    def test_env_false_disables_auto_install(self):
        with patch.dict(os.environ, {"GAITKIT_AUTO_INSTALL_DEPS": "0"}):
            self.assertFalse(_core._auto_install_enabled(None))

    def test_env_true_enables_auto_install(self):
        with patch.dict(os.environ, {"GAITKIT_AUTO_INSTALL_DEPS": "true"}):
            self.assertTrue(_core._auto_install_enabled(None))

    def test_explicit_argument_overrides_env(self):
        with patch.dict(os.environ, {"GAITKIT_AUTO_INSTALL_DEPS": "0"}):
            self.assertTrue(_core._auto_install_enabled(True))


class TestFactoryAutoInstall(unittest.TestCase):
    def test_intellevent_triggers_auto_install_then_retries_import(self):
        fake_module = SimpleNamespace(IntellEventDetector=_FakeIntellEventDetector)
        with patch.object(_core, "_install_optional_dependencies") as install_mock:
            with patch("importlib.import_module", side_effect=[ModuleNotFoundError("onnxruntime"), fake_module]):
                det = _core._make_detector("intellevent", 120.0, auto_install_deps=True)
        install_mock.assert_called_once_with("intellevent")
        self.assertIsInstance(det, _FakeIntellEventDetector)
        self.assertEqual(det.fps, 120.0)

    def test_intellevent_without_auto_install_returns_actionable_error(self):
        with patch("importlib.import_module", side_effect=ModuleNotFoundError("onnxruntime")):
            with self.assertRaises(ImportError) as ctx:
                _core._make_detector("intellevent", 100.0, auto_install_deps=False)
        self.assertIn("gaitkit[onnx]", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
