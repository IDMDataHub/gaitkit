"""Unit tests for visualization input validation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python" / "src"))

from gaitkit import _viz


class _DummyResult:
    def __init__(self, fps=100.0, n_frames=10, angle_frames=None):
        self.fps = fps
        self.n_frames = n_frames
        self.method = "bike"
        self._angle_frames = angle_frames
        self.left_hs = []
        self.right_hs = []
        self.left_to = []
        self.right_to = []


class TestVizValidation(unittest.TestCase):
    def test_compare_plot_rejects_empty_methods_before_mpl_import(self):
        with mock.patch.object(_viz, "_import_mpl", side_effect=AssertionError("should not import")):
            with self.assertRaises(ValueError):
                _viz.compare_plot(data=[], methods=[])

    def test_compare_plot_rejects_invalid_fps_before_mpl_import(self):
        with mock.patch.object(_viz, "_import_mpl", side_effect=AssertionError("should not import")):
            with self.assertRaises(ValueError):
                _viz.compare_plot(data=[], methods=["bike"], fps=0)

    def test_plot_result_rejects_invalid_result_before_mpl_import(self):
        with mock.patch.object(_viz, "_import_mpl", side_effect=AssertionError("should not import")):
            with self.assertRaises(ValueError):
                _viz.plot_result(_DummyResult(fps=0, n_frames=10))
            with self.assertRaises(ValueError):
                _viz.plot_result(_DummyResult(fps=100, n_frames=0))
            with self.assertRaises(ValueError):
                _viz.plot_result(_DummyResult(), signals=[])
            with self.assertRaises(ValueError):
                _viz.plot_result(_DummyResult(), signals=["left_knee_angle", ""])

    def test_plot_cycles_requires_non_empty_variable_and_frames(self):
        with mock.patch.object(_viz, "_import_mpl", side_effect=AssertionError("should not import")):
            with self.assertRaises(ValueError):
                _viz.plot_cycles(_DummyResult(angle_frames=[{"left_knee_angle": 1.0}]), variable="")
            with self.assertRaises(ValueError):
                _viz.plot_cycles(_DummyResult(angle_frames=None))


if __name__ == "__main__":
    unittest.main()
