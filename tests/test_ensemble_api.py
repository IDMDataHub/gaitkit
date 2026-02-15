"""Tests for ensemble API behavior and method alias support."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python" / "src"))

import gaitkit
from gaitkit import _ensemble


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
        with self.assertRaises(ValueError):
            gaitkit.detect_ensemble(self.trial, methods=["bike", "   "])
        with self.assertRaises(ValueError):
            gaitkit.detect_ensemble(self.trial, methods=["bike", None])  # type: ignore[list-item]

    def test_rejects_invalid_numeric_parameters(self):
        with self.assertRaises(ValueError):
            gaitkit.detect_ensemble(self.trial, methods=["bike", "zeni"], fps=0)
        with self.assertRaises(ValueError):
            gaitkit.detect_ensemble(self.trial, methods=["bike", "zeni"], min_votes=0)
        with self.assertRaises(ValueError):
            gaitkit.detect_ensemble(self.trial, methods=["bike", "zeni"], tolerance_ms=-1)
        with self.assertRaises(ValueError):
            gaitkit.detect_ensemble(self.trial, methods="bike")

    def test_rejects_invalid_custom_weights(self):
        with self.assertRaises(ValueError):
            gaitkit.detect_ensemble(
                self.trial,
                methods=["bike", "zeni"],
                weights={"bayesian_bis": -1},
            )
        with self.assertRaises(ValueError):
            gaitkit.detect_ensemble(
                self.trial,
                methods=["bike", "zeni"],
                weights={"bayesian_bis": float("nan")},
            )
        with self.assertRaises(ValueError):
            gaitkit.detect_ensemble(
                self.trial,
                methods=["bike", "zeni"],
                weights="invalid_mode",
            )

    def test_method_normalization_deduplicates_aliases(self):
        normalized = _ensemble._normalize_methods(["bike", "bayesian_bis", "zeni", "bike"])
        self.assertEqual(normalized, ["bayesian_bis", "zeni"])

    def test_failed_detector_reported_in_metadata(self):
        class _Ev:
            def __init__(self, frame_index: int, side: str):
                self.frame_index = frame_index
                self.side = side
                self.time = frame_index / 100.0

        class _DetOk:
            def detect_gait_events(self, _data):
                hs = [_Ev(10, "left"), _Ev(30, "right")]
                to = [_Ev(20, "left"), _Ev(40, "right")]
                return hs, to, []

        def _fake_get_detector(name, fps=100.0, **kwargs):
            if name == "oconnor":
                raise RuntimeError("backend unavailable")
            return _DetOk()

        with mock.patch.object(
            _ensemble,
            "get_detector",
            side_effect=_fake_get_detector,
        ):
            with mock.patch.object(
                _ensemble,
                "DETECTOR_REGISTRY",
                {"bayesian_bis": object(), "zeni": object(), "oconnor": object()},
            ):
                result = _ensemble.detect_ensemble(
                    data=[object()],
                    methods=["bike", "zeni", "oconnor"],
                    fps=100.0,
                    min_votes=1,
                )
        self.assertIn("oconnor", result.metadata["methods_failed"])
        self.assertIn("bayesian_bis", result.metadata["methods_succeeded"])
        self.assertIn("zeni", result.metadata["methods_succeeded"])


if __name__ == "__main__":
    unittest.main()
