"""Unit tests for evaluation.metrics helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python" / "src"))

from gaitkit.evaluation.metrics import compute_cadence_error, compute_event_metrics


class TestEvaluationMetrics(unittest.TestCase):
    def test_compute_event_metrics_rejects_invalid_controls(self):
        with self.assertRaises(ValueError):
            compute_event_metrics([], [], tolerance_ms=10, fps=0)
        with self.assertRaises(ValueError):
            compute_event_metrics([], [], tolerance_ms=-1, fps=100)
        with self.assertRaises(ValueError):
            compute_event_metrics([], [], tolerance_ms=10, fps=100, valid_frame_range=[0, 10])
        with self.assertRaises(ValueError):
            compute_event_metrics([], [], tolerance_ms=10, fps=100, valid_frame_range=(10, 0))

    def test_compute_event_metrics_zone_diagnostics(self):
        m = compute_event_metrics(
            detected=[5, 15, 25],
            ground_truth=[16],
            tolerance_ms=20,
            fps=100.0,
            valid_frame_range=(10, 20),
        )
        self.assertIn("n_detected_total", m)
        self.assertIn("n_detected_in_zone", m)
        self.assertEqual(m["n_detected_total"], 3)
        self.assertEqual(m["n_detected_in_zone"], 1)

    def test_compute_cadence_error_requires_positive_fps(self):
        with self.assertRaises(ValueError):
            compute_cadence_error([1, 2], gt_cadence=100.0, fps=0)


if __name__ == "__main__":
    unittest.main()
