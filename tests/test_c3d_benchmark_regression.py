"""C3D benchmark regression tests based on bundled reference files.

This test validates BIKE performance on six curated C3D files using the
expected scores documented in ``tests/test_c3d_benchmark.Rmd``.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python" / "src"))

import gaitkit
from gaitkit.evaluation.metrics import compute_event_metrics
from gaitkit.extractors.figshare_pd_extractor import FigsharePDExtractor
from gaitkit.extractors.fukuchi_extractor import FukuchiExtractor
from gaitkit.extractors.vanderzee_extractor import VanderzeeExtractor

try:
    import ezc3d  # noqa: F401
except Exception:  # pragma: no cover - optional dependency
    ezc3d = None


class TestC3DBenchmarkRegression(unittest.TestCase):
    """Non-regression checks for BIKE on bundled C3D benchmark samples."""

    _BASE_DIR = PROJECT_ROOT / "python" / "src" / "gaitkit" / "data" / "c3d"
    _TOLERANCE_MS = 50.0

    # Reference values from tests/test_c3d_benchmark.Rmd
    _CASES = {
        "vanderzee_p10_trial3.c3d": {
            "extractor": VanderzeeExtractor,
            "expected_hs_f1": 0.872,
            "expected_to_f1": 0.991,
            "min_hs_f1": 0.82,
            "min_to_f1": 0.94,
        },
        "fukuchi_WBDS42walkT03.c3d": {
            "extractor": FukuchiExtractor,
            "expected_hs_f1": 0.913,
            "expected_to_f1": 0.891,
            "min_hs_f1": 0.86,
            "min_to_f1": 0.84,
        },
        "fukuchi_WBDS19walkT01.c3d": {
            "extractor": FukuchiExtractor,
            "expected_hs_f1": 0.968,
            "expected_to_f1": 0.774,
            "min_hs_f1": 0.92,
            "min_to_f1": 0.72,
        },
        "figshare_pd_SUB07_off_walk_7.c3d": {
            "extractor": FigsharePDExtractor,
            "expected_hs_f1": 0.857,
            "expected_to_f1": 0.793,
            "min_hs_f1": 0.81,
            "min_to_f1": 0.74,
        },
        "figshare_pd_SUB01_off_walk_12b.c3d": {
            "extractor": FigsharePDExtractor,
            "expected_hs_f1": 0.957,
            "expected_to_f1": 1.000,
            "min_hs_f1": 0.91,
            "min_to_f1": 0.95,
        },
        "figshare_pd_SUB05_off_walk_8.c3d": {
            "extractor": FigsharePDExtractor,
            "expected_hs_f1": 1.000,
            "expected_to_f1": 0.833,
            "min_hs_f1": 0.95,
            "min_to_f1": 0.78,
        },
    }

    def test_bike_scores_match_benchmark_references(self):
        if ezc3d is None:
            self.skipTest("ezc3d is not installed")

        for filename, cfg in self._CASES.items():
            with self.subTest(c3d=filename):
                path = self._BASE_DIR / filename
                self.assertTrue(path.exists(), f"Missing benchmark file: {path}")

                extractor = cfg["extractor"](str(self._BASE_DIR))
                extracted = extractor.extract_file(path)
                trial = {
                    "angle_frames": [af.__dict__ for af in extracted.angle_frames],
                    "fps": extracted.fps,
                    "n_frames": extracted.n_frames,
                }

                result = gaitkit.detect(trial, method="bike")
                hs_detected = sorted(
                    [e["frame"] for e in result.left_hs] +
                    [e["frame"] for e in result.right_hs]
                )
                to_detected = sorted(
                    [e["frame"] for e in result.left_to] +
                    [e["frame"] for e in result.right_to]
                )
                hs_gt = sorted(
                    extracted.ground_truth.hs_frames["left"] +
                    extracted.ground_truth.hs_frames["right"]
                )
                to_gt = sorted(
                    extracted.ground_truth.to_frames["left"] +
                    extracted.ground_truth.to_frames["right"]
                )

                hs_metrics = compute_event_metrics(
                    hs_detected,
                    hs_gt,
                    tolerance_ms=self._TOLERANCE_MS,
                    fps=extracted.fps,
                    valid_frame_range=extracted.ground_truth.valid_frame_range,
                )
                to_metrics = compute_event_metrics(
                    to_detected,
                    to_gt,
                    tolerance_ms=self._TOLERANCE_MS,
                    fps=extracted.fps,
                    valid_frame_range=extracted.ground_truth.valid_frame_range,
                )

                # Strong reproducibility check (rounded to 3 decimals as in Rmd table)
                self.assertAlmostEqual(round(hs_metrics["f1"], 3), cfg["expected_hs_f1"], places=3)
                self.assertAlmostEqual(round(to_metrics["f1"], 3), cfg["expected_to_f1"], places=3)

                # CI guardrails from Rmd "Regression thresholds" section
                self.assertGreaterEqual(
                    hs_metrics["f1"],
                    cfg["min_hs_f1"],
                    msg=f"{filename} HS F1 regressed: {hs_metrics['f1']:.3f} < {cfg['min_hs_f1']:.3f}",
                )
                self.assertGreaterEqual(
                    to_metrics["f1"],
                    cfg["min_to_f1"],
                    msg=f"{filename} TO F1 regressed: {to_metrics['f1']:.3f} < {cfg['min_to_f1']:.3f}",
                )


if __name__ == "__main__":
    unittest.main()
