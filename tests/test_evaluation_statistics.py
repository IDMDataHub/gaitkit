"""Unit tests for evaluation.statistics helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python" / "src"))

from gaitkit.evaluation.statistics import (
    bootstrap_confidence_interval,
    compute_summary_table,
    wilcoxon_signed_rank,
)


class TestEvaluationStatistics(unittest.TestCase):
    def test_bootstrap_confidence_interval_validates_controls(self):
        x = np.array([1.0, 2.0, 3.0], dtype=float)
        with self.assertRaises(ValueError):
            bootstrap_confidence_interval(x, n_bootstrap=0)
        with self.assertRaises(ValueError):
            bootstrap_confidence_interval(x, confidence=0.0)
        with self.assertRaises(ValueError):
            bootstrap_confidence_interval(x, statistic="mode")

    def test_wilcoxon_signed_rank_validates_alternative(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
        b = np.array([1.1, 1.9, 3.2, 3.8, 4.9], dtype=float)
        with self.assertRaises(ValueError):
            wilcoxon_signed_rank(a, b, alternative="invalid")

    def test_compute_summary_table_validates_group_columns(self):
        with self.assertRaises(ValueError):
            compute_summary_table([], group_cols=["dataset"])  # type: ignore[arg-type]
        df = pd.DataFrame(
            {
                "dataset": ["d1", "d1"],
                "detector_name": ["m1", "m1"],
                "hs_f1": [0.5, 0.6],
            }
        )
        with self.assertRaises(ValueError):
            compute_summary_table(df, group_cols=[])  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            compute_summary_table(df, group_cols=["missing"])


if __name__ == "__main__":
    unittest.main()
