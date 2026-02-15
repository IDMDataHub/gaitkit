"""Unit tests for evaluation.matching helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python" / "src"))

from gaitkit.evaluation.matching import match_events


class TestEvaluationMatching(unittest.TestCase):
    def test_rejects_negative_tolerance(self):
        with self.assertRaises(ValueError):
            match_events([1], [1], tolerance_frames=-1)

    def test_empty_fast_paths(self):
        matches, unmatched = match_events([], [10, 20], tolerance_frames=5)
        self.assertEqual(matches, [])
        self.assertEqual(unmatched, [10, 20])

        matches, unmatched = match_events([10, 20], [], tolerance_frames=5)
        self.assertEqual(len(matches), 2)
        self.assertEqual(unmatched, [])
        self.assertTrue(all(m.gt_frame is None for m in matches))


if __name__ == "__main__":
    unittest.main()
