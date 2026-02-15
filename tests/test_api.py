"""Tests for gaitkit public API."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python" / "src"))

import unittest
import gaitkit


class TestListMethods(unittest.TestCase):
    def test_returns_list(self):
        methods = gaitkit.list_methods()
        self.assertIsInstance(methods, list)
        self.assertIn("bike", methods)
        self.assertEqual(methods[0], "bike")

    def test_all_ten_methods(self):
        self.assertEqual(len(gaitkit.list_methods()), 10)


class TestListExamples(unittest.TestCase):
    def test_returns_list(self):
        examples = gaitkit.list_examples()
        self.assertIn("healthy", examples)
        self.assertIn("parkinson", examples)


class TestLoadExample(unittest.TestCase):
    def test_healthy(self):
        trial = gaitkit.load_example("healthy")
        self.assertIn("angle_frames", trial)
        self.assertIn("fps", trial)
        self.assertGreater(len(trial["angle_frames"]), 100)

    def test_parkinson(self):
        trial = gaitkit.load_example("parkinson")
        self.assertEqual(trial["population"], "parkinsons")

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            gaitkit.load_example("nonexistent")


class TestDetect(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trial = gaitkit.load_example("healthy")

    def test_bike_default(self):
        result = gaitkit.detect(self.trial)
        self.assertIsInstance(result, gaitkit.GaitResult)
        self.assertEqual(result.method, "bayesian_bis")
        self.assertGreater(len(result.left_hs) + len(result.right_hs), 0)

    def test_zeni(self):
        result = gaitkit.detect(self.trial, method="zeni")
        self.assertEqual(result.method, "zeni")

    def test_events_dataframe(self):
        result = gaitkit.detect(self.trial)
        df = result.events
        self.assertIn("time", df.columns)
        self.assertIn("event_type", df.columns)
        self.assertIn("side", df.columns)

    def test_cycles_dataframe(self):
        result = gaitkit.detect(self.trial)
        cyc = result.cycles
        self.assertIn("stride_time", cyc.columns)
        self.assertIn("stance_pct", cyc.columns)

    def test_to_csv(self):
        import tempfile, os
        result = gaitkit.detect(self.trial)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            result.to_csv(f.name)
            self.assertTrue(os.path.getsize(f.name) > 0)
            os.unlink(f.name)

    def test_summary(self):
        result = gaitkit.detect(self.trial)
        text = result.summary()
        self.assertIn("Heel-strikes", text)

    def test_repr(self):
        result = gaitkit.detect(self.trial)
        r = repr(result)
        self.assertIn("GaitResult", r)

    def test_unknown_method(self):
        with self.assertRaises(ValueError):
            gaitkit.detect(self.trial, method="nonexistent")


class TestDetectAllMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trial = gaitkit.load_example("healthy")

    def test_all_training_free(self):
        for m in ["bike", "zeni", "oconnor", "hreljac", "mickelborough",
                   "ghoussayni", "vancanneyt", "dgei"]:
            with self.subTest(method=m):
                result = gaitkit.detect(self.trial, method=m)
                total = (len(result.left_hs) + len(result.right_hs) +
                         len(result.left_to) + len(result.right_to))
                self.assertGreater(total, 0, f"{m} detected no events")


if __name__ == "__main__":
    unittest.main()
