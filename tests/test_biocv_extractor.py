"""Unit tests for BioCVExtractor."""

from __future__ import annotations

import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python" / "src"))


class TestBioCVConstructor(unittest.TestCase):
    """Test BioCVExtractor constructor validation."""

    def _make_cls(self):
        from gaitkit.extractors.biocv_extractor import BioCVExtractor
        return BioCVExtractor

    def test_valid_data_dir(self):
        cls = self._make_cls()
        with tempfile.TemporaryDirectory() as td:
            ext = cls(td)
            self.assertEqual(ext.data_dir, Path(td))

    def test_invalid_empty_data_dir(self):
        cls = self._make_cls()
        with self.assertRaises(ValueError):
            cls("")

    def test_invalid_missing_data_dir(self):
        cls = self._make_cls()
        with self.assertRaises(ValueError):
            cls("/definitely/missing/path")

    def test_name_and_description(self):
        cls = self._make_cls()
        with tempfile.TemporaryDirectory() as td:
            ext = cls(td)
            self.assertEqual(ext.name, "BioCV")
            self.assertIn("15", ext.description)


class TestParseTrialInfo(unittest.TestCase):
    """Test _parse_trial_info on known path patterns."""

    def _make_extractor(self, tmpdir):
        from gaitkit.extractors.biocv_extractor import BioCVExtractor
        return BioCVExtractor(tmpdir)

    def test_walk_trial(self):
        with tempfile.TemporaryDirectory() as td:
            ext = self._make_extractor(td)
            p = Path(td) / "P03" / "P03_WALK_01" / "markers.c3d"
            info = ext._parse_trial_info(p)
            self.assertEqual(info['subject_id'], 'P03')
            self.assertEqual(info['condition'], 'walk')
            self.assertEqual(info['trial_id'], 'walk_01')

    def test_run_trial(self):
        with tempfile.TemporaryDirectory() as td:
            ext = self._make_extractor(td)
            p = Path(td) / "P10" / "P10_RUN_05" / "markers.c3d"
            info = ext._parse_trial_info(p)
            self.assertEqual(info['subject_id'], 'P10')
            self.assertEqual(info['condition'], 'run')
            self.assertEqual(info['trial_id'], 'run_05')

    def test_cmjm_trial(self):
        with tempfile.TemporaryDirectory() as td:
            ext = self._make_extractor(td)
            p = Path(td) / "P16" / "P16_CMJM_03" / "markers.c3d"
            info = ext._parse_trial_info(p)
            self.assertEqual(info['subject_id'], 'P16')
            self.assertEqual(info['condition'], 'cmjm')

    def test_hop_trial(self):
        with tempfile.TemporaryDirectory() as td:
            ext = self._make_extractor(td)
            p = Path(td) / "P28" / "P28_HOP_01" / "markers.c3d"
            info = ext._parse_trial_info(p)
            self.assertEqual(info['subject_id'], 'P28')
            self.assertEqual(info['condition'], 'hop')
            self.assertEqual(info['trial_id'], 'hop_01')

    def test_unknown_path(self):
        with tempfile.TemporaryDirectory() as td:
            ext = self._make_extractor(td)
            p = Path(td) / "random" / "markers.c3d"
            info = ext._parse_trial_info(p)
            self.assertEqual(info['subject_id'], 'unknown')


class TestParseEventsFile(unittest.TestCase):
    """Test _parse_events_file with mock event files."""

    def _make_extractor(self, tmpdir):
        from gaitkit.extractors.biocv_extractor import BioCVExtractor
        return BioCVExtractor(tmpdir)

    def test_standard_walk_events(self):
        """Parse a typical WALK events file."""
        content = textwrap.dedent("""\
            Event\tLHS\tRHS\tLTO\tRTO\tLOFF\tROFF\tLON\tRON
            Item 1\t450\t261\t360\t303\t809\t696\t672\t561
            Item 2\t672\t323\t584\t472\t\t\t\t
            Item 3\t895\t561\t809\t696\t\t\t\t
            Item 4\t1070\t783\t1033\t920\t\t\t\t
            Item 5\t\t997\t\t\t\t\t\t""")
        with tempfile.TemporaryDirectory() as td:
            ext = self._make_extractor(td)
            events_path = Path(td) / "markers.events.frame"
            events_path.write_text(content)
            events = ext._parse_events_file(events_path)
            self.assertEqual(events['hs_left'], [450, 672, 895, 1070])
            self.assertEqual(events['hs_right'], [261, 323, 561, 783, 997])
            self.assertEqual(events['to_left'], [360, 584, 809, 1033])
            self.assertEqual(events['to_right'], [303, 472, 696, 920])

    def test_events_with_nan(self):
        """NaN values should be skipped."""
        content = textwrap.dedent("""\
            Event\tLHS\tRHS\tLTO\tRTO
            Item 1\t100\tnan\t200\tNaN
            Item 2\tNaN\t300\t\t400""")
        with tempfile.TemporaryDirectory() as td:
            ext = self._make_extractor(td)
            events_path = Path(td) / "markers.events.frame"
            events_path.write_text(content)
            events = ext._parse_events_file(events_path)
            self.assertEqual(events['hs_left'], [100])
            self.assertEqual(events['hs_right'], [300])
            self.assertEqual(events['to_left'], [200])
            self.assertEqual(events['to_right'], [400])

    def test_events_with_dashes(self):
        """Dash characters should be treated as missing."""
        content = textwrap.dedent("""\
            Event\tLHS\tRHS\tLTO\tRTO
            Item 1\t100\t—\t200\t-""")
        with tempfile.TemporaryDirectory() as td:
            ext = self._make_extractor(td)
            events_path = Path(td) / "markers.events.frame"
            events_path.write_text(content)
            events = ext._parse_events_file(events_path)
            self.assertEqual(events['hs_left'], [100])
            self.assertEqual(events['hs_right'], [])
            self.assertEqual(events['to_left'], [200])
            self.assertEqual(events['to_right'], [])

    def test_missing_events_file(self):
        """Missing file should return empty events, not raise."""
        with tempfile.TemporaryDirectory() as td:
            ext = self._make_extractor(td)
            events_path = Path(td) / "nonexistent.events.frame"
            events = ext._parse_events_file(events_path)
            self.assertEqual(events['hs_left'], [])
            self.assertEqual(events['hs_right'], [])

    def test_empty_events_file(self):
        with tempfile.TemporaryDirectory() as td:
            ext = self._make_extractor(td)
            events_path = Path(td) / "markers.events.frame"
            events_path.write_text("")
            events = ext._parse_events_file(events_path)
            self.assertEqual(events['hs_left'], [])

    def test_visual3d_fixed_width_format(self):
        """Parse the real Visual3D export format (space-padded, multi-header)."""
        content = (
            "                           P03_WALK_01.c3d      P03_WALK_01.c3d      P03_WALK_01.c3d      P03_WALK_01.c3d      P03_WALK_01.c3d      P03_WALK_01.c3d      P03_WALK_01.c3d      P03_WALK_01.c3d\n"
            "                                       End                  LHS                 LOFF                  LON                  LTO                  RHS                 ROFF                  RON                  RTO\n"
            "                               EVENT_LABEL          EVENT_LABEL          EVENT_LABEL          EVENT_LABEL          EVENT_LABEL          EVENT_LABEL          EVENT_LABEL          EVENT_LABEL          EVENT_LABEL\n"
            "                                  ORIGINAL             ORIGINAL             ORIGINAL             ORIGINAL             ORIGINAL             ORIGINAL             ORIGINAL             ORIGINAL             ORIGINAL\n"
            "                 ITEM                    X                    X                    X                    X                    X                    X                    X                    X                    X\n"
            "                    1                  942                  450                  809                  672                  360                  261                  696                  561                  303\n"
            "                    2                  NaN                  672                  NaN                  NaN                  584                  323                  NaN                  NaN                  472\n"
            "                    3                  NaN                  895                  NaN                  NaN                  809                  561                  NaN                  NaN                  696\n"
            "                    4                  NaN                 1070                  NaN                  NaN                 1033                  783                  NaN                  NaN                  920\n"
            "                    5                  NaN                  NaN                  NaN                  NaN                  NaN                  997                  NaN                  NaN                  NaN\n"
        )
        with tempfile.TemporaryDirectory() as td:
            ext = self._make_extractor(td)
            events_path = Path(td) / "markers.events.frame"
            events_path.write_text(content)
            events = ext._parse_events_file(events_path)
            self.assertEqual(events['hs_left'], [450, 672, 895, 1070])
            self.assertEqual(events['hs_right'], [261, 323, 561, 783, 997])
            self.assertEqual(events['to_left'], [360, 584, 809, 1033])
            self.assertEqual(events['to_right'], [303, 472, 696, 920])

    def test_events_are_sorted(self):
        """Events should be returned sorted regardless of file order."""
        content = textwrap.dedent("""\
            Event\tLHS\tRHS
            Item 1\t500\t100
            Item 2\t200\t600""")
        with tempfile.TemporaryDirectory() as td:
            ext = self._make_extractor(td)
            events_path = Path(td) / "markers.events.frame"
            events_path.write_text(content)
            events = ext._parse_events_file(events_path)
            self.assertEqual(events['hs_left'], [200, 500])
            self.assertEqual(events['hs_right'], [100, 600])


class TestListFiles(unittest.TestCase):
    """Test list_files with mock directory structure."""

    def _make_extractor(self, tmpdir):
        from gaitkit.extractors.biocv_extractor import BioCVExtractor
        return BioCVExtractor(tmpdir)

    def _create_trial(self, base_dir, subject, trial_name):
        """Create a mock trial directory with a markers.c3d placeholder."""
        trial_dir = Path(base_dir) / subject / f"{subject}_{trial_name}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "markers.c3d").write_bytes(b"")
        return trial_dir

    def test_finds_walk_run_cmj_hop(self):
        with tempfile.TemporaryDirectory() as td:
            self._create_trial(td, "P03", "WALK_01")
            self._create_trial(td, "P03", "WALK_02")
            self._create_trial(td, "P03", "RUN_01")
            self._create_trial(td, "P03", "CMJM_01")
            self._create_trial(td, "P03", "CMJS_01")
            self._create_trial(td, "P03", "HOP_01")
            ext = self._make_extractor(td)
            files = ext.list_files()
            self.assertEqual(len(files), 6)

    def test_excludes_ml_static_calib(self):
        with tempfile.TemporaryDirectory() as td:
            self._create_trial(td, "P03", "WALK_01")
            self._create_trial(td, "P03", "STATIC_01")
            # ML_ trials: directory structure is ML_WALK_01 not P03_ML_WALK_01
            ml_dir = Path(td) / "P03" / "ML_WALK_01"
            ml_dir.mkdir(parents=True)
            (ml_dir / "markers.c3d").write_bytes(b"")
            # calib
            cal_dir = Path(td) / "P03" / "calib_00"
            cal_dir.mkdir(parents=True)
            (cal_dir / "markers.c3d").write_bytes(b"")
            ext = self._make_extractor(td)
            files = ext.list_files()
            self.assertEqual(len(files), 1)
            self.assertIn("WALK_01", str(files[0]))

    def test_multiple_subjects(self):
        with tempfile.TemporaryDirectory() as td:
            self._create_trial(td, "P03", "WALK_01")
            self._create_trial(td, "P10", "WALK_01")
            self._create_trial(td, "P28", "RUN_03")
            ext = self._make_extractor(td)
            files = ext.list_files()
            self.assertEqual(len(files), 3)

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as td:
            ext = self._make_extractor(td)
            files = ext.list_files()
            self.assertEqual(len(files), 0)


class TestModuleRegistration(unittest.TestCase):
    """Test that BioCVExtractor is accessible from the extractors module."""

    def test_biocv_in_all(self):
        import gaitkit.extractors as extractors
        # Should be registered if ezc3d is available
        try:
            import ezc3d
            self.assertIn("BioCVExtractor", extractors.__all__)
            self.assertTrue(hasattr(extractors, "BioCVExtractor"))
        except ImportError:
            # Without ezc3d, it won't be registered (optional import)
            pass


class TestIntegrationP03Walk01(unittest.TestCase):
    """Integration test on real P03_WALK_01 data (skipped if unavailable)."""

    DATA_DIR = Path.home() / "gait_benchmark_project" / "data" / "biocv"

    def setUp(self):
        if not self.DATA_DIR.exists():
            self.skipTest(f"BioCV data not found at {self.DATA_DIR}")
        try:
            import ezc3d  # noqa: F401
        except ImportError:
            self.skipTest("ezc3d not installed")

    def _make_extractor(self):
        from gaitkit.extractors.biocv_extractor import BioCVExtractor
        return BioCVExtractor(str(self.DATA_DIR))

    def test_list_files_non_empty(self):
        ext = self._make_extractor()
        files = ext.list_files()
        self.assertGreater(len(files), 0, "Should find at least one trial")

    def test_extract_walk_trial(self):
        ext = self._make_extractor()
        files = ext.list_files()
        # Find a WALK trial
        walk_files = [f for f in files if '_WALK_' in str(f)]
        if not walk_files:
            self.skipTest("No WALK trial found in data")

        result = ext.extract_file(walk_files[0])

        # Basic sanity checks
        self.assertEqual(result.fps, 200.0)
        self.assertGreater(result.n_frames, 0)
        self.assertGreater(result.duration_s, 0)
        self.assertNotEqual(result.subject_id, 'unknown')
        self.assertEqual(result.condition, 'walk')

        # Ground truth should have events
        self.assertTrue(result.ground_truth.has_hs)
        self.assertTrue(result.ground_truth.has_to)
        self.assertEqual(result.ground_truth.event_source, "annotated")
        self.assertTrue(result.ground_truth.has_forces)

        # HS/TO frames should be non-empty
        total_hs = (len(result.ground_truth.hs_frames['left'])
                     + len(result.ground_truth.hs_frames['right']))
        self.assertGreater(total_hs, 0, "Should have HS events")

        # Angle frames should have non-zero values
        mid = len(result.angle_frames) // 2
        af = result.angle_frames[mid]
        has_nonzero = (af.left_knee_angle != 0.0
                       or af.right_knee_angle != 0.0
                       or af.left_hip_angle != 0.0)
        self.assertTrue(has_nonzero, "Angles should be non-zero at mid-trial")


if __name__ == "__main__":
    unittest.main()
