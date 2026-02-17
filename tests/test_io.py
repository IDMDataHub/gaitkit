"""Unit tests for gaitkit I/O helpers."""

from __future__ import annotations

import sys
import types
import unittest
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
from scipy.io import savemat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python" / "src"))

from gaitkit import _io, _core


class TestLoadC3D(unittest.TestCase):
    def test_load_example_name_must_be_non_empty_string(self):
        with self.assertRaises(ValueError):
            _io.load_example("")
        with self.assertRaises(ValueError):
            _io.load_example(None)  # type: ignore[arg-type]

    def test_load_example_accepts_human_friendly_alias_spelling(self):
        base = _io.load_example("parkinson")
        self.assertEqual(_io.load_example("ParkinSon")["population"], base["population"])
        self.assertEqual(_io.load_example("par kin son")["population"], base["population"])
        self.assertEqual(_io.load_example("par-kin-son")["population"], base["population"])
        self.assertEqual(_io.load_example("par\tkin\nson")["population"], base["population"])

    def test_load_c3d_rejects_missing_or_invalid_path_before_import(self):
        with self.assertRaises(ValueError):
            _io.load_c3d("not_a_c3d.txt")
        with self.assertRaises(ValueError):
            _io.load_c3d("")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            _io.load_c3d(None)  # type: ignore[arg-type]
        with self.assertRaises(FileNotFoundError):
            _io.load_c3d("missing_file.c3d")

    def test_load_c3d_wraps_ezc3d_open_error_with_actionable_hints(self):
        fake_ezc3d = types.SimpleNamespace(
            c3d=lambda _path: (_ for _ in ()).throw(OSError("iostream stream error"))
        )
        with tempfile.TemporaryDirectory() as tmp:
            c3d_path = Path(tmp) / "broken.c3d"
            c3d_path.write_bytes(b"")
            with mock.patch.dict(sys.modules, {"ezc3d": fake_ezc3d}):
                with self.assertRaises(OSError) as ctx:
                    _io.load_c3d(str(c3d_path))
        msg = str(ctx.exception).lower()
        self.assertIn("could not open c3d file", msg)
        self.assertIn("onedrive", msg)

    def test_invalid_marker_set_raises(self):
        fake_ezc3d = types.SimpleNamespace(
            c3d=lambda _path: {
                "header": {"points": {"frame_rate": 100.0}},
                "data": {"points": np.zeros((4, 1, 1))},
                "parameters": {"POINT": {"LABELS": {"value": ["LHEE"]}}},
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            c3d_path = Path(tmp) / "dummy.c3d"
            c3d_path.write_bytes(b"")
            with mock.patch.dict(sys.modules, {"ezc3d": fake_ezc3d}):
                with self.assertRaises(ValueError):
                    _io.load_c3d(str(c3d_path), marker_set="unknown")
                with self.assertRaises(ValueError):
                    _io.load_c3d(str(c3d_path), marker_set="")

    def test_extract_hs_frames_uses_same_open_error_diagnostics(self):
        fake_ezc3d = types.SimpleNamespace(
            c3d=lambda _path: (_ for _ in ()).throw(OSError("iostream stream error"))
        )
        with tempfile.TemporaryDirectory() as tmp:
            c3d_path = Path(tmp) / "broken_hs.c3d"
            c3d_path.write_bytes(b"")
            with mock.patch.dict(sys.modules, {"ezc3d": fake_ezc3d}):
                with self.assertRaises(OSError) as ctx:
                    _io._extract_hs_frames_from_c3d(str(c3d_path))
        self.assertIn("Could not open C3D file", str(ctx.exception))

    def test_auto_detects_imy_marker_set(self):
        labels = ["LCAL", "RCAL", "LFMH1", "RFMH1", "LMM", "LLM", "RMM", "RLM", "LASIS", "RASIS"]
        points = np.zeros((4, len(labels), 3), dtype=float)
        points[0, labels.index("LCAL"), :] = [1.0, 1.1, 1.2]
        points[0, labels.index("RCAL"), :] = [2.0, 2.1, 2.2]
        points[0, labels.index("LFMH1"), :] = [3.0, 3.1, 3.2]
        points[0, labels.index("RFMH1"), :] = [4.0, 4.1, 4.2]

        fake_ezc3d = types.SimpleNamespace(
            c3d=lambda _path: {
                "header": {"points": {"frame_rate": 200.0}},
                "data": {"points": points},
                "parameters": {"POINT": {"LABELS": {"value": labels}}},
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            c3d_path = Path(tmp) / "imy.c3d"
            c3d_path.write_bytes(b"")
            with mock.patch.dict(sys.modules, {"ezc3d": fake_ezc3d}):
                out = _io.load_c3d(str(c3d_path), marker_set="auto")
        lp0 = out["angle_frames"][0]["landmark_positions"]
        self.assertIsNotNone(lp0)
        self.assertIn("left_heel", lp0)
        self.assertIn("right_heel", lp0)
        self.assertIn("left_toe", lp0)
        self.assertIn("right_toe", lp0)

    def test_custom_marker_map_is_supported(self):
        labels = ["XH", "YH", "XT", "YT", "XA", "YA"]
        points = np.zeros((4, len(labels), 2), dtype=float)
        points[:, labels.index("XH"), :] = 1.0
        points[:, labels.index("YH"), :] = 2.0
        points[:, labels.index("XT"), :] = 3.0
        points[:, labels.index("YT"), :] = 4.0
        points[:, labels.index("XA"), :] = 5.0
        points[:, labels.index("YA"), :] = 6.0

        fake_ezc3d = types.SimpleNamespace(
            c3d=lambda _path: {
                "header": {"points": {"frame_rate": 100.0}},
                "data": {"points": points},
                "parameters": {"POINT": {"LABELS": {"value": labels}}},
            }
        )
        marker_map = {
            "left_heel": "XH",
            "right_heel": "YH",
            "left_toe": "XT",
            "right_toe": "YT",
            "left_ankle": "XA",
            "right_ankle": "YA",
        }
        with tempfile.TemporaryDirectory() as tmp:
            c3d_path = Path(tmp) / "custom.c3d"
            c3d_path.write_bytes(b"")
            with mock.patch.dict(sys.modules, {"ezc3d": fake_ezc3d}):
                out = _io.load_c3d(str(c3d_path), marker_map=marker_map)
        lp0 = out["angle_frames"][0]["landmark_positions"]
        self.assertIn("left_heel", lp0)
        self.assertIn("right_toe", lp0)

    def test_no_supported_marker_error_lists_detected_labels(self):
        labels = ["A", "B", "C"]
        points = np.zeros((4, len(labels), 2), dtype=float)
        fake_ezc3d = types.SimpleNamespace(
            c3d=lambda _path: {
                "header": {"points": {"frame_rate": 100.0}},
                "data": {"points": points},
                "parameters": {"POINT": {"LABELS": {"value": labels}}},
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            c3d_path = Path(tmp) / "none.c3d"
            c3d_path.write_bytes(b"")
            with mock.patch.dict(sys.modules, {"ezc3d": fake_ezc3d}):
                with self.assertRaises(ValueError) as ctx:
                    _io.load_c3d(str(c3d_path), marker_set="auto")
        self.assertIn("Detected labels", str(ctx.exception))

    def test_load_c3d_computes_proxy_angles_when_model_angles_absent(self):
        labels = [
            "LASI", "RASI", "SACR",
            "LKNE", "RKNE",
            "LANK", "RANK",
            "LTOE", "RTOE",
            "LHEE", "RHEE",
        ]
        n_frames = 6
        points = np.zeros((4, len(labels), n_frames), dtype=float)

        def set_series(label: str, x0: float, x_step: float, z: float):
            i = labels.index(label)
            for f in range(n_frames):
                points[0, i, f] = x0 + x_step * f
                points[1, i, f] = 0.0
                points[2, i, f] = z

        # Simple sagittal motion with varying knee/toe trajectories.
        set_series("LASI", 0.10, 0.01, 1.00)
        set_series("RASI", 0.12, 0.01, 1.00)
        set_series("SACR", 0.11, 0.01, 0.95)
        set_series("LKNE", 0.15, 0.015, 0.60)
        set_series("RKNE", 0.13, 0.015, 0.60)
        set_series("LANK", 0.18, 0.020, 0.20)
        set_series("RANK", 0.16, 0.020, 0.20)
        set_series("LTOE", 0.24, 0.025, 0.02)
        set_series("RTOE", 0.22, 0.025, 0.02)
        set_series("LHEE", 0.12, 0.020, 0.02)
        set_series("RHEE", 0.10, 0.020, 0.02)

        fake_ezc3d = types.SimpleNamespace(
            c3d=lambda _path: {
                "header": {"points": {"frame_rate": 100.0}},
                "data": {"points": points},
                "parameters": {"POINT": {"LABELS": {"value": labels}}},
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            c3d_path = Path(tmp) / "no_model_angles.c3d"
            c3d_path.write_bytes(b"")
            with mock.patch.dict(sys.modules, {"ezc3d": fake_ezc3d}):
                out = _io.load_c3d(str(c3d_path), marker_set="pig")

        lk = np.array([fr["left_knee_angle"] for fr in out["angle_frames"]], dtype=float)
        la = np.array([fr["left_ankle_angle"] for fr in out["angle_frames"]], dtype=float)
        self.assertTrue(np.isfinite(lk).all())
        self.assertTrue(np.isfinite(la).all())
        self.assertFalse(np.allclose(lk, 0.0))
        self.assertFalse(np.allclose(la, 0.0))

    def test_load_angles_file_from_mapping(self):
        angles = {
            "Lhip": np.arange(10, dtype=float),
            "Rhip": np.arange(10, dtype=float) + 1,
            "Lknee": np.arange(10, dtype=float) + 2,
            "Rknee": np.arange(10, dtype=float) + 3,
            "Lankle": np.arange(10, dtype=float) + 4,
            "Rankle": np.arange(10, dtype=float) + 5,
        }
        out = _io.load_angles_file(angles, n_frames=8)
        self.assertEqual(set(out.keys()), {
            "left_hip_angle", "right_hip_angle",
            "left_knee_angle", "right_knee_angle",
            "left_ankle_angle", "right_ankle_angle",
        })
        self.assertEqual(len(out["left_hip_angle"]), 8)
        self.assertEqual(out["right_ankle_angle"][0], 5.0)

    def test_load_angles_file_from_mat_res_angles_t(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "angles.mat"
            dtype = [
                ("Lhip", "O"), ("Rhip", "O"),
                ("Lknee", "O"), ("Rknee", "O"),
                ("Lankle", "O"), ("Rankle", "O"),
            ]
            arr = np.zeros((1, 1), dtype=dtype)
            base = np.column_stack([np.arange(6, dtype=float), np.zeros(6), np.zeros(6)])
            for name in ("Lhip", "Rhip", "Lknee", "Rknee", "Lankle", "Rankle"):
                arr[name][0, 0] = base
            savemat(path, {"res_angles_t": arr})
            out = _io.load_angles_file(path, n_frames=6)
        self.assertIn("left_hip_angle", out)
        self.assertEqual(out["left_hip_angle"][3], 3.0)

    def test_normalize_input_passes_angles_to_load_c3d(self):
        with mock.patch("gaitkit._io.load_c3d", return_value={"angle_frames": [{"frame_index": 0}], "fps": 100.0}):
            _core._normalize_input("dummy.c3d", fps=None, angles="angles.mat")
            _io.load_c3d.assert_called_once_with("dummy.c3d", angles="angles.mat")

    def test_angle_extraction_failure_is_non_blocking(self):
        # Include an angle label that does not exist in points data to force IndexError
        labels = ["LHEE", "RHEE", "LTOE", "RTOE", "LKneeAngles"]
        points = np.zeros((4, 4, 2), dtype=float)  # only 4 markers available

        fake_ezc3d = types.SimpleNamespace(
            c3d=lambda _path: {
                "header": {"points": {"frame_rate": 100.0}},
                "data": {"points": points},
                "parameters": {"POINT": {"LABELS": {"value": labels}}},
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            c3d_path = Path(tmp) / "dummy.c3d"
            c3d_path.write_bytes(b"")
            with mock.patch.dict(sys.modules, {"ezc3d": fake_ezc3d}):
                with mock.patch.object(_io.logger, "debug") as debug_mock:
                    out = _io.load_c3d(str(c3d_path), marker_set="pig")
        self.assertEqual(out["n_frames"], 2)
        self.assertEqual(len(out["angle_frames"]), 2)
        self.assertTrue(debug_mock.called)

    def test_external_angles_auto_aligns_on_second_hs_when_shorter(self):
        labels = ["LHEE", "RHEE", "LTOE", "RTOE"]
        points = np.zeros((4, len(labels), 10), dtype=float)

        fake_ezc3d = types.SimpleNamespace(
            c3d=lambda _path: {
                "header": {"points": {"frame_rate": 100.0}},
                "data": {"points": points},
                "parameters": {"POINT": {"LABELS": {"value": labels}}},
            }
        )
        angles = {
            "Lhip": [10.0, 11.0, 12.0, 13.0],
            "Rhip": [20.0, 21.0, 22.0, 23.0],
            "Lknee": [30.0, 31.0, 32.0, 33.0],
            "Rknee": [40.0, 41.0, 42.0, 43.0],
            "Lankle": [50.0, 51.0, 52.0, 53.0],
            "Rankle": [60.0, 61.0, 62.0, 63.0],
        }

        with tempfile.TemporaryDirectory() as tmp:
            c3d_path = Path(tmp) / "aligned.c3d"
            c3d_path.write_bytes(b"")
            with mock.patch.dict(sys.modules, {"ezc3d": fake_ezc3d}):
                with mock.patch("gaitkit._io._extract_hs_frames_from_c3d", return_value=[2, 5]):
                    out = _io.load_c3d(str(c3d_path), marker_set="pig", angles=angles, angles_align="auto")

        self.assertEqual(out["angle_frames"][0]["left_hip_angle"], 10.0)
        self.assertEqual(out["angle_frames"][4]["left_hip_angle"], 10.0)
        self.assertEqual(out["angle_frames"][5]["left_hip_angle"], 10.0)
        self.assertEqual(out["angle_frames"][8]["left_hip_angle"], 13.0)
        self.assertEqual(out["angle_frames"][9]["left_hip_angle"], 13.0)

    def test_external_angles_auto_resamples_when_no_hs(self):
        labels = ["LHEE", "RHEE", "LTOE", "RTOE"]
        points = np.zeros((4, len(labels), 6), dtype=float)

        fake_ezc3d = types.SimpleNamespace(
            c3d=lambda _path: {
                "header": {"points": {"frame_rate": 100.0}},
                "data": {"points": points},
                "parameters": {"POINT": {"LABELS": {"value": labels}}},
            }
        )
        angles = {
            "Lhip": [0.0, 1.0, 2.0],
            "Rhip": [0.0, 1.0, 2.0],
            "Lknee": [0.0, 1.0, 2.0],
            "Rknee": [0.0, 1.0, 2.0],
            "Lankle": [0.0, 1.0, 2.0],
            "Rankle": [0.0, 1.0, 2.0],
        }

        with tempfile.TemporaryDirectory() as tmp:
            c3d_path = Path(tmp) / "resample.c3d"
            c3d_path.write_bytes(b"")
            with mock.patch.dict(sys.modules, {"ezc3d": fake_ezc3d}):
                with mock.patch("gaitkit._io._extract_hs_frames_from_c3d", return_value=[]):
                    out = _io.load_c3d(str(c3d_path), marker_set="pig", angles=angles, angles_align="auto")

        vals = [fr["left_hip_angle"] for fr in out["angle_frames"]]
        self.assertEqual(len(vals), 6)
        self.assertAlmostEqual(vals[0], 0.0)
        self.assertAlmostEqual(vals[-1], 2.0)
        self.assertGreater(vals[3], vals[2])


if __name__ == "__main__":
    unittest.main()
