"""
Native-accelerated Hreljac detector (non-intrusive integration).
"""

from __future__ import annotations

import os
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from gaitkit.utils.preprocessing import interpolate_nan

try:
    from gaitkit.native import _gait_native as _native_solver
    _HAS_NATIVE = True
except ImportError:
    _native_solver = None
    _HAS_NATIVE = False

# Keep SciPy peak picking as default for exact parity; native path is opt-in.
_USE_EXPERIMENTAL_NATIVE_PEAKS = os.getenv("GAIT_NATIVE_EXPERIMENTAL_PEAKS", "0") == "1"

from gaitkit.detectors.hreljac_detector import (
    HreljacDetector,
    GaitEvent,
    _build_cycles,
)
from gaitkit.detectors.axis_utils import detect_axes


class HreljacNativeDetector(HreljacDetector):
    """Drop-in replacement for HreljacDetector with native peak selection."""

    def detect(self, angle_frames):
        n = len(angle_frames)
        if n < 30:
            return [], [], []

        _, vax = detect_axes(angle_frames)

        left_heel_z = np.array(
            [
                f.landmark_positions.get("left_heel", (0, 0, 0.5))[vax]
                if f.landmark_positions
                else 0.5
                for f in angle_frames
            ]
        )
        right_heel_z = np.array(
            [
                f.landmark_positions.get("right_heel", (0, 0, 0.5))[vax]
                if f.landmark_positions
                else 0.5
                for f in angle_frames
            ]
        )

        left_toe_z = np.array(
            [
                (
                    f.landmark_positions.get("left_toe", None)
                    or f.landmark_positions.get("left_foot_index", (0, 0, 0.5))
                )[vax]
                if f.landmark_positions
                else 0.5
                for f in angle_frames
            ]
        )
        right_toe_z = np.array(
            [
                (
                    f.landmark_positions.get("right_toe", None)
                    or f.landmark_positions.get("right_foot_index", (0, 0, 0.5))
                )[vax]
                if f.landmark_positions
                else 0.5
                for f in angle_frames
            ]
        )

        left_heel_z = interpolate_nan(left_heel_z)
        right_heel_z = interpolate_nan(right_heel_z)
        left_toe_z = interpolate_nan(left_toe_z)
        right_toe_z = interpolate_nan(right_toe_z)

        if n > self.smooth_window:
            left_heel_z = savgol_filter(left_heel_z, self.smooth_window, 3)
            right_heel_z = savgol_filter(right_heel_z, self.smooth_window, 3)
            left_toe_z = savgol_filter(left_toe_z, self.smooth_window, 3)
            right_toe_z = savgol_filter(right_toe_z, self.smooth_window, 3)

        dt = 1 / self.fps

        left_heel_acc = np.gradient(np.gradient(left_heel_z, dt), dt)
        right_heel_acc = np.gradient(np.gradient(right_heel_z, dt), dt)
        left_toe_acc = np.gradient(np.gradient(left_toe_z, dt), dt)
        right_toe_acc = np.gradient(np.gradient(right_toe_z, dt), dt)

        p_hs_l = float(np.std(left_heel_acc) * 0.5)
        p_hs_r = float(np.std(right_heel_acc) * 0.5)
        p_to_l = float(np.std(left_toe_acc) * 0.3)
        p_to_r = float(np.std(right_toe_acc) * 0.3)

        if _HAS_NATIVE and _USE_EXPERIMENTAL_NATIVE_PEAKS:
            hs_l = np.asarray(
                _native_solver.find_peaks_filtered(
                    np.asarray(left_heel_acc, dtype=np.float64),
                    int(self.min_interval),
                    p_hs_l,
                    1,
                ),
                dtype=np.int32,
            )
            hs_r = np.asarray(
                _native_solver.find_peaks_filtered(
                    np.asarray(right_heel_acc, dtype=np.float64),
                    int(self.min_interval),
                    p_hs_r,
                    1,
                ),
                dtype=np.int32,
            )
            to_l = np.asarray(
                _native_solver.find_peaks_filtered(
                    np.asarray(left_toe_acc, dtype=np.float64),
                    int(self.min_interval),
                    p_to_l,
                    -1,
                ),
                dtype=np.int32,
            )
            to_r = np.asarray(
                _native_solver.find_peaks_filtered(
                    np.asarray(right_toe_acc, dtype=np.float64),
                    int(self.min_interval),
                    p_to_r,
                    -1,
                ),
                dtype=np.int32,
            )
        else:
            hs_l, _ = find_peaks(left_heel_acc, distance=self.min_interval, prominence=p_hs_l)
            hs_r, _ = find_peaks(right_heel_acc, distance=self.min_interval, prominence=p_hs_r)
            to_l, _ = find_peaks(-left_toe_acc, distance=self.min_interval, prominence=p_to_l)
            to_r, _ = find_peaks(-right_toe_acc, distance=self.min_interval, prominence=p_to_r)

        heel_strikes = sorted(
            [GaitEvent(int(f), f / self.fps, "heel_strike", "left") for f in hs_l]
            + [GaitEvent(int(f), f / self.fps, "heel_strike", "right") for f in hs_r],
            key=lambda e: e.frame_index,
        )
        toe_offs = sorted(
            [GaitEvent(int(f), f / self.fps, "toe_off", "left") for f in to_l]
            + [GaitEvent(int(f), f / self.fps, "toe_off", "right") for f in to_r],
            key=lambda e: e.frame_index,
        )

        return heel_strikes, toe_offs, _build_cycles(heel_strikes, toe_offs, self.fps)

    detect_gait_events = detect
