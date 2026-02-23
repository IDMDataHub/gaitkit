"""
Native-accelerated Ghoussayni detector (non-intrusive integration).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from gaitkit.utils.preprocessing import interpolate_nan

try:
    from gaitkit.native import _gait_native as _native_solver
    _HAS_NATIVE = True
except ImportError:
    _native_solver = None
    _HAS_NATIVE = False

from gaitkit.detectors.ghoussayni_detector import (
    GhoussayniDetector,
    GaitEvent,
    _build_cycles,
)
from gaitkit.detectors.axis_utils import detect_axes


class GhoussayniNativeDetector(GhoussayniDetector):
    """Drop-in replacement for GhoussayniDetector with native zero-crossings."""

    def detect(self, angle_frames):
        n = len(angle_frames)
        if n < 30:
            return [], [], []

        _, vax = detect_axes(angle_frames)

        left_heel_z = np.zeros(n)
        right_heel_z = np.zeros(n)
        left_toe_z = np.zeros(n)
        right_toe_z = np.zeros(n)

        for i, f in enumerate(angle_frames):
            if f.landmark_positions:
                lh = f.landmark_positions.get("left_heel")
                rh = f.landmark_positions.get("right_heel")
                if lh is None or (lh[0] == 0 and lh[1] == 0 and lh[2] == 0):
                    lh = f.landmark_positions.get("left_ankle", (0, 0, 0.5))
                if rh is None or (rh[0] == 0 and rh[1] == 0 and rh[2] == 0):
                    rh = f.landmark_positions.get("right_ankle", (0, 0, 0.5))
                left_heel_z[i] = lh[vax]
                right_heel_z[i] = rh[vax]

                lt = f.landmark_positions.get("left_toe")
                if lt is None:
                    lt = f.landmark_positions.get("left_foot_index")
                if lt is None:
                    lt = f.landmark_positions.get("left_ankle", (0, 0, 0.5))
                rt = f.landmark_positions.get("right_toe")
                if rt is None:
                    rt = f.landmark_positions.get("right_foot_index")
                if rt is None:
                    rt = f.landmark_positions.get("right_ankle", (0, 0, 0.5))
                left_toe_z[i] = lt[vax]
                right_toe_z[i] = rt[vax]
            else:
                left_heel_z[i] = 0.5
                right_heel_z[i] = 0.5
                left_toe_z[i] = 0.5
                right_toe_z[i] = 0.5

        if np.std(left_heel_z) < 0.001 or np.std(right_heel_z) < 0.001:
            return [], [], []

        left_heel_z = interpolate_nan(left_heel_z)
        right_heel_z = interpolate_nan(right_heel_z)
        left_toe_z = interpolate_nan(left_toe_z)
        right_toe_z = interpolate_nan(right_toe_z)

        if n > self.smooth_window:
            left_heel_z = savgol_filter(left_heel_z, self.smooth_window, 3)
            right_heel_z = savgol_filter(right_heel_z, self.smooth_window, 3)
            left_toe_z = savgol_filter(left_toe_z, self.smooth_window, 3)
            right_toe_z = savgol_filter(right_toe_z, self.smooth_window, 3)
        else:
            left_heel_z = gaussian_filter1d(left_heel_z, 2)
            right_heel_z = gaussian_filter1d(right_heel_z, 2)
            left_toe_z = gaussian_filter1d(left_toe_z, 2)
            right_toe_z = gaussian_filter1d(right_toe_z, 2)

        left_heel_vel = np.gradient(left_heel_z, 1 / self.fps)
        right_heel_vel = np.gradient(right_heel_z, 1 / self.fps)
        left_toe_vel = np.gradient(left_toe_z, 1 / self.fps)
        right_toe_vel = np.gradient(right_toe_z, 1 / self.fps)

        if _HAS_NATIVE:
            hs_l = np.asarray(
                _native_solver.zero_crossings_neg_to_pos(
                    np.asarray(left_heel_vel, dtype=np.float64),
                    int(self.min_interval),
                ),
                dtype=np.int32,
            )
            hs_r = np.asarray(
                _native_solver.zero_crossings_neg_to_pos(
                    np.asarray(right_heel_vel, dtype=np.float64),
                    int(self.min_interval),
                ),
                dtype=np.int32,
            )
            to_l = np.asarray(
                _native_solver.zero_crossings_neg_to_pos(
                    np.asarray(left_toe_vel, dtype=np.float64),
                    int(self.min_interval),
                ),
                dtype=np.int32,
            )
            to_r = np.asarray(
                _native_solver.zero_crossings_neg_to_pos(
                    np.asarray(right_toe_vel, dtype=np.float64),
                    int(self.min_interval),
                ),
                dtype=np.int32,
            )
        else:
            return super().detect(angle_frames)

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
