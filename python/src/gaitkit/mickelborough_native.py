"""
Native-accelerated Mickelborough detector (non-intrusive integration).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

try:
    from gaitkit.native import _gait_native as _native_solver
    _HAS_NATIVE = True
except ImportError:
    _native_solver = None
    _HAS_NATIVE = False

from gaitkit.detectors.mickelborough_detector import (
    MickelboroughDetector,
    GaitEvent,
    _build_cycles,
)
from gaitkit.detectors.axis_utils import detect_axes


class MickelboroughNativeDetector(MickelboroughDetector):
    """Drop-in replacement for MickelboroughDetector with native crossings."""

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

        if n > self.smooth_window:
            left_heel_z = savgol_filter(left_heel_z, self.smooth_window, 3)
            right_heel_z = savgol_filter(right_heel_z, self.smooth_window, 3)

        left_vel_z = np.gradient(left_heel_z, 1 / self.fps)
        right_vel_z = np.gradient(right_heel_z, 1 / self.fps)

        if _HAS_NATIVE:
            hs_l, to_l = _native_solver.mickelborough_events_raw(
                np.asarray(left_vel_z, dtype=np.float64),
                float(self.threshold_fraction),
                int(self.min_interval),
            )
            hs_r, to_r = _native_solver.mickelborough_events_raw(
                np.asarray(right_vel_z, dtype=np.float64),
                float(self.threshold_fraction),
                int(self.min_interval),
            )
            hs_l = np.asarray(hs_l, dtype=np.int32)
            to_l = np.asarray(to_l, dtype=np.int32)
            hs_r = np.asarray(hs_r, dtype=np.int32)
            to_r = np.asarray(to_r, dtype=np.int32)
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
