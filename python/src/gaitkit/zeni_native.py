"""
Native-accelerated Zeni detector (non-intrusive integration).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

try:
    from gaitkit.native import _gait_native as _native_solver
    _HAS_NATIVE = True
except Exception:
    _native_solver = None
    _HAS_NATIVE = False

try:
    from gaitkit.detectors.zeni_detector import (
        ZeniDetector,
        GaitEvent,
        _build_cycles,
    )
    from gaitkit.detectors.axis_utils import detect_axes, detect_walking_direction
except Exception:
    from gaitkit.detectors.zeni_detector import (
        ZeniDetector,
        GaitEvent,
        _build_cycles,
    )
    from gaitkit.detectors.axis_utils import detect_axes, detect_walking_direction


class ZeniNativeDetector(ZeniDetector):
    """Drop-in replacement for ZeniDetector with native peak selection."""

    def detect(self, angle_frames):
        n = len(angle_frames)
        if n < 30:
            return [], [], []

        ap_axis, _ = detect_axes(angle_frames)
        direction = detect_walking_direction(angle_frames, ap_axis)

        pelvis_ap = np.array(
            [
                (
                    f.landmark_positions.get("left_hip", (0.5, 0, 0))[ap_axis]
                    + f.landmark_positions.get("right_hip", (0.5, 0, 0))[ap_axis]
                )
                / 2
                if f.landmark_positions
                else 0.5
                for f in angle_frames
            ]
        )

        left_ankle_rel = np.array(
            [
                f.landmark_positions.get("left_ankle", (0.5, 0, 0))[ap_axis] - pelvis_ap[i]
                if f.landmark_positions
                else 0
                for i, f in enumerate(angle_frames)
            ]
        )
        right_ankle_rel = np.array(
            [
                f.landmark_positions.get("right_ankle", (0.5, 0, 0))[ap_axis] - pelvis_ap[i]
                if f.landmark_positions
                else 0
                for i, f in enumerate(angle_frames)
            ]
        )

        if direction < 0:
            left_ankle_rel = -left_ankle_rel
            right_ankle_rel = -right_ankle_rel

        if self.smooth_window > 1 and n > self.smooth_window:
            left_ankle_rel = gaussian_filter1d(left_ankle_rel, self.smooth_window / 3)
            right_ankle_rel = gaussian_filter1d(right_ankle_rel, self.smooth_window / 3)

        if _HAS_NATIVE:
            hs_l = np.asarray(
                _native_solver.find_peaks_filtered(
                    np.asarray(left_ankle_rel, dtype=np.float64),
                    int(self.min_peak_distance),
                    0.01,
                    1,
                ),
                dtype=np.int32,
            )
            hs_r = np.asarray(
                _native_solver.find_peaks_filtered(
                    np.asarray(right_ankle_rel, dtype=np.float64),
                    int(self.min_peak_distance),
                    0.01,
                    1,
                ),
                dtype=np.int32,
            )
            to_l = np.asarray(
                _native_solver.find_peaks_filtered(
                    np.asarray(left_ankle_rel, dtype=np.float64),
                    int(self.min_peak_distance),
                    0.01,
                    -1,
                ),
                dtype=np.int32,
            )
            to_r = np.asarray(
                _native_solver.find_peaks_filtered(
                    np.asarray(right_ankle_rel, dtype=np.float64),
                    int(self.min_peak_distance),
                    0.01,
                    -1,
                ),
                dtype=np.int32,
            )
        else:
            hs_l, _ = find_peaks(left_ankle_rel, distance=self.min_peak_distance, prominence=0.01)
            hs_r, _ = find_peaks(right_ankle_rel, distance=self.min_peak_distance, prominence=0.01)
            to_l, _ = find_peaks(-left_ankle_rel, distance=self.min_peak_distance, prominence=0.01)
            to_r, _ = find_peaks(-right_ankle_rel, distance=self.min_peak_distance, prominence=0.01)

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

        cycles = _build_cycles(heel_strikes, toe_offs, self.fps)
        return heel_strikes, toe_offs, cycles

    detect_gait_events = detect
