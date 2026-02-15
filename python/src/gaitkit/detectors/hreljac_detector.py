"""
Hreljac Gait Event Detector.

Reference
---------
Hreljac, A., & Marshall, R. N. (2000).
Algorithms to determine event timing during normal walking using
kinematic data.
*Journal of Biomechanics*, 33(6), 783--786.
https://doi.org/10.1016/S0021-9290(00)00014-2

Principle
---------
HS is detected at peaks in the vertical acceleration of the heel marker
(impact transient).  TO is identified from negative acceleration peaks
of the TOE marker (push-off phase).
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass
from scipy.signal import find_peaks, savgol_filter

from .axis_utils import detect_axes


@dataclass
class GaitEvent:
    """A single gait event."""
    frame_index: int
    time: float
    event_type: str
    side: str
    confidence: float = 1.0


@dataclass
class GaitCycle:
    """A complete gait cycle."""
    cycle_id: int
    side: str
    start_frame: int
    toe_off_frame: Optional[int]
    end_frame: int
    duration: float
    stance_percentage: Optional[float]


def _build_cycles(heel_strikes, toe_offs, fps):
    cycles = []
    for side in ['left', 'right']:
        side_hs = sorted([e for e in heel_strikes if e.side == side],
                         key=lambda x: x.frame_index)
        side_to = sorted([e for e in toe_offs if e.side == side],
                         key=lambda x: x.frame_index)
        for i in range(len(side_hs) - 1):
            start, end = side_hs[i], side_hs[i + 1]
            duration = (end.frame_index - start.frame_index) / fps
            to_in = next((t for t in side_to
                          if start.frame_index < t.frame_index < end.frame_index), None)
            stance_pct = None
            if to_in:
                stance_pct = ((to_in.frame_index - start.frame_index) /
                              (end.frame_index - start.frame_index) * 100)
            cycles.append(GaitCycle(
                cycle_id=len(cycles), side=side,
                start_frame=start.frame_index,
                toe_off_frame=to_in.frame_index if to_in else None,
                end_frame=end.frame_index,
                duration=duration, stance_percentage=stance_pct))
    return sorted(cycles, key=lambda c: c.start_frame)


class HreljacDetector:
    """Heel/toe vertical acceleration peak detector (Hreljac & Marshall, 2000).

    HS = maximum of heel vertical acceleration (impact transient).
    TO = minimum of toe vertical acceleration (push-off).
    """

    def __init__(self, fps: float = 100.0, smooth_window: int = 9,
                 min_cycle_duration: float = 0.4):
        if fps <= 0:
            raise ValueError("fps must be strictly positive")
        if min_cycle_duration <= 0:
            raise ValueError("min_cycle_duration must be strictly positive")
        self.fps = fps
        self.smooth_window = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
        self.min_interval = int(min_cycle_duration * fps / 2)

    def detect(self, angle_frames):
        n = len(angle_frames)
        if n < 30:
            return [], [], []

        # Auto-detect vertical axis
        _, vax = detect_axes(angle_frames)

        # Extract HEEL vertical position (for HS)
        left_heel_z = np.array([
            f.landmark_positions.get('left_heel', (0, 0, 0.5))[vax]
            if f.landmark_positions else 0.5 for f in angle_frames])
        right_heel_z = np.array([
            f.landmark_positions.get('right_heel', (0, 0, 0.5))[vax]
            if f.landmark_positions else 0.5 for f in angle_frames])

        # Extract TOE vertical position (for TO)
        left_toe_z = np.array([
            (f.landmark_positions.get('left_toe', None) or
             f.landmark_positions.get('left_foot_index', (0, 0, 0.5)))[vax]
            if f.landmark_positions else 0.5 for f in angle_frames])
        right_toe_z = np.array([
            (f.landmark_positions.get('right_toe', None) or
             f.landmark_positions.get('right_foot_index', (0, 0, 0.5)))[vax]
            if f.landmark_positions else 0.5 for f in angle_frames])

        if n > self.smooth_window:
            left_heel_z = savgol_filter(left_heel_z, self.smooth_window, 3)
            right_heel_z = savgol_filter(right_heel_z, self.smooth_window, 3)
            left_toe_z = savgol_filter(left_toe_z, self.smooth_window, 3)
            right_toe_z = savgol_filter(right_toe_z, self.smooth_window, 3)

        dt = 1 / self.fps

        # Heel vertical acceleration (for HS)
        left_heel_acc = np.gradient(np.gradient(left_heel_z, dt), dt)
        right_heel_acc = np.gradient(np.gradient(right_heel_z, dt), dt)

        # Toe vertical acceleration (for TO)
        left_toe_acc = np.gradient(np.gradient(left_toe_z, dt), dt)
        right_toe_acc = np.gradient(np.gradient(right_toe_z, dt), dt)

        # HS: maximum of heel vertical acceleration (impact transient)
        hs_l, _ = find_peaks(left_heel_acc, distance=self.min_interval,
                              prominence=np.std(left_heel_acc) * 0.5)
        hs_r, _ = find_peaks(right_heel_acc, distance=self.min_interval,
                              prominence=np.std(right_heel_acc) * 0.5)

        # TO: minimum of toe vertical acceleration (push-off)
        to_l, _ = find_peaks(-left_toe_acc, distance=self.min_interval,
                              prominence=np.std(left_toe_acc) * 0.3)
        to_r, _ = find_peaks(-right_toe_acc, distance=self.min_interval,
                              prominence=np.std(right_toe_acc) * 0.3)

        heel_strikes = sorted(
            [GaitEvent(int(f), f / self.fps, 'heel_strike', 'left') for f in hs_l] +
            [GaitEvent(int(f), f / self.fps, 'heel_strike', 'right') for f in hs_r],
            key=lambda e: e.frame_index)
        toe_offs = sorted(
            [GaitEvent(int(f), f / self.fps, 'toe_off', 'left') for f in to_l] +
            [GaitEvent(int(f), f / self.fps, 'toe_off', 'right') for f in to_r],
            key=lambda e: e.frame_index)

        return heel_strikes, toe_offs, _build_cycles(heel_strikes, toe_offs, self.fps)

    detect_gait_events = detect
