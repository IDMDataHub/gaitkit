"""
Ghoussayni Gait Event Detector.

Reference
---------
Ghoussayni, S., Stevens, C., Durham, S., & Ewins, D. (2004).
Assessment and validation of a simple automated method for the
detection of gait events and intervals.
*Gait & Posture*, 20(3), 266--272.
https://doi.org/10.1016/j.gaitpost.2003.10.001

Principle
---------
HS is detected when the heel vertical velocity crosses zero from
negative to positive (the heel descends then reverses at contact).
TO is detected when the toe vertical velocity crosses zero from
negative to positive (the toe descends then lifts off).
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

from ..utils.preprocessing import interpolate_nan
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


class GhoussayniDetector:
    """Vertical velocity zero-crossing detector (Ghoussayni et al., 2004).

    HS = heel vertical velocity zero-crossing from negative to positive
         (heel descends then reverses at ground contact).
    TO = toe vertical velocity zero-crossing from negative to positive
         (toe descends then lifts off the ground).
    """

    def __init__(self, fps: float = 100.0, smooth_window: int = 7,
                 min_cycle_duration: float = 0.4):
        if fps <= 0:
            raise ValueError("fps must be strictly positive")
        if smooth_window < 5:
            raise ValueError("smooth_window must be >= 5")
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
        left_heel_z = np.zeros(n)
        right_heel_z = np.zeros(n)
        # Extract TOE vertical position (for TO)
        left_toe_z = np.zeros(n)
        right_toe_z = np.zeros(n)

        for i, f in enumerate(angle_frames):
            if f.landmark_positions:
                lh = f.landmark_positions.get('left_heel')
                rh = f.landmark_positions.get('right_heel')
                if lh is None or (lh[0] == 0 and lh[1] == 0 and lh[2] == 0):
                    lh = f.landmark_positions.get('left_ankle', (0, 0, 0.5))
                if rh is None or (rh[0] == 0 and rh[1] == 0 and rh[2] == 0):
                    rh = f.landmark_positions.get('right_ankle', (0, 0, 0.5))
                left_heel_z[i] = lh[vax]
                right_heel_z[i] = rh[vax]

                lt = f.landmark_positions.get('left_toe')
                if lt is None:
                    lt = f.landmark_positions.get('left_foot_index')
                if lt is None:
                    lt = f.landmark_positions.get('left_ankle', (0, 0, 0.5))
                rt = f.landmark_positions.get('right_toe')
                if rt is None:
                    rt = f.landmark_positions.get('right_foot_index')
                if rt is None:
                    rt = f.landmark_positions.get('right_ankle', (0, 0, 0.5))
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

        # Compute vertical velocities
        left_heel_vel = np.gradient(left_heel_z, 1 / self.fps)
        right_heel_vel = np.gradient(right_heel_z, 1 / self.fps)
        left_toe_vel = np.gradient(left_toe_z, 1 / self.fps)
        right_toe_vel = np.gradient(right_toe_z, 1 / self.fps)

        def _zero_crossings_neg_to_pos(vel):
            """Find zero-crossings from negative to positive."""
            crossings = []
            for i in range(1, len(vel)):
                if vel[i - 1] < 0 and vel[i] >= 0:
                    crossings.append(i)
            return crossings

        def _filter_close(events, min_dist):
            if len(events) < 2:
                return events
            out = [events[0]]
            for e in events[1:]:
                if e - out[-1] >= min_dist:
                    out.append(e)
            return out

        # HS: heel vertical velocity zero-crossing from negative to positive
        hs_l = _filter_close(_zero_crossings_neg_to_pos(left_heel_vel), self.min_interval)
        hs_r = _filter_close(_zero_crossings_neg_to_pos(right_heel_vel), self.min_interval)

        # TO: toe vertical velocity zero-crossing from negative to positive
        to_l = _filter_close(_zero_crossings_neg_to_pos(left_toe_vel), self.min_interval)
        to_r = _filter_close(_zero_crossings_neg_to_pos(right_toe_vel), self.min_interval)

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
