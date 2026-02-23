"""
Mickelborough Gait Event Detector.

Reference
---------
Mickelborough, J., van der Linden, M. L., Richards, J., & Ennos, A. R.
(2000). Validity and reliability of a kinematic protocol for determining
foot contact events.
*Gait & Posture*, 11(1), 32--37.
https://doi.org/10.1016/S0966-6362(99)00050-8

Principle
---------
HS is detected when the heel vertical velocity crosses an adaptive
threshold near zero as the foot descends (velocity goes from negative
to near-zero = foot stops descending).  TO is detected when the heel
vertical velocity crosses the threshold as the foot ascends (velocity
goes from near-zero to positive = foot starts rising).
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass
from scipy.signal import savgol_filter

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


class MickelboroughDetector:
    """Heel vertical velocity threshold detector (Mickelborough et al., 2000).

    HS = heel vertical velocity crosses adaptive threshold from negative
         to near-zero (foot stops descending).
    TO = heel vertical velocity crosses adaptive threshold from near-zero
         to positive (foot starts rising).
    """

    def __init__(self, fps: float = 100.0, smooth_window: int = 7,
                 min_cycle_duration: float = 0.4,
                 threshold_fraction: float = 0.05):
        if fps <= 0:
            raise ValueError("fps must be strictly positive")
        if smooth_window < 5:
            raise ValueError("smooth_window must be >= 5")
        if min_cycle_duration <= 0:
            raise ValueError("min_cycle_duration must be strictly positive")
        if threshold_fraction < 0:
            raise ValueError("threshold_fraction must be >= 0")
        self.fps = fps
        self.smooth_window = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
        self.min_interval = int(min_cycle_duration * fps / 2)
        self.threshold_fraction = threshold_fraction

    def detect(self, angle_frames):
        n = len(angle_frames)
        if n < 30:
            return [], [], []

        # Auto-detect vertical axis
        _, vax = detect_axes(angle_frames)

        # Extract HEEL vertical position
        left_heel_z = np.array([
            f.landmark_positions.get('left_heel', (0, 0, 0.5))[vax]
            if f.landmark_positions else 0.5 for f in angle_frames])
        right_heel_z = np.array([
            f.landmark_positions.get('right_heel', (0, 0, 0.5))[vax]
            if f.landmark_positions else 0.5 for f in angle_frames])

        left_heel_z = interpolate_nan(left_heel_z)
        right_heel_z = interpolate_nan(right_heel_z)

        if n > self.smooth_window:
            left_heel_z = savgol_filter(left_heel_z, self.smooth_window, 3)
            right_heel_z = savgol_filter(right_heel_z, self.smooth_window, 3)

        # Compute vertical velocity
        left_vel_z = np.gradient(left_heel_z, 1 / self.fps)
        right_vel_z = np.gradient(right_heel_z, 1 / self.fps)

        def _detect_events_side(vel_z):
            """Detect HS and TO from heel vertical velocity using adaptive threshold."""
            vel_range = np.ptp(vel_z)
            threshold = self.threshold_fraction * vel_range

            hs_frames = []
            to_frames = []

            for i in range(1, len(vel_z)):
                # HS: velocity crosses threshold from negative to near-zero
                # (foot descending then stopping)
                if vel_z[i - 1] < -threshold and vel_z[i] >= -threshold:
                    hs_frames.append(i)
                # TO: velocity crosses threshold from near-zero to positive
                # (foot starting to rise)
                elif vel_z[i - 1] <= threshold and vel_z[i] > threshold:
                    to_frames.append(i)

            # Filter events that are too close together
            hs_frames = _filter_close(hs_frames, self.min_interval)
            to_frames = _filter_close(to_frames, self.min_interval)

            return hs_frames, to_frames

        def _filter_close(events, min_dist):
            if len(events) < 2:
                return events
            out = [events[0]]
            for e in events[1:]:
                if e - out[-1] >= min_dist:
                    out.append(e)
            return out

        hs_l, to_l = _detect_events_side(left_vel_z)
        hs_r, to_r = _detect_events_side(right_vel_z)

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
