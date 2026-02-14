"""
O'Connor Gait Event Detector.

Reference
---------
O'Connor, C. M., Thorpe, S. K., O'Malley, M. J., & Vaughan, C. L. (2007).
Automatic detection of gait events using kinematic data.
*Gait & Posture*, 25(3), 469--474.
https://doi.org/10.1016/j.gaitpost.2006.05.016

Principle
---------
HS is detected at the negative peak (local minimum) of the heel vertical
velocity -- the fastest descent just before ground contact.  TO is
detected at the positive peak (local maximum) of the toe vertical
velocity -- the fastest ascent at lift-off.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from scipy.signal import find_peaks, savgol_filter, butter, filtfilt
from scipy.ndimage import gaussian_filter1d

from .axis_utils import detect_axes

@dataclass
class GaitEvent:
    """A single gait event (heel strike or toe off).

    Attributes
    ----------
    frame_index : int
        Zero-based frame number within the trial.
    time : float
        Timestamp in seconds (frame_index / fps).
    event_type : str
        Either ``'heel_strike'`` or ``'toe_off'``.
    side : str
        ``'left'`` or ``'right'``.
    confidence : float
        Detection confidence in [0, 1].
    """
    frame_index: int
    time: float
    event_type: str
    side: str
    confidence: float = 1.0


@dataclass
class GaitCycle:
    """A complete gait cycle delimited by two consecutive heel strikes.

    Attributes
    ----------
    cycle_id : int
        Sequential cycle identifier.
    side : str
        ``'left'`` or ``'right'``.
    start_frame : int
        Frame of the initial heel strike.
    toe_off_frame : int or None
        Frame of the intervening toe off, if detected.
    end_frame : int
        Frame of the terminal heel strike.
    duration : float
        Cycle duration in seconds.
    stance_percentage : float or None
        Stance phase as a percentage of the full cycle.
    """
    cycle_id: int
    side: str
    start_frame: int
    toe_off_frame: Optional[int]
    end_frame: int
    duration: float
    stance_percentage: Optional[float]

def _build_cycles(heel_strikes: List[GaitEvent], toe_offs: List[GaitEvent],
                  fps: float) -> List[GaitCycle]:
    """Build gait cycles from detected heel strikes and toe offs."""
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


class OConnorDetector:
    """Vertical velocity peak detector (O'Connor et al., 2007).

    HS = negative peak (local minimum) of heel vertical velocity.
    TO = positive peak (local maximum) of toe vertical velocity.

    Parameters
    ----------
    fps : float
        Sampling rate in Hz.
    smooth_window : int
        Savitzky-Golay window length (must be odd).
    min_cycle_duration : float
        Minimum allowed gait cycle duration in seconds.
    """

    def __init__(self, fps: float = 100.0, smooth_window: int = 7,
                 min_cycle_duration: float = 0.4):
        self.fps = fps
        self.smooth_window = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
        self.min_interval = int(min_cycle_duration * fps / 2)

    def detect(self, angle_frames) -> Tuple[List[GaitEvent], List[GaitEvent], List[GaitCycle]]:
        n = len(angle_frames)
        if n < 30:
            return [], [], []

        # Auto-detect vertical axis
        _, vax = detect_axes(angle_frames)

        # Extract HEEL vertical position (for HS detection)
        left_heel_z = np.array([
            f.landmark_positions.get('left_heel', (0, 0, 0.5))[vax]
            if f.landmark_positions else 0.5 for f in angle_frames])
        right_heel_z = np.array([
            f.landmark_positions.get('right_heel', (0, 0, 0.5))[vax]
            if f.landmark_positions else 0.5 for f in angle_frames])

        # Extract TOE vertical position (for TO detection)
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

        # Compute vertical velocities
        left_heel_vz = np.gradient(left_heel_z, 1 / self.fps)
        right_heel_vz = np.gradient(right_heel_z, 1 / self.fps)
        left_toe_vz = np.gradient(left_toe_z, 1 / self.fps)
        right_toe_vz = np.gradient(right_toe_z, 1 / self.fps)

        # HS = negative peak of heel vertical velocity
        # Use find_peaks on -heel_vz to find the most negative velocities
        hs_l, _ = find_peaks(-left_heel_vz, distance=self.min_interval,
                              prominence=np.std(left_heel_vz) * 0.3)
        hs_r, _ = find_peaks(-right_heel_vz, distance=self.min_interval,
                              prominence=np.std(right_heel_vz) * 0.3)

        # TO = positive peak of toe vertical velocity
        # Use find_peaks on toe_vz to find the most positive velocities
        to_l, _ = find_peaks(left_toe_vz, distance=self.min_interval,
                              prominence=np.std(left_toe_vz) * 0.3)
        to_r, _ = find_peaks(right_toe_vz, distance=self.min_interval,
                              prominence=np.std(right_toe_vz) * 0.3)

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
