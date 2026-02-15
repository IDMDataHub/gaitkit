"""
Zeni Gait Event Detector.

Reference
---------
Zeni, J. A., Richards, J. G., & Higginson, J. S. (2008).
Two simple methods for determining gait events during treadmill and
overground walking using kinematic data.
*Gait & Posture*, 27(4), 710--714.
https://doi.org/10.1016/j.gaitpost.2007.07.007

Principle
---------
Heel strike (HS) is detected when the ankle (or heel) is at its maximum
anterior position relative to the pelvis.  Toe off (TO) is detected at
the maximum posterior position.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

from .axis_utils import detect_axes, detect_walking_direction

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


class ZeniDetector:
    """Coordinate-based gait event detector (Zeni et al., 2008).

    HS = maximum of ankle-pelvis AP distance (foot maximally anterior).
    TO = minimum of ankle-pelvis AP distance (foot maximally posterior).

    The AP axis is auto-detected from the data (largest range of ankle
    trajectory).

    Parameters
    ----------
    fps : float
        Sampling rate in Hz.
    min_cycle_duration : float
        Minimum allowed gait cycle duration in seconds.
    max_cycle_duration : float
        Maximum allowed gait cycle duration in seconds.
    smooth_window : int
        Gaussian smoothing window (number of frames).
    """

    def __init__(self, fps: float = 100.0, min_cycle_duration: float = 0.4,
                 max_cycle_duration: float = 2.0, smooth_window: int = 5):
        if fps <= 0:
            raise ValueError("fps must be strictly positive")
        if min_cycle_duration <= 0:
            raise ValueError("min_cycle_duration must be strictly positive")
        if max_cycle_duration <= 0:
            raise ValueError("max_cycle_duration must be strictly positive")
        if max_cycle_duration < min_cycle_duration:
            raise ValueError("max_cycle_duration must be >= min_cycle_duration")
        self.fps = fps
        self.min_peak_distance = int(min_cycle_duration * fps / 2)
        self.max_peak_distance = int(max_cycle_duration * fps)
        self.smooth_window = smooth_window

    def detect(self, angle_frames) -> Tuple[List[GaitEvent], List[GaitEvent], List[GaitCycle]]:
        """Detect gait events from a sequence of angle frames.

        Parameters
        ----------
        angle_frames : list
            Sequence of AngleFrame objects with ``landmark_positions``.

        Returns
        -------
        heel_strikes : list of GaitEvent
        toe_offs : list of GaitEvent
        cycles : list of GaitCycle
        """
        n = len(angle_frames)
        if n < 30:
            return [], [], []

        # Auto-detect AP axis and walking direction
        ap_axis, _ = detect_axes(angle_frames)
        direction = detect_walking_direction(angle_frames, ap_axis)

        pelvis_ap = np.array([
            (f.landmark_positions.get('left_hip', (0.5, 0, 0))[ap_axis] +
             f.landmark_positions.get('right_hip', (0.5, 0, 0))[ap_axis]) / 2
            if f.landmark_positions else 0.5
            for f in angle_frames])

        left_ankle_rel = np.array([
            f.landmark_positions.get('left_ankle', (0.5, 0, 0))[ap_axis] - pelvis_ap[i]
            if f.landmark_positions else 0
            for i, f in enumerate(angle_frames)])
        right_ankle_rel = np.array([
            f.landmark_positions.get('right_ankle', (0.5, 0, 0))[ap_axis] - pelvis_ap[i]
            if f.landmark_positions else 0
            for i, f in enumerate(angle_frames)])

        # Negate when walking in negative direction so peaks always = HS
        if direction < 0:
            left_ankle_rel = -left_ankle_rel
            right_ankle_rel = -right_ankle_rel

        if self.smooth_window > 1 and n > self.smooth_window:
            left_ankle_rel = gaussian_filter1d(left_ankle_rel, self.smooth_window / 3)
            right_ankle_rel = gaussian_filter1d(right_ankle_rel, self.smooth_window / 3)

        hs_l, _ = find_peaks(left_ankle_rel, distance=self.min_peak_distance, prominence=0.01)
        hs_r, _ = find_peaks(right_ankle_rel, distance=self.min_peak_distance, prominence=0.01)
        to_l, _ = find_peaks(-left_ankle_rel, distance=self.min_peak_distance, prominence=0.01)
        to_r, _ = find_peaks(-right_ankle_rel, distance=self.min_peak_distance, prominence=0.01)

        heel_strikes = sorted(
            [GaitEvent(int(f), f / self.fps, 'heel_strike', 'left') for f in hs_l] +
            [GaitEvent(int(f), f / self.fps, 'heel_strike', 'right') for f in hs_r],
            key=lambda e: e.frame_index)

        toe_offs = sorted(
            [GaitEvent(int(f), f / self.fps, 'toe_off', 'left') for f in to_l] +
            [GaitEvent(int(f), f / self.fps, 'toe_off', 'right') for f in to_r],
            key=lambda e: e.frame_index)

        cycles = _build_cycles(heel_strikes, toe_offs, self.fps)
        return heel_strikes, toe_offs, cycles

    # Alias so the benchmark runner finds a uniform interface
    detect_gait_events = detect
