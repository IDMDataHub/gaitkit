# -*- coding: utf-8 -*-
"""
Shared axis auto-detection utilities for gait event detectors.

Standard Vicon convention: X=mediolateral, Y=anteroposterior, Z=vertical.
Some labs use different conventions, so these functions auto-detect axes
from the data.

Author: Frederic Fer (f.fer@institut-myologie.org)
Affiliation: Myodata, Institut de Myologie, Paris, France
License: Apache-2.0
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def detect_axes(angle_frames):
    """Auto-detect progression (AP) and vertical axes from trajectory data.

    Parameters
    ----------
    angle_frames : list
        Sequence of AngleFrame objects with ``landmark_positions``.

    Returns
    -------
    ap_axis : int
        Index of the anteroposterior (progression) axis (0, 1, or 2).
        This is the axis with the largest range for ankle trajectory.
    vertical_axis : int
        Index of the vertical axis (0, 1, or 2).
        Defaults to Z (index 2); falls back to the axis with the smallest
        range among the non-AP axes.
    """
    # Extract ankle positions over time
    positions = []
    for af in angle_frames:
        if af.landmark_positions and "left_ankle" in af.landmark_positions:
            pos = af.landmark_positions["left_ankle"]
            if pos != (0.0, 0.0, 0.0):
                positions.append(pos)

    if len(positions) < 10:
        return 1, 2  # default: Y=AP, Z=vertical

    positions = np.array(positions)
    ranges = np.ptp(positions, axis=0)  # range per axis

    # AP axis = largest range (walking direction has biggest displacement)
    ap_axis = int(np.argmax(ranges))

    # Vertical axis = typically Z (index 2), confirmed by having smallest range
    # and characteristic double-bump oscillation pattern
    remaining = [i for i in range(3) if i != ap_axis]
    # Vertical usually has smaller range than mediolateral for walking
    vertical_axis = 2  # default Z
    if 2 == ap_axis:
        # If Z has largest range (unusual), pick the axis with smaller range
        vertical_axis = remaining[int(np.argmin([ranges[i] for i in remaining]))]

    return ap_axis, vertical_axis


def _direction_score_acceleration(angle_frames, ap_axis, direction, fps,
                                  smoothing_window=11):
    """Score a candidate walking direction using acceleration asymmetry.

    Biomechanical basis: at heel strike, the foot decelerates sharply
    (high-magnitude negative acceleration at position maxima).  At toe-off,
    the foot accelerates more gradually (lower-magnitude positive acceleration
    at position minima).  The correct direction is the one where position
    maxima (HS candidates) have a sharper curvature than position minima
    (TO candidates).

    Parameters
    ----------
    angle_frames : list
        Sequence of AngleFrame objects with ``landmark_positions``.
    ap_axis : int
        Index of the anteroposterior axis.
    direction : int
        Candidate direction sign (+1 or -1).
    fps : float
        Sampling rate in Hz (used to set minimum peak distance).
    smoothing_window : int
        Savitzky-Golay smoothing window length.

    Returns
    -------
    float
        Positive score favours this direction; negative disfavours it.
    """
    n = len(angle_frames)
    if n < 20:
        return 0.0

    # Extract pelvis and ankle AP positions, skipping invalid (0,0,0) frames
    px = np.zeros(n)
    la_raw = np.zeros(n)
    ra_raw = np.zeros(n)
    valid_mask = np.zeros(n, dtype=bool)
    for i in range(n):
        lp = angle_frames[i].landmark_positions
        if lp is None:
            continue
        lh = lp.get("left_hip", (0.0, 0, 0))
        rh = lp.get("right_hip", (0.0, 0, 0))
        la = lp.get("left_ankle", (0.0, 0, 0))
        ra = lp.get("right_ankle", (0.0, 0, 0))
        # Skip frames where key markers are missing (0,0,0)
        if lh == (0.0, 0.0, 0.0) or la == (0.0, 0.0, 0.0):
            continue
        valid_mask[i] = True
        px[i] = (lh[ap_axis] + rh[ap_axis]) / 2.0
        la_raw[i] = la[ap_axis] - px[i]
        ra_raw[i] = ra[ap_axis] - px[i]

    # Interpolate over short gaps in valid data
    if valid_mask.sum() < 20:
        return 0.0

    if direction < 0:
        la_raw = -la_raw
        ra_raw = -ra_raw

    sigma = max(1.0, smoothing_window / 6.0)
    la = gaussian_filter1d(la_raw, sigma)
    ra = gaussian_filter1d(ra_raw, sigma)

    score = 0.0
    count = 0
    min_dist = max(int(fps * 0.2), 5)

    for sig in [la, ra]:
        # Second derivative (acceleration)
        acc = np.gradient(np.gradient(sig))
        acc = gaussian_filter1d(acc, max(1.0, sigma * 0.7))

        max_peaks, _ = find_peaks(sig, distance=min_dist)
        min_peaks, _ = find_peaks(-sig, distance=min_dist)

        if len(max_peaks) > 0 and len(min_peaks) > 0:
            acc_at_max = np.mean([acc[p] for p in max_peaks])
            acc_at_min = np.mean([acc[p] for p in min_peaks])
            # In the correct direction, |acc at maxima (HS)| > |acc at minima (TO)|
            score += abs(acc_at_max) - abs(acc_at_min)
            count += 1

    return score / max(count, 1)


def detect_walking_direction(angle_frames, ap_axis=None, fps=None):
    """Detect the walking direction sign along the AP axis.

    Uses a two-tier strategy:

    1. **Acceleration asymmetry** (v3.6) -- Compare the sharpness of
       ankle AP position maxima versus minima.  At heel strike, the
       foot decelerates abruptly (sharp peak), while at toe-off the
       foot accelerates more gradually (broad trough).  The direction
       that produces sharper maxima than minima is correct.  This
       method is robust on treadmill data where net displacement is
       near zero.

    2. **Net displacement fallback** -- If the acceleration method is
       inconclusive (score ratio < 1.5), fall back to the original net
       ankle displacement method.

    Parameters
    ----------
    angle_frames : list
        Sequence of AngleFrame objects with ``landmark_positions``.
    ap_axis : int or None
        Index of the anteroposterior axis.  If *None*, it is auto-detected
        via :func:`detect_axes`.
    fps : float or None
        Sampling rate in Hz.  If *None*, defaults to 100.

    Returns
    -------
    direction_sign : int
        +1 if the subject walks in the positive AP direction (or if the
        direction cannot be determined), -1 if in the negative AP direction.
    """
    if ap_axis is None:
        ap_axis, _ = detect_axes(angle_frames)

    if fps is None:
        fps = 100.0

    # --- Strategy 1: Net displacement (robust for overground walking) ---
    # For overground walking, the net displacement is unambiguous.
    # Only fall back to acceleration asymmetry for treadmill-like data.
    positions = []
    for af in angle_frames:
        if af.landmark_positions and "left_ankle" in af.landmark_positions:
            pos = af.landmark_positions["left_ankle"]
            if pos != (0.0, 0.0, 0.0):
                positions.append(pos[ap_axis])

    if len(positions) >= 10:
        positions = np.array(positions)
        net_displacement = positions[-1] - positions[0]
        # If net displacement is substantial (> 500mm = clearly overground),
        # use it directly — it is always correct for overground walking.
        step_length_estimate = abs(net_displacement)
        if step_length_estimate > 500:
            return -1 if net_displacement < 0 else 1

    # --- Strategy 2: Acceleration asymmetry (for treadmill/short trials) ---
    score_pos = _direction_score_acceleration(angle_frames, ap_axis, +1, fps)
    score_neg = _direction_score_acceleration(angle_frames, ap_axis, -1, fps)
    if score_pos > 0 and score_neg <= 0:
        return 1
    if score_neg > 0 and score_pos <= 0:
        return -1

    return 1  # default: positive direction
