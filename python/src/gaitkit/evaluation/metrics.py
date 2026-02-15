"""
Detection metrics for gait event evaluation.

Computes precision, recall, F1-score, and temporal accuracy (MAE)
for heel-strike and toe-off detection, given a tolerance window.

Supports zone-aware evaluation for force-plate datasets where ground
truth only covers a portion of the trial (e.g. overground walkways).
When valid_frame_range is provided, both detected and ground-truth
events are filtered to that range before scoring, preventing false
positives from being counted outside the instrumented zone.
"""

import numpy as np
from typing import List, Optional, Tuple

from .matching import match_events


def _filter_to_range(events, valid_range):
    """Keep only events within [start, end] (inclusive)."""
    start, end = valid_range
    return [e for e in events if start <= e <= end]


def compute_event_metrics(detected, ground_truth, tolerance_ms, fps,
                          valid_frame_range=None):
    """Compute detection performance metrics.

    Parameters
    ----------
    detected : list of int
        Frame indices of detected events.
    ground_truth : list of int
        Frame indices of ground-truth events.
    tolerance_ms : float
        Matching tolerance in milliseconds.
    fps : float
        Sampling rate used to convert ms to frames.
    valid_frame_range : tuple of (int, int) or None
        If provided, only events within [start_frame, end_frame] are
        scored.  This handles force-plate datasets where GT only covers
        a portion of the trial.  Events outside the range are silently
        excluded from both detected and ground_truth lists *before*
        matching.  Pass None (the default) to score the full trial.

    Returns
    -------
    dict
        Keys: precision, recall, f1, mae_frames, mae_ms, n_tp, n_fp, n_fn.
        When valid_frame_range is not None the dict also includes
        n_detected_in_zone and n_detected_total for diagnostics.
    """
    if detected is None or ground_truth is None:
        raise ValueError("detected and ground_truth must be sequences of frame indices")
    if fps is None or fps <= 0:
        raise ValueError("fps must be strictly positive")
    if tolerance_ms is None or tolerance_ms < 0:
        raise ValueError("tolerance_ms must be >= 0")
    if valid_frame_range is not None:
        if not isinstance(valid_frame_range, tuple) or len(valid_frame_range) != 2:
            raise ValueError("valid_frame_range must be a (start, end) tuple")
        start, end = valid_frame_range
        if start > end:
            raise ValueError("valid_frame_range start must be <= end")

    tolerance_frames = int(tolerance_ms * fps / 1000)

    # --- Zone-aware filtering ------------------------------------------------
    n_detected_total = len(detected)
    if valid_frame_range is not None:
        detected = _filter_to_range(detected, valid_frame_range)
        ground_truth = _filter_to_range(ground_truth, valid_frame_range)
    n_detected_in_zone = len(detected)

    # --- Edge cases ----------------------------------------------------------
    if not ground_truth:
        if not detected:
            result = dict(precision=1.0, recall=1.0, f1=1.0,
                          mae_frames=0.0, mae_ms=0.0,
                          n_tp=0, n_fp=0, n_fn=0)
        else:
            result = dict(precision=0.0, recall=0.0, f1=0.0,
                          mae_frames=float("inf"), mae_ms=float("inf"),
                          n_tp=0, n_fp=len(detected), n_fn=0)
        if valid_frame_range is not None:
            result["n_detected_in_zone"] = n_detected_in_zone
            result["n_detected_total"] = n_detected_total
        return result

    if not detected:
        result = dict(precision=0.0, recall=0.0, f1=0.0,
                      mae_frames=float("inf"), mae_ms=float("inf"),
                      n_tp=0, n_fp=0, n_fn=len(ground_truth))
        if valid_frame_range is not None:
            result["n_detected_in_zone"] = n_detected_in_zone
            result["n_detected_total"] = n_detected_total
        return result

    # --- Matching ------------------------------------------------------------
    matches, unmatched_gt = match_events(detected, ground_truth,
                                         tolerance_frames)

    tp = sum(1 for m in matches if m.gt_frame is not None)
    fp = sum(1 for m in matches if m.gt_frame is None)
    fn = len(unmatched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    errors = [abs(m.error_frames) for m in matches
              if m.error_frames is not None]
    mae_frames = float(np.mean(errors)) if errors else float("inf")
    mae_ms = (mae_frames / fps * 1000
              if mae_frames != float("inf") else float("inf"))

    result = dict(precision=precision, recall=recall, f1=f1,
                  mae_frames=mae_frames, mae_ms=mae_ms,
                  n_tp=tp, n_fp=fp, n_fn=fn)

    if valid_frame_range is not None:
        result["n_detected_in_zone"] = n_detected_in_zone
        result["n_detected_total"] = n_detected_total

    return result


def compute_cadence_error(detected_hs, gt_cadence, fps):
    """Compute absolute cadence error (steps/min).

    Parameters
    ----------
    detected_hs : list of int
        Detected heel-strike frame indices.
    gt_cadence : float
        Ground-truth cadence in steps/min.
    fps : float
        Sampling rate.

    Returns
    -------
    float
        Absolute cadence error, or -1 if not computable.
    """
    if fps is None or fps <= 0:
        raise ValueError("fps must be strictly positive")
    if len(detected_hs) < 2 or gt_cadence <= 0:
        return -1.0
    intervals = np.diff(sorted(detected_hs)) / fps
    detected_cadence = (60.0 / np.mean(intervals)
                        if np.mean(intervals) > 0 else 0.0)
    return abs(detected_cadence - gt_cadence)
