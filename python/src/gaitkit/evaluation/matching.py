"""
Event matching between detected and ground-truth gait events.

Implements a greedy nearest-neighbour matching strategy with a
frame-tolerance window converted from milliseconds.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class EventMatch:
    """Result of matching a single detected event to ground truth.

    Attributes
    ----------
    detected_frame : int
        Frame index of the detected event.
    gt_frame : int or None
        Frame index of the matched ground-truth event, or None (FP).
    error_frames : float or None
        Signed error in frames (detected - gt), or None if unmatched.
    """
    detected_frame: int
    gt_frame: Optional[int]
    error_frames: Optional[float]


def match_events(detected: List[int], ground_truth: List[int],
                 tolerance_frames: int) -> Tuple[List[EventMatch], List[int]]:
    """Match detected events to ground-truth events within a tolerance.

    Uses a greedy strategy: for each detected event (in temporal order),
    the closest unmatched ground-truth event within *tolerance_frames* is
    selected.

    Parameters
    ----------
    detected : list of int
        Frame indices of detected events, need not be sorted.
    ground_truth : list of int
        Frame indices of ground-truth events, need not be sorted.
    tolerance_frames : int
        Maximum allowed distance (in frames) for a valid match.

    Returns
    -------
    matches : list of EventMatch
        One entry per detected event (matched or false positive).
    unmatched_gt : list of int
        Ground-truth frames that were not matched (false negatives).
    """
    det_sorted = sorted(detected)
    gt_sorted = sorted(ground_truth)
    gt_used = [False] * len(gt_sorted)

    matches = []
    for d in det_sorted:
        best_j = -1
        best_dist = float("inf")
        for j, g in enumerate(gt_sorted):
            if gt_used[j]:
                continue
            dist = abs(d - g)
            if dist <= tolerance_frames and dist < best_dist:
                best_dist = dist
                best_j = j
        if best_j >= 0:
            gt_used[best_j] = True
            matches.append(EventMatch(
                detected_frame=d,
                gt_frame=gt_sorted[best_j],
                error_frames=float(d - gt_sorted[best_j]),
            ))
        else:
            matches.append(EventMatch(detected_frame=d, gt_frame=None, error_frames=None))

    unmatched_gt = [gt_sorted[j] for j in range(len(gt_sorted)) if not gt_used[j]]
    return matches, unmatched_gt
