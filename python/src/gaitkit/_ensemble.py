# -*- coding: utf-8 -*-
"""
Ensemble voting module for gait event detection.

Pools detections from multiple methods and applies temporal clustering
with configurable voting thresholds to produce consensus events with
calibrated confidence scores.

Algorithm
---------
1. Run each requested detector on the input data.
2. Pool all detected events (HS and TO, left and right, separately).
3. Cluster events by temporal proximity (greedy, tolerance-based).
4. Keep clusters where the voter count >= min_votes.
5. Assign confidence = n_voters / n_methods (optionally weighted).

Author: Frederic Fer (f.fer@institut-myologie.org)
Affiliation: Myodata, Institut de Myologie, Paris, France
License: MIT
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .detectors import DETECTOR_REGISTRY, get_detector
from ._core import _normalize_input

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Benchmark-calibrated F1 weights (overall HS + TO macro average)
# ---------------------------------------------------------------------------
BENCHMARK_WEIGHTS: Dict[str, float] = {
    "bayesian_bis": 0.80,
    "intellevent": 0.77,
    "zeni": 0.64,
    "vancanneyt": 0.41,
    "dgei": 0.36,
    "ghoussayni": 0.36,
    "oconnor": 0.32,
    "hreljac": 0.28,
    "mickelborough": 0.16,
}

# Training-free methods (no learned parameters, always available)
DEFAULT_METHODS: List[str] = [
    "bayesian_bis",
    "zeni",
    "oconnor",
    "hreljac",
    "mickelborough",
    "ghoussayni",
    "vancanneyt",
    "dgei",
]

_METHOD_ALIASES = {
    "bike": "bayesian_bis",
    "bayesian": "bayesian_bis",
    "bis": "bayesian_bis",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EnsembleEvent:
    """A single consensus gait event produced by ensemble voting.

    Attributes
    ----------
    frame : int
        Consensus frame index (median of voter frames).
    time : float
        Consensus timestamp in seconds.
    event_type : str
        ``'heel_strike'`` or ``'toe_off'``.
    side : str
        ``'left'`` or ``'right'``.
    confidence : float
        Fraction of methods that voted for this event, in [0, 1].
        When weights are used, this is the sum of voter weights
        divided by the sum of all method weights.
    voters : List[str]
        Names of the methods that contributed to this cluster.
    voter_frames : Dict[str, int]
        Mapping from method name to its detected frame index.
    """
    frame: int
    time: float
    event_type: str
    side: str
    confidence: float
    voters: List[str] = field(default_factory=list)
    voter_frames: Dict[str, int] = field(default_factory=dict)


@dataclass
class GaitResult:
    """Container for gait detection results.

    Works as a unified result object for both single-detector and
    ensemble outputs.  Provides convenience methods for accessing
    events by type/side, extracting gait cycles, and plotting.

    Attributes
    ----------
    method : str
        Name of the detector or ``'ensemble'``.
    heel_strikes : list
        Detected heel-strike events (EnsembleEvent or GaitEvent).
    toe_offs : list
        Detected toe-off events (EnsembleEvent or GaitEvent).
    cycles : list
        Gait cycles built from heel strikes and toe offs.
    fps : float
        Sampling rate.
    metadata : dict
        Extra information (voter details, weights used, etc.).
    _angle_frames : list or None
        Raw angle frame data, stored for plotting if available.
    """
    method: str
    heel_strikes: list = field(default_factory=list)
    toe_offs: list = field(default_factory=list)
    cycles: list = field(default_factory=list)
    fps: float = 100.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    _angle_frames: Optional[list] = field(default=None, repr=False)

    # -- Convenience accessors -----------------------------------------------

    def hs(self, side: Optional[str] = None) -> list:
        """Return heel-strike events, optionally filtered by side."""
        events = self.heel_strikes
        if side is not None:
            events = [e for e in events if e.side == side]
        return sorted(events, key=lambda e: _event_frame(e))

    def to(self, side: Optional[str] = None) -> list:
        """Return toe-off events, optionally filtered by side."""
        events = self.toe_offs
        if side is not None:
            events = [e for e in events if e.side == side]
        return sorted(events, key=lambda e: _event_frame(e))

    def hs_frames(self, side: Optional[str] = None) -> List[int]:
        """Return heel-strike frame indices."""
        return [_event_frame(e) for e in self.hs(side)]

    def to_frames(self, side: Optional[str] = None) -> List[int]:
        """Return toe-off frame indices."""
        return [_event_frame(e) for e in self.to(side)]

    # -- Compatibility with _core.GaitResult (left_hs/right_hs/left_to/right_to)

    @property
    def left_hs(self) -> list:
        """Left heel-strike events (compatibility with _core.GaitResult)."""
        return self.hs("left")

    @property
    def right_hs(self) -> list:
        """Right heel-strike events (compatibility with _core.GaitResult)."""
        return self.hs("right")

    @property
    def left_to(self) -> list:
        """Left toe-off events (compatibility with _core.GaitResult)."""
        return self.to("left")

    @property
    def right_to(self) -> list:
        """Right toe-off events (compatibility with _core.GaitResult)."""
        return self.to("right")

    @property
    def n_frames(self) -> int:
        """Number of frames in the input data."""
        if self._angle_frames is not None:
            return len(self._angle_frames)
        return 0

    def summary(self) -> Dict[str, Any]:
        """Return a concise summary dictionary."""
        return {
            "method": self.method,
            "n_hs_left": len(self.hs("left")),
            "n_hs_right": len(self.hs("right")),
            "n_to_left": len(self.to("left")),
            "n_to_right": len(self.to("right")),
            "n_cycles": len(self.cycles),
        }

    def plot(self, **kwargs):
        """Delegate to the visualization module."""
        from ._viz import plot_result
        return plot_result(self, **kwargs)

    def __repr__(self) -> str:
        hs_l = len(self.hs("left"))
        hs_r = len(self.hs("right"))
        to_l = len(self.to("left"))
        to_r = len(self.to("right"))
        return (
            f"GaitResult(method={self.method!r}, "
            f"HS={hs_l}L/{hs_r}R, TO={to_l}L/{to_r}R, "
            f"cycles={len(self.cycles)})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _event_frame(event) -> int:
    """Extract frame index from any event type (GaitEvent or EnsembleEvent)."""
    if hasattr(event, "frame"):
        return event.frame
    if hasattr(event, "frame_index"):
        return event.frame_index
    raise AttributeError(
        f"Cannot extract frame from {type(event).__name__}: "
        f"expected 'frame' or 'frame_index' attribute"
    )


def _event_time(event) -> float:
    """Extract time from any event type."""
    return event.time


def _normalize_methods(methods: Sequence[str]) -> List[str]:
    """Normalize method aliases and deduplicate while preserving order."""
    out: List[str] = []
    seen = set()
    for method in methods:
        if not isinstance(method, str):
            raise ValueError("methods must contain only string method names")
        raw = method.lower().strip()
        if not raw:
            raise ValueError("methods cannot contain empty entries")
        key = _METHOD_ALIASES.get(raw, raw)
        if key in seen:
            continue
        out.append(key)
        seen.add(key)
    return out


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute the weighted median of a set of values.

    Parameters
    ----------
    values : ndarray
        Array of values.
    weights : ndarray
        Corresponding non-negative weights.

    Returns
    -------
    float
        Weighted median.
    """
    if len(values) == 0:
        return 0.0
    if len(values) == 1:
        return float(values[0])

    sorted_idx = np.argsort(values)
    sorted_vals = values[sorted_idx]
    sorted_wts = weights[sorted_idx]

    cumulative = np.cumsum(sorted_wts)
    half_total = cumulative[-1] / 2.0

    # Find the first index where cumulative weight >= half total
    median_idx = np.searchsorted(cumulative, half_total)
    median_idx = min(median_idx, len(sorted_vals) - 1)

    return float(sorted_vals[median_idx])


# ---------------------------------------------------------------------------
# Clustering engine
# ---------------------------------------------------------------------------

def _cluster_events(
    events_by_method: Dict[str, List[int]],
    tolerance_frames: int,
    min_votes: int,
    weights: Optional[Dict[str, float]] = None,
    fps: float = 100.0,
) -> List[EnsembleEvent]:
    """Cluster candidate events from multiple detectors.

    Algorithm
    ---------
    1. Pool all (method, frame) pairs, sorted by frame.
    2. Greedy clustering: if the next event is within *tolerance_frames*
       of the current cluster center, absorb it; otherwise start a new
       cluster.
    3. For each cluster with >= *min_votes* distinct voters, compute the
       consensus frame (weighted median) and confidence.

    Parameters
    ----------
    events_by_method : dict
        ``{method_name: [frame1, frame2, ...], ...}``
    tolerance_frames : int
        Maximum distance (in frames) between events in the same cluster.
    min_votes : int
        Minimum number of distinct methods that must agree.
    weights : dict or None
        ``{method_name: weight}`` for weighted median and confidence.
        If None, all methods have equal weight.
    fps : float
        Sampling rate, used to compute timestamps.

    Returns
    -------
    list of EnsembleEvent
    """
    # Build sorted list of (frame, method_name)
    all_candidates: List[Tuple[int, str]] = []
    for method, frames in events_by_method.items():
        for f in frames:
            all_candidates.append((int(f), method))

    if not all_candidates:
        return []

    all_candidates.sort(key=lambda x: x[0])

    # Greedy clustering
    clusters: List[List[Tuple[int, str]]] = []
    current_cluster: List[Tuple[int, str]] = [all_candidates[0]]
    current_center = float(all_candidates[0][0])

    for frame, method in all_candidates[1:]:
        if abs(frame - current_center) <= tolerance_frames:
            current_cluster.append((frame, method))
            # Update center as running mean of frames in cluster
            current_center = np.mean([f for f, _ in current_cluster])
        else:
            clusters.append(current_cluster)
            current_cluster = [(frame, method)]
            current_center = float(frame)

    clusters.append(current_cluster)

    # Resolve clusters
    all_methods = list(events_by_method.keys())

    # Default equal weights
    if weights is None:
        method_weights = {m: 1.0 for m in all_methods}
    else:
        method_weights = {m: weights.get(m, 1.0) for m in all_methods}

    total_weight = sum(method_weights[m] for m in all_methods)

    results: List[EnsembleEvent] = []

    for cluster in clusters:
        # Deduplicate: keep only one frame per method (closest to center)
        center_approx = np.mean([f for f, _ in cluster])
        method_frames: Dict[str, int] = {}
        for frame, method in cluster:
            if method not in method_frames:
                method_frames[method] = frame
            else:
                # Keep the frame closest to the cluster center
                if abs(frame - center_approx) < abs(method_frames[method] - center_approx):
                    method_frames[method] = frame

        voters = sorted(method_frames.keys())
        n_voters = len(voters)

        if n_voters < min_votes:
            continue

        # Compute consensus frame (weighted median)
        frames_arr = np.array([method_frames[m] for m in voters], dtype=np.float64)
        weights_arr = np.array([method_weights[m] for m in voters], dtype=np.float64)

        consensus_frame = int(round(_weighted_median(frames_arr, weights_arr)))

        # Confidence: weighted voter fraction
        voter_weight = sum(method_weights[m] for m in voters)
        confidence = voter_weight / total_weight if total_weight > 0 else 0.0

        results.append(EnsembleEvent(
            frame=consensus_frame,
            time=consensus_frame / fps,
            event_type="",  # filled by caller
            side="",        # filled by caller
            confidence=round(confidence, 4),
            voters=voters,
            voter_frames=dict(method_frames),
        ))

    return results


# ---------------------------------------------------------------------------
# Main ensemble API
# ---------------------------------------------------------------------------

def detect_ensemble(
    data=None,
    *,
    methods: Optional[Sequence[str]] = None,
    fps: Optional[float] = None,
    min_votes: int = 2,
    tolerance_ms: float = 50.0,
    weights=None,
    **kwargs,
) -> GaitResult:
    """Detect gait events by ensemble voting across multiple detectors.

    Parameters
    ----------
    data : list of AngleFrame
        Motion capture data (sequence of AngleFrame objects).
    methods : list of str or None
        Detector names to include.  If None, all training-free methods
        are used (see DEFAULT_METHODS).
    fps : float or None
        Sampling rate in Hz.  If None, defaults to 100.0.
    min_votes : int
        Minimum number of detectors that must agree for an event to be
        confirmed.  Default: 2.
    tolerance_ms : float
        Maximum temporal distance (in milliseconds) between detections
        to consider them as the same event.  Default: 50 ms.
    weights : dict, str, or None
        Per-method weights for the consensus computation.
        - None: all methods weighted equally.
        - ``"benchmark"``: use benchmark-calibrated F1 weights.
        - dict: ``{method_name: weight}`` custom weights.
    **kwargs
        Additional keyword arguments passed to each detector constructor.

    Returns
    -------
    GaitResult
        Ensemble result with method="ensemble", containing EnsembleEvent
        objects that carry confidence scores and voter information.

    Examples
    --------
    >>> result = detect_ensemble(angle_frames, fps=100.0, min_votes=3)
    >>> print(result)
    GaitResult(method='ensemble', HS=12L/11R, TO=11L/11R, cycles=20)
    >>> for hs in result.hs("left"):
    ...     print(f"frame={hs.frame}, confidence={hs.confidence:.2f}, "
    ...           f"voters={hs.voters}")
    """
    if data is None:
        raise ValueError("data (angle_frames) is required")
    if fps is None:
        fps = 100.0
    if fps <= 0:
        raise ValueError("fps must be strictly positive")
    if min_votes < 1:
        raise ValueError("min_votes must be >= 1")
    if tolerance_ms < 0:
        raise ValueError("tolerance_ms must be >= 0")

    # Resolve methods
    if methods is None:
        methods = [m for m in DEFAULT_METHODS if m in DETECTOR_REGISTRY]
    else:
        if isinstance(methods, (str, bytes)):
            raise ValueError("methods must be a sequence of method names, not a single string")
        methods = _normalize_methods(methods)
        for m in methods:
            if m not in DETECTOR_REGISTRY:
                available = ", ".join(sorted(DETECTOR_REGISTRY.keys()))
                raise ValueError(
                    f"Unknown detector '{m}'. Available: {available}"
                )

    if len(methods) < 2:
        raise ValueError(
            f"Ensemble requires at least 2 methods, got {len(methods)}: {methods}"
        )

    # Resolve weights
    weight_dict: Optional[Dict[str, float]] = None
    if weights == "benchmark":
        weight_dict = {m: BENCHMARK_WEIGHTS.get(m, 0.5) for m in methods}
    elif isinstance(weights, dict):
        weight_dict = {m: weights.get(m, 1.0) for m in methods}
        for method_name, value in weight_dict.items():
            if not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                raise ValueError(
                    f"Invalid weight for method '{method_name}': {value!r}. "
                    "Weights must be finite numbers >= 0."
                )
    elif weights is not None:
        raise ValueError(
            "weights must be None, 'benchmark', or a dict of non-negative numeric weights"
        )

    tolerance_frames = max(1, int(round(tolerance_ms * fps / 1000.0)))

    # ---- Normalize input (dict → AngleFrame objects) -----------------------
    # Detectors expect AngleFrame objects with .landmark_positions attribute,
    # not raw dicts.  _normalize_input handles all input formats (C3D path,
    # dict with "angle_frames" key, list of dicts, ExtractionResult).
    data, fps = _normalize_input(data, fps)

    # ---- Run each detector -------------------------------------------------
    # Collect events per (event_type, side, method)
    hs_left_by_method: Dict[str, List[int]] = {}
    hs_right_by_method: Dict[str, List[int]] = {}
    to_left_by_method: Dict[str, List[int]] = {}
    to_right_by_method: Dict[str, List[int]] = {}
    per_method_results: Dict[str, Any] = {}
    methods_failed: Dict[str, str] = {}

    for method_name in methods:
        try:
            detector = get_detector(method_name, fps=fps, **kwargs)
            if hasattr(detector, "detect"):
                hs_events, to_events, cycles = detector.detect(data)
            else:
                hs_events, to_events, cycles = detector.detect_gait_events(data)
            per_method_results[method_name] = (hs_events, to_events, cycles)

            # Partition by side
            hs_left_by_method[method_name] = [
                _event_frame(e) for e in hs_events if e.side == "left"
            ]
            hs_right_by_method[method_name] = [
                _event_frame(e) for e in hs_events if e.side == "right"
            ]
            to_left_by_method[method_name] = [
                _event_frame(e) for e in to_events if e.side == "left"
            ]
            to_right_by_method[method_name] = [
                _event_frame(e) for e in to_events if e.side == "right"
            ]

            logger.debug(
                "%-16s  HS: %dL %dR  TO: %dL %dR",
                method_name,
                len(hs_left_by_method[method_name]),
                len(hs_right_by_method[method_name]),
                len(to_left_by_method[method_name]),
                len(to_right_by_method[method_name]),
            )
        except Exception as exc:
            # Ensemble mode is designed to be fault-tolerant: one detector
            # failing should not abort the full multi-method vote.
            logger.warning("Detector '%s' failed: %s", method_name, exc)
            methods_failed[method_name] = str(exc)
            # Remove from pools so it does not affect voter count
            continue

    # Check we still have enough methods after failures
    successful_methods = sorted(
        set(hs_left_by_method) | set(hs_right_by_method)
        | set(to_left_by_method) | set(to_right_by_method)
    )
    if len(successful_methods) < min_votes:
        logger.warning(
            "Only %d methods succeeded (need min_votes=%d). "
            "Returning empty result.",
            len(successful_methods), min_votes,
        )
        return GaitResult(
            method="ensemble",
            fps=fps,
            metadata={
                "methods_requested": list(methods),
                "methods_succeeded": successful_methods,
                "methods_failed": methods_failed,
                "min_votes": min_votes,
                "tolerance_ms": tolerance_ms,
            },
        )

    # ---- Cluster each (event_type, side) independently ---------------------
    all_heel_strikes: List[EnsembleEvent] = []
    all_toe_offs: List[EnsembleEvent] = []

    for event_type, side, pool in [
        ("heel_strike", "left", hs_left_by_method),
        ("heel_strike", "right", hs_right_by_method),
        ("toe_off", "left", to_left_by_method),
        ("toe_off", "right", to_right_by_method),
    ]:
        # For clustering, include methods with empty lists so voter
        # count reflects "could have voted but did not"
        full_pool = {m: pool.get(m, []) for m in successful_methods}

        consensus_events = _cluster_events(
            events_by_method=full_pool,
            tolerance_frames=tolerance_frames,
            min_votes=min_votes,
            weights=weight_dict,
            fps=fps,
        )

        # Fill in event_type and side
        for ev in consensus_events:
            ev.event_type = event_type
            ev.side = side

        if event_type == "heel_strike":
            all_heel_strikes.extend(consensus_events)
        else:
            all_toe_offs.extend(consensus_events)

    # Sort by frame
    all_heel_strikes.sort(key=lambda e: e.frame)
    all_toe_offs.sort(key=lambda e: e.frame)

    # ---- Build gait cycles from consensus events ---------------------------
    cycles = _build_ensemble_cycles(all_heel_strikes, all_toe_offs, fps)

    # ---- Assemble result ---------------------------------------------------
    result = GaitResult(
        method="ensemble",
        heel_strikes=all_heel_strikes,
        toe_offs=all_toe_offs,
        cycles=cycles,
        fps=fps,
        metadata={
            "methods_requested": list(methods),
            "methods_succeeded": successful_methods,
            "methods_failed": methods_failed,
            "min_votes": min_votes,
            "tolerance_ms": tolerance_ms,
            "tolerance_frames": tolerance_frames,
            "weights": weight_dict,
            "per_method_counts": {
                m: {
                    "hs_left": len(hs_left_by_method.get(m, [])),
                    "hs_right": len(hs_right_by_method.get(m, [])),
                    "to_left": len(to_left_by_method.get(m, [])),
                    "to_right": len(to_right_by_method.get(m, [])),
                }
                for m in successful_methods
            },
        },
        _angle_frames=data,
    )

    logger.info(
        "Ensemble (%d methods, min_votes=%d): %s",
        len(successful_methods), min_votes, result,
    )

    return result


# ---------------------------------------------------------------------------
# Cycle extraction from ensemble events
# ---------------------------------------------------------------------------

def _build_ensemble_cycles(
    heel_strikes: List[EnsembleEvent],
    toe_offs: List[EnsembleEvent],
    fps: float,
) -> list:
    """Build gait cycles from ensemble consensus events.

    A gait cycle spans two consecutive heel strikes on the same side,
    with an optional toe-off in between.

    Returns
    -------
    list of dict
        Each dict contains: cycle_id, side, start_frame, end_frame,
        toe_off_frame, duration, stance_percentage, confidence.
    """
    cycles = []
    cycle_id = 0

    for side in ("left", "right"):
        side_hs = sorted(
            [e for e in heel_strikes if e.side == side],
            key=lambda e: e.frame,
        )
        side_to = sorted(
            [e for e in toe_offs if e.side == side],
            key=lambda e: e.frame,
        )

        for i in range(len(side_hs) - 1):
            start = side_hs[i]
            end = side_hs[i + 1]
            duration = (end.frame - start.frame) / fps

            # Skip implausibly short or long cycles
            if duration < 0.3 or duration > 3.0:
                continue

            # Find toe-off between these two heel strikes
            to_in_cycle = [
                t for t in side_to
                if start.frame < t.frame < end.frame
            ]
            to_event = to_in_cycle[0] if to_in_cycle else None

            stance_pct = None
            if to_event is not None:
                stance_pct = round(
                    (to_event.frame - start.frame)
                    / (end.frame - start.frame) * 100.0,
                    1,
                )

            # Cycle confidence = mean of bounding HS confidences
            cycle_confidence = round((start.confidence + end.confidence) / 2.0, 4)

            cycles.append({
                "cycle_id": cycle_id,
                "side": side,
                "start_frame": start.frame,
                "end_frame": end.frame,
                "toe_off_frame": to_event.frame if to_event else None,
                "start_time": start.time,
                "end_time": end.time,
                "toe_off_time": to_event.time if to_event else None,
                "duration": round(duration, 4),
                "stance_percentage": stance_pct,
                "confidence": cycle_confidence,
            })
            cycle_id += 1

    cycles.sort(key=lambda c: c["start_frame"])
    return cycles


# ---------------------------------------------------------------------------
# Convenience: wrap a single-detector result in GaitResult
# ---------------------------------------------------------------------------

def wrap_result(
    method: str,
    heel_strikes: list,
    toe_offs: list,
    cycles: list,
    fps: float = 100.0,
    angle_frames=None,
) -> GaitResult:
    """Wrap raw detector output into a GaitResult.

    Parameters
    ----------
    method : str
        Detector name.
    heel_strikes, toe_offs, cycles : list
        Output from any detector's ``detect()`` method.
    fps : float
        Sampling rate.
    angle_frames : list or None
        Raw data for plotting.

    Returns
    -------
    GaitResult
    """
    return GaitResult(
        method=method,
        heel_strikes=heel_strikes,
        toe_offs=toe_offs,
        cycles=cycles,
        fps=fps,
        _angle_frames=angle_frames,
    )


__all__ = [
    "detect_ensemble",
    "GaitResult",
    "EnsembleEvent",
    "wrap_result",
    "BENCHMARK_WEIGHTS",
    "DEFAULT_METHODS",
]
