"""Core data structures and detection dispatch for gaitkit."""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Method name mapping ──────────────────────────────────────────────
_METHOD_ALIASES = {
    "bike": "bayesian_bis",
    "bayesian": "bayesian_bis",
    "bis": "bayesian_bis",
}

_ALL_METHODS = [
    "bike", "zeni", "oconnor", "hreljac", "mickelborough",
    "ghoussayni", "vancanneyt", "dgei", "intellevent", "deepevent",
]


def list_methods() -> List[str]:
    """Return names of all available detection methods.

    Returns
    -------
    list of str
        Method identifiers accepted by :func:`detect`.
        "bike" is the default (Bayesian Inference for Kinematic Events).
    """
    return list(_ALL_METHODS)


# ── GaitResult ───────────────────────────────────────────────────────

@dataclass
class GaitResult:
    """Container for gait event detection results.

    Attributes
    ----------
    left_hs : list of dict
        Left heel-strikes.  Each dict has keys *frame* (int) and *time* (float, s).
    right_hs : list of dict
        Right heel-strikes.
    left_to : list of dict
        Left toe-offs.
    right_to : list of dict
        Right toe-offs.
    fps : float
        Sampling frequency (Hz).
    method : str
        Detector name that produced these results.
    n_frames : int
        Total number of frames in the input data.
    """

    left_hs: List[Dict[str, Any]] = field(default_factory=list)
    right_hs: List[Dict[str, Any]] = field(default_factory=list)
    left_to: List[Dict[str, Any]] = field(default_factory=list)
    right_to: List[Dict[str, Any]] = field(default_factory=list)
    fps: float = 100.0
    method: str = ""
    n_frames: int = 0
    _angle_frames: Optional[list] = field(default=None, repr=False)

    # ── convenience properties ────────────────────────────────────

    @property
    def events(self):
        """All events as a pandas DataFrame.

        Columns: time, frame, side, event_type, confidence, method.
        """
        import pandas as pd
        rows = []
        for evlist, side, etype in [
            (self.left_hs, "left", "HS"),
            (self.right_hs, "right", "HS"),
            (self.left_to, "left", "TO"),
            (self.right_to, "right", "TO"),
        ]:
            for ev in evlist:
                rows.append({
                    "time": ev["time"],
                    "frame": ev["frame"],
                    "side": side,
                    "event_type": etype,
                    "confidence": ev.get("confidence", 1.0),
                    "method": self.method,
                })
        df = pd.DataFrame(rows)
        if len(df):
            df = df.sort_values("time").reset_index(drop=True)
        return df

    @property
    def cycles(self):
        """Gait cycles as a pandas DataFrame.

        Each row is one stride (HS → TO → HS). Columns:
        side, hs1_time, to_time, hs2_time, stance_pct, swing_pct, stride_time, cadence.
        """
        import pandas as pd
        rows = []
        for side, hs_list, to_list in [
            ("left", self.left_hs, self.left_to),
            ("right", self.right_hs, self.right_to),
        ]:
            hs_times = sorted(e["time"] for e in hs_list)
            to_times = sorted(e["time"] for e in to_list)
            for i in range(len(hs_times) - 1):
                t0, t2 = hs_times[i], hs_times[i + 1]
                stride = t2 - t0
                if stride <= 0:
                    continue
                # Find the TO between these two HS
                to_mid = [t for t in to_times if t0 < t < t2]
                if to_mid:
                    to_t = to_mid[0]
                    stance = (to_t - t0) / stride * 100
                else:
                    to_t = None
                    stance = None
                rows.append({
                    "side": side,
                    "hs1_time": round(t0, 4),
                    "to_time": round(to_t, 4) if to_t is not None else None,
                    "hs2_time": round(t2, 4),
                    "stance_pct": round(stance, 1) if stance is not None else None,
                    "swing_pct": round(100 - stance, 1) if stance is not None else None,
                    "stride_time": round(stride, 4),
                    "cadence": round(60.0 / stride, 1) if stride > 0 else None,
                })
        return pd.DataFrame(rows)

    # ── export ────────────────────────────────────────────────────

    def to_dataframe(self):
        """Alias for :attr:`events`."""
        return self.events

    def to_csv(self, path: str) -> None:
        """Write events to a CSV file."""
        self.events.to_csv(path, index=False)

    def to_json(self, path: str) -> None:
        """Write events to a JSON file."""
        self.events.to_json(path, orient="records", indent=2)

    # ── visualisation delegates ───────────────────────────────────

    def plot(self, **kwargs):
        """Plot gait angles with event markers.  See :func:`gaitkit._viz.plot_result`."""
        from ._viz import plot_result
        return plot_result(self, **kwargs)

    def plot_cycles(self, **kwargs):
        """Butterfly plot of gait cycles.  See :func:`gaitkit._viz.plot_cycles`."""
        from ._viz import plot_cycles
        return plot_cycles(self, **kwargs)

    # ── summary ───────────────────────────────────────────────────

    def summary(self) -> str:
        """Print a concise summary of detection results."""
        n_hs = len(self.left_hs) + len(self.right_hs)
        n_to = len(self.left_to) + len(self.right_to)
        lines = [
            f"GaitResult  method={self.method}  fps={self.fps:.0f}  frames={self.n_frames}",
            f"  Heel-strikes: {n_hs} (L={len(self.left_hs)}, R={len(self.right_hs)})",
            f"  Toe-offs:     {n_to} (L={len(self.left_to)}, R={len(self.right_to)})",
        ]
        cyc = self.cycles
        if len(cyc):
            st = cyc["stride_time"].mean()
            cad = cyc["cadence"].mean()
            stance = cyc["stance_pct"].dropna().mean()
            lines.append(f"  Stride time:  {st:.3f} s  (cadence {cad:.0f} steps/min)")
            lines.append(f"  Stance phase: {stance:.1f}%")
        text = "\n".join(lines)
        print(text)
        return text

    def __repr__(self) -> str:
        n_hs = len(self.left_hs) + len(self.right_hs)
        n_to = len(self.left_to) + len(self.right_to)
        return f"GaitResult(method={self.method!r}, HS={n_hs}, TO={n_to}, frames={self.n_frames})"


# ── Detector factory ─────────────────────────────────────────────────

def _resolve_method(name: str) -> str:
    """Map user-facing name to internal detector key."""
    low = name.lower().strip()
    return _METHOD_ALIASES.get(low, low)


def _make_detector(method: str, fps: float):
    """Instantiate the requested detector, preferring native C backends."""
    key = _resolve_method(method)

    # Try native (C-accelerated) first
    _native_map = {
        "bayesian_bis": ("gaitkit.bayesian_bis_native", "BayesianBisNativeGaitDetector"),
        "zeni":         ("gaitkit.zeni_native",         "ZeniNativeGaitDetector"),
        "oconnor":      ("gaitkit.oconnor_native",      "OConnorNativeGaitDetector"),
        "hreljac":      ("gaitkit.hreljac_native",      "HreljacNativeGaitDetector"),
        "mickelborough":("gaitkit.mickelborough_native","MickelboroughNativeGaitDetector"),
        "ghoussayni":   ("gaitkit.ghoussayni_native",   "GhoussayniNativeGaitDetector"),
        "dgei":         ("gaitkit.dgei_native",         "DGEINativeGaitDetector"),
        "vancanneyt":   ("gaitkit.vancanneyt_native",   "VancanneytNativeGaitDetector"),
    }

    _python_map = {
        "bayesian_bis": ("gaitkit.detectors.bayesian_bis",   "BayesianBisGaitDetector"),
        "zeni":         ("gaitkit.detectors.zeni_detector",   "ZeniDetector"),
        "oconnor":      ("gaitkit.detectors.oconnor_detector","OConnorDetector"),
        "hreljac":      ("gaitkit.detectors.hreljac_detector","HreljacDetector"),
        "mickelborough":("gaitkit.detectors.mickelborough_detector","MickelboroughDetector"),
        "ghoussayni":   ("gaitkit.detectors.ghoussayni_detector", "GhoussayniDetector"),
        "dgei":         ("gaitkit.detectors.dgei_detector",   "DGEIDetector"),
        "vancanneyt":   ("gaitkit.detectors.vancanneyt_detector","VancanneytDetector"),
        "intellevent":  ("gaitkit.detectors.intellevent_detector","IntellEventDetector"),
        "deepevent":    ("gaitkit.detectors.deepevent_detector",  "DeepEventDetector"),
    }

    # Try native
    if key in _native_map:
        mod_name, cls_name = _native_map[key]
        try:
            import importlib
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, cls_name)
            det = cls(fps=fps)
            return det
        except Exception as exc:
            logger.debug("Native detector '%s' unavailable, falling back to Python backend: %s", key, exc)

    # Fall back to pure Python
    if key in _python_map:
        mod_name, cls_name = _python_map[key]
        import importlib
        mod = importlib.import_module(mod_name)
        cls = getattr(mod, cls_name)
        det = cls(fps=fps)
        return det

    raise ValueError(
        f"Unknown method {method!r}. Available: {list_methods()}"
    )


def _events_to_dicts(events, fps: float) -> List[Dict[str, Any]]:
    """Convert a list of GaitEvent dataclasses to list-of-dicts."""
    result = []
    for ev in events:
        result.append({
            "frame": int(ev.frame_index),
            "time": round(ev.frame_index / fps, 4),
            "side": ev.side,
            "confidence": getattr(ev, "probability", 1.0),
        })
    return result


# ── Main detect function ─────────────────────────────────────────────

def detect(
    data=None,
    *,
    method: str = "bike",
    fps: float = None,
    units: Optional[Dict[str, str]] = None,
) -> GaitResult:
    """Detect gait events (heel-strikes and toe-offs).

    Parameters
    ----------
    data : str, Path, list of dict, or ExtractionResult
        Input data. Can be:
        - Path to a C3D file (requires ezc3d)
        - List of angle-frame dicts (as returned by extractors)
        - An ExtractionResult from a dataset extractor
    method : str, default "bike"
        Detection method.  See :func:`list_methods` for options.
    fps : float, optional
        Sampling frequency in Hz.  Required unless *data* is a C3D file
        or an ExtractionResult (which carry their own fps).
    units : dict, optional
        Unit hints, e.g. {"position": "mm", "angles": "deg"}.

    Returns
    -------
    GaitResult
        Detection results with events, cycles, export and plot helpers.

    Examples
    --------
    >>> import gaitkit
    >>> trial = gaitkit.load_example("healthy")
    >>> result = gaitkit.detect(trial, method="bike")
    >>> result.summary()
    """
    angle_frames, resolved_fps = _normalize_input(data, fps)
    det = _make_detector(method, resolved_fps)
    hs, to, _cycles = det.detect_gait_events(angle_frames)

    # Split by side
    left_hs = _events_to_dicts([e for e in hs if e.side == "left"], resolved_fps)
    right_hs = _events_to_dicts([e for e in hs if e.side == "right"], resolved_fps)
    left_to = _events_to_dicts([e for e in to if e.side == "left"], resolved_fps)
    right_to = _events_to_dicts([e for e in to if e.side == "right"], resolved_fps)

    return GaitResult(
        left_hs=left_hs,
        right_hs=right_hs,
        left_to=left_to,
        right_to=right_to,
        fps=resolved_fps,
        method=_resolve_method(method),
        n_frames=len(angle_frames),
        _angle_frames=angle_frames,
    )


def _normalize_input(data, fps):
    """Convert various input formats to (angle_frames, fps)."""
    if fps is not None and fps <= 0:
        raise ValueError("fps must be strictly positive")

    # Case 1: string/Path → C3D file
    if isinstance(data, (str, Path)):
        p = Path(data)
        if p.suffix.lower() == ".c3d":
            from ._io import load_c3d
            trial = load_c3d(str(p))
            if not trial["angle_frames"]:
                raise ValueError("No angle frames were found in the C3D file")
            if trial["fps"] <= 0:
                raise ValueError("Invalid sampling frequency found in the C3D file")
            return trial["angle_frames"], trial["fps"]
        raise ValueError(f"Unsupported file format: {p.suffix}")

    # Case 2: ExtractionResult (has .angle_frames and .fps)
    if hasattr(data, "angle_frames") and hasattr(data, "fps"):
        if not data.angle_frames:
            raise ValueError("Input contains no angle frames")
        if data.fps <= 0:
            raise ValueError("Input sampling frequency must be strictly positive")
        return data.angle_frames, data.fps

    # Case 3: dict with 'angle_frames' key (from load_example)
    if isinstance(data, dict) and "angle_frames" in data:
        resolved_fps = fps or data.get("fps", 100.0)
        if resolved_fps <= 0:
            raise ValueError("Input sampling frequency must be strictly positive")
        af = data["angle_frames"]
        if not af:
            raise ValueError("Input contains no angle frames")
        # If raw dicts, convert to AngleFrame-like objects
        if af and isinstance(af[0], dict):
            af = _dicts_to_angle_frames(af)
        return af, resolved_fps

    # Case 4: list of angle frames directly
    if isinstance(data, (list, tuple)):
        if fps is None:
            raise ValueError("fps is required when data is a list of frames")
        if fps <= 0:
            raise ValueError("fps must be strictly positive")
        af = data
        if not af:
            raise ValueError("Input contains no angle frames")
        if af and isinstance(af[0], dict):
            af = _dicts_to_angle_frames(af)
        return af, fps

    raise TypeError(
        f"Cannot interpret data of type {type(data).__name__}. "
        "Pass a C3D path, ExtractionResult, dict, or list of frames."
    )


def _dicts_to_angle_frames(dicts):
    """Convert list of plain dicts to AngleFrame-like objects."""
    from dataclasses import dataclass, field as dc_field
    from typing import Optional as Opt, Dict

    @dataclass
    class _Frame:
        frame_index: int = 0
        left_hip_angle: float = 0.0
        right_hip_angle: float = 0.0
        left_knee_angle: float = 0.0
        right_knee_angle: float = 0.0
        left_ankle_angle: float = 0.0
        right_ankle_angle: float = 0.0
        pelvis_tilt: float = 0.0
        trunk_angle: float = 0.0
        landmark_positions: Opt[Dict] = None
        is_valid: bool = True

    frames = []
    for i, d in enumerate(dicts):
        f = _Frame(frame_index=d.get("frame_index", i))
        for attr in ("left_hip_angle", "right_hip_angle",
                     "left_knee_angle", "right_knee_angle",
                     "left_ankle_angle", "right_ankle_angle",
                     "pelvis_tilt", "trunk_angle"):
            if attr in d:
                setattr(f, attr, float(d[attr]))
        if "landmark_positions" in d and d["landmark_positions"]:
            lp = {}
            for name, coords in d["landmark_positions"].items():
                lp[name] = tuple(float(c) for c in coords)
            f.landmark_positions = lp
        frames.append(f)
    return frames
