"""I/O utilities: load C3D files and bundled example datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

_DATA_DIR = Path(__file__).parent / "data"

# ── Bundled examples ─────────────────────────────────────────────────

_EXAMPLE_MAP = {
    "healthy":    "example_healthy.json",
    "fukuchi":    "example_healthy.json",
    "parkinson":  "example_parkinson.json",
    "pd":         "example_parkinson.json",
    "parkinsons": "example_parkinson.json",
    "kuopio":     "example_kuopio.json",
    "stroke":     "example_stroke.json",
    "avc":        "example_stroke.json",
}


def load_example(name: str = "healthy") -> dict:
    """Load a bundled example gait trial.

    Parameters
    ----------
    name : str
        Example name.  Available: "healthy", "parkinson", "kuopio", "stroke".
        Aliases: "fukuchi" -> "healthy", "pd" -> "parkinson", "avc" -> "stroke".

    Returns
    -------
    dict
        Keys: *angle_frames* (list of dicts), *fps* (float),
        *n_frames* (int), *description*, *source*, *doi*,
        *population*, and optionally *ground_truth*.

    Examples
    --------
    >>> import gaitkit
    >>> trial = gaitkit.load_example("healthy")
    >>> len(trial["angle_frames"])
    374
    """
    key = name.lower().strip()
    if key not in _EXAMPLE_MAP:
        avail = sorted(set(_EXAMPLE_MAP.keys()) - {"fukuchi", "pd", "parkinsons", "avc"})
        raise ValueError(f"Unknown example {name!r}. Available: {avail}")
    path = _DATA_DIR / _EXAMPLE_MAP[key]
    if not path.exists():
        raise FileNotFoundError(f"Example data not found at {path}")
    with open(path) as f:
        return json.load(f)


def list_examples() -> list:
    """List available example names.

    Returns
    -------
    list of str
        Primary names (without aliases).
    """
    return ["healthy", "parkinson", "kuopio", "stroke"]


# ── C3D loading ──────────────────────────────────────────────────────

def load_c3d(path: str, marker_set: str = "auto") -> dict:
    """Load a C3D file and extract angle frames.

    Parameters
    ----------
    path : str or Path
        Path to a .c3d file.
    marker_set : str
        Marker naming convention: "pig" (Plug-in Gait), "isb", or "auto".

    Returns
    -------
    dict
        Same format as :func:`load_example`: keys *angle_frames*, *fps*, *n_frames*.

    Raises
    ------
    ImportError
        If ezc3d is not installed.  Install with pip install gaitkit[c3d]``.
    """
    try:
        import ezc3d
    except ImportError:
        raise ImportError(
            "ezc3d is required to read C3D files. "
            "Install with: pip install gaitkit[c3d]"
        )

    c3d = ezc3d.c3d(str(path))
    fps = float(c3d["header"]["points"]["frame_rate"])
    points = c3d["data"]["points"]           # (4, n_markers, n_frames)
    labels = c3d["parameters"]["POINT"]["LABELS"]["value"]
    labels = [l.strip() for l in labels]
    n_frames = points.shape[2]

    # Build label -> index mapping
    idx = {l.upper(): i for i, l in enumerate(labels)}

    # Detect marker set
    if marker_set == "auto":
        if "LHEE" in idx or "RHEE" in idx:
            marker_set = "pig"
        elif "LEFTHEEL" in idx:
            marker_set = "isb"
        else:
            marker_set = "pig"

    # Marker name mapping
    if marker_set == "pig":
        _map = {
            "left_heel": "LHEE", "right_heel": "RHEE",
            "left_toe": "LTOE", "right_toe": "RTOE",
            "left_ankle": "LANK", "right_ankle": "RANK",
            "left_knee": "LKNE", "right_knee": "RKNE",
            "left_hip": "LASI", "right_hip": "RASI",
            "left_shoulder": "LSHO", "right_shoulder": "RSHO",
            "sacrum": "SACR",
        }
    elif marker_set == "isb":
        _map = {
            "left_heel": "LEFTHEEL",
            "right_heel": "RIGHTHEEL",
            "left_toe": "LEFTTOE",
            "right_toe": "RIGHTTOE",
            "left_ankle": "LEFTANKLE",
            "right_ankle": "RIGHTANKLE",
            "left_knee": "LEFTKNEE",
            "right_knee": "RIGHTKNEE",
            "left_hip": "LEFTHIP",
            "right_hip": "RIGHTHIP",
        }
    else:
        _map = {}

    # Extract landmarks per frame
    angle_frames = []
    for fi in range(n_frames):
        lp = {}
        for name, label in _map.items():
            li = idx.get(label.upper())
            if li is not None:
                lp[name] = (
                    float(points[0, li, fi]),
                    float(points[1, li, fi]),
                    float(points[2, li, fi]),
                )
        angle_frames.append({
            "frame_index": fi,
            "landmark_positions": lp if lp else None,
            "left_hip_angle": 0.0,
            "right_hip_angle": 0.0,
            "left_knee_angle": 0.0,
            "right_knee_angle": 0.0,
            "left_ankle_angle": 0.0,
            "right_ankle_angle": 0.0,
        })

    # Try to extract angles from MODEL outputs
    try:
        analogs = c3d["parameters"]["POINT"]["LABELS"]["value"]
        # Check for computed angles (PiG model output)
        for angle_key, angle_label in [
            ("left_knee_angle", "LKneeAngles"),
            ("right_knee_angle", "RKneeAngles"),
            ("left_hip_angle", "LHipAngles"),
            ("right_hip_angle", "RHipAngles"),
            ("left_ankle_angle", "LAnkleAngles"),
            ("right_ankle_angle", "RAnkleAngles"),
        ]:
            li = idx.get(angle_label.upper())
            if li is not None:
                for fi in range(n_frames):
                    angle_frames[fi][angle_key] = float(points[0, li, fi])
    except Exception:
        pass

    return {
        "angle_frames": angle_frames,
        "fps": fps,
        "n_frames": n_frames,
        "source_file": str(path),
    }
