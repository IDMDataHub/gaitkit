"""I/O utilities: load C3D files and bundled example datasets."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

_DATA_DIR = Path(__file__).parent / "data"
logger = logging.getLogger(__name__)

Point3D = Tuple[float, float, float]
MarkerSpec = Union[str, Sequence[str]]
AnglesLike = Union[str, Path, Mapping[str, Sequence[float]]]


_MARKER_MAPS: Dict[str, Dict[str, List[MarkerSpec]]] = {
    # Plug-in Gait (+ common aliases)
    "pig": {
        "left_heel": ["LHEE", "LHEEL", "LHEE1", "LCAL", "LEFTHEEL"],
        "right_heel": ["RHEE", "RHEEL", "RHEE1", "RCAL", "RIGHTHEEL"],
        "left_toe": ["LTOE", "LTOE1", "LFMH1", "LTT2", "LEFTTOE"],
        "right_toe": ["RTOE", "RTOE1", "RFMH1", "RTT2", "RIGHTTOE"],
        "left_ankle": ["LANK", "LEFTANKLE", ("LMM", "LLM"), "LMM", "LLM"],
        "right_ankle": ["RANK", "RIGHTANKLE", ("RMM", "RLM"), "RMM", "RLM"],
        "left_knee": ["LKNE", "LEFTKNEE", ("LLFE", "LMFE"), "LLFE", "LMFE"],
        "right_knee": ["RKNE", "RIGHTKNEE", ("RLFE", "RMFE"), "RLFE", "RMFE"],
        "left_hip": ["LASI", "LASIS", "LEFTHIP"],
        "right_hip": ["RASI", "RASIS", "RIGHTHIP"],
        "sacrum": ["SACR", ("LPSIS", "RPSIS"), "LPSIS", "RPSIS"],
    },
    # ISB-style labels
    "isb": {
        "left_heel": ["LEFTHEEL"],
        "right_heel": ["RIGHTHEEL"],
        "left_toe": ["LEFTTOE"],
        "right_toe": ["RIGHTTOE"],
        "left_ankle": ["LEFTANKLE"],
        "right_ankle": ["RIGHTANKLE"],
        "left_knee": ["LEFTKNEE"],
        "right_knee": ["RIGHTKNEE"],
        "left_hip": ["LEFTHIP"],
        "right_hip": ["RIGHTHIP"],
    },
    # Institut de Myologie / trial_07 naming convention
    "imy": {
        "left_heel": ["LCAL"],
        "right_heel": ["RCAL"],
        "left_toe": ["LTT2", "LFMH1", "LFMH5"],
        "right_toe": ["RTT2", "RFMH1", "RFMH5"],
        "left_ankle": [("LMM", "LLM"), "LMM", "LLM"],
        "right_ankle": [("RMM", "RLM"), "RMM", "RLM"],
        "left_knee": [("LLFE", "LMFE"), "LLFE", "LMFE"],
        "right_knee": [("RLFE", "RMFE"), "RLFE", "RMFE"],
        "left_hip": ["LHJC", "LASIS"],
        "right_hip": ["RHJC", "RASIS"],
        "sacrum": [("LPSIS", "RPSIS"), "LPSIS", "RPSIS"],
    },
}


def _get_point(idx: Mapping[str, int], points, frame_index: int, label: str) -> Optional[Point3D]:
    li = idx.get(label.upper())
    if li is None:
        return None
    return (
        float(points[0, li, frame_index]),
        float(points[1, li, frame_index]),
        float(points[2, li, frame_index]),
    )


def _mean_points(a: Optional[Point3D], b: Optional[Point3D]) -> Optional[Point3D]:
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, (a[2] + b[2]) / 2.0)


def _point_from_spec(idx: Mapping[str, int], points, frame_index: int, spec: MarkerSpec) -> Optional[Point3D]:
    if isinstance(spec, str):
        return _get_point(idx, points, frame_index, spec)
    spec_list = list(spec)
    if not spec_list:
        return None
    if len(spec_list) == 1:
        return _get_point(idx, points, frame_index, spec_list[0])
    # For pairs (e.g. malleoli/condyles), use center when both available.
    p0 = _get_point(idx, points, frame_index, spec_list[0])
    p1 = _get_point(idx, points, frame_index, spec_list[1])
    return _mean_points(p0, p1)


def _resolve_marker_point(
    idx: Mapping[str, int], points, frame_index: int, specs: Sequence[MarkerSpec]
) -> Optional[Point3D]:
    for spec in specs:
        p = _point_from_spec(idx, points, frame_index, spec)
        if p is not None:
            return p
    return None


def _normalize_custom_marker_map(marker_map: Mapping[str, MarkerSpec]) -> Dict[str, List[MarkerSpec]]:
    out: Dict[str, List[MarkerSpec]] = {}
    for canonical, spec in marker_map.items():
        if not isinstance(canonical, str) or not canonical.strip():
            raise ValueError("marker_map keys must be non-empty strings")
        if isinstance(spec, str):
            out[canonical] = [spec]
        else:
            seq = list(spec)
            if not seq:
                raise ValueError(f"marker_map entry for '{canonical}' is empty")
            out[canonical] = [tuple(seq) if len(seq) > 1 else seq[0]]
    return out

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


def _normalize_example_name(name: str) -> str:
    # Remove all whitespace and common separators for user-friendly aliases.
    return "".join(name.lower().split()).replace("-", "").replace("_", "")


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
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Example name must be a non-empty string")
    key = _normalize_example_name(name)
    if key not in _EXAMPLE_MAP:
        avail = sorted(set(_EXAMPLE_MAP.keys()) - {"fukuchi", "pd", "parkinsons", "avc"})
        raise ValueError(f"Unknown example {name!r}. Available: {avail}")
    path = _DATA_DIR / _EXAMPLE_MAP[key]
    if not path.exists():
        raise FileNotFoundError(f"Example data not found at {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def list_examples() -> list:
    """List available example names.

    Returns
    -------
    list of str
        Primary names (without aliases).
    """
    return ["healthy", "parkinson", "kuopio", "stroke"]


_ANGLE_KEYS = [
    "left_hip_angle",
    "right_hip_angle",
    "left_knee_angle",
    "right_knee_angle",
    "left_ankle_angle",
    "right_ankle_angle",
]

_ANGLE_ALIASES: Dict[str, Tuple[str, ...]] = {
    "left_hip_angle": ("left_hip_angle", "lhip", "hip_l", "hip_left", "l_hip", "lhipangles"),
    "right_hip_angle": ("right_hip_angle", "rhip", "hip_r", "hip_right", "r_hip", "rhipangles"),
    "left_knee_angle": ("left_knee_angle", "lknee", "knee_l", "knee_left", "l_knee", "lkneeangles"),
    "right_knee_angle": ("right_knee_angle", "rknee", "knee_r", "knee_right", "r_knee", "rkneeangles"),
    "left_ankle_angle": ("left_ankle_angle", "lankle", "ankle_l", "ankle_left", "l_ankle", "lankleangles"),
    "right_ankle_angle": ("right_ankle_angle", "rankle", "ankle_r", "ankle_right", "r_ankle", "rankleangles"),
}


def _norm_key(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _extract_1d_angle(values, n_frames: int) -> Optional[List[float]]:
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        return None
    arr = np.asarray(values)
    if arr.size == 0:
        return None
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        series = arr.astype(float)
    elif arr.ndim == 2:
        # Convention used in VICON exports: first column = flex/ext
        series = arr[:, 0].astype(float)
    else:
        return None
    if series.shape[0] < n_frames:
        return None
    return series[:n_frames].tolist()


def _load_angles_from_mapping(angles: Mapping[str, Sequence[float]], n_frames: int) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    keyed = {_norm_key(k): v for k, v in angles.items()}
    for target in _ANGLE_KEYS:
        for alias in _ANGLE_ALIASES[target]:
            raw = keyed.get(_norm_key(alias))
            if raw is None:
                continue
            series = _extract_1d_angle(raw, n_frames)
            if series is not None:
                out[target] = series
                break
    return out


def _load_angles_from_mat(path: Path, n_frames: int) -> Dict[str, List[float]]:
    from scipy.io import loadmat

    mat = loadmat(str(path))
    # Preferred structure from premanip-pipeline: res_angles_t struct with fields Lhip/Lknee/...
    if "res_angles_t" in mat:
        struct = mat["res_angles_t"][0, 0]
        extracted = {}
        for target, aliases in _ANGLE_ALIASES.items():
            for alias in aliases:
                # Mat field names are case-sensitive. Try direct matches first.
                candidates = [alias, alias.upper(), alias.capitalize()]
                for c in candidates:
                    if c in struct.dtype.names:
                        series = _extract_1d_angle(struct[c], n_frames)
                        if series is not None:
                            extracted[target] = series
                            break
                if target in extracted:
                    break
        if extracted:
            return extracted

    # Fallback: flat variables in .mat
    return _load_angles_from_mapping(mat, n_frames)


def _load_angles_from_csv(path: Path, n_frames: int) -> Dict[str, List[float]]:
    import pandas as pd

    df = pd.read_csv(path)
    return _load_angles_from_mapping(df.to_dict(orient="list"), n_frames)


def _load_angles_from_json(path: Path, n_frames: int) -> Dict[str, List[float]]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, Mapping):
        return {}
    return _load_angles_from_mapping(payload, n_frames)


def load_angles_file(angles: AnglesLike, n_frames: int) -> Dict[str, List[float]]:
    """Load external joint-angle series from MAT/CSV/JSON or in-memory mapping.

    Returns a dict with canonical keys:
    ``left_hip_angle``, ``right_hip_angle``, ``left_knee_angle``,
    ``right_knee_angle``, ``left_ankle_angle``, ``right_ankle_angle``.
    """
    if isinstance(angles, Mapping):
        return _load_angles_from_mapping(angles, n_frames)

    p = Path(angles)
    if not p.exists():
        raise FileNotFoundError(f"Angle file not found: {p}")
    suffix = p.suffix.lower()
    if suffix == ".mat":
        return _load_angles_from_mat(p, n_frames)
    if suffix in {".csv", ".txt"}:
        return _load_angles_from_csv(p, n_frames)
    if suffix in {".json"}:
        return _load_angles_from_json(p, n_frames)
    raise ValueError(f"Unsupported angle file format: {suffix or '<none>'}")


# ── C3D loading ──────────────────────────────────────────────────────

def load_c3d(
    path: str,
    marker_set: str = "auto",
    marker_map: Optional[Mapping[str, MarkerSpec]] = None,
    angles: Optional[AnglesLike] = None,
) -> dict:
    """Load a C3D file and extract angle frames.

    Parameters
    ----------
    path : str or Path
        Path to a .c3d file.
    marker_set : str
        Marker naming convention: "pig" (Plug-in Gait), "isb", "imy", or "auto".
    marker_map : dict, optional
        Custom mapping ``canonical_name -> label_or_pair`` used instead of presets.
        Example: ``{"left_heel": "LCAL", "left_ankle": ("LMM", "LLM")}``.
    angles : str, Path, or dict, optional
        Optional external angle source (MAT/CSV/JSON or in-memory mapping).
        When provided, these values override angle fields extracted from C3D.

    Returns
    -------
    dict
        Same format as :func:`load_example`: keys *angle_frames*, *fps*, *n_frames*.

    Raises
    ------
    ImportError
        If ezc3d is not installed. Install with ``pip install gaitkit[c3d]``.
    """
    if not isinstance(path, (str, Path)) or not str(path).strip():
        raise ValueError("path must be a non-empty string or Path")
    c3d_path = Path(path)
    if c3d_path.suffix.lower() != ".c3d":
        raise ValueError(f"Unsupported file format: {c3d_path.suffix or '<none>'}")
    if not c3d_path.exists():
        raise FileNotFoundError(f"C3D file not found: {c3d_path}")

    if not isinstance(marker_set, str) or not marker_set.strip():
        raise ValueError("marker_set must be 'auto', 'pig', 'isb', or 'imy'")

    try:
        import ezc3d
    except ImportError:
        raise ImportError(
            "ezc3d is required to read C3D files. "
            "Install with: pip install gaitkit[c3d]"
        )

    c3d = ezc3d.c3d(str(c3d_path))
    fps = float(c3d["header"]["points"]["frame_rate"])
    points = c3d["data"]["points"]           # (4, n_markers, n_frames)
    labels = c3d["parameters"]["POINT"]["LABELS"]["value"]
    labels = [l.strip() for l in labels]
    n_frames = points.shape[2]

    # Build label -> index mapping
    idx = {l.upper(): i for i, l in enumerate(labels)}

    # Detect marker set
    marker_set = marker_set.lower().strip()
    if marker_set == "auto":
        if any(k in idx for k in ("LCAL", "RCAL", "LFMH1", "RFMH1", "LMM", "RMM", "LLM", "RLM")):
            marker_set = "imy"
        elif "LHEE" in idx or "RHEE" in idx:
            marker_set = "pig"
        elif "LEFTHEEL" in idx:
            marker_set = "isb"
        else:
            marker_set = "pig"

    if marker_map is not None:
        resolved_map = _normalize_custom_marker_map(marker_map)
    else:
        if marker_set not in _MARKER_MAPS:
            raise ValueError(
                f"Unknown marker_set {marker_set!r}. Expected 'auto', 'pig', 'isb', or 'imy'."
            )
        resolved_map = _MARKER_MAPS[marker_set]

    # Extract landmarks per frame
    angle_frames = []
    for fi in range(n_frames):
        lp = {}
        for name, specs in resolved_map.items():
            p = _resolve_marker_point(idx, points, fi, specs)
            if p is not None:
                lp[name] = p
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

    if not any(frame["landmark_positions"] for frame in angle_frames):
        label_preview = ", ".join(labels[:20])
        raise ValueError(
            "No supported gait markers were found in the C3D file. "
            "Expected Plug-in Gait labels (e.g., LHEE/RHEE/LTOE/RTOE), "
            "ISB-style labels (e.g., LEFTHEEL/RIGHTHEEL), or IMY labels "
            "(e.g., LCAL/RCAL/LFMH1/RFMH1). "
            f"Detected labels (first 20): {label_preview}"
        )

    # Try to extract angles from MODEL outputs
    try:
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
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        logger.debug("Could not extract model angles from C3D payload: %s", exc)

    # Optional external angles override
    if angles is not None:
        ext = load_angles_file(angles, n_frames=n_frames)
        if not ext:
            raise ValueError(
                f"No usable angle columns were found in '{angles}'. "
                "Expected aliases like Lhip/Rhip/Lknee/Rknee/Lankle/Rankle or "
                "left_hip_angle/right_hip_angle/etc."
            )
        for fi in range(n_frames):
            for key, series in ext.items():
                angle_frames[fi][key] = float(series[fi])

    return {
        "angle_frames": angle_frames,
        "fps": fps,
        "n_frames": n_frames,
        "source_file": str(c3d_path),
    }
