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


def _extract_1d_angle(values, n_frames: Optional[int]) -> Optional[List[float]]:
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
    n_src = int(series.shape[0])
    if n_src <= 0:
        return None
    if n_frames is None or n_src == n_frames:
        return series.tolist()
    if n_frames is not None and n_src == 1:
        return [float(series[0])] * n_frames
    # Robust fallback for mismatched lengths (cropped/segmented exports):
    # resample on normalized time [0, 1] to match C3D frame count.
    if n_frames is None:
        return series.tolist()
    x_old = np.linspace(0.0, 1.0, num=n_src)
    x_new = np.linspace(0.0, 1.0, num=n_frames)
    out = np.interp(x_new, x_old, series)
    logger.debug(
        "Resampled external angle series from %d to %d frames.",
        n_src,
        n_frames,
    )
    return out.tolist()


def _load_angles_from_mapping(
    angles: Mapping[str, Sequence[float]], n_frames: Optional[int]
) -> Dict[str, List[float]]:
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


def _load_angles_from_mat(path: Path, n_frames: Optional[int]) -> Dict[str, List[float]]:
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


def _load_angles_from_csv(path: Path, n_frames: Optional[int]) -> Dict[str, List[float]]:
    import pandas as pd

    df = pd.read_csv(path)
    return _load_angles_from_mapping(df.to_dict(orient="list"), n_frames)


def _load_angles_from_json(path: Path, n_frames: Optional[int]) -> Dict[str, List[float]]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, Mapping):
        return {}
    return _load_angles_from_mapping(payload, n_frames)


def load_angles_file(angles: AnglesLike, n_frames: Optional[int] = None) -> Dict[str, List[float]]:
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


def _signed_angle_deg(v1, v2) -> float:
    import numpy as np

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0.0 or n2 == 0.0:
        return float("nan")
    a = v1 / n1
    b = v2 / n2
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    ang = float(np.degrees(np.arccos(dot)))
    cross = a[0] * b[1] - a[1] * b[0]
    return -ang if cross < 0 else ang


def compute_marker_proxy_angles(angle_frames: Sequence[Mapping[str, object]]) -> Dict[str, List[float]]:
    """Compute proxy sagittal angles from landmark positions.

    This function is designed for sanity-checking against external angle exports.
    It does not claim strict clinical equivalence with VICON model-based kinematics.
    """
    import numpy as np

    if not angle_frames:
        return {k: [] for k in _ANGLE_KEYS}

    # Progression axis from heel displacement (same principle as IntellEvent).
    def _coord(frame, name):
        lp = frame.get("landmark_positions")
        if not lp or name not in lp:
            return None
        p = lp[name]
        return np.array([float(p[0]), float(p[1]), float(p[2])], dtype=float)

    first = None
    last = None
    for fr in angle_frames:
        p = _coord(fr, "left_heel")
        if p is None:
            p = _coord(fr, "right_heel")
        if p is not None:
            if first is None:
                first = p
            last = p
    prog_axis = 0
    if first is not None and last is not None:
        dx = abs(last[0] - first[0])
        dy = abs(last[1] - first[1])
        prog_axis = 0 if dx >= dy else 1
    vert_axis = 2

    out = {k: [] for k in _ANGLE_KEYS}

    for fr in angle_frames:
        lp = fr.get("landmark_positions") or {}

        def p2(name):
            if name not in lp:
                return None
            p = lp[name]
            return np.array([float(p[prog_axis]), float(p[vert_axis])], dtype=float)

        for side in ("left", "right"):
            hip = p2(f"{side}_hip")
            knee = p2(f"{side}_knee")
            ankle = p2(f"{side}_ankle")
            toe = p2(f"{side}_toe")
            sacrum = p2("sacrum")

            # Knee flexion proxy: 180 - angle(HIP-KNEE-ANKLE)
            if hip is not None and knee is not None and ankle is not None:
                v1 = hip - knee
                v2 = ankle - knee
                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)
                if n1 > 0 and n2 > 0:
                    ang = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))
                    out[f"{side}_knee_angle"].append(float(180.0 - ang))
                else:
                    out[f"{side}_knee_angle"].append(float("nan"))
            else:
                out[f"{side}_knee_angle"].append(float("nan"))

            # Ankle proxy: angle(KNEE-ANKLE-TOE) - 90
            if knee is not None and ankle is not None and toe is not None:
                v1 = knee - ankle
                v2 = toe - ankle
                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)
                if n1 > 0 and n2 > 0:
                    ang = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))
                    out[f"{side}_ankle_angle"].append(float(ang - 90.0))
                else:
                    out[f"{side}_ankle_angle"].append(float("nan"))
            else:
                out[f"{side}_ankle_angle"].append(float("nan"))

            # Hip proxy: signed angle between trunk (SACR-HIP) and thigh (KNEE-HIP)
            if sacrum is not None and hip is not None and knee is not None:
                trunk = sacrum - hip
                thigh = knee - hip
                out[f"{side}_hip_angle"].append(_signed_angle_deg(trunk, thigh))
            else:
                out[f"{side}_hip_angle"].append(float("nan"))

    return out


def _extract_hs_frames_from_c3d(path: Union[str, Path], fps_hint: Optional[float] = None) -> List[int]:
    import numpy as np
    import ezc3d

    c3d = ezc3d.c3d(str(path))
    fps = float(fps_hint or c3d["header"]["points"]["frame_rate"])
    ev = c3d.get("parameters", {}).get("EVENT", {})
    labels = [str(x).strip() for x in ev.get("LABELS", {}).get("value", [])]
    contexts = [str(x).strip() for x in ev.get("CONTEXTS", {}).get("value", [])]
    raw_times = np.array(ev.get("TIMES", {}).get("value", []), dtype=float)
    if raw_times.size == 0 or not labels:
        return []
    if raw_times.ndim == 2 and raw_times.shape[0] == 2:
        times = (raw_times[0, :] * 60.0 + raw_times[1, :]).tolist()
    else:
        times = raw_times.reshape(-1).tolist()

    hs = []
    for i, lab in enumerate(labels):
        lab_low = lab.lower()
        if "foot strike" in lab_low or "heel strike" in lab_low or lab_low in {"hs", "h.s."}:
            if i < len(times):
                hs.append(int(round(float(times[i]) * fps)))
    hs = sorted(set(hs))
    return hs


def verify_angles_against_external(
    c3d_path: Union[str, Path],
    angles: AnglesLike,
    *,
    marker_set: str = "auto",
    align_start: str = "second_hs",
) -> Dict[str, object]:
    """Compare marker-derived proxy angles against external angle series.

    Parameters
    ----------
    c3d_path : str or Path
        Input C3D file.
    angles : path or mapping
        External angles (MAT/CSV/JSON or dict).
    marker_set : str
        Marker set preset for C3D loading.
    align_start : {"second_hs", "first_hs", "none"}
        How to place external angle frame 0 on the C3D timeline.
    """
    import numpy as np

    trial = load_c3d(str(c3d_path), marker_set=marker_set)
    n = int(trial["n_frames"])
    angle_frames = trial["angle_frames"]
    computed = compute_marker_proxy_angles(angle_frames)
    external_raw = load_angles_file(angles, n_frames=None)

    hs_frames = _extract_hs_frames_from_c3d(c3d_path, fps_hint=float(trial["fps"]))
    if align_start == "second_hs":
        start = hs_frames[1] if len(hs_frames) >= 2 else (hs_frames[0] if hs_frames else 0)
    elif align_start == "first_hs":
        start = hs_frames[0] if hs_frames else 0
    elif align_start == "none":
        start = 0
    else:
        raise ValueError("align_start must be 'second_hs', 'first_hs', or 'none'")

    metrics = {}
    for key in _ANGLE_KEYS:
        c = np.array(computed.get(key, []), dtype=float)
        ext_src = np.array(external_raw.get(key, []), dtype=float)
        if c.size == 0 or ext_src.size == 0:
            metrics[key] = {"n": 0, "rmse": None, "mae": None, "bias": None, "corr": None}
            continue

        ext = np.full(n, np.nan, dtype=float)
        end = min(n, start + ext_src.size)
        if end > start:
            ext[start:end] = ext_src[: end - start]
        valid = np.isfinite(c) & np.isfinite(ext)
        if not np.any(valid):
            metrics[key] = {"n": 0, "rmse": None, "mae": None, "bias": None, "corr": None}
            continue
        d = c[valid] - ext[valid]
        rmse = float(np.sqrt(np.mean(d ** 2)))
        mae = float(np.mean(np.abs(d)))
        bias = float(np.mean(d))
        corr = None
        if np.sum(valid) >= 3:
            with np.errstate(invalid="ignore"):
                cc = np.corrcoef(c[valid], ext[valid])[0, 1]
            if np.isfinite(cc):
                corr = float(cc)
        metrics[key] = {"n": int(np.sum(valid)), "rmse": rmse, "mae": mae, "bias": bias, "corr": corr}

    return {
        "fps": float(trial["fps"]),
        "n_frames_c3d": n,
        "external_length": {k: len(v) for k, v in external_raw.items()},
        "hs_frames": hs_frames,
        "alignment_start_frame": int(start),
        "metrics": metrics,
    }


def _align_external_angle_series(
    series: Sequence[float],
    *,
    n_frames: int,
    start: int,
    mode: str,
) -> List[float]:
    """Align one external angle series to the C3D timeline."""
    import numpy as np

    src = np.asarray(series, dtype=float).reshape(-1)
    if src.size == 0:
        return [0.0] * int(n_frames)

    n = int(n_frames)
    if n <= 0:
        return []

    if mode == "resample":
        if src.size == n:
            return src.astype(float).tolist()
        if src.size == 1:
            return [float(src[0])] * n
        x_old = np.linspace(0.0, 1.0, num=int(src.size))
        x_new = np.linspace(0.0, 1.0, num=n)
        return np.interp(x_new, x_old, src).astype(float).tolist()

    # Placement mode: insert source at `start`, pad edges with first/last value.
    out = np.full(n, float(src[-1]), dtype=float)
    start_i = max(0, int(start))
    if start_i > 0:
        out[:start_i] = float(src[0])
    end_i = min(n, start_i + int(src.size))
    if end_i > start_i:
        out[start_i:end_i] = src[: end_i - start_i]
    return out.tolist()


def _resolve_external_angle_alignment_mode(
    *,
    angles_align: str,
    hs_frames: Sequence[int],
    has_length_mismatch: bool,
) -> Tuple[str, int]:
    align_key = str(angles_align or "auto").strip().lower()
    if align_key not in {"auto", "none", "first_hs", "second_hs", "resample"}:
        raise ValueError("angles_align must be 'auto', 'none', 'first_hs', 'second_hs', or 'resample'")

    if align_key == "none":
        return "place", 0
    if align_key == "first_hs":
        return "place", int(hs_frames[0]) if hs_frames else 0
    if align_key == "second_hs":
        if len(hs_frames) >= 2:
            return "place", int(hs_frames[1])
        return "place", int(hs_frames[0]) if hs_frames else 0
    if align_key == "resample":
        return "resample", 0

    # auto
    if has_length_mismatch and hs_frames:
        if len(hs_frames) >= 2:
            return "place", int(hs_frames[1])
        return "place", int(hs_frames[0])
    return "resample", 0


# ── C3D loading ──────────────────────────────────────────────────────

def load_c3d(
    path: str,
    marker_set: str = "auto",
    marker_map: Optional[Mapping[str, MarkerSpec]] = None,
    angles: Optional[AnglesLike] = None,
    angles_align: str = "auto",
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
    angles_align : str
        External angle alignment mode when lengths differ from C3D:
        ``"auto"`` (default), ``"second_hs"``, ``"first_hs"``, ``"none"``,
        or ``"resample"``.

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
        ext_raw = load_angles_file(angles, n_frames=None)
        hs_frames = _extract_hs_frames_from_c3d(c3d_path, fps_hint=fps)
        has_mismatch = any(len(v) != n_frames for v in ext_raw.values())
        align_mode, align_start = _resolve_external_angle_alignment_mode(
            angles_align=angles_align,
            hs_frames=hs_frames,
            has_length_mismatch=has_mismatch,
        )
        ext = {
            key: _align_external_angle_series(
                series,
                n_frames=n_frames,
                start=align_start,
                mode=align_mode,
            )
            for key, series in ext_raw.items()
        }
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
