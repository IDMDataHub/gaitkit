"""Compatibility helpers for legacy structured API consumers.

These helpers keep historical interfaces working (CLI/MATLAB/tests)
while delegating core detection to :func:`gaitkit.detect`.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from ._core import _dicts_to_angle_frames, detect, list_methods


def _normalize_units(units: Mapping[str, str] | None) -> Dict[str, str]:
    """Validate and normalize unit hints.

    Parameters
    ----------
    units:
        Optional mapping with keys ``position`` and/or ``angles``.
        Allowed values are ``mm``/``m`` for positions and ``deg``/``rad``
        for angles.

    Returns
    -------
    dict
        Normalized mapping including both keys.
    """
    if units is None:
        return {"position": "mm", "angles": "deg"}
    if not isinstance(units, Mapping):
        raise ValueError("units must be a mapping with optional keys 'position' and 'angles'")
    out = {k: str(v).lower().strip() for k, v in dict(units).items()}
    out.setdefault("position", "mm")
    out.setdefault("angles", "deg")
    if out["position"] not in {"mm", "m"}:
        raise ValueError("units.position must be 'mm' or 'm'")
    if out["angles"] not in {"deg", "rad"}:
        raise ValueError("units.angles must be 'deg' or 'rad'")
    return out


def build_angle_frames(
    frames: Sequence[Any],
    units: Mapping[str, str] | None = None,
):
    """Build AngleFrame-like objects from structured frames.

    Parameters
    ----------
    frames:
        Sequence of dictionaries or AngleFrame-like objects.
    units:
        Unit hints. Supported conversions:
        - position: ``m`` -> ``mm``
        - angles: ``rad`` -> ``deg``
    """
    if frames is None:
        return []
    if isinstance(frames, (str, bytes)):
        raise ValueError("frames must be a sequence of frame dictionaries, not a string")

    units_norm = _normalize_units(units)
    pos_scale = 1000.0 if str(units_norm.get("position", "mm")).lower() == "m" else 1.0
    ang_scale = 180.0 / math.pi if str(units_norm.get("angles", "deg")).lower() == "rad" else 1.0

    # Convert to dictionaries so scaling is homogeneous across inputs.
    dict_frames: List[Dict[str, Any]] = []
    for i, frame in enumerate(frames):
        if isinstance(frame, Mapping):
            d = dict(frame)
        else:
            d = {
                "frame_index": getattr(frame, "frame_index", i),
                "left_hip_angle": getattr(frame, "left_hip_angle", 0.0),
                "right_hip_angle": getattr(frame, "right_hip_angle", 0.0),
                "left_knee_angle": getattr(frame, "left_knee_angle", 0.0),
                "right_knee_angle": getattr(frame, "right_knee_angle", 0.0),
                "left_ankle_angle": getattr(frame, "left_ankle_angle", 0.0),
                "right_ankle_angle": getattr(frame, "right_ankle_angle", 0.0),
                "pelvis_tilt": getattr(frame, "pelvis_tilt", 0.0),
                "trunk_angle": getattr(frame, "trunk_angle", 0.0),
                "landmark_positions": getattr(frame, "landmark_positions", None),
            }

        for ang_key in (
            "left_hip_angle",
            "right_hip_angle",
            "left_knee_angle",
            "right_knee_angle",
            "left_ankle_angle",
            "right_ankle_angle",
            "pelvis_tilt",
            "trunk_angle",
        ):
            if ang_key in d:
                if d[ang_key] is None:
                    d.pop(ang_key, None)
                else:
                    d[ang_key] = float(d[ang_key]) * ang_scale

        lp = d.get("landmark_positions")
        if isinstance(lp, Mapping):
            scaled = {}
            for name, coords in lp.items():
                scaled[name] = tuple(float(c) * pos_scale for c in coords)
            d["landmark_positions"] = scaled

        dict_frames.append(d)

    return _dicts_to_angle_frames(dict_frames)


def _normalize_myogait_angle_frame(frame: Mapping[str, Any], index: int) -> Dict[str, Any]:
    """Map one MyoGait frame to gaitkit canonical angle fields."""
    out: Dict[str, Any] = {
        "frame_index": int(frame.get("frame_idx", index)),
        "trunk_angle": frame.get("trunk_angle"),
        "pelvis_tilt": frame.get("pelvis_tilt"),
        "left_hip_angle": frame.get("hip_L"),
        "right_hip_angle": frame.get("hip_R"),
        "left_knee_angle": frame.get("knee_L"),
        "right_knee_angle": frame.get("knee_R"),
        "left_ankle_angle": frame.get("ankle_L"),
        "right_ankle_angle": frame.get("ankle_R"),
    }
    lp = frame.get("landmark_positions")
    if isinstance(lp, Mapping):
        out["landmark_positions"] = lp
    return out


def _extract_frames_and_fps(
    frames: Sequence[Any] | Mapping[str, Any] | str | Path,
    fps: float,
) -> Tuple[Sequence[Any], float]:
    """Resolve input frames from raw frames, MyoGait payload, or JSON path.

    Supported inputs are:
    - list/sequence of frame dictionaries,
    - mapping payload with ``angles.frames`` (MyoGait style),
    - path to a ``.json`` file containing such a payload.
    """
    # Path to JSON payload
    if isinstance(frames, (str, Path)):
        p = Path(frames)
        if p.suffix.lower() == ".json" and p.exists():
            payload = json.loads(p.read_text(encoding="utf-8"))
            return _extract_frames_and_fps(payload, fps)
        raise ValueError("frames path must point to an existing .json file")

    # MyoGait payload as mapping
    if isinstance(frames, Mapping):
        if "angles" in frames and isinstance(frames["angles"], Mapping):
            angle_frames = frames["angles"].get("frames", [])
            if not isinstance(angle_frames, Sequence):
                raise ValueError("Invalid myogait payload: angles.frames must be a sequence")
            norm = [
                _normalize_myogait_angle_frame(fr, i)
                for i, fr in enumerate(angle_frames)
                if isinstance(fr, Mapping)
            ]
            resolved_fps = float(
                frames.get("meta", {}).get("fps", fps) if isinstance(frames.get("meta"), Mapping) else fps
            )
            return norm, resolved_fps
        if "frames" in frames and isinstance(frames["frames"], Sequence):
            # Generic mapping wrapper around raw frames
            return frames["frames"], fps
        raise ValueError("Unsupported mapping input: expected myogait payload with angles.frames")

    # Original list-of-frames path
    return frames, fps


def _cycles_from_events(
    left_hs: Sequence[Dict[str, Any]],
    right_hs: Sequence[Dict[str, Any]],
    left_to: Sequence[Dict[str, Any]],
    right_to: Sequence[Dict[str, Any]],
    fps: float,
) -> List[Dict[str, Any]]:
    """Build simple gait cycles from HS/TO events."""
    cycles: List[Dict[str, Any]] = []
    cycle_id = 0
    for side, hs_list, to_list in (
        ("left", left_hs, left_to),
        ("right", right_hs, right_to),
    ):
        hs_sorted = sorted(hs_list, key=lambda x: x["frame"])
        to_sorted = sorted(to_list, key=lambda x: x["frame"])
        for i in range(len(hs_sorted) - 1):
            hs0 = hs_sorted[i]["frame"]
            hs1 = hs_sorted[i + 1]["frame"]
            duration = (hs1 - hs0) / fps if fps > 0 else 0.0
            to_mid = next((e["frame"] for e in to_sorted if hs0 < e["frame"] < hs1), None)
            stance = None
            if to_mid is not None and hs1 > hs0:
                stance = (to_mid - hs0) / (hs1 - hs0) * 100.0
            cycles.append(
                {
                    "cycle_id": cycle_id,
                    "side": side,
                    "start_frame": hs0,
                    "toe_off_frame": to_mid,
                    "end_frame": hs1,
                    "duration": round(duration, 6),
                    "stance_percentage": round(stance, 4) if stance is not None else None,
                }
            )
            cycle_id += 1
    return cycles


def detect_events_structured(
    method: str,
    frames: Sequence[Any] | Mapping[str, Any] | str | Path,
    fps: float,
    units: Mapping[str, str] | None = None,
) -> Dict[str, Any]:
    """Legacy structured detection API.

    Parameters
    ----------
    method:
        Detection method name (e.g. ``"bike"``).
    frames:
        One of:
        - sequence of frame dictionaries / AngleFrame-like objects,
        - MyoGait payload mapping with ``angles.frames``,
        - path to a MyoGait ``.json`` file.
    fps:
        Sampling rate in Hz. Used as fallback when no ``meta.fps`` is present.
    units:
        Optional unit hints for structured frames.

    Returns a plain dictionary designed for easy JSON serialisation.
    """
    if not isinstance(fps, (int, float)) or fps <= 0:
        raise ValueError("fps must be a positive number")
    if not isinstance(method, str) or not method.strip():
        raise ValueError("method must be a non-empty string")

    resolved_frames, resolved_fps = _extract_frames_and_fps(frames, float(fps))
    af = build_angle_frames(resolved_frames, units=units)
    result = detect(af, method=method, fps=float(resolved_fps))

    heel_strikes = []
    toe_offs = []
    for ev in result.left_hs:
        heel_strikes.append(
            {
                "frame_index": int(ev["frame"]),
                "time_s": float(ev["time"]),
                "side": "left",
                "confidence": float(ev.get("confidence", 1.0)),
            }
        )
    for ev in result.right_hs:
        heel_strikes.append(
            {
                "frame_index": int(ev["frame"]),
                "time_s": float(ev["time"]),
                "side": "right",
                "confidence": float(ev.get("confidence", 1.0)),
            }
        )
    heel_strikes.sort(key=lambda x: x["frame_index"])

    for ev in result.left_to:
        toe_offs.append(
            {
                "frame_index": int(ev["frame"]),
                "time_s": float(ev["time"]),
                "side": "left",
                "confidence": float(ev.get("confidence", 1.0)),
            }
        )
    for ev in result.right_to:
        toe_offs.append(
            {
                "frame_index": int(ev["frame"]),
                "time_s": float(ev["time"]),
                "side": "right",
                "confidence": float(ev.get("confidence", 1.0)),
            }
        )
    toe_offs.sort(key=lambda x: x["frame_index"])

    cycles = _cycles_from_events(result.left_hs, result.right_hs, result.left_to, result.right_to, float(fps))

    return {
        "meta": {
            "detector": result.method,
            "fps_hz": float(resolved_fps),
            "n_frames": int(result.n_frames),
            "available_methods": list_methods(),
        },
        "heel_strikes": heel_strikes,
        "toe_offs": toe_offs,
        "cycles": cycles,
    }


def export_detection(
    payload: Mapping[str, Any],
    output_prefix: str | Path,
    formats: Iterable[str] = ("json",),
) -> Dict[str, str]:
    """Export structured detection payload to one or multiple formats.

    Supported formats are ``json``, ``csv``, ``xlsx``, and ``myogait``.
    The ``myogait`` export writes only the events block, compatible with
    MyoGait-like consumers.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("payload must be a mapping")

    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(formats, str):
        wanted = [formats.lower().strip()]
    else:
        wanted = [str(f).lower().strip() for f in formats]
    if not wanted:
        raise ValueError("formats must not be empty")
    allowed = {"json", "csv", "xlsx", "myogait"}
    unknown = [f for f in wanted if f not in allowed]
    if unknown:
        raise ValueError(f"Unknown export format(s): {unknown}. Allowed: {sorted(allowed)}")
    wanted = list(dict.fromkeys(wanted))
    written: Dict[str, str] = {}

    if "json" in wanted:
        p = prefix.with_suffix(".json")
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        written["json"] = str(p)

    events = []
    for ev in payload.get("heel_strikes", []):
        row = dict(ev)
        row["event_type"] = "HS"
        events.append(row)
    for ev in payload.get("toe_offs", []):
        row = dict(ev)
        row["event_type"] = "TO"
        events.append(row)
    events.sort(key=lambda x: int(x.get("frame_index", 0)))
    cycles = list(payload.get("cycles", []))

    if "csv" in wanted:
        ev_path = prefix.with_name(prefix.name + "_events").with_suffix(".csv")
        cy_path = prefix.with_name(prefix.name + "_cycles").with_suffix(".csv")
        with ev_path.open("w", newline="", encoding="utf-8") as f:
            cols = ["frame_index", "time_s", "event_type", "side", "confidence"]
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for row in events:
                w.writerow({k: row.get(k) for k in cols})
        with cy_path.open("w", newline="", encoding="utf-8") as f:
            cols = ["cycle_id", "side", "start_frame", "toe_off_frame", "end_frame", "duration", "stance_percentage"]
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for row in cycles:
                w.writerow({k: row.get(k) for k in cols})
        written["csv_events"] = str(ev_path)
        written["csv_cycles"] = str(cy_path)

    if "xlsx" in wanted:
        try:
            from openpyxl import Workbook
        except ImportError as exc:
            raise RuntimeError("openpyxl is required for xlsx export") from exc
        x_path = prefix.with_suffix(".xlsx")
        wb = Workbook()
        ws = wb.active
        ws.title = "events"
        ev_cols = ["frame_index", "time_s", "event_type", "side", "confidence"]
        ws.append(ev_cols)
        for row in events:
            ws.append([row.get(c) for c in ev_cols])
        ws2 = wb.create_sheet("cycles")
        cy_cols = ["cycle_id", "side", "start_frame", "toe_off_frame", "end_frame", "duration", "stance_percentage"]
        ws2.append(cy_cols)
        for row in cycles:
            ws2.append([row.get(c) for c in cy_cols])
        wb.save(x_path)
        written["xlsx"] = str(x_path)

    if "myogait" in wanted:
        mg_path = prefix.with_name(prefix.name + "_myogait_events").with_suffix(".json")
        fps = float(payload.get("meta", {}).get("fps_hz", 0.0)) if isinstance(payload.get("meta"), Mapping) else 0.0

        def _pack(side_events):
            out = []
            for ev in side_events:
                frame = int(ev.get("frame_index", 0))
                out.append(
                    {
                        "frame": frame,
                        "time": float(ev.get("time_s", frame / fps if fps > 0 else 0.0)),
                        "confidence": float(ev.get("confidence", 1.0)),
                    }
                )
            return out

        left_hs = [e for e in payload.get("heel_strikes", []) if str(e.get("side", "")).lower() == "left"]
        right_hs = [e for e in payload.get("heel_strikes", []) if str(e.get("side", "")).lower() == "right"]
        left_to = [e for e in payload.get("toe_offs", []) if str(e.get("side", "")).lower() == "left"]
        right_to = [e for e in payload.get("toe_offs", []) if str(e.get("side", "")).lower() == "right"]
        mg = {
            "events": {
                "method": payload.get("meta", {}).get("detector") if isinstance(payload.get("meta"), Mapping) else None,
                "fps": fps if fps > 0 else None,
                "left_hs": _pack(left_hs),
                "right_hs": _pack(right_hs),
                "left_to": _pack(left_to),
                "right_to": _pack(right_to),
            }
        }
        mg_path.write_text(json.dumps(mg, ensure_ascii=False, indent=2), encoding="utf-8")
        written["myogait"] = str(mg_path)

    return written
