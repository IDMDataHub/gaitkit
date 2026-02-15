"""Compatibility helpers for legacy structured API consumers.

These helpers keep historical interfaces working (CLI/MATLAB/tests)
while delegating core detection to :func:`gaitkit.detect`.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from ._core import _dicts_to_angle_frames, detect, list_methods


def _normalize_units(units: Mapping[str, str] | None) -> Dict[str, str]:
    if units is None:
        return {"position": "mm", "angles": "deg"}
    out = dict(units)
    out.setdefault("position", "mm")
    out.setdefault("angles", "deg")
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
            if ang_key in d and d[ang_key] is not None:
                d[ang_key] = float(d[ang_key]) * ang_scale

        lp = d.get("landmark_positions")
        if isinstance(lp, Mapping):
            scaled = {}
            for name, coords in lp.items():
                scaled[name] = tuple(float(c) * pos_scale for c in coords)
            d["landmark_positions"] = scaled

        dict_frames.append(d)

    return _dicts_to_angle_frames(dict_frames)


def _cycles_from_events(
    left_hs: Sequence[Dict[str, Any]],
    right_hs: Sequence[Dict[str, Any]],
    left_to: Sequence[Dict[str, Any]],
    right_to: Sequence[Dict[str, Any]],
    fps: float,
) -> List[Dict[str, Any]]:
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
    frames: Sequence[Any],
    fps: float,
    units: Mapping[str, str] | None = None,
) -> Dict[str, Any]:
    """Legacy structured detection API.

    Returns a plain dictionary designed for easy JSON serialisation.
    """
    if fps is None or float(fps) <= 0:
        raise ValueError("fps must be a positive number")

    af = build_angle_frames(frames, units=units)
    result = detect(af, method=method, fps=float(fps))

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
            "fps_hz": float(fps),
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
    """Export legacy structured payload to JSON/CSV/XLSX."""
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    wanted = [str(f).lower().strip() for f in formats]
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

    return written
