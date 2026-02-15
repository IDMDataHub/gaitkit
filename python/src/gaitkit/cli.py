"""
Command line bridge for structured gaitkit inputs.

Input JSON format:
{
  "method": "bayesian_bis",
  "fps": 100.0,
  "units": {"position": "mm", "angles": "deg"},
  "frames": [...]
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

import gaitkit


def _load_payload(input_path: Path) -> Dict[str, Any]:
    try:
        with input_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {input_path}: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Input JSON must be an object")
    return payload


def _write_json(path: Path | None, payload: Dict[str, Any]) -> None:
    txt = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    if path is None:
        print(txt)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(txt)


def _parse_formats(value: str) -> Sequence[str]:
    if not isinstance(value, str):
        raise ValueError("--formats must be a comma-separated string")
    parts = [p.strip().lower() for p in value.split(",") if p.strip()]
    if not parts:
        raise ValueError("--formats must not be empty")
    allowed = {"json", "csv", "xlsx"}
    unknown = [p for p in parts if p not in allowed]
    if unknown:
        raise ValueError(f"Unknown format(s): {unknown}. Allowed: {sorted(allowed)}")
    # Preserve user order while removing duplicates.
    parts = list(dict.fromkeys(parts))
    return parts


def _normalize_units(units: Any) -> Dict[str, str]:
    if units is None:
        return {}
    if not isinstance(units, dict):
        raise ValueError("'units' must be a JSON object with optional keys 'position' and 'angles'")
    out: Dict[str, str] = {}
    if "position" in units:
        pos = str(units["position"]).strip().lower()
        if pos not in {"mm", "m"}:
            raise ValueError("units.position must be 'mm' or 'm'")
        out["position"] = pos
    if "angles" in units:
        ang = str(units["angles"]).strip().lower()
        if ang not in {"deg", "rad"}:
            raise ValueError("units.angles must be 'deg' or 'rad'")
        out["angles"] = ang
    return out


def _normalize_method(method: Any) -> str:
    value = str(method).strip() if method is not None else ""
    if not value:
        raise ValueError("'method' must be a non-empty string")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Run gaitkit on structured JSON frames.")
    parser.add_argument("--input", required=True, type=Path, help="Input JSON file path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path. For multi-format export, treated as output prefix.",
    )
    parser.add_argument("--method", type=str, default=None, help="Override method from input JSON.")
    parser.add_argument("--fps", type=float, default=None, help="Override fps from input JSON.")
    parser.add_argument("--position-unit", type=str, default=None, help="Override position unit: mm|m")
    parser.add_argument("--angle-unit", type=str, default=None, help="Override angle unit: deg|rad")
    parser.add_argument(
        "--formats",
        type=str,
        default="json",
        help="Comma-separated output formats: json,csv,xlsx",
    )
    args = parser.parse_args()

    payload = _load_payload(args.input)
    method = _normalize_method(args.method if args.method is not None else payload.get("method", "bayesian_bis"))
    fps = float(args.fps if args.fps is not None else payload.get("fps", 100.0))
    if fps <= 0:
        raise ValueError("fps must be strictly positive")
    frames = payload.get("frames", [])
    if not isinstance(frames, list):
        raise ValueError("'frames' must be a JSON array")

    units = _normalize_units(payload.get("units", None))
    if args.position_unit is not None:
        pos = str(args.position_unit).strip().lower()
        if pos not in {"mm", "m"}:
            raise ValueError("--position-unit must be 'mm' or 'm'")
        units["position"] = pos
    if args.angle_unit is not None:
        ang = str(args.angle_unit).strip().lower()
        if ang not in {"deg", "rad"}:
            raise ValueError("--angle-unit must be 'deg' or 'rad'")
        units["angles"] = ang

    formats = _parse_formats(args.formats)

    if formats == ["json"]:
        result = gaitkit.detect_events_structured(method, frames, fps, units=units)
        _write_json(args.output, result)
        return 0

    if args.output is None:
        raise ValueError("--output is required for multi-format export")

    result = gaitkit.detect_events_structured(method, frames, fps, units=units)
    paths = gaitkit.export_detection(result, output_prefix=args.output, formats=formats)
    _write_json(None, {"written": paths})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
