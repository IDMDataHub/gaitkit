"""
Command line bridge for structured BIKEgait inputs.

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

import BIKEgait


def _load_payload(input_path: Path) -> Dict[str, Any]:
    with input_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
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
    parts = [p.strip().lower() for p in value.split(",") if p.strip()]
    if not parts:
        raise ValueError("--formats must not be empty")
    return parts


def main() -> int:
    parser = argparse.ArgumentParser(description="Run BIKEgait on structured JSON frames.")
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
    method = args.method or str(payload.get("method", "bayesian_bis"))
    fps = float(args.fps if args.fps is not None else payload.get("fps", 100.0))
    frames = payload.get("frames", [])

    units = payload.get("units", None)
    if units is None:
        units = {}
    if args.position_unit is not None:
        units["position"] = args.position_unit
    if args.angle_unit is not None:
        units["angles"] = args.angle_unit

    formats = _parse_formats(args.formats)

    if formats == ["json"]:
        result = BIKEgait.detect_events_structured(method, frames, fps, units=units)
        _write_json(args.output, result)
        return 0

    if args.output is None:
        raise ValueError("--output is required for multi-format export")

    result = BIKEgait.detect_events_structured(method, frames, fps, units=units)
    paths = BIKEgait.export_detection(result, output_prefix=args.output, formats=formats)
    _write_json(None, {"written": paths})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
