"""Unit tests for CLI input parsing helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python" / "src"))

from gaitkit import cli


class TestCliHelpers(unittest.TestCase):
    def test_parse_formats_deduplicates_and_preserves_order(self):
        out = cli._parse_formats("json,csv,json,xlsx,myogait")
        self.assertEqual(out, ["json", "csv", "xlsx", "myogait"])

    def test_parse_formats_rejects_unknown_values(self):
        with self.assertRaises(ValueError):
            cli._parse_formats("json,xml")
        with self.assertRaises(ValueError):
            cli._parse_formats(1)  # type: ignore[arg-type]

    def test_load_payload_requires_object(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "in.json"
            p.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
            with self.assertRaises(ValueError):
                cli._load_payload(p)

    def test_load_payload_rejects_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "in.json"
            p.write_text("{invalid-json", encoding="utf-8")
            with self.assertRaises(ValueError):
                cli._load_payload(p)

    def test_normalize_units_accepts_supported_values(self):
        out = cli._normalize_units({"position": "MM", "angles": "Deg"})
        self.assertEqual(out, {"position": "mm", "angles": "deg"})

    def test_normalize_units_rejects_invalid_shapes(self):
        with self.assertRaises(ValueError):
            cli._normalize_units(["mm", "deg"])
        with self.assertRaises(ValueError):
            cli._normalize_units({"position": "cm"})
        with self.assertRaises(ValueError):
            cli._normalize_units({"angles": "grad"})

    def test_normalize_method_validates_non_empty_input(self):
        self.assertEqual(cli._normalize_method(" bike "), "bike")
        with self.assertRaises(ValueError):
            cli._normalize_method("")
        with self.assertRaises(ValueError):
            cli._normalize_method(None)

    def test_extract_detection_input_accepts_legacy_frames_payload(self):
        payload = {"fps": 120.0, "frames": [{"frame_index": 0}]}
        frames, fps = cli._extract_detection_input(payload)
        self.assertEqual(frames, [{"frame_index": 0}])
        self.assertEqual(fps, 120.0)

    def test_extract_detection_input_accepts_myogait_payload(self):
        payload = {
            "meta": {"fps": 100.0},
            "angles": {"frames": [{"frame_idx": 0, "hip_L": 1.0}]},
        }
        data, fps = cli._extract_detection_input(payload)
        self.assertIs(data, payload)
        self.assertEqual(fps, 100.0)

    def test_extract_detection_input_rejects_missing_frames(self):
        with self.assertRaises(ValueError):
            cli._extract_detection_input({"meta": {"fps": 100.0}})


if __name__ == "__main__":
    unittest.main()
