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
        out = cli._parse_formats("json,csv,json,xlsx")
        self.assertEqual(out, ["json", "csv", "xlsx"])

    def test_parse_formats_rejects_unknown_values(self):
        with self.assertRaises(ValueError):
            cli._parse_formats("json,xml")

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


if __name__ == "__main__":
    unittest.main()
