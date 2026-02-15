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


if __name__ == "__main__":
    unittest.main()
