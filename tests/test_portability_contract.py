"""Static portability/regression guards for repository metadata and tests."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestPortabilityContract(unittest.TestCase):
    def test_no_hardcoded_local_repo_path_in_tests(self):
        forbidden = "/".join(["", "home", "ffer", "gaitkit"])
        for test_file in (PROJECT_ROOT / "tests").glob("*.py"):
            txt = test_file.read_text(encoding="utf-8")
            self.assertNotIn(forbidden, txt, f"Hardcoded path found in {test_file.name}")

    def test_no_legacy_repository_slug_in_public_docs(self):
        legacy = "fferlab/gaitkit"
        targets = [
            PROJECT_ROOT / "README.md",
            PROJECT_ROOT / "r" / "README.md",
            PROJECT_ROOT / "r" / "vignettes" / "getting-started.Rmd",
            PROJECT_ROOT / "python" / "pyproject.toml",
            PROJECT_ROOT / "r" / "DESCRIPTION",
        ]
        for path in targets:
            txt = path.read_text(encoding="utf-8")
            self.assertNotIn(legacy, txt, f"Legacy slug found in {path}")

    def test_no_bare_except_or_silent_exception_pass_in_python_sources(self):
        base_extractor = PROJECT_ROOT / "python" / "src" / "gaitkit" / "extractors" / "base_extractor.py"
        for path in (PROJECT_ROOT / "python" / "src" / "gaitkit").rglob("*.py"):
            if path == base_extractor:
                continue
            txt = path.read_text(encoding="utf-8")
            self.assertNotRegex(txt, r"(?m)^\s*except:\s*$", f"Bare except found in {path}")
            self.assertNotRegex(
                txt,
                r"(?ms)except\s+Exception\s*:\s*\n\s*pass\b",
                f"Silent 'except Exception: pass' found in {path}",
            )


if __name__ == "__main__":
    unittest.main()
