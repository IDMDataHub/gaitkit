"""Static tests for documentation links expected by maintainers/users."""

from __future__ import annotations

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestDocsContract(unittest.TestCase):
    def test_readme_links_core_project_docs(self):
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("[TODO.md](TODO.md)", readme)
        self.assertIn("[CONTRIBUTING.md](CONTRIBUTING.md)", readme)
        self.assertIn("[SECURITY.md](SECURITY.md)", readme)
        self.assertIn("[REPRODUCIBILITY.md](REPRODUCIBILITY.md)", readme)


if __name__ == "__main__":
    unittest.main()
