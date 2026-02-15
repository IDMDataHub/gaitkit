"""Static tests for packaging contracts (PyPI/installation expectations)."""

from __future__ import annotations

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestPackagingContract(unittest.TestCase):
    def test_h5_not_in_python_package_payload(self):
        pyproject = (PROJECT_ROOT / "python" / "pyproject.toml").read_text(encoding="utf-8")
        manifest = (PROJECT_ROOT / "python" / "MANIFEST.in").read_text(encoding="utf-8")

        self.assertNotIn("data/*.h5", pyproject)
        self.assertNotIn("data/*.h5", manifest)

    def test_weights_kept_in_repo_assets(self):
        weights_path = PROJECT_ROOT / "assets" / "DeepEventWeight.h5"
        self.assertTrue(weights_path.exists(), "Expected DeepEvent weights in assets/")

    def test_no_recursive_self_reference_in_all_extra(self):
        pyproject = (PROJECT_ROOT / "python" / "pyproject.toml").read_text(encoding="utf-8")
        self.assertNotIn("gaitkit[onnx,deep,viz,c3d]", pyproject)

    def test_test_extra_is_declared(self):
        pyproject = (PROJECT_ROOT / "python" / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn("test = [", pyproject)


if __name__ == "__main__":
    unittest.main()
