"""Contract tests for native-wrapper module imports."""

from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python" / "src"))


NATIVE_MODULES = {
    "gaitkit.oconnor_native": "OConnorNativeDetector",
    "gaitkit.hreljac_native": "HreljacNativeDetector",
    "gaitkit.mickelborough_native": "MickelboroughNativeDetector",
    "gaitkit.ghoussayni_native": "GhoussayniNativeDetector",
    "gaitkit.zeni_native": "ZeniNativeDetector",
    "gaitkit.dgei_native": "DGEINativeDetector",
    "gaitkit.bayesian_bis_native": "BayesianBisNativeGaitDetector",
    "gaitkit.vancanneyt_native": "VancanneytNativeDetector",
}


class TestNativeWrapperImports(unittest.TestCase):
    def test_native_wrapper_modules_import_and_expose_expected_classes(self):
        for module_name, class_name in NATIVE_MODULES.items():
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                self.assertTrue(hasattr(module, class_name))


if __name__ == "__main__":
    unittest.main()
