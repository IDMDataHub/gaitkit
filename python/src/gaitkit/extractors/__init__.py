"""
Data Extractors Module.

Each extractor converts a specific motion capture dataset into the common
AngleFrame format used by all detectors, along with associated ground truth
gait events.
"""

from importlib import import_module

from .base_extractor import BaseExtractor, ExtractionResult, AngleFrame, GroundTruth

__all__ = [
    "BaseExtractor",
    "ExtractionResult",
    "AngleFrame",
    "GroundTruth",
]


def _optional_import(module_name, names):
    try:
        mod = import_module(module_name, package=__name__)
    except ModuleNotFoundError:
        return
    for name in names:
        globals()[name] = getattr(mod, name)
        __all__.append(name)


_optional_import(".nature_extractor", ["NatureC3DExtractor"])
_optional_import(".figshare_pd_extractor", ["FigsharePDExtractor"])
_optional_import(".fukuchi_extractor", ["FukuchiExtractor"])
_optional_import(
    ".vancriekinge_extractor",
    ["VanCriekingeExtractor", "VanCriekingeHealthyExtractor", "VanCriekingeStrokeExtractor"],
)
_optional_import(".schreiber_extractor", ["SchreiberExtractor"])
_optional_import(".vanderzee_extractor", ["VanderzeeExtractor"])
_optional_import(".kuopio_extractor", ["KuopioExtractor"])
_optional_import(".kuopio_openpose_extractor", ["KuopioOpenPoseExtractor"])
_optional_import(".hood_amputee_extractor", ["HoodAmputeeExtractor"])
_optional_import(".elderly_fallrisk_extractor", ["ElderlyFallriskExtractor"])
_optional_import(".camargo_extractor", ["CamargoExtractor"])
