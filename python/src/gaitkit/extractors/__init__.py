"""
Data Extractors Module.

Each extractor converts a specific motion capture dataset into the common
AngleFrame format used by all detectors, along with associated ground truth
gait events.
"""

from .base_extractor import BaseExtractor, ExtractionResult, AngleFrame, GroundTruth
from .nature_extractor import NatureC3DExtractor
from .figshare_pd_extractor import FigsharePDExtractor
from .fukuchi_extractor import FukuchiExtractor
from .vancriekinge_extractor import VanCriekingeExtractor, VanCriekingeHealthyExtractor, VanCriekingeStrokeExtractor
from .schreiber_extractor import SchreiberExtractor
from .vanderzee_extractor import VanderzeeExtractor
from .kuopio_extractor import KuopioExtractor
from .kuopio_openpose_extractor import KuopioOpenPoseExtractor
from .hood_amputee_extractor import HoodAmputeeExtractor
from .elderly_fallrisk_extractor import ElderlyFallriskExtractor
from .camargo_extractor import CamargoExtractor

__all__ = [
    "BaseExtractor", "ExtractionResult", "AngleFrame", "GroundTruth",
    "NatureC3DExtractor", "FigsharePDExtractor", "FukuchiExtractor",
    "VanCriekingeExtractor", "VanCriekingeHealthyExtractor", "VanCriekingeStrokeExtractor",
    "SchreiberExtractor",
    "VanderzeeExtractor",
    "KuopioExtractor",
    "KuopioOpenPoseExtractor",
    "HoodAmputeeExtractor",
    "ElderlyFallriskExtractor",
    "CamargoExtractor",
]
