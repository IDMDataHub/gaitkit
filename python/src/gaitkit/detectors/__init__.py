"""
Gait Event Detectors Module.

Registry of detection algorithms for the benchmark.

Our method:
    - bayesian_bis: Bayesian detector with velocity budget ratios (publication)

Literature:
    - Zeni et al. (2008), O'Connor et al. (2007), Hreljac & Marshall (2000)
    - Mickelborough et al. (2000), Ghoussayni et al. (2004)
    - Vancanneyt/Bonnyaud & Dubois (2025)
    - IntellEvent - Horsak & Kranzl (2023), DeepEvent - Lempereur et al. (2020)
    - DGEI - Arnrich et al. (2024)
"""

from .zeni_detector import ZeniDetector
from .oconnor_detector import OConnorDetector
from .hreljac_detector import HreljacDetector
from .mickelborough_detector import MickelboroughDetector
from .ghoussayni_detector import GhoussayniDetector
from .dgei_detector import DGEIDetector
from .vancanneyt_detector import VancanneytDetector
from .bayesian_bis import BayesianBisGaitDetector

# IntellEvent requires onnxruntime -- graceful fallback
try:
    from .intellevent_detector import IntellEventDetector
    _HAS_INTELLEVENT = True
except ImportError:
    IntellEventDetector = None
    _HAS_INTELLEVENT = False

# DeepEvent requires tensorflow -- graceful fallback
try:
    from .deepevent_detector import DeepEventDetector
    _HAS_DEEPEVENT = True
except ImportError:
    DeepEventDetector = None
    _HAS_DEEPEVENT = False

# ---- Detector registry ----------------------------------------------------
DETECTOR_REGISTRY = {
    "zeni": ZeniDetector,
    "oconnor": OConnorDetector,
    "hreljac": HreljacDetector,
    "mickelborough": MickelboroughDetector,
    "ghoussayni": GhoussayniDetector,
    "dgei": DGEIDetector,
    "vancanneyt": VancanneytDetector,
    "bayesian_bis": BayesianBisGaitDetector,
}
if _HAS_INTELLEVENT:
    DETECTOR_REGISTRY["intellevent"] = IntellEventDetector
if _HAS_DEEPEVENT:
    DETECTOR_REGISTRY["deepevent"] = DeepEventDetector


def get_detector(name: str, fps: float = 100.0, **kwargs):
    key = name.lower()
    if key not in DETECTOR_REGISTRY:
        available = ", ".join(sorted(DETECTOR_REGISTRY.keys()))
        raise ValueError(f"Unknown detector '{name}'. Available: {available}")
    return DETECTOR_REGISTRY[key](fps=fps, **kwargs)


def list_detectors():
    return sorted(DETECTOR_REGISTRY.keys())


__all__ = ["DETECTOR_REGISTRY", "get_detector", "list_detectors"]
