"""gaitkit — Universal gait event detection toolkit.

Quick start
-----------
>>> import gaitkit
>>> trial = gaitkit.load_example("healthy")
>>> result = gaitkit.detect(trial)
>>> result.summary()

Available methods: bike (default), zeni, oconnor, hreljac, mickelborough,
ghoussayni, vancanneyt, dgei, intellevent, deepevent.
"""

__version__ = "1.2.1"

from ._core import detect, list_methods, GaitResult
from ._ensemble import detect_ensemble
from ._io import load_example, list_examples, load_c3d, verify_angles_against_external
from ._viz import compare_plot
from ._compat import build_angle_frames, detect_events_structured, export_detection

__all__ = [
    "detect",
    "detect_events_structured",
    "detect_ensemble",
    "export_detection",
    "build_angle_frames",
    "list_methods",
    "load_example",
    "list_examples",
    "load_c3d",
    "verify_angles_against_external",
    "compare_plot",
    "GaitResult",
    "__version__",
]
