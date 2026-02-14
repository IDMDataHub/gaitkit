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

__version__ = "0.1.0"

from ._core import detect, list_methods, GaitResult
from ._ensemble import detect_ensemble
from ._io import load_example, list_examples, load_c3d
from ._viz import compare_plot

__all__ = [
    "detect",
    "detect_ensemble",
    "list_methods",
    "load_example",
    "list_examples",
    "load_c3d",
    "compare_plot",
    "GaitResult",
    "__version__",
]
