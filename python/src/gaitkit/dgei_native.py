"""
Native-accelerated DGEI detector (non-intrusive integration).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

try:
    from gaitkit.native import _gait_native as _native_solver
    _HAS_NATIVE = True
except ImportError:
    _native_solver = None
    _HAS_NATIVE = False

from gaitkit.detectors.dgei_detector import DGEIDetector


class DGEINativeDetector(DGEIDetector):
    """Drop-in replacement for DGEIDetector with C-accelerated core curve."""

    def _compute_dgei_curve(self, signal: np.ndarray):
        if _HAS_NATIVE:
            pos_raw, neg_raw = _native_solver.compute_dgei_curves_raw(
                np.asarray(signal, dtype=np.float64),
                float(self.fps),
                float(self.bar_threshold),
            )
            dgei_pos = gaussian_filter1d(np.asarray(pos_raw, dtype=np.float64), sigma=2)
            dgei_neg = gaussian_filter1d(np.asarray(neg_raw, dtype=np.float64), sigma=2)
            return dgei_pos, dgei_neg
        return super()._compute_dgei_curve(signal)
