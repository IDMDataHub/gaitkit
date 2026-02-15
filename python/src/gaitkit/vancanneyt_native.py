"""
Native-optimized Vancanneyt detector (non-intrusive integration).

Current optimization:
- cache Butterworth coefficients to avoid repeated filter design cost.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

from gaitkit.detectors.vancanneyt_detector import VancanneytDetector


class VancanneytNativeDetector(VancanneytDetector):
    """Drop-in replacement with cached low-pass filter coefficients."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lp_b = None
        self._lp_a = None
        self._lp_min_len = None

        nyquist = self.fps / 2.0
        if self.filter_cutoff < nyquist:
            b, a = butter(4, self.filter_cutoff / nyquist, btype="low")
            self._lp_b = b
            self._lp_a = a
            self._lp_min_len = 3 * max(len(a), len(b))

    def _lowpass_filter(self, data):
        if self._lp_b is None or self._lp_a is None:
            return data
        if data.shape[0] < self._lp_min_len:
            return data
        filtered = np.zeros_like(data)
        for col in range(data.shape[1]):
            filtered[:, col] = filtfilt(self._lp_b, self._lp_a, data[:, col])
        return filtered
