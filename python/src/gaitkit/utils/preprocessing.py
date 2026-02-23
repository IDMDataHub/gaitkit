"""
Signal preprocessing utilities for gait event detection.

Provides filtering, normalization, and period estimation functions
used by multiple detectors.
"""

import numpy as np
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.ndimage import gaussian_filter1d
from scipy.fft import fft, fftfreq
from typing import Tuple


def interpolate_nan(signal: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN values in a 1-D signal.

    Edge NaNs are filled with the nearest valid value (forward/backward fill).
    Returns a copy; the original array is not modified.
    If the signal is all-NaN, returns zeros.

    Parameters
    ----------
    signal : ndarray, shape (N,)
        Input signal possibly containing NaN values.

    Returns
    -------
    ndarray, shape (N,)
        Signal with NaN gaps filled by linear interpolation.
    """
    out = signal.copy()
    nans = np.isnan(out)
    if not nans.any():
        return out
    if nans.all():
        return np.zeros_like(out)
    valid = ~nans
    out[nans] = np.interp(
        np.flatnonzero(nans),
        np.flatnonzero(valid),
        out[valid],
    )
    return out


def lowpass_butterworth(signal: np.ndarray, cutoff: float, fps: float,
                        order: int = 4) -> np.ndarray:
    """Apply a zero-phase Butterworth low-pass filter.

    Parameters
    ----------
    signal : ndarray
        Input signal.
    cutoff : float
        Cutoff frequency in Hz.
    fps : float
        Sampling rate in Hz.
    order : int
        Filter order (default 4).

    Returns
    -------
    ndarray
        Filtered signal.
    """
    nyquist = 0.5 * fps
    if cutoff >= nyquist:
        return signal.copy()
    b, a = butter(order, cutoff / nyquist, btype="low")
    padlen = min(len(signal) - 1, 3 * max(len(a), len(b)))
    return filtfilt(b, a, signal, padlen=padlen)


def smooth_signal(signal: np.ndarray, window: int = 11,
                  method: str = "savgol") -> np.ndarray:
    """Smooth a 1-D signal using Savitzky-Golay or Gaussian filter.

    Parameters
    ----------
    signal : ndarray
        Input signal.
    window : int
        Window length (must be odd for Savitzky-Golay).
    method : str
        ``'savgol'`` or ``'gaussian'``.

    Returns
    -------
    ndarray
        Smoothed signal.
    """
    if method == "savgol":
        w = window if window % 2 == 1 else window + 1
        if len(signal) > w:
            return savgol_filter(signal, w, 3)
        return gaussian_filter1d(signal, w / 6.0)
    elif method == "gaussian":
        return gaussian_filter1d(signal, window / 6.0)
    else:
        raise ValueError(f"Unknown smoothing method: {method}")


def estimate_period_fft(signal: np.ndarray, fps: float,
                         freq_range: Tuple[float, float] = (0.3, 3.0)
                         ) -> Tuple[float, float]:
    """Estimate the dominant period of a quasi-periodic signal via FFT.

    Parameters
    ----------
    signal : ndarray
        Input signal (e.g. crossing signal d(t)).
    fps : float
        Sampling rate in Hz.
    freq_range : tuple of float
        (min_freq, max_freq) search range in Hz.

    Returns
    -------
    tau : float
        Estimated period in seconds.
    sigma_tau : float
        Uncertainty on the period estimate (seconds).
    """
    n = len(signal)
    n_padded = 2 ** int(np.ceil(np.log2(n)) + 1)

    window = np.hanning(n)
    windowed = (signal - np.mean(signal)) * window

    spectrum = np.abs(fft(windowed, n_padded)) ** 2
    freqs = fftfreq(n_padded, 1 / fps)

    mask = (freqs > freq_range[0]) & (freqs < freq_range[1])
    if not np.any(mask):
        return 1.0, 0.2

    spec_pos = spectrum[mask]
    freqs_pos = freqs[mask]

    peak_idx = np.argmax(spec_pos)
    f0 = freqs_pos[peak_idx]

    # Estimate spectral width at -3 dB for uncertainty
    threshold = spec_pos[peak_idx] / 2
    width_bins = np.sum(spec_pos > threshold)
    df = freqs_pos[1] - freqs_pos[0] if len(freqs_pos) > 1 else 0.1
    sigma_f = width_bins * df / 2

    tau = 1.0 / f0 if f0 > 0 else 1.0
    sigma_tau = sigma_f / (f0 ** 2) if f0 > 0 else 0.2

    return tau, sigma_tau


def normalize_robust(signal: np.ndarray,
                      percentiles: Tuple[float, float] = (5, 95)) -> np.ndarray:
    """Robust min-max normalization to [0, 1] using percentiles.

    Parameters
    ----------
    signal : ndarray
        Input signal.
    percentiles : tuple of float
        Lower and upper percentiles for clipping.

    Returns
    -------
    ndarray
        Normalized signal clipped to [0, 1].
    """
    lo, hi = np.percentile(signal, percentiles)
    if hi - lo < 1e-10:
        return np.zeros_like(signal)
    return np.clip((signal - lo) / (hi - lo), 0, 1)
