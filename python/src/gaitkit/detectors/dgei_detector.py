"""
Dynamic Gait Event Identifier (DGEI) Detector.

Reference
---------
Burdack, J., Horst, F., Giesselbach, S., Hassan, I., Schollhorn, W. I.,
& Arnrich, B. (2024). Adaptive Detection in Real-Time Gait Analysis
through the Dynamic Gait Event Identifier.
*Bioengineering*, 11(4), 391. (MDPI)

Principle
---------
First-order differences of foot vertical trajectories are split into
positive (foot rise = swing) and negative (foot descent = strike)
components.  An adaptive sleep-time mechanism and a weighted-average
peak queue control the detection threshold.  Originally designed for
IMU data; adapted here for 3-D marker positions.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


@dataclass
class GaitEvent:
    """A single gait event."""
    frame_index: int
    time: float
    event_type: str  # 'heel_strike' ou 'toe_off'
    side: str  # 'left' ou 'right'
    probability: float = 1.0


class DGEIDetector:
    """
    Dynamic Gait Event Identifier.

    Utilise les différences de premier ordre des positions des pieds
    pour détecter les événements de marche.
    """

    def __init__(self, fps: float = 100.0,
                 sleep_frames: int = None,
                 bar_threshold: float = 0.01,
                 peak_ratio: float = 0.6,
                 queue_length: int = 5):
        """
        Args:
            fps: Fréquence d'échantillonnage
            sleep_frames: Frames à ignorer après une détection (auto si None)
            bar_threshold: Seuil pour distinguer positive/negative
            peak_ratio: Ratio minimum pour valider un pic (0.6 = 60%)
            queue_length: Longueur de la queue pour le seuil adaptatif
        """
        self.fps = fps
        self.sleep_frames = sleep_frames if sleep_frames else int(0.3 * fps)  # ~300ms
        self.bar_threshold = bar_threshold
        self.peak_ratio = peak_ratio
        self.queue_length = queue_length

    def _extract_foot_signals(self, angle_frames) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract vertical foot position signals.

        Returns:
            left_heel_z, right_heel_z
        """
        n = len(angle_frames)
        left_z = np.zeros(n)
        right_z = np.zeros(n)

        for i, frame in enumerate(angle_frames):
            if frame.landmark_positions:
                # Utiliser heel si disponible, sinon ankle
                if 'left_heel' in frame.landmark_positions:
                    left_z[i] = frame.landmark_positions['left_heel'][2]
                elif 'left_ankle' in frame.landmark_positions:
                    left_z[i] = frame.landmark_positions['left_ankle'][2]

                if 'right_heel' in frame.landmark_positions:
                    right_z[i] = frame.landmark_positions['right_heel'][2]
                elif 'right_ankle' in frame.landmark_positions:
                    right_z[i] = frame.landmark_positions['right_ankle'][2]

        return left_z, right_z

    def _compute_dgei_curve(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute positive and negative DGEI curves.

        DGEIpos = somme des variations positives (montée du pied = swing)
        DGEIneg = somme des variations négatives (descente du pied = strike)

        Returns:
            dgei_pos, dgei_neg
        """
        n = len(signal)

        # First-order differences
        dy = np.diff(signal, prepend=signal[0])

        # Calculer les coefficients adaptatifs
        sigma_dy = np.std(dy) + 1e-10
        mu_dv = np.mean(np.abs(dy)) + 1e-10
        alpha = sigma_dy / (sigma_dy + mu_dv)
        beta = mu_dv / (sigma_dy + mu_dv)

        # DGEI with sliding window
        window_size = max(3, int(0.05 * self.fps))  # ~50ms

        dgei_pos = np.zeros(n)
        dgei_neg = np.zeros(n)

        for i in range(window_size, n):
            window = dy[i-window_size:i]

            # Positive differences (pied monte)
            pos_mask = window > self.bar_threshold
            if pos_mask.any():
                dgei_pos[i] = np.sum(alpha * window[pos_mask] + beta * window[pos_mask])

            # Negative differences (pied descend)
            neg_mask = window < -self.bar_threshold
            if neg_mask.any():
                dgei_neg[i] = np.sum(alpha * np.abs(window[neg_mask]) +
                                     beta * np.abs(window[neg_mask]))

        # Lisser les courbes
        dgei_pos = gaussian_filter1d(dgei_pos, sigma=2)
        dgei_neg = gaussian_filter1d(dgei_neg, sigma=2)

        return dgei_pos, dgei_neg

    def _detect_peaks_with_sleep(self, signal: np.ndarray,
                                  min_height_ratio: float = 0.3) -> List[int]:
        """
        Detect peaks with sleep-time mechanism.

        Args:
            signal: Signal DGEI
            min_height_ratio: Ratio minimum par rapport au max

        Returns:
            Liste des indices des pics
        """
        if len(signal) < 10:
            return []

        # Minimum threshold based on signal
        min_height = min_height_ratio * np.max(signal)

        # Detect all candidate peaks
        candidates, properties = find_peaks(
            signal,
            height=min_height,
            distance=int(0.2 * self.fps),  # Min 200ms entre pics
            prominence=min_height * 0.3
        )

        if len(candidates) == 0:
            return []

        # Appliquer le sleep time et seuil adaptatif
        peaks = []
        peak_values = []
        weights = np.arange(1, self.queue_length + 1)

        last_peak_frame = -self.sleep_frames - 1

        for idx in candidates:
            # Check the sleep time
            if idx - last_peak_frame < self.sleep_frames:
                continue

            peak_value = signal[idx]

            # Adaptive threshold based on previous peaks
            if len(peak_values) >= 2:
                recent_peaks = peak_values[-self.queue_length:]
                w = weights[:len(recent_peaks)]
                weighted_avg = np.sum(np.array(recent_peaks) * w) / np.sum(w)
                threshold = self.peak_ratio * weighted_avg

                if peak_value < threshold:
                    continue

            peaks.append(idx)
            peak_values.append(peak_value)
            last_peak_frame = idx

        return peaks

    def detect_gait_events(self, angle_frames) -> Tuple[List[GaitEvent], List[GaitEvent], dict]:
        """
        Detect gait events.

        Returns:
            heel_strikes, toe_offs, debug_data
        """
        n = len(angle_frames)
        if n < 30:
            return [], [], {'detector': 'DGEI', 'error': 'Too few frames'}

        # Extraire les signaux des pieds
        left_z, right_z = self._extract_foot_signals(angle_frames)

        # Calculer les courbes DGEI pour chaque pied
        left_pos, left_neg = self._compute_dgei_curve(left_z)
        right_pos, right_neg = self._compute_dgei_curve(right_z)

        # Detect events
        # HS = peaks in negative DGEI (foot descending toward the ground)
        # TO = pics dans DGEI positif (pied monte)

        left_hs_peaks = self._detect_peaks_with_sleep(left_neg)
        left_to_peaks = self._detect_peaks_with_sleep(left_pos)
        right_hs_peaks = self._detect_peaks_with_sleep(right_neg)
        right_to_peaks = self._detect_peaks_with_sleep(right_pos)

        # Create events
        heel_strikes = []
        for frame in left_hs_peaks:
            if 0 <= frame < n:
                heel_strikes.append(GaitEvent(
                    frame_index=frame,
                    time=frame / self.fps,
                    event_type='heel_strike',
                    side='left',
                    probability=float(left_neg[frame]) if frame < len(left_neg) else 0.5
                ))
        for frame in right_hs_peaks:
            if 0 <= frame < n:
                heel_strikes.append(GaitEvent(
                    frame_index=frame,
                    time=frame / self.fps,
                    event_type='heel_strike',
                    side='right',
                    probability=float(right_neg[frame]) if frame < len(right_neg) else 0.5
                ))
        heel_strikes.sort(key=lambda e: e.frame_index)

        toe_offs = []
        for frame in left_to_peaks:
            if 0 <= frame < n:
                toe_offs.append(GaitEvent(
                    frame_index=frame,
                    time=frame / self.fps,
                    event_type='toe_off',
                    side='left',
                    probability=float(left_pos[frame]) if frame < len(left_pos) else 0.5
                ))
        for frame in right_to_peaks:
            if 0 <= frame < n:
                toe_offs.append(GaitEvent(
                    frame_index=frame,
                    time=frame / self.fps,
                    event_type='toe_off',
                    side='right',
                    probability=float(right_pos[frame]) if frame < len(right_pos) else 0.5
                ))
        toe_offs.sort(key=lambda e: e.frame_index)

        debug_data = {
            'detector': 'DGEI',
            'n_left_hs': len(left_hs_peaks),
            'n_right_hs': len(right_hs_peaks),
            'n_left_to': len(left_to_peaks),
            'n_right_to': len(right_to_peaks),
        }

        return heel_strikes, toe_offs, debug_data
