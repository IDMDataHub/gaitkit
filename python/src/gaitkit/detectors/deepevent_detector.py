"""
DeepEvent Gait Event Detector.

Reference
---------
Lempereur, M., Rousseau, F., Brochard, S., & Remy-Neris, O. (2020).
A new deep learning-based method for the detection of gait events in
children with gait disorders: Proof-of-concept and concurrent validity.
*Journal of Biomechanics*, 98, 109490.
https://doi.org/10.1016/j.jbiomech.2019.109490

GitHub: https://github.com/LempereurMat/deepevent

Principle
---------
A bidirectional LSTM network (3 layers, 800 units each) processes
3-D marker trajectories and their velocities for 6 foot/ankle markers
(LANK, RANK, LTOE, RTOE, LHEE, RHEE).

Input:  (1, 1536, 36)  -- positions (18) + velocities (18)
Output: (1, 1536, 5)   -- softmax over {no_event, LFS, RFS, LFO, RFO}

Gait events are extracted by thresholding the output probabilities and
selecting local maxima.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import argrelextrema, butter, filtfilt

from .axis_utils import detect_axes

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes (matching the project convention)
# ---------------------------------------------------------------------------

@dataclass
class GaitEvent:
    """A single gait event (heel strike or toe off)."""
    frame_index: int
    time: float
    event_type: str
    side: str
    confidence: float = 1.0


@dataclass
class GaitCycle:
    """A complete gait cycle delimited by two consecutive heel strikes."""
    cycle_id: int
    side: str
    start_frame: int
    toe_off_frame: Optional[int]
    end_frame: int
    duration: float
    stance_percentage: Optional[float]


def _build_cycles(heel_strikes: List[GaitEvent], toe_offs: List[GaitEvent],
                  fps: float) -> List[GaitCycle]:
    """Build gait cycles from detected heel strikes and toe offs."""
    cycles = []
    for side in ['left', 'right']:
        side_hs = sorted([e for e in heel_strikes if e.side == side],
                         key=lambda x: x.frame_index)
        side_to = sorted([e for e in toe_offs if e.side == side],
                         key=lambda x: x.frame_index)
        for i in range(len(side_hs) - 1):
            start, end = side_hs[i], side_hs[i + 1]
            duration = (end.frame_index - start.frame_index) / fps
            to_in = next((t for t in side_to
                          if start.frame_index < t.frame_index < end.frame_index), None)
            stance_pct = None
            if to_in:
                stance_pct = ((to_in.frame_index - start.frame_index) /
                              (end.frame_index - start.frame_index) * 100)
            cycles.append(GaitCycle(
                cycle_id=len(cycles), side=side,
                start_frame=start.frame_index,
                toe_off_frame=to_in.frame_index if to_in else None,
                end_frame=end.frame_index,
                duration=duration, stance_percentage=stance_pct))
    return sorted(cycles, key=lambda c: c.start_frame)


# ---------------------------------------------------------------------------
# Signal preprocessing (from DeepEvent's utils.py, adapted for raw arrays)
# ---------------------------------------------------------------------------

def _butter_lowpass_filter(data: np.ndarray, cutoff: float, fs: float,
                           order: int = 4) -> np.ndarray:
    """Apply a zero-phase Butterworth low-pass filter.

    Replicates the ``filter`` function in DeepEvent's utils.py.
    """
    nyq = fs / 2.0
    normal_cutoff = cutoff / nyq
    if normal_cutoff >= 1.0:
        normal_cutoff = 0.99
    b, a = butter(order, normal_cutoff, btype='low')
    mean_val = np.mean(data, axis=0)
    centered = data - mean_val
    filtered = filtfilt(b, a, centered, axis=0)
    return filtered + mean_val


def _central_difference_velocity(positions: np.ndarray, n_valid: int,
                                 freq: float) -> np.ndarray:
    """Compute velocity via central differences.

    Replicates ``derive_centre`` from DeepEvent's utils.py.
    """
    vel = np.zeros_like(positions)
    if n_valid > 2:
        vel[1:n_valid - 1, :] = (
            (positions[2:n_valid, :] - positions[0:n_valid - 2, :]) / (2.0 / freq)
        )
    return vel


# ---------------------------------------------------------------------------
# Progression frame detection (adapted from DeepEvent's utils.py)
# ---------------------------------------------------------------------------

def _detect_progression_frame(ankle_positions: np.ndarray):
    """Detect the global coordinate frame and forward progression direction.

    Parameters
    ----------
    ankle_positions : np.ndarray, shape (N, 3)
        Left ankle trajectory.

    Returns
    -------
    rotation_matrix : np.ndarray, shape (3, 3)
    forward_progression : bool
    """
    diff_x = ankle_positions[-1, 0] - ankle_positions[0, 0]
    diff_y = ankle_positions[-1, 1] - ankle_positions[0, 1]
    abs_diffs = [abs(diff_x), abs(diff_y)]
    ind = int(np.argmax(abs_diffs))

    if ind == 0:
        # X is the progression axis
        rot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        forward = diff_x > 0
    else:
        # Y is the progression axis => rotate so X becomes progression
        rot = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=float)
        forward = diff_y > 0

    return rot, forward


def _apply_rotation(positions: np.ndarray, rot: np.ndarray,
                    forward: bool) -> np.ndarray:
    """Rotate marker positions into the canonical frame.

    Replicates ``applyRotation`` from DeepEvent's utils.py.
    """
    rotated = positions @ rot.T
    if not forward:
        flip = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        rotated = rotated @ flip.T
    return rotated


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

_MODEL_CACHE = {}  # singleton cache for the loaded model


def _build_deepevent_model(weights_path: Optional[str] = None):
    """Build (or load from cache) the DeepEvent Bi-LSTM model.

    The architecture is reconstructed from the original JSON to avoid
    Keras version incompatibilities.

    Architecture (Lempereur et al. 2020):
        Input  -> (batch, 1536, 36)
        BiLSTM(800) -> Dropout(0.2) ->
        BiLSTM(800) -> Dropout(0.2) ->
        BiLSTM(800) -> Dropout(0.2) ->
        TimeDistributed(Dense(5, sigmoid))
        Output -> (batch, 1536, 5)
    """
    cache_key = weights_path or '__no_weights__'
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    # Try tf_keras first (Keras 2 API, for loading old .h5 weights)
    # then fall back to keras 3
    try:
        os.environ['TF_USE_LEGACY_KERAS'] = '1'
        from tf_keras.models import Sequential, model_from_json
        from tf_keras.layers import (Bidirectional, LSTM, Dropout,
                                     TimeDistributed, Dense)
        use_tf_keras = True
    except ImportError:
        from keras.models import Sequential
        from keras.layers import (Bidirectional, LSTM, Dropout,
                                  TimeDistributed, Dense)
        use_tf_keras = False

    NFRAMES = 1536
    NB_FEATURES = 36  # 6 markers * 3 axes * 2 (pos + vel)

    # Try loading from the original JSON first (works with tf_keras)
    json_paths = [
        Path(__file__).parent.parent.parent.parent / 'gait_benchmark_project' / 'deepevent' / 'deepevent' / 'data' / 'DeepEventModel.json',
        Path.home() / 'gait_benchmark_project' / 'deepevent' / 'deepevent' / 'data' / 'DeepEventModel.json',
    ]

    model = None
    if use_tf_keras:
        for jp in json_paths:
            if jp.exists():
                try:
                    with open(jp) as f:
                        model_json = f.read()
                    model = model_from_json(model_json)
                    logger.info(f"Loaded DeepEvent architecture from {jp}")
                    break
                except Exception as e:
                    logger.warning(f"Could not load JSON from {jp}: {e}")

    # Fallback: build programmatically
    if model is None:
        logger.info("Building DeepEvent model architecture programmatically")
        model = Sequential([
            Bidirectional(
                LSTM(800, return_sequences=True),
                input_shape=(NFRAMES, NB_FEATURES),
                merge_mode='concat'
            ),
            Dropout(0.2),
            Bidirectional(
                LSTM(800, return_sequences=True),
                merge_mode='concat'
            ),
            Dropout(0.2),
            Bidirectional(
                LSTM(800, return_sequences=True),
                merge_mode='concat'
            ),
            Dropout(0.2),
            TimeDistributed(Dense(5, activation='sigmoid')),
        ])

    # Try loading weights
    weights_loaded = False
    if weights_path and Path(weights_path).exists():
        try:
            model.load_weights(weights_path)
            weights_loaded = True
            logger.info(f"Loaded DeepEvent weights from {weights_path}")
        except Exception as e:
            logger.warning(f"Could not load weights from {weights_path}: {e}")

    if not weights_loaded:
        # Search standard locations for the weights file
        weight_paths = [
            Path(__file__).parent.parent.parent.parent / 'gait_benchmark_project' / 'deepevent' / 'deepevent' / 'data' / 'DeepEventWeight.h5',
            Path.home() / 'gait_benchmark_project' / 'deepevent' / 'deepevent' / 'data' / 'DeepEventWeight.h5',
        ]
        # Also check the installed deepevent package
        try:
            import deepevent as de_pkg
            pkg_weight = Path(de_pkg.__file__).parent / 'data' / 'DeepEventWeight.h5'
            weight_paths.append(pkg_weight)
        except ImportError:
            pass

        for wp in weight_paths:
            if wp.exists() and wp.stat().st_size > 10000:
                try:
                    model.load_weights(str(wp))
                    weights_loaded = True
                    logger.info(f"Loaded DeepEvent weights from {wp}")
                    break
                except Exception as e:
                    logger.warning(f"Could not load weights from {wp}: {e}")

    if not weights_loaded:
        logger.warning(
            "DeepEvent weights not found. The model will run with random "
            "weights. Download the weights file (DeepEventWeight.h5) from "
            "https://github.com/LempereurMat/deepevent and place it in "
            "~/gait_benchmark_project/deepevent/deepevent/data/"
        )

    _MODEL_CACHE[cache_key] = model
    return model


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------

class DeepEventDetector:
    """DeepEvent gait event detector (Lempereur et al., 2020).

    A bidirectional LSTM network that detects gait events from 6-marker
    foot/ankle trajectories (LANK, RANK, LTOE, RTOE, LHEE, RHEE).

    Parameters
    ----------
    fps : float
        Sampling rate in Hz.
    weights_path : str or None
        Path to the .h5 weights file.  Searched automatically if None.
    event_threshold : float
        Probability threshold for event detection (0--1).
    filter_cutoff : float
        Butterworth low-pass filter cutoff frequency in Hz.
    """

    # Marker names expected by DeepEvent (Vicon Plug-in-Gait convention)
    # and their mapping to the landmark_positions keys used in this project.
    DEEPEVENT_MARKERS = ['LANK', 'RANK', 'LTOE', 'RTOE', 'LHEE', 'RHEE']
    LANDMARK_MAPPING = {
        'LANK': ['left_ankle'],
        'RANK': ['right_ankle'],
        'LTOE': ['left_toe', 'left_foot_index'],
        'RTOE': ['right_toe', 'right_foot_index'],
        'LHEE': ['left_heel'],
        'RHEE': ['right_heel'],
    }

    NFRAMES = 1536  # Fixed input length for the model

    def __init__(self, fps: float = 100.0,
                 weights_path: Optional[str] = None,
                 event_threshold: float = 0.01,
                 filter_cutoff: float = 6.0):
        self.fps = fps
        self.weights_path = weights_path
        self.event_threshold = event_threshold
        self.filter_cutoff = filter_cutoff
        self._model = None

    def _get_model(self):
        """Lazy-load the model."""
        if self._model is None:
            self._model = _build_deepevent_model(self.weights_path)
        return self._model

    def _extract_marker_positions(self, angle_frames) -> Dict[str, np.ndarray]:
        """Extract 6 marker trajectories from AngleFrame list.

        Returns
        -------
        markers : dict
            ``{marker_name: np.ndarray of shape (n_frames, 3)}``
        """
        n = len(angle_frames)
        markers = {}
        for de_name, candidates in self.LANDMARK_MAPPING.items():
            positions = np.zeros((n, 3))
            for i, af in enumerate(angle_frames):
                if af.landmark_positions:
                    for cand in candidates:
                        if cand in af.landmark_positions:
                            pos = af.landmark_positions[cand]
                            positions[i, :] = pos
                            break
            markers[de_name] = positions
        return markers

    def _preprocess(self, markers: Dict[str, np.ndarray],
                    n_valid: int) -> np.ndarray:
        """Build the (1, 1536, 36) input tensor.

        Processing follows DeepEvent's original pipeline:
        1. Detect progression axis and rotate all markers
        2. Low-pass filter at ``filter_cutoff`` Hz
        3. Compute central-difference velocity
        4. Zero-pad to 1536 frames
        5. Concatenate positions (18 channels) and velocities (18 channels)
        """
        # 1. Detect progression frame and rotate
        lank = markers['LANK'][:n_valid]
        if np.ptp(lank, axis=0).max() > 1e-3:
            rot, forward = _detect_progression_frame(lank)
        else:
            rot = np.eye(3)
            forward = True

        rotated = {}
        for name, pos in markers.items():
            rotated[name] = _apply_rotation(pos[:n_valid], rot, forward)

        # 2. Filter + 3. Velocity
        ordered_markers = self.DEEPEVENT_MARKERS
        nb_markers = len(ordered_markers)

        inputs = np.zeros((1, self.NFRAMES, nb_markers * 3 * 2))

        for k, mname in enumerate(ordered_markers):
            raw = rotated[mname]  # (n_valid, 3)
            # Apply low-pass filter
            if n_valid > 10:
                filtered = _butter_lowpass_filter(raw, self.filter_cutoff, self.fps)
            else:
                filtered = raw

            # Positions: channels [k*3 : (k+1)*3]
            inputs[0, :n_valid, k * 3:(k + 1) * 3] = filtered

            # Velocities: channels [nb_markers*3 + k*3 : nb_markers*3 + (k+1)*3]
            vel = _central_difference_velocity(
                inputs[0, :, k * 3:(k + 1) * 3], n_valid, self.fps
            )
            offset = nb_markers * 3
            inputs[0, :, offset + k * 3:offset + (k + 1) * 3] = vel

        return inputs

    def _postprocess(self, predicted: np.ndarray,
                     n_valid: int) -> Tuple[List[int], List[int], List[int], List[int]]:
        """Extract event frame indices from the model output.

        Parameters
        ----------
        predicted : np.ndarray, shape (1, 1536, 5)
            Channel mapping: 0=no_event, 1=LFS, 2=RFS, 3=LFO, 4=RFO
        n_valid : int
            Number of actual frames (rest is zero-padded).

        Returns
        -------
        lfs, rfs, lfo, rfo : list of int
            Frame indices for each event type.
        """
        nframes = min(n_valid, predicted.shape[1])

        # Apply threshold
        pred = predicted[0, :nframes, :].copy()
        for ch in range(1, 5):
            pred[pred[:, ch] <= self.event_threshold, ch] = 0

        # Find local maxima in each event channel
        event_frames = {}
        for ch, label in [(1, 'LFS'), (2, 'RFS'), (3, 'LFO'), (4, 'RFO')]:
            maxima = argrelextrema(pred[:, ch], np.greater)[0]
            # Filter: keep only where the value is positive
            maxima = [int(m) for m in maxima if pred[m, ch] > 0]
            event_frames[label] = maxima

        return (event_frames['LFS'], event_frames['RFS'],
                event_frames['LFO'], event_frames['RFO'])

    def detect(self, angle_frames) -> Tuple[List[GaitEvent], List[GaitEvent], List[GaitCycle]]:
        """Detect gait events using the DeepEvent model.

        Parameters
        ----------
        angle_frames : list
            Sequence of AngleFrame objects with ``landmark_positions``.

        Returns
        -------
        heel_strikes : list of GaitEvent
        toe_offs : list of GaitEvent
        cycles : list of GaitCycle
        """
        n = len(angle_frames)
        if n < 10:
            return [], [], []

        model = self._get_model()

        # Extract marker positions
        markers = self._extract_marker_positions(angle_frames)
        n_valid = min(n, self.NFRAMES)

        # For sequences longer than 1536, process in overlapping windows
        if n > self.NFRAMES:
            return self._detect_long_sequence(angle_frames, markers, model)

        # Preprocess
        inputs = self._preprocess(markers, n_valid)

        # Predict
        predicted = model.predict(inputs, verbose=0)

        # Postprocess
        lfs, rfs, lfo, rfo = self._postprocess(predicted, n_valid)

        # Convert to GaitEvent objects
        heel_strikes = sorted(
            [GaitEvent(f, f / self.fps, 'heel_strike', 'left') for f in lfs] +
            [GaitEvent(f, f / self.fps, 'heel_strike', 'right') for f in rfs],
            key=lambda e: e.frame_index
        )
        toe_offs = sorted(
            [GaitEvent(f, f / self.fps, 'toe_off', 'left') for f in lfo] +
            [GaitEvent(f, f / self.fps, 'toe_off', 'right') for f in rfo],
            key=lambda e: e.frame_index
        )

        cycles = _build_cycles(heel_strikes, toe_offs, self.fps)
        return heel_strikes, toe_offs, cycles

    def _detect_long_sequence(self, angle_frames, markers, model):
        """Handle sequences longer than 1536 frames using overlapping windows."""
        n = len(angle_frames)
        stride = self.NFRAMES // 2  # 50% overlap
        all_lfs, all_rfs, all_lfo, all_rfo = [], [], [], []

        for start in range(0, n, stride):
            end = min(start + self.NFRAMES, n)
            n_valid = end - start

            # Build markers dict for this window
            window_markers = {}
            for name, pos in markers.items():
                window_markers[name] = pos[start:end]

            inputs = self._preprocess(window_markers, n_valid)
            predicted = model.predict(inputs, verbose=0)
            lfs, rfs, lfo, rfo = self._postprocess(predicted, n_valid)

            # Offset frame indices to global coordinates
            all_lfs.extend([f + start for f in lfs])
            all_rfs.extend([f + start for f in rfs])
            all_lfo.extend([f + start for f in lfo])
            all_rfo.extend([f + start for f in rfo])

            if end >= n:
                break

        # Remove duplicate events from overlap regions (keep unique within
        # a tolerance of 5 frames)
        def _deduplicate(frames, tolerance=5):
            if not frames:
                return frames
            frames = sorted(set(frames))
            deduped = [frames[0]]
            for f in frames[1:]:
                if f - deduped[-1] > tolerance:
                    deduped.append(f)
            return deduped

        all_lfs = _deduplicate(all_lfs)
        all_rfs = _deduplicate(all_rfs)
        all_lfo = _deduplicate(all_lfo)
        all_rfo = _deduplicate(all_rfo)

        heel_strikes = sorted(
            [GaitEvent(f, f / self.fps, 'heel_strike', 'left') for f in all_lfs] +
            [GaitEvent(f, f / self.fps, 'heel_strike', 'right') for f in all_rfs],
            key=lambda e: e.frame_index
        )
        toe_offs = sorted(
            [GaitEvent(f, f / self.fps, 'toe_off', 'left') for f in all_lfo] +
            [GaitEvent(f, f / self.fps, 'toe_off', 'right') for f in all_rfo],
            key=lambda e: e.frame_index
        )

        cycles = _build_cycles(heel_strikes, toe_offs, self.fps)
        return heel_strikes, toe_offs, cycles

    detect_gait_events = detect
