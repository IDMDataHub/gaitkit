# -*- coding: utf-8 -*-
"""
IntellEvent Detector -- ONNX bidirectional-LSTM wrapper.

Reference
---------
Dumphart, B., Slijepcevic, D., Zeppelzauer, M., Breiteneder, C.,
Horsak, B., & Kranzl, A. (2023). Robust deep learning-based gait
event detection across various pathologies.
*PLOS ONE*, 18(8), e0288555.
https://doi.org/10.1371/journal.pone.0288555

Source code: https://github.com/fhstp/IntellEvent

Principle
---------
Marker velocities are resampled to 150 Hz, normalized, and fed through
two pre-trained bidirectional LSTM models (one for initial contact, one
for foot off).  The output probability is peak-detected, and the
laterality of each event is determined from the heel positions at the
detected frame.

CORRECTED IMPLEMENTATION (2026-02-07)
--------------------------------------
This version matches the original IntellEvent v2.0 source code exactly:
  1. Normalization: ONLY progression-axis channels (first 6 for IC,
     first 12 for FO), ONLY when initial frames of LHEE/RHEE are negative.
     Z channels are NEVER touched.
  2. Resampling: pandas DataFrame.resample() + interpolate(method='linear'),
     matching the original time-domain linear interpolation.
  3. Peak detection: find_peaks(height=0.2, distance=25), matching the
     original threshold-based detection.
  4. Progression axis: mean(abs()) comparison, matching the original.

Author: Frederic Fer (f.fer@institut-myologie.org)
Affiliation: Myodata, Institut de Myologie, Paris, France
License: MIT
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
import logging
import os
import tempfile
import urllib.error
import urllib.request
from scipy.signal import find_peaks
import pandas as pd

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

logger = logging.getLogger(__name__)

_INTELLEVENT_MODEL_URLS = {
    "ic_intellevent.onnx": [
        "https://raw.githubusercontent.com/IDMDataHub/gaitkit/master/python/src/gaitkit/data/ic_intellevent.onnx",
    ],
    "fo_intellevent.onnx": [
        "https://raw.githubusercontent.com/IDMDataHub/gaitkit/master/python/src/gaitkit/data/fo_intellevent.onnx",
    ],
}


def _looks_like_lfs_pointer(path: Path) -> bool:
    """Return True when a file appears to be a Git LFS pointer text file."""
    try:
        with open(path, "rb") as f:
            head = f.read(200)
    except OSError:
        return False
    return head.startswith(b"version https://git-lfs.github.com/spec/v1")


def _download_intellevent_model(target_path: Path, model_name: str) -> Optional[Path]:
    """Download one IntellEvent ONNX model to target_path."""
    urls = _INTELLEVENT_MODEL_URLS.get(model_name, [])
    if not urls:
        return None
    target_path.parent.mkdir(parents=True, exist_ok=True)

    for url in urls:
        logger.warning(
            "IntellEvent model '%s' missing/invalid. Downloading from: %s",
            model_name,
            url,
        )
        tmp_path = None
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                fd, tmp_name = tempfile.mkstemp(prefix=f"{model_name}_", suffix=".onnx")
                tmp_path = Path(tmp_name)
                with os.fdopen(fd, "wb") as out:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
            if _looks_like_lfs_pointer(tmp_path):
                raise RuntimeError("downloaded file is a Git LFS pointer")
            tmp_path.replace(target_path)
            logger.warning("IntellEvent model downloaded and cached at: %s", target_path)
            return target_path
        except (OSError, urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            logger.warning("Could not download IntellEvent model from %s: %s", url, exc)
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
    return None


@dataclass
class GaitEvent:
    """A single gait event."""
    frame_index: int
    time: float
    event_type: str  # 'heel_strike' or 'toe_off'
    side: str  # 'left' or 'right'
    probability: float = 1.0


@dataclass
class GaitCycle:
    """A complete gait cycle."""
    cycle_id: int
    side: str
    start_frame: int
    toe_off_frame: Optional[int]
    end_frame: int
    duration: float
    stance_percentage: Optional[float]


class IntellEventDetector:
    """
    IntellEvent detector based on bidirectional LSTM models.

    This implementation faithfully reproduces the preprocessing pipeline
    from the official IntellEvent v2.0 source code:
    https://github.com/fhstp/IntellEvent
    """

    TARGET_FPS = 150.0
    MIN_PEAK_THRESHOLD = 0.2
    MIN_PEAK_DISTANCE = 25  # frames at 150 Hz

    def __init__(self, fps: float = 100.0, models_dir: str = None):
        if fps <= 0:
            raise ValueError("fps must be strictly positive")
        self.fps = fps

        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime is required for IntellEventDetector")

        if models_dir is None:
            # Search: 1) bundled in gaitkit/data/  2) legacy models/ dir
            _data_dir = Path(__file__).parent.parent / 'data'
            _legacy_dir = Path(__file__).parent.parent.parent / 'models'
            if (_data_dir / 'ic_intellevent.onnx').exists():
                models_dir = _data_dir
            elif (_legacy_dir / 'ic_intellevent.onnx').exists():
                models_dir = _legacy_dir
            else:
                models_dir = _data_dir  # will raise below
        else:
            models_dir = Path(models_dir)

        ic_path = models_dir / 'ic_intellevent.onnx'
        fo_path = models_dir / 'fo_intellevent.onnx'

        if not ic_path.exists() or not fo_path.exists():
            _download_intellevent_model(ic_path, "ic_intellevent.onnx")
            _download_intellevent_model(fo_path, "fo_intellevent.onnx")
            if not ic_path.exists() or not fo_path.exists():
                raise FileNotFoundError(
                    f"IntellEvent ONNX models not found in {models_dir}. "
                    "Install with: pip install gaitkit[onnx]"
                )
        self.ic_session = self._load_onnx_session(ic_path, "IC")
        self.fo_session = self._load_onnx_session(fo_path, "FO")

    def _load_onnx_session(self, model_path: Path, model_name: str):
        """Load one ONNX session with actionable diagnostics on failure."""
        if _looks_like_lfs_pointer(model_path):
            model_file = model_path.name
            if _download_intellevent_model(model_path, model_file) is not None:
                return ort.InferenceSession(str(model_path))
            raise RuntimeError(
                f"IntellEvent {model_name} model at '{model_path}' is a Git LFS pointer, "
                "not the real ONNX binary. Reinstall gaitkit from PyPI with:\n"
                "  pip install --no-cache-dir --force-reinstall \"gaitkit[onnx]\""
            )
        try:
            return ort.InferenceSession(str(model_path))
        except Exception as exc:
            model_file = model_path.name
            if _download_intellevent_model(model_path, model_file) is not None:
                try:
                    return ort.InferenceSession(str(model_path))
                except Exception as retry_exc:
                    logger.debug(
                        "Retry loading IntellEvent model failed after re-download (%s): %s",
                        model_path,
                        retry_exc,
                    )
            size = model_path.stat().st_size if model_path.exists() else 0
            raise RuntimeError(
                f"Failed to load IntellEvent {model_name} model from '{model_path}' "
                f"(size={size} bytes). The file may be corrupted or incomplete "
                "(e.g., cloud sync/partial download). Reinstall gaitkit with:\n"
                "  pip uninstall -y gaitkit onnxruntime\n"
                "  pip cache purge\n"
                "  pip install --no-cache-dir --force-reinstall \"gaitkit[onnx]\""
            ) from exc

    def _get_marker_trajectories(self, angle_frames) -> Dict[str, np.ndarray]:
        """Extract marker trajectories from angle frames."""
        n = len(angle_frames)

        # Mapping to IntellEvent marker names -> our names (with fallbacks)
        markers = {
            'LHEE': ['left_heel', 'left_ankle'],
            'LTOE': ['left_toe', 'left_foot_index', 'left_ankle'],
            'LANK': ['left_ankle'],
            'RHEE': ['right_heel', 'right_ankle'],
            'RTOE': ['right_toe', 'right_foot_index', 'right_ankle'],
            'RANK': ['right_ankle']
        }

        trajectories = {}
        for intell_name, name_candidates in markers.items():
            traj = np.zeros((n, 3))
            for i, frame in enumerate(angle_frames):
                if frame.landmark_positions:
                    for candidate in name_candidates:
                        if candidate in frame.landmark_positions:
                            pos = frame.landmark_positions[candidate]
                            if pos[0] != 0 or pos[1] != 0:
                                traj[i] = pos
                                break
            trajectories[intell_name] = traj

        return trajectories

    def _detect_progression_axis(self, trajectories: Dict[str, np.ndarray]) -> str:
        """Detect the progression axis (X or Y).

        Matches original: np.mean(np.abs(prog_x)) > np.mean(np.abs(prog_y))
        """
        lhee = trajectories['LHEE']
        prog_x = lhee[:, 0]
        prog_y = lhee[:, 1]
        # Original uses mean(abs()) comparison
        if np.mean(np.abs(prog_x)) > np.mean(np.abs(prog_y)):
            return 'x'
        else:
            return 'y'

    def _prepare_trajectories(self, trajectories: Dict[str, np.ndarray],
                               progression_axis: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare position trajectories for IC and FO models.

        Following the original code exactly:
        IC: 12 channels = [prog_axis x 6 markers, z_axis x 6 markers]
        FO: 18 channels = [prog_axis x 6, other_axis x 6, z_axis x 6]

        Original code:
            if np.mean(np.abs(prog_x)) > np.mean(np.abs(prog_y)):
                ic_traj = np.concatenate([x_traj, z_traj])
                fo_traj = np.concatenate([x_traj, y_traj, z_traj])
            else:
                ic_traj = np.concatenate([y_traj, z_traj])
                fo_traj = np.concatenate([y_traj, x_traj, z_traj])
        """
        marker_order = ['LHEE', 'LTOE', 'LANK', 'RHEE', 'RTOE', 'RANK']

        # Build x_traj, y_traj, z_traj lists (each list of 6 1D arrays)
        x_traj = []
        y_traj = []
        z_traj = []
        for marker in marker_order:
            x_traj.append(trajectories[marker][:, 0])
            y_traj.append(trajectories[marker][:, 1])
            z_traj.append(trajectories[marker][:, 2])

        # Build prog_traj and other_traj from actual x/y based on progression axis
        if progression_axis == 'x':
            prog_traj = x_traj  # list of 6 arrays
            other_traj = y_traj
        else:
            prog_traj = y_traj
            other_traj = x_traj

        # IC: [progression x 6, z x 6] = 12 channels
        ic_traj = np.array(prog_traj + z_traj)  # shape (12, n)

        # FO: [progression x 6, other x 6, z x 6] = 18 channels
        fo_traj = np.array(prog_traj + other_traj + z_traj)  # shape (18, n)

        return ic_traj, fo_traj

    def _normalize_direction(self, ic_traj: np.ndarray,
                              fo_traj: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize walking direction, matching the original IntellEvent code exactly.

        Original code:
            if any(ic_traj[0, 0:10] < 0) or any(ic_traj[3, 0:10] < 0):
                ic_traj[0:6, :] = (ic_traj[0:6, :] - np.mean(ic_traj[0:6, :], axis=1).reshape(6,1)) * (-1)
                fo_traj[0:12, :] = (fo_traj[0:12, :] - np.mean(fo_traj[0:12, :], axis=1).reshape(12,1)) * (-1)

        Key points:
        - Only triggers when first 10 frames of LHEE or RHEE progression are negative
        - Only normalizes progression-axis channels (first 6 for IC, first 12 for FO)
        - Z channels (ic_traj[6:12] and fo_traj[12:18]) are NEVER modified
        - Centers each channel around its mean, then negates
        """
        ic_out = ic_traj.copy()
        fo_out = fo_traj.copy()

        # Check condition: any of first 10 frames of LHEE (row 0) or RHEE (row 3)
        # progression-axis values are negative
        n_check = min(10, ic_traj.shape[1])
        if np.any(ic_traj[0, 0:n_check] < 0) or np.any(ic_traj[3, 0:n_check] < 0):
            # Center and negate ONLY progression-axis channels
            # IC: channels 0-5 (progression axis for 6 markers)
            ic_out[0:6, :] = (ic_out[0:6, :] - np.mean(ic_out[0:6, :], axis=1).reshape(6, 1)) * (-1)
            # FO: channels 0-11 (progression + other axis for 6 markers each)
            fo_out[0:12, :] = (fo_out[0:12, :] - np.mean(fo_out[0:12, :], axis=1).reshape(12, 1)) * (-1)

        return ic_out, fo_out

    def _compute_velocities(self, traj: np.ndarray) -> np.ndarray:
        """Compute velocities (first derivative via gradient)."""
        return np.gradient(traj, axis=1)

    def _scale_velocities(self, velo: np.ndarray) -> np.ndarray:
        """Scale velocities to [0.1, 1.1] per feature.

        Equivalent to sklearn minmax_scale(..., feature_range=(0.1, 1.1), axis=1),
        implemented with NumPy to avoid a hard dependency on scikit-learn.
        """
        vmin = np.min(velo, axis=1, keepdims=True)
        vmax = np.max(velo, axis=1, keepdims=True)
        span = vmax - vmin
        safe_span = np.where(span == 0, 1.0, span)
        scaled = (velo - vmin) / safe_span
        return scaled + 0.1

    def _resample_to_target(self, data: np.ndarray) -> np.ndarray:
        """Resample to TARGET_FPS using pandas time-domain linear interpolation.

        Matches the original IntellEvent code exactly:
            period = '{}N'.format(int(1e9 / sample_frequ))
            index = pd.date_range(0, periods=len(traj[0, :]), freq=period)
            resampled = [pd.DataFrame(val, index=index)
                         .resample('{}N'.format(int(1e9 / frequ_to_sample))).mean()
                         for val in traj]
            resampled = [np.array(traj.interpolate(method='linear'))
                         for traj in resampled]
            resampled = np.concatenate(resampled, axis=1)
            return resampled  # note: returns transposed relative to input
        """
        if abs(self.fps - self.TARGET_FPS) < 1:
            return data

        n_channels, n_frames = data.shape
        if n_frames < 2:
            return data

        # Build DatetimeIndex at source frequency
        source_period_ns = int(1e9 / self.fps)
        source_index = pd.date_range(0, periods=n_frames,
                                      freq=f'{source_period_ns}ns')

        # Target period
        target_period_ns = int(1e9 / self.TARGET_FPS)
        target_freq = f'{target_period_ns}ns'

        # Resample each channel independently, matching original exactly
        resampled_channels = []
        for ch in range(n_channels):
            df_ch = pd.DataFrame(data[ch, :], index=source_index)
            # .resample().mean() bins the data, then .interpolate() fills NaN
            rs = df_ch.resample(target_freq).mean()
            rs = rs.interpolate(method='linear')
            resampled_channels.append(np.array(rs).flatten())

        # The original does np.concatenate(resampled, axis=1) which gives
        # shape (n_resampled_frames, n_channels). We need (n_channels, n_frames).
        # Stack and ensure correct shape
        result = np.array(resampled_channels)  # (n_channels, n_resampled_frames)

        # Handle any remaining NaN at edges
        if np.any(np.isnan(result)):
            # Forward-fill then backward-fill
            for ch in range(result.shape[0]):
                mask = np.isnan(result[ch])
                if mask.any():
                    valid = ~mask
                    if valid.any():
                        # Simple linear interpolation for edge NaN
                        xp = np.where(valid)[0]
                        fp = result[ch, valid]
                        result[ch] = np.interp(np.arange(result.shape[1]),
                                                xp, fp)
                    else:
                        result[ch] = 0.0

        return result

    def _reshape_for_model(self, data: np.ndarray) -> np.ndarray:
        """Reshape (features, frames) -> (1, frames, features).

        Original: np.transpose(np.array(traj).reshape(1, n_feat, n_frames), (0, 2, 1))
        """
        return np.transpose(
            data.reshape(1, data.shape[0], data.shape[1]),
            (0, 2, 1)
        ).astype(np.float32)

    def _determine_side_ic(self, frame_idx: int, l_heel_prog: np.ndarray,
                           r_heel_prog: np.ndarray) -> str:
        """Determine side for IC.

        Original: if l_heel[ic] < r_heel[ic]: Left Foot Strike
        At initial contact, the striking foot is the one reaching forward.
        After normalization, the interpretation depends on the direction convention.
        """
        if frame_idx >= len(l_heel_prog):
            frame_idx = len(l_heel_prog) - 1
        if l_heel_prog[frame_idx] < r_heel_prog[frame_idx]:
            return 'left'
        else:
            return 'right'

    def _determine_side_fo(self, frame_idx: int, l_heel_prog: np.ndarray,
                           r_heel_prog: np.ndarray) -> str:
        """Determine side for FO.

        Original: if l_heel[fo] > r_heel[fo]: Left Foot Off
        At foot off, the foot leaving is the one behind.
        """
        if frame_idx >= len(l_heel_prog):
            frame_idx = len(l_heel_prog) - 1
        if l_heel_prog[frame_idx] > r_heel_prog[frame_idx]:
            return 'left'
        else:
            return 'right'

    def detect_gait_events(self, angle_frames) -> Tuple[List[GaitEvent], List[GaitEvent], dict]:
        """Detect gait events using the IntellEvent pipeline.

        Follows the original code flow:
        1. Extract marker trajectories
        2. Detect progression axis
        3. Build IC (12ch) and FO (18ch) trajectory arrays
        4. Conditional direction normalization (progression axes only)
        5. Compute velocities (gradient)
        6. Scale to [0.1, 1.1]
        7. Resample to 150Hz (pandas time-domain interpolation)
        8. Reshape for model
        9. Run ONNX inference
        10. Peak detection (height=0.2, distance=25)
        11. Convert to original frame indices
        12. Determine laterality
        """
        n = len(angle_frames)
        if n < 30:
            return [], [], {'detector': 'IntellEvent', 'error': 'Too few frames'}

        # 1. Extract marker trajectories
        trajectories = self._get_marker_trajectories(angle_frames)

        # 2. Detect progression axis (mean(abs()) method)
        prog_axis = self._detect_progression_axis(trajectories)

        # 3. Build IC and FO trajectory arrays
        ic_traj, fo_traj = self._prepare_trajectories(trajectories, prog_axis)

        # 4. Conditional direction normalization
        # CRITICAL: Only normalizes progression-axis channels, NEVER Z channels
        ic_traj, fo_traj = self._normalize_direction(ic_traj, fo_traj)

        # Keep heel progression positions for laterality determination
        # Original passes ic_traj[0:6] (normalized progression positions) for this
        l_heel_prog = ic_traj[0, :]  # LHEE progression (after normalization)
        r_heel_prog = ic_traj[3, :]  # RHEE progression (after normalization)

        # 5. Compute velocities (gradient)
        ic_velo = self._compute_velocities(ic_traj)
        fo_velo = self._compute_velocities(fo_traj)

        # 6. Scale to [0.1, 1.1]
        ic_velo = self._scale_velocities(ic_velo)
        fo_velo = self._scale_velocities(fo_velo)

        # 7. Resample to 150Hz
        if abs(self.fps - self.TARGET_FPS) >= 1:
            ic_velo_rs = self._resample_to_target(ic_velo)
            fo_velo_rs = self._resample_to_target(fo_velo)
        else:
            ic_velo_rs = ic_velo
            fo_velo_rs = fo_velo

        # 8. Reshape for model input
        ic_input = self._reshape_for_model(ic_velo_rs)
        fo_input = self._reshape_for_model(fo_velo_rs)

        # 9. ONNX inference
        ic_preds = self.ic_session.run(None, {'input_1': ic_input})[0][0]  # (frames, 2)
        fo_preds = self.fo_session.run(None, {'input_1': fo_input})[0][0]

        # 10. Peak detection -- height=0.2, distance=25 (matching original exactly)
        ic_peaks, _ = find_peaks(
            ic_preds[:, 1],
            height=self.MIN_PEAK_THRESHOLD,
            distance=self.MIN_PEAK_DISTANCE
        )
        fo_peaks, _ = find_peaks(
            fo_preds[:, 1],
            height=self.MIN_PEAK_THRESHOLD,
            distance=self.MIN_PEAK_DISTANCE
        )

        # 11-12. Convert to original frame indices and determine laterality
        # Original: loc = np.ceil((loc / base_frequency) * cam_frequency).astype(int)
        cam_frequency = self.fps

        heel_strikes = []
        for peak_rs in ic_peaks:
            frame_orig = int(np.ceil((peak_rs / self.TARGET_FPS) * cam_frequency))
            if 0 <= frame_orig < n:
                side = self._determine_side_ic(frame_orig, l_heel_prog, r_heel_prog)
                prob = float(ic_preds[peak_rs, 1])
                heel_strikes.append(GaitEvent(
                    frame_index=frame_orig,
                    time=frame_orig / self.fps,
                    event_type='heel_strike',
                    side=side,
                    probability=prob
                ))

        toe_offs = []
        for peak_rs in fo_peaks:
            frame_orig = int(np.ceil((peak_rs / self.TARGET_FPS) * cam_frequency))
            if 0 <= frame_orig < n:
                side = self._determine_side_fo(frame_orig, l_heel_prog, r_heel_prog)
                prob = float(fo_preds[peak_rs, 1])
                toe_offs.append(GaitEvent(
                    frame_index=frame_orig,
                    time=frame_orig / self.fps,
                    event_type='toe_off',
                    side=side,
                    probability=prob
                ))

        heel_strikes.sort(key=lambda e: e.frame_index)
        toe_offs.sort(key=lambda e: e.frame_index)

        debug_data = {
            'detector': 'IntellEvent',
            'progression_axis': prog_axis,
            'n_ic_peaks': len(ic_peaks),
            'n_fo_peaks': len(fo_peaks),
            'ic_max_prob': float(ic_preds[:, 1].max()) if len(ic_preds) > 0 else 0,
            'fo_max_prob': float(fo_preds[:, 1].max()) if len(fo_preds) > 0 else 0,
        }

        return heel_strikes, toe_offs, debug_data
