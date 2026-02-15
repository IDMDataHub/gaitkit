"""
Vancanneyt Multi-Condition Gait Event Detector (MC7/MC2).

Reference
---------
Vancanneyt, T., Le Moal, C., Blard, M., Lenoir, J., Roche, N.,
Bonnyaud, C., & Dubois, F. (2025).
Automatic detection of human gait events: a simple but versatile 3D
algorithm.
*Journal of NeuroEngineering and Rehabilitation*, 22(1), 110.
https://doi.org/10.1186/s12984-025-01544-9

Source code: https://github.com/FDuRPC/GaitEvent_MultiCondition_algo

Principle
---------
The Multi-Condition algorithm identifies Foot Strike when the *first* of
the foot markers loses its degrees of freedom (i.e. all three velocity
components drop below speed-relative thresholds), and Foot Off when the
*last* of the foot markers regains its degrees of freedom.

Marker positions are low-pass filtered (4th-order Butterworth, 9 Hz
cut-off), then differentiated to obtain velocities. Three binary
threshold vectors (one per axis) are computed for each foot marker.
Foot Strike is the earliest frame across all markers where all three
velocity components are below threshold simultaneously. Foot Off is the
latest such frame.

Calibration parameters (optimised on 819 C3D files / 10 910 events):
    - Low-pass filter cut-off frequency: 9 Hz
    - VX threshold: 18% of mean AP walking speed (medio-lateral)
    - VY threshold: 31% of mean AP walking speed (antero-posterior)
    - VZ threshold:  8% of mean AP walking speed (vertical)
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass
from scipy.signal import butter, filtfilt, find_peaks

try:
    from .axis_utils import detect_axes
except ImportError:
    detect_axes = None


# ---------------------------------------------------------------------------
# Data classes (matching project convention)
# ---------------------------------------------------------------------------

@dataclass
class GaitEvent:
    """A single gait event."""
    frame_index: int
    time: float
    event_type: str   # 'heel_strike' or 'toe_off'
    side: str         # 'left' or 'right'
    confidence: float = 1.0


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


def _build_cycles(heel_strikes, toe_offs, fps):
    """Build gait cycles from detected events."""
    cycles = []
    for side in ('left', 'right'):
        side_hs = sorted(
            [e for e in heel_strikes if e.side == side],
            key=lambda x: x.frame_index)
        side_to = sorted(
            [e for e in toe_offs if e.side == side],
            key=lambda x: x.frame_index)
        for i in range(len(side_hs) - 1):
            start, end = side_hs[i], side_hs[i + 1]
            duration = (end.frame_index - start.frame_index) / fps
            to_in = next(
                (t for t in side_to
                 if start.frame_index < t.frame_index < end.frame_index),
                None)
            stance_pct = None
            if to_in is not None:
                stance_pct = (
                    (to_in.frame_index - start.frame_index) /
                    (end.frame_index - start.frame_index) * 100)
            cycles.append(GaitCycle(
                cycle_id=len(cycles), side=side,
                start_frame=start.frame_index,
                toe_off_frame=to_in.frame_index if to_in else None,
                end_frame=end.frame_index,
                duration=duration, stance_percentage=stance_pct))
    return sorted(cycles, key=lambda c: c.start_frame)


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------

class VancanneytDetector:
    """
    Multi-Condition gait event detector (Vancanneyt et al., 2025).

    Parameters
    ----------
    fps : float
        Sampling rate in Hz.
    filter_cutoff : float
        4th-order Butterworth low-pass cut-off frequency (Hz).
    vx_threshold : float
        Fraction of mean AP walking speed used as ML-velocity threshold.
    vy_threshold : float
        Fraction of mean AP walking speed used as AP-velocity threshold.
    vz_threshold : float
        Fraction of mean AP walking speed used as vertical-velocity threshold.
    windowing_mph_coeff : float
        Coefficient for MinPeakHeight in windowing (fraction of max |Vy|).
    windowing_mpd : int
        MinPeakDistance for windowing peak detection (frames).
    fo_frame_correction : int
        Frame offset added to Foot Off events (paper: +1 for Hallux
        marker position compensation). Set to 0 if no Hallux marker.
    """

    # Optimal calibration from parametric optimisation (6534 simulations)
    DEFAULT_FILTER_CUTOFF = 9.0    # Hz
    DEFAULT_VX_THRESH = 0.18       # 18% of mean AP speed (ML axis)
    DEFAULT_VY_THRESH = 0.31       # 31% of mean AP speed (AP axis)
    DEFAULT_VZ_THRESH = 0.08       #  8% of mean AP speed (vertical axis)

    def __init__(self, fps=100.0, filter_cutoff=None, vx_threshold=None,
                 vy_threshold=None, vz_threshold=None,
                 windowing_mph_coeff=0.5, windowing_mpd=50,
                 fo_frame_correction=0):
        if fps <= 0:
            raise ValueError("fps must be strictly positive")
        if filter_cutoff is not None and filter_cutoff <= 0:
            raise ValueError("filter_cutoff must be strictly positive")
        if vx_threshold is not None and vx_threshold < 0:
            raise ValueError("vx_threshold must be >= 0")
        if vy_threshold is not None and vy_threshold < 0:
            raise ValueError("vy_threshold must be >= 0")
        if vz_threshold is not None and vz_threshold < 0:
            raise ValueError("vz_threshold must be >= 0")
        if windowing_mph_coeff < 0:
            raise ValueError("windowing_mph_coeff must be >= 0")
        if windowing_mpd <= 0:
            raise ValueError("windowing_mpd must be strictly positive")
        self.fps = fps
        self.filter_cutoff = filter_cutoff if filter_cutoff is not None else self.DEFAULT_FILTER_CUTOFF
        self.vx_thresh = vx_threshold if vx_threshold is not None else self.DEFAULT_VX_THRESH
        self.vy_thresh = vy_threshold if vy_threshold is not None else self.DEFAULT_VY_THRESH
        self.vz_thresh = vz_threshold if vz_threshold is not None else self.DEFAULT_VZ_THRESH
        self.windowing_mph_coeff = windowing_mph_coeff
        self.windowing_mpd = windowing_mpd
        self.fo_frame_correction = fo_frame_correction

    # ------------------------------------------------------------------
    # Marker extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_position(frame, name):
        """Retrieve a marker position from an AngleFrame, or None."""
        if not frame.landmark_positions:
            return None
        pos = frame.landmark_positions.get(name)
        if pos is None:
            return None
        if pos[0] == 0.0 and pos[1] == 0.0 and pos[2] == 0.0:
            return None
        return pos

    def _extract_trajectories(self, angle_frames, marker_names):
        """Extract Nx3 trajectory arrays for each marker name."""
        n = len(angle_frames)
        trajectories = {}
        for name in marker_names:
            traj = np.full((n, 3), np.nan)
            for i, af in enumerate(angle_frames):
                pos = self._get_position(af, name)
                if pos is not None:
                    traj[i] = pos
            trajectories[name] = traj
        return trajectories

    def _detect_axes(self, angle_frames):
        """Detect AP and vertical axes using the utility or fallback."""
        if detect_axes is not None:
            try:
                ap_axis, vert_axis = detect_axes(angle_frames)
                return ap_axis, vert_axis
            except (TypeError, ValueError) as exc:
                logger.debug("Axis auto-detection failed, using fallback: %s", exc)
        # Fallback: infer from ankle range
        positions = []
        for af in angle_frames:
            if af.landmark_positions and 'left_ankle' in af.landmark_positions:
                pos = af.landmark_positions['left_ankle']
                if pos != (0.0, 0.0, 0.0):
                    positions.append(pos)
        if len(positions) < 10:
            return 1, 2  # default Y=AP, Z=vert
        positions = np.array(positions)
        ranges = np.ptp(positions, axis=0)
        ap_axis = int(np.argmax(ranges))
        remaining = [i for i in range(3) if i != ap_axis]
        vert_axis = 2 if 2 != ap_axis else remaining[
            int(np.argmin([ranges[i] for i in remaining]))]
        return ap_axis, vert_axis

    # ------------------------------------------------------------------
    # Signal processing helpers
    # ------------------------------------------------------------------

    def _interpolate_gaps(self, traj):
        """Fill NaN gaps in trajectory with linear interpolation."""
        result = traj.copy()
        for col in range(traj.shape[1]):
            y = traj[:, col]
            nans = np.isnan(y)
            if nans.all():
                result[:, col] = 0.0
                continue
            if nans.any():
                x = np.arange(len(y))
                result[:, col] = np.interp(x, x[~nans], y[~nans])
        return result

    def _lowpass_filter(self, data):
        """Apply 4th-order Butterworth low-pass filter."""
        nyquist = self.fps / 2.0
        if self.filter_cutoff >= nyquist:
            return data
        b, a = butter(4, self.filter_cutoff / nyquist, btype='low')
        min_len = 3 * max(len(a), len(b))
        if data.shape[0] < min_len:
            return data
        filtered = np.zeros_like(data)
        for col in range(data.shape[1]):
            filtered[:, col] = filtfilt(b, a, data[:, col])
        return filtered

    def _compute_velocities(self, positions):
        """Compute velocities by finite differences (matching MATLAB diff)."""
        dt = 1.0 / self.fps
        vel = np.diff(positions, axis=0) / dt
        return vel

    def _compute_mean_ap_speed(self, angle_frames, ap_axis):
        """Compute mean AP walking speed from hip/sacrum displacement.

        Following the MATLAB code:
            abs(SACR(end, Y) - SACR(1, Y)) / n_frames * fps
        """
        first_pos = None
        last_pos = None
        for af in angle_frames:
            if af.landmark_positions:
                lh = af.landmark_positions.get('left_hip')
                rh = af.landmark_positions.get('right_hip')
                if (lh and rh and lh != (0.0, 0.0, 0.0) and
                        rh != (0.0, 0.0, 0.0)):
                    sacr = [(lh[i] + rh[i]) / 2.0 for i in range(3)]
                    if first_pos is None:
                        first_pos = sacr
                    last_pos = sacr

        if first_pos is None or last_pos is None:
            # Fallback: use ankle
            for af in angle_frames:
                if af.landmark_positions:
                    la = af.landmark_positions.get('left_ankle')
                    if la and la != (0.0, 0.0, 0.0):
                        if first_pos is None:
                            first_pos = list(la)
                        last_pos = list(la)

        if first_pos is None or last_pos is None:
            return 1000.0  # fallback: assume 1 m/s in mm/s

        n = len(angle_frames)
        ap_disp = abs(last_pos[ap_axis] - first_pos[ap_axis])
        mean_speed = ap_disp / n * self.fps
        return max(mean_speed, 1.0)

    # ------------------------------------------------------------------
    # Windowing
    # ------------------------------------------------------------------

    def _compute_windows(self, vy):
        """Window the trial using AP velocity peaks (swing phase centres).

        Corresponds to MATLAB findpeaks on abs(Vy) with MinPeakHeight
        and MinPeakDistance.
        """
        abs_vy = np.abs(vy)
        max_abs_vy = np.max(abs_vy) if len(abs_vy) > 0 else 1.0
        mph = self.windowing_mph_coeff * max_abs_vy
        mpd = self.windowing_mpd

        peaks, _ = find_peaks(abs_vy, height=mph, distance=mpd)
        frames_win = peaks.tolist()

        if len(frames_win) == 0:
            return [0, len(vy) - 1]
        if frames_win[0] > 4:
            frames_win.insert(0, 0)
        if frames_win[-1] < len(vy) - 5:
            frames_win.append(len(vy) - 1)

        return frames_win

    # ------------------------------------------------------------------
    # Core detection for one side
    # ------------------------------------------------------------------

    def _detect_side_events(self, angle_frames, side, ap_axis, vert_axis,
                            mean_ap_speed):
        """Detect FS and FO frames for one side."""
        prefix = 'left' if side == 'left' else 'right'

        # The 7 markers from the paper: ANK, HEE, TOE, FMH, SMH, VMH, HLX
        # Map to whatever markers are available in our data
        marker_candidates = [
            (prefix + '_ankle',),                             # ANK
            (prefix + '_heel',),                              # HEE
            (prefix + '_toe', prefix + '_foot_index'),        # TOE
            (prefix + '_mt5',),                               # VMH
        ]

        marker_names = []
        sample_size = min(50, len(angle_frames))
        for candidates in marker_candidates:
            for cand in candidates:
                count = 0
                for af in angle_frames[:sample_size]:
                    if af.landmark_positions and cand in af.landmark_positions:
                        pos = af.landmark_positions[cand]
                        if pos != (0.0, 0.0, 0.0):
                            count += 1
                if count > 5:
                    marker_names.append(cand)
                    break

        if len(marker_names) == 0:
            return [], [], {'error': 'No foot markers found for ' + side}

        trajectories = self._extract_trajectories(angle_frames, marker_names)

        # Determine medio-lateral axis
        ml_axis = [i for i in range(3) if i != ap_axis and i != vert_axis][0]

        # Map thresholds to axes (MATLAB: VX=ML=18%, VY=AP=31%, VZ=vert=8%)
        threshold_map = {
            ml_axis:   self.vx_thresh * mean_ap_speed,
            ap_axis:   self.vy_thresh * mean_ap_speed,
            vert_axis: self.vz_thresh * mean_ap_speed,
        }

        # Process first marker (ANK equivalent) to get windowing
        first_marker = marker_names[0]
        first_traj = self._interpolate_gaps(trajectories[first_marker])
        first_traj_filt = self._lowpass_filter(first_traj)
        first_vel = self._compute_velocities(first_traj_filt)
        windows = self._compute_windows(first_vel[:, ap_axis])

        n_windows = len(windows) - 1
        if n_windows < 1:
            return [], [], {'error': 'Not enough windows'}

        all_marker_fs = []
        all_marker_fo = []

        for m_idx, m_name in enumerate(marker_names):
            traj = self._interpolate_gaps(trajectories[m_name])
            traj_filt = self._lowpass_filter(traj)
            vel = self._compute_velocities(traj_filt)
            n_vel = vel.shape[0]

            marker_fs = []
            marker_fo = []

            for w in range(n_windows):
                w_start = windows[w]
                w_end = windows[w + 1]
                w_start_v = max(0, w_start)
                w_end_v = min(n_vel - 1, w_end)

                if w_end_v <= w_start_v + 2:
                    marker_fs.append(np.nan)
                    marker_fo.append(np.nan)
                    continue

                # Build binary vectors following the MATLAB code exactly
                cross = {}
                for axis_idx in (ml_axis, ap_axis, vert_axis):
                    cvec = np.zeros(n_vel, dtype=int)
                    thresh = threshold_map[axis_idx]
                    for j in range(w_start_v + 1, w_end_v - 1):
                        if abs(vel[j, axis_idx]) < thresh:
                            # MATLAB sets j-1, j, j+1, j+2 to 1
                            for offset in (-1, 0, 1, 2):
                                idx = j + offset
                                if 0 <= idx < n_vel:
                                    cvec[idx] = 1
                        else:
                            if cvec[j] != 1:
                                cvec[j] = 0
                    cross[axis_idx] = cvec

                # Find frames where all three axes are below threshold
                overlap = []
                for k in range(w_start_v, min(w_end_v, n_vel)):
                    sigma = (cross[ml_axis][k] + cross[ap_axis][k] +
                             cross[vert_axis][k])
                    if sigma == 3:
                        overlap.append(k)

                if overlap:
                    marker_fs.append(overlap[0])
                    marker_fo.append(overlap[-1])
                else:
                    marker_fs.append(np.nan)
                    marker_fo.append(np.nan)

            all_marker_fs.append(marker_fs)
            all_marker_fo.append(marker_fo)

        # Convert to arrays: shape (n_windows, n_markers)
        n_markers = len(marker_names)
        fs_matrix = np.full((n_windows, n_markers), np.nan)
        fo_matrix = np.full((n_windows, n_markers), np.nan)
        for m in range(n_markers):
            for w in range(min(n_windows, len(all_marker_fs[m]))):
                fs_matrix[w, m] = all_marker_fs[m][w]
                fo_matrix[w, m] = all_marker_fo[m][w]

        # Remove edge windows (false positives from boundaries)
        if n_windows > 0 and not np.all(np.isnan(fs_matrix[0, :])):
            first_row_min = np.nanmin(fs_matrix[0, :])
            if not np.isnan(first_row_min) and first_row_min < 4:
                fs_matrix = fs_matrix[1:, :]
                fo_matrix = fo_matrix[1:, :]
                n_windows = fs_matrix.shape[0]
        if n_windows > 0 and not np.all(np.isnan(fo_matrix[-1, :])):
            n_vel_total = len(angle_frames) - 1
            last_row_max = np.nanmax(fo_matrix[-1, :])
            if not np.isnan(last_row_max) and last_row_max > n_vel_total - 5:
                fs_matrix = fs_matrix[:-1, :]
                fo_matrix = fo_matrix[:-1, :]
                n_windows = fs_matrix.shape[0]

        # FS = min across markers (first to lose DoF)
        # FO = max across markers (last to regain DoF)
        fs_frames = []
        fo_frames = []
        decisive_fs = []
        decisive_fo = []

        for w in range(n_windows):
            fs_row = fs_matrix[w, :]
            fo_row = fo_matrix[w, :]

            if np.all(np.isnan(fs_row)):
                continue
            fs_val = int(np.nanmin(fs_row))
            fs_idx = int(np.nanargmin(fs_row))
            fs_frames.append(fs_val)
            decisive_fs.append(marker_names[fs_idx])

            if np.all(np.isnan(fo_row)):
                continue
            fo_val = int(np.nanmax(fo_row))
            fo_idx = int(np.nanargmax(fo_row))
            fo_val += self.fo_frame_correction
            fo_frames.append(fo_val)
            decisive_fo.append(marker_names[fo_idx])

        debug = {
            'markers_used': marker_names,
            'n_windows': n_windows,
            'decisive_fs_markers': decisive_fs,
            'decisive_fo_markers': decisive_fo,
        }

        return fs_frames, fo_frames, debug

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def detect(self, angle_frames):
        """Detect gait events from angle frames.

        Parameters
        ----------
        angle_frames : list of AngleFrame
            Sequence of AngleFrame objects with ``landmark_positions``.

        Returns
        -------
        heel_strikes : list of GaitEvent
        toe_offs : list of GaitEvent
        cycles : list of GaitCycle
        """
        n = len(angle_frames)
        if n < 30:
            return [], [], []

        ap_axis, vert_axis = self._detect_axes(angle_frames)
        mean_ap_speed = self._compute_mean_ap_speed(angle_frames, ap_axis)

        heel_strikes = []
        toe_offs = []

        for side in ('left', 'right'):
            fs_frames, fo_frames, debug = self._detect_side_events(
                angle_frames, side, ap_axis, vert_axis, mean_ap_speed)

            for frame in fs_frames:
                if 0 <= frame < n:
                    heel_strikes.append(GaitEvent(
                        frame_index=int(frame),
                        time=frame / self.fps,
                        event_type='heel_strike',
                        side=side,
                        confidence=1.0))

            for frame in fo_frames:
                if 0 <= frame < n:
                    toe_offs.append(GaitEvent(
                        frame_index=int(frame),
                        time=frame / self.fps,
                        event_type='toe_off',
                        side=side,
                        confidence=1.0))

        # Post-processing: ensure proper event ordering (MATLAB reference)
        # FO is deleted if it begins the C3D (gait cycle starts with FS)
        for side in ('left', 'right'):
            contra = 'right' if side == 'left' else 'left'
            side_to = [e for e in toe_offs if e.side == side]
            contra_hs = [e for e in heel_strikes if e.side == contra]
            if side_to and contra_hs:
                first_contra_hs = min(e.frame_index for e in contra_hs)
                toe_offs = [e for e in toe_offs
                            if not (e.side == side and
                                    e.frame_index < first_contra_hs)]
            side_hs = [e for e in heel_strikes if e.side == side]
            if side_hs and side_to:
                last_hs = max(e.frame_index for e in side_hs)
                toe_offs = [e for e in toe_offs
                            if not (e.side == side and
                                    e.frame_index > last_hs)]

        heel_strikes.sort(key=lambda e: e.frame_index)
        toe_offs.sort(key=lambda e: e.frame_index)

        cycles = _build_cycles(heel_strikes, toe_offs, self.fps)

        return heel_strikes, toe_offs, cycles

    # Alias for uniform benchmark interface
    detect_gait_events = detect
