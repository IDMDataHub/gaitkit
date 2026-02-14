"""
Camargo Georgia Tech Extractor -- Able-bodied gait on multiple terrains.

Dataset: An open dataset of inertial, magnetic, foot-pressure, and gait data
         during typical daily activities (Camargo et al., Georgia Tech).
Source:  https://doi.org/10.6084/m9.figshare.c.4892463

Content:
- ~20 able-bodied subjects (AB01-AB20) across multiple data collection parts
- Activities: levelground, treadmill, ramp, stair (we use levelground + treadmill)
- Ground truth gait events derived from gait cycle percentage signals
  (gcLeft / gcRight .mat files with HeelStrike and ToeOff columns, 0-100 %)
- 200 Hz optical motion capture (markers)
- Coordinates in mm
- 28 markers with 3 axes each (85 columns including Header)

Directory structure:
    camargo_gatech/{partN}/AB{nn}/{date}/{activity}/{datatype}/filename.mat
    e.g. part3/AB06/2019-03-12/levelground/markers/Levelground_01_00_01_00.mat

Data types used:
    markers/   - 3-D marker trajectories (28 markers x 3 axes)
    gcLeft/    - Left gait cycle % (HeelStrike, ToeOff)
    gcRight/   - Right gait cycle % (HeelStrike, ToeOff)

Dependency: matio  (pip install git+https://github.com/foreverallama/matio)
"""

import logging
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from .base_extractor import (
    BaseExtractor, ExtractionResult, AngleFrame, GroundTruth,
    compute_angle_from_3points
)

logger = logging.getLogger(__name__)

try:
    from matio import load_from_mat
    HAS_MATIO = True
except ImportError:
    HAS_MATIO = False
    logger.warning("matio not installed. Install with: pip install git+https://github.com/foreverallama/matio")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLING_RATE = 200  # Hz

# Activities we consider as walking
WALKING_ACTIVITIES = {'levelground', 'treadmill'}

# Mapping from Camargo marker names -> standard landmark names used by the
# benchmark framework.  The marker DataFrame columns follow the pattern
# {MarkerName}_{axis} where axis is x, y, or z.
MARKER_TO_LANDMARK = {
    'L_Ankle_Lat': 'left_ankle',
    'R_Ankle_Lat': 'right_ankle',
    'L_Heel':      'left_heel',
    'R_Heel':      'right_heel',
    'L_Toe_Tip':   'left_toe',
    'R_Toe_Tip':   'right_toe',
    'L_ASIS':      'left_asis',
    'R_ASIS':      'right_asis',
    'L_PSIS':      'left_psis',
    'R_PSIS':      'right_psis',
    'L_Knee_Lat':  'left_knee',
    'R_Knee_Lat':  'right_knee',
    'L_Toe_Med':   'left_toe_med',
    'R_Toe_Med':   'right_toe_med',
    'L_Toe_Lat':   'left_toe_lat',
    'R_Toe_Lat':   'right_toe_lat',
    'L_Shank_Upper': 'left_tibia',
    'R_Shank_Upper': 'right_tibia',
    'L_Thigh_Upper': 'left_thigh',
    'R_Thigh_Upper': 'right_thigh',
}


class CamargoExtractor(BaseExtractor):
    """Extractor for the Camargo Georgia Tech gait dataset (MAT files)."""

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = str(Path.home() / 'gait_benchmark_project/data/camargo_gatech')
        super().__init__(data_dir)

    # ------------------------------------------------------------------
    # BaseExtractor interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "Camargo Georgia Tech"

    @property
    def description(self) -> str:
        return (
            "Able-bodied subjects, level-ground and treadmill walking, "
            "200 Hz optical markers, Georgia Tech"
        )

    def list_files(self) -> List[Path]:
        """Return a list of *marker* .mat files for walking trials.

        We use the markers directory as the canonical list of trials;
        the corresponding gcLeft / gcRight files are located by
        swapping the ``markers`` path component.
        """
        if not HAS_MATIO:
            return []

        marker_files: List[Path] = []
        for activity in WALKING_ACTIVITIES:
            # Glob across all parts / subjects / dates
            pattern = f'*/AB*/*/{activity}/markers/*.mat'
            marker_files.extend(self.data_dir.glob(pattern))

        return sorted(marker_files)

    def extract_file(self, filepath: Path) -> ExtractionResult:
        """Extract data from a single trial (identified by its markers file).

        Parameters
        ----------
        filepath : Path
            Path to the markers .mat file.

        Returns
        -------
        ExtractionResult
        """
        if not HAS_MATIO:
            raise RuntimeError("matio not installed")

        file_info = self._parse_filepath(filepath)

        # ----- Load marker data ------------------------------------------------
        markers_df = self._load_mat_dataframe(filepath)
        if markers_df is None:
            raise RuntimeError(f"Could not load markers from {filepath}")

        n_frames = len(markers_df)
        fps = SAMPLING_RATE
        header = markers_df['Header'].values
        duration_s = header[-1] - header[0] if n_frames > 1 else 0.0

        # ----- Load gait cycle files ------------------------------------------
        gc_left_path = self._gc_path(filepath, 'gcLeft')
        gc_right_path = self._gc_path(filepath, 'gcRight')

        gc_left_df = self._load_mat_dataframe(gc_left_path)
        gc_right_df = self._load_mat_dataframe(gc_right_path)

        # ----- Extract gait events --------------------------------------------
        hs_left_frames, to_left_frames = self._extract_gait_events(
            gc_left_df, header, 'left', filepath
        )
        hs_right_frames, to_right_frames = self._extract_gait_events(
            gc_right_df, header, 'right', filepath
        )

        # ----- Build AngleFrame list ------------------------------------------
        angle_frames = self._build_angle_frames(markers_df)

        # ----- Fix: TO double-counting on treadmill -------------------------
        # In treadmill data, gcLeft/ToeOff == gcRight/ToeOff (identical signals).
        # Combining both would double the GT count. Detect and deduplicate.
        if (len(to_left_frames) > 2 and len(to_right_frames) > 2 and
                len(to_left_frames) == len(to_right_frames)):
            # Check if TO events from both sides are identical (within 2 frames)
            diffs = [abs(a - b) for a, b in zip(to_left_frames, to_right_frames)]
            if max(diffs) <= 2:
                # Identical: keep only one set, split by alternation with HS
                all_to = sorted(set(to_left_frames))
                # Assign L/R based on which side has the closest HS before
                to_l_new, to_r_new = [], []
                for tf in all_to:
                    # Find closest preceding HS on each side
                    dl = min([tf - h for h in hs_left_frames if h < tf], default=9999)
                    dr = min([tf - h for h in hs_right_frames if h < tf], default=9999)
                    if dl < dr:
                        to_l_new.append(tf)
                    else:
                        to_r_new.append(tf)
                to_left_frames = sorted(to_l_new)
                to_right_frames = sorted(to_r_new)

        # ----- Ground truth ---------------------------------------------------
        has_hs = len(hs_left_frames) > 0 or len(hs_right_frames) > 0
        has_to = len(to_left_frames) > 0 or len(to_right_frames) > 0

        gt = GroundTruth(
            has_hs=has_hs,
            has_to=has_to,
            has_cadence=has_hs,
            has_angles=True,
            has_forces=False,
            hs_frames={'left': hs_left_frames, 'right': hs_right_frames},
            to_frames={'left': to_left_frames, 'right': to_right_frames},
        )

        # Cadence from combined HS events
        if has_hs:
            all_hs = sorted(hs_left_frames + hs_right_frames)
            if len(all_hs) > 1:
                intervals = np.diff(all_hs) / fps
                mean_interval = np.mean(intervals)
                if mean_interval > 0:
                    gt.cadence = 60.0 / mean_interval

        # ----- valid_frame_range: exclude non-walking phases ------------------
        # Levelground trials contain standing/turning at start/end.
        # Restrict evaluation to the frame range spanned by GT events.
        all_gt_frames = sorted(
            hs_left_frames + hs_right_frames +
            to_left_frames + to_right_frames
        )
        if len(all_gt_frames) > 2:
            margin = int(0.5 * fps)  # 0.5s margin
            gt.valid_frame_range = (
                max(0, min(all_gt_frames) - margin),
                min(n_frames - 1, max(all_gt_frames) + margin),
            )

        # ----- Warnings / quality ---------------------------------------------
        warnings: List[str] = []
        if gc_left_df is None:
            warnings.append(f"Missing gcLeft file: {gc_left_path}")
        if gc_right_df is None:
            warnings.append(f"Missing gcRight file: {gc_right_path}")
        if not has_hs:
            warnings.append("No HS events found")
        if not has_to:
            warnings.append("No TO events found")

        quality = 1.0
        if not has_hs:
            quality -= 0.2
        if not has_to:
            quality -= 0.1
        if gc_left_df is None or gc_right_df is None:
            quality -= 0.1

        return ExtractionResult(
            source_file=str(filepath),
            subject_id=file_info['subject_id'],
            trial_id=file_info['trial_id'],
            condition=file_info['activity'],
            fps=fps,
            n_frames=n_frames,
            duration_s=duration_s,
            angle_frames=angle_frames,
            raw_data=None,
            ground_truth=gt,
            quality_score=max(quality, 0.0),
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_mat_dataframe(filepath: Path):
        """Load a .mat file via matio and return the 'data' DataFrame.

        Returns None if the file does not exist or cannot be read.
        """
        if filepath is None or not filepath.exists():
            return None
        try:
            mat = load_from_mat(str(filepath))
            return mat.get('data', None)
        except Exception as exc:
            logger.warning("Failed to load %s: %s", filepath, exc)
            return None

    @staticmethod
    def _gc_path(markers_path: Path, gc_dir: str) -> Path:
        """Derive the gait-cycle file path from a markers file path.

        markers/.../markers/Levelground_01_00_01_00.mat
        ->
        markers/.../gcLeft/Levelground_01_00_01_00.mat
        """
        parts = list(markers_path.parts)
        # Replace the 'markers' directory component with the gc directory name
        try:
            idx = len(parts) - 1 - parts[::-1].index('markers')
            parts[idx] = gc_dir
        except ValueError:
            logger.warning(
                "Could not find 'markers' directory component in %s",
                markers_path,
            )
            return markers_path.parent / gc_dir / markers_path.name
        return Path(*parts)

    @staticmethod
    def _parse_filepath(filepath: Path) -> Dict:
        """Extract metadata from the directory hierarchy and filename.

        Expected path fragment:
            .../part3/AB06/2019-03-12/levelground/markers/Levelground_01_00_01_00.mat

        Filename pattern: {Activity}_{direction}_{speed}_{trial}_{gaitcycle}.mat
        """
        info = {
            'subject_id': 'unknown',
            'activity': 'walking',
            'speed': 'unknown',
            'direction': 'unknown',
            'trial': '00',
            'gait_cycle': '00',
            'trial_id': filepath.stem,
            'date': 'unknown',
            'part': 'unknown',
        }

        parts = filepath.parts

        # Walk up the path to find relevant components
        for i, component in enumerate(parts):
            # Part number
            part_match = re.match(r'(part\d+)', component, re.IGNORECASE)
            if part_match:
                info['part'] = part_match.group(1).lower()

            # Subject ID  (AB01 .. AB20)
            subj_match = re.match(r'(AB\d+)', component)
            if subj_match:
                info['subject_id'] = subj_match.group(1)

            # Date (YYYY-MM-DD)
            date_match = re.match(r'(\d{4}-\d{2}-\d{2})', component)
            if date_match:
                info['date'] = date_match.group(1)

            # Activity
            if component.lower() in WALKING_ACTIVITIES:
                info['activity'] = component.lower()

        # Parse filename: Activity_direction_speed_trial_gaitcycle.mat
        stem = filepath.stem
        fname_parts = stem.split('_')
        if len(fname_parts) >= 5:
            info['direction'] = fname_parts[1]
            info['speed'] = fname_parts[2]
            info['trial'] = fname_parts[3]
            info['gait_cycle'] = fname_parts[4]
        elif len(fname_parts) >= 3:
            # Partial match
            info['direction'] = fname_parts[1] if len(fname_parts) > 1 else 'unknown'
            info['speed'] = fname_parts[2] if len(fname_parts) > 2 else 'unknown'

        info['trial_id'] = (
            f"{info['subject_id']}_{info['activity']}_"
            f"{info['direction']}_{info['speed']}_"
            f"{info['trial']}_{info['gait_cycle']}"
        )

        return info

    @staticmethod
    def _extract_gait_events(
        gc_df,
        header: np.ndarray,
        side: str,
        filepath: Path,
    ) -> Tuple[List[int], List[int]]:
        """Extract heel-strike and toe-off frame indices from a gait cycle DataFrame.

        The gait cycle columns (HeelStrike, ToeOff) go from 0 to 100 and then
        reset.  A reset (drop > 50 between consecutive samples) marks a new
        gait cycle boundary.  The frame *after* the drop corresponds to the
        event time.

        We skip the first cycle only if its preceding duration is abnormally
        long (> 2x median cycle duration), indicating a non-walking start.

        Parameters
        ----------
        gc_df : DataFrame | None
            DataFrame with columns Header, HeelStrike, ToeOff.
        header : np.ndarray
            Time stamps from the markers file (used to convert gc times
            to frame indices).
        side : str
            'left' or 'right' (for logging only).
        filepath : Path
            Source file path (for logging only).

        Returns
        -------
        hs_frames, to_frames : list of int
            Frame indices (0-based into the markers array).
        """
        if gc_df is None:
            return [], []

        hs_frames: List[int] = []
        to_frames: List[int] = []

        try:
            hs_vals = gc_df['HeelStrike'].values
            to_vals = gc_df['ToeOff'].values
            gc_header = gc_df['Header'].values

            # --- Heel strikes ---
            hs_diff = np.diff(hs_vals)
            hs_drop_idx = np.where(hs_diff < -50)[0]

            # Only skip the first cycle if its preceding duration is abnormally
            # long (> 2x median), indicating a non-walking start.
            for idx_i, drop_idx in enumerate(hs_drop_idx):
                if idx_i == 0 and len(hs_drop_idx) > 2:
                    # Check if first cycle is abnormally long
                    median_gap = np.median(np.diff(hs_drop_idx))
                    if drop_idx > 2 * median_gap:
                        continue  # skip non-walking start
                event_time = gc_header[drop_idx + 1]
                frame = _time_to_frame(event_time, header)
                if frame is not None:
                    hs_frames.append(frame)

            # --- Toe offs ---
            to_diff = np.diff(to_vals)
            to_drop_idx = np.where(to_diff < -50)[0]

            for idx_i, drop_idx in enumerate(to_drop_idx):
                if idx_i == 0 and len(to_drop_idx) > 2:
                    median_gap = np.median(np.diff(to_drop_idx))
                    if drop_idx > 2 * median_gap:
                        continue
                # Previous: skipped first unconditionally
                event_time = gc_header[drop_idx + 1]
                frame = _time_to_frame(event_time, header)
                if frame is not None:
                    to_frames.append(frame)

        except KeyError as exc:
            logger.warning(
                "Missing expected column in gc%s for %s: %s",
                side.capitalize(), filepath, exc,
            )
        except Exception as exc:
            logger.warning(
                "Error extracting %s gait events from %s: %s",
                side, filepath, exc,
            )

        return sorted(set(hs_frames)), sorted(set(to_frames))

    def _build_angle_frames(self, markers_df) -> List[AngleFrame]:
        """Construct an AngleFrame for every row of the markers DataFrame.

        Landmark positions are extracted from the marker columns.  Joint
        angles are computed geometrically from the 3-D marker positions.
        """
        n_frames = len(markers_df)
        columns = set(markers_df.columns)
        angle_frames: List[AngleFrame] = []

        # Pre-extract numpy arrays for the markers we need.
        # Each marker has columns {name}_x, {name}_y, {name}_z.
        marker_arrays: Dict[str, Optional[np.ndarray]] = {}
        for marker_name, landmark_name in MARKER_TO_LANDMARK.items():
            cx, cy, cz = f'{marker_name}_x', f'{marker_name}_y', f'{marker_name}_z'
            if cx in columns and cy in columns and cz in columns:
                arr = markers_df[[cx, cy, cz]].values  # (n_frames, 3)
                marker_arrays[landmark_name] = arr
            else:
                marker_arrays[landmark_name] = None

        # Pre-compute pelvis center: mean of ASIS + PSIS markers
        pelvis_parts = ['left_asis', 'right_asis', 'left_psis', 'right_psis']
        has_pelvis = all(marker_arrays.get(p) is not None for p in pelvis_parts)

        for i in range(n_frames):
            # --- landmark positions ---
            positions: Dict[str, Tuple[float, float, float]] = {}
            pos_np: Dict[str, Optional[np.ndarray]] = {}

            for landmark_name, arr in marker_arrays.items():
                if arr is not None:
                    vec = arr[i]
                    if not np.any(np.isnan(vec)):
                        positions[landmark_name] = (
                            float(vec[0]), float(vec[1]), float(vec[2]),
                        )
                        pos_np[landmark_name] = vec
                    else:
                        positions[landmark_name] = (0.0, 0.0, 0.0)
                        pos_np[landmark_name] = None

            # Compute hip centres from ASIS / PSIS
            for side in ('left', 'right'):
                asis = pos_np.get(f'{side}_asis')
                psis = pos_np.get(f'{side}_psis')
                if asis is not None and psis is not None:
                    hip = (asis + psis) / 2.0
                    positions[f'{side}_hip'] = (
                        float(hip[0]), float(hip[1]), float(hip[2]),
                    )
                    pos_np[f'{side}_hip'] = hip

            # Pelvis centre
            if has_pelvis:
                parts = [marker_arrays[p][i] for p in pelvis_parts]
                if all(not np.any(np.isnan(p)) for p in parts):
                    pelvis = np.mean(parts, axis=0)
                    positions['pelvis_center'] = (
                        float(pelvis[0]), float(pelvis[1]), float(pelvis[2]),
                    )

            # --- Joint angles ---
            left_knee_angle = right_knee_angle = 0.0
            left_hip_angle = right_hip_angle = 0.0
            left_ankle_angle = right_ankle_angle = 0.0
            trunk_angle = 0.0

            # Knee angle: hip-knee-ankle
            lh = pos_np.get('left_hip')
            rh = pos_np.get('right_hip')
            lk = pos_np.get('left_knee')
            rk = pos_np.get('right_knee')
            la = pos_np.get('left_ankle')
            ra = pos_np.get('right_ankle')
            lt = pos_np.get('left_toe')
            rt = pos_np.get('right_toe')

            if lh is not None and lk is not None and la is not None:
                left_knee_angle = 180 - compute_angle_from_3points(lh, lk, la)
            if rh is not None and rk is not None and ra is not None:
                right_knee_angle = 180 - compute_angle_from_3points(rh, rk, ra)

            # Hip angle: thigh relative to vertical
            # Detect vertical axis from data (Y-up in Camargo, Z-up in others)
            if i == 0:
                # Determine vertical from hip height: axis with largest hip value
                if lh is not None:
                    vert_axis = int(np.argmax(np.abs(lh)))
                    _vertical = np.zeros(3)
                    _vertical[vert_axis] = -np.sign(lh[vert_axis])  # point downward
                else:
                    _vertical = np.array([0.0, 0.0, -1.0])
            vertical = _vertical
            if lh is not None and lk is not None:
                thigh_vec = lk - lh
                norm = np.linalg.norm(thigh_vec)
                if norm > 1e-10:
                    left_hip_angle = np.degrees(np.arccos(np.clip(
                        np.dot(thigh_vec, vertical) / norm, -1, 1)))
            if rh is not None and rk is not None:
                thigh_vec = rk - rh
                norm = np.linalg.norm(thigh_vec)
                if norm > 1e-10:
                    right_hip_angle = np.degrees(np.arccos(np.clip(
                        np.dot(thigh_vec, vertical) / norm, -1, 1)))

            # Ankle angle: knee-ankle-toe
            if lk is not None and la is not None and lt is not None:
                left_ankle_angle = compute_angle_from_3points(lk, la, lt) - 90
            if rk is not None and ra is not None and rt is not None:
                right_ankle_angle = compute_angle_from_3points(rk, ra, rt) - 90

            # Trunk angle (simplified – no dedicated shoulder markers in this
            # dataset, so we leave it at 0.0)
            trunk_angle = 0.0

            af = AngleFrame(
                frame_index=i,
                left_hip_angle=left_hip_angle,
                right_hip_angle=right_hip_angle,
                left_knee_angle=left_knee_angle,
                right_knee_angle=right_knee_angle,
                left_ankle_angle=left_ankle_angle,
                right_ankle_angle=right_ankle_angle,
                trunk_angle=trunk_angle,
                pelvis_tilt=0.0,
                landmark_positions=positions,
                is_valid=True,
            )
            angle_frames.append(af)

        return angle_frames


# --------------------------------------------------------------------------
# Module-level helpers
# --------------------------------------------------------------------------

def _time_to_frame(
    event_time: float,
    header: np.ndarray,
) -> Optional[int]:
    """Convert a time stamp to the nearest frame index.

    Uses the *header* array (time column from the markers file) to find the
    closest matching frame.  Returns ``None`` if the time falls outside the
    recorded range.
    """
    if len(header) == 0:
        return None
    idx = int(np.argmin(np.abs(header - event_time)))
    # Sanity: reject if the nearest frame is more than 2 samples away
    if abs(header[idx] - event_time) > 2.0 / SAMPLING_RATE:
        return None
    return idx


def test_extractor():
    """Quick test of the extractor (requires data on disk)."""
    if not HAS_MATIO:
        print("matio not installed")
        return

    ext = CamargoExtractor()
    files = ext.list_files()
    print(f"Found {len(files)} marker files")

    if files:
        f = files[0]
        print(f"  Extracting: {f}")
        result = ext.extract_file(f)
        print(f"  Subject: {result.subject_id}, Trial: {result.trial_id}")
        print(f"  Condition: {result.condition}")
        print(f"  Frames: {result.n_frames}, FPS: {result.fps}, "
              f"Duration: {result.duration_s:.2f}s")
        gt = result.ground_truth
        print(f"  GT HS L: {len(gt.hs_frames['left'])}, "
              f"R: {len(gt.hs_frames['right'])}")
        print(f"  GT TO L: {len(gt.to_frames['left'])}, "
              f"R: {len(gt.to_frames['right'])}")
        if gt.cadence:
            print(f"  Cadence: {gt.cadence:.1f} steps/min")
        print(f"  Quality: {result.quality_score:.2f}")
        print(f"  Warnings: {result.warnings}")


if __name__ == '__main__':
    test_extractor()
