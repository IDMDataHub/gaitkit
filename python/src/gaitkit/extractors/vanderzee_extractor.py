"""
Van der Zee Treadmill Gait Dataset Extractor.

Dataset: A biomechanics dataset of healthy human walking at various
         speeds, step lengths and step widths
URL: https://doi.org/10.1371/journal.pone.0266185
Publication: Van der Zee et al. 2022

Content:
- 10 healthy subjects
- Instrumented split-belt treadmill (2 force plates)
- 33 trials per subject at varying speeds (0.7-2.0 m/s),
  step lengths and step widths
- 120 Hz marker data, 1200 Hz analog data
- Ground truth: gait events from force plates (HS/TO)

Marker set:
    SACR, LGTR, RGTR (pelvis/hip)
    LTH1-3, RTH1-3 (thigh clusters)
    LSH1-3, RSH1-3 (shank clusters)
    LCAL, RCAL (calcaneus / heel)
    L5TH, R5TH (5th metatarsal)
    LLML, RLML (lateral malleolus = ankle)
    LASI, RASI (ASIS)
    LLEP, RLEP (lateral femoral epicondyle = knee lateral)
    LMEP, RMEP (medial femoral epicondyle = knee medial)
    LMML, RMML (medial malleolus)
    LAC, RAC (acromion)
    LEP, REP (elbow)
    LWR, RWR (wrist)

Force plates: Type 4 with calibration matrix, 2 plates (split-belt).
    Plate assignment (left/right) is determined by comparing heel marker
    X positions against plate corner X ranges.

Trial lookup (trial_look_up.xlsx):
    Trials 1-33 with varying speeds and conditions (Preferred Walking,
    Constant Step Length, Constant Step Frequency, Fixed speed with
    % preferred frequency, Step width).

Dependency: pip install ezc3d scipy
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from scipy.signal import medfilt
import re
import logging

from .base_extractor import (
    BaseExtractor, ExtractionResult, AngleFrame, GroundTruth,
    compute_angle_from_3points
)

logger = logging.getLogger(__name__)

try:
    import ezc3d
    HAS_EZC3D = True
except ImportError:
    HAS_EZC3D = False
    logger.debug("ezc3d not installed. Install with: pip install ezc3d")


# Trial lookup table: trial_number -> (speed_m_s, condition)
TRIAL_LOOKUP = {
    1:  (0.7, 'constant_step_length'),
    2:  (0.7, 'preferred'),
    3:  (0.7, 'constant_step_frequency'),
    4:  (0.9, 'constant_step_length'),
    5:  (0.9, 'preferred'),
    6:  (0.9, 'constant_step_frequency'),
    7:  (1.1, 'constant_step_length'),
    8:  (1.1, 'preferred'),
    9:  (1.1, 'constant_step_frequency'),
    10: (1.6, 'constant_step_frequency'),
    11: (1.6, 'preferred'),
    12: (1.6, 'constant_step_length'),
    13: (1.8, 'constant_step_frequency'),
    14: (1.8, 'preferred'),
    15: (1.8, 'constant_step_length'),
    16: (2.0, 'preferred'),
    17: (1.25, 'fixed_speed_70pct_freq'),
    18: (1.25, 'fixed_speed_80pct_freq'),
    19: (1.25, 'fixed_speed_90pct_freq'),
    20: (1.25, 'preferred'),
    21: (1.25, 'preferred_repeat1'),
    22: (1.25, 'preferred_repeat2'),
    23: (1.25, 'fixed_speed_110pct_freq'),
    24: (1.25, 'fixed_speed_120pct_freq'),
    25: (1.25, 'fixed_speed_130pct_freq'),
    26: (1.25, 'step_width_0cm'),
    27: (1.25, 'step_width_10cm'),
    28: (1.25, 'step_width_20cm'),
    29: (1.25, 'step_width_30cm'),
    30: (1.25, 'step_width_40cm'),
    31: (1.4, 'constant_step_length'),
    32: (1.4, 'preferred'),
    33: (1.4, 'constant_step_frequency'),
}


class VanderzeeExtractor(BaseExtractor):
    """Extractor for Van der Zee treadmill walking dataset."""

    # Marker mapping to standardized landmark names
    MARKER_MAPPING = {
        # Pelvis
        'SACR': 'sacrum',
        'LASI': 'left_asis', 'RASI': 'right_asis',
        # Hip (greater trochanter as proxy)
        'LGTR': 'left_hip_gtr', 'RGTR': 'right_hip_gtr',
        # Knee (lateral epicondyle)
        'LLEP': 'left_knee_lat', 'RLEP': 'right_knee_lat',
        # Knee (medial epicondyle)
        'LMEP': 'left_knee_med', 'RMEP': 'right_knee_med',
        # Ankle (lateral malleolus)
        'LLML': 'left_ankle_lat', 'RLML': 'right_ankle_lat',
        # Ankle (medial malleolus)
        'LMML': 'left_ankle_med', 'RMML': 'right_ankle_med',
        # Heel (calcaneus)
        'LCAL': 'left_heel', 'RCAL': 'right_heel',
        # Toe (5th metatarsal)
        'L5TH': 'left_toe', 'R5TH': 'right_toe',
        # Shoulder (acromion)
        'LAC': 'left_shoulder', 'RAC': 'right_shoulder',
    }

    # Minimum contact duration in seconds to filter out noise
    MIN_CONTACT_DURATION_S = 0.15
    # Force threshold in Newtons for contact detection (after calibration)
    FORCE_THRESHOLD_N = -20.0
    # Median filter kernel size for force signal denoising (analog samples)
    MEDFILT_KERNEL = 51

    @property
    def name(self) -> str:
        return "Van der Zee Treadmill"

    @property
    def description(self) -> str:
        return "10 healthy subjects, split-belt treadmill, various speeds/step lengths/widths"

    def list_files(self) -> List[Path]:
        """List all walking trial C3D files (exclude standing/static)."""
        if not HAS_EZC3D:
            return []
        files = list(self.data_dir.rglob('*.c3d'))
        # Exclude static/standing calibration files
        files = [f for f in files if 'standing' not in f.stem.lower()
                 and 'static' not in f.stem.lower()]
        return sorted(files)

    def _parse_filename(self, filepath: Path) -> Dict:
        """Parse filename for metadata.

        Format: p1_trial20.c3d
        - p1 = subject (participant 1-10)
        - trial20 = trial number (1-33)
        """
        info = {
            'subject_id': 'unknown',
            'trial_num': 0,
            'trial_id': 'unknown',
            'condition': 'healthy',
            'speed_ms': 0.0,
            'walking_condition': 'unknown',
        }

        filename = filepath.stem

        # Extract subject
        match = re.match(r'p(\d+)', filename)
        if match:
            info['subject_id'] = 'P' + match.group(1).zfill(2)

        # Extract trial number
        trial_match = re.search(r'trial(\d+)', filename)
        if trial_match:
            trial_num = int(trial_match.group(1))
            info['trial_num'] = trial_num
            if trial_num in TRIAL_LOOKUP:
                speed, condition = TRIAL_LOOKUP[trial_num]
                info['speed_ms'] = speed
                info['walking_condition'] = condition
                info['trial_id'] = f"trial{trial_num}_{condition}_{speed}ms"
            else:
                info['trial_id'] = f"trial{trial_num}"

        return info

    def _get_marker_position(self, c3d_data, marker_name: str, frame: int) -> Optional[np.ndarray]:
        """Get 3D position of a marker at a given frame."""
        try:
            labels = c3d_data['parameters']['POINT']['LABELS']['value']
            for i, label in enumerate(labels):
                if label.strip().upper() == marker_name.upper():
                    pos = c3d_data['data']['points'][:3, i, frame]
                    if not np.any(np.isnan(pos)) and np.linalg.norm(pos) > 0:
                        return pos
            return None
        except Exception:
            return None

    def _find_landmark(self, c3d_data, landmark: str, frame: int) -> Optional[np.ndarray]:
        """Find landmark position by trying multiple marker names."""
        for marker, lm in self.MARKER_MAPPING.items():
            if lm == landmark:
                pos = self._get_marker_position(c3d_data, marker, frame)
                if pos is not None:
                    return pos
        return None

    def _compute_joint_center(self, c3d_data, frame: int, lat_lm: str, med_lm: str) -> Optional[np.ndarray]:
        """Compute joint center as midpoint of lateral and medial markers."""
        lat = self._find_landmark(c3d_data, lat_lm, frame)
        med = self._find_landmark(c3d_data, med_lm, frame)
        if lat is not None and med is not None:
            return (lat + med) / 2.0
        return lat or med  # fallback to whichever is available

    def _compute_hip_center(self, c3d_data, frame: int, side: str) -> Optional[np.ndarray]:
        """Estimate hip center from greater trochanter."""
        return self._find_landmark(c3d_data, f'{side}_hip_gtr', frame)

    def _determine_plate_foot_mapping(self, c3d_data) -> Dict[int, str]:
        """Determine which force plate corresponds to which foot.

        Compares the mean X position of LCAL vs RCAL markers against
        the X range of each force plate's corners.

        Returns:
            Dict mapping plate index (0-based) to 'left' or 'right'.
        """
        mapping = {}
        try:
            labels = c3d_data['parameters']['POINT']['LABELS']['value']
            points = c3d_data['data']['points']
            corners = c3d_data['parameters']['FORCE_PLATFORM']['CORNERS']['value']
            n_plates = c3d_data['parameters']['FORCE_PLATFORM']['USED']['value'][0]

            # Get mean X position of each heel over a stable portion of data
            lcal_idx = next((i for i, l in enumerate(labels) if l.strip() == 'LCAL'), -1)
            rcal_idx = next((i for i, l in enumerate(labels) if l.strip() == 'RCAL'), -1)

            if lcal_idx < 0 or rcal_idx < 0:
                return mapping

            n_frames = points.shape[2]
            mid = n_frames // 2
            # Use middle portion of data to avoid edge effects
            start = max(0, mid - 100)
            end = min(n_frames, mid + 100)

            lcal_x_vals = points[0, lcal_idx, start:end]
            rcal_x_vals = points[0, rcal_idx, start:end]
            lcal_x_vals = lcal_x_vals[~np.isnan(lcal_x_vals)]
            rcal_x_vals = rcal_x_vals[~np.isnan(rcal_x_vals)]

            if len(lcal_x_vals) == 0 or len(rcal_x_vals) == 0:
                return mapping

            lcal_mean_x = np.mean(lcal_x_vals)
            rcal_mean_x = np.mean(rcal_x_vals)

            for plate in range(n_plates):
                plate_x_center = np.mean(corners[0, :, plate])
                # Assign to closest foot
                dist_left = abs(lcal_mean_x - plate_x_center)
                dist_right = abs(rcal_mean_x - plate_x_center)
                mapping[plate] = 'left' if dist_left < dist_right else 'right'

        except Exception:
            pass

        return mapping

    def _extract_force_plate_events(self, c3d_data) -> Dict[str, List[int]]:
        """Extract gait events (HS/TO) from force plate data.

        For type 4 force plates, applies the calibration matrix to raw
        analog channels to get calibrated forces, then detects contact
        onset/offset from vertical force (Fz).

        Uses a relative threshold (10% of force range) to handle treadmill
        data where the baseline force may not reach zero due to cross-talk
        between split belts.

        Returns:
            Dict with keys 'hs_left', 'hs_right', 'to_left', 'to_right',
            each containing a sorted list of marker-rate frame indices.
        """
        events = {
            'hs_left': [],
            'hs_right': [],
            'to_left': [],
            'to_right': [],
        }

        try:
            fp = c3d_data['parameters']['FORCE_PLATFORM']
            n_plates = fp['USED']['value'][0]
            fp_type = fp['TYPE']['value']
            channels = fp['CHANNEL']['value']

            analog = c3d_data['data']['analogs'][0]
            analog_rate = c3d_data['parameters']['ANALOG']['RATE']['value'][0]
            marker_rate = c3d_data['parameters']['POINT']['RATE']['value'][0]
            ratio = analog_rate / marker_rate

            # Determine plate-foot mapping
            plate_foot = self._determine_plate_foot_mapping(c3d_data)
            if not plate_foot:
                return events

            has_cal = 'CAL_MATRIX' in fp
            cal = fp['CAL_MATRIX']['value'] if has_cal else None

            min_contact_samples = int(self.MIN_CONTACT_DURATION_S * analog_rate)

            for plate in range(n_plates):
                side = plate_foot.get(plate)
                if side is None:
                    continue

                ch_idx = channels[:, plate] - 1  # 0-indexed channel indices

                # Get Fz based on plate type
                if fp_type[plate] == 4 and cal is not None:
                    # Type 4: apply calibration matrix
                    raw = analog[ch_idx, :]
                    cal_mat = cal[:, :, plate]
                    forces = cal_mat @ raw
                    fz = forces[2, :]  # Fz component
                elif fp_type[plate] in [1, 2, 3]:
                    # Type 1-3: Fz is the 3rd channel directly
                    fz = analog[ch_idx[2], :]
                else:
                    continue

                # Apply median filter to denoise
                fz_filt = medfilt(fz, kernel_size=self.MEDFILT_KERNEL)

                # Use absolute value of Fz (negative = downward force)
                fz_abs = np.abs(fz_filt)

                # Relative threshold: 10% of force range
                # This handles treadmill data where baseline may not reach zero
                fz_range = fz_abs.max() - fz_abs.min()
                if fz_range < 50:  # Less than 50N range = no meaningful signal
                    continue
                rel_threshold = fz_abs.min() + 0.10 * fz_range

                # Detect contact: |Fz| > relative threshold
                contact = fz_abs > rel_threshold
                transitions = np.diff(contact.astype(int))
                onsets_raw = np.where(transitions == 1)[0] + 1
                offsets_raw = np.where(transitions == -1)[0] + 1

                # Debounce: merge events that are too close together
                onsets = self._debounce(onsets_raw, min_contact_samples)
                offsets = self._debounce(offsets_raw, min_contact_samples)

                # Convert to marker frame indices
                hs_frames = (onsets / ratio).astype(int).tolist()
                to_frames = (offsets / ratio).astype(int).tolist()

                events[f'hs_{side}'].extend(hs_frames)
                events[f'to_{side}'].extend(to_frames)

        except Exception:
            pass

        # Sort and deduplicate
        for key in events:
            events[key] = sorted(list(set(events[key])))

        return events

    @staticmethod
    def _debounce(events: np.ndarray, min_gap: int) -> np.ndarray:
        """Remove duplicate events within min_gap samples."""
        if len(events) == 0:
            return events
        result = [events[0]]
        for e in events[1:]:
            if e - result[-1] >= min_gap:
                result.append(e)
        return np.array(result)

    def extract_file(self, filepath: Path) -> ExtractionResult:
        """Extract data from a Van der Zee C3D file."""
        if not HAS_EZC3D:
            raise RuntimeError("ezc3d not installed")

        c3d = ezc3d.c3d(str(filepath))
        file_info = self._parse_filename(filepath)

        fps = c3d['parameters']['POINT']['RATE']['value'][0]
        n_frames = c3d['data']['points'].shape[2]
        duration_s = n_frames / fps

        # Extract gait events from force plates
        events = self._extract_force_plate_events(c3d)

        warnings = []

        # Build angle frames
        angle_frames = []
        for frame in range(n_frames):
            # Get positions
            left_hip = self._compute_hip_center(c3d, frame, 'left')
            right_hip = self._compute_hip_center(c3d, frame, 'right')

            left_knee = self._compute_joint_center(c3d, frame, 'left_knee_lat', 'left_knee_med')
            right_knee = self._compute_joint_center(c3d, frame, 'right_knee_lat', 'right_knee_med')

            left_ankle = self._compute_joint_center(c3d, frame, 'left_ankle_lat', 'left_ankle_med')
            right_ankle = self._compute_joint_center(c3d, frame, 'right_ankle_lat', 'right_ankle_med')

            left_heel = self._find_landmark(c3d, 'left_heel', frame)
            right_heel = self._find_landmark(c3d, 'right_heel', frame)
            left_toe = self._find_landmark(c3d, 'left_toe', frame)
            right_toe = self._find_landmark(c3d, 'right_toe', frame)

            sacrum = self._find_landmark(c3d, 'sacrum', frame)
            left_shoulder = self._find_landmark(c3d, 'left_shoulder', frame)
            right_shoulder = self._find_landmark(c3d, 'right_shoulder', frame)

            # Pelvis center
            pelvis = sacrum
            left_asis = self._find_landmark(c3d, 'left_asis', frame)
            right_asis = self._find_landmark(c3d, 'right_asis', frame)
            if left_asis is not None and right_asis is not None and sacrum is not None:
                pelvis = (left_asis + right_asis + sacrum) / 3.0

            def to_tuple(pos):
                return (float(pos[0]), float(pos[1]), float(pos[2])) if pos is not None else (0.0, 0.0, 0.0)

            landmark_positions = {
                'left_hip': to_tuple(left_hip), 'right_hip': to_tuple(right_hip),
                'left_knee': to_tuple(left_knee), 'right_knee': to_tuple(right_knee),
                'left_ankle': to_tuple(left_ankle), 'right_ankle': to_tuple(right_ankle),
                'left_heel': to_tuple(left_heel), 'right_heel': to_tuple(right_heel),
                'left_toe': to_tuple(left_toe), 'right_toe': to_tuple(right_toe),
                'left_shoulder': to_tuple(left_shoulder), 'right_shoulder': to_tuple(right_shoulder),
                'pelvis': to_tuple(pelvis),
            }

            # Compute joint angles
            left_knee_angle = right_knee_angle = 0.0
            left_hip_angle = right_hip_angle = 0.0
            left_ankle_angle = right_ankle_angle = 0.0
            trunk_angle = 0.0
            pelvis_tilt = 0.0

            # Knee flexion: 180 - angle(hip, knee, ankle)
            if left_hip is not None and left_knee is not None and left_ankle is not None:
                left_knee_angle = 180 - compute_angle_from_3points(left_hip, left_knee, left_ankle)

            if right_hip is not None and right_knee is not None and right_ankle is not None:
                right_knee_angle = 180 - compute_angle_from_3points(right_hip, right_knee, right_ankle)

            # Hip flexion: angle of thigh relative to vertical
            if left_hip is not None and left_knee is not None:
                thigh_vec = left_knee - left_hip
                vertical = np.array([0, 0, -1])
                left_hip_angle = np.degrees(np.arccos(np.clip(
                    np.dot(thigh_vec, vertical) / (np.linalg.norm(thigh_vec) + 1e-10), -1, 1)))

            if right_hip is not None and right_knee is not None:
                thigh_vec = right_knee - right_hip
                vertical = np.array([0, 0, -1])
                right_hip_angle = np.degrees(np.arccos(np.clip(
                    np.dot(thigh_vec, vertical) / (np.linalg.norm(thigh_vec) + 1e-10), -1, 1)))

            # Ankle dorsiflexion: angle(knee, ankle, toe) - 90
            if left_knee is not None and left_ankle is not None and left_toe is not None:
                left_ankle_angle = compute_angle_from_3points(left_knee, left_ankle, left_toe) - 90

            if right_knee is not None and right_ankle is not None and right_toe is not None:
                right_ankle_angle = compute_angle_from_3points(right_knee, right_ankle, right_toe) - 90

            af = AngleFrame(
                frame_index=frame,
                left_hip_angle=left_hip_angle,
                right_hip_angle=right_hip_angle,
                left_knee_angle=left_knee_angle,
                right_knee_angle=right_knee_angle,
                left_ankle_angle=left_ankle_angle,
                right_ankle_angle=right_ankle_angle,
                trunk_angle=trunk_angle,
                pelvis_tilt=pelvis_tilt,
                landmark_positions=landmark_positions,
                is_valid=True
            )
            angle_frames.append(af)

        # Ground truth
        has_hs = len(events['hs_left']) > 0 or len(events['hs_right']) > 0
        has_to = len(events['to_left']) > 0 or len(events['to_right']) > 0

        gt = GroundTruth(
            has_hs=has_hs,
            has_to=has_to,
            has_cadence=has_hs,
            has_angles=False,
            has_forces=True,
            hs_frames={'left': events['hs_left'], 'right': events['hs_right']},
            to_frames={'left': events['to_left'], 'right': events['to_right']},
            # Van der Zee is a split-belt treadmill: force plates cover the
            # entire walking surface so every step is captured. We mark the
            # source but leave valid_frame_range=None (no zone restriction).
            event_source="force_plate",
            valid_frame_range=None,
        )

        # Compute cadence from HS events
        if has_hs:
            all_hs = sorted(events['hs_left'] + events['hs_right'])
            if len(all_hs) > 1:
                intervals = np.diff(all_hs) / fps
                valid_intervals = intervals[(intervals > 0.3) & (intervals < 2.0)]
                if len(valid_intervals) > 0:
                    gt.cadence = 60.0 / np.mean(valid_intervals)

        if not has_hs:
            warnings.append("No HS events detected from force plates")
        if file_info['speed_ms'] > 0:
            warnings.append(f"Speed: {file_info['speed_ms']} m/s, condition: {file_info['walking_condition']}")

        return ExtractionResult(
            source_file=str(filepath),
            subject_id=file_info['subject_id'],
            trial_id=file_info['trial_id'],
            condition=file_info['condition'],
            fps=fps,
            n_frames=n_frames,
            duration_s=duration_s,
            angle_frames=angle_frames,
            raw_data=None,
            ground_truth=gt,
            quality_score=0.9 if has_hs else 0.6,
            warnings=warnings
        )


def test_extractor():
    """Quick test of the Van der Zee extractor."""
    if not HAS_EZC3D:
        print("ezc3d not installed")
        return

    data_dir = Path.home() / 'gait_benchmark_project/data/vanderzee'
    if not data_dir.exists():
        print(f"Directory not found: {data_dir}")
        return

    extractor = VanderzeeExtractor(str(data_dir))
    files = extractor.list_files()
    print(f"Found {len(files)} C3D files")

    # Test on 3 files
    for f in files[:3]:
        try:
            result = extractor.extract_file(f)
            print(f"\nExtracted: {f.name}")
            print(f"  Subject: {result.subject_id}, Trial: {result.trial_id}")
            print(f"  Frames: {result.n_frames}, FPS: {result.fps}, Duration: {result.duration_s:.1f}s")
            print(f"  GT HS L: {len(result.ground_truth.hs_frames['left'])}")
            print(f"  GT HS R: {len(result.ground_truth.hs_frames['right'])}")
            print(f"  GT TO L: {len(result.ground_truth.to_frames['left'])}")
            print(f"  GT TO R: {len(result.ground_truth.to_frames['right'])}")
            if result.ground_truth.cadence:
                print(f"  Cadence: {result.ground_truth.cadence:.1f} steps/min")
            print(f"  Quality: {result.quality_score}")
            print(f"  Warnings: {result.warnings}")
        except Exception as e:
            print(f"\nError on {f.name}: {e}")


if __name__ == '__main__':
    test_extractor()
