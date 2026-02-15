"""
Van Criekinge Extractor -- Healthy Controls + Stroke Survivors.

Datasets:
- 138_HealthyPiG_10.05: 138 healthy adults, Plug-in Gait, 100 Hz
- 50_StrokePiG: 50 stroke survivors, Plug-in Gait, 100 Hz

Source: figshare collection 10.6084/m9.figshare.c.6503791
Publication: Van Criekinge T, Saeys W, Hallemans A, et al. (2023)

Key features:
- Plug-in Gait marker set (LASI, RASI, SACR/LPSI/RPSI, LKNE, RKNE, LANK, RANK, etc.)
- C3D includes PRE-COMPUTED joint angles (LHipAngles, LKneeAngles, LAnkleAngles, etc.)
- Ground truth events: 'Foot Strike' and 'Foot Off' with Left/Right CONTEXTS
- 100 Hz sampling rate

Dependency: pip install ezc3d
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
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


class VanCriekingeExtractor(BaseExtractor):
    """Extractor for Van Criekinge healthy + stroke datasets (Plug-in Gait)."""

    # Plug-in Gait marker mapping
    MARKER_MAPPING = {
        # Pelvis
        'LASI': 'left_asis', 'RASI': 'right_asis',
        'LPSI': 'left_psis', 'RPSI': 'right_psis',
        'SACR': 'sacrum',
        # Thigh
        'LTHI': 'left_thigh', 'RTHI': 'right_thigh',
        # Knee
        'LKNE': 'left_knee', 'RKNE': 'right_knee',
        # Shank
        'LTIB': 'left_tibia', 'RTIB': 'right_tibia',
        # Ankle
        'LANK': 'left_ankle', 'RANK': 'right_ankle',
        # Foot
        'LHEE': 'left_heel', 'RHEE': 'right_heel',
        'LTOE': 'left_toe', 'RTOE': 'right_toe',
        # Upper body
        'C7': 'c7', 'T10': 't10',
        'CLAV': 'clavicle', 'STRN': 'sternum',
        'LSHO': 'left_shoulder', 'RSHO': 'right_shoulder',
    }

    # Pre-computed angle labels in C3D (Plug-in Gait model outputs)
    ANGLE_LABELS = {
        'LHipAngles': 'left_hip',
        'RHipAngles': 'right_hip',
        'LKneeAngles': 'left_knee',
        'RKneeAngles': 'right_knee',
        'LAnkleAngles': 'left_ankle',
        'RAnkleAngles': 'right_ankle',
        'LPelvisAngles': 'left_pelvis',
        'RPelvisAngles': 'right_pelvis',
    }

    def __init__(self, data_dir: str, dataset_type: str = 'healthy'):
        """
        Args:
            data_dir: Path to dataset directory
            dataset_type: 'healthy' or 'stroke'
        """
        super().__init__(data_dir)
        self.dataset_type = dataset_type

    @property
    def name(self) -> str:
        return f"Van Criekinge {self.dataset_type.capitalize()}"

    @property
    def description(self) -> str:
        if self.dataset_type == 'stroke':
            return "50 stroke survivors, Plug-in Gait, overground walking"
        return "138 healthy adults across the life span, Plug-in Gait, overground walking"

    def list_files(self) -> List[Path]:
        """List all walking C3D files (exclude static/calibration)."""
        if not HAS_EZC3D:
            return []
        files = list(self.data_dir.rglob('*.c3d'))
        # Exclude calibration files (contain 'Cal' in name)
        files = [f for f in files if 'Cal' not in f.stem and '_ST' not in f.stem.upper()]
        return sorted(files)

    def _parse_filename(self, filepath: Path) -> Dict:
        """Parse filename for metadata.

        Healthy: SUBJ05/SUBJ5 (1).c3d -> subject_id=SUBJ05, trial=1
        Stroke:  TVC39/BWA 03.c3d -> subject_id=TVC39, trial=03
        """
        info = {
            'subject_id': 'unknown',
            'condition': self.dataset_type,
            'trial': '01',
        }

        parent = filepath.parent.name  # e.g., 'SUBJ05' or 'TVC39'

        if self.dataset_type == 'healthy':
            # Extract subject from parent folder
            match = re.match(r'SUBJ(\d+)', parent)
            if match:
                info['subject_id'] = f'SUBJ{int(match.group(1)):03d}'

            # Extract trial from filename: SUBJ5 (1).c3d
            trial_match = re.search(r'\((\d+)\)', filepath.stem)
            if trial_match:
                info['trial'] = trial_match.group(1)

        elif self.dataset_type == 'stroke':
            # Subject from parent: TVC39
            info['subject_id'] = parent

            # Trial from filename: BWA 03.c3d or BWA3.c3d
            trial_match = re.search(r'BWA\s*(\d+)', filepath.stem)
            if trial_match:
                info['trial'] = trial_match.group(1)

        return info

    def _get_marker_position(self, c3d_data, marker_name: str, frame: int) -> Optional[np.ndarray]:
        """Get 3D position of a marker at a given frame."""
        try:
            labels = c3d_data['parameters']['POINT']['LABELS']['value']
            for i, label in enumerate(labels):
                clean = label.strip().split(":")[-1] if ":" in label.strip() else label.strip()
                if clean.upper() == marker_name.upper():
                    pos = c3d_data['data']['points'][:3, i, frame]
                    if not np.any(np.isnan(pos)) and np.linalg.norm(pos) > 0:
                        return pos
            return None
        except (KeyError, IndexError, TypeError, ValueError):
            return None

    def _find_landmark(self, c3d_data, landmark: str, frame: int) -> Optional[np.ndarray]:
        """Find landmark position by trying multiple marker names."""
        for marker, lm in self.MARKER_MAPPING.items():
            if lm == landmark:
                pos = self._get_marker_position(c3d_data, marker, frame)
                if pos is not None:
                    return pos
        return None

    def _get_precomputed_angle(self, c3d_data, angle_label: str, frame: int, component: int = 0) -> Optional[float]:
        """Get pre-computed angle from Plug-in Gait model output.

        Args:
            angle_label: e.g., 'LHipAngles'
            frame: frame index
            component: 0=sagittal (flexion/extension), 1=frontal, 2=transverse
        """
        try:
            labels = c3d_data['parameters']['POINT']['LABELS']['value']
            for i, label in enumerate(labels):
                clean = label.strip().split(":")[-1] if ":" in label.strip() else label.strip()
                if clean == angle_label:
                    val = c3d_data['data']['points'][component, i, frame]
                    if not np.isnan(val):
                        return float(val)
            return None
        except (KeyError, IndexError, TypeError, ValueError):
            return None

    def _extract_events(self, c3d_data) -> Dict[str, List[int]]:
        """Extract gait events from C3D EVENT group.

        Events: 'Foot Strike' + 'Foot Off' with Left/Right CONTEXTS
        """
        events = {
            'hs_left': [],
            'hs_right': [],
            'to_left': [],
            'to_right': [],
        }

        try:
            event_params = c3d_data['parameters']['EVENT']
            labels = event_params['LABELS']['value']
            contexts = event_params['CONTEXTS']['value']
            times = event_params['TIMES']['value']
            fps = c3d_data['parameters']['POINT']['RATE']['value'][0]
            first_frame = c3d_data["header"]["points"]["first_frame"]
            n_frames = c3d_data["data"]["points"].shape[2]

            for i, label in enumerate(labels):
                label_upper = label.strip().upper()
                context = contexts[i].strip().upper() if i < len(contexts) else ''

                # Get time
                if times.ndim > 1:
                    time = times[1, i]
                else:
                    time = times[i]
                # Convert to frame index relative to data array
                frame = int(round(time * fps)) - first_frame
                if frame < 0 or frame >= n_frames:
                    continue  # Skip events outside data range

                # Determine side
                if 'LEFT' in context:
                    side = 'left'
                elif 'RIGHT' in context:
                    side = 'right'
                else:
                    continue  # Skip if no context

                # Classify event type
                if 'STRIKE' in label_upper or 'HS' in label_upper:
                    events[f'hs_{side}'].append(frame)
                elif 'OFF' in label_upper or 'TO' in label_upper:
                    events[f'to_{side}'].append(frame)

        except Exception as exc:
            logger.debug("No readable C3D EVENT entries for %s", exc)

        for key in events:
            events[key] = sorted(list(set(events[key])))

        return events

    def _compute_hip_center(self, c3d_data, frame: int, side: str) -> Optional[np.ndarray]:
        """Estimate hip center from ASIS/PSIS or SACR."""
        asis = self._find_landmark(c3d_data, f'{side}_asis', frame)

        # Try PSIS first (stroke dataset), then SACR (healthy dataset)
        psis = self._find_landmark(c3d_data, f'{side}_psis', frame)
        sacrum = self._find_landmark(c3d_data, 'sacrum', frame)

        if asis is not None and psis is not None:
            return (asis + psis) / 2
        elif asis is not None and sacrum is not None:
            return (asis + sacrum) / 2
        return asis

    def extract_file(self, filepath: Path) -> ExtractionResult:
        """Extract data from a C3D file.

        Strategy: Use pre-computed angles from Plug-in Gait model when available,
        fall back to computing from marker positions.
        """
        if not HAS_EZC3D:
            raise RuntimeError("ezc3d not installed")

        c3d = ezc3d.c3d(str(filepath))

        file_info = self._parse_filename(filepath)

        fps = c3d['parameters']['POINT']['RATE']['value'][0]
        n_frames = c3d['data']['points'].shape[2]
        duration_s = n_frames / fps

        events = self._extract_events(c3d)

        # Check if pre-computed angles are available
        labels = [l.strip() for l in c3d['parameters']['POINT']['LABELS']['value']]
        has_precomputed = any(l.strip().endswith('LHipAngles') for l in labels)

        angle_frames = []
        warnings = []

        for frame in range(n_frames):
            # Try pre-computed angles first (more accurate, from Plug-in Gait model)
            left_hip_angle = 0.0
            right_hip_angle = 0.0
            left_knee_angle = 0.0
            right_knee_angle = 0.0
            left_ankle_angle = 0.0
            right_ankle_angle = 0.0
            trunk_angle = 0.0
            pelvis_tilt = 0.0

            if has_precomputed:
                lh = self._get_precomputed_angle(c3d, 'LHipAngles', frame, 0)
                rh = self._get_precomputed_angle(c3d, 'RHipAngles', frame, 0)
                lk = self._get_precomputed_angle(c3d, 'LKneeAngles', frame, 0)
                rk = self._get_precomputed_angle(c3d, 'RKneeAngles', frame, 0)
                la = self._get_precomputed_angle(c3d, 'LAnkleAngles', frame, 0)
                ra = self._get_precomputed_angle(c3d, 'RAnkleAngles', frame, 0)
                lp = self._get_precomputed_angle(c3d, 'LPelvisAngles', frame, 0)

                if lh is not None: left_hip_angle = lh
                if rh is not None: right_hip_angle = rh
                if lk is not None: left_knee_angle = lk
                if rk is not None: right_knee_angle = rk
                if la is not None: left_ankle_angle = la
                if ra is not None: right_ankle_angle = ra
                if lp is not None: pelvis_tilt = lp

            else:
                # Fall back to computing from markers
                left_hip = self._compute_hip_center(c3d, frame, 'left')
                right_hip = self._compute_hip_center(c3d, frame, 'right')
                left_knee = self._find_landmark(c3d, 'left_knee', frame)
                right_knee = self._find_landmark(c3d, 'right_knee', frame)
                left_ankle = self._find_landmark(c3d, 'left_ankle', frame)
                right_ankle = self._find_landmark(c3d, 'right_ankle', frame)
                left_toe = self._find_landmark(c3d, 'left_toe', frame)
                right_toe = self._find_landmark(c3d, 'right_toe', frame)

                if left_hip is not None and left_knee is not None and left_ankle is not None:
                    left_knee_angle = 180 - compute_angle_from_3points(left_hip, left_knee, left_ankle)
                if right_hip is not None and right_knee is not None and right_ankle is not None:
                    right_knee_angle = 180 - compute_angle_from_3points(right_hip, right_knee, right_ankle)
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
                if left_knee is not None and left_ankle is not None and left_toe is not None:
                    left_ankle_angle = compute_angle_from_3points(left_knee, left_ankle, left_toe) - 90
                if right_knee is not None and right_ankle is not None and right_toe is not None:
                    right_ankle_angle = compute_angle_from_3points(right_knee, right_ankle, right_toe) - 90

            # Landmark positions for detectors that use them
            left_heel = self._find_landmark(c3d, 'left_heel', frame)
            right_heel = self._find_landmark(c3d, 'right_heel', frame)
            left_toe = self._find_landmark(c3d, 'left_toe', frame)
            right_toe = self._find_landmark(c3d, 'right_toe', frame)
            left_ankle = self._find_landmark(c3d, 'left_ankle', frame)
            right_ankle = self._find_landmark(c3d, 'right_ankle', frame)
            left_knee = self._find_landmark(c3d, 'left_knee', frame)
            right_knee = self._find_landmark(c3d, 'right_knee', frame)
            left_hip = self._compute_hip_center(c3d, frame, 'left')
            right_hip = self._compute_hip_center(c3d, frame, 'right')
            left_shoulder = self._find_landmark(c3d, 'left_shoulder', frame)
            right_shoulder = self._find_landmark(c3d, 'right_shoulder', frame)

            def to_tuple(pos):
                return (float(pos[0]), float(pos[1]), float(pos[2])) if pos is not None else (0.0, 0.0, 0.0)

            landmark_positions = {
                'left_hip': to_tuple(left_hip), 'right_hip': to_tuple(right_hip),
                'left_knee': to_tuple(left_knee), 'right_knee': to_tuple(right_knee),
                'left_ankle': to_tuple(left_ankle), 'right_ankle': to_tuple(right_ankle),
                'left_heel': to_tuple(left_heel), 'right_heel': to_tuple(right_heel),
                'left_toe': to_tuple(left_toe), 'right_toe': to_tuple(right_toe),
                'left_shoulder': to_tuple(left_shoulder), 'right_shoulder': to_tuple(right_shoulder),
            }

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
            has_angles=has_precomputed,
            has_forces='LGroundReactionForce' in labels,
            hs_frames={'left': events['hs_left'], 'right': events['hs_right']},
            to_frames={'left': events['to_left'], 'right': events['to_right']},
        )

        if has_hs:
            all_hs = sorted(events['hs_left'] + events['hs_right'])
            if len(all_hs) > 1:
                intervals = np.diff(all_hs) / fps
                gt.cadence = 60.0 / np.mean(intervals) if np.mean(intervals) > 0 else 0.0

        if not has_hs:
            warnings.append("No HS events found")
        if has_precomputed:
            warnings.append("Using pre-computed Plug-in Gait angles")

        return ExtractionResult(
            source_file=str(filepath),
            subject_id=file_info['subject_id'],
            trial_id=file_info['trial'],
            condition=file_info['condition'],
            fps=fps,
            n_frames=n_frames,
            duration_s=duration_s,
            angle_frames=angle_frames,
            raw_data=None,
            ground_truth=gt,
            quality_score=0.95 if (has_hs and has_precomputed) else 0.7,
            warnings=warnings
        )


class VanCriekingeHealthyExtractor(VanCriekingeExtractor):
    """Convenience class for the healthy dataset."""
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = str(Path.home() / 'gait_benchmark_project/data/vancriekinge_healthy/138_HealthyPiG_10.05')
        super().__init__(data_dir, dataset_type='healthy')


class VanCriekingeStrokeExtractor(VanCriekingeExtractor):
    """Convenience class for the stroke dataset."""
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = str(Path.home() / 'gait_benchmark_project/data/vancriekinge_stroke/50_StrokePiG')
        super().__init__(data_dir, dataset_type='stroke')


def test_extractor():
    """Quick test of the extractor."""
    if not HAS_EZC3D:
        print("ezc3d not installed")
        return

    # Test healthy
    print("=== Testing Healthy ===")
    try:
        ext = VanCriekingeHealthyExtractor()
        files = ext.list_files()
        print(f"Found {len(files)} C3D files")
        if files:
            result = ext.extract_file(files[0])
            print(f"  File: {result.source_file}")
            print(f"  Subject: {result.subject_id}, Trial: {result.trial_id}")
            print(f"  Frames: {result.n_frames}, FPS: {result.fps}")
            print(f"  GT HS L: {len(result.ground_truth.hs_frames['left'])}")
            print(f"  GT HS R: {len(result.ground_truth.hs_frames['right'])}")
            print(f"  GT TO L: {len(result.ground_truth.to_frames['left'])}")
            print(f"  GT TO R: {len(result.ground_truth.to_frames['right'])}")
            print(f"  Warnings: {result.warnings}")
    except Exception as e:
        print(f"Error: {e}")

    # Test stroke
    print("\n=== Testing Stroke ===")
    try:
        ext = VanCriekingeStrokeExtractor()
        files = ext.list_files()
        print(f"Found {len(files)} C3D files")
        if files:
            result = ext.extract_file(files[0])
            print(f"  File: {result.source_file}")
            print(f"  Subject: {result.subject_id}, Trial: {result.trial_id}")
            print(f"  Frames: {result.n_frames}, FPS: {result.fps}")
            print(f"  GT HS L: {len(result.ground_truth.hs_frames['left'])}")
            print(f"  GT HS R: {len(result.ground_truth.hs_frames['right'])}")
            print(f"  Warnings: {result.warnings}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    test_extractor()
