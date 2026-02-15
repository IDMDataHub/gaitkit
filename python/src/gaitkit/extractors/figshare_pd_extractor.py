"""
Figshare PD Extractor - Parkinson's Disease Gait Dataset.

Dataset: A dataset of overground walking full-body kinematics and kinetics
         in individuals with Parkinson's disease
URL: https://figshare.com/articles/dataset/14896881

Content:
- 26 subjects with Parkinson's disease
- ON and OFF medication conditions
- 935 C3D files with MoCap + force plates
- Ground truth events: RHS, LHS, RTO, LTO in C3D EVENT group

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


class FigsharePDExtractor(BaseExtractor):
    """Extractor for Figshare Parkinson's Disease dataset."""

    # Marker mapping (Leardini full body model - Figshare PD naming)
    MARKER_MAPPING = {
        # Pelvis (note: with dots)
        'R.ASIS': 'right_asis', 'L.ASIS': 'left_asis',
        'R.PSIS': 'right_psis', 'L.PSIS': 'left_psis',
        # Thigh / Greater trochanter
        'R.GTR': 'right_thigh', 'L.GTR': 'left_thigh',
        # Knee
        'R.Knee': 'right_knee', 'L.Knee': 'left_knee',
        'R.Knee.Medial': 'right_knee_med', 'L.Knee.Medial': 'left_knee_med',
        # Shank
        'R.TT': 'right_tibia', 'L.TT': 'left_tibia',
        # Ankle
        'R.Ankle': 'right_ankle', 'L.Ankle': 'left_ankle',
        'R.Ankle.Medial': 'right_ankle_med', 'L.Ankle.Medial': 'left_ankle_med',
        # Foot
        'R.Heel': 'right_heel', 'L.Heel': 'left_heel',
        'R.MT1': 'right_toe', 'L.MT1': 'left_toe',
        'R.MT5': 'right_mt5', 'L.MT5': 'left_mt5',
        # Trunk
        'C7': 'c7', 'T10': 't10', 'STRN': 'sternum', 'CLAV': 'clavicle',
        'RSHO': 'right_shoulder', 'LSHO': 'left_shoulder',
        # Virtual joint centers (computed by system)
        'V_R.Hip_JC': 'right_hip_jc', 'V_L.Hip_JC': 'left_hip_jc',
        'V_R.Knee_JC_Dynamic': 'right_knee_jc', 'V_L.Knee_JC_Dynamic': 'left_knee_jc',
        'V_R.Ankle_JC_Dynamic': 'right_ankle_jc', 'V_L.Ankle_JC_Dynamic': 'left_ankle_jc',
    }

    @property
    def name(self) -> str:
        return "Figshare Parkinson's Disease"

    @property
    def description(self) -> str:
        return "26 subjects with PD (ON/OFF medication), C3D MoCap + forces"

    def list_files(self) -> List[Path]:
        """List all walk C3D files (exclude static)."""
        if not HAS_EZC3D:
            return []
        files = list(self.data_dir.rglob('*_walk_*.c3d'))
        return sorted(files)

    def _parse_filename(self, filepath: Path) -> Dict:
        """Parse filename for metadata.

        Format: SUB01_off_walk_1.c3d
        - SUB01 = subject ID
        - off/on = medication condition
        - walk_1 = trial number
        """
        info = {
            'subject_id': 'unknown',
            'condition': 'unknown',
            'trial': '01',
            'medication': 'unknown',
        }

        filename = filepath.stem
        parent = filepath.parent.name  # e.g., 'SUB01_off'

        # Extract from parent folder
        match = re.match(r'(SUB\d+)_(on|off)', parent, re.IGNORECASE)
        if match:
            info['subject_id'] = match.group(1).upper()
            info['medication'] = match.group(2).lower()
            info['condition'] = f'parkinson_{info["medication"]}'

        # Extract trial from filename
        trial_match = re.search(r'walk_(\d+)', filename)
        if trial_match:
            info['trial'] = trial_match.group(1)

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

    def _extract_events(self, c3d_data) -> Dict[str, List[int]]:
        """Extract gait events from C3D EVENT group."""
        events = {
            'hs_left': [],
            'hs_right': [],
            'to_left': [],
            'to_right': [],
        }

        try:
            event_params = c3d_data['parameters']['EVENT']
            contexts = event_params['CONTEXTS']['value']  # ['RHS', 'LTO', ...]
            times = event_params['TIMES']['value']  # 2D array, row 1 = times
            fps = c3d_data['parameters']['POINT']['RATE']['value'][0]

            for i, ctx in enumerate(contexts):
                ctx = ctx.strip().upper()
                time = times[1, i] if times.ndim > 1 else times[i]
                frame = int(round(time * fps))

                if ctx == 'RHS':
                    events['hs_right'].append(frame)
                elif ctx == 'LHS':
                    events['hs_left'].append(frame)
                elif ctx == 'RTO':
                    events['to_right'].append(frame)
                elif ctx == 'LTO':
                    events['to_left'].append(frame)

        except Exception as exc:
            logger.debug("No readable C3D EVENT entries: %s", exc)

        for key in events:
            events[key] = sorted(events[key])

        return events

    def _compute_hip_center(self, c3d_data, frame: int, side: str) -> Optional[np.ndarray]:
        """Estimate hip center from ASIS/PSIS."""
        asis = self._find_landmark(c3d_data, f'{side}_asis', frame)
        psis = self._find_landmark(c3d_data, f'{side}_psis', frame)

        if asis is not None and psis is not None:
            return (asis + psis) / 2
        return asis or psis

    def extract_file(self, filepath: Path) -> ExtractionResult:
        """Extract data from a C3D file."""
        if not HAS_EZC3D:
            raise RuntimeError("ezc3d not installed")

        c3d = ezc3d.c3d(str(filepath))

        file_info = self._parse_filename(filepath)

        fps = c3d['parameters']['POINT']['RATE']['value'][0]
        n_frames = c3d['data']['points'].shape[2]
        duration_s = n_frames / fps

        events = self._extract_events(c3d)

        angle_frames = []
        for frame in range(n_frames):
            # Get positions - prefer virtual joint centers if available
            left_hip = self._find_landmark(c3d, 'left_hip_jc', frame)
            if left_hip is None:
                left_hip = self._compute_hip_center(c3d, frame, 'left')
            right_hip = self._find_landmark(c3d, 'right_hip_jc', frame)
            if right_hip is None:
                right_hip = self._compute_hip_center(c3d, frame, 'right')

            left_knee = self._find_landmark(c3d, 'left_knee_jc', frame)
            if left_knee is None:
                left_knee = self._find_landmark(c3d, 'left_knee', frame)
            right_knee = self._find_landmark(c3d, 'right_knee_jc', frame)
            if right_knee is None:
                right_knee = self._find_landmark(c3d, 'right_knee', frame)

            left_ankle = self._find_landmark(c3d, 'left_ankle_jc', frame)
            if left_ankle is None:
                left_ankle = self._find_landmark(c3d, 'left_ankle', frame)
            right_ankle = self._find_landmark(c3d, 'right_ankle_jc', frame)
            if right_ankle is None:
                right_ankle = self._find_landmark(c3d, 'right_ankle', frame)

            left_heel = self._find_landmark(c3d, 'left_heel', frame)
            right_heel = self._find_landmark(c3d, 'right_heel', frame)
            left_toe = self._find_landmark(c3d, 'left_toe', frame)
            right_toe = self._find_landmark(c3d, 'right_toe', frame)
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

            # Compute angles
            left_knee_angle = right_knee_angle = 0.0
            left_hip_angle = right_hip_angle = 0.0
            left_ankle_angle = right_ankle_angle = 0.0
            trunk_angle = 0.0

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

            if all(p is not None for p in [left_shoulder, right_shoulder, left_hip, right_hip]):
                shoulder_center = (left_shoulder + right_shoulder) / 2
                hip_center = (left_hip + right_hip) / 2
                trunk_vec = shoulder_center - hip_center
                trunk_angle = np.degrees(np.arctan2(trunk_vec[0], trunk_vec[2]))

            af = AngleFrame(
                frame_index=frame,
                left_hip_angle=left_hip_angle,
                right_hip_angle=right_hip_angle,
                left_knee_angle=left_knee_angle,
                right_knee_angle=right_knee_angle,
                left_ankle_angle=left_ankle_angle,
                right_ankle_angle=right_ankle_angle,
                trunk_angle=trunk_angle,
                pelvis_tilt=0.0,
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
            has_angles=True,
            has_forces=True,  # Force plates available
            hs_frames={'left': events['hs_left'], 'right': events['hs_right']},
            to_frames={'left': events['to_left'], 'right': events['to_right']},
        )

        if has_hs:
            all_hs = sorted(events['hs_left'] + events['hs_right'])
            if len(all_hs) > 1:
                intervals = np.diff(all_hs) / fps
                gt.cadence = 60.0 / np.mean(intervals) if np.mean(intervals) > 0 else 0.0

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
            quality_score=0.9 if has_hs else 0.7,
            warnings=[]
        )


def test_extractor():
    """Quick test of the extractor."""
    if not HAS_EZC3D:
        print("ezc3d not installed")
        return

    data_dir = Path.home() / 'gait_benchmark_project/data/figshare_pd/extracted/C3Dfiles'
    if not data_dir.exists():
        print(f"Directory not found: {data_dir}")
        return

    extractor = FigsharePDExtractor(str(data_dir))
    files = extractor.list_files()
    print(f"Found {len(files)} C3D files")

    if files:
        result = extractor.extract_file(files[0])
        print(f"\nExtracted: {result.source_file}")
        print(f"  Subject: {result.subject_id}, Condition: {result.condition}")
        print(f"  Frames: {result.n_frames}, FPS: {result.fps}")
        print(f"  GT HS L: {len(result.ground_truth.hs_frames['left'])}")
        print(f"  GT HS R: {len(result.ground_truth.hs_frames['right'])}")
        print(f"  GT TO L: {len(result.ground_truth.to_frames['left'])}")
        print(f"  GT TO R: {len(result.ground_truth.to_frames['right'])}")


if __name__ == '__main__':
    test_extractor()
