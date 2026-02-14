"""
Schreiber Extractor -- Multimodal Gait Dataset at Different Walking Speeds.

Dataset: A multimodal dataset of human gait at different walking speeds
         (same data as Nature Scientific Data 2019)
Authors: Schreiber C, Moissenet F
Year: 2019
DOI: 10.6084/m9.figshare.7734767.v2

Content:
- 50 healthy adult subjects
- 5 walking speeds (C1=very slow to C5=very fast), 5 trials each
- C3D with ISB markers + EMG + force plates
- Ground truth events: Foot Strike1, Foot Strike2, Foot Off (with Left/Right CONTEXTS)

This extractor extends the NatureC3DExtractor with corrected event parsing.

Dependency: pip install ezc3d
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import re

from .base_extractor import (
    BaseExtractor, ExtractionResult, AngleFrame, GroundTruth,
    compute_angle_from_3points
)

try:
    import ezc3d
    HAS_EZC3D = True
except ImportError:
    HAS_EZC3D = False
    print("Warning: ezc3d not installed. Install with: pip install ezc3d")


class SchreiberExtractor(BaseExtractor):
    """Extractor for Schreiber/Moissenet multimodal gait dataset (ISB markers)."""

    # ISB marker naming convention
    MARKER_MAPPING = {
        # Pelvis
        'L_IAS': 'left_asis', 'R_IAS': 'right_asis',
        'L_IPS': 'left_psis', 'R_IPS': 'right_psis',
        # Knee
        'L_FLE': 'left_knee_lat', 'L_FME': 'left_knee_med',
        'R_FLE': 'right_knee_lat', 'R_FME': 'right_knee_med',
        # Ankle
        'L_FAL': 'left_ankle_lat', 'L_TAM': 'left_ankle_med',
        'R_FAL': 'right_ankle_lat', 'R_TAM': 'right_ankle_med',
        # Foot
        'L_FCC': 'left_heel', 'R_FCC': 'right_heel',
        'L_FM1': 'left_toe', 'R_FM1': 'right_toe',
        'L_FM2': 'left_toe2', 'R_FM2': 'right_toe2',
        'L_FM5': 'left_mt5', 'R_FM5': 'right_mt5',
        # Upper body
        'CV7': 'c7', 'TV10': 't10',
        'SXS': 'sternum', 'SJN': 'jugular_notch',
        'L_SAE': 'left_shoulder', 'R_SAE': 'right_shoulder',
        # Thigh (for femoral landmarks)
        'L_FTC': 'left_thigh_cluster', 'R_FTC': 'right_thigh_cluster',
        'L_FAX': 'left_thigh_axis', 'R_FAX': 'right_thigh_axis',
        # Tibia
        'L_TTC': 'left_tibia_cluster', 'R_TTC': 'right_tibia_cluster',
    }

    @property
    def name(self) -> str:
        return "Schreiber Multimodal"

    @property
    def description(self) -> str:
        return "50 healthy subjects, 5 speeds (ISB markers, C3D + EMG + forces)"

    def list_files(self) -> List[Path]:
        """List all walking C3D files (exclude static)."""
        if not HAS_EZC3D:
            return []
        files = list(self.data_dir.rglob('*.c3d'))
        # Exclude static files
        files = [f for f in files if '_ST' not in f.stem.upper()]
        return sorted(files)

    def _parse_filename(self, filepath: Path) -> Dict:
        """Parse filename for metadata.

        Format: 2014051/2014051_C3_05.c3d
        - 2014051 = subject_id
        - C3 = condition (speed C1-C5)
        - 05 = trial number
        """
        info = {
            'subject_id': 'unknown',
            'condition': 'unknown',
            'trial': '01',
            'speed': 'unknown',
        }

        filename = filepath.stem

        match = re.match(r'(\d+)_C(\d+)_(\d+)', filename)
        if match:
            info['subject_id'] = match.group(1)
            speed_num = int(match.group(2))
            info['condition'] = f'speed_{speed_num}'
            info['trial'] = match.group(3)
            speed_names = {1: 'very_slow', 2: 'slow', 3: 'normal', 4: 'fast', 5: 'very_fast'}
            info['speed'] = speed_names.get(speed_num, f'speed_{speed_num}')

        return info

    def _get_marker_position(self, c3d_data, marker_name: str, frame: int) -> Optional[np.ndarray]:
        """Get 3D position of a marker at a given frame."""
        try:
            labels = c3d_data['parameters']['POINT']['LABELS']['value']
            for i, label in enumerate(labels):
                if label.strip() == marker_name:
                    pos = c3d_data['data']['points'][:3, i, frame]
                    if not np.any(np.isnan(pos)) and np.linalg.norm(pos) > 0:
                        return pos
            return None
        except Exception:
            return None

    def _find_landmark(self, c3d_data, landmark: str, frame: int) -> Optional[np.ndarray]:
        """Find landmark position by trying marker names."""
        for marker, lm in self.MARKER_MAPPING.items():
            if lm == landmark:
                pos = self._get_marker_position(c3d_data, marker, frame)
                if pos is not None:
                    return pos
        return None

    def _get_joint_center(self, c3d_data, frame: int, side: str, joint: str) -> Optional[np.ndarray]:
        """Compute joint center from lateral + medial markers."""
        lat = self._find_landmark(c3d_data, f'{side}_{joint}_lat', frame)
        med = self._find_landmark(c3d_data, f'{side}_{joint}_med', frame)

        if lat is not None and med is not None:
            return (lat + med) / 2
        return lat or med

    def _compute_hip_center(self, c3d_data, frame: int, side: str) -> Optional[np.ndarray]:
        """Estimate hip center from ASIS/PSIS."""
        asis = self._find_landmark(c3d_data, f'{side}_asis', frame)
        psis = self._find_landmark(c3d_data, f'{side}_psis', frame)

        if asis is not None and psis is not None:
            return (asis + psis) / 2
        return asis or psis

    def _extract_events(self, c3d_data) -> Dict[str, List[int]]:
        """Extract gait events from C3D EVENT group.

        Schreiber uses: Foot Strike1, Foot Strike2, Foot Off
        with Left/Right CONTEXTS.
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

                if times.ndim > 1:
                    time = times[1, i]
                else:
                    time = times[i]
                # Convert to frame index relative to data array
                frame = int(round(time * fps)) - first_frame
                if frame < 0 or frame >= n_frames:
                    continue  # Skip events outside data range

                # Determine side
                if 'LEFT' in context or context == 'L':
                    side = 'left'
                elif 'RIGHT' in context or context == 'R':
                    side = 'right'
                else:
                    continue

                # Classify: Foot Strike (any variant) = HS, Foot Off = TO
                if 'STRIKE' in label_upper or 'FOOT STRIKE' in label_upper:
                    events[f'hs_{side}'].append(frame)
                elif 'OFF' in label_upper:
                    events[f'to_{side}'].append(frame)

        except Exception:
            pass

        for key in events:
            events[key] = sorted(list(set(events[key])))

        return events

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
            # Get joint positions
            left_hip = self._compute_hip_center(c3d, frame, 'left')
            right_hip = self._compute_hip_center(c3d, frame, 'right')
            left_knee = self._get_joint_center(c3d, frame, 'left', 'knee')
            right_knee = self._get_joint_center(c3d, frame, 'right', 'knee')
            left_ankle = self._get_joint_center(c3d, frame, 'left', 'ankle')
            right_ankle = self._get_joint_center(c3d, frame, 'right', 'ankle')
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
                sc = (left_shoulder + right_shoulder) / 2
                hc = (left_hip + right_hip) / 2
                tv = sc - hc
                trunk_angle = np.degrees(np.arctan2(tv[0], tv[2]))

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
            has_forces=True,
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
            trial_id=f"{file_info['speed']}_{file_info['trial']}",
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
    """Quick test."""
    if not HAS_EZC3D:
        print("ezc3d not installed")
        return

    data_dir = Path.home() / 'gait_benchmark_project/data/schreiber/A multimodal dataset of human gait at different walking speeds'
    if not data_dir.exists():
        print(f"Not found: {data_dir}")
        return

    ext = SchreiberExtractor(str(data_dir))
    files = ext.list_files()
    print(f"Found {len(files)} C3D files")

    if files:
        result = ext.extract_file(files[0])
        print(f"  File: {result.source_file}")
        print(f"  Subject: {result.subject_id}, Trial: {result.trial_id}")
        print(f"  Condition: {result.condition}")
        print(f"  Frames: {result.n_frames}, FPS: {result.fps}")
        print(f"  GT HS L: {len(result.ground_truth.hs_frames['left'])}")
        print(f"  GT HS R: {len(result.ground_truth.hs_frames['right'])}")
        print(f"  GT TO L: {len(result.ground_truth.to_frames['left'])}")
        print(f"  GT TO R: {len(result.ground_truth.to_frames['right'])}")


if __name__ == '__main__':
    test_extractor()
