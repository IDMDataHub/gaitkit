"""
Hood Amputee Extractor -- Above-Knee (Transfemoral) Amputees.

Dataset: Kinematic and kinetic gait data of above-knee amputees
         walking at different speeds on a treadmill.
Source: https://doi.org/10.6084/m9.figshare.c.4494755

Content:
- 13 transfemoral amputees (TF07-TF20)
- 364 C3D files from treadmill walking at different speeds (0.6-1.4 m/s)
- Ground truth events: 'Foot Strike' and 'Foot Off' with Left/Right CONTEXTS
- 200 Hz sampling rate
- Full-body Vicon marker set (LANK, RANK, LHEE, RHEE, LTOE, RTOE, etc.)

Directory structure:
    hood_amputee/TF{nn}/Vicon Workspace/{speed}_{trial}.c3d
    e.g. TF07/Vicon Workspace/0p6_01.c3d (0.6 m/s, trial 01)

Dependency: pip install ezc3d
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
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


class HoodAmputeeExtractor(BaseExtractor):
    """Extractor for Hood Above-Knee Amputee Dataset (Vicon markers)."""

    # Vicon marker mapping
    MARKER_MAPPING = {
        # Pelvis
        'LASI': 'left_asis', 'RASI': 'right_asis',
        'LPSI': 'left_psis', 'RPSI': 'right_psis',
        # Greater trochanter (hip approximation)
        'LGTR': 'left_thigh', 'RGTR': 'right_thigh',
        # Knee
        'LKNE': 'left_knee', 'RKNE': 'right_knee',
        # Ankle
        'LANK': 'left_ankle', 'RANK': 'right_ankle',
        # Foot
        'LHEE': 'left_heel', 'RHEE': 'right_heel',
        'LTOE': 'left_toe', 'RTOE': 'right_toe',
        'L5FT': 'left_mt5', 'R5FT': 'right_mt5',
        # Upper body
        'C7': 'c7', 'T10': 't10',
        'CLAV': 'clavicle', 'STRN': 'sternum',
        'LSHO': 'left_shoulder', 'RSHO': 'right_shoulder',
        # Shank
        'LTIB': 'left_tibia', 'RTIB': 'right_tibia',
    }

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = str(Path.home() / 'gait_benchmark_project/data/hood_amputee')
        super().__init__(data_dir)

    @property
    def name(self) -> str:
        return "Hood Amputee"

    @property
    def description(self) -> str:
        return "13 transfemoral amputees, treadmill walking at multiple speeds"

    def list_files(self) -> List[Path]:
        """List all walking C3D files (exclude static/functional calibrations)."""
        if not HAS_EZC3D:
            return []
        files = list(self.data_dir.rglob('*.c3d'))
        # Exclude functional calibration and static files
        files = [f for f in files
                 if 'functional' not in f.stem.lower()
                 and 'static' not in f.stem.lower()
                 and 'cal' not in f.stem.lower()]
        return sorted(files)

    def _parse_filename(self, filepath: Path) -> Dict:
        """Parse filename and directory for metadata.

        Path: .../TF07/Vicon Workspace/0p6_01.c3d
        - TF07 = subject ID (transfemoral amputee #7)
        - 0p6 = 0.6 m/s treadmill speed
        - 01 = trial number
        """
        info = {
            'subject_id': 'unknown',
            'condition': 'amputee',
            'trial': '01',
            'speed': 'unknown',
            'speed_ms': 0.0,
        }

        # Extract subject from parent directories
        for parent in filepath.parents:
            match = re.match(r'(TF\d+)', parent.name)
            if match:
                info['subject_id'] = match.group(1)
                break

        # Extract speed and trial from filename: 0p6_01.c3d
        filename = filepath.stem
        speed_match = re.match(r'(\d+)p(\d+)_(\d+)', filename)
        if speed_match:
            whole = speed_match.group(1)
            decimal = speed_match.group(2)
            info['speed_ms'] = float(f'{whole}.{decimal}')
            info['speed'] = f'{whole}p{decimal}'
            info['trial'] = speed_match.group(3)

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

    def _extract_events(self, c3d_data) -> Dict[str, List[int]]:
        """Extract gait events from C3D EVENT group.

        Events: 'Foot Strike' + 'Foot Off' with Left/Right CONTEXTS.
        Frame indices are computed from event times and adjusted for first_frame offset.
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

                # Get time (row 0 = minutes, row 1 = seconds)
                if times.ndim > 1:
                    time = times[1, i]
                else:
                    time = times[i]
                # Convert to frame index relative to data array
                frame = int(round(time * fps)) - first_frame
                if frame < 0 or frame >= n_frames:
                    continue

                # Determine side
                if 'LEFT' in context:
                    side = 'left'
                elif 'RIGHT' in context:
                    side = 'right'
                else:
                    continue

                # Classify event type
                if 'STRIKE' in label_upper:
                    events[f'hs_{side}'].append(frame)
                elif 'OFF' in label_upper:
                    events[f'to_{side}'].append(frame)

        except Exception:
            pass

        for key in events:
            events[key] = sorted(list(set(events[key])))

        return events

    def _compute_hip_center(self, c3d_data, frame: int, side: str) -> Optional[np.ndarray]:
        """Estimate hip center from ASIS/PSIS or greater trochanter."""
        asis = self._find_landmark(c3d_data, f'{side}_asis', frame)
        psis = self._find_landmark(c3d_data, f'{side}_psis', frame)
        gtr = self._find_landmark(c3d_data, f'{side}_thigh', frame)

        # Use greater trochanter as hip approximation if available
        if gtr is not None:
            return gtr
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
        warnings = []

        for frame in range(n_frames):
            # Get positions
            left_hip = self._compute_hip_center(c3d, frame, 'left')
            right_hip = self._compute_hip_center(c3d, frame, 'right')
            left_knee = self._find_landmark(c3d, 'left_knee', frame)
            right_knee = self._find_landmark(c3d, 'right_knee', frame)
            left_ankle = self._find_landmark(c3d, 'left_ankle', frame)
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

            # Compute angles from markers
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

        if not has_hs:
            warnings.append("No HS events found")

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
            warnings=warnings
        )


def test_extractor():
    """Quick test of the extractor."""
    if not HAS_EZC3D:
        print("ezc3d not installed")
        return

    ext = HoodAmputeeExtractor()
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
        print(f"  Warnings: {result.warnings}")


if __name__ == '__main__':
    test_extractor()
