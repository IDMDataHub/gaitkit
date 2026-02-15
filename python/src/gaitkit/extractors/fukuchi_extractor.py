"""
Fukuchi Walking Dataset Extractor.

Dataset: A public data set of overground and treadmill walking kinematics
         and kinetics of healthy individuals
URL: https://doi.org/10.6084/m9.figshare.5722711.v4

Content:
- 42 subjects (24 young adults + 18 older adults)
- 1969 C3D walking files with ground truth events
- Ground truth events: LHS, RHS, LTO, RTO, LON, RON, LOFF, ROFF
- Overground (walkO) and treadmill (walkT) conditions
- Multiple speeds: C (comfortable), F (fast), S (slow)

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


class FukuchiExtractor(BaseExtractor):
    """Extractor for Fukuchi Walking Dataset."""

    # Marker mapping
    MARKER_MAPPING = {
        # Pelvis
        'R.ASIS': 'right_asis', 'L.ASIS': 'left_asis',
        'R.PSIS': 'right_psis', 'L.PSIS': 'left_psis',
        'R.Iliac.Crest': 'right_iliac', 'L.Iliac.Crest': 'left_iliac',
        # Thigh / Greater trochanter
        'R.GTR': 'right_thigh', 'L.GTR': 'left_thigh',
        'R.HF': 'right_thigh_front', 'L.HF': 'left_thigh_front',
        # Knee
        'R.Knee': 'right_knee', 'L.Knee': 'left_knee',
        # Shank (tibial tuberosity)
        'R.TT': 'right_tibia', 'L.TT': 'left_tibia',
        # Ankle
        'R.Ankle': 'right_ankle', 'L.Ankle': 'left_ankle',
        # Foot
        'R.Heel': 'right_heel', 'L.Heel': 'left_heel',
        'R.MT1': 'right_toe', 'L.MT1': 'left_toe',
        'R.MT5': 'right_mt5', 'L.MT5': 'left_mt5',
    }

    @property
    def name(self) -> str:
        return "Fukuchi Walking"

    @property
    def description(self) -> str:
        return "42 healthy subjects (young+older), overground & treadmill walking"

    def list_files(self) -> List[Path]:
        """List all walking C3D files (exclude static)."""
        if not HAS_EZC3D:
            return []
        # Look in c3d_with_events subdirectory
        c3d_dir = self.data_dir / 'c3d_with_events'
        if not c3d_dir.exists():
            c3d_dir = self.data_dir
        files = list(c3d_dir.rglob('WBDS*walk*.c3d'))
        return sorted(files)

    def _parse_filename(self, filepath: Path) -> Dict:
        """Parse filename for metadata.

        Format: WBDS01walkO01C.c3d
        - WBDS01 = subject ID (01-42)
        - walkO/walkT = overground/treadmill
        - 01 = trial number
        - C/F/S = comfortable/fast/slow speed
        """
        info = {
            'subject_id': 'unknown',
            'condition': 'healthy',
            'trial': '01',
            'walking_type': 'unknown',
            'speed': 'comfortable',
        }

        filename = filepath.stem

        # Extract subject ID
        match = re.match(r'WBDS(\d+)', filename)
        if match:
            subj_num = int(match.group(1))
            info['subject_id'] = f'WBDS{subj_num:02d}'
            # Subjects 1-24 are young, 25-42 are older
            if subj_num <= 24:
                info['condition'] = 'healthy_young'
            else:
                info['condition'] = 'healthy_older'

        # Extract walking type (O=overground, T=treadmill)
        if 'walkO' in filename:
            info['walking_type'] = 'overground'
        elif 'walkT' in filename:
            info['walking_type'] = 'treadmill'

        # Extract trial number and speed
        trial_match = re.search(r'walk[OT](\d+)([CFS])', filename)
        if trial_match:
            info['trial'] = trial_match.group(1)
            speed_code = trial_match.group(2)
            speed_map = {'C': 'comfortable', 'F': 'fast', 'S': 'slow'}
            info['speed'] = speed_map.get(speed_code, 'unknown')

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
        except:
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

        Events in Fukuchi: LHS, RHS, LTO, RTO, LON, RON, LOFF, ROFF
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
            times = event_params['TIMES']['value']  # 2D array [2, n_events]
            fps = c3d_data['parameters']['POINT']['RATE']['value'][0]
            first_frame = c3d_data['parameters']['POINT']['FRAMES']['value'][0] if 'FRAMES' in c3d_data['parameters']['POINT'] else 0

            for i, label in enumerate(labels):
                label = label.strip().upper()
                # Times is [2, n_events], row 0 is minutes, row 1 is seconds
                if times.ndim > 1:
                    time = times[1, i]
                else:
                    time = times[i]
                frame = int(round(time * fps))

                # Map event labels
                if label in ['RHS', 'RON']:  # Right heel strike or right foot on
                    events['hs_right'].append(frame)
                elif label in ['LHS', 'LON']:  # Left heel strike or left foot on
                    events['hs_left'].append(frame)
                elif label in ['RTO', 'ROFF']:  # Right toe off or right foot off
                    events['to_right'].append(frame)
                elif label in ['LTO', 'LOFF']:  # Left toe off or left foot off
                    events['to_left'].append(frame)

        except Exception as exc:
            logger.debug("No readable C3D EVENT entries: %s", exc)

        for key in events:
            events[key] = sorted(list(set(events[key])))  # Remove duplicates and sort

        return events

    def _compute_hip_center(self, c3d_data, frame: int, side: str) -> Optional[np.ndarray]:
        """Estimate hip center from ASIS/PSIS/GTR."""
        asis = self._find_landmark(c3d_data, f'{side}_asis', frame)
        psis = self._find_landmark(c3d_data, f'{side}_psis', frame)
        gtr = self._find_landmark(c3d_data, f'{side}_thigh', frame)

        # Use GTR (greater trochanter) as hip approximation if available
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

            def to_tuple(pos):
                return (float(pos[0]), float(pos[1]), float(pos[2])) if pos is not None else (0.0, 0.0, 0.0)

            landmark_positions = {
                'left_hip': to_tuple(left_hip), 'right_hip': to_tuple(right_hip),
                'left_knee': to_tuple(left_knee), 'right_knee': to_tuple(right_knee),
                'left_ankle': to_tuple(left_ankle), 'right_ankle': to_tuple(right_ankle),
                'left_heel': to_tuple(left_heel), 'right_heel': to_tuple(right_heel),
                'left_toe': to_tuple(left_toe), 'right_toe': to_tuple(right_toe),
                'left_shoulder': (0.0, 0.0, 0.0), 'right_shoulder': (0.0, 0.0, 0.0),
            }

            # Compute angles
            left_knee_angle = right_knee_angle = 0.0
            left_hip_angle = right_hip_angle = 0.0
            left_ankle_angle = right_ankle_angle = 0.0

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

            af = AngleFrame(
                frame_index=frame,
                left_hip_angle=left_hip_angle,
                right_hip_angle=right_hip_angle,
                left_knee_angle=left_knee_angle,
                right_knee_angle=right_knee_angle,
                left_ankle_angle=left_ankle_angle,
                right_ankle_angle=right_ankle_angle,
                trunk_angle=0.0,
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
            trial_id=f"{file_info['walking_type']}_{file_info['trial']}_{file_info['speed']}",
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

    data_dir = Path.home() / 'gait_benchmark_project/data/fukuchi'
    if not data_dir.exists():
        print(f"Directory not found: {data_dir}")
        return

    extractor = FukuchiExtractor(str(data_dir))
    files = extractor.list_files()
    print(f"Found {len(files)} C3D files")

    if files:
        result = extractor.extract_file(files[0])
        print(f"\nExtracted: {result.source_file}")
        print(f"  Subject: {result.subject_id}, Condition: {result.condition}")
        print(f"  Trial: {result.trial_id}")
        print(f"  Frames: {result.n_frames}, FPS: {result.fps}")
        print(f"  GT HS L: {len(result.ground_truth.hs_frames['left'])}")
        print(f"  GT HS R: {len(result.ground_truth.hs_frames['right'])}")
        print(f"  GT TO L: {len(result.ground_truth.to_frames['left'])}")
        print(f"  GT TO R: {len(result.ground_truth.to_frames['right'])}")


if __name__ == '__main__':
    test_extractor()
