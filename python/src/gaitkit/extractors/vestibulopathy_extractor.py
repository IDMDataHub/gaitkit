"""
Vestibulopathy Extractor -- Bilateral/Unilateral Vestibulopathy Gait Dataset.

Dataset: Gait analysis in patients with vestibular disorders
Source: Zenodo 14179651
Content:
- 240 C3D gait trials (excl. static), 30 subjects (10 BV + 10 UV + 10 HS)
- CGM 1.0 markers (120 channels), 100 Hz
- 3 force plates
- 8 walking conditions: ChangeSpeed, DoubleTask, EyesClosed, HeadHorTurns,
  HeadVerTurns, Obstacle, UTurn, (Static excluded)
- Events: Foot Strike / Foot Off with Left/Right CONTEXTS (~30 per trial)

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


class VestibulopathyExtractor(BaseExtractor):
    """Extractor for vestibulopathy gait dataset (CGM 1.0 / PiG markers)."""

    MARKER_MAPPING = {
        'LHEE': 'left_heel', 'RHEE': 'right_heel',
        'LTOE': 'left_toe', 'RTOE': 'right_toe',
        'LANK': 'left_ankle', 'RANK': 'right_ankle',
        'LKNE': 'left_knee', 'RKNE': 'right_knee',
        'LASI': 'left_asis', 'RASI': 'right_asis',
        'LPSI': 'left_psis', 'RPSI': 'right_psis',
        'LTHI': 'left_thigh', 'RTHI': 'right_thigh',
        'LTIB': 'left_tibia', 'RTIB': 'right_tibia',
    }

    GROUP_POPULATION = {
        'BV': 'bilateral_vestibulopathy',
        'UV': 'unilateral_vestibulopathy',
        'HS': 'healthy',
    }

    @property
    def name(self) -> str:
        return "Vestibulopathy"

    @property
    def description(self) -> str:
        return "Vestibular disorders: 240 trials, 30 subjects, 8 conditions, 100 Hz CGM"

    def list_files(self) -> List[Path]:
        if not HAS_EZC3D:
            return []
        files = list(self.data_dir.rglob('*.c3d'))
        files = [f for f in files
                 if 'static' not in f.stem.lower()
                 and 'Static' not in f.stem]
        return sorted(files)

    def _parse_filename(self, filepath: Path) -> Dict:
        """Parse filename for metadata.

        Format: {GROUP}_{NN}_{CONDITION}_{TT}.c3d
        e.g. BV_01_ChangeSpeed_01.c3d
        """
        info = {
            'subject_id': 'unknown',
            'condition': 'unknown',
            'trial_id': '01',
            'population': 'unknown',
        }

        filename = filepath.stem
        m = re.match(r'(BV|UV|HS)_(\d+)_([A-Za-z]+)_(\d+)', filename)
        if m:
            group = m.group(1)
            subject_num = m.group(2)
            condition = m.group(3)
            trial_num = m.group(4)

            info['subject_id'] = f'{group}_{subject_num}'
            info['condition'] = condition.lower()
            info['trial_id'] = trial_num
            info['population'] = self.GROUP_POPULATION.get(group, 'unknown')

        return info

    def _get_marker_position(self, c3d_data, marker_name: str, frame: int) -> Optional[np.ndarray]:
        try:
            labels = c3d_data['parameters']['POINT']['LABELS']['value']
            for i, label in enumerate(labels):
                if label.strip() == marker_name:
                    pos = c3d_data['data']['points'][:3, i, frame]
                    if not np.any(np.isnan(pos)) and np.linalg.norm(pos) > 0:
                        return pos
            return None
        except (KeyError, IndexError, TypeError, ValueError):
            return None

    def _find_landmark(self, c3d_data, landmark: str, frame: int) -> Optional[np.ndarray]:
        for marker, lm in self.MARKER_MAPPING.items():
            if lm == landmark:
                pos = self._get_marker_position(c3d_data, marker, frame)
                if pos is not None:
                    return pos
        return None

    def _compute_hip_center(self, c3d_data, frame: int, side: str) -> Optional[np.ndarray]:
        asis = self._find_landmark(c3d_data, f'{side}_asis', frame)
        psis = self._find_landmark(c3d_data, f'{side}_psis', frame)
        if asis is not None and psis is not None:
            return (asis + psis) / 2
        return asis or psis

    def _extract_events(self, c3d_data) -> Dict[str, List[int]]:
        events = {'hs_left': [], 'hs_right': [], 'to_left': [], 'to_right': []}

        try:
            event_params = c3d_data['parameters']['EVENT']
            labels = event_params['LABELS']['value']
            contexts = event_params['CONTEXTS']['value']
            times = event_params['TIMES']['value']
            fps = c3d_data['parameters']['POINT']['RATE']['value'][0]
            first_frame = c3d_data['header']['points']['first_frame']
            n_frames = c3d_data['data']['points'].shape[2]

            for i, label in enumerate(labels):
                label_upper = label.strip().upper()
                context = contexts[i].strip().upper() if i < len(contexts) else ''

                if times.ndim > 1:
                    time = times[1, i]
                else:
                    time = times[i]
                frame = int(round(time * fps)) - first_frame
                if frame < 0 or frame >= n_frames:
                    continue

                if 'LEFT' in context or context == 'L':
                    side = 'left'
                elif 'RIGHT' in context or context == 'R':
                    side = 'right'
                else:
                    continue

                if 'STRIKE' in label_upper:
                    events[f'hs_{side}'].append(frame)
                elif 'OFF' in label_upper:
                    events[f'to_{side}'].append(frame)

        except Exception as exc:
            logger.debug("Failed to parse C3D EVENT entries: %s", exc)

        for key in events:
            events[key] = sorted(list(set(events[key])))
        return events

    def extract_file(self, filepath: Path) -> ExtractionResult:
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
            }

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

        has_hs = len(events['hs_left']) > 0 or len(events['hs_right']) > 0
        has_to = len(events['to_left']) > 0 or len(events['to_right']) > 0

        gt = GroundTruth(
            has_hs=has_hs,
            has_to=has_to,
            has_cadence=has_hs,
            has_angles=True,
            has_forces=True,
            event_source="force_plate",
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
            trial_id=file_info['trial_id'],
            condition=file_info['condition'],
            fps=fps,
            n_frames=n_frames,
            duration_s=duration_s,
            angle_frames=angle_frames,
            ground_truth=gt,
            quality_score=0.9 if has_hs else 0.7,
        )
