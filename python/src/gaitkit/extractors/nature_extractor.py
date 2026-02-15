"""
Nature C3D Extractor.

Bases Nature Scientific Data:
- Multimodal Gait Dataset (50 sujets, 5 vitesses): C3D avec MoCap + EMG + forces
- Multi-sensor Gait Dataset (25 sujets): IMU + MoCap synchronisés

Les fichiers C3D contiennent:
- Positions 3D des marqueurs
- Événements de marche (HS, TO) dans certains fichiers
- Données analogiques (EMG, forces)

Dépendance: pip install ezc3d
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

# Import optionnel de ezc3d
try:
    import ezc3d
    HAS_EZC3D = True
except ImportError:
    HAS_EZC3D = False
    logger.debug("ezc3d not installed. Install with: pip install ezc3d")


class NatureC3DExtractor(BaseExtractor):
    """Extracteur pour les fichiers C3D des bases Nature."""

    # Mapping des marqueurs vers les landmarks
    # Plug-in Gait naming convention
    MARKER_MAPPING_PIG = {
        # Pelvis/Hip
        'LASI': 'left_asis',
        'RASI': 'right_asis',
        'LPSI': 'left_psis',
        'RPSI': 'right_psis',
        'SACR': 'sacrum',
        # Thigh
        'LTHI': 'left_thigh',
        'RTHI': 'right_thigh',
        'LKNE': 'left_knee',
        'RKNE': 'right_knee',
        # Shank
        'LTIB': 'left_tibia',
        'RTIB': 'right_tibia',
        'LANK': 'left_ankle',
        'RANK': 'right_ankle',
        # Foot
        'LHEE': 'left_heel',
        'RHEE': 'right_heel',
        'LTOE': 'left_toe',
        'RTOE': 'right_toe',
        'LMT5': 'left_mt5',
        'RMT5': 'right_mt5',
        # Shoulder (pour trunk)
        'LSHO': 'left_shoulder',
        'RSHO': 'right_shoulder',
        'C7': 'c7',
    }

    # ISB naming convention (Nature Multimodal dataset)
    MARKER_MAPPING_ISB = {
        # Pelvis - ASIS/PSIS
        'L_IAS': 'left_asis',
        'R_IAS': 'right_asis',
        'L_IPS': 'left_psis',
        'R_IPS': 'right_psis',
        # Knee - lateral/medial epicondyles
        'L_FLE': 'left_knee_lat',
        'L_FME': 'left_knee_med',
        'R_FLE': 'right_knee_lat',
        'R_FME': 'right_knee_med',
        # Ankle - fibula/tibia
        'L_FAL': 'left_ankle_lat',
        'L_TAM': 'left_ankle_med',
        'R_FAL': 'right_ankle_lat',
        'R_TAM': 'right_ankle_med',
        # Foot
        'L_FCC': 'left_heel',
        'R_FCC': 'right_heel',
        'L_FM1': 'left_toe',
        'R_FM1': 'right_toe',
        'L_FM2': 'left_toe2',
        'R_FM2': 'right_toe2',
        'L_FM5': 'left_mt5',
        'R_FM5': 'right_mt5',
        # Upper body
        'CV7': 'c7',
        'L_SIA': 'left_shoulder',
        'R_SIA': 'right_shoulder',
    }

    # Generic mapping
    MARKER_MAPPING_GENERIC = {
        'L_HIP': 'left_hip',
        'R_HIP': 'right_hip',
        'L_KNEE': 'left_knee',
        'R_KNEE': 'right_knee',
        'L_ANK': 'left_ankle',
        'R_ANK': 'right_ankle',
        'L_HEEL': 'left_heel',
        'R_HEEL': 'right_heel',
        'L_TOE': 'left_toe',
        'R_TOE': 'right_toe',
    }

    @property
    def name(self) -> str:
        return "Nature C3D"

    @property
    def description(self) -> str:
        return "Bases Nature Scientific Data (C3D MoCap + EMG + forces)"

    def list_files(self) -> List[Path]:
        """Liste tous les fichiers .c3d."""
        if not HAS_EZC3D:
            return []
        files = list(self.data_dir.rglob('*.c3d'))
        # Exclure les fichiers statiques (calibration)
        files = [f for f in files if '_ST' not in f.stem.upper()]
        return sorted(files)

    def _parse_filename(self, filepath: Path) -> Dict:
        """Parse le nom de fichier pour extraire les métadonnées."""
        # Format typique: 2014001_C1_01.c3d
        # 2014001 = subject_id
        # C1 = condition (C1-C5 = different speeds)
        # 01 = trial number

        info = {
            'subject_id': 'unknown',
            'condition': 'unknown',
            'trial': '01',
        }

        filename = filepath.stem

        # Pattern Nature Multimodal
        match = re.match(r'(\d+)_C(\d+)_(\d+)', filename)
        if match:
            info['subject_id'] = match.group(1)
            info['condition'] = f'speed_{match.group(2)}'
            info['trial'] = match.group(3)
            return info

        # Generic pattern
        parts = filename.split('_')
        if len(parts) >= 1:
            info['subject_id'] = parts[0]
        if len(parts) >= 2:
            info['condition'] = parts[1]
        if len(parts) >= 3:
            info['trial'] = parts[2]

        return info

    def _get_marker_position(self, c3d_data, marker_name: str, frame: int) -> Optional[np.ndarray]:
        """Récupère la position 3D d'un marqueur à une frame donnée."""
        try:
            labels = c3d_data['parameters']['POINT']['LABELS']['value']
            # Trouver l'index du marqueur
            for i, label in enumerate(labels):
                if label.strip().upper() == marker_name.upper():
                    pos = c3d_data['data']['points'][:3, i, frame]
                    if not np.any(np.isnan(pos)) and np.linalg.norm(pos) > 0:
                        return pos
            return None
        except:
            return None

    def _find_landmark_position(self, c3d_data, landmark: str, frame: int) -> Optional[np.ndarray]:
        """Trouve la position d'un landmark en essayant plusieurs noms de marqueurs."""
        # Construire la liste des noms possibles depuis tous les mappings
        possible_names = []
        all_mappings = [self.MARKER_MAPPING_PIG, self.MARKER_MAPPING_ISB, self.MARKER_MAPPING_GENERIC]

        for mapping in all_mappings:
            for marker, lm in mapping.items():
                if lm == landmark:
                    possible_names.append(marker)

        # Essayer chaque nom
        for name in possible_names:
            pos = self._get_marker_position(c3d_data, name, frame)
            if pos is not None:
                return pos

        return None

    def _get_joint_center(self, c3d_data, frame: int, side: str, joint: str) -> Optional[np.ndarray]:
        """Calcule le centre d'une articulation depuis les marqueurs lat/med."""
        lat = self._find_landmark_position(c3d_data, f'{side}_{joint}_lat', frame)
        med = self._find_landmark_position(c3d_data, f'{side}_{joint}_med', frame)

        if lat is not None and med is not None:
            return (lat + med) / 2
        elif lat is not None:
            return lat
        elif med is not None:
            return med

        # Fallback: chercher le marqueur direct
        direct = self._find_landmark_position(c3d_data, f'{side}_{joint}', frame)
        return direct

    def _compute_hip_center(self, c3d_data, frame: int, side: str) -> Optional[np.ndarray]:
        """Estime le centre de hanche depuis ASIS/PSIS."""
        asis = self._find_landmark_position(c3d_data, f'{side}_asis', frame)
        psis = self._find_landmark_position(c3d_data, f'{side}_psis', frame)

        if asis is not None and psis is not None:
            # Approximation simple: milieu entre ASIS et PSIS
            return (asis + psis) / 2

        # Fallback: utiliser knee et projeter
        knee = self._find_landmark_position(c3d_data, f'{side}_knee', frame)
        if knee is not None:
            # Estimer la hanche au-dessus du genou
            return knee + np.array([0, 0, 400])  # 40cm au-dessus

        return None

    def _extract_events(self, c3d_data) -> Dict[str, List[int]]:
        """Extrait les événements de marche depuis le C3D."""
        events = {
            'hs_left': [],
            'hs_right': [],
            'to_left': [],
            'to_right': [],
        }

        try:
            event_labels = c3d_data['parameters']['EVENT']['LABELS']['value']
            event_times = c3d_data['parameters']['EVENT']['TIMES']['value']
            fps = c3d_data['parameters']['POINT']['RATE']['value'][0]

            # Try to get contexts (may not exist)
            try:
                event_contexts = c3d_data['parameters']['EVENT']['CONTEXTS']['value']
            except Exception as exc:
                logger.debug("No EVENT CONTEXTS available: %s", exc)
                event_contexts = [''] * len(event_labels)

            for i, label in enumerate(event_labels):
                label_upper = label.strip().upper()
                context = event_contexts[i].strip().upper() if i < len(event_contexts) else ''

                # Get time - handle 2D array format (row 1 contains times)
                if event_times.ndim > 1:
                    time = event_times[1, i]
                else:
                    time = event_times[i]
                frame = int(time * fps)

                # Determine side from context or label
                if 'LEFT' in context or 'L' == context:
                    side = 'left'
                elif 'RIGHT' in context or 'R' == context:
                    side = 'right'
                elif '1' in label_upper:
                    side = 'left'  # Convention: 1 = left, 2 = right
                elif '2' in label_upper:
                    side = 'right'
                else:
                    side = 'left'  # Default

                # Classify event type
                if 'STRIKE' in label_upper or 'HS' in label_upper or 'HEEL' in label_upper or 'IC' in label_upper:
                    events[f'hs_{side}'].append(frame)
                elif 'OFF' in label_upper or 'TO' in label_upper:
                    events[f'to_{side}'].append(frame)

        except Exception as exc:
            logger.debug("No readable C3D EVENT entries: %s", exc)

        # Sort events
        for key in events:
            events[key] = sorted(events[key])

        return events

    def extract_file(self, filepath: Path) -> ExtractionResult:
        """Extrait les données d'un fichier C3D."""
        if not HAS_EZC3D:
            raise RuntimeError("ezc3d not installed")

        # Charger le C3D
        c3d = ezc3d.c3d(str(filepath))

        # Metadata
        file_info = self._parse_filename(filepath)

        # Parameters
        fps = c3d['parameters']['POINT']['RATE']['value'][0]
        n_frames = c3d['data']['points'].shape[2]
        duration_s = n_frames / fps

        # Extract events
        events = self._extract_events(c3d)

        # Create AngleFrames
        angle_frames = []

        for frame in range(n_frames):
            # Retrieve landmark positions
            landmark_positions = {}

            # Hanches (from ASIS/PSIS)
            left_hip = self._compute_hip_center(c3d, frame, 'left')
            right_hip = self._compute_hip_center(c3d, frame, 'right')

            # Genoux (average of lateral/medial epicondyles)
            left_knee = self._get_joint_center(c3d, frame, 'left', 'knee')
            right_knee = self._get_joint_center(c3d, frame, 'right', 'knee')

            # Chevilles (average of lateral/medial malleoli)
            left_ankle = self._get_joint_center(c3d, frame, 'left', 'ankle')
            right_ankle = self._get_joint_center(c3d, frame, 'right', 'ankle')

            # Talons
            left_heel = self._find_landmark_position(c3d, 'left_heel', frame)
            right_heel = self._find_landmark_position(c3d, 'right_heel', frame)

            # Orteils
            left_toe = self._find_landmark_position(c3d, 'left_toe', frame)
            right_toe = self._find_landmark_position(c3d, 'right_toe', frame)

            # Shoulders (for trunk)
            left_shoulder = self._find_landmark_position(c3d, 'left_shoulder', frame)
            right_shoulder = self._find_landmark_position(c3d, 'right_shoulder', frame)

            # Convertir en dict (x, y, z)
            def to_tuple(pos):
                if pos is not None:
                    return (float(pos[0]), float(pos[1]), float(pos[2]))
                return (0.0, 0.0, 0.0)

            landmark_positions = {
                'left_hip': to_tuple(left_hip),
                'right_hip': to_tuple(right_hip),
                'left_knee': to_tuple(left_knee),
                'right_knee': to_tuple(right_knee),
                'left_ankle': to_tuple(left_ankle),
                'right_ankle': to_tuple(right_ankle),
                'left_heel': to_tuple(left_heel),
                'right_heel': to_tuple(right_heel),
                'left_toe': to_tuple(left_toe),
                'right_toe': to_tuple(right_toe),
                'left_shoulder': to_tuple(left_shoulder),
                'right_shoulder': to_tuple(right_shoulder),
            }

            # Calculer les angles
            left_knee_angle = 0.0
            right_knee_angle = 0.0
            left_hip_angle = 0.0
            right_hip_angle = 0.0
            left_ankle_angle = 0.0
            right_ankle_angle = 0.0
            trunk_angle = 0.0

            if left_hip is not None and left_knee is not None and left_ankle is not None:
                left_knee_angle = 180 - compute_angle_from_3points(left_hip, left_knee, left_ankle)

            if right_hip is not None and right_knee is not None and right_ankle is not None:
                right_knee_angle = 180 - compute_angle_from_3points(right_hip, right_knee, right_ankle)

            # Hip angles (relative to vertical)
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

            # Ankle angles
            if left_knee is not None and left_ankle is not None and left_toe is not None:
                left_ankle_angle = compute_angle_from_3points(left_knee, left_ankle, left_toe) - 90

            if right_knee is not None and right_ankle is not None and right_toe is not None:
                right_ankle_angle = compute_angle_from_3points(right_knee, right_ankle, right_toe) - 90

            # Trunk angle
            if left_shoulder is not None and right_shoulder is not None and left_hip is not None and right_hip is not None:
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
            has_angles=True,  # Angles computed from MoCap
            has_forces=False,  # To be verified in analog data
            has_emg=False,
            hs_frames={'left': events['hs_left'], 'right': events['hs_right']},
            to_frames={'left': events['to_left'], 'right': events['to_right']},
        )

        # Calculer cadence si HS disponibles
        if has_hs:
            all_hs = sorted(events['hs_left'] + events['hs_right'])
            if len(all_hs) > 1:
                intervals = np.diff(all_hs) / fps
                mean_step = np.mean(intervals)
                gt.cadence = 60.0 / mean_step if mean_step > 0 else 0.0

        warnings = []
        if not has_hs:
            warnings.append("Pas d'événements HS dans le fichier")
        warnings.append(f"FPS: {fps}, Frames: {n_frames}")

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
            warnings=warnings
        )


def test_extractor():
    """Test rapide de l'extracteur."""
    if not HAS_EZC3D:
        print("ezc3d not installed. Install with: pip install ezc3d")
        return

    data_dir = Path.home() / 'gait_benchmark_project/data/nature/extracted'
    if not data_dir.exists():
        print(f"Directory not found: {data_dir}")
        return

    extractor = NatureC3DExtractor(str(data_dir))
    files = extractor.list_files()
    print(f"Found {len(files)} C3D files")

    if files:
        result = extractor.extract_file(files[0])
        print(f"\nExtracted: {result.source_file}")
        print(f"  Subject: {result.subject_id}, Condition: {result.condition}")
        print(f"  Frames: {result.n_frames}, FPS: {result.fps}")
        print(f"  Has HS: {result.ground_truth.has_hs}")
        print(f"  Warnings: {result.warnings}")


if __name__ == '__main__':
    test_extractor()
