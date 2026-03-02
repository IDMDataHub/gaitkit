"""
BioCV Dataset Extractor.

Dataset: BioCV (BATH-01258)
URL: https://doi.org/10.15125/BATH-01258
Authors: Evans, Needham, Wade, Parsons, Colyer, McGuigan, Bilzon, Cosker
Publication date: September 2024

Content:
- 15 healthy participants (8F, 7M)
- Activities: walk (10 trials), run (10), CMJ max (5), CMJ self (5), hop (1)
- 200 Hz MoCap (Qualisys + Visual3D processed)
- 9 synchronised cameras at 200 fps
- Ground truth: kinematic HS/TO events (validated 0-frame vs force plate)
- Force plate analog data in C3D

File structure per trial:
    P{XX}/P{XX}_{TYPE}_{NN}/
        markers.c3d              - Visual3D processed (filtered + joint centres)
        markers.events.frame     - Gait events indexed by frame (tab-separated)
        markers.events.time      - Gait events indexed by time
        raw.c3d                  - Raw Qualisys export

Marker set (markers.c3d): 106 markers including Visual3D-computed joint centres:
    LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
    HEEL_L, HEEL_R, TOE_L, TOE_R, ACROM_L, ACROM_R, C7, etc.

Ground truth events (markers.events.frame):
    Tab-separated, columns LHS/RHS/LTO/RTO (+ LOFF/ROFF/LON/RON for force plates)
    Frame indices at 200 Hz, NaN for missing values.

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


class BioCVExtractor(BaseExtractor):
    """Extractor for BioCV dataset (BATH-01258, 15 healthy subjects)."""

    # Marker mapping: C3D label -> standardized landmark name
    # Uses Visual3D-computed joint centres from markers.c3d
    MARKER_MAPPING = {
        # Computed joint centres (Visual3D)
        'LEFT_HIP': 'left_hip',
        'RIGHT_HIP': 'right_hip',
        'LEFT_KNEE': 'left_knee',
        'RIGHT_KNEE': 'right_knee',
        'LEFT_ANKLE': 'left_ankle',
        'RIGHT_ANKLE': 'right_ankle',
        # Foot markers
        'HEEL_L': 'left_heel',
        'HEEL_R': 'right_heel',
        'TOE_L': 'left_toe',
        'TOE_R': 'right_toe',
        # Upper body
        'ACROM_L': 'left_shoulder',
        'ACROM_R': 'right_shoulder',
        'C7': 'c7',
        # Pelvis (for pelvis tilt / midpoint)
        'ASIS_L': 'left_asis',
        'ASIS_R': 'right_asis',
        'PSIS_L': 'left_psis',
        'PSIS_R': 'right_psis',
        'HIP_MIDPOINT': 'hip_midpoint',
    }

    # Trial type patterns that contain gait events
    _VALID_TRIAL_PATTERN = re.compile(
        r'^P\d+_(WALK|RUN|CMJM|CMJS|HOP)_\d+$'
    )

    @property
    def name(self) -> str:
        return "BioCV"

    @property
    def description(self) -> str:
        return "BioCV dataset (15 healthy subjects, walk/run/jump, 200 Hz MoCap)"

    def list_files(self) -> List[Path]:
        """List all markers.c3d files from primary trials."""
        if not HAS_EZC3D:
            return []
        files = list(self.data_dir.rglob('markers.c3d'))
        valid_files = []
        for f in files:
            trial_dir = f.parent.name
            # Exclude ML_*, STATIC_*, calib_*
            if trial_dir.startswith(('ML_', 'calib')):
                continue
            if '_STATIC_' in trial_dir:
                continue
            # Must match valid trial pattern
            if self._VALID_TRIAL_PATTERN.match(trial_dir):
                valid_files.append(f)
        return sorted(valid_files)

    def _parse_trial_info(self, filepath: Path) -> Dict:
        """Extract metadata from trial directory path.

        Path: .../P03/P03_WALK_01/markers.c3d
        Dir name: P{XX}_{TYPE}_{NN}
        """
        info = {
            'subject_id': 'unknown',
            'trial_id': 'unknown',
            'condition': 'unknown',
        }

        trial_dir = filepath.parent.name
        match = re.match(r'(P\d+)_(WALK|RUN|CMJM|CMJS|HOP)_(\d+)', trial_dir)
        if match:
            info['subject_id'] = match.group(1)
            activity = match.group(2).lower()
            trial_num = match.group(3)
            info['trial_id'] = f"{activity}_{trial_num}"
            info['condition'] = activity

        return info

    def _parse_events_file(self, events_path: Path) -> Dict[str, List[int]]:
        """Parse markers.events.frame Visual3D export format.

        The file uses fixed-width (space-padded) columns with multiple
        header rows.  The actual layout (from P03_WALK_01) looks like::

            [row 0]  P03_WALK_01.c3d  P03_WALK_01.c3d  ...
            [row 1]  End  First_Movement  LHS  LOFF  LON  LTO  RHS  ...
            [row 2]  EVENT_LABEL  EVENT_LABEL  ...
            [row 3]  ORIGINAL  ORIGINAL  ...
            [row 4]  ITEM  X  X  X  ...
            [row 5+] 1  942  NaN  450  809  672  360  261  ...

        Strategy: find the header row containing event type names (LHS,
        RHS, etc.), split every row by whitespace, then map columns
        positionally.

        Also handles simpler tab-separated formats (used in unit tests
        and some trials).

        Returns dict with keys hs_left, hs_right, to_left, to_right.
        """
        events = {
            'hs_left': [],
            'hs_right': [],
            'to_left': [],
            'to_right': [],
        }

        if not events_path.exists():
            return events

        try:
            text = events_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            text = events_path.read_text(encoding='latin-1')

        lines = [l for l in text.splitlines() if l.strip()]
        if len(lines) < 2:
            return events

        # --- Locate the header row that contains event type names --------
        EVENT_MAP = {
            'LHS': 'hs_left',
            'RHS': 'hs_right',
            'LTO': 'to_left',
            'RTO': 'to_right',
        }
        header_idx = None
        header_tokens = []
        for i, line in enumerate(lines):
            tokens = line.split()
            token_set = {t.upper() for t in tokens}
            # The header row must contain at least LHS or RHS
            if token_set & {'LHS', 'RHS'}:
                header_idx = i
                header_tokens = tokens
                break

        if header_idx is None:
            # Fallback: try tab-separated first line (unit-test format)
            tokens = lines[0].split('\t')
            tokens = [t.strip() for t in tokens]
            token_set = {t.upper() for t in tokens}
            if token_set & {'LHS', 'RHS'}:
                header_idx = 0
                header_tokens = tokens

        if header_idx is None:
            return events

        # --- Build column map: positional index -> event key -------------
        col_map = {}
        for ci, tok in enumerate(header_tokens):
            key = EVENT_MAP.get(tok.upper().strip())
            if key:
                col_map[ci] = key

        if not col_map:
            return events

        # --- Detect column offset between header and data rows -----------
        # In the Visual3D fixed-width format, the header row has N tokens
        # (End, LHS, LOFF, ...) but data rows have N+1 tokens because the
        # first token is the item number.  Detect this by checking whether
        # the header's first token is a known non-data label.
        first_header = header_tokens[0].upper()
        is_tab_format = '\t' in lines[header_idx]
        # In the space-padded format, first token is an event name (e.g. "End").
        # In the tab format, first token is "Event" (column label).
        # Data rows always start with an item number.
        # We detect the offset when parsing the first data row below.

        # --- Parse data rows (everything after header metadata) ----------
        for line in lines[header_idx + 1:]:
            tokens = line.split() if '\t' not in line else [t.strip() for t in line.split('\t')]
            if not tokens:
                continue
            # Skip non-data rows (EVENT_LABEL, ORIGINAL, ITEM header)
            first = tokens[0].strip()
            is_data_row = False
            data_offset = 0
            try:
                int(first)
                is_data_row = True
                # Data row starts with item number — offset by 1 if
                # header row didn't start with a numeric-like column
                if len(tokens) > len(header_tokens):
                    data_offset = len(tokens) - len(header_tokens)
            except ValueError:
                if first.lower().startswith('item'):
                    # Tab-separated "Item N val val ..." format
                    is_data_row = True
                    data_offset = 0  # "Item" is already col 0 in header too

            if not is_data_row:
                continue

            for col_idx, event_key in col_map.items():
                actual_idx = col_idx + data_offset
                if actual_idx >= len(tokens):
                    continue
                val = tokens[actual_idx].strip()
                if not val or val.lower() == 'nan' or val in ('—', '-'):
                    continue
                try:
                    frame = int(float(val))
                    events[event_key].append(frame)
                except (ValueError, OverflowError):
                    continue

        # Sort
        for key in events:
            events[key] = sorted(events[key])

        return events

    def _get_marker_position(self, c3d_data, marker_name: str, frame: int) -> Optional[np.ndarray]:
        """Get 3D position of a marker at a given frame."""
        try:
            labels = c3d_data['parameters']['POINT']['LABELS']['value']
            for i, label in enumerate(labels):
                if label.strip() == marker_name:
                    pos = c3d_data['data']['points'][:3, i, frame]
                    if not np.any(np.isnan(pos)) and np.linalg.norm(pos) > 0.001:
                        return pos
            return None
        except (KeyError, IndexError, TypeError, ValueError):
            return None

    def _find_landmark(self, c3d_data, landmark: str, frame: int) -> Optional[np.ndarray]:
        """Find landmark position by trying all mapped marker names."""
        for marker, lm in self.MARKER_MAPPING.items():
            if lm == landmark:
                pos = self._get_marker_position(c3d_data, marker, frame)
                if pos is not None:
                    return pos
        return None

    def extract_file(self, filepath: Path) -> ExtractionResult:
        """Extract data from a BioCV trial (markers.c3d + markers.events.frame)."""
        if not HAS_EZC3D:
            raise RuntimeError("ezc3d not installed")

        c3d = ezc3d.c3d(str(filepath))
        file_info = self._parse_trial_info(filepath)

        fps = c3d['parameters']['POINT']['RATE']['value'][0]
        n_frames = c3d['data']['points'].shape[2]
        duration_s = n_frames / fps

        # Parse ground truth events from sibling file
        events_path = filepath.parent / 'markers.events.frame'
        events = self._parse_events_file(events_path)

        warnings = []

        # Build angle frames
        angle_frames = []
        for frame in range(n_frames):
            # Joint centres from Visual3D
            left_hip = self._find_landmark(c3d, 'left_hip', frame)
            right_hip = self._find_landmark(c3d, 'right_hip', frame)
            left_knee = self._find_landmark(c3d, 'left_knee', frame)
            right_knee = self._find_landmark(c3d, 'right_knee', frame)
            left_ankle = self._find_landmark(c3d, 'left_ankle', frame)
            right_ankle = self._find_landmark(c3d, 'right_ankle', frame)

            # Foot markers
            left_heel = self._find_landmark(c3d, 'left_heel', frame)
            right_heel = self._find_landmark(c3d, 'right_heel', frame)
            left_toe = self._find_landmark(c3d, 'left_toe', frame)
            right_toe = self._find_landmark(c3d, 'right_toe', frame)

            # Upper body
            left_shoulder = self._find_landmark(c3d, 'left_shoulder', frame)
            right_shoulder = self._find_landmark(c3d, 'right_shoulder', frame)
            c7 = self._find_landmark(c3d, 'c7', frame)

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

            # Hip flexion: thigh relative to vertical
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

            # Trunk angle from shoulders and hips
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
                pelvis_tilt=pelvis_tilt,
                landmark_positions=landmark_positions,
                is_valid=True,
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
            event_source="annotated",
            hs_frames={'left': events['hs_left'], 'right': events['hs_right']},
            to_frames={'left': events['to_left'], 'right': events['to_right']},
        )

        # Compute cadence from HS events
        if has_hs:
            all_hs = sorted(events['hs_left'] + events['hs_right'])
            if len(all_hs) > 1:
                intervals = np.diff(all_hs) / fps
                valid_intervals = intervals[(intervals > 0.2) & (intervals < 2.0)]
                if len(valid_intervals) > 0:
                    gt.cadence = 60.0 / np.mean(valid_intervals)

        if not has_hs:
            warnings.append("No HS events found in events file")

        total_hs = len(events['hs_left']) + len(events['hs_right'])
        quality = 0.95 if total_hs >= 4 else (0.8 if total_hs >= 2 else 0.6)

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
            quality_score=quality,
            warnings=warnings,
        )


def test_extractor():
    """Quick test of the BioCV extractor."""
    if not HAS_EZC3D:
        print("ezc3d not installed")
        return

    data_dir = Path.home() / 'gait_benchmark_project/data/biocv'
    if not data_dir.exists():
        print(f"Directory not found: {data_dir}")
        return

    extractor = BioCVExtractor(str(data_dir))
    files = extractor.list_files()
    print(f"Found {len(files)} markers.c3d files")

    test_files = files[:3]
    for f in test_files:
        try:
            result = extractor.extract_file(f)
            print(f"\nExtracted: {f.parent.name}")
            print(f"  Subject: {result.subject_id}, Trial: {result.trial_id}")
            print(f"  Condition: {result.condition}")
            print(f"  Frames: {result.n_frames}, FPS: {result.fps}, Duration: {result.duration_s:.1f}s")
            print(f"  GT HS L: {len(result.ground_truth.hs_frames['left'])}")
            print(f"  GT HS R: {len(result.ground_truth.hs_frames['right'])}")
            print(f"  GT TO L: {len(result.ground_truth.to_frames['left'])}")
            print(f"  GT TO R: {len(result.ground_truth.to_frames['right'])}")
            if result.ground_truth.cadence:
                print(f"  Cadence: {result.ground_truth.cadence:.1f} steps/min")
            print(f"  Quality: {result.quality_score}")
            print(f"  Warnings: {result.warnings}")
            if result.angle_frames:
                mid = len(result.angle_frames) // 2
                af = result.angle_frames[mid]
                print(f"  Sample frame {mid}: L_knee={af.left_knee_angle:.1f}, R_knee={af.right_knee_angle:.1f}")
        except Exception as e:
            print(f"\nError on {f.parent.name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    test_extractor()
