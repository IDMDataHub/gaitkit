"""
Kuopio Gait Dataset Extractor.

Dataset: Kuopio gait dataset
URL: https://doi.org/10.1038/s41597-024-03882-2
Publication: Reijonen et al. 2024

Content:
- 51 healthy subjects (currently subjects 01-17 extracted)
- 3 speeds: comfortable (comf), slow, fast
- Overground walking with 5 force plates in a walkway
- 100 Hz marker data, 1000 Hz analog data
- MoCap + IMU + Video (we use MoCap C3D only)
- Ground truth: gait events from force plates (HS/TO)

Marker set (cluster-based with model-computed virtual joints):
    Physical clusters: Pelvis1-4, Torso1-4, LFemur1-6, RFemur1-6,
                       LTibia1-6, RTibia1-6, LFoot1-5, RFoot1-5
    Virtual joint centers (from SCORE/SARA functional calibration):
        Pelvis_LFemur_score  -> left hip joint center
        Pelvis_RFemur_score  -> right hip joint center
        LFemur_LTibia_score  -> left knee joint center (= LKnee)
        RFemur_RTibia_score  -> right knee joint center (= RKnee)
        LTibia_LFoot_score   -> left ankle joint center
        RTibia_RFoot_score   -> right ankle joint center

Force plates: Type 2 (already calibrated in N/Nmm), 5 plates.
    Plates 1-3 form the walking path; plates 4-5 are typically unused.
    Foot assignment is determined by comparing foot marker positions
    against force plate corner positions at the time of contact.

File naming: {side}_{speed}_{trial}.c3d
    e.g., l_comf_01.c3d = left-start, comfortable speed, trial 1
    Calibration files (calib_*, mass.*) are excluded.

Dependency: pip install ezc3d
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
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


class KuopioExtractor(BaseExtractor):
    """Extractor for Kuopio gait dataset (overground walking)."""

    # Marker mapping: C3D label -> standardized landmark name
    # Uses the SCORE/SARA functional joint centers when available
    MARKER_MAPPING = {
        # Hip joint centers (from functional calibration)
        'Pelvis_LFemur_score': 'left_hip',
        'Pelvis_RFemur_score': 'right_hip',
        # Knee joint centers
        'LKnee': 'left_knee',
        'RKnee': 'right_knee',
        'LFemur_LTibia_score': 'left_knee_score',
        'RFemur_RTibia_score': 'right_knee_score',
        # Ankle joint centers
        'LTibia_LFoot_score': 'left_ankle',
        'RTibia_RFoot_score': 'right_ankle',
        # Pelvis cluster markers (for pelvis center estimation)
        'Pelvis1': 'pelvis1', 'Pelvis2': 'pelvis2',
        'Pelvis3': 'pelvis3', 'Pelvis4': 'pelvis4',
        # Foot markers (for heel/toe approximation)
        'LFoot1': 'left_foot1', 'LFoot2': 'left_foot2',
        'LFoot3': 'left_foot3', 'LFoot4': 'left_foot4',
        'LFoot5': 'left_foot5',
        'RFoot1': 'right_foot1', 'RFoot2': 'right_foot2',
        'RFoot3': 'right_foot3', 'RFoot4': 'right_foot4',
        'RFoot5': 'right_foot5',
        # Torso (for trunk angle)
        'Torso1': 'torso1', 'Torso2': 'torso2',
        'Torso3': 'torso3', 'Torso4': 'torso4',
    }

    # Force threshold in Newtons for contact detection
    FORCE_THRESHOLD_N = -20.0
    # Only use first 3 walkway plates (plates 4-5 are off to the side)
    MAX_WALKWAY_PLATES = 3

    @property
    def name(self) -> str:
        return "Kuopio Gait"

    @property
    def description(self) -> str:
        return "51 healthy subjects, overground walking, 3 speeds (slow/comfortable/fast)"

    def list_files(self) -> List[Path]:
        """List all walking trial C3D files (exclude calibration/static)."""
        if not HAS_EZC3D:
            return []
        files = list(self.data_dir.rglob('*.c3d'))
        # Keep only files in mocap/ subdirectories
        # Exclude calibration files (calib_*, mass.*)
        walking_files = []
        for f in files:
            stem = f.stem.lower()
            if ('mocap' in str(f.parent).lower()
                and 'calib' not in stem
                and stem != 'mass'
                and not stem.startswith('.')):
                walking_files.append(f)
        return sorted(walking_files)

    def _parse_filename(self, filepath: Path) -> Dict:
        """Parse filename and path for metadata.

        Path: .../kuopio/03/mocap/l_comf_01.c3d
        Filename format: {side}_{speed}_{trial_num}.c3d
        - side: l or r (starting foot, not strictly relevant for us)
        - speed: comf, slow, fast
        - trial_num: 01-10
        """
        info = {
            'subject_id': 'unknown',
            'trial_id': 'unknown',
            'condition': 'healthy',
            'speed': 'comfortable',
            'starting_foot': 'unknown',
        }

        # Extract subject from directory structure
        # Path like .../kuopio/03/mocap/file.c3d
        parts = filepath.parts
        for i, part in enumerate(parts):
            if part == 'mocap' and i > 0:
                info['subject_id'] = 'KUO' + parts[i - 1].zfill(2)
                break
            elif part == 'kuopio' and i + 1 < len(parts):
                # Might be kuopio/03/mocap/...
                try:
                    subj_num = int(parts[i + 1])
                    info['subject_id'] = f'KUO{subj_num:02d}'
                except (ValueError, IndexError):
                    pass

        # Parse filename
        filename = filepath.stem
        match = re.match(r'([lr])_(comf|slow|fast)_(\d+)', filename)
        if match:
            side = match.group(1)
            speed = match.group(2)
            trial = match.group(3)

            info['starting_foot'] = 'left' if side == 'l' else 'right'
            speed_map = {'comf': 'comfortable', 'slow': 'slow', 'fast': 'fast'}
            info['speed'] = speed_map.get(speed, speed)
            info['trial_id'] = f"{info['speed']}_{side}_{trial}"
            info['condition'] = f"healthy_{info['speed']}"

        return info

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

    def _get_foot_centroid(self, c3d_data, side: str, frame: int) -> Optional[np.ndarray]:
        """Get centroid of foot cluster markers as a foot position estimate."""
        prefix = 'left_foot' if side == 'left' else 'right_foot'
        positions = []
        for num in range(1, 6):
            pos = self._find_landmark(c3d_data, f'{prefix}{num}', frame)
            if pos is not None:
                positions.append(pos)
        if len(positions) >= 2:
            return np.mean(positions, axis=0)
        return positions[0] if positions else None

    def _get_heel_proxy(self, c3d_data, side: str, frame: int) -> Optional[np.ndarray]:
        """Estimate heel position from foot cluster.

        For heel, use the most posterior foot marker (highest Y or lowest Y
        depending on walking direction). We use the foot marker that is
        closest to the ankle as a heel proxy.
        """
        ankle = self._find_landmark(c3d_data, f'{side}_ankle', frame)
        if ankle is None:
            return self._get_foot_centroid(c3d_data, side, frame)

        prefix = 'left_foot' if side == 'left' else 'right_foot'
        best_pos = None
        best_dist = float('inf')
        for num in range(1, 6):
            pos = self._find_landmark(c3d_data, f'{prefix}{num}', frame)
            if pos is not None:
                dist = np.linalg.norm(pos - ankle)
                if dist < best_dist:
                    best_dist = dist
                    best_pos = pos

        return best_pos

    def _get_toe_proxy(self, c3d_data, side: str, frame: int) -> Optional[np.ndarray]:
        """Estimate toe position from foot cluster.

        Use the foot marker farthest from the ankle as a toe proxy.
        """
        ankle = self._find_landmark(c3d_data, f'{side}_ankle', frame)
        if ankle is None:
            return self._get_foot_centroid(c3d_data, side, frame)

        prefix = 'left_foot' if side == 'left' else 'right_foot'
        best_pos = None
        best_dist = 0.0
        for num in range(1, 6):
            pos = self._find_landmark(c3d_data, f'{prefix}{num}', frame)
            if pos is not None:
                dist = np.linalg.norm(pos - ankle)
                if dist > best_dist:
                    best_dist = dist
                    best_pos = pos

        return best_pos

    def _get_pelvis_center(self, c3d_data, frame: int) -> Optional[np.ndarray]:
        """Compute pelvis center from pelvis cluster markers."""
        positions = []
        for num in range(1, 5):
            pos = self._find_landmark(c3d_data, f'pelvis{num}', frame)
            if pos is not None:
                positions.append(pos)
        if len(positions) >= 2:
            return np.mean(positions, axis=0)
        # Fallback: midpoint of hip centers
        lh = self._find_landmark(c3d_data, 'left_hip', frame)
        rh = self._find_landmark(c3d_data, 'right_hip', frame)
        if lh is not None and rh is not None:
            return (lh + rh) / 2.0
        return positions[0] if positions else None

    def _determine_foot_for_plate(self, c3d_data, plate_idx: int,
                                   contact_frame: int) -> Optional[str]:
        """Determine which foot is on a given force plate at the contact frame.

        Compares the position of left and right foot centroids against
        the plate corners at the time of contact.

        Args:
            c3d_data: C3D data dict
            plate_idx: 0-based force plate index
            contact_frame: marker-rate frame of contact onset

        Returns:
            'left' or 'right', or None if cannot determine.
        """
        try:
            corners = c3d_data['parameters']['FORCE_PLATFORM']['CORNERS']['value']
            plate_y_min = corners[1, :, plate_idx].min()
            plate_y_max = corners[1, :, plate_idx].max()

            # Get foot positions at contact frame
            lfoot = self._get_foot_centroid(c3d_data, 'left', contact_frame)
            rfoot = self._get_foot_centroid(c3d_data, 'right', contact_frame)

            if lfoot is None and rfoot is None:
                return None

            # Check which foot's Y position falls within the plate Y range
            l_on = False
            r_on = False
            if lfoot is not None:
                l_on = plate_y_min <= lfoot[1] <= plate_y_max
            if rfoot is not None:
                r_on = plate_y_min <= rfoot[1] <= plate_y_max

            if l_on and not r_on:
                return 'left'
            elif r_on and not l_on:
                return 'right'
            elif l_on and r_on:
                # Both feet in range -- use distance to plate center
                plate_center_y = (plate_y_min + plate_y_max) / 2.0
                l_dist = abs(lfoot[1] - plate_center_y) if lfoot is not None else float('inf')
                r_dist = abs(rfoot[1] - plate_center_y) if rfoot is not None else float('inf')
                return 'left' if l_dist < r_dist else 'right'

            return None
        except Exception:
            return None

    def _extract_force_plate_events(self, c3d_data) -> Dict[str, List[int]]:
        """Extract gait events (HS/TO) from force plate data.

        For the Kuopio dataset, force plates are type 2 (pre-calibrated).
        Each plate in the walkway typically captures 0-1 foot contacts per trial.

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
            n_plates = min(fp['USED']['value'][0], self.MAX_WALKWAY_PLATES)

            analog = c3d_data['data']['analogs'][0]
            an_labels = c3d_data['parameters']['ANALOG']['LABELS']['value']
            analog_rate = c3d_data['parameters']['ANALOG']['RATE']['value'][0]
            marker_rate = c3d_data['parameters']['POINT']['RATE']['value'][0]
            ratio = analog_rate / marker_rate

            for plate in range(n_plates):
                # Find Fz channel for this plate
                fz_label = f'Force.Fz{plate + 1}'
                fz_idx = None
                for i, label in enumerate(an_labels):
                    if label.strip() == fz_label:
                        fz_idx = i
                        break

                if fz_idx is None:
                    # Fallback: try by channel position (6 channels per plate)
                    fz_idx = plate * 6 + 2
                    if fz_idx >= analog.shape[0]:
                        continue

                fz = analog[fz_idx, :]

                # Detect contact: Fz < threshold (negative = downward)
                contact = fz < self.FORCE_THRESHOLD_N
                transitions = np.diff(contact.astype(int))
                onsets = np.where(transitions == 1)[0] + 1
                offsets = np.where(transitions == -1)[0] + 1

                if len(onsets) == 0:
                    continue

                # For overground, typically 1 contact per plate per trial
                # Take the first valid onset and offset
                for onset in onsets:
                    # Find matching offset
                    matching_offsets = offsets[offsets > onset]
                    if len(matching_offsets) == 0:
                        continue
                    offset = matching_offsets[0]

                    # Contact must be at least 100ms
                    contact_duration = (offset - onset) / analog_rate
                    if contact_duration < 0.1:
                        continue

                    # Convert to marker frames
                    hs_frame = int(onset / ratio)
                    to_frame = int(offset / ratio)

                    # Determine which foot hit this plate
                    foot = self._determine_foot_for_plate(c3d_data, plate, hs_frame)
                    if foot is None:
                        continue

                    events[f'hs_{foot}'].append(hs_frame)
                    events[f'to_{foot}'].append(to_frame)

        except Exception:
            pass

        # Sort and deduplicate
        for key in events:
            events[key] = sorted(list(set(events[key])))

        return events

    def extract_file(self, filepath: Path) -> ExtractionResult:
        """Extract data from a Kuopio C3D file."""
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
            # Get joint center positions (from SCORE/SARA calibration)
            left_hip = self._find_landmark(c3d, 'left_hip', frame)
            right_hip = self._find_landmark(c3d, 'right_hip', frame)

            # Prefer LKnee/RKnee, fall back to score
            left_knee = self._find_landmark(c3d, 'left_knee', frame)
            if left_knee is None:
                left_knee = self._find_landmark(c3d, 'left_knee_score', frame)
            right_knee = self._find_landmark(c3d, 'right_knee', frame)
            if right_knee is None:
                right_knee = self._find_landmark(c3d, 'right_knee_score', frame)

            left_ankle = self._find_landmark(c3d, 'left_ankle', frame)
            right_ankle = self._find_landmark(c3d, 'right_ankle', frame)

            # Foot positions (heel/toe proxies from cluster)
            left_heel = self._get_heel_proxy(c3d, 'left', frame)
            right_heel = self._get_heel_proxy(c3d, 'right', frame)
            left_toe = self._get_toe_proxy(c3d, 'left', frame)
            right_toe = self._get_toe_proxy(c3d, 'right', frame)

            # Pelvis center
            pelvis = self._get_pelvis_center(c3d, frame)

            def to_tuple(pos):
                return (float(pos[0]), float(pos[1]), float(pos[2])) if pos is not None else (0.0, 0.0, 0.0)

            landmark_positions = {
                'left_hip': to_tuple(left_hip), 'right_hip': to_tuple(right_hip),
                'left_knee': to_tuple(left_knee), 'right_knee': to_tuple(right_knee),
                'left_ankle': to_tuple(left_ankle), 'right_ankle': to_tuple(right_ankle),
                'left_heel': to_tuple(left_heel), 'right_heel': to_tuple(right_heel),
                'left_toe': to_tuple(left_toe), 'right_toe': to_tuple(right_toe),
                'left_shoulder': (0.0, 0.0, 0.0), 'right_shoulder': (0.0, 0.0, 0.0),
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
        )

        # Compute cadence from HS events (limited precision for overground)
        if has_hs:
            all_hs = sorted(events['hs_left'] + events['hs_right'])
            if len(all_hs) > 1:
                intervals = np.diff(all_hs) / fps
                valid_intervals = intervals[(intervals > 0.2) & (intervals < 2.0)]
                if len(valid_intervals) > 0:
                    gt.cadence = 60.0 / np.mean(valid_intervals)

        # --- Force-plate zone-aware evaluation --------------------------------
        # Kuopio is an overground walkway with 3-5 force plates covering only
        # a small portion of the walking path. GT events only exist where the
        # subject stepped on a plate (typically 1-3 steps per pass). To avoid
        # penalising detectors for events outside the instrumented zone, we
        # restrict evaluation to the frame range spanned by the GT events
        # plus a small margin.
        gt.event_source = "force_plate"
        all_gt_frames = sorted(
            events["hs_left"] + events["hs_right"] +
            events["to_left"] + events["to_right"]
        )
        if len(all_gt_frames) > 0:
            margin = int(0.5 * fps)  # 0.5 s margin around GT events
            crop_start = max(0, min(all_gt_frames) - margin)
            crop_end = min(n_frames - 1, max(all_gt_frames) + margin)

            # --- Crop angle_frames to valid region ---
            # ~50% of frames have (0,0,0) markers (outside capture volume).
            # This corrupts smoothing filters and walking direction detection.
            # Within the force-plate zone, marker availability is 99-100%.
            angle_frames = angle_frames[crop_start:crop_end + 1]
            # Re-index: update frame_index in each AngleFrame
            for idx, af in enumerate(angle_frames):
                af.frame_index = idx
            # Adjust GT frame indices
            for key in events:
                events[key] = [f - crop_start for f in events[key]
                               if crop_start <= f <= crop_end]
            gt.hs_frames = {'left': events['hs_left'], 'right': events['hs_right']}
            gt.to_frames = {'left': events['to_left'], 'right': events['to_right']}
            # Update valid_frame_range relative to cropped array
            gt.valid_frame_range = (0, len(angle_frames) - 1)
            # Update n_frames and duration
            n_frames = len(angle_frames)
            duration_s = n_frames / fps

        if not has_hs:
            warnings.append("No HS events detected from force plates")

        total_hs = len(events['hs_left']) + len(events['hs_right'])
        quality = 0.9 if total_hs >= 2 else (0.7 if total_hs >= 1 else 0.5)

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
            warnings=warnings
        )


def test_extractor():
    """Quick test of the Kuopio extractor."""
    if not HAS_EZC3D:
        print("ezc3d not installed")
        return

    data_dir = Path.home() / 'gait_benchmark_project/data/kuopio'
    if not data_dir.exists():
        print(f"Directory not found: {data_dir}")
        return

    extractor = KuopioExtractor(str(data_dir))
    files = extractor.list_files()
    print(f"Found {len(files)} C3D files")

    # Test on 3 files from different subjects/speeds
    test_files = files[:3]
    for f in test_files:
        try:
            result = extractor.extract_file(f)
            print(f"\nExtracted: {f.name} (from {f.parent.parent.name})")
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
            # Show a sample landmark
            if result.angle_frames:
                mid = len(result.angle_frames) // 2
                af = result.angle_frames[mid]
                print(f"  Sample frame {mid}: L_knee={af.left_knee_angle:.1f}, R_knee={af.right_knee_angle:.1f}")
        except Exception as e:
            print(f"\nError on {f.name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    test_extractor()
