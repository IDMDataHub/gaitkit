"""
Kuopio OpenPose Extractor.

Extracts 2D pose data from OpenPose JSON output for the Kuopio gait dataset
and converts it into the standard AngleFrame format for gait event detection.

The OpenPose data lives alongside the MoCap data:
    ~/gait_benchmark_project/data/kuopio/XX/openpose/json_{trial_name}/
    Each folder contains one JSON per video frame with 25-keypoint BODY_25 poses.

Ground truth events come from the CORRESPONDING C3D file (force plate data)
with frame indices converted via the MoCap-to-video fps ratio.

OpenPose BODY_25 keypoint indices:
    0=Nose, 1=Neck, 2=RShoulder, 3=RElbow, 4=RWrist,
    5=LShoulder, 6=LElbow, 7=LWrist, 8=MidHip,
    9=RHip, 10=RKnee, 11=RAnkle, 12=RBigToe, 13=RSmallToe, 14=RHeel,
    15=LHip, 16=LKnee, 17=LAnkle, 18=LBigToe, 19=LSmallToe, 20=LHeel,
    21=REye, 22=LEye, 23=REar, 24=LEar

Coordinate convention:
    - Sagittal video: X = anterior-posterior (horizontal pixel direction)
    - Y pixel increases downward, but anatomically "up" is decreasing Y
    - Pseudo-3D: [x_pixel, 0, -y_pixel] so col0=AP, col2=vertical(up-positive)

Sagittal view note:
    In sagittal video, only the near-side joints are reliably detected.
    For l_* trials (camera from left): right-side OpenPose joints are
    the near-side (fully visible). For r_* trials: left-side joints are
    near-side. The far-side ankle (kp17 or kp11) is often occluded.
    When ankle data is missing, we fall back to heel position as proxy.
"""

import json
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


class KuopioOpenPoseExtractor(BaseExtractor):
    """Extractor for Kuopio gait dataset - OpenPose 2D video data."""

    # OpenPose BODY_25 keypoint indices
    KP_NOSE = 0
    KP_NECK = 1
    KP_RSHOULDER = 2
    KP_LSHOULDER = 5
    KP_MIDHIP = 8
    KP_RHIP = 9
    KP_RKNE = 10
    KP_RANK = 11
    KP_RTOE = 12  # RBigToe
    KP_RHEE = 14
    KP_LHIP = 15
    KP_LKNE = 16
    KP_LANK = 17
    KP_LTOE = 18  # LBigToe
    KP_LHEE = 20

    # Minimum confidence to accept a keypoint
    MIN_CONFIDENCE = 0.3

    @property
    def name(self) -> str:
        return "Kuopio OpenPose"

    @property
    def description(self) -> str:
        return "51 healthy subjects, OpenPose 2D from sagittal video, 3 speeds"

    def list_files(self) -> List[Path]:
        """List all OpenPose trial folders that have a matching C3D file."""
        trials = []
        for subj_dir in sorted(self.data_dir.iterdir()):
            if not subj_dir.is_dir() or not subj_dir.name.isdigit():
                continue
            op_dir = subj_dir / 'openpose'
            mocap_dir = subj_dir / 'mocap'
            if not op_dir.exists() or not mocap_dir.exists():
                continue
            for json_dir in sorted(op_dir.iterdir()):
                if not json_dir.is_dir() or not json_dir.name.startswith('json_'):
                    continue
                trial_name = json_dir.name.replace('json_', '')
                c3d_path = mocap_dir / f'{trial_name}.c3d'
                if c3d_path.exists():
                    trials.append(json_dir)
        return trials

    def _parse_trial_info(self, json_dir: Path) -> Dict:
        """Parse trial directory path for metadata."""
        info = {
            'subject_id': 'unknown',
            'trial_id': 'unknown',
            'condition': 'healthy',
            'speed': 'comfortable',
            'trial_name': 'unknown',
        }
        trial_name = json_dir.name.replace('json_', '')
        info['trial_name'] = trial_name

        parts = json_dir.parts
        for i, part in enumerate(parts):
            if part == 'openpose' and i > 0:
                info['subject_id'] = 'KUO' + parts[i - 1].zfill(2)
                break
            elif part == 'kuopio' and i + 1 < len(parts):
                try:
                    subj_num = int(parts[i + 1])
                    info['subject_id'] = f'KUO{subj_num:02d}'
                except (ValueError, IndexError):
                    continue

        match = re.match(r'([lr])_(comf|slow|fast)_(\d+)', trial_name)
        if match:
            side = match.group(1)
            speed = match.group(2)
            trial = match.group(3)
            speed_map = {'comf': 'comfortable', 'slow': 'slow', 'fast': 'fast'}
            info['speed'] = speed_map.get(speed, speed)
            info['trial_id'] = f"{info['speed']}_{side}_{trial}"
            info['condition'] = f"healthy_{info['speed']}"

        return info

    def _load_openpose_frames(self, json_dir: Path) -> List[Optional[Dict]]:
        """Load all OpenPose JSON files from a trial directory.

        Returns a list of dicts, one per frame. Each dict maps
        keypoint_index -> (x, y, confidence). Frames with no valid
        detection return None.
        """
        json_files = sorted(json_dir.glob('*_keypoints.json'))
        frames = []

        for jf in json_files:
            try:
                with open(jf) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError):
                frames.append(None)
                continue

            people = data.get('people', [])
            if not people:
                frames.append(None)
                continue

            # Select the walking subject: person with largest vertical extent
            best_person = None
            best_height = 0

            for person in people:
                kps = person.get('pose_keypoints_2d', [])
                if len(kps) < 75:  # 25 keypoints * 3
                    continue

                valid_count = sum(1 for i in range(25) if kps[i*3+2] > self.MIN_CONFIDENCE)
                if valid_count < 8:
                    continue

                valid_ys = []
                for i in range(25):
                    if kps[i*3+2] > self.MIN_CONFIDENCE:
                        valid_ys.append(kps[i*3+1])
                if len(valid_ys) < 4:
                    continue

                height = max(valid_ys) - min(valid_ys)
                if height > best_height:
                    best_height = height
                    best_person = kps

            if best_person is None:
                frames.append(None)
                continue

            kp_dict = {}
            for i in range(25):
                x, y, c = best_person[i*3], best_person[i*3+1], best_person[i*3+2]
                if c > self.MIN_CONFIDENCE and (x > 0 or y > 0):
                    kp_dict[i] = (x, y, c)
            frames.append(kp_dict if kp_dict else None)

        return frames

    def _interpolate_keypoints(self, frames: List[Optional[Dict]],
                                keypoint_indices: List[int]) -> Dict[int, np.ndarray]:
        """Build continuous time series for each keypoint, interpolating gaps.

        Returns dict: keypoint_index -> Nx2 array of (x, y) pixel positions.
        Missing keypoints (no valid detections at all) get NaN arrays.
        """
        n_frames = len(frames)
        result = {}

        for kp_idx in keypoint_indices:
            xs = np.full(n_frames, np.nan)
            ys = np.full(n_frames, np.nan)

            for fi, frame in enumerate(frames):
                if frame is not None and kp_idx in frame:
                    xs[fi] = frame[kp_idx][0]
                    ys[fi] = frame[kp_idx][1]

            valid = ~np.isnan(xs)
            if valid.sum() < 2:
                result[kp_idx] = np.column_stack([xs, ys])
                continue

            indices = np.arange(n_frames)
            xs[~valid] = np.interp(indices[~valid], indices[valid], xs[valid])
            ys[~valid] = np.interp(indices[~valid], indices[valid], ys[valid])

            result[kp_idx] = np.column_stack([xs, ys])

        return result

    def _get_c3d_events(self, c3d_path: Path, video_fps: float) -> Dict[str, List[int]]:
        """Extract ground truth events from C3D file and convert to video frame indices."""
        from .kuopio_extractor import KuopioExtractor

        events = {'hs_left': [], 'hs_right': [], 'to_left': [], 'to_right': []}

        if not HAS_EZC3D or not c3d_path.exists():
            return events

        try:
            kuopio_root = c3d_path.parent.parent.parent
            ext = KuopioExtractor(str(kuopio_root))
            c3d = ezc3d.c3d(str(c3d_path))

            mocap_fps = c3d['parameters']['POINT']['RATE']['value'][0]
            fps_ratio = video_fps / mocap_fps

            mocap_events = ext._extract_force_plate_events(c3d)

            for key in events:
                for mocap_frame in mocap_events[key]:
                    video_frame = int(round(mocap_frame * fps_ratio))
                    events[key].append(video_frame)
                events[key] = sorted(events[key])

        except Exception as exc:
            logger.debug("Failed to extract C3D events from %s: %s", c3d_path, exc)

        return events

    def _to_pseudo3d(self, x: float, y: float) -> np.ndarray:
        """Convert 2D pixel coords to pseudo-3D: [x, 0, -y]."""
        return np.array([x, 0.0, -y])

    def extract_file(self, filepath: Path) -> ExtractionResult:
        """Extract data from a Kuopio OpenPose trial directory."""
        json_dir = Path(filepath)
        info = self._parse_trial_info(json_dir)

        raw_frames = self._load_openpose_frames(json_dir)
        n_frames = len(raw_frames)

        if n_frames == 0:
            raise ValueError(f"No JSON frames found in {json_dir}")

        # Determine video fps from C3D duration
        trial_name = info['trial_name']
        subj_dir = json_dir.parent.parent
        c3d_path = subj_dir / 'mocap' / f'{trial_name}.c3d'

        video_fps = 15.0  # default fallback
        if HAS_EZC3D and c3d_path.exists():
            try:
                c3d = ezc3d.c3d(str(c3d_path))
                mocap_fps = c3d['parameters']['POINT']['RATE']['value'][0]
                mocap_frames = c3d['data']['points'].shape[2]
                if mocap_fps > 0 and mocap_frames > 0:
                    mocap_duration = mocap_frames / mocap_fps
                    if mocap_duration > 0:
                        video_fps = n_frames / mocap_duration
            except Exception as exc:
                logger.debug("Failed to infer video fps from %s: %s", c3d_path, exc)

        if video_fps <= 0:
            video_fps = 15.0

        duration_s = n_frames / video_fps

        needed_kps = [
            self.KP_NOSE, self.KP_NECK, self.KP_RSHOULDER, self.KP_LSHOULDER,
            self.KP_MIDHIP, self.KP_RHIP, self.KP_RKNE, self.KP_RANK,
            self.KP_RTOE, self.KP_RHEE, self.KP_LHIP, self.KP_LKNE,
            self.KP_LANK, self.KP_LTOE, self.KP_LHEE,
        ]

        kp_series = self._interpolate_keypoints(raw_frames, needed_kps)

        # Check availability: if ankle is fully missing, use heel as fallback
        def _has_valid_data(kp_idx):
            if kp_idx not in kp_series:
                return False
            return not np.all(np.isnan(kp_series[kp_idx][:, 0]))

        if not _has_valid_data(self.KP_LANK) and _has_valid_data(self.KP_LHEE):
            kp_series[self.KP_LANK] = kp_series[self.KP_LHEE].copy()
        if not _has_valid_data(self.KP_RANK) and _has_valid_data(self.KP_RHEE):
            kp_series[self.KP_RANK] = kp_series[self.KP_RHEE].copy()
        # Similarly for hips: if missing, use knee as rough proxy
        if not _has_valid_data(self.KP_LHIP) and _has_valid_data(self.KP_LKNE):
            kp_series[self.KP_LHIP] = kp_series[self.KP_LKNE].copy()
        if not _has_valid_data(self.KP_RHIP) and _has_valid_data(self.KP_RKNE):
            kp_series[self.KP_RHIP] = kp_series[self.KP_RKNE].copy()

        events = self._get_c3d_events(c3d_path, video_fps)

        warnings = []

        n_valid = sum(1 for f in raw_frames if f is not None)
        detection_rate = n_valid / n_frames if n_frames > 0 else 0
        if detection_rate < 0.8:
            warnings.append(f"Low detection rate: {detection_rate:.1%}")

        angle_frames = []
        for frame in range(n_frames):
            def get_pos(kp_idx, fr=frame):
                if kp_idx in kp_series:
                    xy = kp_series[kp_idx][fr]
                    if not np.isnan(xy[0]):
                        return self._to_pseudo3d(xy[0], xy[1])
                return None

            left_hip = get_pos(self.KP_LHIP)
            right_hip = get_pos(self.KP_RHIP)
            left_knee = get_pos(self.KP_LKNE)
            right_knee = get_pos(self.KP_RKNE)
            left_ankle = get_pos(self.KP_LANK)
            right_ankle = get_pos(self.KP_RANK)
            left_heel = get_pos(self.KP_LHEE)
            right_heel = get_pos(self.KP_RHEE)
            left_toe = get_pos(self.KP_LTOE)
            right_toe = get_pos(self.KP_RTOE)
            left_shoulder = get_pos(self.KP_LSHOULDER)
            right_shoulder = get_pos(self.KP_RSHOULDER)
            midhip = get_pos(self.KP_MIDHIP)

            pelvis = midhip
            if pelvis is None and left_hip is not None and right_hip is not None:
                pelvis = (left_hip + right_hip) / 2.0

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
                'pelvis': to_tuple(pelvis),
            }

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
                is_valid=(raw_frames[frame] is not None),
            )
            angle_frames.append(af)

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

        if has_hs:
            all_hs = sorted(events['hs_left'] + events['hs_right'])
            if len(all_hs) > 1:
                intervals = np.diff(all_hs) / video_fps
                valid_intervals = intervals[(intervals > 0.2) & (intervals < 2.0)]
                if len(valid_intervals) > 0:
                    gt.cadence = 60.0 / np.mean(valid_intervals)

        if not has_hs:
            warnings.append("No HS events from C3D force plates")

        total_hs = len(events['hs_left']) + len(events['hs_right'])
        quality = 0.7 if total_hs >= 2 else (0.5 if total_hs >= 1 else 0.3)
        quality *= detection_rate

        return ExtractionResult(
            source_file=str(json_dir),
            subject_id=info['subject_id'],
            trial_id=info['trial_id'],
            condition=info['condition'],
            fps=video_fps,
            n_frames=n_frames,
            duration_s=duration_s,
            angle_frames=angle_frames,
            raw_data=None,
            ground_truth=gt,
            quality_score=quality,
            warnings=warnings,
        )


def test_extractor():
    """Quick test of the Kuopio OpenPose extractor."""
    data_dir = Path.home() / 'gait_benchmark_project/data/kuopio'
    if not data_dir.exists():
        print(f"Directory not found: {data_dir}")
        return

    extractor = KuopioOpenPoseExtractor(str(data_dir))
    files = extractor.list_files()
    print(f"Found {len(files)} OpenPose trial directories")

    for f in files[:3]:
        try:
            result = extractor.extract_file(f)
            print(f"\nExtracted: {f.name} (subject {result.subject_id})")
            print(f"  Trial: {result.trial_id}, Condition: {result.condition}")
            print(f"  Frames: {result.n_frames}, FPS: {result.fps:.1f}, Duration: {result.duration_s:.1f}s")
            print(f"  GT HS L: {len(result.ground_truth.hs_frames['left'])}")
            print(f"  GT HS R: {len(result.ground_truth.hs_frames['right'])}")
            print(f"  GT TO L: {len(result.ground_truth.to_frames['left'])}")
            print(f"  GT TO R: {len(result.ground_truth.to_frames['right'])}")
            if result.ground_truth.cadence:
                print(f"  Cadence: {result.ground_truth.cadence:.1f} steps/min")
            print(f"  Quality: {result.quality_score:.2f}")
            print(f"  Warnings: {result.warnings}")
        except Exception as e:
            print(f"\nError on {f.name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    test_extractor()
