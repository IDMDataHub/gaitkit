"""
Base Extractor - Interface commune pour tous les extracteurs.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path


@dataclass
class GroundTruth:
    """Ground truth disponible pour une séquence."""
    has_hs: bool = False  # Heel Strike events
    has_to: bool = False  # Toe Off events
    has_cadence: bool = False  # Cadence/stride time
    has_angles: bool = False  # Joint angles
    has_forces: bool = False  # Ground reaction forces
    has_emg: bool = False  # EMG signals

    # Force-plate zone-aware evaluation fields.
    # event_source: how GT events were obtained.
    #   "annotated" = manual annotation or full-coverage system (default)
    #   "force_plate" = derived from force plates (may cover only part of trial)
    #   "derived" = algorithmically derived from other signals
    event_source: str = "annotated"
    # valid_frame_range: if event_source is "force_plate" on an overground
    # walkway, this is the (start_frame, end_frame) range covered by the
    # force plates.  Only events within this range should be scored to avoid
    # penalising detections outside the instrumented zone.  None means the
    # entire trial is valid (e.g. treadmill, full annotation).
    valid_frame_range: Optional[Tuple[int, int]] = None

    hs_frames: Optional[Dict[str, List[int]]] = None  # {'left': [...], 'right': [...]}
    to_frames: Optional[Dict[str, List[int]]] = None
    cadence: Optional[float] = None  # steps/min
    stride_times: Optional[List[float]] = None  # list of stride durations
    angles_ref: Optional[pd.DataFrame] = None  # reference angles

    def __post_init__(self):
        if self.hs_frames is None:
            self.hs_frames = {'left': [], 'right': []}
        if self.to_frames is None:
            self.to_frames = {'left': [], 'right': []}


@dataclass
class AngleFrame:
    """Frame d'angles articulaires - format commun."""
    frame_index: int
    left_hip_angle: float
    right_hip_angle: float
    left_knee_angle: float
    right_knee_angle: float
    left_ankle_angle: float
    right_ankle_angle: float
    trunk_angle: float
    pelvis_tilt: float
    landmark_positions: Optional[Dict[str, Tuple[float, float, float]]] = None
    is_valid: bool = True

    def __post_init__(self):
        if self.landmark_positions is None:
            self.landmark_positions = {}


@dataclass
class ExtractionResult:
    """Résultat d'extraction d'une séquence."""
    # Metadata
    source_file: str
    subject_id: str
    trial_id: str
    condition: str  # ex: 'walking', 'fast', 'slow', etc.

    # Temporal data
    fps: float
    n_frames: int
    duration_s: float

    # Extracted data
    angle_frames: List[AngleFrame]
    raw_data: Optional[pd.DataFrame] = None  # optional raw data

    # Ground truth
    ground_truth: GroundTruth = field(default_factory=GroundTruth)

    # Quality
    quality_score: float = 1.0  # 0-1, data quality
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convertit en dictionnaire pour sérialisation."""
        return {
            'source_file': self.source_file,
            'subject_id': self.subject_id,
            'trial_id': self.trial_id,
            'condition': self.condition,
            'fps': self.fps,
            'n_frames': self.n_frames,
            'duration_s': self.duration_s,
            'quality_score': self.quality_score,
            'ground_truth': {
                'has_hs': self.ground_truth.has_hs,
                'has_to': self.ground_truth.has_to,
                'has_cadence': self.ground_truth.has_cadence,
                'has_angles': self.ground_truth.has_angles,
                'has_forces': self.ground_truth.has_forces,
                'has_emg': self.ground_truth.has_emg,
                'cadence': self.ground_truth.cadence,
                'n_hs_left': len(self.ground_truth.hs_frames.get('left', [])),
                'n_hs_right': len(self.ground_truth.hs_frames.get('right', [])),
            },
            'warnings': self.warnings,
        }


class BaseExtractor(ABC):
    """Classe de base pour tous les extracteurs."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Directory not found: {data_dir}")

    @property
    @abstractmethod
    def name(self) -> str:
        """Nom de la base de données."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Description de la base."""
        pass

    @abstractmethod
    def list_files(self) -> List[Path]:
        """Liste tous les fichiers extractibles."""
        pass

    @abstractmethod
    def extract_file(self, filepath: Path) -> ExtractionResult:
        """Extrait les données d'un fichier."""
        pass

    def extract_all(self, max_files: Optional[int] = None) -> List[ExtractionResult]:
        """Extrait toutes les données de la base."""
        files = self.list_files()
        if max_files:
            files = files[:max_files]

        results = []
        for f in files:
            try:
                result = self.extract_file(f)
                results.append(result)
            except Exception as e:
                print(f"Error extracting {f}: {e}")

        return results

    def get_summary(self) -> Dict:
        """Retourne un résumé de la base."""
        files = self.list_files()
        return {
            'name': self.name,
            'description': self.description,
            'data_dir': str(self.data_dir),
            'n_files': len(files),
        }


def compute_angle_from_3points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Calcule l'angle au point p2 formé par p1-p2-p3.

    Args:
        p1, p2, p3: Points 3D ou 2D

    Returns:
        Angle en degrés
    """
    v1 = p1 - p2
    v2 = p3 - p2

    v1_norm = v1 / (np.linalg.norm(v1) + 1e-10)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-10)

    cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def compute_signed_angle_2d(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calcule l'angle signé entre deux vecteurs 2D.

    Returns:
        Angle en degrés (-180 à 180)
    """
    angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
    return np.degrees(angle)
