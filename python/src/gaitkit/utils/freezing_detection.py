"""
Utilitaires pour segmenter les données de marche.

Permet d'exclure les zones problématiques (freezing, pieds derrière bassin)
et de traiter chaque segment valide séparément.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class GaitSegment:
    """Un segment de marche valide."""
    start_frame: int
    end_frame: int
    angle_frames: list  # Liste des AngleFrame du segment
    is_valid: bool = True


@dataclass
class FreezingZone:
    """Zone de freezing détectée."""
    start_frame: int
    end_frame: int
    reason: str = "feet_behind_pelvis"


def detect_freezing_zones(angle_frames,
                          threshold_mm: float = 50.0,
                          min_duration_frames: int = 30) -> List[FreezingZone]:
    """
    Détecte les zones où les deux pieds sont derrière le bassin.

    Ces zones correspondent à du freezing ou des situations non-marche.

    Args:
        angle_frames: Liste des frames
        threshold_mm: Seuil en mm pour considérer "derrière"
        min_duration_frames: Durée minimum pour constituer une zone

    Returns:
        Liste des zones de freezing
    """
    n = len(angle_frames)
    if n < min_duration_frames:
        return []

    # Calculer la position du COM (pelvis) et des pieds
    pelvis_x = np.zeros(n)
    left_ankle_x = np.zeros(n)
    right_ankle_x = np.zeros(n)

    for i, frame in enumerate(angle_frames):
        if not frame.landmark_positions:
            continue

        # Position du pelvis (milieu des hanches)
        left_hip = frame.landmark_positions.get('left_hip', (0, 0, 0))
        right_hip = frame.landmark_positions.get('right_hip', (0, 0, 0))
        pelvis_x[i] = (left_hip[0] + right_hip[0]) / 2

        # Position des chevilles
        left_ankle_x[i] = frame.landmark_positions.get('left_ankle', (0, 0, 0))[0]
        right_ankle_x[i] = frame.landmark_positions.get('right_ankle', (0, 0, 0))[0]

    # Detect frames where both feet are behind the pelvis
    both_behind = np.zeros(n, dtype=bool)

    for i in range(n):
        left_behind = left_ankle_x[i] < pelvis_x[i] - threshold_mm
        right_behind = right_ankle_x[i] < pelvis_x[i] - threshold_mm
        both_behind[i] = left_behind and right_behind

    # Regrouper en zones continues
    zones = []
    in_zone = False
    zone_start = 0

    for i in range(n):
        if both_behind[i] and not in_zone:
            in_zone = True
            zone_start = i
        elif not both_behind[i] and in_zone:
            in_zone = False
            if i - zone_start >= min_duration_frames:
                zones.append(FreezingZone(
                    start_frame=zone_start,
                    end_frame=i - 1,
                    reason="feet_behind_pelvis"
                ))

    # Close the last zone if necessary
    if in_zone and n - zone_start >= min_duration_frames:
        zones.append(FreezingZone(
            start_frame=zone_start,
            end_frame=n - 1,
            reason="feet_behind_pelvis"
        ))

    return zones


def segment_by_freezing(angle_frames,
                         freezing_zones: List[FreezingZone],
                         min_segment_frames: int = 50) -> List[GaitSegment]:
    """
    Segmente les données en excluant les zones de freezing.

    Args:
        angle_frames: Liste complète des frames
        freezing_zones: Zones à exclure
        min_segment_frames: Taille minimum d'un segment valide

    Returns:
        Liste de segments valides
    """
    n = len(angle_frames)

    if not freezing_zones:
        # Pas de freezing, retourner tout comme un segment
        return [GaitSegment(
            start_frame=0,
            end_frame=n - 1,
            angle_frames=angle_frames,
            is_valid=True
        )]

    # Sort zones by start frame
    sorted_zones = sorted(freezing_zones, key=lambda z: z.start_frame)

    segments = []
    current_start = 0

    for zone in sorted_zones:
        # Segment avant cette zone
        if zone.start_frame > current_start:
            segment_frames = angle_frames[current_start:zone.start_frame]

            if len(segment_frames) >= min_segment_frames:
                segments.append(GaitSegment(
                    start_frame=current_start,
                    end_frame=zone.start_frame - 1,
                    angle_frames=segment_frames,
                    is_valid=True
                ))

        current_start = zone.end_frame + 1

    # Segment after the last zone
    if current_start < n:
        segment_frames = angle_frames[current_start:]

        if len(segment_frames) >= min_segment_frames:
            segments.append(GaitSegment(
                start_frame=current_start,
                end_frame=n - 1,
                angle_frames=segment_frames,
                is_valid=True
            ))

    return segments


def filter_gt_by_segments(gt_frames: List[int],
                           segments: List[GaitSegment]) -> List[Tuple[List[int], GaitSegment]]:
    """
    Filtre les frames GT pour chaque segment.

    Args:
        gt_frames: Liste des frames GT (globaux)
        segments: Liste des segments

    Returns:
        Liste de (gt_frames_locaux, segment) pour chaque segment
    """
    result = []

    for segment in segments:
        # Filtrer les GT qui sont dans ce segment
        segment_gt = [
            f for f in gt_frames
            if segment.start_frame <= f <= segment.end_frame
        ]

        # Convertir en indices locaux
        local_gt = [f - segment.start_frame for f in segment_gt]

        result.append((local_gt, segment))

    return result


def get_valid_segments(angle_frames,
                        threshold_mm: float = 50.0,
                        min_freezing_frames: int = 30,
                        min_segment_frames: int = 50) -> Tuple[List[GaitSegment], List[FreezingZone]]:
    """
    Fonction principale: détecte les zones freezing et retourne les segments valides.

    Args:
        angle_frames: Liste des frames
        threshold_mm: Seuil pour détecter "derrière pelvis"
        min_freezing_frames: Durée min pour une zone freezing
        min_segment_frames: Durée min pour un segment valide

    Returns:
        (segments_valides, zones_freezing)
    """
    freezing_zones = detect_freezing_zones(
        angle_frames,
        threshold_mm=threshold_mm,
        min_duration_frames=min_freezing_frames
    )

    segments = segment_by_freezing(
        angle_frames,
        freezing_zones,
        min_segment_frames=min_segment_frames
    )

    return segments, freezing_zones
