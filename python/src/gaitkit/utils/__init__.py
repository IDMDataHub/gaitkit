"""
Utility modules for signal preprocessing and gait segmentation.
"""

from .freezing_detection import (
    detect_freezing_zones, segment_by_freezing, get_valid_segments,
    GaitSegment, FreezingZone,
)

__all__ = [
    "detect_freezing_zones", "segment_by_freezing", "get_valid_segments",
    "GaitSegment", "FreezingZone",
]
