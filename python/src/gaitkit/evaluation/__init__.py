"""
Evaluation module for gait event detection.

Provides metrics computation, event matching, and statistical analysis
tools for comparing detector performance.
"""

from .metrics import compute_event_metrics, compute_cadence_error
from .matching import match_events, EventMatch
from .statistics import (
    bootstrap_confidence_interval, wilcoxon_signed_rank, compute_summary_table,
)

__all__ = [
    "compute_event_metrics", "compute_cadence_error",
    "match_events", "EventMatch",
    "bootstrap_confidence_interval", "wilcoxon_signed_rank", "compute_summary_table",
]
