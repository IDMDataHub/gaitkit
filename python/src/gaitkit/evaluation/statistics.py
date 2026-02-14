"""
Statistical analysis tools for the benchmark.

Provides bootstrap confidence intervals, Wilcoxon signed-rank tests,
and summary table generation for publication.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from scipy import stats


def bootstrap_confidence_interval(values: np.ndarray, n_bootstrap: int = 10000,
                                   confidence: float = 0.95,
                                   statistic: str = "mean") -> Tuple[float, float, float]:
    """Compute a bootstrap confidence interval.

    Parameters
    ----------
    values : ndarray
        Sample values.
    n_bootstrap : int
        Number of bootstrap resamples.
    confidence : float
        Confidence level (e.g. 0.95 for 95% CI).
    statistic : str
        ``'mean'`` or ``'median'``.

    Returns
    -------
    estimate : float
        Point estimate of the statistic.
    ci_lower : float
        Lower bound of the confidence interval.
    ci_upper : float
        Upper bound of the confidence interval.
    """
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(42)
    stat_fn = np.mean if statistic == "mean" else np.median
    estimate = float(stat_fn(values))

    boot_stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_stats[i] = stat_fn(sample)

    alpha = 1 - confidence
    ci_lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return estimate, ci_lower, ci_upper


def wilcoxon_signed_rank(values_a: np.ndarray, values_b: np.ndarray,
                          alternative: str = "two-sided") -> Tuple[float, float]:
    """Wilcoxon signed-rank test comparing paired samples.

    Parameters
    ----------
    values_a, values_b : ndarray
        Paired metric values (e.g. F1 scores per file for two detectors).
    alternative : str
        ``'two-sided'``, ``'greater'``, or ``'less'``.

    Returns
    -------
    statistic : float
        The Wilcoxon test statistic.
    p_value : float
        Associated p-value.
    """
    a = np.asarray(values_a)
    b = np.asarray(values_b)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 5:
        return np.nan, np.nan
    result = stats.wilcoxon(a, b, alternative=alternative)
    return float(result.statistic), float(result.pvalue)


def compute_summary_table(df: pd.DataFrame,
                           group_cols: Optional[List[str]] = None
                           ) -> pd.DataFrame:
    """Aggregate per-file results into a summary table.

    Parameters
    ----------
    df : DataFrame
        Raw per-file benchmark results with columns including
        ``dataset``, ``detector_name``, ``hs_f1``, ``to_f1``, etc.
    group_cols : list of str or None
        Columns to group by (default: ``['dataset', 'detector_name']``).

    Returns
    -------
    DataFrame
        Summary table with mean metrics, subject counts, etc.
    """
    if group_cols is None:
        group_cols = ["dataset", "detector_name"]

    agg_dict = {
        "hs_precision": "mean",
        "hs_recall": "mean",
        "hs_f1": "mean",
        "to_precision": "mean",
        "to_recall": "mean",
        "to_f1": "mean",
        "processing_time_ms": "mean",
        "source_file": "count",
    }
    # Only aggregate columns that exist
    agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}

    summary = df.groupby(group_cols).agg(agg_dict).round(3)
    if "source_file" in summary.columns:
        summary = summary.rename(columns={"source_file": "n_files"})

    return summary.sort_values(group_cols)
