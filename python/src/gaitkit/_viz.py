"""Visualization functions for gait events and cycles."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from ._core import GaitResult


def _import_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install gaitkit[viz]"
        )


# ── Colours ──────────────────────────────────────────────────────────
_LEFT_COLOR = "#2166ac"
_RIGHT_COLOR = "#b2182b"
_HS_MARKER = "v"   # downward triangle
_TO_MARKER = "^"   # upward triangle


def plot_result(
    result: "GaitResult",
    signals: Optional[List[str]] = None,
    ax=None,
    figsize=(14, 5),
    title: Optional[str] = None,
):
    """Plot angle signals with HS/TO markers overlaid.

    Parameters
    ----------
    result : GaitResult
        Detection results (from :func:`gaitkit.detect`).
    signals : list of str, optional
        Angle names to plot, e.g. ["left_knee_angle"].
        Defaults to left and right knee angles.
    ax : matplotlib Axes, optional
        If provided, draw on this axes.
    figsize : tuple
        Figure size when creating a new figure.
    title : str, optional
        Plot title. Defaults to "Gait Events — {method}".

    Returns
    -------
    matplotlib.figure.Figure
    """
    if result is None:
        raise ValueError("result is required")
    if getattr(result, "fps", 0) <= 0:
        raise ValueError("result.fps must be strictly positive")
    if getattr(result, "n_frames", 0) <= 0:
        raise ValueError("result.n_frames must be strictly positive")
    if signals is not None:
        if not isinstance(signals, list) or not signals:
            raise ValueError("signals must be a non-empty list of variable names")
        if any(not isinstance(s, str) or not s.strip() for s in signals):
            raise ValueError("signals must contain non-empty string names")

    plt = _import_mpl()
    af = result._angle_frames

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    t = np.arange(result.n_frames) / result.fps

    # Default signals
    if signals is None:
        signals = ["left_knee_angle", "right_knee_angle"]

    # Plot angle curves
    for sig in signals:
        if af and hasattr(af[0], sig):
            vals = [getattr(f, sig) for f in af]
        elif af and isinstance(af[0], dict) and sig in af[0]:
            vals = [f.get(sig, 0) for f in af]
        else:
            continue
        color = _LEFT_COLOR if "left" in sig else _RIGHT_COLOR
        label = sig.replace("_", " ").title()
        ax.plot(t, vals, color=color, alpha=0.7, linewidth=1.2, label=label)

    # Plot HS markers
    y_range = ax.get_ylim()
    for ev_list, color, side_label in [
        (result.left_hs, _LEFT_COLOR, "L"),
        (result.right_hs, _RIGHT_COLOR, "R"),
    ]:
        if ev_list:
            frames = [e["frame"] for e in ev_list]
            times = [e["time"] for e in ev_list]
            ax.scatter(times, [y_range[0] + 2] * len(times),
                       marker=_HS_MARKER, s=80, color=color, zorder=5,
                       label=f"HS {side_label} (n={len(ev_list)})")

    # Plot TO markers
    for ev_list, color, side_label in [
        (result.left_to, _LEFT_COLOR, "L"),
        (result.right_to, _RIGHT_COLOR, "R"),
    ]:
        if ev_list:
            times = [e["time"] for e in ev_list]
            ax.scatter(times, [y_range[1] - 2] * len(times),
                       marker=_TO_MARKER, s=80, color=color, zorder=5,
                       label=f"TO {side_label} (n={len(ev_list)})")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (deg)")
    ax.set_title(title or f"Gait Events \u2014 {result.method}")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


def compare_plot(
    data,
    methods: Sequence[str],
    fps: float = None,
    figsize=(14, None),
    title: str = "Method Comparison",
):
    """Run multiple detectors on the same data and plot side by side.

    Parameters
    ----------
    data : any
        Input accepted by :func:`gaitkit.detect`.
    methods : list of str
        Methods to compare.
    fps : float, optional
        Sampling frequency.
    figsize : tuple
        Figure width and height. Height auto-scales with number of methods.
    title : str
        Super-title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from ._core import detect
    if not methods:
        raise ValueError("methods must contain at least one detector name")
    if fps is not None and fps <= 0:
        raise ValueError("fps must be strictly positive when provided")

    plt = _import_mpl()

    n = len(methods)
    h = figsize[1] or 2.5 * n
    fig, axes = plt.subplots(n, 1, figsize=(figsize[0], h), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        result = detect(data, method=method, fps=fps)
        # Plot a simple knee angle if available
        af = result._angle_frames
        t = np.arange(result.n_frames) / result.fps
        if af:
            if hasattr(af[0], "left_knee_angle"):
                vals = [f.left_knee_angle for f in af]
            elif isinstance(af[0], dict):
                vals = [f.get("left_knee_angle", 0) for f in af]
            else:
                vals = np.zeros(result.n_frames)
            ax.plot(t, vals, color="#666666", linewidth=0.8, alpha=0.6)

        # HS/TO markers
        for ev_list, color in [(result.left_hs, _LEFT_COLOR), (result.right_hs, _RIGHT_COLOR)]:
            if ev_list:
                times = [e["time"] for e in ev_list]
                ax.scatter(times, [0] * len(times), marker=_HS_MARKER,
                           s=60, color=color, zorder=5)
        for ev_list, color in [(result.left_to, _LEFT_COLOR), (result.right_to, _RIGHT_COLOR)]:
            if ev_list:
                times = [e["time"] for e in ev_list]
                ax.scatter(times, [0] * len(times), marker=_TO_MARKER,
                           s=60, color=color, zorder=5)

        n_hs = len(result.left_hs) + len(result.right_hs)
        n_to = len(result.left_to) + len(result.right_to)
        ax.set_ylabel(method, fontsize=10, fontweight="bold")
        ax.text(0.98, 0.85, f"{n_hs} HS, {n_to} TO",
                transform=ax.transAxes, ha="right", fontsize=8, color="#444")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_cycles(
    result: "GaitResult",
    variable: str = "left_knee_angle",
    ax=None,
    figsize=(8, 5),
):
    """Butterfly plot of normalised gait cycles.

    Parameters
    ----------
    result : GaitResult
        Must contain angle frames in *_angle_frames*.
    variable : str
        Angle name to plot (default "left_knee_angle").
    ax : matplotlib Axes, optional
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not isinstance(variable, str) or not variable.strip():
        raise ValueError("variable must be a non-empty string")
    af = result._angle_frames
    if af is None or len(af) == 0:
        raise ValueError("No angle frames stored in result; cannot plot cycles.")
    plt = _import_mpl()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Determine side from variable name
    side = "left" if "left" in variable else "right"
    hs_list = result.left_hs if side == "left" else result.right_hs
    to_list = result.left_to if side == "left" else result.right_to

    hs_frames = sorted(e["frame"] for e in hs_list)
    to_frames = sorted(e["frame"] for e in to_list)

    # Extract signal
    if hasattr(af[0], variable):
        vals = np.array([getattr(f, variable) for f in af])
    elif isinstance(af[0], dict):
        vals = np.array([f.get(variable, 0) for f in af])
    else:
        raise ValueError(f"Cannot find {variable} in angle frames.")

    # Normalize each cycle to 0-100%
    cycles_norm = []
    to_pcts = []
    for i in range(len(hs_frames) - 1):
        f0, f1 = hs_frames[i], hs_frames[i + 1]
        if f1 - f0 < 10:
            continue
        seg = vals[f0:f1 + 1]
        x_norm = np.linspace(0, 100, len(seg))
        x_interp = np.linspace(0, 100, 101)
        seg_interp = np.interp(x_interp, x_norm, seg)
        cycles_norm.append(seg_interp)

        # Find TO percentage in this cycle
        to_in = [t for t in to_frames if f0 < t < f1]
        if to_in:
            to_pcts.append((to_in[0] - f0) / (f1 - f0) * 100)

    if not cycles_norm:
        ax.text(0.5, 0.5, "No complete cycles", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    arr = np.array(cycles_norm)
    mean_c = arr.mean(axis=0)
    std_c = arr.std(axis=0)
    x = np.linspace(0, 100, 101)

    # Individual cycles (thin)
    color = _LEFT_COLOR if side == "left" else _RIGHT_COLOR
    for c in cycles_norm:
        ax.plot(x, c, color=color, alpha=0.15, linewidth=0.7)
    # Mean + std band
    ax.plot(x, mean_c, color=color, linewidth=2.5, label=f"Mean (n={len(cycles_norm)})")
    ax.fill_between(x, mean_c - std_c, mean_c + std_c, color=color, alpha=0.15)

    # TO vertical line
    if to_pcts:
        mean_to = np.mean(to_pcts)
        ax.axvline(mean_to, color="#999", linestyle="--", linewidth=1,
                   label=f"TO ({mean_to:.0f}%)")

    ax.set_xlabel("Gait Cycle (%)")
    ax.set_ylabel(f"{variable.replace('_', ' ').title()} (deg)")
    ax.set_title(f"Gait Cycles \u2014 {result.method}")
    ax.set_xlim(0, 100)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def plot_ensemble(
    result: "GaitResult",
    ax=None,
    figsize=(14, 4),
):
    """Plot ensemble results with confidence color-coding.

    Parameters
    ----------
    result : GaitResult
        Must have confidence and voters fields in events.
    ax : matplotlib Axes, optional
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _import_mpl()
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    norm = Normalize(vmin=0, vmax=1)
    cmap = plt.cm.RdYlGn

    for ev_list, marker, y_pos, label in [
        (result.left_hs, _HS_MARKER, 1.0, "HS Left"),
        (result.right_hs, _HS_MARKER, 0.8, "HS Right"),
        (result.left_to, _TO_MARKER, 0.4, "TO Left"),
        (result.right_to, _TO_MARKER, 0.2, "TO Right"),
    ]:
        if not ev_list:
            continue
        times = [e["time"] for e in ev_list]
        confs = [e.get("confidence", 1.0) for e in ev_list]
        colors = [cmap(norm(c)) for c in confs]
        ax.scatter(times, [y_pos] * len(times), marker=marker,
                   s=100, c=colors, edgecolors="#333", linewidths=0.5, zorder=5)
        ax.text(-0.02, y_pos, label, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=9)

    ax.set_ylim(-0.1, 1.3)
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Ensemble Detection \u2014 {result.method}")
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.02)
    cb.set_label("Confidence", fontsize=9)

    fig.tight_layout()
    return fig
