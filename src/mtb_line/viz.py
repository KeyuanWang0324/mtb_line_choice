"""The Gate 1 chart: every rider's line, in trail coordinates, on one axis.

Plotting `d` against `s` rather than drawing trajectories over a 3D model is the
whole point of the (s, d) representation -- a 30cm difference is invisible in a
3D render of a 120m descent, and obvious here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mtb_line.evaluate import SeparationProfile, separation_profile
from mtb_line.frenet import FrenetProfile


def plot_lines(
    profiles: list[FrenetProfile],
    out_path: str | Path,
    labels: dict[str, str] | None = None,
    split_window: tuple[float, float] | None = None,
    title: str = "Rider lines in trail coordinates",
) -> Path:
    """Write a PNG of lateral offset vs arc length, one trace per run."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plots: pip install 'mtb-line-choice[viz]'") from exc

    sep: SeparationProfile = separation_profile(profiles)
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, height_ratios=[3, 1], constrained_layout=True
    )

    groups = sorted(set(labels.values())) if labels else []
    colors = {g: c for g, c in zip(groups, ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"])}
    seen: set[str] = set()
    for p in profiles:
        group = labels.get(p.run_id) if labels else None
        color = colors.get(group, "#666666")
        label = group if group and group not in seen else None
        if group:
            seen.add(group)
        ax.plot(p.s, p.d, lw=1.2, alpha=0.85, color=color, label=label)

    if split_window:
        for a in (ax, ax2):
            a.axvspan(*split_window, color="#ffd166", alpha=0.25, zorder=0)
    ax.axhline(0.0, color="#333333", lw=0.8, ls="--")
    ax.set_ylabel("lateral offset d (m)\n<- rider's left    rider's right ->")
    ax.set_title(title)
    if seen:
        ax.legend(loc="upper right", frameon=False)
    ax.grid(alpha=0.25)

    spread = sep.spread
    ax2.fill_between(sep.s, 0, spread, color="#118ab2", alpha=0.5)
    ax2.axhline(sep.baseline_spread, color="#333333", lw=0.8, ls=":",
                label=f"baseline {sep.baseline_spread:.2f} m")
    for dp in sep.decision_points():
        ax2.annotate(
            f"{dp['spread_m']:.1f} m", xy=(dp["s"], dp["spread_m"]),
            xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8,
        )
        ax.axvline(dp["s"], color="#ef476f", lw=0.9, alpha=0.6)
    ax2.set_ylabel("spread (m)")
    ax2.set_xlabel("arc length along trail s (m)")
    ax2.legend(loc="upper right", frameon=False, fontsize=8)
    ax2.grid(alpha=0.25)

    out_path = Path(out_path)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
