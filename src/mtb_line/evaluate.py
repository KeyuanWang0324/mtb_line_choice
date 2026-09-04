"""Gate 1 scoring: can the pipeline tell two riders' lines apart at all?

Gate 1a is deliberately the easy version of the question. It runs on a section
where riders take a visible A/B split -- around two metres apart, and obvious
enough that the correct answer can be read off the footage by eye. If the
pipeline cannot recover a separation that large with a free and uncontested
ground truth, it will never recover the ~30cm variation the product needs
(Gate 1b), and the project stops there.

Grouping, not absolute sign, is the criterion. The centerline is the mean of the
bundle, so where the split is lopsided it sits inside the larger group and that
group's `d` is pushed toward zero. What must survive is the *ordering* of the two
groups and the assignment of each run to the right side of the boundary between
them.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from mtb_line.frenet import FrenetProfile


def common_s_grid(profiles: list[FrenetProfile], spacing: float = 0.25) -> np.ndarray:
    """Arc-length grid spanning the range every run has in common."""
    if not profiles:
        raise ValueError("no profiles")
    lo = max(p.s_range[0] for p in profiles)
    hi = min(p.s_range[1] for p in profiles)
    if hi - lo < spacing:
        raise ValueError(
            f"runs share only {hi - lo:.2f}m of trail -- they are probably not "
            "registered into the same frame"
        )
    return np.arange(lo, hi, spacing)


@dataclass
class SeparationProfile:
    """How far apart the runs' lines are, as a function of position on trail."""

    s: np.ndarray
    d: np.ndarray  # (n_runs, n_s), NaN outside a run's coverage
    run_ids: list[str]

    @property
    def spread(self) -> np.ndarray:
        """Max-min lateral disagreement at each s, in metres."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmax(self.d, axis=0) - np.nanmin(self.d, axis=0)

    @property
    def baseline_spread(self) -> float:
        """Median disagreement -- the segment's ordinary rider-to-rider wander."""
        return float(np.nanmedian(self.spread))

    def decision_points(
        self, min_ratio: float = 2.0, min_spread: float = 1.0, min_gap_m: float = 5.0
    ) -> list[dict]:
        """Local maxima of disagreement -- where the trail actually offers a choice.

        This is the product-facing output of Gate 1: found from the data, with no
        hand-labelling. On a segment where every rider takes the same line it
        should correctly return nothing.

        A real choice has to stand out against the segment's own baseline wander,
        not just clear an absolute threshold -- every rider drifts half a metre
        everywhere, and calling that a decision point buries the actual splits in
        noise. Hence `min_ratio` against the median spread, with `min_spread` as
        an absolute floor for segments that are uniformly tight.
        """
        spread = self.spread
        threshold = max(min_spread, min_ratio * self.baseline_spread)
        found: list[dict] = []
        for i in np.argsort(spread)[::-1]:
            if not np.isfinite(spread[i]) or spread[i] < threshold:
                break
            if any(abs(self.s[i] - f["s"]) < min_gap_m for f in found):
                continue
            found.append({"s": float(self.s[i]), "spread_m": float(spread[i]),
                          "vs_baseline": float(spread[i] / max(self.baseline_spread, 1e-6))})
        return sorted(found, key=lambda f: f["s"])


def separation_profile(profiles: list[FrenetProfile], spacing: float = 0.25) -> SeparationProfile:
    grid = common_s_grid(profiles, spacing)
    stacked = np.stack([p.resample(grid).d for p in profiles], axis=0)
    return SeparationProfile(s=grid, d=stacked, run_ids=[p.run_id for p in profiles])


@dataclass
class GateResult:
    passed: bool
    accuracy: float  # fraction of runs assigned to the labelled group
    separation_m: float  # distance between the two group means
    effect_size: float  # separation / pooled within-group std
    per_run: dict[str, dict]
    reasons: list[str]

    def report(self) -> str:
        head = "PASS" if self.passed else "FAIL"
        lines = [
            f"Gate 1a: {head}",
            f"  group separation : {self.separation_m:.2f} m",
            f"  effect size      : {self.effect_size:.1f}",
            f"  side accuracy    : {self.accuracy:.0%} "
            f"({sum(r['correct'] for r in self.per_run.values())}/{len(self.per_run)} runs)",
            "",
            f"  {'run':<16}{'label':<8}{'mean d':>9}{'predicted':>12}{'':>4}",
        ]
        for run_id, r in self.per_run.items():
            mark = "ok" if r["correct"] else "MISS"
            lines.append(
                f"  {run_id:<16}{r['label']:<8}{r['mean_d']:>8.2f}m{r['predicted']:>12}{mark:>6}"
            )
        if self.reasons:
            lines += ["", "  " + "\n  ".join(self.reasons)]
        return "\n".join(lines)


def evaluate_split(
    profiles: list[FrenetProfile],
    labels: dict[str, str],
    s_window: tuple[float, float],
    min_separation_m: float = 1.0,
) -> GateResult:
    """Score one hand-labelled A/B split.

    Args:
        labels: run_id -> group name, read off the footage by eye. Exactly two
            distinct groups, both non-empty -- a split where every rider chose
            the same branch carries no information and is rejected rather than
            scored, since the centerline would simply follow them.
        s_window: (start, end) arc length of the split, metres.
        min_separation_m: group means must differ by at least this.
    """
    groups = sorted(set(labels.values()))
    if len(groups) != 2:
        raise ValueError(f"expected exactly 2 groups, got {groups}")

    lo, hi = s_window
    mean_d: dict[str, float] = {}
    for p in profiles:
        if p.run_id not in labels:
            continue
        mask = (p.s >= lo) & (p.s <= hi)
        if mask.sum() < 3:
            raise ValueError(
                f"run {p.run_id!r} has {mask.sum()} samples in s=[{lo}, {hi}] -- "
                "wrong window, or this run failed to localize"
            )
        mean_d[p.run_id] = float(np.mean(p.d[mask]))

    missing = set(labels) - set(mean_d)
    if missing:
        raise ValueError(f"labelled runs with no profile: {sorted(missing)}")

    by_group = {g: [mean_d[r] for r, lab in labels.items() if lab == g] for g in groups}
    if not all(by_group.values()):
        raise ValueError("both groups must contain at least one run")

    means = {g: float(np.mean(v)) for g, v in by_group.items()}
    low_group, high_group = sorted(groups, key=lambda g: means[g])
    boundary = (means[low_group] + means[high_group]) / 2.0
    separation = means[high_group] - means[low_group]

    within = np.concatenate([np.asarray(v) - means[g] for g, v in by_group.items()])
    pooled_std = float(np.std(within))
    effect = separation / pooled_std if pooled_std > 1e-9 else float("inf")

    per_run: dict[str, dict] = {}
    for run_id, value in sorted(mean_d.items(), key=lambda kv: kv[1]):
        predicted = low_group if value < boundary else high_group
        per_run[run_id] = {
            "label": labels[run_id],
            "mean_d": value,
            "predicted": predicted,
            "correct": predicted == labels[run_id],
        }

    accuracy = float(np.mean([r["correct"] for r in per_run.values()]))
    reasons: list[str] = []
    if accuracy < 1.0:
        reasons.append("Runs landed on the wrong side of the split -- relocalization is unreliable.")
    if separation < min_separation_m:
        reasons.append(
            f"Group separation {separation:.2f}m is below the {min_separation_m}m floor; "
            "the split is being smeared out, not resolved."
        )
    return GateResult(
        passed=accuracy == 1.0 and separation >= min_separation_m,
        accuracy=accuracy,
        separation_m=separation,
        effect_size=effect,
        per_run=per_run,
        reasons=reasons,
    )


def quality_report(profiles: list[FrenetProfile]) -> str:
    """Per-run health check. Read this before believing any Gate verdict."""
    lines = [
        "  |d| is line choice and is expected to be large; |h| is mostly vertical",
        "  drift and should stay small. A low monotonic % means the run's recovered",
        "  position jitters backwards down the trail -- a relocalization failure.",
        "",
        f"  {'run':<16}{'samples':>9}{'s span':>10}{'monotonic':>11}{'|d| p95':>10}{'|h| p95':>10}",
    ]
    for p in profiles:
        lo, hi = p.s_range
        d95 = float(np.nanpercentile(np.abs(p.d), 95))
        h95 = float(np.nanpercentile(np.abs(p.h), 95))
        flags = []
        if p.monotonic_fraction <= 0.98:
            flags.append("jittery")
        if h95 > 1.0:
            flags.append("vertical drift")
        note = ("  <- " + ", ".join(flags)) if flags else ""
        lines.append(
            f"  {p.run_id:<16}{len(p):>9}{hi - lo:>9.1f}m"
            f"{p.monotonic_fraction:>10.0%}{d95:>9.2f}m{h95:>9.2f}m{note}"
        )
    return "\n".join(lines)
