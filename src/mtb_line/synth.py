"""Synthetic trail segments, for exercising the analysis half of the pipeline.

Scope warning, because it is easy to over-read a green demo: this generator
validates the *analysis* stages only -- centerline fitting, (s, d) projection,
and Gate scoring. It says nothing about whether Depth Anything 3 can reconstruct
a forest or whether hloc can relocalize a motion-blurred run into it, which is
the actual risk Gate 1 exists to test. Passing here is necessary, not sufficient.

What it does buy: the analysis code can be developed and regression-tested on a
laptop with no GPU, with ground truth known exactly, before any footage exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mtb_line.centerline import Centerline
from mtb_line.types import Reconstruction, TrailSegment, Trajectory


def _smooth_noise(n: int, scale: float, correlation: int, rng: np.random.Generator) -> np.ndarray:
    """Low-frequency noise -- rider wander and reloc drift, not white jitter."""
    raw = rng.normal(0.0, 1.0, n + 4 * correlation)
    kernel = np.hanning(2 * correlation + 1)
    smoothed = np.convolve(raw, kernel / kernel.sum(), mode="same")[:n]
    peak = np.abs(smoothed).max()
    return smoothed / max(peak, 1e-9) * scale


def true_centerline(length_m: float = 120.0, spacing: float = 0.05) -> Centerline:
    """An S-bend descending at roughly 12% -- a plausible trail segment."""
    t = np.linspace(0.0, 1.0, int(length_m / spacing))
    xyz = np.stack([
        t * length_m,
        6.0 * np.sin(2.0 * np.pi * 1.2 * t),
        -0.12 * t * length_m,
    ], axis=1)
    return Centerline.from_polyline(xyz, spacing=spacing, smooth_window_m=1.0)


@dataclass
class SyntheticTruth:
    """Exactly what the generator put in, so recovery error can be measured.

    Separating these two is the whole point of having a synthetic mode:

      `lateral` is where the rider actually was -- signal plus rider wander.
      `measurement_error` is what was added on top to stand in for imperfect
      relocalization.

    A pipeline can only ever be held responsible for the second. The first is
    the sport, and no amount of accuracy removes it.
    """

    centerline: Centerline
    labels: dict[str, str]
    split_window: tuple[float, float]
    lateral: dict[str, np.ndarray] = field(default_factory=dict)
    s: dict[str, np.ndarray] = field(default_factory=dict)


def build_synthetic(
    n_left: int = 3,
    n_right: int = 3,
    split_window: tuple[float, float] = (48.0, 62.0),
    split_offset_m: float = 1.8,
    wander_m: float = 0.35,
    reloc_drift_m: float = 0.15,
    reloc_noise_m: float = 0.04,
    common_drift_m: float = 0.0,
    seed: int = 0,
) -> tuple[TrailSegment, SyntheticTruth]:
    """Build a segment where riders take a known A/B split, with ground truth.

    The runs differ in sample count (riders move at different speeds), carry
    correlated lateral wander, and each gets its own slow drift standing in for
    residual relocalization error -- the three things that break a naive
    "just average the trajectories" centerline.

    Args:
        wander_m: how much a rider varies run to run. Set to 0 to isolate
            measurement error from rider behaviour.
        reloc_drift_m: slow per-run registration error. This is the error that
            actually matters, because it does NOT cancel in `d`.
        common_drift_m: registration error shared by every run, standing in for
            error in the map itself. Expected to cancel almost entirely, since
            `d` is measured against a centerline fitted to the same bundle --
            which is why absolute map accuracy is far less critical here than
            run-to-run consistency.
    """
    rng = np.random.default_rng(seed)
    # Drawn unconditionally, even at zero amplitude, so that two segments built
    # with the same seed but different noise settings share an identical
    # underlying rider path. The clean twin is the ground truth for the noisy
    # one, which only works if the rng stream stays in lockstep.
    shared = np.stack([_smooth_noise(4096, max(common_drift_m, 0.0), 512, rng) for _ in range(3)], axis=1)
    cl = true_centerline()
    labels: dict[str, str] = {}
    trajectories: list[Trajectory] = []

    truth = SyntheticTruth(centerline=cl, labels=labels, split_window=split_window)
    plan = [("left", -1.0)] * n_left + [("right", +1.0)] * n_right
    for i, (side, sign) in enumerate(plan):
        run_id = f"{side}{i:02d}"
        labels[run_id] = side

        n = int(rng.integers(280, 460))  # speed differences -> sample-count differences
        s = np.linspace(2.0, cl.length - 2.0, n)

        lateral = _smooth_noise(n, wander_m, max(2, n // 12), rng)
        lo, hi = split_window
        # Ramp the branch in and out so the split has soft edges, as a real
        # line choice does -- riders commit before the feature and rejoin after.
        ramp = np.clip(np.minimum(s - lo, hi - s) / 4.0, 0.0, 1.0)
        lateral += sign * split_offset_m * ramp

        base = cl.interp_xyz(s)
        right = cl.interp_right(s)
        xyz = base + lateral[:, None] * right
        xyz[:, 2] += rng.normal(0.0, 0.03, n)  # rider height bob

        truth.lateral[run_id] = lateral.copy()
        truth.s[run_id] = s.copy()

        drift = np.stack([_smooth_noise(n, reloc_drift_m, max(2, n // 6), rng) for _ in range(3)], 1)
        # Same map error for every run, sampled at each run's own position.
        idx = np.linspace(0, len(shared) - 1, n).astype(int)
        drift = drift + shared[idx]
        xyz = xyz + drift + rng.normal(0.0, reloc_noise_m, xyz.shape)

        speed = np.full(n, cl.length / (n / 30.0))  # crude: constant m/s at 30fps
        trajectories.append(
            Trajectory(
                run_id=run_id,
                xyz=xyz,
                t=np.arange(n) / 30.0,
                speed=speed,
                rider=run_id,
                bike="synthetic",
                conditions="synthetic",
            )
        )

    recon = Reconstruction(
        points=cl.xyz.copy(),
        scale_source="unknown",
        backend="synthetic",
        notes="Generated by mtb_line.synth -- analysis-stage validation only.",
    )
    segment = TrailSegment(segment_id="synthetic-s-bend", reconstruction=recon,
                           trajectories=trajectories)
    return segment, truth


def synthetic_segment(**kwargs) -> tuple[TrailSegment, dict[str, str], tuple[float, float]]:
    """`build_synthetic` without the ground truth, for callers that only score."""
    segment, truth = build_synthetic(**kwargs)
    return segment, truth.labels, truth.split_window
