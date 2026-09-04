"""Fitting a trail centerline from a bundle of registered rider trajectories.

There is no surveyed centerline for a mountain bike trail, and the GPS track on
Trailforks is nowhere near accurate enough to serve as one. So the centerline is
derived from the runs themselves: it is the principal curve of the trajectory
bundle -- roughly "where riders go on average" -- and exists only as a reference
axis for measuring how far each individual line deviates from it.

This means the centerline carries no authority. It is not the recommended line
and it is not the trail's physical middle; if every rider in the bundle takes
the same wide entry, the centerline follows them there. Only differences in `d`
between runs are meaningful, never `d` itself.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

from mtb_line.frenet import _normalize, project_points_to_polyline

DEFAULT_SPACING = 0.05  # metres between centerline vertices


def resample_by_arclength(points: np.ndarray, spacing: float = DEFAULT_SPACING) -> np.ndarray:
    """Resample a polyline to uniform arc-length spacing."""
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        raise ValueError("need at least 2 points to resample")
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(steps)])
    keep = np.concatenate([[True], steps > 1e-9])
    cum, points = cum[keep], points[keep]
    if cum[-1] < spacing:
        return points
    grid = np.arange(0.0, cum[-1], spacing)
    return np.stack([np.interp(grid, cum, points[:, i]) for i in range(3)], axis=1)


def _smooth(points: np.ndarray, window_m: float, spacing: float) -> np.ndarray:
    """Savitzky-Golay along each axis, with the window given in metres."""
    win = int(round(window_m / spacing))
    win = max(5, win | 1)  # odd, at least 5
    if win >= len(points):
        win = max(5, (len(points) - 1) | 1)
    if win >= len(points) or len(points) < 7:
        return points
    return savgol_filter(points, window_length=win, polyorder=2, axis=0)


@dataclass
class Centerline:
    """A densely resampled reference axis with a Frenet-like frame attached.

    The frame is built against a fixed `up` vector rather than the curve's own
    normal, so it does not twist through inflection points -- Frenet normals flip
    sign where curvature vanishes, which on a trail would swap left and right
    mid-segment.
    """

    xyz: np.ndarray  # (M,3) vertices, uniform arc-length spacing
    s: np.ndarray  # (M,) cumulative arc length
    tangent: np.ndarray  # (M,3) unit forward
    right: np.ndarray  # (M,3) unit rider's-right
    up: np.ndarray  # (3,) reference up used to build the frame

    @property
    def length(self) -> float:
        return float(self.s[-1])

    @classmethod
    def from_polyline(
        cls,
        points: np.ndarray,
        spacing: float = DEFAULT_SPACING,
        up: np.ndarray | None = None,
        smooth_window_m: float = 2.0,
    ) -> "Centerline":
        up_vec = _normalize(np.asarray(up if up is not None else [0.0, 0.0, 1.0], dtype=float))
        xyz = resample_by_arclength(points, spacing)
        xyz = _smooth(xyz, smooth_window_m, spacing)

        steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(steps)])
        tangent = _normalize(np.gradient(xyz, axis=0))

        right = np.cross(tangent, up_vec)
        norms = np.linalg.norm(right, axis=1)
        # Where the trail runs (near-)parallel to `up` the cross product
        # degenerates; carry the last good frame through instead of emitting NaN.
        bad = norms < 1e-6
        if bad.any():
            good = np.flatnonzero(~bad)
            if len(good) == 0:
                raise ValueError("centerline is everywhere parallel to `up`")
            right[bad] = right[good[np.searchsorted(good, np.flatnonzero(bad)).clip(0, len(good) - 1)]]
        right = _normalize(right)

        return cls(xyz=xyz, s=s, tangent=tangent, right=right, up=up_vec)

    def _interp(self, field: np.ndarray, s_query: np.ndarray) -> np.ndarray:
        s_query = np.atleast_1d(np.asarray(s_query, dtype=float))
        return np.stack([np.interp(s_query, self.s, field[:, i]) for i in range(3)], axis=1)

    def interp_right(self, s_query: np.ndarray) -> np.ndarray:
        return self._interp(self.right, s_query)

    def interp_tangent(self, s_query: np.ndarray) -> np.ndarray:
        return self._interp(self.tangent, s_query)

    def interp_xyz(self, s_query: np.ndarray) -> np.ndarray:
        return self._interp(self.xyz, s_query)


def _run_on_grid(traj: np.ndarray, centerline: Centerline, grid: np.ndarray) -> np.ndarray:
    """Resample one run onto a shared arc-length grid. NaN outside its coverage."""
    s, _, _ = project_points_to_polyline(traj, centerline.xyz, centerline.s)
    order = np.argsort(s)
    s_sorted, xyz_sorted = s[order], traj[order]
    keep = np.concatenate([[True], np.diff(s_sorted) > 1e-6])
    s_sorted, xyz_sorted = s_sorted[keep], xyz_sorted[keep]
    out = np.full((len(grid), 3), np.nan)
    if len(s_sorted) < 2:
        return out
    inside = (grid >= s_sorted[0]) & (grid <= s_sorted[-1])
    for i in range(3):
        out[inside, i] = np.interp(grid[inside], s_sorted, xyz_sorted[:, i])
    return out


def fit_centerline(
    trajectories: list[np.ndarray],
    spacing: float = DEFAULT_SPACING,
    step: float = 0.5,
    n_iter: int = 8,
    min_runs: int = 2,
    tol: float = 0.01,
    up: np.ndarray | None = None,
    smooth_window_m: float = 2.0,
) -> Centerline:
    """Principal-curve fit over a bundle of trajectories in a shared frame.

    Iterates: project every run onto the current centerline, resample each run
    onto a common arc-length grid, average across runs, refit.

    The aggregation is interpolation onto a shared grid, not binning of raw
    samples, and the difference is not cosmetic. Riders are sampled at fixed
    frame rate, so sample density along the trail is a function of *speed* --
    bin a fast run and a slow run together and the slow one dominates. Worse, at
    a genuine A/B split, bins whose membership happens to be all-left followed by
    bins that happen to be all-right make the mean curve jump the full width of
    the split, which smears the very feature Gate 1 is trying to measure.
    Interpolation gives every run a value at every grid point it spans, so
    coverage changes only at the ends of a run, where it is handled by trimming.

    Args:
        trajectories: list of (N_i, 3) arrays, all already registered into one
            world frame -- this is what Gate 1's relocalization step produces.
        step: arc-length spacing of the aggregation grid, metres.
        min_runs: grid points backed by fewer runs are trimmed, dropping the
            ragged ends where only one run has coverage.
        tol: stop once the centerline moves less than this (metres, RMS).
    """
    trajectories = [np.atleast_2d(np.asarray(t, dtype=float)) for t in trajectories]
    trajectories = [t for t in trajectories if len(t) >= 2]
    if not trajectories:
        raise ValueError("need at least one trajectory with >= 2 points")

    # Seed with the longest run: most likely to span the whole segment.
    lengths = [np.linalg.norm(np.diff(t, axis=0), axis=1).sum() for t in trajectories]
    seed = resample_by_arclength(trajectories[int(np.argmax(lengths))], spacing)
    centerline = Centerline.from_polyline(seed, spacing, up, smooth_window_m)
    if len(trajectories) == 1:
        return centerline

    required = min(min_runs, len(trajectories))
    for _ in range(n_iter):
        grid = np.arange(0.0, centerline.length, step)
        if len(grid) < 3:
            break
        stack = np.stack([_run_on_grid(t, centerline, grid) for t in trajectories], axis=0)
        coverage = np.sum(~np.isnan(stack[:, :, 0]), axis=0)
        keep = coverage >= required
        if keep.sum() < 3:
            break
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns are expected
            mean_curve = np.nanmean(stack, axis=0)

        first, last = np.flatnonzero(keep)[[0, -1]]
        span = mean_curve[first : last + 1]
        rows = np.flatnonzero(~np.isnan(span[:, 0]))
        if len(rows) < 3:
            break
        span = np.stack(
            [np.interp(np.arange(len(span)), rows, span[rows, i]) for i in range(3)], axis=1
        )

        previous = centerline
        centerline = Centerline.from_polyline(span, spacing, up, smooth_window_m)
        shift = _rms_shift(previous, centerline)
        if shift < tol:
            break

    return centerline


def _rms_shift(a: Centerline, b: Centerline) -> float:
    """RMS distance between two centerlines, sampled on the shorter one."""
    n = min(len(a.xyz), len(b.xyz), 400)
    probe = b.interp_xyz(np.linspace(0.0, b.length, n))
    _, _, dist = project_points_to_polyline(probe, a.xyz, a.s)
    return float(np.sqrt(np.mean(dist**2)))
