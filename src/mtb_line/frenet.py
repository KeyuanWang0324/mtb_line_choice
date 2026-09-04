"""Projection of 3D rider trajectories into trail coordinates (s, d, h).

Why this representation, and not 6DoF pose:

    s  arc length along the trail centerline   -- "how far down the track am I"
    d  signed lateral offset from centerline   -- "how far left/right is my line"
    h  signed vertical offset from centerline  -- "am I in the rut or on the berm"

Global 6DoF relocalization accuracy is a much stronger requirement than the
product needs, and drift in height/pitch/roll -- the components that degrade
first in forest reconstructions -- does not contaminate `d`. Gate 1 is therefore
scored on (s, d), not on pose error. See docs/next_steps.md.

Sign convention: `right = normalize(cross(tangent, up))`, so d > 0 means the
rider is to the rider's right of the centerline when travelling in +s.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import cKDTree

if TYPE_CHECKING:  # pragma: no cover
    from mtb_line.centerline import Centerline


def _normalize(v: np.ndarray, axis: int = -1) -> np.ndarray:
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, 1e-12)


def project_points_to_polyline(
    points: np.ndarray,
    verts: np.ndarray,
    cum_s: np.ndarray,
    window: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest point on a polyline, for each query point.

    Coarse nearest-vertex lookup via KD-tree, then exact projection onto the
    ``2*window`` segments around it. With a densely resampled centerline the
    coarse step is already within a few centimetres, so the window only has to
    cover local curvature.

    Returns ``(s, closest_xyz, dist)``.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    verts = np.asarray(verts, dtype=float)
    cum_s = np.asarray(cum_s, dtype=float)
    if len(verts) < 2:
        raise ValueError("polyline needs at least 2 vertices")

    n_seg = len(verts) - 1
    _, nearest = cKDTree(verts).query(points, k=1)
    nearest = np.asarray(nearest).reshape(-1)

    a = verts[:-1]
    ab = verts[1:] - a
    seg_len = np.linalg.norm(ab, axis=1)
    ab_sq = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-24)

    best_dist = np.full(len(points), np.inf)
    best_s = np.zeros(len(points))
    best_pt = np.zeros((len(points), 3))

    for offset in range(-window, window):
        seg = np.clip(nearest + offset, 0, n_seg - 1)
        t = np.einsum("ij,ij->i", points - a[seg], ab[seg]) / ab_sq[seg]
        t = np.clip(t, 0.0, 1.0)
        proj = a[seg] + t[:, None] * ab[seg]
        dist = np.linalg.norm(points - proj, axis=1)
        better = dist < best_dist
        best_dist = np.where(better, dist, best_dist)
        best_s = np.where(better, cum_s[seg] + t * seg_len[seg], best_s)
        best_pt = np.where(better[:, None], proj, best_pt)

    return best_s, best_pt, best_dist


@dataclass
class FrenetProfile:
    """One rider's line expressed in trail coordinates."""

    run_id: str
    s: np.ndarray  # (N,) arc length along centerline, metres
    d: np.ndarray  # (N,) signed lateral offset, metres (+ = rider's right)
    h: np.ndarray  # (N,) signed vertical offset, metres
    residual: np.ndarray  # (N,) 3D distance to centerline, metres
    xyz: np.ndarray | None = None  # (N,3) source points, world frame
    speed: np.ndarray | None = None  # (N,) m/s, if telemetry was available

    def __len__(self) -> int:
        return len(self.s)

    @property
    def s_range(self) -> tuple[float, float]:
        return float(self.s.min()), float(self.s.max())

    @property
    def monotonic_fraction(self) -> float:
        """Fraction of consecutive samples that advance down the trail.

        A quality signal, not a correctness check: a run whose recovered `s`
        jitters backwards is a relocalization failure, and this catches it
        before the number reaches a chart. Healthy runs sit above ~0.98.
        """
        if len(self.s) < 2:
            return 1.0
        return float(np.mean(np.diff(self.s) > 0))

    def resample(self, s_grid: np.ndarray) -> "FrenetProfile":
        """Interpolate onto a shared s grid so runs can be compared pointwise.

        Samples outside this run's own s range become NaN rather than being
        extrapolated -- a run that only covers half the segment must not
        silently contribute a fabricated line to the other half.
        """
        s_grid = np.asarray(s_grid, dtype=float)
        order = np.argsort(self.s)
        s_sorted = self.s[order]
        lo, hi = self.s_range
        inside = (s_grid >= lo) & (s_grid <= hi)

        def interp(values: np.ndarray | None) -> np.ndarray | None:
            if values is None:
                return None
            out = np.full(len(s_grid), np.nan)
            out[inside] = np.interp(s_grid[inside], s_sorted, values[order])
            return out

        return FrenetProfile(
            run_id=self.run_id,
            s=s_grid,
            d=interp(self.d),
            h=interp(self.h),
            residual=interp(self.residual),
            xyz=None,
            speed=interp(self.speed),
        )


def project_to_frenet(
    centerline: "Centerline",
    points: np.ndarray,
    run_id: str = "run",
    speed: np.ndarray | None = None,
    window: int = 8,
) -> FrenetProfile:
    """Express a trajectory in the centerline's trail coordinates."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    s, closest, residual = project_points_to_polyline(
        points, centerline.xyz, centerline.s, window=window
    )

    # Interpolate the frame at the projected arc length. `offset` is already
    # perpendicular to the tangent, so decomposing it onto (right, up_local)
    # is exact rather than approximate.
    right = _normalize(centerline.interp_right(s))
    tangent = _normalize(centerline.interp_tangent(s))
    up_local = _normalize(np.cross(right, tangent))

    offset = points - closest
    d = np.einsum("ij,ij->i", offset, right)
    h = np.einsum("ij,ij->i", offset, up_local)

    return FrenetProfile(
        run_id=run_id,
        s=s,
        d=d,
        h=h,
        residual=residual,
        xyz=points,
        speed=None if speed is None else np.asarray(speed, dtype=float),
    )
