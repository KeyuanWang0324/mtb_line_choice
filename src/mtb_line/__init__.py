"""Recover per-rider 3D lines from onboard MTB video and compare them in trail coordinates.

The pipeline is deliberately staged so the risky part runs first (see docs/next_steps.md):

    video -> frames -> reconstruction -> per-run 6DoF trajectories
          -> centerline -> per-run (s, d) profiles -> Gate 1 verdict

Only the last three stages are novel; everything before them delegates to
Depth Anything 3 and hloc. The (s, d) representation is the point: it is a far
weaker requirement than globally accurate 6DoF pose, and it is what the product
actually needs -- "how far along the trail" and "how far off the centerline".
"""

from mtb_line.types import Reconstruction, TrailSegment, Trajectory
from mtb_line.centerline import Centerline, fit_centerline
from mtb_line.frenet import FrenetProfile, project_to_frenet

__all__ = [
    "Reconstruction",
    "TrailSegment",
    "Trajectory",
    "Centerline",
    "fit_centerline",
    "FrenetProfile",
    "project_to_frenet",
]
