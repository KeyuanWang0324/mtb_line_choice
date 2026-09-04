"""The one data structure worth freezing early.

Every downstream question -- "where do riders disagree", "which line was fastest",
"which line is within my ability" -- is a query against a TrailSegment. Pinning
it down now means the reconstruction backend and the relocalizer can both be
swapped later without rewriting the analysis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1


@dataclass
class Reconstruction:
    """Sparse/dense geometry for one trail segment, in an arbitrary world frame.

    `scale_source` records how metric scale was established, because monocular
    reconstruction has none of its own and a segment whose scale came from a
    guess must never be compared against one whose scale came from GNSS.
    """

    points: np.ndarray  # (P,3)
    scale_source: str = "unknown"  # gpmf_gps | arkit_vio | known_object | barometer | unknown
    up: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    backend: str = "unknown"  # e.g. "depth-anything-3/DA3-METRIC-LARGE"
    notes: str = ""

    @property
    def is_metric(self) -> bool:
        return self.scale_source != "unknown"


@dataclass
class Trajectory:
    """One rider's registered run through a Reconstruction's world frame."""

    run_id: str
    xyz: np.ndarray  # (N,3) camera centres, world frame
    t: np.ndarray | None = None  # (N,) seconds from clip start
    speed: np.ndarray | None = None  # (N,) m/s from telemetry, if available
    rider: str = ""
    bike: str = ""  # wheel size / travel -- lines are not transferable across these
    conditions: str = ""  # dry | damp | wet | loose ...
    reloc_inliers: np.ndarray | None = None  # (N,) per-frame localization inlier count
    source_video: str = ""

    def __len__(self) -> int:
        return len(self.xyz)


@dataclass
class TrailSegment:
    """A reconstruction plus every run registered into it."""

    segment_id: str
    reconstruction: Reconstruction
    trajectories: list[Trajectory] = field(default_factory=list)
    semantics: list[dict] = field(default_factory=list)  # VLM labels: roots, rock, berm, split
    outcomes: dict[str, dict] = field(default_factory=dict)  # run_id -> {"time_s": ..., ...}

    def trajectory(self, run_id: str) -> Trajectory:
        for t in self.trajectories:
            if t.run_id == run_id:
                return t
        raise KeyError(run_id)

    def save(self, path: str | Path) -> Path:
        """Arrays to .npz, everything else to a sidecar .json."""
        path = Path(path).with_suffix(".npz")
        arrays: dict[str, np.ndarray] = {"recon_points": self.reconstruction.points,
                                         "recon_up": self.reconstruction.up}
        meta: dict = {
            "schema_version": SCHEMA_VERSION,
            "segment_id": self.segment_id,
            "reconstruction": {
                "scale_source": self.reconstruction.scale_source,
                "backend": self.reconstruction.backend,
                "notes": self.reconstruction.notes,
            },
            "trajectories": [],
            "semantics": self.semantics,
            "outcomes": self.outcomes,
        }
        for i, tr in enumerate(self.trajectories):
            for name in ("xyz", "t", "speed", "reloc_inliers"):
                value = getattr(tr, name)
                if value is not None:
                    arrays[f"traj{i}_{name}"] = np.asarray(value)
            meta["trajectories"].append(
                {"index": i, "run_id": tr.run_id, "rider": tr.rider, "bike": tr.bike,
                 "conditions": tr.conditions, "source_video": tr.source_video}
            )
        np.savez_compressed(path, **arrays)
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TrailSegment":
        path = Path(path).with_suffix(".npz")
        arrays = np.load(path)
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if meta.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"{path} has schema version {meta.get('schema_version')}, "
                f"this build expects {SCHEMA_VERSION}"
            )
        recon = Reconstruction(
            points=arrays["recon_points"],
            up=arrays["recon_up"],
            **meta["reconstruction"],
        )
        trajectories = []
        for entry in meta["trajectories"]:
            i = entry.pop("index")
            trajectories.append(
                Trajectory(
                    xyz=arrays[f"traj{i}_xyz"],
                    t=arrays.get(f"traj{i}_t"),
                    speed=arrays.get(f"traj{i}_speed"),
                    reloc_inliers=arrays.get(f"traj{i}_reloc_inliers"),
                    **entry,
                )
            )
        return cls(
            segment_id=meta["segment_id"],
            reconstruction=recon,
            trajectories=trajectories,
            semantics=meta.get("semantics", []),
            outcomes=meta.get("outcomes", {}),
        )
