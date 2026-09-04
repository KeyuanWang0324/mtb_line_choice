"""Getting per-run 3D trajectories out of a localization run.

Deliberate boundary: this module does not wrap hloc. hloc and COLMAP are run
directly by the operator (recipe below), and what lives here is the conversion
of their output into `Trajectory` objects -- the part that is easy to get subtly
wrong and worth testing, as opposed to the part that is a shell command.

Recipe, per segment. Build a map from the sharpest run, then register the others
against it as queries:

    # 1. map from run A's frames
    python -m hloc.pipelines.build  # or the SfM notebook in cvg/Hierarchical-Localization
    #    features: superpoint_aachen, matcher: superpoint+lightglue

    # 2. register run B..N frames as queries against that map

    # 3. export to text so this module can read it
    colmap model_converter --input_path <sfm_dir> \\
        --output_path <sfm_dir>/txt --output_type TXT

Then `trajectory_from_colmap(".../txt/images.txt", run_id="right03")`.

The camera *centre* is what matters here, not the pose: it is the rider's head
position, which is what the line is made of.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mtb_line.types import Trajectory


def quat_to_rotation(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """COLMAP stores world-to-camera rotation as a Hamilton quaternion (w,x,y,z)."""
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n < 1e-12:
        raise ValueError("degenerate quaternion")
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def read_colmap_images(path: str | Path) -> dict[str, dict]:
    """Parse a COLMAP ``images.txt`` into ``{image_name: {R, t, center, n_points}}``.

    COLMAP's text format alternates a pose line with a 2D-point line per image;
    the point line is skipped but its length is kept, since the number of
    observed 3D points is the most direct per-frame confidence signal available.
    """
    entries: dict[str, dict] = {}
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    body = [ln for ln in lines if ln.strip() and not ln.startswith("#")]
    for i in range(0, len(body) - 1, 2):
        fields = body[i].split()
        if len(fields) < 10:
            continue
        qw, qx, qy, qz, tx, ty, tz = (float(v) for v in fields[1:8])
        name = fields[9]
        rot = quat_to_rotation(qw, qx, qy, qz)
        t = np.array([tx, ty, tz])
        point_fields = body[i + 1].split()
        # A 2D observation is (x, y, point3D_id); -1 means "not triangulated".
        n_points = sum(1 for j in range(2, len(point_fields), 3) if point_fields[j] != "-1")
        entries[name] = {"R": rot, "t": t, "center": -rot.T @ t, "n_points": n_points}
    return entries


def trajectory_from_colmap(
    images_txt: str | Path,
    run_id: str,
    name_filter: str | None = None,
    fps: float | None = None,
    min_points: int = 20,
    **meta,
) -> Trajectory:
    """Build one run's Trajectory from a COLMAP model.

    Frames are ordered by filename, which is why `extract_frames` emits
    zero-padded names -- lexical order is temporal order.

    Frames with fewer than `min_points` triangulated observations are dropped.
    A frame that localized against a handful of points is not a weak measurement
    to be smoothed over, it is usually a wrong one, and a single wrong camera
    centre puts a metre-scale spike straight into the line.
    """
    entries = read_colmap_images(images_txt)
    names = sorted(n for n in entries if name_filter is None or name_filter in n)
    if not names:
        raise ValueError(f"no images matching {name_filter!r} in {images_txt}")

    kept = [n for n in names if entries[n]["n_points"] >= min_points]
    if len(kept) < 2:
        raise ValueError(
            f"run {run_id!r}: only {len(kept)} of {len(names)} frames localized with "
            f">= {min_points} points -- this run did not register against the map"
        )

    xyz = np.stack([entries[n]["center"] for n in kept])
    inliers = np.array([entries[n]["n_points"] for n in kept])
    t = None
    if fps:
        # Index within the *full* frame list, so dropped frames leave real gaps
        # in time rather than compressing the run.
        order = {n: i for i, n in enumerate(names)}
        t = np.array([order[n] / fps for n in kept])

    return Trajectory(run_id=run_id, xyz=xyz, t=t, reloc_inliers=inliers, **meta)
