"""Turning raw onboard footage into frames a reconstructor can use.

Two things matter here and both are about motion blur. Reconstruction and
feature matching both degrade sharply on blurred frames, and an onboard MTB clip
is mostly blurred frames -- so the frame *selection* step is not housekeeping,
it is a large part of whether Gate 1 works at all.

Downscaling to 1080p is deliberate, not a compromise: SuperPoint/LightGlue
matching gains nothing from 4K here, while 4K would exhaust local disk within a
handful of clips.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

DEFAULT_WIDTH = 1920


def require_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it with `brew install ffmpeg` "
            "(macOS) or your distro's package manager."
        )
    return exe


def extract_frames(
    video: str | Path,
    out_dir: str | Path,
    fps: float = 6.0,
    max_width: int = DEFAULT_WIDTH,
    start: float | None = None,
    duration: float | None = None,
) -> list[Path]:
    """Decode a clip to JPEG frames at a fixed rate.

    `fps` trades reconstruction density against cost. 6fps on a 30-second
    segment gives ~180 frames, which is comfortably inside the 20-200 range
    that photogrammetry and 3DGS pipelines expect.
    """
    exe = require_ffmpeg()
    video, out_dir = Path(video), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [exe, "-hide_banner", "-loglevel", "error", "-y"]
    if start is not None:
        cmd += ["-ss", str(start)]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += [
        "-i", str(video),
        "-vf", f"fps={fps},scale='min({max_width},iw)':-2",
        "-q:v", "2",
        str(out_dir / "frame_%05d.jpg"),
    ]
    subprocess.run(cmd, check=True)
    return sorted(out_dir.glob("frame_*.jpg"))


def sharpness(paths: list[Path]) -> np.ndarray:
    """Variance of the Laplacian per frame -- higher is sharper.

    Absolute values are not comparable across trails (a rock garden is texturally
    busier than loam), so callers should threshold on a percentile within one
    clip rather than on a fixed number.
    """
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required for blur filtering: pip install pillow") from exc

    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float)
    scores = []
    for path in paths:
        img = np.asarray(Image.open(path).convert("L").resize((480, 270)), dtype=float)
        lap = sum(
            kernel[i + 1, j + 1] * np.roll(np.roll(img, i, 0), j, 1)
            for i in (-1, 0, 1) for j in (-1, 0, 1) if kernel[i + 1, j + 1]
        )
        scores.append(float(np.var(lap[1:-1, 1:-1])))
    return np.array(scores)


def drop_blurred(paths: list[Path], keep_fraction: float = 0.7) -> tuple[list[Path], list[Path]]:
    """Split frames into (kept, rejected) by within-clip sharpness percentile."""
    if not paths:
        return [], []
    scores = sharpness(paths)
    cutoff = np.percentile(scores, (1.0 - keep_fraction) * 100.0)
    kept = [p for p, s in zip(paths, scores) if s >= cutoff]
    dropped = [p for p, s in zip(paths, scores) if s < cutoff]
    return kept, dropped
