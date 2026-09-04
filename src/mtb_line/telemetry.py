"""GoPro GPMF telemetry: the cheapest source of metric scale.

A monocular reconstruction has no scale of its own, and every number the product
eventually reports -- a 40cm line difference, a 12% gradient -- is meaningless
without one. GoPro embeds gyro at 400Hz, accelerometer at 200Hz and GPS at 18Hz
directly in the MP4, which gives scale, gravity direction (hence a trustworthy
`up` for the trail frame) and a speed profile for free.

GPS alone is not enough for the *line* -- Racecraft Labs abandoned GPS timing for
exactly that reason -- but it is entirely adequate for fixing overall scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Telemetry:
    t: np.ndarray  # (N,) seconds from clip start
    gps_lla: np.ndarray | None = None  # (N,3) lat, lon, alt
    speed: np.ndarray | None = None  # (N,) m/s
    accel: np.ndarray | None = None  # (M,3) m/s^2
    gyro: np.ndarray | None = None  # (M,3) rad/s
    gravity: np.ndarray | None = None  # (3,) unit, in camera frame

    @property
    def has_scale(self) -> bool:
        return self.gps_lla is not None or self.speed is not None


def read_gpmf(video: str | Path) -> Telemetry:
    """Extract telemetry streams from a GoPro MP4.

    Requires `pip install py-gpmf-parser`. Falls back with a clear message
    rather than silently returning an empty Telemetry, because a segment that
    quietly loses its scale source is worse than one that fails loudly.
    """
    try:
        from py_gpmf_parser.gopro_telemetry_extractor import GoProTelemetryExtractor
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "py-gpmf-parser not installed. `pip install py-gpmf-parser`, or pass "
            "--scale-source manually if this footage is not from a GoPro."
        ) from exc

    extractor = GoProTelemetryExtractor(str(video))
    extractor.open_source()
    try:
        gps, gps_t = extractor.extract_data("GPS5")
        accel, _ = extractor.extract_data("ACCL")
        gyro, _ = extractor.extract_data("GYRO")
        try:
            grav, _ = extractor.extract_data("GRAV")
        except Exception:
            grav = None
    finally:
        extractor.close_source()

    gps = np.asarray(gps, dtype=float)
    speed = gps[:, 4] if gps.ndim == 2 and gps.shape[1] >= 5 else None
    gravity = None
    if grav is not None and len(np.asarray(grav)):
        g = np.asarray(grav, dtype=float).mean(axis=0)
        gravity = g / max(np.linalg.norm(g), 1e-9)

    return Telemetry(
        t=np.asarray(gps_t, dtype=float),
        gps_lla=gps[:, :3] if gps.ndim == 2 else None,
        speed=speed,
        accel=np.asarray(accel, dtype=float) if accel is not None else None,
        gyro=np.asarray(gyro, dtype=float) if gyro is not None else None,
        gravity=gravity,
    )


def scale_from_gps(trajectory_xyz: np.ndarray, telem: Telemetry) -> float:
    """Ratio that converts reconstruction units to metres.

    Compares the reconstructed path length against the GPS-integrated distance
    over the same clip. Deliberately a single global scalar: GPS is too noisy to
    correct scale locally, and a per-segment scalar is all a rigid reconstruction
    needs.
    """
    if telem.speed is None or telem.t is None or len(telem.t) < 2:
        raise ValueError("telemetry has no speed track to derive scale from")
    gps_distance = float(np.trapezoid(telem.speed, telem.t))
    recon_length = float(np.linalg.norm(np.diff(trajectory_xyz, axis=0), axis=1).sum())
    if recon_length < 1e-9:
        raise ValueError("reconstructed trajectory has zero length")
    return gps_distance / recon_length
