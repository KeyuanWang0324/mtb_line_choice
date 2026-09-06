"""Render a synthetic trail corridor with exact ground-truth camera poses.

Gate 0 asks whether a reconstructor recovers usable geometry from trail footage.
Real footage cannot answer that cleanly, because there is no ground truth to
check against -- which is why Gate 0's criteria fall back on a barometer and hand
measurement. This renderer supplies a scene where the answer is known exactly,
so the reconstruction stage can be debugged before any filming happens.

What it deliberately does NOT reproduce: motion blur, rolling shutter, exposure
shifts, wet glare, foliage that moves between runs. Those are the reasons real
trail reconstruction is hard. A pipeline that fails here is broken; a pipeline
that passes here has only earned the right to be tried on real video.

The ground is a heightfield rather than a plane on purpose. A planar scene is
degenerate for structure-from-motion -- pose recovery is ambiguous on it -- so a
flat test track would flatter the pipeline in a way a real trail never does.

    python scripts/render_synthetic_trail.py --out /tmp/trail --n 24
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _value_noise(x: np.ndarray, y: np.ndarray, freq: float, seed: int) -> np.ndarray:
    """Cheap smooth noise: hashed lattice with cubic interpolation."""
    xf, yf = x * freq, y * freq
    x0, y0 = np.floor(xf).astype(np.int64), np.floor(yf).astype(np.int64)
    tx, ty = xf - x0, yf - y0
    tx, ty = tx * tx * (3 - 2 * tx), ty * ty * (3 - 2 * ty)

    mask = np.int64(0x7FFFFFFF)

    def h(i: np.ndarray, j: np.ndarray) -> np.ndarray:
        # Constants kept small enough that every intermediate stays inside
        # int64; a wider mixing constant silently overflows on this path.
        n = (i * np.int64(374761393)) ^ (j * np.int64(668265263)) ^ np.int64(seed * 1013904223)
        n &= mask
        n = ((n ^ (n >> 13)) * np.int64(1274126177)) & mask
        return n / float(1 << 31)

    n00, n10 = h(x0, y0), h(x0 + 1, y0)
    n01, n11 = h(x0, y0 + 1), h(x0 + 1, y0 + 1)
    return (n00 * (1 - tx) + n10 * tx) * (1 - ty) + (n01 * (1 - tx) + n11 * tx) * ty


def height(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Trail corridor: descending, with banks either side and surface roughness."""
    descent = -0.12 * x
    corridor = 0.55 * np.clip(np.abs(y) - 1.6, 0.0, None) ** 1.6  # banks
    rocks = 0.06 * _value_noise(x, y, 1.7, 11) + 0.02 * _value_noise(x, y, 6.0, 12)
    return descent + corridor + rocks


def albedo(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """High-frequency texture so feature matching has something to lock onto."""
    dirt = 0.30 + 0.30 * _value_noise(x, y, 3.0, 21) + 0.18 * _value_noise(x, y, 11.0, 22)
    litter = 0.22 * _value_noise(x, y, 27.0, 23)
    base = np.clip(dirt + litter, 0.05, 1.0)
    tint = np.stack([base * 1.00, base * 0.86, base * 0.66], axis=-1)  # brown
    mossy = (_value_noise(x, y, 2.2, 24) > 0.62)[..., None]
    return np.where(mossy, tint * np.array([0.65, 0.95, 0.55]), tint)


def landmarks(n: int = 26, length: float = 30.0, seed: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Boulders beside the trail, at distinct positions with distinct colours.

    These are load-bearing, not scenery. Without them the corridor is
    statistically homogeneous along its length: every metre of it looks like
    every other metre, so a reconstructor cannot observe forward translation at
    all and correctly reports the camera as stationary. Real trails are full of
    landmarks -- one particular root, one particular rock -- and a synthetic
    scene that omits them tests a problem the real one does not have.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(1.0, length, n) + rng.normal(0, 0.35, n)
    side = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    y = side * rng.uniform(1.7, 3.2, n)
    radius = rng.uniform(0.28, 0.65, n)
    centres = np.stack([x, y, height(x, y) + radius * 0.45], axis=1)
    hue = rng.uniform(0, 1, (n, 3)) * 0.55 + 0.35
    return centres, radius, hue


_LM_C, _LM_R, _LM_HUE = landmarks()


def _ray_spheres(origin: np.ndarray, dirs: np.ndarray, max_dist: float):
    """Nearest sphere hit per ray. Returns (t, colour, hit)."""
    best_t = np.full(dirs.shape[:2], np.inf)
    best_c = np.zeros(dirs.shape[:2] + (3,))
    for centre, radius, hue in zip(_LM_C, _LM_R, _LM_HUE):
        oc = origin - centre
        b = np.einsum("ijk,k->ij", dirs, oc)
        disc = b * b - (oc @ oc - radius * radius)
        ok = disc > 0
        if not ok.any():
            continue
        t = np.where(ok, -b - np.sqrt(np.maximum(disc, 0)), np.inf)
        closer = ok & (t > 0.2) & (t < best_t) & (t < max_dist)
        best_t = np.where(closer, t, best_t)
        best_c = np.where(closer[..., None], hue, best_c)
    return best_t, best_c, np.isfinite(best_t)


def render(cam_pos: np.ndarray, look_at: np.ndarray, width: int, height_px: int,
           fov_deg: float = 75.0, max_dist: float = 45.0) -> np.ndarray:
    """Raymarch the heightfield from one camera pose. Returns uint8 RGB."""
    forward = look_at - cam_pos
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)

    f = 0.5 * width / np.tan(np.radians(fov_deg) / 2)
    px, py = np.meshgrid(np.arange(width), np.arange(height_px))
    dirs = (
        forward[None, None, :]
        + ((px - width / 2) / f)[..., None] * right[None, None, :]
        - ((py - height_px / 2) / f)[..., None] * up[None, None, :]
    )
    dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)

    t = np.full(dirs.shape[:2], 0.25)
    hit = np.zeros(dirs.shape[:2], dtype=bool)
    for _ in range(96):
        p = cam_pos[None, None, :] + t[..., None] * dirs
        hit |= p[..., 2] < height(p[..., 0], p[..., 1])
        t = np.where(hit, t, np.minimum(t * 1.06 + 0.05, max_dist))

    sphere_t, sphere_c, sphere_hit = _ray_spheres(cam_pos, dirs, max_dist)
    use_sphere = sphere_hit & (sphere_t < t)

    p = cam_pos[None, None, :] + t[..., None] * dirs
    colour = albedo(p[..., 0], p[..., 1])

    # Lambertian shade off the heightfield normal, so geometry reads as shading
    # and not only as texture.
    eps = 0.05
    hx = height(p[..., 0] + eps, p[..., 1]) - height(p[..., 0] - eps, p[..., 1])
    hy = height(p[..., 0], p[..., 1] + eps) - height(p[..., 0], p[..., 1] - eps)
    normal = np.stack([-hx / (2 * eps), -hy / (2 * eps), np.ones_like(hx)], axis=-1)
    normal = normal / np.linalg.norm(normal, axis=-1, keepdims=True)
    sun = np.array([0.35, 0.25, 0.90])
    sun = sun / np.linalg.norm(sun)
    shade = 0.35 + 0.65 * np.clip(normal @ sun, 0, 1)
    colour = colour * shade[..., None]

    # Composite landmarks over the ground, flat-shaded so they stay distinctive.
    t = np.where(use_sphere, sphere_t, t)
    hit |= use_sphere
    colour = np.where(use_sphere[..., None], sphere_c * 0.85, colour)

    fog = np.clip(t / max_dist, 0, 1)[..., None]
    sky = np.array([0.62, 0.70, 0.82])
    colour = colour * (1 - fog) + sky * fog
    colour = np.where(hit[..., None], colour, sky)
    return (np.clip(colour, 0, 1) * 255).astype(np.uint8)


def camera_path(n: int, lateral_offset: float = 0.0, length: float = 22.0,
                sway: float = 0.9) -> np.ndarray:
    """Camera centres for one run down the corridor, at bar height.

    `sway` is the lateral amplitude. It is a real parameter of the test, not a
    styling choice: pure forward motion is the degenerate case for
    structure-from-motion -- baseline lies along the viewing direction, so
    triangulation is ill-conditioned. A test track with no sway measures the
    reconstructor at its worst rather than at its typical.
    """
    s = np.linspace(2.0, 2.0 + length, n)
    y = sway * np.sin(s / 7.0) + lateral_offset
    z = height(s, y) + 1.15
    return np.stack([s, y, z], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=24, help="frames")
    ap.add_argument("--width", type=int, default=518)
    ap.add_argument("--height", type=int, default=392)
    ap.add_argument("--lateral", type=float, default=0.0, help="offset from corridor centre, m")
    ap.add_argument("--run-id", default="run")
    ap.add_argument("--sway", type=float, default=0.9, help="lateral amplitude, m")
    args = ap.parse_args()

    from PIL import Image

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    centres = camera_path(args.n, args.lateral, sway=args.sway)

    for i, c in enumerate(centres):
        if i < len(centres) - 3:
            target = centres[i + 3].copy()
        else:  # keep looking forward past the end of the run
            target = c + (c - centres[i - 3])
        target[2] -= 0.25  # eyes down the trail, as a rider looks
        Image.fromarray(render(c, target, args.width, args.height)).save(
            out / f"frame_{i:05d}.jpg", quality=92
        )

    (out / "poses.json").write_text(json.dumps({
        "run_id": args.run_id,
        "lateral_offset_m": args.lateral,
        "centres": centres.tolist(),
        "note": "ground-truth camera centres, world frame, metres",
    }, indent=2))
    print(f"rendered {args.n} frames to {out}")
    print(f"path length {np.linalg.norm(np.diff(centres, axis=0), axis=1).sum():.1f} m")


if __name__ == "__main__":
    main()
