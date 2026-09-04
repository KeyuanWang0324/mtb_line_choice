"""Depth Anything 3 adapter.

UNVALIDATED: written against DA3's documented CLI but not executed, because the
development machine has no CUDA device. Treat the first real run as part of
Gate 0, not as a regression.

Model choice is a licensing decision, not only a quality one. DA3-SMALL, DA3-BASE
and DA3METRIC-LARGE are Apache-2.0; the larger and nested variants are CC BY-NC
4.0 and cannot ship in a product. The default here is the largest Apache-2.0
metric model, so that a passing Gate 0 does not have to be re-run on a different
model later.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

from mtb_line.types import Reconstruction

APACHE_MODELS = {"da3-small", "da3-base", "da3metric-large"}
DEFAULT_MODEL = "da3metric-large"


def da3_available() -> bool:
    return shutil.which("da3") is not None


def read_ply(path: str | Path) -> np.ndarray:
    """Minimal binary/ascii PLY vertex reader -- positions only."""
    path = Path(path)
    with path.open("rb") as fh:
        if fh.readline().strip() != b"ply":
            raise ValueError(f"{path} is not a PLY file")
        fmt, count, props = None, 0, []
        while True:
            line = fh.readline().decode("ascii", "replace").strip()
            if not line:
                raise ValueError(f"{path}: unexpected end of header")
            parts = line.split()
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element" and parts[1] == "vertex":
                count = int(parts[2])
            elif parts[0] == "property" and count and not line.startswith("property list"):
                props.append((parts[1], parts[2]))
            elif parts[0] == "end_header":
                break
        names = [p[1] for p in props]
        for axis in ("x", "y", "z"):
            if axis not in names:
                raise ValueError(f"{path}: vertex element has no '{axis}' property")

        if fmt == "ascii":
            rows = [fh.readline().split() for _ in range(count)]
            data = np.array(rows, dtype=float)
            return data[:, [names.index("x"), names.index("y"), names.index("z")]]

        numpy_types = {"float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
                       "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
                       "short": "i2", "ushort": "u2", "int": "i4", "uint": "u4"}
        endian = "<" if fmt == "binary_little_endian" else ">"
        dtype = np.dtype([(n, endian + numpy_types[t]) for t, n in props])
        arr = np.frombuffer(fh.read(count * dtype.itemsize), dtype=dtype, count=count)
        return np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(float)


def reconstruct(
    frames_dir: str | Path,
    out_dir: str | Path,
    model: str = DEFAULT_MODEL,
    allow_noncommercial: bool = False,
    export: str = "ply",
) -> Reconstruction:
    """Run `da3 auto` over a frame directory and load the exported geometry."""
    if not da3_available():
        raise RuntimeError(
            "`da3` not on PATH. Install Depth Anything 3 "
            "(https://github.com/ByteDance-Seed/Depth-Anything-3) on a CUDA host; "
            "DA3-Streaming needs under 12GB VRAM, so a single 4090 or A10 is enough."
        )
    if model.lower() not in APACHE_MODELS and not allow_noncommercial:
        raise ValueError(
            f"{model!r} is CC BY-NC 4.0 and cannot be used in a product. Pick one of "
            f"{sorted(APACHE_MODELS)}, or pass allow_noncommercial=True for a "
            "research-only comparison."
        )

    frames_dir, out_dir = Path(frames_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["da3", "auto", "-i", str(frames_dir), "-o", str(out_dir),
         "--model", model, "--export", export],
        check=True,
    )
    plys = sorted(out_dir.rglob("*.ply"))
    if not plys:
        raise RuntimeError(f"da3 produced no .ply under {out_dir}")

    return Reconstruction(
        points=read_ply(plys[0]),
        scale_source="unknown",  # set by telemetry.scale_from_gps once a run is registered
        backend=f"depth-anything-3/{model}",
        notes=f"frames={len(list(frames_dir.glob('*.jpg')))}",
    )
