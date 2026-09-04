"""Command line entry points. Run `mtbline demo` first -- it needs no footage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _analyze(segment, labels=None, split_window=None, plot=None, verbose=True):
    from mtb_line.centerline import fit_centerline
    from mtb_line.evaluate import evaluate_split, quality_report, separation_profile
    from mtb_line.frenet import project_to_frenet

    centerline = fit_centerline(
        [t.xyz for t in segment.trajectories], up=segment.reconstruction.up
    )
    profiles = [
        project_to_frenet(centerline, t.xyz, run_id=t.run_id, speed=t.speed)
        for t in segment.trajectories
    ]
    if verbose:
        print(f"\ncenterline: {centerline.length:.1f} m from {len(profiles)} runs\n")
        print(quality_report(profiles))

    sep = separation_profile(profiles)
    if verbose:
        print(f"\nbaseline rider-to-rider spread: {sep.baseline_spread:.2f} m")
        points = sep.decision_points()
        if points:
            print("decision points (where the trail offers a real choice):")
            for dp in points:
                print(f"  s = {dp['s']:6.1f} m   spread {dp['spread_m']:5.2f} m "
                      f"({dp['vs_baseline']:.1f}x baseline)")
        else:
            print("no decision points: every rider took effectively the same line.")

    result = None
    if labels and split_window:
        result = evaluate_split(profiles, labels, split_window)
        if verbose:
            print()
            print(result.report())

    if plot:
        from mtb_line.viz import plot_lines
        path = plot_lines(profiles, plot, labels=labels, split_window=split_window)
        if verbose:
            print(f"\nchart written to {path}")
    return centerline, profiles, result


def cmd_demo(args) -> int:
    """End-to-end on synthetic data: no GPU, no footage, ground truth known."""
    from mtb_line.synth import synthetic_segment

    segment, labels, window = synthetic_segment(seed=args.seed)
    print("Synthetic segment -- validates the ANALYSIS stages only.")
    print("It says nothing about whether DA3 can reconstruct a forest or whether")
    print("hloc can localize a blurred run into it. That is what Gate 1 is for.")
    _, _, result = _analyze(segment, labels, window, plot=args.plot)
    return 0 if (result and result.passed) else 1


def cmd_frames(args) -> int:
    from mtb_line.video import drop_blurred, extract_frames

    paths = extract_frames(args.video, args.out, fps=args.fps, max_width=args.width,
                           start=args.start, duration=args.duration)
    print(f"extracted {len(paths)} frames to {args.out}")
    if args.keep_sharpest < 1.0:
        kept, dropped = drop_blurred(paths, args.keep_sharpest)
        for p in dropped:
            p.unlink()
        print(f"dropped {len(dropped)} blurred frames, kept {len(kept)}")
    return 0


def cmd_telemetry(args) -> int:
    from mtb_line.telemetry import read_gpmf

    telem = read_gpmf(args.video)
    print(f"samples      : {len(telem.t)}")
    print(f"duration     : {telem.t[-1] - telem.t[0]:.1f} s")
    if telem.speed is not None:
        print(f"speed        : mean {np.mean(telem.speed):.1f} m/s, "
              f"max {np.max(telem.speed):.1f} m/s")
        print(f"distance     : {np.trapezoid(telem.speed, telem.t):.1f} m")
    if telem.gravity is not None:
        print(f"gravity (cam): {np.round(telem.gravity, 3).tolist()}")
    return 0


def cmd_analyze(args) -> int:
    from mtb_line.types import TrailSegment

    segment = TrailSegment.load(args.segment)
    labels = json.loads(Path(args.labels).read_text()) if args.labels else None
    window = tuple(args.split) if args.split else None
    if bool(labels) != bool(window):
        print("error: --labels and --split must be given together", file=sys.stderr)
        return 2
    _, _, result = _analyze(segment, labels, window, plot=args.plot)
    return 0 if (result is None or result.passed) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mtbline", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("demo", help="run the analysis pipeline on synthetic data")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plot", type=Path, default=None, help="write a PNG here")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("frames", help="decode a clip to frames for reconstruction")
    p.add_argument("video")
    p.add_argument("--out", required=True)
    p.add_argument("--fps", type=float, default=6.0)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--start", type=float, default=None, help="seconds")
    p.add_argument("--duration", type=float, default=None, help="seconds")
    p.add_argument("--keep-sharpest", type=float, default=0.7,
                   help="fraction of frames to keep by sharpness (1.0 = keep all)")
    p.set_defaults(func=cmd_frames)

    p = sub.add_parser("telemetry", help="dump GoPro GPMF telemetry summary")
    p.add_argument("video")
    p.set_defaults(func=cmd_telemetry)

    p = sub.add_parser("analyze", help="score a TrailSegment (Gate 1)")
    p.add_argument("segment", help="path to a .npz written by TrailSegment.save")
    p.add_argument("--labels", help="JSON: {run_id: group}")
    p.add_argument("--split", nargs=2, type=float, metavar=("S_START", "S_END"))
    p.add_argument("--plot", type=Path, default=None)
    p.set_defaults(func=cmd_analyze)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
