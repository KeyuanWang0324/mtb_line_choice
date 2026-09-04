import numpy as np

from mtb_line.centerline import Centerline, fit_centerline, resample_by_arclength
from mtb_line.frenet import project_to_frenet
from mtb_line.synth import synthetic_segment, true_centerline


def test_resample_gives_uniform_spacing():
    pts = np.stack([np.linspace(0, 10, 7), np.zeros(7), np.zeros(7)], axis=1)
    out = resample_by_arclength(pts, 0.1)
    steps = np.linalg.norm(np.diff(out, axis=0), axis=1)
    assert np.allclose(steps, 0.1, atol=1e-6)


def test_duplicate_points_do_not_break_resampling():
    pts = np.array([[0.0, 0, 0], [0.0, 0, 0], [1.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])
    assert len(resample_by_arclength(pts, 0.1)) > 10


def test_frame_does_not_flip_through_an_inflection():
    # An S-bend: the Frenet normal flips sign at the inflection, a fixed-up
    # frame must not.
    t = np.linspace(0, 1, 600)
    xyz = np.stack([t * 60, 5 * np.sin(2 * np.pi * t), np.zeros_like(t)], axis=1)
    cl = Centerline.from_polyline(xyz, spacing=0.05, smooth_window_m=1.0)
    dots = np.einsum("ij,ij->i", cl.right[:-1], cl.right[1:])
    assert np.all(dots > 0.9), "right vector reversed along the curve"


def test_fit_recovers_the_true_centerline():
    segment, _, _ = synthetic_segment(seed=3)
    fitted = fit_centerline([t.xyz for t in segment.trajectories])
    truth = true_centerline()
    # Sample the fit and measure its distance to the truth. A balanced 3-vs-3
    # split should average out, leaving only wander and drift.
    probe = fitted.interp_xyz(np.linspace(5, fitted.length - 5, 300))
    residual = project_to_frenet(truth, probe).residual
    assert np.percentile(residual, 95) < 0.5, f"p95 {np.percentile(residual, 95):.2f} m"


def test_single_trajectory_is_its_own_centerline():
    segment, _, _ = synthetic_segment(n_left=1, n_right=0, seed=1)
    cl = fit_centerline([segment.trajectories[0].xyz])
    assert 100.0 < cl.length < 145.0


def test_uneven_sample_density_does_not_bias_the_fit():
    """A slow rider contributes more frames; it must not drag the centerline."""
    segment, labels, _ = synthetic_segment(seed=5)
    trajs = [t.xyz for t in segment.trajectories]
    dense = [t if labels[s.run_id] == "left" else t[::3]
             for t, s in zip(trajs, segment.trajectories)]
    a = fit_centerline(trajs)
    b = fit_centerline(dense)
    probe = np.linspace(10, min(a.length, b.length) - 10, 200)
    shift = np.abs(project_to_frenet(a, b.interp_xyz(probe)).d)
    assert np.percentile(shift, 95) < 0.25, f"centerline moved {np.percentile(shift, 95):.2f} m"
