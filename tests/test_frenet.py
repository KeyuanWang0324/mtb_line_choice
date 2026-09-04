import numpy as np
import pytest

from mtb_line.centerline import Centerline
from mtb_line.frenet import project_to_frenet


def straight_line(length=50.0):
    xyz = np.stack([np.linspace(0, length, 500), np.zeros(500), np.zeros(500)], axis=1)
    return Centerline.from_polyline(xyz, spacing=0.05, smooth_window_m=1.0)


def test_arc_length_matches_distance_travelled():
    cl = straight_line()
    pts = np.stack([np.linspace(5, 45, 20), np.zeros(20), np.zeros(20)], axis=1)
    p = project_to_frenet(cl, pts)
    assert np.allclose(p.s, np.linspace(5, 45, 20), atol=0.05)
    assert np.allclose(p.d, 0.0, atol=0.02)


def test_sign_convention_positive_is_riders_right():
    # Travelling +x with up=+z, the rider's right is -y.
    cl = straight_line()
    p = project_to_frenet(cl, np.array([[25.0, -2.0, 0.0]]))
    assert p.d[0] > 0, "a point at -y must read as positive d (rider's right)"
    p = project_to_frenet(cl, np.array([[25.0, +2.0, 0.0]]))
    assert p.d[0] < 0


def test_lateral_and_vertical_offsets_are_separated():
    cl = straight_line()
    p = project_to_frenet(cl, np.array([[25.0, -1.5, 0.8]]))
    assert p.d[0] == pytest.approx(1.5, abs=0.02)
    assert p.h[0] == pytest.approx(0.8, abs=0.02)
    assert p.residual[0] == pytest.approx(np.hypot(1.5, 0.8), abs=0.02)


def test_offset_recovered_around_a_curve():
    t = np.linspace(0, np.pi, 800)
    radius = 30.0
    cl = Centerline.from_polyline(
        np.stack([radius * np.cos(t), radius * np.sin(t), np.zeros_like(t)], axis=1),
        spacing=0.05, smooth_window_m=1.0,
    )
    # A concentric arc 2m inside the centerline holds a constant offset.
    t2 = np.linspace(0.3, np.pi - 0.3, 60)
    inner = np.stack([28.0 * np.cos(t2), 28.0 * np.sin(t2), np.zeros_like(t2)], axis=1)
    p = project_to_frenet(cl, inner)
    assert np.allclose(np.abs(p.d), 2.0, atol=0.05)
    assert len(set(np.sign(p.d))) == 1, "sign must not flip along a smooth curve"


def test_resample_does_not_extrapolate_beyond_coverage():
    cl = straight_line()
    pts = np.stack([np.linspace(10, 30, 50), np.zeros(50), np.zeros(50)], axis=1)
    p = project_to_frenet(cl, pts)
    grid = np.arange(0.0, 50.0, 1.0)
    r = p.resample(grid)
    assert np.all(np.isnan(r.d[grid < 9.5]))
    assert np.all(np.isnan(r.d[grid > 30.5]))
    assert np.all(np.isfinite(r.d[(grid > 11) & (grid < 29)]))


def test_monotonic_fraction_flags_a_jittery_run():
    cl = straight_line()
    s = np.linspace(5, 45, 200)
    s[::5] -= 3.0  # inject backwards jumps
    pts = np.stack([s, np.zeros(200), np.zeros(200)], axis=1)
    assert project_to_frenet(cl, pts).monotonic_fraction < 0.9
