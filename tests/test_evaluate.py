import numpy as np
import pytest

from mtb_line.centerline import fit_centerline
from mtb_line.evaluate import evaluate_split, quality_report, separation_profile
from mtb_line.frenet import project_to_frenet
from mtb_line.synth import synthetic_segment


def profiles_for(**kwargs):
    segment, labels, window = synthetic_segment(**kwargs)
    cl = fit_centerline([t.xyz for t in segment.trajectories], up=segment.reconstruction.up)
    profiles = [project_to_frenet(cl, t.xyz, run_id=t.run_id, speed=t.speed)
                for t in segment.trajectories]
    return profiles, labels, window


def test_gate_1a_passes_on_a_clear_split():
    profiles, labels, window = profiles_for(seed=0)
    result = evaluate_split(profiles, labels, window)
    assert result.passed
    assert result.accuracy == 1.0
    assert result.separation_m > 2.0


def test_gate_1a_is_stable_across_seeds():
    for seed in range(6):
        profiles, labels, window = profiles_for(seed=seed)
        assert evaluate_split(profiles, labels, window).passed, f"seed {seed}"


def test_no_decision_points_when_everyone_takes_the_same_line():
    """The most important negative: this must not hallucinate a split."""
    profiles, _, _ = profiles_for(split_offset_m=0.0, seed=2)
    assert separation_profile(profiles).decision_points() == []


def test_decision_point_lands_inside_the_true_split():
    profiles, _, window = profiles_for(seed=1)
    points = separation_profile(profiles).decision_points()
    assert points, "a 1.8m split must be detected"
    assert any(window[0] <= dp["s"] <= window[1] for dp in points)


def test_lopsided_split_still_groups_correctly():
    """5 riders left, 1 right: the centerline sits inside the big group, so the
    absolute sign of d is uninformative and only the grouping can survive."""
    profiles, labels, window = profiles_for(n_left=5, n_right=1, seed=4)
    result = evaluate_split(profiles, labels, window)
    assert result.accuracy == 1.0


def test_rejects_a_single_group():
    profiles, labels, window = profiles_for(seed=0)
    with pytest.raises(ValueError, match="exactly 2 groups"):
        evaluate_split(profiles, {k: "left" for k in labels}, window)


def test_rejects_a_window_with_no_samples():
    profiles, labels, window = profiles_for(seed=0)
    with pytest.raises(ValueError, match="wrong window"):
        evaluate_split(profiles, labels, (9000.0, 9100.0))


def test_rejects_unregistered_runs():
    """Runs in disjoint frames share no arc length -- that must be an error,
    not a silently empty comparison."""
    profiles, _, _ = profiles_for(seed=0)
    profiles[0].s = profiles[0].s + 10_000.0
    with pytest.raises(ValueError, match="not registered into the same frame"):
        separation_profile(profiles)


def test_quality_report_mentions_every_run():
    profiles, _, _ = profiles_for(seed=0)
    report = quality_report(profiles)
    for p in profiles:
        assert p.run_id in report
