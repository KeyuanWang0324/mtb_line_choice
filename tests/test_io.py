import numpy as np
import pytest

from mtb_line.localize import quat_to_rotation, read_colmap_images, trajectory_from_colmap
from mtb_line.synth import synthetic_segment
from mtb_line.types import TrailSegment

COLMAP_SAMPLE = """# Image list with two lines of data per image:
1 1 0 0 0 0 0 0 1 frame_00001.jpg
100.0 200.0 5 110.0 210.0 7 120.0 220.0 -1
2 0.7071068 0 0.7071068 0 -1 0 0 1 frame_00002.jpg
130.0 230.0 9 140.0 240.0 -1
"""


def test_identity_quaternion():
    assert np.allclose(quat_to_rotation(1, 0, 0, 0), np.eye(3))


def test_quaternion_is_a_rotation():
    R = quat_to_rotation(0.5, 0.5, 0.5, 0.5)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_colmap_centre_is_minus_R_transpose_t(tmp_path):
    path = tmp_path / "images.txt"
    path.write_text(COLMAP_SAMPLE)
    entries = read_colmap_images(path)
    assert set(entries) == {"frame_00001.jpg", "frame_00002.jpg"}
    # Identity rotation, zero translation -> camera sits at the origin.
    assert np.allclose(entries["frame_00001.jpg"]["center"], 0.0)
    e = entries["frame_00002.jpg"]
    assert np.allclose(e["center"], -e["R"].T @ e["t"])


def test_colmap_counts_only_triangulated_observations(tmp_path):
    path = tmp_path / "images.txt"
    path.write_text(COLMAP_SAMPLE)
    entries = read_colmap_images(path)
    assert entries["frame_00001.jpg"]["n_points"] == 2  # third has id -1
    assert entries["frame_00002.jpg"]["n_points"] == 1


def test_poorly_localized_run_is_rejected_not_smoothed(tmp_path):
    path = tmp_path / "images.txt"
    path.write_text(COLMAP_SAMPLE)
    with pytest.raises(ValueError, match="did not register"):
        trajectory_from_colmap(path, run_id="bad", min_points=20)


def test_trajectory_keeps_real_time_gaps(tmp_path):
    path = tmp_path / "images.txt"
    path.write_text(COLMAP_SAMPLE)
    traj = trajectory_from_colmap(path, run_id="ok", min_points=1, fps=10.0)
    assert len(traj) == 2
    assert np.allclose(traj.t, [0.0, 0.1])


def test_segment_round_trips(tmp_path):
    segment, _, _ = synthetic_segment(seed=7)
    segment.outcomes = {"left00": {"time_s": 41.2}}
    segment.semantics = [{"s": 50.0, "label": "rock garden"}]
    loaded = TrailSegment.load(segment.save(tmp_path / "seg"))

    assert loaded.segment_id == segment.segment_id
    assert len(loaded.trajectories) == len(segment.trajectories)
    assert loaded.outcomes == segment.outcomes
    assert loaded.semantics == segment.semantics
    assert np.allclose(loaded.trajectory("left00").xyz, segment.trajectory("left00").xyz)
    assert np.allclose(loaded.reconstruction.points, segment.reconstruction.points)


def test_schema_version_mismatch_is_loud(tmp_path):
    import json
    segment, _, _ = synthetic_segment(seed=7)
    path = segment.save(tmp_path / "seg")
    meta_path = path.with_suffix(".json")
    meta = json.loads(meta_path.read_text())
    meta["schema_version"] = 999
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="schema version"):
        TrailSegment.load(path)
