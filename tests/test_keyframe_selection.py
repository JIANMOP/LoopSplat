import numpy as np
import pytest
import torch

import src.entities.gaussian_slam as gaussian_slam_module
from src.entities.gaussian_slam import (
    evaluate_gi_keyframe,
    mapping_keyframe_decision,
)
from src.utils.keyframe_selection import (
    KeyframeDecision,
    compute_keyframe_motion,
    forced_keyframe_decision,
    gi_slam_keyframe_decision,
)
from src.utils.mapper_utils import compute_gaussian_visibility
from src.utils.mapper_utils import compute_gaussian_iou


class DatasetWhoseImuAccessRaises:
    timestamps = np.array([1.0, 1.033275], dtype=np.float64)

    def get_imu_data_for_frame(self, frame_id):
        raise AssertionError("GI-KF must not read IMU when gyro aid is disabled")

    def __len__(self):
        return 3

    def __getitem__(self, frame_id):
        return frame_id, None, np.ones((2, 2)), np.eye(4)


class HighMotionDataset(DatasetWhoseImuAccessRaises):
    timestamps = np.arange(5, dtype=np.float64) / 30.0

    def __len__(self):
        return 5


class InvalidDepthHighMotionDataset(HighMotionDataset):
    def __getitem__(self, frame_id):
        return frame_id, None, np.zeros((2, 2)), np.eye(4)


class LongHighMotionDataset(HighMotionDataset):
    timestamps = np.arange(12, dtype=np.float64) / 30.0

    def __len__(self):
        return 12


class LongInvalidDepthHighMotionDataset(LongHighMotionDataset):
    def __getitem__(self, frame_id):
        return frame_id, None, np.zeros((2, 2)), np.eye(4)


def translated_pose(x):
    pose = np.eye(4)
    pose[0, 3] = x
    return pose


@pytest.mark.parametrize(
    "intervals",
    [(1, 0, 10), (1, "3", 10), (1, 11, 10), (3, 2, 10)],
)
def test_keyframe_interval_validation_rejects_invalid_stable_gap(
        intervals):
    with pytest.raises(ValueError, match="keyframe intervals"):
        gaussian_slam_module.validate_keyframe_intervals(*intervals)


def test_frustum_proxy_uses_c2w_and_keeps_camera_center_ray(cuda_device):
    c2w = translated_pose(10.0)
    intrinsics = np.array([
        [2.0, 0.0, 2.0],
        [0.0, 2.0, 2.0],
        [0.0, 0.0, 1.0],
    ])
    depth = np.array([
        [0.5, 0.5, 0.5, 0.5],
        [0.5, 1.0, 1.0, 0.5],
        [0.5, 1.0, 2.0, 0.5],
        [0.5, 0.5, 0.5, 0.5],
    ])
    gaussian_xyz = torch.tensor(
        [[10.0, 0.0, 1.0], [20.0, 0.0, 1.0]],
        device=cuda_device)

    visible = compute_gaussian_visibility(
        gaussian_xyz, c2w, intrinsics, depth)

    np.testing.assert_array_equal(visible, np.array([0]))


def test_submap_boundary_is_the_single_primary_selection_reason():
    decision = mapping_keyframe_decision(
        None, 5, None, None, submap_boundary=True)

    assert decision == KeyframeDecision(
        True, 0.0, "submap_boundary", {})


def test_gi_kf_does_not_read_gyro_when_keyframe_imu_is_disabled():
    dataset = DatasetWhoseImuAccessRaises()

    components = compute_keyframe_motion(
        dataset=dataset,
        frame_id=1,
        previous_frame_id=0,
        c2w_current=translated_pose(0.033275),
        c2w_previous=np.eye(4),
        fallback_fps=30.0,
        use_imu_gyro=False,
    )

    assert components["linear_velocity_mps"] == pytest.approx(1.0)
    assert components["gyro_assistance_used"] is False


def test_active_slam_selector_keeps_gyro_disabled():
    class EmptyModel:
        def get_xyz(self):
            return torch.empty((0, 3))

    slam = type("SlamState", (), {})()
    slam.dataset = DatasetWhoseImuAccessRaises()
    slam.mapping_frame_ids = [0]
    slam._gi_min_interval = 1
    slam._gi_max_gap = 30
    slam._gi_stable_gap = 3
    slam._gi_prev_frame_id = 0
    slam._gi_prev_c2w = np.eye(4)
    slam._gi_fps = 30.0
    slam._gi_use_imu_gyro = False
    slam._gi_kf_c2ws = {0: np.eye(4)}
    slam._gi_score_threshold = 0.5
    slam._gi_w_covis = 1.0
    slam._gi_w_base = 1.0
    slam._gi_w_mot = 2.0
    slam._gi_v_max = 1.5
    slam._gi_omega_max = 120.0
    slam.dataset.intrinsics = np.eye(3)

    decision = evaluate_gi_keyframe(
        slam, 1, EmptyModel(), translated_pose(0.1))

    assert isinstance(decision, KeyframeDecision)
    assert decision.selected is False
    assert decision.reason == "high_motion_reject"
    assert decision.components["motion_penalty"] == pytest.approx(2.0)
    assert decision.components["gyro_assistance_used"] is False


def test_high_motion_frame_is_rejected_before_emergency_gap():
    class EmptyModel:
        def get_xyz(self):
            return torch.empty((0, 3))

    slam = type("SlamState", (), {})()
    slam.dataset = HighMotionDataset()
    slam.dataset.intrinsics = np.eye(3)
    slam.mapping_frame_ids = [0]
    slam._gi_min_interval = 1
    slam._gi_max_gap = 10
    slam._gi_stable_gap = 3
    slam._gi_prev_frame_id = 0
    slam._gi_prev_c2w = np.eye(4)
    slam._gi_fps = 30.0
    slam._gi_use_imu_gyro = False
    slam._gi_kf_c2ws = {0: np.eye(4)}
    slam._gi_score_threshold = 0.1
    slam._gi_w_covis = 1.0
    slam._gi_w_base = 1.0
    slam._gi_w_mot = 2.0
    slam._gi_v_max = 1.5
    slam._gi_omega_max = 120.0

    decision = evaluate_gi_keyframe(
        slam, 1, EmptyModel(), translated_pose(0.1))

    assert decision.selected is False
    assert decision.reason == "high_motion_reject"
    assert decision.components["frame_gap"] == 1
    assert decision.components["linear_velocity_mps"] == pytest.approx(3.0)
    assert decision.components["motion_penalty"] == pytest.approx(2.0)


def test_high_motion_policy_does_not_select_invalid_depth_frame():
    class EmptyModel:
        def get_xyz(self):
            return torch.empty((0, 3))

    slam = type("SlamState", (), {})()
    slam.dataset = InvalidDepthHighMotionDataset()
    slam.dataset.intrinsics = np.eye(3)
    slam.mapping_frame_ids = [0]
    slam._gi_min_interval = 1
    slam._gi_max_gap = 10
    slam._gi_stable_gap = 3
    slam._gi_prev_frame_id = 2
    slam._gi_prev_c2w = np.eye(4)
    slam._gi_fps = 30.0
    slam._gi_use_imu_gyro = False
    slam._gi_kf_c2ws = {0: np.eye(4)}
    slam._gi_score_threshold = 0.1
    slam._gi_w_covis = 1.0
    slam._gi_w_base = 1.0
    slam._gi_w_mot = 2.0
    slam._gi_v_max = 1.5
    slam._gi_omega_max = 120.0

    decision = evaluate_gi_keyframe(
        slam, 3, EmptyModel(), translated_pose(0.1))

    assert decision.selected is False
    assert decision.reason == "invalid_depth"


@pytest.mark.parametrize("frame_id", [10, 11])
def test_forced_selection_does_not_override_invalid_depth(frame_id):
    class EmptyModel:
        def get_xyz(self):
            return torch.empty((0, 3))

    slam = type("SlamState", (), {})()
    slam.dataset = LongInvalidDepthHighMotionDataset()
    slam.dataset.intrinsics = np.eye(3)
    slam.mapping_frame_ids = [0]
    slam._gi_min_interval = 1
    slam._gi_max_gap = 10
    slam._gi_stable_gap = 3
    slam._gi_prev_frame_id = frame_id - 1
    slam._gi_prev_c2w = np.eye(4)
    slam._gi_fps = 30.0
    slam._gi_use_imu_gyro = False
    slam._gi_kf_c2ws = {0: np.eye(4)}
    slam._gi_score_threshold = 0.1
    slam._gi_w_covis = 1.0
    slam._gi_w_base = 1.0
    slam._gi_w_mot = 2.0
    slam._gi_v_max = 0.8
    slam._gi_omega_max = 50.0

    decision = evaluate_gi_keyframe(
        slam, frame_id, EmptyModel(), translated_pose(0.1))

    assert decision.selected is False
    assert decision.reason == "invalid_depth"


def test_emergency_gap_selects_after_continuous_high_motion():
    class EmptyModel:
        def get_xyz(self):
            return torch.empty((0, 3))

    slam = type("SlamState", (), {})()
    slam.dataset = LongHighMotionDataset()
    slam.dataset.intrinsics = np.eye(3)
    slam.mapping_frame_ids = [0]
    slam._gi_min_interval = 1
    slam._gi_max_gap = 10
    slam._gi_stable_gap = 3
    slam._gi_prev_frame_id = 9
    slam._gi_prev_c2w = np.eye(4)
    slam._gi_fps = 30.0
    slam._gi_use_imu_gyro = False
    slam._gi_kf_c2ws = {0: np.eye(4)}
    slam._gi_score_threshold = 0.1
    slam._gi_w_covis = 1.0
    slam._gi_w_base = 1.0
    slam._gi_w_mot = 2.0
    slam._gi_v_max = 1.5
    slam._gi_omega_max = 120.0

    decision = evaluate_gi_keyframe(
        slam, 10, EmptyModel(), translated_pose(0.1))

    assert decision.selected is True
    assert decision.reason == "emergency_gap"


def test_stable_gap_selects_without_expensive_frustum_scoring():
    class ModelWhoseAccessRaises:
        def get_xyz(self):
            raise AssertionError("stable-gap selection must skip IoU scoring")

    slam = type("SlamState", (), {})()
    slam.dataset = HighMotionDataset()
    slam.dataset.intrinsics = np.eye(3)
    slam.mapping_frame_ids = [0]
    slam._gi_min_interval = 2
    slam._gi_stable_gap = 3
    slam._gi_max_gap = 10
    slam._gi_prev_frame_id = 2
    slam._gi_prev_c2w = np.eye(4)
    slam._gi_fps = 30.0
    slam._gi_use_imu_gyro = False
    slam._gi_kf_c2ws = {0: np.eye(4)}
    slam._gi_score_threshold = 0.1
    slam._gi_w_covis = 1.0
    slam._gi_w_base = 1.0
    slam._gi_w_mot = 2.0
    slam._gi_v_max = 1.5
    slam._gi_omega_max = 120.0

    decision = evaluate_gi_keyframe(
        slam, 3, ModelWhoseAccessRaises(), translated_pose(0.01))

    assert decision.selected is True
    assert decision.reason == "stable_gap"
    assert decision.components["frame_gap"] == 3
    assert decision.components["motion_penalty"] == pytest.approx(0.0)


def test_selector_reprojects_same_current_gaussian_set_for_both_views(
        monkeypatch):
    current_gaussians = torch.tensor([[1.0, 2.0, 3.0]])
    calls = []

    def fake_frustum_ids(gaussians, w2c, intrinsics, depth):
        calls.append((gaussians, w2c.copy()))
        return np.array([0], dtype=np.int64)

    monkeypatch.setattr(
        gaussian_slam_module, "compute_gaussian_frustum_ids",
        fake_frustum_ids)

    class Model:
        def get_xyz(self):
            return current_gaussians

    slam = type("SlamState", (), {})()
    slam.dataset = DatasetWhoseImuAccessRaises()
    slam.dataset.intrinsics = np.eye(3)
    slam.mapping_frame_ids = [0]
    slam._gi_min_interval = 1
    slam._gi_max_gap = 30
    slam._gi_stable_gap = 3
    slam._gi_prev_frame_id = 0
    slam._gi_prev_c2w = np.eye(4)
    slam._gi_fps = 30.0
    slam._gi_use_imu_gyro = False
    slam._gi_kf_c2ws = {0: np.eye(4)}
    slam._gi_score_threshold = 0.5
    slam._gi_w_covis = 1.0
    slam._gi_w_base = 1.0
    slam._gi_w_mot = 2.0
    slam._gi_v_max = 1.5
    slam._gi_omega_max = 120.0

    decision = evaluate_gi_keyframe(
        slam, 1, Model(), translated_pose(0.033275))

    assert len(calls) == 2
    assert calls[0][0] is current_gaussians
    assert calls[1][0] is current_gaussians
    assert decision.components["frustum_center_iou"] == pytest.approx(1.0)


def test_frustum_visibility_is_empty_for_all_invalid_depth():
    visible = compute_gaussian_visibility(
        torch.tensor([[0.0, 0.0, 1.0]], device="cuda"),
        np.eye(4), np.eye(3), np.zeros((4, 4), dtype=np.float32))

    assert visible.size == 0


def test_two_empty_visibility_sets_have_no_novelty():
    assert compute_gaussian_iou(
        np.array([], dtype=np.int64),
        np.array([], dtype=np.int64)) == pytest.approx(1.0)


def test_invalid_depth_candidate_is_rejected_by_gi_score():
    decision = gi_slam_keyframe_decision(
        np.array([], dtype=np.int64), np.array([0], dtype=np.int64),
        np.eye(4), np.eye(4), np.zeros((2, 2)),
        0.0, 0.0, 0.5)

    assert decision.selected is False
    assert decision.reason == "invalid_depth"


def test_gyro_assistance_is_controlled_by_separate_switch():
    class Dataset(DatasetWhoseImuAccessRaises):
        def get_imu_data_for_frame(self, frame_id):
            return {"angular_velocity": np.array([0.0, 0.0, 2.0])}

    components = compute_keyframe_motion(
        dataset=Dataset(),
        frame_id=1,
        previous_frame_id=0,
        c2w_current=np.eye(4),
        c2w_previous=np.eye(4),
        fallback_fps=30.0,
        use_imu_gyro=True,
    )

    assert components["gyro_assistance_used"] is True
    assert components["angular_velocity_dps"] == pytest.approx(
        np.degrees(2.0))


def test_score_decision_names_frustum_center_proxy_accurately():
    decision = gi_slam_keyframe_decision(
        frustum_ids_current=np.array([1, 2, 3]),
        frustum_ids_keyframe=np.array([2, 3, 4]),
        c2w_current=translated_pose(0.2),
        c2w_keyframe=np.eye(4),
        depth_map_current=np.ones((2, 2)),
        linear_velocity_mps=0.1,
        angular_velocity_dps=1.0,
        score_threshold=0.1,
    )

    assert isinstance(decision, KeyframeDecision)
    assert decision.selected is True
    assert decision.reason == "score"
    assert decision.components["frustum_center_iou"] == pytest.approx(0.5)
    assert all("visibility" not in key for key in decision.components)


@pytest.mark.parametrize(
    ("frame_id", "num_frames", "last_keyframe_id", "expected_reason"),
    [
        (0, 20, None, "first_frame"),
        (19, 20, 5, "last_frame"),
        (8, 20, 3, "max_gap"),
        (4, 20, 3, "min_interval"),
    ],
)
def test_forced_rules_return_explicit_reasons(
        frame_id, num_frames, last_keyframe_id, expected_reason):
    decision = forced_keyframe_decision(
        frame_id=frame_id,
        num_frames=num_frames,
        last_keyframe_id=last_keyframe_id,
        min_interval=2,
        max_gap=5,
    )

    assert decision is not None
    assert decision.reason == expected_reason
    assert decision.selected is (expected_reason != "min_interval")
