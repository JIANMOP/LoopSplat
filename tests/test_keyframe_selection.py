import numpy as np
import pytest
import torch

from src.entities.gaussian_slam import evaluate_gi_keyframe
from src.utils.keyframe_selection import (
    KeyframeDecision,
    compute_keyframe_motion,
    forced_keyframe_decision,
    gi_slam_keyframe_decision,
)


class DatasetWhoseImuAccessRaises:
    timestamps = np.array([1.0, 1.033275], dtype=np.float64)

    def get_imu_data_for_frame(self, frame_id):
        raise AssertionError("GI-KF must not read IMU when gyro aid is disabled")

    def __len__(self):
        return 3

    def __getitem__(self, frame_id):
        return frame_id, None, np.ones((2, 2)), np.eye(4)


def translated_pose(x):
    pose = np.eye(4)
    pose[0, 3] = x
    return pose


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
    slam._gi_prev_frame_id = 0
    slam._gi_prev_c2w = np.eye(4)
    slam._gi_fps = 30.0
    slam._gi_use_imu_gyro = False
    slam._gi_kf_frustum_ids = {0: np.array([], dtype=np.int64)}
    slam._gi_kf_c2ws = {0: np.eye(4)}
    slam._gi_score_threshold = 0.5
    slam._gi_w_covis = 1.0
    slam._gi_w_base = 1.0
    slam._gi_w_mot = 2.0
    slam._gi_v_max = 0.8
    slam._gi_omega_max = 50.0
    slam.dataset.intrinsics = np.eye(3)

    decision = evaluate_gi_keyframe(
        slam, 1, EmptyModel(), translated_pose(0.033275))

    assert isinstance(decision, KeyframeDecision)
    assert decision.components["gyro_assistance_used"] is False


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
