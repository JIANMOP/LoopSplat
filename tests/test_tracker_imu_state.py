import numpy as np
import pytest
import torch

from src.entities.gaussian_slam import should_use_dataset_pose
from src.entities.imu_preintegration import IMUPrediction, so3_exp
from src.entities.imu_types import IMUInterval
from src.entities.tracker import (
    IMUTrackingState,
    Tracker,
    mean_over_valid_tracking_pixels,
    relative_camera_motion_from_tracking,
)


class IntervalDataset:
    def __init__(self):
        self.timestamps = np.array([0.0, 0.1], dtype=np.float64)

    def get_imu_measurements(self, start_frame_id, end_frame_id,
                             time_offset_s=0.0):
        assert (start_frame_id, end_frame_id) == (0, 1)
        return IMUInterval(
            timestamps_s=np.array([0.0, 0.05, 0.1]),
            accelerations=np.repeat([[0.0, 0.0, 9.81]], 3, axis=0),
            angular_velocities=np.repeat([[0.0, 0.0, 0.2]], 3, axis=0),
            dt_s=0.1,
            valid=True,
            reason="",
        )


def make_tracker(device):
    tracker = Tracker.__new__(Tracker)
    tracker.dataset = IntervalDataset()
    tracker.use_imu = True
    tracker.lambda_imu_trans = 0.01
    tracker.lambda_imu_rot = 0.01
    tracker.imu_translation_huber_m = 0.1
    tracker.imu_rotation_huber_rad = 0.1
    tracker.imu_translation_residual_scale_m = 0.05
    tracker.imu_rotation_residual_scale_rad = 0.01
    tracker.imu_config = {
        "accel_bias": [0.0, 0.0, 0.0],
        "gyro_bias": [0.0, 0.0, 0.0],
        "gravity_mps2": 9.81,
        "gravity_max_accel_std": 0.2,
        "gravity_max_gyro_norm": 0.5,
        "T_cam_imu": np.eye(4),
    }
    tracker.imu_state = IMUTrackingState.create(device)
    tracker.imu_committed_frame_ids = []
    return tracker


def make_prediction(device):
    dtype = torch.float64
    return IMUPrediction(
        delta_R=so3_exp(torch.tensor(
            [0.0, 0.0, 0.02], dtype=dtype, device=device)),
        delta_v=torch.tensor(
            [0.01, 0.0, 0.0], dtype=dtype, device=device),
        delta_p=torch.tensor(
            [0.001, 0.0, 0.0], dtype=dtype, device=device),
        total_dt=0.1,
        sample_count=3,
        valid=True,
        translation_valid=True,
        reason="",
    )


def snapshot(state):
    return {
        "velocity": state.velocity.clone(),
        "gravity_cam": (
            None if state.gravity_cam is None else state.gravity_cam.clone()),
        "last_committed_frame_id": state.last_committed_frame_id,
        "last_c2w": None if state.last_c2w is None else state.last_c2w.clone(),
    }


def assert_snapshot_equal(actual, expected):
    torch.testing.assert_close(actual.velocity, expected["velocity"])
    assert actual.last_committed_frame_id == expected["last_committed_frame_id"]
    if expected["gravity_cam"] is None:
        assert actual.gravity_cam is None
    else:
        torch.testing.assert_close(actual.gravity_cam, expected["gravity_cam"])
    if expected["last_c2w"] is None:
        assert actual.last_c2w is None
    else:
        torch.testing.assert_close(actual.last_c2w, expected["last_c2w"])


def test_tracking_transform_is_converted_to_previous_camera_motion(
        cuda_device):
    previous_c2w = torch.eye(
        4, dtype=torch.float64, device=cuda_device)
    previous_c2w[:3, :3] = so3_exp(torch.tensor(
        [0.0, 0.0, 0.7], dtype=torch.float64, device=cuda_device))
    previous_c2w[:3, 3] = torch.tensor(
        [1.0, -2.0, 0.5], dtype=torch.float64, device=cuda_device)
    current_c2w = torch.eye(
        4, dtype=torch.float64, device=cuda_device)
    current_c2w[:3, :3] = so3_exp(torch.tensor(
        [0.2, -0.1, 0.3], dtype=torch.float64, device=cuda_device))
    current_c2w[:3, 3] = torch.tensor(
        [-0.4, 0.8, 1.2], dtype=torch.float64, device=cuda_device)
    previous_w2c = torch.linalg.inv(previous_c2w)
    tracking_transform = previous_c2w @ torch.linalg.inv(current_c2w)

    actual = relative_camera_motion_from_tracking(
        previous_w2c, tracking_transform)

    expected = previous_w2c @ current_c2w
    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, torch.linalg.inv(tracking_transform))


def test_tracking_losses_are_means_over_valid_pixels(cuda_device):
    loss_map = torch.tensor(
        [[[1.0, 100.0]], [[3.0, 100.0]], [[5.0, 100.0]]],
        device=cuda_device,
    )
    valid_mask = torch.tensor(
        [[[True, False]]], device=cuda_device)

    actual = mean_over_valid_tracking_pixels(loss_map, valid_mask)

    assert actual.item() == pytest.approx(3.0)


def test_repeated_imu_loss_does_not_mutate_tracker_state(cuda_device):
    tracker = make_tracker(cuda_device)
    prediction = make_prediction(cuda_device)
    rotation_vector = torch.tensor(
        [0.0, 0.0, 0.01], dtype=torch.float64,
        device=cuda_device, requires_grad=True)
    relative_pose = torch.eye(
        4, dtype=torch.float64, device=cuda_device)
    relative_pose = relative_pose.clone()
    relative_pose[:3, :3] = so3_exp(rotation_vector)
    before = snapshot(tracker.imu_state)

    first = tracker.compute_imu_loss(relative_pose, prediction)
    second = tracker.compute_imu_loss(relative_pose, prediction)
    (first + second).backward()

    assert_snapshot_equal(tracker.imu_state, before)
    assert torch.isfinite(rotation_vector.grad).all()
    assert torch.linalg.vector_norm(rotation_vector.grad).item() > 0.0


def test_imu_rotation_residual_uses_configured_physical_scale(cuda_device):
    tracker = make_tracker(cuda_device)
    prediction = make_prediction(cuda_device)
    prediction = IMUPrediction(
        delta_R=torch.eye(3, dtype=torch.float64, device=cuda_device),
        delta_v=prediction.delta_v,
        delta_p=prediction.delta_p,
        total_dt=prediction.total_dt,
        sample_count=prediction.sample_count,
        valid=True,
        translation_valid=False,
        reason="rotation_only",
    )
    relative_pose = torch.eye(
        4, dtype=torch.float64, device=cuda_device)
    relative_pose[:3, :3] = so3_exp(torch.tensor(
        [0.02, 0.0, 0.0], dtype=torch.float64, device=cuda_device))

    fine_scale_loss = tracker.compute_imu_loss(relative_pose, prediction)
    tracker.imu_rotation_residual_scale_rad = 0.02
    coarse_scale_loss = tracker.compute_imu_loss(relative_pose, prediction)

    assert fine_scale_loss.item() == pytest.approx(
        4.0 * coarse_scale_loss.item(), rel=1e-5)


def test_commit_advances_state_once(cuda_device):
    tracker = make_tracker(cuda_device)
    prediction = make_prediction(cuda_device)
    final_c2w = torch.eye(4, dtype=torch.float64, device=cuda_device)

    optimized_relative_pose = torch.eye(
        4, dtype=torch.float64, device=cuda_device)
    tracker.commit_imu_state(
        3, final_c2w, prediction, optimized_relative_pose)
    once = snapshot(tracker.imu_state)

    with pytest.raises(RuntimeError, match="already committed"):
        tracker.commit_imu_state(
            3, final_c2w, prediction, optimized_relative_pose)
    assert_snapshot_equal(tracker.imu_state, once)
    assert tracker.imu_committed_frame_ids == [3]


def test_commit_is_noop_when_imu_is_disabled(cuda_device):
    tracker = make_tracker(cuda_device)
    tracker.use_imu = False
    prediction = make_prediction(cuda_device)
    before = snapshot(tracker.imu_state)

    tracker.commit_imu_state(
        3, torch.eye(4, dtype=torch.float64, device=cuda_device), prediction,
        torch.eye(4, dtype=torch.float64, device=cuda_device))

    assert_snapshot_equal(tracker.imu_state, before)
    assert tracker.imu_committed_frame_ids == []


def test_prepare_prediction_uses_full_interval_and_initializes_gravity(
        cuda_device):
    tracker = make_tracker(cuda_device)

    prediction = tracker.prepare_imu_prediction(1)

    assert prediction.valid
    assert prediction.translation_valid
    assert prediction.sample_count == 3
    assert prediction.total_dt == pytest.approx(0.1)
    assert tracker.imu_state.gravity_cam is not None
    assert torch.linalg.vector_norm(prediction.delta_p).item() < 1e-8


@pytest.mark.parametrize(
    ("frame_id", "gt_camera", "has_ground_truth", "expected"),
    [
        (0, False, False, True),
        (1, False, False, False),
        (1, True, False, False),
        (1, True, True, True),
    ],
)
def test_dataset_pose_policy_requires_explicit_available_ground_truth(
        frame_id, gt_camera, has_ground_truth, expected):
    assert should_use_dataset_pose(
        frame_id, gt_camera, has_ground_truth) is expected
