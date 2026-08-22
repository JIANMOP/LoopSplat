import numpy as np
import pytest
import torch

from src.entities.imu_preintegration import (
    estimate_gravity,
    preintegrate_imu,
    so3_exp,
    so3_log,
)
from src.entities.imu_types import IMUInterval


def make_interval(acceleration, angular_velocity, duration=0.2):
    timestamps = np.array([0.0, duration / 2.0, duration], dtype=np.float64)
    accelerations = np.repeat(
        np.asarray(acceleration, dtype=np.float64)[None], 3, axis=0)
    angular_velocities = np.repeat(
        np.asarray(angular_velocity, dtype=np.float64)[None], 3, axis=0)
    return IMUInterval(
        timestamps_s=timestamps,
        accelerations=accelerations,
        angular_velocities=angular_velocities,
        dt_s=duration,
        valid=True,
        reason="",
    )


def zeros(device):
    return torch.zeros(3, dtype=torch.float64, device=device)


def test_constant_z_angular_velocity_integrates_expected_rotation(cuda_device):
    interval = make_interval((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))

    prediction = preintegrate_imu(
        interval,
        bias_accel=zeros(cuda_device),
        bias_gyro=zeros(cuda_device),
        gravity_cam=None,
    )

    assert prediction.valid
    assert prediction.sample_count == 3
    assert prediction.total_dt == pytest.approx(0.2)
    assert so3_log(prediction.delta_R)[2].item() == pytest.approx(
        0.2, abs=1e-6)


def test_static_acceleration_has_no_translation_after_gravity_compensation(
        cuda_device):
    interval = make_interval((0.0, 0.0, 9.81), (0.0, 0.0, 0.0))
    gravity = torch.tensor(
        [0.0, 0.0, 9.81], dtype=torch.float64, device=cuda_device)

    prediction = preintegrate_imu(
        interval,
        bias_accel=zeros(cuda_device),
        bias_gyro=zeros(cuda_device),
        gravity_cam=gravity,
    )

    assert prediction.translation_valid
    assert torch.linalg.vector_norm(prediction.delta_v).item() < 1e-9
    assert torch.linalg.vector_norm(prediction.delta_p).item() < 1e-9


def test_camera_imu_lever_arm_affects_rotational_translation(cuda_device):
    interval = make_interval(
        (0.0, 0.0, 9.81), (0.0, 0.0, 1.0))
    gravity = torch.tensor(
        [0.0, 0.0, 9.81], dtype=torch.float64, device=cuda_device)
    t_cam_imu = np.eye(4)
    t_cam_imu[0, 3] = 1.0

    prediction = preintegrate_imu(
        interval,
        bias_accel=zeros(cuda_device),
        bias_gyro=zeros(cuda_device),
        gravity_cam=gravity,
        t_cam_imu=t_cam_imu,
    )

    lever_arm = torch.tensor(
        [1.0, 0.0, 0.0], dtype=torch.float64, device=cuda_device)
    expected = lever_arm - prediction.delta_R @ lever_arm
    torch.testing.assert_close(prediction.delta_p, expected)
    assert torch.linalg.vector_norm(prediction.delta_p).item() > 0.0


def test_preintegration_rejects_excessive_interval_and_sensor_magnitude(
        cuda_device):
    interval = make_interval(
        (0.0, 0.0, 9.81), (0.0, 0.0, 0.0))
    too_long = preintegrate_imu(
        interval, zeros(cuda_device), zeros(cuda_device), None,
        max_interval_s=0.005)
    assert too_long.valid is False
    assert too_long.reason == "interval_too_long"

    extreme = make_interval(
        (100.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    too_fast = preintegrate_imu(
        extreme, zeros(cuda_device), zeros(cuda_device), None,
        max_accel_norm_mps2=50.0)
    assert too_fast.valid is False
    assert too_fast.reason == "acceleration_limit"


def test_so3_residual_has_finite_nonzero_rotation_gradient(cuda_device):
    target_vector = torch.tensor(
        [0.02, -0.01, 0.03], dtype=torch.float64, device=cuda_device)
    optimized_vector = torch.tensor(
        [0.04, -0.02, 0.01], dtype=torch.float64,
        device=cuda_device, requires_grad=True)

    residual = so3_log(
        so3_exp(target_vector).transpose(0, 1)
        @ so3_exp(optimized_vector))
    residual.square().sum().backward()

    assert torch.isfinite(optimized_vector.grad).all()
    assert torch.linalg.vector_norm(optimized_vector.grad).item() > 0.0


def test_preintegration_is_repeatable_and_has_no_hidden_state(cuda_device):
    interval = make_interval((0.1, -0.2, 9.81), (0.0, 0.0, 0.2))
    gravity = torch.tensor(
        [0.0, 0.0, 9.81], dtype=torch.float64, device=cuda_device)

    first = preintegrate_imu(
        interval, zeros(cuda_device), zeros(cuda_device), gravity)
    second = preintegrate_imu(
        interval, zeros(cuda_device), zeros(cuda_device), gravity)

    torch.testing.assert_close(first.delta_R, second.delta_R)
    torch.testing.assert_close(first.delta_v, second.delta_v)
    torch.testing.assert_close(first.delta_p, second.delta_p)


def test_gravity_estimation_rejects_nonstationary_window(cuda_device):
    interval = IMUInterval(
        timestamps_s=np.array([0.0, 0.01, 0.02]),
        accelerations=np.array([
            [0.0, 0.0, 9.81],
            [4.0, 0.0, 9.81],
            [-4.0, 0.0, 9.81],
        ]),
        angular_velocities=np.zeros((3, 3)),
        dt_s=0.02,
        valid=True,
        reason="",
    )

    estimate = estimate_gravity(
        interval,
        device=cuda_device,
        gravity_magnitude=9.81,
        max_accel_std=0.2,
        max_gyro_norm=0.1,
    )

    assert estimate.valid is False
    assert estimate.reason == "nonstationary_acceleration"


def test_gravity_estimation_rejects_wrong_stationary_magnitude(cuda_device):
    interval = make_interval((0.0, 0.0, 1.0), (0.0, 0.0, 0.0))

    estimate = estimate_gravity(
        interval, device=cuda_device, gravity_magnitude=9.81,
        max_accel_std=0.2, max_gyro_norm=0.1,
        magnitude_tolerance=1.0)

    assert estimate.valid is False
    assert estimate.reason == "gravity_magnitude_mismatch"


def test_preintegration_rejects_non_rigid_camera_imu_transform(cuda_device):
    transform = np.eye(4)
    transform[0, 0] = 2.0

    with pytest.raises(ValueError, match="rigid"):
        preintegrate_imu(
            make_interval((0.0, 0.0, 9.81), (0.0, 0.0, 0.0)),
            zeros(cuda_device), zeros(cuda_device), None,
            t_cam_imu=transform)
