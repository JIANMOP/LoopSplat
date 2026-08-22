from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class IMUPrediction:
    delta_R: torch.Tensor
    delta_v: torch.Tensor
    delta_p: torch.Tensor
    total_dt: float
    sample_count: int
    valid: bool
    translation_valid: bool
    reason: str


@dataclass(frozen=True)
class GravityEstimate:
    gravity_cam: torch.Tensor
    valid: bool
    reason: str


def validate_rigid_transform(transform, dtype, device):
    transform = torch.as_tensor(transform, dtype=dtype, device=device)
    if transform.shape != (4, 4) or not torch.isfinite(transform).all():
        raise ValueError("T_cam_imu must be a finite rigid 4x4 transform")
    rotation = transform[:3, :3]
    identity = torch.eye(3, dtype=dtype, device=device)
    expected_bottom = torch.tensor(
        [0.0, 0.0, 0.0, 1.0], dtype=dtype, device=device)
    if (not torch.allclose(rotation.transpose(0, 1) @ rotation,
                           identity, atol=1e-5, rtol=1e-5)
            or not torch.isclose(torch.linalg.det(rotation),
                                 rotation.new_tensor(1.0),
                                 atol=1e-5, rtol=1e-5)
            or not torch.allclose(transform[3], expected_bottom,
                                  atol=1e-7, rtol=0.0)):
        raise ValueError("T_cam_imu must be a finite rigid 4x4 transform")
    return transform


def so3_exp(rotation_vector):
    if rotation_vector.shape != (3,):
        raise ValueError("rotation_vector must have shape (3,)")
    x, y, z = rotation_vector.unbind()
    zero = torch.zeros((), dtype=rotation_vector.dtype,
                       device=rotation_vector.device)
    skew = torch.stack((
        torch.stack((zero, -z, y)),
        torch.stack((z, zero, -x)),
        torch.stack((-y, x, zero)),
    ))
    theta_squared = torch.dot(rotation_vector, rotation_vector)
    theta = torch.sqrt(theta_squared)
    theta_safe = torch.clamp(theta, min=1e-8)
    theta_squared_safe = torch.clamp(theta_squared, min=1e-16)
    small = theta_squared < 1e-8
    a = torch.where(
        small,
        1.0 - theta_squared / 6.0 + theta_squared * theta_squared / 120.0,
        torch.sin(theta) / theta_safe,
    )
    b = torch.where(
        small,
        0.5 - theta_squared / 24.0
        + theta_squared * theta_squared / 720.0,
        (1.0 - torch.cos(theta)) / theta_squared_safe,
    )
    identity = torch.eye(
        3, dtype=rotation_vector.dtype, device=rotation_vector.device)
    return identity + a * skew + b * (skew @ skew)


def so3_log(rotation_matrix):
    if rotation_matrix.shape != (3, 3):
        raise ValueError("rotation_matrix must have shape (3, 3)")
    vee = torch.stack((
        rotation_matrix[2, 1] - rotation_matrix[1, 2],
        rotation_matrix[0, 2] - rotation_matrix[2, 0],
        rotation_matrix[1, 0] - rotation_matrix[0, 1],
    ))
    cosine = torch.clamp(
        (torch.trace(rotation_matrix) - 1.0) / 2.0,
        min=-1.0 + 1e-7,
        max=1.0 - 1e-7,
    )
    theta = torch.acos(cosine)
    sine = torch.sin(theta)
    factor = torch.where(
        theta < 1e-4,
        0.5 + theta * theta / 12.0,
        theta / (2.0 * torch.clamp(sine, min=1e-8)),
    )
    return factor * vee


def preintegrate_imu(interval, bias_accel, bias_gyro, gravity_cam,
                     t_cam_imu=None, max_interval_s=None,
                     max_accel_norm_mps2=None, max_gyro_norm_rps=None):
    device = bias_accel.device
    dtype = bias_accel.dtype
    sample_count = len(interval.timestamps_s)
    zero_vector = torch.zeros(3, dtype=dtype, device=device)
    identity = torch.eye(3, dtype=dtype, device=device)
    for name, value in (
            ("max_interval_s", max_interval_s),
            ("max_accel_norm_mps2", max_accel_norm_mps2),
            ("max_gyro_norm_rps", max_gyro_norm_rps)):
        if value is not None and (not torch.isfinite(torch.as_tensor(value))
                                  or value <= 0):
            raise ValueError(f"{name} must be finite and positive")
    if not interval.valid:
        return IMUPrediction(
            identity, zero_vector, zero_vector.clone(), 0.0, sample_count,
            False, False, interval.reason)
    if sample_count < 2:
        return IMUPrediction(
            identity, zero_vector, zero_vector.clone(), 0.0, sample_count,
            False, False, "insufficient_samples")

    timestamps = torch.as_tensor(
        interval.timestamps_s, dtype=dtype, device=device)
    accelerations = torch.as_tensor(
        interval.accelerations, dtype=dtype, device=device)
    angular_velocities = torch.as_tensor(
        interval.angular_velocities, dtype=dtype, device=device)
    if (not torch.isfinite(timestamps).all()
            or not torch.isfinite(accelerations).all()
            or not torch.isfinite(angular_velocities).all()):
        return IMUPrediction(
            identity, zero_vector, zero_vector.clone(), 0.0, sample_count,
            False, False, "non_finite_measurement")
    time_steps = timestamps[1:] - timestamps[:-1]
    if torch.any(time_steps <= 0):
        return IMUPrediction(
            identity, zero_vector, zero_vector.clone(), 0.0, sample_count,
            False, False, "non_monotonic_time")
    total_dt = float(time_steps.sum().item())
    if max_interval_s is not None and total_dt > max_interval_s:
        return IMUPrediction(
            identity, zero_vector, zero_vector.clone(), total_dt,
            sample_count, False, False, "interval_too_long")
    if (max_accel_norm_mps2 is not None
            and torch.linalg.vector_norm(
                accelerations, dim=1).max().item() > max_accel_norm_mps2):
        return IMUPrediction(
            identity, zero_vector, zero_vector.clone(), total_dt,
            sample_count, False, False, "acceleration_limit")
    if (max_gyro_norm_rps is not None
            and torch.linalg.vector_norm(
                angular_velocities, dim=1).max().item() > max_gyro_norm_rps):
        return IMUPrediction(
            identity, zero_vector, zero_vector.clone(), total_dt,
            sample_count, False, False, "angular_velocity_limit")

    if t_cam_imu is None:
        rotation_cam_imu = identity
        translation_cam_imu = zero_vector
    else:
        t_cam_imu = validate_rigid_transform(t_cam_imu, dtype, device)
        rotation_cam_imu = t_cam_imu[:3, :3]
        translation_cam_imu = t_cam_imu[:3, 3]

    delta_rotation = identity
    delta_velocity = zero_vector
    delta_position = zero_vector.clone()
    translation_valid = gravity_cam is not None
    if translation_valid:
        gravity_cam = torch.as_tensor(
            gravity_cam, dtype=dtype, device=device)

    for index, time_step in enumerate(time_steps):
        omega_imu = (
            0.5 * (angular_velocities[index]
                   + angular_velocities[index + 1])
            - bias_gyro
        )
        omega_cam = rotation_cam_imu @ omega_imu
        rotation_step = so3_exp(omega_cam * time_step)

        if translation_valid:
            accel_imu = (
                0.5 * (accelerations[index] + accelerations[index + 1])
                - bias_accel
            )
            accel_cam = rotation_cam_imu @ accel_imu
            midpoint_rotation = delta_rotation @ so3_exp(
                omega_cam * time_step * 0.5)
            linear_acceleration = midpoint_rotation @ accel_cam - gravity_cam
            delta_position = (
                delta_position + delta_velocity * time_step
                + 0.5 * linear_acceleration * time_step * time_step)
            delta_velocity = delta_velocity + linear_acceleration * time_step

        delta_rotation = delta_rotation @ rotation_step

    delta_position = (
        delta_position + translation_cam_imu
        - delta_rotation @ translation_cam_imu)

    return IMUPrediction(
        delta_R=delta_rotation,
        delta_v=delta_velocity,
        delta_p=delta_position,
        total_dt=total_dt,
        sample_count=sample_count,
        valid=True,
        translation_valid=translation_valid,
        reason="" if translation_valid else "rotation_only_no_gravity",
    )


def estimate_gravity(interval, device, gravity_magnitude, max_accel_std,
                     max_gyro_norm, t_cam_imu=None,
                     magnitude_tolerance=1.0):
    dtype = torch.float64
    zero = torch.zeros(3, dtype=dtype, device=device)
    for name, value in (
            ("gravity_magnitude", gravity_magnitude),
            ("max_accel_std", max_accel_std),
            ("max_gyro_norm", max_gyro_norm),
            ("magnitude_tolerance", magnitude_tolerance)):
        if not torch.isfinite(torch.as_tensor(value)) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not interval.valid or len(interval.timestamps_s) < 2:
        return GravityEstimate(zero, False, "invalid_interval")
    accelerations = torch.as_tensor(
        interval.accelerations, dtype=dtype, device=device)
    angular_velocities = torch.as_tensor(
        interval.angular_velocities, dtype=dtype, device=device)
    if (not torch.isfinite(accelerations).all()
            or not torch.isfinite(angular_velocities).all()):
        return GravityEstimate(zero, False, "non_finite_measurement")

    identity = torch.eye(3, dtype=dtype, device=device)
    if t_cam_imu is None:
        rotation_cam_imu = identity
    else:
        transform = validate_rigid_transform(t_cam_imu, dtype, device)
        rotation_cam_imu = transform[:3, :3]
    accelerations_cam = (rotation_cam_imu @ accelerations.T).T
    angular_velocities_cam = (rotation_cam_imu @ angular_velocities.T).T

    accel_std = torch.linalg.vector_norm(
        accelerations_cam.std(dim=0, unbiased=False))
    if accel_std.item() > max_accel_std:
        return GravityEstimate(zero, False, "nonstationary_acceleration")
    gyro_peak = torch.linalg.vector_norm(
        angular_velocities_cam, dim=1).max()
    if gyro_peak.item() > max_gyro_norm:
        return GravityEstimate(zero, False, "nonstationary_rotation")

    mean_acceleration = accelerations_cam.mean(dim=0)
    magnitude = torch.linalg.vector_norm(mean_acceleration)
    if magnitude.item() < 1e-8:
        return GravityEstimate(zero, False, "zero_gravity_direction")
    if abs(magnitude.item() - gravity_magnitude) > magnitude_tolerance:
        return GravityEstimate(zero, False, "gravity_magnitude_mismatch")
    gravity_cam = mean_acceleration / magnitude * gravity_magnitude
    return GravityEstimate(gravity_cam, True, "")
