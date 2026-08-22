from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IMUInterval:
    timestamps_s: np.ndarray
    accelerations: np.ndarray
    angular_velocities: np.ndarray
    dt_s: float
    valid: bool
    reason: str

    @classmethod
    def invalid(cls, reason: str) -> "IMUInterval":
        empty_times = np.empty(0, dtype=np.float64)
        empty_vectors = np.empty((0, 3), dtype=np.float64)
        return cls(
            timestamps_s=empty_times,
            accelerations=empty_vectors,
            angular_velocities=empty_vectors.copy(),
            dt_s=0.0,
            valid=False,
            reason=reason,
        )


def build_imu_interval(frame_timestamps, imu_data, start_frame_id,
                       end_frame_id, time_offset_s=0.0) -> IMUInterval:
    if end_frame_id <= start_frame_id:
        return IMUInterval.invalid("frame_order")
    if start_frame_id < 0 or end_frame_id >= len(frame_timestamps):
        return IMUInterval.invalid("frame_bounds")
    if len(imu_data) < 2:
        return IMUInterval.invalid("insufficient_samples")

    if not np.isfinite(time_offset_s):
        return IMUInterval.invalid("non_finite_time_offset")
    start_s = frame_timestamps[start_frame_id] + time_offset_s
    end_s = frame_timestamps[end_frame_id] + time_offset_s
    imu_times = np.asarray(
        [sample["timestamp"] for sample in imu_data], dtype=np.float64)
    if start_s < imu_times[0] or end_s > imu_times[-1]:
        return IMUInterval.invalid("imu_coverage")

    interior = imu_times[(imu_times > start_s) & (imu_times < end_s)]
    sample_times = np.concatenate(([start_s], interior, [end_s]))
    if len(sample_times) < 2 or np.any(np.diff(sample_times) <= 0):
        return IMUInterval.invalid("non_monotonic_time")

    def interpolate(key):
        values = np.asarray(
            [sample[key] for sample in imu_data], dtype=np.float64)
        return np.column_stack([
            np.interp(sample_times, imu_times, values[:, axis])
            for axis in range(3)
        ])

    return IMUInterval(
        timestamps_s=sample_times,
        accelerations=interpolate("acceleration"),
        angular_velocities=interpolate("angular_velocity"),
        dt_s=end_s - start_s,
        valid=True,
        reason="",
    )
