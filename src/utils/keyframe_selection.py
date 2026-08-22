from dataclasses import dataclass

import numpy as np

from src.utils.mapper_utils import (
    compute_camera_velocity,
    compute_gaussian_iou,
    compute_gaussian_visibility,
)


@dataclass(frozen=True)
class KeyframeDecision:
    selected: bool
    score: float
    reason: str
    components: dict


def compute_gaussian_frustum_ids(*args, **kwargs):
    """Return Gaussian-center IDs inside the frustum-only proxy."""
    return compute_gaussian_visibility(*args, **kwargs)


def compute_keyframe_motion(dataset, frame_id: int, previous_frame_id: int,
                            c2w_current: np.ndarray,
                            c2w_previous: np.ndarray,
                            fallback_fps: float,
                            use_imu_gyro: bool) -> dict:
    if fallback_fps <= 0:
        raise ValueError("keyframing.fps must be positive")
    timestamps = getattr(dataset, "timestamps", None)
    if (timestamps is not None
            and previous_frame_id < len(timestamps)
            and frame_id < len(timestamps)):
        dt_s = float(timestamps[frame_id] - timestamps[previous_frame_id])
    else:
        dt_s = 1.0 / fallback_fps
    if not np.isfinite(dt_s) or dt_s <= 0:
        dt_s = 1.0 / fallback_fps

    linear_velocity, visual_angular_velocity = compute_camera_velocity(
        c2w_current, c2w_previous, dt_s)
    angular_velocity = visual_angular_velocity
    gyro_assistance_used = False
    if use_imu_gyro and hasattr(dataset, "get_imu_data_for_frame"):
        imu_data = dataset.get_imu_data_for_frame(frame_id)
        if imu_data is not None and "angular_velocity" in imu_data:
            gyro_rps = float(np.linalg.norm(imu_data["angular_velocity"]))
            if np.isfinite(gyro_rps):
                angular_velocity = max(
                    angular_velocity, float(np.degrees(gyro_rps)))
                gyro_assistance_used = True

    return {
        "dt_s": dt_s,
        "linear_velocity_mps": linear_velocity,
        "visual_angular_velocity_dps": visual_angular_velocity,
        "angular_velocity_dps": angular_velocity,
        "gyro_assistance_used": gyro_assistance_used,
    }


def forced_keyframe_decision(frame_id: int, num_frames: int,
                             last_keyframe_id: int | None,
                             min_interval: int,
                             max_gap: int) -> KeyframeDecision | None:
    if min_interval < 1 or max_gap < 1:
        raise ValueError("keyframe intervals must be positive")
    if frame_id == 0:
        return KeyframeDecision(True, 0.0, "first_frame", {})
    if frame_id == num_frames - 1:
        return KeyframeDecision(True, 0.0, "last_frame", {})
    if last_keyframe_id is None:
        return KeyframeDecision(True, 0.0, "no_previous_keyframe", {})

    gap = frame_id - last_keyframe_id
    components = {"frame_gap": gap}
    if gap >= max_gap:
        return KeyframeDecision(True, 0.0, "max_gap", components)
    if gap < min_interval:
        return KeyframeDecision(False, 0.0, "min_interval", components)
    return None


def _median_valid_depth(depth_map):
    valid = depth_map[np.isfinite(depth_map) & (depth_map > 0)]
    return float(np.median(valid)) if valid.size else 1.0


def gi_slam_keyframe_decision(frustum_ids_current: np.ndarray,
                              frustum_ids_keyframe: np.ndarray,
                              c2w_current: np.ndarray,
                              c2w_keyframe: np.ndarray,
                              depth_map_current: np.ndarray,
                              linear_velocity_mps: float,
                              angular_velocity_dps: float,
                              score_threshold: float,
                              w_covis: float = 1.0,
                              w_base: float = 1.0,
                              w_mot: float = 2.0,
                              v_max: float = 0.8,
                              omega_max: float = 50.0) -> KeyframeDecision:
    frustum_iou = compute_gaussian_iou(
        frustum_ids_current, frustum_ids_keyframe)
    frustum_term = w_covis * (1.0 - frustum_iou)
    baseline_m = float(np.linalg.norm(
        c2w_current[:3, 3] - c2w_keyframe[:3, 3]))
    median_depth_m = _median_valid_depth(depth_map_current)
    baseline_term = w_base * baseline_m / max(median_depth_m, 1e-6)
    motion_penalty = (
        w_mot
        if (linear_velocity_mps > v_max
            or angular_velocity_dps > omega_max)
        else 0.0)
    score = float(frustum_term + baseline_term - motion_penalty)
    selected = score > score_threshold
    return KeyframeDecision(
        selected=selected,
        score=score,
        reason="score" if selected else "below_threshold",
        components={
            "frustum_center_iou": frustum_iou,
            "frustum_center_novelty_term": frustum_term,
            "baseline_m": baseline_m,
            "median_depth_m": median_depth_m,
            "baseline_term": baseline_term,
            "linear_velocity_mps": linear_velocity_mps,
            "angular_velocity_dps": angular_velocity_dps,
            "motion_penalty": motion_penalty,
        },
    )
