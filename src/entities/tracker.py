""" This module includes the Mapper class, which is responsible scene mapping: Paper Section 3.4  """
from argparse import ArgumentParser
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from scipy.spatial.transform import Rotation as R

from src.entities.arguments import OptimizationParams
from src.entities.losses import l1_loss
from src.entities.gaussian_model import GaussianModel
from src.entities.logger import Logger
from src.entities.datasets import BaseDataset
from src.entities.imu_preintegration import (
    IMUPrediction,
    estimate_gravity,
    preintegrate_imu,
    so3_log,
    validate_rigid_transform,
)
from src.entities.visual_odometer import VisualOdometer
from src.utils.gaussian_model_utils import build_rotation
from src.utils.tracker_utils import (compute_camera_opt_params,
                                     extrapolate_poses, multiply_quaternions,
                                     transformation_to_quaternion)
from src.utils.utils import (get_render_settings, np2torch,
                             render_gaussian_model, torch2np)


def relative_camera_motion_from_tracking(previous_w2c, tracking_transform):
    previous_w2c = torch.as_tensor(
        previous_w2c,
        dtype=tracking_transform.dtype,
        device=tracking_transform.device,
    )
    current_w2c = previous_w2c @ tracking_transform
    current_c2w = torch.linalg.inv(current_w2c)
    return previous_w2c @ current_c2w


def mean_over_valid_tracking_pixels(loss_map, valid_mask):
    expanded_mask = torch.broadcast_to(valid_mask, loss_map.shape)
    valid_count = expanded_mask.sum().clamp_min(1)
    return torch.where(
        expanded_mask, loss_map, torch.zeros_like(loss_map)).sum() / valid_count


@dataclass
class IMUTrackingState:
    velocity: torch.Tensor
    gravity_cam: torch.Tensor | None
    last_committed_frame_id: int | None
    last_c2w: torch.Tensor | None

    @classmethod
    def create(cls, device):
        return cls(
            velocity=torch.zeros(3, dtype=torch.float64, device=device),
            gravity_cam=None,
            last_committed_frame_id=None,
            last_c2w=None,
        )


class Tracker(object):
    def __init__(self, config: dict, dataset: BaseDataset, logger: Logger) -> None:
        """ Initializes the Tracker with a given configuration, dataset, and logger.
        Args:
            config: Configuration dictionary specifying hyperparameters and operational settings.
            dataset: The dataset object providing access to the sequence of frames.
            logger: Logger object for logging the tracking process.
        """
        self.dataset = dataset
        self.logger = logger
        self.config = config
        self.filter_alpha = self.config["filter_alpha"]
        self.filter_outlier_depth = self.config["filter_outlier_depth"]
        self.alpha_thre = self.config["alpha_thre"]
        self.soft_alpha = self.config["soft_alpha"]
        self.mask_invalid_depth_in_color_loss = self.config["mask_invalid_depth"]
        self.w_color_loss = self.config["w_color_loss"]
        self.transform = torchvision.transforms.ToTensor()
        self.opt = OptimizationParams(ArgumentParser(description="Training script parameters"))
        self.frame_depth_loss = []
        self.frame_color_loss = []
        self.odometry_type = self.config["odometry_type"]
        self.help_camera_initialization = self.config["help_camera_initialization"]
        self.init_err_ratio = self.config["init_err_ratio"]
        self.enable_exposure = self.config["enable_exposure"]
        self.odometer = VisualOdometer(
            self.dataset.intrinsics,
            self.config["odometer_method"],
            device=self.config.get("odometer_device", "cuda"),
            cpu_fallback=self.config.get("odometer_cpu_fallback", True),
            max_translation_m=self.config.get(
                "odometer_max_translation_m"),
            max_rotation_deg=self.config.get("odometer_max_rotation_deg"),
        )
        self.odometry_fallback_records = []

        # IMU loss function configuration (following GI-SLAM paper)
        self.use_imu = self.config.get("use_imu", False)
        self.lambda_imu_trans = self.config.get("lambda_imu_trans", 1.0)
        self.lambda_imu_rot = self.config.get("lambda_imu_rot", 1.0)
        self.imu_config = self.config.get("imu", {})
        if (not np.isfinite(self.lambda_imu_trans)
                or not np.isfinite(self.lambda_imu_rot)
                or self.lambda_imu_trans < 0 or self.lambda_imu_rot < 0):
            raise ValueError("IMU loss weights must be non-negative")
        if self.use_imu and self.lambda_imu_trans == self.lambda_imu_rot == 0:
            raise ValueError("enabled IMU tracking requires a positive loss weight")
        if self.use_imu:
            required = {"T_cam_imu", "accel_bias", "gyro_bias"}
            missing = sorted(required.difference(self.imu_config))
            if missing:
                raise ValueError(
                    "IMU tracking requires calibrated values: "
                    + ", ".join(missing))
            validate_rigid_transform(
                self.imu_config["T_cam_imu"], torch.float64, "cuda")
        self.imu_translation_huber_m = self.imu_config.get(
            "translation_huber_m", 0.1)
        self.imu_rotation_huber_rad = self.imu_config.get(
            "rotation_huber_rad", 0.1)
        self.imu_translation_residual_scale_m = self.imu_config.get(
            "translation_residual_scale_m", 0.05)
        self.imu_rotation_residual_scale_rad = self.imu_config.get(
            "rotation_residual_scale_rad", 0.01)
        if (not np.isfinite(self.imu_translation_huber_m)
                or self.imu_translation_huber_m <= 0):
            raise ValueError("tracking.imu.translation_huber_m must be positive")
        if (not np.isfinite(self.imu_rotation_huber_rad)
                or self.imu_rotation_huber_rad <= 0):
            raise ValueError("tracking.imu.rotation_huber_rad must be positive")
        if (not np.isfinite(self.imu_translation_residual_scale_m)
                or self.imu_translation_residual_scale_m <= 0):
            raise ValueError(
                "tracking.imu.translation_residual_scale_m must be positive")
        if (not np.isfinite(self.imu_rotation_residual_scale_rad)
                or self.imu_rotation_residual_scale_rad <= 0):
            raise ValueError(
                "tracking.imu.rotation_residual_scale_rad must be positive")
        positive_imu_limits = (
            "max_interval_s",
            "max_accel_norm_mps2",
            "max_gyro_norm_rps",
            "gravity_mps2",
            "gravity_max_accel_std",
            "gravity_max_gyro_norm",
            "gravity_magnitude_tolerance_mps2",
        )
        for name in positive_imu_limits:
            value = self.imu_config.get(name, {
                "max_interval_s": 0.2,
                "max_accel_norm_mps2": 50.0,
                "max_gyro_norm_rps": 20.0,
                "gravity_mps2": 9.81,
                "gravity_max_accel_std": 0.2,
                "gravity_max_gyro_norm": 0.1,
                "gravity_magnitude_tolerance_mps2": 1.0,
            }[name])
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"tracking.imu.{name} must be positive")
        if not np.isfinite(self.imu_config.get("time_offset_s", 0.0)):
            raise ValueError("tracking.imu.time_offset_s must be finite")
        self.imu_state = IMUTrackingState.create("cuda")
        self.imu_committed_frame_ids = []
        self.imu_constraint_frame_ids = []
        self.imu_prediction_records = []
        self.tracking_loss_records = []

    def estimate_odometry(self, frame_id, image, depth):
        transform = self.odometer.estimate_rel_pose(image, depth)
        diagnostic = self.odometer.last_diagnostic
        if diagnostic["source"] != "primary":
            self.odometry_fallback_records.append({
                "frame_id": int(frame_id),
                **diagnostic,
            })
        return transform

    def odometry_diagnostics(self):
        return {
            **self.odometer.diagnostics(),
            "fallback_records": list(self.odometry_fallback_records),
        }

    def compute_losses(self, gaussian_model: GaussianModel, render_settings: dict,
                       opt_cam_rot: torch.Tensor, opt_cam_trans: torch.Tensor,
                       gt_color: torch.Tensor, gt_depth: torch.Tensor, depth_mask: torch.Tensor,
                       exposure_ab=None, imu_prediction=None,
                       reference_w2c=None) -> tuple:
        """ Computes the tracking losses with respect to ground truth color and depth.
        Args:
            gaussian_model: The current state of the Gaussian model of the scene.
            render_settings: Dictionary containing rendering settings such as image dimensions and camera intrinsics.
            opt_cam_rot: Optimizable tensor representing the camera's rotation.
            opt_cam_trans: Optimizable tensor representing the camera's translation.
            gt_color: Ground truth color image tensor.
            gt_depth: Ground truth depth image tensor.
            depth_mask: Binary mask indicating valid depth values in the ground truth depth image.
        Returns:
            A tuple containing losses and renders
        """
        rel_transform = torch.eye(4).cuda().float()
        rel_transform[:3, :3] = build_rotation(F.normalize(opt_cam_rot[None]))[0]
        rel_transform[:3, 3] = opt_cam_trans

        pts = gaussian_model.get_xyz()
        pts_ones = torch.ones(pts.shape[0], 1).cuda().float()
        pts4 = torch.cat((pts, pts_ones), dim=1)
        transformed_pts = (rel_transform @ pts4.T).T[:, :3]

        quat = F.normalize(opt_cam_rot[None])
        _rotations = multiply_quaternions(gaussian_model.get_rotation(), quat.unsqueeze(0)).squeeze(0)

        render_dict = render_gaussian_model(gaussian_model, render_settings,
                                            override_means_3d=transformed_pts, override_rotations=_rotations)
        rendered_color, rendered_depth = render_dict["color"], render_dict["depth"]
        if self.enable_exposure:
            rendered_color = torch.clamp(torch.exp(exposure_ab[0]) * rendered_color + exposure_ab[1], 0, 1.)
        alpha_mask = render_dict["alpha"] > self.alpha_thre

        tracking_mask = torch.ones_like(alpha_mask).bool()
        tracking_mask &= depth_mask
        depth_err = torch.abs(rendered_depth - gt_depth) * depth_mask

        if self.filter_alpha:
            tracking_mask &= alpha_mask
        if self.filter_outlier_depth and torch.median(depth_err) > 0:
            tracking_mask &= depth_err < 50 * torch.median(depth_err)

        color_loss = l1_loss(rendered_color, gt_color, agg="none")
        depth_loss = l1_loss(rendered_depth, gt_depth, agg="none")

        if self.soft_alpha:
            alpha = render_dict["alpha"] ** 3
            color_loss *= alpha
            depth_loss *= alpha
            if self.mask_invalid_depth_in_color_loss:
                color_loss *= tracking_mask
        else:
            color_loss *= tracking_mask

        color_mask = (
            tracking_mask
            if not self.soft_alpha or self.mask_invalid_depth_in_color_loss
            else torch.ones_like(tracking_mask, dtype=torch.bool))
        color_loss = mean_over_valid_tracking_pixels(color_loss, color_mask)
        depth_loss = mean_over_valid_tracking_pixels(
            depth_loss, tracking_mask)

        # Compute IMU loss if enabled
        if self.use_imu and imu_prediction is not None:
            candidate_motion = relative_camera_motion_from_tracking(
                reference_w2c, rel_transform)
            imu_loss = self.compute_imu_loss(
                candidate_motion, imu_prediction)
        else:
            imu_loss = torch.tensor(0.0, device="cuda")

        return color_loss, depth_loss, imu_loss, rendered_color, rendered_depth, alpha_mask

    def _invalid_imu_prediction(self, reason):
        device = self.imu_state.velocity.device
        dtype = self.imu_state.velocity.dtype
        return IMUPrediction(
            delta_R=torch.eye(3, dtype=dtype, device=device),
            delta_v=torch.zeros(3, dtype=dtype, device=device),
            delta_p=torch.zeros(3, dtype=dtype, device=device),
            total_dt=0.0,
            sample_count=0,
            valid=False,
            translation_valid=False,
            reason=reason,
        )

    def prepare_imu_prediction(self, frame_id):
        if not self.use_imu:
            return self._invalid_imu_prediction("imu_disabled")
        if frame_id <= 0:
            return self._invalid_imu_prediction("no_previous_frame")
        if not hasattr(self.dataset, "get_imu_measurements"):
            return self._invalid_imu_prediction("dataset_has_no_imu_interval")

        interval = self.dataset.get_imu_measurements(
            frame_id - 1, frame_id,
            time_offset_s=self.imu_config.get("time_offset_s", 0.0))
        device = self.imu_state.velocity.device
        dtype = self.imu_state.velocity.dtype
        transform = self.imu_config.get("T_cam_imu")
        if self.imu_state.gravity_cam is None:
            gravity_estimate = estimate_gravity(
                interval,
                device=device,
                gravity_magnitude=self.imu_config.get("gravity_mps2", 9.81),
                max_accel_std=self.imu_config.get(
                    "gravity_max_accel_std", 0.2),
                max_gyro_norm=self.imu_config.get(
                    "gravity_max_gyro_norm", 0.1),
                t_cam_imu=transform,
                magnitude_tolerance=self.imu_config.get(
                    "gravity_magnitude_tolerance_mps2", 1.0),
            )
            if gravity_estimate.valid:
                self.imu_state.gravity_cam = gravity_estimate.gravity_cam

        prediction = preintegrate_imu(
            interval,
            bias_accel=torch.as_tensor(
                self.imu_config.get("accel_bias", [0.0, 0.0, 0.0]),
                dtype=dtype, device=device),
            bias_gyro=torch.as_tensor(
                self.imu_config.get("gyro_bias", [0.0, 0.0, 0.0]),
                dtype=dtype, device=device),
            gravity_cam=self.imu_state.gravity_cam,
            t_cam_imu=transform,
            max_interval_s=self.imu_config.get("max_interval_s", 0.2),
            max_accel_norm_mps2=self.imu_config.get(
                "max_accel_norm_mps2", 50.0),
            max_gyro_norm_rps=self.imu_config.get(
                "max_gyro_norm_rps", 20.0),
        )
        if not prediction.valid or not prediction.translation_valid:
            return prediction
        return IMUPrediction(
            delta_R=prediction.delta_R,
            delta_v=prediction.delta_v,
            delta_p=(
                self.imu_state.velocity * prediction.total_dt
                + prediction.delta_p),
            total_dt=prediction.total_dt,
            sample_count=prediction.sample_count,
            valid=True,
            translation_valid=True,
            reason=prediction.reason,
        )

    @staticmethod
    def _huber_loss(residual, delta):
        absolute = residual.abs()
        delta_tensor = torch.as_tensor(
            delta, dtype=residual.dtype, device=residual.device)
        return torch.where(
            absolute <= delta_tensor,
            0.5 * residual.square(),
            delta_tensor * (absolute - 0.5 * delta_tensor),
        ).sum()

    def compute_imu_loss_terms(self, relative_pose, prediction):
        zero = relative_pose.sum() * 0.0
        if not self.use_imu or prediction is None or not prediction.valid:
            return {
                "rotation_residual_rad": zero,
                "translation_residual_m": None,
                "weighted_rotation_loss": zero,
                "weighted_translation_loss": zero,
                "total_loss": zero,
            }

        dtype = relative_pose.dtype
        device = relative_pose.device
        predicted_rotation = prediction.delta_R.to(
            dtype=dtype, device=device)
        rotation_residual = so3_log(
            predicted_rotation.transpose(0, 1) @ relative_pose[:3, :3])
        normalized_rotation_residual = (
            rotation_residual / self.imu_rotation_residual_scale_rad)
        rotation_loss = self._huber_loss(
            normalized_rotation_residual,
            self.imu_rotation_huber_rad
            / self.imu_rotation_residual_scale_rad)

        translation_loss = zero
        translation_residual_m = None
        if prediction.translation_valid:
            translation_residual = (
                relative_pose[:3, 3]
                - prediction.delta_p.to(dtype=dtype, device=device))
            translation_residual_m = torch.linalg.vector_norm(
                translation_residual)
            normalized_translation_residual = (
                translation_residual
                / self.imu_translation_residual_scale_m)
            translation_loss = self._huber_loss(
                normalized_translation_residual,
                self.imu_translation_huber_m
                / self.imu_translation_residual_scale_m)

        weighted_translation_loss = self.lambda_imu_trans * translation_loss
        weighted_rotation_loss = self.lambda_imu_rot * rotation_loss
        return {
            "rotation_residual_rad": torch.linalg.vector_norm(
                rotation_residual),
            "translation_residual_m": translation_residual_m,
            "weighted_rotation_loss": weighted_rotation_loss,
            "weighted_translation_loss": weighted_translation_loss,
            "total_loss": weighted_translation_loss + weighted_rotation_loss,
        }

    def compute_imu_loss(self, relative_pose, prediction):
        return self.compute_imu_loss_terms(
            relative_pose, prediction)["total_loss"]

    def commit_imu_state(self, frame_id, final_c2w, prediction,
                         optimized_relative_pose):
        if not self.use_imu:
            return
        previous_id = self.imu_state.last_committed_frame_id
        if previous_id is not None and frame_id <= previous_id:
            raise RuntimeError(
                f"IMU state for frame {frame_id} already committed")

        rotation = optimized_relative_pose[:3, :3].to(
            self.imu_state.velocity)
        if (prediction is not None and prediction.valid
                and prediction.total_dt > 0.0):
            visual_velocity = (
                optimized_relative_pose[:3, 3].to(self.imu_state.velocity)
                / prediction.total_dt)
            self.imu_state.velocity = (
                rotation.transpose(0, 1) @ visual_velocity)
        else:
            self.imu_state.velocity = (
                rotation.transpose(0, 1) @ self.imu_state.velocity)
        if self.imu_state.gravity_cam is not None:
            self.imu_state.gravity_cam = (
                rotation.to(self.imu_state.gravity_cam).transpose(0, 1)
                @ self.imu_state.gravity_cam)

        self.imu_state.last_committed_frame_id = frame_id
        if self.use_imu:
            self.imu_committed_frame_ids.append(frame_id)
            if prediction is not None and prediction.valid:
                self.imu_constraint_frame_ids.append(frame_id)
        self.imu_state.last_c2w = torch.as_tensor(
            final_c2w,
            dtype=self.imu_state.velocity.dtype,
            device=self.imu_state.velocity.device,
        ).detach().clone()

    def track(self, frame_id: int, gaussian_model: GaussianModel, prev_c2ws: np.ndarray) -> np.ndarray:
        """
        Updates the camera pose estimation for the current frame based on the provided image and depth, using either ground truth poses,
        constant speed assumption, or visual odometry.
        Args:
            frame_id: Index of the current frame being processed.
            gaussian_model: The current Gaussian model of the scene.
            prev_c2ws: Array containing the camera-to-world transformation matrices for the frames (0, i - 2, i - 1)
        Returns:
            The updated camera-to-world transformation matrix for the current frame.
        """
        _, image, depth, gt_c2w = self.dataset[frame_id]

        imu_prediction = self.prepare_imu_prediction(frame_id)
        if self.use_imu:
            self.imu_prediction_records.append({
                "frame_id": int(frame_id),
                "valid": bool(imu_prediction.valid),
                "translation_valid": bool(
                    imu_prediction.translation_valid),
                "sample_count": int(imu_prediction.sample_count),
                "total_dt_s": float(imu_prediction.total_dt),
                "reason": imu_prediction.reason,
            })
            self.logger.log_imu_prediction(frame_id, imu_prediction)

        if (self.help_camera_initialization or self.odometry_type == "odometer") and self.odometer.last_rgbd is None:
            _, last_image, last_depth, _ = self.dataset[frame_id - 1]
            self.odometer.update_last_rgbd(last_image, last_depth)

        if self.odometry_type == "gt":
            return gt_c2w
        elif self.odometry_type == "const_speed":
            init_c2w = extrapolate_poses(prev_c2ws[1:])
        elif self.odometry_type == "odometer":
            odometer_rel = self.estimate_odometry(frame_id, image, depth)
            init_c2w = prev_c2ws[-1] @ odometer_rel
        elif self.odometry_type == "previous":
            init_c2w = prev_c2ws[-1]

        last_c2w = prev_c2ws[-1]
        last_w2c = np.linalg.inv(last_c2w)
        init_rel = init_c2w @ np.linalg.inv(last_c2w)
        init_rel_w2c = np.linalg.inv(init_rel)
        reference_w2c = last_w2c
        render_settings = get_render_settings(
            self.dataset.width, self.dataset.height, self.dataset.intrinsics, reference_w2c)
        opt_cam_rot, opt_cam_trans = compute_camera_opt_params(init_rel_w2c)
        if self.enable_exposure:
            exposure_ab = torch.nn.Parameter(torch.tensor(
                0.0, device="cuda")), torch.nn.Parameter(torch.tensor(0.0, device="cuda"))
        else:
            exposure_ab = None
        gaussian_model.training_setup_camera(opt_cam_rot, opt_cam_trans, self.config, exposure_ab)

        gt_color = self.transform(image).cuda()
        gt_depth = np2torch(depth, "cuda")
        depth_mask = gt_depth > 0.0
        gt_trans = np2torch(gt_c2w[:3, 3])
        gt_quat = np2torch(R.from_matrix(gt_c2w[:3, :3]).as_quat(canonical=True)[[3, 0, 1, 2]])
        num_iters = self.config["iterations"]
        current_min_loss = float("inf")

        print(f"\nTracking frame {frame_id}")
        # Initial loss check
        color_loss, depth_loss, imu_loss, _, _, _ = self.compute_losses(gaussian_model, render_settings, opt_cam_rot,
                                                                        opt_cam_trans, gt_color, gt_depth, depth_mask,
                                                                        exposure_ab, imu_prediction, reference_w2c)
        if len(self.frame_color_loss) > 0 and (
            color_loss.item() > self.init_err_ratio * np.median(self.frame_color_loss)
            or depth_loss.item() > self.init_err_ratio * np.median(self.frame_depth_loss)
        ):
            num_iters *= 2
            print(f"Higher initial loss, increasing num_iters to {num_iters}")
            if self.help_camera_initialization and self.odometry_type != "odometer":
                _, last_image, last_depth, _ = self.dataset[frame_id - 1]
                self.odometer.update_last_rgbd(last_image, last_depth)
                odometer_rel = self.estimate_odometry(
                    frame_id, image, depth)
                init_c2w = last_c2w @ odometer_rel
                init_rel = init_c2w @ np.linalg.inv(last_c2w)
                init_rel_w2c = np.linalg.inv(init_rel)
                opt_cam_rot, opt_cam_trans = compute_camera_opt_params(init_rel_w2c)
                gaussian_model.training_setup_camera(opt_cam_rot, opt_cam_trans, self.config, exposure_ab)
                render_settings = get_render_settings(
                    self.dataset.width, self.dataset.height, self.dataset.intrinsics, last_w2c)
                print(f"re-init with odometer for frame {frame_id}")

        for iter in range(num_iters):
            color_loss, depth_loss, imu_loss, _, _, _ = self.compute_losses(
                gaussian_model, render_settings, opt_cam_rot, opt_cam_trans, gt_color, gt_depth, depth_mask,
                exposure_ab, imu_prediction, reference_w2c)

            # Total loss includes IMU term
            total_loss = (self.w_color_loss * color_loss + (1 - self.w_color_loss) * depth_loss + imu_loss)
            with torch.no_grad():
                if total_loss.item() < current_min_loss:
                    current_min_loss = total_loss.item()
                    best_w2c = torch.eye(4)
                    best_w2c[:3, :3] = build_rotation(F.normalize(opt_cam_rot[None].clone().detach().cpu()))[0]
                    best_w2c[:3, 3] = opt_cam_trans.clone().detach().cpu()
            total_loss.backward()
            gaussian_model.optimizer.step()
            # gaussian_model.scheduler.step(total_loss, epoch=iter)
            gaussian_model.optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                cur_quat, cur_trans = F.normalize(opt_cam_rot[None].clone().detach()), opt_cam_trans.clone().detach()
                cur_rel_w2c = torch.eye(4)
                cur_rel_w2c[:3, :3] = build_rotation(cur_quat)[0]
                cur_rel_w2c[:3, 3] = cur_trans
                if iter == num_iters - 1:
                    cur_w2c = torch.from_numpy(reference_w2c) @ best_w2c
                else:
                    cur_w2c = torch.from_numpy(reference_w2c) @ cur_rel_w2c
                cur_c2w = torch.inverse(cur_w2c)
                cur_cam = transformation_to_quaternion(cur_c2w)
                if (gt_quat * cur_cam[:4]).sum() < 0:  # for logging purpose
                    gt_quat *= -1
                if iter == num_iters - 1:
                    self.frame_color_loss.append(color_loss.item())
                    self.frame_depth_loss.append(depth_loss.item())
                    self.tracking_loss_records.append({
                        "frame_id": int(frame_id),
                        "color_loss": float(color_loss.item()),
                        "depth_loss": float(depth_loss.item()),
                        "weighted_color_loss": float(
                            self.w_color_loss * color_loss.item()),
                        "weighted_depth_loss": float(
                            (1 - self.w_color_loss) * depth_loss.item()),
                        "imu_loss": float(imu_loss.item()),
                        "total_loss": float(total_loss.item()),
                    })
                    # Log with IMU loss information
                    if self.use_imu:
                        print(f"  IMU Loss: {imu_loss.item():.6e}")
                    self.logger.log_tracking_iteration(
                        frame_id, cur_cam, gt_quat, gt_trans, total_loss, color_loss, depth_loss, iter, num_iters,
                        wandb_output=True, print_output=True)
                elif iter % 20 == 0:
                    self.logger.log_tracking_iteration(
                        frame_id, cur_cam, gt_quat, gt_trans, total_loss, color_loss, depth_loss, iter, num_iters,
                        wandb_output=False, print_output=True)

        final_c2w = torch.inverse(torch.from_numpy(reference_w2c) @ best_w2c)
        final_c2w[-1, :] = torch.tensor([0., 0., 0., 1.], dtype=final_c2w.dtype, device=final_c2w.device)
        optimized_relative_pose = relative_camera_motion_from_tracking(
            reference_w2c, best_w2c)
        imu_terms = self.compute_imu_loss_terms(
            optimized_relative_pose, imu_prediction)
        tracking_record = self.tracking_loss_records[-1]
        tracking_record.update({
            "imu_rotation_residual_rad": float(
                imu_terms["rotation_residual_rad"].item()),
            "imu_translation_residual_m": (
                None if imu_terms["translation_residual_m"] is None
                else float(imu_terms["translation_residual_m"].item())),
            "imu_rotation_loss": float(
                imu_terms["weighted_rotation_loss"].item()),
            "imu_translation_loss": float(
                imu_terms["weighted_translation_loss"].item()),
            "imu_loss_at_best_pose": float(
                imu_terms["total_loss"].item()),
        })
        self.commit_imu_state(
            frame_id, final_c2w, imu_prediction, optimized_relative_pose)
        return torch2np(final_c2w), exposure_ab
