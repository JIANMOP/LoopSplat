""" This module includes the Gaussian-SLAM class, which is responsible for controlling Mapper and Tracker
    It also decides when to start a new submap and when to update the estimated camera poses.
"""
import os
import pprint
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
import torch
import roma

from src.entities.arguments import OptimizationParams
from src.entities.datasets import get_dataset
from src.entities.gaussian_model import GaussianModel
from src.entities.mapper import Mapper
from src.entities.tracker import Tracker
from src.entities.lc import Loop_closure
from src.entities.logger import Logger
from src.utils.io_utils import save_dict_to_ckpt, save_dict_to_yaml
from src.utils.mapper_utils import exceeds_motion_thresholds
from src.utils.utils import np2torch, setup_seed, torch2np
from src.utils.vis_utils import *  # noqa - needed for debugging
from src.utils.keyframe_selection import (
    KeyframeDecision,
    compute_gaussian_frustum_ids,
    compute_keyframe_motion,
    forced_keyframe_decision,
    gi_slam_keyframe_decision,
)


def should_use_dataset_pose(frame_id, gt_camera, has_ground_truth):
    return frame_id == 0 or (gt_camera and has_ground_truth)


def build_dataset_config(config):
    dataset_config = {**config["data"], **config["cam"]}
    if "frame_limit" in config:
        dataset_config["frame_limit"] = config["frame_limit"]
    return dataset_config


def _record_keyframe_decision(slam, frame_id, decision):
    previous = slam._gi_decisions.get(frame_id)
    if previous is not None:
        slam._gi_decision_counts[previous.reason] -= 1
    slam._gi_decisions[frame_id] = decision
    slam._gi_decision_counts[decision.reason] = (
        slam._gi_decision_counts.get(decision.reason, 0) + 1)
    record = {
        "frame_id": frame_id,
        "selected": decision.selected,
        "score": decision.score,
        "reason": decision.reason,
        "components": decision.components,
        "cumulative_reason_counts": dict(slam._gi_decision_counts),
    }
    with open(slam.output_path / "keyframe_decisions.jsonl", "a") as file:
        file.write(json.dumps(record, sort_keys=True) + "\n")


def evaluate_gi_keyframe(slam, frame_id, gaussian_model, estimated_c2w):
    last_keyframe_id = (
        slam.mapping_frame_ids[-1] if slam.mapping_frame_ids else None)
    forced = forced_keyframe_decision(
        frame_id=frame_id,
        num_frames=len(slam.dataset),
        last_keyframe_id=last_keyframe_id,
        min_interval=slam._gi_min_interval,
        max_gap=slam._gi_max_gap,
    )
    if forced is not None:
        return forced

    motion = compute_keyframe_motion(
        dataset=slam.dataset,
        frame_id=frame_id,
        previous_frame_id=slam._gi_prev_frame_id,
        c2w_current=estimated_c2w,
        c2w_previous=slam._gi_prev_c2w,
        fallback_fps=slam._gi_fps,
        use_imu_gyro=slam._gi_use_imu_gyro,
    )
    _, _, depth, _ = slam.dataset[frame_id]
    frustum_ids_current = compute_gaussian_frustum_ids(
        gaussian_model.get_xyz(), np.linalg.inv(estimated_c2w),
        slam.dataset.intrinsics, depth)
    reference_frustum_ids = slam._gi_kf_frustum_ids.get(
        last_keyframe_id, np.array([], dtype=np.int64))
    reference_c2w = slam._gi_kf_c2ws.get(
        last_keyframe_id, estimated_c2w)
    decision = gi_slam_keyframe_decision(
        frustum_ids_current=frustum_ids_current,
        frustum_ids_keyframe=reference_frustum_ids,
        c2w_current=estimated_c2w,
        c2w_keyframe=reference_c2w,
        depth_map_current=depth,
        linear_velocity_mps=motion["linear_velocity_mps"],
        angular_velocity_dps=motion["angular_velocity_dps"],
        score_threshold=slam._gi_score_threshold,
        w_covis=slam._gi_w_covis,
        w_base=slam._gi_w_base,
        w_mot=slam._gi_w_mot,
        v_max=slam._gi_v_max,
        omega_max=slam._gi_omega_max,
    )
    components = {
        **decision.components,
        **motion,
        "reference_keyframe_id": last_keyframe_id,
        "reference_policy": "temporal_last",
    }
    return KeyframeDecision(
        decision.selected, decision.score, decision.reason, components)


def register_gi_keyframe(slam, frame_id, gaussian_model, estimated_c2w):
    _, _, depth, _ = slam.dataset[frame_id]
    frustum_ids = compute_gaussian_frustum_ids(
        gaussian_model.get_xyz(), np.linalg.inv(estimated_c2w),
        slam.dataset.intrinsics, depth)
    slam._gi_kf_frustum_ids[frame_id] = frustum_ids
    slam._gi_kf_c2ws[frame_id] = estimated_c2w.copy()


class GaussianSLAM(object):

    def __init__(self, config: dict) -> None:

        self._setup_output_path(config)
        self.device = "cuda"
        self.config = config

        self.scene_name = config["data"]["scene_name"]
        self.dataset_name = config["dataset_name"]
        self.dataset = get_dataset(config["dataset_name"])(
            build_dataset_config(config))

        n_frames = len(self.dataset)
        frame_ids = list(range(n_frames))
        self.mapping_frame_ids = frame_ids[::config["mapping"]["map_every"]] + [n_frames - 1]

        self.estimated_c2ws = torch.empty(len(self.dataset), 4, 4)
        self.estimated_c2ws[0] = torch.from_numpy(self.dataset[0][3])
        self.exposures_ab = torch.zeros(len(self.dataset), 2)

        save_dict_to_yaml(config, "config.yaml", directory=self.output_path)

        self.submap_using_motion_heuristic = config["mapping"]["submap_using_motion_heuristic"]

        # ── GI-SLAM keyframing config ─────────────────────────────────
        kf_cfg = config.get("keyframing", {})
        self._gi_enabled = kf_cfg.get("enable_gi_slam", False)
        self._gi_w_covis = kf_cfg.get("w_covis", 1.0)
        self._gi_w_base = kf_cfg.get("w_base", 1.0)
        self._gi_w_mot = kf_cfg.get("w_mot", 2.0)
        self._gi_score_threshold = kf_cfg.get("score_threshold", 0.5)
        self._gi_v_max = kf_cfg.get("v_max", 0.8)
        self._gi_omega_max = kf_cfg.get("omega_max", 50.0)
        self._gi_min_interval = kf_cfg.get("min_keyframe_interval", 1)
        self._gi_fps = kf_cfg.get("fps", 30.0)  # fallback when no timestamps

        # Pre-compute mapping frame IDs (may be overridden dynamically by GI-SLAM)
        if self._gi_enabled:
            # Start empty; keyframes will be selected online
            self.mapping_frame_ids = []
        else:
            self.mapping_frame_ids = frame_ids[::config["mapping"]["map_every"]] + [n_frames - 1]

        # GI-SLAM state: cache visibility + poses of keyframes for IoU scoring
        self._gi_kf_visible_ids = {}  # frame_id → np.ndarray of visible Gaussian indices
        self._gi_kf_c2ws = {}         # frame_id → np.ndarray (4, 4) estimated c2w
        self._gi_prev_c2w = None      # previous frame c2w for velocity computation
        self._gi_prev_frame_id = None # previous frame id for dt computation
        # ────────────────────────────────────────────────────────────────

        self.keyframes_info = {}
        self.opt = OptimizationParams(ArgumentParser(description="Training script parameters"))

        if self.submap_using_motion_heuristic:
            self.new_submap_frame_ids = [0]
        else:
            self.new_submap_frame_ids = frame_ids[::config["mapping"]["new_submap_every"]] + [n_frames - 1]
            self.new_submap_frame_ids.pop(0)

        self.logger = Logger(self.output_path, config["use_wandb"])
        self.mapper = Mapper(
            config["mapping"], config.get("gaussian_pyramid", {}),
            self.dataset, self.logger)
        self._gi_use_imu_gyro = kf_cfg.get("use_imu_gyro", False)
        self._gi_max_gap = kf_cfg.get("max_keyframe_gap", 30)
        if not isinstance(self._gi_use_imu_gyro, bool):
            raise TypeError("keyframing.use_imu_gyro must be bool")
        if (not isinstance(self._gi_min_interval, int)
                or not isinstance(self._gi_max_gap, int)
                or self._gi_min_interval < 1 or self._gi_max_gap < 1):
            raise ValueError("keyframe intervals must be positive integers")
        self._gi_kf_frustum_ids = {}
        self._gi_decisions = {}
        self._gi_decision_counts = {}
        self.tracker = Tracker(config["tracking"], self.dataset, self.logger)
        self.enable_exposure = self.tracker.enable_exposure
        self.loop_closer = Loop_closure(config, self.dataset, self.logger)
        self.loop_closer.submap_path = self.output_path / "submaps"
        save_dict_to_yaml(
            {
                "imu": self.tracker.use_imu,
                "gaussian_pyramid": self.mapper.effective_pyramid_config(),
                "gi_keyframing": self._gi_enabled,
                "gi_keyframing_imu_gyro": self._gi_use_imu_gyro,
            },
            "effective_features.yaml",
            directory=self.output_path,
        )
        
        print('Tracking config')
        pprint.PrettyPrinter().pprint(config["tracking"])
        print('Mapping config')
        pprint.PrettyPrinter().pprint(config["mapping"])
        print('Loop closure config')
        pprint.PrettyPrinter().pprint(config["lc"])
        

    def _setup_output_path(self, config: dict) -> None:
        """ Sets up the output path for saving results based on the provided configuration. If the output path is not
        specified in the configuration, it creates a new directory with a timestamp.
        Args:
            config: A dictionary containing the experiment configuration including data and output path information.
        """
        output_path = config.get("data", {}).get("output_path")
        if not output_path:
            raise ValueError("data.output_path is required")
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True, parents=True)
        
        os.makedirs(self.output_path / "mapping_vis", exist_ok=True)
        os.makedirs(self.output_path / "tracking_vis", exist_ok=True)

    def should_start_new_submap(self, frame_id: int) -> bool:
        """ Determines whether a new submap should be started based on the motion heuristic or specific frame IDs.
        Args:
            frame_id: The ID of the current frame being processed.
        Returns:
            A boolean indicating whether to start a new submap.
        """
        if self.submap_using_motion_heuristic:
            if exceeds_motion_thresholds(
                self.estimated_c2ws[frame_id], self.estimated_c2ws[self.new_submap_frame_ids[-1]],
                    rot_thre=50, trans_thre=0.5):
                print(f"\nNew submap at {frame_id}")
                return True
        elif frame_id in self.new_submap_frame_ids:
            return True
        return False

    def save_current_submap(self, gaussian_model: GaussianModel):
        """Saving the current submap's checkpoint and resetting the Gaussian model

        Args:
            gaussian_model (GaussianModel): The current GaussianModel instance to capture and reset for the new submap.
        """
        
        gaussian_params = gaussian_model.capture_dict()
        submap_ckpt_name = str(self.submap_id).zfill(6)
        submap_ckpt = {
            "gaussian_params": gaussian_params,
            "submap_keyframes": sorted(list(self.keyframes_info.keys()))
        }
        save_dict_to_ckpt(
            submap_ckpt, f"{submap_ckpt_name}.ckpt", directory=self.output_path / "submaps")
    
    def start_new_submap(self, frame_id: int, gaussian_model: GaussianModel) -> None:
        """ Initializes a new submap.
        This function updates the submap count and optionally marks the current frame ID for new submap initiation.
        Args:
            frame_id: The ID of the current frame at which the new submap is started.
            gaussian_model: The current GaussianModel instance to capture and reset for the new submap.
        Returns:
            A new, reset GaussianModel instance for the new submap.
        """
        
        gaussian_model = GaussianModel(0)
        gaussian_model.training_setup(self.opt)
        self.mapper.keyframes = []
        self.mapper.reset_pyramid_state()
        self.keyframes_info = {}
        if self._gi_enabled:
            self._gi_kf_frustum_ids.clear()
            _record_keyframe_decision(
                self,
                frame_id,
                KeyframeDecision(True, 0.0, "submap_boundary", {}),
            )
        if self.submap_using_motion_heuristic:
            self.new_submap_frame_ids.append(frame_id)
        self.mapping_frame_ids.append(frame_id) if frame_id not in self.mapping_frame_ids else self.mapping_frame_ids
        self.submap_id += 1
        self.loop_closer.submap_id += 1

        # Clear GI-SLAM keyframe state on submap reset (Gaussian model is reset)
        self._gi_kf_visible_ids.clear()
        self._gi_kf_c2ws.clear()

        return gaussian_model

    def _should_map_frame_gi_slam(self, frame_id: int,
                                   gaussian_model: GaussianModel,
                                   estimated_c2w: np.ndarray) -> bool:
        return evaluate_gi_keyframe(
            self, frame_id, gaussian_model, estimated_c2w).selected

        """Evaluate whether the current frame should become a mapping keyframe
        based on GI-SLAM's content-aware selection (Sec. 3.3).

        Scoring: s_i = w_covis*(1-IoU_G) + w_base*||t_ij||/d_med
                       - w_mot*𝕀(v_i>v_max ∨ ω_i>ω_max)

        If s_i > threshold, the frame is selected.

        Args:
            frame_id: Current frame index.
            gaussian_model: Current submap Gaussian model.
            estimated_c2w: Estimated camera-to-world pose of current frame.

        Returns:
            True if the frame should be a keyframe.
        """
        # Always select frame 0 and the last frame as keyframes
        if frame_id == 0 or frame_id == len(self.dataset) - 1:
            return True

        # Enforce minimum frame interval between keyframes
        if len(self.mapping_frame_ids) > 0:
            last_kf = self.mapping_frame_ids[-1]
            if frame_id - last_kf < self._gi_min_interval:
                return False

        # ── Safety: force keyframe if gap is too large ───────────────
        # Prevents model starvation when GI-KF score is consistently below
        # threshold (e.g. fast motion + odometer noise = perpetual motion
        # penalty).  The default map_every=1 would map every frame; this
        # safety net ensures we map at least every 30 frames.
        _max_gap = max(30, self._gi_min_interval * 10)
        if len(self.mapping_frame_ids) > 0:
            last_kf = self.mapping_frame_ids[-1]
            if frame_id - last_kf > _max_gap:
                print(f"\n⚠️  GI-SLAM SAFETY: frame {frame_id} forced as keyframe "
                      f"(gap={frame_id - last_kf} > {_max_gap}, last_kf={last_kf}). "
                      f"GI-KF threshold too strict for this dataset.")
                return True
        # ──────────────────────────────────────────────────────────────

        # ── Velocity estimation ──────────────────────────────────────
        linear_vel, angular_vel = 0.0, 0.0

        # Time delta from dataset timestamps or fallback FPS
        if hasattr(self.dataset, 'timestamps') and \
           self._gi_prev_frame_id is not None and \
           self._gi_prev_frame_id < len(self.dataset.timestamps) and \
           frame_id < len(self.dataset.timestamps):
            dt = self.dataset.timestamps[frame_id] - self.dataset.timestamps[self._gi_prev_frame_id]
        else:
            dt = 1.0 / self._gi_fps

        # Visual odometry velocity
        if self._gi_prev_c2w is not None and dt > 0:
            linear_vel, angular_vel = compute_camera_velocity(
                estimated_c2w, self._gi_prev_c2w, dt)

        # Supplement with IMU angular velocity when available (deg/s)
        if hasattr(self.dataset, 'get_imu_data_for_frame'):
            imu_data = self.dataset.get_imu_data_for_frame(frame_id)
            if imu_data is not None and 'angular_velocity' in imu_data:
                imu_omega = np.linalg.norm(imu_data['angular_velocity'])
                # rad/s → deg/s, take max of visual and IMU
                angular_vel = max(angular_vel, imu_omega * 180.0 / np.pi)

        # ── Gaussian visibility of current frame ─────────────────────
        _, gt_color, gt_depth, _ = self.dataset[frame_id]
        estimate_w2c = np.linalg.inv(estimated_c2w)
        gaussian_xyz = gaussian_model.get_xyz()

        visible_ids_curr = compute_gaussian_visibility(
            gaussian_xyz, estimate_w2c, self.dataset.intrinsics, gt_depth)

        # ── Score against nearest keyframe ───────────────────────────
        last_kf_id = self.mapping_frame_ids[-1]
        nearest_kf_visible = self._gi_kf_visible_ids.get(last_kf_id, np.array([], dtype=np.int64))
        nearest_kf_c2w = self._gi_kf_c2ws.get(last_kf_id, estimated_c2w)

        score = gi_slam_keyframe_score(
            visible_ids_curr, nearest_kf_visible,
            estimated_c2w, nearest_kf_c2w, gt_depth,
            linear_vel, angular_vel,
            w_covis=self._gi_w_covis,
            w_base=self._gi_w_base,
            w_mot=self._gi_w_mot,
            v_max=self._gi_v_max,
            omega_max=self._gi_omega_max,
        )

        selected = score > self._gi_score_threshold
        if selected:
            from src.utils.mapper_utils import compute_gaussian_iou
            iou = compute_gaussian_iou(visible_ids_curr, nearest_kf_visible)
            print(f"\nGI-SLAM: frame {frame_id} selected as keyframe "
                  f"(score={score:.3f}, covis=1-IoU={1-iou:.3f}, "
                  f"baseline={np.linalg.norm(estimated_c2w[:3, 3] - nearest_kf_c2w[:3, 3]):.3f}m, "
                  f"vel={linear_vel:.2f}/{angular_vel:.1f})")

        return selected

    def _register_gi_keyframe(self, frame_id: int, gaussian_model: GaussianModel,
                               estimated_c2w: np.ndarray) -> None:
        register_gi_keyframe(
            self, frame_id, gaussian_model, estimated_c2w)
        return

        """Cache the visible Gaussian IDs and pose of a newly-mapped keyframe
        for subsequent GI-SLAM covisibility scoring.

        Args:
            frame_id: Frame just mapped as a keyframe.
            gaussian_model: Updated Gaussian model after mapping.
            estimated_c2w: Estimated camera-to-world pose.
        """
        _, gt_color, gt_depth, _ = self.dataset[frame_id]
        estimate_w2c = np.linalg.inv(estimated_c2w)
        gaussian_xyz = gaussian_model.get_xyz()

        visible_ids = compute_gaussian_visibility(
            gaussian_xyz, estimate_w2c, self.dataset.intrinsics, gt_depth)

        self._gi_kf_visible_ids[frame_id] = visible_ids
        self._gi_kf_c2ws[frame_id] = estimated_c2w.copy()
    
    def rigid_transform_gaussians(self, gaussian_params, tsfm_matrix):
        '''
        Apply a rigid transformation to the Gaussian parameters.
        
        Args:
            gaussian_params (dict): Dictionary containing Gaussian parameters.
            tsfm_matrix (torch.Tensor): 4x4 rigid transformation matrix.
            
        Returns:
            dict: Updated Gaussian parameters after applying the transformation.
        '''
        # Transform Gaussian centers (xyz)
        tsfm_matrix = torch.from_numpy(tsfm_matrix).float()
        xyz = gaussian_params['xyz']
        pts_ones = torch.ones((xyz.shape[0], 1))
        pts_homo = torch.cat([xyz, pts_ones], dim=1)
        transformed_xyz = (tsfm_matrix @ pts_homo.T).T[:, :3]
        gaussian_params['xyz'] = transformed_xyz

        # Rotate covariance matrix (rotation)
        rotation = gaussian_params['rotation']
        cur_rot = roma.unitquat_to_rotmat(rotation)
        rot_mat = tsfm_matrix[:3, :3].unsqueeze(0)  # Adding batch dimension
        new_rot = rot_mat @ cur_rot
        new_quat = roma.rotmat_to_unitquat(new_rot)
        gaussian_params['rotation'] = new_quat.squeeze()

        return gaussian_params
    
    def update_keyframe_poses(self, lc_output, submaps_kf_ids, cur_frame_id):
        '''
        Update the keyframe poses using the correction from pgo, currently update the frame range that covered by the keyframes.
        
        '''
        for correction in lc_output:
            submap_id = correction['submap_id']
            correct_tsfm = correction['correct_tsfm']
            submap_kf_ids = submaps_kf_ids[submap_id]
            min_id, max_id = min(submap_kf_ids), max(submap_kf_ids)
            self.estimated_c2ws[min_id:max_id + 1] = torch.from_numpy(correct_tsfm).float() @ self.estimated_c2ws[min_id:max_id + 1]
        
        # last tracked frame is based on last submap, update it as well
        self.estimated_c2ws[cur_frame_id] = torch.from_numpy(lc_output[-1]['correct_tsfm']).float() @ self.estimated_c2ws[cur_frame_id]
        
        
    def apply_correction_to_submaps(self, correction_list):
        submaps_kf_ids= {}
        for correction in correction_list:
            submap_id = correction['submap_id']
            correct_tsfm = correction['correct_tsfm']

            submap_ckpt_name = str(submap_id).zfill(6) + ".ckpt"
            submap_ckpt = torch.load(self.output_path / "submaps" / submap_ckpt_name)
            submaps_kf_ids[submap_id] = submap_ckpt["submap_keyframes"]

            gaussian_params = submap_ckpt["gaussian_params"]
            updated_gaussian_params = self.rigid_transform_gaussians(
                gaussian_params, correct_tsfm)

            submap_ckpt["gaussian_params"] = updated_gaussian_params
            torch.save(submap_ckpt, self.output_path / "submaps" / submap_ckpt_name)
            self.loop_closer.invalidate_submap_cache([submap_id])
        return submaps_kf_ids
    
    def run(self) -> None:
        """ Starts the main program flow for Gaussian-SLAM, including tracking and mapping. """
        setup_seed(self.config["seed"])
        gaussian_model = GaussianModel(0)
        gaussian_model.training_setup(self.opt)
        self.submap_id = 0

        for frame_id in range(len(self.dataset)):

            use_dataset_pose = should_use_dataset_pose(
                frame_id,
                self.config["tracking"].get("gt_camera", False),
                self.dataset.has_ground_truth,
            )
            if use_dataset_pose:
                estimated_c2w = self.dataset[frame_id][-1]
                exposure_ab = torch.nn.Parameter(torch.tensor(
                    0.0, device="cuda")), torch.nn.Parameter(torch.tensor(0.0, device="cuda"))
            else:
                pose_ids = [0, max(0, frame_id - 2), frame_id - 1]
                estimated_c2w, exposure_ab = self.tracker.track(
                    frame_id, gaussian_model,
                    torch2np(self.estimated_c2ws[torch.tensor(pose_ids)]))
            exposure_ab = exposure_ab if self.enable_exposure else None
            self.estimated_c2ws[frame_id] = np2torch(estimated_c2w)

            # ── GI-SLAM dynamic keyframe selection ───────────────────
            if self._gi_enabled:
                decision = evaluate_gi_keyframe(
                    self, frame_id, gaussian_model, estimated_c2w)
                _record_keyframe_decision(self, frame_id, decision)
                if decision.selected:
                    self.mapping_frame_ids.append(frame_id)
            # ──────────────────────────────────────────────────────────

            # Reinitialize gaussian model for new segment
            if self.should_start_new_submap(frame_id):
                # first save current submap and its keyframe info
                self.save_current_submap(gaussian_model)
                
                # update submap infomation for loop closer
                self.loop_closer.update_submaps_info(self.keyframes_info)
                
                # apply loop closure
                lc_output = self.loop_closer.loop_closure(self.estimated_c2ws)
                
                if len(lc_output) > 0:
                    submaps_kf_ids = self.apply_correction_to_submaps(lc_output)
                    self.update_keyframe_poses(lc_output, submaps_kf_ids, frame_id)
                
                save_dict_to_ckpt(self.estimated_c2ws[:frame_id + 1], "estimated_c2w.ckpt", directory=self.output_path)
                
                gaussian_model = self.start_new_submap(frame_id, gaussian_model)

            if frame_id in self.mapping_frame_ids:
                print("\nMapping frame", frame_id)
                gaussian_model.training_setup(self.opt, exposure_ab) 
                estimate_c2w = torch2np(self.estimated_c2ws[frame_id])
                new_submap = not bool(self.keyframes_info)
                opt_dict = self.mapper.map(
                    frame_id, estimate_c2w, gaussian_model, new_submap, exposure_ab)

                # Keyframes info update
                self.keyframes_info[frame_id] = {
                    "keyframe_id": frame_id, 
                    "opt_dict": opt_dict,
                }
                if self.enable_exposure:
                    self.keyframes_info[frame_id]["exposure_a"] = exposure_ab[0].item()
                    self.keyframes_info[frame_id]["exposure_b"] = exposure_ab[1].item()

                # Register frustum-center IDs; this is not occlusion visibility.
                if self._gi_enabled:
                    register_gi_keyframe(
                        self, frame_id, gaussian_model,
                        torch2np(self.estimated_c2ws[frame_id]))
                # ───────────────────────────────────────────────────────────────────

            if frame_id == len(self.dataset) - 1 and self.config['lc']['final']:
                print("\n Final loop closure ...")
                self.loop_closer.update_submaps_info(self.keyframes_info)
                lc_output = self.loop_closer.loop_closure(self.estimated_c2ws, final=True)
                if len(lc_output) > 0:
                    submaps_kf_ids = self.apply_correction_to_submaps(lc_output)
                    self.update_keyframe_poses(lc_output, submaps_kf_ids, frame_id)
            if self.enable_exposure:
                self.exposures_ab[frame_id] = torch.tensor([exposure_ab[0].item(), exposure_ab[1].item()])

            # ── GI-SLAM: track previous frame for velocity estimation ──
            if self._gi_enabled:
                self._gi_prev_c2w = estimated_c2w.copy()
                self._gi_prev_frame_id = frame_id
            # ────────────────────────────────────────────────────────────

        # Save final submap if there are unsaved keyframes
        if len(self.keyframes_info) > 0:
            print(f"\nSaving final submap {self.submap_id} with {len(self.keyframes_info)} keyframes")
            self.save_current_submap(gaussian_model)
            self.loop_closer.update_submaps_info(self.keyframes_info)

        save_dict_to_ckpt(self.estimated_c2ws[:frame_id + 1], "estimated_c2w.ckpt", directory=self.output_path)
        if self.enable_exposure:
            save_dict_to_ckpt(self.exposures_ab, "exposures_ab.ckpt", directory=self.output_path)
        save_dict_to_yaml(
            {
                "enabled": self.tracker.use_imu,
                "committed_frame_ids": self.tracker.imu_committed_frame_ids,
                "commit_count": len(self.tracker.imu_committed_frame_ids),
                "last_committed_frame_id": (
                    self.tracker.imu_state.last_committed_frame_id),
                "translation_initialized": (
                    self.tracker.imu_state.gravity_cam is not None),
            },
            "imu_tracking_summary.yaml",
            directory=self.output_path,
        )
        pyramid_usage = self.mapper.pyramid_lifetime_usage_summary()
        pyramid_config = self.mapper.effective_pyramid_config()
        level_totals = {
            level_id: sum(
                counts.get(level_id, 0) for counts in pyramid_usage.values())
            for level_id in range(
                pyramid_config["num_sub_levels"] + 1)
        }
        save_dict_to_yaml(
            {
                **pyramid_config,
                "frame_level_usage": pyramid_usage,
                "level_totals": level_totals,
                "optimizer_step_count": sum(level_totals.values()),
            },
            "gaussian_pyramid_summary.yaml",
            directory=self.output_path,
        )
