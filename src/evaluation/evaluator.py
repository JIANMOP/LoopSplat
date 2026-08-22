""" This module is responsible for evaluating rendering, trajectory and reconstruction metrics"""
import traceback
from argparse import ArgumentParser
from copy import deepcopy
from itertools import cycle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Set non-GUI backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
import torchvision
import json
from pytorch_msssim import ms_ssim
from torch.utils.data import DataLoader
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision.utils import save_image
from tqdm import tqdm

from src.entities.arguments import OptimizationParams
from src.entities.datasets import get_dataset
from src.entities.gaussian_model import GaussianModel
from src.evaluation.evaluate_merged_map import (RenderFrames, merge_submaps,
                                                refine_global_map)
from src.evaluation.evaluate_reconstruction import evaluate_reconstruction, clean_mesh
from src.evaluation.evaluate_trajectory import evaluate_trajectory
from src.evaluation.protocol import (
    assign_frames_to_submaps,
    build_evaluation_frame_ids,
    masked_depth_l1,
    trajectory_status,
)
from src.utils.io_utils import load_config, save_dict_to_json, log_metrics_to_wandb
from src.utils.mapper_utils import calc_psnr
from src.utils.utils import (get_render_settings, np2torch,
                             render_gaussian_model, setup_seed, 
                             torch2np, filter_depth_outliers)


class Evaluator(object):

    def __init__(self, checkpoint_path, config_path, config=None, save_render=False) -> None:
        if config is None:
            self.config = load_config(config_path)
        else:
            self.config = config
        setup_seed(self.config["seed"])

        self.checkpoint_path = Path(checkpoint_path)
        self.use_wandb = self.config["use_wandb"]
        self.device = "cuda"
        dataset_config = {**self.config["data"], **self.config["cam"]}
        if "frame_limit" in self.config:
            dataset_config["frame_limit"] = self.config["frame_limit"]
        self.dataset = get_dataset(self.config["dataset_name"])(
            dataset_config)
        self.scene_name = self.config["data"]["scene_name"]
        self.dataset_name = self.config["dataset_name"]
        self.gt_poses = (
            np.array(self.dataset.poses)
            if self.dataset.has_ground_truth else None)
        self.evaluation_config = self.config.get("evaluation", {})
        self.fx, self.fy = self.dataset.intrinsics[0, 0], self.dataset.intrinsics[1, 1]
        self.cx, self.cy = self.dataset.intrinsics[0,
                                                   2], self.dataset.intrinsics[1, 2]
        self.width, self.height = self.dataset.width, self.dataset.height
        self.save_render = save_render
        if self.save_render:
            self.render_path = self.checkpoint_path / "rendered_imgs"
            self.render_path.mkdir(exist_ok=True, parents=True)

        c2w_path = self.checkpoint_path / "estimated_c2w.ckpt"
        if not c2w_path.exists():
            raise FileNotFoundError(
                f"Missing {c2w_path} — SLAM run may not have completed successfully")
        self.estimated_c2w = torch2np(torch.load(c2w_path, map_location=self.device))
        submaps_dir = self.checkpoint_path / "submaps"
        self.submaps_paths = sorted(list(submaps_dir.glob('*'))) if submaps_dir.exists() else []
        self.exposures_ab = None
        if (self.checkpoint_path / "exposures_ab.ckpt").exists():
            self.exposures_ab = torch2np(torch.load(self.checkpoint_path / "exposures_ab.ckpt", map_location=self.device))
            print(f"Loaded trained exposures paramters for scene {self.scene_name}")

    def run_trajectory_eval(self):
        """ Evaluates the estimated trajectory """
        print("Running trajectory evaluation...")
        status = trajectory_status(self.dataset)
        save_dict_to_json(
            {"status": status}, "trajectory_status.json",
            directory=self.checkpoint_path)
        if status != "available":
            print("Skipping trajectory evaluation: dataset has no ground truth")
            return
        evaluate_trajectory(self.estimated_c2w, self.gt_poses, self.checkpoint_path)

    def run_rendering_eval(self):
        """Evaluate fixed observed frames, independent of keyframe selection."""
        print("Running fixed observed-view rendering evaluation...")
        stride = self.evaluation_config.get("observed_view_stride", 10)
        frame_ids = build_evaluation_frame_ids(len(self.dataset), stride)
        save_dict_to_json(
            frame_ids, "evaluation_frame_ids.json",
            directory=self.checkpoint_path)

        refinement = self.evaluation_config.get("global_refinement", {})
        protocol = {
            "frame_ids": frame_ids,
            "observed_view_stride": stride,
            "formal_map_source": "submap_checkpoint",
            "global_refinement_enabled": False,
            "global_refinement_iterations": 0,
            "optional_global_refinement_enabled": bool(
                refinement.get("enabled", False)),
            "optional_global_refinement_iterations": int(
                refinement.get("iterations", 0)),
        }
        save_dict_to_json(
            protocol, "evaluation_protocol.json",
            directory=self.checkpoint_path)

        submap_paths = sorted(
            (self.checkpoint_path / "submaps").glob("*.ckpt"))
        if not submap_paths:
            raise RuntimeError("no submap checkpoints available for evaluation")
        submap_keyframe_ids = []
        for submap_path in submap_paths:
            submap = torch.load(submap_path, map_location="cpu")
            submap_keyframe_ids.append(submap["submap_keyframes"])
        assignments = assign_frames_to_submaps(
            frame_ids, submap_keyframe_ids)

        color_transform = torchvision.transforms.ToTensor()
        lpips_model = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True).to(self.device)
        opt_settings = OptimizationParams(ArgumentParser(
            description="Training script parameters"))
        psnr_values, lpips_values, ssim_values, depth_values = [], [], [], []
        depth_valid_pixels = 0

        for submap_index, submap_path in enumerate(submap_paths):
            assigned_ids = assignments[submap_index]
            if not assigned_ids:
                continue
            submap = torch.load(submap_path, map_location=self.device)
            gaussian_model = GaussianModel()
            gaussian_model.training_setup(opt_settings)
            gaussian_model.restore_from_params(
                submap["gaussian_params"], opt_settings)

            for frame_id in assigned_ids:
                _, gt_color, gt_depth, _ = self.dataset[frame_id]
                gt_color = color_transform(gt_color).to(self.device)
                gt_depth = np2torch(gt_depth).to(self.device)
                estimate_w2c = np.linalg.inv(self.estimated_c2w[frame_id])
                render_dict = render_gaussian_model(
                    gaussian_model,
                    get_render_settings(
                        self.width, self.height, self.dataset.intrinsics,
                        estimate_w2c),
                )
                rendered_color = torch.clamp(
                    render_dict["color"].detach(), 0.0, 1.0)
                rendered_depth = render_dict["depth"][0].detach()
                if self.exposures_ab is not None:
                    exposure = self.exposures_ab[frame_id]
                    rendered_color = torch.clamp(
                        torch.exp(torch.as_tensor(
                            exposure[0], device=self.device))
                        * rendered_color + exposure[1], 0.0, 1.0)
                if self.save_render:
                    torchvision.utils.save_image(
                        rendered_color,
                        self.render_path / f"{frame_id:05d}.png")

                mse = torch.nn.functional.mse_loss(
                    rendered_color, gt_color)
                psnr_values.append((-10.0 * torch.log10(mse)).item())
                lpips_values.append(lpips_model(
                    rendered_color[None], gt_color[None]).item())
                ssim_values.append(ms_ssim(
                    rendered_color[None], gt_color[None],
                    data_range=1.0, size_average=True).item())
                depth_l1, valid_count = masked_depth_l1(
                    rendered_depth, gt_depth)
                if valid_count:
                    depth_values.append(depth_l1.item())
                    depth_valid_pixels += valid_count

            del gaussian_model, submap
            torch.cuda.empty_cache()

        if len(psnr_values) != len(frame_ids):
            raise RuntimeError("not every formal evaluation frame was rendered")
        metrics = {
            "psnr": float(np.mean(psnr_values)),
            "lpips": float(np.mean(lpips_values)),
            "ssim": float(np.mean(ssim_values)),
            "depth_l1_observed_view": (
                float(np.mean(depth_values)) if depth_values else None),
            "depth_valid_pixels": depth_valid_pixels,
            "num_renders": len(frame_ids),
        }
        save_dict_to_json(
            metrics, "rendering_metrics_observed_view.json",
            directory=self.checkpoint_path)
        print(metrics)

    def run_keyframe_rendering_diagnostic(self):
        """ Renders the submaps and global splats and evaluates the PSNR, LPIPS, SSIM and depth L1 metrics."""
        print("Running rendering evaluation...")
        psnr, lpips, ssim, depth_l1 = [], [], [], []
        color_transform = torchvision.transforms.ToTensor()
        lpips_model = LearnedPerceptualImagePatchSimilarity(
            net_type='alex', normalize=True).to(self.device)
        opt_settings = OptimizationParams(ArgumentParser(
            description="Training script parameters"))

        submaps_paths = sorted(
            list((self.checkpoint_path / "submaps").glob('*.ckpt')))
        for submap_path in tqdm(submaps_paths):
            submap = torch.load(submap_path, map_location=self.device)
            gaussian_model = GaussianModel()
            gaussian_model.training_setup(opt_settings)
            gaussian_model.restore_from_params(
                submap["gaussian_params"], opt_settings)

            for keyframe_id in submap["submap_keyframes"]:

                _, gt_color, gt_depth, _ = self.dataset[keyframe_id]
                gt_color = color_transform(gt_color).to(self.device)
                gt_depth = np2torch(gt_depth).to(self.device)

                estimate_c2w = self.estimated_c2w[keyframe_id]
                estimate_w2c = np.linalg.inv(estimate_c2w)
                render_dict = render_gaussian_model(
                    gaussian_model, get_render_settings(self.width, self.height, self.dataset.intrinsics, estimate_w2c))
                rendered_color, rendered_depth = render_dict["color"].detach(
                ), render_dict["depth"][0].detach()
                rendered_color = torch.clamp(rendered_color, min=0.0, max=1.0)
                if self.save_render:
                    torchvision.utils.save_image(
                        rendered_color, self.render_path / f"{keyframe_id:05d}.png")

                mse_loss = torch.nn.functional.mse_loss(
                    rendered_color, gt_color)
                psnr_value = (-10. * torch.log10(mse_loss)).item()
                lpips_value = lpips_model(
                    rendered_color[None], gt_color[None]).item()
                ssim_value = ms_ssim(
                    rendered_color[None], gt_color[None], data_range=1.0, size_average=True).item()
                depth_l1_value = torch.abs(
                    (rendered_depth - gt_depth)).mean().item()

                psnr.append(psnr_value)
                lpips.append(lpips_value)
                ssim.append(ssim_value)
                depth_l1.append(depth_l1_value)

        num_frames = len(psnr)
        metrics = {
            "psnr": sum(psnr) / num_frames,
            "lpips": sum(lpips) / num_frames,
            "ssim": sum(ssim) / num_frames,
            "depth_l1_train_view": sum(depth_l1) / num_frames,
            "num_renders": num_frames
        }
        save_dict_to_json(metrics, "rendering_metrics_keyframe_diagnostic.json",
                          directory=self.checkpoint_path)

        x = list(range(len(psnr)))
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].plot(x, psnr, label="PSNR")
        axs[0].legend()
        axs[0].set_title("PSNR")
        axs[1].plot(x, ssim, label="SSIM")
        axs[1].legend()
        axs[1].set_title("SSIM")
        axs[2].plot(x, depth_l1, label="Depth L1 (Train view)")
        axs[2].legend()
        axs[2].set_title("Depth L1 Render")
        plt.tight_layout()
        plt.savefig(str(self.checkpoint_path /
                    "rendering_metrics.png"), dpi=300)
        print(metrics)

    def run_reconstruction_eval(self):
        """ Reconstructs the mesh, evaluates it, render novel view depth maps from it, and evaluates them as well """
        print("Running reconstruction evaluation...")

        (self.checkpoint_path / "mesh").mkdir(exist_ok=True, parents=True)
        opt_settings = OptimizationParams(ArgumentParser(
            description="Training script parameters"))
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            self.width, self.height, self.fx, self.fy, self.cx, self.cy)
        scale = 1.0
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=5.0 * scale / 512.0,
            sdf_trunc=0.04 * scale,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

        submaps_paths = sorted(list((self.checkpoint_path / "submaps").glob('*.ckpt')))
        for submap_path in tqdm(submaps_paths):
            submap = torch.load(submap_path, map_location=self.device)
            gaussian_model = GaussianModel()
            gaussian_model.training_setup(opt_settings)
            gaussian_model.restore_from_params(
                submap["gaussian_params"], opt_settings)

            for keyframe_id in submap["submap_keyframes"]:
                estimate_c2w = self.estimated_c2w[keyframe_id]
                estimate_w2c = np.linalg.inv(estimate_c2w)
                render_dict = render_gaussian_model(
                    gaussian_model, get_render_settings(self.width, self.height, self.dataset.intrinsics, estimate_w2c))
                rendered_color, rendered_depth = render_dict["color"].detach(
                ), render_dict["depth"][0].detach()
                rendered_color = torch.clamp(rendered_color, min=0.0, max=1.0)

                rendered_color = (
                    torch2np(rendered_color.permute(1, 2, 0)) * 255).astype(np.uint8)
                rendered_depth = torch2np(rendered_depth)

                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(np.ascontiguousarray(rendered_color)),
                    o3d.geometry.Image(rendered_depth),
                    depth_scale=scale,
                    depth_trunc=30,
                    convert_rgb_to_intensity=False)
                volume.integrate(rgbd, intrinsic, estimate_w2c)

        o3d_mesh = volume.extract_triangle_mesh()
        compensate_vector = (-0.0 * scale / 512.0, 2.5 *
                             scale / 512.0, -2.5 * scale / 512.0)
        o3d_mesh = o3d_mesh.translate(compensate_vector)
        o3d_mesh = clean_mesh(o3d_mesh)
        file_name = self.checkpoint_path / "mesh" / "cleaned_mesh.ply"
        o3d.io.write_triangle_mesh(str(file_name), o3d_mesh)
        print(f"Reconstructed mesh saved to {file_name}")
        if self.config["dataset_name"] == "replica":
            evaluate_reconstruction(file_name,
                                    f"data/Replica-SLAM/cull_replica/{self.scene_name}/gt_mesh_cull_virt_cams.ply",
                                    f"data/Replica-SLAM/cull_replica/{self.scene_name}/gt_pc_unseen.npy",
                                    self.checkpoint_path)


    def run_global_map_eval(self, init_from='mesh'):
        """ Merges the map, evaluates it over training and novel views 
        
        Args:
            init_from (str, optional): 'mesh' or 'splats'. Initialization method for the global splats. Defaults to mesh vertices reconstructed before.
        """
        print("Running global map evaluation...")

        training_frames = RenderFrames(
            self.dataset, self.estimated_c2w, self.height, self.width,
            self.fx, self.fy, self.exposures_ab,
            frame_ids=build_evaluation_frame_ids(
                len(self.dataset),
                self.evaluation_config.get("observed_view_stride", 10)))
        training_frames = DataLoader(
            training_frames, batch_size=1, shuffle=True)
        len_frames = len(training_frames)
        training_frames = cycle(training_frames)

        # Check if submaps exist before merging
        if len(self.submaps_paths) == 0:
            print("Warning: No submaps found, skipping global map evaluation")
            return

        merged_cloud = merge_submaps(self.submaps_paths) if init_from == 'splats' else None

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            self.width, self.height, self.fx, self.fy, self.cx, self.cy)
        refinement_config = self.evaluation_config.get(
            "global_refinement", {})
        refinement_iterations = refinement_config.get("iterations", 0)
        if type(refinement_iterations) is not int or refinement_iterations < 1:
            raise ValueError(
                "enabled global refinement requires positive iterations")
        refined_merged_gaussian_model = refine_global_map(merged_cloud, training_frames, refinement_iterations, export_refine_mesh=False,
                                                          output_dir=self.checkpoint_path, len_frames=len_frames, 
                                                          o3d_intrinsic=intrinsic)
        ply_path = self.checkpoint_path / \
            f"{self.config['data']['scene_name']}_global_splats.ply"
        refined_merged_gaussian_model.save_ply(ply_path)
        print(f"Refined global splats saved to {ply_path}")

        if self.config["dataset_name"] == "scannetpp":
            # "NVS evaluation only supported for scannetpp"

            eval_config = deepcopy(self.config)
            print(f"✨ Eval NVS for scene {self.config['data']['scene_name']}...")
            (self.checkpoint_path / "nvs_eval").mkdir(exist_ok=True, parents=True)
            eval_config["data"]["use_train_split"] = False
            test_set = get_dataset(eval_config["dataset_name"])({**eval_config["data"], **eval_config["cam"]})
            test_poses = torch.stack([torch.from_numpy(test_set[i][3]) for i in range(len(test_set))], dim=0)
            test_frames = RenderFrames(test_set, test_poses, self.height, self.width, self.fx, self.fy)

            psnr_list = []
            for i in tqdm(range(len(test_set))):
                gt_color, _, render_settings = (
                    test_frames[i]["color"],
                    test_frames[i]["depth"],
                    test_frames[i]["render_settings"])
                render_dict = render_gaussian_model(refined_merged_gaussian_model, render_settings)
                rendered_color, _ = (render_dict["color"].permute(1, 2, 0), render_dict["depth"],)
                rendered_color = torch.clip(rendered_color, 0, 1)
                save_image(rendered_color.permute(2, 0, 1), self.checkpoint_path / f"nvs_eval/{i:04d}.jpg")
                psnr = calc_psnr(gt_color, rendered_color)
                psnr_list.append(psnr.item())
            print(f"PSNR List: {psnr_list}")
            print(f"Avg. NVS PSNR: {np.array(psnr_list).mean()}")
            with open(self.checkpoint_path / 'nvs_eval' / "results.json", "w") as f:
                data = {"avg_nvs_psnr": np.mean(psnr_list)}
                json.dump(data, f, indent=4)
        
        else: # evaluate rendering performance on the global submap
            print("Running rendering evaluation on global map ...")
            psnr, lpips, ssim, depth_l1 = [], [], [], []
            color_transform = torchvision.transforms.ToTensor()
            lpips_model = LearnedPerceptualImagePatchSimilarity(
                net_type='alex', normalize=True).to(self.device)
            
            submaps_paths = sorted(list((self.checkpoint_path / "submaps").glob('*.ckpt')))
            for submap_path in tqdm(submaps_paths):
                submap = torch.load(submap_path, map_location=self.device)

                for keyframe_id in submap["submap_keyframes"]:

                    _, gt_color, gt_depth, _ = self.dataset[keyframe_id]
                    gt_color = color_transform(gt_color).to(self.device)
                    gt_depth = np2torch(gt_depth).to(self.device)

                    estimate_c2w = self.estimated_c2w[keyframe_id]
                    estimate_w2c = np.linalg.inv(estimate_c2w)
                    render_dict = render_gaussian_model(
                        refined_merged_gaussian_model, get_render_settings(self.width, self.height, self.dataset.intrinsics, estimate_w2c))
                    rendered_color, rendered_depth = render_dict["color"].detach(
                    ), render_dict["depth"][0].detach()
                    rendered_color = torch.clamp(rendered_color, min=0.0, max=1.0)
                    if self.save_render:
                        torchvision.utils.save_image(
                            rendered_color, self.render_path / f"{keyframe_id:05d}.png")

                    mse_loss = torch.nn.functional.mse_loss(
                        rendered_color, gt_color)
                    psnr_value = (-10. * torch.log10(mse_loss)).item()
                    lpips_value = lpips_model(
                        rendered_color[None], gt_color[None]).item()
                    ssim_value = ms_ssim(
                        rendered_color[None], gt_color[None], data_range=1.0, size_average=True).item()
                    depth_l1_value = torch.abs(
                        (rendered_depth - gt_depth)).mean().item()

                    psnr.append(psnr_value)
                    lpips.append(lpips_value)
                    ssim.append(ssim_value)
                    depth_l1.append(depth_l1_value)

            num_frames = len(psnr)
            metrics = {
                "psnr": sum(psnr) / num_frames,
                "lpips": sum(lpips) / num_frames,
                "ssim": sum(ssim) / num_frames,
                "depth_l1_train_view": sum(depth_l1) / num_frames,
                "num_renders": num_frames
            }
            save_dict_to_json(metrics, "rendering_metrics_global.json",
                            directory=self.checkpoint_path)

            x = list(range(len(psnr)))
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].plot(x, psnr, label="PSNR")
            axs[0].legend()
            axs[0].set_title("PSNR")
            axs[1].plot(x, ssim, label="SSIM")
            axs[1].legend()
            axs[1].set_title("SSIM")
            axs[2].plot(x, depth_l1, label="Depth L1 (Train view)")
            axs[2].legend()
            axs[2].set_title("Depth L1 Render")
            plt.tight_layout()
            plt.savefig(str(self.checkpoint_path /
                        "rendering_metrics_global.png"), dpi=300)
            print(metrics)

    def run(self):
        """ Runs the general evaluation flow """

        print("Starting evaluation...🍺")

        try:
            self.run_trajectory_eval()
        except Exception:
            print("Could not run trajectory eval")
            traceback.print_exc()

        try:
            self.run_rendering_eval()
        except Exception:
            print("Could not run fixed observed-view rendering eval")
            traceback.print_exc()

        if self.evaluation_config.get("run_keyframe_diagnostic", False):
            try:
                self.run_keyframe_rendering_diagnostic()
            except Exception:
                print("Could not run keyframe rendering diagnostic")
                traceback.print_exc()

        if self.evaluation_config.get("run_reconstruction", False):
            try:
                self.run_reconstruction_eval()
            except Exception:
                print("Could not run reconstruction eval")
                traceback.print_exc()

        refinement = self.evaluation_config.get("global_refinement", {})
        refinement_enabled = bool(refinement.get("enabled", False))
        save_dict_to_json(
            {
                "enabled": refinement_enabled,
                "iterations": int(refinement.get("iterations", 0)),
            },
            "global_refinement_status.json",
            directory=self.checkpoint_path)
        if refinement_enabled:
            try:
                self.run_global_map_eval()
            except Exception:
                print("Could not run global map eval")
                traceback.print_exc()

        if self.use_wandb: 
            evals = ["rendering_metrics_observed_view.json", "reconstruction_metrics.json", "ate_aligned.json", "nvs_eval/results.json"]
            log_metrics_to_wandb(evals, self.checkpoint_path, "Evaluation")
