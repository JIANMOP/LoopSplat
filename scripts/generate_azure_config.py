#!/usr/bin/env python3
"""
Azure Kinect Configuration Generator

This script automatically generates YAML configuration files for Azure Kinect datasets
by reading camera_parameters.json and frame_info.json files.

Usage:
    python scripts/generate_azure_config.py data/capture_data/room5
    python scripts/generate_azure_config.py data/capture_data/room5 --output configs/AzureKinect/room5.yaml
"""

import argparse
import json
import yaml
from pathlib import Path


def load_camera_parameters(data_path):
    """Load camera parameters from camera_parameters.json"""
    camera_params_path = Path(data_path) / "camera_parameters.json"
    
    if not camera_params_path.exists():
        raise FileNotFoundError(f"Camera parameters file not found: {camera_params_path}")
    
    with open(camera_params_path, 'r') as f:
        return json.load(f)


def load_frame_info(data_path):
    """Load frame information from frame_info.json"""
    frame_info_path = Path(data_path) / "frame_info.json"
    
    if not frame_info_path.exists():
        raise FileNotFoundError(f"Frame info file not found: {frame_info_path}")
    
    with open(frame_info_path, 'r') as f:
        return json.load(f)


def generate_config(data_path, scene_name=None, output_path=None, resize_mode=None, custom_resolution=None):
    """Generate YAML configuration file for Azure Kinect dataset

    Args:
        data_path: Path to Azure Kinect data directory
        scene_name: Name of the scene (default: directory name)
        output_path: Output configuration file path
        resize_mode: Resize strategy - "rergb", "redepth", or "reall" (default: from base config)
        custom_resolution: Custom resolution for "reall" mode (tuple: width, height, default: from base config)
    """

    # Determine scene config path for priority loading
    if scene_name is None:
        scene_name = Path(data_path).name
    scene_config_path = Path("configs/AzureKinect") / f"{scene_name}.yaml"

    # Load default settings with priority: terminal > scene config > base config
    if resize_mode is None or custom_resolution is None:
        defaults = load_config_defaults(scene_config_path if scene_config_path.exists() else None)
        if resize_mode is None:
            resize_mode = defaults["resize"]
        if custom_resolution is None:
            re_res = defaults["Re_resolution"]
            custom_resolution = (re_res["W"], re_res["H"])

    # Load data
    camera_params = load_camera_parameters(data_path)
    frame_info = load_frame_info(data_path)

    # Extract scene name from path if not provided
    if scene_name is None:
        scene_name = Path(data_path).name

    # Extract camera parameters
    color_intrinsics = camera_params["intrinsics"]["color_camera"]
    depth_intrinsics = camera_params["intrinsics"]["depth_camera"]
    device_config = camera_params["device_config"]

    # Determine main camera parameters based on resize mode
    if resize_mode == "rergb":
        # Use depth camera as main parameters (RGB will be resized to depth)
        main_h = depth_intrinsics["height"]
        main_w = depth_intrinsics["width"]
        main_fx = depth_intrinsics["fx"]
        main_fy = depth_intrinsics["fy"]
        main_cx = depth_intrinsics["cx"]
        main_cy = depth_intrinsics["cy"]
    elif resize_mode == "redepth":
        # Use color camera as main parameters (depth will be resized to RGB)
        main_h = color_intrinsics["height"]
        main_w = color_intrinsics["width"]
        main_fx = color_intrinsics["fx"]
        main_fy = color_intrinsics["fy"]
        main_cx = color_intrinsics["cx"]
        main_cy = color_intrinsics["cy"]
    elif resize_mode == "reall":
        # Use custom resolution with scaled color camera parameters
        if custom_resolution is None:
            custom_resolution = (640, 480)  # Default resolution
        main_w, main_h = custom_resolution

        # Scale color camera parameters to custom resolution
        scale_x = main_w / color_intrinsics["width"]
        scale_y = main_h / color_intrinsics["height"]
        main_fx = color_intrinsics["fx"] * scale_x
        main_fy = color_intrinsics["fy"] * scale_y
        main_cx = color_intrinsics["cx"] * scale_x
        main_cy = color_intrinsics["cy"] * scale_y
    else:
        raise ValueError(f"Invalid resize mode: {resize_mode}. Must be 'rergb', 'redepth', or 'reall'")

    # Create cam configuration with desired order
    cam_config = {}

    # Main camera parameters first (H and W together)
    cam_config["H"] = main_h
    cam_config["W"] = main_w

    # Then Re_resolution
    cam_config["Re_resolution"] = {
        "H": custom_resolution[1],
        "W": custom_resolution[0]
    }

    # Other main camera parameters
    cam_config["fx"] = main_fx
    cam_config["fy"] = main_fy
    cam_config["cx"] = main_cx
    cam_config["cy"] = main_cy
    cam_config["depth_scale"] = device_config["depth_scale"]
    cam_config["depth_trunc"] = device_config.get("depth_trunc", 4.0)

    # Color camera parameters (full resolution)
    cam_config["color_camera"] = {
        "H": color_intrinsics["height"],
        "W": color_intrinsics["width"],
        "fx": color_intrinsics["fx"],
        "fy": color_intrinsics["fy"],
        "cx": color_intrinsics["cx"],
        "cy": color_intrinsics["cy"],
        "distortion": color_intrinsics["distortion_coefficients"]
    }

    # Depth camera parameters (full resolution)
    cam_config["depth_camera"] = {
        "H": depth_intrinsics["height"],
        "W": depth_intrinsics["width"],
        "fx": depth_intrinsics["fx"],
        "fy": depth_intrinsics["fy"],
        "cx": depth_intrinsics["cx"],
        "cy": depth_intrinsics["cy"],
        "distortion": depth_intrinsics["distortion_coefficients"]
    }

    # Resize strategy and crop edge at the end
    cam_config["resize"] = resize_mode

    # Add crop_edge from defaults (only from base config, not scene config to avoid circular dependency)
    defaults = load_config_defaults(None)
    cam_config["crop_edge"] = defaults["crop_edge"]

    # Create output path with resize mode suffix
    output_scene_name = f"{scene_name}_{resize_mode}"

    # Create main configuration dictionary
    config = {
        "inherit_from": "configs/AzureKinect/azure_kinect.yaml",
        "data": {
            "input_path": str(Path(data_path).resolve()),
            "output_path": f"output/AzureKinect/{output_scene_name}",
            "scene_name": scene_name,
            "frame_limit": frame_info["total_frames"]
        },
        "cam": cam_config
    }

    # Add main distortion coefficients if significant (based on resize mode)
    if resize_mode == "rergb":
        main_distortion = depth_intrinsics["distortion_coefficients"]
    else:
        main_distortion = color_intrinsics["distortion_coefficients"]

    if any(abs(coeff) > 1e-6 for coeff in main_distortion):
        config["cam"]["distortion"] = main_distortion
    
    # Determine output path
    if output_path is None:
        output_dir = Path("configs/AzureKinect")
        output_dir.mkdir(parents=True, exist_ok=True)
        # Add suffix for non-default resize modes
        suffix = f"_{resize_mode}" if resize_mode != "redepth" else ""
        output_path = output_dir / f"{scene_name}{suffix}.yaml"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write configuration file with custom order
    with open(output_path, 'w') as f:
        # Write inherit_from first
        f.write(f"inherit_from: {config['inherit_from']}\n")

        # Write data section
        f.write("data:\n")
        for key, value in config['data'].items():
            if isinstance(value, str):
                f.write(f"  {key}: {value}\n")
            else:
                f.write(f"  {key}: {value}\n")

        # Write cam section with controlled order
        f.write("cam:\n")
        cam = config['cam']

        # Main resolution parameters first
        f.write(f"  H: {cam['H']}\n")
        f.write(f"  W: {cam['W']}\n")

        # Re_resolution next
        f.write("  Re_resolution:\n")
        f.write(f"    H: {cam['Re_resolution']['H']}\n")
        f.write(f"    W: {cam['Re_resolution']['W']}\n")

        # Other main parameters
        f.write(f"  fx: {cam['fx']}\n")
        f.write(f"  fy: {cam['fy']}\n")
        f.write(f"  cx: {cam['cx']}\n")
        f.write(f"  cy: {cam['cy']}\n")
        f.write(f"  depth_scale: {cam['depth_scale']}\n")
        f.write(f"  depth_trunc: {cam['depth_trunc']}\n")

        # Color camera section
        f.write("  color_camera:\n")
        color_cam = cam['color_camera']
        for key, value in color_cam.items():
            if key == 'distortion':
                f.write(f"    {key}:\n")
                for coeff in value:
                    f.write(f"    - {coeff}\n")
            else:
                f.write(f"    {key}: {value}\n")

        # Depth camera section
        f.write("  depth_camera:\n")
        depth_cam = cam['depth_camera']
        for key, value in depth_cam.items():
            if key == 'distortion':
                f.write(f"    {key}:\n")
                for coeff in value:
                    f.write(f"    - {coeff}\n")
            else:
                f.write(f"    {key}: {value}\n")

        # Resize strategy and crop edge
        f.write(f"  resize: {cam['resize']}\n")
        f.write(f"  crop_edge: {cam['crop_edge']}\n")

    print(f"Generated configuration: {output_path}")
    print(f"Scene: {scene_name}")
    print(f"Resize mode: {resize_mode}")
    print(f"Main resolution: {main_w}x{main_h}")
    print(f"Color resolution: {color_intrinsics['width']}x{color_intrinsics['height']}")
    print(f"Depth resolution: {depth_intrinsics['width']}x{depth_intrinsics['height']}")
    print(f"Total frames: {frame_info['total_frames']}")

    return output_path


def load_config_defaults(scene_config_path=None):
    """Load resize settings with priority: scene config > base config > hardcoded defaults

    Args:
        scene_config_path: Path to existing scene-specific config file (optional)

    Returns:
        dict: Default resize and resolution settings
    """
    defaults = {"resize": "redepth", "Re_resolution": {"W": 640, "H": 480}, "crop_edge": 150}

    # Load base config defaults
    base_config_path = Path("configs/AzureKinect/azure_kinect.yaml")
    if base_config_path.exists():
        with open(base_config_path, 'r') as f:
            base_config = yaml.safe_load(f)
            cam_config = base_config.get("cam", {})
            if "resize" in cam_config:
                defaults["resize"] = cam_config["resize"]
            if "Re_resolution" in cam_config:
                defaults["Re_resolution"] = cam_config["Re_resolution"]
            if "crop_edge" in cam_config:
                defaults["crop_edge"] = cam_config["crop_edge"]

    # Load scene-specific config defaults (higher priority)
    if scene_config_path and Path(scene_config_path).exists():
        with open(scene_config_path, 'r') as f:
            scene_config = yaml.safe_load(f)
            cam_config = scene_config.get("cam", {})
            if "resize" in cam_config:
                defaults["resize"] = cam_config["resize"]
            if "Re_resolution" in cam_config:
                defaults["Re_resolution"] = cam_config["Re_resolution"]
            if "crop_edge" in cam_config:
                defaults["crop_edge"] = cam_config["crop_edge"]

    return defaults


def create_base_config():
    """Create base Azure Kinect configuration file"""
    base_config = {
        "project_name": "LoopSplat_SLAM_azure_kinect",
        "dataset_name": "azure_kinect",
        "checkpoint_path": None,
        "use_wandb": True,
        "frame_limit": -1,
        "seed": 0,
        "mapping": {
            "new_submap_every": 50,
            "map_every": 1,
            "iterations": 100,
            "new_submap_iterations": 100,
            "new_submap_points_num": 100000,
            "new_submap_gradient_points_num": 50000,
            "new_frame_sample_size": 30000,
            "new_points_radius": 0.0001,
            "current_view_opt_iterations": 0.4,
            "alpha_thre": 0.6,
            "pruning_thre": 0.5,
            "submap_using_motion_heuristic": True
        },
        "tracking": {
            "gt_camera": False,
            "w_color_loss": 0.6,
            "iterations": 200,
            "cam_rot_lr": 0.002,
            "cam_trans_lr": 0.01,
            "odometry_type": "const_speed",
            "help_camera_initialization": False,
            "init_err_ratio": 5,
            "odometer_method": "hybrid",
            "filter_alpha": False,
            "filter_outlier_depth": False,
            "alpha_thre": 0.98,
            "soft_alpha": True,
            "mask_invalid_depth": True,
            "enable_exposure": False
        },
        "cam": {
            "crop_edge": 150,  # Crop edges to remove distortion artifacts (150px for 1920x1080)
            "depth_scale": 1000.0  # Default, will be overridden by specific configs
        },
        "lc": {
            "min_similarity": 0.5,
            "pgo_edge_prune_thres": 0.25,
            "voxel_size": 0.02,
            "pgo_max_iterations": 500,
            "registration": {
                "method": "gs_reg",
                "base_lr": 5e-3,
                "min_overlap_ratio": 0.2,
                "use_render": False
            },
            "min_interval": 3,
            "final": False
        }
    }
    
    # Create base config file
    base_config_path = Path("configs/AzureKinect/azure_kinect.yaml")
    base_config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(base_config_path, 'w') as f:
        yaml.dump(base_config, f, default_flow_style=False, indent=2)
    
    print(f"Created base configuration: {base_config_path}")
    return base_config_path


def main():
    parser = argparse.ArgumentParser(description="Generate Azure Kinect configuration files")
    parser.add_argument("data_path", help="Path to Azure Kinect data directory")
    parser.add_argument("--scene-name", help="Scene name (default: directory name)")
    parser.add_argument("--output", help="Output configuration file path")
    parser.add_argument("--resize", choices=["rergb", "redepth", "reall"], default=None,
                       help="Resize strategy: rergb (resize RGB to depth), redepth (resize depth to RGB), reall (resize both to custom resolution). Default: from base config")
    parser.add_argument("--resolution", help="Custom resolution for 'all' mode (format: WIDTHxHEIGHT, e.g., 640x480)")
    parser.add_argument("--create-base", action="store_true",
                       help="Create base configuration file")

    args = parser.parse_args()

    # Parse custom resolution if provided
    custom_resolution = None
    if args.resolution:
        try:
            width, height = map(int, args.resolution.split('x'))
            custom_resolution = (width, height)
        except ValueError:
            print(f"Invalid resolution format: {args.resolution}. Use WIDTHxHEIGHT (e.g., 640x480)")
            return 1

    # Validate resolution parameter for 'reall' mode
    if args.resize == "reall" and custom_resolution is None:
        print("Warning: Using default resolution 640x480 for 'reall' mode. Use --resolution to specify custom resolution.")

    # Create base configuration if requested
    if args.create_base:
        create_base_config()

    # Generate scene-specific configuration
    try:
        generate_config(args.data_path, args.scene_name, args.output, args.resize, custom_resolution)
    except Exception as e:
        print(f"Error generating configuration: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
