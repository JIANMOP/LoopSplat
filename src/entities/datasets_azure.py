import json
import math
import os
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

from .datasets import BaseDataset


class AzureKinect(BaseDataset):
    """Azure Kinect dataset loader that supports dynamic resolution and camera parameters

    Features:
    - Dynamic resize strategies (rergb, redepth, reall)
    - Distortion correction for color images
    - Edge cropping to remove distortion artifacts (inherited from BaseDataset)
    - Automatic intrinsic parameter adjustment for cropped images
    """
    
    def __init__(self, dataset_config: dict):
        self.dataset_path = Path(dataset_config["input_path"])

        # Extract resize strategy and camera parameters from config
        self.resize_mode = dataset_config.get("resize", "redepth")

        # Load original camera parameters from config
        self.color_camera = dataset_config["color_camera"]
        self.depth_camera = dataset_config["depth_camera"]

        # Set up resolution parameters based on resize mode
        if self.resize_mode == "reall":
            self.re_resolution = dataset_config["Re_resolution"]

        # K4A transformation configuration
        self.use_k4a_transformation = dataset_config.get("use_k4a_transformation", False)
        self.k4a_transformation_mode = dataset_config.get("k4a_transformation_mode", "depth_to_color")
        self.preprocessing_strategy = dataset_config.get("preprocessing_strategy", "resize_only")

        # Check K4A SDK availability if transformation is requested
        if self.use_k4a_transformation:
            try:
                import pyk4a
                self.k4a_available = True
                print("Azure Kinect SDK available for transformation")
            except ImportError:
                print("Warning: Azure Kinect SDK not available, falling back to resize")
                self.use_k4a_transformation = False
                self.k4a_available = False
        else:
            self.k4a_available = False

        super().__init__(dataset_config)

        # Load frame information
        self.load_frame_info()

        # Load IMU data if available
        self.load_imu_data()

        # Setup processed images directory
        self.processed_dir = self.dataset_path / "processed_images" / self.resize_mode
        self.processed_color_dir = self.processed_dir / "color"
        self.processed_depth_dir = self.processed_dir / "depth"

        # Create processed images if they don't exist
        self.ensure_processed_images()

        print(f"Loaded {len(self.color_paths)} Azure Kinect frames")
        print(f"Resize mode: {self.resize_mode}")
        print(f"Preprocessing strategy: {self.preprocessing_strategy}")
        print(f"K4A transformation: {'Enabled' if self.use_k4a_transformation else 'Disabled'}")
        print(f"Main resolution: {self.width}x{self.height}")
        print(f"Color camera: {self.color_camera['W']}x{self.color_camera['H']}")
        print(f"Depth camera: {self.depth_camera['W']}x{self.depth_camera['H']}")
        print(f"IMU data: {'Available' if self.has_imu else 'Not available'}")
        print(f"Processed images directory: {self.processed_dir}")
    

    def load_frame_info(self):
        """Load frame information from frame_info.json"""
        frame_info_path = self.dataset_path / "frame_info.json"
        
        if not frame_info_path.exists():
            raise FileNotFoundError(f"Frame info file not found: {frame_info_path}")
        
        with open(frame_info_path, 'r') as f:
            frame_info = json.load(f)
        
        self.total_frames = frame_info["total_frames"]
        frames = frame_info["frames"]
        
        # Build file paths
        self.color_paths = []
        self.depth_paths = []
        self.timestamps = []
        
        for frame in frames:
            # Convert Windows-style paths to Unix-style
            color_path = frame["color_path"].replace("\\", "/")
            depth_path = frame["depth_path"].replace("\\", "/")

            # Build full paths
            full_color_path = self.dataset_path / color_path
            full_depth_path = self.dataset_path / depth_path

            # Verify files exist
            if full_color_path.exists() and full_depth_path.exists():
                self.color_paths.append(str(full_color_path))
                self.depth_paths.append(str(full_depth_path))
                self.timestamps.append(frame["timestamp"])
        
        # Generate dummy poses (identity matrices) since Azure Kinect data doesn't include poses
        # In real SLAM, these will be estimated
        self.poses = []
        for i in range(len(self.color_paths)):
            pose = np.eye(4, dtype=np.float32)
            self.poses.append(pose)

    def load_imu_data(self):
        """Load IMU data from TC-MMS format imu.txt file"""
        imu_path = self.dataset_path / "imu.txt"

        if not imu_path.exists():
            print(f"IMU data file not found: {imu_path}")
            self.has_imu = False
            self.imu_data = []
            return

        try:
            self.imu_data = []
            with open(imu_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#') or not line:
                        continue

                    parts = line.split()
                    if len(parts) == 7:
                        timestamp = float(parts[0])
                        acc = [float(parts[1]), float(parts[2]), float(parts[3])]
                        gyro = [float(parts[4]), float(parts[5]), float(parts[6])]

                        self.imu_data.append({
                            'timestamp': timestamp,
                            'acceleration': acc,
                            'angular_velocity': gyro
                        })

            self.has_imu = len(self.imu_data) > 0
            if self.has_imu:
                print(f"Loaded {len(self.imu_data)} IMU samples")
            else:
                print("IMU file exists but contains no valid data")

        except Exception as e:
            print(f"Error loading IMU data: {e}")
            self.has_imu = False
            self.imu_data = []

    def get_imu_data_for_frame(self, frame_id):
        """Get IMU data for a specific frame by finding the closest timestamp"""
        if not self.has_imu or frame_id >= len(self.timestamps):
            return None

        frame_timestamp = self.timestamps[frame_id]

        # Find the closest IMU sample by timestamp
        closest_imu = min(self.imu_data,
                         key=lambda x: abs(x['timestamp'] - frame_timestamp))

        return closest_imu

    def apply_k4a_transformation(self, color_data, depth_data):
        """
        Apply Azure Kinect SDK transformation functions

        Note: This is a framework implementation. Full implementation would require
        access to the actual calibration data and K4A transformation functions.
        """
        if not self.k4a_available:
            print("K4A SDK not available, skipping transformation")
            return color_data, depth_data

        try:
            # This is a placeholder for actual K4A transformation
            # In a real implementation, you would:
            # 1. Load calibration data from camera_parameters.json
            # 2. Use k4a_transformation_depth_image_to_color_camera or
            #    k4a_transformation_color_image_to_depth_camera
            # 3. Handle edge cases and invalid regions

            if self.k4a_transformation_mode == "depth_to_color":
                # Transform depth to color camera perspective
                print("K4A transformation: depth_to_color (placeholder)")
                # transformed_depth = k4a_transformation_depth_image_to_color_camera(depth_data)
                return color_data, depth_data  # Placeholder
            else:
                # Transform color to depth camera perspective
                print("K4A transformation: color_to_depth (placeholder)")
                # transformed_color = k4a_transformation_color_image_to_depth_camera(color_data)
                return color_data, depth_data  # Placeholder

        except Exception as e:
            print(f"K4A transformation failed: {e}, falling back to original data")
            return color_data, depth_data

    def apply_preprocessing_strategy(self, color_data, depth_data):
        """Apply the configured preprocessing strategy"""
        if self.preprocessing_strategy == "resize_only":
            return self.apply_resize_strategy(color_data, depth_data)
        elif self.preprocessing_strategy == "k4a_only":
            # Check if K4A transformation is actually enabled
            if self.use_k4a_transformation:
                return self.apply_k4a_transformation(color_data, depth_data)
            else:
                print("Warning: preprocessing_strategy is 'k4a_only' but use_k4a_transformation is False")
                print("Falling back to resize_only strategy")
                return self.apply_resize_strategy(color_data, depth_data)
        elif self.preprocessing_strategy == "k4a_then_resize":
            # Hybrid approach: K4A transformation followed by resize
            if self.use_k4a_transformation:
                color_data, depth_data = self.apply_k4a_transformation(color_data, depth_data)
            else:
                print("Warning: preprocessing_strategy includes K4A but use_k4a_transformation is False")
                print("Skipping K4A transformation, using resize only")
            return self.apply_resize_strategy(color_data, depth_data)
        else:
            # Default to resize_only for unknown strategies
            print(f"Warning: Unknown preprocessing_strategy '{self.preprocessing_strategy}', using resize_only")
            return self.apply_resize_strategy(color_data, depth_data)
    
    def apply_resize_strategy(self, color_data, depth_data):
        """Apply the configured resize strategy to color and depth images"""
        if self.resize_mode == "rergb":
            # Resize RGB to depth resolution
            target_w = self.depth_camera["W"]
            target_h = self.depth_camera["H"]
            color_data = cv2.resize(color_data, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        elif self.resize_mode == "redepth":
            # Resize depth to RGB resolution
            target_w = self.color_camera["W"]
            target_h = self.color_camera["H"]
            depth_data = cv2.resize(depth_data, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        elif self.resize_mode == "reall":
            # Resize both to custom resolution
            target_w = self.re_resolution["W"]
            target_h = self.re_resolution["H"]
            color_data = cv2.resize(color_data, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            depth_data = cv2.resize(depth_data, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        return color_data, depth_data

    def apply_distortion_correction(self, color_data, depth_data):
        """Apply distortion correction based on resize mode"""
        # Apply color distortion correction if significant
        color_distortion = np.array(self.color_camera["distortion"])
        if np.any(np.abs(color_distortion) > 1e-6):
            if self.resize_mode == "rergb":
                # Use depth camera intrinsics for undistortion (RGB resized to depth)
                fx, fy = self.depth_camera["fx"], self.depth_camera["fy"]
                cx, cy = self.depth_camera["cx"], self.depth_camera["cy"]
            elif self.resize_mode == "redepth":
                # Use color camera intrinsics for undistortion
                fx, fy = self.color_camera["fx"], self.color_camera["fy"]
                cx, cy = self.color_camera["cx"], self.color_camera["cy"]
            else:  # "reall" mode
                # Use scaled color camera intrinsics
                fx, fy = self.fx, self.fy
                cx, cy = self.cx, self.cy

            intrinsics_matrix = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])
            color_data = cv2.undistort(color_data, intrinsics_matrix, color_distortion)

        # Apply depth distortion correction if significant (only for rergb mode)
        if self.resize_mode == "rergb":
            depth_distortion = np.array(self.depth_camera["distortion"])
            if np.any(np.abs(depth_distortion) > 1e-6):
                depth_intrinsics_matrix = np.array([
                    [self.depth_camera["fx"], 0, self.depth_camera["cx"]],
                    [0, self.depth_camera["fy"], self.depth_camera["cy"]],
                    [0, 0, 1]
                ])
                depth_data = cv2.undistort(depth_data, depth_intrinsics_matrix, depth_distortion)

        return color_data, depth_data

    def ensure_processed_images(self):
        """Ensure processed images exist for the current resize mode"""
        # Check if processed images already exist
        if self.processed_color_dir.exists() and self.processed_depth_dir.exists():
            # Check if all images are processed
            expected_count = len(self.color_paths)
            existing_color = len(list(self.processed_color_dir.glob("*.png")))
            existing_depth = len(list(self.processed_depth_dir.glob("*.png")))

            if existing_color == expected_count and existing_depth == expected_count:
                print(f"Processed images already exist for {self.resize_mode} mode")
                return

        print(f"Creating processed images for {self.resize_mode} mode...")
        self.create_processed_images()

    def create_processed_images(self):
        """Create and save processed images for the current resize mode"""
        # Create directories
        self.processed_color_dir.mkdir(parents=True, exist_ok=True)
        self.processed_depth_dir.mkdir(parents=True, exist_ok=True)

        # Process and save all images
        for i in tqdm(range(len(self.color_paths)), desc=f"Processing {self.resize_mode} images"):
            # Load original images
            color_data = cv2.imread(self.color_paths[i])
            color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)

            depth_data = cv2.imread(self.depth_paths[i], cv2.IMREAD_UNCHANGED)
            depth_data = depth_data.astype(np.float32) / self.depth_scale

            # Apply preprocessing strategy (resize, K4A transformation, or hybrid)
            color_resized, depth_resized = self.apply_preprocessing_strategy(color_data, depth_data)

            # Apply distortion correction
            color_final, depth_final = self.apply_distortion_correction(color_resized, depth_resized)

            # Apply depth truncation
            depth_trunc = self.dataset_config.get("depth_trunc", 4.0)
            depth_final[depth_final > depth_trunc] = 0

            # Save processed images
            color_filename = f"frame_{i:06d}.png"
            depth_filename = f"frame_{i:06d}.png"

            # Save color image (convert back to BGR for cv2)
            color_bgr = cv2.cvtColor(color_final, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(self.processed_color_dir / color_filename), color_bgr)

            # Save depth image (scale back and convert to uint16)
            depth_scaled = (depth_final * self.depth_scale).astype(np.uint16)
            cv2.imwrite(str(self.processed_depth_dir / depth_filename), depth_scaled)

        print(f"Processed images saved to {self.processed_dir}")

    def get_processed_image_paths(self, index):
        """Get paths to processed images for given index"""
        color_filename = f"frame_{index:06d}.png"
        depth_filename = f"frame_{index:06d}.png"

        color_path = self.processed_color_dir / color_filename
        depth_path = self.processed_depth_dir / depth_filename

        return str(color_path), str(depth_path)

    def __getitem__(self, index):
        """Get color image, depth image, and pose for given index

        Processing pipeline:
        1. Load preprocessed images (resized and distortion-corrected)
        2. Apply edge cropping to remove distortion artifacts
        3. Return processed data consistent with TUM dataset format
        """
        # Load processed images directly
        color_path, depth_path = self.get_processed_image_paths(index)

        # Load color image
        color_data = cv2.imread(color_path)
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)

        # Load depth image
        depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth_data = depth_data.astype(np.float32) / self.depth_scale

        # Apply crop to remove edge artifacts from distortion correction
        edge = self.crop_edge
        if edge > 0:
            color_data = color_data[edge:-edge, edge:-edge]
            depth_data = depth_data[edge:-edge, edge:-edge]

        return index, color_data, depth_data, self.poses[index]


def get_azure_dataset(dataset_name: str):
    """Get Azure Kinect dataset class"""
    if dataset_name == "azure_kinect":
        return AzureKinect
    raise NotImplementedError(f"Azure dataset {dataset_name} not implemented")
