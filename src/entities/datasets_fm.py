"""
FM Dataset loader for LoopSplat.

Supports the FM RGB-D+IMU dataset format:
    scene_dir/
      color/           ← RGB images (640×480 PNG)
      depth/           ← depth maps (640×480 PNG, uint16)
      filtered/        ← filtered depth (optional, not used by default)
      IMU.txt          ← #timestamp(us) gx gy gz ax ay az
      TIMESTAMP.txt    ← #timestamp(us) colorname depthname

No ground-truth camera poses are available; the dataset relies on
IMU + visual odometry for tracking.
"""

import math
from pathlib import Path

import cv2
import numpy as np


from src.entities.datasets import BaseDataset
from src.entities.imu_types import build_imu_interval
from src.utils.rgbd_registration import register_depth_to_color


class FMDataset(BaseDataset):
    """FM RGB-D + IMU dataset."""

    def __init__(self, dataset_config: dict):
        # Call BaseDataset first (sets up width/height/intrinsics/fov)
        super().__init__(dataset_config)
        self.has_ground_truth = False

        self.dataset_path = Path(dataset_config["input_path"])
        self.use_filtered_depth = dataset_config.get("use_filtered_depth", False)
        self.register_depth_to_color = dataset_config.get(
            "register_depth_to_color", False)
        self.max_malformed_imu_rows = int(
            dataset_config.get("max_malformed_imu_rows", 0))
        if self.max_malformed_imu_rows < 0:
            raise ValueError("max_malformed_imu_rows must be non-negative")
        self.imu_rows_dropped = 0
        if self.register_depth_to_color:
            try:
                self.depth_intrinsics = np.asarray(
                    dataset_config["depth_intrinsics"], dtype=np.float64)
                self.t_color_depth = np.asarray(
                    dataset_config["T_color_depth"], dtype=np.float64)
            except KeyError as error:
                raise ValueError(
                    "registered FM depth requires depth_intrinsics and "
                    "T_color_depth") from error

        # Load timestamp → image mapping
        self.timestamps = []
        self._load_timestamps()

        # Load IMU data if available
        self.has_imu = False
        self.imu_data = []
        self._load_imu()

        # Build color/depth path lists matching timestamps
        self.color_paths = []
        self.depth_paths = []
        depth_subdir = "filtered" if self.use_filtered_depth else "depth"
        for _, color_name, depth_name in self._ts_mapping:
            self.color_paths.append(self.dataset_path / "color" / color_name)
            self.depth_paths.append(self.dataset_path / depth_subdir / depth_name)

        # No GT poses — store identity as dummy
        self.poses = [np.eye(4, dtype=np.float32) for _ in range(len(self))]

        n_frames = len(self)
        print(f"FM Dataset loaded: {n_frames} frames"
              + (f", IMU available ({len(self.imu_data)} samples)" if self.has_imu else ""))

    # ── TIMESTAMP parsing ──────────────────────────────────────────

    def _load_timestamps(self):
        ts_path = self.dataset_path / "TIMESTAMP.txt"
        if not ts_path.exists():
            raise FileNotFoundError(f"TIMESTAMP.txt not found at {ts_path}")

        self._ts_mapping = []  # (timestamp_s, color_name, depth_name)
        with open(ts_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(',')
                if len(parts) >= 3:
                    ts = float(parts[0]) * 1e-6
                    color_name = parts[1].strip()
                    depth_name = parts[2].strip()
                    self._ts_mapping.append((ts, color_name, depth_name))
                    self.timestamps.append(ts)

        print(f"  Loaded {len(self._ts_mapping)} timestamp entries")

    # ── IMU parsing ─────────────────────────────────────────────────

    def _load_imu(self):
        imu_path = self.dataset_path / "IMU.txt"
        if not imu_path.exists():
            print(f"  IMU file not found: {imu_path}")
            return

        with open(imu_path) as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                # Format: timestamp(us), gx, gy, gz, ax, ay, az
                parts = (
                    [part.strip() for part in line.split(',')]
                    if ',' in line else line.split())
                if len(parts) < 7:
                    self.imu_rows_dropped += 1
                    if self.imu_rows_dropped > self.max_malformed_imu_rows:
                        raise ValueError(
                            f"invalid IMU row {line_number}: expected 7 fields")
                    continue
                values = np.asarray(
                    [float(value) for value in parts[:7]],
                    dtype=np.float64)
                if not np.isfinite(values).all():
                    raise ValueError(
                        f"invalid IMU row {line_number}: non-finite value")
                self.imu_data.append({
                    'timestamp': values[0] * 1e-6,
                    'acceleration': values[4:7].tolist(),
                    'angular_velocity': values[1:4].tolist(),
                })

        if len(self.imu_data) < 2:
            raise ValueError("IMU.txt must contain at least two valid samples")
        imu_times = np.asarray([
            sample['timestamp'] for sample in self.imu_data])
        if np.any(np.diff(imu_times) <= 0):
            raise ValueError("IMU timestamps must be strictly increasing")
        self.has_imu = True
        print(
            f"  Loaded {len(self.imu_data)} IMU samples "
            f"(dropped {self.imu_rows_dropped} explicitly allowed rows)")

    def get_imu_data_for_frame(self, frame_id: int):
        """Return IMU sample closest in time to frame_id's timestamp."""
        if not self.has_imu or frame_id >= len(self.timestamps):
            return None
        frame_ts = self.timestamps[frame_id]
        closest = min(self.imu_data, key=lambda x: abs(x['timestamp'] - frame_ts))
        return closest

    def get_imu_measurements(self, start_frame_id: int, end_frame_id: int,
                             time_offset_s=0.0):
        return build_imu_interval(
            self.timestamps, self.imu_data, start_frame_id, end_frame_id,
            time_offset_s=time_offset_s)

    # ── Dataset interface ───────────────────────────────────────────

    def __len__(self):
        n = len(self._ts_mapping)
        if self.frame_limit > 0 and self.frame_limit < n:
            return self.frame_limit
        return n

    def __getitem__(self, index):
        color_data = cv2.imread(str(self.color_paths[index]))
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)

        depth_data = cv2.imread(
            str(self.depth_paths[index]), cv2.IMREAD_UNCHANGED)
        depth_data = depth_data.astype(np.float32) / self.depth_scale

        if self.register_depth_to_color:
            depth_data = register_depth_to_color(
                depth_m=depth_data,
                depth_intrinsics=self.depth_intrinsics,
                color_intrinsics=self.intrinsics,
                t_color_depth=self.t_color_depth,
                output_shape=color_data.shape[:2],
            )

        # No distortion / crop for FM dataset
        return index, color_data, depth_data, self.poses[index]
