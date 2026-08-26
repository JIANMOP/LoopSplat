""" This module includes the Odometer class, which is allows for fast pose estimation from RGBD neighbor frames  """
import numpy as np
import open3d as o3d
import open3d.core as o3c


class VisualOdometer(object):

    def __init__(self, intrinsics: np.ndarray, method_name="hybrid",
                 device="cuda", cpu_fallback=True,
                 max_translation_m=None, max_rotation_deg=None):
        """ Initializes the visual odometry system with specified intrinsics, method, and device.
        Args:
            intrinsics: Camera intrinsic parameters.
            method_name: The name of the odometry computation method to use ('hybrid' or 'point_to_plane').
            device: The computation device ('cuda' or 'cpu').
        """
        device_name = "CUDA:0" if device == "cuda" else "CPU:0"
        self.device = o3c.Device(device_name)
        self.cpu_device = (
            o3c.Device("CPU:0")
            if cpu_fallback and device == "cuda" else None)
        self.intrinsics = o3d.core.Tensor(intrinsics, o3d.core.Dtype.Float64)
        self.last_abs_pose = None
        self.last_frame = None
        self.criteria_list = [
            o3d.t.pipelines.odometry.OdometryConvergenceCriteria(500),
            o3d.t.pipelines.odometry.OdometryConvergenceCriteria(500),
            o3d.t.pipelines.odometry.OdometryConvergenceCriteria(500)]
        self.setup_method(method_name)
        self.max_depth = 10.0
        self.depth_scale = 1.0
        self.last_rgbd = None
        self.cpu_last_rgbd = None
        self.max_translation_m = self._positive_limit(
            max_translation_m, "max_translation_m")
        self.max_rotation_deg = self._positive_limit(
            max_rotation_deg, "max_rotation_deg")
        self.last_diagnostic = None
        self._estimate_count = 0
        self._cpu_fallback_count = 0
        self._identity_fallback_count = 0
        self._reason_counts = {}

    @staticmethod
    def _positive_limit(value, name):
        if value is None:
            return float("inf")
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite value")
        return value

    @staticmethod
    def _make_rgbd(image, depth, device):
        return o3d.t.geometry.RGBDImage(
            o3d.t.geometry.Image(np.ascontiguousarray(
                image).astype(np.float32)).to(device),
            o3d.t.geometry.Image(np.ascontiguousarray(
                depth).astype(np.float32)).to(device))

    def setup_method(self, method_name: str) -> None:
        """ Sets up the odometry computation method based on the provided method name.
        Args:
            method_name: The name of the odometry method to use ('hybrid' or 'point_to_plane').
        """
        if method_name == "hybrid":
            self.method = o3d.t.pipelines.odometry.Method.Hybrid
        elif method_name == "point_to_plane":
            self.method = o3d.t.pipelines.odometry.Method.PointToPlane
        else:
            raise ValueError("Odometry method does not exist!")

    def update_last_rgbd(self, image: np.ndarray, depth: np.ndarray) -> None:
        """ Updates the last RGB-D frame stored in the system with a new RGB-D frame constructed from provided image and depth.
        Args:
            image: The new RGB image as a numpy ndarray.
            depth: The new depth image as a numpy ndarray.
        """
        self.last_rgbd = self._make_rgbd(image, depth, self.device)
        if self.cpu_device is not None:
            self.cpu_last_rgbd = self._make_rgbd(
                image, depth, self.cpu_device)

    def _solve(self, source, target, init_transform):
        return o3d.t.pipelines.odometry.rgbd_odometry_multi_scale(
            source, target, self.intrinsics, o3c.Tensor(init_transform),
            self.depth_scale, self.max_depth, self.criteria_list,
            self.method)

    @staticmethod
    def _convert_transform(result):
        transform = result.transformation.cpu().numpy()
        transform[0, [1, 2, 3]] *= -1
        transform[1, [0, 2, 3]] *= -1
        transform[2, [0, 1, 3]] *= -1
        return transform

    def _validate_transform(self, transform):
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            return "invalid_transform", None, None
        translation_m = float(np.linalg.norm(transform[:3, 3]))
        cosine = np.clip(
            (np.trace(transform[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
        rotation_deg = float(np.degrees(np.arccos(cosine)))
        if (translation_m > self.max_translation_m
                or rotation_deg > self.max_rotation_deg):
            return "motion_outlier", translation_m, rotation_deg
        return None, translation_m, rotation_deg

    def _record_diagnostic(self, source, reason, translation_m,
                           rotation_deg):
        self.last_diagnostic = {
            "source": source,
            "reason": reason,
            "translation_m": translation_m,
            "rotation_deg": rotation_deg,
        }
        if source == "cpu_fallback":
            self._cpu_fallback_count += 1
        elif source == "identity_fallback":
            self._identity_fallback_count += 1
        if reason:
            self._reason_counts[reason] = self._reason_counts.get(reason, 0) + 1

    def diagnostics(self):
        return {
            "estimate_count": self._estimate_count,
            "cpu_fallback_count": self._cpu_fallback_count,
            "identity_fallback_count": self._identity_fallback_count,
            "reason_counts": dict(self._reason_counts),
        }

    def estimate_rel_pose(self, image: np.ndarray, depth: np.ndarray, init_transform=np.eye(4)):
        """ Estimates the relative pose of the current frame with respect to the last frame using RGB-D odometry.
        Args:
            image: The current RGB image as a numpy ndarray.
            depth: The current depth image as a numpy ndarray.
            init_transform: An initial transformation guess as a numpy ndarray. Defaults to the identity matrix.
        Returns:
            The relative transformation matrix as a numpy ndarray.
        """
        rgbd = self._make_rgbd(image, depth, self.device)
        cpu_rgbd = (
            self._make_rgbd(image, depth, self.cpu_device)
            if self.cpu_device is not None else None)
        self._estimate_count += 1
        primary_reason = None
        try:
            transform = self._convert_transform(
                self._solve(self.last_rgbd, rgbd, init_transform))
            invalid_reason, translation_m, rotation_deg = (
                self._validate_transform(transform))
            if invalid_reason is None:
                self._record_diagnostic(
                    "primary", "", translation_m, rotation_deg)
                if cpu_rgbd is not None:
                    self.cpu_last_rgbd = cpu_rgbd.clone()
                return transform
            primary_reason = f"primary_{invalid_reason}"
        except RuntimeError:
            primary_reason = "primary_solver_error"
        finally:
            self.last_rgbd = rgbd.clone()

        if cpu_rgbd is not None and self.cpu_last_rgbd is not None:
            try:
                transform = self._convert_transform(
                    self._solve(
                        self.cpu_last_rgbd, cpu_rgbd, init_transform))
                invalid_reason, translation_m, rotation_deg = (
                    self._validate_transform(transform))
                if invalid_reason is None:
                    self._record_diagnostic(
                        "cpu_fallback", primary_reason,
                        translation_m, rotation_deg)
                    print(
                        "Visual odometry CPU fallback: "
                        f"{primary_reason}")
                    return transform
                final_reason = f"cpu_{invalid_reason}"
            except RuntimeError:
                final_reason = "cpu_solver_error"
            finally:
                self.cpu_last_rgbd = cpu_rgbd.clone()
        else:
            final_reason = primary_reason

        self._record_diagnostic(
            "identity_fallback", final_reason, 0.0, 0.0)
        print(f"Visual odometry identity fallback: {final_reason}")
        return np.eye(4, dtype=np.float64)
