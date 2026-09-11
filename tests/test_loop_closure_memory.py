import numpy as np
import pytest
import torch
from types import SimpleNamespace

import src.entities.lc as loop_closure_module
import src.gsr.solver as solver_module
from src.entities.lc import Loop_closure
from src.gsr.camera import Camera


def replica_gradient_config():
    return {
        "Training": {"edge_threshold": 4.0},
        "Dataset": {"type": "replica"},
    }


def make_camera(rgb, cuda_device):
    height = rgb.shape[-2] if rgb.ndim >= 2 else 32
    width = rgb.shape[-1] if rgb.ndim >= 2 else 32
    return Camera(
        0,
        rgb,
        np.ones((height, width), dtype=np.float32),
        torch.eye(4, device=cuda_device),
        torch.eye(4, device=cuda_device),
        1.0,
        1.0,
        0.5,
        0.5,
        1.0,
        1.0,
        height,
        width,
        device=str(cuda_device),
    )


def test_camera_materializes_cpu_uint8_rgb_on_cuda(cuda_device):
    rgb = torch.zeros((3, 32, 32), dtype=torch.uint8)
    rgb[:, 0, :2] = torch.tensor(
        [[0, 255], [64, 128], [255, 0]], dtype=torch.uint8)
    camera = make_camera(rgb, cuda_device)
    camera.config = replica_gradient_config()

    camera.load_rgb()

    assert camera.original_image.is_cuda
    assert camera.original_image.dtype == torch.float32
    torch.testing.assert_close(
        camera.original_image.cpu(), rgb.float() / 255.0)
    assert camera.grad_mask.is_cuda


def test_camera_rejects_invalid_cpu_rgb_shape(cuda_device):
    camera = make_camera(torch.zeros((2, 2), dtype=torch.uint8), cuda_device)
    camera.config = replica_gradient_config()

    with pytest.raises(ValueError, match="RGB observation"):
        camera.load_rgb()


def make_loop_closer(cuda_device):
    loop_closer = Loop_closure.__new__(Loop_closure)
    loop_closer.device = str(cuda_device)
    loop_closer.config = replica_gradient_config()
    loop_closer.proj_matrix = torch.eye(4, device=cuda_device)
    loop_closer.dataset = SimpleNamespace(
        intrinsics=np.eye(3),
        fovx=1.0,
        fovy=1.0,
        height=4,
        width=5,
    )
    return loop_closer


def test_loop_camera_keeps_array_observation_on_cpu(cuda_device):
    loop_closer = make_loop_closer(cuda_device)
    rgb = np.arange(3 * 4 * 5, dtype=np.uint8).reshape(4, 5, 3)
    depth = np.ones((4, 5), dtype=np.float32)

    camera = loop_closer._make_camera(
        0, np.eye(4, dtype=np.float32), torch.eye(4), depth, rgb)

    assert camera.original_image.device.type == "cpu"
    assert camera.original_image.dtype == torch.uint8
    assert camera.original_image.shape == (3, 4, 5)
    assert camera.grad_mask is None


def test_pose_only_camera_copy_drops_observations_and_isolates_pose(
        cuda_device):
    camera = make_camera(
        torch.zeros((3, 2, 2), dtype=torch.uint8), cuda_device)
    camera.depth = np.ones((2, 2), dtype=np.float32)
    camera.grad_mask = torch.ones((1, 2, 2), device=cuda_device)

    clone = loop_closure_module.pose_only_camera_copy(camera)
    clone.update_RT(
        torch.eye(3, device=cuda_device),
        torch.ones(3, device=cuda_device),
    )

    assert clone.original_image is None
    assert clone.depth is None
    assert clone.grad_mask is None
    assert not torch.equal(clone.T, camera.T)


class CopyTrackedCamera:
    def __init__(self, uid):
        self.uid = uid
        self.copy_count = 0

    def __deepcopy__(self, memo):
        self.copy_count += 1
        return CopyTrackedCamera(self.uid)


def tracked_submap(descriptors, prefix):
    return {
        "kf_desc": torch.tensor(descriptors, dtype=torch.float32),
        "cameras": [
            CopyTrackedCamera(f"{prefix}{index}")
            for index in range(len(descriptors))
        ],
    }


def test_registration_view_selection_preserves_topk_and_limits_copies():
    source = tracked_submap(
        [[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]], "s")
    target = tracked_submap(
        [[1.0, 0.0], [0.0, 0.9], [0.7, 0.7]], "t")

    src_views, tgt_views = solver_module.select_registration_views(
        source, target, device="cpu")

    assert [camera.uid for camera in src_views] == ["s0", "s1"]
    assert [camera.uid for camera in tgt_views] == ["t0", "t1"]
    assert sum(camera.copy_count for camera in source["cameras"]) == 2
    assert sum(camera.copy_count for camera in target["cameras"]) == 2
