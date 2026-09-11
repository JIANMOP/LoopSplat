import numpy as np
import pytest
import torch

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
