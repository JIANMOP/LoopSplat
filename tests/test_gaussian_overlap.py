import pytest
import torch

from src.gsr.overlap import compute_overlap_gaussians


class GaussianPositions:
    def __init__(self, positions):
        self.positions = positions

    def get_xyz(self):
        return self.positions


def test_gaussian_overlap_compares_faiss_squared_distances(cuda_device):
    source = GaussianPositions(torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.float32,
        device=cuda_device,
    ))
    target = GaussianPositions(torch.tensor(
        [[0.2, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=torch.float32,
        device=cuda_device,
    ))

    overlap = compute_overlap_gaussians(source, target, threshold=0.1)

    assert float(overlap) == pytest.approx(0.0)
