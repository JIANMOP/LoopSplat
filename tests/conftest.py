import pytest
import torch


@pytest.fixture(scope="session")
def cuda_device():
    assert torch.cuda.is_available(), "publication tests require CUDA"
    return torch.device("cuda")
