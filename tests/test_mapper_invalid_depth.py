import numpy as np
import pytest

from src.entities.mapper import Mapper


def test_mapper_rejects_frame_without_finite_positive_depth():
    mapper = Mapper.__new__(Mapper)
    mapper.dataset = type("Dataset", (), {
        "__getitem__": lambda self, index: (
            index,
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.zeros((2, 2), dtype=np.float32),
            np.eye(4),
        ),
    })()

    with pytest.raises(ValueError, match="valid depth"):
        mapper.map(0, np.eye(4), object(), True)
