import numpy as np
import pytest
import torch

from src.entities.mapper import Mapper
from src.utils.mapper_utils import (
    build_depth_pyramid,
    get_pyramid_render_settings,
)
from src.utils.utils import get_render_settings


def mapping_config():
    return {
        "iterations": 1,
        "new_submap_iterations": 1,
        "new_submap_points_num": 1,
        "new_submap_gradient_points_num": 1,
        "new_frame_sample_size": 1,
        "new_points_radius": 0.01,
        "alpha_thre": 0.5,
        "pruning_thre": 0.5,
        "current_view_opt_iterations": 0.5,
    }


def make_mapper():
    return Mapper(
        mapping_config(),
        {"enabled": True, "num_sub_levels": 2, "uses_per_level": 2},
        object(),
        object(),
    )


def test_zero_depth_does_not_dilute_valid_depth_after_downsampling(
        cuda_device):
    depth = torch.tensor(
        [[0.0, 0.0], [0.0, 2.0]], device=cuda_device)

    levels = build_depth_pyramid(depth, depth > 0, 1)
    low_depth, low_valid = levels[0]

    assert low_valid.item() is True
    assert low_depth.item() == pytest.approx(2.0)


def test_all_invalid_depth_stays_zero_and_invalid(cuda_device):
    depth = torch.zeros((2, 2), device=cuda_device)

    low_depth, low_valid = build_depth_pyramid(
        depth, depth > 0, 1)[0]

    assert low_valid.item() is False
    assert low_depth.item() == 0.0


def test_pyramid_schedule_uses_each_level_exactly_n_times():
    mapper = make_mapper()

    levels = [mapper.next_pyramid_level(7) for _ in range(6)]

    assert levels == [0, 0, 1, 1, 2, 2]
    assert mapper.pyramid_usage_summary()[7] == {0: 2, 1: 2, 2: 2}


def test_pyramid_state_reset_clears_frame_schedule():
    mapper = make_mapper()
    mapper.next_pyramid_level(7)

    mapper.reset_pyramid_state()

    assert mapper.pyramid_usage_summary() == {}
    assert mapper.next_pyramid_level(7) == 0


def test_real_raster_settings_preserve_camera_fields(cuda_device):
    settings = get_render_settings(
        640, 480,
        np.array([[500.0, 0.0, 320.0],
                  [0.0, 500.0, 240.0],
                  [0.0, 0.0, 1.0]]),
        np.eye(4),
    )

    low = get_pyramid_render_settings(settings, 160, 120)

    assert low.image_width == 160
    assert low.image_height == 120
    assert low._fields == settings._fields
    torch.testing.assert_close(low.campos, settings.campos)
    torch.testing.assert_close(low.projmatrix, settings.projmatrix)
