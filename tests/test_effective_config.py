import pytest

from src.entities.mapper import Mapper


def mapping_config():
    return {
        "iterations": 10,
        "new_submap_iterations": 20,
        "new_submap_points_num": 100,
        "new_submap_gradient_points_num": 50,
        "new_frame_sample_size": 30,
        "new_points_radius": 0.001,
        "alpha_thre": 0.6,
        "pruning_thre": 0.5,
        "current_view_opt_iterations": 0.4,
    }


def make_mapper(pyramid_config):
    return Mapper(mapping_config(), pyramid_config, object(), object())


def test_mapper_reports_requested_pyramid_configuration():
    mapper = make_mapper({
        "enabled": True,
        "num_sub_levels": 2,
        "uses_per_level": 8,
    })

    assert mapper.effective_pyramid_config() == {
        "enabled": True,
        "num_sub_levels": 2,
        "uses_per_level": 8,
    }


@pytest.mark.parametrize(("pyramid_config", "message"), [
    ({"enabled": 1, "num_sub_levels": 2, "uses_per_level": 8}, "enabled"),
    ({"enabled": True, "num_sub_levels": 0, "uses_per_level": 8}, "positive"),
    ({"enabled": True, "num_sub_levels": 2, "uses_per_level": 0}, "positive"),
])
def test_mapper_rejects_invalid_pyramid_configuration(pyramid_config, message):
    with pytest.raises((TypeError, ValueError), match=message):
        make_mapper(pyramid_config)
