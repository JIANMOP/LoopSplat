import copy

import numpy as np
import pytest

from src.entities.mapper import Mapper


def make_mapper_for_support_test():
    mapper = Mapper.__new__(Mapper)
    mapper.keyframes = [(0, {"persistent": True})]
    mapper._pyramid_step_counts = {0: 7}
    mapper._pyramid_level_usage = {0: {0: 3, 1: 4}}
    mapper._pyramid_lifetime_level_usage = {0: {0: 3, 1: 4}}
    return mapper


def test_support_update_uses_only_transient_full_resolution_view():
    mapper = make_mapper_for_support_test()
    original_keyframes = list(mapper.keyframes)
    original_step_counts = dict(mapper._pyramid_step_counts)
    original_usage = copy.deepcopy(mapper._pyramid_level_usage)
    original_lifetime_usage = copy.deepcopy(
        mapper._pyramid_lifetime_level_usage)
    calls = {}
    transient_keyframe = {"transient": True}

    def build_keyframe(frame_id, estimate_c2w, exposure_ab=None,
                       include_pyramid=True):
        calls["build"] = {
            "frame_id": frame_id,
            "estimate_c2w": estimate_c2w,
            "exposure_ab": exposure_ab,
            "include_pyramid": include_pyramid,
        }
        return None, None, transient_keyframe

    def optimize_submap(keyframes, gaussian_model, iterations=100,
                        use_pyramid=True, prune=True):
        calls["optimize"] = {
            "keyframes": keyframes,
            "gaussian_model": gaussian_model,
            "iterations": iterations,
            "use_pyramid": use_pyramid,
            "prune": prune,
        }
        return {"optimization_time": 0.25}

    def forbidden_call(*args, **kwargs):
        raise AssertionError("support update must not seed or grow Gaussians")

    mapper._build_keyframe = build_keyframe
    mapper.optimize_submap = optimize_submap
    mapper.compute_seeding_mask = forbidden_call
    mapper.seed_new_gaussians = forbidden_call
    mapper.grow_submap = forbidden_call
    gaussian_model = object()
    estimate_c2w = np.eye(4)
    exposure_ab = object()

    result = mapper.support_update(
        frame_id=2,
        estimate_c2w=estimate_c2w,
        gaussian_model=gaussian_model,
        iterations=20,
        exposure_ab=exposure_ab,
    )

    assert result == {"optimization_time": 0.25}
    assert calls["build"] == {
        "frame_id": 2,
        "estimate_c2w": estimate_c2w,
        "exposure_ab": exposure_ab,
        "include_pyramid": False,
    }
    assert calls["optimize"] == {
        "keyframes": [(2, transient_keyframe)],
        "gaussian_model": gaussian_model,
        "iterations": 20,
        "use_pyramid": False,
        "prune": False,
    }
    assert mapper.keyframes == original_keyframes
    assert mapper._pyramid_step_counts == original_step_counts
    assert mapper._pyramid_level_usage == original_usage
    assert mapper._pyramid_lifetime_level_usage == original_lifetime_usage


@pytest.mark.parametrize("iterations", [0, -1, 1.5, True])
def test_support_update_rejects_invalid_iteration_budget(iterations):
    mapper = make_mapper_for_support_test()

    with pytest.raises(
            ValueError, match="support update iterations must be a positive"):
        mapper.support_update(
            frame_id=2,
            estimate_c2w=np.eye(4),
            gaussian_model=object(),
            iterations=iterations,
        )
