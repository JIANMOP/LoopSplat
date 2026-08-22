from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.run_ablation import EXPERIMENTS, GSR_MAX_ITERS
from src.entities.gaussian_slam import GaussianSLAM
from src.entities.lc import Loop_closure
import src.entities.lc as loop_closure_module
from src.gsr.solver import validate_gsr_max_iters


def test_cache_invalidation_removes_only_corrected_submaps():
    loop_closer = Loop_closure.__new__(Loop_closure)
    loop_closer._sm_cache = {0: {"value": "old"}, 1: {"value": "keep"}}

    loop_closer.invalidate_submap_cache([0])

    assert 0 not in loop_closer._sm_cache
    assert loop_closer._sm_cache[1] == {"value": "keep"}


def test_pgo_checkpoint_update_invalidates_cached_gaussians(tmp_path):
    submap_dir = tmp_path / "submaps"
    submap_dir.mkdir()
    checkpoint = {
        "submap_keyframes": [0],
        "gaussian_params": {
            "xyz": torch.tensor([[1.0, 0.0, 0.0]]),
            "rotation": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        },
    }
    torch.save(checkpoint, submap_dir / "000000.ckpt")

    loop_closer = Loop_closure.__new__(Loop_closure)
    loop_closer._sm_cache = {0: {"gaussian_params": checkpoint["gaussian_params"]}}
    slam = GaussianSLAM.__new__(GaussianSLAM)
    slam.output_path = Path(tmp_path)
    slam.loop_closer = loop_closer
    correction = np.eye(4)
    correction[0, 3] = 2.0

    slam.apply_correction_to_submaps([
        {"submap_id": 0, "correct_tsfm": correction},
    ])

    assert 0 not in loop_closer._sm_cache
    reloaded = torch.load(submap_dir / "000000.ckpt")
    torch.testing.assert_close(
        reloaded["gaussian_params"]["xyz"],
        torch.tensor([[3.0, 0.0, 0.0]]),
    )


def test_loop_closure_does_not_mutate_experiment_config(monkeypatch):
    monkeypatch.setattr(loop_closure_module, "GlobalDesc", lambda: object())
    config = {
        "lc": {
            "min_interval": 5,
            "voxel_size": 0.02,
            "registration": {"gsr_max_iters": 100},
        },
    }
    original = deepcopy(config)
    dataset = SimpleNamespace(
        width=640,
        height=480,
        intrinsics=np.eye(3),
    )

    loop_closer = Loop_closure(config, dataset, logger=None)

    assert config == original
    assert loop_closer.config["Training"] == {"edge_threshold": 4.0}
    assert loop_closer.config["Dataset"] == {"type": "replica"}


@pytest.mark.parametrize("value", [0, -1, 1.5, True, None])
def test_gsr_iterations_must_be_a_positive_integer(value):
    with pytest.raises((TypeError, ValueError, KeyError)):
        validate_gsr_max_iters({"gsr_max_iters": value})


def test_gsr_iterations_are_required_and_uniform_in_ablation_runner():
    with pytest.raises(KeyError):
        validate_gsr_max_iters({})
    assert validate_gsr_max_iters(
        {"gsr_max_iters": GSR_MAX_ITERS}) == GSR_MAX_ITERS
    assert {
        experiment["overrides"]["lc"]["registration"]["gsr_max_iters"]
        for experiment in EXPERIMENTS
    } == {GSR_MAX_ITERS}
