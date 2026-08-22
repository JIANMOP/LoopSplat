import json

import numpy as np
import pytest
import torch

from src.evaluation.evaluator import Evaluator
from src.evaluation.protocol import (
    assert_compatible_protocols,
    assign_frames_to_submaps,
    build_evaluation_frame_ids,
    masked_depth_l1,
    trajectory_status,
)


def test_evaluation_ids_are_fixed_sorted_unique_and_include_last():
    assert build_evaluation_frame_ids(10, 3) == [0, 3, 6, 9]
    assert build_evaluation_frame_ids(11, 3) == [0, 3, 6, 9, 10]


def test_evaluation_stride_must_be_positive():
    with pytest.raises(ValueError, match="stride"):
        build_evaluation_frame_ids(10, 0)


def test_no_gt_dataset_skips_trajectory_files(tmp_path):
    dataset = type("Dataset", (), {"has_ground_truth": False})()
    evaluator = Evaluator.__new__(Evaluator)
    evaluator.dataset = dataset
    evaluator.checkpoint_path = tmp_path
    evaluator.estimated_c2w = np.repeat(np.eye(4)[None], 2, axis=0)
    evaluator.gt_poses = None

    evaluator.run_trajectory_eval()

    assert trajectory_status(dataset) == "skipped_no_ground_truth"
    assert not (tmp_path / "ate.json").exists()
    assert not (tmp_path / "ate_aligned.json").exists()
    status = json.loads((tmp_path / "trajectory_status.json").read_text())
    assert status["status"] == "skipped_no_ground_truth"


def test_depth_metric_ignores_invalid_ground_truth_pixels():
    rendered = torch.tensor([[100.0, 2.5], [4.0, 6.0]])
    ground_truth = torch.tensor([[0.0, 2.0], [float("nan"), 5.0]])

    value, valid_count = masked_depth_l1(rendered, ground_truth)

    assert value.item() == pytest.approx(0.75)
    assert valid_count == 2


def test_fixed_frames_are_assigned_once_to_temporal_submap_ranges():
    assignments = assign_frames_to_submaps(
        frame_ids=[0, 3, 6, 9],
        submap_keyframe_ids=[[0, 2], [5, 8]],
    )

    assert assignments == {0: [0, 3], 1: [6, 9]}
    assert sorted(sum(assignments.values(), [])) == [0, 3, 6, 9]


def test_aggregator_rejects_mixed_formal_protocols():
    with pytest.raises(ValueError, match="protocol"):
        assert_compatible_protocols([
            {"frame_ids": [0, 5, 9], "global_refinement_iterations": 0},
            {"frame_ids": [0, 5, 9], "global_refinement_iterations": 100},
        ])
