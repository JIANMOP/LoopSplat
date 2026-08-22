import json

import numpy as np
import pytest
import torch

from scripts.aggregate_results import read_trajectory_metrics
from src.evaluation.evaluator import Evaluator
from src.evaluation.evaluate_trajectory import compute_relative_pose_errors
from src.evaluation.protocol import (
    aggregate_weighted_depth_l1,
    assert_compatible_protocols,
    assign_frames_to_submaps,
    build_evaluation_frame_ids,
    concatenate_gaussian_params,
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


def test_depth_metric_is_aggregated_by_valid_pixel_count():
    assert aggregate_weighted_depth_l1(
        [(1.0, 1), (3.0, 3)]) == pytest.approx(2.5)


def test_submap_gaussians_are_concatenated_without_retraining():
    first = {
        "xyz": torch.tensor([[1.0, 2.0, 3.0]]),
        "features_dc": torch.ones(1, 1, 3),
        "features_rest": torch.empty(1, 0, 3),
        "opacity": torch.tensor([[0.1]]),
        "scaling": torch.tensor([[0.2, 0.2, 0.2]]),
        "rotation": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    }
    second = {
        key: value + 1 if value.numel() else value.clone()
        for key, value in first.items()
    }

    merged = concatenate_gaussian_params([first, second])

    assert merged["xyz"].shape == (2, 3)
    torch.testing.assert_close(merged["xyz"][0], first["xyz"][0])
    torch.testing.assert_close(merged["xyz"][1], second["xyz"][0])
    assert all(not value.requires_grad for value in merged.values())


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


def test_relative_pose_errors_report_translation_rotation_and_pairs():
    ground_truth = np.repeat(np.eye(4)[None], 3, axis=0)
    estimated = ground_truth.copy()
    ground_truth[:, 0, 3] = [0.0, 1.0, 2.0]
    estimated[:, 0, 3] = [0.0, 1.1, 2.2]

    metrics = compute_relative_pose_errors(estimated, ground_truth)

    assert metrics["valid_pairs"] == 2
    assert metrics["translation_rmse_m"] == pytest.approx(0.1)
    assert metrics["rotation_rmse_deg"] == pytest.approx(0.0)


def test_aggregator_extracts_ate_and_rpe_for_gt_dataset():
    extracted = read_trajectory_metrics(
        {"status": "available"},
        {"rmse": 0.031},
        {
            "alignment_mode": "se3_horn_translation_no_scale",
            "rpe_consecutive": {
                "translation_rmse_m": 0.012,
                "rotation_rmse_deg": 0.7,
                "valid_pairs": 42,
            },
        },
    )

    assert extracted == {
        "ate_rmse_cm": 3.1,
        "rpe_translation_cm": 1.2,
        "rpe_rotation_deg": 0.7,
        "rpe_valid_pairs": 42,
        "trajectory_alignment": "se3_horn_translation_no_scale",
    }
