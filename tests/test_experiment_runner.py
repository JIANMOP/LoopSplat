from argparse import Namespace
from datetime import datetime, timezone
import json
import yaml
import pytest

from run_slam import update_config_with_args
from scripts.aggregate_results import (
    render_markdown,
    summarize_seed_metrics,
    validate_formal_run_group,
)
from scripts.run_ablation import EXPERIMENTS, config_has_results
from src.entities.gaussian_slam import build_run_statistics
from src.utils.experiment_utils import (
    create_run_directory,
    config_sha256,
    discover_completed_runs,
    prepare_run_directory,
    formal_outputs_complete,
    write_manifest,
    write_status,
)


def write_valid_formal_outputs(run_dir):
    (run_dir / "manifest.json").write_text(json.dumps({
        "git_commit": "a" * 40,
        "git_dirty": False,
        "evaluation_frame_ids": [0],
        "requested_features": {
            "imu": False,
            "gaussian_pyramid": False,
            "gi_keyframing": False,
            "gi_keyframing_imu_gyro": False,
        },
        "effective_features": {
            "imu": False,
            "gaussian_pyramid": {"enabled": False},
            "gi_keyframing": False,
            "gi_keyframing_imu_gyro": False,
        },
    }))
    (run_dir / "rendering_metrics_observed_view.json").write_text(
        json.dumps({
            "psnr": 20.0,
            "ssim": 0.8,
            "lpips": 0.2,
            "depth_l1_observed_view": 0.1,
            "depth_valid_pixels": 100,
            "num_renders": 1,
        }))
    (run_dir / "evaluation_protocol.json").write_text(json.dumps({
        "frame_ids": [0],
        "formal_map_source": "unrefined_global_gaussian_concatenation",
        "global_refinement_enabled": False,
        "global_refinement_iterations": 0,
    }))
    (run_dir / "trajectory_status.json").write_text(
        '{"status": "skipped_no_ground_truth"}')
    (run_dir / "imu_tracking_summary.yaml").write_text(
        yaml.safe_dump({"enabled": False, "commit_count": 0}))
    (run_dir / "gaussian_pyramid_summary.yaml").write_text(
        yaml.safe_dump({"enabled": False, "optimizer_step_count": 0}))
    (run_dir / "run_statistics.yaml").write_text(yaml.safe_dump({
        "frame_count": 1,
        "keyframe_count": 1,
        "submap_count": 1,
        "slam_elapsed_seconds": 1.0,
        "slam_peak_gpu_memory_bytes": 100,
    }))


def test_two_runs_never_share_a_directory(tmp_path):
    fixed_time = datetime(2026, 8, 22, 4, 0, tzinfo=timezone.utc)

    first = create_run_directory(
        tmp_path, "C1_0", 0, fixed_time, "aaaa")
    second = create_run_directory(
        tmp_path, "C1_0", 0, fixed_time, "bbbb")

    assert first != second
    assert first == tmp_path / "C1_0" / "seed_0" / "20260822T040000Z_aaaa"
    assert first.is_dir() and second.is_dir()


def test_only_succeeded_complete_run_is_discovered(tmp_path):
    complete = create_run_directory(
        tmp_path, "C1_0", 0, suffix="complete")
    failed = create_run_directory(
        tmp_path, "C1_0", 0, suffix="failed")
    write_manifest(
        failed, {"seed": 0}, ["run_slam.py"], {"imu": False})
    write_valid_formal_outputs(complete)
    write_status(complete, "succeeded")
    write_status(failed, "failed")

    records = discover_completed_runs(tmp_path)

    assert [record.path for record in records] == [complete]
    assert records[0].experiment_id == "C1_0"
    assert records[0].seed == 0


def test_formal_outputs_reject_empty_metrics_and_dirty_manifest(tmp_path):
    write_valid_formal_outputs(tmp_path)
    (tmp_path / "rendering_metrics_observed_view.json").write_text("{}")
    assert formal_outputs_complete(tmp_path) is False

    write_valid_formal_outputs(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["git_dirty"] = True
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert formal_outputs_complete(tmp_path) is False


def test_formal_outputs_reject_requested_effective_feature_mismatch(tmp_path):
    write_valid_formal_outputs(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["effective_features"]["gaussian_pyramid"]["enabled"] = True
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    assert formal_outputs_complete(tmp_path) is False


def test_formal_imu_run_requires_at_least_one_valid_prediction(tmp_path):
    write_valid_formal_outputs(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["requested_features"]["imu"] = True
    manifest["effective_features"]["imu"] = True
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "imu_tracking_summary.yaml").write_text(yaml.safe_dump({
        "enabled": True,
        "commit_count": 1,
        "prediction_records": [
            {"frame_id": 1, "valid": False, "reason": "coverage"}],
    }))

    assert formal_outputs_complete(tmp_path) is False


def test_formal_trajectory_rejects_nonfinite_rpe(tmp_path):
    write_valid_formal_outputs(tmp_path)
    (tmp_path / "trajectory_status.json").write_text(
        '{"status": "available"}')
    (tmp_path / "ate_aligned.json").write_text('{"rmse": 0.1}')
    (tmp_path / "rpe.json").write_text(json.dumps({
        "valid_pairs": 1,
        "translation_rmse_m": float("nan"),
        "rotation_rmse_deg": 1.0,
    }))
    (tmp_path / "trajectory_metrics.json").write_text(json.dumps({
        "alignment_mode": "se3_horn_translation_no_scale",
        "valid_poses": 2,
        "ate_aligned": {"rmse": 0.1},
        "rpe_consecutive": {
            "valid_pairs": 1,
            "translation_rmse_m": float("nan"),
            "rotation_rmse_deg": 1.0,
        },
    }))

    assert formal_outputs_complete(tmp_path) is False


def test_formal_group_requires_three_seeds_and_shared_commit():
    def record(seed, commit="a"):
        return type("Record", (), {
            "seed": seed,
            "manifest": {
                "git_commit": commit,
                "gsr_max_iters": 100,
                "environment": {"gpu": "GPU", "cuda_runtime": "12.1"},
            },
        })()

    with pytest.raises(ValueError, match="seeds"):
        validate_formal_run_group([record(0), record(1)], "C1_0")
    with pytest.raises(ValueError, match="commit"):
        validate_formal_run_group(
            [record(0), record(1), record(2, "b")], "C1_0")


def test_manifest_records_config_hash_command_and_effective_features(tmp_path):
    run_dir = create_run_directory(tmp_path, "A1_0", 0, suffix="manifest")

    manifest = write_manifest(
        run_dir,
        {"seed": 0, "tracking": {"use_imu": False}},
        ["run_slam.py", "config.yaml"],
        {"imu": False, "pyramid": False},
    )

    assert len(manifest["config_sha256"]) == 64
    assert manifest["command"] == ["run_slam.py", "config.yaml"]
    assert manifest["effective_features"]["imu"] is False
    assert "git_commit" in manifest
    assert "pytorch" in manifest["environment"]


def test_config_hash_ignores_run_directory_but_not_seed():
    first = {
        "seed": 0,
        "data": {"output_path": "/tmp/run-a", "run_directory_prepared": True},
        "tracking": {"use_imu": False},
    }
    second = {
        **first,
        "data": {"output_path": "/tmp/run-b", "run_directory_prepared": True},
    }

    assert config_sha256(first) == config_sha256(second)
    assert config_sha256(first) != config_sha256({**second, "seed": 1})


def test_resume_requires_matching_seed_config_and_commit(tmp_path):
    config = {"seed": 0, "data": {"output_path": str(tmp_path)}}
    run_dir = create_run_directory(
        tmp_path, "C1_0", 0, suffix="complete")
    write_valid_formal_outputs(run_dir)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update({
        "config_sha256": config_sha256(config),
        "git_commit": "b" * 40,
    })
    manifest_path.write_text(json.dumps(manifest))
    write_status(run_dir, "succeeded")

    output_dir = tmp_path / "C1_0"
    assert config_has_results(output_dir, 0, config, "b" * 40)
    assert not config_has_results(output_dir, 1, {**config, "seed": 1}, "b" * 40)
    assert not config_has_results(
        output_dir, 0, {**config, "tracking": {"use_imu": True}}, "b" * 40)
    assert not config_has_results(output_dir, 0, config, "c" * 40)


def test_seed_metrics_report_mean_sample_std_and_count():
    summary = summarize_seed_metrics([
        {"psnr": 20.0, "keyframe_count": 10},
        {"psnr": 22.0, "keyframe_count": 12},
        {"psnr": 24.0, "keyframe_count": 14},
    ])

    assert summary["psnr"] == pytest.approx(22.0)
    assert summary["psnr_std"] == pytest.approx(2.0)
    assert summary["keyframe_count"] == pytest.approx(12.0)
    assert summary["seed_count"] == 3


def test_cli_applies_gt_camera_and_seed_zero():
    config = {
        "seed": 5,
        "data": {},
        "tracking": {"gt_camera": False},
        "mapping": {},
    }
    args = Namespace(
        input_path="", output_path="", track_w_color_loss=None,
        track_alpha_thre=None, track_iters=None, track_filter_alpha=False,
        track_filter_outlier=False, track_wo_filter_alpha=False,
        track_wo_filter_outlier=False, track_cam_trans_lr=None,
        alpha_seeding_thre=None, map_every=None, map_iters=None,
        new_submap_every=None, project_name=None, gt_camera=True,
        help_camera_initialization=False, soft_alpha=False, seed=0,
        submap_using_motion_heuristic=False, new_submap_points_num=None,
    )

    updated = update_config_with_args(config, args)

    assert updated["tracking"]["gt_camera"] is True
    assert updated["seed"] == 0


def test_missing_output_path_fails_before_slam_initialization():
    config = {"seed": 0, "data": {"scene_name": "scene"}}

    try:
        prepare_run_directory(config)
    except ValueError as error:
        assert "data.output_path" in str(error)
    else:
        raise AssertionError("missing output path must fail")


def test_azure_matrix_excludes_imu_without_camera_imu_calibration():
    azure_experiments = [
        experiment for experiment in EXPERIMENTS
        if experiment["group"] == "B"
    ]

    assert azure_experiments
    assert all(
        experiment["overrides"]["tracking"]["use_imu"] is False
        for experiment in azure_experiments
    )


def test_azure_report_uses_rgbd_only_matrix_without_trajectory_columns():
    results = {
        f"B1_{suffix}": {"error": "no results"}
        for suffix in range(6)
    }
    results["B1_0"] = {
        "psnr": 20.0,
        "ssim": 0.8,
        "lpips": 0.2,
        "depth_l1": 0.1,
        "keyframe_count": 12,
        "submap_count": 3,
        "slam_elapsed_seconds": 45.0,
        "slam_peak_gpu_memory_gib": 2.5,
    }

    report = render_markdown(results)
    azure_section = report.split("## Group B", 1)[1].split(
        "## Group C", 1)[0]

    assert "+IMU" not in azure_section
    assert "+ALL" not in azure_section
    assert "ATE↓" not in azure_section
    assert "RPE-t↓" not in azure_section
    assert "KF↓" in azure_section
    assert "Submaps↓" in azure_section
    assert "SLAM s↓" in azure_section
    assert "Peak GiB↓" in azure_section
    assert "12" in azure_section
    assert "2.50" in azure_section


def test_run_statistics_count_unique_keyframes_and_submaps():
    statistics = build_run_statistics(
        mapping_frame_ids=[0, 3, 3, 5],
        frame_count=6,
        submap_count=2,
        elapsed_seconds=12.5,
        peak_gpu_memory_bytes=2_000_000_000,
    )

    assert statistics["mapping_frame_ids"] == [0, 3, 5]
    assert statistics["keyframe_count"] == 3
    assert statistics["submap_count"] == 2
    assert statistics["frame_count"] == 6
    assert statistics["slam_elapsed_seconds"] == 12.5
    assert statistics["slam_peak_gpu_memory_bytes"] == 2_000_000_000


def test_succeeded_status_includes_persisted_run_statistics(tmp_path):
    (tmp_path / "run_statistics.yaml").write_text(
        "keyframe_count: 3\nsubmap_count: 2\n"
        "slam_peak_gpu_memory_bytes: 2000000000\n"
    )

    write_status(tmp_path, "succeeded", elapsed_seconds=15.0)

    status = json.loads((tmp_path / "status.json").read_text())
    assert status["keyframe_count"] == 3
    assert status["submap_count"] == 2
    assert status["slam_peak_gpu_memory_bytes"] == 2_000_000_000
    assert status["elapsed_seconds"] == 15.0
