from argparse import Namespace
from datetime import datetime, timezone

from run_slam import update_config_with_args
from scripts.aggregate_results import render_markdown
from scripts.run_ablation import EXPERIMENTS
from src.utils.experiment_utils import (
    create_run_directory,
    discover_completed_runs,
    prepare_run_directory,
    write_manifest,
    write_status,
)


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
        complete, {"seed": 0}, ["run_slam.py"], {"imu": False})
    write_manifest(
        failed, {"seed": 0}, ["run_slam.py"], {"imu": False})
    for filename in (
            "rendering_metrics_observed_view.json",
            "evaluation_protocol.json"):
        (complete / filename).write_text("{}")
    (complete / "trajectory_status.json").write_text(
        '{"status": "skipped_no_ground_truth"}')
    write_status(complete, "succeeded")
    write_status(failed, "failed")

    records = discover_completed_runs(tmp_path)

    assert [record.path for record in records] == [complete]
    assert records[0].experiment_id == "C1_0"
    assert records[0].seed == 0


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

    report = render_markdown(results)
    azure_section = report.split("## Group B", 1)[1].split(
        "## Group C", 1)[0]

    assert "+IMU" not in azure_section
    assert "+ALL" not in azure_section
    assert "ATE↓" not in azure_section
    assert "RPE-t↓" not in azure_section
