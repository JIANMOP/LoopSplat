import json
from pathlib import Path

import pytest

from thesis.scripts.build_paper_assets import (
    load_latest_successful_runs,
    macro_average,
)


def _write_run(
    root: Path,
    experiment_id: str,
    run_name: str,
    *,
    state: str = "succeeded",
    psnr: float = 20.0,
    depth_l1: float = 0.1,
    with_trajectory: bool = True,
) -> Path:
    run_dir = root / experiment_id / "seed_0" / run_name
    run_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text(
        json.dumps(
            {
                "state": state,
                "frame_count": 100,
                "keyframe_count": 20,
                "submap_count": 3,
                "slam_elapsed_seconds": 7200.0,
                "slam_peak_gpu_memory_bytes": 2**30,
                "gi_keyframing": {
                    "support_update_count": 4,
                    "support_update_elapsed_seconds": 12.0,
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "seed": 0,
                "environment": {"gpu": "RTX test", "cuda_runtime": "12.1"},
                "updated_utc": f"2026-09-01T{run_name[:2]}:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "rendering_metrics_observed_view.json").write_text(
        json.dumps(
            {
                "psnr": psnr,
                "ssim": 0.8,
                "lpips": 0.2,
                "depth_l1_observed_view": depth_l1,
            }
        ),
        encoding="utf-8",
    )
    if with_trajectory:
        (run_dir / "ate_aligned.json").write_text(json.dumps({"rmse": 0.02}), encoding="utf-8")
        (run_dir / "rpe.json").write_text(
            json.dumps({"translation_rmse_m": 0.01, "rotation_rmse_deg": 0.5}),
            encoding="utf-8",
        )
    return run_dir


def test_selects_latest_successful_run_per_experiment_and_seed(tmp_path):
    _write_run(tmp_path, "A1_0", "01-old")
    latest = _write_run(tmp_path, "A1_0", "02-new", psnr=25.0)
    _write_run(tmp_path, "A1_0", "03-failed", state="failed", psnr=30.0)

    records = load_latest_successful_runs(tmp_path)

    assert len(records) == 1
    assert records[0].run_dir == latest
    assert records[0].psnr == 25.0


def test_rejects_nonfinite_metrics(tmp_path):
    _write_run(tmp_path, "A1_0", "01-run", psnr=float("nan"))

    with pytest.raises(ValueError, match="non-finite"):
        load_latest_successful_runs(tmp_path)


def test_preserves_missing_trajectory_as_null_for_fm(tmp_path):
    _write_run(tmp_path, "C1_0", "01-run", with_trajectory=False)

    record = load_latest_successful_runs(tmp_path)[0]

    assert record.dataset_group == "C"
    assert record.ate_cm is None
    assert record.rpe_translation is None
    assert record.rpe_rotation is None


def test_macro_average_groups_by_dataset_and_strategy(tmp_path):
    _write_run(tmp_path, "A1_0", "01-run", psnr=10.0, depth_l1=0.2)
    _write_run(tmp_path, "A2_0", "01-run", psnr=14.0, depth_l1=0.4)
    _write_run(tmp_path, "A1_1", "01-run", psnr=30.0, depth_l1=0.1)

    summaries = macro_average(load_latest_successful_runs(tmp_path))
    baseline = next(
        summary
        for summary in summaries
        if summary.dataset_group == "A" and summary.strategy_id == "Baseline"
    )

    assert baseline.scene_count == 2
    assert baseline.psnr == 12.0
    assert baseline.depth_l1 == 0.3


def test_does_not_compute_seed_standard_deviation_from_scenes(tmp_path):
    _write_run(tmp_path, "R1_0", "01-run", psnr=10.0)
    _write_run(tmp_path, "R2_0", "01-run", psnr=14.0)

    summary = macro_average(load_latest_successful_runs(tmp_path))[0]

    assert summary.scene_count == 2
    assert "std" not in summary.__dataclass_fields__
