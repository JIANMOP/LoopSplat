from dataclasses import dataclass
from datetime import datetime, timezone
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import secrets
import subprocess
import sys

import torch
import yaml


VALID_STATES = {"running", "succeeded", "failed"}
REQUIRED_FORMAL_OUTPUTS = (
    "manifest.json",
    "rendering_metrics_observed_view.json",
    "evaluation_protocol.json",
    "trajectory_status.json",
)


@dataclass(frozen=True)
class RunRecord:
    path: Path
    experiment_id: str
    seed: int
    status: dict
    manifest: dict


def _validate_path_segment(value, label):
    value = str(value)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value) or value in {".", ".."}:
        raise ValueError(f"invalid {label}: {value!r}")
    return value


def create_run_directory(output_root, experiment_id, seed, now_utc=None,
                         suffix=None) -> Path:
    experiment_id = _validate_path_segment(experiment_id, "experiment_id")
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    now_utc = now_utc or datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    now_utc = now_utc.astimezone(timezone.utc)
    suffix = _validate_path_segment(
        suffix or secrets.token_hex(4), "run suffix")
    run_name = f"{now_utc.strftime('%Y%m%dT%H%M%SZ')}_{suffix}"
    run_dir = (
        Path(output_root) / experiment_id / f"seed_{seed}" / run_name)
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def prepare_run_directory(config) -> Path:
    data_config = config.get("data", {})
    configured_output = data_config.get("output_path")
    if not configured_output:
        raise ValueError("data.output_path is required")
    if data_config.get("run_directory_prepared", False):
        run_dir = Path(configured_output)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    experiment_id = config.get(
        "experiment_id", data_config.get("scene_name", "experiment"))
    experiment_id = str(experiment_id).replace("/", "-").replace(" ", "-")
    run_dir = create_run_directory(
        configured_output, experiment_id, config.get("seed", 0))
    data_config["output_path"] = str(run_dir)
    data_config["run_directory_prepared"] = True
    return run_dir


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git_value(project_root, args):
    result = subprocess.run(
        ["git", *args], cwd=project_root, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def current_git_commit():
    project_root = Path(__file__).resolve().parents[2]
    return _git_value(project_root, ["rev-parse", "HEAD"])


def current_git_dirty():
    project_root = Path(__file__).resolve().parents[2]
    return bool(_git_value(project_root, ["status", "--porcelain"]))


def config_sha256(config, exclude_seed=False):
    normalized = deepcopy(config)
    data_config = normalized.get("data", {})
    data_config.pop("output_path", None)
    data_config.pop("run_directory_prepared", None)
    if exclude_seed:
        normalized.pop("seed", None)
    serialized = yaml.safe_dump(normalized, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def write_manifest(run_dir, config, argv, effective_features) -> dict:
    run_dir = Path(run_dir)
    project_root = Path(__file__).resolve().parents[2]
    manifest_path = run_dir / "manifest.json"
    existing = (
        json.loads(manifest_path.read_text()) if manifest_path.exists() else {})
    evaluation_ids_path = run_dir / "evaluation_frame_ids.json"
    evaluation_frame_ids = (
        json.loads(evaluation_ids_path.read_text())
        if evaluation_ids_path.exists() else None)
    effective_path = run_dir / "effective_features.yaml"
    observed_effective_features = (
        yaml.safe_load(effective_path.read_text())
        if effective_path.exists() else effective_features)
    cuda_available = torch.cuda.is_available()
    manifest = {
        "created_utc": existing.get(
            "created_utc", datetime.now(timezone.utc).isoformat()),
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_value(project_root, ["rev-parse", "HEAD"]),
        "git_dirty": bool(_git_value(project_root, ["status", "--porcelain"])),
        "config_sha256": config_sha256(config),
        "experiment_config_sha256": config_sha256(
            config, exclude_seed=True),
        "seed": config.get("seed"),
        "experiment_id": config.get("experiment_id"),
        "formal_experiment": bool(config.get("formal_experiment", False)),
        "command": [str(value) for value in argv],
        "requested_features": effective_features,
        "effective_features": observed_effective_features,
        "gsr_max_iters": config.get(
            "lc", {}).get("registration", {}).get("gsr_max_iters"),
        "evaluation_frame_ids": evaluation_frame_ids,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": cuda_available,
            "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
        },
    }
    _write_json(manifest_path, manifest)
    return manifest


def write_status(run_dir, state, **fields) -> None:
    if state not in VALID_STATES:
        raise ValueError(f"invalid run state: {state}")
    run_dir = Path(run_dir)
    status_path = run_dir / "status.json"
    existing = (
        json.loads(status_path.read_text()) if status_path.exists() else {})
    statistics_path = run_dir / "run_statistics.yaml"
    statistics = (
        yaml.safe_load(statistics_path.read_text())
        if state == "succeeded" and statistics_path.exists() else {})
    status = {
        **existing,
        **statistics,
        "state": state,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    _write_json(status_path, status)


def effective_features_from_config(config):
    tracking = config.get("tracking", {})
    keyframing = config.get("keyframing", {})
    pyramid = config.get("gaussian_pyramid", {})
    return {
        "imu": bool(tracking.get("use_imu", False)),
        "gaussian_pyramid": bool(pyramid.get("enabled", False)),
        "gi_keyframing": bool(keyframing.get("enable_gi_slam", False)),
        "gi_keyframing_imu_gyro": bool(
            keyframing.get("use_imu_gyro", False)),
    }


def formal_outputs_complete(run_dir):
    run_dir = Path(run_dir)
    if not all((run_dir / name).exists() for name in REQUIRED_FORMAL_OUTPUTS):
        return False
    try:
        manifest = json.loads((run_dir / "manifest.json").read_text())
        metrics = json.loads(
            (run_dir / "rendering_metrics_observed_view.json").read_text())
        protocol = json.loads(
            (run_dir / "evaluation_protocol.json").read_text())
        trajectory = json.loads(
            (run_dir / "trajectory_status.json").read_text())

        frame_ids = protocol.get("frame_ids")
        if (not isinstance(frame_ids, list) or not frame_ids
                or any(type(frame_id) is not int or frame_id < 0
                       for frame_id in frame_ids)
                or frame_ids != sorted(set(frame_ids))):
            return False
        if protocol.get(
                "formal_map_source"
        ) != "unrefined_global_gaussian_concatenation":
            return False
        if protocol.get("global_refinement_enabled") is not False:
            return False
        if protocol.get("global_refinement_iterations") != 0:
            return False

        if manifest.get("git_dirty") is not False:
            return False
        if not manifest.get("git_commit"):
            return False
        if manifest.get("evaluation_frame_ids") != frame_ids:
            return False
        if manifest.get("formal_experiment"):
            formal_paths = (
                run_dir / "config.yaml",
                run_dir / "run.log",
                run_dir / "evaluation_frame_ids.json",
            )
            if not all(path.exists() for path in formal_paths):
                return False
            saved_frame_ids = json.loads(formal_paths[2].read_text())
            if saved_frame_ids != frame_ids:
                return False
            seed_dir = run_dir.parent.name
            if (not seed_dir.startswith("seed_")
                    or manifest.get("seed") != int(
                        seed_dir.removeprefix("seed_"))
                    or manifest.get("experiment_id")
                    != run_dir.parent.parent.name):
                return False
            if (not isinstance(manifest.get("command"), list)
                    or not manifest["command"]
                    or not re.fullmatch(
                        r"[0-9a-f]{64}",
                        str(manifest.get("config_sha256", "")))
                    or not re.fullmatch(
                        r"[0-9a-f]{64}",
                        str(manifest.get(
                            "experiment_config_sha256", "")))):
                return False
            environment = manifest.get("environment", {})
            if (environment.get("cuda_available") is not True
                    or not environment.get("gpu")
                    or not environment.get("pytorch")
                    or not environment.get("cuda_runtime")):
                return False

        def feature_flags(features):
            pyramid = features.get("gaussian_pyramid", False)
            if isinstance(pyramid, dict):
                pyramid = pyramid.get("enabled", False)
            return {
                "imu": bool(features.get("imu", False)),
                "gaussian_pyramid": bool(pyramid),
                "gi_keyframing": bool(features.get("gi_keyframing", False)),
                "gi_keyframing_imu_gyro": bool(
                    features.get("gi_keyframing_imu_gyro", False)),
            }

        requested = feature_flags(manifest.get("requested_features", {}))
        effective = feature_flags(manifest.get("effective_features", {}))
        if requested != effective:
            return False
        if effective["gi_keyframing_imu_gyro"] and not (
                effective["gi_keyframing"] and effective["imu"]):
            return False

        finite_metric_keys = ("psnr", "ssim", "lpips")
        if any(not isinstance(metrics.get(key), (int, float))
               or not math.isfinite(metrics[key])
               for key in finite_metric_keys):
            return False
        if metrics["lpips"] < 0 or not 0 <= metrics["ssim"] <= 1:
            return False
        if metrics.get("num_renders") != len(frame_ids):
            return False
        depth_value = metrics.get("depth_l1_observed_view")
        if (not isinstance(depth_value, (int, float))
                or not math.isfinite(depth_value) or depth_value < 0
                or type(metrics.get("depth_valid_pixels")) is not int
                or metrics["depth_valid_pixels"] <= 0):
            return False

        summary_paths = {
            "imu": run_dir / "imu_tracking_summary.yaml",
            "pyramid": run_dir / "gaussian_pyramid_summary.yaml",
            "statistics": run_dir / "run_statistics.yaml",
        }
        if not all(path.exists() for path in summary_paths.values()):
            return False
        imu_summary = yaml.safe_load(summary_paths["imu"].read_text())
        pyramid_summary = yaml.safe_load(summary_paths["pyramid"].read_text())
        statistics = yaml.safe_load(summary_paths["statistics"].read_text())
        if not all(isinstance(value, dict) for value in (
                imu_summary, pyramid_summary, statistics)):
            return False
        if bool(imu_summary.get("enabled")) != effective["imu"]:
            return False
        if bool(pyramid_summary.get("enabled")) != effective[
                "gaussian_pyramid"]:
            return False
        if effective["imu"]:
            prediction_records = imu_summary.get("prediction_records")
            if (not isinstance(prediction_records, list)
                    or not prediction_records
                    or not any(
                        record.get("valid") is True
                        for record in prediction_records)):
                return False
            valid_prediction_count = sum(
                record.get("valid") is True
                for record in prediction_records)
            if imu_summary.get(
                    "valid_prediction_count") != valid_prediction_count:
                return False
            if (type(imu_summary.get("dataset_imu_samples")) is not int
                    or imu_summary["dataset_imu_samples"] < 2
                    or type(imu_summary.get(
                        "dataset_imu_rows_dropped")) is not int
                    or type(imu_summary.get(
                        "dataset_max_malformed_imu_rows")) is not int
                    or imu_summary["dataset_imu_rows_dropped"]
                    > imu_summary["dataset_max_malformed_imu_rows"]):
                return False
        if (effective["gaussian_pyramid"]
                and pyramid_summary.get("optimizer_step_count", 0) <= 0):
            return False
        for key in ("frame_count", "keyframe_count", "submap_count"):
            if type(statistics.get(key)) is not int or statistics[key] <= 0:
                return False
        for key in ("slam_elapsed_seconds", "slam_peak_gpu_memory_bytes"):
            value = statistics.get(key)
            if (not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                return False
        if effective["gi_keyframing"]:
            decisions_path = run_dir / "keyframe_decisions.jsonl"
            if not decisions_path.exists() or not decisions_path.read_text().strip():
                return False
            for line in decisions_path.read_text().splitlines():
                decision = json.loads(line)
                if "frame_id" not in decision or "selected" not in decision:
                    return False

        if trajectory.get("status") == "available":
            trajectory_paths = (
                run_dir / "ate_aligned.json",
                run_dir / "rpe.json",
                run_dir / "trajectory_metrics.json",
            )
            if not all(path.exists() for path in trajectory_paths):
                return False
            ate = json.loads(trajectory_paths[0].read_text())
            rpe = json.loads(trajectory_paths[1].read_text())
            trajectory_metrics = json.loads(
                trajectory_paths[2].read_text())
            rpe_values = (
                rpe.get("translation_rmse_m"),
                rpe.get("rotation_rmse_deg"),
            )
            if (not isinstance(ate.get("rmse"), (int, float))
                    or not math.isfinite(ate["rmse"])
                    or ate["rmse"] < 0
                    or type(rpe.get("valid_pairs")) is not int
                    or rpe["valid_pairs"] <= 0
                    or any(not isinstance(value, (int, float))
                           or not math.isfinite(value) or value < 0
                           for value in rpe_values)):
                return False
            return (
                trajectory_metrics.get("alignment_mode")
                == "se3_horn_translation_no_scale"
                and type(trajectory_metrics.get("valid_poses")) is int
                and trajectory_metrics["valid_poses"] >= 2
                and trajectory_metrics.get("rpe_consecutive") == rpe
                and trajectory_metrics.get(
                    "ate_aligned", {}).get("rmse") == ate["rmse"]
            )
        return trajectory.get("status") == "skipped_no_ground_truth"
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, yaml.YAMLError):
        return False


def discover_completed_runs(output_root) -> list[RunRecord]:
    output_root = Path(output_root)
    records = []
    if not output_root.exists():
        return records
    for status_path in output_root.rglob("status.json"):
        run_dir = status_path.parent
        status = json.loads(status_path.read_text())
        if status.get("state") != "succeeded":
            continue
        if not formal_outputs_complete(run_dir):
            continue
        relative = run_dir.relative_to(output_root)
        if len(relative.parts) < 3 or not relative.parts[1].startswith("seed_"):
            continue
        manifest = json.loads((run_dir / "manifest.json").read_text())
        records.append(RunRecord(
            path=run_dir,
            experiment_id=relative.parts[0],
            seed=int(relative.parts[1].removeprefix("seed_")),
            status=status,
            manifest=manifest,
        ))
    return sorted(records, key=lambda record: str(record.path))
