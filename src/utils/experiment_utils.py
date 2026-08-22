from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
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


def write_manifest(run_dir, config, argv, effective_features) -> dict:
    run_dir = Path(run_dir)
    project_root = Path(__file__).resolve().parents[2]
    serialized_config = yaml.safe_dump(config, sort_keys=True)
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
        "config_sha256": hashlib.sha256(
            serialized_config.encode("utf-8")).hexdigest(),
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
    trajectory = json.loads(
        (run_dir / "trajectory_status.json").read_text())
    if trajectory.get("status") == "available":
        return all((run_dir / name).exists() for name in (
            "ate_aligned.json", "rpe.json", "trajectory_metrics.json"))
    return trajectory.get("status") == "skipped_no_ground_truth"


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
