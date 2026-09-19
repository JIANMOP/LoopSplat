#!/usr/bin/env python3
"""Extract compact, auditable manuscript metrics from LoopSplat runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from statistics import mean
from typing import Iterable, Optional


EXPERIMENT_RE = re.compile(r"^(?P<group>[ARC])(?P<scene>\d+)_(?P<strategy>\d+)$")
STRATEGIES = {
    "A": {"0": "Baseline", "1": "GI-KF", "2": "Pyramid", "3": "GI-KF+Pyramid"},
    "R": {"0": "Baseline", "1": "GI-KF", "2": "Pyramid", "3": "GI-KF+Pyramid"},
    "C": {
        "0": "Baseline",
        "1": "IMU",
        "2": "GI-KF",
        "3": "Pyramid",
        "4": "GI-KF+Pyramid",
        "5": "ALL",
    },
}


@dataclass(frozen=True)
class RunRecord:
    dataset_group: str
    scene_id: str
    strategy_id: str
    experiment_id: str
    seed: int
    run_dir: Path
    frame_count: int
    keyframe_count: int
    submap_count: int
    slam_time_hours: float
    peak_gpu_gib: float
    ate_cm: Optional[float]
    rpe_translation: Optional[float]
    rpe_rotation: Optional[float]
    psnr: float
    ssim: float
    lpips: float
    depth_l1: float
    support_update_count: int
    support_update_seconds: float
    gpu_name: str
    cuda_runtime: str
    internal_source_id: str


@dataclass(frozen=True)
class SummaryRecord:
    dataset_group: str
    strategy_id: str
    scene_count: int
    frame_count: float
    keyframe_count: float
    submap_count: float
    slam_time_hours: float
    peak_gpu_gib: float
    ate_cm: Optional[float]
    rpe_translation: Optional[float]
    rpe_rotation: Optional[float]
    psnr: float
    ssim: float
    lpips: float
    depth_l1: float
    support_update_count: float
    support_update_seconds: float


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _finite_float(value, *, label: str, path: Path) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite {label} in {path}")
    return result


def _optional_metric(data: dict, keys: Iterable[str]) -> Optional[float]:
    for key in keys:
        if key in data and data[key] is not None:
            value = float(data[key])
            return value if math.isfinite(value) else None
    return None


def _parse_rpe(path: Path) -> tuple[Optional[float], Optional[float]]:
    if not path.exists():
        return None, None
    data = _read_json(path)
    translation = _optional_metric(
        data,
        ("translation_rmse_m", "translation_rmse", "trans_rmse", "rmse_translation"),
    )
    rotation = _optional_metric(
        data,
        ("rotation_rmse_deg", "rotation_rmse", "rot_rmse", "rmse_rotation"),
    )
    if translation is None and isinstance(data.get("translation"), dict):
        translation = _optional_metric(data["translation"], ("rmse",))
    if rotation is None and isinstance(data.get("rotation"), dict):
        rotation = _optional_metric(data["rotation"], ("rmse",))
    return translation, rotation


def _parse_run(experiment_id: str, seed: int, run_dir: Path) -> RunRecord:
    match = EXPERIMENT_RE.match(experiment_id)
    if match is None:
        raise ValueError(f"unsupported experiment id: {experiment_id}")
    group = match.group("group")
    strategy_suffix = match.group("strategy")
    try:
        strategy_id = STRATEGIES[group][strategy_suffix]
    except KeyError as exc:
        raise ValueError(f"unsupported strategy: {experiment_id}") from exc

    status = _read_json(run_dir / "status.json")
    manifest = _read_json(run_dir / "manifest.json")
    rendering_path = run_dir / "rendering_metrics_observed_view.json"
    rendering = _read_json(rendering_path)

    psnr = _finite_float(rendering["psnr"], label="PSNR", path=rendering_path)
    ssim = _finite_float(rendering["ssim"], label="SSIM", path=rendering_path)
    lpips = _finite_float(rendering["lpips"], label="LPIPS", path=rendering_path)
    depth_l1 = _finite_float(
        rendering["depth_l1_observed_view"], label="Depth L1", path=rendering_path
    )

    ate_cm = None
    ate_path = run_dir / "ate_aligned.json"
    if ate_path.exists():
        ate = _read_json(ate_path)
        ate_cm = _finite_float(ate["rmse"], label="ATE", path=ate_path) * 100.0
    rpe_translation, rpe_rotation = _parse_rpe(run_dir / "rpe.json")

    gi = status.get("gi_keyframing") or {}
    environment = manifest.get("environment") or {}
    source_id = str(
        manifest.get("experiment_source_sha256")
        or manifest.get("config_sha256")
        or manifest.get("git_commit")
        or "unrecorded"
    )
    return RunRecord(
        dataset_group=group,
        scene_id=f"{group}{match.group('scene')}",
        strategy_id=strategy_id,
        experiment_id=experiment_id,
        seed=seed,
        run_dir=run_dir,
        frame_count=int(status.get("frame_count", 0)),
        keyframe_count=int(status.get("keyframe_count", 0)),
        submap_count=int(status.get("submap_count", 0)),
        slam_time_hours=_finite_float(
            status.get("slam_elapsed_seconds", status.get("elapsed_seconds", 0.0)),
            label="SLAM elapsed seconds",
            path=run_dir / "status.json",
        )
        / 3600.0,
        peak_gpu_gib=_finite_float(
            status.get("slam_peak_gpu_memory_bytes", 0.0),
            label="peak GPU memory",
            path=run_dir / "status.json",
        )
        / (2**30),
        ate_cm=ate_cm,
        rpe_translation=rpe_translation,
        rpe_rotation=rpe_rotation,
        psnr=psnr,
        ssim=ssim,
        lpips=lpips,
        depth_l1=depth_l1,
        support_update_count=int(gi.get("support_update_count", 0)),
        support_update_seconds=_finite_float(
            gi.get("support_update_elapsed_seconds", 0.0),
            label="support update seconds",
            path=run_dir / "status.json",
        ),
        gpu_name=str(environment.get("gpu", "unknown")),
        cuda_runtime=str(environment.get("cuda_runtime", "unknown")),
        internal_source_id=source_id,
    )


def load_latest_successful_runs(root: Path) -> list[RunRecord]:
    """Load the latest complete successful run for every experiment and seed."""
    root = Path(root)
    records: list[RunRecord] = []
    if not root.exists():
        raise FileNotFoundError(root)

    for experiment_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        experiment_id = experiment_dir.name
        if EXPERIMENT_RE.match(experiment_id) is None:
            continue
        for seed_dir in sorted(experiment_dir.glob("seed_*")):
            if not seed_dir.is_dir():
                continue
            try:
                seed = int(seed_dir.name.split("_", 1)[1])
            except (IndexError, ValueError) as exc:
                raise ValueError(f"invalid seed directory: {seed_dir}") from exc
            successful: list[Path] = []
            for run_dir in sorted(path for path in seed_dir.iterdir() if path.is_dir()):
                required = (
                    run_dir / "status.json",
                    run_dir / "manifest.json",
                    run_dir / "rendering_metrics_observed_view.json",
                )
                if not all(path.exists() for path in required):
                    continue
                status = _read_json(run_dir / "status.json")
                if status.get("state") == "succeeded":
                    successful.append(run_dir)
            if successful:
                records.append(_parse_run(experiment_id, seed, successful[-1]))
    return records


def _mean_optional(values: Iterable[Optional[float]]) -> Optional[float]:
    present = [value for value in values if value is not None]
    return round(mean(present), 12) if present else None


def macro_average(records: list[RunRecord]) -> list[SummaryRecord]:
    """Macro-average scenes for each dataset and strategy without seed variance."""
    grouped: dict[tuple[str, str], list[RunRecord]] = {}
    for record in records:
        grouped.setdefault((record.dataset_group, record.strategy_id), []).append(record)

    summaries: list[SummaryRecord] = []
    numeric = (
        "frame_count",
        "keyframe_count",
        "submap_count",
        "slam_time_hours",
        "peak_gpu_gib",
        "psnr",
        "ssim",
        "lpips",
        "depth_l1",
        "support_update_count",
        "support_update_seconds",
    )
    for (group, strategy), group_records in sorted(grouped.items()):
        values = {
            name: round(
                mean(float(getattr(record, name)) for record in group_records), 12
            )
            for name in numeric
        }
        summaries.append(
            SummaryRecord(
                dataset_group=group,
                strategy_id=strategy,
                scene_count=len({record.scene_id for record in group_records}),
                ate_cm=_mean_optional(record.ate_cm for record in group_records),
                rpe_translation=_mean_optional(record.rpe_translation for record in group_records),
                rpe_rotation=_mean_optional(record.rpe_rotation for record in group_records),
                **values,
            )
        )
    return summaries


def _write_csv(path: Path, rows: list[dict], field_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(records: list[RunRecord], output_root: Path) -> None:
    output_root = Path(output_root)
    table_root = output_root / "tables"
    internal_root = output_root / "internal"
    public_fields = [field.name for field in fields(RunRecord) if field.name != "internal_source_id"]
    public_rows = []
    for record in records:
        row = asdict(record)
        row["run_dir"] = str(record.run_dir)
        row.pop("internal_source_id")
        public_rows.append(row)
    _write_csv(table_root / "result-source.csv", public_rows, public_fields)

    summaries = macro_average(records)
    summary_fields = [field.name for field in fields(SummaryRecord)]
    _write_csv(
        table_root / "dataset-summary.csv",
        [asdict(summary) for summary in summaries],
        summary_fields,
    )

    internal_root.mkdir(parents=True, exist_ok=True)
    audit = [
        {
            "experiment_id": record.experiment_id,
            "seed": record.seed,
            "run_dir": str(record.run_dir),
            "internal_source_id": record.internal_source_id,
            "gpu_name": record.gpu_name,
            "cuda_runtime": record.cuda_runtime,
        }
        for record in records
    ]
    (internal_root / "result-source-audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    records = load_latest_successful_runs(args.results_root)
    if not records:
        raise SystemExit(f"no successful experiment records found in {args.results_root}")
    write_outputs(records, args.output_root)
    print(f"wrote {len(records)} records and {len(macro_average(records))} summaries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
