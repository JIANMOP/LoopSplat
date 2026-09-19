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
STRATEGY_ORDER = ("Baseline", "IMU", "GI-KF", "Pyramid", "GI-KF+Pyramid", "ALL")
TABLE_FILES = {"A": "tum-results.md", "R": "replica-results.md", "C": "fm-results.md"}


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


def load_records_csv(path: Path) -> list[RunRecord]:
    """Load the normalized result CSV emitted by :func:`write_outputs`."""
    optional = {"ate_cm", "rpe_translation", "rpe_rotation"}
    integers = {
        "seed",
        "frame_count",
        "keyframe_count",
        "submap_count",
        "support_update_count",
    }
    floats = {
        "slam_time_hours",
        "peak_gpu_gib",
        "ate_cm",
        "rpe_translation",
        "rpe_rotation",
        "psnr",
        "ssim",
        "lpips",
        "depth_l1",
        "support_update_seconds",
    }
    records: list[RunRecord] = []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            values = dict(row)
            for name in integers:
                values[name] = int(values[name])
            for name in floats:
                if name in optional and not values[name]:
                    values[name] = None
                else:
                    values[name] = float(values[name])
            values["run_dir"] = Path(values["run_dir"])
            values["internal_source_id"] = "omitted-from-public-csv"
            records.append(RunRecord(**values))
    return records


def relative_change(value: float, baseline: float) -> float:
    """Return percentage change using the baseline as denominator."""
    if baseline == 0:
        raise ValueError("baseline must be non-zero")
    return (value - baseline) / baseline * 100.0


def hardware_compatible_pairs(
    records: list[RunRecord], strategy_id: str
) -> list[tuple[RunRecord, RunRecord]]:
    """Pair a strategy with its same-scene baseline on compatible hardware."""
    baselines = {
        (record.dataset_group, record.scene_id): record
        for record in records
        if record.strategy_id == "Baseline"
    }
    pairs = []
    for record in records:
        if record.strategy_id != strategy_id:
            continue
        baseline = baselines.get((record.dataset_group, record.scene_id))
        if baseline is None:
            continue
        if (record.gpu_name, record.cuda_runtime) != (
            baseline.gpu_name,
            baseline.cuda_runtime,
        ):
            continue
        pairs.append((baseline, record))
    return pairs


def _strategy_rank(strategy: str) -> int:
    try:
        return STRATEGY_ORDER.index(strategy)
    except ValueError:
        return len(STRATEGY_ORDER)


def _fmt(value: Optional[float], digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _table_header() -> str:
    return (
        "| Scene | Strategy | ATE (cm) ↓ | RPE-t (m) ↓ | RPE-r (°) ↓ | "
        "PSNR (dB) ↑ | MS-SSIM ↑ | LPIPS ↓ | Depth L1 ↓ | Keyframes ↓ |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )


def generate_manuscript_tables(
    records: list[RunRecord], output_root: Path
) -> dict[str, Path]:
    """Generate per-dataset quality tables and a hardware-safe efficiency table."""
    table_root = Path(output_root) / "tables"
    table_root.mkdir(parents=True, exist_ok=True)
    summaries = macro_average(records)
    paths: dict[str, Path] = {}
    for group, file_name in TABLE_FILES.items():
        group_records = sorted(
            (record for record in records if record.dataset_group == group),
            key=lambda record: (record.scene_id, _strategy_rank(record.strategy_id)),
        )
        if not group_records:
            continue
        lines = [_table_header().rstrip("\n")]
        for record in group_records:
            lines.append(
                "| {scene} | {strategy} | {ate} | {rpet} | {rper} | {psnr} | "
                "{ssim} | {lpips} | {depth} | {keyframes} |".format(
                    scene=record.scene_id,
                    strategy=record.strategy_id,
                    ate=_fmt(record.ate_cm),
                    rpet=_fmt(record.rpe_translation, 4),
                    rper=_fmt(record.rpe_rotation),
                    psnr=_fmt(record.psnr),
                    ssim=_fmt(record.ssim, 4),
                    lpips=_fmt(record.lpips, 4),
                    depth=_fmt(record.depth_l1, 4),
                    keyframes=record.keyframe_count,
                )
            )
        group_summaries = sorted(
            (summary for summary in summaries if summary.dataset_group == group),
            key=lambda summary: _strategy_rank(summary.strategy_id),
        )
        for summary in group_summaries:
            lines.append(
                "| Macro average | {strategy} | {ate} | {rpet} | {rper} | {psnr} | "
                "{ssim} | {lpips} | {depth} | {keyframes} |".format(
                    strategy=summary.strategy_id,
                    ate=_fmt(summary.ate_cm),
                    rpet=_fmt(summary.rpe_translation, 4),
                    rper=_fmt(summary.rpe_rotation),
                    psnr=_fmt(summary.psnr),
                    ssim=_fmt(summary.ssim, 4),
                    lpips=_fmt(summary.lpips, 4),
                    depth=_fmt(summary.depth_l1, 4),
                    keyframes=_fmt(summary.keyframe_count, 1),
                )
            )
        path = table_root / file_name
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        paths[group] = path

    efficiency_lines = [
        "| Dataset | Scene | Strategy | GPU | Time (h) ↓ | Time Δ (%) ↓ | "
        "Peak PyTorch allocation (GiB) ↓ | Keyframes ↓ | Keyframe Δ (%) ↓ | "
        "Support updates | Support time (s) |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    baseline_by_scene = {
        (record.dataset_group, record.scene_id): record
        for record in records
        if record.strategy_id == "Baseline"
    }
    for record in sorted(
        records,
        key=lambda item: (item.dataset_group, item.scene_id, _strategy_rank(item.strategy_id)),
    ):
        baseline = baseline_by_scene[(record.dataset_group, record.scene_id)]
        compatible = (record.gpu_name, record.cuda_runtime) == (
            baseline.gpu_name,
            baseline.cuda_runtime,
        )
        if record.strategy_id == "Baseline":
            time_delta = keyframe_delta = 0.0
        elif compatible:
            time_delta = relative_change(record.slam_time_hours, baseline.slam_time_hours)
            keyframe_delta = relative_change(record.keyframe_count, baseline.keyframe_count)
        else:
            time_delta = keyframe_delta = None
        efficiency_lines.append(
            "| {group} | {scene} | {strategy} | {gpu} | {time} | {time_delta} | "
            "{memory} | {keyframes} | {keyframe_delta} | {support_count} | "
            "{support_time} |".format(
                group=record.dataset_group,
                scene=record.scene_id,
                strategy=record.strategy_id,
                gpu=record.gpu_name,
                time=_fmt(record.slam_time_hours),
                time_delta=_fmt(time_delta, 1),
                memory=_fmt(record.peak_gpu_gib),
                keyframes=record.keyframe_count,
                keyframe_delta=_fmt(keyframe_delta, 1),
                support_count=record.support_update_count,
                support_time=_fmt(record.support_update_seconds),
            )
        )
    efficiency_lines.extend(
        [
            "",
            "Time and keyframe deltas use the same-scene Baseline as denominator. "
            "A blank delta indicates incompatible recorded hardware/runtime environments. "
            "Peak allocation is measured with `torch.cuda.max_memory_allocated()`; "
            "support observations are excluded from persistent-keyframe counts.",
        ]
    )
    efficiency_path = table_root / "efficiency-results.md"
    efficiency_path.write_text("\n".join(efficiency_lines) + "\n", encoding="utf-8")
    paths["efficiency"] = efficiency_path
    return paths


def generate_figures(records: list[RunRecord], output_root: Path) -> list[Path]:
    """Generate hardware-safe, dataset-faceted publication plots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    figure_root = Path(output_root) / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    colors = {
        "IMU": "#0072B2",
        "GI-KF": "#D55E00",
        "Pyramid": "#009E73",
        "GI-KF+Pyramid": "#CC79A7",
        "ALL": "#E69F00",
    }
    groups = ("A", "R", "C")

    def figure_legend(figure, plot_axes, *, top: float) -> None:
        by_label = {}
        for axis in plot_axes:
            handles, labels = axis.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                by_label.setdefault(label, handle)
        ordered_labels = [name for name in STRATEGY_ORDER if name in by_label]
        if ordered_labels:
            figure.legend(
                [by_label[name] for name in ordered_labels],
                ordered_labels,
                loc="upper center",
                ncol=min(5, len(ordered_labels)),
                frameon=False,
            )
        figure.tight_layout(rect=(0, 0, 1, top))

    fig, axes = plt.subplots(2, 3, figsize=(11.2, 5.8), sharey="row")
    for column, group in enumerate(groups):
        ax_psnr, ax_depth = axes[0, column], axes[1, column]
        for strategy in STRATEGY_ORDER[1:]:
            pairs = [
                pair
                for pair in hardware_compatible_pairs(records, strategy)
                if pair[0].dataset_group == group
            ]
            if not pairs:
                continue
            x = list(range(len(pairs)))
            labels = [baseline.scene_id for baseline, _ in pairs]
            ax_psnr.plot(
                x,
                [candidate.psnr - baseline.psnr for baseline, candidate in pairs],
                marker="o",
                linewidth=1.2,
                color=colors[strategy],
                label=strategy,
            )
            ax_depth.plot(
                x,
                [
                    relative_change(candidate.depth_l1, baseline.depth_l1)
                    for baseline, candidate in pairs
                ],
                marker="o",
                linewidth=1.2,
                color=colors[strategy],
                label=strategy,
            )
            ax_psnr.set_xticks(x, labels, rotation=45, ha="right")
            ax_depth.set_xticks(x, labels, rotation=45, ha="right")
        ax_psnr.axhline(0, color="#666666", linewidth=0.7)
        ax_depth.axhline(0, color="#666666", linewidth=0.7)
        ax_psnr.set_xlabel(f"{group} scene")
        ax_depth.set_xlabel(f"{group} scene")
        ax_psnr.grid(axis="y", alpha=0.25)
        ax_depth.grid(axis="y", alpha=0.25)
    axes[0, 0].set_ylabel("PSNR change (dB)")
    axes[1, 0].set_ylabel("Depth L1 change (%)")
    figure_legend(fig, axes.flat, top=0.94)
    metric_pdf = figure_root / "metric-deltas.pdf"
    metric_png = figure_root / "metric-deltas.png"
    fig.savefig(metric_pdf, bbox_inches="tight")
    fig.savefig(metric_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.7), squeeze=False)
    for column, group in enumerate(groups):
        ax = axes[0, column]
        for strategy in STRATEGY_ORDER[1:]:
            pairs = [
                pair
                for pair in hardware_compatible_pairs(records, strategy)
                if pair[0].dataset_group == group
            ]
            if not pairs:
                continue
            time_change = mean(
                relative_change(candidate.slam_time_hours, baseline.slam_time_hours)
                for baseline, candidate in pairs
            )
            psnr_change = mean(
                candidate.psnr - baseline.psnr for baseline, candidate in pairs
            )
            keyframe_ratio = mean(
                candidate.keyframe_count / baseline.keyframe_count
                for baseline, candidate in pairs
            )
            ax.scatter(
                time_change,
                psnr_change,
                s=max(30.0, 180.0 * keyframe_ratio),
                color=colors[strategy],
                label=strategy,
                alpha=0.85,
                edgecolor="white",
                linewidth=0.6,
            )
        ax.axhline(0, color="#666666", linewidth=0.7)
        ax.axvline(0, color="#666666", linewidth=0.7)
        ax.set_xlabel(f"{group}: SLAM time change (%)")
        ax.grid(alpha=0.25)
    axes[0, 0].set_ylabel("PSNR change (dB)")
    figure_legend(fig, axes.flat, top=0.90)
    efficiency_pdf = figure_root / "quality-efficiency.pdf"
    efficiency_png = figure_root / "quality-efficiency.png"
    fig.savefig(efficiency_pdf, bbox_inches="tight")
    fig.savefig(efficiency_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [metric_pdf, metric_png, efficiency_pdf, efficiency_png]


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
    parser.add_argument("results_root", type=Path, nargs="?")
    parser.add_argument("--input-csv", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if (args.results_root is None) == (args.input_csv is None):
        parser.error("provide exactly one of results_root or --input-csv")
    if args.input_csv is not None:
        records = load_records_csv(args.input_csv)
    else:
        records = load_latest_successful_runs(args.results_root)
        if not records:
            raise SystemExit(f"no successful experiment records found in {args.results_root}")
        write_outputs(records, args.output_root)
    tables = generate_manuscript_tables(records, args.output_root)
    figures = generate_figures(records, args.output_root)
    print(
        f"processed {len(records)} records; wrote {len(tables)} tables and "
        f"{len(figures)} figures"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
