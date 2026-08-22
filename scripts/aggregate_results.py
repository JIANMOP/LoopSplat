#!/usr/bin/env python3
"""
LoopSplat Ablation Results Aggregator
======================================
Scans output/ablation/ directories, collects metrics, renders tables.

Usage:
  python scripts/aggregate_results.py --format markdown
  python scripts/aggregate_results.py --format terminal
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.protocol import assert_compatible_protocols
from src.utils.experiment_utils import discover_completed_runs

# Scene labels for display
SCENE_LABELS = {
    "A1": "TUM fr1/desk", "A2": "TUM fr1/desk2", "A3": "TUM fr1/room",
    "A4": "TUM fr2/xyz", "A5": "TUM fr3/long",
    "B1": "Azure 144_5FPS_720p_IMU",
    "C1": "FM dorm1_fast1", "C2": "FM dorm2_fast", "C3": "FM hotel_fast1",
}

STRATEGY_LABELS_A = {
    "_0": "Baseline", "_1": "+GI-KF", "_2": "+Pyramid", "_3": "+KF+Pyramid",
}
STRATEGY_LABELS_BC = {
    "_0": "Baseline", "_1": "+IMU", "_2": "+GI-KF",
    "_3": "+Pyramid", "_4": "+KF+Pyramid", "_5": "+ALL",
}

# Which experiments belong to each scene (built from scene + strategy)
SCENE_IDS = ["A1","A2","A3","A4","A5","B1","C1","C2","C3"]
STRATEGY_SUFFIXES_A = ["_0","_1","_2","_3"]
STRATEGY_SUFFIXES_BC = ["_0","_1","_2","_3","_4","_5"]


def strategy_spec(scene_id):
    if scene_id.startswith(("A", "B")):
        return STRATEGY_SUFFIXES_A, STRATEGY_LABELS_A
    return STRATEGY_SUFFIXES_BC, STRATEGY_LABELS_BC


def find_result_dir(base: Path) -> Path | None:
    records = [
        record for record in discover_completed_runs(base.parent)
        if record.experiment_id == base.name]
    return records[-1].path if records else None


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def read_trajectory_metrics(status, ate_aligned, trajectory_metrics):
    if not status or status.get("status") != "available":
        return {}
    if not ate_aligned or not trajectory_metrics:
        return {}
    rpe = trajectory_metrics.get("rpe_consecutive", {})
    result = {
        "ate_rmse_cm": round(ate_aligned["rmse"] * 100, 2),
        "trajectory_alignment": trajectory_metrics.get("alignment_mode"),
    }
    if rpe.get("translation_rmse_m") is not None:
        result["rpe_translation_cm"] = round(
            rpe["translation_rmse_m"] * 100, 2)
    if rpe.get("rotation_rmse_deg") is not None:
        result["rpe_rotation_deg"] = round(
            rpe["rotation_rmse_deg"], 3)
    result["rpe_valid_pairs"] = rpe.get("valid_pairs", 0)
    return result


def collect_results() -> dict:
    results = {}
    protocols_by_scene = {scene_id: [] for scene_id in SCENE_IDS}
    base = PROJECT_ROOT / "output" / "ablation"

    for sid in SCENE_IDS:
        suffixes, labels = strategy_spec(sid)
        for suffix in suffixes:
            eid = sid + suffix
            exp_dir = base / eid
            rd = find_result_dir(exp_dir)
            if rd is None:
                results[eid] = {"error": "no results"}
                continue

            m = {"scene": SCENE_LABELS.get(sid, sid),
                 "strategy": labels.get(suffix, suffix)}
            status = read_json(rd / "status.json") or {}
            for field in (
                    "keyframe_count", "submap_count",
                    "slam_elapsed_seconds"):
                if status.get(field) is not None:
                    m[field] = status[field]
            peak_bytes = status.get("slam_peak_gpu_memory_bytes")
            if peak_bytes is not None:
                m["slam_peak_gpu_memory_gib"] = peak_bytes / (1024 ** 3)

            protocol = read_json(rd / "evaluation_protocol.json")
            if protocol is None:
                results[eid] = {"error": "missing formal evaluation protocol"}
                continue
            protocols_by_scene[sid].append(protocol)

            ate = read_json(rd / "ate_aligned.json")
            trajectory = read_json(rd / "trajectory_status.json")
            trajectory_metrics = read_json(rd / "trajectory_metrics.json")
            m.update(read_trajectory_metrics(
                trajectory, ate, trajectory_metrics))

            render = read_json(rd / "rendering_metrics_observed_view.json")
            if render:
                m["psnr"] = round(render.get("psnr", 0), 2)
                m["ssim"] = round(render.get("ssim", 0), 4)
                m["lpips"] = round(render.get("lpips", 0), 4)
                depth_l1 = render.get("depth_l1_observed_view")
                if depth_l1 is not None:
                    m["depth_l1"] = round(depth_l1, 4)

            results[eid] = m

    for protocols in protocols_by_scene.values():
        assert_compatible_protocols(protocols)
    return results


def _fmt(v, precision=2):
    if isinstance(v, (int, float)):
        return f"{v:.{precision}f}"
    return str(v)


def render_markdown(results: dict) -> str:
    lines = ["# LoopSplat Ablation Results\n"]
    lines.append("_Auto-generated_\n")

    # ── Helper: render one scene table ──
    def scene_table(sid, show_ate=True):
        suffixes, slabels = strategy_spec(sid)
        cols = ["Exp"]
        if show_ate:
            cols += ["ATE↓", "RPE-t↓", "RPE-R↓"]
        cols += [
            "PSNR↑", "SSIM↑", "LPIPS↓", "Depth L1↓",
            "KF↓", "Submaps↓", "SLAM s↓", "Peak GiB↓",
        ]
        header = "| " + " | ".join(cols) + " |"
        sep = "|" + "|".join(["---"] * len(cols)) + "|"
        rows = [header, sep]
        for suffix in suffixes:
            eid = sid + suffix
            r = results.get(eid, {})
            label = slabels.get(suffix, suffix)
            if "error" in r:
                row = [label] + ["—"] * (len(cols) - 1)
            else:
                row = [label]
                if show_ate:
                    row += [
                        _fmt(r.get("ate_rmse_cm", "—")),
                        _fmt(r.get("rpe_translation_cm", "—")),
                        _fmt(r.get("rpe_rotation_deg", "—"), 3),
                    ]
                row += [_fmt(r.get("psnr", "—")),
                        _fmt(r.get("ssim", "—"), 4),
                        _fmt(r.get("lpips", "—"), 4),
                        _fmt(r.get("depth_l1", "—"), 4),
                        _fmt(r.get("keyframe_count", "—"), 0),
                        _fmt(r.get("submap_count", "—"), 0),
                        _fmt(r.get("slam_elapsed_seconds", "—"), 2),
                        _fmt(r.get("slam_peak_gpu_memory_gib", "—"), 2)]
            rows.append("| " + " | ".join(row) + " |")
        return "\n".join(rows)

    # ── Group A: TUM ──
    lines.append("## Group A — TUM RGB-D (no IMU)\n")
    for sid in ["A1","A2","A3","A4","A5"]:
        lines.append(f"### {sid}: {SCENE_LABELS[sid]}\n")
        lines.append(scene_table(sid, show_ate=True))
        lines.append("")

    # ── Group B: AzureKinect ──
    lines.append(
        "## Group B — AzureKinect (uncalibrated IMU, no GT poses)\n")
    for sid in ["B1"]:
        lines.append(f"### {sid}: {SCENE_LABELS[sid]}\n")
        lines.append(scene_table(sid, show_ate=False))
        lines.append("")

    # ── Group C: FMDataset ──
    lines.append("## Group C — FMDataset (has IMU, no GT poses)\n")
    for sid in ["C1","C2","C3"]:
        lines.append(f"### {sid}: {SCENE_LABELS[sid]}\n")
        lines.append(scene_table(sid, show_ate=False))
        lines.append("")

    return "\n".join(lines)


def render_terminal(results: dict) -> str:
    lines = []
    for sid in SCENE_IDS:
        lines.append(f"\n{sid} — {SCENE_LABELS.get(sid,sid)}")
        suffixes, slabels = strategy_spec(sid)
        for suffix in suffixes:
            eid = sid + suffix
            r = results.get(eid, {})
            label = slabels.get(suffix, suffix)
            if "error" in r:
                lines.append(f"  {label:14s}  ❌")
            else:
                psnr = _fmt(r.get("psnr", "?"))
                ate = _fmt(r.get("ate_rmse_cm", "?"))
                keyframes = _fmt(r.get("keyframe_count", "?"), 0)
                elapsed = _fmt(r.get("slam_elapsed_seconds", "?"), 2)
                lines.append(
                    f"  {label:14s}  ATE={ate}cm  PSNR={psnr}  "
                    f"KF={keyframes}  SLAM={elapsed}s")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["markdown","terminal","json"], default="markdown")
    args = parser.parse_args()

    print("Scanning...", file=sys.stderr)
    results = collect_results()
    found = sum(1 for v in results.values() if "error" not in v)
    print(f"Found {found}/{len(results)} results.\n", file=sys.stderr)

    if args.format == "json":
        json.dump(results, sys.stdout, indent=2, default=str)
    elif args.format == "terminal":
        print(render_terminal(results))
    else:
        print(render_markdown(results))


if __name__ == "__main__":
    main()
