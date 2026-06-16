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


def find_result_dir(base: Path) -> Path | None:
    if not base.exists():
        return None
    dirs = sorted([d for d in base.iterdir() if d.is_dir() and d.name[0].isdigit()], reverse=True)
    return dirs[0] if dirs else None


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def collect_results() -> dict:
    results = {}
    base = PROJECT_ROOT / "output" / "ablation"

    for sid in SCENE_IDS:
        suffixes = STRATEGY_SUFFIXES_A if sid.startswith("A") else STRATEGY_SUFFIXES_BC
        for suffix in suffixes:
            eid = sid + suffix
            exp_dir = base / eid
            rd = find_result_dir(exp_dir)
            if rd is None:
                results[eid] = {"error": "no results"}
                continue

            labels = STRATEGY_LABELS_A if sid.startswith("A") else STRATEGY_LABELS_BC
            m = {"scene": SCENE_LABELS.get(sid, sid),
                 "strategy": labels.get(suffix, suffix)}

            ate = read_json(rd / "ate_aligned.json")
            if ate:
                m["ate_rmse_cm"] = round(ate.get("rmse", 0) * 100, 2)

            render = read_json(rd / "rendering_metrics.json")
            if render:
                m["psnr"] = round(render.get("psnr", 0), 2)
                m["ssim"] = round(render.get("ssim", 0), 4)
                m["lpips"] = round(render.get("lpips", 0), 4)
                m["depth_l1"] = round(render.get("depth_l1_train_view", 0), 4)

            results[eid] = m

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
        suffixes = STRATEGY_SUFFIXES_A if sid.startswith("A") else STRATEGY_SUFFIXES_BC
        slabels = STRATEGY_LABELS_A if sid.startswith("A") else STRATEGY_LABELS_BC
        cols = ["Exp"]
        if show_ate:
            cols.append("ATE↓")
        cols += ["PSNR↑", "SSIM↑", "LPIPS↓", "Depth L1↓"]
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
                    row.append(_fmt(r.get("ate_rmse_cm", "—")))
                row += [_fmt(r.get("psnr", "—")),
                        _fmt(r.get("ssim", "—"), 4),
                        _fmt(r.get("lpips", "—"), 4),
                        _fmt(r.get("depth_l1", "—"), 4)]
            rows.append("| " + " | ".join(row) + " |")
        return "\n".join(rows)

    # ── Group A: TUM ──
    lines.append("## Group A — TUM RGB-D (no IMU)\n")
    for sid in ["A1","A2","A3","A4","A5"]:
        lines.append(f"### {sid}: {SCENE_LABELS[sid]}\n")
        lines.append(scene_table(sid, show_ate=True))
        lines.append("")

    # ── Group B: AzureKinect ──
    lines.append("## Group B — AzureKinect (has IMU)\n")
    for sid in ["B1"]:
        lines.append(f"### {sid}: {SCENE_LABELS[sid]}\n")
        lines.append(scene_table(sid, show_ate=True))
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
        suffixes = STRATEGY_SUFFIXES_A if sid.startswith("A") else STRATEGY_SUFFIXES_BC
        slabels = STRATEGY_LABELS_A if sid.startswith("A") else STRATEGY_LABELS_BC
        for suffix in suffixes:
            eid = sid + suffix
            r = results.get(eid, {})
            label = slabels.get(suffix, suffix)
            if "error" in r:
                lines.append(f"  {label:14s}  ❌")
            else:
                psnr = _fmt(r.get("psnr", "?"))
                ate = _fmt(r.get("ate_rmse_cm", "?"))
                lines.append(f"  {label:14s}  ATE={ate}cm  PSNR={psnr}")
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
