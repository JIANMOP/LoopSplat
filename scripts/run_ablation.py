#!/usr/bin/env python3
"""
LoopSplat Ablation Experiment Runner
=====================================
Three dataset groups × all strategy combinations.

Group A  TUM RGB-D (no IMU)     5 scenes × 4 combos = 20 experiments
Group B  AzureKinect (has IMU)   1 scene  × 6 combos =  6 experiments
Group C  FMDataset (has IMU)     3 scenes × 6 combos = 18 experiments
                                                  Total = 44 experiments

Strategy codes per experiment:
  _0 = Baseline (all off)
  _1-5 = see STRATEGIES_BY_GROUP below

Usage:
  python scripts/run_ablation.py --dry-run
  python scripts/run_ablation.py --experiment A1_0
  python scripts/run_ablation.py --group A
"""

import argparse
import os
import subprocess
import sys
import time
import yaml
from datetime import datetime
from pathlib import Path
from copy import deepcopy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)


# ── Strategy definitions ─────────────────────────────────────────────

# Group A: TUM RGB-D — no IMU, so only GI-KF and Pyramid apply
STRATEGIES_A = [
    # id_suffix, name, desc, overrides
    ("_0", "Baseline",            "All OFF",
     {"keyframing": {"enable_gi_slam": False},
      "gaussian_pyramid": {"enabled": False}}),
    ("_1", "+GI-KF",              "GI-SLAM keyframe selection ON",
     {"keyframing": {"enable_gi_slam": True, "score_threshold": 0.5,
                     "w_covis": 1.0, "w_base": 1.0, "w_mot": 2.0},
      "gaussian_pyramid": {"enabled": False}}),
    ("_2", "+Pyramid",            "Photo-SLAM Gaussian Pyramid ON",
     {"keyframing": {"enable_gi_slam": False},
      "gaussian_pyramid": {"enabled": True,
                           "num_sub_levels": 2, "uses_per_level": 8}}),
    ("_3", "+GI-KF+Pyramid",      "GI-KF + Pyramid",
     {"keyframing": {"enable_gi_slam": True, "score_threshold": 0.5,
                     "w_covis": 1.0, "w_base": 1.0, "w_mot": 2.0},
      "gaussian_pyramid": {"enabled": True,
                           "num_sub_levels": 2, "uses_per_level": 8}}),
]

# Group B/C: has IMU — all three strategies available
STRATEGIES_BC = [
    ("_0", "Baseline",              "All OFF",
     {"keyframing": {"enable_gi_slam": False},
      "gaussian_pyramid": {"enabled": False},
      "tracking": {"use_imu": False}}),
    ("_1", "+IMU",                  "IMU loss only",
     {"keyframing": {"enable_gi_slam": False},
      "gaussian_pyramid": {"enabled": False},
      "tracking": {"use_imu": True,
                   "lambda_imu_trans": 0.01, "lambda_imu_rot": 0.01}}),
    ("_2", "+GI-KF",                "GI-SLAM keyframe only",
     {"keyframing": {"enable_gi_slam": True, "score_threshold": 0.5,
                     "w_covis": 1.0, "w_base": 1.0, "w_mot": 2.0},
      "gaussian_pyramid": {"enabled": False},
      "tracking": {"use_imu": False}}),
    ("_3", "+Pyramid",              "Photo-SLAM pyramid only",
     {"keyframing": {"enable_gi_slam": False},
      "gaussian_pyramid": {"enabled": True,
                           "num_sub_levels": 2, "uses_per_level": 8},
      "tracking": {"use_imu": False}}),
    ("_4", "+KF+Pyramid",           "GI-KF + Pyramid (no IMU)",
     {"keyframing": {"enable_gi_slam": True, "score_threshold": 0.5,
                     "w_covis": 1.0, "w_base": 1.0, "w_mot": 2.0},
      "gaussian_pyramid": {"enabled": True,
                           "num_sub_levels": 2, "uses_per_level": 8},
      "tracking": {"use_imu": False}}),
    ("_5", "+ALL",                  "IMU + GI-KF + Pyramid",
     {"keyframing": {"enable_gi_slam": True, "score_threshold": 0.5,
                     "w_covis": 1.0, "w_base": 1.0, "w_mot": 2.0},
      "gaussian_pyramid": {"enabled": True,
                           "num_sub_levels": 2, "uses_per_level": 8},
      "tracking": {"use_imu": True,
                   "lambda_imu_trans": 0.01, "lambda_imu_rot": 0.01}}),
]


# ── Scene definitions ─────────────────────────────────────────────────

SCENES_A = [
    ("A1", "TUM fr1/desk",
     "configs/TUM_RGBD/rgbd_dataset_freiburg1_desk.yaml"),
    ("A2", "TUM fr1/desk2",
     "configs/TUM_RGBD/rgbd_dataset_freiburg1_desk2.yaml"),
    ("A3", "TUM fr1/room",
     "configs/TUM_RGBD/rgbd_dataset_freiburg1_room.yaml"),
    ("A4", "TUM fr2/xyz",
     "configs/TUM_RGBD/rgbd_dataset_freiburg2_xyz.yaml"),
    ("A5", "TUM fr3/long",
     "configs/TUM_RGBD/rgbd_dataset_freiburg3_long_office_household.yaml"),
]

SCENES_B = [
    ("B1", "Azure 144_5FPS_720p_IMU",
     "configs/AzureKinect/144_5FPS_720p_IMU.yaml"),
]

SCENES_C = [
    ("C1", "FM dorm1_fast1",
     "configs/FMDataset/dorm1_fast1.yaml"),
    ("C2", "FM dorm2_fast",
     "configs/FMDataset/dorm2_fast.yaml"),
    ("C3", "FM hotel_fast1",
     "configs/FMDataset/hotel_fast1.yaml"),
]


# ── Build experiment list ─────────────────────────────────────────────

def build_experiments():
    exps = []

    for scene_id, scene_name, config_path in SCENES_A:
        for suffix, sname, sdesc, overrides in STRATEGIES_A:
            eid = scene_id + suffix
            exps.append({
                "id": eid,
                "name": f"{scene_name} — {sname}",
                "desc": f"{sdesc}",
                "config": config_path,
                "group": "A",
                "overrides": {
                    "data": {"output_path": f"output/ablation/{eid}/"},
                    **deepcopy(overrides),
                },
            })

    for scene_id, scene_name, config_path in SCENES_B:
        for suffix, sname, sdesc, overrides in STRATEGIES_BC:
            eid = scene_id + suffix
            exps.append({
                "id": eid,
                "name": f"{scene_name} — {sname}",
                "desc": f"{sdesc}",
                "config": config_path,
                "group": "B",
                "overrides": {
                    "data": {"output_path": f"output/ablation/{eid}/"},
                    **deepcopy(overrides),
                },
            })

    for scene_id, scene_name, config_path in SCENES_C:
        for suffix, sname, sdesc, overrides in STRATEGIES_BC:
            eid = scene_id + suffix
            exps.append({
                "id": eid,
                "name": f"{scene_name} — {sname}",
                "desc": f"{sdesc}",
                "config": config_path,
                "group": "C",
                "overrides": {
                    "data": {"output_path": f"output/ablation/{eid}/"},
                    **deepcopy(overrides),
                },
            })

    return exps


EXPERIMENTS = build_experiments()


# ── Helpers ───────────────────────────────────────────────────────────

def deep_merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = deepcopy(v) if isinstance(v, dict) else v
    return result


def update_recursive(d1: dict, d2: dict) -> None:
    for k, v in d2.items():
        if k not in d1:
            d1[k] = {}
        if isinstance(v, dict):
            update_recursive(d1[k], v)
        else:
            d1[k] = v


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    inherit = cfg.pop("inherit_from", None)
    if inherit is not None:
        base = load_yaml(PROJECT_ROOT / inherit)
        update_recursive(base, cfg)
        cfg = base
    return cfg


def config_has_results(output_dir: Path) -> bool:
    if not output_dir.exists():
        return False
    subdirs = sorted(
        [d for d in output_dir.iterdir() if d.is_dir() and d.name[0].isdigit()],
        reverse=True,
    )
    if not subdirs:
        return False
    return (subdirs[0] / "ate_aligned.json").exists() or \
           (subdirs[0] / "rendering_metrics.json").exists()


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LoopSplat Ablation Runner")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Run single experiment, e.g. A1_0")
    parser.add_argument("--group", type=str, default=None,
                        help="Run all experiments in group A/B/C")
    args = parser.parse_args()

    experiments = EXPERIMENTS
    if args.experiment:
        experiments = [e for e in experiments if e["id"] == args.experiment]
        if not experiments:
            print(f"Error: no experiment '{args.experiment}'")
            ids = [e["id"] for e in EXPERIMENTS]
            print(f"Available ({len(ids)}): {', '.join(ids[:20])}...")
            sys.exit(1)
    elif args.group:
        experiments = [e for e in experiments if e["group"] == args.group]
        if not experiments:
            print(f"Error: no group '{args.group}'")
            sys.exit(1)

    total = len(experiments)
    completed = skipped = failed = 0
    start_time = time.time()

    for i, exp in enumerate(experiments):
        eid = exp["id"]
        ename = exp["name"]
        edesc = exp["desc"]

        base_config = load_yaml(exp["config"])
        merged = deep_merge(base_config, exp["overrides"])
        merged["use_wandb"] = False
        output_base = Path(merged["data"]["output_path"])

        print(f"\n{'='*70}")
        print(f"[{i+1}/{total}] {eid} — {ename}")
        print(f"      {edesc}")
        print(f"      Output: {output_base}")
        print(f"{'='*70}")

        if args.dry_run:
            continue

        if not args.force and config_has_results(output_base):
            print(f"      ⏭  Skipped (results exist)")
            skipped += 1
            continue

        runner = "run_slam_azure.py" if merged.get("dataset_name") == "azure_kinect" else "run_slam.py"
        tmp_config = Path(f"_ablation_{eid}.yaml")

        with open(tmp_config, "w") as f:
            yaml.dump(merged, f, default_flow_style=False)

        print(f"      Running {runner} ...")
        t0 = time.time()
        try:
            result = subprocess.run(
                [sys.executable, runner, str(tmp_config)],
                cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=21600)
            elapsed = time.time() - t0
            if result.returncode == 0:
                print(f"      ✅ Completed in {elapsed:.0f}s")
                completed += 1
            else:
                print(f"      ❌ Failed (exit {result.returncode}) in {elapsed:.0f}s")
                for line in result.stderr.strip().split("\n")[-8:]:
                    print(f"        {line}")
                failed += 1
        except subprocess.TimeoutExpired:
            print(f"      ❌ Timeout after {time.time()-t0:.0f}s")
            failed += 1
        finally:
            if tmp_config.exists():
                tmp_config.unlink()

    total_elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"SUMMARY: {completed}/{total} done, {skipped} skipped, "
          f"{failed} failed in {total_elapsed:.0f}s")
    print(f"{'='*70}")
    if not args.dry_run and completed > 0:
        print("\n  Aggregate results: python scripts/aggregate_results.py")


if __name__ == "__main__":
    main()
