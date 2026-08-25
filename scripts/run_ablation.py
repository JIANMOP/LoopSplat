#!/usr/bin/env python3
"""
LoopSplat Ablation Experiment Runner
=====================================
Three dataset groups × all strategy combinations.

Group A  TUM RGB-D (no IMU)      5 scenes × 4 combos = 20 configurations
Group R  Replica (no IMU)        8 scenes × 4 combos = 32 configurations
Group C  FMDataset (has IMU)     3 scenes × 6 combos = 18 configurations
                          70 configurations × 3 seeds = 210 formal runs

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
sys.path.insert(0, str(PROJECT_ROOT))
GSR_MAX_ITERS = 100

from src.utils.experiment_utils import (
    ablation_output_root,
    create_run_directory,
    config_sha256,
    current_git_commit,
    discover_completed_runs,
    effective_features_from_config,
    formal_outputs_complete,
    verify_formal_source_state,
    write_manifest,
    write_status,
)


# ── Strategy definitions ─────────────────────────────────────────────

# Group A: TUM RGB-D — no IMU, so only GI-KF and Pyramid apply
STRATEGIES_A = [
    # id_suffix, name, desc, overrides
    ("_0", "Baseline",            "All OFF",
     {"keyframing": {"enable_gi_slam": False},
      "gaussian_pyramid": {"enabled": False}}),
    ("_1", "+GI-KF",              "GI-SLAM keyframe selection ON",
     {"keyframing": {"enable_gi_slam": True, "score_threshold": 0.1,
                     "w_covis": 1.0, "w_base": 1.0, "w_mot": 2.0,
                     "max_keyframe_gap": 10,
                     "high_motion_max_gap": 3},
      "gaussian_pyramid": {"enabled": False}}),
    ("_2", "+Pyramid",            "Photo-SLAM Gaussian Pyramid ON",
     {"keyframing": {"enable_gi_slam": False},
      "gaussian_pyramid": {"enabled": True,
                           "num_sub_levels": 2, "uses_per_level": 8}}),
    ("_3", "+GI-KF+Pyramid",      "GI-KF + Pyramid",
     {"keyframing": {"enable_gi_slam": True, "score_threshold": 0.1,
                     "w_covis": 1.0, "w_base": 1.0, "w_mot": 2.0,
                     "max_keyframe_gap": 10,
                     "high_motion_max_gap": 3},
      "gaussian_pyramid": {"enabled": True,
                           "num_sub_levels": 2, "uses_per_level": 8}}),
]

# Group C: calibrated IMU — all three strategies available
STRATEGIES_BC = [
    ("_0", "Baseline",              "All OFF",
     {"keyframing": {"enable_gi_slam": False},
      "gaussian_pyramid": {"enabled": False},
      "tracking": {"use_imu": False}}),
    ("_1", "+IMU",                  "weak rotation-only IMU prior",
     {"keyframing": {"enable_gi_slam": False},
      "gaussian_pyramid": {"enabled": False},
      "tracking": {"use_imu": True,
                   "lambda_imu_trans": 0.0, "lambda_imu_rot": 0.001}}),
    ("_2", "+GI-KF",                "GI-SLAM keyframe only",
     {"keyframing": {"enable_gi_slam": True, "score_threshold": 0.1,
                     "w_covis": 1.0, "w_base": 1.0, "w_mot": 2.0,
                     "max_keyframe_gap": 10,
                     "high_motion_max_gap": 3},
      "gaussian_pyramid": {"enabled": False},
      "tracking": {"use_imu": False}}),
    ("_3", "+Pyramid",              "Photo-SLAM pyramid only",
     {"keyframing": {"enable_gi_slam": False},
      "gaussian_pyramid": {"enabled": True,
                           "num_sub_levels": 2, "uses_per_level": 8},
      "tracking": {"use_imu": False}}),
    ("_4", "+KF+Pyramid",           "GI-KF + Pyramid (no IMU)",
     {"keyframing": {"enable_gi_slam": True, "score_threshold": 0.1,
                     "w_covis": 1.0, "w_base": 1.0, "w_mot": 2.0,
                     "max_keyframe_gap": 10,
                     "high_motion_max_gap": 3},
      "gaussian_pyramid": {"enabled": True,
                           "num_sub_levels": 2, "uses_per_level": 8},
      "tracking": {"use_imu": False}}),
    ("_5", "+ALL",                  "IMU + GI-KF + Pyramid",
     {"keyframing": {"enable_gi_slam": True, "score_threshold": 0.1,
                     "w_covis": 1.0, "w_base": 1.0, "w_mot": 2.0,
                     "max_keyframe_gap": 10,
                     "high_motion_max_gap": 3},
      "gaussian_pyramid": {"enabled": True,
                           "num_sub_levels": 2, "uses_per_level": 8},
      "tracking": {"use_imu": True,
                   "lambda_imu_trans": 0.0, "lambda_imu_rot": 0.001}}),
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

SCENES_R = [
    ("R1", "Replica office0", "configs/Replica/office0.yaml"),
    ("R2", "Replica office1", "configs/Replica/office1.yaml"),
    ("R3", "Replica office2", "configs/Replica/office2.yaml"),
    ("R4", "Replica office3", "configs/Replica/office3.yaml"),
    ("R5", "Replica office4", "configs/Replica/office4.yaml"),
    ("R6", "Replica room0", "configs/Replica/room0.yaml"),
    ("R7", "Replica room1", "configs/Replica/room1.yaml"),
    ("R8", "Replica room2", "configs/Replica/room2.yaml"),
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

def build_experiments(output_root=None):
    output_root = Path(output_root or ablation_output_root())
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
                    "data": {"output_path": str(output_root / eid)},
                    "lc": {"registration": {
                        "gsr_max_iters": GSR_MAX_ITERS}},
                    **deepcopy(overrides),
                },
            })

    for scene_id, scene_name, config_path in SCENES_R:
        for suffix, sname, sdesc, overrides in STRATEGIES_A:
            eid = scene_id + suffix
            exps.append({
                "id": eid,
                "name": f"{scene_name} — {sname}",
                "desc": f"{sdesc}",
                "config": config_path,
                "group": "R",
                "overrides": {
                    "data": {"output_path": str(output_root / eid)},
                    "lc": {"registration": {
                        "gsr_max_iters": GSR_MAX_ITERS}},
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
                    "data": {"output_path": str(output_root / eid)},
                    "lc": {"registration": {
                        "gsr_max_iters": GSR_MAX_ITERS}},
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
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    inherit = cfg.pop("inherit_from", None)
    if inherit is not None:
        base = load_yaml(PROJECT_ROOT / inherit)
        update_recursive(base, cfg)
        cfg = base
    return cfg


def config_has_results(output_dir: Path, seed: int, config: dict,
                       source_fingerprint: str) -> bool:
    records = discover_completed_runs(output_dir.parent)
    return any(
        record.experiment_id == output_dir.name
        and record.seed == seed
        and record.manifest.get("config_sha256") == config_sha256(config)
        and record.manifest.get(
            "experiment_source_sha256") == source_fingerprint
        for record in records)


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LoopSplat Ablation Runner")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[0, 1, 2],
        help="Random seeds for repeated formal runs (default: 0 1 2)")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Run single experiment, e.g. A1_0")
    parser.add_argument("--group", type=str, default=None,
                        help="Run all experiments in group A/R/C")
    args = parser.parse_args()

    available_experiments = build_experiments()
    experiments = available_experiments
    if args.experiment:
        experiments = [e for e in experiments if e["id"] == args.experiment]
        if not experiments:
            print(f"Error: no experiment '{args.experiment}'")
            ids = [e["id"] for e in available_experiments]
            print(f"Available ({len(ids)}): {', '.join(ids[:20])}...")
            sys.exit(1)
    elif args.group:
        experiments = [e for e in experiments if e["group"] == args.group]
        if not experiments:
            print(f"Error: no group '{args.group}'")
            sys.exit(1)

    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must contain unique integers")
    jobs = [
        (experiment, seed)
        for experiment in experiments
        for seed in args.seeds
    ]
    total = len(jobs)
    completed = skipped = failed = 0
    start_time = time.time()
    frozen_commit = current_git_commit()
    if not frozen_commit:
        raise RuntimeError("formal runs require a Git commit")
    frozen_source_fingerprint = (
        None if args.dry_run else verify_formal_source_state())

    for i, (exp, seed) in enumerate(jobs):
        eid = exp["id"]
        ename = exp["name"]
        edesc = exp["desc"]

        base_config = load_yaml(exp["config"])
        merged = deep_merge(base_config, exp["overrides"])
        merged["use_wandb"] = False
        merged["formal_experiment"] = True
        merged["seed"] = seed
        merged["experiment_id"] = eid
        requested_output = Path(merged["data"]["output_path"])

        print(f"\n{'='*70}")
        print(f"[{i+1}/{total}] {eid} seed={seed} — {ename}")
        print(f"      {edesc}")
        print(f"      Output root: {requested_output}")
        print(f"{'='*70}")

        if args.dry_run:
            continue

        verify_formal_source_state(frozen_source_fingerprint)

        if (not args.force and config_has_results(
                requested_output, seed, merged, frozen_source_fingerprint)):
            print(f"      ⏭  Skipped (results exist)")
            skipped += 1
            continue

        runner = "run_slam_azure.py" if merged.get("dataset_name") == "azure_kinect" else "run_slam.py"
        run_dir = create_run_directory(
            requested_output.parent, eid, seed)
        merged["data"]["output_path"] = str(run_dir)
        merged["data"]["run_directory_prepared"] = True
        tmp_config = run_dir / "config.input.yaml"

        with open(tmp_config, "w", encoding="utf-8") as f:
            yaml.dump(merged, f, default_flow_style=False)

        command = [sys.executable, runner, str(tmp_config)]
        features = effective_features_from_config(merged)
        write_manifest(run_dir, merged, command, features)
        write_status(run_dir, "running")

        print(f"      Running {runner} ...")
        t0 = time.time()
        try:
            with open(
                    run_dir / "run.log", "w", buffering=1,
                    encoding="utf-8") as log_file:
                process = subprocess.Popen(
                    command, cwd=PROJECT_ROOT, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, bufsize=1,
                    encoding="utf-8", errors="replace")
                for line in process.stdout:
                    print(line, end="")
                    log_file.write(line)
                return_code = process.wait()
            elapsed = time.time() - t0
            write_manifest(run_dir, merged, command, features)
            if return_code == 0 and formal_outputs_complete(run_dir):
                write_status(run_dir, "succeeded", elapsed_seconds=elapsed)
                print(f"      ✅ Completed in {elapsed:.0f}s")
                completed += 1
            else:
                reason = (
                    "missing formal evaluation outputs"
                    if return_code == 0 else f"exit {return_code}")
                write_status(
                    run_dir, "failed", elapsed_seconds=elapsed,
                    reason=reason)
                print(f"      ❌ Failed ({reason}) in {elapsed:.0f}s")
                failed += 1
        except Exception as error:
            write_status(
                run_dir, "failed", elapsed_seconds=time.time() - t0,
                reason=repr(error))
            print(f"      ❌ Failed: {error}")
            failed += 1

    total_elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"SUMMARY: {completed}/{total} done, {skipped} skipped, "
          f"{failed} failed in {total_elapsed:.0f}s")
    print(f"{'='*70}")
    if not args.dry_run and completed > 0:
        print("\n  Aggregate results: python scripts/aggregate_results.py")


if __name__ == "__main__":
    main()
