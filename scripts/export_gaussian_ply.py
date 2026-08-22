#!/usr/bin/env python3
"""Export the unrefined global Gaussian map stored in submap checkpoints."""

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.entities.arguments import OptimizationParams
from src.entities.gaussian_model import GaussianModel
from src.evaluation.protocol import (
    GAUSSIAN_PARAM_KEYS,
    concatenate_gaussian_params,
)


def export_gaussian_ply(run_dir, output_path=None, force=False):
    run_dir = Path(run_dir)
    submap_paths = sorted((run_dir / "submaps").glob("*.ckpt"))
    if not submap_paths:
        raise FileNotFoundError(f"no submap checkpoints found in {run_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("Gaussian PLY export requires a CUDA GPU")

    output_path = Path(
        output_path or run_dir / "unrefined_global_splats.ply")
    if output_path.exists() and not force:
        raise FileExistsError(
            f"output already exists: {output_path}; use --force to replace it")

    parameter_sets = []
    for path in submap_paths:
        saved_params = torch.load(
            path, map_location="cpu")["gaussian_params"]
        parameter_sets.append({
            key: saved_params[key] for key in GAUSSIAN_PARAM_KEYS})
    merged_params = {
        key: value.cuda()
        for key, value in concatenate_gaussian_params(parameter_sets).items()
    }
    gaussian_model = GaussianModel()
    gaussian_model.restore_from_params(
        merged_params,
        OptimizationParams(argparse.ArgumentParser(description="PLY export")),
    )
    gaussian_model.save_ply(output_path)
    gaussian_count = gaussian_model.get_size()
    print(
        f"Exported {gaussian_count} Gaussians from {len(submap_paths)} "
        f"submaps to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Export an unrefined global Gaussian PLY from a run")
    parser.add_argument("run_dir", help="Formal experiment run directory")
    parser.add_argument(
        "--output", default=None,
        help="Output PLY path (default: <run_dir>/unrefined_global_splats.ply)")
    parser.add_argument(
        "--force", action="store_true", help="Replace an existing PLY")
    args = parser.parse_args()
    export_gaussian_ply(args.run_dir, args.output, args.force)


if __name__ == "__main__":
    main()
