from pathlib import Path
import subprocess
import sys

from plyfile import PlyData
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def gaussian_params(x):
    return {
        "xyz": torch.tensor([[x, 0.0, 0.0]]),
        "features_dc": torch.ones(1, 1, 3),
        "features_rest": torch.empty(1, 0, 3),
        "opacity": torch.tensor([[0.1]]),
        "scaling": torch.tensor([[0.2, 0.2, 0.2]]),
        "rotation": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    }


def test_export_script_rebuilds_unrefined_global_splats_from_checkpoints(
        tmp_path):
    submaps = tmp_path / "submaps"
    submaps.mkdir()
    torch.save(
        {"gaussian_params": gaussian_params(1.0), "submap_keyframes": [0]},
        submaps / "000000.ckpt")
    torch.save(
        {"gaussian_params": gaussian_params(2.0), "submap_keyframes": [1]},
        submaps / "000001.ckpt")

    result = subprocess.run(
        [sys.executable, "scripts/export_gaussian_ply.py", str(tmp_path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    output_path = tmp_path / "unrefined_global_splats.ply"
    assert result.returncode == 0, result.stdout + result.stderr
    assert output_path.is_file()
    ply = PlyData.read(output_path)
    assert len(ply["vertex"]) == 2
    assert list(ply["vertex"]["x"]) == [1.0, 2.0]

    refused = subprocess.run(
        [sys.executable, "scripts/export_gaussian_ply.py", str(tmp_path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "output already exists" in refused.stderr

    forced = subprocess.run(
        [sys.executable, "scripts/export_gaussian_ply.py", str(tmp_path),
         "--force"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert forced.returncode == 0, forced.stdout + forced.stderr
