from pathlib import Path
import json

import cv2
import numpy as np
import pytest

from src.entities.datasets import TUM_RGBD
from src.entities.datasets_azure import AzureKinect
from src.entities.datasets_fm import FMDataset
from src.entities.gaussian_slam import build_dataset_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_top_level_frame_limit_reaches_dataset_loader():
    config = {
        "frame_limit": 8,
        "data": {"input_path": "data/example"},
        "cam": {"H": 10, "W": 20},
    }

    dataset_config = build_dataset_config(config)

    assert dataset_config["frame_limit"] == 8
    assert dataset_config["input_path"] == "data/example"


@pytest.fixture(scope="module")
def fm_dataset():
    return FMDataset({
        "input_path": str(
            PROJECT_ROOT / "data/FMDataset/dorm1/dorm1_fast1"),
        "frame_limit": 3,
        "H": 480,
        "W": 640,
        "fx": 608.446,
        "fy": 607.847,
        "cx": 331.332,
        "cy": 245.549,
        "depth_scale": 1000.0,
        "use_filtered_depth": False,
    })


@pytest.fixture(scope="module")
def tum_dataset():
    return TUM_RGBD({
        "input_path": str(
            PROJECT_ROOT
            / "data/TUM_RGBD-SLAM/rgbd_dataset_freiburg1_desk"),
        "frame_limit": 3,
        "H": 480,
        "W": 640,
        "fx": 517.3,
        "fy": 516.5,
        "cx": 318.6,
        "cy": 255.3,
        "crop_edge": 50,
        "distortion": [0.2624, -0.9531, -0.0054, 0.0026, 1.1633],
        "depth_scale": 5000.0,
    })


def test_fm_converts_microsecond_timestamps_to_seconds(fm_dataset):
    assert fm_dataset.timestamps[0] == pytest.approx(0.755517)
    assert fm_dataset.timestamps[1] == pytest.approx(0.788792)
    assert fm_dataset.timestamps[1] - fm_dataset.timestamps[0] == pytest.approx(
        0.033275, abs=1e-9)
    assert fm_dataset.imu_data[0]["timestamp"] == pytest.approx(0.001876)


def test_fm_interval_uses_all_samples_and_interpolates_frame_boundaries(
        fm_dataset):
    interval = fm_dataset.get_imu_measurements(0, 1)

    assert interval.valid
    assert interval.dt_s == pytest.approx(0.033275, abs=1e-9)
    assert interval.timestamps_s[0] == pytest.approx(0.755517)
    assert interval.timestamps_s[-1] == pytest.approx(0.788792)
    assert np.all(np.diff(interval.timestamps_s) > 0)
    assert len(interval.timestamps_s) > 2
    assert interval.accelerations.shape == (len(interval.timestamps_s), 3)
    assert interval.angular_velocities.shape == (len(interval.timestamps_s), 3)


def test_fm_rejects_reversed_frame_interval(fm_dataset):
    interval = fm_dataset.get_imu_measurements(1, 0)

    assert interval.valid is False
    assert interval.reason == "frame_order"


def test_datasets_declare_ground_truth_capability(fm_dataset, tum_dataset):
    assert fm_dataset.has_ground_truth is False
    assert tum_dataset.has_ground_truth is True


def test_tum_preserves_associated_rgb_timestamps_in_seconds(tum_dataset):
    assert len(tum_dataset.timestamps) == len(tum_dataset.color_paths)
    assert np.all(np.diff(tum_dataset.timestamps) > 0)
    assert 0.02 < tum_dataset.timestamps[1] - tum_dataset.timestamps[0] < 0.05


def test_azure_requires_and_applies_explicit_timestamp_unit(tmp_path):
    (tmp_path / "color").mkdir()
    (tmp_path / "depth").mkdir()
    cv2.imwrite(
        str(tmp_path / "color/0.png"),
        np.zeros((1, 2, 3), dtype=np.uint8),
    )
    cv2.imwrite(
        str(tmp_path / "depth/0.png"),
        np.full((1, 2), 1000, dtype=np.uint16),
    )
    (tmp_path / "frame_info.json").write_text(json.dumps({
        "total_frames": 2,
        "frames": [
            {
                "color_path": "color/0.png",
                "depth_path": "depth/0.png",
                "timestamp": 1_000_000,
            },
            {
                "color_path": "color/0.png",
                "depth_path": "depth/0.png",
                "timestamp": 1_010_000,
            },
        ],
    }))
    (tmp_path / "imu.txt").write_text(
        "1000000 0 0 9.81 0 0 0\n"
        "1005000 0 0 9.81 0 0 0.1\n"
        "1010000 0 0 9.81 0 0 0.2\n")
    camera = {
        "H": 1, "W": 2, "fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0,
        "distortion": [0.0] * 8,
    }
    dataset = AzureKinect({
        "input_path": str(tmp_path),
        "frame_limit": -1,
        "H": 1,
        "W": 2,
        "fx": 1.0,
        "fy": 1.0,
        "cx": 0.0,
        "cy": 0.0,
        "depth_scale": 1000.0,
        "crop_edge": 0,
        "resize": "rergb",
        "preprocessing_strategy": "resize_only",
        "use_k4a_transformation": False,
        "color_camera": camera,
        "depth_camera": camera,
        "timestamp_unit": "us",
    })

    assert dataset.timestamps == pytest.approx([1.0, 1.01])
    assert dataset.imu_data[0]["timestamp"] == pytest.approx(1.0)
    interval = dataset.get_imu_measurements(0, 1)
    assert interval.valid
    assert interval.dt_s == pytest.approx(0.01)
    assert interval.timestamps_s == pytest.approx([1.0, 1.005, 1.01])


def test_azure_rejects_missing_timestamp_unit(tmp_path):
    dataset = AzureKinect.__new__(AzureKinect)
    with pytest.raises(ValueError, match="timestamp_unit"):
        dataset._timestamp_scale({})
