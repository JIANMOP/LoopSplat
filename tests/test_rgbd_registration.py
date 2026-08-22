import numpy as np
import pytest
import cv2

from src.entities.datasets_fm import FMDataset
from src.utils.io_utils import load_config
from src.utils.rgbd_registration import register_depth_to_color


def test_identity_depth_to_color_registration_preserves_depth():
    depth = np.array([[1.0, 0.0], [2.0, 3.0]], dtype=np.float32)

    registered = register_depth_to_color(
        depth_m=depth,
        depth_intrinsics=np.eye(3),
        color_intrinsics=np.eye(3),
        t_color_depth=np.eye(4),
        output_shape=(2, 2),
    )

    np.testing.assert_allclose(registered, depth)


def test_depth_registration_keeps_nearest_surface_on_pixel_collision():
    depth = np.array([[2.0, 1.0]], dtype=np.float32)
    t_color_depth = np.eye(4)
    t_color_depth[0, 3] = -0.6

    registered = register_depth_to_color(
        depth_m=depth,
        depth_intrinsics=np.eye(3),
        color_intrinsics=np.eye(3),
        t_color_depth=t_color_depth,
        output_shape=(1, 2),
    )

    assert registered[0, 0] == pytest.approx(1.0)
    assert registered[0, 1] == 0.0


def test_depth_registration_rejects_non_rigid_transform_shape():
    with pytest.raises(ValueError, match="4x4"):
        register_depth_to_color(
            depth_m=np.ones((2, 2), dtype=np.float32),
            depth_intrinsics=np.eye(3),
            color_intrinsics=np.eye(3),
            t_color_depth=np.eye(3),
            output_shape=(2, 2),
        )


def test_fm_dataset_applies_configured_depth_to_color_registration(tmp_path):
    (tmp_path / "color").mkdir()
    (tmp_path / "depth").mkdir()
    cv2.imwrite(
        str(tmp_path / "color/1000.png"),
        np.zeros((1, 2, 3), dtype=np.uint8),
    )
    cv2.imwrite(
        str(tmp_path / "depth/1000.png"),
        np.array([[2000, 1000]], dtype=np.uint16),
    )
    (tmp_path / "TIMESTAMP.txt").write_text(
        "#timestamp(us) colorname depthname\n1000,1000.png,1000.png\n")
    t_color_depth = np.eye(4)
    t_color_depth[0, 3] = -0.6
    dataset = FMDataset({
        "input_path": str(tmp_path),
        "frame_limit": -1,
        "H": 1,
        "W": 2,
        "fx": 1.0,
        "fy": 1.0,
        "cx": 0.0,
        "cy": 0.0,
        "depth_scale": 1000.0,
        "use_filtered_depth": False,
        "register_depth_to_color": True,
        "depth_intrinsics": np.eye(3).tolist(),
        "T_color_depth": t_color_depth.tolist(),
    })

    _, _, registered_depth, _ = dataset[0]

    np.testing.assert_allclose(registered_depth, [[1.0, 0.0]])


def test_fm_formal_config_registers_raw_depth_into_color_camera():
    config = load_config("configs/FMDataset/dorm1_fast1.yaml")
    config["data"]["frame_limit"] = 1
    dataset = FMDataset({**config["data"], **config["cam"]})
    raw_depth = cv2.imread(
        str(dataset.depth_paths[0]), cv2.IMREAD_UNCHANGED,
    ).astype(np.float32) / dataset.depth_scale

    _, color, registered_depth, _ = dataset[0]

    assert registered_depth.shape == color.shape[:2]
    assert not np.array_equal(registered_depth, raw_depth)
