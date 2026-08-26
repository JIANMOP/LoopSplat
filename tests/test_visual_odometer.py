from types import SimpleNamespace

import numpy as np
import pytest

import src.entities.visual_odometer as odometer_module
from src.entities.visual_odometer import VisualOdometer


class FakeTransformation:
    def __init__(self, matrix):
        self.matrix = np.asarray(matrix, dtype=np.float64)

    def cpu(self):
        return self

    def numpy(self):
        return self.matrix.copy()


def odometry_result(matrix):
    return SimpleNamespace(transformation=FakeTransformation(matrix))


def make_frames():
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    depth = np.ones((2, 2), dtype=np.float32)
    return image, depth


def test_gpu_solver_failure_retries_on_cpu_and_records_diagnostic(
        cuda_device, monkeypatch):
    calls = []
    cpu_transform = np.eye(4)
    cpu_transform[0, 3] = 0.1

    def solve(*args):
        calls.append(args)
        if len(calls) == 1:
            raise RuntimeError("Singular 6x6 linear system")
        return odometry_result(cpu_transform)

    monkeypatch.setattr(
        odometer_module.o3d.t.pipelines.odometry,
        "rgbd_odometry_multi_scale", solve)
    odometer = VisualOdometer(
        np.eye(3), device="cuda", cpu_fallback=True,
        max_translation_m=0.5, max_rotation_deg=60.0)
    previous_image, previous_depth = make_frames()
    image, depth = make_frames()
    odometer.update_last_rgbd(previous_image, previous_depth)

    transform = odometer.estimate_rel_pose(image, depth)

    assert len(calls) == 2
    assert str(calls[0][0].color.device) == "CUDA:0"
    assert str(calls[1][0].color.device) == "CPU:0"
    assert transform[0, 3] == pytest.approx(-0.1)
    assert odometer.last_diagnostic == {
        "source": "cpu_fallback",
        "reason": "primary_solver_error",
        "translation_m": pytest.approx(0.1),
        "rotation_deg": pytest.approx(0.0),
    }
    assert odometer.diagnostics()["cpu_fallback_count"] == 1


def test_motion_outlier_is_recomputed_on_cpu(cuda_device, monkeypatch):
    primary_transform = np.eye(4)
    primary_transform[0, 3] = 2.0
    cpu_transform = np.eye(4)
    cpu_transform[0, 3] = 0.2
    results = iter([
        odometry_result(primary_transform),
        odometry_result(cpu_transform),
    ])
    monkeypatch.setattr(
        odometer_module.o3d.t.pipelines.odometry,
        "rgbd_odometry_multi_scale", lambda *args: next(results))
    odometer = VisualOdometer(
        np.eye(3), device="cuda", cpu_fallback=True,
        max_translation_m=0.5, max_rotation_deg=60.0)
    previous_image, previous_depth = make_frames()
    image, depth = make_frames()
    odometer.update_last_rgbd(previous_image, previous_depth)

    transform = odometer.estimate_rel_pose(image, depth)

    assert transform[0, 3] == pytest.approx(-0.2)
    assert odometer.last_diagnostic["source"] == "cpu_fallback"
    assert odometer.last_diagnostic["reason"] == "primary_motion_outlier"


def test_primary_success_advances_cpu_fallback_reference(
        cuda_device, monkeypatch):
    monkeypatch.setattr(
        odometer_module.o3d.t.pipelines.odometry,
        "rgbd_odometry_multi_scale",
        lambda *args: odometry_result(np.eye(4)))
    odometer = VisualOdometer(
        np.eye(3), device="cuda", cpu_fallback=True,
        max_translation_m=0.5, max_rotation_deg=60.0)
    previous_image, previous_depth = make_frames()
    image, depth = make_frames()
    odometer.update_last_rgbd(previous_image, previous_depth)
    previous_cpu = odometer.cpu_last_rgbd

    odometer.estimate_rel_pose(image, depth)

    assert odometer.cpu_last_rgbd is not previous_cpu


def test_failed_primary_and_cpu_solvers_advance_reference_and_freeze_pose(
        cuda_device, monkeypatch):
    def solve(*args):
        raise RuntimeError("tracking failed")

    monkeypatch.setattr(
        odometer_module.o3d.t.pipelines.odometry,
        "rgbd_odometry_multi_scale", solve)
    odometer = VisualOdometer(
        np.eye(3), device="cuda", cpu_fallback=True,
        max_translation_m=0.5, max_rotation_deg=60.0)
    previous_image, previous_depth = make_frames()
    image, depth = make_frames()
    odometer.update_last_rgbd(previous_image, previous_depth)
    previous_primary = odometer.last_rgbd
    previous_cpu = odometer.cpu_last_rgbd

    transform = odometer.estimate_rel_pose(image, depth)

    np.testing.assert_array_equal(transform, np.eye(4))
    assert odometer.last_rgbd is not previous_primary
    assert odometer.cpu_last_rgbd is not previous_cpu
    assert odometer.last_diagnostic["source"] == "identity_fallback"
    assert odometer.last_diagnostic["reason"] == "cpu_solver_error"
    assert odometer.diagnostics()["identity_fallback_count"] == 1
