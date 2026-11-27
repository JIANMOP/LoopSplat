import importlib.util
from pathlib import Path

import numpy as np
import pytest


def geometry_module():
    path = Path(__file__).parents[1] / 'scripts/evaluate_revision_geometry.py'
    spec = importlib.util.spec_from_file_location('revision_geometry', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_surface_distances_use_metres_and_empty_predictions_are_not_success():
    module = geometry_module()
    points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    same = module.surface_metrics(points, points, [0.005, 0.02])
    assert same['accuracy_m'] == 0
    assert same['completeness_m'] == 0
    assert same['thresholds']['0.005']['fscore'] == 1
    shifted = module.surface_metrics(points + [0., 0., 0.01], points, [0.005, 0.02])
    assert shifted['accuracy_m'] == pytest.approx(0.01)
    assert shifted['completeness_m'] == pytest.approx(0.01)
    assert shifted['thresholds']['0.005']['fscore'] == 0
    assert shifted['thresholds']['0.02']['fscore'] == 1
    with pytest.raises(ValueError, match='nonempty'):
        module.surface_metrics(np.empty((0, 3)), points, [0.05])


def test_rigid_gauge_alignment_recovers_rotation_without_scale():
    module = geometry_module()
    points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    reference = points @ rotation.T + [2., 3., 4.]
    transform = module.rigid_alignment(points, reference)
    np.testing.assert_allclose(points @ transform[:3, :3].T + transform[:3, 3], reference, atol=1e-12)
    assert np.linalg.det(transform[:3, :3]) == pytest.approx(1)
    scaled = module.rigid_alignment(points * 2, points)
    assert not np.allclose(points * 2 @ scaled[:3, :3].T + scaled[:3, 3], points)


def test_gt_visibility_excludes_occluded_and_behind_camera_points():
    module = geometry_module()
    points = np.array([[0., 0., 1.], [0., 0., 3.], [0., 0., -1.]])
    intrinsic = np.array([[1., 0., 1.], [0., 1., 1.], [0., 0., 1.]])
    mask = module.visible_in_depth(points, np.eye(4), intrinsic, np.ones((3, 3)), 0.02)
    assert mask.tolist() == [True, False, False]
