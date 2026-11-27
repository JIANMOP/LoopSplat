#!/usr/bin/env python3
"""Evaluate frozen Replica maps with fixed GT cameras and sampled surfaces.

No optimization, scale fitting, ICP, or method-dependent keyframe culling.
Distances are in metres. This is geometric evaluation, not held-out rendering.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial import cKDTree


def surface_metrics(predicted, target, thresholds):
    if len(predicted) == 0 or len(target) == 0:
        raise ValueError('surface samples must be nonempty')
    if not np.isfinite(predicted).all() or not np.isfinite(target).all():
        raise ValueError('surface samples must be finite')
    accuracy = cKDTree(target).query(predicted, workers=4)[0]
    completeness = cKDTree(predicted).query(target, workers=4)[0]
    result = {
        'accuracy_m': float(accuracy.mean()),
        'completeness_m': float(completeness.mean()),
        'chamfer_l1_m': float((accuracy.mean() + completeness.mean()) / 2),
        'thresholds': {},
    }
    for threshold in thresholds:
        precision = float((accuracy < threshold).mean())
        recall = float((completeness < threshold).mean())
        result['thresholds'][str(threshold)] = {
            'precision': precision, 'recall': recall,
            'fscore': 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        }
    return result


def rigid_alignment(estimated, reference):
    estimated_mean, reference_mean = estimated.mean(0), reference.mean(0)
    u, _, vt = np.linalg.svd((estimated - estimated_mean).T @ (reference - reference_mean))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(vt.T @ u.T)
    transform = np.eye(4)
    transform[:3, :3] = vt.T @ correction @ u.T
    transform[:3, 3] = reference_mean - transform[:3, :3] @ estimated_mean
    return transform


def visible_in_depth(points, c2w, intrinsic, depth, tolerance):
    camera = (points - c2w[:3, 3]) @ c2w[:3, :3]
    front = camera[:, 2] > 0
    ids = np.flatnonzero(front)
    pixels = camera[ids] @ intrinsic.T
    uv = np.rint(pixels[:, :2] / pixels[:, 2:3]).astype(np.int64)
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < depth.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < depth.shape[0])
    ids, uv = ids[inside], uv[inside]
    measured = depth[uv[:, 1], uv[:, 0]]
    result = np.zeros(len(points), dtype=bool)
    result[ids] = (measured > 0) & np.isfinite(measured) & (np.abs(measured - camera[ids, 2]) <= tolerance)
    return result


def sample_surface(mesh, count, seed):
    rng = np.random.default_rng(seed)
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    xyz = vertices[triangles]
    area = np.linalg.norm(np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1)
    if len(area) == 0 or area.sum() <= 0:
        raise ValueError('mesh must have positive-area triangles')
    selected = xyz[rng.choice(len(area), size=count, p=area / area.sum())]
    root = np.sqrt(rng.random(count))
    other = rng.random(count)
    weights = np.column_stack([1 - root, root * (1 - other), root * other])
    return (selected * weights[:, :, None]).sum(axis=1)


def load_surface_arrays(path):
    import trimesh
    mesh = trimesh.load(path, force='mesh', process=False)
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    if not np.isfinite(vertices).all() or len(faces) == 0:
        raise ValueError('GT mesh must have finite vertices and nonempty faces')
    return vertices, faces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--gt-mesh', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frame-limit', type=int, default=-1, help='Positive values mark a pilot, never a formal result')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import open3d as o3d
    import torch
    import yaml
    from src.entities.arguments import OptimizationParams
    from src.entities.datasets import get_dataset
    from src.entities.gaussian_model import GaussianModel
    from src.evaluation.protocol import GAUSSIAN_PARAM_KEYS, build_evaluation_frame_ids, concatenate_gaussian_params
    from src.utils.utils import get_render_settings, render_gaussian_model

    if (args.output / 'metrics.json').exists():
        raise FileExistsError('Existing result is preserved; choose a new output path')
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    config_bytes = (args.run_dir / 'config.yaml').read_bytes()
    config = yaml.safe_load(config_bytes)
    if config['dataset_name'] != 'replica':
        raise ValueError('This fixed protocol is currently defined for Replica only')
    data_config = {**config['data'], **config['cam']}
    data_config['frame_limit'] = -1
    dataset = get_dataset('replica')(data_config)
    estimated = torch.load(args.run_dir / 'estimated_c2w.ckpt', map_location='cpu').numpy()
    reference = np.asarray(dataset.poses, dtype=np.float64)
    if len(estimated) != len(dataset) or len(reference) != len(dataset):
        raise ValueError('Full saved trajectory must match the dataset')
    if not np.isfinite(estimated).all():
        raise ValueError('Saved poses must be finite')
    gauge = rigid_alignment(estimated[:, :3, 3], reference[:, :3, 3])
    count = len(dataset) if args.frame_limit < 0 else min(args.frame_limit, len(dataset))
    frame_ids = build_evaluation_frame_ids(count, 10)
    protocol = {
        'version': 'replica-frozen-global-tsdf-v1',
        'pilot': args.frame_limit >= 0,
        'frame_ids': frame_ids,
        'camera_source': 'fixed_GT_poses_mapped_back_by_single_SE3_gauge',
        'gauge_estimation': 'all_saved_camera_positions_no_scale_no_ICP',
        'gt_from_map': gauge.tolist(),
        'voxel_m': 0.01, 'sdf_trunc_m': 0.04, 'depth_trunc_m': 10.0,
        'minimum_render_alpha': 0.5, 'gt_visibility_tolerance_m': 0.02,
        'surface_sample_count': 200000, 'surface_sample_seed': 0,
        'distance_thresholds_m': [0.01, 0.02, 0.05],
        'predicted_surface_culling': 'none; no GT-distance filtering',
        'gt_surface_culling': 'visible in the same fixed GT RGB-D cameras',
        'depth_source': 'unmodified_rasterizer_depth_output',
        'config_sha256': hashlib.sha256(config_bytes).hexdigest(),
        'gt_mesh_sha256': hashlib.sha256(args.gt_mesh.read_bytes()).hexdigest(),
    }
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    vertices, faces = load_surface_arrays(args.gt_mesh)
    print(f'Loaded complete GT mesh: {len(vertices)} vertices, {len(faces)} faces', flush=True)
    gt_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
    target_candidates = sample_surface(gt_mesh, 200000, 0)
    seen = np.zeros(len(target_candidates), dtype=bool)
    for frame_id in frame_ids:
        depth = dataset[frame_id][2]
        seen |= visible_in_depth(target_candidates, reference[frame_id], dataset.intrinsics, depth, 0.02)
    target = target_candidates[seen]
    if len(target) < 1000:
        raise ValueError(f'Only {len(target)} visible GT samples; inspect coordinates and depth units')
    print(f'GT visibility: {len(target)} / 200000 samples, {len(frame_ids)} fixed cameras', flush=True)

    parameter_sets = []
    for path in sorted((args.run_dir / 'submaps').glob('*.ckpt')):
        saved = torch.load(path, map_location='cpu')['gaussian_params']
        parameter_sets.append({key: saved[key] for key in GAUSSIAN_PARAM_KEYS})
    merged = {key: value.cuda() for key, value in concatenate_gaussian_params(parameter_sets).items()}
    del parameter_sets, saved
    model = GaussianModel()
    model.restore_from_params(merged, OptimizationParams(argparse.ArgumentParser()))
    del merged
    print(f'Loaded {model.get_size()} frozen Gaussians', flush=True)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(dataset.width, dataset.height,
                                                dataset.intrinsics[0, 0], dataset.intrinsics[1, 1],
                                                dataset.intrinsics[0, 2], dataset.intrinsics[1, 2])
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.01, sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    inverse_gauge = np.linalg.inv(gauge)
    frame_diagnostics = []
    for index, frame_id in enumerate(frame_ids):
        query_pose = inverse_gauge @ reference[frame_id]
        with torch.no_grad():
            rendered = render_gaussian_model(model, get_render_settings(
                dataset.width, dataset.height, dataset.intrinsics, np.linalg.inv(query_pose)))
            depth = rendered['depth'][0].detach().cpu().numpy().copy()
            alpha = rendered['alpha'].detach().cpu().numpy().squeeze()
            color = (rendered['color'].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        valid = np.isfinite(depth) & (depth > 0) & (depth < 10) & (alpha > 0.5)
        depth[~valid] = 0
        frame_diagnostics.append({'frame_id': frame_id, 'valid_render_fraction': float(valid.mean())})
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(color)),
            o3d.geometry.Image(np.ascontiguousarray(depth, dtype=np.float32)),
            depth_scale=1.0, depth_trunc=10.0, convert_rgb_to_intensity=False)
        volume.integrate(rgbd, intrinsic, np.linalg.inv(reference[frame_id]))
        del rendered
        print(f'Geometry render {index + 1}/{len(frame_ids)} frame={frame_id} valid={valid.mean():.3f}', flush=True)
    del model
    torch.cuda.empty_cache()
    mesh = volume.extract_triangle_mesh()
    o3d.io.write_triangle_mesh(str(args.output / 'reconstructed.ply'), mesh)
    predicted = sample_surface(mesh, 200000, 0)
    metrics = surface_metrics(predicted, target, [0.01, 0.02, 0.05])
    metrics.update({'pilot': args.frame_limit >= 0, 'predicted_sample_count': len(predicted),
                    'visible_gt_sample_count': len(target), 'elapsed_seconds': time.time() - started})
    (args.output / 'render-coverage.json').write_text(json.dumps(frame_diagnostics, indent=2))
    np.savez_compressed(args.output / 'surface-samples.npz', predicted=predicted, target=target)
    (args.output / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
