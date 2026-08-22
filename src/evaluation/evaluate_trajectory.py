import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class NumpyFloatValuesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return JSONEncoder.default(self, obj)


def align(model, data):
    """Align two trajectories using the method of Horn (closed-form).

    Input:
    model -- first trajectory (3xn)
    data -- second trajectory (3xn)

    Output:
    rot -- rotation matrix (3x3)
    trans -- translation vector (3x1)
    trans_error -- translational error per point (1xn)

    """
    np.set_printoptions(precision=3, suppress=True)
    model_zerocentered = model - model.mean(1)
    data_zerocentered = data - data.mean(1)

    W = np.zeros((3, 3))
    for column in range(model.shape[1]):
        W += np.outer(model_zerocentered[:,
                                         column], data_zerocentered[:, column])
    U, d, Vh = np.linalg.linalg.svd(W.transpose())
    S = np.matrix(np.identity(3))
    if (np.linalg.det(U) * np.linalg.det(Vh) < 0):
        S[2, 2] = -1
    rot = U * S * Vh
    trans = data.mean(1) - rot * model.mean(1)

    model_aligned = rot * model + trans
    alignment_error = model_aligned - data

    trans_error = np.sqrt(
        np.sum(np.multiply(alignment_error, alignment_error), 0)).A[0]

    return rot, trans, trans_error


def align_trajectories(t_pred: np.ndarray, t_gt: np.ndarray):
    """
    Args:
        t_pred: (n, 3) translations
        t_gt: (n, 3) translations
    Returns:
        t_align: (n, 3) aligned translations
    """
    t_align = np.matrix(t_pred).transpose()
    R, t, _ = align(t_align, np.matrix(t_gt).transpose())
    t_align = R * t_align + t
    t_align = np.asarray(t_align).T
    return t_align


def pose_error(t_pred: np.ndarray, t_gt: np.ndarray, align=False):
    """
    Args:
        t_pred: (n, 3) translations
        t_gt: (n, 3) translations
    Returns:
        dict: error dict
    """
    n = t_pred.shape[0]
    trans_error = np.linalg.norm(t_pred - t_gt, axis=1)
    return {
        "compared_pose_pairs": n,
        "rmse": np.sqrt(np.dot(trans_error, trans_error) / n),
        "mean": np.mean(trans_error),
        "median": np.median(trans_error),
        "std": np.std(trans_error),
        "min": np.min(trans_error),
        "max": np.max(trans_error)
    }


def compute_relative_pose_errors(estimated_poses: np.ndarray,
                                 gt_poses: np.ndarray) -> dict:
    num_poses = min(len(estimated_poses), len(gt_poses))
    translation_errors = []
    rotation_errors_rad = []
    for index in range(num_poses - 1):
        pose_group = (
            estimated_poses[index], estimated_poses[index + 1],
            gt_poses[index], gt_poses[index + 1])
        if not all(np.isfinite(pose).all() for pose in pose_group):
            continue
        estimated_relative = (
            np.linalg.inv(estimated_poses[index])
            @ estimated_poses[index + 1])
        gt_relative = (
            np.linalg.inv(gt_poses[index]) @ gt_poses[index + 1])
        error = np.linalg.inv(gt_relative) @ estimated_relative
        translation_errors.append(np.linalg.norm(error[:3, 3]))
        cosine = np.clip(
            (np.trace(error[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
        rotation_errors_rad.append(np.arccos(cosine))

    valid_pairs = len(translation_errors)
    if valid_pairs == 0:
        return {
            "valid_pairs": 0,
            "translation_rmse_m": None,
            "rotation_rmse_deg": None,
        }
    translation_errors = np.asarray(translation_errors)
    rotation_errors_rad = np.asarray(rotation_errors_rad)
    return {
        "valid_pairs": valid_pairs,
        "translation_rmse_m": float(np.sqrt(np.mean(
            translation_errors ** 2))),
        "rotation_rmse_deg": float(np.degrees(np.sqrt(np.mean(
            rotation_errors_rad ** 2)))),
    }


def plot_2d(pts, ax=None, color="green", label="None", title="3D Trajectory in 2D"):
    if ax is None:
        _, ax = plt.subplots()
    ax.scatter(pts[:, 0], pts[:, 1], color=color, label=label, s=0.7)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    return ax


def evaluate_trajectory(estimated_poses: np.ndarray, gt_poses: np.ndarray, output_path: Path):
    output_path.mkdir(exist_ok=True, parents=True)
    num_poses = min(gt_poses.shape[0], estimated_poses.shape[0])
    gt_poses = gt_poses[:num_poses]
    estimated_poses = estimated_poses[:num_poses]
    valid = (
        np.isfinite(gt_poses).all(axis=(1, 2))
        & np.isfinite(estimated_poses).all(axis=(1, 2)))
    gt_poses = gt_poses[valid]
    estimated_poses = estimated_poses[valid]

    gt_t = gt_poses[:, :3, 3]
    estimated_t = estimated_poses[:, :3, 3]
    estimated_t_aligned = align_trajectories(estimated_t, gt_t)
    ate = pose_error(estimated_t, gt_t)
    ate_aligned = pose_error(estimated_t_aligned, gt_t)
    rpe = compute_relative_pose_errors(estimated_poses, gt_poses)

    with open(str(output_path / "ate.json"), "w") as f:
        f.write(json.dumps(ate, cls=NumpyFloatValuesEncoder))

    with open(str(output_path / "ate_aligned.json"), "w") as f:
        f.write(json.dumps(ate_aligned, cls=NumpyFloatValuesEncoder))

    with open(str(output_path / "rpe.json"), "w") as f:
        f.write(json.dumps(rpe, cls=NumpyFloatValuesEncoder))

    trajectory_metrics = {
        "alignment_mode": "se3_horn_translation_no_scale",
        "valid_poses": int(len(gt_poses)),
        "ate_unaligned": ate,
        "ate_aligned": ate_aligned,
        "rpe_consecutive": rpe,
    }
    with open(str(output_path / "trajectory_metrics.json"), "w") as f:
        f.write(json.dumps(
            trajectory_metrics, cls=NumpyFloatValuesEncoder))

    ate_rmse, ate_rmse_aligned = ate["rmse"], ate_aligned["rmse"]
    ax = plot_2d(
        estimated_t, label=f"ate-rmse: {round(ate_rmse * 100, 2)} cm", color="orange")
    ax = plot_2d(estimated_t_aligned, ax,
                 label=f"ate-rsme (aligned): {round(ate_rmse_aligned * 100, 2)} cm", color="lightskyblue")
    ax = plot_2d(gt_t, ax, label="GT", color="green")
    ax.legend()
    plt.savefig(str(output_path / "eval_trajectory.png"), dpi=300)
    plt.close()
    print(
        f"ATE-RMSE: {ate_rmse * 100:.2f} cm, ATE-RMSE (aligned): {ate_rmse_aligned * 100:.2f} cm")
