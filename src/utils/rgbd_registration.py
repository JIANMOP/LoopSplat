import numpy as np


def register_depth_to_color(depth_m, depth_intrinsics, color_intrinsics,
                            t_color_depth, output_shape):
    depth_m = np.asarray(depth_m, dtype=np.float64)
    depth_intrinsics = np.asarray(depth_intrinsics, dtype=np.float64)
    color_intrinsics = np.asarray(color_intrinsics, dtype=np.float64)
    t_color_depth = np.asarray(t_color_depth, dtype=np.float64)
    if depth_m.ndim != 2:
        raise ValueError("depth_m must be a 2D array")
    if depth_intrinsics.shape != (3, 3) or color_intrinsics.shape != (3, 3):
        raise ValueError("camera intrinsics must be 3x3")
    if t_color_depth.shape != (4, 4):
        raise ValueError("t_color_depth must be 4x4")

    output_height, output_width = output_shape
    if output_height < 1 or output_width < 1:
        raise ValueError("output_shape must be positive")

    rows, cols = np.nonzero(np.isfinite(depth_m) & (depth_m > 0))
    if len(rows) == 0:
        return np.zeros(output_shape, dtype=np.float32)

    z_depth = depth_m[rows, cols]
    fx_d, fy_d = depth_intrinsics[0, 0], depth_intrinsics[1, 1]
    cx_d, cy_d = depth_intrinsics[0, 2], depth_intrinsics[1, 2]
    points_depth = np.column_stack((
        (cols - cx_d) * z_depth / fx_d,
        (rows - cy_d) * z_depth / fy_d,
        z_depth,
        np.ones_like(z_depth),
    ))
    points_color = (t_color_depth @ points_depth.T).T[:, :3]
    z_color = points_color[:, 2]

    valid_z = np.isfinite(points_color).all(axis=1) & (z_color > 0)
    points_color = points_color[valid_z]
    z_color = z_color[valid_z]
    fx_c, fy_c = color_intrinsics[0, 0], color_intrinsics[1, 1]
    cx_c, cy_c = color_intrinsics[0, 2], color_intrinsics[1, 2]
    color_cols = np.rint(
        fx_c * points_color[:, 0] / z_color + cx_c).astype(np.int64)
    color_rows = np.rint(
        fy_c * points_color[:, 1] / z_color + cy_c).astype(np.int64)

    in_bounds = (
        (color_cols >= 0) & (color_cols < output_width)
        & (color_rows >= 0) & (color_rows < output_height)
    )
    flat_ids = color_rows[in_bounds] * output_width + color_cols[in_bounds]
    registered = np.full(output_height * output_width, np.inf, dtype=np.float64)
    np.minimum.at(registered, flat_ids, z_color[in_bounds])
    registered[~np.isfinite(registered)] = 0.0
    return registered.reshape(output_shape).astype(np.float32)
