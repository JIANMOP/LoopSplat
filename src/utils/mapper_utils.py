
import cv2
import faiss
import faiss.contrib.torch_utils
import numpy as np
import torch


def compute_opt_views_distribution(keyframes_num, iterations_num, current_frame_iter) -> np.ndarray:
    """ Computes the probability distribution for selecting views based on the current iteration.
    Args:
        keyframes_num: The total number of keyframes.
        iterations_num: The total number of iterations planned.
        current_frame_iter: The current iteration number.
    Returns:
        An array representing the probability distribution of keyframes.
    """
    if keyframes_num == 1:
        return np.array([1.0])
    prob = np.full(keyframes_num, (iterations_num - current_frame_iter) / (keyframes_num - 1))
    prob[0] = current_frame_iter
    prob /= prob.sum()
    return prob


def compute_camera_frustum_corners(depth_map: np.ndarray, pose: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """ Computes the 3D coordinates of the camera frustum corners based on the depth map, pose, and intrinsics.
    Args:
        depth_map: The depth map of the scene.
        pose: The camera pose matrix.
        intrinsics: The camera intrinsic matrix.
    Returns:
        An array of 3D coordinates for the frustum corners.
    """
    height, width = depth_map.shape
    depth_map = depth_map[depth_map > 0]
    min_depth, max_depth = depth_map.min(), depth_map.max()
    corners = np.array(
        [
            [0, 0, min_depth],
            [width, 0, min_depth],
            [0, height, min_depth],
            [width, height, min_depth],
            [0, 0, max_depth],
            [width, 0, max_depth],
            [0, height, max_depth],
            [width, height, max_depth],
        ]
    )
    x = (corners[:, 0] - intrinsics[0, 2]) * corners[:, 2] / intrinsics[0, 0]
    y = (corners[:, 1] - intrinsics[1, 2]) * corners[:, 2] / intrinsics[1, 1]
    z = corners[:, 2]
    corners_3d = np.vstack((x, y, z, np.ones(x.shape[0]))).T
    corners_3d = pose @ corners_3d.T
    return corners_3d.T[:, :3]


def compute_camera_frustum_planes(frustum_corners: np.ndarray) -> torch.Tensor:
    """ Computes the planes of the camera frustum from its corners.
    Args:
        frustum_corners: An array of 3D coordinates representing the corners of the frustum.

    Returns:
        A tensor of frustum planes.
    """
    # near, far, left, right, top, bottom
    planes = torch.stack(
        [
            torch.cross(
                frustum_corners[2] - frustum_corners[0],
                frustum_corners[1] - frustum_corners[0]
            ),
            torch.cross(
                frustum_corners[6] - frustum_corners[4],
                frustum_corners[5] - frustum_corners[4]
            ),
            torch.cross(
                frustum_corners[4] - frustum_corners[0],
                frustum_corners[2] - frustum_corners[0]
            ),
            torch.cross(
                frustum_corners[7] - frustum_corners[3],
                frustum_corners[1] - frustum_corners[3]
            ),
            torch.cross(
                frustum_corners[5] - frustum_corners[1], 
                frustum_corners[0] - frustum_corners[1]
            ),
            torch.cross(
                frustum_corners[6] - frustum_corners[2], 
                frustum_corners[3] - frustum_corners[2]
            ),
        ]
    )
    D = torch.stack([-torch.dot(plane, frustum_corners[i]) for i, plane in enumerate(planes)])
    return torch.cat([planes, D[:, None]], dim=1).float()


def compute_frustum_aabb(frustum_corners: torch.Tensor):
    """ Computes a mask indicating which points lie inside a given axis-aligned bounding box (AABB).
    Args:
        points: An array of 3D points.
        min_corner: The minimum corner of the AABB.
        max_corner: The maximum corner of the AABB.
    Returns:
        A boolean array indicating whether each point lies inside the AABB.
    """
    return torch.min(frustum_corners, axis=0).values, torch.max(frustum_corners, axis=0).values


def points_inside_aabb_mask(points: np.ndarray, min_corner: np.ndarray, max_corner: np.ndarray) -> np.ndarray:
    """ Computes a mask indicating which points lie inside the camera frustum.
    Args:
        points: A tensor of 3D points.
        frustum_planes: A tensor representing the planes of the frustum.
    Returns:
        A boolean tensor indicating whether each point lies inside the frustum.
    """
    return (
        (points[:, 0] >= min_corner[0])
        & (points[:, 0] <= max_corner[0])
        & (points[:, 1] >= min_corner[1])
        & (points[:, 1] <= max_corner[1])
        & (points[:, 2] >= min_corner[2])
        & (points[:, 2] <= max_corner[2]))


def points_inside_frustum_mask(points: torch.Tensor, frustum_planes: torch.Tensor) -> torch.Tensor:
    """ Computes a mask indicating which points lie inside the camera frustum.
    Args:
        points: A tensor of 3D points.
        frustum_planes: A tensor representing the planes of the frustum.
    Returns:
        A boolean tensor indicating whether each point lies inside the frustum.
    """
    num_pts = points.shape[0]
    ones = torch.ones(num_pts, 1).to(points.device)
    plane_product = torch.cat([points, ones], axis=1) @ frustum_planes.T
    return torch.all(plane_product <= 0, axis=1)


def compute_frustum_point_ids(pts: torch.Tensor, frustum_corners: torch.Tensor, device: str = "cuda"):
    """ Identifies points within the camera frustum, optimizing for computation on a specified device.
    Args:
        pts: A tensor of 3D points.
        frustum_corners: A tensor of 3D coordinates representing the corners of the frustum.
        device: The computation device ("cuda" or "cpu").
    Returns:
        Indices of points lying inside the frustum.
    """
    if pts.shape[0] == 0:
        return torch.tensor([], dtype=torch.int64, device=device)
    # Broad phase
    pts = pts.to(device)
    frustum_corners = frustum_corners.to(device)

    min_corner, max_corner = compute_frustum_aabb(frustum_corners)
    inside_aabb_mask = points_inside_aabb_mask(pts, min_corner, max_corner)

    # Narrow phase
    frustum_planes = compute_camera_frustum_planes(frustum_corners)
    frustum_planes = frustum_planes.to(device)
    inside_frustum_mask = points_inside_frustum_mask(pts[inside_aabb_mask], frustum_planes)

    inside_aabb_mask[inside_aabb_mask == 1] = inside_frustum_mask
    return torch.where(inside_aabb_mask)[0]


def sample_pixels_based_on_gradient(image: np.ndarray, num_samples: int) -> np.ndarray:
    """ Samples pixel indices based on the gradient magnitude of an image.
    Args:
        image: The image from which to sample pixels.
        num_samples: The number of pixels to sample.
    Returns:
        Indices of the sampled pixels.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = cv2.magnitude(grad_x, grad_y)

    # Normalize the gradient magnitude to create a probability map
    prob_map = grad_magnitude / np.sum(grad_magnitude)

    # Flatten the probability map
    prob_map_flat = prob_map.flatten()

    # Sample pixel indices based on the probability map
    sampled_indices = np.random.choice(prob_map_flat.size, size=num_samples, p=prob_map_flat)
    return sampled_indices.T


def compute_new_points_ids(frustum_points: torch.Tensor, new_pts: torch.Tensor,
                           radius: float = 0.03, device: str = "cpu") -> torch.Tensor:
    """ Having newly initialized points, decides which of them should be added to the submap.
        For every new point, if there are no neighbors within the radius in the frustum points,
        it is added to the submap.
    Args:
        frustum_points: Point within a current frustum of the active submap of shape (N, 3)
        new_pts: New 3D Gaussian means which are about to be added to the submap of shape (N, 3)
        radius: Radius whithin which the points are considered to be neighbors
        device: Execution device
    Returns:
        Indicies of the new points that should be added to the submap of shape (N)
    """
    if frustum_points.shape[0] == 0:
        return torch.arange(new_pts.shape[0])
    if device == "cpu":
        pts_index = faiss.IndexFlatL2(3)
    else:
        pts_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, faiss.IndexFlatL2(3))
    frustum_points = frustum_points.to(device)
    new_pts = new_pts.to(device)
    pts_index.add(frustum_points)

    split_pos = torch.split(new_pts, 65535, dim=0)
    distances, ids = [], []
    for split_p in split_pos:
        distance, id = pts_index.search(split_p.float(), 8)
        distances.append(distance)
        ids.append(id)
    distances = torch.cat(distances, dim=0)
    ids = torch.cat(ids, dim=0)
    neighbor_num = (distances < radius).sum(axis=1).int()
    pts_index.reset()
    return torch.where(neighbor_num == 0)[0]


def rotation_to_euler(R: torch.Tensor) -> torch.Tensor:
    """
    Converts a rotation matrix to Euler angles.
    Args:
        R: A rotation matrix.
    Returns:
        Euler angles corresponding to the rotation matrix.
    """
    sy = torch.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        x = torch.atan2(R[2, 1], R[2, 2])
        y = torch.atan2(-R[2, 0], sy)
        z = torch.atan2(R[1, 0], R[0, 0])
    else:
        x = torch.atan2(-R[1, 2], R[1, 1])
        y = torch.atan2(-R[2, 0], sy)
        z = 0

    return torch.tensor([x, y, z]) * (180 / np.pi)


def exceeds_motion_thresholds(current_c2w: torch.Tensor, last_submap_c2w: torch.Tensor,
                              rot_thre: float = 50, trans_thre: float = 0.5) -> bool:
    """  Checks if a camera motion exceeds certain rotation and translation thresholds
    Args:
        current_c2w: The current camera-to-world transformation matrix.
        last_submap_c2w: The last submap's camera-to-world transformation matrix.
        rot_thre: The rotation threshold for triggering a new submap.
        trans_thre: The translation threshold for triggering a new submap.

    Returns:
        A boolean indicating whether a new submap is required.
    """
    delta_pose = torch.matmul(torch.linalg.inv(last_submap_c2w).float(), current_c2w.float())
    translation_diff = torch.norm(delta_pose[:3, 3])
    rot_euler_diff_deg = torch.abs(rotation_to_euler(delta_pose[:3, :3]))
    exceeds_thresholds = (translation_diff > trans_thre) or torch.any(rot_euler_diff_deg > rot_thre)
    return exceeds_thresholds.item()


def geometric_edge_mask(rgb_image: np.ndarray, dilate: bool = True, RGB: bool = False) -> np.ndarray:
    """ Computes an edge mask for an RGB image using geometric edges.
    Args:
        rgb_image: The RGB image.
        dilate: Whether to dilate the edges.
        RGB: Indicates if the image format is RGB (True) or BGR (False).
    Returns:
        An edge mask of the input image.
    """
    # Convert the image to grayscale as Canny edge detection requires a single channel image
    gray_image = cv2.cvtColor(
        rgb_image, cv2.COLOR_BGR2GRAY if not RGB else cv2.COLOR_RGB2GRAY)
    if gray_image.dtype != np.uint8:
        gray_image = gray_image.astype(np.uint8)
    edges = cv2.Canny(gray_image, threshold1=100, threshold2=200, apertureSize=3, L2gradient=True)
    # Define the structuring element for dilation, you can change the size for a thicker/thinner mask
    if dilate:
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
    return edges


def calc_psnr(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """ Calculates the Peak Signal-to-Noise Ratio (PSNR) between two images.
    Args:
        img1: The first image.
        img2: The second image.
    Returns:
        The PSNR value.
    """
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).mean()


def create_point_cloud(image: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """
    Creates a point cloud from an image, depth map, camera intrinsics, and pose.

    Args:
        image: The RGB image of shape (H, W, 3)
        depth: The depth map of shape (H, W)
        intrinsics: The camera intrinsic parameters of shape (3, 3)
        pose: The camera pose of shape (4, 4)
    Returns:
        A point cloud of shape (N, 6) with last dimension representing (x, y, z, r, g, b)
    """
    height, width = depth.shape
    # Create a mesh grid of pixel coordinates
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    # Convert pixel coordinates to camera coordinates
    x = (u - intrinsics[0, 2]) * depth / intrinsics[0, 0]
    y = (v - intrinsics[1, 2]) * depth / intrinsics[1, 1]
    z = depth
    # Stack the coordinates together
    points = np.stack((x, y, z, np.ones_like(z)), axis=-1)
    # Reshape the coordinates for matrix multiplication
    points = points.reshape(-1, 4)
    # Transform points to world coordinates
    posed_points = pose @ points.T
    posed_points = posed_points.T[:, :3]
    # Flatten the image to get colors for each point
    colors = image.reshape(-1, 3)
    # Concatenate posed points with their corresponding color
    point_cloud = np.concatenate((posed_points, colors), axis=-1)

    return point_cloud


def compute_gaussian_visibility(gaussian_xyz: torch.Tensor, estimate_w2c: np.ndarray,
                                intrinsics: np.ndarray, depth_map: np.ndarray,
                                device: str = "cuda") -> np.ndarray:
    """Compute the set of visible Gaussian point IDs from a given camera pose.

    Uses camera frustum culling to determine which 3D Gaussian means project
    into the current view. This is a key building block for the GI-SLAM
    keyframe selection strategy (covisibility IoU).

    Args:
        gaussian_xyz: Tensor of Gaussian center positions (N, 3).
        estimate_w2c: World-to-camera transformation matrix (4, 4).
        intrinsics: Camera intrinsic matrix (3, 3).
        depth_map: Depth map of the frame (H, W), used to estimate
            frustum near/far planes.
        device: Computation device.

    Returns:
        Numpy array of visible point indices (can be empty).
    """
    if gaussian_xyz.shape[0] == 0:
        return np.array([], dtype=np.int64)

    frustum_corners = compute_camera_frustum_corners(
        depth_map, estimate_w2c, intrinsics)
    frustum_corners_t = torch.from_numpy(frustum_corners).float().to(device)

    visible_ids = compute_frustum_point_ids(gaussian_xyz, frustum_corners_t, device=device)
    return visible_ids.cpu().numpy()


def compute_gaussian_iou(visible_ids_a: np.ndarray,
                          visible_ids_b: np.ndarray) -> float:
    """Compute the IoU (intersection over union) of two sets of visible
    Gaussian point IDs. Used for GI-SLAM covisibility score.

    Args:
        visible_ids_a: Indices of visible Gaussians from view A.
        visible_ids_b: Indices of visible Gaussians from view B.

    Returns:
        IoU value in [0.0, 1.0]. Returns 0.0 when union is empty.
    """
    if len(visible_ids_a) == 0 and len(visible_ids_b) == 0:
        return 0.0

    set_a = set(visible_ids_a.tolist())
    set_b = set(visible_ids_b.tolist())

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)

    if union == 0:
        return 0.0
    return intersection / union


def compute_camera_velocity(c2w_curr: np.ndarray, c2w_prev: np.ndarray,
                            dt: float) -> tuple:
    """Compute camera linear and angular velocity from two consecutive poses.

    Used for GI-SLAM motion-blur penalty: frames with excessive velocity
    are penalized from becoming keyframes.

    Args:
        c2w_curr: Current camera-to-world transformation (4, 4).
        c2w_prev: Previous camera-to-world transformation (4, 4).
        dt: Time delta between the two frames in seconds.

    Returns:
        Tuple (linear_vel, angular_vel) in m/s and deg/s respectively.
        Returns (0.0, 0.0) when dt <= 0.
    """
    if dt <= 0:
        return 0.0, 0.0

    # Delta w2c: delta_pose = c2w_prev^{-1} @ c2w_curr
    delta_pose = np.linalg.inv(c2w_prev) @ c2w_curr

    # Linear velocity: ||translation|| / dt (m/s)
    linear_vel = np.linalg.norm(delta_pose[:3, 3]) / dt

    # Angular velocity: extract axis-angle from rotation matrix (deg/s)
    R_delta = delta_pose[:3, :3]
    # cos(theta) = (trace(R) - 1) / 2
    cos_theta = (np.trace(R_delta) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    angular_vel = np.degrees(angle_rad) / dt

    return float(linear_vel), float(angular_vel)


def compute_median_depth(depth_map: np.ndarray) -> float:
    """Compute the median valid depth value from a depth map.

    Used to normalize the baseline distance in GI-SLAM keyframe scoring.

    Args:
        depth_map: Depth map (H, W).

    Returns:
        Median depth among valid (> 0) pixels, or 1.0 when all invalid.
    """
    valid = depth_map[depth_map > 0]
    if len(valid) == 0:
        return 1.0
    return float(np.median(valid))


def gi_slam_keyframe_score(visible_ids_curr: np.ndarray,
                           visible_ids_keyframe: np.ndarray,
                           c2w_curr: np.ndarray,
                           c2w_keyframe: np.ndarray,
                           depth_map_curr: np.ndarray,
                           linear_vel: float,
                           angular_vel: float,
                           w_covis: float = 1.0,
                           w_base: float = 1.0,
                           w_mot: float = 2.0,
                           v_max: float = 0.8,
                           omega_max: float = 50.0) -> float:
    """Compute GI-SLAM keyframe selection score.

    Implements the scoring formula from GI-SLAM (Section 3.3):
        s_i = w_covis * (1 - IoU_𝒢)
            + w_base * ||t_ij|| / d_med
            - w_mot * 𝕀(v_i > v_max ∨ ω_i > ω_max)

    A higher score indicates the frame is more suitable as a new keyframe.
    Frames with scores above a threshold should be selected.

    Args:
        visible_ids_curr: Visible Gaussian IDs from current frame.
        visible_ids_keyframe: Visible Gaussian IDs from nearest keyframe.
        c2w_curr: Current camera-to-world pose (4, 4).
        c2w_keyframe: Nearest keyframe camera-to-world pose (4, 4).
        depth_map_curr: Current frame depth map (H, W).
        linear_vel: Camera linear velocity (m/s).
        angular_vel: Camera angular velocity (deg/s).
        w_covis: Covisibility weight.
        w_base: Baseline distance weight.
        w_mot: Motion blur penalty weight.
        v_max: Max linear velocity threshold (m/s).
        omega_max: Max angular velocity threshold (deg/s).

    Returns:
        Keyframe selection score.
    """
    # 1. Covisibility score: 1 - IoU of visible Gaussian sets
    iou = compute_gaussian_iou(visible_ids_curr, visible_ids_keyframe)
    covis_score = w_covis * (1.0 - iou)

    # 2. Baseline distance score: ||t_ij|| / d_med
    baseline = np.linalg.norm(c2w_curr[:3, 3] - c2w_keyframe[:3, 3])
    d_med = compute_median_depth(depth_map_curr)
    baseline_score = w_base * baseline / max(d_med, 1e-6)

    # 3. Motion blur penalty
    motion_penalty = 0.0
    if linear_vel > v_max or angular_vel > omega_max:
        motion_penalty = w_mot

    score = covis_score + baseline_score - motion_penalty
    return float(score)


# ── Photo-SLAM Gaussian Pyramid utilities ────────────────────────────


def build_image_pyramid(image: "torch.Tensor", num_sub_levels: int) -> list:
    """Build a Gaussian image pyramid from a single image tensor.

    Applies repeated 2x downsampling with Gaussian pre-smoothing to produce
    a list of progressively coarser images. Image is assumed to be a GPU
    tensor in (C, H, W) format with values in [0, 1].

    The returned list has length ``num_sub_levels``; index 0 is the coarsest
    level and index ``num_sub_levels - 1`` is the finest sub-level (just below
    full resolution). The full-resolution image is *not* included.

    Scale factor for level ``l`` (0-indexed)::
        0.5 ** (num_sub_levels - l)

    e.g. with num_sub_levels=2 → [0.25x, 0.5x]

    Args:
        image: (C, H, W) torch tensor on GPU, values in [0, 1].
        num_sub_levels: Number of sub-resolution levels (≥ 1).

    Returns:
        List of (C, h_i, w_i) tensors, coarsest first.
    """
    if num_sub_levels < 1:
        return []

    pyramid = []
    _, H, W = image.shape

    for level in range(num_sub_levels):
        scale = 0.5 ** (num_sub_levels - level)
        new_h, new_w = max(1, int(H * scale)), max(1, int(W * scale))
        # Gaussian smoothing before downsampling (kernel size ~ scale)
        blur_sigma = (0.5 / scale - 0.5) if scale < 1.0 else 0.0
        kernel_size = max(3, 2 * int(2 * blur_sigma) + 1)
        if blur_sigma > 0.1:
            smoothed = torch.nn.functional.avg_pool2d(
                image.unsqueeze(0),
                kernel_size=kernel_size, stride=1, padding=kernel_size // 2
            ).squeeze(0)
        else:
            smoothed = image
        downsampled = torch.nn.functional.interpolate(
            smoothed.unsqueeze(0), size=(new_h, new_w),
            mode='bilinear', align_corners=False
        ).squeeze(0)
        pyramid.append(downsampled)

    return pyramid


def build_depth_pyramid(depth: "torch.Tensor", num_sub_levels: int) -> list:
    """Build a depth pyramid from a single depth map tensor.

    Differs from ``build_image_pyramid`` in that depth values should NOT be
    Gaussian-smoothed before downsampling (to preserve geometric edges).
    Simple bilinear interpolation (area-weighted) is used instead.

    Args:
        depth: (H, W) or (1, H, W) torch tensor.
        num_sub_levels: Number of sub-resolution levels (≥ 1).

    Returns:
        List of tensors with the same number of dimensions, coarsest first.
    """
    if num_sub_levels < 1:
        return []

    pyramid = []
    if depth.dim() == 2:
        depth = depth.unsqueeze(0)  # (1, H, W)
    squeeze_out = depth.dim() == 3 and depth.shape[0] == 1
    _, H, W = depth.shape if depth.dim() == 3 else (1, depth.shape[0], depth.shape[1])

    for level in range(num_sub_levels):
        scale = 0.5 ** (num_sub_levels - level)
        new_h, new_w = max(1, int(H * scale)), max(1, int(W * scale))
        downsampled = torch.nn.functional.interpolate(
            depth.unsqueeze(0) if depth.dim() == 2 else depth.unsqueeze(0),
            size=(new_h, new_w),
            mode='bilinear', align_corners=False
        ).squeeze(0)
        if squeeze_out:
            downsampled = downsampled.squeeze(0)
        pyramid.append(downsampled)

    return pyramid


def get_pyramid_level_dims(W: int, H: int, level: int,
                           num_sub_levels: int) -> tuple:
    """Return (width, height) for a given pyramid sub-level.

    Args:
        W: Full-resolution width.
        H: Full-resolution height.
        level: Sub-level index (0 = coarsest).
        num_sub_levels: Total number of sub-levels.

    Returns:
        (width, height) tuple.
    """
    scale = 0.5 ** (num_sub_levels - level)
    return max(1, int(W * scale)), max(1, int(H * scale))


def get_pyramid_render_settings(full_render_settings, W_level: int,
                                H_level: int):
    """Create render settings adjusted for a pyramid-level resolution.

    Scales the camera intrinsics proportionally so that the field-of-view
    remains unchanged while the rasterization target size decreases.

    Args:
        full_render_settings: The original render settings dict or
            GaussianRasterizationSettings object for full resolution.
        W_level: Target pyramid-level image width.
        H_level: Target pyramid-level image height.

    Returns:
        A new ``GaussianRasterizationSettings`` object (if the input is one),
        or a dict with ``image_height``/``image_width`` updated.
    """
    # Determine full-res dimensions and compute scale factors
    if hasattr(full_render_settings, 'image_height'):
        # It's a GaussianRasterizationSettings namedtuple-like object
        from diff_gaussian_rasterization import GaussianRasterizationSettings
        H_full = full_render_settings.image_height
        W_full = full_render_settings.image_width
        scale_x = W_level / max(W_full, 1)
        scale_y = H_level / max(H_full, 1)

        # Scale intrinsics
        new_tanfovx = full_render_settings.tanfovx  # unchanged — fov is same
        new_tanfovy = full_render_settings.tanfovy
        new_viewmatrix = full_render_settings.viewmatrix
        new_projmatrix = full_render_settings.projmatrix
        new_cam_center = full_render_settings.camera_center

        # Build new projection matrix for the scaled image
        # We need to scale fx,fy,cx,cy, but the projection matrix already
        # encodes the full-res intrinsics. Re-use the existing projmatrix;
        # the rasterizer uses tanfovx/y + image dimensions for the actual
        # projection, so the stored projmatrix is advisory.
        return GaussianRasterizationSettings(
            image_height=H_level,
            image_width=W_level,
            tanfovx=new_tanfovx,
            tanfovy=new_tanfovy,
            bg=full_render_settings.bg,
            scale_modifier=full_render_settings.scale_modifier,
            viewmatrix=new_viewmatrix,
            projmatrix=new_projmatrix,
            sh_degree=full_render_settings.sh_degree,
            campos=new_cam_center,
            prefiltered=full_render_settings.prefiltered,
            debug=full_render_settings.debug,
        )
    else:
        # It's a dict
        import copy
        settings = copy.deepcopy(full_render_settings)
        settings['image_height'] = H_level
        settings['image_width'] = W_level
        return settings
