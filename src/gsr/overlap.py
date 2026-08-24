import open3d as o3d
import numpy as np
import torch
import faiss
import faiss.contrib.torch_utils
from pycg import vis



FAISS_TEMP_MEMORY_BYTES = 256 * 1024 * 1024
FAISS_SEARCH_BATCH_SIZE = 65535


def _directional_overlap_count(reference_points, query_points,
                               squared_threshold):
    resources = faiss.StandardGpuResources()
    resources.setTempMemory(FAISS_TEMP_MEMORY_BYTES)
    cpu_index = faiss.IndexFlatL2(3)
    device_index = reference_points.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    gpu_index = faiss.index_cpu_to_gpu(
        resources, device_index, cpu_index)
    gpu_index.add(reference_points.float())

    match_count = torch.zeros(
        (), dtype=torch.int64, device=query_points.device)
    for query_batch in torch.split(
            query_points, FAISS_SEARCH_BATCH_SIZE, dim=0):
        distances, indices = gpu_index.search(query_batch.float(), 1)
        match_count += (distances[:, 0] < squared_threshold).sum()
        del distances, indices

    del gpu_index, cpu_index, resources
    return match_count

def get_correspondences(src_pcd, tgt_pcd, trans, search_voxel_size, K=None):
    src_pcd.transform(trans)
    pcd_tree = o3d.geometry.KDTreeFlann(tgt_pcd)

    correspondences = []
    for i, point in enumerate(src_pcd.points):
        [count, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            correspondences.append([i, j])
    
    correspondences = np.array(correspondences)
    correspondences = torch.from_numpy(correspondences)
    return correspondences

def get_overlap_ratio(source,target,threshold=0.03):
    """
    We compute overlap ratio from source point cloud to target point cloud
    """
    pcd_tree = o3d.geometry.KDTreeFlann(target)
    
    match_count=0
    for i, point in enumerate(source.points):
        [count, _, _] = pcd_tree.search_radius_vector_3d(point, threshold)
        if(count!=0):
            match_count+=1

    overlap_ratio = match_count / min(len(source.points), len(target.points))
    return overlap_ratio

def compute_overlap_gaussians(src_gs, tgt_gs, threshold=0.03):
    """compute the overlap ratio and correspondences between two gaussians

    Args:
        src_gs: _description_
        tgt_ts: _description_
        threshold (float, optional): _description_. Defaults to 0.03.
    """
    src_tensor = src_gs.get_xyz().detach()
    tgt_tensor = tgt_gs.get_xyz().detach()
    squared_threshold = threshold * threshold
    matched_src = _directional_overlap_count(
        tgt_tensor, src_tensor, squared_threshold)
    matched_tgt = _directional_overlap_count(
        src_tensor, tgt_tensor, squared_threshold)
    faiss_overlap_ratio = torch.minimum(
        matched_src / len(src_tensor), matched_tgt / len(tgt_tensor))
    
    return faiss_overlap_ratio

def visualize_overlap(pc1, pc2, corr):
    import matplotlib
    matplotlib.use('Agg')  # Set non-GUI backend before importing matplotlib components
    import matplotlib.cm as cm
    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(pc1.cpu().numpy())
    tgt_pcd = o3d.geometry.PointCloud()
    tgt_pcd.points = o3d.utility.Vector3dVector(pc2.cpu().numpy())
    # corr = get_correspondences(src_pcd, tgt_pcd, np.eye(4), 0.05)
    
    color_1 = cm.tab10(0)
    color_2 = cm.tab10(1)
    overlap_color = cm.tab10(2)
    
    color_src = np.ones_like(pc1.cpu().numpy())
    color_tgt = np.ones_like(pc2.cpu().numpy())
    
    if len(corr)>0:
        color_src[corr[:,0].cpu().numpy()] = np.array(color_1)[:3]
        color_tgt[corr[:,1].cpu().numpy()] = np.array(color_2)[:3]
    
    vis_src = vis.pointcloud(pc1.cpu().numpy(), color=color_src, is_sphere=True)
    vis_tgt = vis.pointcloud(pc2.cpu().numpy(), color=color_tgt, is_sphere=True)
    vis.show_3d([vis_src, vis_tgt],[vis_src], [vis_tgt], use_new_api=True)
