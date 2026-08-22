import torch


def build_evaluation_frame_ids(num_frames: int, stride: int) -> list[int]:
    if stride < 1:
        raise ValueError("evaluation stride must be positive")
    if num_frames < 0:
        raise ValueError("num_frames must be non-negative")
    if num_frames == 0:
        return []
    frame_ids = list(range(0, num_frames, stride))
    frame_ids.append(num_frames - 1)
    return sorted(set(frame_ids))


def trajectory_status(dataset) -> str:
    return (
        "available"
        if bool(getattr(dataset, "has_ground_truth", False))
        else "skipped_no_ground_truth"
    )


def masked_depth_l1(rendered_depth, ground_truth_depth):
    valid = (
        torch.isfinite(rendered_depth)
        & torch.isfinite(ground_truth_depth)
        & (ground_truth_depth > 0)
    )
    valid_count = int(valid.sum().item())
    if valid_count == 0:
        return rendered_depth.new_tensor(float("nan")), 0
    return torch.abs(
        rendered_depth[valid] - ground_truth_depth[valid]).mean(), valid_count


def assign_frames_to_submaps(frame_ids, submap_keyframe_ids):
    if not submap_keyframe_ids:
        raise ValueError("at least one submap is required")
    intervals = []
    for keyframe_ids in submap_keyframe_ids:
        if not keyframe_ids:
            raise ValueError("submaps must contain at least one keyframe")
        intervals.append((min(keyframe_ids), max(keyframe_ids)))

    assignments = {index: [] for index in range(len(intervals))}
    for frame_id in frame_ids:
        def interval_distance(interval):
            start, end = interval
            if frame_id < start:
                return start - frame_id
            if frame_id > end:
                return frame_id - end
            return 0

        submap_id = min(
            range(len(intervals)),
            key=lambda index: (interval_distance(intervals[index]), index),
        )
        assignments[submap_id].append(frame_id)
    return assignments


def assert_compatible_protocols(protocols):
    if not protocols:
        return None
    reference = protocols[0]
    for protocol in protocols[1:]:
        if protocol != reference:
            raise ValueError("formal evaluation protocol mismatch")
    return reference
