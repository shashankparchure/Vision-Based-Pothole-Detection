from typing import Any, Dict, Optional

import numpy as np


def extract_depth_features(
    mask: np.ndarray, depth_map: np.ndarray
) -> Optional[Dict[str, Any]]:
    """
    Extract depth statistics from pixels where mask == 1.

    Args:
        mask: Binary mask array with foreground as 1.
        depth_map: Depth map array with same shape as mask.

    Returns:
        Dictionary containing:
        mean_depth, max_depth, min_depth, depth_std, depth_range,
        area, and p90_depth.
        Returns None when no pothole pixels are present in mask.
    """
    mask_array = np.asarray(mask)
    depth_array = np.asarray(depth_map, dtype=np.float32)

    if mask_array.shape != depth_array.shape:
        raise ValueError("mask and depth_map must have the same shape")

    # Normalize depth map to [0, 1] before computing features.
    depth_min_all = float(np.min(depth_array))
    depth_max_all = float(np.max(depth_array))
    if depth_max_all > depth_min_all:
        depth_array = (depth_array - depth_min_all) / (depth_max_all - depth_min_all)
    else:
        depth_array = np.zeros_like(depth_array, dtype=np.float32)

    depth_values = depth_array[mask_array == 1]
    area = int(depth_values.size)

    if area == 0:
        return None

    max_depth = float(np.max(depth_values))
    min_depth = float(np.min(depth_values))

    return {
        "mean_depth": float(np.mean(depth_values)),
        "max_depth": max_depth,
        "min_depth": min_depth,
        "depth_std": float(np.std(depth_values)),
        "depth_range": float(max_depth - min_depth),
        "area": area,
        "p90_depth": float(np.percentile(depth_values, 90)),
    }
