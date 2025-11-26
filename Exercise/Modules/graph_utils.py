"""
Graph Utilities Module for Neural LSH Project

This module provides k-NN graph construction using pure NumPy brute-force
distance computation with batched processing for memory efficiency.
"""

import numpy as np
from typing import Tuple, Optional
from tqdm import tqdm


def build_knn(
    points: np.ndarray,
    k: int,
    batch_size: int = 1000,
    max_points: Optional[int] = None,
    show_progress: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build k-NN graph using brute-force Euclidean distance.
    
    Uses batched distance computation for memory efficiency on large datasets.
    Points are processed in batches to avoid memory overflow.
    
    Args:
        points: Dataset points, shape [n, d]
        k: Number of nearest neighbors per point
        batch_size: Process points in batches for memory efficiency (default: 1000)
        max_points: Optional limit on number of points to process (e.g., first 100k
                   for SIFT). If None, process all points. Useful for testing or
                   reducing computational cost on large datasets.
        show_progress: Show tqdm progress bar during construction (default: True)
    
    Returns:
        indices: Neighbor indices, shape [n_used, k] (int64)
        distances: Neighbor distances, shape [n_used, k] (float32)
        
        where n_used = min(n, max_points) if max_points is set, else n
    
    Raises:
        ValueError: If k <= 0, points is not 2D, or k >= n_used
    
    Example:
        >>> points = np.random.randn(1000, 128).astype(np.float32)
        >>> indices, distances = build_knn(points, k=10)
        >>> print(indices.shape)  # (1000, 10)
        
        >>> # Process only first 100k points from large dataset
        >>> indices, distances = build_knn(points, k=10, max_points=100000)
    """
    # Validate inputs
    if points.ndim != 2:
        raise ValueError(f"Points must be 2D array, got shape {points.shape}")
    
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    
    n, d = points.shape
    
    # Apply max_points limit if specified
    if max_points is not None and max_points < n:
        points = points[:max_points]
        n = max_points
    
    if k >= n:
        raise ValueError(
            f"k ({k}) must be less than number of points ({n}). "
            f"Cannot find {k} neighbors from {n} points."
        )
    
    # Prepare output arrays
    knn_indices = np.zeros((n, k), dtype=np.int64)
    knn_distances = np.zeros((n, k), dtype=np.float32)
    
    # Compute k-NN in batches
    n_batches = (n + batch_size - 1) // batch_size
    
    iterator = range(n_batches)
    if show_progress:
        iterator = tqdm(iterator, desc="Building k-NN graph", unit="batch")
    
    for batch_idx in iterator:
        # Get batch range
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n)
        batch_points = points[start_idx:end_idx]
        
        # Compute pairwise squared distances for this batch
        # Using formula: ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x·y
        # This is more memory efficient than computing (x - y)^2 directly
        batch_sq_norms = np.sum(batch_points ** 2, axis=1, keepdims=True)  # [batch, 1]
        points_sq_norms = np.sum(points ** 2, axis=1)  # [n]
        dot_products = np.dot(batch_points, points.T)  # [batch, n]
        
        squared_distances = batch_sq_norms + points_sq_norms - 2 * dot_products
        
        # Handle numerical errors (negative values due to floating point precision)
        squared_distances = np.maximum(squared_distances, 0)
        
        # Take square root to get actual Euclidean distances
        distances = np.sqrt(squared_distances).astype(np.float32)
        
        # For each point in batch, find k+1 nearest neighbors
        # (k+1 because the point itself will be among them with distance 0)
        # Then exclude the point itself
        
        # Use argpartition for efficiency (only partially sorts)
        # k+1 to account for self
        k_plus_1 = min(k + 1, n)
        nearest_indices = np.argpartition(distances, k_plus_1 - 1, axis=1)[:, :k_plus_1]
        
        # Get corresponding distances
        batch_indices = np.arange(end_idx - start_idx).reshape(-1, 1)
        nearest_distances = distances[batch_indices, nearest_indices]
        
        # Sort the k+1 nearest neighbors by distance
        sort_order = np.argsort(nearest_distances, axis=1)
        nearest_indices = np.take_along_axis(nearest_indices, sort_order, axis=1)
        nearest_distances = np.take_along_axis(nearest_distances, sort_order, axis=1)
        
        # Remove self (should be first with distance ~0)
        # Check if first neighbor is self
        global_indices = np.arange(start_idx, end_idx).reshape(-1, 1)
        is_self = (nearest_indices[:, 0:1] == global_indices)
        
        if np.all(is_self):
            # Self is first neighbor, remove it
            knn_indices[start_idx:end_idx] = nearest_indices[:, 1:k+1]
            knn_distances[start_idx:end_idx] = nearest_distances[:, 1:k+1]
        else:
            # Self might not be first (shouldn't happen with exact distances)
            # Remove self wherever it appears
            for i in range(end_idx - start_idx):
                global_idx = start_idx + i
                mask = nearest_indices[i] != global_idx
                knn_indices[global_idx] = nearest_indices[i][mask][:k]
                knn_distances[global_idx] = nearest_distances[i][mask][:k]
    
    return knn_indices, knn_distances


def get_graph_statistics(indices: np.ndarray, distances: np.ndarray) -> dict:
    """
    Compute statistics about the k-NN graph.
    
    Args:
        indices: Neighbor indices, shape [n, k]
        distances: Neighbor distances, shape [n, k]
    
    Returns:
        Dictionary with statistics:
        - n_points: Number of points
        - k: Number of neighbors per point
        - mean_distance: Mean distance to neighbors
        - std_distance: Std deviation of distances
        - min_distance: Minimum neighbor distance
        - max_distance: Maximum neighbor distance
        - mean_kth_distance: Mean distance to k-th neighbor
    """
    n, k = indices.shape
    
    stats = {
        'n_points': n,
        'k': k,
        'mean_distance': float(np.mean(distances)),
        'std_distance': float(np.std(distances)),
        'min_distance': float(np.min(distances)),
        'max_distance': float(np.max(distances)),
        'mean_kth_distance': float(np.mean(distances[:, -1])),  # Distance to farthest neighbor
    }
    
    return stats
