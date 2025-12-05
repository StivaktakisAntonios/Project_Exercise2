import numpy as np
from typing import Tuple, Optional, List, Set, Dict
from tqdm import tqdm


def build_knn(
    points: np.ndarray,
    k: int,
    batch_size: int = 1000,
    max_points: Optional[int] = None,
    show_progress: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Build k-NN graph using brute-force Euclidean distance."""
    # Validate input
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
    """Compute statistics about the k-NN graph."""
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


def symmetrize_graph(
    indices: np.ndarray,
    distances: np.ndarray
) -> Tuple[List[Set[int]], Dict[Tuple[int, int], dict]]:
    """Convert directed k-NN graph to undirected graph with edge mutuality tracking."""
    n, k = indices.shape
    
    # Track directed edges and their distances
    directed_edges = {}  # (u, v) -> distance
    
    # Collect all directed edges from k-NN graph
    for u in range(n):
        for j in range(k):
            v = indices[u, j]
            # Φιλτράρουμε invalid neighbors (π.χ. -1 από C++)
            if v < 0 or v >= n:
                continue
            dist = distances[u, j]
            directed_edges[(u, v)] = dist
    
    # Build undirected graph and edge info
    adjacency = [set() for _ in range(n)]
    edge_info = {}
    processed_edges = set()  # Track which undirected edges we've processed
    
    # Process all directed edges
    for (u, v), dist_uv in directed_edges.items():
        # Canonical edge representation (smaller index first)
        edge_key = (min(u, v), max(u, v))
        
        if edge_key in processed_edges:
            continue  # Already processed this undirected edge
        
        processed_edges.add(edge_key)
        
        # Check if reverse edge exists
        reverse_exists = (v, u) in directed_edges
        
        if reverse_exists:
            # Mutual edge: both u→v and v→u exist
            dist_vu = directed_edges[(v, u)]
            avg_distance = (dist_uv + dist_vu) / 2.0
            
            edge_info[edge_key] = {
                'distance': float(avg_distance),
                'mutual': True
            }
        else:
            # Non-mutual edge: only one direction exists
            edge_info[edge_key] = {
                'distance': float(dist_uv),
                'mutual': False
            }
        
        # Add to adjacency list (both directions for undirected graph)
        adjacency[u].add(v)
        adjacency[v].add(u)
    
    return adjacency, edge_info


def assign_edge_weights(edge_info: Dict[Tuple[int, int], dict]) -> Dict[Tuple[int, int], int]:
    """Assign integer weights to edges: weight=2 for mutual edges, weight=1 for non-mutual."""
    edge_weights = {}
    
    for edge_key, info in edge_info.items():
        if info['mutual']:
            edge_weights[edge_key] = 2
        else:
            edge_weights[edge_key] = 1
    
    return edge_weights


def build_weighted_graph(
    indices: np.ndarray,
    distances: np.ndarray
) -> Tuple[List[Set[int]], Dict[Tuple[int, int], dict]]:
    """Build weighted undirected graph from k-NN graph for KaHIP partitioning."""
    # Symmetrize graph and get edge info
    adjacency, edge_info = symmetrize_graph(indices, distances)
    
    # Assign weights based on mutuality
    edge_weights = assign_edge_weights(edge_info)
    
    # Combine into single edge_data dict
    edge_data = {}
    for edge_key, info in edge_info.items():
        edge_data[edge_key] = {
            'distance': info['distance'],
            'mutual': info['mutual'],
            'weight': edge_weights[edge_key]
        }
    
    return adjacency, edge_data


import numpy as np

def load_knn_indices_bin(path: str):
    """Load k-NN indices from C++ binary file (format: [n][k][n*k ids])."""
    with open(path, "rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=2)
        if header.size < 2:
            raise ValueError(f"File {path} is too short or invalid.")
        n, k = int(header[0]), int(header[1])
        ids = np.fromfile(f, dtype=np.int32)

    if ids.size != n * k:
        raise ValueError(
            f"File {path} is inconsistent: expected {n*k} ids, got {ids.size}."
        )

    indices = ids.reshape(n, k).astype(np.int64)

    # Για KaHIP/weights χρειαζόμαστε μόνο τη δομή του γράφου (ποιοι είναι γείτονες).
    # Οι αποστάσεις δεν επηρεάζουν τα βάρη 1/2 (mutual/non-mutual),
    # άρα μπορούμε να βάλουμε dummy τιμές.
    distances = np.ones((n, k), dtype=np.float32)

    return indices, distances
