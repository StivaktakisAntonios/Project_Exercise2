"""
Search module for Neural LSH.
Implements multi-probe search, exact distance computation, and metrics.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple
import time


def search_query(
    query: np.ndarray,
    model: torch.nn.Module,
    bins: Dict[int, np.ndarray],
    dataset: np.ndarray,
    T: int = 5,
    N: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Multi-probe search for a single query.
    
    Args:
        query: Query vector (d,)
        model: Trained MLP classifier
        bins: Dictionary {partition_id: array of dataset indices}
        dataset: Full dataset (n, d)
        T: Number of top bins to probe
        N: Number of nearest neighbors to return
        
    Returns:
        candidate_indices: Array of candidate indices from top-T bins
        candidate_distances: Distances to candidates
        top_n_indices: Top N neighbor indices (from candidates)
        top_n_distances: Top N neighbor distances
    """
    # Enforce CPU
    device = torch.device("cpu")
    
    # Convert query to torch tensor
    query_tensor = torch.from_numpy(query).float().unsqueeze(0).to(device)
    
    # Get model predictions and compute softmax probabilities
    model.eval()
    with torch.no_grad():
        logits = model(query_tensor)  # (1, num_classes)
        probs = F.softmax(logits, dim=1).squeeze(0)  # (num_classes,)
    
    # Select top-T bins by probability
    top_T_probs, top_T_bins = torch.topk(probs, min(T, len(probs)))
    top_T_bins = top_T_bins.cpu().numpy()
    
    # Gather candidates from selected bins
    candidates = []
    for bin_id in top_T_bins:
        if bin_id in bins:
            candidates.extend(bins[bin_id].tolist())
    
    # Remove duplicates and convert to array
    candidates = np.unique(candidates)
    
    if len(candidates) == 0:
        # No candidates found, return empty results
        return (
            np.array([], dtype=int),
            np.array([], dtype=float),
            np.array([], dtype=int),
            np.array([], dtype=float)
        )
    
    # Compute exact distances to candidates
    candidate_vectors = dataset[candidates]
    distances = np.linalg.norm(candidate_vectors - query, axis=1)
    
    # Sort by distance and select top-N
    sorted_indices = np.argsort(distances)
    top_n_idx = sorted_indices[:N]
    
    top_n_indices = candidates[top_n_idx]
    top_n_distances = distances[top_n_idx]
    
    return candidates, distances, top_n_indices, top_n_distances


def compute_ground_truth(
    query: np.ndarray,
    dataset: np.ndarray,
    N: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute exact nearest neighbors over full dataset (ground truth).
    
    Args:
        query: Query vector (d,)
        dataset: Full dataset (n, d)
        N: Number of nearest neighbors
        
    Returns:
        true_indices: Ground truth neighbor indices
        true_distances: Ground truth neighbor distances
    """
    distances = np.linalg.norm(dataset - query, axis=1)
    sorted_indices = np.argsort(distances)
    
    true_indices = sorted_indices[:N]
    true_distances = distances[true_indices]
    
    return true_indices, true_distances


def compute_recall_at_n(
    approx_indices: np.ndarray,
    true_indices: np.ndarray,
    N: int
) -> float:
    """
    Compute Recall@N: fraction of true neighbors found in approximate results.
    
    Args:
        approx_indices: Approximate neighbor indices (up to N)
        true_indices: Ground truth neighbor indices (N)
        N: Number of neighbors
        
    Returns:
        Recall@N as float
    """
    if N == 0:
        return 1.0
    
    # Count how many true neighbors appear in approximate results
    approx_set = set(approx_indices[:N])
    true_set = set(true_indices[:N])
    
    intersection = len(approx_set & true_set)
    recall = intersection / N
    
    return recall


def compute_approximation_factor(
    approx_distances: np.ndarray,
    true_distances: np.ndarray
) -> float:
    """
    Compute average approximation factor (AF) over all neighbors.
    
    AF = mean(d_approx / d_true) where d_approx is distance to approximate neighbor,
    d_true is distance to corresponding ground truth neighbor.
    
    Args:
        approx_distances: Distances to approximate neighbors
        true_distances: Distances to ground truth neighbors
        
    Returns:
        Average AF
    """
    if len(true_distances) == 0 or len(approx_distances) == 0:
        return 1.0
    
    # Compute ratio for each position
    # Avoid division by zero
    ratios = []
    for i in range(min(len(approx_distances), len(true_distances))):
        if true_distances[i] > 0:
            ratios.append(approx_distances[i] / true_distances[i])
        else:
            ratios.append(1.0)
    
    return float(np.mean(ratios)) if ratios else 1.0


def find_r_near_neighbors(
    dataset: np.ndarray,
    query: np.ndarray,
    R: float
) -> np.ndarray:
    """
    Find all points within distance R from query (range search).
    
    Args:
        dataset: Full dataset (n, d)
        query: Query vector (d,)
        R: Distance threshold
        
    Returns:
        Array of indices within distance R
    """
    distances = np.linalg.norm(dataset - query, axis=1)
    within_R = np.where(distances <= R)[0]
    return within_R


def batch_search(
    queries: np.ndarray,
    model: torch.nn.Module,
    bins: Dict[int, np.ndarray],
    dataset: np.ndarray,
    T: int = 5,
    N: int = 1,
    R: float = None,
    range_search: bool = True
) -> Tuple[List, Dict]:
    """
    Perform batch search over multiple queries and compute aggregate metrics.
    
    Args:
        queries: Query set (q, d)
        model: Trained MLP classifier
        bins: Dictionary {partition_id: array of dataset indices}
        dataset: Full dataset (n, d)
        T: Number of top bins to probe
        N: Number of nearest neighbors to return
        R: Distance threshold for range search
        range_search: Whether to perform range search
        
    Returns:
        results: List of per-query results (dicts)
        aggregate_metrics: Dict with aggregate metrics
    """
    num_queries = len(queries)
    results = []
    
    # Timing accumulators
    total_approx_time = 0.0
    total_true_time = 0.0
    
    # Metric accumulators
    total_recall = 0.0
    total_af = 0.0
    
    for query_id in range(num_queries):
        query = queries[query_id]
        
        # Approximate search
        t_start_approx = time.time()
        candidates, cand_dist, approx_indices, approx_distances = search_query(
            query, model, bins, dataset, T, N
        )
        t_approx = time.time() - t_start_approx
        
        # Ground truth (exact search)
        t_start_true = time.time()
        true_indices, true_distances = compute_ground_truth(query, dataset, N)
        t_true = time.time() - t_start_true
        
        # Range search if enabled
        r_near_indices = []
        if range_search and R is not None:
            r_near_indices = find_r_near_neighbors(dataset, query, R)
        
        # Compute metrics
        recall = compute_recall_at_n(approx_indices, true_indices, N)
        af = compute_approximation_factor(approx_distances, true_distances)
        
        # Store per-query result
        result = {
            'query_id': query_id,
            'approx_indices': approx_indices,
            'approx_distances': approx_distances,
            'true_indices': true_indices,
            'true_distances': true_distances,
            'r_near_indices': r_near_indices,
            'recall': recall,
            'af': af,
            't_approx': t_approx,
            't_true': t_true,
            'num_candidates': len(candidates)
        }
        results.append(result)
        
        # Accumulate
        total_recall += recall
        total_af += af
        total_approx_time += t_approx
        total_true_time += t_true
    
    # Compute aggregate metrics
    avg_recall = total_recall / num_queries if num_queries > 0 else 0.0
    avg_af = total_af / num_queries if num_queries > 0 else 1.0
    avg_approx_time = total_approx_time / num_queries if num_queries > 0 else 0.0
    avg_true_time = total_true_time / num_queries if num_queries > 0 else 0.0
    qps = num_queries / total_approx_time if total_approx_time > 0 else 0.0
    
    aggregate_metrics = {
        'recall_at_n': avg_recall,
        'average_af': avg_af,
        'qps': qps,
        't_approximate_average': avg_approx_time,
        't_true_average': avg_true_time,
        'num_queries': num_queries
    }
    
    return results, aggregate_metrics


def write_output_file(
    results: List[Dict],
    aggregate_metrics: Dict,
    output_path: str,
    N: int, range_search: bool = True,
):
    """
    Write search results to output file in Assignment 1 format.
    
    Format (exact match required):
    - METHOD NAME: Neural LSH
    - For each query:
      - Neural LSH
      - Query: <query_id>
      - Nearest neighbor-<i>: <data_id>
      - distanceApproximate: <double>
      - distanceTrue: <double>
      - (repeat for each of N neighbors)
      - R-near neighbors:
      - <data_id> (one per line, or none)
    - Aggregate metrics at end:
      - Average AF: <double>
      - Recall@N: <double>
      - QPS: <double>
      - tApproximateAverage: <double>
      - tTrueAverage: <double>
    
    Args:
        results: List of per-query result dicts
        aggregate_metrics: Dict with aggregate metrics
        output_path: Path to output file
        N: Number of neighbors
    """
    with open(output_path, 'w') as f:
        # Write method name header
        f.write("METHOD NAME: Neural LSH\n")
        
        # Write per-query results
        for result in results:
            query_id = result['query_id']
            approx_indices = result['approx_indices']
            approx_distances = result['approx_distances']
            true_distances = result['true_distances']
            r_near_indices = result['r_near_indices']
            
            # Query header
            f.write(f"\n")
            f.write(f"Query: {query_id}\n")
            
            # Write N neighbors
            for i in range(min(N, len(approx_indices))):
                neighbor_idx = approx_indices[i]
                dist_approx = approx_distances[i]
                
                # Get corresponding true distance for this position
                if i < len(true_distances):
                    dist_true = true_distances[i]
                else:
                    dist_true = dist_approx  # Fallback
                
                f.write(f"Nearest neighbor-{i+1}: {neighbor_idx}\n")
                f.write(f"distanceApproximate: {dist_approx}\n")
                f.write(f"distanceTrue: {dist_true}\n")
                f.write(f"\n")
            
            # R-near neighbors section (μόνο αν range_search == True)
            if range_search:
                f.write("R-near neighbors:\n")
                for idx in r_near_indices:
                    f.write(f"{idx}\n")

        
        # Write aggregate metrics
        f.write(f"Average AF: {aggregate_metrics['average_af']}\n")
        f.write(f"Recall@N: {aggregate_metrics['recall_at_n']}\n")
        f.write(f"QPS: {aggregate_metrics['qps']}\n")
        f.write(f"tApproximateAverage: {aggregate_metrics['t_approximate_average']}\n")
        f.write(f"tTrueAverage: {aggregate_metrics['t_true_average']}\n")

