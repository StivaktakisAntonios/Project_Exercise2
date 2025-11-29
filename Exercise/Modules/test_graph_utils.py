"""
Test Graph Utilities Module

This script validates k-NN graph construction to ensure neighbor indices
and distances are correct, and graph structure is sensible.
"""

import numpy as np
import sys
from graph_utils import build_knn, get_graph_statistics


def test_basic_correctness():
    """
    Test k-NN on a controlled dataset with known structure.
    """
    print("Testing basic correctness with clustered data...")
    
    # Create 3 clusters of 10 points each
    np.random.seed(42)
    cluster1 = np.random.randn(10, 2) * 0.1 + np.array([0, 0])
    cluster2 = np.random.randn(10, 2) * 0.1 + np.array([5, 0])
    cluster3 = np.random.randn(10, 2) * 0.1 + np.array([0, 5])
    
    points = np.vstack([cluster1, cluster2, cluster3]).astype(np.float32)
    
    # Build k-NN with k=5
    indices, distances = build_knn(points, k=5, show_progress=False)
    
    # Validate shape
    assert indices.shape == (30, 5), f"Expected shape (30, 5), got {indices.shape}"
    assert distances.shape == (30, 5), f"Expected shape (30, 5), got {distances.shape}"
    
    # Check that neighbors are sorted by distance
    for i in range(30):
        for j in range(4):
            assert distances[i, j] <= distances[i, j+1], \
                f"Distances not sorted for point {i}: {distances[i]}"
    
    # Check that points in same cluster are neighbors
    # Point 0 (cluster1) should have neighbors mostly from cluster1 (indices 0-9)
    neighbors_0 = indices[0]
    cluster1_neighbors = np.sum((neighbors_0 >= 0) & (neighbors_0 < 10))
    assert cluster1_neighbors >= 4, \
        f"Point 0 should have mostly cluster1 neighbors, got {neighbors_0}"
    
    print("✓ Basic correctness: Shape, sorting, and cluster structure correct")


def test_self_loop_exclusion():
    """
    Validate that points are not their own neighbors.
    """
    print("Testing self-loop exclusion...")
    
    np.random.seed(42)
    points = np.random.randn(50, 10).astype(np.float32)
    indices, distances = build_knn(points, k=5, show_progress=False)
    
    # Check no self-loops
    for i in range(50):
        assert i not in indices[i], \
            f"Point {i} is its own neighbor: {indices[i]}"
    
    # Check all distances are positive (no zero distance to self)
    assert np.all(distances > 0), \
        f"Found zero or negative distances (self-loops)"
    
    print("✓ Self-loop exclusion: No point is its own neighbor")


def test_distance_correctness():
    """
    Validate distance computation accuracy by manual calculation.
    """
    print("Testing distance computation accuracy...")
    
    np.random.seed(42)
    points = np.random.randn(20, 5).astype(np.float32)
    indices, distances = build_knn(points, k=3, show_progress=False)
    
    # Manually compute distances for first point
    point_0 = points[0]
    neighbor_indices = indices[0]
    
    for j, neighbor_idx in enumerate(neighbor_indices):
        neighbor = points[neighbor_idx]
        manual_dist = np.sqrt(np.sum((point_0 - neighbor) ** 2))
        reported_dist = distances[0, j]
        
        # Check within floating point precision
        assert np.abs(manual_dist - reported_dist) < 1e-5, \
            f"Distance mismatch: manual={manual_dist}, reported={reported_dist}"
    
    # Verify that reported neighbors are actually the k nearest
    all_distances = np.sqrt(np.sum((points - point_0) ** 2, axis=1))
    all_distances[0] = np.inf  # Exclude self
    k_nearest_manual = np.argsort(all_distances)[:3]
    
    assert set(neighbor_indices) == set(k_nearest_manual), \
        f"Neighbor mismatch: reported={neighbor_indices}, expected={k_nearest_manual}"
    
    print("✓ Distance correctness: Distances match manual computation")


def test_max_points_parameter():
    """
    Verify subset processing with max_points parameter.
    """
    print("Testing max_points parameter...")
    
    np.random.seed(42)
    points = np.random.randn(1000, 10).astype(np.float32)
    
    # Build k-NN with max_points=500
    indices, distances = build_knn(points, k=5, max_points=500, show_progress=False)
    
    # Verify output shape is [500, 5] not [1000, 5]
    assert indices.shape == (500, 5), \
        f"Expected shape (500, 5) with max_points=500, got {indices.shape}"
    assert distances.shape == (500, 5), \
        f"Expected shape (500, 5) with max_points=500, got {distances.shape}"
    
    # Verify all indices are within [0, 500) range
    assert np.all(indices >= 0) and np.all(indices < 500), \
        "Indices should be in range [0, 500) when max_points=500"
    
    print("✓ max_points parameter: Subset processing works correctly")


def test_edge_cases():
    """
    Test edge cases: small k, large k, different batch sizes.
    """
    print("Testing edge cases...")
    
    np.random.seed(42)
    points = np.random.randn(100, 10).astype(np.float32)
    
    # Test k=1
    indices, distances = build_knn(points, k=1, show_progress=False)
    assert indices.shape == (100, 1), f"k=1 failed: got shape {indices.shape}"
    print("  ✓ k=1 works")
    
    # Test larger k
    indices, distances = build_knn(points, k=20, show_progress=False)
    assert indices.shape == (100, 20), f"k=20 failed: got shape {indices.shape}"
    print("  ✓ k=20 works")
    
    # Test different batch sizes
    indices1, distances1 = build_knn(points, k=5, batch_size=10, show_progress=False)
    indices2, distances2 = build_knn(points, k=5, batch_size=50, show_progress=False)
    
    # Results should be identical regardless of batch size
    assert np.array_equal(indices1, indices2), "Different batch sizes give different results"
    assert np.allclose(distances1, distances2), "Different batch sizes give different distances"
    print("  ✓ Different batch sizes produce consistent results")
    
    # Test error handling: k >= n
    try:
        build_knn(points, k=100, show_progress=False)
        assert False, "Should raise ValueError when k >= n"
    except ValueError as e:
        assert "must be less than" in str(e).lower()
        print("  ✓ k >= n properly raises ValueError")
    
    print("✓ Edge cases: All handled correctly")


def test_graph_statistics():
    """
    Verify get_graph_statistics() returns reasonable values.
    """
    print("Testing graph statistics...")
    
    np.random.seed(42)
    points = np.random.randn(100, 10).astype(np.float32)
    indices, distances = build_knn(points, k=5, show_progress=False)
    
    stats = get_graph_statistics(indices, distances)
    
    # Check all expected keys present
    expected_keys = ['n_points', 'k', 'mean_distance', 'std_distance', 
                     'min_distance', 'max_distance', 'mean_kth_distance']
    for key in expected_keys:
        assert key in stats, f"Missing key: {key}"
    
    # Check values are reasonable
    assert stats['n_points'] == 100
    assert stats['k'] == 5
    assert stats['mean_distance'] > 0
    assert stats['std_distance'] >= 0
    assert stats['min_distance'] > 0  # No self-loops
    assert stats['max_distance'] >= stats['mean_distance']
    assert stats['mean_kth_distance'] >= stats['mean_distance']  # k-th neighbor should be farther
    
    print("✓ Graph statistics: All keys present and values reasonable")


def run_all_tests():
    """
    Run all validation tests.
    """
    print("=" * 60)
    print("k-NN Graph Construction Validation Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_basic_correctness,
        test_self_loop_exclusion,
        test_distance_correctness,
        test_max_points_parameter,
        test_edge_cases,
        test_graph_statistics,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
