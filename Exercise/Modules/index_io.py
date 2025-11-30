#!/usr/bin/env python3
"""
index_io.py — Index Persistence Module

Handles saving and loading of Neural LSH index components:
- Inverted index (bins.npz): mapping from partition ID to list of point indices
- Trained model (model.pth): PyTorch state dict
- Metadata (meta.json): configuration and parameters
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import torch

from Exercise.Modules.models import MLPClassifier


def build_inverted_index(labels: np.ndarray) -> Dict[int, np.ndarray]:
    """
    Build inverted index from partition labels.
    
    Args:
        labels: Array of shape [n] with partition assignments (0 to m-1)
        
    Returns:
        Dictionary mapping partition_id -> array of point indices
        
    Example:
        >>> labels = np.array([0, 2, 0, 1, 2])
        >>> bins = build_inverted_index(labels)
        >>> bins[0]  # Points 0 and 2 in partition 0
        array([0, 2])
        >>> bins[1]  # Point 3 in partition 1
        array([3])
    """
    n = len(labels)
    num_partitions = int(labels.max()) + 1
    
    # Build bins dictionary
    bins = {}
    for partition_id in range(num_partitions):
        indices = np.where(labels == partition_id)[0]
        bins[partition_id] = indices
    
    # Verify all points mapped
    total_mapped = sum(len(v) for v in bins.values())
    assert total_mapped == n, f"Mismatch: {total_mapped} mapped vs {n} total points"
    
    return bins


def save_index(
    index_path: str,
    bins: Dict[int, np.ndarray],
    model: MLPClassifier,
    metadata: Dict[str, Any]
) -> None:
    """
    Save Neural LSH index to disk.
    
    Creates directory structure:
        index_path/
            bins.npz       - inverted index
            model.pth      - trained MLP state dict
            meta.json      - metadata
    
    Args:
        index_path: Directory path for index (will be created if doesn't exist)
        bins: Dictionary mapping partition_id -> array of point indices
        model: Trained MLPClassifier instance
        metadata: Dictionary with configuration:
            - num_partitions (int): number of partitions
            - input_dim (int): feature dimension
            - dataset_type (str): 'mnist' or 'sift'
            - knn (int): k for k-NN graph
            - partition_params (dict): KaHIP parameters
            - model_config (dict): MLP architecture
            - training_params (dict): training hyperparameters
            
    Raises:
        ValueError: If metadata is missing required fields
        IOError: If saving fails
    """
    index_path = Path(index_path)
    index_path.mkdir(parents=True, exist_ok=True)
    
    # Validate metadata
    required_fields = ['num_partitions', 'input_dim', 'dataset_type']
    for field in required_fields:
        if field not in metadata:
            raise ValueError(f"Metadata missing required field: {field}")
    
    # Save inverted index (bins.npz)
    # Convert dict to arrays for npz format
    bins_path = index_path / 'bins.npz'
    bin_data = {}
    for partition_id, indices in bins.items():
        bin_data[f'bin_{partition_id}'] = indices
    np.savez_compressed(bins_path, **bin_data)
    
    # Save model (model.pth)
    model_path = index_path / 'model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model.get_config()
    }, model_path)
    
    # Save metadata (meta.json)
    meta_path = index_path / 'meta.json'
    # Add bin statistics to metadata
    bin_stats = {
        'num_bins': len(bins),
        'bin_sizes': {int(k): len(v) for k, v in bins.items()},
        'total_points': sum(len(v) for v in bins.values()),
        'min_bin_size': min(len(v) for v in bins.values()),
        'max_bin_size': max(len(v) for v in bins.values()),
        'avg_bin_size': sum(len(v) for v in bins.values()) / len(bins)
    }
    metadata['bin_statistics'] = bin_stats
    
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Index saved to: {index_path}")
    print(f"  bins.npz: {len(bins)} partitions, {bin_stats['total_points']} points")
    print(f"  model.pth: {model.get_config()}")
    print(f"  meta.json: {len(metadata)} fields")


def load_index(index_path: str) -> Tuple[Dict[int, np.ndarray], MLPClassifier, Dict[str, Any]]:
    """
    Load Neural LSH index from disk.
    
    Args:
        index_path: Directory path containing index files
        
    Returns:
        Tuple of (bins, model, metadata):
            - bins: Dictionary mapping partition_id -> array of point indices
            - model: Loaded MLPClassifier instance
            - metadata: Configuration dictionary
            
    Raises:
        FileNotFoundError: If index files are missing
        ValueError: If index structure is invalid
    """
    index_path = Path(index_path)
    
    if not index_path.exists():
        raise FileNotFoundError(f"Index directory not found: {index_path}")
    
    bins_path = index_path / 'bins.npz'
    model_path = index_path / 'model.pth'
    meta_path = index_path / 'meta.json'
    
    # Check all required files exist
    for path in [bins_path, model_path, meta_path]:
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")
    
    # Load metadata
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
    
    # Validate metadata
    required_fields = ['num_partitions', 'input_dim', 'dataset_type']
    for field in required_fields:
        if field not in metadata:
            raise ValueError(f"Metadata missing required field: {field}")
    
    # Load bins
    bins_data = np.load(bins_path)
    bins = {}
    for key in bins_data.keys():
        if key.startswith('bin_'):
            partition_id = int(key.split('_')[1])
            bins[partition_id] = bins_data[key]
    
    # Verify bin count
    if len(bins) != metadata['num_partitions']:
        raise ValueError(
            f"Bin count mismatch: {len(bins)} loaded vs "
            f"{metadata['num_partitions']} in metadata"
        )
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    model_config = checkpoint['model_config']
    
    # Create model instance
    model = MLPClassifier(
        input_dim=model_config['input_dim'],
        num_partitions=model_config['num_partitions'],
        hidden_dim=model_config['hidden_dim'],
        num_layers=model_config['num_layers']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode
    
    print(f"Index loaded from: {index_path}")
    print(f"  {len(bins)} partitions, {metadata['bin_statistics']['total_points']} points")
    print(f"  Model: {model_config}")
    
    return bins, model, metadata


def validate_index(
    bins: Dict[int, np.ndarray],
    model: MLPClassifier,
    metadata: Dict[str, Any],
    dataset_size: int = None
) -> bool:
    """
    Validate loaded index structure and consistency.
    
    Args:
        bins: Inverted index dictionary
        model: Loaded model
        metadata: Metadata dictionary
        dataset_size: Expected dataset size (optional)
        
    Returns:
        True if validation passes
        
    Raises:
        ValueError: If validation fails
    """
    # Check bin structure
    num_partitions = metadata['num_partitions']
    if len(bins) != num_partitions:
        raise ValueError(f"Bin count {len(bins)} != metadata {num_partitions}")
    
    # Check all partition IDs present
    for i in range(num_partitions):
        if i not in bins:
            raise ValueError(f"Missing partition {i}")
    
    # Check model config matches metadata
    model_config = model.get_config()
    if model_config['num_partitions'] != num_partitions:
        raise ValueError(
            f"Model partitions {model_config['num_partitions']} != "
            f"metadata {num_partitions}"
        )
    
    if model_config['input_dim'] != metadata['input_dim']:
        raise ValueError(
            f"Model input_dim {model_config['input_dim']} != "
            f"metadata {metadata['input_dim']}"
        )
    
    # Check total points if dataset size provided
    total_points = sum(len(v) for v in bins.values())
    if dataset_size is not None and total_points != dataset_size:
        raise ValueError(
            f"Total points in bins {total_points} != dataset size {dataset_size}"
        )
    
    # Check for duplicate indices across bins
    all_indices = []
    for indices in bins.values():
        all_indices.extend(indices.tolist())
    if len(all_indices) != len(set(all_indices)):
        raise ValueError("Duplicate indices found across bins")
    
    print("✓ Index validation passed")
    return True


if __name__ == '__main__':
    """
    Test index_io module functionality
    """
    print("Testing index_io module...")
    print()
    
    # Test 1: Build inverted index
    print("=== Test 1: Build Inverted Index ===")
    labels = np.array([0, 2, 0, 1, 2, 1, 0, 3, 3, 2])
    bins = build_inverted_index(labels)
    print(f"Labels: {labels}")
    print(f"Bins: {bins}")
    assert len(bins) == 4, "Should have 4 bins"
    assert len(bins[0]) == 3, "Bin 0 should have 3 points"
    assert len(bins[1]) == 2, "Bin 1 should have 2 points"
    print("✓ Inverted index built correctly")
    print()
    
    # Test 2: Save and load index
    print("=== Test 2: Save and Load Index ===")
    
    # Create synthetic data
    from Exercise.Modules.models import train_partition_classifier
    np.random.seed(42)
    points = np.random.randn(100, 20).astype(np.float32)
    labels_train = np.random.randint(0, 5, 100)
    
    # Train a small model
    model, _ = train_partition_classifier(
        points=points,
        labels=labels_train,
        input_dim=20,
        num_partitions=5,
        hidden_dim=16,
        num_layers=2,
        epochs=2,
        batch_size=32,
        learning_rate=0.001,
        val_split=0.1,
        seed=42,
        verbose=False
    )
    
    # Build bins
    bins_test = build_inverted_index(labels_train)
    
    # Prepare metadata
    metadata = {
        'num_partitions': 5,
        'input_dim': 20,
        'dataset_type': 'test',
        'knn': 10,
        'partition_params': {'imbalance': 0.03, 'mode': 2},
        'model_config': model.get_config(),
        'training_params': {'epochs': 2, 'lr': 0.001}
    }
    
    # Save index
    test_index_path = '/tmp/test_nlsh_index'
    save_index(test_index_path, bins_test, model, metadata)
    print()
    
    # Load index
    print("Loading index...")
    bins_loaded, model_loaded, metadata_loaded = load_index(test_index_path)
    print()
    
    # Validate
    print("Validating index...")
    validate_index(bins_loaded, model_loaded, metadata_loaded, dataset_size=100)
    print()
    
    # Verify bins match
    assert len(bins_loaded) == len(bins_test), "Bin count mismatch"
    for k in bins_test.keys():
        assert np.array_equal(bins_loaded[k], bins_test[k]), f"Bin {k} mismatch"
    print("✓ Bins match after save/load")
    
    # Verify model produces same output
    test_input = torch.randn(10, 20)
    with torch.no_grad():
        output_original = model(test_input)
        output_loaded = model_loaded(test_input)
    assert torch.allclose(output_original, output_loaded, atol=1e-6), "Model output mismatch"
    print("✓ Model produces identical output after save/load")
    print()
    
    # Clean up
    import shutil
    shutil.rmtree(test_index_path)
    
    print("✓ All tests passed!")
