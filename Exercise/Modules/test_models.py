#!/usr/bin/env python3
"""
Test script for models.py module

Tests MLPClassifier class and train_partition_classifier function.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import torch
from Exercise.Modules.models import MLPClassifier, train_partition_classifier


def test_1_mlp_classifier_import():
    """Test 1: MLPClassifier import and instantiation"""
    print('=== Test 1: MLPClassifier import ===')
    model = MLPClassifier(input_dim=784, num_partitions=100, hidden_dim=64, num_layers=3)
    print(f'✓ MLPClassifier imported successfully')
    print(f'  Class name: {model.__class__.__name__}')
    print(f'  Config: {model.get_config()}')
    print()
    return model


def test_2_forward_pass(model):
    """Test 2: Forward pass"""
    print('=== Test 2: Forward pass ===')
    x = torch.randn(32, 784)
    logits = model(x)
    print(f'  Input shape: {x.shape}')
    print(f'  Output shape: {logits.shape}')
    assert logits.shape == (32, 100), f"Expected (32, 100), got {logits.shape}"
    print(f'✓ Forward pass works')
    print()


def test_3_backward_compatibility():
    """Test 3: Backward compatibility with PartitionMLP"""
    print('=== Test 3: Backward compatibility ===')
    from Exercise.Modules.models import PartitionMLP
    model2 = PartitionMLP(input_dim=128, num_partitions=50, hidden_dim=32, num_layers=2)
    print(f'  PartitionMLP class name: {model2.__class__.__name__}')
    print(f'  Are they the same class? {PartitionMLP is MLPClassifier}')
    assert PartitionMLP is MLPClassifier, "PartitionMLP should be an alias for MLPClassifier"
    print(f'✓ PartitionMLP alias works')
    print()


def test_4_training_function():
    """Test 4: Training function uses train_partition_classifier"""
    print('=== Test 4: Training function (train_partition_classifier) ===')
    
    # Create synthetic dataset
    np.random.seed(42)
    points = np.random.randn(100, 20).astype(np.float32)
    labels = np.random.randint(0, 5, 100)
    
    # Train using train_partition_classifier
    trained_model, history = train_partition_classifier(
        points=points,
        labels=labels,
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
    
    print(f'  Trained model class: {trained_model.__class__.__name__}')
    print(f'  History keys: {list(history.keys())}')
    print(f'  Epochs trained: {len(history["train_loss"])}')
    print(f'  Final train accuracy: {history["train_acc"][-1]:.2f}%')
    print(f'  Final val accuracy: {history["val_acc"][-1]:.2f}%')
    
    assert trained_model.__class__.__name__ == 'MLPClassifier', "Should return MLPClassifier"
    assert 'train_loss' in history, "History should contain train_loss"
    assert 'train_acc' in history, "History should contain train_acc"
    assert 'val_loss' in history, "History should contain val_loss"
    assert 'val_acc' in history, "History should contain val_acc"
    assert len(history['train_loss']) == 2, "Should have 2 epochs of history"
    
    print(f'✓ train_partition_classifier works correctly')
    print()
    return trained_model


def test_5_model_inference(model):
    """Test 5: Model inference methods"""
    print('=== Test 5: Model inference ===')
    
    x = torch.randn(10, 20)
    
    # Test predict_probabilities
    probs = model.predict_probabilities(x)
    print(f'  Probabilities shape: {probs.shape}')
    assert probs.shape == (10, 5), f"Expected (10, 5), got {probs.shape}"
    assert torch.allclose(probs.sum(dim=1), torch.ones(10), atol=1e-5), "Probabilities should sum to 1"
    
    # Test predict_top_k
    top_3 = model.predict_top_k(x, k=3)
    print(f'  Top-3 shape: {top_3.shape}')
    assert top_3.shape == (10, 3), f"Expected (10, 3), got {top_3.shape}"
    
    print(f'✓ Model inference works')
    print()


if __name__ == '__main__':
    print('Testing models.py module...')
    print()
    
    # Run tests
    model = test_1_mlp_classifier_import()
    test_2_forward_pass(model)
    test_3_backward_compatibility()
    trained_model = test_4_training_function()
    test_5_model_inference(trained_model)
    
    print('✓ All tests passed!')
