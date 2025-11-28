"""
Models Module for Neural LSH Project

This module provides the MLP classifier architecture for partition prediction.
The MLP serves as the learned hash function, mapping data points to partition bins.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List


# CPU-only configuration (matches config.py)
DEVICE = "cpu"
RANDOM_SEED = 42


class MLPClassifier(nn.Module):
    """
    MLP classifier for Neural LSH partition prediction.
    
    This network learns to map input vectors to partition labels assigned by KaHIP.
    The learned mapping serves as a hash function for approximate nearest neighbor search.
    
    Architecture:
    - Input layer: input_dim → hidden_dim (ReLU)
    - Hidden layers: (num_layers - 1) × [hidden_dim → hidden_dim (ReLU)]
    - Output layer: hidden_dim → num_partitions (logits, no activation)
    
    Training uses CrossEntropyLoss which applies softmax internally.
    At inference, softmax converts logits to probabilities for top-T partition selection.
    
    Args:
        input_dim: Dimensionality of input vectors (d)
                  e.g., 784 for MNIST, 128 for SIFT
        num_partitions: Number of output classes/partitions (m)
                       e.g., 100 for typical Neural LSH configuration
        hidden_dim: Width of hidden layers (default 64)
        num_layers: Number of hidden layers (default 3)
                   Includes input and intermediate layers, excludes output
    
    Example:
        >>> model = MLPClassifier(input_dim=784, num_partitions=100, hidden_dim=64, num_layers=3)
        >>> model.to(DEVICE)
        >>> x = torch.randn(32, 784)  # batch of 32 MNIST images
        >>> logits = model(x)  # shape: [32, 100]
        >>> probs = torch.softmax(logits, dim=1)  # convert to probabilities
    """
    
    def __init__(
        self,
        input_dim: int,
        num_partitions: int,
        hidden_dim: int = 64,
        num_layers: int = 3
    ):
        super(MLPClassifier, self).__init__()
        
        # Store configuration
        self.input_dim = input_dim
        self.num_partitions = num_partitions
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Build network layers
        layers = []
        
        # Input layer: input_dim → hidden_dim
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        # Hidden layers: hidden_dim → hidden_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # Output layer: hidden_dim → num_partitions (no activation)
        layers.append(nn.Linear(hidden_dim, num_partitions))
        
        # Combine into sequential model
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input batch, shape [batch_size, input_dim]
               Should be float32 tensor on same device as model
        
        Returns:
            logits: Output logits, shape [batch_size, num_partitions]
                   Raw scores before softmax (for use with CrossEntropyLoss)
        """
        return self.network(x)
    
    def get_config(self) -> Dict[str, int]:
        """
        Return model configuration for serialization.
        
        This configuration can be saved with the model weights to enable
        reconstruction of the model architecture during loading.
        
        Returns:
            Dictionary with model architecture parameters:
            - input_dim: Input dimensionality
            - num_partitions: Number of output classes
            - hidden_dim: Hidden layer width
            - num_layers: Number of hidden layers
        """
        return {
            'input_dim': self.input_dim,
            'num_partitions': self.num_partitions,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers
        }
    
    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict partition probabilities (with softmax).
        
        Convenience method for inference that applies softmax to logits.
        
        Args:
            x: Input batch, shape [batch_size, input_dim]
        
        Returns:
            probs: Partition probabilities, shape [batch_size, num_partitions]
                  Each row sums to 1.0
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)
    
    def predict_top_k(self, x: torch.Tensor, k: int = 1) -> torch.Tensor:
        """
        Predict top-k partition indices.
        
        Args:
            x: Input batch, shape [batch_size, input_dim]
            k: Number of top partitions to return
        
        Returns:
            top_k_indices: Top-k partition indices, shape [batch_size, k]
                          Indices sorted by probability (highest first)
        """
        probs = self.predict_probabilities(x)
        _, top_k_indices = torch.topk(probs, k, dim=1)
        return top_k_indices


def train_partition_classifier(
    points: np.ndarray,
    labels: np.ndarray,
    input_dim: int,
    num_partitions: int,
    hidden_dim: int = 64,
    num_layers: int = 3,
    epochs: int = 50,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    val_split: float = 0.1,
    seed: int = 42,
    verbose: bool = True
) -> tuple:
    """
    Train MLP classifier for partition prediction.
    
    Trains a PartitionMLP to map data points to their KaHIP partition labels.
    Uses CrossEntropyLoss and Adam optimizer with train/val split for monitoring.
    
    Args:
        points: Data points, shape [n, d]
        labels: Partition labels, shape [n], values in [0, m-1]
        input_dim: Input dimensionality (should match points.shape[1])
        num_partitions: Number of partitions (should match max(labels) + 1)
        hidden_dim: Hidden layer width (default 64)
        num_layers: Number of hidden layers (default 3)
        epochs: Number of training epochs (default 50)
        batch_size: Batch size for training (default 128)
        learning_rate: Learning rate for Adam optimizer (default 0.001)
        val_split: Validation set fraction, 0.1 = 10% (default 0.1)
        seed: Random seed for reproducibility (default 42)
        verbose: Print training progress (default True)
    
    Returns:
        model: Trained MLPClassifier on CPU
        history: Training history dict with keys:
                - 'train_loss': List of training losses per epoch
                - 'train_acc': List of training accuracies per epoch
                - 'val_loss': List of validation losses per epoch
                - 'val_acc': List of validation accuracies per epoch
    
    Example:
        >>> points = np.random.randn(10000, 784).astype(np.float32)
        >>> labels = np.random.randint(0, 100, 10000)
        >>> model, history = train_partition_classifier(
        ...     points, labels, input_dim=784, num_partitions=100, epochs=10
        ... )
    """
    from torch.utils.data import TensorDataset, DataLoader

    # Basic argument checks
    if not (0.0 < val_split < 1.0):
        raise ValueError(
            f"val_split must be in (0, 1), got {val_split}"
        )
    
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Sanity checks on data shapes / label ranges
    if points.ndim != 2:
        raise ValueError(f"points must be a 2D array, got shape {points.shape}")
    
    if points.shape[1] != input_dim:
        raise ValueError(
            f"input_dim={input_dim} but points have dimensionality {points.shape[1]}"
        )
    
    if labels.ndim != 1:
        raise ValueError(f"labels must be a 1D array, got shape {labels.shape}")
    
    if labels.shape[0] != points.shape[0]:
        raise ValueError(
            f"labels length {labels.shape[0]} does not match "
            f"number of points {points.shape[0]}"
        )
    
    if labels.min() < 0 or labels.max() >= num_partitions:
        raise ValueError(
            f"Partition labels out of range: min={labels.min()}, max={labels.max()}, "
            f"expected in [0, {num_partitions - 1}]"
        )
    
    n_samples = len(points)
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val
    
    if verbose:
        print(f"Training partition classifier:")
        print(f"  Data: {n_samples} samples ({n_train} train, {n_val} val)")
        print(f"  Architecture: {input_dim} → {hidden_dim}×{num_layers} → {num_partitions}")
        print(f"  Hyperparams: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")
        print(f"  Device: {DEVICE}")
        print()
    
    # Create train/val split
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_points = points[train_indices]
    train_labels = labels[train_indices]
    val_points = points[val_indices]
    val_labels = labels[val_indices]
    
    # Convert to torch tensors
    train_points_t = torch.from_numpy(train_points).float()
    train_labels_t = torch.from_numpy(train_labels).long()
    val_points_t = torch.from_numpy(val_points).float()
    val_labels_t = torch.from_numpy(val_labels).long()
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(train_points_t, train_labels_t)
    val_dataset = TensorDataset(val_points_t, val_labels_t)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Initialize model
    model = MLPClassifier(input_dim, num_partitions, hidden_dim, num_layers)
    model.to(DEVICE)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_points, batch_labels in train_loader:
            batch_points = batch_points.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(batch_points)
            loss = criterion(logits, batch_labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item() * len(batch_labels)
            predictions = torch.argmax(logits, dim=1)
            train_correct += (predictions == batch_labels).sum().item()
            train_total += len(batch_labels)
        
        # Average training metrics
        avg_train_loss = train_loss / train_total
        avg_train_acc = 100.0 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_points, batch_labels in val_loader:
                batch_points = batch_points.to(DEVICE)
                batch_labels = batch_labels.to(DEVICE)
                
                # Forward pass
                logits = model(batch_points)
                loss = criterion(logits, batch_labels)
                
                # Track metrics
                val_loss += loss.item() * len(batch_labels)
                predictions = torch.argmax(logits, dim=1)
                val_correct += (predictions == batch_labels).sum().item()
                val_total += len(batch_labels)
        
        # Average validation metrics
        avg_val_loss = val_loss / val_total
        avg_val_acc = 100.0 * val_correct / val_total
        
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)
        
        # Print progress
        if verbose:
            print(f"Epoch [{epoch+1:3d}/{epochs}] - "
                  f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:5.2f}%, "
                  f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:5.2f}%")
    
    if verbose:
        print()
        print(f"Training complete!")
        print(f"  Final Train Acc: {history['train_acc'][-1]:.2f}%")
        print(f"  Final Val Acc: {history['val_acc'][-1]:.2f}%")
    
    return model, history


# Alias for backward compatibility
PartitionMLP = MLPClassifier
