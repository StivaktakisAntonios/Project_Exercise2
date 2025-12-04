"""
Dataset Parser Module for Neural LSH Project

This module provides binary data loaders for MNIST and SIFT datasets.
Compatible with Assignment 1 data formats:
- MNIST: idx3-ubyte format (input.idx3-ubyte, query.idx3-ubyte)
- SIFT: fvecs format (sift_base.fvecs, sift_query.fvecs)
"""

import numpy as np
import os
import struct
from typing import Literal


def load_points(path: str, dtype: Literal["mnist", "sift"]) -> np.ndarray:
    """
    Load dataset points from binary file.
    
    Args:
        path: Path to binary file containing dataset vectors
        dtype: Dataset type, either "mnist" or "sift"
    
    Returns:
        numpy.ndarray: Array of shape [n, d] where n is number of vectors
                      and d is dimensionality
    
    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If binary format is invalid or dtype is unsupported
    
    Example:
        >>> points = load_points("input.idx3-ubyte", "mnist")
        >>> print(points.shape)  # (60000, 784) for MNIST
        >>> points = load_points("sift_base.fvecs", "sift")
        >>> print(points.shape)  # (1000000, 128) for SIFT
    """
    if dtype == "mnist":
        return _load_mnist_idx(path)
    elif dtype == "sift":
        return _load_sift_fvecs(path)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}. Must be 'mnist' or 'sift'")


def load_queries(path: str, dtype: Literal["mnist", "sift"]) -> np.ndarray:
    """
    Load query points from binary file.
    
    Args:
        path: Path to binary file containing query vectors
        dtype: Dataset type, either "mnist" or "sift"
    
    Returns:
        numpy.ndarray: Array of shape [q, d] where q is number of queries
                      and d is dimensionality
    
    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If binary format is invalid or dtype is unsupported
    
    Example:
        >>> queries = load_queries("query.idx3-ubyte", "mnist")
        >>> print(queries.shape)  # (100, 784) for MNIST queries
        >>> queries = load_queries("sift_query.fvecs", "sift")
        >>> print(queries.shape)  # (10000, 128) for SIFT queries
    """
    if dtype == "mnist":
        return _load_mnist_idx(path)
    elif dtype == "sift":
        return _load_sift_fvecs(path)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}. Must be 'mnist' or 'sift'")


def _load_mnist_idx(path: str) -> np.ndarray:
    """
    Load MNIST data from idx3-ubyte format.
    
    IDX file format for 3D data (images):
    - Magic number: 0x00000803 (4 bytes, big-endian int32) = 2051
    - Number of images: n (4 bytes, big-endian int32)
    - Number of rows: 28 (4 bytes, big-endian int32)
    - Number of columns: 28 (4 bytes, big-endian int32)
    - Image data: n × 28 × 28 unsigned bytes
    
    Args:
        path: Path to idx3-ubyte file
    
    Returns:
        numpy.ndarray: Flattened images with shape [n, 784]
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"MNIST file not found: {path}")
    
    with open(path, 'rb') as f:
        # Read magic number (big-endian)
        magic = struct.unpack('>I', f.read(4))[0]
        
        if magic != 2051:  # 0x00000803
            raise ValueError(f"Invalid MNIST magic number: {magic} (expected 2051)")
        
        # Read dimensions (big-endian)
        n_images = struct.unpack('>I', f.read(4))[0]
        n_rows = struct.unpack('>I', f.read(4))[0]
        n_cols = struct.unpack('>I', f.read(4))[0]
        
        # Validate dimensions
        if n_images <= 0 or n_rows <= 0 or n_cols <= 0:
            raise ValueError(f"Invalid MNIST dimensions: n={n_images}, rows={n_rows}, cols={n_cols}")
        
        # Read image data as unsigned bytes
        data = np.fromfile(f, dtype=np.uint8, count=n_images * n_rows * n_cols)
        
        # Validate data size
        expected_size = n_images * n_rows * n_cols
        if len(data) != expected_size:
            raise ValueError(
                f"Invalid MNIST data size: expected {expected_size} bytes, got {len(data)}"
            )
        
        # Reshape to [n, rows, cols] then flatten to [n, rows*cols]
        images = data.reshape(n_images, n_rows, n_cols)
        vectors = images.reshape(n_images, n_rows * n_cols)
        
        # Convert to float32
        vectors = vectors.astype(np.float32)
    
    return vectors


def _load_sift_fvecs(path: str) -> np.ndarray:
    """
    Load SIFT data from fvecs format.
    
    FVECS file format:
    - For each vector:
        - Dimension: d (4 bytes, little-endian int32)
        - Components: d float32 values (little-endian)
    - All vectors have the same dimension
    
    Args:
        path: Path to fvecs file
    
    Returns:
        numpy.ndarray: Vectors with shape [n, d]
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"SIFT file not found: {path}")
    
    vectors = []
    dim = None
    
    with open(path, 'rb') as f:
        while True:
            # Read dimension (4 bytes, little-endian)
            dim_bytes = f.read(4)
            if not dim_bytes:
                break  # End of file
            
            if len(dim_bytes) != 4:
                raise ValueError(f"Incomplete dimension field in {path}")
            
            d = struct.unpack('<i', dim_bytes)[0]
            
            # Validate dimension
            if d <= 0:
                raise ValueError(f"Invalid dimension in {path}: d={d}")
            
            # Check dimension consistency
            if dim is None:
                dim = d
            elif dim != d:
                raise ValueError(
                    f"Inconsistent dimensions in {path}: expected {dim}, got {d}"
                )
            
            # Read vector components (d float32 values, little-endian)
            vec_bytes = f.read(4 * d)
            if len(vec_bytes) != 4 * d:
                raise ValueError(f"Incomplete vector data in {path}")
            
            vec = struct.unpack(f'<{d}f', vec_bytes)
            vectors.append(vec)
    
    if not vectors:
        raise ValueError(f"No vectors found in {path}")
    
    # Convert to numpy array
    result = np.array(vectors, dtype=np.float32)
    
    return result
