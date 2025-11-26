"""
Test Dataset Parser Module

This script validates the MNIST idx3-ubyte and SIFT fvecs loaders
by creating synthetic binary data and testing the parsing logic.
"""

import numpy as np
import struct
import tempfile
import os
from dataset_parser import load_points, load_queries


def test_mnist_loader():
    """
    Test MNIST idx3-ubyte loader with synthetic data.
    
    Validates:
    - Magic number parsing (2051, big-endian)
    - Dimension extraction (n, 28, 28)
    - Flattening to [n, 784]
    - uint8 → float32 conversion
    - Normalization to [0, 1]
    """
    print("Testing MNIST loader...")
    
    # Create synthetic MNIST data
    n_images = 10
    n_rows = 28
    n_cols = 28
    
    # Generate random image data (uint8)
    image_data = np.random.randint(0, 256, size=(n_images, n_rows, n_cols), dtype=np.uint8)
    
    # Create temporary file with idx3-ubyte format
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.idx3-ubyte') as f:
        temp_path = f.name
        
        # Write magic number (2051, big-endian)
        f.write(struct.pack('>I', 2051))
        
        # Write dimensions (big-endian)
        f.write(struct.pack('>I', n_images))
        f.write(struct.pack('>I', n_rows))
        f.write(struct.pack('>I', n_cols))
        
        # Write image data
        f.write(image_data.tobytes())
    
    try:
        # Load using our loader
        loaded = load_points(temp_path, "mnist")
        
        # Validate shape
        assert loaded.shape == (n_images, n_rows * n_cols), \
            f"Expected shape ({n_images}, {n_rows * n_cols}), got {loaded.shape}"
        
        # Validate data type
        assert loaded.dtype == np.float32, \
            f"Expected dtype float32, got {loaded.dtype}"
        
        # Validate normalization range [0, 1]
        assert loaded.min() >= 0.0 and loaded.max() <= 1.0, \
            f"Expected values in [0, 1], got range [{loaded.min()}, {loaded.max()}]"
        
        # Validate actual values (should match normalized original)
        expected = image_data.reshape(n_images, n_rows * n_cols).astype(np.float32) / 255.0
        assert np.allclose(loaded, expected), \
            "Loaded data does not match expected normalized values"
        
        print("✓ MNIST loader: Shape, dtype, normalization correct")
        
    finally:
        # Cleanup
        os.unlink(temp_path)


def test_sift_loader():
    """
    Test SIFT fvecs loader with synthetic data.
    
    Validates:
    - Dimension prefix parsing (little-endian)
    - Vector extraction (float32)
    - Shape [n, d]
    - Multiple vector handling
    - Dimension consistency
    """
    print("Testing SIFT loader...")
    
    # Create synthetic SIFT data
    n_vectors = 5
    dim = 128
    
    # Generate random vectors (float32)
    vectors = np.random.randn(n_vectors, dim).astype(np.float32)
    
    # Create temporary file with fvecs format
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.fvecs') as f:
        temp_path = f.name
        
        for i in range(n_vectors):
            # Write dimension (little-endian int32)
            f.write(struct.pack('<i', dim))
            
            # Write vector components (little-endian float32)
            f.write(vectors[i].tobytes())
    
    try:
        # Load using our loader
        loaded = load_points(temp_path, "sift")
        
        # Validate shape
        assert loaded.shape == (n_vectors, dim), \
            f"Expected shape ({n_vectors}, {dim}), got {loaded.shape}"
        
        # Validate data type
        assert loaded.dtype == np.float32, \
            f"Expected dtype float32, got {loaded.dtype}"
        
        # Validate actual values
        assert np.allclose(loaded, vectors), \
            "Loaded data does not match expected vector values"
        
        print("✓ SIFT loader: Shape, dtype, values correct")
        
    finally:
        # Cleanup
        os.unlink(temp_path)


def test_mnist_invalid_magic():
    """
    Test MNIST loader error handling for invalid magic number.
    """
    print("Testing MNIST invalid magic number...")
    
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.idx3-ubyte') as f:
        temp_path = f.name
        
        # Write invalid magic number
        f.write(struct.pack('>I', 9999))
        f.write(struct.pack('>I', 10))
        f.write(struct.pack('>I', 28))
        f.write(struct.pack('>I', 28))
    
    try:
        try:
            load_points(temp_path, "mnist")
            assert False, "Should have raised ValueError for invalid magic number"
        except ValueError as e:
            assert "magic number" in str(e).lower(), \
                f"Expected magic number error, got: {e}"
            print("✓ MNIST loader: Invalid magic number properly detected")
    finally:
        os.unlink(temp_path)


def test_sift_inconsistent_dimensions():
    """
    Test SIFT loader error handling for inconsistent dimensions.
    """
    print("Testing SIFT inconsistent dimensions...")
    
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.fvecs') as f:
        temp_path = f.name
        
        # Write first vector with dim=128
        f.write(struct.pack('<i', 128))
        f.write(np.random.randn(128).astype(np.float32).tobytes())
        
        # Write second vector with different dim=64
        f.write(struct.pack('<i', 64))
        f.write(np.random.randn(64).astype(np.float32).tobytes())
    
    try:
        try:
            load_points(temp_path, "sift")
            assert False, "Should have raised ValueError for inconsistent dimensions"
        except ValueError as e:
            assert "inconsistent" in str(e).lower(), \
                f"Expected inconsistent dimension error, got: {e}"
            print("✓ SIFT loader: Inconsistent dimensions properly detected")
    finally:
        os.unlink(temp_path)


def test_file_not_found():
    """
    Test error handling for non-existent files.
    """
    print("Testing file not found error...")
    
    try:
        load_points("/nonexistent/path.idx3-ubyte", "mnist")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        print("✓ File not found: Properly detected for MNIST")
    
    try:
        load_points("/nonexistent/path.fvecs", "sift")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        print("✓ File not found: Properly detected for SIFT")


def test_invalid_dtype():
    """
    Test error handling for invalid dtype parameter.
    """
    print("Testing invalid dtype...")
    
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        temp_path = f.name
        f.write(b"dummy data")
    
    try:
        try:
            load_points(temp_path, "invalid")
            assert False, "Should have raised ValueError for invalid dtype"
        except ValueError as e:
            assert "unsupported dtype" in str(e).lower(), \
                f"Expected dtype error, got: {e}"
            print("✓ Invalid dtype: Properly detected")
    finally:
        os.unlink(temp_path)


def run_all_tests():
    """
    Run all validation tests.
    """
    print("=" * 60)
    print("Dataset Parser Validation Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_mnist_loader,
        test_sift_loader,
        test_mnist_invalid_magic,
        test_sift_inconsistent_dimensions,
        test_file_not_found,
        test_invalid_dtype,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
        print()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
