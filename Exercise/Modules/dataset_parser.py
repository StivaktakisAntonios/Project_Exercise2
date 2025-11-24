# dataset_parser.py
import numpy as np
from pathlib import Path
from typing import Tuple


def _load_mnist_images(path: str) -> np.ndarray:
    p = Path(path)
    with p.open("rb") as f:
        # Διαβάζουμε header: magic, num, rows, cols (big-endian 32-bit)
        magic = int.from_bytes(f.read(4), byteorder="big", signed=False)
        num = int.from_bytes(f.read(4), byteorder="big", signed=False)
        rows = int.from_bytes(f.read(4), byteorder="big", signed=False)
        cols = int.from_bytes(f.read(4), byteorder="big", signed=False)

        if magic != 2051:
            raise RuntimeError(f"MNIST magic != 2051 for {path} (got {magic})")

        n = num
        dim = rows * cols

        # Διαβάζουμε όλα τα δεδομένα ως uint8
        data = np.frombuffer(f.read(n * dim), dtype=np.uint8)
        if data.size != n * dim:
            raise RuntimeError(f"Short read in {path}: expected {n*dim} bytes, got {data.size}")

    # Μετατροπή σε float32 και reshape σε (n, dim)
    data = data.astype(np.float32).reshape(n, dim)

    # Για κανονικοποίηση 
    # data /= 255.0

    return data


def _load_sift_fvecs(path: str) -> np.ndarray:
    p = Path(path)
    with p.open("rb") as f:
        # μέγεθος αρχείου
        f.seek(0, 2)
        file_size = f.tell()
        f.seek(0)

        # Διαβάζουμε την πρώτη dim
        d_first = np.fromfile(f, dtype="<i4", count=1)  # little-endian int32
        if d_first.size != 1:
            raise RuntimeError(f"Short read when reading first dim from {path}")
        dim = int(d_first[0])
        if dim <= 0:
            raise RuntimeError(f"Invalid first dim in fvecs: {path}")

        stride = 4 + dim * 4  # 4 bytes dim + dim*4 bytes float32

        if file_size % stride != 0:
            raise RuntimeError(
                f"fvecs file size is not a multiple of record stride in {path}"
            )

        n = file_size // stride

        # Επιστρέφουμε στην αρχή του αρχείου για να διαβάσουμε όλα τα vectors
        f.seek(0)

        # Θα αποθηκεύσουμε όλα τα vectors σε έναν πίνακα (n, dim)
        data = np.empty((n, dim), dtype=np.float32)

        for i in range(n):
            # διαβάζουμε dim (και το ελέγχουμε)
            d_i = np.fromfile(f, dtype="<i4", count=1)
            if d_i.size != 1:
                raise RuntimeError(f"Short read (dim) in {path} at vector {i}")
            if int(d_i[0]) != dim:
                raise RuntimeError(f"Mixed dims in fvecs: {path}")

            # διαβάζουμε dim float32 τιμές
            vec = np.fromfile(f, dtype="<f4", count=dim)
            if vec.size != dim:
                raise RuntimeError(f"Short read (values) in {path} at vector {i}")

            data[i, :] = vec

    # L2-normalization
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    # αποφυγή διαίρεσης με μηδέν
    norms[norms == 0.0] = 1.0
    data = data / norms

    return data


def load_dataset(path: str, data_type: str) -> np.ndarray:
    if data_type == "mnist":
        return _load_mnist_images(path)
    elif data_type == "sift":
        return _load_sift_fvecs(path)
    else:
        raise ValueError(f"Unsupported data type: {data_type}")
