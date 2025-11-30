#!/usr/bin/env python3
"""
nlsh_build.py — Neural LSH Index Builder

Builds Neural LSH index through the following pipeline:
1. Load dataset
2. Build k-NN graph using IVFFlat (C++ from Assignment 1)
3. Partition k-NN graph using KaHIP (via partition_knn_graph)
4. Train MLP classifier on partition labels
5. Save index (model + inverted file + metadata)

Usage:
    python nlsh_build.py -d <input_file> -i <index_path> -type <sift|mnist> [options]

Required Arguments:
    -d              Path to input dataset file
    -i              Path to output index directory
    -type           Dataset type: 'sift' or 'mnist'

Optional Arguments:
    --knn           Number of nearest neighbors (default: 10)
    -m              Number of partitions/bins (default: 100)
    --imbalance     KaHIP imbalance tolerance (default: 0.03)
    --kahip_mode    KaHIP mode: 0=FAST, 1=ECO, 2=STRONG (default: 2)
    --layers        MLP hidden layers (default: 3)
    --nodes         MLP hidden layer width (default: 64)
    --epochs        Training epochs (default: 10)
    --batch_size    Training batch size (default: 128)
    --lr            Learning rate (default: 0.001)
    --seed          Random seed (default: 1)
"""

import argparse
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import os
import subprocess

from Exercise.Modules.config import DEVICE
from Exercise.Modules.dataset_parser import load_points
from Exercise.Modules.graph_utils import build_knn, load_knn_indices_bin
from Exercise.Modules.partitioner import partition_knn_graph
from Exercise.Modules.models import train_partition_classifier
from Exercise.Modules.index_io import build_inverted_index, save_index


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Neural LSH Index Builder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build MNIST index with defaults
  python nlsh_build.py -d data/input.idx3-ubyte -i index/mnist -type mnist

  # Build SIFT index with custom parameters
  python nlsh_build.py -d data/sift_base.fvecs -i index/sift -type sift \\
      --knn 20 -m 200 --epochs 20
        """
    )
    
    # Required arguments
    parser.add_argument('-d', '--dataset', required=True,
                        help='Path to input dataset file')
    parser.add_argument('-i', '--index', required=True,
                        help='Path to output index directory')
    parser.add_argument('-type', '--type', required=True, choices=['sift', 'mnist'],
                        help='Dataset type: sift or mnist')
    
    # k-NN graph parameters
    parser.add_argument('--knn', type=int, default=10,
                        help='Number of nearest neighbors (default: 10)')
    parser.add_argument('--max_points', type=int, default=None,
                        help='Maximum points for k-NN graph (default: all). Use for large datasets like SIFT.')
    
    # Partitioning parameters
    parser.add_argument('-m', '--partitions', type=int, default=100,
                        help='Number of partitions/bins (default: 100)')
    parser.add_argument('--imbalance', type=float, default=0.03,
                        help='KaHIP imbalance tolerance (default: 0.03)')
    parser.add_argument('--kahip_mode', type=int, default=2, choices=[0, 1, 2],
                        help='KaHIP mode: 0=FAST, 1=ECO, 2=STRONG (default: 2)')
    
    # MLP architecture parameters
    parser.add_argument('--layers', type=int, default=3,
                        help='Number of MLP hidden layers (default: 3)')
    parser.add_argument('--nodes', type=int, default=64,
                        help='MLP hidden layer width (default: 64)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10,
                        help='Training epochs (default: 10)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Training batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed for reproducibility (default: 1)')
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, 'cuda') and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def main():
    """Main build pipeline."""
    args = parse_args()
    
    print("=" * 80)
    print("Neural LSH Index Builder")
    print("=" * 80)
    print(f"Dataset:      {args.dataset}")
    print(f"Index path:   {args.index}")
    print(f"Type:         {args.type}")
    print(f"Device:       {DEVICE}")
    print(f"k-NN:         {args.knn}")
    print(f"Max points:   {args.max_points if args.max_points else 'all'}")
    print(f"Partitions:   {args.partitions}")
    print(f"Imbalance:    {args.imbalance}")
    print(f"KaHIP mode:   {args.kahip_mode} ({'FAST' if args.kahip_mode == 0 else 'ECO' if args.kahip_mode == 1 else 'STRONG'})")
    print(f"MLP layers:   {args.layers}")
    print(f"MLP nodes:    {args.nodes}")
    print(f"Epochs:       {args.epochs}")
    print(f"Batch size:   {args.batch_size}")
    print(f"Learn rate:   {args.lr}")
    print(f"Seed:         {args.seed}")
    print("=" * 80)
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Phase 1: Load dataset
    print("\n[1/5] Loading dataset...")
    t_start = time.time()
    try:
        dataset = load_points(args.dataset, args.type)
        print(f"  Loaded {dataset.shape[0]} points, dimension {dataset.shape[1]}")
        print(f"  Time: {time.time() - t_start:.2f}s")
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Phase 2: Build k-NN graph
    print("\n[2/5] Building k-NN graph...")
    t_start = time.time()

    try:
        # Χρησιμοποιούμε C++ IVFFlat-based k-NN για όλα τα datasets (SIFT και MNIST)
        script_dir = os.path.dirname(os.path.abspath(__file__))  # .../Exercise/NeuralLSH
        
        if args.type == "sift":
            # Για SIFT χρησιμοποιούμε build_knn_sift
            knn_bin_path = os.path.join("Raw_Data", "SIFT", f"knn_sift_k{args.knn}.bin")
            exe_name = "build_knn_sift"
        else:  # mnist
            # Για MNIST χρησιμοποιούμε build_knn_mnist
            knn_bin_path = os.path.join("Raw_Data", "MNIST", f"knn_mnist_k{args.knn}.bin")
            exe_name = "build_knn_mnist"

        # Αν υπάρχει ήδη το k-NN file, δεν το ξαναυπολογίζουμε
        if not os.path.exists(knn_bin_path):
            print(f"  k-NN file {knn_bin_path} not found.")
            print(f"  Running C++ executable '{exe_name}' to compute k-NN graph using IVFFlat...")

            # Βρίσκουμε το path του executable
            exe_path = os.path.join(script_dir, "..", exe_name)
            exe_path = os.path.normpath(exe_path)

            if not os.path.exists(exe_path):
                raise FileNotFoundError(
                    f"C++ executable not found: {exe_path}\n"
                    f"Please compile it first:\n"
                    f"  cd Exercise\n"
                    f"  g++ -O3 -std=c++17 Modules/Models/{exe_name}.cpp "
                    f"Modules/Models/Template/data_io.cpp Modules/Models/IVFFlat/ivfflat.cpp "
                    f"-o {exe_name}"
                )

            # Δημιουργούμε τον κατάλογο για το output αν δεν υπάρχει
            os.makedirs(os.path.dirname(knn_bin_path), exist_ok=True)

            cmd = [
                exe_path,
                "-d", args.dataset,
                "-k", str(args.knn),
                "-o", knn_bin_path,
            ]
            subprocess.run(cmd, check=True)

        # Διαβάζουμε το .bin που έφτιαξε ο C++ κώδικας
        knn_indices, knn_distances = load_knn_indices_bin(knn_bin_path)
        actual_points = knn_indices.shape[0]
        print(
            f"  Loaded k-NN graph from {knn_bin_path} "
            f"for {actual_points} points, k={args.knn}"
        )

        print(f"  Time: {time.time() - t_start:.2f}s")

    except Exception as e:
        print(f"ERROR: Failed to build k-NN graph: {e}", file=sys.stderr)
        sys.exit(1)

    
    # Phase 3: Partition k-NN graph with KaHIP
    print("\n[3/5] Partitioning k-NN graph with KaHIP...")
    t_start = time.time()
    try:
        labels = partition_knn_graph(
            indices=knn_indices,
            distances=knn_distances,
            n_parts=args.partitions,
            imbalance=args.imbalance,
            mode=args.kahip_mode,
            seed=args.seed
        )
        
        unique_labels = np.unique(labels)
        bin_sizes = [np.sum(labels == lab) for lab in unique_labels]
        print(f"  Partitioned into {len(unique_labels)} bins")
        print(f"  Bin size range: [{min(bin_sizes)}, {max(bin_sizes)}]")
        print(f"  Bin size mean: {np.mean(bin_sizes):.1f} ± {np.std(bin_sizes):.1f}")
        print(f"  Time: {time.time() - t_start:.2f}s")
    except Exception as e:
        print(f"ERROR: Failed to partition graph: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Phase 4: Train MLP classifier
    print("\n[4/5] Training MLP classifier...")
    t_start = time.time()
    try:
        # Use subset of data if max_points was specified
        training_data = dataset[:actual_points]

        #training_data = dataset[:actual_points] if args.max_points else dataset
        
        model, history = train_partition_classifier(
            points=training_data,
            labels=labels,
            input_dim=dataset.shape[1],
            num_partitions=args.partitions,
            hidden_dim=args.nodes,
            num_layers=args.layers,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            val_split=0.1,
            seed=args.seed,
            verbose=True
        )
        print(f"  Training complete")
        print(f"  Time: {time.time() - t_start:.2f}s")
    except Exception as e:
        print(f"ERROR: Failed to train classifier: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Phase 5: Save index
    print("\n[5/5] Saving index...")
    t_start = time.time()
    try:
        # Build inverted index from labels
        bins = build_inverted_index(labels)
        
        metadata = {
            'dataset_type': args.type,
            'dataset_path': args.dataset,
            'num_points': int(dataset.shape[0]),
            'num_points_used': int(actual_points),
            'input_dim': int(dataset.shape[1]),
            'dimension': int(dataset.shape[1]),
            'knn': args.knn,
            'max_points': args.max_points,
            'num_partitions': args.partitions,
            'partition_params': {
                'imbalance': args.imbalance,
                'mode': args.kahip_mode,
            },
            'mlp_layers': args.layers,
            'mlp_nodes': args.nodes,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'seed': args.seed,
            'device': DEVICE
        }
        
        save_index(args.index, bins, model, metadata)
        print(f"  Index saved to: {args.index}")
        print(f"  Time: {time.time() - t_start:.2f}s")
    except Exception as e:
        print(f"ERROR: Failed to save index: {e}", file=sys.stderr)
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("Index building complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
