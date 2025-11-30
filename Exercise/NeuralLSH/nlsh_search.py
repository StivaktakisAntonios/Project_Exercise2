#!/usr/bin/env python3
"""
nlsh_search.py — Neural LSH Query Search

Performs approximate nearest neighbor search using a pre-built Neural LSH index.

Pipeline:
1. Load index (model + inverted file + metadata)
2. Load dataset and queries
3. For each query:
   - Compute softmax probabilities over partitions
   - Select top-T bins (multi-probe)
   - Collect candidate points from selected bins
   - Perform exact distance computation on candidates
   - Return top-N nearest neighbors
4. Compute metrics (Recall@N, Average AF, QPS, timing)
5. Write output in Assignment 1 format

Usage:
    python nlsh_search.py -d <dataset> -q <queries> -i <index> -o <output> -type <sift|mnist> [options]

Required Arguments:
    -d              Path to dataset file
    -q              Path to query file
    -i              Path to index directory
    -o              Path to output file
    -type           Dataset type: 'sift' or 'mnist'

Optional Arguments:
    -N              Number of nearest neighbors to return (default: 1)
    -R              Radius for R-near neighbors (default: 2000 for MNIST, 2800 for SIFT)
    -T              Number of top bins to probe (default: 5)
    -range          Compute R-near neighbors: true or false (default: true)
"""

import argparse
import sys
import os
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch

from Exercise.Modules.config import DEVICE
from Exercise.Modules.dataset_parser import load_points, load_queries
from Exercise.Modules.index_io import load_index
from Exercise.Modules.search import batch_search, write_output_file


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Neural LSH Query Search',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search MNIST with defaults
  python nlsh_search.py -d data/input.idx3-ubyte -q data/query.idx3-ubyte \\
      -i index/mnist -o output.txt -type mnist

  # Search SIFT with custom parameters
  python nlsh_search.py -d data/sift_base.fvecs -q data/sift_query.fvecs \\
      -i index/sift -o output.txt -type sift -N 10 -T 10 -R 2800
        """
    )
    
    # Required arguments
    parser.add_argument('-d', '--dataset', required=True,
                        help='Path to dataset file')
    parser.add_argument('-q', '--queries', required=True,
                        help='Path to query file')
    parser.add_argument('-i', '--index', required=True,
                        help='Path to index directory')
    parser.add_argument('-o', '--output', required=True,
                        help='Path to output file')
    parser.add_argument('-type', '--type', required=True, choices=['sift', 'mnist'],
                        help='Dataset type: sift or mnist')
    
    # Search parameters
    parser.add_argument('-N', '--neighbors', type=int, default=1,
                        help='Number of nearest neighbors (default: 1)')
    parser.add_argument('-R', '--radius', type=float, default=None,
                        help='Radius for R-near neighbors (default: 2000 for MNIST, 2800 for SIFT)')
    parser.add_argument('-T', '--top_bins', type=int, default=5,
                        help='Number of top bins to probe (default: 5)')
    parser.add_argument('-range', '--range_search', type=str, default='true',
                        choices=['true', 'false'],
                        help='Compute R-near neighbors (default: true)')
    parser.add_argument('--max_queries', type=int, default=None,
                        help='Limit number of queries to process (default: all)')
    
    return parser.parse_args()


def main():
    """Main search pipeline."""
    args = parse_args()
    
    # Set default radius based on dataset type if not specified
    if args.radius is None:
        args.radius = 2000.0 if args.type == 'mnist' else 2800.0
    
    # Parse range search flag
    range_search = (args.range_search.lower() == 'true')
    
    print("=" * 80)
    print("Neural LSH Query Search")
    print("=" * 80)
    print(f"Dataset:      {args.dataset}")
    print(f"Queries:      {args.queries}")
    print(f"Index:        {args.index}")
    print(f"Output:       {args.output}")
    print(f"Type:         {args.type}")
    print(f"Device:       {DEVICE}")
    print(f"N neighbors:  {args.neighbors}")
    print(f"R radius:     {args.radius}")
    print(f"T bins:       {args.top_bins}")
    print(f"Range search: {range_search}")
    if args.max_queries:
        print(f"Max queries:  {args.max_queries}")
    print("=" * 80)
    
    # Phase 1: Load index
    print("\n[1/4] Loading index...")
    t_start = time.time()
    try:
        bins, model, metadata = load_index(args.index)
        model.to(DEVICE)
        model.eval()
        print(f"  Loaded index with {metadata['num_partitions']} partitions")
        print(f"  Dataset: {metadata['num_points']} points, dimension {metadata['input_dim']}")
        print(f"  Time: {time.time() - t_start:.2f}s")
    except Exception as e:
        print(f"ERROR: Failed to load index: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Phase 2: Load dataset and queries
    print("\n[2/4] Loading dataset and queries...")
    t_start = time.time()
    try:
        dataset = load_points(args.dataset, args.type)
        queries = load_queries(args.queries, args.type)
        # Optionally limit number of queries for faster iteration
        if args.max_queries is not None and args.max_queries > 0:
            queries = queries[:args.max_queries]
        print(f"  Dataset: {dataset.shape[0]} points, dimension {dataset.shape[1]}")
        print(f"  Queries: {queries.shape[0]} queries")
        print(f"  Time: {time.time() - t_start:.2f}s")
    except Exception as e:
        print(f"ERROR: Failed to load data: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Validate dimensions
    if dataset.shape[1] != metadata['input_dim']:
        print(f"ERROR: Dataset dimension mismatch. Expected {metadata['input_dim']}, got {dataset.shape[1]}", 
              file=sys.stderr)
        sys.exit(1)
    if queries.shape[1] != metadata['input_dim']:
        print(f"ERROR: Query dimension mismatch. Expected {metadata['input_dim']}, got {queries.shape[1]}", 
              file=sys.stderr)
        sys.exit(1)
    
    # Phase 3: Perform search
    print(f"\n[3/4] Searching {queries.shape[0]} queries...")
    t_start = time.time()
    try:
        results, aggregate_metrics = batch_search(
            queries=queries,
            model=model,
            bins=bins,
            dataset=dataset,
            N=args.neighbors,
            T=args.top_bins,
            R=args.radius,
            range_search=range_search
        )
        print(f"  Search complete")
        print(f"  Total time: {time.time() - t_start:.2f}s")
        print(f"  QPS (approx only): {aggregate_metrics['qps']:.2f}")
    except Exception as e:
        print(f"ERROR: Search failed: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Phase 4: Write output
    print("\n[4/4] Writing output...")
    t_start = time.time()
    try:
        write_output_file(
            results=results,
            aggregate_metrics=aggregate_metrics,
            output_path=args.output,
            N=args.neighbors,
            range_search=range_search #extra gia R-neigbours tipoma
        )
        print(f"  Output written to: {args.output}")
        print(f"  Time: {time.time() - t_start:.2f}s")
    except Exception as e:
        print(f"ERROR: Failed to write output: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Print summary metrics
    print("\n" + "=" * 80)
    print("Search Results Summary")
    print("=" * 80)
    print(f"Average AF:       {aggregate_metrics['average_af']:.4f}")
    print(f"Recall@{args.neighbors}:        {aggregate_metrics['recall_at_n']:.4f}")
    print(f"QPS:              {aggregate_metrics['qps']:.2f}")
    print(f"tApproximate:     {aggregate_metrics['t_approximate_average']:.6f}s")
    print(f"tTrue:            {aggregate_metrics['t_true_average']:.6f}s")
    print("=" * 80)


if __name__ == '__main__':
    main()
