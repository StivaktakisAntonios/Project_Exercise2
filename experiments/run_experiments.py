#!/usr/bin/env python3
"""
run_experiments.py - Neural LSH Experimental Validation

Runs end-to-end experiments for Neural LSH index building and querying.
Measures performance metrics (recall, query time, throughput).

Usage:
    python run_experiments.py --dataset mnist --config configs/mnist_config.json
    python run_experiments.py --dataset sift --config configs/sift_config.json
    python run_experiments.py --all  # Run all predefined experiments

Output:
    Results saved to experiments/results/{dataset}_{timestamp}.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime
import subprocess

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


def run_build(config, dataset_type, dataset_path, index_path):
    """
    Run Neural LSH index building.
    
    Args:
        config: Configuration dict with hyperparameters
        dataset_type: 'mnist' or 'sift'
        dataset_path: Path to input dataset
        index_path: Path to output index
    
    Returns:
        Execution time in seconds
    """
    cmd = [
        'python', 'Exercise/NeuralLSH/nlsh_build.py',
        '-d', dataset_path,
        '-i', index_path,
        '-type', dataset_type,
        '--knn', str(config.get('knn', 10)),
        '-m', str(config.get('num_partitions', 100)),
        '--imbalance', str(config.get('imbalance', 0.03)),
        '--kahip_mode', str(config.get('kahip_mode', 2)),
        '--layers', str(config.get('layers', 3)),
        '--nodes', str(config.get('nodes', 64)),
        '--epochs', str(config.get('epochs', 10)),
        '--batch_size', str(config.get('batch_size', 128)),
        '--lr', str(config.get('lr', 0.001)),
        '--seed', str(config.get('seed', 1))
    ]
    
    print(f"\n{'='*60}")
    print(f"Building index: {index_path}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    build_time = time.time() - start_time
    
    if result.returncode != 0:
        raise RuntimeError(f"Index building failed with code {result.returncode}")
    
    print(f"\n✓ Index built in {build_time:.2f}s")
    return build_time


def run_search(config, dataset_type, query_path, index_path, output_path):
    """
    Run Neural LSH query search.
    
    Args:
        config: Configuration dict with search parameters
        dataset_type: 'mnist' or 'sift'
        query_path: Path to query dataset
        index_path: Path to index
        output_path: Path to output file
    
    Returns:
        Dict with metrics (query_time, qps, etc.)
    """
    cmd = [
        'python', 'Exercise/NeuralLSH/nlsh_search.py',
        '-q', query_path,
        '-i', index_path,
        '-o', output_path,
        '-type', dataset_type,
        '-N', str(config.get('N', 10)),
        '-T', str(config.get('top_bins', 5)),
        '-R', str(config.get('rerank', 50))
    ]
    
    print(f"\n{'='*60}")
    print(f"Running search: {output_path}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    search_time = time.time() - start_time
    
    if result.returncode != 0:
        raise RuntimeError(f"Search failed with code {result.returncode}\n{result.stderr}")
    
    # Parse metrics from output
    metrics = {
        'total_time': search_time,
        'output': result.stdout
    }
    
    # Extract metrics from stdout if available
    for line in result.stdout.split('\n'):
        if 'queries' in line.lower():
            parts = line.split()
            for i, part in enumerate(parts):
                if part.isdigit():
                    metrics['num_queries'] = int(part)
        if 'qps' in line.lower() or 'throughput' in line.lower():
            parts = line.split()
            for part in parts:
                try:
                    metrics['qps'] = float(part)
                    break
                except ValueError:
                    continue
    
    print(f"\n✓ Search completed in {search_time:.2f}s")
    return metrics


def calculate_recall(output_file, groundtruth_file, N=10):
    """
    Calculate recall@N by comparing output with groundtruth.
    
    Args:
        output_file: Path to search output file
        groundtruth_file: Path to groundtruth file (.ivecs for SIFT)
        N: Number of neighbors to consider
    
    Returns:
        Average recall@N across all queries
    """
    # Read output file
    with open(output_file, 'r') as f:
        lines = f.readlines()
    
    # Parse search results
    results = []
    for line in lines:
        if line.strip() and not line.startswith('Query'):
            parts = line.strip().split()
            if parts:
                neighbor_ids = [int(x) for x in parts]
                results.append(neighbor_ids[:N])
    
    # For SIFT, read groundtruth from .ivecs
    if groundtruth_file.endswith('.ivecs'):
        groundtruth = read_ivecs(groundtruth_file)
    else:
        # For MNIST, we'd need to compute true k-NN
        print("Warning: Groundtruth computation for MNIST not implemented")
        return None
    
    # Calculate recall
    recalls = []
    for i, pred in enumerate(results):
        if i >= len(groundtruth):
            break
        true_nn = set(groundtruth[i][:N])
        pred_nn = set(pred[:N])
        recall = len(true_nn & pred_nn) / N
        recalls.append(recall)
    
    avg_recall = np.mean(recalls) if recalls else 0.0
    return avg_recall


def read_ivecs(filename):
    """Read .ivecs file (SIFT groundtruth format)."""
    with open(filename, 'rb') as f:
        data = []
        while True:
            # Read dimension
            d_bytes = f.read(4)
            if not d_bytes:
                break
            d = int(np.frombuffer(d_bytes, dtype=np.int32)[0])
            # Read vector
            vec = np.frombuffer(f.read(d * 4), dtype=np.int32)
            data.append(vec)
        return np.array(data)


def run_experiment(experiment_config, results_dir):
    """
    Run a single experiment configuration.
    
    Args:
        experiment_config: Dict with dataset paths and hyperparameters
        results_dir: Directory to save results
    
    Returns:
        Dict with all metrics
    """
    dataset_type = experiment_config['dataset_type']
    dataset_path = experiment_config['dataset_path']
    query_path = experiment_config['query_path']
    groundtruth_path = experiment_config.get('groundtruth_path')
    
    # Create unique index path
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    index_path = f"experiments/indices/{dataset_type}_{timestamp}"
    output_path = f"{results_dir}/{dataset_type}_{timestamp}_output.txt"
    
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Run build
    build_time = run_build(
        experiment_config['build_params'],
        dataset_type,
        dataset_path,
        index_path
    )
    
    # Run search
    search_metrics = run_search(
        experiment_config['search_params'],
        dataset_type,
        query_path,
        index_path,
        output_path
    )
    
    # Calculate recall if groundtruth available
    recall = None
    if groundtruth_path:
        try:
            recall = calculate_recall(
                output_path,
                groundtruth_path,
                N=experiment_config['search_params'].get('N', 10)
            )
            if recall is not None:
                print(f"\n✓ Recall@{experiment_config['search_params']['N']}: {recall:.4f}")
        except Exception as e:
            print(f"Warning: Could not calculate recall: {e}")
    
    # Compile results
    results = {
        'timestamp': timestamp,
        'dataset_type': dataset_type,
        'dataset_path': dataset_path,
        'query_path': query_path,
        'build_params': experiment_config['build_params'],
        'search_params': experiment_config['search_params'],
        'build_time': build_time,
        'search_metrics': search_metrics,
        'recall': recall,
        'index_path': index_path,
        'output_path': output_path
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Neural LSH Experimental Validation')
    parser.add_argument('--config', help='Path to experiment config JSON file')
    parser.add_argument('--dataset', choices=['mnist', 'sift'], help='Dataset to use')
    parser.add_argument('--all', action='store_true', help='Run all predefined experiments')
    parser.add_argument('--results_dir', default='experiments/results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    experiments = []
    
    if args.config:
        # Load custom config
        with open(args.config, 'r') as f:
            config = json.load(f)
            experiments = [config]
    
    elif args.dataset:
        # Use predefined config for specified dataset
        if args.dataset == 'mnist':
            experiments = [{
                'dataset_type': 'mnist',
                'dataset_path': 'Raw_Data/MNIST/input.idx3-ubyte',
                'query_path': 'Raw_Data/MNIST/query.idx3-ubyte',
                'groundtruth_path': None,
                'build_params': {
                    'knn': 10,
                    'num_partitions': 100,
                    'imbalance': 0.03,
                    'kahip_mode': 2,
                    'layers': 3,
                    'nodes': 64,
                    'epochs': 10,
                    'batch_size': 128,
                    'lr': 0.001,
                    'seed': 1
                },
                'search_params': {
                    'N': 10,
                    'top_bins': 5,
                    'rerank': 50
                }
            }]
        else:  # sift
            experiments = [{
                'dataset_type': 'sift',
                'dataset_path': 'Raw_Data/SIFT/sift_base.fvecs',
                'query_path': 'Raw_Data/SIFT/sift_query.fvecs',
                'groundtruth_path': 'Raw_Data/SIFT/sift_groundtruth.ivecs',
                'build_params': {
                    'knn': 10,
                    'num_partitions': 100,
                    'imbalance': 0.03,
                    'kahip_mode': 2,
                    'layers': 3,
                    'nodes': 128,
                    'epochs': 15,
                    'batch_size': 256,
                    'lr': 0.001,
                    'seed': 1
                },
                'search_params': {
                    'N': 10,
                    'top_bins': 5,
                    'rerank': 100
                }
            }]
    
    elif args.all:
        # Run both MNIST and SIFT
        experiments = [
            {
                'dataset_type': 'mnist',
                'dataset_path': 'Raw_Data/MNIST/input.idx3-ubyte',
                'query_path': 'Raw_Data/MNIST/query.idx3-ubyte',
                'groundtruth_path': None,
                'build_params': {
                    'knn': 10,
                    'num_partitions': 100,
                    'imbalance': 0.03,
                    'kahip_mode': 2,
                    'layers': 3,
                    'nodes': 64,
                    'epochs': 10,
                    'batch_size': 128,
                    'lr': 0.001,
                    'seed': 1
                },
                'search_params': {
                    'N': 10,
                    'top_bins': 5,
                    'rerank': 50
                }
            },
            {
                'dataset_type': 'sift',
                'dataset_path': 'Raw_Data/SIFT/sift_base.fvecs',
                'query_path': 'Raw_Data/SIFT/sift_query.fvecs',
                'groundtruth_path': 'Raw_Data/SIFT/sift_groundtruth.ivecs',
                'build_params': {
                    'knn': 10,
                    'num_partitions': 100,
                    'imbalance': 0.03,
                    'kahip_mode': 2,
                    'layers': 3,
                    'nodes': 128,
                    'epochs': 15,
                    'batch_size': 256,
                    'lr': 0.001,
                    'seed': 1
                },
                'search_params': {
                    'N': 10,
                    'top_bins': 5,
                    'rerank': 100
                }
            }
        ]
    else:
        parser.print_help()
        return
    
    # Run experiments
    all_results = []
    for i, exp_config in enumerate(experiments, 1):
        print(f"\n{'#'*60}")
        print(f"# Experiment {i}/{len(experiments)}: {exp_config['dataset_type'].upper()}")
        print(f"{'#'*60}")
        
        try:
            results = run_experiment(exp_config, str(results_dir))
            all_results.append(results)
            
            # Save individual result
            result_file = results_dir / f"{results['dataset_type']}_{results['timestamp']}.json"
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n✓ Results saved to {result_file}")
            
        except Exception as e:
            print(f"\n✗ Experiment failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Save summary
    if all_results:
        summary_file = results_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"All experiments completed!")
        print(f"Summary saved to {summary_file}")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
