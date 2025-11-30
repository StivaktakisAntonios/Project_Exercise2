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


def run_search(config, dataset_type, dataset_path, query_path, index_path, output_path):
    """
    Run Neural LSH query search.
    
    Args:
        config: Configuration dict with search parameters
        dataset_type: 'mnist' or 'sift'
        dataset_path: Path to dataset file
        query_path: Path to query dataset
        index_path: Path to index
        output_path: Path to output file
    
    Returns:
        Dict with metrics (recall, af, qps, etc.)
    """
    # Set default radius based on dataset type
    default_radius = 2000.0 if dataset_type == 'mnist' else 2800.0
    
    cmd = [
        'python', 'Exercise/NeuralLSH/nlsh_search.py',
        '-d', dataset_path,
        '-q', query_path,
        '-i', index_path,
        '-o', output_path,
        '-type', dataset_type,
        '-N', str(config.get('N', 10)),
        '-T', str(config.get('top_bins', 5)),
        '-R', str(config.get('radius', default_radius)),
        '-range', 'false'
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
    
    # Parse metrics from output file (aggregate metrics at end)
    metrics = {
        'total_time': search_time,
        'recall': None,
        'average_af': None,
        'qps': None,
        't_approximate': None,
        't_true': None
    }
    
    try:
        with open(output_path, 'r') as f:
            lines = f.readlines()
        
        # Parse aggregate metrics from end of file
        for line in lines:
            line = line.strip()
            if line.startswith('Average AF:'):
                metrics['average_af'] = float(line.split(':')[1].strip())
            elif line.startswith('Recall@N:'):
                metrics['recall'] = float(line.split(':')[1].strip())
            elif line.startswith('QPS:'):
                metrics['qps'] = float(line.split(':')[1].strip())
            elif line.startswith('tApproximateAverage:'):
                metrics['t_approximate'] = float(line.split(':')[1].strip())
            elif line.startswith('tTrueAverage:'):
                metrics['t_true'] = float(line.split(':')[1].strip())
    except Exception as e:
        print(f"Warning: Could not parse metrics from output file: {e}")
    
    print(f"\n✓ Search completed in {search_time:.2f}s")
    if metrics['recall'] is not None:
        print(f"  Recall@N: {metrics['recall']:.4f}")
        print(f"  Average AF: {metrics['average_af']:.4f}")
        print(f"  QPS: {metrics['qps']:.2f}")
    return metrics


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
        dataset_path,
        query_path,
        index_path,
        output_path
    )
    
    # Compile results
    results = {
        'timestamp': timestamp,
        'dataset_type': dataset_type,
        'dataset_path': dataset_path,
        'query_path': query_path,
        'build_params': experiment_config['build_params'],
        'search_params': experiment_config['search_params'],
        'build_time': build_time,
        'recall': search_metrics.get('recall'),
        'average_af': search_metrics.get('average_af'),
        'qps': search_metrics.get('qps'),
        't_approximate': search_metrics.get('t_approximate'),
        't_true': search_metrics.get('t_true'),
        'total_search_time': search_metrics.get('total_time'),
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
