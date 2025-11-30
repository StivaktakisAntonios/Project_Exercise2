# Neural LSH - Approximate Nearest Neighbor Search

**Project:** Κ23γ – 2η Προγραμματιστική Εργασία  
**Course:** Ανάπτυξη Λογισμικού για Αλγοριθμικά Προβλήματα  
**Semester:** Χειμερινό εξάμηνο 2025-26

**Team 15:**  
- **Ανέστης Θεοδωρίδης** – ΑΜ: 1115201500212 – Email: sdi1500212@di.uoa.gr
- **Αντώνιος-Ραφαήλ Στιβακτάκης** – ΑΜ: 1115202200258 – Email: sdi2200258@di.uoa.gr

---

## Overview

Implementation of Neural LSH (Locality-Sensitive Hashing) for approximate nearest neighbor search using neural network-based partitioning.

The algorithm combines:
- k-NN graph construction using IVFFlat (C++ from Assignment 1)
- Balanced graph partitioning using KaHIP
- Multi-layer perceptron (MLP) classifier for learning partition assignments
- Multi-probe search strategy for efficient approximate nearest neighbor retrieval

This implementation follows the specification of Assignment 2 (Κ23γ) and produces output compatible with Assignment 1 methods (LSH, Hypercube, IVFFlat, IVFPQ) for direct performance comparison.

**k-NN Graph Construction:** Uses optimized C++ IVFFlat implementation from Assignment 1 for fast approximate k-NN graph construction on both MNIST and SIFT datasets. The k-NN graphs are precomputed and cached for reuse across multiple experiments.

## Project Structure

```
Project_Exercise2/
├── Exercise/
│   ├── Modules/              # Core implementation modules
│   │   ├── config.py         # Configuration and constants
│   │   ├── dataset_parser.py # Dataset loading (MNIST, SIFT)
│   │   ├── graph_utils.py    # k-NN graph construction
│   │   ├── partitioner.py    # KaHIP graph partitioning
│   │   ├── models.py         # MLP classifier training
│   │   ├── index_io.py       # Index persistence
│   │   ├── search.py         # Query search algorithms
│   │   └── Models/           # C++ implementations from Assignment 1
│   │       ├── build_knn_mnist.cpp  # MNIST k-NN builder
│   │       ├── build_knn_sift.cpp   # SIFT k-NN builder
│   │       ├── IVFFlat/      # IVF-Flat implementation
│   │       ├── LSH/          # LSH implementation
│   │       ├── Hypercube/    # Hypercube implementation
│   │       ├── IVFPQ/        # IVF-PQ implementation
│   │       └── Template/     # Utilities (L2, data I/O)
│   ├── NeuralLSH/            # CLI scripts
│   │   ├── nlsh_build.py     # Index building script
│   │   └── nlsh_search.py    # Query search script
│   ├── build_knn_mnist       # Compiled MNIST k-NN executable
│   └── build_knn_sift        # Compiled SIFT k-NN executable
├── experiments/              # Experimental validation
│   ├── run_experiments.py    # Automated experiment runner
│   └── results/              # Experiment results
├── Raw_Data/                 # Input datasets
│   ├── MNIST/
│   └── SIFT/
├── build_knn_executables.sh  # Compile k-NN executables
└── requirements.txt          # Python dependencies
```

## Installation

### Prerequisites

- Python 3.10 or higher
- KaHIP graph partitioning tool
- C++ compiler with C++17 support (g++ recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/StivaktakisAntonios/Project_Exercise2.git
cd Project_Exercise2
```

2. Create and activate virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Linux/Mac
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Install KaHIP:
```bash
# On Ubuntu/Debian
sudo apt-get install kahip

# Or build from source
git clone https://github.com/KaHIP/KaHIP.git
cd KaHIP
./compile_withcmake.sh
export PATH=$PATH:$(pwd)/deploy
```

5. Compile C++ k-NN graph builders:
```bash
./build_knn_executables.sh
```

This will compile:
- `Exercise/build_knn_mnist` - k-NN graph builder for MNIST using IVFFlat
- `Exercise/build_knn_sift` - k-NN graph builder for SIFT using IVFFlat

6. Verify installation:
```bash
which kaffpa
python -c "import torch; print(torch.__version__)"
./Exercise/build_knn_mnist -h
./Exercise/build_knn_sift -h
```

## Usage

### Building an Index

Build a Neural LSH index from a dataset:

```bash
python Exercise/NeuralLSH/nlsh_build.py \
    -d Raw_Data/MNIST/input.idx3-ubyte \
    -i indices/mnist_index \
    -type mnist \
    --knn 10 \
    -m 100 \
    --epochs 10
```

**Required Arguments:**
- `-d, --dataset`: Path to input dataset file
- `-i, --index`: Path to output index directory
- `-type`: Dataset type (`mnist` or `sift`)

**Optional Arguments:**
- `--knn`: Number of nearest neighbors for graph construction (default: 10)
- `-m, --partitions`: Number of partitions/bins (default: 100)
- `--imbalance`: KaHIP imbalance tolerance (default: 0.03)
- `--kahip_mode`: KaHIP quality mode: 0=FAST, 1=ECO, 2=STRONG (default: 2)
- `--layers`: Number of MLP hidden layers (default: 3)
- `--nodes`: MLP hidden layer width (default: 64)
- `--epochs`: Training epochs (default: 10)
- `--batch_size`: Training batch size (default: 128)
- `--lr`: Learning rate (default: 0.001)
- `--seed`: Random seed for reproducibility (default: 1)

### Querying an Index

Search for nearest neighbors using a built index:

```bash
python Exercise/NeuralLSH/nlsh_search.py \
    -q Raw_Data/MNIST/query.idx3-ubyte \
    -i indices/mnist_index \
    -o results/mnist_output.txt \
    -type mnist \
    -N 10 \
    -T 5 \
    -R 50
```

**Required Arguments:**
- `-q, --query`: Path to query dataset file
- `-i, --index`: Path to index directory
- `-o, --output`: Path to output results file
- `-type`: Dataset type (`mnist` or `sift`)

**Optional Arguments:**
- `-N`: Number of nearest neighbors to return (default: 10)
- `-T, --top_bins`: Number of top bins to probe (default: 5)
- `-R, --rerank`: Number of candidates to rerank (default: 50)

### Running Experiments

Run predefined experiments:

```bash
# Run MNIST experiment
python experiments/run_experiments.py --dataset mnist

# Run SIFT experiment
python experiments/run_experiments.py --dataset sift

# Run all experiments
python experiments/run_experiments.py --all
```

## Algorithm Pipeline

### Index Building

1. **Load Dataset**: Parse input data (MNIST or SIFT format)
2. **Build k-NN Graph**: Construct approximate k-NN graph using brute-force search
3. **Partition Graph**: Use KaHIP to create balanced partitions of the k-NN graph
4. **Train Classifier**: Train MLP to predict partition assignments from data points
5. **Save Index**: Persist inverted index, trained model, and metadata

### Query Search

1. **Load Index**: Load inverted index and trained classifier
2. **Predict Partitions**: Use MLP to predict top-T most likely bins for each query
3. **Multi-Probe Search**: Search candidates from selected bins
4. **Rerank**: Compute exact distances and return top-N nearest neighbors

## Configuration

Key parameters in `Exercise/Modules/config.py`:

```python
DEVICE = "cpu"          # Force CPU-only execution
RANDOM_SEED = 1         # Default random seed
EPSILON = 1e-10         # Numerical stability constant
```

## Dataset Formats

### MNIST
- Format: IDX3-ubyte (binary format with 4-byte header)
- Dimensions: 784 (28×28 pixels)
- Input: `Raw_Data/MNIST/input.idx3-ubyte`
- Query: `Raw_Data/MNIST/query.idx3-ubyte`

### SIFT
- Format: fvecs (binary format with 4-byte dimension prefix)
- Dimensions: 128
- Base: `Raw_Data/SIFT/sift_base.fvecs`
- Query: `Raw_Data/SIFT/sift_query.fvecs`
- Ground truth: `Raw_Data/SIFT/sift_groundtruth.ivecs`

## Output Format

Search results are written in the format:

```
Query_0: neighbor_1 neighbor_2 ... neighbor_N
Query_1: neighbor_1 neighbor_2 ... neighbor_N
...
```

Each line contains space-separated neighbor IDs (0-indexed) for a single query.

## Performance Tuning

### Index Building
- **More partitions** (`-m`): Better selectivity but more storage
- **Higher k** (`--knn`): Better graph quality but slower construction
- **More epochs**: Better classifier but longer training time
- **Stronger KaHIP mode**: Better partitions but slower partitioning

### Query Search
- **More bins** (`-T`): Higher recall but slower search
- **More rerank candidates** (`-R`): Better precision but more computation

## Development Notes

- **CPU-only**: Implementation uses CPU-only PyTorch (no GPU required)
- **Deterministic**: Fixed random seeds ensure reproducible results
- **Memory efficient**: Batch processing for large datasets
- **Modular design**: Clear separation between modules for maintainability
- **Assignment 1 Compatibility**: Output format matches Assignment 1 for direct comparison with LSH, Hypercube, IVFFlat, and IVFPQ methods

## Experimental Comparison

For the experimental report comparing Neural LSH with Assignment 1 methods:

1. **Run Neural LSH experiments**:
   ```bash
   python experiments/run_experiments.py --all
   ```

2. **Compare metrics** (from both assignments):
   - Recall@N: Fraction of true neighbors found
   - Average AF: Approximation factor (distance ratio)
   - QPS: Queries per second (throughput)
   - tApproximate: Average query time
   - tTrue: Average ground truth computation time

3. **Hyperparameter tuning**: Use `experiments/configs/` to test different settings:
   - Number of partitions (`m`)
   - k-NN graph size (`k`)
   - MLP architecture (`layers`, `nodes`)
   - Training parameters (`epochs`, `batch_size`, `lr`)
   - Multi-probe depth (`T`)

Results will be saved in `experiments/results/` as JSON files for analysis.

## Troubleshooting

### KaHIP not found
```bash
which kaffpa
export PATH=$PATH:/path/to/KaHIP/deploy
```

### Out of memory
- Reduce batch size: `--batch_size 64`
- Reduce partitions: `-m 50`
- Use smaller k: `--knn 5`

### Low recall
- Increase probed bins: `-T 10`
- Increase rerank candidates: `-R 100`
- Use more partitions: `-m 200`
- Train longer: `--epochs 20`

## References

- KaHIP: https://github.com/KaHIP/KaHIP
- PyTorch: https://pytorch.org/