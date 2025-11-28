# Neural LSH Project - Development Summary

## Project Completion Status

### ✅ Completed Phases

#### Phase 1-8: Core Implementation
- Configuration module with CPU-only enforcement
- Dataset parsers for MNIST and SIFT formats
- k-NN graph construction and utilities
- KaHIP-based graph partitioning
- MLP classifier training module
- Index persistence (save/load)
- Batch search with multi-probe strategy
- Comprehensive unit tests

#### Phase 9: CLI Tools
- `nlsh_build.py`: Full pipeline for index building
- `nlsh_search.py`: Query processing with configurable parameters
- Argument validation and error handling
- Progress reporting and logging

#### Phase 10: Experimental Validation
- Automated experiment runner (`run_experiments.py`)
- Predefined configurations for MNIST and SIFT
- Metric collection (build time, search time, recall)
- JSON result persistence

#### Phase 11: Documentation
- Comprehensive README with installation and usage
- Troubleshooting guide
- Performance tuning recommendations
- Project structure documentation

#### Phase 12: Git Hygiene
- Organized commit history
- Comprehensive .gitignore
- Clean separation of code and data

## Key Design Decisions

1. **CPU-Only Execution**: All PyTorch operations forced to CPU for portability
2. **Modular Architecture**: Clear separation between data loading, graph construction, partitioning, training, and search
3. **Deterministic Behavior**: Fixed random seeds throughout for reproducibility
4. **Batch Processing**: Memory-efficient handling of large datasets
5. **Flexible Configuration**: CLI arguments and JSON configs for easy experimentation

## Technical Highlights

- **KaHIP Integration**: Subprocess-based invocation with CSR format conversion
- **Multi-Probe Search**: Classifier-guided bin selection with configurable depth
- **Assignment 1 Compatibility**: Output format matches previous assignment for fair comparison
- **Error Handling**: Comprehensive validation and informative error messages

## Performance Characteristics

### MNIST (784-dim)
- Default: 100 partitions, 10-NN graph, 3-layer MLP
- Training: ~10 epochs, batch_size=128
- Search: Top-5 bins, rerank=50

### SIFT (128-dim)
- Default: 100 partitions, 10-NN graph, 3-layer MLP
- Training: ~15 epochs, batch_size=256
- Search: Top-5 bins, rerank=100

## Testing Coverage

- Unit tests for all core modules
- Integration tests via CLI scripts
- Validation against expected formats
- Error condition handling

## Known Limitations

1. **MNIST Groundtruth**: No precomputed groundtruth available
2. **Memory Usage**: Full dataset loaded into memory during graph construction
3. **KaHIP Dependency**: External executable required (not pure Python)
4. **CPU-Only**: No GPU acceleration implemented

## Future Improvements

- GPU support for training and search
- Incremental index updates
- Compressed index storage
- Parallel query processing
- Advanced reranking strategies

## Repository Structure

```
Project_Exercise2/
├── Exercise/
│   ├── Modules/           # Core algorithms
│   └── NeuralLSH/        # CLI tools
├── experiments/           # Validation scripts
│   ├── configs/          # Experiment configurations
│   └── run_experiments.py
├── Raw_Data/             # Datasets (gitignored)
├── README.md             # User documentation
└── requirements.txt      # Dependencies
```

## Commit Summary

1. Initial project structure and configuration
2. Dataset parsing modules (MNIST, SIFT)
3. k-NN graph construction and utilities
4. KaHIP partitioning integration
5. MLP classifier implementation
6. Index I/O and persistence
7. Search algorithms and batch processing
8. CLI tools (build and search)
9. Bug fixes and import corrections
10. Experimental validation framework
11. Comprehensive documentation
12. Git hygiene and finalization

## Dependencies

- Python 3.10+
- PyTorch 2.2.0 (CPU-only)
- NumPy 1.26.4
- tqdm 4.66.6
- KaHIP (external)

## Team

- Ανέστης Θεοδωρίδης (sdi1500212@di.uoa.gr)
- Αντώνιος-Ραφαήλ Στιβακτάκης (sdi2200258@di.uoa.gr)

**Project:** Κ23γ – 2η Προγραμματιστική Εργασία  
**Course:** Ανάπτυξη Λογισμικού για Αλγοριθμικά Προβλήματα  
**Semester:** Winter 2025-26

---

*Document generated: November 28, 2025*
