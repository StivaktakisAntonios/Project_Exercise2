"""
Neural LSH Project Configuration

This module defines global configuration constants for the Neural LSH project.
All modules should import settings from this file to ensure consistent behavior.

CRITICAL: This is a CPU-only project. No CUDA operations are permitted.
"""

# ============================================================================
# Device Configuration
# ============================================================================

# Force CPU-only execution throughout the project
# This device constant is used for all PyTorch operations
# The project is designed for CPU-only execution without GPU support
DEVICE = "cpu"

# ============================================================================
# Reproducibility Configuration
# ============================================================================

# Default random seed for reproducible experiments
# Use this seed for: numpy.random.seed(), torch.manual_seed(), random.seed()
RANDOM_SEED = 1

# ============================================================================
# Usage Instructions
# ============================================================================
#
# Import in other modules as:
#   from Exercise.Modules.config import DEVICE, RANDOM_SEED
#
# Example usage:
#   import torch
#   from Exercise.Modules.config import DEVICE, RANDOM_SEED
#   
#   torch.manual_seed(RANDOM_SEED)
#   model = MyModel().to(DEVICE)
#   tensor = torch.tensor(data, device=DEVICE)
#
# ============================================================================
