# KaHIP Installation Guide

KaHIP (Karlsruhe High Quality Partitioning) is required for graph partitioning in Neural LSH.

## Option 1: Install from Ubuntu Repository

```bash
sudo apt-get update
sudo apt-get install kahip
```

Verify installation:
```bash
which kaffpa
kaffpa --help
```

## Option 2: Build from Source

If the package is not available in your distribution:

```bash
# Install dependencies
sudo apt-get install build-essential cmake libopenmpi-dev

# Clone KaHIP repository
git clone https://github.com/KaHIP/KaHIP.git
cd KaHIP

# Build
./compile_withcmake.sh

# Add to PATH (add this to ~/.bashrc for persistence)
export PATH=$PATH:$(pwd)/deploy

# Verify
which kaffpa
```

## Option 3: Using pip (Python wrapper - experimental)

Some systems may have a Python wrapper:

```bash
pip install kahip
```

Note: This may not include the `kaffpa` executable needed by our implementation.

## Troubleshooting

### kaffpa not found after installation

Add KaHIP to your PATH:
```bash
# Find where kaffpa is installed
find /usr -name kaffpa 2>/dev/null

# Add to PATH (example)
export PATH=$PATH:/usr/bin
# or
export PATH=$PATH:/path/to/KaHIP/deploy
```

### Permission denied when running kaffpa

```bash
chmod +x /path/to/kaffpa
```

## Testing the Installation

After installation, test with:

```bash
# Should show usage help
kaffpa --help

# Or test with our build script
cd /path/to/Project_Exercise2
source .venv/bin/activate
python Exercise/NeuralLSH/nlsh_build.py -d Raw_Data/MNIST/input.idx3-ubyte \
    -i test_index -type mnist --knn 5 -m 20 --epochs 2
```

## Alternative: Mock Partitioner for Testing

For testing without KaHIP installation, see `INSTALL_KAHIP.md` for a simple random partitioner fallback.
