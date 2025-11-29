#!/bin/bash
# Quick KaHIP Installation Script for Ubuntu/Debian

set -e

echo "========================================="
echo "KaHIP Installation for Neural LSH"
echo "========================================="

# Check if kaffpa already exists
if command -v kaffpa &> /dev/null; then
    echo "✓ KaHIP is already installed!"
    kaffpa --help | head -5
    exit 0
fi

echo "Installing KaHIP from source..."
echo ""

# Install dependencies
echo "[1/4] Installing build dependencies..."
sudo apt-get update
sudo apt-get install -y build-essential cmake git

# Create temp directory
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

# Clone KaHIP
echo "[2/4] Cloning KaHIP repository..."
git clone https://github.com/KaHIP/KaHIP.git
cd KaHIP

# Build
echo "[3/4] Building KaHIP..."
./compile_withcmake.sh

# Add to PATH
echo "[4/4] Setting up PATH..."
KAHIP_PATH="$(pwd)/deploy"

# Add to current session
export PATH=$PATH:$KAHIP_PATH

# Add to ~/.bashrc for persistence
if ! grep -q "KaHIP" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# KaHIP for Neural LSH" >> ~/.bashrc
    echo "export PATH=\$PATH:$KAHIP_PATH" >> ~/.bashrc
    echo "Added KaHIP to ~/.bashrc"
fi

# Verify installation
echo ""
echo "========================================="
echo "Installation Complete!"
echo "========================================="
if command -v kaffpa &> /dev/null; then
    echo "✓ kaffpa is now available"
    kaffpa --help | head -5
    echo ""
    echo "Note: Restart your terminal or run:"
    echo "  source ~/.bashrc"
    echo "to use KaHIP in new terminals."
else
    echo "⚠ Warning: kaffpa not found in PATH"
    echo "Please add this to your PATH:"
    echo "  export PATH=\$PATH:$KAHIP_PATH"
fi

echo ""
echo "KaHIP installed at: $KAHIP_PATH"
