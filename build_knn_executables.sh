#!/bin/bash
# Build script for k-NN graph construction executables

set -e  # Exit on error

echo "=================================="
echo "Building k-NN Graph Executables"
echo "=================================="

cd Exercise

# Build build_knn_mnist
echo ""
echo "[1/2] Building build_knn_mnist..."
g++ -O3 -std=c++17 \
    -I./Modules/Models/Template \
    -I./Modules/Models/IVFFlat \
    Modules/Models/build_knn_mnist.cpp \
    Modules/Models/Template/data_io.cpp \
    Modules/Models/Template/L2.cpp \
    Modules/Models/IVFFlat/ivfflat.cpp \
    -o build_knn_mnist

if [ -f build_knn_mnist ]; then
    echo "build_knn_mnist compiled successfully"
else
    echo "Failed to compile build_knn_mnist"
    exit 1
fi

# Build build_knn_sift
echo ""
echo "[2/2] Building build_knn_sift..."
g++ -O3 -std=c++17 \
    -I./Modules/Models/Template \
    -I./Modules/Models/IVFFlat \
    Modules/Models/build_knn_sift.cpp \
    Modules/Models/Template/data_io.cpp \
    Modules/Models/Template/L2.cpp \
    Modules/Models/IVFFlat/ivfflat.cpp \
    -o build_knn_sift

if [ -f build_knn_sift ]; then
    echo "build_knn_sift compiled successfully"
else
    echo "Failed to compile build_knn_sift"
    exit 1
fi

cd ..

echo ""
echo "=================================="
echo "Build Complete!"
echo "=================================="
echo ""
echo "Executables created:"
ls -lh Exercise/build_knn_mnist Exercise/build_knn_sift
echo ""
echo "Usage:"
echo "  ./Exercise/build_knn_mnist -d <mnist.idx3-ubyte> -k <knn> -o <output.bin>"
echo "  ./Exercise/build_knn_sift -d <sift.fvecs> -k <knn> -o <output.bin>"
