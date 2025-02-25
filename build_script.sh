#!/bin/bash

# Exit on error
set -e

# Create build directory
mkdir -p build
cd build

# Configure CMake
cmake .. 

# Build
cmake --build . -- -j$(nproc)

# Return to original directory
cd ..

echo "Build completed successfully."
echo "Usage: ./build/transformer_example <model_path> [quantization_mode]"
