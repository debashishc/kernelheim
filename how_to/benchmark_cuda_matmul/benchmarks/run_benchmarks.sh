#!/bin/bash

# Build the project
make clean
make

# Matrix sizes to test
SIZES=(256 512 1024 2048)

# Implementations to test
IMPLEMENTATIONS=("basic" "shared_memory" "tiled")

# Create results directory if it doesn't exist
mkdir -p results

for size in "${SIZES[@]}"; do
  for impl in "${IMPLEMENTATIONS[@]}"; do
    echo "Running $impl implementation with size $size x $size"
    ./benchmark $impl $size >> results/${impl}_${size}.txt
  done
done
