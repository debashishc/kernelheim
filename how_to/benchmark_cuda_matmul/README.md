# cuda_matrix_multiplication_benchmark (DRAFT)

**Status**: DRAFT - The initial project skeleton has been set up, and the first implementations are currently being developed.

This project is aimed at benchmarking different CUDA matrix multiplication implementations. The goal is to compare performance across various approaches, such as basic, shared memory, and tiled matrix multiplication. As of now, the basic structure is in place, but kernel implementations are still under development.

## Table of Contents

- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Building the Project](#building-the-project)
  - [Running Benchmarks](#running-benchmarks)
- [Implementations Included](#implementations-included)
- [TODOs](#todos)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Project Structure

```
cuda_matrix_multiplication_benchmark/
├── src/
│   ├── kernels/
│   │   ├── mm_basic.cu            # Basic matrix multiplication kernel (in progress)
│   │   ├── mm_shared_memory.cu    # Shared memory optimized kernel (in progress)
│   │   ├── mm_tiled.cu            # Tiled matrix multiplication kernel (in progress)
│   ├── include/
│   │   ├── matrix_utils.h         # Utility functions for matrices (in progress)
│   ├── main.cu                    # Main program to run benchmarks (skeleton in place)
│   └── Makefile                   # Build automation (initial version)
├── benchmarks/
│   ├── run_benchmarks.sh          # Script to run benchmarks (skeleton in place)
│   ├── results/
│   │   └── ...                    # Placeholder for benchmark results
│   └── plots/
│       ├── plot_results.py        # Script to plot results (to be developed)
├── docs/
│   └── ...                        # Documentation
├── README.md                      # Project overview and instructions (this file)
```

---

## Getting Started

### Prerequisites

Ensure you have the following installed:

- **Hardware**: CUDA-enabled GPU
- **Software**:
  - NVIDIA CUDA Toolkit
  - GNU Make
  - Bash shell (for running scripts)
  - Python 3 (for plotting results in future updates)
  - Python libraries: `matplotlib`, `numpy` (for future visualizations)

### Building the Project

The build process is currently being set up. The following steps outline the future build process:

```bash
cd src/
make
```

This will compile the project and produce an executable named `benchmark`. At present, some kernel implementations are incomplete, so building may not yet succeed.

### Running Benchmarks

Once the kernel implementations are complete, you will be able to run the benchmarks via the script in the `benchmarks/` directory:

```bash
cd benchmarks/
chmod +x run_benchmarks.sh
./run_benchmarks.sh
```

This script will execute different matrix multiplication kernels across various matrix sizes and store the results in the `results/` directory.

---

## Implementations Included

Currently, the following implementations are being built:

- **Basic Matrix Multiplication** (`mm_basic.cu`):
  - Each thread computes one element of the output matrix.
  - Currently under development.

- **Shared Memory Optimization** (`mm_shared_memory.cu`):
  - Utilizes shared memory to reduce global memory accesses.
  - Development is in progress.

- **Tiled Matrix Multiplication** (`mm_tiled.cu`):
  - Breaks matrices into sub-matrices (tiles) to optimize memory access patterns.
  - In development.

---

## TODOs

The project is still in its early stages. Below are the key tasks currently planned:

- [ ] **Complete Basic Implementation** (`mm_basic.cu`)
- [ ] **Build Shared Memory Implementation** (`mm_shared_memory.cu`)
- [ ] **Develop Tiled Matrix Multiplication** (`mm_tiled.cu`)
- [ ] **Set Up Benchmarking Script** (`run_benchmarks.sh`)
- [ ] **Add Basic Matrix Utilities** (allocation, initialization, etc.)
- [ ] **Implement Kernel Selection via Command Line in `main.cu`**

---

## Contributing

Contributions are welcome once the project has a more stable foundation. For now, the focus is on completing the initial implementation. If you'd like to contribute:

1. Fork the repository.
2. Create a new branch for your feature.
3. Commit your changes with clear descriptions.
4. Push your branch to your fork.
5. Submit a pull request.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This project is in the early stages of development, and most implementations are incomplete. Keep an eye out for updates!