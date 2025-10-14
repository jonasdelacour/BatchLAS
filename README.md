<div align="center">
  <img src="BatchLAS_logo_transparent.png" alt="BatchLAS Logo" width="200">
</div>

BatchLAS is a high-performance library for batched linear algebra operations that supports multiple backends. It provides an abstraction layer over different vendor-specific libraries while maintaining high performance.

## Features

- Unified API for different hardware backends
- Batched matrix operations
- Support for dense and sparse matrices
- SYCL interoperability for cross-platform performance

## Currently Implemented Operations

### Dense Matrix Operations
- **Basic BLAS operations**
  - Matrix-matrix multiplication (gemm)
  - Matrix-vector multiplication (gemv) 
  - Triangular solve (trsm)

- **LAPACK operations**
  - Cholesky factorization (potrf)
  - LU factorization with partial pivoting (getrf)
  - Solution of linear systems using LU factorization (getrs)
  - Matrix inversion (getri, inv)
  - QR factorization (geqrf)
  - Generation of orthogonal matrix from QR factorization (orgqr)
  - Multiplication by orthogonal matrix from QR factorization (ormqr)
  - Symmetric eigenvalue decomposition (syev)

- **Matrix orthogonalization with multiple algorithms**
  - Cholesky-based methods (Chol2, Cholesky, ShiftChol3)
  - Classical Gram-Schmidt with reorthogonalization (CGS2)
  - Householder QR-based orthogonalization
  - SVQB (SVD-based orthogonalization)

- **Utility operations**
  - Matrix norms (Frobenius, 1-norm, infinity-norm)
  - Condition number computation (cond)
  - Matrix transpose
  - Matrix creation utilities (Identity, Random, Zeros, Ones, Diagonal, Triangular, TriDiagToeplitz)

### Sparse Matrix Operations
- **Basic operations**
  - Sparse matrix-dense matrix multiplication (spmm)
  - Support for CSR (Compressed Sparse Row) format
  - Format conversion between dense and sparse

- **Sparse eigensolvers**
  - Batched LOBPCG for partial eigendecomposition (syevx)
    - Finds largest or smallest eigen-pairs
    - Supports both sparse and dense matrices
    - Configurable orthogonalization algorithms and tolerances
  - Batched Lanczos algorithm for full eigendecompositions (lanczos)
    - Supports sparse and dense matrices
    - Configurable orthogonalization and sorting options
  - Specialized tridiagonal eigensolvers for Lanczos
  - Ritz values computation (ritz_values)
    - Computes eigenvalue approximations from trial vectors
    - Uses Rayleigh quotient for each trial vector
    - Supports both sparse and dense matrices
    - See [RITZ_VALUES.md](RITZ_VALUES.md) for detailed usage

### Advanced Features
- **Batched operations**: All operations support processing multiple matrices simultaneously
- **Multiple data formats**: Dense and CSR sparse matrix support
- **Memory management**: Unified memory vectors and spans for cross-platform compatibility
- **Backend abstraction**: Automatic backend selection or manual specification
- **SYCL integration**: Full SYCL interoperability for cross-platform GPU computing
- **Python bindings**: Complete Python interface with NumPy integration

## Working Backends
- NVIDIA CUDA (cuBLAS, cuSOLVER, cuSPARSE)
- AMD ROCm (rocBLAS, rocSOLVER, rocSPARSE)
- CPU (CBLAS, LAPACKE)

## Requirements

- C++17 compatible compiler
- CMake 3.14 or higher
- SYCL implementation (Intel oneAPI DPC++)
- Optional: CUDA toolkit for NVIDIA GPUs
- Optional: ROCm for AMD GPUs
- Optional: Netlib BLAS/LAPACK for CPU
- Optional: Intel oneMKL for optimized CPU backend (currently experimental)
- For oneMKL support, set the `MKLROOT` environment variable to your oneAPI installation
- Optional: Python 3.x (for Python bindings)

## Installation

### Basic Installation

```bash
git clone https://github.com/yourusername/BatchLAS.git
cd BatchLAS
mkdir build && cd build
cmake ..
make -j$(nproc)
make install
```

### Configuration Options

BatchLAS can be configured with various options:

```bash
cmake .. \
  -DBATCHLAS_BUILD_TESTS=ON \
  -DBATCHLAS_BUILD_EXAMPLES=ON \
  -DBATCHLAS_ENABLE_CUDA=ON \
  -DBATCHLAS_ENABLE_ROCM=ON \
  -DBATCHLAS_ENABLE_OPENMP=ON \
  -DBATCHLAS_BUILD_PYTHON=ON
```

Available options:
- `BATCHLAS_BUILD_TESTS`: Build test suite (default: ON)
- `BATCHLAS_BUILD_EXAMPLES`: Build examples (default: OFF)
- `BATCHLAS_BUILD_DOCS`: Build documentation (default: OFF)
- `BATCHLAS_ENABLE_CUDA`: Enable CUDA support (default: OFF)
- `BATCHLAS_ENABLE_ROCM`: Enable ROCm support (default: OFF)
- `BATCHLAS_ENABLE_OPENMP`: Enable OpenMP support (default: OFF)
- `BATCHLAS_BUILD_PYTHON`: Build Python bindings (default: ON)
- `BATCHLAS_ENABLE_MKL`: Enable Intel oneMKL backend (default: OFF, experimental)
- `BATCHLAS_AMD_ARCH`: AMD GPU architecture when building ROCm backend (default: gfx942)
- `BATCHLAS_NVIDIA_ARCH`: NVIDIA GPU architecture when building CUDA backend (default: sm_50)

## Quick Start

Here's a simple example of using BatchLAS for matrix multiplication:

```cpp
#include <batchlas.hh>

using namespace batchlas;

int main() {
    // Create a context
    auto ctx = Queue(Device::default_device());
    
    // Define matrix dimensions
    const int rows = 1000;
    const int cols = 1000;
    const int k = 1000;
    const int batch_size = 10;
    
    // Create matrices using factory methods
    auto A = Matrix<float>::Random(rows, k, batch_size);
    auto B = Matrix<float>::Random(k, cols, batch_size);
    auto C = Matrix<float>::Zeros(rows, cols, batch_size);
    
    // Initialize data (if needed, Random already initializes)
    // A.fill(1.0f); // Example: fill A with 1.0f
    // B.fill(2.0f); // Example: fill B with 2.0f
    
    // Perform batched matrix multiplication using views of the matrices
    gemm<Backend::AUTO>(ctx, A, B, C, 1.0f, 0.0f, Transpose::NoTrans, Transpose::NoTrans);
    
    // Wait for completion
    ctx.wait();
    
    return 0;
}
```

## Advanced Features

### Matrix Creation and Utilities

BatchLAS provides comprehensive matrix creation utilities:

```cpp
// Create various matrix types
auto identity = Matrix<float>::Identity(100, batch_size);          // Identity matrices
auto random = Matrix<float>::Random(100, 100, false, batch_size);  // Random matrices
auto zeros = Matrix<float>::Zeros(100, 100, batch_size);           // Zero matrices  
auto ones = Matrix<float>::Ones(100, 100, batch_size);             // Matrices filled with ones
auto tridiag = Matrix<float>::TriDiagToeplitz(100, 2.0f, -1.0f, -1.0f, batch_size); // Tridiagonal matrices

// Create sparse matrices in CSR format
auto sparse_A = Matrix<float, MatrixFormat::CSR>(rows, cols, nnz, batch_size);

// Matrix utilities
auto norms = norm<float, MatrixFormat::Dense>(ctx, A, NormType::Frobenius);
auto conditions = cond<Backend::AUTO>(ctx, A, NormType::Frobenius);
auto A_transposed = transpose(ctx, A);
auto A_inverse = inv<Backend::AUTO>(ctx, A);
```

### Orthogonalization

BatchLAS provides various orthogonalization algorithms with configurable parameters:

```cpp
// Allocate workspace memory
UnifiedVector<std::byte> workspace(ortho_buffer_size<Backend::AUTO>(
    ctx, matrices, Transpose::NoTrans, OrthoAlgorithm::ShiftChol3));

// Orthogonalize matrices using different algorithms
ortho<Backend::AUTO>(ctx, matrices, Transpose::NoTrans, workspace, OrthoAlgorithm::CGS2);       // Classical Gram-Schmidt
ortho<Backend::AUTO>(ctx, matrices, Transpose::NoTrans, workspace, OrthoAlgorithm::Chol2);      // Cholesky-based
ortho<Backend::AUTO>(ctx, matrices, Transpose::NoTrans, workspace, OrthoAlgorithm::Householder); // QR-based
ortho<Backend::AUTO>(ctx, matrices, Transpose::NoTrans, workspace, OrthoAlgorithm::SVQB);       // SVD-based

// Orthogonalize with respect to an external metric
ortho<Backend::AUTO>(ctx, A, M, Transpose::NoTrans, Transpose::NoTrans, workspace, OrthoAlgorithm::Chol2, 2);
```

### Sparse Eigensolvers

For large-scale eigenvalue problems:

```cpp
// LOBPCG for finding specific eigenvalues
SyevxParams<float> lobpcg_params;
lobpcg_params.find_largest = true;           // Find largest eigenvalues
lobpcg_params.iterations = 100;              // Maximum iterations
lobpcg_params.extra_directions = 10;         // Extra search directions
lobpcg_params.algorithm = OrthoAlgorithm::CGS2; // Orthogonalization method

UnifiedVector<std::byte> syevx_workspace(syevx_buffer_size<Backend::AUTO>(
    ctx, sparse_A, eigenvalues, neigs, JobType::EigenVectors, eigenvectors, lobpcg_params));

syevx<Backend::AUTO>(ctx, sparse_A, eigenvalues, neigs, syevx_workspace, 
                     JobType::EigenVectors, eigenvectors, lobpcg_params);

// Lanczos for full eigendecomposition
LanczosParams<float> lanczos_params;
lanczos_params.ortho_algorithm = OrthoAlgorithm::CGS2;
lanczos_params.sort_enabled = true;
lanczos_params.sort_order = SortOrder::Ascending;

UnifiedVector<std::byte> lanczos_workspace(lanczos_buffer_size<Backend::AUTO>(
    ctx, sparse_A, all_eigenvalues, JobType::EigenVectors, all_eigenvectors, lanczos_params));

lanczos<Backend::AUTO>(ctx, sparse_A, all_eigenvalues, lanczos_workspace,
                       JobType::EigenVectors, all_eigenvectors, lanczos_params);
```

### Matrix and Vector Views

Work efficiently with matrix and vector subsets:

```cpp
// Create views into existing data
auto A_view = A.view(50, 50);  // View first 50x50 submatrix
auto batch_item = A[0];        // View single matrix from batch
auto col_vector = VectorView<float>(matrix_data + col_offset, rows, 1, 0, batch_size);

// Access and manipulate individual elements
float value = A_view.at(10, 20, 0);  // Element at row 10, col 20, batch 0
A_view.at(5, 5, 0) = 2.5f;           // Set element value
```

## Testing

To run the test suite:

```bash
cd build
ctest
```

## Performance Tuning

BatchLAS automatically selects the most suitable backend for your hardware, but you can manually specify a backend for optimal performance in specific use cases:

```cpp
// Use CUDA backend explicitly on NVIDIA hardware
gemm<Backend::CUDA>(ctx, A, B, C, alpha, beta, Transpose::NoTrans, Transpose::NoTrans);

// Use ROCm backend explicitly on AMD hardware  
gemm<Backend::ROCM>(ctx, A, B, C, alpha, beta, Transpose::NoTrans, Transpose::NoTrans);
```

## Benchmarks

Benchmark executables are built in the `benchmarks` directory. Each benchmark
registers a default set of input sizes, but you can override these at runtime by
providing custom sizes on the command line. Arguments may be integers,
comma&#8209;separated lists or `start:end:num` ranges. When custom sizes are
supplied they replace the registered ones for all benchmarks. You can further
limit execution to specific backends or floating point types using the
`--backend` and `--type` options.

Example:

```bash
./gemm_benchmark 512 512 128 10
./gemm_benchmark 64:256:4 64:256:4 64:256:4 1,2,4
./gemm_benchmark --backend=CUDA --type=float 256 256 64 8
./ortho_benchmark --backend=ROCM --type=double 1024 512 4
./syevx_benchmark --backend=AUTO 2048 2048 50 2
```

Available benchmarks include:
- `gemm_benchmark`: Dense matrix multiplication
- `gemv_benchmark`: Matrix-vector multiplication  
- `ortho_benchmark`: Orthogonalization algorithms
- `syevx_benchmark`: Sparse eigenvalue solvers
- `lanczos_benchmark`: Lanczos eigenvalue algorithm
- `spmm_benchmark`: Sparse matrix-dense matrix multiplication
- `trsm_benchmark`: Triangular solve operations

## License

TBD
