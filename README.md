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
- **Dense Matrix Operations**
  - Basic BLAS operations (gemm, gemv, trsm)
  - Cholesky factorization (potrf)
  - Symmetric eigenvalue decomposition (syev)
  - Matrix orthogonalization with multiple algorithms:
    - Cholesky-based methods (Chol2, ShiftChol3)
    - Classical Gram-Schmidt with reorthogonalization (CGS2)
    - Householder
    - SVQB

- **Sparse Matrix Operations**
  - Sparse matrix-dense matrix multiplication (spmm)
  - Sparse symmetric eigensolvers:
    - Batched LOBPCG for partial eigendecomposition (finding largest or smallest eigen-pairs)
    - Batched Lanczos algorithm for full eigendecompositions
  - CSR format support

## Working Backends
- NVIDIA CUDA (cuBLAS, cuSOLVER, cuSPARSE)
- CPU (CBLAS, LAPACKE)

## Requirements

- C++17 compatible compiler
- CMake 3.14 or higher
- SYCL implementation (Intel oneAPI DPC++)
- Optional: CUDA toolkit for NVIDIA GPUs
- Optional: Netlib BLAS/LAPACK for CPU

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
  -DBATCHLAS_ENABLE_OPENMP=ON
```

Available options:
- `BATCHLAS_BUILD_TESTS`: Build test suite (default: ON)
- `BATCHLAS_BUILD_EXAMPLES`: Build examples (default: OFF)
- `BATCHLAS_BUILD_DOCS`: Build documentation (default: OFF)
- `BATCHLAS_ENABLE_CUDA`: Enable CUDA support (default: OFF)
- `BATCHLAS_ENABLE_OPENMP`: Enable OpenMP support (default: OFF)

## Quick Start

Here's a simple example of using BatchLAS for matrix multiplication:

```cpp
#include <batchlas.hh>

using namespace batchlas;

int main() {
    // Create a context
    auto ctx = Queue(Device::default_device());
    
    // Allocate memory for matrices
    const int rows = 1000;
    const int cols = 1000;
    const int k = 1000;
    const int batch_size = 10;
    const int ld = rows; // Leading dimension
    
    // Allocate matrices on device
    UnifiedVector<float> A_data(rows * k * batch_size);
    UnifiedVector<float> B_data(k * cols * batch_size);
    UnifiedVector<float> C_data(rows * cols * batch_size);
    
    // Initialize data...
    
    // Create matrix handles
    DenseMatHandle<float, BatchType::Batched> A(A_data.data(), rows, k, ld, rows*k, batch_size);
    DenseMatHandle<float, BatchType::Batched> B(B_data.data(), k, cols, k, k*cols, batch_size);
    DenseMatHandle<float, BatchType::Batched> C(C_data.data(), rows, cols, ld, rows*cols, batch_size);
    
    
    // Perform batched matrix multiplication
    gemm<Backend::AUTO>(ctx, A(), B(), C(), 1.0f, 0.0f, Transpose::NoTrans, Transpose::NoTrans);
    
    // Wait for completion
    ctx.wait();
    
    return 0;
}
```

## Advanced Features

### Orthogonalization

BatchLAS provides various orthogonalization algorithms:

```cpp
// Allocate workspace memory
UnifiedVector<std::byte> workspace(ortho_buffer_size<Backend::CUDA>(
    ctx, matrices, Transpose::NoTrans, OrthoAlgorithm::ShiftChol3));

// Orthogonalize matrices
ortho<Backend::CUDA>(ctx, matrices, Transpose::NoTrans, workspace, OrthoAlgorithm::ShiftChol3);
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
gemm<Backend::CUDA>(ctx, A(), B(), C(), alpha, beta, Transpose::NoTrans, Transpose::NoTrans);
```


## License

TBD
