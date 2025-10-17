# Ritz Values Function

## Overview

The `ritz_values` function computes Ritz values from a matrix and a set of trial vectors. Ritz values are approximations to eigenvalues of a matrix, computed using the Rayleigh quotient.

## Mathematical Background

For a matrix **A** and a set of trial vectors **V** (where each column is a potential eigenvector), the Ritz value for the j-th trial vector v_j is computed as:

```
ritz_value[j] = (v_j^T * A * v_j) / (v_j^T * v_j)
```

This represents the Rayleigh quotient, which provides an approximation to an eigenvalue when v_j is close to an actual eigenvector.

## Usage

### Basic Example

```cpp
#include <blas/linalg.hh>
#include <blas/extensions.hh>

using namespace batchlas;

// Create a queue for GPU computation
auto ctx = std::make_shared<Queue>(Device::default_device());

// Define dimensions
constexpr int n = 100;      // Matrix size
constexpr int k = 10;       // Number of trial vectors
constexpr int batch = 1;    // Batch size

// Create a symmetric matrix A (e.g., CSR format for sparse)
Matrix<float, MatrixFormat::CSR> A = /* initialize your matrix */;
auto A_view = A.view();

// Create trial vectors V (columns are potential eigenvectors)
Matrix<float, MatrixFormat::Dense> V(n, k, batch);
auto V_view = V.view();
// Initialize V with your trial vectors...

// Allocate output for Ritz values
Vector<float> ritz_vals(k, batch);
auto ritz_vals_view = ritz_vals.view();

// Compute required workspace size
size_t workspace_size = ritz_values_workspace<Backend::SYCL, float, MatrixFormat::CSR>(
    *ctx, A_view, V_view, ritz_vals_view);

// Allocate workspace
UnifiedVector<std::byte> workspace(workspace_size);

// Compute Ritz values
ritz_values<Backend::SYCL, float, MatrixFormat::CSR>(
    *ctx, A_view, V_view, ritz_vals_view, workspace);

// Wait for completion
ctx->wait();

// Access results
auto ritz_host = ritz_vals.data().to_vector();
for (int i = 0; i < k; ++i) {
    std::cout << "Ritz value " << i << ": " << ritz_host[i] << std::endl;
}
```

### With Dense Matrices

```cpp
// For dense matrices, simply use MatrixFormat::Dense
Matrix<float, MatrixFormat::Dense> A_dense(n, n, batch);
// ... initialize A_dense ...

size_t workspace_size = ritz_values_workspace<Backend::SYCL, float, MatrixFormat::Dense>(
    *ctx, A_dense.view(), V.view(), ritz_vals.view());
UnifiedVector<std::byte> workspace(workspace_size);

ritz_values<Backend::SYCL, float, MatrixFormat::Dense>(
    *ctx, A_dense.view(), V.view(), ritz_vals.view(), workspace);
```

### Batched Computation

```cpp
constexpr int batch = 10;  // Process 10 matrices simultaneously

// Create batched matrices
Matrix<float, MatrixFormat::Dense> A(n, n, batch);
Matrix<float, MatrixFormat::Dense> V(n, k, batch);
Vector<float> ritz_vals(k, batch);

// Initialize matrices for each batch...

// Compute Ritz values for all batches
size_t workspace_size = ritz_values_workspace<Backend::SYCL, float, MatrixFormat::Dense>(
    *ctx, A.view(), V.view(), ritz_vals.view());
UnifiedVector<std::byte> workspace(workspace_size);

ritz_values<Backend::SYCL, float, MatrixFormat::Dense>(
    *ctx, A.view(), V.view(), ritz_vals.view(), workspace);

ctx->wait();

// Results for batch b are at indices [b*k, (b+1)*k)
auto ritz_host = ritz_vals.data().to_vector();
for (int b = 0; b < batch; ++b) {
    std::cout << "Batch " << b << " Ritz values:" << std::endl;
    for (int i = 0; i < k; ++i) {
        std::cout << "  [" << i << "]: " << ritz_host[b * k + i] << std::endl;
    }
}
```

## Use Cases

1. **Eigenvalue Approximation**: Quickly estimate eigenvalues using approximate eigenvectors from iterative methods like Lanczos or Arnoldi.

2. **Convergence Checking**: Monitor how well trial vectors approximate actual eigenvectors by tracking changes in Ritz values across iterations.

3. **Subspace Methods**: Compute eigenvalues of the projected matrix V^T * A * V in subspace iteration methods.

4. **Quality Assessment**: Evaluate the quality of approximate eigenvectors obtained from various algorithms.

## Function Signatures

### Core Function
```cpp
template <Backend B, typename T, MatrixFormat MFormat>
Event ritz_values(Queue& ctx,
                  const MatrixView<T, MFormat>& A,
                  const MatrixView<T, MatrixFormat::Dense>& V,
                  const VectorView<typename base_type<T>::type>& ritz_vals,
                  Span<std::byte> workspace);
```

### Workspace Computation
```cpp
template <Backend B, typename T, MatrixFormat MFormat>
size_t ritz_values_workspace(Queue& ctx,
                             const MatrixView<T, MFormat>& A,
                             const MatrixView<T, MatrixFormat::Dense>& V,
                             const VectorView<typename base_type<T>::type>& ritz_vals);
```

## Parameters

- **ctx**: Execution context/device queue for GPU operations
- **A**: Input matrix (can be sparse CSR or dense format)
- **V**: Trial vectors stored as a dense matrix (columns are trial eigenvectors)
- **ritz_vals**: Output vector to store computed Ritz values
- **workspace**: Pre-allocated workspace buffer (size from `ritz_values_workspace`)

## Supported Types

- **Backends**: SYCL, CUDA, ROCM, NETLIB, MKL
- **Data Types**: `float`, `double`, `std::complex<float>`, `std::complex<double>`
- **Matrix Formats**: Dense, CSR (Compressed Sparse Row)

## Notes

- The function computes the Rayleigh quotient for each column of V independently
- For complex types, the function uses conjugate transpose in the computation
- The trial vectors in V do not need to be normalized; normalization is handled internally
- For best results, trial vectors should be orthogonal or nearly orthogonal
- Workspace memory is automatically managed using the `ritz_values_workspace` function
