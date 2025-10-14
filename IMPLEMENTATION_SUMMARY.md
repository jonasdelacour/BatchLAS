# Implementation Summary: Ritz Values Function

## Overview
This PR implements a utility function `ritz_values` that computes Ritz values (eigenvalue approximations) from a matrix A and a set of trial vectors V.

## Files Modified/Created

### Core Implementation (188 lines)
- **`src/extensions/ritz_values.cc`**: Main implementation
  - Computes Rayleigh quotient for each trial vector: `ritz_value[j] = (v_j^T * A * v_j) / (v_j^T * v_j)`
  - Uses SYCL kernels for parallel computation on GPU
  - Supports both dense and sparse (CSR) matrices
  - Handles complex and real data types correctly
  - Includes explicit template instantiations for all supported backends

### Header Declarations (92 lines added)
- **`include/blas/extensions.hh`**: Function declarations
  - Main `ritz_values` function template
  - `ritz_values_workspace` function for memory allocation
  - Multiple forwarding overloads for convenience (owning Matrix/Vector types)
  - Comprehensive documentation comments

### Tests (242 lines)
- **`tests/ritz_values_tests.cc`**: Comprehensive test suite
  - `TridiagonalMatrix` test: Validates against known eigenvalues
  - `DiagonalMatrix` test: Tests with simple identity eigenvectors
  - `SparseCSRMatrix` test: Verifies sparse matrix support
  - All tests use proper tolerances and batch processing

### Documentation (167 lines)
- **`RITZ_VALUES.md`**: Detailed usage guide
  - Mathematical background
  - Basic usage examples
  - Dense and sparse matrix examples
  - Batched computation examples
  - Function signatures and parameters
  - Use cases and applications

### Build System Updates
- **`src/extensions/CMakeLists.txt`**: Added `ritz_values.cc` to sources
- **`tests/CMakeLists.txt`**: Added `ritz_values_tests` to test targets
- **`README.md`**: Updated to list the new function

## Key Features

### Mathematical Computation
- Computes Rayleigh quotient: `(v^T * A * v) / (v^T * v)` for each trial vector
- Properly handles conjugate transpose for complex types
- Normalizes by vector norm (denominator in Rayleigh quotient)

### Performance Optimizations
- Uses efficient matrix-matrix multiplication (gemm/spmm) for A*V computation
- Single SYCL kernel launch for all Ritz value computations
- Parallel computation across batches and trial vectors
- Minimal memory overhead with workspace management

### Flexibility
- Supports multiple backends: SYCL, CUDA, ROCM, NETLIB, MKL
- Works with both dense and sparse (CSR) matrices
- Handles batched operations efficiently
- Supports float, double, complex<float>, complex<double>

## Usage Pattern

Following the repository conventions:

```cpp
// 1. Create matrices and vectors
Matrix<float, MatrixFormat::CSR> A = /* sparse matrix */;
Matrix<float, MatrixFormat::Dense> V = /* trial vectors */;
Vector<float> ritz_vals(k, batch);

// 2. Compute workspace size
size_t ws_size = ritz_values_workspace<Backend::SYCL, float, MatrixFormat::CSR>(
    *ctx, A.view(), V.view(), ritz_vals.view());

// 3. Allocate workspace
UnifiedVector<std::byte> workspace(ws_size);

// 4. Compute Ritz values
ritz_values<Backend::SYCL, float, MatrixFormat::CSR>(
    *ctx, A.view(), V.view(), ritz_vals.view(), workspace);

// 5. Wait and use results
ctx->wait();
```

## Testing

The implementation includes three comprehensive tests:

1. **TridiagonalMatrix**: Tests with a well-conditioned tridiagonal matrix with known eigenvalues
2. **DiagonalMatrix**: Tests with diagonal matrix (simplest case with exact eigenvectors)
3. **SparseCSRMatrix**: Validates sparse matrix support with CSR format

All tests:
- Verify against analytically known eigenvalues
- Use appropriate numerical tolerances
- Test batched operations
- Cover different matrix formats

## Design Decisions

### Workspace Management
Following the repository pattern, the function requires pre-allocated workspace:
- `ritz_values_workspace()` computes required memory
- User allocates workspace buffer
- `ritz_values()` uses the workspace for temporary storage

### Computation Approach
Two-step process:
1. Compute `AV = A * V` using optimized BLAS/sparse routines
2. Launch SYCL kernel to compute Rayleigh quotients in parallel

This is more efficient than computing each Ritz value separately.

### Memory Layout
- Trial vectors V are stored as columns of a dense matrix
- Output ritz_vals is a vector with k elements per batch
- Batch-first layout: `ritz_vals[batch * k + trial_vector_index]`

## Building and Testing

The implementation follows standard CMake build process:

```bash
# Configure
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build -j

# Test
cd build && ctest --output-on-failure -R ritz_values
```

**Note**: Requires Intel oneAPI DPC++ compiler or equivalent SYCL implementation.

## Integration Points

The function integrates naturally with:
- **Lanczos algorithm**: Verify Ritz values during iterations
- **LOBPCG (syevx)**: Check convergence of approximate eigenvectors
- **Custom eigensolvers**: Evaluate trial vectors from any source

## Future Enhancements (Optional)

Possible improvements if needed:
1. Add option to compute residual norms: `||A*v - λ*v||`
2. Optimize for case where V is orthonormal (skip normalization)
3. Add GPU kernel optimizations for very large k values
4. Support block-wise computation for memory-constrained cases

## Statistics

- **Total lines added**: 696 lines
- **New source files**: 1 (ritz_values.cc)
- **New test files**: 1 (ritz_values_tests.cc)
- **Documentation files**: 1 (RITZ_VALUES.md)
- **Modified files**: 4 (CMakeLists, README, extensions.hh)
