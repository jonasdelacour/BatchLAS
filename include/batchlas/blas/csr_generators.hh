#pragma once

#include <batchlas/export.hh>
#include <batchlas/blas/matrix.hh>

#include <complex>
#include <stdexcept>

namespace batchlas {
namespace csr_generators {

/// @file
/// @brief Random CSR test-matrix generators.

/// @brief Generates a batch of random sparse symmetric (real) or Hermitian (complex) n x n CSR matrices.
///
/// Every row holds its diagonal entry. Off-diagonal entries come in (i, j) / (j, i)
/// pairs with conjugate values, drawn uniformly from [-1, 1] (real and imaginary
/// parts independently). Each diagonal is real and set to the row's off-diagonal
/// absolute sum (|re| + |im| for complex) plus @p diagonal_boost, so the matrix is
/// strictly diagonally dominant with a positive diagonal. Column indices are sorted
/// within each row. Storage is allocated and
/// filled on the device; the call waits before returning. Equivalent to
/// Matrix::RandomSparseHermitian().
/// @tparam T  `float`, `double`, `std::complex<float>` or `std::complex<double>`
/// @param n               matrix order
/// @param density         target fraction of stored entries over the full n x n
///                        matrix, diagonal included; clamped to [0, 1]. The count is
///                        rounded up to at least n and to an even off-diagonal count.
/// @param batch_size      number of matrices
/// @param seed            RNG seed; equal seeds give identical matrices
/// @param diagonal_boost  margin added to each diagonal above its off-diagonal row sum
/// @param shared_pattern  true: every batch item has the same sparsity pattern
///                        (values still differ); false: one pattern per item
/// @return an owning CSR Matrix; `nnz()` is the same for every item
/// @throws batchlas::invalid_argument if n <= 0 or batch_size <= 0
/// @ingroup sparse
template <typename T>
BATCHLAS_API Matrix<T, MatrixFormat::CSR> random_sparse_hermitian_csr(int n,
                                                                      float density,
                                                                      int batch_size = 1,
                                                                      unsigned seed = 42,
                                                                      typename base_type<T>::type diagonal_boost = typename base_type<T>::type(1),
                                                                      bool shared_pattern = true);

}  // namespace csr_generators
}  // namespace batchlas
