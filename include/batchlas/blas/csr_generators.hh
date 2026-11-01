// SPDX-License-Identifier: MIT
#pragma once

#include <batchlas/export.hh>
#include <batchlas/blas/matrix.hh>

#include <complex>
#include <stdexcept>

/// @file
/// @brief Random CSR test-matrix generators (`batchlas::csr_generators`).

namespace batchlas {
/// @brief Random sparse test-matrix generators.
namespace csr_generators {

/// @brief Generates a batch of random sparse symmetric (real) or Hermitian (complex) \f$n \times n\f$ CSR matrices.
///
/// Every row holds its diagonal entry. Off-diagonal entries come in \f$(i, j)\f$ / \f$(j, i)\f$
/// pairs with conjugate values, \f$a_{ji} = \overline{a_{ij}}\f$, drawn uniformly from \f$[-1, 1]\f$
/// (real and imaginary parts independently). Each diagonal is real and set to the row's
/// off-diagonal absolute sum plus @p diagonal_boost \f$\delta\f$,
/// \f[ a_{ii} = \sum_{j \ne i} \bigl( \lvert \mathrm{Re}\, a_{ij} \rvert + \lvert \mathrm{Im}\, a_{ij} \rvert \bigr) + \delta, \f]
/// so the matrix is strictly diagonally dominant with a positive diagonal. Column indices are sorted
/// within each row. Storage is allocated and
/// filled on the device; the call waits before returning. Equivalent to
/// Matrix::RandomSparseHermitian().
/// @tparam T  `float`, `double`, `std::complex<float>` or `std::complex<double>`
/// @param[in] n               matrix order \f$n\f$
/// @param[in] density         target fraction of stored entries over the full \f$n \times n\f$
///                            matrix, diagonal included; clamped to \f$[0, 1]\f$. The count is
///                            rounded to nearest, raised to at least \f$n\f$, its off-diagonal part
///                            rounded up to even, and capped at \f$n^2\f$.
/// @param[in] batch_size      number of matrices
/// @param[in] seed            RNG seed; equal seeds give identical matrices
/// @param[in] diagonal_boost  margin \f$\delta\f$ added to each diagonal above its off-diagonal row sum
/// @param[in] shared_pattern  true: every batch item has the same sparsity pattern
///                            (values still differ); false: one pattern per item
/// @return an owning CSR Matrix; `nnz()` is the same for every item
/// @throws batchlas::invalid_argument if \f$n \le 0\f$ or \f$\mathit{batch\_size} \le 0\f$
/// @ingroup api_sparse_lowlevel
template <typename T>
BATCHLAS_API Matrix<T, MatrixFormat::CSR> random_sparse_hermitian_csr(int n,
                                                                      float density,
                                                                      int batch_size = 1,
                                                                      unsigned seed = 42,
                                                                      typename base_type<T>::type diagonal_boost = typename base_type<T>::type(1),
                                                                      bool shared_pattern = true);

}  // namespace csr_generators
}  // namespace batchlas
