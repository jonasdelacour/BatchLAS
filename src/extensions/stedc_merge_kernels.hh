#pragma once
#include <blas/matrix.hh>
#include <blas/functions.hh>
#include <blas/extensions.hh>

namespace batchlas {

// Dispatch the merge step (secular solve + eigenvector formation) according to params.merge_variant.
// Replaces the 3-kernel sequence: StedcSecularSolve + StedcRescaleV + StedcMatrixUpdate.
//
// Inputs:
//   eigenvalues  – sorted poles D[0..dd-1] per batch item (dd = n_reduced[bid])
//   v            – secular vector z (already permuted, squared weights for Legacy path)
//   rho          – rank-1 update coefficient per batch item
//   n_reduced    – number of non-deflated poles per batch item
//   e_m_minus_1  – e(m-1) per batch item, used for sign of rho
//   n            – full problem size (for Qprime dimensioning)
//
// Outputs:
//   Qprime       – n×n eigenvector matrix (identity for deflated columns)
//   temp_lambdas – merged eigenvalues (before final sort)
template <Backend B, typename T>
void stedc_merge_dispatch(Queue& ctx,
                          const VectorView<T>& eigenvalues,
                          const VectorView<T>& v,
                          const Span<T>& rho,
                          const Span<int32_t>& n_reduced,
                          const VectorView<T>& e,  // original e vector (for sign at m-1)
                          int64_t m,                // split point
                          int64_t n,                // full problem size
                          const MatrixView<T, MatrixFormat::Dense>& Qprime,
                          const VectorView<T>& temp_lambdas,
                          const StedcParams<T>& params);

} // namespace batchlas
