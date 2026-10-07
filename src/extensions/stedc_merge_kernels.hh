#pragma once
#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/extensions.hh>

namespace batchlas {

template <Backend B, typename T>
void stedc_merge_fused(Queue& ctx,
                       const VectorView<T>& eigenvalues,
                       const VectorView<T>& v,
                       const Span<T>& rho,
                       const Span<int32_t>& n_reduced,
                       const MatrixView<T, MatrixFormat::Dense>& Qprime,
                       const VectorView<T>& temp_lambdas,
                       const StedcParams<T>& params,
                       int32_t* info,
                       int64_t info_nodes_per_item);

template <Backend B, typename T>
void stedc_merge_fused_cta(Queue& ctx,
                           const VectorView<T>& eigenvalues,
                           const VectorView<T>& v,
                           const Span<T>& rho,
                           const Span<int32_t>& n_reduced,
                           const MatrixView<T, MatrixFormat::Dense>& Qprime,
                           const VectorView<T>& temp_lambdas,
                           const StedcParams<T>& params,
                           int32_t* info,
                           int64_t info_nodes_per_item);

// Merge step (secular solve + eigenvector formation), dispatched on params.merge_variant.
//   eigenvalues  – sorted poles D[0..dd-1] per batch item (dd = n_reduced[bid])
//   v            – secular vector z (already permuted, squared weights for Legacy path)
//   rho          – signed rank-1 update coefficient per batch item
//   n_reduced    – number of non-deflated poles per batch item
// Outputs: Qprime (n×n, identity for deflated columns), temp_lambdas (unsorted),
// info (per-item status or nullptr; divide the node index by info_nodes_per_item).
template <Backend B, typename T>
void stedc_merge_dispatch(Queue& ctx,
                          const VectorView<T>& eigenvalues,
                          const VectorView<T>& v,
                          const Span<T>& rho,
                          const Span<int32_t>& n_reduced,
                          const MatrixView<T, MatrixFormat::Dense>& Qprime,
                          const VectorView<T>& temp_lambdas,
                          const StedcParams<T>& params,
                          // Per-item status or nullptr; see info_nodes_per_item above.
                          int32_t* info,
                          int64_t info_nodes_per_item);

} // namespace batchlas
