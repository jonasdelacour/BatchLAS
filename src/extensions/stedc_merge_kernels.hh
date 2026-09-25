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

// Dispatch the merge step (secular solve + eigenvector formation) according to params.merge_variant.
// Replaces the 3-kernel sequence: StedcSecularSolve + StedcRescaleV + StedcMatrixUpdate.
//
// Inputs:
//   eigenvalues  – sorted poles D[0..dd-1] per batch item (dd = n_reduced[bid])
//   v            – secular vector z (already permuted, squared weights for Legacy path)
//   rho          – signed rank-1 update coefficient per batch item
//   n_reduced    – number of non-deflated poles per batch item
//
// Outputs:
//   Qprime       – n×n eigenvector matrix (identity for deflated columns)
//   temp_lambdas – merged eigenvalues (before final sort)
//   info         – per-batch-item convergence status, or nullptr when not
//                  requested. `info_nodes_per_item` is the number of merge nodes
//                  this launch runs per batch item: the level-synchronous driver
//                  merges 2^l siblings of the same item in one launch, so the
//                  kernel index has to be divided down before it indexes `info`.
//                  See src/extensions/info_span.hh.
template <Backend B, typename T>
void stedc_merge_dispatch(Queue& ctx,
                          const VectorView<T>& eigenvalues,
                          const VectorView<T>& v,
                          const Span<T>& rho,
                          const Span<int32_t>& n_reduced,
                          const MatrixView<T, MatrixFormat::Dense>& Qprime,
                          const VectorView<T>& temp_lambdas,
                          const StedcParams<T>& params,
                          // Per-item convergence status, or nullptr when the caller did
                          // not ask for it -- info_report is then a no-op. nodes_per_item
                          // is how many merge nodes this launch covers per batch item, so
                          // a node's status can be folded onto the right item.
                          int32_t* info,
                          int64_t info_nodes_per_item);

} // namespace batchlas
