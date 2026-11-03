// SPDX-License-Identifier: MIT
#pragma once
#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/extensions.hh>
#include <batchlas/util/group-collectives.hh>
#include <src/ops/syev/stedc/stedc_v2.hh>
#include <sycl/sycl.hpp>

namespace batchlas {

// Löwner rescale on executor `ex` (v2::RtGroup or v2::SgpPartition<P>): `first`/`stride`
// select the eigen-indices it owns, (0, 1) every index (WG reference variant), (part_id,
// parts_per_wg) one per partition with no work-group barrier. Iteration `eid` reads row `eid`
// of Q (never written here) and writes only v(eid). The product MUST be stedc_v2::prod_scaled:
// narrow partitions lengthen each serial product and the naive one leaves float range at n=64.
// evidence: docs/perf/stedc.md#stedc-partition-parallel-rescale-and-normalize
template <typename T, typename Ex, typename QBatch, typename VView>
inline void maybe_rescale_vectors(bool do_rescale,
                                  const Ex& ex,
                                  int32_t first,
                                  int32_t stride,
                                  int32_t dd,
                                  QBatch& Q_bid,
                                  const VectorView<T>& d_prob,
                                  const VView& v,
                                  int32_t bid) {
    if (!do_rescale) {
        return;
    }

    // One trip count for every reduction group, work masked: the reductions are sub-group
    // shuffles, and a partition that left the loop early makes them read garbage.
    const int32_t iters = (dd + stride - 1) / stride;

    for (int32_t it = 0; it < iters; ++it) {
        const int32_t eid = first + it * stride;
        const bool active = (eid < dd);
        const T Di = active ? d_prob(eid) : T(0);
        // |z_eid|^2 = prod_j Q(eid, j) / (D_eid - D_j), the j == eid factor Q(eid, eid) itself
        const auto prod = stedc_v2::prod_scaled(ex, active ? dd : 0, [&](int32_t j) {
            const T q_elem = Q_bid(eid, j);
            T ratio;
            if (j == eid) {
                ratio = q_elem;
            } else {
                const T denom = Di - d_prob(j);
                ratio = q_elem / denom;
            }
            return ratio;
        });
        if (active && ex.leader()) {
            const T sgn = (v(eid, bid) >= T(0)) ? T(1) : T(-1);
            v(eid, bid) = sgn * stedc_v2::sqrt_scaled(prod);
        }
        stedc_v2::group_sync(ex);
    }
}


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
