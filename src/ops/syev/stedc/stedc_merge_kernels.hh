// SPDX-License-Identifier: MIT
#pragma once
#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/extensions.hh>
#include <batchlas/util/group-collectives.hh>
#include <src/ops/syev/stedc/loewner.hh>
#include <sycl/sycl.hpp>

namespace batchlas {

struct WorkgroupAdapter {
    sycl::group<1> wg;
    int32_t tid;
    int32_t bdim;

    inline int32_t lane_id() const {
        return tid;
    }

    inline int32_t width() const {
        return bdim;
    }

    template <typename T>
    inline T reduce_product(T value) const {
        return batchlas::portable::reduce_over_group(wg, value, sycl::multiplies<T>());
    }

    inline void sync() const {
        sycl::group_barrier(wg);
    }

    template <typename T>
    inline T reduce_sum(T value) const {
        return batchlas::portable::reduce_over_group(wg, value, sycl::plus<T>());
    }

    template <typename T>
    inline T broadcast(T value) const {
        return batchlas::portable::group_broadcast(wg, value, 0);
    }
};

// zhat <- sign(z) |zhat| (loewner_weight) for the indices this reduction group owns: (0, 1)
// walks every index (WG reference variant), (part_id, parts_per_wg) gives one index per
// partition with no work-group barrier. Index i reads row i of Q and writes only v(i).
template <typename T, typename Adapter, typename QBatch, typename VView>
inline void maybe_rescale_vectors(bool do_rescale,
                                  const Adapter& adapter,
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

    // One trip count for every reduction group, work masked (see loewner_weight).
    const int32_t iters = (dd + stride - 1) / stride;

    for (int32_t it = 0; it < iters; ++it) {
        const int32_t eid = first + it * stride;
        const bool active = (eid < dd);
        const tridiag::ScaledProduct<T> zhat_sq = tridiag::loewner_weight(adapter, active, eid, dd, Q_bid, d_prob);
        if (active && adapter.lane_id() == 0) {
            const T mag = zhat_sq.sqrt_value();
            const T sgn = (v(eid, bid) >= T(0)) ? T(1) : T(-1);
            v(eid, bid) = sgn * mag;
        }
        adapter.sync();
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
//   rho          – signed cut off-diagonal beta per batch item (tridiag::Tear); the update
//                  coefficient is Tear<T>{beta}.rho() = 2|beta|
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
