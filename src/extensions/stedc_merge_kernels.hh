#pragma once
#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/extensions.hh>
#include <batchlas/util/group-collectives.hh>
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

// Löwner rescale. `first`/`stride` select which eigen-indices this reduction group
// owns: (0, 1) walks every index (WG reference variant), (part_id, parts_per_wg)
// gives one index per partition with no work-group barrier. Iteration `eid` reads
// row `eid` of Q (never written here) and writes only v(eid).
//
// The product is a (mantissa, exponent) pair, REQUIRED: narrow partitions lengthen
// each serial product and the naive one leaves float range at n=64. frexp scaling is
// by exact powers of two, so in-range results are bit-identical to the naive product.
// evidence: docs/perf/stedc.md#stedc-partition-parallel-rescale-and-normalize
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

    const int32_t lane = adapter.lane_id();
    const int32_t width = adapter.width();

    // One trip count for every reduction group, work masked: the reductions are sub-group
    // shuffles, and a partition that left the loop early makes them read garbage.
    const int32_t iters = (dd + stride - 1) / stride;

    for (int32_t it = 0; it < iters; ++it) {
        const int32_t eid = first + it * stride;
        const bool active = (eid < dd);
        const T Di = active ? d_prob(eid) : T(0);
        T mant = T(1);
        int32_t expo = 0;
        for (int32_t j = lane; active && j < dd; j += width) {
            const T q_elem = Q_bid(eid, j);
            T ratio;
            if (j == eid) {
                ratio = q_elem;
            } else {
                const T denom = Di - d_prob(j);
                ratio = q_elem / denom;
            }
            mant *= ratio;
            // Renormalize every factor so |mant| stays in [0.5, 1); exact,
            // since frexp only rescales by a power of two.
            if (mant != T(0) && sycl::isfinite(mant)) {
                int e = 0;
                mant = sycl::frexp(mant, &e);
                expo += e;
            }
        }

        // Each lane's |mant| is in [0.5, 1), so one product over a wide group bottoms out at
        // 0.5^width (2^-256 at width 256 is zero in float). Reduce 32 lanes per round and
        // renormalize between rounds: every partial stays >= 2^-33.
        T mant_all = T(1);
        int32_t expo_all = adapter.template reduce_sum<int32_t>(expo);
        for (int32_t r0 = 0; r0 < width; r0 += 32) {
            const bool mine = lane >= r0 && lane < r0 + 32;
            mant_all *= adapter.reduce_product(mine ? mant : T(1));
            if (mant_all != T(0) && sycl::isfinite(mant_all)) {
                int e = 0;
                mant_all = sycl::frexp(mant_all, &e);
                expo_all += e;
            }
        }

        if (active && lane == 0) {
            // mag = sqrt(|mant_all| * 2^expo_all), split so neither the square
            // root's argument nor its result can leave range.
            const int32_t half = expo_all >> 1;              // floor(expo_all / 2)
            const int32_t rem = expo_all - 2 * half;         // 0 or 1
            T mag = sycl::sqrt(sycl::fabs(mant_all) * (rem ? T(2) : T(1)));
            mag = sycl::ldexp(mag, half);
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
