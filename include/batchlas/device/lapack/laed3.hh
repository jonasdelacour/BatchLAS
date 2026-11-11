// SPDX-License-Identifier: MIT
#pragma once
// xLAED3: a deflated merge's roots (laed4) and eigenvectors Q(:, j) = z~ ./ (d - lambda_j),
// z~ the Löwner-rescaled weights (Gu-Eisenstat). laed3 is the host fused merges' body and
// device::stedc's; laed3_detail holds its steps (moved from src/ops/syev/stedc unchanged).

#include <cstdint>
#include <limits>

#include <sycl/sycl.hpp>

#include <batchlas/device.hh>
#include <batchlas/device/lapack/laed4.hh>

namespace batchlas::device {

namespace laed3_detail {

/// q(i) = (d(i) - origin) - tau, the root's denominators (at the origin pole exactly -tau; an
/// exact zero is clamped to eps (|d(i)| + 1) to keep the Löwner reciprocal finite).
template <typename T, typename Ex, typename QCol, typename DV>
BATCHLAS_DEVICE_HELPER void denominators(const Ex& ex, const QCol& q, const DV& d_prob, T origin, T tau) {
    map(
        ex, q,
        [=](T di) {
            T denom = (di - origin) - tau;
            if (denom == T(0)) {
                denom = std::numeric_limits<T>::epsilon() * (sycl::fabs(di) + T(1));
            }
            return denom;
        },
        d_prob);
}

/// Löwner rescale on ex (a group, or a partition): `first`/`stride` select the eigen-indices it
/// owns, (0, 1) every index, (part_id, parts_per_wg) one per partition with no work-group
/// barrier. Iteration eid reads row eid of Q and writes only v(eid).
// prod_scaled: the naive product leaves float range at n = 64.
// evidence: docs/perf/stedc.md#stedc-partition-parallel-rescale-and-normalize
template <typename T, typename Ex, typename QBatch, typename DV, typename VView>
BATCHLAS_DEVICE_HELPER void rescale(bool do_rescale, const Ex& ex, std::int32_t first, std::int32_t stride, std::int32_t dd, QBatch& Q_bid,
                                 const DV& d_prob, const VView& v, std::int32_t bid) {
    if (!do_rescale) {
        return;
    }

    // One trip count for every reduction group: a partition leaving early breaks the shuffles.
    const std::int32_t iters = (dd + stride - 1) / stride;

    for (std::int32_t it = 0; it < iters; ++it) {
        const std::int32_t eid = first + it * stride;
        const bool active = (eid < dd);
        const T Di = active ? d_prob(eid) : T(0);
        // |z_eid|^2 = prod_j Q(eid, j) / (D_eid - D_j), the j == eid factor Q(eid, eid) itself
        const auto prod = prod_scaled(ex, active ? dd : 0, [&](std::int32_t j) {
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
            v(eid, bid) = sgn * sqrt_scaled(prod);
        }
        if constexpr (laed4_detail::is_group_v<Ex>) sync<scope::Group>(ex, Mem::Global);
    }
}

/// Q(:, eig) = z ./ Q(:, eig), divided by its Blue 2-norm; `first`/`stride` as in rescale
/// (column eig reads z and touches only Q(:, eig)).
template <typename T, typename Ex, typename QMat, typename VView>
BATCHLAS_DEVICE_HELPER void form_vectors(const Ex& ex, std::int32_t first, std::int32_t stride, std::int32_t dd, const QMat& Q, const VView& v,
                                   std::int32_t bid) {
    // Uniform trip count (see rescale): an inactive group runs every op on an empty column.
    const std::int32_t iters = (dd + stride - 1) / stride;
    const auto z = vec(v.batch_item(bid));

    for (std::int32_t it = 0; it < iters; ++it) {
        const std::int32_t eig = first + it * stride;
        const bool active = (eig < dd);
        const auto q = Q.col(active ? eig : 0).head(active ? dd : 0);
        map(ex, q, [](T zi, T den) { return zi / den; }, z, q);
        // No barrier: nrm2 walks q with map's owner stride (each item reads what it wrote).
        const T nrm = nrm2<alg::Blue>(ex, q);
        map(ex, q, [=](T qi) { return qi / nrm; }, q);
        if constexpr (laed4_detail::is_group_v<Ex>) sync<scope::Group>(ex, Mem::Global);
    }
}

}  // namespace laed3_detail

/// One merge's secular step, COLLECTIVE over the work-group wex. Roots k = first, first +
/// stride, ... < dd are solved on pex (a partition of wex, or wex itself with (0, 1)); for each,
/// on_root(k, root) runs on pex's leader and Q(:, k) receives the root's denominators. Then the
/// weights v are Löwner-rescaled (do_rescale) and Q(:, 0..dd) becomes the merged eigenvectors.
/// d_prob, z_prob: the dd sorted poles and weights (any memory); abs_2rho = |2 rho|; Q_bid: Q
/// by (i, j) for the rescale; v(i, bid) / v.batch_item(bid): the weights (written).
template <class T, class WEx, class PEx, class DV, class QMat, class QBatch, class VView, class OnRoot>
BATCHLAS_DEVICE_INLINE void laed3(const WEx& wex, const PEx& pex, std::int32_t first, std::int32_t stride, std::int32_t dd, const DV& d_prob,
                               const DV& z_prob, T abs_2rho, std::int32_t max_iter, bool do_rescale, const QMat& Q, QBatch& Q_bid,
                               const VView& v, std::int32_t bid, const OnRoot& on_root) {
    for (std::int32_t root_ix = first; root_ix < dd; root_ix += stride) {
        laed4_root<T> root;
        if (root_ix == dd - 1) {
            root = laed4_detail::solve_last(pex, d_prob, z_prob, dd, abs_2rho, max_iter);
        } else {
            root = laed4_detail::solve_inner(pex, d_prob, z_prob, dd, root_ix, abs_2rho, max_iter);
        }
        if (pex.leader()) on_root(root_ix, root);
        laed3_detail::denominators(pex, Q.col(root_ix).head(dd), d_prob, root.origin, root.tau);
    }

    sync<scope::Group>(wex, Mem::Global);

    // Rescale reads Q, writes v; form_vectors the reverse. On the group each step syncs itself.
    laed3_detail::rescale<T>(do_rescale, pex, first, stride, dd, Q_bid, d_prob, v, bid);
    if constexpr (!laed4_detail::is_group_v<PEx>) sync<scope::Group>(wex, Mem::Global);
    laed3_detail::form_vectors<T>(pex, first, stride, dd, Q, v, bid);
}

}  // namespace batchlas::device
