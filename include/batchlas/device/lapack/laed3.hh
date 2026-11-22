// SPDX-License-Identifier: MIT
#pragma once
/// @file
/// @brief `laed3` (xLAED3): a deflated divide-and-conquer merge's roots and eigenvectors, as
/// `stedc` and the host merges run it. Not covered by the compatibility promise: its signature
/// follows those callers.
/// @ingroup api_device_lowlevel
// xLAED3: a deflated merge's roots (laed4) and eigenvectors Q(:, j) = z~ ./ (d - lambda_j),
// z~ the Löwner-rescaled weights (Gu-Eisenstat). laed3 is the host fused merges' body and
// device::stedc's; detail holds its steps (moved from src/ops/syev/stedc unchanged).

#include <cstdint>
#include <limits>

#include <sycl/sycl.hpp>

#include <batchlas/device.hh>
#include <batchlas/device/lapack/laed4.hh>

namespace batchlas::device {

namespace detail {

/// q(i) = (d(i) - origin) - tau, the root's denominators (at the origin pole exactly -tau; an
/// exact zero is clamped to eps (|d(i)| + 1) to keep the Löwner reciprocal finite).
template <class Real, class Ex, class QCol, class DV>
BATCHLAS_DEVICE_HELPER void denominators(const Ex& ex, const QCol& q, const DV& d_prob, Real origin, Real tau) {
    map(
        ex, q,
        [=](Real di) {
            Real denom = (di - origin) - tau;
            if (denom == Real(0)) {
                denom = std::numeric_limits<Real>::epsilon() * (sycl::fabs(di) + Real(1));
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
template <class Real, class Ex, class QBatch, class DV, class VView>
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
        const Real Di = active ? d_prob(eid) : Real(0);
        // |z_eid|^2 = prod_j Q(eid, j) / (D_eid - D_j), the j == eid factor Q(eid, eid) itself
        const auto prod = prod_scaled(ex, active ? dd : 0, [&](std::int32_t j) {
            const Real q_elem = Q_bid(eid, j);
            Real ratio;
            if (j == eid) {
                ratio = q_elem;
            } else {
                const Real denom = Di - d_prob(j);
                ratio = q_elem / denom;
            }
            return ratio;
        });
        if (active && ex.leader()) {
            const Real sgn = (v(eid, bid) >= Real(0)) ? Real(1) : Real(-1);
            v(eid, bid) = sgn * sqrt_scaled(prod);
        }
        if constexpr (detail::is_group_v<Ex>) sync<Scope::Group>(ex, Mem::Global);
    }
}

/// Q(:, eig) = z ./ Q(:, eig), divided by its Blue 2-norm; `first`/`stride` as in rescale
/// (column eig reads z and touches only Q(:, eig)).
template <class Real, class Ex, class QMat, class VView>
BATCHLAS_DEVICE_HELPER void form_vectors(const Ex& ex, std::int32_t first, std::int32_t stride, std::int32_t dd, const QMat& Q, const VView& v,
                                   std::int32_t bid) {
    // Uniform trip count (see rescale): an inactive group runs every op on an empty column.
    const std::int32_t iters = (dd + stride - 1) / stride;
    const auto z = vec(v.batch_item(bid));

    for (std::int32_t it = 0; it < iters; ++it) {
        const std::int32_t eig = first + it * stride;
        const bool active = (eig < dd);
        const auto q = Q.col(active ? eig : 0).head(active ? dd : 0);
        map(ex, q, [](Real zi, Real den) { return zi / den; }, z, q);
        // No barrier: nrm2 walks q with map's owner stride (each item reads what it wrote).
        const Real nrm = nrm2<alg::Blue>(ex, q);
        map(ex, q, [=](Real qi) { return qi / nrm; }, q);
        if constexpr (detail::is_group_v<Ex>) sync<Scope::Group>(ex, Mem::Global);
    }
}

}  // namespace detail

/// @brief xLAED3: one deflated merge's secular step, the roots and then the merged eigenvectors,
/// COLLECTIVE over the work-group `wex`.
///
/// Roots k = first, first + stride, ... below dd are solved on `pex`; for each,
/// `on_root(k, origin, tau, converged)` runs on pex's leader (the root is origin + tau) and
/// Q(:, k) receives the root's denominators. Then the weights are Löwner-rescaled (Gu-Eisenstat)
/// and Q(:, 0) to Q(:, dd - 1) become the merged eigenvectors.
/// @param[in]     wex         the work-group executor
/// @param[in]     pex         a partition of wex, or wex itself with first = 0 and stride = 1
/// @param[in]     first       the first root this executor solves
/// @param[in]     stride      the step between the roots it solves
/// @param[in]     dd          the number of kept poles
/// @param[in]     d_prob      the dd sorted poles, any memory
/// @param[in]     z_prob      the dd weights, any memory
/// @param[in]     abs_2rho    \f$|2 \rho|\f$
/// @param[in]     max_iter    the secular iteration cap per root
/// @param[in]     do_rescale  apply the Löwner rescale to the weights
/// @param[in,out] Q           the denominators, then the eigenvectors
/// @param[in]     Q_bid       Q by (i, j), for the rescale
/// @param[out]    v           the rescaled weights, v(i, bid) or `v.batch_item(bid)`
/// @param[in]     bid         this merge's batch item in v
/// @param[in]     on_root     called once per root on pex's leader
/// @ingroup api_device_lowlevel
template <class Real, class WEx, class PEx, class DV, class QMat, class QBatch, class VView, class OnRoot>
BATCHLAS_DEVICE_INLINE void laed3(const WEx& wex, const PEx& pex, std::int32_t first, std::int32_t stride, std::int32_t dd, const DV& d_prob,
                               const DV& z_prob, Real abs_2rho, std::int32_t max_iter, bool do_rescale, const QMat& Q, QBatch& Q_bid,
                               const VView& v, std::int32_t bid, const OnRoot& on_root) {
    for (std::int32_t root_ix = first; root_ix < dd; root_ix += stride) {
        detail::SecularRoot<Real> root;
        if (root_ix == dd - 1) {
            root = detail::solve_last(pex, d_prob, z_prob, dd, abs_2rho, max_iter);
        } else {
            root = detail::solve_inner(pex, d_prob, z_prob, dd, root_ix, abs_2rho, max_iter);
        }
        if (pex.leader()) on_root(root_ix, root.origin, root.tau, root.converged);
        detail::denominators(pex, Q.col(root_ix).head(dd), d_prob, root.origin, root.tau);
    }

    sync<Scope::Group>(wex, Mem::Global);

    // Rescale reads Q, writes v; form_vectors the reverse. On the group each step syncs itself.
    detail::rescale<Real>(do_rescale, pex, first, stride, dd, Q_bid, d_prob, v, bid);
    if constexpr (!detail::is_group_v<PEx>) sync<Scope::Group>(wex, Mem::Global);
    detail::form_vectors<Real>(pex, first, stride, dd, Q, v, bid);
}

}  // namespace batchlas::device
