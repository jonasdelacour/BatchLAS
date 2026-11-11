// SPDX-License-Identifier: MIT
#pragma once
// xSTEBZ / xLAEBZ on a symmetric tridiagonal (d, e): d has n entries, e(i) couples rows i and
// i + 1. Each keeps the legacy stebz kernel's expression shape (bit-identical ports):
//   stebz_gershgorin   the Gershgorin interval and dstebz's pivmin scale (three reductions)
//   laebz_count        the Sturm sign count, number of eigenvalues <= x
//   laebz_bisect       one eigenvalue by bisection on that count
//   stebz              the public op (design C): selected eigenvalues of one matrix on one
//                      executor, any n; one item per wanted eigenvalue
// Tier: one body for every n (no panel, nothing of the matrix in local memory): tier::Blocked<1>.
// RANGE and ORDER stay run-time (stebz_detail::Params): the host route serves every range from
// one kernel, and the paths differ by one pair of Sturm counts (design C.2's tax is nil).

#include <cstdint>
#include <limits>
#include <type_traits>

#include <batchlas/device/arith.hh>
#include <batchlas/device/exec.hh>
#include <batchlas/device/level1.hh>
#include <batchlas/device/tier.hh>
#include <batchlas/device/workspace.hh>

namespace batchlas::device {

namespace stebz_detail {
template <class R> struct Interval {
    R lo, hi;
    R e2max;  // max_i e_i^2 (dstebz's pivmin scale)
};

// dstebz's selection. value_range: RANGE = 'V', the eigenvalues in (vl, vu]; else RANGE = 'I'
// (or 'A' with il = 0, iu = n - 1), indices il..iu, 0-based. descending: ORDER = descending
// output. abstol <= 0: eps * max(||T||, 1). max_iter: the bisection cap per eigenvalue (>= 1).
template <class R> struct Params {
    bool value_range;
    std::int64_t il, iu;
    R vl, vu, abstol;
    bool descending;
    std::int32_t max_iter;
};

// The workspace plan (design C.5): the leader's interval (lo, hi, pivmin) and the selection's
// (count, first index), both in local memory; nothing global.
template <class R> struct Layout {
    ws::Slot<R> bounds;
    ws::Slot<std::int32_t> found;
    WorkspaceSize size;
};
template <class R> constexpr Layout<R> layout() {
    Plan p;
    const auto bounds = p.take<R>(ws::Space::Local, 3);
    const auto found = p.take<std::int32_t>(ws::Space::Local, 2);
    return {bounds, found, p.size()};
}
}  // namespace stebz_detail

// [min_i d_i - r_i, max_i d_i + r_i], r_i = |e_{i-1}| + |e_i|: min, then max, then max.
template <Scope S = Scope::Auto, class Ex, class D, class E>
BATCHLAS_DEVICE_INLINE auto stebz_gershgorin(const Ex& ex, const D& d, const E& e) {
    using R = elem_t<D>;
    const int n = d.size();
    R lo = std::numeric_limits<R>::max(), hi = -std::numeric_limits<R>::max(), e2 = R(0);
    for_each_owned(ex, n, [&](int i) {
        const R left = (i > 0) ? sycl::fabs(e(i - 1)) : R(0);
        const R right = (i < n - 1) ? sycl::fabs(e(i)) : R(0);
        const R radius = left + right;
        lo = sycl::min(lo, d(i) - radius);
        hi = sycl::max(hi, d(i) + radius);
        e2 = sycl::max(e2, right * right);
    });
    const R glo = allreduce<S>(ex, lo, sycl::minimum<R>());
    const R ghi = allreduce<S>(ex, hi, sycl::maximum<R>());
    const R ge2 = allreduce<S>(ex, e2, sycl::maximum<R>());
    return stebz_detail::Interval<R>{glo, ghi, ge2};
}

// Sign count of q_i = d_i - x - e_{i-1}^2 / q_{i-1}; |q| < pivmin is clamped to -pivmin
// (dlaebz), which keeps the count monotone in x.
template <class D, class E, class R> BATCHLAS_DEVICE_INLINE int laebz_count(const Thread&, const D& d, const E& e, R x, R pivmin) {
    const int n = d.size();
    int count = 0;
    R q = d(0) - x;
    if (sycl::fabs(q) < pivmin) q = -pivmin;
    if (q <= R(0)) ++count;
    for (int i = 1; i < n; ++i) {
        const R ei = e(i - 1);
        q = d(i) - x - (ei * ei) / q;
        if (sycl::fabs(q) < pivmin) q = -pivmin;
        if (q <= R(0)) ++count;
    }
    return count;
}

// Eigenvalue j (0-based, ascending) in (lo, hi] with count(lo) <= j < count(hi): halves until
// width <= tol, a midpoint that no longer moves, or max_iter; returns the final midpoint.
template <class D, class E, class R>
BATCHLAS_DEVICE_INLINE R laebz_bisect(const Thread& th, const D& d, const E& e, int j, R lo, R hi, R tol, R pivmin, int max_iter) {
    R left = lo, right = hi;
    for (int iter = 0; iter < max_iter; ++iter) {
        if ((right - left) <= tol) break;
        const R mid = left + (right - left) * R(0.5);
        if (mid <= left || mid >= right) break;
        if (laebz_count(th, d, e, mid, pivmin) >= j + 1) right = mid;
        else left = mid;
    }
    return left + (right - left) * R(0.5);
}

namespace routine {
struct stebz {
    static constexpr const char* name = "stebz";
};
}  // namespace routine

template <class T> struct tiers<routine::stebz, T> {
    using type = tier::list<tier::Blocked<1>>;
};
// wg_size is the work-group with_tier checks and the host route's ceiling; the op itself runs
// on any executor width (items past the wanted count idle). Real T only (xSTEBZ).
template <class T> struct tier_info<routine::stebz, tier::Blocked<1>, T> {
    static_assert(std::is_floating_point_v<T>, "stebz: a real T");
    static constexpr int wg_size = 256, sg_size = 32;
    static constexpr bool admits(int n) { return n >= 1; }
    static constexpr WorkspaceSize buffer_size(int) { return stebz_detail::layout<T>().size; }
};

template <class Tier, class T> constexpr WorkspaceSize stebz_buffer_size(int n) {
    return tier_info<routine::stebz, resolve_tier_t<routine::stebz, T, Tier>, T>::buffer_size(n);
}

// Selected eigenvalues of one tridiagonal (d, e) by bisection, host order (d, e, w, m, ws,
// params). COLLECTIVE over ex; d and e must be visible to every item on entry. w receives the
// m selected eigenvalues (ascending unless p.descending); m is set on every item. Returns 0
// (xSTEBZ's info: the bisection is capped, never fails). Exit: outputs visible on return.
// Workspace: stebz_buffer_size<Tier, T>(n) (local only). The interval and the selection are
// published through local memory by the leader, as the host route does.
template <class Tier, class Ex, class D, class E, class W, class R>
BATCHLAS_DEVICE_INLINE int stebz(const Ex& ex, const D& d, const E& e, const W& w, int& m, const Workspace& ws,
                              const stebz_detail::Params<R>& p) {
    using T = elem_t<D>;
    static_assert(std::is_same_v<T, R>, "stebz: params of the matrix's real type");
    static_assert(std::is_same_v<resolve_tier_t<routine::stebz, T, Tier>, tier::Blocked<1>>, "stebz: tier::Blocked<1>");
    constexpr auto L = stebz_detail::layout<T>();
    const auto bounds = carve_local(ws, L.bounds);
    const auto found = carve_local(ws, L.found);
    const Thread th{};

    const auto g = stebz_gershgorin(ex, d, e);
    if (ex.leader()) {
        const T eps = std::numeric_limits<T>::epsilon();
        const T norm = sycl::max(sycl::fabs(g.lo), sycl::fabs(g.hi));
        // Widen slightly so that count(lo) == 0 and count(hi) == n strictly, which the
        // bisection invariant relies on.
        const T pad = eps * norm * T(2) + std::numeric_limits<T>::min();
        bounds[0] = g.lo - pad;
        bounds[1] = g.hi + pad;
        bounds[2] = std::numeric_limits<T>::min() * sycl::max(T(1), g.e2max);
        found[0] = 0;
        found[1] = 0;
    }
    sync(ex, Mem::Global);

    const T glo = bounds[0];
    const T ghi = bounds[1];
    const T pivmin = bounds[2];
    const T eps = std::numeric_limits<T>::epsilon();
    const T norm = sycl::max(sycl::fabs(glo), sycl::fabs(ghi));
    const T tol = (p.abstol > T(0)) ? p.abstol : (eps * sycl::max(norm, T(1)));

    // A value range (vl, vu] is the index range [count(vl), count(vu) - 1].
    std::int64_t first = p.il;
    std::int64_t last = p.iu;
    if (p.value_range) {
        if (ex.leader()) {
            const int c_lo = laebz_count(th, d, e, p.vl, pivmin);
            const int c_hi = laebz_count(th, d, e, p.vu, pivmin);
            found[0] = c_hi - c_lo;
            found[1] = c_lo;
        }
        sync(ex, Mem::Global);
        first = static_cast<std::int64_t>(found[1]);
        last = first + static_cast<std::int64_t>(found[0]) - 1;
    } else if (ex.leader()) {
        found[0] = static_cast<std::int32_t>(last - first + 1);
    }

    // One item per wanted eigenvalue; no interaction between them.
    const int count_wanted = static_cast<int>(last - first + 1);
    for_each_owned(ex, count_wanted, [&](int slot) {
        const T lambda = laebz_bisect(th, d, e, int(first) + slot, glo, ghi, tol, pivmin, p.max_iter);
        w(p.descending ? (count_wanted - 1 - slot) : slot) = lambda;
    });

    sync(ex, Mem::Global);
    m = found[0];
    return 0;
}

}  // namespace batchlas::device
