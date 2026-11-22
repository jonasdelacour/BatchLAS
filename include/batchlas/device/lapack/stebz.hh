// SPDX-License-Identifier: MIT
#pragma once
/// @file
/// @brief Selected eigenvalues of a real symmetric tridiagonal by bisection, `stebz` (xSTEBZ),
/// inside a kernel.
///
/// d has n entries and e(i) couples rows i and i + 1. `stebz` selects by index or by value range
/// (`StebzOptions`) and bisects one wanted eigenvalue per item on any executor, any n, in one
/// tier, `tier::Blocked<1>` (nothing of the matrix in local memory). `with_tier` checks a
/// `WorkGroup<256>` at sub-group size 32. Pair it with `stein` for the eigenvectors. The
/// Gershgorin bound, the Sturm count and the bisection step (`stebz_gershgorin`, `laebz_count`,
/// `laebz_bisect`) are in @ref api_device_lowlevel.
/// @ingroup api_device_lapack_eig
// Each keeps the legacy stebz kernel's expression shape (bit-identical ports). RANGE and ORDER
// stay run-time (StebzOptions): the host route serves every range from one kernel, and the paths
// differ by one pair of Sturm counts (design C.2's tax is nil).

#include <cstdint>
#include <limits>
#include <type_traits>

#include <batchlas/device/arith.hh>
#include <batchlas/device/exec.hh>
#include <batchlas/device/level1.hh>
#include <batchlas/device/tier.hh>
#include <batchlas/device/workspace.hh>

namespace batchlas::device {

/// @brief Run-time options of `stebz`: dstebz's selection, order and tolerance.
///
/// With `value_range` the eigenvalues in (vl, vu] are selected (LAPACK's RANGE = 'V'); otherwise
/// the indices il to iu, 0-based (RANGE = 'I', or 'A' with il = 0 and iu = n - 1). The defaults
/// select index 0 alone.
/// @tparam Real  the tridiagonal's real type
/// @ingroup api_device_lapack_eig
template <class Real> struct StebzOptions {
    /// Select by value range (vl, vu] instead of by index.
    bool value_range = false;
    /// The first and last selected index, 0-based (index selection only).
    std::int64_t il = 0, iu = 0;
    /// The value range (vl, vu], and the absolute tolerance of each eigenvalue: abstol <= 0 means
    /// eps * max(norm(T), 1).
    Real vl = Real(0), vu = Real(0), abstol = Real(0);
    /// Write the selected eigenvalues in descending order (LAPACK's ORDER).
    bool descending = false;
    /// The bisection cap per eigenvalue, >= 1 (the host StebzParams::max_iterations).
    std::int32_t max_iter = 128;
};

namespace detail {
template <class Real> struct Interval {
    Real lo, hi;
    Real e2max;  // max_i e_i^2 (dstebz's pivmin scale)
};

// The workspace plan (design C.5): the leader's interval (lo, hi, pivmin) and the selection's
// (count, first index), both in local memory; nothing global.
template <class Real> struct StebzLayout {
    WsSlot<Real> bounds;
    WsSlot<std::int32_t> found;
    WorkspaceSize size;
};
template <class Real> constexpr StebzLayout<Real> stebz_layout() {
    Plan p;
    const auto bounds = p.take<Real>(WsSpace::Local, 3);
    const auto found = p.take<std::int32_t>(WsSpace::Local, 2);
    return {bounds, found, p.size()};
}
}  // namespace detail

/// @brief The Gershgorin interval of a tridiagonal and dstebz's pivmin scale.
///
/// The interval is \f$[\min_i (d_i - r_i), \max_i (d_i + r_i)]\f$ with
/// \f$r_i = |e_{i-1}| + |e_i|\f$; three reductions over `ex` (min, then max, then max), so the
/// result is the same on every item.
/// @tparam S  the reduction scope
/// @param[in] ex  the executor
/// @param[in] d   n reals, the diagonal
/// @param[in] e   n - 1 reals, the off-diagonal
/// @return `{lo, hi, e2max}`, e2max the largest \f$e_i^2\f$
/// @ingroup api_device_lowlevel
template <Scope S = Scope::Own, class Ex, class D, class E>
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
    return detail::Interval<R>{glo, ghi, ge2};
}

/// @brief The Sturm count of a tridiagonal at x: the number of eigenvalues <= x, serial on one
/// item.
///
/// Counts the nonpositive \f$q_i = d_i - x - e_{i-1}^2 / q_{i-1}\f$; a \f$|q_i|\f$ below pivmin is
/// clamped to -pivmin (dlaebz), which keeps the count monotone in x.
/// @param[in] d       n reals, the diagonal
/// @param[in] e       n - 1 reals, the off-diagonal
/// @param[in] x       the shift
/// @param[in] pivmin  the smallest pivot magnitude allowed
/// @return the count, 0 to n
/// @ingroup api_device_lowlevel
template <class D, class E, class Real> BATCHLAS_DEVICE_INLINE int laebz_count(const Thread&, const D& d, const E& e, Real x, Real pivmin) {
    const int n = d.size();
    int count = 0;
    Real q = d(0) - x;
    if (sycl::fabs(q) < pivmin) q = -pivmin;
    if (q <= Real(0)) ++count;
    for (int i = 1; i < n; ++i) {
        const Real ei = e(i - 1);
        q = d(i) - x - (ei * ei) / q;
        if (sycl::fabs(q) < pivmin) q = -pivmin;
        if (q <= Real(0)) ++count;
    }
    return count;
}

/// @brief Eigenvalue j of a tridiagonal by bisection on the Sturm count, serial on one item.
///
/// Halves (lo, hi] until its width is at most tol, the midpoint no longer moves, or max_iter
/// steps have run.
/// @param[in] th        the calling item
/// @param[in] d         n reals, the diagonal
/// @param[in] e         n - 1 reals, the off-diagonal
/// @param[in] j         the eigenvalue's index, 0-based ascending
/// @param[in] lo        the lower end, with count(lo) <= j
/// @param[in] hi        the upper end, with j < count(hi)
/// @param[in] tol       the absolute width to stop at
/// @param[in] pivmin    the smallest pivot magnitude allowed, as for laebz_count()
/// @param[in] max_iter  the step cap
/// @return the final midpoint
/// @ingroup api_device_lowlevel
template <class D, class E, class Real>
BATCHLAS_DEVICE_INLINE Real laebz_bisect(const Thread& th, const D& d, const E& e, int j, Real lo, Real hi, Real tol, Real pivmin, int max_iter) {
    Real left = lo, right = hi;
    for (int iter = 0; iter < max_iter; ++iter) {
        if ((right - left) <= tol) break;
        const Real mid = left + (right - left) * Real(0.5);
        if (mid <= left || mid >= right) break;
        if (laebz_count(th, d, e, mid, pivmin) >= j + 1) right = mid;
        else left = mid;
    }
    return left + (right - left) * Real(0.5);
}

namespace routine {
/// @brief The routine tag of `stebz` (xSTEBZ) for `with_tier`, `launch` and `buffer_size`.
/// @ingroup api_device_lapack_eig
struct stebz {
    /// The name `with_tier` puts in its error message.
    static constexpr const char* name = "stebz";
};
}  // namespace routine

/// @brief The tiers `with_tier` tries for `stebz`: `tier::Blocked<1>` only.
/// @ingroup api_device_lapack_eig
template <class Real> struct tiers<routine::stebz, Real> {
    using type = tier::list<tier::Blocked<1>>;
};
/// @brief The one tier of `stebz`, `tier::Blocked<1>`: real data, any n >= 1, local workspace only.
///
/// `wg_size` 256 and `sg_size` 32 are what `with_tier` checks and the host route's ceiling; the
/// op itself runs on any executor width (items past the wanted count idle).
/// @ingroup api_device_lapack_eig
template <class Real> struct tier_info<routine::stebz, tier::Blocked<1>, Real> {
    static_assert(std::is_floating_point_v<Real>, "stebz: a real T");
    static constexpr int wg_size = 256, sg_size = 32;
    using executor = WorkGroup<wg_size, sg_size>;
    static constexpr int per_group = 1;
    static constexpr bool admits(int n) { return n >= 1; }
    static constexpr WorkspaceSize buffer_size(int) { return detail::stebz_layout<Real>().size; }
};

/// @brief xSTEBZ on one real symmetric tridiagonal (d, e) inside a kernel: the selected
/// eigenvalues by bisection, one item per wanted eigenvalue.
///
/// Arguments in the host order (d, e, w, m, ws, opts). COLLECTIVE over `ex`; d and e must be
/// visible to every item on entry, and every output is visible on return. The leader publishes
/// the interval and the selection through local memory, as the host route does.
/// @tparam Tier  `tier::Blocked<1>`
/// @param[in]  ex  the executor, any width
/// @param[in]  d   n reals, the diagonal
/// @param[in]  e   n - 1 reals, the off-diagonal
/// @param[out] w   the m selected eigenvalues, ascending unless `p.descending`
/// @param[out] m   the number selected, set on every item
/// @param      ws  this matrix's slice of `buffer_size<routine::stebz, Tier, T>(n)` (local only)
/// @param[in]  p   the selection, order, tolerance and iteration cap
/// @retval 0  always: the bisection is capped and never fails (xSTEBZ's info)
/// @ingroup api_device_lapack_eig
template <class Tier, class Ex, class D, class E, class W, class Real>
BATCHLAS_DEVICE_INLINE int stebz(const Ex& ex, const D& d, const E& e, const W& w, int& m, const Workspace& ws,
                                 const StebzOptions<Real>& p) {
    using T = elem_t<D>;
    static_assert(std::is_same_v<T, Real>, "stebz: options of the matrix's real type");
    static_assert(std::is_same_v<resolve_tier_t<routine::stebz, T, Tier>, tier::Blocked<1>>, "stebz: tier::Blocked<1>");
    constexpr auto L = detail::stebz_layout<T>();
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
