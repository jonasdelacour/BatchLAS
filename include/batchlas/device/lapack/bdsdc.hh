// SPDX-License-Identifier: MIT
#pragma once
// Bidiagonal SVD of a real upper bidiagonal B = (d, e) by stedc, NOT a port of LAPACK's dlasd*:
// the perfect-shuffled Golub-Kahan matrix [0 B^T; B 0] is tridiagonal with zero diagonal and
// off-diagonal (d0, e0, d1, ..., d(n-1)), eigenvalues +/- sigma. Nothing is squared; an
// eigenvector's even rows are v, its odd rows u. ONE body for every executor:
//   detail::build_gk        the Golub-Kahan tridiagonal on one item (a prefix product)
//   detail::extract_column  sigma_i, row i of V^H, column i of U (a work-group)
//   detail::repair          the vectors of the numerically zero sigma (a work-group)
//   bdsdc<tier::Blocked<1>>       all three around the in-group stedc on one WorkGroup<256, 32>;
//                                 the host route (gesvd/bdsdc/bdsdc.cc) launches each step as its
//                                 own kernel around the host stedc
// evidence: docs/design/gesvd.md#gesvd-design-why-bdsdc-goes-through-golub-kahan

#include <cstddef>
#include <limits>
#include <type_traits>

#include <sycl/sycl.hpp>

#include <batchlas/blas/enums.hh>
#include <batchlas/device.hh>
#include <batchlas/device/lapack/gesvd.hh>
#include <batchlas/device/lapack/stedc.hh>
#include <batchlas/device/tier.hh>
#include <batchlas/device/workspace.hh>

namespace batchlas::device {

namespace routine {
struct bdsdc {
    static constexpr const char* name = "bdsdc";
};
}  // namespace routine

namespace detail {

// gd := 0 (2n), ge := (d0, e0, d1, ..., d(n-1)) made non-negative (2n - 1), gs := the signs
// (2n) that undo it; one item: f(2k) = d(k) couples v(k), u(k); f(2k + 1) = e(k) u(k), v(k + 1).
template <class D, class E, class GD, class GE, class GS>
BATCHLAS_DEVICE_INLINE void build_gk(const D& d, const E& e, const GD& gd, const GE& ge, const GS& gs) {
    using T = elem_t<GD>;
    const Thread th{};
    map(th, gd, [] { return T(0); });
    interleave(th, d, e, ge);
    detail::tridiag_abs_offdiag(th, ge, gs);
}

// Eigenvector z (2n) of the Golub-Kahan matrix with eigenvalue lam(src): sig(i) := max(lam, 0),
// vrow := its even rows and ucol := its odd rows (n each), signed by gs and normalised. A half
// below 1/16 (exactly 1/sqrt(2) in theory) is zeroed, not amplified: a half rotated away comes
// back at ~1e-20, and 1/1e-20 would write +-inf, NaN every later dot product and make repair
// reject every axis. COLLECTIVE over the work-group wg.
template <class Ex, class Z, class GS, class L, class SG, class VR, class UC>
BATCHLAS_DEVICE_INLINE void extract_column(const Ex& wg, const Z& z, const GS& gs, int n, const L& lam, int src, const SG& sig, int i,
                                        const VR& vrow, const UC& ucol, bool wu, bool wv) {
    using T = std::remove_cv_t<elem_t<GS>>;
    const auto zv = slice(z, 0, n, 2), zu = slice(z, 1, n, 2);
    const auto sv = slice(gs, 0, n, 2), su = slice(gs, 1, n, 2);
    const auto [p2, q2] = sumsq<alg::Fused, Scope::Group>(wg, zv, zu);
    const T pn = sycl::sqrt(p2);
    const T qn = sycl::sqrt(q2);
    if (wg.leader()) sig(i) = sycl::fmax(static_cast<T>(lam(src)), T(0));
    const T half_present = T(0.0625);
    const T pinv = (pn > half_present) ? (T(1) / pn) : T(0);
    const T qinv = (qn > half_present) ? (T(1) / qn) : T(0);
    map_many(
        wg, n,
        [&](int k) {
            if (wv) vrow(k) = tag_mul(as_tag(pinv), Arith<T>::mul(static_cast<T>(sv(k)), static_cast<T>(zv(k))));
        },
        [&](int k) {
            if (wu) ucol(k) = tag_mul(as_tag(qinv), Arith<T>::mul(static_cast<T>(su(k)), static_cast<T>(zu(k))));
        });
}

// Rebuilds the vectors of sigma_i <= eps sigma_max / (2 kOrthTarget): below it the +/- pair is
// unresolved and U gets parallel columns (no factor of n: it would discard informative sigma;
// the values keep gebrd's eps ||A|| floor). The criterion is sigma, not the column's shape. Each
// such j seeds e_c on the axis c least covered by the final vectors (every non-degenerate one
// and the rebuilt t < j), projects them out twice, and normalises. U (n x n) and Vh (n x n) are
// read and written in global memory; sg holds the n sigma. COLLECTIVE over the work-group wg.
// evidence: docs/perf/gesvd.md#gesvd-bdsdc-null-space-repair-criteria-that-failed
// evidence: docs/perf/gesvd.md#gesvd-bdsdc-the-repair-threshold-from-the-resolution-limit
template <class Ex, class MU, class MV, class SG>
BATCHLAS_DEVICE_INLINE void repair(const Ex& wg, const MU& U, const MV& Vh, const SG& sg, int n, bool wu, bool wv) {
    using T = std::remove_cv_t<elem_t<SG>>;
    constexpr int W = Ex::size;
    constexpr T kOrthTarget = T(1e-3);
    const T tol_factor = (T(1) / (T(2) * kOrthTarget)) * std::numeric_limits<T>::epsilon();
    const int lid = wg.lid();
    T smax = T(0);
    for (int i = 0; i < n; ++i) smax = sycl::fmax(smax, static_cast<T>(sg(i)));
    const T tol = smax * tol_factor;
    int ndeg = 0;
    for (int i = 0; i < n; ++i) ndeg += (sg(i) <= tol) ? 1 : 0;
    if (ndeg == 0) return;

    // which = 0 rebuilds V (rows of Vh), which = 1 U (columns of U).
    for (int which = 0; which < 2; ++which) {
        if (which == 0 && !wv) continue;
        if (which == 1 && !wu) continue;
        for (int j = 0; j < n; ++j) {
            if (sg(j) > tol) continue;
            // Ranked against every final vector: the rebuilt t < j and ALL non-degenerate ones,
            // including those after j (skipping those loses orthogonality under ascending order).
            T best = std::numeric_limits<T>::max();
            int best_c = lid;
            for (int c = lid; c < n; c += W) {
                T acc = T(0);
                for (int t = 0; t < n; ++t) {
                    if (t == j) continue;
                    if (t > j && sg(t) <= tol) continue;
                    const T q = (which == 0) ? Vh(t, c) : U(c, t);
                    acc += q * q;
                }
                if (acc < best) {
                    best = acc;
                    best_c = c;
                }
            }
            const T best_all = wg.group_reduce(best, sycl::minimum<T>());
            const int c_star = wg.group_reduce((best == best_all) ? best_c : n, sycl::minimum<int>());

            // w := e_c*, the final vectors projected out twice (once is not enough when e_c* is
            // nearly in their span).
            for (int r = lid; r < n; r += W) {
                const T seed = (r == c_star) ? T(1) : T(0);
                if (which == 0) Vh(j, r) = seed;
                else U(r, j) = seed;
            }
            wg.barrier(Mem::Global);
            for (int pass = 0; pass < 2; ++pass) {
                for (int t = 0; t < n; ++t) {
                    if (t == j) continue;
                    if (t > j && sg(t) <= tol) continue;
                    T dot = T(0);
                    for (int r = lid; r < n; r += W) {
                        const T wr = (which == 0) ? Vh(j, r) : U(r, j);
                        const T tr = (which == 0) ? Vh(t, r) : U(r, t);
                        dot += wr * tr;
                    }
                    dot = wg.group_reduce(dot, sycl::plus<T>());
                    for (int r = lid; r < n; r += W) {
                        const T tr = (which == 0) ? Vh(t, r) : U(r, t);
                        if (which == 0) Vh(j, r) -= dot * tr;
                        else U(r, j) -= dot * tr;
                    }
                    wg.barrier(Mem::Global);
                }
            }
            T acc = T(0);
            for (int r = lid; r < n; r += W) {
                const T wr = (which == 0) ? Vh(j, r) : U(r, j);
                acc += wr * wr;
            }
            const T nrm2 = wg.group_reduce(acc, sycl::plus<T>());
            // c_star is the best axis there is: a residual too small to normalise means the final
            // vectors already span the space, so the column stays as extracted.
            if (nrm2 > std::numeric_limits<T>::min()) {
                const T inv = T(1) / sycl::sqrt(nrm2);
                for (int r = lid; r < n; r += W) {
                    if (which == 0) Vh(j, r) *= inv;
                    else U(r, j) *= inv;
                }
            }
            wg.barrier(Mem::Global);
        }
    }
}

// tier::Blocked<1>'s plan of one matrix: the Golub-Kahan (gd, ge, gs), its eigenvalues w and
// vectors Z (2n x 2n) in global memory, then the nested stedc's slices.
template <class R> struct BdsdcLayout {
    ws::Slot<R> gd, ge, gs, w, z;
    ws::Slot<std::byte> stedc_local, stedc_global;
    WorkspaceSize size;
};
template <class R> constexpr BdsdcLayout<R> plan(int n) {
    const int N = 2 * (n > 0 ? n : 1);
    const WorkspaceSize q = stedc_buffer_size<JobType::EigenVectors, tier::Blocked<1>, R>(N);
    Plan p(q.align > 16 ? q.align : 16);
    BdsdcLayout<R> L{};
    L.gd = p.take<R>(ws::Space::Global, std::size_t(N));
    L.ge = p.take<R>(ws::Space::Global, std::size_t(N - 1));
    L.gs = p.take<R>(ws::Space::Global, std::size_t(N));
    L.w = p.take<R>(ws::Space::Global, std::size_t(N));
    L.z = p.take<R>(ws::Space::Global, std::size_t(N) * std::size_t(N));
    L.stedc_local = p.take<std::byte>(ws::Space::Local, q.local_bytes);
    L.stedc_global = p.take<std::byte>(ws::Space::Global, q.global_bytes);
    L.size = p.size();
    return L;
}

}  // namespace detail

template <class T> struct tiers<routine::bdsdc, T> {
    using type = tier::list<tier::Blocked<1>>;
};
// stedc's executor (WorkGroup<256, 32>) at order 2n; real T only (B is real for every gebrd T).
template <class T> struct tier_info<routine::bdsdc, tier::Blocked<1>, T> {
    static_assert(std::is_floating_point_v<T>, "bdsdc: a real T");
    static constexpr int wg_size = detail::wg_size, sg_size = detail::sg_size;
    static constexpr bool admits(int n) { return n >= 1; }
    static constexpr WorkspaceSize buffer_size(int n) { return detail::plan<T>(n).size; }
};
template <class Tier, class T> constexpr WorkspaceSize bdsdc_buffer_size(int n) {
    return tier_info<routine::bdsdc, resolve_tier_t<routine::bdsdc, T, Tier>, T>::buffer_size(n);
}

// The SVD of one upper bidiagonal, host order (d, e, s, ws, U, Vh, sort_desc): s receives the n
// singular values (descending when sort_desc, else ascending), U (n x n or larger: its leading
// n x n block) the left vectors as columns, Vh (the same) the right vectors as rows; an operand
// smaller than n x n is not formed. d (n) and e (n - 1) are read only. COLLECTIVE over ex, a
// WorkGroup<256, 32> launched at sub-group size 32, with the same arguments; d, e visible on
// entry, outputs on return. Workspace: bdsdc_buffer_size<Tier, T>(n). Returns stedc's info: 0, or
// 1 when a secular root ran out of iterations.
template <class Tier = tier::Blocked<1>, class Ex, class D, class E, class S, class MU, class MV>
BATCHLAS_DEVICE_HELPER int bdsdc(const Ex& ex, const D& d, const E& e, const S& s, const Workspace& ws, const MU& U, const MV& Vh,
                              bool sort_desc = true) {
    using R = std::remove_cv_t<elem_t<D>>;
    static_assert(std::is_floating_point_v<R>, "bdsdc: a real T");
    static_assert(std::is_same_v<resolve_tier_t<routine::bdsdc, R, Tier>, tier::Blocked<1>>, "bdsdc: tier::Blocked<1>");
    static_assert(is_work_group_v<Ex> && Ex::size == detail::wg_size && Ex::sg_size == detail::sg_size,
                  "bdsdc: executor WorkGroup<256, 32>");
    const int n = d.size();
    if (n <= 0) return 0;
    const int N = 2 * n;
    const auto L = detail::plan<R>(n);
    const auto gd = device::vec(carve_global(ws, L.gd), N);
    const auto ge = device::vec(carve_global(ws, L.ge), N - 1);
    const auto gs = device::vec(carve_global(ws, L.gs), N);
    const auto w = device::vec(carve_global(ws, L.w), N);
    const auto Z = device::mat(carve_global(ws, L.z), N, N, N);
    if (ex.leader()) detail::build_gk(d, e, gd, ge, gs);
    ex.barrier(Mem::Global);
    const int info = stedc<JobType::EigenVectors, tier::Blocked<1>>(
        ex, gd, ge, w, detail::nested_workspace(ws, L.stedc_local, L.stedc_global), detail::stedc_params<R>{}, Z);
    ex.barrier(Mem::Global);
    const bool wu = U.rows() >= n && U.cols() >= n;
    const bool wv = Vh.rows() >= n && Vh.cols() >= n;
    if (!wu && !wv) {
        for_each_owned(ex, n, [&](int i) { s(i) = sycl::fmax(static_cast<R>(w(sort_desc ? N - 1 - i : n + i)), R(0)); });
        ex.barrier(Mem::Global);
        return info;
    }
    const auto Un = U.sub(0, 0, wu ? n : 0, wu ? n : 0);
    const auto Vn = Vh.sub(0, 0, wv ? n : 0, wv ? n : 0);
    for (int i = 0; i < n; ++i) {
        const int src = sort_desc ? N - 1 - i : n + i;
        detail::extract_column(ex, Z.col(src), gs, n, w, src, s, i, Vn.row(wv ? i : 0), Un.col(wu ? i : 0), wu, wv);
    }
    ex.barrier(Mem::Global);
    detail::repair(ex, Un, Vn, s, n, wu, wv);
    ex.barrier(Mem::Global);
    return info;
}

}  // namespace batchlas::device
