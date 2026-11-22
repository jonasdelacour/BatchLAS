// SPDX-License-Identifier: MIT
#pragma once
/// @file
/// @brief The first stage of the two-stage tridiagonal reduction, `sy2sb` (xSYTRD_SY2SB /
/// xHETRD_HE2HB): a symmetric or Hermitian matrix to a lower band of semibandwidth kd, inside a
/// kernel.
///
/// One matrix per work-group, any n and kd, in `tier::Blocked<NB>` with NB = 32 or 64 (the level 3
/// tile): A and the panel scratch in global memory, only the tile staging in local memory. Per
/// panel: geqr2, larft, then the two-sided update with the in-group gemm, hemm and her2k. Size
/// the workspace with `buffer_size<routine::sy2sb<UL, KDMax>, Tier, T>(n, kd)`. The executor is a
/// `WorkGroup` in sub-groups of 32: the tier needs 32-wide sub-groups (its larft), which is why
/// the host syevx DirectSubset route, built on it, is refused on a device without them. The
/// second stage is `sb2st`.
/// @ingroup api_device_lapack_eig
// The host sytrd_sy2sb's algorithm on in-group ops; copy_band_col is also the host route's band
// copy.

#include <algorithm>
#include <complex>
#include <cstddef>
#include <type_traits>

#include <sycl/sycl.hpp>

#include <batchlas/device.hh>
#include <batchlas/device/lapack/geqrf.hh>
#include <batchlas/device/lapack/larft.hh>
#include <batchlas/device/level3/group.hh>

namespace batchlas::device {

namespace detail {

template <class Tier, class Scalar> struct sy2sb_cfg_of;  // only tier::Blocked<NB> is a sy2sb tier
template <int NB, class A, class Scalar> struct sy2sb_cfg_of<tier::Blocked<NB, A>, Scalar> {
    static_assert(NB == 32 || NB == 64, "sy2sb: tier::Blocked<NB> with NB = 32 or 64 (the Level 3 tile)");
    using type = l3::cfg::RegTiled<Scalar, NB, NB, 8, NB / 16, NB / 16>;
};
template <class Tier, class Scalar> using cfg_t = typename sy2sb_cfg_of<Tier, Scalar>::type;

// One matrix's workspace (kd > 0 and n > kd; else none), sized for the first panel (pn0 x pk0),
// which bounds every later one: the tile staging (local), then (global) the panel's tau, its
// unit-lower V, P = V T, Y = A22 P, T, and the pk x kd products M1, M2.
template <class Scalar> struct Sy2sbLayout {
    WsSlot<Scalar> smem, taup, v, p, y, tf, m1, m2;
    WorkspaceSize size;
};
template <class Tier, class Scalar> constexpr Sy2sbLayout<Scalar> panel_layout(int n, int kd) {
    Sy2sbLayout<Scalar> l{};
    Plan p;
    if (kd > 0 && n > kd) {
        constexpr WsSpace g = WsSpace::Global;
        const std::size_t pn0 = static_cast<std::size_t>(n - kd);
        const std::size_t kdz = static_cast<std::size_t>(kd);
        const std::size_t pk0 = pn0 < kdz ? pn0 : kdz;
        l.smem = p.take<Scalar>(WsSpace::Local, static_cast<std::size_t>(l3::smem_elems<cfg_t<Tier, Scalar>>()));
        l.taup = p.take<Scalar>(g, pk0);
        l.v = p.take<Scalar>(g, pn0 * pk0);
        l.p = p.take<Scalar>(g, pn0 * pk0);
        l.y = p.take<Scalar>(g, pn0 * pk0);
        l.tf = p.take<Scalar>(g, pk0 * pk0);
        l.m1 = p.take<Scalar>(g, pk0 * kdz);
        l.m2 = p.take<Scalar>(g, pk0 * kdz);
    }
    l.size = p.size();
    return l;
}

// The length of band column j of an n x n matrix: min(kd, n - 1 - j) + 1.
BATCHLAS_DEVICE_INLINE constexpr int band_len(int kd, int n, int j) { return std::min(kd, n - 1 - j) + 1; }
// AB(0 : lk, j) := A(j : j + lk, j), lk = band_len(kd, n, j): column j of the lower band. lk is
// an argument: computed in the host launcher's own body its SASS stays identical to before.
template <class Ex, class MA, class MB> BATCHLAS_DEVICE_INLINE void copy_band_col(const Ex& ex, const MA& a, const MB& ab, int lk, int j) {
    copy(ex, a.col(j).tail(j).head(lk), ab.col(j).head(lk));
}

template <class Scalar> constexpr Scalar minus_half() {
    if constexpr (Arith<Scalar>::is_complex) return Arith<Scalar>::make(real_t<Scalar>(-0.5), real_t<Scalar>(0));
    else return Scalar(-0.5);
}

// A := H^H A H on the lower triangle for the panel's H = diag(I, I - W T W^H): the strip left of
// A22 (columns c0 .. c0 + sw, a short last panel only) gets the left side, C -= W (T^H (W^H C)),
// and A22 both (LAPACK xSYTRD_SY2SB): P = W T, Y = A22 P, Y -= W (T^H (W^H Y)) / 2,
// A22 -= W Y^H + Y W^H. Every Level 3 op ends on a global barrier.
template <class Cfg, class Ex, class MA, class MW, class MT>
BATCHLAS_DEVICE_INLINE void two_sided_lower(const Ex& ex, const MA& strip, const MA& a22, const MW& w, const MT& tf, typename Cfg::value_type* pp,
                                         typename Cfg::value_type* yp, typename Cfg::value_type* m1p, typename Cfg::value_type* m2p, const Workspace& sm) {
    using T = typename Cfg::value_type;
    constexpr Transpose NT = Transpose::NoTrans;
    constexpr Transpose CT = Arith<T>::is_complex ? Transpose::ConjTrans : Transpose::Trans;
    const int pn = w.rows(), pk = w.cols(), sw = strip.cols();
    if (sw > 0) {
        const auto m1 = mat(m1p, pk, sw, pk), m2 = mat(m2p, pk, sw, pk);
        gemm<CT, NT, Cfg>(ex, One{}, w, strip, Zero{}, m1, sm);
        gemm<CT, NT, Cfg>(ex, One{}, tf, m1, Zero{}, m2, sm);
        gemm<NT, NT, Cfg>(ex, NegOne{}, w, m2, One{}, strip, sm);
    }
    const auto pm = mat(pp, pn, pk, pn), ym = mat(yp, pn, pk, pn);
    const auto m1 = mat(m1p, pk, pk, pk), m2 = mat(m2p, pk, pk, pk);
    gemm<NT, NT, Cfg>(ex, One{}, w, tf, Zero{}, pm, sm);
    hemm<Side::Left, Uplo::Lower, Cfg>(ex, One{}, a22, pm, Zero{}, ym, sm);
    gemm<CT, NT, Cfg>(ex, One{}, w, ym, Zero{}, m1, sm);
    gemm<CT, NT, Cfg>(ex, One{}, tf, m1, Zero{}, m2, sm);
    gemm<NT, NT, Cfg>(ex, minus_half<T>(), w, m2, One{}, ym, sm);
    her2k<Uplo::Lower, NT, Cfg>(ex, NegOne{}, w, ym, One{}, a22, sm);
}

}  // namespace detail

namespace routine {
/// @brief The routine tag of `sy2sb` for `with_tier`, `launch` and `buffer_size`.
/// @tparam UL     the triangle of A that is read (`Uplo::Lower` only)
/// @tparam KDMax  the semibandwidth `with_tier` sizes the workspace for (>= 0)
/// @ingroup api_device_lapack_eig
template <Uplo UL, int KDMax> struct sy2sb {
    static_assert(KDMax >= 0, "routine::sy2sb: KDMax >= 0");
    /// The name `with_tier` puts in its error message.
    static constexpr const char* name = "sy2sb";
    /// The semibandwidth bound, `KDMax`.
    static constexpr int kd_max = KDMax;
    /// The semibandwidth used at order n: min(KDMax, max(n - 1, 0)).
    static constexpr int kd_at(int n) { return n - 1 < KDMax ? (n - 1 < 0 ? 0 : n - 1) : KDMax; }
};
}  // namespace routine

/// @brief The `tier::Blocked<NB>` tier of `sy2sb`, NB = 32 or 64: one matrix per work-group of
/// the level 3 tile's item count, `sg_size` 32, any n.
///
/// `buffer_size(n, kd)` is one matrix's workspace at (n, kd), zero when nothing is reduced
/// (kd = 0 or n <= kd); `buffer_size(n)` is the same at the tag's `kd_at(n)`.
/// @ingroup api_device_lapack_eig
template <Uplo UL, int KDMax, int NB, class A, class Scalar> struct tier_info<routine::sy2sb<UL, KDMax>, tier::Blocked<NB, A>, Scalar> {
    static constexpr int wg_size = detail::cfg_t<tier::Blocked<NB, A>, Scalar>::threads, sg_size = 32;
    using executor = WorkGroup<wg_size, sg_size>;
    static constexpr int per_group = 1;
    static constexpr bool admits(int n) { return n >= 0; }
    static constexpr WorkspaceSize buffer_size(int n, int kd) {
        static_assert(UL == Uplo::Lower, "sy2sb: Uplo::Lower only (as host sytrd_sy2sb)");
        return detail::panel_layout<tier::Blocked<NB, A>, std::remove_cv_t<Scalar>>(n, kd).size;
    }
    static constexpr WorkspaceSize buffer_size(int n) { return buffer_size(n, routine::sy2sb<UL, KDMax>::kd_at(n)); }
};
/// @brief The tiers `with_tier` tries for `sy2sb`: `tier::Blocked<32>`.
/// @ingroup api_device_lapack_eig
template <Uplo UL, int KDMax, class Scalar> struct tiers<routine::sy2sb<UL, KDMax>, Scalar> {
    using type = tier::list<tier::Blocked<32>>;
};

/// @brief xSYTRD_SY2SB / xHETRD_HE2HB on one n x n symmetric or Hermitian matrix inside a
/// kernel: its lower band of semibandwidth kd, with the reflectors kept in A.
///
/// The lower triangle of A is read; the strict upper triangle is never read or written. On
/// return ab(r, j) = A(j + r, j) of the band for r <= kd, rows past min(kd, n - 1 - j) zero; A's
/// lower triangle below the band holds the reflectors and tau their scalars, as the host
/// sytrd_sy2sb leaves them. With kd = 0 or n <= kd the band is copied and tau is not written.
///
/// COLLECTIVE over `ex`, a `WorkGroup` of `tier_info::wg_size` items in sub-groups of 32
/// (`BATCHLAS_REQD_SG_SIZE(32)`); A must be visible on entry, and every output is visible on
/// return.
/// @tparam UL    `Uplo::Lower` (the only form)
/// @tparam Tier  `tier::Blocked<NB>`, NB = 32 or 64
/// @param[in]     ex     the work-group executor
/// @param[in,out] a_in   A, n x n in global memory (a `Mat`, `ItemMat` or `l3::Global`); the
///                       reflectors on return
/// @param[out]    ab_in  the band, at least kd + 1 rows and n columns, same operand kinds
/// @param[out]    tau    at least n - kd elements, the reflector scalars
/// @param[in]     kd     the semibandwidth, >= 0
/// @param         ws     this matrix's slices of `buffer_size<routine::sy2sb<UL, KDMax>, Tier, T>(n, kd)`
///                       (any KDMax)
/// @retval 0   success
/// @retval -1  A is not square
/// @retval -2  ab has fewer than kd + 1 rows or not n columns
/// @retval -3  tau is shorter than n - kd
/// @retval -4  kd < 0
/// @return     The value is uniform; on a negative value nothing is written.
/// @ingroup api_device_lapack_eig
template <Uplo UL, class Tier, class Ex, class MA, class MB, class Tau>
BATCHLAS_DEVICE_INLINE int sy2sb(const Ex& ex, const MA& a_in, const MB& ab_in, const Tau& tau, int kd, const Workspace& ws) {
    static_assert(UL == Uplo::Lower, "sy2sb: Uplo::Lower only (as host sytrd_sy2sb)");
    const auto ag = l3::global_of(a_in);
    const auto bg = l3::global_of(ab_in);
    using T = std::remove_cv_t<std::remove_reference_t<decltype(ag(0, 0))>>;
    using Cfg = detail::cfg_t<Tier, T>;
    static_assert(Ex::size == Cfg::threads && Ex::sg_size == 32, "sy2sb: ex is a WorkGroup<tier_info::wg_size, 32>");
    const int n = ag.m;
    if (kd < 0) return -4;
    if (ag.n != n) return -1;
    if (bg.m < kd + 1 || bg.n != n) return -2;
    if (tau.size() < (n > kd ? n - kd : 0)) return -3;
    if (n == 0) return 0;

    const auto a = mat(ag.p, n, n, ag.ld);
    const auto ab = mat(bg.p, kd + 1, n, bg.ld);
    // alg::Flat: the x86 OpenCL CPU compiler hoists Walk's wg_size / (kd + 1) to the kernel entry,
    // above the kd < 0 return (SIGFPE at kd = -1); Flat divides only inside its loop.
    laset<Part::Full, alg::Flat>(ex, Zero{}, Zero{}, ab);
    ex.barrier(Mem::Global);
    if (kd == 0 || n <= kd) {
        for_each_owned(ex, n, [&](int j) BATCHLAS_DEVICE_BODY {
            detail::copy_band_col(Thread{}, a, ab, detail::band_len(kd, n, j), j);
        });
        ex.barrier(Mem::Global);
        return 0;
    }

    const auto lay = detail::panel_layout<Tier, T>(n, kd);
    const Workspace sm = sub_workspace(ws, lay.smem);
    T* const taup = carve_global(ws, lay.taup);
    T* const vp = carve_global(ws, lay.v);
    T* const tfp = carve_global(ws, lay.tf);
    const RtGroup<Scope::Group, 32> rex{ex.it};
    for (int i = 0; i + kd < n; i += kd) {
        const int pn = n - i - kd, pk = pn < kd ? pn : kd;
        const auto v = a.sub(i + kd, i, pn, pk);
        geqr2(rex, v, vec(taup, pk));
        ex.barrier(Mem::Global);
        // Columns [i, i + pk) are final; the strip and A22 below write columns >= i + pk only.
        for_each_owned(ex, pk, [&](int c) BATCHLAS_DEVICE_BODY {
            detail::copy_band_col(Thread{}, a, ab, detail::band_len(kd, n, i + c), i + c);
            tau(i + c) = taup[c];
        });
        const auto w = mat(vp, pn, pk, pn);
        lacpy<Part::Lower, Transpose::NoTrans, alg::Walk, alg::EdgeZeroPad, DiagFill::Unit>(ex, v, w);
        ex.barrier(Mem::Global);
        const auto tf = mat(tfp, pk, pk, pk);
        larft(ex, w, vec(taup, pk), tf);
        ex.barrier(Mem::Global);
        detail::two_sided_lower<Cfg>(ex, a.sub(i + kd, i + pk, pn, kd - pk), a.sub(i + kd, i + kd, pn, pn), w, tf,
                                           carve_global(ws, lay.p), carve_global(ws, lay.y), carve_global(ws, lay.m1),
                                           carve_global(ws, lay.m2), sm);
    }
    for_each_owned(ex, kd, [&](int c) BATCHLAS_DEVICE_BODY {
        detail::copy_band_col(Thread{}, a, ab, detail::band_len(kd, n, n - kd + c), n - kd + c);
    });
    ex.barrier(Mem::Global);
    return 0;
}

}  // namespace batchlas::device
