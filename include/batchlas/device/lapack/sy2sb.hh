// SPDX-License-Identifier: MIT
#pragma once
// sy2sb: a Hermitian (real: symmetric) matrix -> lower band of semibandwidth kd (LAPACK
// xSYTRD_SY2SB / xHETRD_HE2HB) inside the caller's kernel, one matrix per work-group, any n and kd.
// The host sytrd_sy2sb's algorithm on in-group ops: per panel geqr2, larft, then Q^H A Q on the
// lower triangle with gemm / symm / hemm / syr2k / her2k (device/level3/group.hh).
// Public op (v22 design C.2/C.4/C.5): sy2sb<UL, Tier>(ex, A, AB, tau, kd, ws); tier::Blocked<NB>
// (NB = 32 or 64, the Level 3 tile): A and the panel scratch in global memory, only the tile
// staging in local memory. Sizes: sy2sb_buffer_size<UL, Tier, T>(n, kd); with_tier:
// routine::sy2sb<UL, KDMax>. copy_band_col is also the host route's band copy.

#include <complex>
#include <cstddef>
#include <type_traits>

#include <sycl/sycl.hpp>

#include <batchlas/device.hh>
#include <batchlas/device/lapack/geqrf.hh>
#include <batchlas/device/lapack/larft.hh>
#include <batchlas/device/level3/group.hh>

namespace batchlas::device::inline v2 {

namespace sy2sb_detail {

template <class Tier, class T> struct cfg_of;  // only tier::Blocked<NB> is a sy2sb tier
template <int NB, class A, class T> struct cfg_of<tier::Blocked<NB, A>, T> {
    static_assert(NB == 32 || NB == 64, "sy2sb: tier::Blocked<NB> with NB = 32 or 64 (the Level 3 tile)");
    using type = l3::cfg::RegTiled<T, NB, NB, 8, NB / 16, NB / 16>;
};
template <class Tier, class T> using cfg_t = typename cfg_of<Tier, T>::type;

// One matrix's workspace (kd > 0 and n > kd; else none), sized for the first panel (pn0 x pk0),
// which bounds every later one: the tile staging (local), then (global) the panel's tau, its
// unit-lower V, P = V T, Y = A22 P, T, and the pk x kd products M1, M2.
template <class T> struct Layout {
    ws::Slot<T> smem, taup, v, p, y, tf, m1, m2;
    WorkspaceSize size;
};
template <class Tier, class T> constexpr Layout<T> panel_layout(int n, int kd) {
    Layout<T> l{};
    Plan p;
    if (kd > 0 && n > kd) {
        constexpr ws::Space g = ws::Space::Global;
        const std::size_t pn0 = static_cast<std::size_t>(n - kd);
        const std::size_t kdz = static_cast<std::size_t>(kd);
        const std::size_t pk0 = pn0 < kdz ? pn0 : kdz;
        l.smem = p.take<T>(ws::Space::Local, static_cast<std::size_t>(l3::smem_elems<cfg_t<Tier, T>>()));
        l.taup = p.take<T>(g, pk0);
        l.v = p.take<T>(g, pn0 * pk0);
        l.p = p.take<T>(g, pn0 * pk0);
        l.y = p.take<T>(g, pn0 * pk0);
        l.tf = p.take<T>(g, pk0 * pk0);
        l.m1 = p.take<T>(g, pk0 * kdz);
        l.m2 = p.take<T>(g, pk0 * kdz);
    }
    l.size = p.size();
    return l;
}

// AB(0 : lk, j) := A(j : j + lk, j), lk = min(kd, n - 1 - j) + 1: column j of the lower band.
template <class Ex, class MA, class MB> BATCHLAS_DLA_INLINE void copy_band_col(const Ex& ex, const MA& a, const MB& ab, int kd, int j) {
    const int last = a.cols() - 1 - j;
    const int lk = (kd < last ? kd : last) + 1;
    copy(ex, a.col(j).tail(j).head(lk), ab.col(j).head(lk));
}

template <class T> constexpr T minus_half() {
    if constexpr (Arith<T>::is_complex) return Arith<T>::make(real_t<T>(-0.5), real_t<T>(0));
    else return T(-0.5);
}

// A := H^H A H on the lower triangle for the panel's H = diag(I, I - W T W^H): the strip left of
// A22 (columns c0 .. c0 + sw, a short last panel only) gets the left side, C -= W (T^H (W^H C)),
// and A22 both (LAPACK xSYTRD_SY2SB): P = W T, Y = A22 P, Y -= W (T^H (W^H Y)) / 2,
// A22 -= W Y^H + Y W^H. Every Level 3 op ends on a global barrier.
template <class Cfg, class Ex, class MA, class MW, class MT>
BATCHLAS_DLA_INLINE void two_sided_lower(const Ex& ex, const MA& strip, const MA& a22, const MW& w, const MT& tf, typename Cfg::value_type* pp,
                                         typename Cfg::value_type* yp, typename Cfg::value_type* m1p, typename Cfg::value_type* m2p, l3::Smem<Cfg> sm) {
    using T = typename Cfg::value_type;
    constexpr Transpose NT = Transpose::NoTrans;
    constexpr Transpose CT = Arith<T>::is_complex ? Transpose::ConjTrans : Transpose::Trans;
    const int pn = w.rows(), pk = w.cols(), sw = strip.cols();
    if (sw > 0) {
        const auto m1 = mat(m1p, pk, sw, pk), m2 = mat(m2p, pk, sw, pk);
        gemm<CT, NT, Cfg>(ex, w, strip, m1, One{}, Zero{}, sm);
        gemm<CT, NT, Cfg>(ex, tf, m1, m2, One{}, Zero{}, sm);
        gemm<NT, NT, Cfg>(ex, w, m2, strip, NegOne{}, One{}, sm);
    }
    const auto pm = mat(pp, pn, pk, pn), ym = mat(yp, pn, pk, pn);
    const auto m1 = mat(m1p, pk, pk, pk), m2 = mat(m2p, pk, pk, pk);
    gemm<NT, NT, Cfg>(ex, w, tf, pm, One{}, Zero{}, sm);
    if constexpr (Arith<T>::is_complex) hemm<Side::Left, Uplo::Lower, Cfg>(ex, a22, pm, ym, One{}, Zero{}, sm);
    else symm<Side::Left, Uplo::Lower, Cfg>(ex, a22, pm, ym, One{}, Zero{}, sm);
    gemm<CT, NT, Cfg>(ex, w, ym, m1, One{}, Zero{}, sm);
    gemm<CT, NT, Cfg>(ex, tf, m1, m2, One{}, Zero{}, sm);
    gemm<NT, NT, Cfg>(ex, w, m2, ym, minus_half<T>(), One{}, sm);
    if constexpr (Arith<T>::is_complex) her2k<Uplo::Lower, NT, Cfg>(ex, w, ym, a22, NegOne{}, One{}, sm);
    else syr2k<Uplo::Lower, NT, Cfg>(ex, w, ym, a22, NegOne{}, One{}, sm);
}

}  // namespace sy2sb_detail

// Bytes of ONE matrix's workspace for sy2sb<UL, Tier> at (n, kd); zero when nothing is reduced
// (kd == 0 or n <= kd).
template <Uplo UL, class Tier, class T> constexpr WorkspaceSize sy2sb_buffer_size(int n, int kd) {
    static_assert(UL == Uplo::Lower, "sy2sb: Uplo::Lower only (as host sytrd_sy2sb)");
    return sy2sb_detail::panel_layout<Tier, std::remove_cv_t<T>>(n, kd).size;
}

namespace routine {
// with_tier's tag: KDMax is the band width the tier filter sizes the workspace for.
template <Uplo UL, int KDMax = 32> struct sy2sb {
    static_assert(KDMax >= 0, "routine::sy2sb: KDMax >= 0");
    static constexpr const char* name = "sy2sb";
    static constexpr int kd_max = KDMax;
    static constexpr int kd_at(int n) { return n - 1 < KDMax ? (n - 1 < 0 ? 0 : n - 1) : KDMax; }
};
}  // namespace routine

template <Uplo UL, int KDMax, int NB, class A, class T> struct tier_info<routine::sy2sb<UL, KDMax>, tier::Blocked<NB, A>, T> {
    static constexpr int wg_size = sy2sb_detail::cfg_t<tier::Blocked<NB, A>, T>::threads, sg_size = 32;
    static constexpr bool admits(int n) { return n >= 0; }
    static constexpr WorkspaceSize buffer_size(int n) { return sy2sb_buffer_size<UL, tier::Blocked<NB, A>, T>(n, routine::sy2sb<UL, KDMax>::kd_at(n)); }
};
template <Uplo UL, int KDMax, class T> struct tiers<routine::sy2sb<UL, KDMax>, T> {
    using type = tier::list<tier::Blocked<32>>;
};

// sy2sb<UL, Tier>(ex, A, AB, tau, kd, ws): the n x n Hermitian A (lower triangle read; the strict
// upper is never read or written) -> its lower band, AB(r, j) = band A(j + r, j) for r <= kd, rows
// past min(kd, n - 1 - j) zero; A's lower triangle below the band holds the reflectors and tau
// (n - kd) their scalars, as host sytrd_sy2sb. kd == 0 or n <= kd: the band is copied, tau not
// written. ex: a WorkGroup of tier_info::wg_size items in sub-groups of 32 (collective; A visible
// on entry). A, AB: a global Mat, ItemMat or l3::Global (AB: at least kd + 1 rows, n columns);
// tau: a vector operand. ws: this matrix's slice of sy2sb_buffer_size<UL, Tier, T>(n, kd).
// Returns 0, or -i for a bad i-th argument after the executor (1 A not square, 2 AB too small,
// 3 tau shorter than n - kd, 4 kd < 0), uniform; nothing is written then. Outputs visible on return.
template <Uplo UL, class Tier, class Ex, class MA, class MB, class Tau>
BATCHLAS_DLA_INLINE int sy2sb(const Ex& ex, const MA& A, const MB& AB, const Tau& tau, int kd, const Workspace& ws) {
    static_assert(UL == Uplo::Lower, "sy2sb: Uplo::Lower only (as host sytrd_sy2sb)");
    const auto ag = l3::global_of(A);
    const auto bg = l3::global_of(AB);
    using T = std::remove_cv_t<std::remove_reference_t<decltype(ag(0, 0))>>;
    using Cfg = sy2sb_detail::cfg_t<Tier, T>;
    static_assert(Ex::size == Cfg::threads && Ex::sg_size == 32, "sy2sb: ex is a WorkGroup<tier_info::wg_size, 32>");
    const int n = ag.m;
    if (kd < 0) return -4;
    if (ag.n != n) return -1;
    if (bg.m < kd + 1 || bg.n != n) return -2;
    if (tau.size() < (n > kd ? n - kd : 0)) return -3;
    if (n == 0) return 0;

    const auto a = mat(ag.p, n, n, ag.ld);
    const auto ab = mat(bg.p, kd + 1, n, bg.ld);
    laset(ex, Zero{}, Zero{}, ab);
    ex.barrier(Mem::Global);
    if (kd == 0 || n <= kd) {
        for_each_owned(ex, n, [&](int j) BATCHLAS_DLA_BODY { sy2sb_detail::copy_band_col(Thread{}, a, ab, kd, j); });
        ex.barrier(Mem::Global);
        return 0;
    }

    const auto lay = sy2sb_detail::panel_layout<Tier, T>(n, kd);
    const l3::Smem<Cfg> sm{carve_local(ws, lay.smem).get_raw()};
    T* const taup = carve_global(ws, lay.taup);
    T* const vp = carve_global(ws, lay.v);
    T* const tfp = carve_global(ws, lay.tf);
    const RtGroup<Scope::Group, 32> rex{ex.it};
    for (int i = 0; i + kd < n; i += kd) {
        const int pn = n - i - kd, pk = pn < kd ? pn : kd;
        const auto v = a.sub(i + kd, i, pn, pk);
        geqr2(rex, v, pk, taup);
        ex.barrier(Mem::Global);
        // Columns [i, i + pk) are final; the strip and A22 below write columns >= i + pk only.
        for_each_owned(ex, pk, [&](int c) BATCHLAS_DLA_BODY {
            sy2sb_detail::copy_band_col(Thread{}, a, ab, kd, i + c);
            tau(i + c) = taup[c];
        });
        const auto w = mat(vp, pn, pk, pn);
        lacpy<Part::Lower, Transpose::NoTrans, alg::Walk, edge::ZeroPad, diag_fill::Unit>(ex, v, w);
        ex.barrier(Mem::Global);
        const auto tf = mat(tfp, pk, pk, pk);
        larft(ex, w, tf, vec(taup, pk));
        ex.barrier(Mem::Global);
        sy2sb_detail::two_sided_lower<Cfg>(ex, a.sub(i + kd, i + pk, pn, kd - pk), a.sub(i + kd, i + kd, pn, pn), w, tf,
                                           carve_global(ws, lay.p), carve_global(ws, lay.y), carve_global(ws, lay.m1),
                                           carve_global(ws, lay.m2), sm);
    }
    for_each_owned(ex, kd, [&](int c) BATCHLAS_DLA_BODY { sy2sb_detail::copy_band_col(Thread{}, a, ab, kd, n - kd + c); });
    ex.barrier(Mem::Global);
    return 0;
}

}  // namespace batchlas::device::inline v2
