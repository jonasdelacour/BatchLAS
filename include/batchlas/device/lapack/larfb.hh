// SPDX-License-Identifier: MIT
#pragma once
// LAPACK xLARFB (DIRECT = Forward, STOREV = Columnwise): apply the block reflector H = I - V T V^H
// of larft to C from the left or the right, on the in-group Level 3 (level3/group.hh): two gemms
// and a trmm over the work-group's tiles, one matrix per work-group, everything in global memory
// but the L3 stage (l3::Smem). larfb_from_panel is the WY step the blocked QR ops share (geqrf's
// trailing update, every block of ormqr / orgqr): pack V from geqrf's storage, larft, larfb.

#include <cstddef>
#include <type_traits>

#include <batchlas/blas/enums.hh>
#include <batchlas/device.hh>
#include <batchlas/device/lapack/larft.hh>
#include <batchlas/device/level3/group.hh>

namespace batchlas::device {

namespace larfb_detail {
// The L3 configuration of a QR tier::Blocked<NB, Alg>: Alg itself when it is an l3::cfg preset,
// else RegTiled<T, 32, 32, 8, 2, 2> (256 items, the in-group L3 tests' tile).
template <class Alg, class T> struct cfg_of {
    using type = l3::cfg::RegTiled<T, 32, 32, 8, 2, 2>;
};
template <l3::Config Alg, class T> struct cfg_of<Alg, T> {
    static_assert(std::is_same_v<typename Alg::value_type, T>, "larfb: the tier's l3 Cfg is for another scalar type");
    using type = Alg;
};
}  // namespace larfb_detail

template <class Alg, class T> using larfb_cfg_t = typename larfb_detail::cfg_of<Alg, T>::type;

/// The QR family's default WY block width, the host blocked routes' (16 for double, else 32).
template <class T> constexpr int larfb_default_nb() { return std::is_same_v<T, double> ? 16 : 32; }

/// Elements of larfb's global work for C (m x n) and k reflectors: W1 and W2, k x n (Left) or
/// m x k (Right).
template <Side SD> constexpr std::size_t larfb_work_elems(int m, int n, int k) {
    const std::size_t q = static_cast<std::size_t>(SD == Side::Left ? (n > 0 ? n : 0) : (m > 0 ? m : 0));
    return 2 * static_cast<std::size_t>(k > 0 ? k : 0) * q;
}

/// The executor a blocked QR op runs on: a WorkGroup of Cfg::threads items in 32-wide sub-groups
/// (larft's gemv packs over 32 lanes).
template <class Cfg, class Ex> BATCHLAS_DEVICE_INLINE constexpr void larfb_check_ex() {
    static_assert(is_work_group_v<Ex>, "larfb: the executor is a WorkGroup<WG, 32>");
    static_assert(Ex::size == Cfg::threads, "larfb: the WorkGroup has the tier's l3 Cfg::threads items");
    static_assert(Ex::sg_size == 32, "larfb: sub-group size 32 (larft)");
}

/// This matrix's tau as a pointer: a T* passes through, a vector operand gives &tau(0) (unit
/// stride, as geqrf writes it).
template <class Tau> BATCHLAS_DEVICE_INLINE auto larfb_tau_base(const Tau& tau) {
    if constexpr (std::is_pointer_v<Tau>) return tau;
    else return &tau(0);
}

/// @brief ?LARFB (Forward, Columnwise) in a kernel: C := op(H) C (Left) or C op(H) (Right).
///
/// H = I - V T V^H; op(H) = H for NoTrans, H^H for ConjTrans (Trans on a real T). V (k columns,
/// m rows for Left, n for Right) has its unit diagonal and zero upper triangle STORED, as larft
/// takes it; T is k x k upper (its strict lower is not read). W1 = V^H C, W2 = op(T) W1, C -= V W2
/// (Left; Right: W1 = C V, W2 = W1 op(T), C -= W2 V^H). Collective over a WorkGroup of
/// Cfg::threads items; inputs visible on entry, C visible on return. Operands: l3::Global, a
/// global Mat or device::item.
/// @param work  larfb_work_elems<SD>(m, n, k) elements of global memory (W1 then W2, packed)
/// @param smem  the work-group's l3::Smem<Cfg>
template <Side SD, Transpose TR, class Direct, class Cfg, class Ex, class MV, class MT, class MC, class W>
    requires l3::Config<Cfg>
BATCHLAS_DEVICE_INLINE void larfb(const Ex& ex, const MV& V, const MT& Tm, const MC& C, W* work, l3::Smem<Cfg> smem) {
    static_assert(std::is_same_v<Direct, dir::Forward>, "larfb: Direct = dir::Forward (larft's only direction)");
    using T = typename Cfg::value_type;
    static_assert(std::is_same_v<std::remove_cv_t<W>, T>, "larfb: work holds the Cfg's scalar type");
    static_assert(!(Arith<T>::is_complex && TR == Transpose::Trans), "larfb: complex Trans is not H^H; use ConjTrans");
    constexpr Transpose OT = TR == Transpose::NoTrans ? Transpose::NoTrans : Transpose::ConjTrans;
    const auto v = l3::global_of(V);
    const auto c = l3::global_of(C);
    const int k = v.n, m = c.m, n = c.n;
    if (k <= 0 || m <= 0 || n <= 0) return;  // uniform: before any barrier
    if constexpr (SD == Side::Left) {
        const l3::Global<T> w1{work, k, k, n}, w2{work + static_cast<std::ptrdiff_t>(k) * n, k, k, n};
        gemm<Transpose::ConjTrans, Transpose::NoTrans, Cfg>(ex, v, c, w1, One{}, Zero{}, smem);
        trmm<Side::Left, Uplo::Upper, OT, Diag::NonUnit, Cfg>(ex, Tm, w1, w2, One{}, smem);
        gemm<Transpose::NoTrans, Transpose::NoTrans, Cfg>(ex, v, w2, c, NegOne{}, One{}, smem);
    } else {
        const l3::Global<T> w1{work, m, m, k}, w2{work + static_cast<std::ptrdiff_t>(m) * k, m, m, k};
        gemm<Transpose::NoTrans, Transpose::NoTrans, Cfg>(ex, c, v, w1, One{}, Zero{}, smem);
        trmm<Side::Right, Uplo::Upper, OT, Diag::NonUnit, Cfg>(ex, Tm, w1, w2, One{}, smem);
        gemm<Transpose::NoTrans, Transpose::ConjTrans, Cfg>(ex, w2, v, c, NegOne{}, One{}, smem);
    }
}

/// The WY step of reflectors i0 .. i0 + ib - 1 of geqrf's storage a (Q of order a.m): V :=
/// a(i0 :, i0 : i0 + ib) unit lower with zeros above (vbuf, ld a.m - i0), T := larft(V,
/// tau(i0 :)) (tbuf, ld ldt >= ib), then larfb<SD, TR> on c (a.m - i0 rows for Left, columns
/// for Right). a's diagonal and upper triangle are not read. Collective; c visible on return.
template <Side SD, Transpose TR, class Cfg, class Ex, class T>
BATCHLAS_DEVICE_INLINE void larfb_from_panel(const Ex& ex, const l3::Global<T>& a, int i0, int ib, const T* tau, const l3::Global<T>& c,
                                          T* vbuf, T* tbuf, int ldt, T* work, l3::Smem<Cfg> smem) {
    const int mv = a.m - i0;
    const auto panel = mat(a.p + i0 + static_cast<std::ptrdiff_t>(i0) * a.ld, mv, ib, a.ld);
    const auto v = mat(vbuf, mv, ib, mv);
    lacpy<Part::Lower, Transpose::NoTrans, alg::Walk, edge::None, diag_fill::Unit>(ex, panel, v);
    laset<Part::StrictUpper>(ex, Zero{}, Zero{}, v);
    ex.barrier(Mem::Global);  // V complete before larft's gemv reads its columns
    larft(ex, v, mat(tbuf, ib, ib, ldt), vec(tau + i0, ib));
    ex.barrier(Mem::Global);  // T complete before the trmm stages it
    larfb<SD, TR, dir::Forward, Cfg>(ex, l3::Global<T>{vbuf, mv, mv, ib}, l3::Global<T>{tbuf, ldt, ib, ib}, c, work, smem);
}

}  // namespace batchlas::device
