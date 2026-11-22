// SPDX-License-Identifier: MIT
#pragma once
/// @file
/// @brief Applying a block reflector inside a kernel: `larfb`, LAPACK's xLARFB (forward,
/// columnwise).
///
/// `larfb<SD, TR, Direct::Forward, Cfg>(ex, v, t, c, ws)` applies \f$H = I - V T V^H\f$ from
/// `larft` to C from the left or the right with two gemms and a trmm of the in-group Level 3
/// (`level3/group.hh`), one matrix per work-group, everything in global memory but the Level 3
/// stage. `larfb_work_elems` sizes its global work. The WY step the blocked QR routines share,
/// `larfb_from_panel`, and the configuration helpers are in @ref api_device_lowlevel. Reference:
/// @ref api_device_lapack_qr.
/// @ingroup api_device_lapack_qr

#include <cstddef>
#include <type_traits>

#include <batchlas/blas/enums.hh>
#include <batchlas/device.hh>
#include <batchlas/device/lapack/larft.hh>
#include <batchlas/device/level3/group.hh>

namespace batchlas::device {

namespace detail {
// The L3 configuration of a QR tier::Blocked<NB, Alg>: Alg itself when it is an l3::cfg preset,
// else RegTiled<Scalar, 32, 32, 8, 2, 2> (256 items, the in-group L3 tests' tile).
template <class Alg, class Scalar> struct larfb_cfg_of {
    using type = l3::cfg::RegTiled<Scalar, 32, 32, 8, 2, 2>;
};
template <l3::Config Alg, class Scalar> struct larfb_cfg_of<Alg, Scalar> {
    static_assert(std::is_same_v<typename Alg::value_type, Scalar>,
                  "larfb: the tier's l3 Cfg is for another scalar type");
    using type = Alg;
};
}  // namespace detail

/// @brief The in-group Level 3 configuration of a QR `tier::Blocked<NB, Alg>`: `Alg` when it is
/// an `l3::cfg` preset for `Scalar`, else `l3::cfg::RegTiled<Scalar, 32, 32, 8, 2, 2>` (256 items).
/// @ingroup api_device_lowlevel
template <class Alg, class Scalar> using larfb_cfg_t = typename detail::larfb_cfg_of<Alg, Scalar>::type;

/// @brief The QR routines' default block width NB, the host blocked routes' (16 for double, else
/// 32).
/// @ingroup api_device_lowlevel
template <class Scalar> constexpr int larfb_default_nb() { return std::is_same_v<Scalar, double> ? 16 : 32; }

/// @brief Elements of `larfb`'s global work for C (m x n) and k reflectors.
///
/// The work is W1 and W2, each k x n (`Side::Left`) or m x k (`Side::Right`); negative sizes count
/// as 0.
/// @tparam SD  the side `larfb` applies H from
/// @param m  rows of C
/// @param n  columns of C
/// @param k  the number of reflectors (columns of V)
/// @return the element count, of `Cfg::value_type`
/// @ingroup api_device_lapack_qr
template <Side SD> constexpr std::size_t larfb_work_elems(int m, int n, int k) {
    const std::size_t q = static_cast<std::size_t>(SD == Side::Left ? (n > 0 ? n : 0) : (m > 0 ? m : 0));
    return 2 * static_cast<std::size_t>(k > 0 ? k : 0) * q;
}

namespace detail {
// The executor a blocked QR op runs on: a WorkGroup of Cfg::threads items in 32-wide sub-groups
// (larft's gemv packs over 32 lanes).
template <class Cfg, class Ex> BATCHLAS_DEVICE_INLINE constexpr void larfb_check_ex() {
    static_assert(is_work_group_v<Ex>, "larfb: the executor is a WorkGroup<WG, 32>");
    static_assert(Ex::size == Cfg::threads, "larfb: the WorkGroup has the tier's l3 Cfg::threads items");
    static_assert(Ex::sg_size == 32, "larfb: sub-group size 32 (larft)");
}
}  // namespace detail

/// @brief xLARFB (Direct = Forward, Store = Columnwise) inside a kernel:
/// \f$C \leftarrow \mathrm{op}(H)\,C\f$ (`Side::Left`) or \f$C \leftarrow C\,\mathrm{op}(H)\f$
/// (`Side::Right`), with \f$H = I - V T V^H\f$.
///
/// op(H) is H for `Transpose::NoTrans` and \f$H^H\f$ for `Transpose::ConjTrans` (`Trans` on a
/// real type only). `v` has k columns and m rows (Left) or n rows (Right), with its unit diagonal
/// and the zeros above it **stored**; `t` is the k x k upper triangular factor from `larft` (its
/// strict lower triangle is not read). Left forms \f$W_1 = V^H C\f$,
/// \f$W_2 = \mathrm{op}(T) W_1\f$, \f$C \leftarrow C - V W_2\f$; Right forms \f$W_1 = C V\f$,
/// \f$W_2 = W_1 \mathrm{op}(T)\f$, \f$C \leftarrow C - W_2 V^H\f$. k = 0 or an empty C returns
/// at once.
///
/// Collective over a `WorkGroup<Cfg::threads, 32>`, every item with the same arguments; inputs
/// visible to every item on entry, `c` visible on return.
/// @pre The kernel runs at sub-group size 32 (`BATCHLAS_REQD_SG_SIZE(32)`).
/// @tparam SD   `Side::Left` or `Side::Right`
/// @tparam TR   `Transpose::NoTrans`, `Transpose::ConjTrans`, or `Transpose::Trans` for a real type
/// @tparam DR   `Direct::Forward`, the only direction
/// @tparam Cfg  the in-group Level 3 configuration (an `l3::cfg` preset); its `threads` is the
///              work-group size and its `value_type` the scalar type
/// @param[in]     ex  the executor
/// @param[in]     v   the reflector vectors: `l3::Global`, a `Mat` over global memory, or
///                    `device::item`
/// @param[in]     t   the k x k triangular factor, the same operand kinds
/// @param[in,out] c   the matrix C, the same operand kinds
/// @param[in]     ws  `ws.local`: the Level 3 stage, `l3::smem_bytes<Cfg>()`; `ws.global`: W1
///                    then W2, packed, `larfb_work_elems<SD>(m, n, k)` elements of
///                    `Cfg::value_type`
/// @ingroup api_device_lapack_qr
template <Side SD, Transpose TR, Direct DR, class Cfg, class Ex, class MV, class MT, class MC>
    requires l3::Config<Cfg>
BATCHLAS_DEVICE_INLINE void larfb(const Ex& ex, const MV& v, const MT& t, const MC& c, const Workspace& ws) {
    static_assert(DR == Direct::Forward, "larfb: Direct::Forward (larft's only direction)");
    using T = typename Cfg::value_type;
    T* const work = carve_global(ws, WsSlot<T>{WsSpace::Global, 0, 0});
    static_assert(!(Arith<T>::is_complex && TR == Transpose::Trans), "larfb: complex Trans is not H^H; use ConjTrans");
    constexpr Transpose OT = TR == Transpose::NoTrans ? Transpose::NoTrans : Transpose::ConjTrans;
    const auto vg = l3::global_of(v);
    const auto cg = l3::global_of(c);
    const int k = vg.n, m = cg.m, n = cg.n;
    if (k <= 0 || m <= 0 || n <= 0) return;  // uniform: before any barrier
    if constexpr (SD == Side::Left) {
        const l3::Global<T> w1{work, k, k, n}, w2{work + static_cast<std::ptrdiff_t>(k) * n, k, k, n};
        gemm<Transpose::ConjTrans, Transpose::NoTrans, Cfg>(ex, One{}, vg, cg, Zero{}, w1, ws);
        trmm_out<Side::Left, Uplo::Upper, OT, Diag::NonUnit, Cfg>(ex, One{}, t, w1, w2, ws);
        gemm<Transpose::NoTrans, Transpose::NoTrans, Cfg>(ex, NegOne{}, vg, w2, One{}, cg, ws);
    } else {
        const l3::Global<T> w1{work, m, m, k}, w2{work + static_cast<std::ptrdiff_t>(m) * k, m, m, k};
        gemm<Transpose::NoTrans, Transpose::NoTrans, Cfg>(ex, One{}, cg, vg, Zero{}, w1, ws);
        trmm_out<Side::Right, Uplo::Upper, OT, Diag::NonUnit, Cfg>(ex, One{}, t, w1, w2, ws);
        gemm<Transpose::NoTrans, Transpose::ConjTrans, Cfg>(ex, NegOne{}, w2, vg, One{}, cg, ws);
    }
}

/// @brief The WY step the blocked QR routines share: apply reflectors i0 to i0 + ib - 1 of
/// `geqrf`'s storage `a` to `c` as one block reflector.
///
/// Packs V from `a(i0 :, i0 : i0 + ib)` as unit lower with zeros above into `vbuf` (leading
/// dimension `a.m - i0`), forms T with `larft` into `tbuf` (leading dimension `ldt >= ib`), then
/// runs `larfb<SD, TR, Direct::Forward, Cfg>` on `c` (`a.m - i0` rows for Left, columns for Right)
/// with `ws`. The diagonal and upper triangle of `a` are not read. Collective over the
/// executor `larfb` takes; `c` visible on return.
/// @ingroup api_device_lowlevel
template <Side SD, Transpose TR, class Cfg, class Ex, class Scalar>
BATCHLAS_DEVICE_INLINE void larfb_from_panel(const Ex& ex, const l3::Global<Scalar>& a, int i0, int ib, const Scalar* tau, const l3::Global<Scalar>& c,
                                          Scalar* vbuf, Scalar* tbuf, int ldt, const Workspace& ws) {
    const int mv = a.m - i0;
    const auto panel = mat(a.p + i0 + static_cast<std::ptrdiff_t>(i0) * a.ld, mv, ib, a.ld);
    const auto v = mat(vbuf, mv, ib, mv);
    lacpy<Part::Lower, Transpose::NoTrans, alg::Walk, alg::EdgeNone, DiagFill::Unit>(ex, panel, v);
    laset<Part::StrictUpper>(ex, Zero{}, Zero{}, v);
    ex.barrier(Mem::Global);  // V complete before larft's gemv reads its columns
    larft(ex, v, vec(tau + i0, ib), mat(tbuf, ib, ib, ldt));
    ex.barrier(Mem::Global);  // T complete before the trmm stages it
    larfb<SD, TR, Direct::Forward, Cfg>(ex, l3::Global<Scalar>{vbuf, mv, mv, ib}, l3::Global<Scalar>{tbuf, ldt, ib, ib},
                                        c, ws);
}

}  // namespace batchlas::device
