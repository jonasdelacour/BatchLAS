// SPDX-License-Identifier: MIT
#pragma once
/// @file
/// @brief Bidiagonal reduction inside a kernel: `gebrd`, LAPACK's xGEBRD,
/// \f$Q^H A P = B\f$ with B upper bidiagonal.
///
/// `gebrd<tier::Cta<P>>(ex, it, A, d, e, tauq, taup, ws, live)` reduces one real n x n matrix
/// (n <= P) per P-lane partition of a sub-group;
/// `gebrd<tier::Blocked<NB>>(ex, A, d, e, tauq, taup, ws)` reduces one m x n matrix (m >= n,
/// every scalar type) per work-group. `ormbr` applies Q and P. The unblocked step `gebd2` (one matrix per work-item) and the panel `labrd` are in
/// @ref api_device_lowlevel. Reference: @ref api_device_lapack_qr.
/// @ingroup api_device_lapack_qr
// detail::gebd2_tile is gebd2's lane form on one problem's n x n tile in local memory (the Cta
// tier, on core larfg_lanes). labrd (real d, e: ZLARFG's real beta) is on core gemv_acc: the
// host's panel (gebrd_blocked.cc labrd_panel, every T) and the Blocked tier's panel, whose
// trailing update is the in-group L3 gemm.

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include <sycl/sycl.hpp>

#include <batchlas/device/exec.hh>
#include <batchlas/device/lapack_aux.hh>
#include <batchlas/device/sg_partition.hh>
#include <batchlas/device/interop.hh>
#include <batchlas/device/level2.hh>
#include <batchlas/device/level3/group.hh>
#include <batchlas/device/operand.hh>
#include <batchlas/device/scalar.hh>
#include <batchlas/device/tier.hh>
#include <batchlas/device/view.hh>
#include <batchlas/device/workspace.hh>

namespace batchlas::device {

namespace detail {
// ||x|| as sqrt(sumsq<Alg>) of x with each part scaled by 2^-e, e the exponent of max |x|
// (larfg_lanes' scaling), so no square under- or overflows: Metal flushes squares below 2^-126,
// and below 2^-149 every device loses them, so an x near 2^-76 formed no reflector. A power of two
// scales exactly: without under- or overflow the bits are the unscaled sum's. Collective over ex.
template <class Alg, class Ex, class X> BATCHLAS_DEVICE_INLINE auto pow2_scaled_norm(const Ex& ex, const X& x) {
    using T = elem_t<X>;
    using A = Arith<T>;
    using R = real_t<T>;
    R mag = R(0);
    for_each_owned(ex, x.size(), [&](int i) {
        const T v = x(i);
        mag = sycl::fmax(mag, sycl::fmax(sycl::fabs(A::re(v)), sycl::fabs(A::im(v))));
    });
    const R m = scope_reduce<Scope::Own>(ex, mag, sycl::maximum<R>());
    const int e = (m > R(0) && sycl::isfinite(m)) ? sycl::clamp(static_cast<int>(sycl::ilogb(m)), -120, 120) : 0;
    const R sc = sycl::ldexp(R(1), -e);
    const R ss = sumsq<Alg>(ex, mapped(x, [sc](const T& v) { return A::make(A::re(v) * sc, A::im(v) * sc); }));
    return sycl::sqrt(ss) * sycl::ldexp(R(1), e);
}
}  // namespace detail

/// @brief xGEBD2 inside a kernel: unblocked reduction of the m x n `A` (m >= n) to upper
/// bidiagonal form on one work-item, every scalar type.
///
/// Reflector \f$H_i\f$ from the left, then \f$G_i\f$ from the right, as LAPACK stores them: `A`
/// holds the reflectors' vectors on exit, row i holding \f$\overline{u_i}\f$ (`ormbr`'s P order),
/// and \f$H_i\f$ is applied as \f$H_i^H = I - \overline{\tau} v v^H\f$ (on a real type every
/// conjugation is the identity). `d` and `e` are real (ZLARFG's real beta); entries of `tauq` and
/// `taup` past the reflectors are zeroed.
/// @param[in]     ex    a `Thread`
/// @param[in,out] A     the matrix; the reflectors on exit
/// @param[out]    d     the n diagonal entries of B
/// @param[out]    e     the n - 1 superdiagonal entries of B
/// @param[out]    tauq  the left reflectors' scalars
/// @param[out]    taup  the right reflectors' scalars
/// @return 0
/// @ingroup api_device_lowlevel
// The norm is pow2_scaled_norm over sumsq's policy, chosen by T: Fused for real (dot(x, x)'s FFMA
// chain), Unfused for complex. A function, not the kernel lambda's body, so the lambda calls no
// view operator() (SemaSYCL quirk).
template <class Ex, class MA, class RV, class TV>
BATCHLAS_DEVICE_INLINE int gebd2(const Ex& ex, const MA& A, const RV& d, const RV& e, const TV& tauq, const TV& taup) {
    static_assert(is_thread_v<Ex>, "gebd2: one matrix per work-item (Thread)");
    using T = typename MA::value_type;
    using Ar = device::Arith<T>;
    using Norm = std::conditional_t<Ar::is_complex, alg::Unfused, alg::Fused>;
    const int m = A.rows(), n = A.cols(), k = std::min(m, n);
    for (int i = 0; i < k; ++i) {
        // H(i) annihilates A(i+1:m, i), then A(i:m, i+1:n) := H(i)^H A(i:m, i+1:n).
        const auto xq = A.col(i).tail(i + 1);
        const auto hl = device::larfg_scalars<device::alg::RealBeta>(static_cast<T>(A(i, i)), detail::pow2_scaled_norm<Norm>(ex, xq), m - i);
        if (hl.tau != T(0)) {
            A(i, i) = T(1);
            device::scal(ex, hl.scale, xq);
            device::larf<Side::Left>(ex, Ar::conj(hl.tau), device::unit_head(A.col(i).tail(i), 0), A.sub(i, i + 1, m - i, n - i - 1));
        }
        A(i, i) = hl.beta;
        d(i) = Ar::re(hl.beta);
        tauq(i) = hl.tau;

        if (i >= n - 1) {
            taup(i) = T(0);
            continue;
        }

        // Row i conjugated (lacgv); G(i) annihilates A(i, i+2:n), then A(i+1:m, i+1:n) := A G(i).
        const auto row = A.row(i).tail(i + 1);
        const auto xp = A.row(i).tail(i + 2);
        device::lacgv(ex, row);
        const auto hr = device::larfg_scalars<device::alg::RealBeta>(static_cast<T>(A(i, i + 1)), detail::pow2_scaled_norm<Norm>(ex, xp), n - (i + 1));
        if (hr.tau != T(0)) {
            A(i, i + 1) = T(1);
            device::scal(ex, hr.scale, xp);
            device::larf<Side::Right>(ex, hr.tau, device::unit_head(row, 0), A.sub(i + 1, i + 1, m - i - 1, n - i - 1));
        }
        device::lacgv(ex, xp);
        A(i, i + 1) = hr.beta;
        e(i) = Ar::re(hr.beta);
        taup(i) = hr.tau;
    }

    for (int i = k; i < tauq.size(); ++i) {
        tauq(i) = T(0);
    }
    for (int i = std::max(0, k - 1); i < taup.size(); ++i) {
        taup(i) = T(0);
    }
    return 0;
}

namespace detail {
// gebd2's lane form (real types): one problem's n x n tile in local memory, lane l of the P-lane
// partition owning row l (column l in the left update), each reflector from larfg_lanes. Every
// barrier is work-group wide: a dead partition runs a zero tile and stores nothing. A function,
// not the kernel lambda's body, so the lambda calls no view operator() (SemaSYCL entry quirk).
template <class Real, class Ex, class TM, class AV, class RV, class TV>
BATCHLAS_DEVICE_INLINE void gebd2_tile(const Ex& ex, const sycl::nd_item<1>& it, const TM& tile, const AV& a, const RV& d,
                                       const RV& e, const TV& tauq, const TV& taup, int32_t n, bool live) {
    const int32_t lane = ex.lid();
    for (int32_t c = 0; c < n; ++c) {
        if (lane < n) {
            tile(lane, c) = live ? a(lane, c) : Real(0);
        }
    }
    it.barrier(sycl::access::fence_space::local_space);

    for (int32_t i = 0; i < n; ++i) {
        // H(i) annihilates tile(i+1:n, i); lane j > i then applies it to column j.
        Real alpha_l = Real(0);
        if (lane == i) {
            alpha_l = tile(i, i);
        }
        Real x_l = Real(0);
        const bool x_l_active = (lane > i && lane < n);
        if (x_l_active) {
            x_l = tile(lane, i);
        }

        const Real tau_l = device::larfg_lanes(ex, n - i, i, alpha_l, x_l, x_l_active);
        if (lane == i) {
            tile(i, i) = alpha_l;
            if (live) {
                d(i) = alpha_l;
                tauq(i) = tau_l;
            }
        } else if (x_l_active) {
            tile(lane, i) = x_l;
        }
        it.barrier(sycl::access::fence_space::local_space);

        if (i < n - 1 && lane > i && lane < n) {
            device::larf<Side::Left>(device::Thread{}, tau_l, device::unit_head(tile.col(i).tail(i), 0), tile.sub(i, lane, n - i, 1));
        }
        it.barrier(sycl::access::fence_space::local_space);

        if (i >= n - 1) continue;

        // G(i) annihilates tile(i, i+2:n); lane r > i then applies it to row r.
        Real alpha_r = Real(0);
        if (lane == i + 1) {
            alpha_r = tile(i, i + 1);
        }
        Real x_r = Real(0);
        const bool x_r_active = (lane > i + 1 && lane < n);
        if (x_r_active) {
            x_r = tile(i, lane);
        }

        const Real tau_r = device::larfg_lanes(ex, n - (i + 1), i + 1, alpha_r, x_r, x_r_active);
        if (lane == i + 1) {
            tile(i, i + 1) = alpha_r;
            if (live) {
                e(i) = alpha_r;
                taup(i) = tau_r;
            }
        } else if (x_r_active) {
            tile(i, lane) = x_r;
        }
        it.barrier(sycl::access::fence_space::local_space);

        if (lane > i && lane < n) {
            device::larf<Side::Right>(device::Thread{}, tau_r, device::unit_head(tile.row(i).tail(i + 1), 0), tile.sub(lane, i + 1, 1, n - i - 1));
        }
        it.barrier(sycl::access::fence_space::local_space);
    }

    if (live && lane == n - 1) {
        taup(n - 1) = Real(0);
    }

    for (int32_t c = 0; c < n; ++c) {
        if (live && lane < n) {
            a(lane, c) = tile(lane, c);
        }
    }
}
}  // namespace detail

// ---- the public op: device::gebrd<Tier> ------------------------------------------------------
namespace routine {
/// @brief The routine tag of `gebrd`, for `with_tier`, `launch` and `buffer_size`.
/// @ingroup api_device_lapack_qr
struct gebrd {
    static constexpr const char* name = "gebrd";
};
}  // namespace routine

namespace detail {
template <class T> struct GebrdCtaPlan {
    WsSlot<T> tile;  // the matrix's P x P tile, ld = P
    WorkspaceSize size;
};
template <int P, class T> constexpr GebrdCtaPlan<T> gebrd_cta_plan() {
    Plan p;
    const WsSlot<T> tile = p.take<T>(WsSpace::Local, std::size_t(P) * std::size_t(P));
    return {tile, p.size()};
}
template <int P> inline constexpr bool cta_width_v = P == 4 || P == 8 || P == 16 || P == 32;

// tier::Blocked<NB>: one matrix per WorkGroup<256>; the trailing update's L3 configuration (its
// threads fix the work-group), labrd's two NB-long products and the trailing gemm's tiles in local
// memory, X (m x NB) and Y (n x NB) in the caller's global slice.
template <class Scalar> using GebrdBlockedCfg = l3::cfg::RegTiled<Scalar, 32, 32, 8, 2, 2>;
inline constexpr int blocked_wg = 256;
inline constexpr int blocked_nb = 16;  // the default panel width (the host's gebrd_block_size)
template <class Scalar> struct GebrdBlockedPlan {
    WsSlot<Scalar> work;  // local: wa (NB), wx (NB)
    WsSlot<Scalar> smem;  // local: the L3 stage of GebrdBlockedCfg<Scalar>
    WsSlot<Scalar> x, y;  // global: X (ld m), Y (ld n)
    WorkspaceSize size;
};
template <int NB, class Scalar> constexpr GebrdBlockedPlan<Scalar> labrd_plan(int m, int n) {
    Plan p;
    const WsSlot<Scalar> work = p.take<Scalar>(WsSpace::Local, std::size_t(2) * NB);
    const WsSlot<Scalar> smem = p.take<Scalar>(WsSpace::Local, std::size_t(l3::smem_elems<GebrdBlockedCfg<Scalar>>()));
    const WsSlot<Scalar> x = p.take<Scalar>(WsSpace::Global, std::size_t(m) * NB);
    const WsSlot<Scalar> y = p.take<Scalar>(WsSpace::Global, std::size_t(n) * NB);
    return {work, smem, x, y, p.size()};
}
}  // namespace detail

/// @brief `gebrd` in `tier::Cta<P>`, P one of 4, 8, 16 and 32: one real n x n matrix (n <= P) per
/// P-lane partition.
///
/// The executor is `SgpPartition<P>` of a launch at sub-group size 32
/// (`BATCHLAS_REQD_SG_SIZE(sg_size)`), with `32 / P` matrices per sub-group
/// (`matrices_per_sub_group`); the work-group size is any multiple of 32 (`wg_size` is 32). Each
/// matrix's P x P tile is in local memory, `buffer_size(n)`. Real scalar types only.
/// @ingroup api_device_lapack_qr
template <int P, class Scalar> struct tier_info<routine::gebrd, tier::Cta<P>, Scalar> {
    static_assert(detail::cta_width_v<P>, "gebrd: tier::Cta<P> needs P in {4, 8, 16, 32}");
    static constexpr int wg_size = 32, sg_size = 32;
    static constexpr int matrices_per_sub_group = 32 / P;
    using executor = SgpPartition<P>;
    static constexpr int per_group = wg_size / P;
    static constexpr bool admits(int n) { return !Arith<Scalar>::is_complex && n >= 1 && n <= P; }
    static constexpr WorkspaceSize buffer_size(int) { return detail::gebrd_cta_plan<P, Scalar>().size; }
};
/// @brief `gebrd` in `tier::Blocked<NB>` (1 <= NB <= 64, the panel width): any m x n with
/// m >= n, every scalar type, one matrix per `WorkGroup<256, 32>`.
///
/// The kernel must run at sub-group size 32 (`BATCHLAS_REQD_SG_SIZE(sg_size)`).
/// `buffer_size(n)` is the square problem's local and global bytes, `buffer_size(m, n)` any other
/// (the Cta tier has no (m, n) form).
/// @ingroup api_device_lapack_qr
template <int NB, class A, class Scalar> struct tier_info<routine::gebrd, tier::Blocked<NB, A>, Scalar> {
    static_assert(NB >= 1 && NB <= 64, "gebrd: tier::Blocked<NB> needs 1 <= NB <= 64 (the host panel's bound)");
    static constexpr int wg_size = detail::blocked_wg, sg_size = 32;
    using executor = WorkGroup<wg_size, sg_size>;
    static constexpr int per_group = 1;
    static constexpr bool admits(int n) { return n >= 1; }
    static constexpr WorkspaceSize buffer_size(int n) { return detail::labrd_plan<NB, Scalar>(n, n).size; }
    static constexpr WorkspaceSize buffer_size(int m, int n) { return detail::labrd_plan<NB, Scalar>(m, n).size; }
};
/// @brief The tiers `with_tier` tries for `gebrd`, in order: `tier::Cta<4>`, `tier::Cta<8>`,
/// `tier::Cta<16>`, `tier::Cta<32>`, then `tier::Blocked<16>`; a complex type gets only
/// `tier::Blocked<16>`.
/// @ingroup api_device_lapack_qr
template <class Scalar> struct tiers<routine::gebrd, Scalar> {
    using type = std::conditional_t<Arith<Scalar>::is_complex, tier::list<tier::Blocked<detail::blocked_nb>>,
                                    tier::list<tier::Cta<4>, tier::Cta<8>, tier::Cta<16>, tier::Cta<32>,
                                               tier::Blocked<detail::blocked_nb>>>;
};

/// @brief xGEBRD inside a kernel, `tier::Cta<P>`: the real n x n `A` (n = A.rows() <= P) of this
/// P-lane partition to upper bidiagonal form.
///
/// `A` is overwritten with the reflectors (xGEBD2's layout), `d` and `e` receive the bidiagonal,
/// `tauq` and `taup` the reflector scalars. Collective over the work-group of `it`: the barriers
/// are work-group wide, so every partition calls it with the same n, and a partition without a
/// matrix passes `live = false` (a zero tile; nothing is read or stored). The outputs are stored
/// by the lanes that own them, after the last barrier: barrier before another item reads them.
/// @pre The kernel runs at sub-group size 32 (`BATCHLAS_REQD_SG_SIZE(32)`).
/// @tparam Tier  `tier::Cta<P>`, P one of 4, 8, 16 and 32
/// @param[in]     ex    this matrix's `SgpPartition<P>`
/// @param[in]     it    the kernel's `nd_item`, for the work-group barriers
/// @param[in,out] A     this matrix; the reflectors on return
/// @param[out]    d     the n diagonal entries of B
/// @param[out]    e     the n - 1 superdiagonal entries of B
/// @param[out]    tauq  the n left reflector scalars
/// @param[out]    taup  the n right reflector scalars (the last is 0)
/// @param[in]     ws    this matrix's slice, `launch<routine::gebrd, tier::Cta<P>, T>::workspace`
/// @param[in]     live  false for a partition with no matrix
/// @return 0 (there is no failure mode)
/// @ingroup api_device_lapack_qr
template <class Tier, class Ex, class MA, class RV, class TV>
BATCHLAS_DEVICE_INLINE int gebrd(const Ex& ex, const sycl::nd_item<1>& it, const MA& A, const RV& d, const RV& e, const TV& tauq,
                              const TV& taup, const Workspace& ws, bool live = true) {
    using T = std::remove_cv_t<std::remove_reference_t<decltype(A(0, 0))>>;
    using Tr = resolve_tier_t<routine::gebrd, T, Tier>;
    static_assert(HasTier<routine::gebrd, Tr, T>, "gebrd: tier not implemented (tier::Cta<4|8|16|32>)");
    constexpr int P = Tr::n_max;
    static_assert(!Arith<T>::is_complex, "gebrd: tier::Cta is real only");
    static_assert(Ex::size == P, "gebrd: tier::Cta<P> runs on a P-lane partition");
    constexpr auto plan = detail::gebrd_cta_plan<P, T>();
    const int n = A.rows();
    const Mat<T, Dyn, Dyn, Ext<P>, Mem::Local, Cyclic, local_mptr<T>> tile{carve_local(ws, plan.tile), {n}, {n}, {}, 0};
    detail::gebd2_tile<T>(ex, it, tile, A, d, e, tauq, taup, n, live);
    return 0;
}

/// @brief xLABRD inside a kernel: reduce panel columns j0 to j0 + ib - 1 of one m x n matrix
/// (m >= n), the panel of `gebrd<tier::Blocked<NB>>`.
///
/// `Xb` (m x nb) and `Yb` (n x nb) are indexed by the global rows of `Ab`; rows hold
/// \f$\overline{u}\f$ on exit; `d` and `e` are real (ZLARFG's real beta). `wa` and `wx` (nb each,
/// local memory) are scratch. `ex` is a `WorkGroup` whose size is a compile-time constant.
/// Collective; outputs visible on return except `taup(n - 1)`, which the leader zeroes after the
/// last barrier.
/// @ingroup api_device_lowlevel
// wa / wx hold the negated small products, so every update below is one gemv_acc: y := y0 + alpha
// (sum of the terms). The body of the host's panel (gebrd_blocked.cc labrd_panel, every T) and of
// gebrd<tier::Blocked>.
template <class Ex, class MA, class MX, class MY, class RV, class TV, class WV>
BATCHLAS_DEVICE_INLINE void labrd(const Ex& ex, const MA& Ab, const MX& Xb, const MY& Yb, const RV& d, const RV& e, const TV& tauq,
                               const TV& taup, const WV& wa, const WV& wx, int j0, int ib) {
    using T = typename MA::value_type;
    using Real = device::real_t<T>;
    constexpr auto C = Transpose::ConjTrans;
    const int32_t m = Ab.rows(), n = Ab.cols();
    const auto pub = [](auto x) { return device::assume_published(x); };  // written before the last barrier
    // -0 everywhere: as gemv_acc's y0 the add folds (-0 + x == x; +0 + -0 would be +0).
    const auto zero = device::fn_vec(n, [](int) { return device::Arith<T>::make(-Real(0), -Real(0)); });

    for (int32_t ii = 0; ii < ib; ++ii) {
        const int32_t gi = j0 + ii;
        const int32_t mt = m - gi, nt = n - gi - 1;  // rows from gi, columns after gi

        // A(gi:m, gi) -= A(gi:m, j0:gi) Y(gi, 0:ii)^H + X(gi:m, 0:ii) A(j0:gi, gi)
        const auto acol = Ab.col(gi).tail(gi);
        device::gemv_acc(ex, device::NegOne{}, acol, acol, device::term(Ab.sub(gi, j0, mt, ii), pub(device::conj_view(Yb.row(gi).head(ii)))),
                     device::term(Xb.sub(gi, 0, mt, ii), pub(Ab.col(gi).tail(j0).head(ii))));
        ex.barrier(device::Mem::Global);

        // H(gi) annihilates A(gi+1:m, gi); v overwrites it, A(gi, gi) = 1. Every item reads alpha
        // before the reduction and forms the same scalars; the leader overwrites it after.
        const auto xq = Ab.col(gi).tail(gi + 1);
        const T alpha_q = Ab(gi, gi);
        const Real xnorm_q = detail::pow2_scaled_norm<device::alg::Unfused>(ex, xq);
        const auto hq = device::larfg_scalars<device::alg::RealBeta>(alpha_q, xnorm_q, mt);
        if (ex.leader()) {
            d(gi) = device::Arith<T>::re(hq.beta);
            tauq(gi) = hq.tau;
            Ab(gi, gi) = T(1);
        }
        if (hq.tau != T(0)) device::scal(ex, hq.scale, xq);
        ex.barrier(device::Mem::Global);

        if (gi >= n - 1) {
            continue;
        }

        // Y(gi+1:n, ii) = tau_q (A(gi:m, gi+1:n)^H v - Y(gi+1:n, 0:ii) A(gi:m, j0:gi)^H v
        //                        - A(j0:gi, gi+1:n)^H X(gi:m, 0:ii)^H v)
        const auto v = pub(Ab.col(gi).tail(gi));
        device::gemv<C, device::alg::GroupReducePerOutput>(ex, device::NegOne{}, Ab.sub(gi, j0, mt, ii), v, device::Zero{}, wa);
        device::gemv<C, device::alg::GroupReducePerOutput>(ex, device::NegOne{}, Xb.sub(gi, 0, mt, ii), v, device::Zero{}, wx);
        ex.barrier(device::Mem::Local);
        device::gemv_acc(ex, hq.tau, zero, Yb.col(ii).tail(gi + 1), device::term(device::op_view<C>(Ab.sub(gi, gi + 1, mt, nt)), v),
                     device::term(Yb.sub(gi + 1, 0, nt, ii), pub(wa)), device::term(device::op_view<C>(Ab.sub(j0, gi + 1, ii, nt)), pub(wx)));
        ex.barrier(device::Mem::Global);

        // Row gi, conjugated (zlabrd's lacgv before the right reflector):
        // A(gi, gi+1:n) = conj(A(gi, gi+1:n)) - Y(gi+1:n, 0:ii+1) conj(A(gi, j0:gi+1)) - A(j0:gi, gi+1:n)^H conj(X(gi, 0:ii))
        const auto arow = Ab.row(gi).tail(gi + 1);
        device::gemv_acc(ex, device::NegOne{}, device::conj_view(arow), arow,
                     device::term(Yb.sub(gi + 1, 0, nt, ii + 1), pub(device::conj_view(Ab.row(gi).tail(j0).head(ii + 1)))),
                     device::term(device::op_view<C>(Ab.sub(j0, gi + 1, ii, nt)), pub(device::conj_view(Xb.row(gi).head(ii)))));
        ex.barrier(device::Mem::Global);

        // G(gi) annihilates A(gi, gi+2:n); u overwrites it, A(gi, gi+1) = 1.
        const auto xp = Ab.row(gi).tail(gi + 2);
        const T alpha_p = Ab(gi, gi + 1);
        const Real xnorm_p = detail::pow2_scaled_norm<device::alg::Unfused>(ex, xp);
        const auto hp = device::larfg_scalars<device::alg::RealBeta>(alpha_p, xnorm_p, nt);
        if (ex.leader()) {
            e(gi) = device::Arith<T>::re(hp.beta);
            taup(gi) = hp.tau;
            Ab(gi, gi + 1) = T(1);
        }
        if (hp.tau != T(0)) device::scal(ex, hp.scale, xp);
        ex.barrier(device::Mem::Global);

        // X(gi+1:m, ii) = tau_p (A(gi+1:m, gi+1:n) u - A(gi+1:m, j0:gi+1) Y(gi+1:n, 0:ii+1)^H u
        //                        - X(gi+1:m, 0:ii) A(j0:gi, gi+1:n) u)
        const auto u = pub(Ab.row(gi).tail(gi + 1));
        device::gemv<C, device::alg::GroupReducePerOutput>(ex, device::NegOne{}, Yb.sub(gi + 1, 0, nt, ii + 1), u, device::Zero{}, wa);
        device::gemv<Transpose::NoTrans>(ex, device::NegOne{}, Ab.sub(j0, gi + 1, ii, nt), u, device::Zero{}, wx);
        ex.barrier(device::Mem::Local);
        device::gemv_acc(ex, hp.tau, zero, Xb.col(ii).tail(gi + 1), device::term(Ab.sub(gi + 1, gi + 1, mt - 1, nt), u),
                     device::term(Ab.sub(gi + 1, j0, mt - 1, ii + 1), pub(wa)), device::term(Xb.sub(gi + 1, 0, mt - 1, ii), pub(wx)));
        ex.barrier(device::Mem::Global);
        device::lacgv(ex, Ab.row(gi).tail(gi + 1));
        ex.barrier(device::Mem::Global);
    }

    if (ex.leader() && n > 0) {
        taup(n - 1) = T(0);
    }
}

/// @brief xGEBRD inside a kernel, `tier::Blocked<NB>`: one m x n `A` (m >= n, any n) to upper
/// bidiagonal form on one work-group.
///
/// LAPACK's blocked loop: a `labrd` panel of NB columns, then
/// \f$A_{22} \leftarrow A_{22} - V Y^H - X U\f$ with the in-group gemm, then d and e written back
/// on A's diagonal and superdiagonal (as the host route does). On return `A` holds the reflectors
/// in xGEBRD's layout with B on its diagonal and superdiagonal. Arguments in the host order
/// (A, d, e, tauq, taup, ws). Collective: every item calls it with the same arguments, `A`
/// visible on entry, outputs visible on return.
/// @pre The kernel runs at sub-group size 32 (`BATCHLAS_REQD_SG_SIZE(32)`).
/// @tparam Tier  `tier::Blocked<NB>`, 1 <= NB <= 64
/// @tparam Ex    `WorkGroup<256, 32>`
/// @param[in]     ex    the executor
/// @param[in,out] A     a `Mat` over global memory (`device::mat`)
/// @param[out]    d     the n diagonal entries of B, real
/// @param[out]    e     the n - 1 superdiagonal entries of B, real
/// @param[out]    tauq  the n left reflector scalars
/// @param[out]    taup  the n right reflector scalars
/// @param[in]     ws    `ws.local` and this matrix's `ws.global` slice of
///                      `device::buffer_size<routine::gebrd, Tier, T>(m, n)`
/// @return `info`, the same on every item
/// @retval 0   success
/// @retval -1  m < n (nothing written)
/// @ingroup api_device_lapack_qr
template <class Tier, class Ex, class MA, class RV, class TV>
    requires tier::is_blocked_v<resolve_tier_t<routine::gebrd, typename MA::value_type, Tier>>
BATCHLAS_DEVICE_INLINE int gebrd(const Ex& ex, const MA& A, const RV& d, const RV& e, const TV& tauq, const TV& taup, const Workspace& ws) {
    using T = typename MA::value_type;
    using Tr = resolve_tier_t<routine::gebrd, T, Tier>;
    using Cfg = detail::GebrdBlockedCfg<T>;
    constexpr int NB = Tr::nb;
    static_assert(HasTier<routine::gebrd, Tr, T>, "gebrd: tier::Blocked<NB> needs 1 <= NB <= 64");
    static_assert(is_work_group_v<Ex> && Ex::size == Cfg::threads, "gebrd: tier::Blocked runs on a WorkGroup<256, 32>");
    const int m = A.rows(), n = A.cols();
    if (m < n) return -1;
    const auto plan = detail::labrd_plan<NB, T>(m, n);
    const auto Xb = device::mat(carve_global(ws, plan.x), m, NB, m);
    const auto Yb = device::mat(carve_global(ws, plan.y), n, NB, n);
    const local_mptr<T> work = carve_local(ws, plan.work);
    const Vec<T, Dyn, Unit, Mem::Local, Cyclic, local_mptr<T>> wa{work, Dyn{NB}}, wx{work, Dyn{NB}, Unit{}, NB};
    const Workspace smem = sub_workspace(ws, plan.smem);
    for (int j0 = 0; j0 < n; j0 += NB) {
        const int ib = sycl::min(NB, n - j0);
        laset(ex, Zero{}, Zero{}, Xb.sub(j0, 0, m - j0, ib));
        laset(ex, Zero{}, Zero{}, Yb.sub(j0, 0, n - j0, ib));
        ex.barrier(Mem::Global);
        labrd(ex, A, Xb, Yb, d, e, tauq, taup, wa, wx, j0, ib);
        const int j2 = j0 + ib;
        if (j2 < n) {
            const auto a22 = A.sub(j2, j2, m - j2, n - j2);
            device::gemm<Transpose::NoTrans, Transpose::ConjTrans, Cfg>(ex, NegOne{}, A.sub(j2, j0, m - j2, ib), Yb.sub(j2, 0, n - j2, ib),
                                                                        One{}, a22, smem);
            device::gemm<Transpose::NoTrans, Transpose::NoTrans, Cfg>(ex, NegOne{}, Xb.sub(j2, 0, m - j2, ib), A.sub(j0, j2, ib, n - j2),
                                                                      One{}, a22, smem);
        }
    }
    for_each_owned(ex, n, [&](int i) {
        A(i, i) = static_cast<T>(d(i));
        if (i < n - 1) A(i, i + 1) = static_cast<T>(e(i));
    });
    ex.barrier(Mem::Global);
    return 0;
}

}  // namespace batchlas::device
