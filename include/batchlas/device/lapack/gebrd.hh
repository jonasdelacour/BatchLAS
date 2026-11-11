// SPDX-License-Identifier: MIT
#pragma once
// LAPACK xGEBRD's unblocked step xGEBD2 (bidiagonal reduction Q^H A P = B, B upper bidiagonal):
// gebd2_real / gebd2_complex on one matrix per work-item (the unblocked route), gebd2_tile on one
// problem's n x n tile in local memory (the CTA route, GebrdCTAKernel).
// detail::larfg_lanes is xLARFG on a lane-held vector whose leader forms the scalars and
// broadcasts them (core larfg's lane form replicates the scalars instead). labrd is xLABRD on one
// matrix (real d, e: ZLARFG's real beta), on core gemv_acc: the host's complex panel (labrd_panel)
// and the panel of gebrd<tier::Blocked<NB>>, whose trailing update is the in-group L3 gemm.

#include <algorithm>
#include <cstdint>

#include <sycl/sycl.hpp>

#include <batchlas/device/exec.hh>
#include <batchlas/device/lapack_aux.hh>
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
// larfg on [alpha; x] held one element per lane: lane `head` holds alpha, the x_active lanes x.
// The leader forms the scalars (alg::Phase: tau = 0, beta = alpha for len <= 1); on return lane
// `head` holds beta, the x lanes v (x * scale when tau != 0), and every lane gets tau.
template <typename Ex, typename T>
inline T larfg_lanes(const Ex& ex, int32_t len, int32_t head, T& alpha, T& x, bool x_active) {
    const T sumsq = ex.team_reduce(x_active ? (x * x) : T(0));
    T xnorm = T(0);
    if (ex.leader()) xnorm = sycl::sqrt(sumsq);
    xnorm = ex.bcast(xnorm, 0u);
    const T alpha_h = ex.bcast(alpha, static_cast<std::uint32_t>(head));
    device::LarfgScalars<T> s{};
    if (ex.leader()) s = device::larfg_scalars<device::alg::Phase>(alpha_h, xnorm, len);
    const T beta = ex.bcast(s.beta, 0u);
    const T tau = ex.bcast(s.tau, 0u);
    const T scale = ex.bcast(s.scale, 0u);
    if (ex.lid() == head) {
        alpha = beta;
    } else if (x_active && tau != T(0)) {
        x *= scale;
    }
    return tau;
}
}  // namespace detail

// dgebd2 on one matrix (one work-item): H(i) from the left, then G(i) from the right. A function,
// not the kernel lambda's body, so the lambda calls no view operator() (SemaSYCL entry quirk).
template <typename T, typename MA, typename RV, typename TV>
BATCHLAS_DEVICE_INLINE void gebd2_real(const MA& Ab, int32_t b, const RV& D, const RV& E, const TV& TAUQ, const TV& TAUP) {
    const device::Thread th{};
    const int32_t m = Ab.rows(), n = Ab.cols(), k = std::min(m, n);
    for (int32_t i = 0; i < k; ++i) {
        // H(i) annihilates A(i+1:m, i), then A(i:m, i+1:n) := H(i) A(i:m, i+1:n).
        const auto xq = Ab.col(i).tail(i + 1);
        const T sigma = device::dot(th, xq, xq);
        const auto hl = device::larfg_scalars<device::alg::Phase>(static_cast<T>(Ab(i, i)), sycl::sqrt(sigma), m - i);
        if (hl.tau != T(0)) {
            Ab(i, i) = T(1);
            device::scal(th, hl.scale, xq);
            device::larf<Side::Left>(th, hl.tau, device::unit_head(Ab.col(i).tail(i), 0), Ab.sub(i, i + 1, m - i, n - i - 1));
        }
        Ab(i, i) = hl.beta;
        D(i, b) = hl.beta;
        TAUQ(i, b) = hl.tau;

        if (i >= n - 1) {
            TAUP(i, b) = T(0);
            continue;
        }

        // G(i) annihilates A(i, i+2:n), then A(i+1:m, i+1:n) := A(i+1:m, i+1:n) G(i).
        const auto xp = Ab.row(i).tail(i + 2);
        const T sigma_r = device::dot(th, xp, xp);
        const auto hr = device::larfg_scalars<device::alg::Phase>(static_cast<T>(Ab(i, i + 1)), sycl::sqrt(sigma_r), n - (i + 1));
        if (hr.tau != T(0)) {
            Ab(i, i + 1) = T(1);
            device::scal(th, hr.scale, xp);
            device::larf<Side::Right>(th, hr.tau, device::unit_head(Ab.row(i).tail(i + 1), 0), Ab.sub(i + 1, i + 1, m - i - 1, n - i - 1));
        }
        Ab(i, i + 1) = hr.beta;
        E(i, b) = hr.beta;
        TAUP(i, b) = hr.tau;
    }

    for (int32_t i = k; i < TAUQ.size(); ++i) {
        TAUQ(i, b) = T(0);
    }
    for (int32_t i = std::max<int32_t>(0, k - 1); i < TAUP.size(); ++i) {
        TAUP(i, b) = T(0);
    }
}

// zgebd2 on one matrix: d, e real (ZLARFG's real beta); row i holds conj(u_i) on exit (unmbr 'P'
// order). H(i) is applied as H^H = I - conj(tau) v v^H, with v(0) = 1 stored.
template <typename T, typename MA, typename RV, typename TV>
BATCHLAS_DEVICE_INLINE void gebd2_complex(const MA& Ab, int32_t b, const RV& D, const RV& E, const TV& TAUQ, const TV& TAUP) {
    using Real = device::real_t<T>;
    const device::Thread th{};
    const int32_t m = Ab.rows(), n = Ab.cols();
    for (int32_t i = 0; i < n; ++i) {
        const auto xq = Ab.col(i).tail(i + 1);
        const Real sigma = device::sumsq<device::alg::Unfused>(th, xq);
        const auto lq = device::larfg_scalars<device::alg::RealBeta>(static_cast<T>(Ab(i, i)), sycl::sqrt(sigma), m - i);
        device::scal(th, lq.scale, xq);
        Ab(i, i) = T(1);
        device::larf<Side::Left>(th, device::Arith<T>::conj(lq.tau), Ab.col(i).tail(i), Ab.sub(i, i + 1, m - i, n - i - 1));
        D(i, b) = device::Arith<T>::re(lq.beta);
        Ab(i, i) = lq.beta;
        TAUQ(i, b) = lq.tau;
        if (i >= n - 1) {
            TAUP(i, b) = T(0);
            continue;
        }
        // Row i conjugated (lacgv); G(i) annihilates A(i, i+2:n) (u(0) = 1 stored); A := A G.
        const auto row = Ab.row(i).tail(i + 1);
        const auto xp = Ab.row(i).tail(i + 2);
        device::lacgv(th, row);
        const Real sigma_r = device::sumsq<device::alg::Unfused>(th, xp);
        const auto rp = device::larfg_scalars<device::alg::RealBeta>(static_cast<T>(Ab(i, i + 1)), sycl::sqrt(sigma_r), n - (i + 1));
        device::scal(th, rp.scale, xp);
        Ab(i, i + 1) = T(1);
        device::larf<Side::Right>(th, rp.tau, row, Ab.sub(i + 1, i + 1, m - i - 1, n - i - 1));
        device::lacgv(th, xp);
        E(i, b) = device::Arith<T>::re(rp.beta);
        Ab(i, i + 1) = rp.beta;
        TAUP(i, b) = rp.tau;
    }
}

// sgebd2 on one problem's n x n tile in local memory, lane l of the P-lane partition owning row l
// (column l in the left update). Every barrier is work-group wide: a dead partition runs a zero
// tile and stores nothing. A function, not the kernel lambda's body, so the lambda calls no view
// operator() (SemaSYCL entry quirk).
template <typename T, typename Ex, typename TM, typename AV, typename RV, typename TV>
BATCHLAS_DEVICE_INLINE void gebd2_tile(const Ex& ex, const sycl::nd_item<1>& it, const TM& tile, const AV& A_prob,
                                    const RV& D_prob, const RV& E_prob, const TV& TAUQ_prob, const TV& TAUP_prob,
                                    int32_t n, bool live) {
    const int32_t lane = ex.lid();
    for (int32_t c = 0; c < n; ++c) {
        if (lane < n) {
            tile(lane, c) = live ? A_prob(lane, c) : T(0);
        }
    }
    it.barrier(sycl::access::fence_space::local_space);

    for (int32_t i = 0; i < n; ++i) {
        // H(i) annihilates tile(i+1:n, i); lane j > i then applies it to column j.
        T alpha_l = T(0);
        if (lane == i) {
            alpha_l = tile(i, i);
        }
        T x_l = T(0);
        const bool x_l_active = (lane > i && lane < n);
        if (x_l_active) {
            x_l = tile(lane, i);
        }

        const T tau_l = detail::larfg_lanes(ex, n - i, i, alpha_l, x_l, x_l_active);
        if (lane == i) {
            tile(i, i) = alpha_l;
            if (live) {
                D_prob(i) = alpha_l;
                TAUQ_prob(i) = tau_l;
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
        T alpha_r = T(0);
        if (lane == i + 1) {
            alpha_r = tile(i, i + 1);
        }
        T x_r = T(0);
        const bool x_r_active = (lane > i + 1 && lane < n);
        if (x_r_active) {
            x_r = tile(i, lane);
        }

        const T tau_r = detail::larfg_lanes(ex, n - (i + 1), i + 1, alpha_r, x_r, x_r_active);
        if (lane == i + 1) {
            tile(i, i + 1) = alpha_r;
            if (live) {
                E_prob(i) = alpha_r;
                TAUP_prob(i) = tau_r;
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
        TAUP_prob(n - 1) = T(0);
    }

    for (int32_t c = 0; c < n; ++c) {
        if (live && lane < n) {
            A_prob(lane, c) = tile(lane, c);
        }
    }
}

// ---- the public op (v22 design C.4-C.7): device::gebrd<Tier> ---------------------------------
namespace routine {
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
template <class T> using GebrdBlockedCfg = l3::cfg::RegTiled<T, 32, 32, 8, 2, 2>;
inline constexpr int blocked_wg = 256;
inline constexpr int blocked_nb = 16;  // the default panel width (the host's gebrd_block_size)
template <class T> struct GebrdBlockedPlan {
    WsSlot<T> work;  // local: wa (NB), wx (NB)
    WsSlot<T> smem;  // local: l3::Smem<BlockedCfg<T>>
    WsSlot<T> x, y;  // global: X (ld m), Y (ld n)
    WorkspaceSize size;
};
template <int NB, class T> constexpr GebrdBlockedPlan<T> labrd_plan(int m, int n) {
    Plan p;
    const WsSlot<T> work = p.take<T>(WsSpace::Local, std::size_t(2) * NB);
    const WsSlot<T> smem = p.take<T>(WsSpace::Local, std::size_t(l3::smem_elems<GebrdBlockedCfg<T>>()));
    const WsSlot<T> x = p.take<T>(WsSpace::Global, std::size_t(m) * NB);
    const WsSlot<T> y = p.take<T>(WsSpace::Global, std::size_t(n) * NB);
    return {work, smem, x, y, p.size()};
}
}  // namespace detail

// tier::Cta<P>, P in {4, 8, 16, 32}: one matrix per P-lane partition (SgpPartition<P>) of a
// sub-group-32 launch, its P x P tile in local memory; the work-group size is any multiple of 32
// (32 / P matrices per sub-group). Real T only.
template <int P, class T> struct tier_info<routine::gebrd, tier::Cta<P>, T> {
    static_assert(detail::cta_width_v<P>, "gebrd: tier::Cta<P> needs P in {4, 8, 16, 32}");
    static constexpr int wg_size = 32, sg_size = 32;
    static constexpr int matrices_per_sub_group = 32 / P;
    static constexpr bool admits(int n) { return !Arith<T>::is_complex && n >= 1 && n <= P; }
    static constexpr WorkspaceSize buffer_size(int) { return detail::gebrd_cta_plan<P, T>().size; }
};
// tier::Blocked<NB> (1 <= NB <= 64, the panel width): any m >= n, every T, on a WorkGroup<256, 32>.
// buffer_size(n) is the square problem's; gebrd_buffer_size<Tier, T>(m, n) any other.
template <int NB, class A, class T> struct tier_info<routine::gebrd, tier::Blocked<NB, A>, T> {
    static_assert(NB >= 1 && NB <= 64, "gebrd: tier::Blocked<NB> needs 1 <= NB <= 64 (the host panel's bound)");
    static constexpr int wg_size = detail::blocked_wg, sg_size = 32;
    static constexpr bool admits(int n) { return n >= 1; }
    static constexpr WorkspaceSize buffer_size(int n) { return detail::labrd_plan<NB, T>(n, n).size; }
};
template <class T> struct tiers<routine::gebrd, T> {
    using type = std::conditional_t<Arith<T>::is_complex, tier::list<tier::Blocked<detail::blocked_nb>>,
                                    tier::list<tier::Cta<4>, tier::Cta<8>, tier::Cta<16>, tier::Cta<32>,
                                               tier::Blocked<detail::blocked_nb>>>;
};

// Local / global bytes of ONE matrix for gebrd<Tier> at order n (Cta: the work-group's local
// buffer holds one such slice per partition), and of an m x n matrix for tier::Blocked.
template <class Tier, class T> constexpr WorkspaceSize gebrd_buffer_size(int n) {
    return tier_info<routine::gebrd, resolve_tier_t<routine::gebrd, T, Tier>, T>::buffer_size(n);
}
template <class Tier, class T> constexpr WorkspaceSize gebrd_buffer_size(int m, int n) {
    using Tr = resolve_tier_t<routine::gebrd, T, Tier>;
    static_assert(tier::is_blocked_v<Tr>, "gebrd_buffer_size(m, n): tier::Blocked (Cta is square)");
    return detail::labrd_plan<Tr::nb, T>(m, n).size;
}

// xGEBRD of the n x n matrix A (n = A.rows() <= P) owned by this P-lane partition ex: A is
// overwritten with the reflectors (xGEBD2's layout), d / e receive the bidiagonal, tauq / taup
// the reflector scalars. ws.local is the WORK-GROUP's local buffer, one gebrd_buffer_size<Tier,
// T>(n).local_bytes slice per partition: partition k of the group (k = sub-group * 32 / P +
// partition in its sub-group) uses slice k, addressed by an int32 element offset from ws.local
// (a local pointer moved per partition costs ~8% more SASS). Collective over the work-group of
// it: the barriers are work-group wide, so every partition calls it with the same n and ws, and
// a partition without a matrix passes live = false (a zero tile, nothing read or stored).
// Returns 0 (no info).
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
    constexpr int slice = static_cast<int>(plan.size.local_bytes / sizeof(T));
    static_assert(plan.tile.offset == 0 && slice * sizeof(T) == plan.size.local_bytes, "gebrd: a slice is whole T");
    const int part = static_cast<int>(it.get_sub_group().get_group_linear_id()) * static_cast<int>(ex.part.get_group_linear_range()) +
                     static_cast<int>(ex.part.get_group_linear_id());
    const int n = A.rows();
    const Mat<T, Dyn, Dyn, Ext<P>, Mem::Local, Cyclic, local_mptr<T>> tile{carve_local(ws, plan.tile), {n}, {n}, {}, part * slice};
    gebd2_tile<T>(ex, it, tile, A, d, e, tauq, taup, n, live);
    return 0;
}

// xLABRD on panel columns j0 .. j0+ib-1 of ONE m x n matrix A (m >= n), its X (m x nb) and Y
// (n x nb) indexed by A's global rows; ex a WorkGroup whose size is a compile-time constant, so
// every Cyclic loop strides by a constant. Rows hold conj(u) on exit, d and e are real (ZLARFG's
// real beta; real T: every conj is the identity). wa / wx (nb each, local) hold the negated small
// products, so every update below is one gemv_acc: y := y0 + alpha (sum of the terms). The body of
// the host's complex panel (labrd_panel) and of gebrd<tier::Blocked>.
template <typename T, typename Ex, typename MA, typename MX, typename MY, typename RV, typename TV, typename WV>
BATCHLAS_DEVICE_INLINE void labrd(const Ex& ex, const MA& Ab, const MX& Xb, const MY& Yb, const RV& d, const RV& e, const TV& tauq,
                               const TV& taup, const WV& wa, const WV& wx, int32_t j0, int32_t ib) {
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
        const Real sigma = device::sumsq<device::alg::Unfused>(ex, xq);
        const auto hq = device::larfg_scalars<device::alg::RealBeta>(alpha_q, sycl::sqrt(sigma), mt);
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
        const Real sigma_r = device::sumsq<device::alg::Unfused>(ex, xp);
        const auto hp = device::larfg_scalars<device::alg::RealBeta>(alpha_p, sycl::sqrt(sigma_r), nt);
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

// zlabrd on panel columns j0 .. j0+ib-1 of item b of the host's batched views: labrd over the
// item's operands (A, X, Y rebased by b, d / e / tau through item(), the host view's address).
template <typename T, typename Ex, typename AV, typename RV, typename TV>
BATCHLAS_DEVICE_INLINE void labrd_panel(const Ex& ex, int32_t b, const AV& A, const AV& X, const AV& Y, const RV& D,
                                     const RV& E, const TV& TAUQ, const TV& TAUP, const sycl::local_accessor<T, 1>& work,
                                     int32_t m, int32_t n, int32_t nb, int32_t j0, int32_t ib) {
    const auto Ab = device::mat(A.data() + static_cast<int64_t>(b) * A.stride(), m, n, A.ld());
    const auto Xb = device::mat(X.data() + static_cast<int64_t>(b) * X.stride(), m, nb, X.ld());
    const auto Yb = device::mat(Y.data() + static_cast<int64_t>(b) * Y.stride(), n, nb, Y.ld());
    labrd<T>(ex, Ab, Xb, Yb, device::item(D, b), device::item(E, b), device::item(TAUQ, b), device::item(TAUP, b),
             device::vec(work, nb, 0), device::vec(work, nb, nb), j0, ib);
}

// xGEBRD of ONE m x n matrix A (m >= n, any n) on the work-group ex (WorkGroup<256, 32>), host
// order (A, d, e, tauq, taup, ws): LAPACK's blocked loop, one labrd panel of NB columns, then
// A22 -= V Y^H + X U with the in-group L3 gemm (the host route's two gemm calls), then d / e back
// on A's diagonal and superdiagonal (gebrd_blocked's restore). A is a Mat over T* in global memory
// (device::mat), d / e real vectors of n / n - 1, tauq / taup vectors of n. COLLECTIVE: every item
// calls with the same arguments, A visible on entry; outputs visible on return. Workspace:
// gebrd_buffer_size<Tier, T>(m, n), ws.global this matrix's slice. Returns 0, or -1 when m < n
// (the host's rows >= cols precondition; nothing is written).
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
    const l3::Smem<Cfg> smem{carve_local(ws, plan.smem).get_raw()};
    for (int j0 = 0; j0 < n; j0 += NB) {
        const int ib = sycl::min(NB, n - j0);
        laset(ex, Zero{}, Zero{}, Xb.sub(j0, 0, m - j0, ib));
        laset(ex, Zero{}, Zero{}, Yb.sub(j0, 0, n - j0, ib));
        ex.barrier(Mem::Global);
        labrd<T>(ex, A, Xb, Yb, d, e, tauq, taup, wa, wx, j0, ib);
        const int j2 = j0 + ib;
        if (j2 < n) {
            const auto a22 = A.sub(j2, j2, m - j2, n - j2);
            device::gemm<Transpose::NoTrans, Transpose::ConjTrans, Cfg>(ex, A.sub(j2, j0, m - j2, ib), Yb.sub(j2, 0, n - j2, ib), a22,
                                                                        NegOne{}, One{}, smem);
            device::gemm<Transpose::NoTrans, Transpose::NoTrans, Cfg>(ex, Xb.sub(j2, 0, m - j2, ib), A.sub(j0, j2, ib, n - j2), a22,
                                                                      NegOne{}, One{}, smem);
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
