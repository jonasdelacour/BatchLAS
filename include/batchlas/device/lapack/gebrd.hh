// SPDX-License-Identifier: MIT
#pragma once
// LAPACK xGEBRD's unblocked step xGEBD2 (bidiagonal reduction Q^H A P = B, B upper bidiagonal):
// gebd2_real / gebd2_complex on one matrix per work-item (the unblocked route), gebd2_tile on one
// problem's n x n tile in local memory (the CTA route, GebrdCTAKernel).
// gebd2_detail::larfg_lanes is xLARFG on a lane-held vector whose leader forms the scalars and
// broadcasts them (core larfg's lane form replicates the scalars instead). labrd_panel is the
// complex xLABRD panel of the blocked route (real d, e: ZLARFG's real beta), on core gemv_acc.

#include <algorithm>
#include <cstdint>

#include <sycl/sycl.hpp>

#include <batchlas/device/exec.hh>
#include <batchlas/device/lapack_aux.hh>
#include <batchlas/device/level2.hh>
#include <batchlas/device/operand.hh>
#include <batchlas/device/scalar.hh>
#include <batchlas/device/view.hh>

namespace batchlas::device::inline v2 {

namespace gebd2_detail {
// larfg on [alpha; x] held one element per lane: lane `head` holds alpha, the x_active lanes x.
// The leader forms the scalars (conv::Phase: tau = 0, beta = alpha for len <= 1); on return lane
// `head` holds beta, the x lanes v (x * scale when tau != 0), and every lane gets tau.
template <typename Ex, typename T>
inline T larfg_lanes(const Ex& ex, int32_t len, int32_t head, T& alpha, T& x, bool x_active) {
    const T sumsq = ex.team_reduce(x_active ? (x * x) : T(0));
    T xnorm = T(0);
    if (ex.leader()) xnorm = sycl::sqrt(sumsq);
    xnorm = ex.bcast(xnorm, 0u);
    const T alpha_h = ex.bcast(alpha, static_cast<std::uint32_t>(head));
    device::LarfgScalars<T> s{};
    if (ex.leader()) s = device::larfg_scalars<device::conv::Phase>(alpha_h, xnorm, len);
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
}  // namespace gebd2_detail

// dgebd2 on one matrix (one work-item): H(i) from the left, then G(i) from the right. A function,
// not the kernel lambda's body, so the lambda calls no view operator() (SemaSYCL entry quirk).
template <typename T, typename MA, typename RV, typename TV>
BATCHLAS_DLA_INLINE void gebd2_real(const MA& Ab, int32_t b, const RV& D, const RV& E, const TV& TAUQ, const TV& TAUP) {
    const device::Thread th{};
    const int32_t m = Ab.rows(), n = Ab.cols(), k = std::min(m, n);
    for (int32_t i = 0; i < k; ++i) {
        // H(i) annihilates A(i+1:m, i), then A(i:m, i+1:n) := H(i) A(i:m, i+1:n).
        const auto xq = Ab.col(i).tail(i + 1);
        const T sigma = device::dot(th, xq, xq);
        const auto hl = device::larfg_scalars<device::conv::Phase>(static_cast<T>(Ab(i, i)), sycl::sqrt(sigma), m - i);
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
        const auto hr = device::larfg_scalars<device::conv::Phase>(static_cast<T>(Ab(i, i + 1)), sycl::sqrt(sigma_r), n - (i + 1));
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
BATCHLAS_DLA_INLINE void gebd2_complex(const MA& Ab, int32_t b, const RV& D, const RV& E, const TV& TAUQ, const TV& TAUP) {
    using Real = device::real_t<T>;
    const device::Thread th{};
    const int32_t m = Ab.rows(), n = Ab.cols();
    for (int32_t i = 0; i < n; ++i) {
        const auto xq = Ab.col(i).tail(i + 1);
        const Real sigma = device::sumsq<device::alg::Unfused>(th, xq);
        const auto lq = device::larfg_scalars<device::conv::RealBeta>(static_cast<T>(Ab(i, i)), sycl::sqrt(sigma), m - i);
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
        const auto rp = device::larfg_scalars<device::conv::RealBeta>(static_cast<T>(Ab(i, i + 1)), sycl::sqrt(sigma_r), n - (i + 1));
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
BATCHLAS_DLA_INLINE void gebd2_tile(const Ex& ex, const sycl::nd_item<1>& it, const TM& tile, const AV& A_prob,
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

        const T tau_l = gebd2_detail::larfg_lanes(ex, n - i, i, alpha_l, x_l, x_l_active);
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

        const T tau_r = gebd2_detail::larfg_lanes(ex, n - (i + 1), i + 1, alpha_r, x_r, x_r_active);
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

// zlabrd on panel columns j0 .. j0+ib-1 of item b (one work-group; ex a WorkGroup whose size is a
// compile-time constant, so every Cyclic loop strides by a constant). Rows hold conj(u) on exit,
// d and e are real (ZLARFG's real beta). wa / wx hold the negated small products, so every
// update below is one gemv_acc: y := y0 + alpha (sum of the terms). A function, not the kernel
// lambda's body, so the lambda calls no view operator() (SemaSYCL entry quirk).
template <typename T, typename Ex, typename AV, typename RV, typename TV>
BATCHLAS_DLA_INLINE void labrd_panel(const Ex& ex, int32_t b, const AV& A, const AV& X, const AV& Y, const RV& D,
                                        const RV& E, const TV& TAUQ, const TV& TAUP, const sycl::local_accessor<T, 1>& work,
                                        int32_t m, int32_t n, int32_t nb, int32_t j0, int32_t ib) {
    using Real = device::real_t<T>;
    constexpr auto C = Transpose::ConjTrans;
    const auto Ab = device::mat(A.data() + static_cast<int64_t>(b) * A.stride(), m, n, A.ld());
    const auto Xb = device::mat(X.data() + static_cast<int64_t>(b) * X.stride(), m, nb, X.ld());
    const auto Yb = device::mat(Y.data() + static_cast<int64_t>(b) * Y.stride(), n, nb, Y.ld());
    const auto wa = device::vec(work, nb, 0);   // -A^H v, then -Y^H u
    const auto wx = device::vec(work, nb, nb);  // -X^H v, then -A u
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
        const auto hq = device::larfg_scalars<device::conv::RealBeta>(alpha_q, sycl::sqrt(sigma), mt);
        if (ex.leader()) {
            D(gi, b) = device::Arith<T>::re(hq.beta);
            TAUQ(gi, b) = hq.tau;
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
        const auto hp = device::larfg_scalars<device::conv::RealBeta>(alpha_p, sycl::sqrt(sigma_r), nt);
        if (ex.leader()) {
            E(gi, b) = device::Arith<T>::re(hp.beta);
            TAUP(gi, b) = hp.tau;
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
        TAUP(n - 1, b) = T(0);
    }
}

}  // namespace batchlas::device
