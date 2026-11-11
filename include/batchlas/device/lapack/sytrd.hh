// SPDX-License-Identifier: MIT
#pragma once
// xSYTRD / xHETRD: sytd2 (the unblocked reduction on a resident tile, one row per item),
// sytrd<UL, tier::Cta<N>> (one matrix per SgpPartition<N>, the tile in local memory) and
// sytrd<Lower, tier::Blocked<NB>> (one matrix per WorkGroup<256>, any n: xLATRD panels on A in
// global memory, the trailing update on the in-group her2k). Strict port: arith::Native and the
// replaced kernels' expression shapes. detail holds the CTA body (sytrd_cta_run, the host
// sytrd_cta kernel's too), the xLATRD lower panel (latrd_lower_run, the host latrd kernel's too)
// and the forms core does not reproduce bit for bit yet: Fit dotc / gemv / her2, lacpy_herm.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <batchlas/device.hh>
#include <batchlas/device/level3/group.hh>
#include <batchlas/device/sg_partition.hh>
#include <batchlas/util/group-collectives.hh>

namespace batchlas::device {

namespace detail {

// x1 conj(y1) + x2 conj(y2): one term of a rank-2 Hermitian update. Real: one expression, so
// it contracts to fma(x1, y1, x2 * y2) as the hand kernel's did.
template <class Ops, class T> BATCHLAS_DEVICE_INLINE T her2_term(const T& x1, const T& y1, const T& x2, const T& y2) {
    using Ar = arith_t<Ops, T>;
    if constexpr (Ar::is_complex) return Ar::add(Ar::mul(x1, Ar::conj(y1)), Ar::mul(x2, Ar::conj(y2)));
    else return x1 * y1 + x2 * y2;
}

// x^H y, one product per item (no accumulator), then the chunk reduction.
template <class Ops, class Ex, class X, class Y> BATCHLAS_DEVICE_INLINE auto dotc(const Fit<Ex>& ex, const X& x, const Y& y) {
    using T = elem_t<X>;
    using Ar = arith_t<Ops, T>;
    const int i = ex.lid();
    const T p = (i >= 0 && i < x.size()) ? Ar::mul(Ar::conj(x(i)), y(i)) : Ar::zero();
    return detail::scope_reduce<Scope::Auto>(ex, p);
}

// y(r) := (A x)(r) * alpha with item r computing output row r (core gemv's RowOwner without
// the loop); x is read across owners. alpha multiplies on the right, as `y *= tau` did.
template <class Ops, class Ex, class T, class MA, class X, class Y>
BATCHLAS_DEVICE_INLINE void gemv_rows(const Fit<Ex>& ex, const T& alpha, const MA& a, const Published<X>& xp, const Y& y) {
    using Ar = arith_t<Ops, T>;
    const auto& x = xp.v;
    const int m = a.rows(), n = a.cols(), r = ex.lid();
    if (r >= 0 && r < m) {
        T acc = Ar::zero();
        for (int c = 0; c < n; ++c) Ar::mac(acc, a(r, c), x(c));
        y(r) = Ar::mul(acc, alpha);
    }
}

// A := A - (x y^H + y x^H) on the FULL square A (both triangles), one row per item.
template <class Ops, class Ex, class X, class Y, class MA>
BATCHLAS_DEVICE_INLINE void her2_full_sub(const Fit<Ex>& ex, const Published<X>& xp, const Published<Y>& yp, const MA& a) {
    using T = typename MA::value_type;
    using Ar = arith_t<Ops, T>;
    const auto& x = xp.v;
    const auto& y = yp.v;
    const int m = a.rows(), n = a.cols(), r = ex.lid();
    if (r >= 0 && r < m) {
        const T xr = x(r), yr = y(r);
        for (int c = 0; c < n; ++c) {
            const T xc = x(c), yc = y(c);
            a(r, c) = Ar::sub(static_cast<T>(a(r, c)), her2_term<Ops>(xr, yc, yr, xc));
        }
    }
}

// tile := the Hermitian matrix whose UL triangle is a (n x n, n = a.rows()), Im(diag) dropped,
// zero past n up to the tile's extent. Item r reads row r of the triangle (coalesced) and writes
// it and its conjugate mirror, so a barrier must precede reads of other rows. active == false
// (a masked-off batch item) zero-fills the tile's row and reads nothing.
template <Uplo UL, class Ops, class Ex, class MA, class MT>
BATCHLAS_DEVICE_INLINE void lacpy_herm(const Fit<Ex>& ex, const MA& a, const MT& tile, bool active) {
    using T = typename MT::value_type;
    using Ar = arith_t<Ops, T>;
    const int n = a.rows(), r = ex.lid();
    if (r < 0 || r >= tile.rows()) return;
    if (active && r < n) {
        for (int c = 0; c < n; ++c) {
            if (UL == Uplo::Upper ? r > c : r < c) continue;
            T v = a(r, c);
            if (r == c) {
                if constexpr (Ar::is_complex) v = T(Ar::re(v), real_t<T>(0));
            } else {
                tile(c, r) = Ar::conj(v);
            }
            tile(r, c) = v;
        }
        for (int c = n; c < tile.cols(); ++c) tile(r, c) = Ar::zero();
    } else {
        for (int c = 0; c < tile.cols(); ++c) tile(r, c) = Ar::zero();
    }
}

// B := A (m x n), item r copying row r (core lacpy owns elements 2-D; this is the row-owner
// form the hand kernels wrote, one guarded row per item).
template <class Ex, class MA, class MB> BATCHLAS_DEVICE_INLINE void lacpy_rows(const Fit<Ex>& ex, const MA& a, const MB& b) {
    const int n = a.cols();
    for_each_owned(ex, a.rows(), [&](int r) {
        for (int c = 0; c < n; ++c) b(r, c) = a(r, c);
    });
}

// One sytd2 step on the active block blk: [alpha; x] = col one element per item (alpha on item
// ar), row is col's conjugate mirror. Reflector; mirrored write-back; then, unless tau == 0,
// v := reflector, w := A v tau, w += (-tau/2)(v^H w) v, A -= v w^H + w v^H.
template <Uplo UL, class Ops, class Ex, class Blk, class Col, class Row, class V, class W, class T, class F>
BATCHLAS_DEVICE_INLINE void sytd2_step(const Ex& ex, int i, int m, int ar, const Blk& blk, const Col& col, const Row& row,
                                    const V& v, const W& w, bool x_active, T& tau_lane, F& on_step) {
    using Ar = arith_t<Ops, T>;
    const Fit<Ex> fx{ex};
    const int lane = ex.lid();
    const auto vm = v.head(m);
    const auto wm = w.head(m);
    const bool is_alpha = lane == ar;
    T alpha = Ar::zero(), x = Ar::zero();
    if (lane < m) {
        const T e = col(lane);
        if (is_alpha) alpha = e;
        else x = e;
    }
    const T taui = larfg<conv::Phase, Ops>(ex, m, ar, alpha, x, x_active);
    if (x_active) {  // both triangles: the block update reads the full block
        col(lane) = x;
        row(lane) = Ar::conj(x);
    }
    if (is_alpha) {
        col(ar) = alpha;
        row(ar) = Ar::conj(alpha);
        if constexpr (UL == Uplo::Upper) tau_lane = taui;
        on_step(i, alpha, taui);
    }
    if (taui != Ar::zero()) {
        copy(fx, unit_head<refl::QR, unit_fill::Keep>(col, ar), vm);
        sync<scope::Part>(ex);
        if (is_alpha) {  // LAPACK's temporary unit; the block ops read v, not the column
            col(ar) = Ar::one();
            row(ar) = Ar::one();
        }
        sync<scope::Part>(ex);
        detail::gemv_rows<Ops>(fx, taui, blk, assume_published(vm), wm);
        sync<scope::Part>(ex);
        const T alpha2 = T(-0.5) * taui * detail::dotc<Ops>(fx, vm, wm);
        axpy<Ops>(fx, Scalar<T>{alpha2}, vm, wm);
        sync<scope::Part>(ex);
        detail::her2_full_sub<Ops>(fx, assume_published(vm), assume_published(wm), blk);
        sync<scope::Part>(ex);
        if (is_alpha) {
            col(ar) = alpha;
            row(ar) = Ar::conj(alpha);
        }
        sync<scope::Part>(ex);
    } else if constexpr (UL == Uplo::Upper) {
        if (is_alpha) row(ar) = Ar::conj(alpha);
    }
}

}  // namespace detail

// xSYTD2 / xHETD2 on the n x n leading block of a resident tile held in FULL storage (both
// triangles, real diagonal), one row per item (n <= Ex::size); v and w are length >= n scratch.
// Leaves LAPACK's packing, every element's conjugate mirror kept:
//   d(i) = A(i,i); Upper: e(i) = A(i,i+1), reflector i in A(0:i-1, i+1) (unit at row i);
//                  Lower: e(i) = A(i+1,i), reflector i in A(i+2:n-1, i) (unit at row i+1).
// on_step(i, beta, tau) runs on the item holding reflector i's alpha right after it is formed.
// Upper returns tau(lid) on item lid (0 past n-2); Lower returns 0 (tau goes through on_step).
// Step k touches only its active block, so column k is final once step k has run. COLLECTIVE.
template <Uplo UL, class Ops = arith::Native, class Ex, class MA, class V, class W, class F = NoOp>
BATCHLAS_DEVICE_INLINE typename MA::value_type sytd2(const Ex& ex, const MA& a, const V& v, const W& w, int n, F&& on_step = {}) {
    using T = typename MA::value_type;
    const int lane = ex.lid();
    T tau_lane = arith_t<Ops, T>::zero();
    if constexpr (UL == Uplo::Upper) {
        for (int k = n - 1; k >= 1; --k) {  // annihilate A(0:k-2, k); alpha = A(k-1, k)
            detail::sytd2_step<UL, Ops>(ex, k - 1, k, k - 1, a.sub(0, 0, k, k), a.col(k).head(k), a.row(k).head(k), v, w,
                                              lane < k - 1, tau_lane, on_step);
            sync<scope::Part>(ex);
        }
    } else {
        for (int i = 0; i <= n - 2; ++i) {  // annihilate A(i+2:n-1, i); alpha = A(i+1, i) on item 0
            const int s = i + 1, m = n - s;
            detail::sytd2_step<UL, Ops>(ex, i, m, 0, a.sub(s, s, m, m), a.col(i).tail(s).head(m), a.row(i).tail(s).head(m),
                                              v, w, lane >= 1 && lane < m, tau_lane, on_step);
        }
    }
    return tau_lane;
}

namespace detail {

// Local memory as the decorated pointer the Upper tile views take (never &acc[off]: generic).
template <class T> BATCHLAS_DEVICE_INLINE typename local_mptr<T>::pointer decorated_ptr(const sycl::local_accessor<T, 1>& a) {
    return a.template get_multi_ptr<sycl::access::decorated::yes>().get();
}
template <class T> BATCHLAS_DEVICE_INLINE T* decorated_ptr(T* p) { return p; }

// sytd2<Upper> on the full Hermitian tile (ld LDA, zero past n) at A_local[base_a]; returns
// tau(lane), 0 for lane >= n-1. The inliner declines this body for complex T, so the call
// carries decorated local pointers by value: an accessor taken by reference is copied to the
// stack (cf/cd Upper STACK 0->344).
template <typename T, int32_t LDA, size_t P, bool M>
inline T sytd2_upper_tile(const SubGroupPartition<P, M>& part, typename local_mptr<T>::pointer A_local, int32_t base_a,
                          typename local_mptr<T>::pointer V_local, int32_t base_v, typename local_mptr<T>::pointer W_local,
                          int32_t base_w, int32_t n) {
    using LP = typename local_mptr<T>::pointer;
    using Tile = Mat<T, Dyn, Dyn, Ext<LDA>, Mem::Local, Cyclic, LP>;
    using Col = Vec<T, Dyn, Unit, Mem::Local, Cyclic, LP>;
    return sytd2<Uplo::Upper>(wrap(part), Tile{A_local, Dyn{n}, Dyn{n}, {}, base_a}, Col{V_local, Dyn{n}, {}, base_v},
                              Col{W_local, Dyn{n}, {}, base_w}, n);
}

// The CTA tier's one body (P x P tile, ld P): the Hermitian tile from A's UL triangle, sytd2,
// d / e / tau out, the reduced tile back to A. Buf is the local memory's base (a
// sycl::local_accessor<T, 1>, or a decorated pointer) and base_* the element offsets of this
// problem's tile, v and w. live == false (a masked-off item) reduces a zero tile and stores
// nothing. The host sytrd_cta kernel and device::sytrd<UL, tier::Cta<P>> both run it.
template <Uplo UL, class T, size_t P, bool M, class Buf, class MA, class DV, class EV, class TV>
BATCHLAS_DEVICE_INLINE void sytrd_cta_run(const SubGroupPartition<P, M>& part, const Buf& A_local, int32_t base_a,
                                       const Buf& V_local, int32_t base_v, const Buf& W_local, int32_t base_w,
                                       const MA& A, DV D_prob, EV E_prob, TV TAU_prob, int32_t n, bool live) {
    constexpr int kP = static_cast<int>(P);
    const auto ex = wrap(part);
    const int32_t lane = static_cast<int32_t>(part.get_local_linear_id());
    // Views with the local base itself: the uniform base keeps [R.X4+UR] addressing.
    using Tile = Mat<T, Ext<kP>, Ext<kP>, Ext<kP>, Mem::Local, Cyclic, Buf>;
    using Col = Vec<T, Dyn, Unit, Mem::Local, Cyclic, Buf>;
    const Tile tile{A_local, {}, {}, {}, base_a};

    // Item r loads row r of the named triangle (coalesced); Im(diag) dropped; zero past n.
    lacpy_herm<UL, arith::Native>(Fit{ex}, A, tile, live);
    group_barrier(part);

    if constexpr (UL == Uplo::Upper) {
        const T tau_lane = sytd2_upper_tile<T, kP>(part, decorated_ptr(A_local), base_a, decorated_ptr(V_local), base_v,
                                                   decorated_ptr(W_local), base_w, n);
        // The tridiagonal straight off the reduced tile.
        if (live && lane < n) {
            D_prob(lane) = A_local[base_a + lane + lane * kP];
        }
        if (live && lane < (n - 1)) {
            E_prob(lane) = A_local[base_a + lane + (lane + 1) * kP];
            TAU_prob(lane) = tau_lane;
        }
    } else {
        // d(i), e(i), tau(i) leave as reflector i is formed (d(i) is final by then). live by
        // value: a flag the nested lambda takes by reference can land in local memory.
        sytd2<Uplo::Lower>(ex, tile, Col{V_local, Dyn{n}, Unit{}, base_v}, Col{W_local, Dyn{n}, Unit{}, base_w}, n,
                           [&, live](int32_t i, const T& beta, const T& taui) {
                               if (live) {
                                   E_prob(i) = beta;
                                   TAU_prob(i) = taui;
                                   D_prob(i) = tile(i, i);
                               }
                           });
        if (live && lane == 0) {
            D_prob(n - 1) = A_local[base_a + (n - 1) + (n - 1) * kP];
        }
    }

    group_barrier(part);

    // Write back A, one row per lane (core lacpy owns elements 2-D Cyclic, not rows).
    if (live) lacpy_rows(Fit{ex}, tile.sub(0, 0, n, n), A);
}

// The CTA tier's workspace: the N x N tile (ld N), then v and w (N each), all local.
template <int N, class T> struct sytrd_cta_plan {
    Plan plan{};
    ws::Slot<T> a{}, v{}, w{};
    constexpr sytrd_cta_plan()
        : a(plan.take<T>(ws::Space::Local, static_cast<std::size_t>(N) * N)), v(plan.take<T>(ws::Space::Local, N)),
          w(plan.take<T>(ws::Space::Local, N)) {}
};

// ---- xLATRD, Lower: one work-group, A in global memory ------------------------------------------
// The lower panel's one body (the host LatrdLowerPanelKernelLegacy's too): reduces columns 0 ..
// ib-1 of the n x n matrix Ab (lower triangle read; ib <= n - 1) and forms W (n x ib), as xLATRD
// UPLO = 'L'. Reflector i leaves in Ab(i+2:, i) with its unit Ab(i+1, i) := 1 (the trailing update
// reads it; the caller writes e back afterwards); on_reflector(i, beta, tau) runs on item 0 as it
// is formed. Ab / Wb: (r, c), data(), ld(). v / wcol: n elements each in memory VM, shared by the
// work-group; vip / wip: ib elements of local memory. The hand kernel's expressions (std::complex
// operators, Native and ExplicitInline policies). Ends on a global barrier. COLLECTIVE.
template <int WG, Mem VM = Mem::Local, class T, class AB, class WB, class Loc, class F>
BATCHLAS_DEVICE_INLINE void latrd_lower_run(const WorkGroup<WG>& ex, const AB& Ab, const WB& Wb, int n, int ib, T* v_ptr, T* wcol_ptr,
                                         const Loc& vip_local, const Loc& wip_local, F&& on_reflector) {
    using Real = real_t<T>;
    using arith::Native;
    const int lid = ex.lid();
    const sycl::group<1> g = ex.it.get_group();
    const auto A = mat(Ab.data(), n, n, Ab.ld());
    const auto W = mat(Wb.data(), n, ib, Wb.ld());
    const auto vfull = vec<VM>(v_ptr, n);
    const auto wfull = vec<VM>(wcol_ptr, n);

    for (int i = 0; i < ib; ++i) {
        if (i >= n - 1) break;

        if (lid < i) {
            const int p = lid;
            vip_local[p] = (i == p + 1) ? T(1) : Ab(i, p);
            wip_local[p] = Wb(i, p);
        }
        ex.barrier(Mem::Local);

        // A(i:, i) -= V(i:, :i) W(i, :i)^H + W(i:, :i) V(i, :i)^H, each term subtracted from the
        // running value (not a gemv: beta = 1 sums first).
        if (lid == 0) {
            T aii = Ab(i, i);
            for (int p = 0; p < i; ++p) {
                const T vip = vip_local[p];
                const T wip = wip_local[p];
                aii -= vip * ::batchlas::device::conj(wip) + wip * ::batchlas::device::conj(vip);
            }
            Ab(i, i) = aii;
        }
        for (int r = i + 1 + lid; r < n; r += WG) {
            T val = Ab(r, i);
            for (int p = 0; p < i; ++p) {
                const T wip = wip_local[p];
                const T vip = vip_local[p];
                const T vrp = unit_head(A.col(p), p + 1)(r);
                const T wrp = Wb(r, p);
                val -= vrp * ::batchlas::device::conj(wip) + wrp * ::batchlas::device::conj(vip);
            }
            Ab(r, i) = val;
        }
        ex.barrier(Mem::Global);

        // Householder reflector of A(i+1:, i); |x|^2 as re*re + im*im of the element in place.
        const auto x = A.col(i).tail(i + 2);
        const Real sumsq = reduce<Scope::Group>(ex, x.size(), [&](int k) {
            const T& xk = x(k);
            if constexpr (Arith<T>::is_complex) {
                const Real re = xk.real();
                const Real im = xk.imag();
                return re * re + im * im;
            } else {
                return xk * xk;
            }
        });
        T alpha = T(0);
        if (lid == 0 && i + 1 < n) {
            alpha = Ab(i + 1, i);
        }
        alpha = ::batchlas::portable::group_broadcast(g, alpha);

        T tau_i = T(0);
        T beta = alpha;
        T scale = T(0);

        if (lid == 0) {
            const Real xnorm = sycl::sqrt(sumsq);
            if constexpr (Arith<T>::is_complex) {
                if (xnorm == Real(0) && alpha.imag() == Real(0)) {
                    tau_i = T(0);
                    beta = alpha;
                    scale = T(0);
                } else {
                    const Real alpha_abs = sycl::hypot(alpha.real(), alpha.imag());
                    const Real beta_abs = sycl::hypot(alpha_abs, xnorm);
                    const T alpha_sign = (alpha_abs == Real(0)) ? T(1) : (alpha / alpha_abs);
                    beta = -alpha_sign * T(beta_abs);
                    tau_i = (beta - alpha) / beta;
                    scale = T(1) / (alpha - beta);
                }
            } else {
                if (xnorm == Real(0)) {
                    tau_i = T(0);
                    beta = alpha;
                    scale = T(0);
                } else {
                    // -sign(alpha) by its sign bit (never 0), as the hand kernel's sign_nonzero.
                    beta = -T(sycl::signbit(static_cast<Real>(alpha)) ? Real(-1) : Real(1)) * T(sycl::hypot(static_cast<Real>(alpha), xnorm));
                    tau_i = (beta - alpha) / beta;
                    scale = T(1) / (alpha - beta);
                }
            }

            on_reflector(i, beta, tau_i);
            Ab(i + 1, i) = T(1);
        }

        tau_i = ::batchlas::portable::group_broadcast(g, tau_i);
        scale = ::batchlas::portable::group_broadcast(g, scale);

        if (tau_i != T(0)) scal<Native>(ex, scale, x);
        ex.barrier(Mem::Global);

        // w = tau (A v - V W^H v - W V^H v), w -= tau/2 (v^H w) v.
        const auto v = vfull.tail(i + 1);
        const auto w = wfull.tail(i + 1);
        // i by value: on AdaptiveCpp (Metal and its CPU device) a by-reference capture of this
        // barrier loop's counter gives NaN eigenvalues (syev_blocked Upper n = 96).
        for_each_owned(ex, i + 1, n, [&, i](int r) { vfull(r) = (r == i + 1) ? T(1) : A(r, i); });
        hemv<Uplo::Lower, alg::RowOwnerSplitStoredDiag, arith::ExplicitInline>(
            ex, One{}, trailing(Ab, i + 1, n), ex.publish(trailing(vfull, i + 1)), Zero{}, trailing(wfull, i + 1));

        for (int p = 0; p < i; ++p) {
            const auto vcol = unit_head(A.col(p), p + 1).tail(i + 1);
            const auto wcol = W.col(p).tail(i + 1);
            const auto gd = dot_many<true, Scope::Group, Native>(ex, v, wcol, vcol);
            const T gamma = gd[0], delta = gd[1];
            map(ex, w, [&](T wr, T vrp, T wrp) { return wr - (vrp * gamma + wrp * delta); }, w, vcol, wcol);
        }

        scal<Native>(ex, tau_i, w);
        const T dot = dotc<Scope::Group, Native>(ex, v, w);
        axpy<Native>(ex, T(-0.5) * tau_i * dot, v, w);
        copy(ex, w, W.col(i).tail(i + 1));
        ex.barrier(Mem::Global);
    }
}

// ---- the Blocked tier ---------------------------------------------------------------------------
// A column-major matrix in global memory as latrd_lower_run reads it.
template <class T> struct ColView {
    T* p;
    int ld_;
    BATCHLAS_DEVICE_INLINE T& operator()(int r, int c) const { return p[r + c * ld_]; }
    BATCHLAS_DEVICE_INLINE T* data() const { return p; }
    BATCHLAS_DEVICE_INLINE int ld() const { return ld_; }
};

// The trailing update's tile: the in-group L3 preset the device_api_l3 tests cover (256 items).
template <class T> using blocked_cfg = l3::cfg::RegTiled<T, 32, 32, 8, 2, 2>;


// The Blocked tier's workspace: W (n x NB, ld n), v and wcol (n each) in global memory; vip, wip
// (NB each) and the L3 stage in local memory. O(NB) local bytes, so any n fits.
template <int NB, class T> struct sytrd_blocked_plan {
    Plan plan{};
    ws::Slot<T> w{}, v{}, wc{}, vip{}, wip{}, smem{};
    constexpr explicit sytrd_blocked_plan(int n)
        : w(plan.take<T>(ws::Space::Global, static_cast<std::size_t>(n) * NB)), v(plan.take<T>(ws::Space::Global, n)),
          wc(plan.take<T>(ws::Space::Global, n)), vip(plan.take<T>(ws::Space::Local, NB)), wip(plan.take<T>(ws::Space::Local, NB)),
          smem(plan.take<T>(ws::Space::Local, l3::smem_elems<blocked_cfg<T>>())) {}
};

// xSYTRD UPLO = 'L' on one n x n matrix per work-group (the host sytrd_blocked's panel loop): per
// NB-column panel, latrd_lower_run on A(j0:, j0:) with W(j0:, :ib), then A22 -= V W^H + W V^H by
// the in-group her2k (Lower triangle only, Im(diag) := 0); finally d(i) = A(i,i) and e(i) back
// into A(i+1, i) and A(i, i+1) = conj(e(i)) (the host restore_tridiag_lower). Im(diag) of the
// input is dropped first. The strictly upper triangle past the superdiagonal is not referenced.
template <int NB, int WG, class T, class DV, class EV, class TV>
BATCHLAS_DEVICE_INLINE void sytrd_blocked_run(const WorkGroup<WG>& ex, const l3::Global<T>& a, const DV& d, const EV& e, const TV& tau,
                                           const Workspace& ws) {
    using Cfg = blocked_cfg<T>;
    using Ar = Arith<T>;
    const int n = a.m, lda = a.ld;
    const sytrd_blocked_plan<NB, T> plan(n);
    T* const wp = carve_global(ws, plan.w);
    T* const vp = carve_global(ws, plan.v);
    T* const wcp = carve_global(ws, plan.wc);
    const auto vip = carve_local(ws, plan.vip).get();
    const auto wip = carve_local(ws, plan.wip).get();
    const l3::Smem<Cfg> sm{carve_local(ws, plan.smem).get_raw()};

    if constexpr (Ar::is_complex) {
        for_each_owned(ex, n, [&](int i) BATCHLAS_DEVICE_BODY { a(i, i) = Ar::make(Ar::re(a(i, i)), real_t<T>(0)); });
        ex.barrier(Mem::Global);
    }
    const int k = n - 1;
    for (int j0 = 0; j0 < k; j0 += NB) {
        const int ib = std::min(NB, k - j0);
        const int j2 = j0 + ib, n2 = n - j2;
        const ColView<T> ap{a.p + j0 + j0 * lda, lda};
        const ColView<T> wv{wp + j0, n};
        latrd_lower_run<WG, Mem::Global>(ex, ap, wv, n - j0, ib, vp, wcp, vip, wip,
                                         [&, j0](int i, const T& beta, const T& t) BATCHLAS_DEVICE_BODY {  // j0 by value: see the latrd i
                                             e(j0 + i) = beta;
                                             tau(j0 + i) = t;
                                         });
        if (n2 > 0)
            her2k<Uplo::Lower, Transpose::NoTrans, Cfg>(ex, l3::Global<T>{a.p + j2 + j0 * lda, lda, n2, ib}, l3::Global<T>{wp + j2, n, n2, ib},
                                                         l3::Global<T>{a.p + j2 + j2 * lda, lda, n2, n2}, NegOne{}, One{}, sm);
    }
    for_each_owned(ex, n, [&](int i) BATCHLAS_DEVICE_BODY {
        d(i) = a(i, i);
        if (i < n - 1) {
            const T ei = e(i);
            a(i + 1, i) = ei;
            a(i, i + 1) = ::batchlas::device::conj(ei);
        }
    });
    ex.barrier(Mem::Global);
}

}  // namespace detail

namespace routine {
// xSYTRD / xHETRD, triangle UL.
template <Uplo UL> struct sytrd {
    static constexpr const char* name = "sytrd";
};
}  // namespace routine

template <Uplo UL, int N, class Alg, class T> struct tier_info<routine::sytrd<UL>, tier::Cta<N, Alg>, T> {
    static_assert(N == 4 || N == 8 || N == 16 || N == 32, "sytrd: tier::Cta<N> has N in {4, 8, 16, 32}");
    static constexpr int wg_size = 32;  // the smallest launch: one sub-group, 32 / N problems
    static constexpr int sg_size = 32;
    static constexpr bool admits(int n) { return n >= 1 && n <= N; }
    static constexpr WorkspaceSize buffer_size(int) { return detail::sytrd_cta_plan<N, T>{}.plan.size(); }
};
// tier::Blocked<NB>: Uplo::Lower only (the host sytrd_blocked's restriction; mirror an Upper input
// with core mirror<Uplo::Upper, true> first, as syev_blocked does).
template <int NB, class Alg, class T> struct tier_info<routine::sytrd<Uplo::Lower>, tier::Blocked<NB, Alg>, T> {
    static constexpr int wg_size = detail::blocked_cfg<T>::threads;  // 256: the her2k tile's items
    static constexpr int sg_size = 32;
    static constexpr bool admits(int n) { return n >= 1; }
    static constexpr WorkspaceSize buffer_size(int n) { return detail::sytrd_blocked_plan<NB, T>(n).plan.size(); }
};
template <Uplo UL, class T> struct tiers<routine::sytrd<UL>, T> {
    using cta = tier::list<tier::Cta<4>, tier::Cta<8>, tier::Cta<16>, tier::Cta<32>>;
    using type = std::conditional_t<UL == Uplo::Lower, tier::list<tier::Cta<4>, tier::Cta<8>, tier::Cta<16>, tier::Cta<32>, tier::Blocked<32>>, cta>;
};

// Bytes of ONE problem's workspace (Cta: local tile + v + w; Blocked: global W, v, wcol and local
// vip, wip and the L3 stage).
template <Uplo UL, class Tier, class T> constexpr WorkspaceSize sytrd_buffer_size(int n) {
    return tier_info<routine::sytrd<UL>, resolve_tier_t<routine::sytrd<UL>, T, Tier>, T>::buffer_size(n);
}

// xSYTRD / xHETRD on one n x n matrix A, reading only A's UL triangle (Im(diag) ignored); d(i) =
// A(i,i), e(i) the off-diagonal, tau(i) (i < n-1) the reflector scalars, all of A's scalar type.
//   tier::Cta<N>      (unblocked: xSYTD2) n = A.rows() <= N, one matrix per SgpPartition<N>; A
//                     leaves in LAPACK's packing with BOTH triangles written (the reflectors in UL,
//                     the conjugate mirror in the other). ws.local only.
//   tier::Blocked<NB> Uplo::Lower, any n, one matrix in global memory per WorkGroup<256>; A leaves
//                     in LAPACK's lower packing plus A(i, i+1) = conj(e(i)); the rest of the upper
//                     triangle is not referenced. ws.global is this matrix's slice.
// ws: sytrd_buffer_size bytes for this problem. COLLECTIVE over the executor. Returns 0, or -2
// when A is not square with 1 <= n (<= N for Cta).
template <Uplo UL, class Tier, class Ex, class MA, class DV, class EV, class TV>
BATCHLAS_DEVICE_INLINE int sytrd(const Ex& ex, const MA& A, const DV& d, const EV& e, const TV& tau, const Workspace& ws) {
    using T = std::remove_cv_t<typename MA::value_type>;
    using R = routine::sytrd<UL>;
    using Tr = resolve_tier_t<R, T, Tier>;
    static_assert(HasTier<R, Tr, T>, "sytrd: the routine has no such tier (tier::Cta<N>; tier::Blocked<NB> for Uplo::Lower)");
    if constexpr (tier::is_blocked_v<Tr>) {
        constexpr int WG = tier_info<R, Tr, T>::wg_size;
        static_assert(std::is_same_v<Ex, WorkGroup<WG>>, "sytrd: tier::Blocked<NB> runs on device::WorkGroup<256> (one matrix per work-group)");
        const auto a = l3::global_of(A);
        if (a.m < 1 || a.n != a.m) return -2;
        detail::sytrd_blocked_run<Tr::nb>(ex, a, d, e, tau, ws);
        return 0;
    } else {
        constexpr int N = Tr::n_max;
        static_assert(std::is_same_v<Ex, SgpPartition<N, false>> || std::is_same_v<Ex, SgpPartition<N, true>>,
                      "sytrd: tier::Cta<N> runs on device::SgpPartition<N> (device::wrap(make_partition<N>(sg)))");
        const int n = A.rows();
        if (n < 1 || n > N || A.cols() != n) return -2;
        const detail::sytrd_cta_plan<N, T> plan{};
        const auto a = carve_local(ws, plan.a).get();
        const auto v = carve_local(ws, plan.v).get();
        const auto w = carve_local(ws, plan.w).get();
        detail::sytrd_cta_run<UL, T, static_cast<size_t>(N)>(ex.part, a, 0, v, 0, w, 0, A, d, e, tau, n, true);
        return 0;
    }
}

}  // namespace batchlas::device
