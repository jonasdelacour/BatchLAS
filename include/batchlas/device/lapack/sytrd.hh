// SPDX-License-Identifier: MIT
#pragma once
// xSYTRD / xHETRD: sytd2 (the unblocked reduction on a resident tile, one row per item) and
// sytrd<UL, tier::Cta<N>> (one matrix per SgpPartition<N>, the tile in local memory). Strict
// port: arith::Native and the replaced kernels' expression shapes. sytrd_detail holds the CTA
// body (sytrd_cta_run, the host sytrd_cta kernel's too) and the forms core does not reproduce
// bit for bit yet: Fit dotc / gemv / her2 (product-only dot, subtracted rank-2 term), lacpy_herm.

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <batchlas/device.hh>
#include <batchlas/device/sg_partition.hh>

namespace batchlas::device::inline v2 {

namespace sytrd_detail {

// x1 conj(y1) + x2 conj(y2): one term of a rank-2 Hermitian update. Real: one expression, so
// it contracts to fma(x1, y1, x2 * y2) as the hand kernel's did.
template <class Ops, class T> BATCHLAS_DLA_INLINE T her2_term(const T& x1, const T& y1, const T& x2, const T& y2) {
    using Ar = arith_t<Ops, T>;
    if constexpr (Ar::is_complex) return Ar::add(Ar::mul(x1, Ar::conj(y1)), Ar::mul(x2, Ar::conj(y2)));
    else return x1 * y1 + x2 * y2;
}

// x^H y, one product per item (no accumulator), then the chunk reduction.
template <class Ops, class Ex, class X, class Y> BATCHLAS_DLA_INLINE auto dotc(const Fit<Ex>& ex, const X& x, const Y& y) {
    using T = elem_t<X>;
    using Ar = arith_t<Ops, T>;
    const int i = ex.lid();
    const T p = (i >= 0 && i < x.size()) ? Ar::mul(Ar::conj(x(i)), y(i)) : Ar::zero();
    return detail::scope_reduce<Scope::Auto>(ex, p);
}

// y(r) := (A x)(r) * alpha with item r computing output row r (core gemv's RowOwner without
// the loop); x is read across owners. alpha multiplies on the right, as `y *= tau` did.
template <class Ops, class Ex, class T, class MA, class X, class Y>
BATCHLAS_DLA_INLINE void gemv_rows(const Fit<Ex>& ex, const T& alpha, const MA& a, const Published<X>& xp, const Y& y) {
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
BATCHLAS_DLA_INLINE void her2_full_sub(const Fit<Ex>& ex, const Published<X>& xp, const Published<Y>& yp, const MA& a) {
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
BATCHLAS_DLA_INLINE void lacpy_herm(const Fit<Ex>& ex, const MA& a, const MT& tile, bool active) {
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
template <class Ex, class MA, class MB> BATCHLAS_DLA_INLINE void lacpy_rows(const Fit<Ex>& ex, const MA& a, const MB& b) {
    const int n = a.cols();
    for_each_owned(ex, a.rows(), [&](int r) {
        for (int c = 0; c < n; ++c) b(r, c) = a(r, c);
    });
}

// One sytd2 step on the active block blk: [alpha; x] = col one element per item (alpha on item
// ar), row is col's conjugate mirror. Reflector; mirrored write-back; then, unless tau == 0,
// v := reflector, w := A v tau, w += (-tau/2)(v^H w) v, A -= v w^H + w v^H.
template <Uplo UL, class Ops, class Ex, class Blk, class Col, class Row, class V, class W, class T, class F>
BATCHLAS_DLA_INLINE void sytd2_step(const Ex& ex, int i, int m, int ar, const Blk& blk, const Col& col, const Row& row,
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
    const T taui = v2::larfg<conv::Phase, Ops>(ex, m, ar, alpha, x, x_active);
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
        v2::copy(fx, v2::unit_head<refl::QR, unit_fill::Keep>(col, ar), vm);
        v2::sync<scope::Part>(ex);
        if (is_alpha) {  // LAPACK's temporary unit; the block ops read v, not the column
            col(ar) = Ar::one();
            row(ar) = Ar::one();
        }
        v2::sync<scope::Part>(ex);
        sytrd_detail::gemv_rows<Ops>(fx, taui, blk, assume_published(vm), wm);
        v2::sync<scope::Part>(ex);
        const T alpha2 = T(-0.5) * taui * sytrd_detail::dotc<Ops>(fx, vm, wm);
        v2::axpy<Ops>(fx, Scalar<T>{alpha2}, vm, wm);
        v2::sync<scope::Part>(ex);
        sytrd_detail::her2_full_sub<Ops>(fx, assume_published(vm), assume_published(wm), blk);
        v2::sync<scope::Part>(ex);
        if (is_alpha) {
            col(ar) = alpha;
            row(ar) = Ar::conj(alpha);
        }
        v2::sync<scope::Part>(ex);
    } else if constexpr (UL == Uplo::Upper) {
        if (is_alpha) row(ar) = Ar::conj(alpha);
    }
}

}  // namespace sytrd_detail

// xSYTD2 / xHETD2 on the n x n leading block of a resident tile held in FULL storage (both
// triangles, real diagonal), one row per item (n <= Ex::size); v and w are length >= n scratch.
// Leaves LAPACK's packing, every element's conjugate mirror kept:
//   d(i) = A(i,i); Upper: e(i) = A(i,i+1), reflector i in A(0:i-1, i+1) (unit at row i);
//                  Lower: e(i) = A(i+1,i), reflector i in A(i+2:n-1, i) (unit at row i+1).
// on_step(i, beta, tau) runs on the item holding reflector i's alpha right after it is formed.
// Upper returns tau(lid) on item lid (0 past n-2); Lower returns 0 (tau goes through on_step).
// Step k touches only its active block, so column k is final once step k has run. COLLECTIVE.
template <Uplo UL, class Ops = arith::Native, class Ex, class MA, class V, class W, class F = NoOp>
BATCHLAS_DLA_INLINE typename MA::value_type sytd2(const Ex& ex, const MA& a, const V& v, const W& w, int n, F&& on_step = {}) {
    using T = typename MA::value_type;
    const int lane = ex.lid();
    T tau_lane = arith_t<Ops, T>::zero();
    if constexpr (UL == Uplo::Upper) {
        for (int k = n - 1; k >= 1; --k) {  // annihilate A(0:k-2, k); alpha = A(k-1, k)
            sytrd_detail::sytd2_step<UL, Ops>(ex, k - 1, k, k - 1, a.sub(0, 0, k, k), a.col(k).head(k), a.row(k).head(k), v, w,
                                              lane < k - 1, tau_lane, on_step);
            v2::sync<scope::Part>(ex);
        }
    } else {
        for (int i = 0; i <= n - 2; ++i) {  // annihilate A(i+2:n-1, i); alpha = A(i+1, i) on item 0
            const int s = i + 1, m = n - s;
            sytrd_detail::sytd2_step<UL, Ops>(ex, i, m, 0, a.sub(s, s, m, m), a.col(i).tail(s).head(m), a.row(i).tail(s).head(m),
                                              v, w, lane >= 1 && lane < m, tau_lane, on_step);
        }
    }
    return tau_lane;
}

namespace sytrd_detail {

// Local memory as the decorated pointer the Upper tile views take (never &acc[off]: generic).
template <class T> BATCHLAS_DLA_INLINE typename local_mptr<T>::pointer decorated_ptr(const sycl::local_accessor<T, 1>& a) {
    return a.template get_multi_ptr<sycl::access::decorated::yes>().get();
}
template <class T> BATCHLAS_DLA_INLINE T* decorated_ptr(T* p) { return p; }

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
BATCHLAS_DLA_INLINE void sytrd_cta_run(const SubGroupPartition<P, M>& part, const Buf& A_local, int32_t base_a,
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
template <int N, class T> struct cta_plan {
    Plan plan{};
    ws::Slot<T> a{}, v{}, w{};
    constexpr cta_plan()
        : a(plan.take<T>(ws::Space::Local, static_cast<std::size_t>(N) * N)), v(plan.take<T>(ws::Space::Local, N)),
          w(plan.take<T>(ws::Space::Local, N)) {}
};

}  // namespace sytrd_detail

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
    static constexpr WorkspaceSize buffer_size(int) { return sytrd_detail::cta_plan<N, T>{}.plan.size(); }
};
template <Uplo UL, class T> struct tiers<routine::sytrd<UL>, T> {
    using type = tier::list<tier::Cta<4>, tier::Cta<8>, tier::Cta<16>, tier::Cta<32>>;
};

// Bytes of ONE problem's workspace (local: tile + v + w; global: none).
template <Uplo UL, class Tier, class T> constexpr WorkspaceSize sytrd_buffer_size(int n) {
    return tier_info<routine::sytrd<UL>, resolve_tier_t<routine::sytrd<UL>, T, Tier>, T>::buffer_size(n);
}

// xSYTRD / xHETRD (unblocked: xSYTD2) on one n x n matrix A (n = A.rows() <= N) per
// SgpPartition<N>, reading only A's UL triangle (Im(diag) ignored). A leaves in LAPACK's packing
// with BOTH triangles written (the reflectors in UL, the conjugate mirror in the other); d(i) =
// A(i,i), e(i) the off-diagonal, tau(i) (i < n-1) the reflector scalars, all of A's scalar type.
// ws.local: sytrd_buffer_size bytes for this problem. COLLECTIVE over the partition. Returns 0,
// or -2 when A is not square with 1 <= n <= N.
template <Uplo UL, class Tier, class Ex, class MA, class DV, class EV, class TV>
BATCHLAS_DLA_INLINE int sytrd(const Ex& ex, const MA& A, const DV& d, const EV& e, const TV& tau, const Workspace& ws) {
    using T = std::remove_cv_t<typename MA::value_type>;
    using R = routine::sytrd<UL>;
    using Tr = resolve_tier_t<R, T, Tier>;
    static_assert(HasTier<R, Tr, T>, "sytrd: the routine has no such tier (tier::Cta<N> only)");
    constexpr int N = Tr::n_max;
    static_assert(std::is_same_v<Ex, SgpPartition<N, false>> || std::is_same_v<Ex, SgpPartition<N, true>>,
                  "sytrd: tier::Cta<N> runs on device::SgpPartition<N> (device::wrap(make_partition<N>(sg)))");
    const int n = A.rows();
    if (n < 1 || n > N || A.cols() != n) return -2;
    const sytrd_detail::cta_plan<N, T> plan{};
    const auto a = carve_local(ws, plan.a).get();
    const auto v = carve_local(ws, plan.v).get();
    const auto w = carve_local(ws, plan.w).get();
    sytrd_detail::sytrd_cta_run<UL, T, static_cast<size_t>(N)>(ex.part, a, 0, v, 0, w, 0, A, d, e, tau, n, true);
    return 0;
}

}  // namespace batchlas::device
