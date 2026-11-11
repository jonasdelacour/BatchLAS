// SPDX-License-Identifier: MIT
#pragma once
// Device LA, Jacobi family (syev_jacobi_cta, gesvdj_cta): the 2x2 Hermitian rotation
// (sym_schur2) and per-item fragment ops on Cyclic rows of a static extent, the pieces still
// waiting for core (inv_sqrt in scalar.hh; the Frag ops in level1 / reduce). The pivot schedule,
// threshold, rotation record, phased rotation and rotation rounds are public
// (batchlas/device/lapack/jacobi.hh, jacobi_detail); reduce_scatter is core (reduce.hh).
// Ops = arith::Native (default) keeps each legacy expression whole (see lapack/jacobi.hh).

#include <src/sycl/inv_sqrt.hh>
#include <cstdint>
#include <type_traits>

#include <batchlas/device/lapack/jacobi.hh>
#include <batchlas/device/operand.hh>
#include <batchlas/device/reduce.hh>
#include <batchlas/device/scalar.hh>
#include <batchlas/device/sg_partition.hh>

namespace batchlas::device {

// ---- Cyclic rows of a static extent N (N % size == 0): item l owns rows l + k*size ----------
// f(k, row) for the item's N / size rows, k a compile-time Frag slot; no bound check.
// U = unroll::Full is a full unroll pragma, unroll::Default none.
template <int N, class U = unroll::Full, class Ex, class F> BATCHLAS_DEVICE_INLINE void for_each_owned_frag(const Ex& ex, F&& f) {
    static_assert(N % Ex::size == 0, "for_each_owned_frag: N must be a multiple of the executor size");
    constexpr int K = N / Ex::size;
    const int l = ex.lid();
    if constexpr (std::is_same_v<U, unroll::Full>) {
#pragma unroll
        for (int k = 0; k < K; ++k) f(k, l + k * Ex::size);
    } else {
        static_assert(std::is_same_v<U, unroll::Default>, "for_each_owned_frag: unroll::Full or unroll::Default");
        for (int k = 0; k < K; ++k) f(k, l + k * Ex::size);
    }
}

// x[k] = v(l + k*size) for rows < n, else 0.
template <class Ex, class V, class T, int K> BATCHLAS_DEVICE_INLINE void load_owned_masked(const Ex& ex, const V& v, int n, Frag<T, K>& x) {
    for_each_owned_frag<K * Ex::size>(ex, [&](int k, int r) { x[k] = (r < n) ? static_cast<T>(v(r)) : T(0); });
}
// x = e_j restricted to the item's rows.
template <class Ex, class T, int K> BATCHLAS_DEVICE_INLINE void set_unit_owned(const Ex& ex, int j, Frag<T, K>& x) {
    for_each_owned_frag<K * Ex::size>(ex, [&](int k, int r) { x[k] = (r == j) ? T(1) : T(0); });
}
// sum_k conj(x[k]) y[k] over the item's fragment (a partial dotc; reduce it with allreduce).
template <class Ops = arith::Native, class T, int K> BATCHLAS_DEVICE_INLINE T dotc_owned(const Frag<T, K>& x, const Frag<T, K>& y) {
    T acc = T(0);
#pragma unroll
    for (int k = 0; k < K; ++k) {
        if constexpr (jacobi_detail::is_native_v<Ops>) acc = acc + conj(x[k]) * y[k];
        else arith_t<Ops, T>::mac_conj(acc, x[k], y[k]);
    }
    return acc;
}
// x[k] = u(l + k*size), y[k] = v(l + k*size) and the partial dotc sum_k conj(x[k]) y[k] in one
// pass (x, y: Frag or T[K]). Two loads plus dotc_owned gave complex T extra BRA.
template <int K, class Ex, class U, class V, class X>
BATCHLAS_DEVICE_INLINE auto load_dotc_owned(const Ex& ex, const U& u, const V& v, X& x, X& y) {
    using T = std::remove_cvref_t<decltype(x[0])>;
    T acc = T(0);
    const int l = ex.lid();
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int r = l + k * Ex::size;
        x[k] = u(r);
        y[k] = v(r);
        acc = acc + conj(x[k]) * y[k];
    }
    return acc;
}
// y[k] := y[k] - x[k] a (one classical Gram-Schmidt projection step).
template <class Ops = arith::Native, class T, int K> BATCHLAS_DEVICE_INLINE void sub_mul(Frag<T, K>& y, const Frag<T, K>& x, T a) {
#pragma unroll
    for (int k = 0; k < K; ++k) {
        if constexpr (jacobi_detail::is_native_v<Ops>) y[k] = y[k] - x[k] * a;
        else arith_t<Ops, T>::msub(y[k], x[k], a);
    }
}
// sum of |x[k]|^2 over the item's rows < n (each abs2 its own rounding, as the hand loop).
template <class Ex, class T, int K> BATCHLAS_DEVICE_INLINE real_t<T> sumsq_owned(const Ex& ex, const Frag<T, K>& x, int n) {
    real_t<T> acc = real_t<T>(0);
    for_each_owned_frag<K * Ex::size>(ex, [&](int k, int r) {
        if (r < n) acc += abs2(x[k]);
    });
    return acc;
}

// Lane l returns ||a(:, l)||^2 for the P = Ex::size columns of a, whose static row count (a.m,
// an Ext) is Cyclic over the executor: per-lane partial sums, then reduce_scatter<P> (log2 P
// exchanges; core sumsq over P columns would be P all-reduces).
template <class Ex, class M> BATCHLAS_DEVICE_INLINE auto col_sumsq(const Ex& ex, const M& a) {
    using T = typename M::value_type;
    using R = real_t<T>;
    constexpr int P = Ex::size;
    constexpr int Rows = decltype(M::m)::value;
    Frag<R, P> x;
#pragma unroll
    for (int c = 0; c < P; ++c) {
        R acc = R(0);
        for_each_owned_frag<Rows>(ex, [&](int, int r) { acc += abs2(static_cast<T>(a(r, c))); });
        x[c] = acc;
    }
    return reduce_scatter<P>(ex, x);
}

// Golub & Van Loan sym.schur2 generalised by a diagonal phase: for H = [app apq; conj(apq) aqq]
// (app, aqq real) U^H H U is diagonal. d = conj(apq) / |apq| makes the 2x2 real, then the
// classical rotation with tau = (aqq - app) / (2 g), t = sign(tau) / (|tau| + sqrt(1 + tau^2)).
// For real T, d = 1 and g carries the sign of apq. A rotation below the threshold or whose s
// rounds to 0 (it could never annihilate apq) is the identity with active == false.
template <class T>
BATCHLAS_DEVICE_INLINE jacobi_detail::JacobiRotation<T> sym_schur2(real_t<T> app, real_t<T> aqq, T apq,
                                                                const jacobi_detail::JacobiThreshold<real_t<T>>& th) {
    using R = real_t<T>;
    jacobi_detail::JacobiRotation<T> r = jacobi_detail::identity_rotation<T>();
    const R g_abs = modulus(apq);
    const R thresh = th.tol * sycl::sqrt(sycl::fabs(app) * sycl::fabs(aqq));
    if (g_abs > thresh && g_abs > th.tiny) {
        R g;
        if constexpr (Arith<T>::is_complex) {
            g = g_abs;
            r.d = T(apq.real() / g_abs, -apq.imag() / g_abs);
        } else {
            g = apq;
            r.d = T(1);
        }
        r.g = g;
        const R tau = (aqq - app) / (R(2) * g);
        R tt;
        if (sycl::fabs(tau) > th.tau_big) {
            tt = R(1) / (R(2) * tau);
        } else {
            tt = sycl::copysign(R(1), tau) / (sycl::fabs(tau) + sycl::sqrt(R(1) + tau * tau));
        }
        r.t = tt;
        r.c = ::batchlas::sycl_device::inv_sqrt<::batchlas::sycl_device::InvSqrtEdge::FiniteGeOne>(R(1) + tt * tt);
        r.s = tt * r.c;
        r.active = !(r.s == R(0));
    }
    if (!r.active) {
        r.c = R(1);
        r.s = R(0);
        r.d = T(1);
    }
    return r;
}

}  // namespace batchlas::device
