// SPDX-License-Identifier: MIT
#pragma once
// Device LA v2 ops private to the LU solves (getrs fused and composed tiers). Everything shared
// comes from v2 core (RtGroup, for_each_owned, lacpy, laswp, select, arith::Fma); these have
// one user:
//   ConjIf, Trans      run-time conjugated and transposed views (core has no adaptors yet)
//   sum_to_lane0       a shift-down sub-group sum, only lane 0 holds the total
//   trsm_lane          op(A)^{-1} Y, one sub-group, lane i owns row i, shuffles only
//   gemm_nr            C -= op(A) X with X.cols() <= NR accumulators
//   lapmr_copy         B(i, :) := A(perm(i), :), out of place
// Arithmetic is arith::Fma: sycl_device::dev_mul / fma_acc / dev_div bit for bit.

#include <sycl/sycl.hpp>

#include <batchlas/device.hh>
#include <batchlas/util/group-collectives.hh>
#include <src/kernels/common/dla_v2_scalar.hh>
#include <batchlas/device/detail/kernel_attrs.hh>

namespace batchlas::device::inline v2::lu {

template <class T> using lu_arith = arith_t<arith::Fma, T>;

template <class V> struct ConjIf {
    using value_type = typename V::value_type;
    V v;
    bool c;
    BATCHLAS_DLA_INLINE value_type operator()(int i, int j) const {
        const value_type a = v(i, j);
        return c ? Arith<value_type>::conj(a) : a;
    }
    BATCHLAS_DLA_INLINE int rows() const { return v.rows(); }
    BATCHLAS_DLA_INLINE int cols() const { return v.cols(); }
    BATCHLAS_DLA_INLINE auto sub(int i, int j, int r, int cc) const {
        return ConjIf<decltype(v.sub(i, j, r, cc))>{v.sub(i, j, r, cc), c};
    }
};
template <class V> struct Trans {
    using value_type = typename V::value_type;
    V v;
    BATCHLAS_DLA_INLINE decltype(auto) operator()(int i, int j) const { return v(j, i); }
    BATCHLAS_DLA_INLINE int rows() const { return v.cols(); }
    BATCHLAS_DLA_INLINE int cols() const { return v.rows(); }
};
template <class V> BATCHLAS_DLA_INLINE ConjIf<V> conj_if(V v, bool c) { return {v, c}; }
template <class V> BATCHLAS_DLA_INLINE Trans<V> trans(V v) { return {v}; }

// Plain inline, the replaced helper's attribute. No static local memory: sycl::reduce_over_group
// adds some and reopens the 48 KB launch hole.
template <int SG, class G, class T> BATCHLAS_DLA_HELPER T sum_to_lane0(const G& g, T v) {
    using A = Arith<T>;
    if constexpr (A::is_complex) {
        auto re = A::re(v);
        auto im = A::im(v);
        for (int off = SG / 2; off > 0; off >>= 1) {
            re += sycl::shift_group_left(g, re, off);
            im += sycl::shift_group_left(g, im, off);
        }
        return A::make(re, im);
    } else {
        for (int off = SG / 2; off > 0; off >>= 1) v += sycl::shift_group_left(g, v, off);
        return v;
    }
}

// Y(:, c) := op(A)^{-1} Y(:, c) for c = c0, c0 + cs, ... < Y.cols(), lane i owning row i (rows
// >= A.rows() idle). Lower runs forward, Upper backward; Unit skips the last pivot, which
// updates nothing. Collective: every lane calls it; the lane guards are inside.
template <Uplo UL, Diag DG, int SG, class MA, class MY>
BATCHLAS_DLA_INLINE void trsm_lane(const SubGroup<SG>& ex, const MA& a, const MY& y, int c0, int cs) {
    using T = typename MY::value_type;
    using Ar = lu_arith<T>;
    const int lane = ex.lid();
    const int jb = a.rows();
    for (int c = c0; c < y.cols(); c += cs) {
        T v = (lane < jb) ? y(lane, c) : Ar::zero();
        auto pivot = [&](int s) {
            if constexpr (DG == Diag::NonUnit) {
                if (lane == s) v = Ar::div(v, a(s, s));
            }
            return batchlas::portable::group_broadcast(ex.sg, v, s);
        };
        if constexpr (UL == Uplo::Lower) {
            const int end = DG == Diag::Unit ? jb - 1 : jb;
            for (int s = 0; s < end; ++s) {
                const T vs = pivot(s);
                if (lane > s && lane < jb) v = Ar::sub(v, Ar::mul(a(lane, s), vs));
            }
        } else {
            const int stop = DG == Diag::Unit ? 1 : 0;
            for (int s = jb - 1; s >= stop; --s) {
                const T vs = pivot(s);
                if (lane < s) v = Ar::sub(v, Ar::mul(a(lane, s), vs));
            }
        }
        if (lane < jb) y(lane, c) = v;
    }
}

// C := C - op(A) X; X.cols() <= NR is read at run time, accumulators past it are never touched.
// NoTrans: item per row of C, k serial (coalesced down A's columns). Trans: sub-group per row
// of C, lanes stride k, sum_to_lane0, lane 0 writes. ConjTrans is Trans of conj_if(A, true).
template <Transpose TA, int NR, class Ex, class MA, class MX, class MC>
BATCHLAS_DLA_INLINE void gemm_nr(const Ex& ex, NegOne, const MA& a, const MX& x, One, const MC& c) {
    using T = typename MC::value_type;
    using Ar = lu_arith<T>;
    const int nrhs = x.cols();
    if constexpr (TA == Transpose::NoTrans) {
        const int k = a.cols();
        for_each_owned(ex, c.rows(), [&](int i) BATCHLAS_DLA_BODY {
            T acc[NR];
BATCHLAS_UNROLL_FULL
            for (int r = 0; r < NR; ++r) acc[r] = Ar::zero();
            for (int kk = 0; kk < k; ++kk) {
                const T av = a(i, kk);
BATCHLAS_UNROLL_FULL
                for (int r = 0; r < NR; ++r)
                    if (r < nrhs) Ar::mac(acc[r], av, x(kk, r));
            }
BATCHLAS_UNROLL_FULL
            for (int r = 0; r < NR; ++r)
                if (r < nrhs) c(i, r) = Ar::sub(c(i, r), acc[r]);
        });
    } else {
        static_assert(TA == Transpose::Trans, "lu::gemm_nr: ConjTrans is Trans of conj_if(A, true)");
        constexpr int SG = Ex::sg_size;
        const int m = c.rows(), k = a.rows();
        const int lane = ex.lane();
        for (int t = ex.team(); t < m; t += detail::nteams_of(ex)) {
            T acc[NR];
BATCHLAS_UNROLL_FULL
            for (int r = 0; r < NR; ++r) acc[r] = Ar::zero();
            for (int i = lane; i < k; i += SG) {
                const T av = a(i, t);
BATCHLAS_UNROLL_FULL
                for (int r = 0; r < NR; ++r)
                    if (r < nrhs) Ar::mac(acc[r], av, x(i, r));
            }
BATCHLAS_UNROLL_FULL
            for (int r = 0; r < NR; ++r)
                if (r < nrhs) {
                    const T s = sum_to_lane0<SG>(ex.sub(), acc[r]);
                    if (lane == 0) c(t, r) = Ar::sub(c(t, r), s);
                }
        }
    }
}

// B(i, :) := A(perm(i), :) (LAPACK lapmr, forward, out of place); perm(i) is 0-based.
template <class Alg = alg::Stepped, class Ex, class MA, class P, class MB>
BATCHLAS_DLA_INLINE void lapmr_copy(const Ex& ex, const MA& a, const P& perm, const MB& b) {
    for_each_owned_2d<Alg>(ex, b.rows(), b.cols(), [&](int i, int c) BATCHLAS_DLA_BODY { b(i, c) = a(perm(i), c); });
}

}  // namespace batchlas::device::lu
