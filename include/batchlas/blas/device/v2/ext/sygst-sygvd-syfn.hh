// SPDX-License-Identifier: MIT
#pragma once
// Device LA v2, generalized-eigen family (sygst / sygvd / syfn):
//   - trsv_frag / trmv_frag: ?TRSV and ?TRMV with op(A) = A^T or A^H on one work-item, x in a
//     register Frag, A's columns read as aligned 16-byte packs, W split accumulators;
//   - frag_load / frag_store: a Frag from / to any vector operand;
//   - herm / conj_vec / in_triangle: the Hermitian matrix one triangle stores (real diagonal),
//     a conjugated vector view;
//   - for_each_tile_col: the column-wise, coalesced sweep of a group over several m x m tiles;
//   - laset_at / dgmm_at: ?LASET and right ?DGMM on one element of a range-launched kernel.
// Complex arithmetic in the triangular ops is the explicit-fma form (fused::), which is what
// sygst_diag wrote by hand, so its port stays bit-identical. Additive: core headers unchanged.

#include <type_traits>

#include <batchlas/backend_config.h>
#include <batchlas/blas/device/v2/level1.hh>

#ifndef BATCHLAS_DLA_UNROLL_FULL
#if BATCHLAS_SYCL_IMPL_ACPP
#define BATCHLAS_DLA_UNROLL_FULL _Pragma("clang loop unroll(full)")
#else
#define BATCHLAS_DLA_UNROLL_FULL _Pragma("unroll")
#endif
#endif

namespace batchlas::device::v2 {

// ---- extents and packs ---------------------------------------------------------------------
// K for Ext<K>, 0 for a run-time extent.
template <class E> inline constexpr int static_extent_v = 0;
template <int K> inline constexpr int static_extent_v<Ext<K>> = K;

// One aligned 16-byte load of consecutive elements (one element when T is 16 bytes).
template <class T> struct alignas(16) Pack16 {
    T e[16 / sizeof(T) > 0 ? 16 / sizeof(T) : 1];
};
template <class T> inline constexpr int pack16_elems = static_cast<int>(sizeof(Pack16<T>::e) / sizeof(T));

// ---- explicit-fma scalar forms (complex: four fma, no Annex-G multiply) ----------------------
namespace fused {
// acc + a b
template <class T> BATCHLAS_DLA_INLINE T mac(T acc, T a, T b) {
    using A = Arith<T>;
    if constexpr (A::is_complex) {
        auto re = sycl::fma(A::re(a), A::re(b), A::re(acc));
        re = sycl::fma(-A::im(a), A::im(b), re);
        auto im = sycl::fma(A::re(a), A::im(b), A::im(acc));
        im = sycl::fma(A::im(a), A::re(b), im);
        return A::make(re, im);
    } else {
        return sycl::fma(a, b, acc);
    }
}
// acc + conj(a) b
template <class T> BATCHLAS_DLA_INLINE T mac_conj(T acc, T a, T b) {
    using A = Arith<T>;
    if constexpr (A::is_complex) {
        auto re = sycl::fma(A::re(a), A::re(b), A::re(acc));
        re = sycl::fma(A::im(a), A::im(b), re);
        auto im = sycl::fma(A::re(a), A::im(b), A::im(acc));
        im = sycl::fma(-A::im(a), A::re(b), im);
        return A::make(re, im);
    } else {
        return sycl::fma(a, b, acc);
    }
}
// acc - a b
template <class T> BATCHLAS_DLA_INLINE T nmac(T acc, T a, T b) {
    using A = Arith<T>;
    if constexpr (A::is_complex) {
        auto re = sycl::fma(-A::re(a), A::re(b), A::re(acc));
        re = sycl::fma(A::im(a), A::im(b), re);
        auto im = sycl::fma(-A::re(a), A::im(b), A::im(acc));
        im = sycl::fma(-A::im(a), A::re(b), im);
        return A::make(re, im);
    } else {
        return sycl::fma(-a, b, acc);
    }
}
// acc - conj(a) b
template <class T> BATCHLAS_DLA_INLINE T nmac_conj(T acc, T a, T b) {
    using A = Arith<T>;
    if constexpr (A::is_complex) {
        auto re = sycl::fma(-A::re(a), A::re(b), A::re(acc));
        re = sycl::fma(-A::im(a), A::im(b), re);
        auto im = sycl::fma(-A::re(a), A::im(b), A::im(acc));
        im = sycl::fma(A::im(a), A::re(b), im);
        return A::make(re, im);
    } else {
        return sycl::fma(-a, b, acc);
    }
}
// x d and x / d for a real d (complex: per component).
template <class T> BATCHLAS_DLA_INLINE T scale_real(T x, real_t<T> d) {
    using A = Arith<T>;
    if constexpr (A::is_complex) return A::make(A::re(x) * d, A::im(x) * d);
    else return x * d;
}
template <class T> BATCHLAS_DLA_INLINE T div_real(T x, real_t<T> d) {
    using A = Arith<T>;
    if constexpr (A::is_complex) return A::make(A::re(x) / d, A::im(x) / d);
    else return x / d;
}
// re(x) as T (the imaginary part dropped).
template <class T> BATCHLAS_DLA_INLINE T real_only(T x) {
    using A = Arith<T>;
    if constexpr (A::is_complex) return A::make(A::re(x), real_t<T>(0));
    else return x;
}
}  // namespace fused

// ---- operand views -------------------------------------------------------------------------
// (i, j) lies in the uplo triangle (diagonal included). uplo is data here, not an algorithm.
BATCHLAS_DLA_INLINE bool in_triangle(Uplo uplo, int i, int j) { return uplo == Uplo::Lower ? i >= j : i <= j; }

template <class V> struct ConjVec {
    V v;
    using value_type = typename V::value_type;
    BATCHLAS_DLA_INLINE value_type operator()(int i) const { return Arith<value_type>::conj(v(i)); }
    BATCHLAS_DLA_INLINE int size() const { return v.size(); }
};
// conj(v(i)), read-only.
template <class V> BATCHLAS_DLA_INLINE ConjVec<V> conj_vec(V v) { return {v}; }

template <class H> struct HermCol {
    H h;
    int j;
    using value_type = typename H::value_type;
    BATCHLAS_DLA_INLINE value_type operator()(int i) const { return h(i, j); }
    BATCHLAS_DLA_INLINE int size() const { return h.rows(); }
};

// The Hermitian (or symmetric) matrix whose uplo triangle M stores: the diagonal is read as
// real, the other triangle as the conjugate of its mirror. Read-only.
template <class M> struct HermView {
    M a;
    Uplo uplo;
    using value_type = typename M::value_type;
    BATCHLAS_DLA_INLINE value_type operator()(int i, int j) const {
        using A = Arith<value_type>;
        return i == j ? fused::real_only<value_type>(a(i, j)) : in_triangle(uplo, i, j) ? a(i, j) : A::conj(a(j, i));
    }
    BATCHLAS_DLA_INLINE int rows() const { return a.rows(); }
    BATCHLAS_DLA_INLINE int cols() const { return a.cols(); }
    BATCHLAS_DLA_INLINE HermCol<HermView> col(int j) const { return {*this, j}; }
};
template <class M> BATCHLAS_DLA_INLINE HermView<M> herm(M a, Uplo uplo) { return {a, uplo}; }

// ---- register fragments on one work-item ---------------------------------------------------
// The trip count is n's compile-time value for an Ext<K> n, else N with an `i < n` guard.
template <class NE, int N> inline constexpr int frag_trip_v = static_extent_v<NE> ? static_extent_v<NE> : N;

// x[i] = v(i) for i < n; x[i] for i >= n unchanged.
template <class NE, class V, class T, int N, class L>
BATCHLAS_DLA_INLINE Frag<T, N, L> frag_load(const Thread&, NE n, const V& v, Frag<T, N, L> x) {
    static_assert(frag_trip_v<NE, N> <= N, "v2::frag_load: the extent exceeds the fragment");
    BATCHLAS_DLA_UNROLL_FULL
    for (int i = 0; i < frag_trip_v<NE, N>; ++i)
        if (i < int(n)) x[i] = v(i);
    return x;
}

// v(i) = x[i] for i < n.
template <class NE, class T, int N, class L, class V>
BATCHLAS_DLA_INLINE void frag_store(const Thread&, NE n, const Frag<T, N, L>& x, const V& v) {
    static_assert(frag_trip_v<NE, N> <= N, "v2::frag_store: the extent exceeds the fragment");
    BATCHLAS_DLA_UNROLL_FULL
    for (int i = 0; i < frag_trip_v<NE, N>; ++i)
        if (i < int(n)) v(i) = x[i];
}

// The diagonal of A is real (a Cholesky or Hermitian factor): its imaginary part is not read.
struct RealDiag {};

namespace alg {
// Column-dot form: x_i combines column i of A (contiguous, read as aligned 16-byte packs) with
// the other entries of x; W partial sums (term p goes to p mod W) shorten the dependent chain.
// Every column start must be 16-byte aligned (A.p and ld * sizeof(T) multiples of 16).
template <int W = 4> struct ColDotPacked {};
}  // namespace alg

namespace detail {
template <class Alg> struct coldot_ways {
    static_assert(sizeof(Alg) == 0, "v2::trsv_frag/trmv_frag: the algorithm is alg::ColDotPacked<W>");
};
template <int W> struct coldot_ways<alg::ColDotPacked<W>> {
    static constexpr int value = W;
};

template <int W, class T> BATCHLAS_DLA_INLINE T tree_sum(T (&acc)[W]) {
    static_assert(is_pow2(W), "v2: ColDotPacked<W> needs a power-of-two W");
#pragma unroll
    for (int w = W / 2; w >= 1; w /= 2) {
#pragma unroll
        for (int k = 0; k < w; ++k) acc[k] = Arith<T>::add(acc[2 * k], acc[2 * k + 1]);
    }
    return acc[0];
}

template <bool Solve, bool Conj, class T> BATCHLAS_DLA_INLINE T coldot_step(T acc, T a, T x) {
    if constexpr (Solve) return Conj ? fused::nmac_conj(acc, a, x) : fused::nmac(acc, a, x);
    else return Conj ? fused::mac_conj(acc, a, x) : fused::mac(acc, a, x);
}

// Below: the terms p < i (A upper), else p > i (A lower). Solve: x_i = (x_i - sum) / a_ii, else
// x_i = a_ii x_i + sum. In place, so the order visits i before any entry its sum still needs.
template <bool Below, bool Solve, bool Conj, int W, class T, int N, class L, class MA>
BATCHLAS_DLA_INLINE Frag<T, N, L> coldot_tri(const MA& a, Frag<T, N, L> x) {
    using A = Arith<T>;
    constexpr int V = pack16_elems<T>;
    constexpr int Trip = frag_trip_v<std::remove_cv_t<decltype(a.n)>, N>;
    static_assert(Trip <= N, "v2::trsv_frag/trmv_frag: the extent exceeds the fragment");
    constexpr bool ascending = Below == Solve;
    const int n = a.cols();
    BATCHLAS_DLA_UNROLL_FULL
    for (int s = 0; s < Trip; ++s) {
        const int i = ascending ? s : Trip - 1 - s;
        if (i < n) {
            T acc[W];
#pragma unroll
            for (int k = 0; k < W; ++k) acc[k] = A::zero();
            const T* const line = &a(0, i);  // column i of A
            if constexpr (Below) {
                BATCHLAS_DLA_UNROLL_FULL
                for (int p0 = 0; p0 < i; p0 += V) {
                    const Pack16<T> pk = *reinterpret_cast<const Pack16<T>*>(line + p0);
                    BATCHLAS_DLA_UNROLL_FULL
                    for (int u = 0; u < V; ++u)
                        if (p0 + u < i) acc[(p0 + u) % W] = coldot_step<Solve, Conj>(acc[(p0 + u) % W], pk.e[u], x[p0 + u]);
                }
            } else {
                BATCHLAS_DLA_UNROLL_FULL
                for (int p0 = (i + 1) / V * V; p0 < Trip; p0 += V) {
                    if (p0 >= n) continue;
                    const Pack16<T> pk = *reinterpret_cast<const Pack16<T>*>(line + p0);
                    BATCHLAS_DLA_UNROLL_FULL
                    for (int u = 0; u < V; ++u)
                        if (p0 + u > i && p0 + u < n)
                            acc[(p0 + u) % W] = coldot_step<Solve, Conj>(acc[(p0 + u) % W], pk.e[u], x[p0 + u]);
                }
            }
            const T sum = tree_sum<W>(acc);
            const auto d = A::re(line[i]);
            if constexpr (Solve) x[i] = fused::div_real(A::add(x[i], sum), d);
            else x[i] = A::add(fused::scale_real(x[i], d), sum);
        }
    }
    return x;
}
}  // namespace detail

// x := op(A)^-1 x, op = Trans or ConjTrans, A n x n triangular (uplo) with a real diagonal.
// By value: an array out-parameter spills once the caller is not inlined.
template <Uplo UP, Transpose TR, class Dg = RealDiag, class Alg = alg::ColDotPacked<4>, class T, int N, class L, class MA>
BATCHLAS_DLA_INLINE Frag<T, N, L> trsv_frag(const Thread&, const MA& a, Frag<T, N, L> x) {
    static_assert(TR != Transpose::NoTrans, "v2::trsv_frag: the column-dot form is op(A) = A^T or A^H");
    static_assert(std::is_same_v<Dg, RealDiag>, "v2::trsv_frag: only RealDiag is implemented");
    return detail::coldot_tri<UP == Uplo::Upper, true, TR == Transpose::ConjTrans, detail::coldot_ways<Alg>::value>(a, x);
}

// x := op(A) x, op = Trans or ConjTrans, A n x n triangular (uplo) with a real diagonal.
template <Uplo UP, Transpose TR, class Dg = RealDiag, class Alg = alg::ColDotPacked<4>, class T, int N, class L, class MA>
BATCHLAS_DLA_INLINE Frag<T, N, L> trmv_frag(const Thread&, const MA& a, Frag<T, N, L> x) {
    static_assert(TR != Transpose::NoTrans, "v2::trmv_frag: the column-dot form is op(A) = A^T or A^H");
    static_assert(std::is_same_v<Dg, RealDiag>, "v2::trmv_frag: only RealDiag is implemented");
    return detail::coldot_tri<UP == Uplo::Upper, false, TR == Transpose::ConjTrans, detail::coldot_ways<Alg>::value>(a, x);
}

// ---- run-time-width group sweeps -----------------------------------------------------------
// A group of `size` items sweeping `tiles` m x m tiles column by column: item lid owns row
// lid % m of the tile columns lid / m, lid / m + size / m, ...; f(tile, row, col). Columns are
// coalesced. When m does not divide size, the last lid's rows also revisit columns another item
// covers (same element, same value): f must be idempotent.
template <class F> BATCHLAS_DLA_INLINE void for_each_tile_col(int lid, int size, int m, int tiles, F&& f) {
    const int row = lid % m, step = size / m;
    for (int idx = lid / m; idx < tiles * m; idx += step) {
        const int g = idx / m, c = idx - g * m;
        f(g, row, c);
    }
}

// ---- element ops for range-launched kernels --------------------------------------------------
template <class T, class S> BATCHLAS_DLA_INLINE T tag_value(S s) {
    if constexpr (std::is_same_v<S, One>) return Arith<T>::one();
    else if constexpr (std::is_same_v<S, Zero>) return Arith<T>::zero();
    else if constexpr (std::is_same_v<S, NegOne>) return Arith<T>::neg(Arith<T>::one());
    else return static_cast<T>(s.v);
}

// ?LASET on element (i, j): beta on the diagonal, alpha off it; the full matrix.
template <class T, class AS, class BS> BATCHLAS_DLA_INLINE void laset_at(T& aij, int i, int j, AS alpha, BS beta) {
    aij = i == j ? tag_value<T>(as_tag(beta)) : tag_value<T>(as_tag(alpha));
}
// ?LASET on element (i, j) of the uplo triangle; an element outside it is not touched.
template <class T, class AS, class BS> BATCHLAS_DLA_INLINE void laset_at(Uplo uplo, T& aij, int i, int j, AS alpha, BS beta) {
    if (!in_triangle(uplo, i, j)) return;
    laset_at(aij, i, j, alpha, beta);
}

// Right ?DGMM on one element: x(i, j) d(j), d real or of x's type.
template <class T, class D> BATCHLAS_DLA_INLINE T dgmm_at(T xij, D dj) {
    if constexpr (std::is_same_v<D, T>) return Arith<T>::mul(xij, dj);
    else return fused::scale_real(xij, static_cast<real_t<T>>(dj));
}

}  // namespace batchlas::device::v2
