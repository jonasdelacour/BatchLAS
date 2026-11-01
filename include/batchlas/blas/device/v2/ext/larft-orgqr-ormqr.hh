// SPDX-License-Identifier: MIT
#pragma once
// Device LA v2 ops for the larft / orgqr / ormqr family: trmv (row owner, out of place), laset
// (2-D, no % or /) and larf on a vector one item owns whole (a ColPerLane register column, or a
// row of an SLM tile). Conventions as in v2: BLAS argument order (alpha first), every algorithm a
// type, caller-owned barriers, Arith<T> in every inner loop.

#include <type_traits>

#include <batchlas/blas/device/v2/level2.hh>

#ifndef BATCHLAS_DLA_UNROLL_FULL
#if BATCHLAS_SYCL_IMPL_ACPP
#define BATCHLAS_DLA_UNROLL_FULL _Pragma("clang loop unroll(full)")  // SSCP JIT: bare unroll is partial
#else
#define BATCHLAS_DLA_UNROLL_FULL _Pragma("unroll")
#endif
#endif

namespace batchlas::device::v2 {

namespace alg {
struct Fused {};    // larf update x -= a*b as one fma (one rounding)
struct Unfused {};  // the product rounded first, then subtracted: v1 ger's form, for bit-identical ports
}  // namespace alg

namespace detail {
template <class T, class S> BATCHLAS_DLA_INLINE T tag_value(S s) {
    if constexpr (std::is_same_v<S, One>) return Arith<T>::one();
    else if constexpr (std::is_same_v<S, Zero>) return Arith<T>::zero();
    else if constexpr (std::is_same_v<S, NegOne>) return Arith<T>::neg(Arith<T>::one());
    else return static_cast<T>(s.v);
}

template <class X> inline constexpr int frag_n_v = 0;  // 0: not a Frag
template <class T, int N, class L> inline constexpr int frag_n_v<Frag<T, N, L>> = N;
template <class X> inline constexpr bool is_frag_v = frag_n_v<X> > 0;

template <class X> struct owned_elem { using type = typename X::value_type; };
template <class T, int N, class L> struct owned_elem<Frag<T, N, L>> { using type = T; };

template <class X> BATCHLAS_DLA_INLINE decltype(auto) at(X& x, int i) {
    if constexpr (is_frag_v<std::remove_cv_t<X>>) return x[i];
    else return x(i);
}

// f(i) for every element of x: fully unrolled over a Frag (constant indices keep it in
// registers), a plain loop over a Vec.
template <class X, class F> BATCHLAS_DLA_INLINE void for_each_elem(const X& x, F&& f) {
    if constexpr (is_frag_v<X>) {
        constexpr int n = frag_n_v<X>;
        BATCHLAS_DLA_UNROLL_FULL
        for (int i = 0; i < n; ++i) f(i);
    } else {
        const int n = x.size();
        for (int i = 0; i < n; ++i) f(i);
    }
}

template <Uplo UL, bool SetDiag, class Ex, class AS, class BS, class MA>
BATCHLAS_DLA_INLINE void laset_impl(const Ex& ex, AS alpha, BS beta, const MA& a) {
    using T = typename MA::value_type;
    const T av = tag_value<T>(as_tag(alpha));
    const int m = a.rows(), n = a.cols();
    for (int c = 0; c < n; ++c) {
        for (int r = ex.lid(); r < m; r += Ex::size) {
            if (r == c) {
                if constexpr (SetDiag) a(r, c) = tag_value<T>(as_tag(beta));
            } else if ((UL == Uplo::Lower) == (r > c)) {
                a(r, c) = av;
            }
        }
    }
}
}  // namespace detail

// y := op(A) x for an n x n triangular A (only the UL triangle is read; Diag::Unit never reads the
// diagonal). Out of place: y must not alias x. alg::RowOwner: item r forms y(r) from row r of
// op(A) in ascending column order (no reduction), so y is in Cyclic layout. A later call that reads
// y(r) back as A(r, :) runs on the item that wrote it: a column-by-column triangular recurrence
// (larft's T) on one executor needs no barrier.
template <Uplo UL, Transpose TR, Diag DG, class Alg = alg::RowOwner, class Ex, class MA, class X, class Y>
BATCHLAS_DLA_INLINE void trmv(const Ex& ex, const MA& a, const X& xp, const Y& y) {
    using T = typename MA::value_type;
    using Ar = Arith<T>;
    static_assert(std::is_same_v<Alg, alg::RowOwner>, "v2::trmv: alg::RowOwner is the only algorithm");
    detail::require_published<Ex, X>();
    const auto& x = detail::unpub(xp);
    constexpr bool upper = (UL == Uplo::Upper) == (TR == Transpose::NoTrans);  // op(A) upper
    auto e = [&](int r, int c) -> T {
        if constexpr (TR == Transpose::NoTrans) return a(r, c);
        else if constexpr (TR == Transpose::Trans) return a(c, r);
        else return Ar::conj(a(c, r));
    };
    const int n = a.rows();
    for (int r = ex.lid(); r < n; r += Ex::size) {
        T acc = Ar::zero();
        if constexpr (!upper) {
#pragma unroll 4
            for (int c = 0; c < r; ++c) Ar::mac(acc, e(r, c), x(c));
        }
        if constexpr (DG == Diag::Unit) acc = Ar::add(acc, static_cast<T>(x(r)));
        else Ar::mac(acc, e(r, r), x(r));
        if constexpr (upper) {
#pragma unroll 4
            for (int c = r + 1; c < n; ++c) Ar::mac(acc, e(r, c), x(c));
        }
        y(r) = acc;
    }
}

// LAPACK laset on an m x n A: the strictly UL part := alpha, the diagonal := beta.
template <Uplo UL, class Ex, class AS, class BS, class MA>
BATCHLAS_DLA_INLINE void laset(const Ex& ex, AS alpha, BS beta, const MA& a) {
    detail::laset_impl<UL, true>(ex, alpha, beta, a);
}
// The strictly UL part := alpha; the diagonal is not touched.
template <Uplo UL, class Ex, class AS, class MA> BATCHLAS_DLA_INLINE void laset(const Ex& ex, AS alpha, const MA& a) {
    detail::laset_impl<UL, false>(ex, alpha, Zero{}, a);
}

// H = I - tau v v^H applied to a vector the calling item owns whole (Thread executor):
// Side::Left  x := H x  (x a column; w = v^H x),  Side::Right  x := x H  (x a row; w = x v).
// v is read in full, so it must be Published (staged under the caller's barrier). x is a
// Frag<T, N> (fully unrolled, v then has N elements) or a Vec of v's length. alg::Fused updates
// with one fma per element; alg::Unfused rounds tau w v first (v1 ger's arithmetic).
template <Side S, class Alg = alg::Fused, class Tau, class V, class X>
BATCHLAS_DLA_INLINE void larf(const Thread&, Tau tau, const V& vp, X&& x) {
    using Xd = std::remove_cvref_t<X>;
    using T = typename detail::owned_elem<Xd>::type;
    using Ar = Arith<T>;
    static_assert(is_published_v<V>, "v2::larf: v is read whole; pass assume_published(v) after the barrier that staged it");
    static_assert(std::is_same_v<Alg, alg::Fused> || std::is_same_v<Alg, alg::Unfused>, "v2::larf: alg::Fused or alg::Unfused");
    const auto& v = detail::unpub(vp);
    T w = Ar::zero();
    detail::for_each_elem(x, [&](int i) {
        if constexpr (S == Side::Left) Ar::mac_conj(w, v(i), detail::at(x, i));
        else Ar::mac(w, detail::at(x, i), v(i));
    });
    const T g = Ar::mul(static_cast<T>(tau), w);
    constexpr bool fused = std::is_same_v<Alg, alg::Fused>;
    detail::for_each_elem(x, [&](int i) {
        T& xi = detail::at(x, i);
        if constexpr (S == Side::Left) {  // x -= v g
            if constexpr (fused) Ar::mac(xi, Ar::neg(v(i)), g);
            else xi = Ar::sub(xi, Ar::mul(v(i), g));
        } else {  // x -= g conj(v)
            if constexpr (fused) Ar::mac(xi, Ar::neg(g), Ar::conj(v(i)));
            else xi = Ar::sub(xi, Ar::mul(g, Ar::conj(v(i))));
        }
    });
}

}  // namespace batchlas::device::v2
