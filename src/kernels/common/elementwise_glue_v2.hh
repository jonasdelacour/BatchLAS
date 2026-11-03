// SPDX-License-Identifier: MIT
#pragma once
// Device LA v2 ops private to the elementwise-glue kernels (src/util/elementwise.cc, norm.cc,
// transpose.cc, src/kernels/common/triangular_expand.hh): geam, trimask on a runtime or static
// triangle, tri_expand, the SLM tile moves of transpose / mirror, and lange. Everything they
// share with other families (Grid2D, for_each_owned_2d, mat_wide, modulus, abs2, reductions)
// is core; an op moves to core when a second family needs it.

#include <type_traits>

#include <sycl/sycl.hpp>

#include <batchlas/blas/device/v2/dla.hh>

namespace batchlas::device::v2::glue {

// Uplo / Diag: a UploC / DiagC tag folds at compile time, a plain enum is read at run time
// (the legacy spelling, so a 1:1 port stays 1:1).
template <Uplo U> struct UploC {
    static constexpr Uplo value = U;
};
template <Diag D> struct DiagC {
    static constexpr Diag value = D;
};
using LowerC = UploC<Uplo::Lower>;
using UpperC = UploC<Uplo::Upper>;

namespace impl {
template <class U> BATCHLAS_DLA_INLINE constexpr bool is_lower([[maybe_unused]] U u) {
    if constexpr (std::is_same_v<U, Uplo>) return u == Uplo::Lower;
    else return U::value == Uplo::Lower;
}
template <class D> BATCHLAS_DLA_INLINE constexpr bool is_unit([[maybe_unused]] D d) {
    if constexpr (std::is_same_v<D, Diag>) return d == Diag::Unit;
    else return D::value == Diag::Unit;
}
template <bool Conj, class T> BATCHLAS_DLA_INLINE T conj_if(T a) {
    if constexpr (Conj) return Arith<T>::conj(a);
    else return a;
}
// (re(a), 0): a Hermitian diagonal.
template <class T> BATCHLAS_DLA_INLINE T real_only(T a) {
    using A = Arith<T>;
    if constexpr (A::is_complex) return A::make(A::re(a), real_t<T>(0));
    else return a;
}
}  // namespace impl

// Upper keeps j - i >= k, Lower keeps j - i <= k (k = 0: the triangle with its diagonal).
template <class U = Uplo> struct Tri {
    U uplo{};
    int k = 0;
};
template <class U> BATCHLAS_DLA_INLINE Tri<U> tri(U uplo, int k = 0) { return {uplo, k}; }

// ---- 2-D elementwise (Grid2D or any Cyclic executor, core for_each_owned_2d) ----------------
// out(i, j) = f(ins(i, j)...)
template <class Ex, class Out, class F, class... In>
BATCHLAS_DLA_INLINE void map2d(const Ex& ex, const Out& out, F&& f, const In&... ins) {
    for_each_owned_2d(ex, out.rows(), out.cols(), [&](int i, int j) { out(i, j) = f(ins(i, j)...); });
}

// C := A inside r, 0 outside (tril / triu with an offset). A may alias C.
template <class Ex, class U, class MA, class MC> BATCHLAS_DLA_INLINE void trimask(const Ex& ex, const Tri<U>& r, const MA& a, const MC& c) {
    using T = typename MC::value_type;
    const bool lower = impl::is_lower(r.uplo);
    for_each_owned_2d(ex, c.rows(), c.cols(), [&](int i, int j) {
        const bool keep = lower ? j - i <= r.k : j - i >= r.k;
        c(i, j) = keep ? static_cast<T>(a(i, j)) : Arith<T>::zero();
    });
}

// O := the dense op(A) of a triangular A stored in uplo: the stored strict triangle, zeros
// opposite, A's diagonal or ones (Diag::Unit: the diagonal is never read). Storage outside the
// referenced triangle is never read.
template <class Ex, class U, class D, class MA, class MO>
BATCHLAS_DLA_INLINE void tri_expand(const Ex& ex, U uplo, D diag, const MA& a, const MO& o) {
    using T = typename MO::value_type;
    const bool lower = impl::is_lower(uplo);
    const bool unit = impl::is_unit(diag);
    for_each_owned_2d(ex, o.rows(), o.cols(), [&](int i, int j) {
        T value;
        if (i == j) {
            value = unit ? Arith<T>::one() : static_cast<T>(a(i, j));
        } else if (lower ? (i > j) : (i < j)) {
            value = a(i, j);
        } else {
            value = Arith<T>::zero();
        }
        o(i, j) = value;
    });
}

// C := alpha A + beta B (explicit arithmetic); beta == Zero never reads B. C may alias A or B.
template <class Ex, class AS, class MA, class BS, class MB, class MC>
BATCHLAS_DLA_INLINE void geam(const Ex& ex, AS alpha_, const MA& a, BS beta_, const MB& b, const MC& c) {
    using T = typename MC::value_type;
    const auto alpha = as_tag(alpha_);
    const auto beta = as_tag(beta_);
    for_each_owned_2d(ex, c.rows(), c.cols(), [&](int i, int j) {
        if constexpr (std::is_same_v<tag_t<BS>, Zero>) c(i, j) = tag_mul(alpha, static_cast<T>(a(i, j)));
        else c(i, j) = Arith<T>::add(tag_mul(alpha, static_cast<T>(a(i, j))), tag_mul(beta, static_cast<T>(b(i, j))));
    });
}

// ---- SLM tiles on a Grid2D with a static group shape ----------------------------------------
// The item owns tile row r = lrow() and columns c = lcol() + s * GC < TN. The tile t is any
// Mat (Mem::Local); elements past the global operand's edge load as zero and are not stored.
// The caller owns the barrier between a load and a store of other items' elements.
namespace impl {
template <int TN, class Ex, class F> BATCHLAS_DLA_INLINE void tile_cols(const Ex& ex, F&& f) {
    constexpr int GC = Ex::static_group_cols;
    static_assert(GC > 0, "glue tile ops: Grid2D needs a static group shape");
    if constexpr (TN == GC) {
        f(ex.lcol());
    } else {
        for (int c = ex.lcol(); c < TN; c += GC) f(c);
    }
}
}  // namespace impl

// t(r, c) = A(i0 + r, j0 + c)
template <int TN, class Ex, class MA, class MT> BATCHLAS_DLA_INLINE void tile_load(const Ex& ex, const MA& a, int i0, int j0, const MT& t) {
    using T = typename MT::value_type;
    const int r = ex.lrow();
    impl::tile_cols<TN>(ex, [&](int c) {
        const int i = i0 + r, j = j0 + c;
        t(r, c) = (i < a.rows() && j < a.cols()) ? static_cast<T>(a(i, j)) : Arith<T>::zero();
    });
}
// t(c, r) = A(i0 + r, j0 + c): the tile stored transposed.
template <int TN, class Ex, class MA, class MT> BATCHLAS_DLA_INLINE void tile_load_trans(const Ex& ex, const MA& a, int i0, int j0, const MT& t) {
    using T = typename MT::value_type;
    const int r = ex.lrow();
    impl::tile_cols<TN>(ex, [&](int c) {
        const int i = i0 + r, j = j0 + c;
        T v = Arith<T>::zero();
        if (i < a.rows() && j < a.cols()) v = a(i, j);
        t(c, r) = v;
    });
}
// B(i0 + r, j0 + c) = t(r, c)
template <int TN, class Ex, class MT, class MB> BATCHLAS_DLA_INLINE void tile_store(const Ex& ex, const MT& t, const MB& b, int i0, int j0) {
    const int r = ex.lrow();
    impl::tile_cols<TN>(ex, [&](int c) {
        const int i = i0 + r, j = j0 + c;
        if (i < b.rows() && j < b.cols()) b(i, j) = t(r, c);
    });
}
// The off-diagonal pair of a mirror in one column loop: B(i0 + r, j0 + c) = t(r, c) and
// B(j0 + r, i0 + c) = op(t(c, r)).
template <bool Conj, int TN, class Ex, class MT, class MB>
BATCHLAS_DLA_INLINE void tile_store_pair(const Ex& ex, const MT& t, const MB& b, int i0, int j0) {
    using T = typename MB::value_type;
    const int r = ex.lrow();
    impl::tile_cols<TN>(ex, [&](int c) {
        if (i0 + r < b.rows() && j0 + c < b.cols()) b(i0 + r, j0 + c) = t(r, c);
        if (j0 + r < b.rows() && i0 + c < b.cols()) b(j0 + r, i0 + c) = impl::conj_if<Conj>(static_cast<T>(t(c, r)));
    });
}
// A diagonal tile (i0 == j0) of a mirror: the referenced member of each pair, op(t(c, r))
// opposite; with Conj the diagonal is forced real.
template <bool Conj, int TN, class Ex, class U, class MT, class MB>
BATCHLAS_DLA_INLINE void tile_store_diag(const Ex& ex, U uplo, const MT& t, const MB& b, int i0) {
    using T = typename MB::value_type;
    const bool lower = impl::is_lower(uplo);
    const int r = ex.lrow();
    impl::tile_cols<TN>(ex, [&](int c) {
        const int i = i0 + r, j = i0 + c;
        if (i >= b.rows() || j >= b.cols()) return;
        const bool referenced = lower ? (r >= c) : (r <= c);
        T value = referenced ? static_cast<T>(t(r, c)) : impl::conj_if<Conj>(static_cast<T>(t(c, r)));
        if constexpr (Conj) {
            if (i == j) value = impl::real_only(value);
        }
        b(i, j) = value;
    });
}

// ---- lange --------------------------------------------------------------------------------
// LAPACK lange on a Cyclic executor (|.| is the modulus). Max / Fro walk the flat column-major
// order; One gives each item whole columns, Inf whole rows (serial sums, no SLM), then one
// Scope S max. Fro: alg::Unscaled = sqrt(sum |a|^2), one reduction (overflows past
// sqrt(max)); alg::Scaled = the two-pass lassq. Max keeps std::max(|a|, m)'s NaN behaviour.
template <NormType NT, class Alg = alg::Unscaled, Scope S = Scope::Auto, class Ex, class MA>
BATCHLAS_DLA_INLINE auto lange(const Ex& ex, const MA& a) {
    using T = typename MA::value_type;
    using R = real_t<T>;
    const int m = a.rows(), n = a.cols();
    if constexpr (NT == NormType::Max) {
        R mx = R(0);
        for_each_owned_2d(ex, m, n, [&](int i, int j) {
            const R v = modulus(static_cast<T>(a(i, j)));
            mx = v < mx ? mx : v;
        });
        return detail::scope_reduce<S>(ex, mx, sycl::maximum<R>());
    } else if constexpr (NT == NormType::One || NT == NormType::Inf) {
        constexpr bool one = NT == NormType::One;
        const int outer = one ? n : m, inner = one ? m : n;
        R mx = R(0);
        for (int k = ex.lid(); k < outer; k += Ex::size) {  // caller-owned shape (core 2.2 rule)
            R s = R(0);
            for (int l = 0; l < inner; ++l) s += modulus(static_cast<T>(one ? a(l, k) : a(k, l)));
            mx = sycl::maximum<R>()(mx, s);
        }
        return detail::scope_reduce<S>(ex, mx, sycl::maximum<R>());
    } else {
        static_assert(NT == NormType::Frobenius, "glue::lange: Max, One, Inf or Frobenius");
        if constexpr (std::is_same_v<Alg, alg::Unscaled>) {
            R ss = R(0);
            for_each_owned_2d(ex, m, n, [&](int i, int j) { ss += abs2(static_cast<T>(a(i, j))); });
            return static_cast<R>(sycl::sqrt(detail::scope_reduce<S>(ex, ss)));
        } else {
            static_assert(std::is_same_v<Alg, alg::Scaled>, "glue::lange: Frobenius is alg::Unscaled or alg::Scaled");
            using A = Arith<T>;
            R sc = R(0);
            auto upd = [&](R v) { sc = (v > sc || sycl::isnan(v)) ? v : sc; };
            for_each_owned_2d(ex, m, n, [&](int i, int j) {
                const T v = a(i, j);
                upd(sycl::fabs(A::re(v)));
                if constexpr (A::is_complex) upd(sycl::fabs(A::im(v)));
            });
            const R scale = detail::scope_reduce<S>(ex, sc, sycl::maximum<R>());
            if (sycl::isinf(scale)) return scale;
            const R inv = scale > R(0) ? R(1) / scale : R(1);
            R ss = R(0);
            for_each_owned_2d(ex, m, n, [&](int i, int j) {
                const T v = a(i, j);
                const R tr = A::re(v) * inv;
                ss += tr * tr;
                if constexpr (A::is_complex) {
                    const R ti = A::im(v) * inv;
                    ss += ti * ti;
                }
            });
            ss = detail::scope_reduce<S>(ex, ss);
            return scale > R(0) ? scale * sycl::sqrt(ss) : sycl::sqrt(ss);
        }
    }
}

}  // namespace batchlas::device::v2::glue
