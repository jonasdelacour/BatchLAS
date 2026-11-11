// SPDX-License-Identifier: MIT
#pragma once
// Device LA ops private to the elementwise-glue kernels (src/util/elementwise.cc,
// transpose.cc, src/kernels/common/triangular_expand.hh): geam, trimask on a runtime or static
// triangle, tri_expand and the SLM tile moves of transpose / mirror. Each waits for its core
// form (lacpy with a run-time part, a diagonal offset and a fill outside the part; 2-D map and
// axpby; tile-owned lacpy / mirror); lange, conj_if and force_real are core already.

#include <type_traits>

#include <sycl/sycl.hpp>

#include <batchlas/device.hh>

namespace batchlas::device::glue {

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
template <class U> BATCHLAS_DEVICE_INLINE constexpr bool is_lower([[maybe_unused]] U u) {
    if constexpr (std::is_same_v<U, Uplo>) return u == Uplo::Lower;
    else return U::value == Uplo::Lower;
}
template <class D> BATCHLAS_DEVICE_INLINE constexpr bool is_unit([[maybe_unused]] D d) {
    if constexpr (std::is_same_v<D, Diag>) return d == Diag::Unit;
    else return D::value == Diag::Unit;
}
}  // namespace impl

// Upper keeps j - i >= k, Lower keeps j - i <= k (k = 0: the triangle with its diagonal).
template <class U = Uplo> struct Tri {
    U uplo{};
    int k = 0;
};
template <class U> BATCHLAS_DEVICE_INLINE Tri<U> tri(U uplo, int k = 0) { return {uplo, k}; }

// ---- 2-D elementwise (Grid2D or any Cyclic executor, core for_each_owned_2d) ----------------
// out(i, j) = f(ins(i, j)...)
template <class Ex, class Out, class F, class... In>
BATCHLAS_DEVICE_INLINE void map2d(const Ex& ex, const Out& out, F&& f, const In&... ins) {
    for_each_owned_2d(ex, out.rows(), out.cols(), [&](int i, int j) { out(i, j) = f(ins(i, j)...); });
}

// C := A inside r, 0 outside (tril / triu with an offset). A may alias C.
template <class Ex, class U, class MA, class MC> BATCHLAS_DEVICE_INLINE void trimask(const Ex& ex, const Tri<U>& r, const MA& a, const MC& c) {
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
BATCHLAS_DEVICE_INLINE void tri_expand(const Ex& ex, U uplo, D diag, const MA& a, const MO& o) {
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
BATCHLAS_DEVICE_INLINE void geam(const Ex& ex, AS alpha_, const MA& a, BS beta_, const MB& b, const MC& c) {
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
template <int TN, class Ex, class F> BATCHLAS_DEVICE_INLINE void tile_cols(const Ex& ex, F&& f) {
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
template <int TN, class Ex, class MA, class MT> BATCHLAS_DEVICE_INLINE void tile_load(const Ex& ex, const MA& a, int i0, int j0, const MT& t) {
    using T = typename MT::value_type;
    const int r = ex.lrow();
    impl::tile_cols<TN>(ex, [&](int c) {
        const int i = i0 + r, j = j0 + c;
        t(r, c) = (i < a.rows() && j < a.cols()) ? static_cast<T>(a(i, j)) : Arith<T>::zero();
    });
}
// t(c, r) = A(i0 + r, j0 + c): the tile stored transposed.
template <int TN, class Ex, class MA, class MT> BATCHLAS_DEVICE_INLINE void tile_load_trans(const Ex& ex, const MA& a, int i0, int j0, const MT& t) {
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
template <int TN, class Ex, class MT, class MB> BATCHLAS_DEVICE_INLINE void tile_store(const Ex& ex, const MT& t, const MB& b, int i0, int j0) {
    const int r = ex.lrow();
    impl::tile_cols<TN>(ex, [&](int c) {
        const int i = i0 + r, j = j0 + c;
        if (i < b.rows() && j < b.cols()) b(i, j) = t(r, c);
    });
}
// The off-diagonal pair of a mirror in one column loop: B(i0 + r, j0 + c) = t(r, c) and
// B(j0 + r, i0 + c) = op(t(c, r)).
template <bool Conj, int TN, class Ex, class MT, class MB>
BATCHLAS_DEVICE_INLINE void tile_store_pair(const Ex& ex, const MT& t, const MB& b, int i0, int j0) {
    using T = typename MB::value_type;
    const int r = ex.lrow();
    impl::tile_cols<TN>(ex, [&](int c) {
        if (i0 + r < b.rows() && j0 + c < b.cols()) b(i0 + r, j0 + c) = t(r, c);
        if (j0 + r < b.rows() && i0 + c < b.cols()) b(j0 + r, i0 + c) = device::conj_if<Conj>(static_cast<T>(t(c, r)));
    });
}
// A diagonal tile (i0 == j0) of a mirror: the referenced member of each pair, op(t(c, r))
// opposite; with Conj the diagonal is forced real.
template <bool Conj, int TN, class Ex, class U, class MT, class MB>
BATCHLAS_DEVICE_INLINE void tile_store_diag(const Ex& ex, U uplo, const MT& t, const MB& b, int i0) {
    using T = typename MB::value_type;
    const bool lower = impl::is_lower(uplo);
    const int r = ex.lrow();
    impl::tile_cols<TN>(ex, [&](int c) {
        const int i = i0 + r, j = i0 + c;
        if (i >= b.rows() || j >= b.cols()) return;
        const bool referenced = lower ? (r >= c) : (r <= c);
        T value = referenced ? static_cast<T>(t(r, c)) : device::conj_if<Conj>(static_cast<T>(t(c, r)));
        if constexpr (Conj) {
            if (i == j) value = device::force_real(value);
        }
        b(i, j) = value;
    });
}

}  // namespace batchlas::device::glue
