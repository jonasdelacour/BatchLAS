// SPDX-License-Identifier: MIT
#pragma once
// Device LA v2, Wave 4 glue (namespace batchlas::device::v2, additive to the core):
//   - Grid2D + batched2d_range: the batched 2-D launch (2-D indices, no % or /, the batch
//     dimension grid-strided past the dim-0 group limit) and a flat column-major walk of a
//     matrix on any Cyclic executor (one division per item, a carry per step);
//   - 2-D LAPACK auxiliaries on either: lacpy laset trimask tri_expand geam dgmm lapmt mirror;
//   - SLM tile moves for transpose / mirror (the caller owns the barrier between load and store);
//   - lange (Max One Inf Fro) and absrange reductions;
//   - RowFrag moves of the register tier (lane r owns row r), partition bcast / argmax_pair;
//   - scalar LAPACK: lartg, larfg_scalars<Conv> (two conventions), lae2, laev2, las2, and larfg
//     on a v2 executor.
// Uplo / Diag arguments follow Ext / Dyn: a UploC / DiagC tag folds at compile time, a plain
// enum is read at run time (the legacy spelling, kept so 1:1 ports stay 1:1).

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <batchlas/blas/device/v2/dla.hh>

#ifndef BATCHLAS_DLA_UNROLL_FULL
#if defined(__ACPP__) || defined(__HIPSYCL__)
#define BATCHLAS_DLA_UNROLL_FULL _Pragma("clang loop unroll(full)")  // SSCP JIT: bare unroll is partial
#else
#define BATCHLAS_DLA_UNROLL_FULL _Pragma("unroll")
#endif
#endif

namespace batchlas::device::v2 {

// ---- static / dynamic Uplo, Diag ----------------------------------------------------------
template <Uplo U> struct UploC {
    static constexpr Uplo value = U;
};
template <Diag D> struct DiagC {
    static constexpr Diag value = D;
};
using LowerC = UploC<Uplo::Lower>;
using UpperC = UploC<Uplo::Upper>;

namespace detail {
template <class U> BATCHLAS_DLA_INLINE constexpr bool is_lower(U u) {
    if constexpr (std::is_same_v<U, Uplo>) return u == Uplo::Lower;
    else return U::value == Uplo::Lower;
}
template <class D> BATCHLAS_DLA_INLINE constexpr bool is_unit(D d) {
    if constexpr (std::is_same_v<D, Diag>) return d == Diag::Unit;
    else return D::value == Diag::Unit;
}
#ifndef BATCHLAS_DLA_V2_INTEG_DETAIL_TAG_VALUE  // integ/dla-v2-all: defined by several ext headers
#define BATCHLAS_DLA_V2_INTEG_DETAIL_TAG_VALUE
template <class T, class S> BATCHLAS_DLA_INLINE T tag_value(S s) {
    if constexpr (std::is_same_v<S, One>) return Arith<T>::one();
    else if constexpr (std::is_same_v<S, Zero>) return Arith<T>::zero();
    else if constexpr (std::is_same_v<S, NegOne>) return Arith<T>::neg(Arith<T>::one());
    else return static_cast<T>(s.v);
}
#endif  // BATCHLAS_DLA_V2_INTEG_DETAIL_TAG_VALUE
}  // namespace detail

// ---- scalar element helpers ---------------------------------------------------------------
// c ? a : b component by component: a ?: over a complex aggregate is not a select, and a
// register array read through one leaves the register file (docs/perf/potrf.md).
template <class T> BATCHLAS_DLA_INLINE T sel(bool c, T a, T b) {
    using A = Arith<T>;
    if constexpr (A::is_complex) return A::make(c ? A::re(a) : A::re(b), c ? A::im(a) : A::im(b));
    else return c ? a : b;
}
// (re(a), 0): a Hermitian diagonal.
template <class T> BATCHLAS_DLA_INLINE T real_only(T a) {
    using A = Arith<T>;
    if constexpr (A::is_complex) return A::make(A::re(a), real_t<T>(0));
    else return a;
}
template <bool Conj, class T> BATCHLAS_DLA_INLINE T conj_if(T a) {
    if constexpr (Conj) return Arith<T>::conj(a);
    else return a;
}
// |a|: hypot for complex (LAPACK abs, as lange), not BLAS abs1.
template <class T> BATCHLAS_DLA_INLINE real_t<T> modulus(T a) {
    using A = Arith<std::remove_cv_t<T>>;
    if constexpr (A::is_complex) return sycl::hypot(A::re(a), A::im(a));
    else return sycl::fabs(a);
}
#ifndef BATCHLAS_DLA_V2_INTEG_ABS2  // integ/dla-v2-all: defined by several ext headers
#define BATCHLAS_DLA_V2_INTEG_ABS2
template <class T> BATCHLAS_DLA_INLINE real_t<T> abs2(T a) {
    using A = Arith<std::remove_cv_t<T>>;
    if constexpr (A::is_complex) return A::re(a) * A::re(a) + A::im(a) * A::im(a);
    else return a * a;
}
#endif  // BATCHLAS_DLA_V2_INTEG_ABS2

// ---- the batched 2-D executor --------------------------------------------------------------
// Local range (LB, GC, GR), dimension 2 fastest, so consecutive items walk a column and both
// sides of a copy coalesce. Item (row(), col()) of batch items batch() + k * global_range(0).
// A template value of 0 is read from the launch.
template <int GR = 0, int GC = 0, int LB = 0> struct Grid2D {
    static constexpr int static_group_rows = GR, static_group_cols = GC, static_group_batch = LB;
    sycl::nd_item<3> it;

    BATCHLAS_DLA_INLINE int group_rows() const {
        if constexpr (GR > 0) return GR;
        else return static_cast<int>(it.get_local_range(2));
    }
    BATCHLAS_DLA_INLINE int group_cols() const {
        if constexpr (GC > 0) return GC;
        else return static_cast<int>(it.get_local_range(1));
    }
    BATCHLAS_DLA_INLINE int lrow() const { return static_cast<int>(it.get_local_id(2)); }
    BATCHLAS_DLA_INLINE int lcol() const { return static_cast<int>(it.get_local_id(1)); }
    BATCHLAS_DLA_INLINE int row0() const { return static_cast<int>(it.get_group(2)) * group_rows(); }
    BATCHLAS_DLA_INLINE int col0() const { return static_cast<int>(it.get_group(1)) * group_cols(); }
    BATCHLAS_DLA_INLINE int row() const { return static_cast<int>(it.get_global_id(2)); }
    BATCHLAS_DLA_INLINE int col() const { return static_cast<int>(it.get_global_id(1)); }
    // The first batch item of this item (the only one when the launch covers the batch).
    BATCHLAS_DLA_INLINE int batch() const {
        if constexpr (LB == 1) return static_cast<int>(it.get_group(0));
        else return static_cast<int>(it.get_global_id(0));
    }
    template <class F> BATCHLAS_DLA_INLINE void for_each_batch(int nbatch, F&& f) const {
        const int step = static_cast<int>(it.get_global_range(0));
        for (int b = batch(); b < nbatch; b += step) f(b);
    }
    BATCHLAS_DLA_INLINE void barrier() const { it.barrier(sycl::access::fence_space::local_space); }
};

template <class Ex> inline constexpr bool is_grid2d_v = false;
template <int GR, int GC, int LB> inline constexpr bool is_grid2d_v<Grid2D<GR, GC, LB>> = true;

// Host side. Rows: the smallest power of two >= min(m, 32); columns: the smallest power of two
// >= min(n, items / rows); batch items fill the rest, so tiny matrices still fill a group.
struct Shape2D {
    int gr, gc, lb;
};
inline Shape2D batched2d_shape(int m, int n, int items = 256) {
    int gr = 1;
    while (gr < 32 && gr < m) gr *= 2;
    int gc = 1;
    while (gc < items / gr && gc < n) gc *= 2;
    return {gr, gc, items / (gr * gc)};
}
// Dimension 0 holds at most max_groups0 groups (CUDA gridDim.z); Grid2D::for_each_batch walks
// the rest. m, n, batch >= 1.
inline sycl::nd_range<3> batched2d_range(int batch, int m, int n, Shape2D s, int max_groups0 = 65535) {
    const auto up = [](int x, int q) { return static_cast<std::size_t>((x + q - 1) / q) * static_cast<std::size_t>(q); };
    const int groups0 = (batch + s.lb - 1) / s.lb;
    const int g0 = groups0 < max_groups0 ? groups0 : max_groups0;
    return {sycl::range<3>(static_cast<std::size_t>(g0) * s.lb, up(n, s.gc), up(m, s.gr)),
            sycl::range<3>(static_cast<std::size_t>(s.lb), static_cast<std::size_t>(s.gc), static_cast<std::size_t>(s.gr))};
}

// f(i, j) for each element of an m x n operand this item owns. Grid2D: its one element.
// Cyclic executors: flat column-major index k = lid + s * size (the legacy grid-stride order,
// so per-item partial sums keep their order) with one division per item and a carry per step.
template <class Ex, class F> BATCHLAS_DLA_INLINE void for_each_owned2d(const Ex& ex, int m, int n, F&& f) {
    if constexpr (is_grid2d_v<Ex>) {
        const int i = ex.row(), j = ex.col();
        if (i < m && j < n) f(i, j);
    } else if constexpr (Ex::size == 1) {
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) f(i, j);
    } else {
        if (m <= 0) return;
        constexpr int S = Ex::size;
        const int dj = S / m, di = S - dj * m;
        int j = ex.lid() / m;
        int i = ex.lid() - j * m;
        while (j < n) {
            f(i, j);
            i += di;
            j += dj;
            if (i >= m) {
                i -= m;
                ++j;
            }
        }
    }
}

// A whole user matrix with 64-bit element offsets (size_t(j) * ld + i), the index math of the
// hand-written whole-matrix kernels, so a 1:1 port keeps their SASS. Mat's int32 j * ld is the
// default for a kernel's own tiles and sub-views.
template <class T, Mem M = Mem::Global> struct MatW {
    using value_type = std::remove_cv_t<T>;
    using layout = Cyclic;
    static constexpr Mem mem = M;
    T* p;
    int m, n;
    std::size_t ld;
    BATCHLAS_DLA_INLINE T& operator()(int i, int j) const {
        return p[static_cast<std::size_t>(j) * ld + static_cast<std::size_t>(i)];
    }
    BATCHLAS_DLA_INLINE int rows() const { return m; }
    BATCHLAS_DLA_INLINE int cols() const { return n; }
};
template <Mem M = Mem::Global, class T> BATCHLAS_DLA_INLINE MatW<T, M> matw(T* p, int m, int n, std::size_t ld) {
    return {p, m, n, ld};
}

// ---- regions --------------------------------------------------------------------------------
struct All {};
// Upper keeps j - i >= k, Lower keeps j - i <= k (k = 0: the triangle with its diagonal;
// k = +1 / -1: the strict upper / lower triangle).
template <class U = Uplo> struct Tri {
    U uplo{};
    int k = 0;
};
template <class U> BATCHLAS_DLA_INLINE Tri<U> tri(U uplo, int k = 0) { return {uplo, k}; }

namespace detail {
BATCHLAS_DLA_INLINE constexpr bool in_region(All, int, int) { return true; }
template <class U> BATCHLAS_DLA_INLINE bool in_region(const Tri<U>& r, int i, int j) {
    return is_lower(r.uplo) ? j - i <= r.k : j - i >= r.k;
}
}  // namespace detail

// ---- 2-D elementwise (Grid2D or any Cyclic executor) -----------------------------------------
// out(i, j) = f(ins(i, j)...)
template <class Ex, class Out, class F, class... In>
BATCHLAS_DLA_INLINE void map2d(const Ex& ex, const Out& out, F&& f, const In&... ins) {
    for_each_owned2d(ex, out.rows(), out.cols(), [&](int i, int j) { out(i, j) = f(ins(i, j)...); });
}

// B := A on the region r (LAPACK lacpy); B outside r is not written.
template <class Ex, class R, class MA, class MB> BATCHLAS_DLA_INLINE void lacpy(const Ex& ex, const R& r, const MA& a, const MB& b) {
    for_each_owned2d(ex, b.rows(), b.cols(), [&](int i, int j) {
        if (detail::in_region(r, i, j)) b(i, j) = a(i, j);
    });
}

// A := alpha off the diagonal and beta on it, on the region r (LAPACK laset).
template <class Ex, class R, class AS, class BS, class MA>
BATCHLAS_DLA_INLINE void laset(const Ex& ex, const R& r, AS alpha_, BS beta_, const MA& a) {
    using T = typename MA::value_type;
    const T alpha = detail::tag_value<T>(as_tag(alpha_));
    const T beta = detail::tag_value<T>(as_tag(beta_));
    for_each_owned2d(ex, a.rows(), a.cols(), [&](int i, int j) {
        if (detail::in_region(r, i, j)) a(i, j) = i == j ? beta : alpha;
    });
}

// C := A inside r, 0 outside (tril / triu with an offset). A may alias C.
template <class Ex, class R, class MA, class MC> BATCHLAS_DLA_INLINE void trimask(const Ex& ex, const R& r, const MA& a, const MC& c) {
    using T = typename MC::value_type;
    for_each_owned2d(ex, c.rows(), c.cols(), [&](int i, int j) {
        c(i, j) = detail::in_region(r, i, j) ? static_cast<T>(a(i, j)) : Arith<T>::zero();
    });
}

// O := the dense op(A) of a triangular A stored in uplo: the stored strict triangle, zeros
// opposite, A's diagonal or ones (Diag::Unit: the diagonal is never read). Storage outside the
// referenced triangle is never read.
template <class Ex, class U, class D, class MA, class MO>
BATCHLAS_DLA_INLINE void tri_expand(const Ex& ex, U uplo, D diag, const MA& a, const MO& o) {
    using T = typename MO::value_type;
    const bool lower = detail::is_lower(uplo);
    const bool unit = detail::is_unit(diag);
    for_each_owned2d(ex, o.rows(), o.cols(), [&](int i, int j) {
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

// C := alpha A + beta B; beta == Zero never reads B. C may alias A or B.
template <class Ex, class AS, class MA, class BS, class MB, class MC>
BATCHLAS_DLA_INLINE void geam(const Ex& ex, AS alpha_, const MA& a, BS beta_, const MB& b, const MC& c) {
    using T = typename MC::value_type;
    const auto alpha = as_tag(alpha_);
    const auto beta = as_tag(beta_);
    for_each_owned2d(ex, c.rows(), c.cols(), [&](int i, int j) {
        if constexpr (std::is_same_v<tag_t<BS>, Zero>) c(i, j) = tag_mul(alpha, static_cast<T>(a(i, j)));
        else c(i, j) = Arith<T>::add(tag_mul(alpha, static_cast<T>(a(i, j))), tag_mul(beta, static_cast<T>(b(i, j))));
    });
}

// C := diag(d) A (Side::Left) or A diag(d) (Side::Right). C may alias A.
template <Side SD, class Ex, class V, class MA, class MC> BATCHLAS_DLA_INLINE void dgmm(const Ex& ex, const V& d, const MA& a, const MC& c) {
    using T = typename MC::value_type;
    for_each_owned2d(ex, c.rows(), c.cols(), [&](int i, int j) {
        if constexpr (SD == Side::Left) c(i, j) = Arith<T>::mul(static_cast<T>(d(i)), static_cast<T>(a(i, j)));
        else c(i, j) = Arith<T>::mul(static_cast<T>(a(i, j)), static_cast<T>(d(j)));
    });
}

// LAPACK lapmt with 1-based k, out of place (B must not alias A): Forward B(:, j) = A(:, k(j)),
// backward B(:, k(j)) = A(:, j).
template <bool Forward = true, class Ex, class K, class MA, class MB>
BATCHLAS_DLA_INLINE void lapmt(const Ex& ex, const K& k, const MA& a, const MB& b) {
    for_each_owned2d(ex, a.rows(), a.cols(), [&](int i, int j) {
        if constexpr (Forward) b(i, j) = a(i, static_cast<int>(k(j)) - 1);
        else b(i, static_cast<int>(k(j)) - 1) = a(i, j);
    });
}

// The unreferenced strict triangle from the referenced one, in place: A(i, j) := op(A(j, i)),
// op = conj when Conj (hermitize; the diagonal is then forced real) else identity (symmetrize).
// Every write reads only referenced storage, so there is no hazard between items.
template <bool Conj, class Ex, class U, class MA> BATCHLAS_DLA_INLINE void mirror(const Ex& ex, U uplo, const MA& a) {
    using T = typename MA::value_type;
    const bool lower = detail::is_lower(uplo);
    for_each_owned2d(ex, a.rows(), a.cols(), [&](int i, int j) {
        if (i == j) {
            if constexpr (Conj && Arith<T>::is_complex) a(i, i) = real_only(static_cast<T>(a(i, i)));
        } else if (lower ? (i < j) : (i > j)) {
            a(i, j) = conj_if<Conj>(static_cast<T>(a(j, i)));
        }
    });
}

// ---- SLM tiles on a Grid2D with a static group shape ----------------------------------------
// The item owns tile row r = lrow() and columns c = lcol() + s * GC < TN. The tile t is any
// Mat (Mem::Local); elements past the global operand's edge load as zero and are not stored.
// The caller owns the barrier between a load and a store of other items' elements.
namespace detail {
template <int TN, class Ex, class F> BATCHLAS_DLA_INLINE void tile_cols(const Ex& ex, F&& f) {
    constexpr int GC = Ex::static_group_cols;
    static_assert(GC > 0, "v2 tile ops: Grid2D needs a static group shape");
    if constexpr (TN == GC) {
        f(ex.lcol());
    } else {
        for (int c = ex.lcol(); c < TN; c += GC) f(c);
    }
}
}  // namespace detail

// t(r, c) = A(i0 + r, j0 + c)
template <int TN, class Ex, class MA, class MT> BATCHLAS_DLA_INLINE void tile_load(const Ex& ex, const MA& a, int i0, int j0, const MT& t) {
    using T = typename MT::value_type;
    const int r = ex.lrow();
    detail::tile_cols<TN>(ex, [&](int c) {
        const int i = i0 + r, j = j0 + c;
        t(r, c) = (i < a.rows() && j < a.cols()) ? static_cast<T>(a(i, j)) : Arith<T>::zero();
    });
}
// t(c, r) = A(i0 + r, j0 + c): the tile stored transposed.
template <int TN, class Ex, class MA, class MT> BATCHLAS_DLA_INLINE void tile_load_trans(const Ex& ex, const MA& a, int i0, int j0, const MT& t) {
    using T = typename MT::value_type;
    const int r = ex.lrow();
    detail::tile_cols<TN>(ex, [&](int c) {
        const int i = i0 + r, j = j0 + c;
        T v = Arith<T>::zero();
        if (i < a.rows() && j < a.cols()) v = a(i, j);
        t(c, r) = v;
    });
}
// B(i0 + r, j0 + c) = t(r, c)
template <int TN, class Ex, class MT, class MB> BATCHLAS_DLA_INLINE void tile_store(const Ex& ex, const MT& t, const MB& b, int i0, int j0) {
    const int r = ex.lrow();
    detail::tile_cols<TN>(ex, [&](int c) {
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
    detail::tile_cols<TN>(ex, [&](int c) {
        if (i0 + r < b.rows() && j0 + c < b.cols()) b(i0 + r, j0 + c) = t(r, c);
        if (j0 + r < b.rows() && i0 + c < b.cols()) b(j0 + r, i0 + c) = conj_if<Conj>(static_cast<T>(t(c, r)));
    });
}
// A diagonal tile (i0 == j0) of a mirror: the referenced member of each pair, op(t(c, r))
// opposite; with Conj the diagonal is forced real.
template <bool Conj, int TN, class Ex, class U, class MT, class MB>
BATCHLAS_DLA_INLINE void tile_store_diag(const Ex& ex, U uplo, const MT& t, const MB& b, int i0) {
    using T = typename MB::value_type;
    const bool lower = detail::is_lower(uplo);
    const int r = ex.lrow();
    detail::tile_cols<TN>(ex, [&](int c) {
        const int i = i0 + r, j = i0 + c;
        if (i >= b.rows() || j >= b.cols()) return;
        const bool referenced = lower ? (r >= c) : (r <= c);
        T value = referenced ? static_cast<T>(t(r, c)) : conj_if<Conj>(static_cast<T>(t(c, r)));
        if constexpr (Conj) {
            if (i == j) value = real_only(value);
        }
        b(i, j) = value;
    });
}

// ---- matrix reductions ----------------------------------------------------------------------
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
        for_each_owned2d(ex, m, n, [&](int i, int j) {
            const R v = modulus(static_cast<T>(a(i, j)));
            mx = v < mx ? mx : v;
        });
        return detail::scope_reduce<S>(ex, mx, sycl::maximum<R>());
    } else if constexpr (NT == NormType::One || NT == NormType::Inf) {
        constexpr bool one = NT == NormType::One;
        const int outer = one ? n : m, inner = one ? m : n;
        R mx = R(0);
        for (int k = ex.lid(); k < outer; k += Ex::size) {
            R s = R(0);
            for (int l = 0; l < inner; ++l) s += modulus(static_cast<T>(one ? a(l, k) : a(k, l)));
            mx = sycl::maximum<R>()(mx, s);
        }
        return detail::scope_reduce<S>(ex, mx, sycl::maximum<R>());
    } else {
        static_assert(NT == NormType::Frobenius, "v2::lange: Max, One, Inf or Frobenius");
        if constexpr (std::is_same_v<Alg, alg::Unscaled>) {
            R ss = R(0);
            for_each_owned2d(ex, m, n, [&](int i, int j) { ss += abs2(static_cast<T>(a(i, j))); });
            return static_cast<R>(sycl::sqrt(detail::scope_reduce<S>(ex, ss)));
        } else {
            static_assert(std::is_same_v<Alg, alg::Scaled>, "v2::lange: Frobenius is alg::Unscaled or alg::Scaled");
            using A = Arith<T>;
            R sc = R(0);
            auto upd = [&](R v) { sc = (v > sc || sycl::isnan(v)) ? v : sc; };
            for_each_owned2d(ex, m, n, [&](int i, int j) {
                const T v = a(i, j);
                upd(sycl::fabs(A::re(v)));
                if constexpr (A::is_complex) upd(sycl::fabs(A::im(v)));
            });
            const R scale = detail::scope_reduce<S>(ex, sc, sycl::maximum<R>());
            if (sycl::isinf(scale)) return scale;
            const R inv = scale > R(0) ? R(1) / scale : R(1);
            R ss = R(0);
            for_each_owned2d(ex, m, n, [&](int i, int j) {
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

// {min |x_i|, max |x_i|} (modulus) with the fmin / fmax NaN policy: a NaN element is skipped
// unless every element is NaN. n == 0 -> {+inf, 0}.
template <class R> struct AbsRange {
    R lo;
    R hi;
};
template <Scope S = Scope::Auto, class Ex, class X> BATCHLAS_DLA_INLINE auto absrange(const Ex& ex, const X& x) {
    using T = elem_t<X>;
    using R = real_t<T>;
    R lo = std::numeric_limits<R>::infinity(), hi = R(0);
    for_each_owned(ex, x.size(), [&](int i) {
        const R v = modulus(static_cast<T>(x(i)));
        lo = sycl::fmin(lo, v);
        hi = sycl::fmax(hi, v);
    });
    return AbsRange<R>{detail::scope_reduce<S>(ex, lo, sycl::minimum<R>()), detail::scope_reduce<S>(ex, hi, sycl::maximum<R>())};
}

// ---- RowFrag moves (lane r holds row r of an N-wide register row; F is T* or Frag) ------------
// Every index into the row is a compile-time constant after the full unroll, and a dead row
// carries the identity so the unrolled factorisation needs no lane guard (docs/perf/potrf.md).
template <int N, class F> BATCHLAS_DLA_INLINE void rowfrag_identity(F&& a, int lane) {
    using T = std::remove_cvref_t<decltype(a[0])>;
    const T one = Arith<T>::one();
    const T zero = Arith<T>::zero();
    BATCHLAS_DLA_UNROLL_FULL
    for (int c = 0; c < N; ++c) a[c] = sel(c == lane, one, zero);
}

// Row `lane` of the UL triangle (Upper: as the conj-transposed row, S(lane, c) = conj(A(c, lane)),
// so one Lower recurrence serves both). Columns c > lane keep the identity.
template <Uplo UL, int N, class F, class T>
BATCHLAS_DLA_INLINE void rowfrag_load_tri(F&& a, const T* __restrict p, int ld, int lane, bool live, bool real_diag) {
    rowfrag_identity<N>(a, lane);
    BATCHLAS_DLA_UNROLL_FULL
    for (int c = 0; c < N; ++c) {
        if (live && c <= lane) {
            T v;
            if constexpr (UL == Uplo::Lower) v = p[lane + static_cast<std::ptrdiff_t>(c) * ld];
            else v = Arith<T>::conj(p[c + static_cast<std::ptrdiff_t>(lane) * ld]);
            if (real_diag && c == lane) v = real_only(v);
            a[c] = v;
        }
    }
}
template <Uplo UL, int N, class F, class T>
BATCHLAS_DLA_INLINE void rowfrag_store_tri(const F& a, T* __restrict p, int ld, int lane, bool live) {
    BATCHLAS_DLA_UNROLL_FULL
    for (int c = 0; c < N; ++c) {
        if (live && c <= lane) {
            if constexpr (UL == Uplo::Lower) p[lane + static_cast<std::ptrdiff_t>(c) * ld] = a[c];
            else p[c + static_cast<std::ptrdiff_t>(lane) * ld] = Arith<T>::conj(a[c]);
        }
    }
}
// The full row (columns c < n; the rest keep the identity).
template <int N, class F, class T>
BATCHLAS_DLA_INLINE void rowfrag_load(F&& a, const T* __restrict p, int ld, int lane, int n, bool live) {
    rowfrag_identity<N>(a, lane);
    BATCHLAS_DLA_UNROLL_FULL
    for (int c = 0; c < N; ++c) {
        if (live && c < n) a[c] = p[lane + static_cast<std::ptrdiff_t>(c) * ld];
    }
}
template <int N, class F, class T>
BATCHLAS_DLA_INLINE void rowfrag_store(const F& a, T* __restrict p, int ld, int row, int n, bool live) {
    BATCHLAS_DLA_UNROLL_FULL
    for (int c = 0; c < N; ++c) {
        if (live && c < n) p[row + static_cast<std::ptrdiff_t>(c) * ld] = a[c];
    }
}
// One element of an identity-padded tile: A(r, c) inside the live n x n block, I(r, c) outside.
template <class T>
BATCHLAS_DLA_INLINE T load_or_identity(const T* __restrict p, int r, int c, int n, std::ptrdiff_t ld, bool live) {
    if (live && r < n && c < n) return p[static_cast<std::ptrdiff_t>(r) + static_cast<std::ptrdiff_t>(c) * ld];
    return (r == c) ? Arith<T>::one() : Arith<T>::zero();
}

namespace detail {
#ifndef BATCHLAS_DLA_V2_INTEG_DETAIL_SELECT_LANE  // integ/dla-v2-all: defined by several ext headers
#define BATCHLAS_DLA_V2_INTEG_DETAIL_SELECT_LANE
template <class Impl, class V> BATCHLAS_DLA_INLINE V select_lane(const Impl& part, V v, std::uint32_t src) {
    if constexpr (sycl::is_group_v<Impl>) return batchlas::portable::select_from_group(part, v, src);
    else return select_from_group(part, v, src);  // ADL: SubGroupPartition (src/kernels/common)
}
#endif  // BATCHLAS_DLA_V2_INTEG_DETAIL_SELECT_LANE
}  // namespace detail

// v of lane src, to every lane of the group / partition. COLLECTIVE: every lane must reach it.
// Complex goes as two reals (select rejects an aggregate).
template <class Impl, class T> BATCHLAS_DLA_INLINE T bcast(const Impl& part, T v, std::uint32_t src) {
    using A = Arith<T>;
    if constexpr (A::is_complex) return A::make(detail::select_lane(part, A::re(v), src), detail::select_lane(part, A::im(v), src));
    else return detail::select_lane(part, v, src);
}

// (value, key) argmax over N lanes by xor butterfly: larger value wins, ties -> the lower key.
// The caller owes NaN-free values (NaN loses every comparison and the lanes would disagree).
template <int N, class Impl, class R> BATCHLAS_DLA_INLINE void argmax_pair(const Impl& part, R& a, int& key) {
    BATCHLAS_DLA_UNROLL_FULL
    for (std::uint32_t mask = 1u; mask < static_cast<std::uint32_t>(N); mask <<= 1) {
        const R ov = detail::xor_shuffle(part, a, mask);
        const int ok = detail::xor_shuffle(part, key, mask);
        if (ov > a || (ov == a && ok < key)) {
            a = ov;
            key = ok;
        }
    }
}

// ---- scalar LAPACK ------------------------------------------------------------------------
// LAPACK dlamch-style safe minimum / maximum.
template <class R> constexpr R safmin() {
    constexpr R eps = std::numeric_limits<R>::epsilon();
    constexpr R s1 = std::numeric_limits<R>::min();
    constexpr R s2 = R(1) / std::numeric_limits<R>::max();
    return s2 > s1 ? s2 * (R(1) + eps) : s1;
}
template <class R> constexpr R safmax() { return R(1) / safmin<R>(); }

// lartg: [c s; -conj(s) c] [f; g] = [r; 0], c real.
template <class T> struct LartgResult {
    real_t<T> c;
    T s;
    T r;
};

namespace detail {
// CLARTG/ZLARTG with f, g != 0, the scalar math in W (scaled against overflow / underflow).
template <class W, class T> BATCHLAS_DLA_INLINE LartgResult<T> lartg_complex_scaled(const T& f, const T& g) {
    using R = real_t<T>;
    const W f_abs_w = sycl::hypot(static_cast<W>(f.real()), static_cast<W>(f.imag()));
    const W g_abs_w = sycl::hypot(static_cast<W>(g.real()), static_cast<W>(g.imag()));

    const W u = sycl::fmin(safmax<W>(), sycl::fmax(safmin<W>(), sycl::fmax(f_abs_w, g_abs_w)));
    const W fs = f_abs_w / u;
    const W gs = g_abs_w / u;
    const W norm = u * sycl::sqrt(fs * fs + gs * gs);

    const std::complex<W> f_w{static_cast<W>(f.real()), static_cast<W>(f.imag())};
    const std::complex<W> g_w{static_cast<W>(g.real()), static_cast<W>(g.imag())};
    const std::complex<W> alpha = f_w / f_abs_w;  // sgn(F)
    const W c_w = f_abs_w / norm;
    const std::complex<W> conjg_w{g_w.real(), -g_w.imag()};
    const std::complex<W> s_w = alpha * conjg_w / norm;
    const std::complex<W> r_w = alpha * norm;

    const T s{static_cast<R>(s_w.real()), static_cast<R>(s_w.imag())};
    const T r{static_cast<R>(r_w.real()), static_cast<R>(r_w.imag())};
    return {static_cast<R>(c_w), s, r};
}
}  // namespace detail

// W: the working precision of the complex branch (void: real_t<T>); the real branch is
// SLARTG/DLARTG with its unscaled fast path.
template <class W = void, class T> inline LartgResult<T> lartg(const T& f, const T& g) {
    using R = real_t<T>;
    const R f_abs = modulus(f);
    const R g_abs = modulus(g);

    if (g_abs == R(0)) return {R(1), T(0), f};  // G = 0: C = 1, S = 0, R = F

    if (f_abs == R(0)) {  // F = 0: C = 0, S chosen so that R is real
        if constexpr (Arith<T>::is_complex) {
            const T conjg = T(g.real(), -g.imag());
            const T s = conjg / T(g_abs);
            const T r = T(g_abs, R(0));
            return {R(0), s, r};
        } else {
            const R s = sycl::copysign(R(1), static_cast<R>(g));
            const R r = g_abs;
            return {R(0), T(s), T(r)};
        }
    }

    if constexpr (Arith<T>::is_complex) {
        return detail::lartg_complex_scaled<std::conditional_t<std::is_void_v<W>, R, W>>(f, g);
    } else {
        const R rtmin = sycl::sqrt(safmin<R>());
        const R rtmax = sycl::sqrt(safmax<R>() / R(2));
        const R f_r = static_cast<R>(f);
        const R g_r = static_cast<R>(g);
        if (f_abs > rtmin && f_abs < rtmax && g_abs > rtmin && g_abs < rtmax) {
            const R d = sycl::sqrt(f_r * f_r + g_r * g_r);
            const R c = f_abs / d;
            const R r = sycl::copysign(d, f_r);
            const R s = g_r / r;
            return {c, T(s), T(r)};
        }
        const R u = sycl::fmin(safmax<R>(), sycl::fmax(safmin<R>(), sycl::fmax(f_abs, g_abs)));
        const R fs = f_r / u;
        const R gs = g_r / u;
        const R d = sycl::sqrt(fs * fs + gs * gs);
        const R c = sycl::fabs(fs) / d;
        const R r_unscaled = sycl::copysign(d, f_r);
        const R s = gs / r_unscaled;
        const R r = r_unscaled * u;
        return {c, T(s), T(r)};
    }
}

// larfg: H^H [alpha; x] = [beta; 0], H = I - tau v v^H, v = [1; scale * x]. Conventions:
//   conv::Phase    beta = -sgn(alpha) ||[alpha; x]|| keeps alpha's phase (complex beta, real
//                  tau); len <= 1, or a real alpha over x == 0 -> tau = 0. (src detail::larfg)
//   conv::RealBeta LAPACK ZLARFG: beta real, tau complex; a complex alpha reflects even at
//                  len == 1. Real T: identical to Phase. (gebrd larfg_real_beta)
namespace conv {
struct Phase {};
struct RealBeta {};
}  // namespace conv

#ifndef BATCHLAS_DLA_V2_INTEG_LARFG_SCALARS3  // integ/dla-v2-all: defined by several ext headers
#define BATCHLAS_DLA_V2_INTEG_LARFG_SCALARS3
template <class T> struct LarfgScalars {
    T tau;
    T beta;
    T scale;
};
#endif  // BATCHLAS_DLA_V2_INTEG_LARFG_SCALARS3

template <class Conv = conv::Phase, class T>
inline LarfgScalars<T> larfg_scalars(const T& alpha, real_t<T> xnorm, int len) {
    static_assert(std::is_same_v<Conv, conv::Phase> || std::is_same_v<Conv, conv::RealBeta>,
                  "v2::larfg_scalars: Conv is conv::Phase or conv::RealBeta");
    using R = real_t<T>;
    LarfgScalars<T> result{T(0), alpha, T(0)};
    if constexpr (std::is_same_v<Conv, conv::RealBeta> && Arith<T>::is_complex) {
        if (len < 1 || (xnorm == R(0) && alpha.imag() == R(0))) return result;
        const R norm = sycl::hypot(sycl::hypot(alpha.real(), alpha.imag()), xnorm);
        if (norm == R(0)) return result;
        const R beta = sycl::signbit(alpha.real()) ? norm : -norm;
        result.beta = T(beta);
        result.tau = T((beta - alpha.real()) / beta, -alpha.imag() / beta);
        result.scale = T(1) / (alpha - result.beta);
        return result;
    } else {
        if (len <= 1) return result;  // LAPACK: N <= 1 -> tau = 0, beta = alpha
        if constexpr (Arith<T>::is_complex) {
            if (xnorm == R(0) && alpha.imag() == R(0)) return result;
            const R alpha_abs = sycl::hypot(alpha.real(), alpha.imag());
            const R beta_abs = sycl::hypot(alpha_abs, xnorm);
            const T alpha_sign = (alpha_abs == R(0)) ? T(1) : (alpha / alpha_abs);
            result.beta = -alpha_sign * T(beta_abs);
            result.tau = (result.beta - alpha) / result.beta;
            result.scale = T(1) / (alpha - result.beta);
            return result;
        } else {
            if (xnorm == R(0)) return result;
            const R alpha_r = static_cast<R>(alpha);
            const R sign = sycl::signbit(alpha_r) ? R(-1) : R(1);
            const R beta = -sign * sycl::hypot(alpha_r, xnorm);
            result.beta = T(beta);
            result.tau = (result.beta - alpha) / result.beta;
            result.scale = T(1) / (alpha - result.beta);
            return result;
        }
    }
}

// larfg on a v2 executor: alpha := beta, x := scale * x, returns tau. alpha must be uniform
// over the executor; every item computes the scalars from the uniform (alpha, ||x||), so there
// is no broadcast. ||x|| is Unscaled (one reduction), as the legacy group larfg. The caller
// owns the barrier before x is read across owners.
template <class Conv = conv::Phase, Scope S = Scope::Auto, class Ex, class T, class X>
BATCHLAS_DLA_INLINE T larfg(const Ex& ex, T& alpha, const X& x) {
    using R = real_t<T>;
    const int n = x.size();
    const R xnorm = n > 0 ? static_cast<R>(sycl::sqrt(Arith<T>::re(dotc<S>(ex, x, x)))) : R(0);
    const LarfgScalars<T> sc = larfg_scalars<Conv>(alpha, xnorm, n + 1);
    alpha = sc.beta;
    if (sc.tau != T(0) && n > 0) scal(ex, sc.scale, x);
    return sc.tau;
}

// LAPACK SLAE2/DLAE2: {rt1, rt2} of [[a, b], [b, c]], |rt1| >= |rt2|.
template <class T> inline constexpr auto lae2(const T& a, const T& b, const T& c) {
    const auto one = T(1);
    const auto two = T(2);
    const auto zero = T(0);
    const auto half = T(0.5);
    const auto sm = a + c;
    const auto df = a - c;
    const auto adf = std::abs(df);
    const auto tb = b + b;
    const auto ab = std::abs(tb);
    const auto acmx = (std::abs(a) > std::abs(c)) ? a : c;
    const auto acmn = (std::abs(a) > std::abs(c)) ? c : a;
    T rt;  // sqrt(adf^2 + (2b)^2) without overflow / underflow
    if (adf > ab) rt = adf * std::sqrt(one + (ab / adf) * (ab / adf));
    else if (adf < ab) rt = ab * std::sqrt(one + (adf / ab) * (adf / ab));
    else rt = ab * std::sqrt(two);  // includes ab = adf = 0
    T rt1, rt2;
    if (sm < zero) {
        rt1 = half * (sm - rt);
        rt2 = (acmx / rt1) * acmn - (b / rt1) * b;  // this order keeps the smaller one accurate
    } else if (sm > zero) {
        rt1 = half * (sm + rt);
        rt2 = (acmx / rt1) * acmn - (b / rt1) * b;
    } else {
        rt1 = half * rt;
        rt2 = -half * rt;
    }
    return std::array{rt1, rt2};
}

// LAPACK SLAEV2/DLAEV2: {rt1, rt2, cs1, sn1}; [cs1, sn1] is the unit eigenvector of rt1.
template <class T> inline constexpr auto laev2(const T& a, const T& b, const T& c) {
    const T one = T(1);
    const T two = T(2);
    const T zero = T(0);
    const T half = T(0.5);
    int sgn1 = 0;
    int sgn2 = 0;
    const T sm = a + c;
    const T df = a - c;
    const T adf = std::abs(df);
    const T tb = b + b;
    const T ab = std::abs(tb);
    T acmx, acmn;
    if (std::abs(a) > std::abs(c)) {
        acmx = a;
        acmn = c;
    } else {
        acmx = c;
        acmn = a;
    }
    T rt;
    if (adf > ab) rt = adf * std::sqrt(one + (ab / adf) * (ab / adf));
    else if (adf < ab) rt = ab * std::sqrt(one + (adf / ab) * (adf / ab));
    else rt = ab * std::sqrt(two);
    T rt1, rt2;
    if (sm < zero) {
        rt1 = half * (sm - rt);
        sgn1 = -1;
        rt2 = (acmx / rt1) * acmn - (b / rt1) * b;
    } else if (sm > zero) {
        rt1 = half * (sm + rt);
        sgn1 = 1;
        rt2 = (acmx / rt1) * acmn - (b / rt1) * b;
    } else {
        rt1 = half * rt;
        rt2 = -half * rt;
        sgn1 = 1;
    }
    T cs;
    if (df >= zero) {
        cs = df + rt;
        sgn2 = 1;
    } else {
        cs = df - rt;
        sgn2 = -1;
    }
    const T acs = std::abs(cs);
    T cs1, sn1;
    if (acs > ab) {
        const T ct = -tb / cs;
        sn1 = one / std::sqrt(one + ct * ct);
        cs1 = ct * sn1;
    } else if (ab == zero) {
        cs1 = one;
        sn1 = zero;
    } else {
        const T tn = -cs / tb;
        cs1 = one / std::sqrt(one + tn * tn);
        sn1 = tn * cs1;
    }
    if (sgn1 == sgn2) {
        const T tn = cs1;
        cs1 = -sn1;
        sn1 = tn;
    }
    return std::array{rt1, rt2, cs1, sn1};
}

// LAPACK ?LAS2: {ssmin, ssmax} of the upper triangular [[f, g], [0, h]], nothing squared.
template <class T> inline constexpr std::array<T, 2> las2(const T& f, const T& g, const T& h) {
    const T one = T(1), two = T(2), zero = T(0);
    const T fa = std::abs(f), ga = std::abs(g), ha = std::abs(h);
    const T fhmn = fa < ha ? fa : ha;
    const T fhmx = fa < ha ? ha : fa;
    if (fhmn == zero) {
        if (fhmx == zero) return {zero, ga};
        const T mx = fhmx > ga ? fhmx : ga, mn = fhmx > ga ? ga : fhmx;
        return {zero, mx * std::sqrt(one + (mn / mx) * (mn / mx))};
    }
    if (ga < fhmx) {
        const T as = one + fhmn / fhmx, at = (fhmx - fhmn) / fhmx, au = (ga / fhmx) * (ga / fhmx);
        const T c = two / (std::sqrt(as * as + au) + std::sqrt(at * at + au));
        return {fhmn * c, fhmx / c};
    }
    const T au = fhmx / ga;
    if (au == zero) return {(fhmn * fhmx) / ga, ga};
    const T as = one + fhmn / fhmx, at = (fhmx - fhmn) / fhmx;
    const T c = one / (std::sqrt(one + (as * au) * (as * au)) + std::sqrt(one + (at * au) * (at * au)));
    return {two * ((fhmn * c) * au), ga / (c + c)};
}

}  // namespace batchlas::device::v2
