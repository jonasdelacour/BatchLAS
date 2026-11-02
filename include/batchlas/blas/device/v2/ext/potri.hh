// SPDX-License-Identifier: MIT
#pragma once
// Device LA v2 extension: what potri's resident trti2 / lauu2 (src/ops/potri/kernels/potri_cta.cc)
// needs beyond the core. Register-tile BLAS on the Thread executor (namespace reg), a work-group
// whose size is a launch value, a column-major 2-D distribution, and the LAPACK auxiliaries
// (trtri's zero-diagonal scan, trttp/tpttr, the Hermitian fill). Every op is DLA-inline and adds
// no barrier; register tiles are indexed by unrolled constants only.
//
// Include src/sycl/kernel_attrs.hh first so BATCHLAS_DLA_UNROLL_FULL spells the kernels' pragma.

#include <climits>
#include <cstddef>
#include <type_traits>

#include <sycl/sycl.hpp>

#include <batchlas/blas/device/v2/core.hh>
#include <batchlas/blas/device/v2/exec.hh>
#include <batchlas/blas/device/v2/operand.hh>
#include <batchlas/blas/device/v2/level1.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/util/group-collectives.hh>

#ifndef BATCHLAS_DLA_UNROLL_FULL
#ifdef BATCHLAS_UNROLL_FULL
#define BATCHLAS_DLA_UNROLL_FULL BATCHLAS_UNROLL_FULL
#else
#define BATCHLAS_DLA_UNROLL_FULL _Pragma("clang loop unroll(full)")
#endif
#endif

namespace batchlas::device::v2 {

// Arith with the explicit-FMA complex forms of src/sycl/device_scalar.hh (dev_mul, fma_acc), so
// a port of a kernel written on those keeps its rounding bit for bit.
template <class T>
struct FmaArith : Arith<T> {
    using A = Arith<T>;
    using R = typename A::real_t;
    BATCHLAS_DLA_INLINE static T mul(T a, T b) {
        if constexpr (A::is_complex)
            return A::make(sycl::fma(A::re(a), A::re(b), -A::im(a) * A::im(b)),
                           sycl::fma(A::re(a), A::im(b), A::im(a) * A::re(b)));
        else
            return a * b;
    }
    // acc += a b: one fma, or four for complex in fma_acc's order.
    BATCHLAS_DLA_INLINE static void fma(T& acc, T a, T b) {
        if constexpr (A::is_complex) {
            R re = sycl::fma(A::re(a), A::re(b), A::re(acc));
            re = sycl::fma(-A::im(a), A::im(b), re);
            R im = sycl::fma(A::re(a), A::im(b), A::im(acc));
            im = sycl::fma(A::im(a), A::re(b), im);
            acc = A::make(re, im);
        } else {
            acc = sycl::fma(a, b, acc);
        }
    }
};

template <class T>
BATCHLAS_DLA_INLINE bool is_exact_zero(T x) {
    using A = Arith<T>;
    using R = typename A::real_t;
    if constexpr (A::is_complex)
        return A::re(x) == R(0) && A::im(x) == R(0);
    else
        return x == T(0);
}

// ---- operands over the kernel's own address arithmetic -----------------------------------
// f(i) / f(i, j) return the element (a reference to write it); the kernel keeps its 64-bit
// index expressions, which a Vec / Mat (int32 indexing) would change.
template <class F>
struct FnVec {
    using value_type = std::remove_cvref_t<std::invoke_result_t<const F&, int>>;
    F f;
    Dyn n;
    BATCHLAS_DLA_INLINE decltype(auto) operator()(int i) const { return f(i); }
    BATCHLAS_DLA_INLINE int size() const { return int(n); }
};
template <class F>
struct FnMat {
    using value_type = std::remove_cvref_t<std::invoke_result_t<const F&, int, int>>;
    F f;
    Dyn m, n;
    BATCHLAS_DLA_INLINE decltype(auto) operator()(int i, int j) const { return f(i, j); }
    BATCHLAS_DLA_INLINE int rows() const { return int(m); }
    BATCHLAS_DLA_INLINE int cols() const { return int(n); }
};
template <class F>
BATCHLAS_DLA_INLINE FnVec<F> vec_fn(int n, F f) {
    return {f, Dyn{n}};
}
template <class F>
BATCHLAS_DLA_INLINE FnMat<F> mat_fn(int m, int n, F f) {
    return {f, Dyn{m}, Dyn{n}};
}

// ---- a work-group sized at launch ------------------------------------------------------
// For kernels that pick WG from the problem (potri_cta: one item per tile, a multiple of 32).
// Only ops written on ex.size take it; the core's Ex::size ops reject it at compile time.
struct DynWorkGroup {
    static constexpr Scope natural_scope = Scope::Group;
    sycl::nd_item<1> it;
    int size;

    BATCHLAS_DLA_INLINE int lid() const { return static_cast<int>(it.get_local_linear_id()); }
    BATCHLAS_DLA_INLINE bool leader() const { return lid() == 0; }
    BATCHLAS_DLA_INLINE void barrier() const { sycl::group_barrier(it.get_group()); }
    template <class T, class Op = sycl::plus<T>>
    BATCHLAS_DLA_INLINE T group_reduce(T v, Op op = {}) const {
        return batchlas::portable::reduce_over_group(it.get_group(), v, op);
    }
};

template <class F>
BATCHLAS_DLA_INLINE void for_each_owned(const DynWorkGroup& ex, int n, F&& f) {
    for (int i = ex.lid(); i < n; i += ex.size)
        f(i);
}

namespace detail {
// (e % m, e / m) for e = start, start + step, ... with no division per step.
struct Walk2d {
    int r, c, dr, dc, m;
    BATCHLAS_DLA_INLINE Walk2d(int start, int step, int m_)
        : r(start % m_), c(start / m_), dr(step % m_), dc(step / m_), m(m_) {}
    BATCHLAS_DLA_INLINE void next() {
        r += dr;
        c += dc;
        if (r >= m) {
            r -= m;
            ++c;
        }
    }
};
} // namespace detail

// f(r, c) for every (r, c) of a rows x cols column-major grid this item owns (element
// r + c rows cyclic over ex.size): consecutive items take consecutive rows, so loads coalesce.
template <class Ex, class F>
BATCHLAS_DLA_INLINE void for_each_owned_2d(const Ex& ex, int rows, int cols, F&& f) {
    for (detail::Walk2d w(ex.lid(), ex.size, rows); w.c < cols; w.next())
        f(w.r, w.c);
}

// The same walk, U positions per round: load(live, r, c) for all U (their loads overlap), then
// use(item) for all U. A dead slot (past the grid) still gets load(false, ...) and use().
template <int U, class Ex, class Load, class Use>
BATCHLAS_DLA_INLINE void for_each_owned_2d_batched(const Ex& ex, int rows, int cols, Load&& load, Use&& use) {
    using Item = decltype(load(true, 0, 0));
    for (detail::Walk2d w(ex.lid(), ex.size, rows); w.c < cols;) {
        Item e[U];
        BATCHLAS_DLA_UNROLL_FULL
        for (int u = 0; u < U; ++u) {
            const bool live = w.c < cols;
            e[u] = load(live, w.r, w.c);
            if (live) w.next();
        }
        BATCHLAS_DLA_UNROLL_FULL
        for (int u = 0; u < U; ++u)
            use(e[u]);
    }
}

// ---- LAPACK auxiliaries --------------------------------------------------------------------
// ?trtri's singularity scan: the smallest i with x(i) == 0 exactly, INT_MAX if none. One group
// reduction; the result is uniform.
template <class Ex, class X>
BATCHLAS_DLA_INLINE int first_zero(const Ex& ex, const X& x) {
    int z = INT_MAX;
    for_each_owned(ex, x.size(), [&](int i) {
        if (is_exact_zero(x(i))) z = sycl::min(z, i);
    });
    return ex.group_reduce(z, sycl::minimum<int>());
}

// ?trttp (to_packed) / ?tpttr of columns [c0, c1) of the n x n uplo triangle of a, packed by
// columns (LAPACK's AP layout). The direction is a launch value: one kernel serves both.
template <class Ex, class MA, class T>
BATCHLAS_DLA_INLINE void tpcopy(const Ex& ex, bool lower, bool to_packed, const MA& a, T* ap, int c0, int c1) {
    const int n = a.rows();
    for (int c = c0; c < c1; ++c) {
        const std::ptrdiff_t cc = c;
        const std::ptrdiff_t off = lower ? cc * n - cc * (cc - 1) / 2 - cc : cc * (cc + 1) / 2;
        for (int r = (lower ? c : 0) + ex.lid(); r < (lower ? n : c + 1); r += ex.size) {
            auto&& g = a(r, c);
            if (to_packed)
                ap[off + r] = g;
            else
                g = ap[off + r];
        }
    }
}

// One element of a Hermitian completion: the diagonal's imaginary part := 0, and (r, c) outside
// the uplo triangle := conj(a(c, r)). Entries inside the triangle are left alone.
template <class MA>
BATCHLAS_DLA_INLINE void hermitian_fill_at(bool lower, const MA& a, int r, int c) {
    using T = typename MA::value_type;
    using A = Arith<T>;
    if (r == c) {
        auto&& d = a(r, c);
        d = A::make(A::re(d), typename A::real_t(0));
    } else if (lower ? r < c : r > c) {
        a(r, c) = A::conj(a(c, r));
    }
}

// (r, c) of entry t of a lower triangle packed by rows (t = r (r + 1) / 2 + c, c <= r), exact
// for t < 2^30: a float sqrt estimate, then at most a step each way.
struct RowCol {
    int r, c;
};
BATCHLAS_DLA_INLINE RowCol tri_index(int t) {
    int r = static_cast<int>((sycl::sqrt(8.0f * static_cast<float>(t) + 1.0f) - 1.0f) * 0.5f);
    while ((r + 1) * (r + 2) / 2 <= t)
        ++r;
    while (r * (r + 1) / 2 > t)
        --r;
    return {r, t - r * (r + 1) / 2};
}

// ---- register-tile BLAS on the Thread executor ------------------------------------------
// A Tile is R x C values in one work-item's registers. Operands outside the tile are callables
// a(i, k) (local or global memory, or a masked view), read in the loop order each op states.
template <class T, int R, int C>
struct Tile {
    T v[R][C];
    BATCHLAS_DLA_INLINE T& operator()(int i, int j) { return v[i][j]; }
    BATCHLAS_DLA_INLINE const T& operator()(int i, int j) const { return v[i][j]; }
};

namespace reg {

template <class Ar, class T>
using arith_t = std::conditional_t<std::is_void_v<Ar>, FmaArith<T>, Ar>;

// ?laset at (i0, j0) of the global matrix: t(x, y) := beta where i0 + x == j0 + y, else alpha.
template <class AS, class BS, class T, int R, int C>
BATCHLAS_DLA_INLINE void laset(const Thread&, AS alpha_, BS beta_, Tile<T, R, C>& t, int i0, int j0) {
    const auto alpha = as_tag(alpha_);
    const auto beta = as_tag(beta_);
    constexpr bool same_tag = std::is_same_v<AS, BS> && (std::is_same_v<AS, Zero> || std::is_same_v<AS, One>);
    BATCHLAS_DLA_UNROLL_FULL
    for (int x = 0; x < R; ++x) {
        BATCHLAS_DLA_UNROLL_FULL
        for (int y = 0; y < C; ++y) {
            if constexpr (same_tag)
                t(x, y) = tag_mul(alpha, Arith<T>::one());
            else
                t(x, y) = i0 + x == j0 + y ? tag_mul(beta, Arith<T>::one()) : tag_mul(alpha, Arith<T>::one());
        }
    }
}

// N values f(0), ..., f(N - 1) into registers, in that order.
template <int N, class F>
BATCHLAS_DLA_INLINE auto load(const Thread&, F&& f) {
    Frag<std::remove_cvref_t<decltype(f(0))>, N> out;
    BATCHLAS_DLA_UNROLL_FULL
    for (int i = 0; i < N; ++i)
        out[i] = f(i);
    return out;
}

// f(i, j, t(x, y)) at the global i = i0 + x, j = j0 + y, row by row.
template <class T, int R, int C, class F>
BATCHLAS_DLA_INLINE void for_each(const Thread&, Tile<T, R, C>& t, int i0, int j0, F&& f) {
    BATCHLAS_DLA_UNROLL_FULL
    for (int x = 0; x < R; ++x) {
        BATCHLAS_DLA_UNROLL_FULL
        for (int y = 0; y < C; ++y)
            f(i0 + x, j0 + y, t(x, y));
    }
}

// B := op(A)^-1 B for an R x R triangular A (Left); forward substitution, row x of B updated
// by rows c < x. Implemented: Left Lower NoTrans Unit (A read strictly below its diagonal).
template <Side S, Uplo UL, Transpose TR, Diag DG, class Ar = void, class MA, class T, int R, int C>
BATCHLAS_DLA_INLINE void trsm(const Thread&, const MA& a, Tile<T, R, C>& b) {
    static_assert(S == Side::Left && UL == Uplo::Lower && TR == Transpose::NoTrans && DG == Diag::Unit,
                  "v2::reg::trsm: only Left Lower NoTrans Unit is implemented");
    using A = arith_t<Ar, T>;
    BATCHLAS_DLA_UNROLL_FULL
    for (int x = 1; x < R; ++x) {
        BATCHLAS_DLA_UNROLL_FULL
        for (int c = 0; c < x; ++c) {
            const T m = a(x, c);
            BATCHLAS_DLA_UNROLL_FULL
            for (int y = 0; y < C; ++y)
                b(x, y) = A::sub(b(x, y), A::mul(m, b(c, y)));
        }
    }
}

// C := alpha A B + beta C with A = a(x, k) (R x K) and B = b(k, y) (K x C), both staged into
// registers first (for k: A's column k, then B's row k), then K rank-1 steps in k order.
// Implemented: alpha = NegOne, beta = One (C - A B, one multiply and one subtract per term).
template <int K, class Ar = void, class AS, class MA, class MB, class BS, class T, int R, int C>
BATCHLAS_DLA_INLINE void gemm(const Thread&, AS, const MA& a, const MB& b, BS, Tile<T, R, C>& c) {
    static_assert(std::is_same_v<AS, NegOne> && std::is_same_v<BS, One>,
                  "v2::reg::gemm: only alpha = NegOne, beta = One is implemented");
    using A = arith_t<Ar, T>;
    T at[R][K], bt[K][C];
    BATCHLAS_DLA_UNROLL_FULL
    for (int k = 0; k < K; ++k) {
        BATCHLAS_DLA_UNROLL_FULL
        for (int x = 0; x < R; ++x)
            at[x][k] = a(x, k);
        BATCHLAS_DLA_UNROLL_FULL
        for (int y = 0; y < C; ++y)
            bt[k][y] = b(k, y);
    }
    BATCHLAS_DLA_UNROLL_FULL
    for (int k = 0; k < K; ++k) {
        BATCHLAS_DLA_UNROLL_FULL
        for (int x = 0; x < R; ++x) {
            BATCHLAS_DLA_UNROLL_FULL
            for (int y = 0; y < C; ++y)
                c(x, y) = A::sub(c(x, y), A::mul(at[x][k], bt[k][y]));
        }
    }
}

// A := alpha x y^T + A (geru) with fused multiply-adds. Implemented: alpha = One.
template <class Ar = void, class AS, class T, int R, int C, class L1, class L2>
BATCHLAS_DLA_INLINE void ger(const Thread&, AS, const Frag<T, R, L1>& x, const Frag<T, C, L2>& y, Tile<T, R, C>& a) {
    static_assert(std::is_same_v<AS, One>, "v2::reg::ger: only alpha = One is implemented");
    using A = arith_t<Ar, T>;
    BATCHLAS_DLA_UNROLL_FULL
    for (int i = 0; i < R; ++i) {
        BATCHLAS_DLA_UNROLL_FULL
        for (int j = 0; j < C; ++j)
            A::fma(a(i, j), x[i], y[j]);
    }
}

} // namespace reg

} // namespace batchlas::device::v2
