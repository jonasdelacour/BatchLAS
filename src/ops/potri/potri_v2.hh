// SPDX-License-Identifier: MIT
#pragma once
// potri-private device LA v2 ops (src/ops/potri/kernels/potri_cta.cc): register-tile BLAS on the
// Thread executor (namespace reg), the 4-deep batched column-major walk, trttp / tpttr of a column
// range and the row-packed triangle index. Everything shared comes from the v2.1 core
// (include/batchlas/blas/device/v2/dla.hh): RtGroup, for_each_owned(_2d), find_first, Tile,
// Frag, load, fill, arith::Fma. Every op is DLA-inline and adds no barrier; register tiles are
// indexed by unrolled constants only.

#include <cstddef>
#include <type_traits>

#include <sycl/sycl.hpp>

#include <src/sycl/kernel_attrs.hh>
#include <src/kernels/common/dla_v2_scalar.hh>
#include <batchlas/blas/device/v2/dla.hh>
#include <batchlas/blas/enums.hh>

namespace batchlas::sycl_potri {

namespace dla = batchlas::device::v2;

// (e % m, e / m) for e = start, start + step, ... with no division per step. (Not in a nested
// `detail`: that would hide batchlas::detail from the kernels in this namespace.)
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

// dla::for_each_owned_2d's walk, U positions per round: load(live, r, c) for all U (their loads
// overlap), then use(item) for all U. A dead slot (past the grid) still gets load(false, ...) and use().
template <int U, class Ex, class Load, class Use>
BATCHLAS_DLA_INLINE void for_each_owned_2d_batched(const Ex& ex, int rows, int cols, Load&& load, Use&& use) {
    using Item = decltype(load(true, 0, 0));
    for (Walk2d w(ex.lid(), ex.width(), rows); w.c < cols;) {
        Item e[U];
        BATCHLAS_UNROLL_FULL
        for (int u = 0; u < U; ++u) {
            const bool live = w.c < cols;
            e[u] = load(live, w.r, w.c);
            if (live) w.next();
        }
        BATCHLAS_UNROLL_FULL
        for (int u = 0; u < U; ++u)
            use(e[u]);
    }
}

// ?trttp (to_packed) / ?tpttr of columns [c0, c1) of the n x n uplo triangle of a(r, c) (a
// reference), packed by columns (LAPACK's AP layout). The direction is a launch value.
template <class Ex, class MA, class T>
BATCHLAS_DLA_INLINE void tpcopy(const Ex& ex, bool lower, bool to_packed, int n, const MA& a, T* ap, int c0, int c1) {
    for (int c = c0; c < c1; ++c) {
        const std::ptrdiff_t cc = c;
        const std::ptrdiff_t off = lower ? cc * n - cc * (cc - 1) / 2 - cc : cc * (cc + 1) / 2;
        dla::for_each_owned(ex, dla::Range{lower ? c : 0, lower ? n : c + 1}, [&](int r) BATCHLAS_DLA_BODY {
            auto&& g = a(r, c);
            if (to_packed)
                ap[off + r] = g;
            else
                g = ap[off + r];
        });
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
// Operands outside the tile are callables a(i, k) (local or global memory, or a masked view),
// read in the loop order each op states. Ops default to arith::Fma (dev_mul / fma_acc bits).
namespace reg {

// ?laset of a tile at (i0, j0) of the global matrix: t(x, y) := beta where i0 + x == j0 + y, else alpha.
template <class AS, class BS, class T, int R, int C>
BATCHLAS_DLA_INLINE void laset(const dla::Thread&, AS alpha_, BS beta_, dla::Tile<T, R, C>& t, int i0, int j0) {
    const T alpha = dla::tag_mul(dla::as_tag(alpha_), dla::Arith<T>::one());
    const T beta = dla::tag_mul(dla::as_tag(beta_), dla::Arith<T>::one());
    BATCHLAS_UNROLL_FULL
    for (int x = 0; x < R; ++x) {
        BATCHLAS_UNROLL_FULL
        for (int y = 0; y < C; ++y)
            t(x, y) = i0 + x == j0 + y ? beta : alpha;
    }
}

// f(i, j, t(x, y)) at the global i = i0 + x, j = j0 + y, row by row.
template <class T, int R, int C, class F>
BATCHLAS_DLA_INLINE void for_each(const dla::Thread&, dla::Tile<T, R, C>& t, int i0, int j0, F&& f) {
    BATCHLAS_UNROLL_FULL
    for (int x = 0; x < R; ++x) {
        BATCHLAS_UNROLL_FULL
        for (int y = 0; y < C; ++y)
            f(i0 + x, j0 + y, t(x, y));
    }
}

// B := A^-1 B for an R x R unit lower A (Left Lower NoTrans Unit; A read strictly below its
// diagonal); forward substitution, row x of B updated by rows c < x, sub(b, mul(m, b)).
template <class Ops = dla::arith::Fma, class MA, class T, int R, int C>
BATCHLAS_DLA_INLINE void trsm_llnu(const dla::Thread&, const MA& a, dla::Tile<T, R, C>& b) {
    using A = dla::arith_t<Ops, T>;
    BATCHLAS_UNROLL_FULL
    for (int x = 1; x < R; ++x) {
        BATCHLAS_UNROLL_FULL
        for (int c = 0; c < x; ++c) {
            const T m = a(x, c);
            BATCHLAS_UNROLL_FULL
            for (int y = 0; y < C; ++y)
                b(x, y) = A::sub(b(x, y), A::mul(m, b(c, y)));
        }
    }
}

// C := C - A B with A = a(x, k) (R x K) and B = b(k, y) (K x C), both staged into registers
// first (for k: A's column k, then B's row k), then K rank-1 steps in k order, sub(c, mul(a, b)).
template <int K, class Ops = dla::arith::Fma, class MA, class MB, class T, int R, int C>
BATCHLAS_DLA_INLINE void gemm_sub(const dla::Thread&, const MA& a, const MB& b, dla::Tile<T, R, C>& c) {
    using A = dla::arith_t<Ops, T>;
    T at[R][K], bt[K][C];
    BATCHLAS_UNROLL_FULL
    for (int k = 0; k < K; ++k) {
        BATCHLAS_UNROLL_FULL
        for (int x = 0; x < R; ++x)
            at[x][k] = a(x, k);
        BATCHLAS_UNROLL_FULL
        for (int y = 0; y < C; ++y)
            bt[k][y] = b(k, y);
    }
    BATCHLAS_UNROLL_FULL
    for (int k = 0; k < K; ++k) {
        BATCHLAS_UNROLL_FULL
        for (int x = 0; x < R; ++x) {
            BATCHLAS_UNROLL_FULL
            for (int y = 0; y < C; ++y)
                c(x, y) = A::sub(c(x, y), A::mul(at[x][k], bt[k][y]));
        }
    }
}

// A := x y^T + A on a register tile, one Ops::mac per entry, row by row (core ger is a
// memory-operand op with fixed arithmetic).
template <class Ops = dla::arith::Fma, class T, int R, int C, class L1, class L2>
BATCHLAS_DLA_INLINE void ger(const dla::Thread&, const dla::Frag<T, R, L1>& x, const dla::Frag<T, C, L2>& y,
                             dla::Tile<T, R, C>& a) {
    using A = dla::arith_t<Ops, T>;
    BATCHLAS_UNROLL_FULL
    for (int i = 0; i < R; ++i) {
        BATCHLAS_UNROLL_FULL
        for (int j = 0; j < C; ++j)
            A::mac(a(i, j), x[i], y[j]);
    }
}

} // namespace reg

} // namespace batchlas::sycl_potri
