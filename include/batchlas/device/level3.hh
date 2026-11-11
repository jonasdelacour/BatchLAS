// SPDX-License-Identifier: MIT
#pragma once
// Device LA Level 3, register tier (v22 design A.2 level3.hh): one work-item's whole operands in
// register Tiles (every index a compile-time constant after unrolling). The in-group forms over
// executors and the tuned gemm components join here in L3a-L3d (design B).
//   gemm<TA, TB, Shape, Ops>   C := alpha op(A) op(B) + beta C  (epilogue store<Shape, Ops>)
//   gemm_sub<Ops>              C -= A B (the factorizations' trailing update; Ops::msub)
//   trsm<Side, UL, TR, DG, Ops>  op(A) X = alpha B (Left) or X op(A) = alpha B (Right), in place

#include <type_traits>

#include <batchlas/blas/enums.hh>
#include <batchlas/device/arith.hh>
#include <batchlas/device/core.hh>
#include <batchlas/device/exec.hh>
#include <batchlas/device/operand.hh>

namespace batchlas::device {

namespace detail {
// op(A)(i, j) of a register tile.
template <Transpose TR, class T, int R, int C> BATCHLAS_DLA_INLINE T tile_op(const Tile<T, R, C>& a, int i, int j) {
    if constexpr (TR == Transpose::NoTrans) return a(i, j);
    else if constexpr (TR == Transpose::Trans) return a(j, i);
    else return Arith<T>::conj(a(j, i));
}
template <Transpose TR, int R, int C> inline constexpr int op_rows = TR == Transpose::NoTrans ? R : C;
template <Transpose TR, int R, int C> inline constexpr int op_cols = TR == Transpose::NoTrans ? C : R;
}  // namespace detail

// C(M x N) := alpha op(A) op(B) + beta C; op(A) is M x K, op(B) is K x N. The k loop runs
// innermost per (i, j) with Ops::mac into a zero accumulator, then one epilogue store per
// element (Shape: epi::Sum default; see arith.hh store).
template <Transpose TA = Transpose::NoTrans, Transpose TB = Transpose::NoTrans, class Shape = epi::Sum, class Ops = arith::Explicit,
          class AS, class BS, class T, int AR, int AC, int BR, int BC, int M, int N>
BATCHLAS_DLA_INLINE void gemm(const Thread&, AS alpha, const Tile<T, AR, AC>& A, const Tile<T, BR, BC>& B, BS beta, Tile<T, M, N>& C) {
    constexpr int K = detail::op_cols<TA, AR, AC>;
    static_assert(detail::op_rows<TA, AR, AC> == M && detail::op_cols<TB, BR, BC> == N && detail::op_rows<TB, BR, BC> == K,
                  "gemm (register tile): op(A) is M x K, op(B) is K x N, C is M x N");
    using Ar = ops_arith_t<Ops, T>;
#pragma unroll
    for (int j = 0; j < N; ++j)
#pragma unroll
        for (int i = 0; i < M; ++i) {
            T acc = Ar::zero();
#pragma unroll
            for (int k = 0; k < K; ++k) Ar::mac(acc, detail::tile_op<TA>(A, i, k), detail::tile_op<TB>(B, k, j));
            store<Shape, Ops>(alpha, acc, beta, C(i, j));
        }
}

// C -= A B on tiles, one Ops::msub per term in k order (Ops = arith::FmaSplit: sub(c, mul(a, b)),
// the two-rounding hand form; arith::Fma: one fused msub).
template <class Ops = arith::Explicit, class T, int M, int K, int N>
BATCHLAS_DLA_INLINE void gemm_sub(const Thread&, const Tile<T, M, K>& A, const Tile<T, K, N>& B, Tile<T, M, N>& C) {
    using Ar = ops_arith_t<Ops, T>;
#pragma unroll
    for (int j = 0; j < N; ++j)
#pragma unroll
        for (int i = 0; i < M; ++i)
#pragma unroll
            for (int k = 0; k < K; ++k) Ar::msub(C(i, j), A(i, k), B(k, j));
}

// Triangular solve on register tiles, B overwritten by X. Left: op(A) X = alpha B with A M x M;
// Right: X op(A) = alpha B with A N x N. UL names the STORED triangle of A (the other is not
// read); DG = Unit never reads the diagonal. Substitution order is the reference BLAS order
// (dtrsm), so a real Ops::msub chain matches it term by term.
template <Side SD, Uplo UL, Transpose TR, Diag DG, class Ops = arith::Explicit, class AS, class T, int NA, int M, int N>
BATCHLAS_DLA_INLINE void trsm(const Thread&, AS alpha_, const Tile<T, NA, NA>& A, Tile<T, M, N>& B) {
    using Ar = ops_arith_t<Ops, T>;
    const auto alpha = as_tag(alpha_);
    if constexpr (!std::is_same_v<std::remove_cv_t<decltype(alpha)>, One>) {
#pragma unroll
        for (int j = 0; j < N; ++j)
#pragma unroll
            for (int i = 0; i < M; ++i) B(i, j) = detail::mul_term<Ar>(alpha, B(i, j));
    }
    // op(A) lower <=> (UL == Lower) == (TR == NoTrans)
    constexpr bool op_lower = (UL == Uplo::Lower) == (TR == Transpose::NoTrans);
    if constexpr (SD == Side::Left) {
        static_assert(NA == M, "trsm (register tile, Left): A is M x M");
#pragma unroll
        for (int j = 0; j < N; ++j) {
            if constexpr (op_lower) {
#pragma unroll
                for (int i = 0; i < M; ++i) {
#pragma unroll
                    for (int k = 0; k < i; ++k) Ar::msub(B(i, j), detail::tile_op<TR>(A, i, k), B(k, j));
                    if constexpr (DG == Diag::NonUnit) B(i, j) = Ar::div(B(i, j), detail::tile_op<TR>(A, i, i));
                }
            } else {
#pragma unroll
                for (int i = M - 1; i >= 0; --i) {
#pragma unroll
                    for (int k = i + 1; k < M; ++k) Ar::msub(B(i, j), detail::tile_op<TR>(A, i, k), B(k, j));
                    if constexpr (DG == Diag::NonUnit) B(i, j) = Ar::div(B(i, j), detail::tile_op<TR>(A, i, i));
                }
            }
        }
    } else {
        static_assert(NA == N, "trsm (register tile, Right): A is N x N");
#pragma unroll
        for (int i = 0; i < M; ++i) {
            if constexpr (!op_lower) {  // X U = B: columns left to right
#pragma unroll
                for (int j = 0; j < N; ++j) {
#pragma unroll
                    for (int k = 0; k < j; ++k) Ar::msub(B(i, j), B(i, k), detail::tile_op<TR>(A, k, j));
                    if constexpr (DG == Diag::NonUnit) B(i, j) = Ar::div(B(i, j), detail::tile_op<TR>(A, j, j));
                }
            } else {  // X L = B: columns right to left
#pragma unroll
                for (int j = N - 1; j >= 0; --j) {
#pragma unroll
                    for (int k = j + 1; k < N; ++k) Ar::msub(B(i, j), B(i, k), detail::tile_op<TR>(A, k, j));
                    if constexpr (DG == Diag::NonUnit) B(i, j) = Ar::div(B(i, j), detail::tile_op<TR>(A, j, j));
                }
            }
        }
    }
}

}  // namespace batchlas::device
