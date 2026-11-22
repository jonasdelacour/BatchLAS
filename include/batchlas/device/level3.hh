// SPDX-License-Identifier: MIT
#pragma once
/// @file
/// @brief In-kernel BLAS level 3, register tier: one work-item's whole operands in register tiles
/// (@ref api_device_blas3).
///
/// Every index is a compile-time constant after unrolling, so the operands stay in registers:
///   - `gemm<TA, TB, Shape, Ops>`: `C := alpha op(A) op(B) + beta C`.
///   - `gemm_sub<K, Ops>`: `C -= A B`, the factorizations' trailing update.
///   - `trsm<SD, UL, TR, DG, Ops>`: `op(A) X = alpha B` (Left) or `X op(A) = alpha B` (Right), in place.
///
/// All three run on the `Thread` executor. The in-group forms (one matrix per work-group, any
/// size) are in `<batchlas/device/level3/group.hh>`. Included by `<batchlas/device.hh>`.
/// @ingroup api_device_blas3

#include <type_traits>

#include <batchlas/blas/enums.hh>
#include <batchlas/device/arith.hh>
#include <batchlas/device/core.hh>
#include <batchlas/device/detail/kernel_attrs.hh>
#include <batchlas/device/exec.hh>
#include <batchlas/device/operand.hh>

namespace batchlas::device {

namespace detail {
// op(A)(i, j) of a register tile or any a(i, j) operand.
template <Transpose TR, class MA> BATCHLAS_DEVICE_INLINE auto tile_op(const MA& a, int i, int j) {
    using T = std::remove_cvref_t<decltype(a(i, j))>;
    if constexpr (TR == Transpose::NoTrans) return static_cast<T>(a(i, j));
    else if constexpr (TR == Transpose::Trans) return static_cast<T>(a(j, i));
    else return Arith<T>::conj(static_cast<T>(a(j, i)));
}
template <class MA> inline constexpr bool is_tile_v = false;
template <class T, int R, int C> inline constexpr bool is_tile_v<Tile<T, R, C>> = true;
template <Transpose TR, int R, int C> inline constexpr int op_rows = TR == Transpose::NoTrans ? R : C;
template <Transpose TR, int R, int C> inline constexpr int op_cols = TR == Transpose::NoTrans ? C : R;
}  // namespace detail

/// @brief `C := alpha op(A) op(B) + beta C` (BLAS gemm) on register tiles, on one item.
///
/// `op(A)` is M x K, `op(B)` is K x N and `C` is M x N; the shapes are checked at compile time.
/// Per element, the k loop runs innermost with `Ops::mac` into a zero accumulator, then one
/// epilogue store. A `Zero` tag for `beta` never reads `C`.
/// @ingroup api_device_blas3
/// @tparam TA `op(A)`: `Transpose::NoTrans` (default), `Trans` or `ConjTrans`.
/// @tparam TB `op(B)`, likewise.
/// @tparam Shape the epilogue, `store_epi<Shape, Ops>` (default `epi::Sum`).
/// @tparam Ops the arithmetic policy (`arith::FmaSplit` selects the split epilogue arithmetic).
/// @param alpha the scalar `alpha`: a value or a tag.
/// @param A the register tile `A`.
/// @param B the register tile `B`.
/// @param beta the scalar `beta`: a value or a tag.
/// @param[in,out] C the register tile `C`.
template <Transpose TA = Transpose::NoTrans, Transpose TB = Transpose::NoTrans, class Shape = epi::Sum, class Ops = arith::Explicit,
          class AS, class BS, class Scalar, int AR, int AC, int BR, int BC, int M, int N>
BATCHLAS_DEVICE_INLINE void gemm(const Thread&, AS alpha, const Tile<Scalar, AR, AC>& A, const Tile<Scalar, BR, BC>& B, BS beta, Tile<Scalar, M, N>& C) {
    constexpr int K = detail::op_cols<TA, AR, AC>;
    static_assert(detail::op_rows<TA, AR, AC> == M && detail::op_cols<TB, BR, BC> == N && detail::op_rows<TB, BR, BC> == K,
                  "gemm (register tile): op(A) is M x K, op(B) is K x N, C is M x N");
    using Ar = ops_arith_t<Ops, Scalar>;
#pragma unroll
    for (int j = 0; j < N; ++j)
#pragma unroll
        for (int i = 0; i < M; ++i) {
            Scalar acc = Ar::zero();
#pragma unroll
            for (int k = 0; k < K; ++k) Ar::mac(acc, detail::tile_op<TA>(A, i, k), detail::tile_op<TB>(B, k, j));
            store_epi<Shape, Ops>(alpha, acc, beta, C(i, j));
        }
}

/// @brief `C -= A B` on a register tile, one `Ops::msub` per term in k order: the factorizations'
/// trailing update.
///
/// `A = a(x, k)` (M x K) and `B = b(k, y)` (K x N) are tiles or any operands (local or global
/// memory, a masked view). Both are staged into registers first (for each k, `A`'s column k then
/// `B`'s row k), then K rank-1 steps in k order.
/// @ingroup api_device_blas3
/// @tparam K the inner dimension.
/// @tparam Ops `arith::FmaSplit`: `sub(c, mul(a, b))`, two roundings; `arith::Fma`: one fused
///   `msub`; default `arith::Explicit`.
/// @param a the M x K operand.
/// @param b the K x N operand.
/// @param[in,out] c the M x N register tile.
template <int K, class Ops = arith::Explicit, class MA, class MB, class Scalar, int M, int N>
BATCHLAS_DEVICE_INLINE void gemm_sub(const Thread&, const MA& a, const MB& b, Tile<Scalar, M, N>& c) {
    using Ar = ops_arith_t<Ops, Scalar>;
    Scalar at[M][K], bt[K][N];
BATCHLAS_UNROLL_FULL
    for (int k = 0; k < K; ++k) {
BATCHLAS_UNROLL_FULL
        for (int x = 0; x < M; ++x) at[x][k] = a(x, k);
BATCHLAS_UNROLL_FULL
        for (int y = 0; y < N; ++y) bt[k][y] = b(k, y);
    }
BATCHLAS_UNROLL_FULL
    for (int k = 0; k < K; ++k)
BATCHLAS_UNROLL_FULL
        for (int x = 0; x < M; ++x)
BATCHLAS_UNROLL_FULL
            for (int y = 0; y < N; ++y) Ar::msub(c(x, y), at[x][k], bt[k][y]);
}
/// @brief `C -= A B` on register tiles, `K` deduced from the tiles.
/// @ingroup api_device_blas3
/// @tparam Ops the arithmetic policy, as for the operand form.
/// @param ex the `Thread` executor.
/// @param A the M x K register tile.
/// @param B the K x N register tile.
/// @param[in,out] C the M x N register tile.
template <class Ops = arith::Explicit, class Scalar, int M, int K, int N>
BATCHLAS_DEVICE_INLINE void gemm_sub(const Thread& ex, const Tile<Scalar, M, K>& A, const Tile<Scalar, K, N>& B, Tile<Scalar, M, N>& C) {
    gemm_sub<K, Ops>(ex, A, B, C);
}

/// @brief Triangular solve with a register tile `B`, overwritten by `X` (BLAS trsm) on one item.
///
/// `Side::Left`: `op(A) X = alpha B` with `A` M x M. `Side::Right`: `X op(A) = alpha B` with `A`
/// N x N. `A` is a tile or any `a(i, j)` operand; a tile's shape is checked at compile time. Each
/// element's terms run in the reference BLAS order (dtrsm), so a real `Ops::msub` chain matches it
/// term by term. A zero diagonal element of a `Diag::NonUnit` `A` gives Inf or NaN.
/// @ingroup api_device_blas3
/// @tparam SD `Side::Left` or `Side::Right`.
/// @tparam UL the stored triangle of `A`; the other is not read.
/// @tparam TR `op(A)`: `Transpose::NoTrans`, `Trans` or `ConjTrans`.
/// @tparam DG `Diag::Unit` never reads the diagonal; `Diag::NonUnit` divides by it.
/// @tparam Ops the arithmetic policy.
/// @param alpha_ the scalar `alpha`: a value or a tag (`One` skips the scaling).
/// @param A the triangular matrix.
/// @param[in,out] B the M x N register tile: the right-hand side on entry, `X` on return.
template <Side SD, Uplo UL, Transpose TR, Diag DG, class Ops = arith::Explicit, class AS, class MA, class Scalar, int M, int N>
BATCHLAS_DEVICE_INLINE void trsm(const Thread&, AS alpha_, const MA& A, Tile<Scalar, M, N>& B) {
    using Ar = ops_arith_t<Ops, Scalar>;
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
        if constexpr (detail::is_tile_v<MA>) static_assert(MA::rows == M && MA::cols == M, "trsm (register tile, Left): A is M x M");
        if constexpr (op_lower) {  // row i by the rows k < i, A(i, k) read once for every column
BATCHLAS_UNROLL_FULL
            for (int i = 0; i < M; ++i) {
BATCHLAS_UNROLL_FULL
                for (int k = 0; k < i; ++k) {
                    const Scalar aik = detail::tile_op<TR>(A, i, k);
BATCHLAS_UNROLL_FULL
                    for (int j = 0; j < N; ++j) Ar::msub(B(i, j), aik, B(k, j));
                }
                if constexpr (DG == Diag::NonUnit) {
BATCHLAS_UNROLL_FULL
                    for (int j = 0; j < N; ++j) B(i, j) = Ar::div(B(i, j), detail::tile_op<TR>(A, i, i));
                }
            }
        } else {
#pragma unroll
            for (int j = 0; j < N; ++j) {
#pragma unroll
                for (int i = M - 1; i >= 0; --i) {
#pragma unroll
                    for (int k = i + 1; k < M; ++k) Ar::msub(B(i, j), detail::tile_op<TR>(A, i, k), B(k, j));
                    if constexpr (DG == Diag::NonUnit) B(i, j) = Ar::div(B(i, j), detail::tile_op<TR>(A, i, i));
                }
            }
        }
    } else {
        if constexpr (detail::is_tile_v<MA>) static_assert(MA::rows == N && MA::cols == N, "trsm (register tile, Right): A is N x N");
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
