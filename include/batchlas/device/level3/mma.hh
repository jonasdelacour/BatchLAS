// SPDX-License-Identifier: MIT
#pragma once
/// @file
/// @brief Level 3 register micro-kernels of the tuned gemm (@ref api_device_lowlevel, not covered
/// by the compatibility promise).
///
///   - `l3::outer_mac`: `acc += a b^T` on a register tile.
///   - `l3::mma_slab`: one staged k slab of rank-1 updates.
///   - `l3::gemv_t_frag`: `acc(j) += sum_l a(l) B(l, j)`, `x` and `y` in registers.
/// @ingroup api_device_lowlevel
// Moved from the tuned gemm kernels unchanged.

#include <type_traits>
#include <utility>

#include <batchlas/device/arith.hh>
#include <batchlas/device/core.hh>
#include <batchlas/device/detail/kernel_attrs.hh>
#include <batchlas/device/operand.hh>

namespace batchlas::device::l3 {

/// @brief `acc += a b^T` on a `TR` x `TC` register tile.
/// @ingroup api_device_lowlevel
/// @tparam Full `true` (default): full-unroll pragmas; `false`: none (unrolled as the compiler decides).
/// @tparam Ops the arithmetic policy (default `arith::Native`).
/// @param[in,out] acc the register tile.
/// @param a the `TR` column values.
/// @param b the `TC` row values.
template <bool Full = true, class Ops = arith::Native, class Scalar, int TR, int TC>
BATCHLAS_DEVICE_INLINE void outer_mac(Tile<Scalar, TR, TC>& acc, const Scalar (&a)[TR], const Scalar (&b)[TC]) {
    using A = arith_t<Ops, Scalar>;
    if constexpr (Full) {
BATCHLAS_UNROLL_FULL
        for (int i = 0; i < TR; ++i) {
BATCHLAS_UNROLL_FULL
            for (int j = 0; j < TC; ++j) A::mac(acc(i, j), a[i], b[j]);
        }
    } else {
        for (int i = 0; i < TR; ++i) {
            for (int j = 0; j < TC; ++j) A::mac(acc(i, j), a[i], b[j]);
        }
    }
}

// k by reference, like the tuned lambda's closure read: by value, SimplifyCFG merged the two
// exits first and the TileK = 8 slab stayed rolled (32x32x8: 34 -> 6 FFMA).
/// @brief One staged k slab: for each `t < TileK` (and `kk0 + t < k` unless `Exact`), `acc += a b^T`
/// with `a(i) = a_at(t, i)` and `b(j) = b_at(t, j)`.
/// @ingroup api_device_lowlevel
/// @tparam TileK the slab depth.
/// @tparam UnrollK steps per outer iteration (no pragmas).
/// @tparam Exact `true`: the slab is full, no `k` test.
/// @tparam Ops the arithmetic policy (default `arith::Native`).
/// @param[in,out] acc the register tile.
/// @param kk0 the slab's k offset.
/// @param k the total depth.
/// @param a_at called as `a_at(int t, int i)`.
/// @param b_at called as `b_at(int t, int j)`.
template <int TileK, int UnrollK, bool Exact, class Ops = arith::Native, class Scalar, int TR, int TC, class FA, class FB>
BATCHLAS_DEVICE_INLINE void mma_slab(Tile<Scalar, TR, TC>& acc, int kk0, const int& k, FA&& a_at, FB&& b_at) {
    auto step = [&](int t) BATCHLAS_DEVICE_BODY {
        Scalar a_frag[TR];
        Scalar b_frag[TC];
        for (int i = 0; i < TR; ++i) a_frag[i] = a_at(t, i);
        for (int j = 0; j < TC; ++j) b_frag[j] = b_at(t, j);
        outer_mac<false, Ops>(acc, a_frag, b_frag);
    };
    if constexpr (Exact) {
        for (int t0 = 0; t0 < TileK; t0 += UnrollK) {
            for (int unroll = 0; unroll < UnrollK; ++unroll) step(t0 + unroll);
        }
    } else {
        for (int t0 = 0; t0 < TileK && kk0 + t0 < k; t0 += UnrollK) {
            for (int unroll = 0; unroll < UnrollK && t0 + unroll < TileK && kk0 + t0 + unroll < k; ++unroll)
                step(t0 + unroll);
        }
    }
}

namespace detail {
template <int L0, int Lc, class Ar, class Scalar, int NA, int NC, class FB>
BATCHLAS_DEVICE_INLINE void gemv_t_frag_chunk(Frag<Scalar, NC>& acc, const Frag<Scalar, NA>& a, FB& b_at, int k) {
BATCHLAS_UNROLL_FULL
    for (int l = L0; l < L0 + Lc; ++l) {
        if (l >= k) continue;  // uniform; `continue` keeps the unroll, so a[] stays in registers
BATCHLAS_UNROLL_FULL
        for (int j = 0; j < NC; ++j) Ar::mac(acc[j], a[l], b_at(l, j));
    }
}
}  // namespace detail

// Chunk = NA / 4: one complex loop over 64 l passes LLVM's unroll threshold and a[] spills; a
// lambda per chunk put 160-184 B on the stack.
/// @brief `acc(j) += sum_{l < k} a(l) B(l, j)`, `l` outermost: a transposed gemv with `x` and `y` in registers.
/// @ingroup api_device_lowlevel
/// @tparam Chunk `NA` (one unrolled loop) or `NA / 4` (four sequential unrolled loops).
/// @tparam Ops the arithmetic policy.
/// @param[in,out] acc the `NC` outputs.
/// @param a the `NA` inputs.
/// @param b_at called as `b_at(int l, int j)`.
/// @param k the live depth, at most `NA`.
template <int Chunk, class Ops, class Scalar, int NA, int NC, class FB>
BATCHLAS_DEVICE_INLINE void gemv_t_frag(Frag<Scalar, NC>& acc, const Frag<Scalar, NA>& a, FB&& b_at, int k) {
    using Ar = arith_t<Ops, Scalar>;
    static_assert(Chunk == NA || Chunk * 4 == NA, "gemv_t_frag: one loop, or four");
    detail::gemv_t_frag_chunk<0, Chunk, Ar>(acc, a, b_at, k);
    if constexpr (NA / Chunk > 1) {
        detail::gemv_t_frag_chunk<Chunk, Chunk, Ar>(acc, a, b_at, k);
        detail::gemv_t_frag_chunk<2 * Chunk, Chunk, Ar>(acc, a, b_at, k);
        detail::gemv_t_frag_chunk<3 * Chunk, Chunk, Ar>(acc, a, b_at, k);
    }
}

}  // namespace batchlas::device::l3
