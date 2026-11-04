// SPDX-License-Identifier: MIT
#pragma once
// Level 3 register micro-kernels (v22 design B.2), moved from the tuned gemm kernels unchanged:
//   outer_mac     acc += a b^T on a register tile
//   mma_slab      one staged k slab of rank-1 updates
//   gemv_t_frag   acc(j) += sum_l a(l) B(l, j), x and y in registers

#include <type_traits>
#include <utility>

#include <batchlas/device/arith.hh>
#include <batchlas/device/core.hh>
#include <batchlas/device/detail/kernel_attrs.hh>
#include <batchlas/device/operand.hh>

namespace batchlas::device::inline v2::l3 {

// acc += a b^T on a TR x TC register tile. Full: the full-unroll pragmas; otherwise none (the
// register-tiled kernel's own loops, unrolled as the compiler decides).
template <bool Full = true, class Ops = arith::Native, class T, int TR, int TC>
BATCHLAS_DLA_INLINE void outer_mac(Tile<T, TR, TC>& acc, const T (&a)[TR], const T (&b)[TC]) {
    using A = arith_t<Ops, T>;
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

// One staged k slab: for each t < TileK (and kk0 + t < k unless Exact) a(i) = a_at(t, i),
// b(j) = b_at(t, j), acc += a b^T; UnrollK steps per outer iteration, no pragmas. k by
// reference, like the tuned lambda's closure read: by value, SimplifyCFG merged the two exits
// first and the TileK = 8 slab stayed rolled (32x32x8: 34 -> 6 FFMA).
template <int TileK, int UnrollK, bool Exact, class Ops = arith::Native, class T, int TR, int TC, class FA, class FB>
BATCHLAS_DLA_INLINE void mma_slab(Tile<T, TR, TC>& acc, int kk0, const int& k, FA&& a_at, FB&& b_at) {
    auto step = [&](int t) BATCHLAS_DLA_BODY {
        T a_frag[TR];
        T b_frag[TC];
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
template <int L0, int Lc, class Ar, class T, int NA, int NC, class FB>
BATCHLAS_DLA_INLINE void gemv_t_frag_chunk(Frag<T, NC>& acc, const Frag<T, NA>& a, FB& b_at, int k) {
BATCHLAS_UNROLL_FULL
    for (int l = L0; l < L0 + Lc; ++l) {
        if (l >= k) continue;  // uniform; `continue` keeps the unroll, so a[] stays in registers
BATCHLAS_UNROLL_FULL
        for (int j = 0; j < NC; ++j) Ar::mac(acc[j], a[l], b_at(l, j));
    }
}
}  // namespace detail

// acc(j) += sum_{l < k} a(l) B(l, j), l outermost: gemv^T with x and y in registers. Chunk = NA
// is one unrolled loop; Chunk = NA / 4 is four sequential ones (one complex loop over 64 l passes
// LLVM's unroll threshold and a[] spills; a lambda per chunk put 160-184 B on the stack).
template <int Chunk, class Ops, class T, int NA, int NC, class FB>
BATCHLAS_DLA_INLINE void gemv_t_frag(Frag<T, NC>& acc, const Frag<T, NA>& a, FB&& b_at, int k) {
    using Ar = arith_t<Ops, T>;
    static_assert(Chunk == NA || Chunk * 4 == NA, "gemv_t_frag: one loop, or four");
    detail::gemv_t_frag_chunk<0, Chunk, Ar>(acc, a, b_at, k);
    if constexpr (NA / Chunk > 1) {
        detail::gemv_t_frag_chunk<Chunk, Chunk, Ar>(acc, a, b_at, k);
        detail::gemv_t_frag_chunk<2 * Chunk, Chunk, Ar>(acc, a, b_at, k);
        detail::gemv_t_frag_chunk<3 * Chunk, Chunk, Ar>(acc, a, b_at, k);
    }
}

}  // namespace batchlas::device::inline v2::l3
