// SPDX-License-Identifier: MIT
#pragma once
// Level 3 C epilogues (v22 design B.2), moved from the tuned gemm kernels unchanged:
//   LinearEpilogue          C := alpha acc + beta C, beta compared with 0 at run time
//   epilogue_store_fused    C := alpha acc + beta C with a host beta_zero flag
//   store_tile_staged       a register tile through local memory to C, coalesced at any ldc

#include <cstddef>
#include <type_traits>

#include <batchlas/device/arith.hh>
#include <batchlas/device/core.hh>
#include <batchlas/device/operand.hh>

namespace batchlas::device::inline v2::l3 {

// \f$\beta = 0\f$ is \f$\alpha \cdot \mathrm{accum}\f$ alone and never reads C: \f$0 \cdot \mathrm{NaN}\f$ is NaN, and \f$-0 + 0 \cdot 0 = +0\f$.
// evidence: docs/design/known-defects.md#defect-11-native-gemm-reads-c-at-beta-zero
template <typename T>
struct LinearEpilogue {
    static T apply(T alpha, T beta, T accum, T prior) {
        return beta == T(0) ? alpha * accum : alpha * accum + beta * prior;
    }

    static void store(T* dst, T alpha, T beta, T accum) {
        *dst = beta == T(0) ? alpha * accum : apply(alpha, beta, accum, *dst);
    }
};

// *dst := alpha acc + beta *dst with the host's beta_zero flag (beta_zero never reads *dst). A
// complex takes Ops' product and accumulate (arith::Fma: sycl_device's dev_mul and fma_acc), a
// real scalar the operators.
template <class Ops, class T> BATCHLAS_DLA_INLINE void epilogue_store_fused(T* dst, T alpha, T beta, bool beta_zero, T acc) {
    using Ar = arith_t<Ops, T>;
    if constexpr (Ar::is_complex) {
        T out = Ar::mul(alpha, acc);
        if (!beta_zero) Ar::mac(out, beta, *dst);
        *dst = out;
    } else {
        *dst = beta_zero ? alpha * acc : alpha * acc + beta * *dst;
    }
}

// C(m0 + r, n0 + c) := alpha acc + beta C for a TileM x TileN tile whose item (local_row,
// local_col) holds rows local_row * TR + i, cols local_col * TC + j. Columns pass through local
// memory (lo: LoElems, then hi: HiElems; odd stride TileM + 1), then lane = row stores run down
// C's columns (coalesced at any ldc), Batch priors in flight per item. Precondition: the caller's
// last read of lo and hi ended before a barrier. beta == 0 never reads C.
template <int TileM, int TileN, int Threads, int LoElems, int HiElems, int Batch = 2, class Ex, class T, int TR, int TC>
BATCHLAS_DLA_INLINE void store_tile_staged(const Ex& ex, int lid, int local_row, int local_col, const Tile<T, TR, TC>& acc,
                                           T* lo, T* hi, T* c_item, int ldc, int m0, int n0, int m, int n, T alpha, T beta) {
    constexpr int CS = TileM + 1;
    constexpr int LoCols = LoElems / CS;
    constexpr int HiCols = HiElems / CS;
    constexpr int CW = LoCols + HiCols;
    constexpr int Passes = (TileN + CW - 1) / CW;
    static_assert(CW >= 1, "store_tile_staged: the stage must hold one C column");
    auto slot = [&](int c, int r) -> T* { return c < LoCols ? lo + c * CS + r : hi + (c - LoCols) * CS + r; };
    const bool beta_zero = beta == T(0);
#pragma unroll 1
    for (int pass = 0; pass < Passes; ++pass) {
        const int c0 = pass * CW;
        const int cw = TileN - c0 < CW ? TileN - c0 : CW;
#pragma unroll
        for (int j = 0; j < TC; ++j) {
            const int c = local_col * TC + j - c0;
            if (c < 0 || c >= cw) continue;
#pragma unroll
            for (int i = 0; i < TR; ++i) *slot(c, local_row * TR + i) = acc(i, j);
        }
        ex.barrier();
        const int count = TileM * cw;
        auto at = [&](int e, bool& ok) {
            const int row = m0 + e % TileM, col = n0 + c0 + e / TileM;
            ok = e < count && row < m && col < n;
            return row + static_cast<std::ptrdiff_t>(col) * ldc;
        };
#pragma unroll 1
        for (int e0 = lid; e0 < count; e0 += Batch * Threads) {
            T prior[Batch];
#pragma unroll
            for (int b = 0; b < Batch; ++b) {
                bool ok;
                const std::ptrdiff_t off = at(e0 + b * Threads, ok);
                prior[b] = (ok && !beta_zero) ? c_item[off] : T(0);
            }
#pragma unroll
            for (int b = 0; b < Batch; ++b) {
                const int e = e0 + b * Threads;
                bool ok;
                const std::ptrdiff_t off = at(e, ok);
                if (!ok) continue;
                const T v = *slot(e / TileM, e % TileM);
                c_item[off] = beta_zero ? alpha * v : LinearEpilogue<T>::apply(alpha, beta, v, prior[b]);
            }
        }
        if (pass + 1 < Passes) ex.barrier();
    }
}

}  // namespace batchlas::device::inline v2::l3
