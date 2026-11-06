// SPDX-License-Identifier: MIT
#pragma once
// Level 3 C epilogues (v22 design B.2), moved from the tuned gemm kernels unchanged:
//   LinearEpilogue          C := alpha acc + beta C, beta compared with 0 at run time
//   NoMask, TriMask         which elements of a tile the epilogue writes
//   store_tile_staged       a register tile through local memory to C, coalesced at any ldc
// C := alpha acc + beta C with a host beta_zero flag is core store<epi::MulMac, Ops> with
// MaybeZero beta (arith.hh).

#include <cstddef>
#include <type_traits>

#include <batchlas/blas/enums.hh>
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

// Which elements of an output tile an epilogue writes (the Mask of design B.2). NoMask: every
// element inside the matrix. TriMask<UL, HermDiag>: the UL triangle of C (row >= col for Lower),
// and with HermDiag a real diagonal (herk / her2k: Im C(i, i) := 0, as zherk does).
struct NoMask {};
template <Uplo UL, bool HermDiag = false> struct TriMask {
    BATCHLAS_DLA_INLINE bool keep(int row, int col) const { return UL == Uplo::Lower ? row >= col : row <= col; }
    template <class T> BATCHLAS_DLA_INLINE T fix(int row, int col, T v) const {
        using A = Arith<T>;
        if constexpr (HermDiag && A::is_complex) {
            if (row == col) return A::make(A::re(v), typename A::real_t(0));
        }
        return v;
    }
};

// C(m0 + r, n0 + c) := alpha acc + beta C for a TileM x TileN tile whose item (local_row,
// local_col) holds rows local_row * TR + i, cols local_col * TC + j. Columns pass through local
// memory (lo: LoElems, then hi: HiElems; odd stride TileM + 1), then lane = row stores run down
// C's columns (coalesced at any ldc), Batch priors in flight per item. Precondition: the caller's
// last read of lo and hi ended before a barrier. beta == 0 never reads C. Ops = arith::Native is
// the tuned expression (LinearEpilogue); another Ops forms alpha v + beta prior through Arith
// (no std::complex operator*). mask drops the elements it does not keep (design B.2 Mask).
template <int TileM, int TileN, int Threads, int LoElems, int HiElems, int Batch = 2, class Ops = arith::Native, class Ex, class T,
          int TR, int TC, class Mask = NoMask>
BATCHLAS_DLA_INLINE void store_tile_staged(const Ex& ex, int lid, int local_row, int local_col, const Tile<T, TR, TC>& acc,
                                           T* lo, T* hi, T* c_item, int ldc, int m0, int n0, int m, int n, T alpha, T beta,
                                           [[maybe_unused]] Mask mask = {}) {
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
                T out;
                if constexpr (std::is_same_v<Ops, arith::Native>) {
                    out = beta_zero ? alpha * v : LinearEpilogue<T>::apply(alpha, beta, v, prior[b]);
                } else {
                    using Ar = arith_t<Ops, T>;
                    out = beta_zero ? Ar::mul(alpha, v) : Ar::add(Ar::mul(alpha, v), Ar::mul(beta, prior[b]));
                }
                if constexpr (std::is_same_v<Mask, NoMask>) {
                    c_item[off] = out;
                } else {
                    const int row = m0 + e % TileM, col = n0 + c0 + e / TileM;
                    if (mask.keep(row, col)) c_item[off] = mask.fix(row, col, out);
                }
            }
        }
        if (pass + 1 < Passes) ex.barrier();
    }
}

}  // namespace batchlas::device::inline v2::l3
