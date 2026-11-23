// SPDX-License-Identifier: MIT
#pragma once
/// @file
/// @brief Level 3 `C` epilogues of the tuned gemm (@ref api_device_lowlevel, not covered by the
/// compatibility promise).
///
///   - `l3::LinearEpilogue`: `C := alpha acc + beta C`, `beta` compared with 0 at run time.
///   - `l3::NoMask`, `l3::TriMask`: which elements of a tile the epilogue writes.
///   - `l3::store_tile_staged`: a register tile through local memory to `C`, coalesced at any `ldc`.
///
/// `C := alpha acc + beta C` with a host `beta_zero` flag is `store_epi<epi::MulMac, Ops>` with a
/// `MaybeZero` beta (`arith.hh`).
/// @ingroup api_device_lowlevel
// Moved from the tuned gemm kernels unchanged.

#include <cstddef>
#include <type_traits>

#include <batchlas/blas/enums.hh>
#include <batchlas/device/arith.hh>
#include <batchlas/device/core.hh>
#include <batchlas/device/operand.hh>

namespace batchlas::device::l3 {

// \f$\beta = 0\f$ is \f$\alpha \cdot \mathrm{accum}\f$ alone and never reads C: \f$0 \cdot \mathrm{NaN}\f$ is NaN, and \f$-0 + 0 \cdot 0 = +0\f$.
// evidence: docs/design/known-defects.md#defect-11-native-gemm-reads-c-at-beta-zero
/// @brief `C := alpha acc + beta C` in `Scalar`'s own operators, with `beta == 0` tested at run time.
///
/// With `beta == 0` the result is `alpha * acc` alone and `C` is not read.
/// @ingroup api_device_lowlevel
/// @tparam Scalar the element type.
template <class Scalar>
struct LinearEpilogue {
    /// @brief `alpha * accum + beta * prior`, or `alpha * accum` when `beta == 0`.
    /// @param alpha the product's scale.
    /// @param beta the prior value's scale.
    /// @param accum the accumulated product.
    /// @param prior the prior value of the element.
    static Scalar apply(Scalar alpha, Scalar beta, Scalar accum, Scalar prior) {
        return beta == Scalar(0) ? alpha * accum : alpha * accum + beta * prior;
    }

    /// @brief `*dst := apply(alpha, beta, accum, *dst)`; `*dst` is not read when `beta == 0`.
    /// @param dst the element.
    /// @param alpha the product's scale.
    /// @param beta the prior value's scale.
    /// @param accum the accumulated product.
    static void store(Scalar* dst, Scalar alpha, Scalar beta, Scalar accum) {
        *dst = beta == Scalar(0) ? alpha * accum : apply(alpha, beta, accum, *dst);
    }
};

/// @brief Epilogue mask: write every element of the tile inside the matrix.
/// @ingroup api_device_lowlevel
struct NoMask {};
/// @brief Epilogue mask: write the `UL` triangle of `C` only, with a real diagonal under `HermDiag`.
/// @ingroup api_device_lowlevel
/// @tparam UL `Uplo::Lower` keeps `row >= col`, `Uplo::Upper` keeps `row <= col`.
/// @tparam HermDiag `true` sets `Im C(i, i)` to 0, as zherk does (herk, her2k).
template <Uplo UL, bool HermDiag = false> struct TriMask {
    /// @brief Whether element `(row, col)` is written.
    /// @param row the row in `C`.
    /// @param col the column in `C`.
    BATCHLAS_DEVICE_INLINE bool keep(int row, int col) const { return UL == Uplo::Lower ? row >= col : row <= col; }
    /// @brief The value written at `(row, col)`: `v`, or its real part on the diagonal under `HermDiag`.
    /// @param row the row in `C`.
    /// @param col the column in `C`.
    /// @param v the epilogue's value.
    template <class Scalar> BATCHLAS_DEVICE_INLINE Scalar fix(int row, int col, Scalar v) const {
        using A = Arith<Scalar>;
        if constexpr (HermDiag && A::is_complex) {
            if (row == col) return A::make(A::re(v), typename A::real_t(0));
        }
        return v;
    }
};

/// @brief `C(m0 + r, n0 + c) := alpha acc + beta C` for a `TileM` x `TileN` tile held in the items' register tiles.
///
/// Item `(local_row, local_col)` holds rows `local_row * TR + i` and columns `local_col * TC + j`.
/// The columns pass through local memory (`lo` holds `LoElems`, then `hi` holds `HiElems`, with
/// the odd column stride `TileM + 1`), then the stores run down `C`'s columns (coalesced at any
/// `ldc`), `Batch` prior values in flight per item. `beta == 0` never reads `C`. COLLECTIVE.
/// @ingroup api_device_lowlevel
/// @pre the caller's last read of `lo` and `hi` ended before a barrier.
/// @tparam TileM tile rows.
/// @tparam TileN tile columns.
/// @tparam Threads the work-group size.
/// @tparam LoElems the elements of the `lo` buffer; with `HiElems`, at least one column.
/// @tparam HiElems the elements of the `hi` buffer.
/// @tparam Batch prior values in flight per item (default 2).
/// @tparam Ops `arith::Native` (default): the tuned expression (`LinearEpilogue`); another policy
///   forms `alpha v + beta prior` through `Arith` (no `std::complex` operator).
/// @tparam Mask `NoMask` (default) or a `TriMask`: the elements it does not keep are dropped.
/// @param ex the executor.
/// @param lid this item's index.
/// @param local_row this item's tile row.
/// @param local_col this item's tile column.
/// @param acc this item's register tile.
/// @param lo the first local buffer.
/// @param hi the second local buffer.
/// @param c_item `C`'s first element.
/// @param ldc `C`'s leading dimension.
/// @param m0 the tile's first row.
/// @param n0 the tile's first column.
/// @param m `C`'s row count.
/// @param n `C`'s column count.
/// @param alpha the product's scale.
/// @param beta `C`'s scale.
/// @param mask the mask object.
template <int TileM, int TileN, int Threads, int LoElems, int HiElems, int Batch = 2, class Ops = arith::Native, class Ex, class Scalar,
          int TR, int TC, class Mask = NoMask, class LP>
BATCHLAS_DEVICE_INLINE void store_tile_staged(const Ex& ex, int lid, int local_row, int local_col, const Tile<Scalar, TR, TC>& acc,
                                           LP lo, LP hi, Scalar* c_item, int ldc, int m0, int n0, int m, int n, Scalar alpha, Scalar beta,
                                           [[maybe_unused]] Mask mask = {}) {
    constexpr int CS = TileM + 1;
    constexpr int LoCols = LoElems / CS;
    constexpr int HiCols = HiElems / CS;
    constexpr int CW = LoCols + HiCols;
    constexpr int Passes = (TileN + CW - 1) / CW;
    static_assert(CW >= 1, "store_tile_staged: the stage must hold one C column");
    auto slot = [&](int c, int r) -> LP { return c < LoCols ? lo + c * CS + r : hi + (c - LoCols) * CS + r; };
    const bool beta_zero = beta == Scalar(0);
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
            Scalar prior[Batch];
#pragma unroll
            for (int b = 0; b < Batch; ++b) {
                bool ok;
                const std::ptrdiff_t off = at(e0 + b * Threads, ok);
                prior[b] = (ok && !beta_zero) ? c_item[off] : Scalar(0);
            }
#pragma unroll
            for (int b = 0; b < Batch; ++b) {
                const int e = e0 + b * Threads;
                bool ok;
                const std::ptrdiff_t off = at(e, ok);
                if (!ok) continue;
                const Scalar v = *slot(e / TileM, e % TileM);
                Scalar out;
                if constexpr (std::is_same_v<Ops, arith::Native>) {
                    out = beta_zero ? alpha * v : LinearEpilogue<Scalar>::apply(alpha, beta, v, prior[b]);
                } else {
                    using Ar = arith_t<Ops, Scalar>;
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

}  // namespace batchlas::device::l3
