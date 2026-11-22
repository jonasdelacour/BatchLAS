// SPDX-License-Identifier: MIT
#pragma once
/// @file
/// @brief Level 3 tile bodies and schedules (@ref api_device_lowlevel, not covered by the
/// compatibility promise).
///
///   - `l3::reg_tile_mma`: `acc += op(A) op(B)` over one output tile of a `RegGeom`: staging, k
///     loop and micro-kernel; the body of the host's register-tiled gemm and of the in-group
///     operations.
///   - `l3::for_each_tile`: every tile of one matrix on one work-group, a barrier after each.
/// @ingroup api_device_lowlevel
// The host kernels take OneTile (their entry reads the group ids); a persistent loop is a separate
// instantiation (it makes m0 / n0 loop-carried) and never a host route (agent-guide section 11).

#include <batchlas/device/core.hh>
#include <batchlas/device/level3/cfg.hh>
#include <batchlas/device/level3/mma.hh>
#include <batchlas/device/level3/pipeline.hh>
#include <batchlas/device/operand.hh>

namespace batchlas::device::l3 {

namespace detail {
// ceil(value / divisor) as the tuned kernel spelled it (a call, run-time divisor): an inline
// constant TileK keeps nsw, the k guard becomes k > 0 and the reg tiles' REG moves.
inline int slab_count(int value, int divisor) { return (value + divisor - 1) / divisor; }
}  // namespace detail

/// @brief `acc += op(A)(m0 + :, 0 : k) op(B)(0 : k, n0 + :)` for the item `(local_row, local_col)` of a `G` tile.
///
/// Ends on a barrier after the last read of both stages (`kloop`), so an epilogue may reuse them.
/// COLLECTIVE.
/// @ingroup api_device_lowlevel
/// @tparam G the geometry (`RegGeom`).
/// @tparam Exact `true`: `k % TileK == 0`, no tail tests.
/// @tparam Ops the arithmetic policy (default `arith::Native`).
/// @param ex the executor.
/// @param local_row this item's tile row.
/// @param local_col this item's tile column.
/// @param tile_a the `op(A)` stages: `Stages * StageASize` local elements (a pointer or a
///   `local_accessor`).
/// @param tile_b the `op(B)` stages: `Stages * StageBSize` local elements.
/// @param stage called as `stage(int kk0, dst_a, dst_b)`: write `op(A)`'s `TileM` x `TileK` slab at
///   k offset `kk0` through `dst_a(r, c)` and `op(B)`'s `TileK` x `TileN` slab through
///   `dst_b(r, c)`, both zero outside the matrix.
/// @param k the depth.
/// @param[in,out] accum this item's register tile.
template <class G, bool Exact, class Ops = arith::Native, class Ex, class Mem, class Stage, class Scalar>
BATCHLAS_DEVICE_INLINE void reg_tile_mma(const Ex& ex, int local_row, int local_col, const Mem& tile_a, const Mem& tile_b, Stage&& stage,
                                      const int& k, Tile<Scalar, G::ThreadTileRows, G::ThreadTileCols>& accum) {
    auto load_stage = [&](int kk0, int s) BATCHLAS_DEVICE_BODY {
        auto* tile_a_stage = &tile_a[s * G::StageASize];
        auto* tile_b_stage = &tile_b[s * G::StageBSize];
        auto dst_a = [&](int r, int c) -> auto& { return tile_a_stage[c * G::TileAStride + r]; };
        auto dst_b = [&](int r, int c) -> auto& { return tile_b_stage[G::b_slot(c, r)]; };
        stage(kk0, dst_a, dst_b);
    };
    auto accumulate_stage = [&](int stage, int kk0) {
        const auto* tile_a_stage = &tile_a[stage * G::StageASize];
        const auto* tile_b_stage = &tile_b[stage * G::StageBSize];
        mma_slab<G::TileK, G::UnrollK, Exact, Ops>(
            accum, kk0, k, [&](int t, int i) { return tile_a_stage[t * G::TileAStride + local_row * G::ThreadTileRows + i]; },
            [&](int t, int j) { return tile_b_stage[G::b_slot(local_col * G::ThreadTileCols + j, t)]; });
    };
    const int tile_count = Exact ? (k / G::TileK) : detail::slab_count(k, G::TileK);
    kloop<G::Stages, Exact, G::TileK>(ex, tile_count, load_stage, accumulate_stage);
}

/// @brief Calls `f(t)` for every `t` in `[0, count)` on the whole executor, with a barrier after each.
///
/// The barrier lets the staged epilogue reuse the stage memory the next tile's first load
/// writes. COLLECTIVE.
/// @ingroup api_device_lowlevel
/// @pre `f` is uniform in `t`: every item makes the same calls.
/// @param ex the executor.
/// @param count the number of tiles.
/// @param f called as `f(int t)`.
template <class Ex, class F> BATCHLAS_DEVICE_INLINE void for_each_tile(const Ex& ex, int count, F&& f) {
    for (int t = 0; t < count; ++t) {
        f(t);
        ex.barrier();
    }
}

}  // namespace batchlas::device::l3
