// SPDX-License-Identifier: MIT
#pragma once
// Level 3 tile bodies and schedules (v22 design B.2):
//   reg_tile_mma   acc += op(A) op(B) over one output tile of a RegGeom: staging, k loop, micro
//                  kernel; the body of GemmRegisterTiledKernel and of the in-group ops
//   for_each_tile  Persistent: every tile of one matrix on one work-group, a barrier after each
// The host kernels take OneTile (their entry reads the group ids); a persistent loop is a separate
// instantiation (it makes m0 / n0 loop-carried) and never a host route (agent-guide section 11).

#include <batchlas/device/core.hh>
#include <batchlas/device/level3/cfg.hh>
#include <batchlas/device/level3/mma.hh>
#include <batchlas/device/level3/pipeline.hh>
#include <batchlas/device/operand.hh>

namespace batchlas::device::inline v2::l3 {

/// acc += op(A)(m0 + :, 0 : k) op(B)(0 : k, n0 + :) for the item (local_row, local_col) of a G
/// tile. stage_a(kk0, dst) writes op(A)'s TileM x TileK slab at k offset kk0 through dst(r, c);
/// stage_b(kk0, dst) op(B)'s TileK x TileN slab (both zero outside the matrix). tile_a / tile_b:
/// Stages * StageASize / StageBSize local elements. Exact: k % TileK == 0, no tail tests. Ends
/// on a barrier after the last read of both stages (kloop), so an epilogue may reuse them.
template <class G, bool Exact, class Ops = arith::Native, class Ex, class T, class StageA, class StageB>
BATCHLAS_DLA_INLINE void reg_tile_mma(const Ex& ex, int local_row, int local_col, T* tile_a, T* tile_b, const int& k, StageA&& stage_a,
                                      StageB&& stage_b, Tile<T, G::ThreadTileRows, G::ThreadTileCols>& accum) {
    auto load_stage = [&](int kk0, int stage) BATCHLAS_DLA_BODY {
        T* tile_a_stage = tile_a + stage * G::StageASize;
        T* tile_b_stage = tile_b + stage * G::StageBSize;
        auto dst_a = [&](int r, int c) -> T& { return tile_a_stage[c * G::TileAStride + r]; };
        auto dst_b = [&](int r, int c) -> T& { return tile_b_stage[G::b_slot(c, r)]; };
        stage_a(kk0, dst_a);
        stage_b(kk0, dst_b);
    };
    auto accumulate_stage = [&](int stage, int kk0) {
        const T* tile_a_stage = tile_a + stage * G::StageASize;
        const T* tile_b_stage = tile_b + stage * G::StageBSize;
        mma_slab<G::TileK, G::UnrollK, Exact, Ops>(
            accum, kk0, k, [&](int t, int i) { return tile_a_stage[t * G::TileAStride + local_row * G::ThreadTileRows + i]; },
            [&](int t, int j) { return tile_b_stage[G::b_slot(local_col * G::ThreadTileCols + j, t)]; });
    };
    const int tile_count = Exact ? (k / G::TileK) : (k + G::TileK - 1) / G::TileK;
    kloop<G::Stages, Exact, G::TileK>(ex, tile_count, load_stage, accumulate_stage);
}

/// f(t) for t = 0 .. count - 1 on the whole executor, one barrier after each: the staged
/// epilogue reuses the stage memory the next tile's first load writes. f must be uniform in t.
template <class Ex, class F> BATCHLAS_DLA_INLINE void for_each_tile(const Ex& ex, int count, F&& f) {
    for (int t = 0; t < count; ++t) {
        f(t);
        ex.barrier();
    }
}

}  // namespace batchlas::device::inline v2::l3
