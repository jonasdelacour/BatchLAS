// SPDX-License-Identifier: MIT
#pragma once
// Level 3 k loops (v22 design B.2), moved from the tuned gemm kernels unchanged; the barriers are
// the pipeline's contract. Any executor with barrier().
//   kloop             1-stage or 2-stage double-buffered local stages
//   kloop_prefetch2   register-prefetched double buffer, one barrier per slab
//   kloop_async       Stages - 1 slabs in flight through asynchronous copies

#include <type_traits>
#include <utility>

#include <batchlas/device/core.hh>

namespace batchlas::device::inline v2::l3 {

// load(kk0, stage) fills local stage `stage` with k slab kk0; compute(stage, kk0) consumes it.
// Stages 1: load, barrier, compute, barrier per slab. Stages 2: slab t + 1 loads while slab t
// computes, one barrier per slab. Exact: no tail, and the first load sits behind tiles > 0.
template <int Stages, bool Exact, int TileK, class Ex, class Load, class Compute>
BATCHLAS_DLA_INLINE void kloop(const Ex& ex, int tile_count, Load&& load, Compute&& compute) {
    static_assert(Stages == 1 || Stages == 2, "kloop: one or two local stages");
    if constexpr (Stages == 1) {
        for (int tile_idx = 0; tile_idx < tile_count; ++tile_idx) {
            const int kk0 = tile_idx * TileK;
            load(kk0, 0);
            ex.barrier();
            compute(0, kk0);
            ex.barrier();
        }
    } else if constexpr (Exact) {
        if (tile_count > 0) {
            load(0, 0);
            ex.barrier();
            for (int tile_idx = 0; tile_idx + 1 < tile_count; ++tile_idx) {
                const int current_stage = tile_idx & 1;
                const int next_stage = current_stage ^ 1;
                load((tile_idx + 1) * TileK, next_stage);
                compute(current_stage, tile_idx * TileK);
                ex.barrier();
            }
            const int final_stage = (tile_count - 1) & 1;
            compute(final_stage, (tile_count - 1) * TileK);
            ex.barrier();
        }
    } else {
        load(0, 0);
        ex.barrier();
        for (int tile_idx = 0; tile_idx < tile_count; ++tile_idx) {
            const int kk0 = tile_idx * TileK;
            const int current_stage = tile_idx & 1;
            const int next_tile_idx = tile_idx + 1;
            if (next_tile_idx < tile_count) load(next_tile_idx * TileK, current_stage ^ 1);
            compute(current_stage, kk0);
            ex.barrier();
        }
    }
}

// Register-prefetched double buffer: gload(k0) reads slab k0 into registers, sstore(buf) writes
// them to local buffer buf, compute(buf) consumes one. One barrier per slab: slab s + 1 goes into
// the buffer slab s - 1 was read from, and that read ended before the barrier closing iteration
// s - 1. Unrolled by two so buf is a constant. Ends without a barrier.
template <int TileK, class Ex, class GLoad, class SStore, class Compute>
BATCHLAS_DLA_INLINE void kloop_prefetch2(const Ex& ex, int slabs, GLoad&& gload, SStore&& sstore, Compute&& compute) {
    gload(0);
    sstore(0);
    ex.barrier();
    int s = 0;
    for (; s + 1 < slabs; s += 2) {
        gload((s + 1) * TileK);
        compute(0);
        sstore(1);
        ex.barrier();
        const bool more = s + 2 < slabs;
        if (more) gload((s + 2) * TileK);
        compute(1);
        if (more) sstore(0);
        ex.barrier();
    }
    if (s < slabs) compute(0);  // an odd slab count leaves the last slab in buffer 0
}

// Stages - 1 slabs in flight through asynchronous copies: issue(k0, buf) starts slab k0's copies
// into buffer buf (an integral_constant), commit() closes a copy group, wait() waits until at most
// Stages - 2 groups are pending, drain() until none. Step s waits for slab s, barriers, refills the
// buffer slab s - 1 was read from and computes; one commit per step (empty past the end).
template <int Stages, int TileK, class Ex, class Issue, class Commit, class Wait, class Drain, class Compute>
BATCHLAS_DLA_INLINE void kloop_async(const Ex& ex, int slabs, Issue&& issue, Commit&& commit, Wait&& wait, Drain&& drain,
                                     Compute&& compute) {
    [&]<int... J>(std::integer_sequence<int, J...>) BATCHLAS_DLA_BODY {
        ((J < slabs ? issue(J * TileK, std::integral_constant<int, J>{}) : void(), commit()), ...);
    }(std::make_integer_sequence<int, Stages - 1>{});
    auto step = [&](int s, auto buf) BATCHLAS_DLA_BODY {
        constexpr int b = decltype(buf)::value;
        wait();
        ex.barrier();
        if (s + Stages - 1 < slabs) issue((s + Stages - 1) * TileK, std::integral_constant<int, (b + Stages - 1) % Stages>{});
        commit();
        compute(b);
    };
    for (int s0 = 0; s0 < slabs; s0 += Stages) {
        [&]<int... J>(std::integer_sequence<int, J...>) BATCHLAS_DLA_BODY {
            ((s0 + J < slabs && (step(s0 + J, std::integral_constant<int, J>{}), true)) && ...);
        }(std::make_integer_sequence<int, Stages>{});
    }
    drain();
}

}  // namespace batchlas::device::inline v2::l3
