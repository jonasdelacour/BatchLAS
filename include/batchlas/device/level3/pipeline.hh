// SPDX-License-Identifier: MIT
#pragma once
/// @file
/// @brief Level 3 k loops of the tuned gemm (@ref api_device_lowlevel, not covered by the
/// compatibility promise).
///
/// The barriers are each loop's contract; any executor with `barrier()` runs them.
///   - `l3::kloop`: one local stage, or two double-buffered.
///   - `l3::kloop_prefetch2`: a register-prefetched double buffer, one barrier per slab.
///   - `l3::kloop_async`: `Stages - 1` slabs in flight through asynchronous copies.
///   - `l3::kloop_async_ring`: the same with a run-time ring index.
/// @ingroup api_device_lowlevel
// Moved from the tuned gemm kernels unchanged.

#include <type_traits>
#include <utility>

#include <batchlas/device/core.hh>
#include <batchlas/device/detail/kernel_attrs.hh>
#include <batchlas/device/level3/copy_async.hh>

namespace batchlas::device::l3 {

/// @brief The staged k loop over `tile_count` slabs.
///
/// `Stages = 1`: load, barrier, compute, barrier per slab. `Stages = 2`: slab `t + 1` loads while
/// slab `t` computes, one barrier per slab. Ends on a barrier after the last compute. COLLECTIVE.
/// @ingroup api_device_lowlevel
/// @tparam Stages 1 or 2.
/// @tparam Exact `true`: no tail, and the first load sits behind `tile_count > 0`.
/// @tparam TileK the slab depth.
/// @param ex the executor.
/// @param tile_count the number of slabs.
/// @param load called as `load(int kk0, int stage)`: fill local stage `stage` with the slab at `kk0`.
/// @param compute called as `compute(int stage, int kk0)`: consume it.
template <int Stages, bool Exact, int TileK, class Ex, class Load, class Compute>
BATCHLAS_DEVICE_INLINE void kloop(const Ex& ex, int tile_count, Load&& load, Compute&& compute) {
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

/// @brief A register-prefetched double-buffered k loop, one barrier per slab.
///
/// Slab `s + 1` goes into the buffer slab `s - 1` was read from, and that read ended before the
/// barrier closing iteration `s - 1`. Unrolled by two, so `buf` is a constant. Ends without a
/// barrier. COLLECTIVE.
/// @ingroup api_device_lowlevel
/// @tparam TileK the slab depth.
/// @param ex the executor.
/// @param slabs the number of slabs.
/// @param gload called as `gload(int k0)`: read the slab at `k0` into registers.
/// @param sstore called as `sstore(int buf)`: write them to local buffer `buf`.
/// @param compute called as `compute(int buf)`: consume a buffer.
template <int TileK, class Ex, class GLoad, class SStore, class Compute>
BATCHLAS_DEVICE_INLINE void kloop_prefetch2(const Ex& ex, int slabs, GLoad&& gload, SStore&& sstore, Compute&& compute) {
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

/// @brief `Stages - 1` slabs in flight through asynchronous copies, the buffer index a compile-time constant.
///
/// Step `s` waits for slab `s`, takes a barrier, refills the buffer slab `s - 1` was read from and
/// computes; one commit per step (an empty group past the end). Ends drained. COLLECTIVE.
/// @ingroup api_device_lowlevel
/// @tparam Stages the ring size.
/// @tparam TileK the slab depth.
/// @param ex the executor.
/// @param slabs the number of slabs.
/// @param issue called as `issue(int k0, buf)` with `buf` a `std::integral_constant<int, b>`: start
///   the slab's copies into buffer `b`.
/// @param commit called as `commit()`: close a copy group.
/// @param wait called as `wait()`: wait until at most `Stages - 2` groups are pending.
/// @param drain called as `drain()`: wait until none is.
/// @param compute called as `compute(int b)`: consume buffer `b`.
template <int Stages, int TileK, class Ex, class Issue, class Commit, class Wait, class Drain, class Compute>
BATCHLAS_DEVICE_INLINE void kloop_async(const Ex& ex, int slabs, Issue&& issue, Commit&& commit, Wait&& wait, Drain&& drain,
                                     Compute&& compute) {
    [&]<int... J>(std::integer_sequence<int, J...>) BATCHLAS_DEVICE_BODY {
        ((J < slabs ? issue(J * TileK, std::integral_constant<int, J>{}) : void(), commit()), ...);
    }(std::make_integer_sequence<int, Stages - 1>{});
    auto step = [&](int s, auto buf) BATCHLAS_DEVICE_BODY {
        constexpr int b = decltype(buf)::value;
        wait();
        ex.barrier();
        if (s + Stages - 1 < slabs) issue((s + Stages - 1) * TileK, std::integral_constant<int, (b + Stages - 1) % Stages>{});
        commit();
        compute(b);
    };
    for (int s0 = 0; s0 < slabs; s0 += Stages) {
        [&]<int... J>(std::integer_sequence<int, J...>) BATCHLAS_DEVICE_BODY {
            ((s0 + J < slabs && (step(s0 + J, std::integral_constant<int, J>{}), true)) && ...);
        }(std::make_integer_sequence<int, Stages>{});
    }
    drain();
}

/// @brief `Stages - 1` slabs in flight through `copy_async_zfill` into a ring whose buffer index is a
/// run-time value (an unrolled ring is `Stages` times the code).
///
/// Slab `s` is at `k_lo + s * TileK`, issued in slab order. Step `s` waits for slab `s` (its own
/// copies, then the barrier for everyone's), refills the buffer slab `s - 1` was read from and
/// runs `compute(buf)`; one commit per step, empty past the end, keeps the wait count at
/// `Stages - 2`. Ends drained, without a barrier. COLLECTIVE.
/// @ingroup api_device_lowlevel
/// @tparam Stages the ring size.
/// @tparam TileK the slab depth.
/// @param ex the executor.
/// @param slabs the number of slabs.
/// @param k_lo the first slab's k offset.
/// @param issue called as `issue(int k0, int buf)`: start the slab's copies into buffer `buf`.
/// @param compute called as `compute(int buf)`: consume buffer `buf`.
template <int Stages, int TileK, class Ex, class Issue, class Compute>
BATCHLAS_DEVICE_INLINE void kloop_async_ring(const Ex& ex, int slabs, int k_lo, Issue&& issue, Compute&& compute) {
BATCHLAS_UNROLL_FULL
    for (int j = 0; j < Stages - 1; ++j) {
        if (j < slabs) issue(k_lo + j * TileK, j);
        copy_async_commit();
    }
    int buf = 0;
#pragma unroll 1
    for (int s = 0; s < slabs; ++s) {
        copy_async_wait<Stages - 2>();
        ex.barrier();
        const int nb = buf == 0 ? Stages - 1 : buf - 1;
        if (s + Stages - 1 < slabs) issue(k_lo + (s + Stages - 1) * TileK, nb);
        copy_async_commit();
        compute(buf);
        buf = buf + 1 == Stages ? 0 : buf + 1;
    }
    copy_async_wait<0>();
}

}  // namespace batchlas::device::l3
