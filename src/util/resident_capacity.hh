#pragma once

// The occupancy rule shared by the local-memory-resident tiers (potrf, getrf, geqrf), in one
// place. Constexpr and SYCL-header-free, so src/backends/'s shape builders may include it.
// evidence: docs/perf/potrf.md#the-shared-helper-budget-slice-walk-and-pack

#include <cstddef>

namespace batchlas::resident {

// THE spelling of the budget; a site that inlines its own disagrees by the reserve.
constexpr std::size_t device_slm_budget(std::size_t local_mem_bytes,
                                        std::size_t reserve_bytes = 4096) {
    return (local_mem_bytes > reserve_bytes) ? (local_mem_bytes - reserve_bytes) : 0;
}

inline constexpr int kMinBlocksPerSm = 4;  // design rule R1's occupancy target

// The slice ONE work-group may own, hence the ADVERTISED capacity; min_blocks_per_sm = 1 asks
// the other question, "can it be held at all". evidence: docs/perf/lu.md#the-panel-leaf-is-not-the-tier-ceiling
constexpr std::size_t occupancy_budget(std::size_t slm_budget_bytes,
                                       int min_blocks_per_sm = kMinBlocksPerSm) {
    const std::size_t blocks =
        (min_blocks_per_sm > 0) ? static_cast<std::size_t>(min_blocks_per_sm) : 1u;
    return slm_budget_bytes / blocks;
}

// Largest n that fits the slice AND for which every smaller n fits too. THE SECOND CLAUSE IS
// LOAD-BEARING: hole padding makes bytes(n) non-monotone while supports() spells capacity as a
// contiguous `order <= cta_max_n`, so `continue` would advertise a range with a hole in it.
// bytes_per_matrix must use 64-bit products: (m|1)*n overflows int at a reachable height.
template <typename BytesFn>
constexpr int resident_max_n(BytesFn bytes_per_matrix,
                             std::size_t slm_budget_bytes,
                             int min_blocks_per_sm = kMinBlocksPerSm,
                             int n_hi = 4096) {
    const std::size_t budget = occupancy_budget(slm_budget_bytes, min_blocks_per_sm);
    int best = 0;
    for (int n = 1; n <= n_hi; ++n) {
        if (bytes_per_matrix(n) > budget) break;
        best = n;
    }
    return best;
}

// Matrices per work-group when one sub-group serves a matrix. NOT A FREE KNOB: G > 1 is correct
// ONLY where every barrier the kernel executes is a SUB-GROUP barrier -- a work-group barrier
// synchronises the G matrices with each other, a race by construction, not a launch failure, so
// callers derive scope and G together. Power of two: matrix = wg_id*G + sg_id stays a shift.
constexpr int pack_matrices_per_wg(std::size_t bytes_per_matrix,
                                   int lanes_per_matrix,
                                   std::size_t wg_slm_budget_bytes,
                                   int max_wg_size,
                                   int target_wg_size = 128,
                                   int max_pack = 4) {
    if (lanes_per_matrix < 1 || bytes_per_matrix == 0 || max_pack < 1) return 1;
    const int wg_cap = (max_wg_size < target_wg_size) ? max_wg_size : target_wg_size;
    if (wg_cap < lanes_per_matrix) return 1;

    int g = 1;
    while (g * 2 <= max_pack &&
           (g * 2) * lanes_per_matrix <= wg_cap &&
           static_cast<std::size_t>(g * 2) * bytes_per_matrix <= wg_slm_budget_bytes) {
        g *= 2;
    }
    return g;  // 1 is the honest answer when not even one matrix fits
}

// THE LAUNCH gate, as against the occupancy gates above: on sm_89 the 65,536 registers of an SM
// are owned by FOUR sub-partitions of 16,384 and a block's warps are dealt over them round-robin,
// so what must fit is ceil(warps / 4) x 32 x allocated_regs <= 16,384. The per-block spelling
// `regs x wg <= 65536` agrees only when the warp count is a multiple of four and is strictly
// looser otherwise -- it accepts launches the driver refuses. Every gate that was on the
// per-block spelling now routes through here; getrf_panel_reg.cc still carries its own copy
// of these constants and its own panel_reg_wg_ceiling, which agrees with sm89_max_work_group
// but has not been collapsed into it.
// evidence: docs/perf/lu.md#the-register-cap-that-binds-is-per-sub-partition
inline constexpr int kRegsPerBlock = 65536;
inline constexpr int kRegsPerPartition = 16384;
inline constexpr int kPartitionsPerBlock = 4;
inline constexpr int kLanesPerWarp = 32;
inline constexpr int kRegAllocGranularity = 8;  // ptxas allocates registers in banks of eight

// The count every gate below divides into is the ALLOCATED one, not the probed one.
constexpr int sm89_alloc_regs(int regs_per_thread) {
    return (regs_per_thread + kRegAllocGranularity - 1) / kRegAllocGranularity *
           kRegAllocGranularity;
}

// The widest work-group this register demand may be LAUNCHED at: the most warps one
// sub-partition may hold, times the four partitions a block is dealt over. 0 for a nonsense
// demand, exactly as resident_max_n's 0 spells "nothing fits".
constexpr int sm89_max_work_group(int regs_per_thread) {
    if (regs_per_thread < 1) return 0;
    const int warps_per_partition =
        kRegsPerPartition / (kLanesPerWarp * sm89_alloc_regs(regs_per_thread));
    return warps_per_partition * kPartitionsPerBlock * kLanesPerWarp;
}

// The same rule as a predicate, so a static_assert reads as prose rather than as arithmetic.
constexpr bool sm89_fits(int regs_per_thread, int work_group_size) {
    if (regs_per_thread < 1 || work_group_size < 1) return false;
    const int warps = (work_group_size + kLanesPerWarp - 1) / kLanesPerWarp;
    const int per_partition = (warps + kPartitionsPerBlock - 1) / kPartitionsPerBlock;
    return per_partition * kLanesPerWarp * sm89_alloc_regs(regs_per_thread) <= kRegsPerPartition;
}

}  // namespace batchlas::resident
