#pragma once

// Device helpers for the register-resident GEQRF ("tiny") tier. geqrf_cta_device.hh's
// LarfgScalars is reused VERBATIM -- tau is a contract ormqr/orgqr/ormbr/sy2sb consume, and a
// second LAPACK real-beta convention is a second thing to get wrong. evidence:
// docs/perf/qr.md#the-two-partition-butterflies, #a-residual-test-cannot-guard-a-convention

#include "geqrf_cta_device.hh"
#include "tiny_device.hh"

#include "../sycl/device_scalar.hh"
#include "../util/resident_capacity.hh"

#include <cstddef>
#include <cstdint>

namespace batchlas::geqrf_native {

// PARTITION-wide XOR butterflies, ALL-reduces on purpose: a WORK-GROUP reduce_over_group would
// re-open the 48 KB hole. evidence: docs/perf/qr.md#the-two-partition-butterflies
template <int N, typename R, typename Part>
inline R geqrf_tiny_reduce_fmax(const Part& part, R value) {
#pragma unroll
    for (uint32_t mask = 1u; mask < static_cast<uint32_t>(N); mask <<= 1) {
        value = sycl::fmax(value, permute_group_by_xor(part, value, mask));
    }
    return value;
}

template <int N, typename R, typename Part>
inline R geqrf_tiny_reduce_sum(const Part& part, R value) {
#pragma unroll
    for (uint32_t mask = 1u; mask < static_cast<uint32_t>(N); mask <<= 1) {
        value += permute_group_by_xor(part, value, mask);
    }
    return value;
}

// THE ONE source of truth for the register count. Keyed on the DEVICE scalar D, never on T:
// Cx<float> and double are 8 bytes each with very different bodies.
// evidence: docs/perf/qr.md#the-tiny-tier-register-table
inline constexpr int kGeqrfTinyRegMargin = 8;

constexpr int geqrf_tiny_reg_slot(int n_pad) {  // -1 = off the ladder; never read
    return (n_pad == 8) ? 0 : (n_pad == 16) ? 1 : (n_pad == 32) ? 2 : -1;
}

// Primary template = NO PROBE ROW. The zero sentinels, and why `pin` is not a free knob:
// evidence: docs/perf/qr.md#the-two-partition-butterflies
template <typename D> struct GeqrfTinyRegs {
    static constexpr int at[3] = {0, 0, 0};
    static constexpr int probed_chunk[3] = {0, 0, 0};
    static constexpr int pin[3] = {0, 0, 0};
};
template <> struct GeqrfTinyRegs<float> {
    static constexpr int at[3] = {64, 91, 142};            // N = 8, 16, 32
    static constexpr int probed_chunk[3] = {8, 16, 32};
    static constexpr int pin[3] = {0, 0, 0};
};
template <> struct GeqrfTinyRegs<double> {
    // N = 32 is PINNED: the wider chunk costs a resident block.
    // evidence: docs/perf/qr.md#the-chunk-width-and-the-one-cell-that-is-pinned
    static constexpr int at[3] = {84, 124, 193};
    static constexpr int probed_chunk[3] = {8, 16, 16};
    static constexpr int pin[3] = {0, 0, 16};
};
template <> struct GeqrfTinyRegs<sycl_device::Cx<float>> {
    static constexpr int at[3] = {72, 128, 168};
    static constexpr int probed_chunk[3] = {8, 16, 16};
    static constexpr int pin[3] = {0, 0, 0};
};
template <> struct GeqrfTinyRegs<sycl_device::Cx<double>> {
    // N = 32 is not instantiated; the slot carries the ceiling it would have to respect, so
    // the gate cannot widen without a re-probe. evidence: docs/perf/qr.md#the-cdouble-n32-cell
    static constexpr int at[3] = {110, 162, 226};
    static constexpr int probed_chunk[3] = {8, 8, 0};
    static constexpr int pin[3] = {0, 0, 0};
};

template <typename D>
constexpr int geqrf_tiny_words() {  // 32-bit words per scalar: float 1, cdouble 4
    return static_cast<int>(sizeof(D) / sizeof(float));
}

// The register MODEL: fallback for a cell with NO probe row only -- it is wrong in both
// directions here. evidence: docs/perf/qr.md#the-tiny-tier-register-table
inline constexpr int kGeqrfTinyRegOverhead = 88;
constexpr int geqrf_tiny_regs(int n_pad, int words) {
    return n_pad * words + kGeqrfTinyRegOverhead;
}

template <typename D, int N>
constexpr int geqrf_tiny_regs_for() {
    constexpr int slot = geqrf_tiny_reg_slot(N);
    static_assert(slot >= 0, "N must be on the {8, 16, 32} ladder");
    constexpr int probed = GeqrfTinyRegs<D>::at[slot];
    return (probed > 0) ? probed : geqrf_tiny_regs(N, geqrf_tiny_words<D>());
}

// The ceil8 below is load-bearing: ptxas allocates registers in banks of eight, so a rule
// reading the raw count admits a block the hardware refuses.
inline constexpr int kGeqrfTinyRegsPerSm = 65536;
inline constexpr int kGeqrfTinyThreadsPerSm = 1536;
inline constexpr int kGeqrfTinyMaxBlocksPerSm = 24;

constexpr int geqrf_tiny_ceil8(int regs) { return (regs + 7) & ~7; }

constexpr int geqrf_tiny_blocks_by_regs(int regs, int wg) {
    if (regs <= 0 || wg <= 0) return 0;
    int b = kGeqrfTinyRegsPerSm / (geqrf_tiny_ceil8(regs) * wg);
    const int by_threads = kGeqrfTinyThreadsPerSm / wg;
    if (b > by_threads) b = by_threads;
    if (b > kGeqrfTinyMaxBlocksPerSm) b = kGeqrfTinyMaxBlocksPerSm;
    return b;
}

// The N x (C+1) tile plus a SEPARATE C-element y -- separate, not a row of the tile, is why
// two barriers per chunk suffice. evidence: docs/perf/qr.md#the-two-partition-butterflies
constexpr std::size_t geqrf_tiny_slm_elems(int n_pad, int c) {
    return static_cast<std::size_t>(n_pad) * static_cast<std::size_t>(c + 1) +
           static_cast<std::size_t>(c);
}

inline constexpr std::size_t kGeqrfTinyReferenceSlm =
    97280;  // COMPILE-TIME derivation only; runtime reads resident::device_slm_budget

constexpr int geqrf_tiny_blocks_by_slm(int n_pad, int c, std::size_t elem_bytes,
                                       int matrices_per_wg) {
    const std::size_t per_wg = static_cast<std::size_t>(matrices_per_wg) *
                               geqrf_tiny_slm_elems(n_pad, c) * elem_bytes;
    return per_wg > 0 ? static_cast<int>(kGeqrfTinyReferenceSlm / per_wg) : 0;
}

// The shared constant, not a forked geometry. No tiny kernel declares reqd_work_group_size:
// .maxntid lets ptxas trade registers against a launch bound, which would make the probed
// table a function of the launch shape. evidence: docs/perf/qr.md#the-launch-shape-64-work-items-not-128
inline constexpr int kGeqrfTinyWg = tiny_native::kTinyWgSize;

// Through the shared helper, not `kGeqrfTinyWg / N`, so the power-of-two guarantee and the
// max-work-group clamp stay where they are owned; the byte arguments are nominal.
template <int N>
constexpr int geqrf_tiny_matrices_per_wg() {
    return resident::pack_matrices_per_wg(/*bytes_per_matrix=*/1u, N,
                                          /*wg_slm_budget_bytes=*/~std::size_t(0),
                                          /*max_wg_size=*/kGeqrfTinyWg, kGeqrfTinyWg,
                                          /*max_pack=*/kGeqrfTinyWg / N);
}

static_assert(geqrf_tiny_matrices_per_wg<8>() * 8 == kGeqrfTinyWg, "N=8: 2 sub-groups x 4");
static_assert(geqrf_tiny_matrices_per_wg<16>() * 16 == kGeqrfTinyWg, "N=16: 2 sub-groups x 2");
static_assert(geqrf_tiny_matrices_per_wg<32>() * 32 == kGeqrfTinyWg, "N=32: 2 sub-groups x 1");
static_assert(kGeqrfTinyWg / tiny_native::kTinySubGroupSize >= 2,
              "R3 wants at least two sub-groups per work-group");

// C is DERIVED, not fixed: the largest of {N, N/2, N/4} at which local-memory residency is at
// least register residency. evidence: docs/perf/qr.md#the-chunk-width-and-the-one-cell-that-is-pinned
template <typename D, int N>
constexpr int geqrf_tiny_chunk() {
    constexpr int pinned = GeqrfTinyRegs<D>::pin[geqrf_tiny_reg_slot(N)];
    if (pinned > 0) return pinned;
    const int mats = geqrf_tiny_matrices_per_wg<N>();
    const int wg = mats * N;
    const int want = geqrf_tiny_blocks_by_regs(geqrf_tiny_regs_for<D, N>(), wg);
    if (geqrf_tiny_blocks_by_slm(N, N, sizeof(D), mats) >= want) return N;
    if (N >= 2 && geqrf_tiny_blocks_by_slm(N, N / 2, sizeof(D), mats) >= want) return N / 2;
    return (N >= 4) ? N / 4 : N;
}

// THE ONE-STEP FIXED POINT of a circular derivation (C moves the register count, the count picks
// C): a re-probe crossing the boundary without updating `probed_chunk` must fail to COMPILE.
template <typename D, int N>
constexpr bool geqrf_tiny_chunk_is_fixed_point() {
    constexpr int slot = geqrf_tiny_reg_slot(N);
    constexpr int recorded = GeqrfTinyRegs<D>::probed_chunk[slot];
    return recorded == 0 || recorded == geqrf_tiny_chunk<D, N>();
}

// THE GATE THAT BINDS -- the raw `regs x wg <= 65536` launch-abort gate has too much slack
// here to guard anything. evidence: docs/perf/qr.md#the-tiny-tier-register-table
template <typename D, int N>
constexpr bool geqrf_tiny_cell_is_resident() {
    return geqrf_tiny_blocks_by_regs(geqrf_tiny_regs_for<D, N>(),
                                     geqrf_tiny_matrices_per_wg<N>() * N) >=
           resident::kMinBlocksPerSm;
}

// The +1 is load-bearing: an ODD row stride spreads a column's N rows over distinct banks at
// every scalar width. evidence: docs/perf/qr.md#the-chunk-width-and-the-one-cell-that-is-pinned
constexpr int geqrf_tiny_slda(int c) { return c + 1; }

}  // namespace batchlas::geqrf_native
