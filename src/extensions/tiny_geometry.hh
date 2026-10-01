#pragma once

// The tiny tier's host-side geometry constants, SYCL-free so a launch plan can read them.
// tiny_device.hh re-exports them to the kernels. evidence: docs/perf/potrf.md#the-shared-tiny-tier-invariants

namespace batchlas::tiny_native {

inline constexpr int kTinySubGroupSize = 32;  // every tiny kernel: reqd_sub_group_size(32)
inline constexpr int kTinySubGroups = 2;      // a tuning constant, not a contract
inline constexpr int kTinyWgSize = kTinySubGroups * kTinySubGroupSize;

constexpr bool tiny_n_is_legal(int N) {  // invariant 2: N must divide the sub-group
    return N == 8 || N == 16 || N == 32;
}

constexpr int tiny_bucket_ge(int n) {  // order -> bucket; 0 means "above the tier"
    if (n < 1) return 0;
    if (n <= 8) return 8;
    if (n <= 16) return 16;
    if (n <= 32) return 32;
    return 0;
}

}  // namespace batchlas::tiny_native
