// SPDX-License-Identifier: MIT
#pragma once
// The local memory one work-group may launch with: the rule is device::local_mem_bytes
// (include/batchlas/device/tier.hh, which with_tier's DeviceLimits::of applies too).
#include <src/sycl/impl.hh>
#include <src/util/resident_capacity.hh>

#include <batchlas/device/tier.hh>
#include <batchlas/error.hh>

#include <cstddef>
#include <string>

namespace batchlas::impl {

inline std::size_t local_mem_scratch_per_item(const sycl::device& d) { return device::local_mem_scratch_per_item(d); }
inline std::size_t local_mem_bytes(const sycl::device& d, std::size_t wg) { return device::local_mem_bytes(d, wg); }
inline std::size_t local_mem_bytes(const sycl::device& d) { return device::local_mem_bytes(d); }

// resident::cta_fit_wg_multiplier, throwing before a launch that cannot fit one step.
inline int cta_wg_multiplier(const char* who, int requested, int probs_per_step,
                             std::size_t bytes_per_prob, std::size_t local_mem) {
    const int m = resident::cta_fit_wg_multiplier(requested, probs_per_step, bytes_per_prob, local_mem);
    if (m < 1) {
        throw batchlas::unsupported(std::string(who) + ": one work-group needs " +
                                    std::to_string(static_cast<std::size_t>(probs_per_step) * bytes_per_prob) +
                                    " B of local memory; the device launches " + std::to_string(local_mem));
    }
    return m;
}

}  // namespace batchlas::impl
