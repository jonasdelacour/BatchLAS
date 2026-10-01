#pragma once

// The device half of an OpShape, filled in ONE place; every *_op_shape builder calls
// fill_device_facts(). evidence: docs/perf/dispatch.md#routing-profiles

#include <batchlas/export.hh>
#include <batchlas/arch/arch_key.hh>
#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/util/sycl-device-queue.hh>

#include <optional>

namespace batchlas::dispatch {

struct DeviceFacts {
    bool is_gpu = false;
    // sub_group_sizes()[0], NOT the largest: a reqd_sub_group_size gate must use
    // Device::supports_sub_group_size().
    int max_sub_group = 0;
    int compute_units = 0;
    arch::ArchKey key{};
};

// Memoized per device; a failing query leaves its default.
BATCHLAS_API DeviceFacts device_facts(const Device& d);

// BATCHLAS_ROUTING_PROFILE; nullopt when unset or empty, THROWS invalid_argument otherwise.
BATCHLAS_API std::optional<arch::RoutingProfile> routing_profile_override();

inline arch::ProfileChoice routing_profile(const Device& d) {
    return arch::select_profile(device_facts(d).key, routing_profile_override());
}

inline void fill_device_facts(OpShape& s, const Queue& q) {
    const DeviceFacts f = device_facts(q.device());
    s.is_gpu = f.is_gpu;
    s.max_sub_group = f.max_sub_group;
    s.compute_units = f.compute_units;
    s.cuda_cc = f.key.cuda_cc;
    const arch::ProfileChoice p = arch::select_profile(f.key, routing_profile_override());
    s.profile = p.profile;
    s.profile_nearest = p.nearest;
}

} // namespace batchlas::dispatch
