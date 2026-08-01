#pragma once

#include <map>
#include <mutex>
#include <string>
#include <utility>

#include <util/sycl-device-queue.hh>

#include <blas/dispatch/provider.hh>

namespace batchlas::blas::dispatch {

struct DeviceCaps {
    bool is_gpu = false;
    int max_sub_group = 0;
    std::string name;
};

struct DispatchContext {
    Queue& q;
    DeviceCaps caps;
    DispatchPolicy policy;
};

// Best-effort querying: never throws.
inline DeviceCaps query_caps_uncached(Queue& q) {
    DeviceCaps out;

    try {
        out.is_gpu = (q.device().type == DeviceType::GPU);
    } catch (...) {
        // leave default
    }

    // Best-effort querying via the public Device wrapper.
    try {
        out.name = q.device().get_name();
    } catch (...) {
        // leave empty
    }

    try {
        out.max_sub_group = static_cast<int>(q.device().get_property(DeviceProperty::MAX_SUB_GROUP_SIZE));
    } catch (...) {
        // leave default
    }

    return out;
}

// Device capabilities are immutable for the life of the process, but every
// dispatch call used to re-query them: get_name() is an uncached SYCL
// get_info<device::name>() returning a fresh std::string, and
// MAX_SUB_GROUP_SIZE allocates a std::vector per call. Operations that dispatch
// in a tight loop (band reduction issues ~6 ormqr per chase step, each hitting
// this twice) paid that cost thousands of times per call.
//
// Cached per (device type, device index); the set of devices is fixed at
// startup, so the cache never needs invalidating.
inline const DeviceCaps& query_caps(Queue& q) {
    static std::mutex mtx;
    static std::map<std::pair<int, size_t>, DeviceCaps> cache;

    std::pair<int, size_t> key{static_cast<int>(DeviceType::HOST), 0};
    try {
        const auto d = q.device();
        key = {static_cast<int>(d.type), d.idx};
    } catch (...) {
        // fall through with the default key
    }

    std::lock_guard<std::mutex> lock(mtx);
    auto it = cache.find(key);
    if (it == cache.end()) {
        it = cache.emplace(key, query_caps_uncached(q)).first;
    }
    return it->second;
}

} // namespace batchlas::blas::dispatch
