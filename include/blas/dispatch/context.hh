#pragma once

#include <string>

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
inline DeviceCaps query_caps(Queue& q) {
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

} // namespace batchlas::blas::dispatch
