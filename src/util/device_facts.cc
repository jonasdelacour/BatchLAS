#include <batchlas/blas/dispatch/device_facts.hh>

#include <batchlas/error.hh>
#include <batchlas/settings.hh>

#include <map>
#include <mutex>
#include <string>
#include <utility>

namespace batchlas::dispatch {

namespace {

arch::ArchVendor arch_vendor(const Device& d, int cuda_cc) {
    if (cuda_cc > 0) return arch::ArchVendor::NVIDIA;   // only the CUDA backend reports one
    if (d.type == DeviceType::CPU) return arch::ArchVendor::CPU;
    switch (d.get_vendor()) {
        case Vendor::NVIDIA: return arch::ArchVendor::NVIDIA;
        case Vendor::AMD:    return arch::ArchVendor::AMD;
        case Vendor::INTEL:  return arch::ArchVendor::Intel;
        default:             return arch::ArchVendor::Other;
    }
}

DeviceFacts query(const Device& d) {
    DeviceFacts f;
    f.is_gpu = d.type == DeviceType::GPU;
    try {
        f.max_sub_group = static_cast<int>(d.get_property(DeviceProperty::MAX_SUB_GROUP_SIZE));
    } catch (...) {
    }
    try {
        f.compute_units = static_cast<int>(d.get_property(DeviceProperty::MAX_COMPUTE_UNITS));
    } catch (...) {
    }
    try {
        f.key.cuda_cc = d.cuda_compute_capability();
    } catch (...) {
    }
    try {
        f.key.vendor = arch_vendor(d, f.key.cuda_cc);
    } catch (...) {
    }
    return f;
}

} // namespace

DeviceFacts device_facts(const Device& d) {
    // Leaked: read from routed calls that may run during static destruction.
    static auto* mu = new std::mutex();
    static auto* memo = new std::map<std::pair<int, size_t>, DeviceFacts>();
    const auto key = std::make_pair(static_cast<int>(d.type), d.idx);
    {
        std::lock_guard<std::mutex> lock(*mu);
        if (auto it = memo->find(key); it != memo->end()) return it->second;
    }
    const DeviceFacts f = query(d);   // outside the lock: get_info can be slow
    std::lock_guard<std::mutex> lock(*mu);
    return memo->emplace(key, f).first->second;
}

std::optional<arch::RoutingProfile> routing_profile_override() {
    const char* v = batchlas::settings().routing.routing_profile.get();
    if (!v || !*v) return std::nullopt;
    if (const auto p = arch::parse_routing_profile(v)) return p;
    throw batchlas::invalid_argument(
        std::string("BatchLAS: BATCHLAS_ROUTING_PROFILE=\"") + v +
        "\" is not a routing profile. Expected one of: sm_89, sm_120; unset it to use "
        "the device's nearest measured profile.");
}

} // namespace batchlas::dispatch
