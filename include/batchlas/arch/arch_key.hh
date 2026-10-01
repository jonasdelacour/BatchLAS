#pragma once

// Which measured architecture a device's routing comes from. An unmeasured device
// borrows the NEAREST profile and is flagged `nearest`, which coverage records.
// SYCL-free and constexpr, so the map is testable without a GPU.
// evidence: docs/perf/dispatch.md#routing-profiles

#include <cstdint>
#include <optional>
#include <string_view>

namespace batchlas::arch {

enum class ArchVendor : uint8_t { Other, NVIDIA, AMD, Intel, CPU };

struct ArchKey {
    ArchVendor vendor = ArchVendor::Other;
    int cuda_cc = 0;   // major*10+minor (89, 120); 0 when not CUDA

    friend constexpr bool operator==(const ArchKey&, const ArchKey&) = default;
};

// `Unset` = a builder skipped fill_device_facts(). APPENDED, never inserted.
enum class RoutingProfile : uint8_t { Unset, sm_89, sm_120 };

struct ProfileChoice {
    RoutingProfile profile = RoutingProfile::Unset;
    bool nearest = false;   // not measured on this exact architecture

    friend constexpr bool operator==(const ProfileChoice&, const ProfileChoice&) = default;
};

inline constexpr std::string_view to_string(RoutingProfile p) {
    switch (p) {
        case RoutingProfile::Unset:  return "unset";
        case RoutingProfile::sm_89:  return "sm_89";
        case RoutingProfile::sm_120: return "sm_120";
    }
    return "?";
}

// Exact spellings only; nullopt must be rejected by the caller, never mean "auto".
inline constexpr std::optional<RoutingProfile> parse_routing_profile(std::string_view s) {
    if (s == "sm_89") return RoutingProfile::sm_89;
    if (s == "sm_120") return RoutingProfile::sm_120;
    return std::nullopt;
}

inline constexpr ProfileChoice nearest_profile(ArchKey k) {
    const int cc = k.vendor == ArchVendor::NVIDIA ? k.cuda_cc : 0;
    if (cc == 89) return {RoutingProfile::sm_89, false};
    if (cc == 120) return {RoutingProfile::sm_120, false};
    if (cc == 80 || cc == 86 || cc == 87 || cc == 90) return {RoutingProfile::sm_89, true};
    if (cc >= 100 && cc <= 121) return {RoutingProfile::sm_120, true};
    return {RoutingProfile::sm_89, true};
}

// A forced profile is still `nearest` unless it IS this device's exact one.
inline constexpr ProfileChoice select_profile(ArchKey k,
                                              std::optional<RoutingProfile> forced) {
    const ProfileChoice natural = nearest_profile(k);
    if (!forced || *forced == RoutingProfile::Unset) return natural;
    const bool exact = !natural.nearest && natural.profile == *forced;
    return {*forced, !exact};
}

} // namespace batchlas::arch
