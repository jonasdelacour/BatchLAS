#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <string>

#include <blas/dispatch/provider.hh>

namespace batchlas::blas::dispatch {

inline std::string uppercase_ascii(std::string s) {
    for (char& ch : s) {
        ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    }
    return s;
}

inline std::string lowercase_ascii(std::string s) {
    for (char& ch : s) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return s;
}

inline Provider parse_provider_value(const char* v) {
    if (!v || !*v) return Provider::Auto;
    std::string s = lowercase_ascii(std::string(v));

    if (s == "auto") return Provider::Auto;
    if (s == "vendor") return Provider::Vendor;

    if (s == "cta" || s == "batchlas_cta" || s == "batchlas-cta") return Provider::BatchLAS_CTA;
    if (s == "blocked" || s == "batchlas_blocked" || s == "batchlas-blocked") return Provider::BatchLAS_Blocked;

    if (s == "netlib") return Provider::Netlib;

    // Unknown value: keep behavior conservative (Auto).
    return Provider::Auto;
}

// Reads BATCHLAS_<OP>_PROVIDER, where <OP> is uppercased.
inline Provider parse_provider_env(const char* opname) {
    if (!opname || !*opname) return Provider::Auto;
    const std::string key = "BATCHLAS_" + uppercase_ascii(std::string(opname)) + "_PROVIDER";
    return parse_provider_value(std::getenv(key.c_str()));
}

inline constexpr std::array<Provider, 4> default_order_cta_blocked_vendor_netlib = {
    Provider::BatchLAS_CTA,
    Provider::BatchLAS_Blocked,
    Provider::Vendor,
    Provider::Netlib,
};

inline constexpr std::array<Provider, 4> default_order_for_op(const char* /*opname*/) {
    // Phase 0: keep a single default order. Extend per-op later.
    return default_order_cta_blocked_vendor_netlib;
}

inline DispatchPolicy policy_from_env(const char* op) {
    DispatchPolicy p;
    p.forced = parse_provider_env(op);
    p.order = default_order_for_op(op);
    // Phase 0: no additional env knobs.
    p.log = false;
    p.require_in_order = false;
    return p;
}

} // namespace batchlas::blas::dispatch
