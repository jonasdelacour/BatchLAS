#pragma once

// GETRI routing: one native arm (a host-driven composition over the routed
// trsm) then the vendor; a native arm must not write A. Shape fields are
// potrf's: s.m == s.n == s.k == the order. docs/perf/lu.md

#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/dispatch/route_resolve.hh>

#include <cstdint>

namespace batchlas::dispatch {

struct GetriShape : OpShape {
    // Is the getri DRIVER compiled in this build -- not merely the routed trsm.
    bool blocked_available = false;

    bool has_sg32 = false;

    int64_t order() const { return k; }
};

inline constexpr Route kGetriOrder[] = {
    {Origin::Native, Algorithm::Blocked},
    {Origin::Vendor, Algorithm::Auto},
};

template <typename T>
struct RouteTable<Op::getri, T> {
    // Gates transcribe trsm's supports(): an omission is a wrong answer.
    static bool supports(Route r, const GetriShape& s) {
        if (is_vendor(r)) return true;
        if (!is_native(r)) return false;

        if (!s.blocked_available) return false;

        if (s.m != s.n) return false;

        if (!s.is_gpu) return false;

        if (!s.has_sg32) return false;

        // The pivot list is read at pivots[b*order + k] with a single order.
        if (s.heterogeneous_batch) return false;

        // No batch floor on purpose; the suite runs getri at batch 2.
        if (s.order() < 1 || s.batch < 1) return false;

        // Wrong-answer gate: GPU backends pack 1-based int32 pivots into the
        // int64 span, netlib writes genuine int64 (is_gpu reads the QUEUE).
        // evidence: docs/perf/lu.md#correctness-findings
        if (s.backend == Backend::NETLIB) return false;

        switch (r.algo) {
            case Algorithm::Blocked:
                return true;
            default:
                // Including Auto: resolve_route expects a SPECIFIC algorithm.
                return false;
        }
    }

    // float from order 128, cfloat from order 256, no batch term.
    // evidence: docs/perf/lu.md#getri-window-evidence
    static bool preferred(Route r, const GetriShape& s) {
        if (!is_native(r)) return false;
        if (r.algo != Algorithm::Blocked) return false;

        if constexpr (std::is_same_v<T, float>)               return s.order() >= 128;
        if constexpr (std::is_same_v<T, std::complex<float>>) return s.order() >= 256;
        return false;
    }

    static constexpr const Route* order_begin() { return kGetriOrder; }
    static constexpr const Route* order_end() {
        return kGetriOrder + (sizeof(kGetriOrder) / sizeof(kGetriOrder[0]));
    }
};

template <typename T>
inline Route resolve_getri_route(Route forced, const GetriShape& s,
                                 bool vendor_available = true) {
    return resolve_route<Op::getri, T>(forced, s, vendor_available);
}

} // namespace batchlas::dispatch
