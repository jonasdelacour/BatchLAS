#pragma once

// GETRF's routing table. evidence: docs/perf/lu.md

#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/dispatch/route_resolve.hh>

#include <cstdint>
#include <type_traits>

namespace batchlas::dispatch {

struct GetrfShape : OpShape {
    // Device-queried; 0 = absent. MUST include the pivot-search SLM scratch or the wide types
    // ask past the cap and the launch is rejected. evidence: docs/perf/lu.md#one-spelling-per-ceiling
    int cta_max_n = 0;

    bool blocked_available = false;

    int tiny_max_n = 0;   // compile-time {8,16,32} ladder (no local memory); 0 = absent

    // MUST come from sycl::info::device::sub_group_sizes: OpShape::max_sub_group reports
    // entry [0], not the max, so it admits a device that rejects the sg32 launch.
    bool has_sg32 = false;

    int64_t order() const { return k; }
};

// Walk order is this array, never Algorithm's numeric value; Tiny first, the narrower tier.
inline constexpr Route kGetrfOrder[] = {
    {Origin::Native, Algorithm::Tiny},
    {Origin::Native, Algorithm::CTA},
    {Origin::Native, Algorithm::Blocked},
    {Origin::Vendor, Algorithm::Auto},
};

template <typename T>
struct RouteTable<Op::getrf, T> {
    // Correctness only: a speed threshold here removes getrf's vendor-free route.
    static bool supports(Route r, const GetrfShape& s) {
        if (is_vendor(r)) return true;   // the vendor serves everything it is given
        if (!is_native(r)) return false;

        if (s.m != s.n) return false;
        if (!s.is_gpu) return false;
        if (!s.has_sg32) return false;
        if (s.heterogeneous_batch) return false;
        if (s.order() < 1 || s.batch < 1) return false;

        // Pivot Span<int64_t> layout is backend-dependent: CUDA/ROCm and the native kernels
        // pack 1-based int32 in its first half, netlib real int64; mixing them is silent garbage.
        if (s.backend == Backend::NETLIB) return false;

        switch (r.algo) {
            case Algorithm::Tiny:
                if (s.tiny_max_n < 1) return false;
                return s.order() <= static_cast<int64_t>(s.tiny_max_n);

            case Algorithm::CTA:
                if (s.cta_max_n < 1) return false;
                return s.order() <= static_cast<int64_t>(s.cta_max_n);

            case Algorithm::Blocked:
                // No order floor, or a forced `blocked` falls through to automatic().
                return s.blocked_available && s.cta_max_n >= 1;

            default:
                return false;
        }
    }

    // Blocked only; Tiny stays absent until its grid exists (R8b: the first native arm true
    // here takes every shape it holds, whatever the tier hook says).
    // evidence: docs/perf/lu.md#getrf-window-evidence
    static bool preferred(Route r, const GetrfShape& s) {
        if (!is_native(r)) return false;
        if (r.algo != Algorithm::Blocked) return false;

        if constexpr (std::is_same_v<T, float>)               return s.order() >= 256;
        if constexpr (std::is_same_v<T, std::complex<float>>) return s.order() >= 512;
        return false;   // double and cdouble earn nothing at any order
    }

    // Native-vs-native tie-break, vendor-free walk only. evidence: docs/perf/lu.md#native_tier_preferred
    static bool native_tier_preferred(Route r, const GetrfShape& s) {
        if (!is_native(r)) return true;

        const int64_t cta_max_order = [] () -> int64_t {
            if constexpr (std::is_same_v<T, double>) {
                return 32;
            } else {
                return 1 << 30;
            }
        }();

        switch (r.algo) {
            // EXPLICIT: `default:` returns TRUE and Tiny leads the order array. Flips with
            // preferred(). evidence: docs/perf/lu.md#why-tiny-is-absent-from-the-window
            case Algorithm::Tiny:
                return false;
            case Algorithm::CTA:
                return s.order() <= cta_max_order;
            case Algorithm::Blocked:
                return s.order() > cta_max_order;
            default:
                return true;
        }
    }

    static constexpr const Route* order_begin() { return kGetrfOrder; }
    static constexpr const Route* order_end() {
        return kGetrfOrder + (sizeof(kGetrfOrder) / sizeof(kGetrfOrder[0]));
    }
};

// vendor_available is factorization_vendor_available<B>, NOT the solver one.
template <typename T>
inline Route resolve_getrf_route(Route forced, const GetrfShape& s,
                                 bool vendor_available = true) {
    return resolve_route<Op::getrf, T>(forced, s, vendor_available);
}

} // namespace batchlas::dispatch
