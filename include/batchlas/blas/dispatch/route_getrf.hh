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

    // DISJOINT windows, one per tier: exactly one may answer true at any order, or
    // the order array becomes the decision (R8b). cfloat 256..511 is BATCH-gated.
    // evidence: docs/perf/lu.md#getrf-window-evidence
    static bool preferred(Route r, const GetrfShape& s) {
        if (!is_native(r)) return false;
        if (r.algo == Algorithm::Tiny) return tiny_window(s);
        if (r.algo == Algorithm::CTA) return cta_window(s);
        if (r.algo != Algorithm::Blocked) return false;
        if (tiny_window(s)) return false;  // defence in depth; no test observes it

        if constexpr (std::is_same_v<T, float>) return s.order() >= 256;
        if constexpr (std::is_same_v<T, std::complex<float>>) {
            return s.order() >= 512 || (s.order() >= 256 && s.batch >= 256);
        }
        return false;   // double and cdouble earn nothing at any order
    }

    // Bounds are measured EDGES. n = 4 ties the vendor at the DRAM roof, cfloat 8 falls
    // under the gate, cfloat 17 pads into N = 32 and loses; float 17..22 is CTA's.
    // evidence: docs/perf/lu.md#the-n4-bucket-and-the-cta-band
    static bool tiny_window(const GetrfShape& s) {
        if (!tiny_fits(s)) return false;
        if constexpr (std::is_same_v<T, float>) {
            return (s.order() >= 5 && s.order() <= 16) || (s.order() >= 23 && s.order() <= 32);
        } else if constexpr (std::is_same_v<T, std::complex<float>>) {
            return (s.order() >= 5 && s.order() <= 7) || (s.order() >= 9 && s.order() <= 16);
        } else {
            return false;   // fp64 on this part runs at 1/64 rate; no grid, no window
        }
    }

    static bool cta_window(const GetrfShape& s) {
        if constexpr (std::is_same_v<T, float>) {
            return s.order() >= 17 && s.order() <= 22;
        } else {
            return false;
        }
    }

    // 0 spells "tier absent".
    static bool tiny_fits(const GetrfShape& s) {
        return s.tiny_max_n >= 1 && s.order() <= static_cast<int64_t>(s.tiny_max_n);
    }

    // Native against native: at n <= 8 tiny is ~3x the CTA tier for both single types,
    // so the vendor-free walk takes it even where the vendor ties.
    static bool tiny_native(const GetrfShape& s) {
        if (tiny_window(s)) return true;
        constexpr bool kSingle =
            std::is_same_v<T, float> || std::is_same_v<T, std::complex<float>>;
        return kSingle && tiny_fits(s) && s.order() <= 8;
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
            // EXPLICIT: `default:` returns TRUE and Tiny leads the order array. Inside
            // tiny_native only, so the vendor-FREE walk lands where the windows point.
            case Algorithm::Tiny:
                return tiny_native(s);
            case Algorithm::CTA:
                return !tiny_native(s) && s.order() <= cta_max_order;
            case Algorithm::Blocked:
                return !tiny_native(s) && s.order() > cta_max_order;
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
