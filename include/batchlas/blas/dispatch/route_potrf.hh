#pragma once

// POTRF's routing table: the correctness gates, the native CTA/Blocked capability
// ladder and the (empty) preferred() window. Evidence: docs/perf/potrf.md

#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/dispatch/route_resolve.hh>

namespace batchlas::dispatch {

struct PotrfShape : OpShape {
    // 0 means the CTA kernel is absent from this build; device-queried by the same function that sizes the kernel's local_accessor.
    int cta_max_n = 0;

    bool blocked_available = false;

    // From sycl::info::device::sub_group_sizes (max_sub_group reports entry [0]); gates Blocked too, whose diagonal leaf is the CTA kernel.
    bool has_sg32 = false;

    int64_t order() const { return k; }
};

inline constexpr Route kPotrfOrder[] = {
    {Origin::Native, Algorithm::CTA},
    {Origin::Native, Algorithm::Blocked},
    {Origin::Vendor, Algorithm::Auto},
};

template <typename T>
struct RouteTable<Op::potrf, T> {
    // Correctness only: a speed threshold here removes potrf's vendor-free route.
    static bool supports(Route r, const PotrfShape& s) {
        if (is_vendor(r)) return true;
        if (!is_native(r)) return false;

        if (s.m != s.n) return false;

        if (!s.is_gpu) return false;

        if (!s.has_sg32) return false;

        if (s.heterogeneous_batch) return false;

        if (s.order() < 1 || s.batch < 1) return false;

        switch (r.algo) {
            case Algorithm::CTA:
                // No uplo gate, unlike Blocked below: Upper is the same recurrence on the transformed tile S(i,c) = conj(A(c,i)).
                if (s.cta_max_n < 1) return false;
                return s.order() <= s.cta_max_n;

            case Algorithm::Blocked:
                // No lower order bound: a floor makes a forced `blocked` fall through to automatic().
                // The uplo gate IS correctness: the driver implements Lower only and would overwrite the wrong triangle.
                if (s.uplo != Uplo::Lower) return false;
                return s.blocked_available && s.cta_max_n >= 1;

            default:
                return false;
        }
    }

    // The window is empty: Auto takes the vendor at every shape, and a vendor-free build still resolves to a supported native arm.
    // evidence: docs/perf/potrf.md#preferred-is-false-everywhere
    static bool preferred(Route r, const PotrfShape& s) {
        static_cast<void>(r);
        static_cast<void>(s);
        return false;
    }

    static constexpr const Route* order_begin() { return kPotrfOrder; }
    static constexpr const Route* order_end() {
        return kPotrfOrder + (sizeof(kPotrfOrder) / sizeof(kPotrfOrder[0]));
    }
};

// Pass vendor_available = dispatch::solver_vendor_available<B>, not factorization_vendor_available -- they differ on CUDA.
template <typename T>
inline Route resolve_potrf_route(Route forced, const PotrfShape& s,
                                 bool vendor_available = true) {
    return resolve_route<Op::potrf, T>(forced, s, vendor_available);
}

} // namespace batchlas::dispatch
