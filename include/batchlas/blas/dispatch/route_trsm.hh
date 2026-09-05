#pragma once

// TRSM's routing table: the native CTA and blocked arms, the vendor arm, and the
// window between them (docs/perf/trsm.md). supports() is correctness only -- a
// speed cutoff there makes trsm THROW on a vendor-free build, not merely run slow.

#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/dispatch/route_resolve.hh>

namespace batchlas::dispatch {

// Defined in the kernel TU; called by the shape builder, never by this table.
template <typename T>
int trsm_cta_max_n();

struct TrsmShape : OpShape {
    // Zero means this build has no native kernel.
    int cta_max_n = 0;

    // Must describe the build: claiming Blocked when unlinked routes to nothing.
    bool blocked_available = false;

    int64_t tri_order() const { return k; }
    int64_t rhs_count() const { return side == Side::Left ? n : m; }
};

inline constexpr Route kTrsmOrder[] = {
    {Origin::Native, Algorithm::CTA},
    {Origin::Native, Algorithm::Blocked},
    {Origin::Vendor, Algorithm::Auto},
};

template <typename T>
struct RouteTable<Op::trsm, T> {
    static bool supports(Route r, const TrsmShape& s) {
        if (is_vendor(r)) return true;
        if (!is_native(r)) return false;

        if (!s.is_gpu) return false;

        // Correctness, not preference: one launch covers the batch with a single
        // (order, q, ld, stride) tuple, and gemm's batch walker has no twin here.
        if (s.heterogeneous_batch) return false;

        const int64_t order = s.tri_order();
        const int64_t q     = s.rhs_count();
        if (order < 1 || q < 1 || s.batch < 1) return false;

        switch (r.algo) {
            case Algorithm::CTA:
                if (s.cta_max_n < 1) return false;
                return order <= s.cta_max_n;

            case Algorithm::Blocked:
                return s.blocked_available && s.cta_max_n >= 1;

            default:
                return false;
        }
    }

    // Native at batch >= 8, except float + Side::Right below batch 128 (order <= 32).
    // evidence: docs/perf/trsm.md#the-preferred-window-as-implemented
    static bool preferred(Route r, const TrsmShape& s) {
        if (!is_native(r)) return false;

        const int64_t order = s.tri_order();

        // evidence: docs/perf/trsm.md#the-batch-floor
        if (s.batch < 8) return false;

        if constexpr (std::is_same_v<T, float>) {
            if (s.side == Side::Left) {
                return true;
            }
            return s.batch >= 128 || order <= 32;
        } else {
            return true;
        }
    }

    static constexpr const Route* order_begin() { return kTrsmOrder; }
    static constexpr const Route* order_end() {
        return kTrsmOrder + (sizeof(kTrsmOrder) / sizeof(kTrsmOrder[0]));
    }
};

// Call THIS, not resolve_route_uninstrumented: it records trsm's coverage row, so
// an added record_level3_route call would double-count.
template <typename T>
inline Route resolve_trsm_route(Route forced, const TrsmShape& s,
                                bool vendor_available = true) {
    return resolve_route<Op::trsm, T>(forced, s, vendor_available);
}

} // namespace batchlas::dispatch
