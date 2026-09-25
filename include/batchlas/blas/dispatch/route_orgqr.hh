#pragma once

// orgqr routing: the one native arm is an identity fill plus a ROUTED ormqr, so supports()
// transcribes ormqr's gates and pinning orgqr needs BATCHLAS_ORMQR_ROUTE set as well.

#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/dispatch/route_resolve.hh>

namespace batchlas::dispatch {

struct OrgqrShape : OpShape {
    bool blocked_available = false;  // the ORGQR driver, not ormqr_blocked (already true)

    int64_t rows() const { return m; }
    int64_t cols() const { return n; }
    int64_t reflectors() const { return k; }
};

inline constexpr Route kOrgqrOrder[] = {
    {Origin::Native, Algorithm::Blocked},
    {Origin::Vendor, Algorithm::Auto},
};

template <typename T>
struct RouteTable<Op::orgqr, T> {
    static bool supports(Route r, const OrgqrShape& s) {
        if (is_vendor(r)) return true;   // the vendor serves everything it is given
        if (!is_native(r)) return false;

        if (!s.is_gpu) return false;

        // Unreachable today (the apply is fixed at NoTrans); kept so a future Q^H
        // spelling inherits ormqr's exclusion rather than silently losing it.
        if constexpr (is_std_complex_v<T>) {
            if (s.transA == Transpose::Trans) return false;
        }

        // Q's columns live in C^m; n > m runs off the end of the identity (OOB, not speed).
        if (s.n > s.m) return false;

        // One identity and one ormqr, single (m, n, ld, stride), serve the whole batch.
        if (s.heterogeneous_batch) return false;

        if (s.m < 1 || s.n < 1 || s.batch < 1) return false;

        switch (r.algo) {
            case Algorithm::Blocked:
                // No extent bound: a speed cutoff here silently drops a forced route.
                return s.blocked_available;

            default:
                return false;
        }
    }

    // Native to n = 512 on both extents, every type; the vendor above it. The doc's ratios
    // mean "beats the per-item cusolverDnXorgqr LOOP", never "beats cuSOLVER".
    // evidence: docs/perf/qr.md#the-shipped-orgqr-ceiling
    static bool preferred(Route r, const OrgqrShape& s) {
        if (!is_native(r)) return false;
        return s.cols() <= 512 && s.rows() <= 512;
    }

    static constexpr const Route* order_begin() { return kOrgqrOrder; }
    static constexpr const Route* order_end() {
        return kOrgqrOrder + (sizeof(kOrgqrOrder) / sizeof(kOrgqrOrder[0]));
    }
};

template <typename T>
inline Route resolve_orgqr_route(Route forced, const OrgqrShape& s,
                                 bool vendor_available = true) {  // facade always passes it
    return resolve_route<Op::orgqr, T>(forced, s, vendor_available);
}

} // namespace batchlas::dispatch
