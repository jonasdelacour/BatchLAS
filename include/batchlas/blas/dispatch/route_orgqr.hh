#pragma once

// orgqr routing: the one native arm, {Native, Blocked}, is an identity fill plus a routed
// ormqr, so supports() transcribes ormqr's gates and pinning orgqr needs both
// BATCHLAS_ORGQR_ROUTE and BATCHLAS_ORMQR_ROUTE. evidence: docs/perf/qr.md#the-vendor-baseline

#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/dispatch/route_resolve.hh>

namespace batchlas::dispatch {

struct OrgqrShape : OpShape {
    // Is the orgqr driver compiled -- not merely ormqr_blocked, which is true already.
    bool blocked_available = false;

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

        // Unreachable today -- the identity apply is fixed at NoTrans -- but kept so a
        // future Q^H spelling inherits ormqr's exclusion instead of silently losing it.
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

    // Route-neutral: false everywhere, so Origin::Auto keeps the vendor and only a
    // vendor-free build resolves to the native arm. Not `is_native(r) && supports(r, s)`
    // -- there are measured losing cells. evidence: docs/perf/qr.md#orgqr-grid
    static bool preferred(Route r, const OrgqrShape& s) {
        static_cast<void>(r);
        static_cast<void>(s);
        return false;
    }

    static constexpr const Route* order_begin() { return kOrgqrOrder; }
    static constexpr const Route* order_end() {
        return kOrgqrOrder + (sizeof(kOrgqrOrder) / sizeof(kOrgqrOrder[0]));
    }
};

// The facade passes vendor_available explicitly; the default hides the vendor-free fallback.
template <typename T>
inline Route resolve_orgqr_route(Route forced, const OrgqrShape& s,
                                 bool vendor_available = true) {
    return resolve_route<Op::orgqr, T>(forced, s, vendor_available);
}

} // namespace batchlas::dispatch
