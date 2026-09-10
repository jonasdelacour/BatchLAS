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

    // Native up to n = 512, which is where the evidence stops. Read the ratios
    // below as "beats the
    // per-item loop", NOT as "beats cuSOLVER": cublas.cc dispatches
    // cusolverDnXorgqr once PER BATCH ITEM on an out-of-order sub-queue, so
    // the vendor arm is batch launches deep and the comparison is against a
    // structure, not against a kernel.
    //
    // That is exactly why the window has no floor. The measured margin is
    // smallest at the largest order and never approaches the flip gate:
    //
    //   T         n = 512   n = 256   n = 64    n <= 16
    //   float       3.64      5.30     27.10     >= 181
    //   cfloat      2.54      4.04     15.09     >= 172
    //   double      4.61      8.46     27.09     >= 98
    //   cdouble     2.77      4.75     11.31     >= 42
    //
    // 66 square cells measured over four types and orders 4..512, zero losses,
    // minimum 2.54.
    //
    // THE 512 CEILING IS LOAD-BEARING, and it is the bracket. There is no
    // losing cell inside the measured range, so the bound comes from the
    // cells ABOVE it, which docs/perf/qr.md#orgqr-grid records as losses:
    // cfloat n = 1024 is 0.82x (0.88 at batch 256, and the record does not
    // claim it crosses), cdouble n = 1024 is 0.78x, and at n = 2048 every
    // type loses -- float 0.41x, cfloat 0.31x, cdouble 0.46x. An unbounded
    // `is_native(r)` would route all of those native. Only float n = 1024
    // recovers with batch (0.84 / 1.11 / 1.27 / 1.33 at batch 32..256), and
    // one type crossing is not a window.
    //
    // The vendor arm also costs 3.3x the workspace (4,870 MB against 1,476 at
    // cdouble n=64, batch 8192), which the window does not weigh but a caller
    // near the memory ceiling will feel.
    // evidence: docs/perf/small-n-baseline.md#orgqr, docs/perf/qr.md#orgqr-grid
    static bool preferred(Route r, const OrgqrShape& s) {
        if (!is_native(r)) return false;
        return s.cols() <= 512 && s.rows() <= 512;
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
