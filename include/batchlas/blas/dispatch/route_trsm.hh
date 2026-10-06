#pragma once

/// @file
/// @brief TRSM's routing table: the native CTA and blocked arms, the vendor arm, and the window between them.
///
/// supports() is correctness only: a speed cutoff there makes trsm THROW on a
/// vendor-free build, not merely run slow.
/// @ingroup dispatch

#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/dispatch/route_resolve.hh>

namespace batchlas::dispatch {

/// @brief Largest triangle order the CTA trsm kernel accepts on this build; 0 when not linked.
///
/// Defined in the kernel TU; called by the shape builder, never by the table.
/// @ingroup dispatch
template <typename T>
int trsm_cta_max_n();

/// @brief TRSM routing shape: k is the triangle's order, and the build's capacities.
/// @ingroup dispatch
struct TrsmShape : OpShape {
    int cta_max_n = 0;   ///< CTA capacity; zero means this build has no native kernel

    bool blocked_available = false;   ///< must describe the build: claiming Blocked when unlinked routes to nothing

    int64_t tri_order() const { return k; }                            ///< order of the triangular matrix
    int64_t rhs_count() const { return side == Side::Left ? n : m; }   ///< right-hand sides
};

/// @brief TRSM walk order: CTA, Blocked, the vendor.
/// @ingroup dispatch
inline constexpr Route kTrsmOrder[] = {
    {Origin::Native, Algorithm::CTA},
    {Origin::Native, Algorithm::Blocked},
    {Origin::Vendor, Algorithm::Auto},
};

/// @brief TRSM routes: `{Native, CTA}`, `{Native, Blocked}` and the vendor.
///
/// preferred(): native at batch >= 8, except float Side::Right below batch 128
/// with order > 32. Evidence: @ref md_docs_2perf_2trsm "docs/perf/trsm.md".
/// @ingroup dispatch
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
    // evidence: docs/perf/trsm.md#trsm-the-preferred-window-as-implemented
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

/// @brief resolve_route() for trsm.
/// @trap Call THIS, not resolve_route_uninstrumented(): it records trsm's coverage row,
///       so an added record_level3_route call would double-count.
/// @ingroup dispatch
template <typename T>
inline Route resolve_trsm_route(Route forced, const TrsmShape& s,
                                bool vendor_available = true) {
    return resolve_route<Op::trsm, T>(forced, s, vendor_available);
}

} // namespace batchlas::dispatch
