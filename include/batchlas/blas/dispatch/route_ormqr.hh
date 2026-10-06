#pragma once

/// @file
/// @brief ORMQR's routing table.
///
/// It replaced a chooser that returned a forced provider without checking
/// support, so forcing could run the blocked kernel on excluded inputs and the
/// buffer-size query could size a different route than the call ran. Splitting
/// supports() from the forced request, and resolving once through a pure
/// table, makes both impossible by construction.
/// @ingroup dispatch
// evidence: docs/perf/dispatch.md#dispatch-the-ormqr-chooser-that-forced-past-supports

#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/dispatch/route_resolve.hh>

namespace batchlas::dispatch {

/// @brief ORMQR walk order: the blocked kernel, then the vendor.
/// @ingroup dispatch
inline constexpr Route kOrmqrOrder[] = {
    {Origin::Native, Algorithm::Blocked},
    {Origin::Vendor, Algorithm::Auto},
};

/// @brief ORMQR routes: `{Native, Blocked}` and the vendor.
///
/// supports(): GPU queue, and for complex T not a plain `Transpose::Trans`.
/// preferred() equals "native and supported": no measured window exists.
/// Evidence: @ref md_docs_2perf_2qr "docs/perf/qr.md".
/// @ingroup dispatch
template <typename T>
struct RouteTable<Op::ormqr, T> {
    // Verbatim ormqr_supports_blocked, and nothing else.
    static bool supports(Route r, const OpShape& s) {
        if (is_vendor(r)) return true;
        if (!is_native(r)) return false;
        if (r.algo != Algorithm::Blocked && r.algo != Algorithm::Auto) return false;

        if (!s.is_gpu) return false;
        if constexpr (is_std_complex_v<T>) {
            // Complex with a plain Trans (as opposed to ConjTrans) is excluded.
            // Transcribed as-is: ormqr_supports_blocked gives no reason for it,
            // and a refactor is not the place to guess at one.
            if (s.transA == Transpose::Trans) return false;
        }
        return true;
    }

    // No measured window: preferred == supported. Kept separate because a future
    // crossover goes HERE; in supports() it would become a correctness claim.
    static bool preferred(Route r, const OpShape& s) {
        return is_native(r) && supports(r, s);
    }

    static constexpr const Route* order_begin() { return kOrmqrOrder; }
    static constexpr const Route* order_end() {
        return kOrmqrOrder + (sizeof(kOrmqrOrder) / sizeof(kOrmqrOrder[0]));
    }
};

/// @brief resolve_route() for ormqr.
/// @ingroup dispatch
template <typename T>
inline Route resolve_ormqr_route(Route forced, const OpShape& s,
                                 bool vendor_available = true) {
    return resolve_route<Op::ormqr, T>(forced, s, vendor_available);
}

} // namespace batchlas::dispatch
