#pragma once

/// @file
/// @brief GEMV's routing table: pure predicates over GemvShape.
///
/// Device- and environment-dependent facts are gathered by the shape builder in
/// src/backends/gemv_route.hh.
/// @ingroup dispatch

#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/dispatch/route_resolve.hh>

namespace batchlas::dispatch {

/// @brief GEMV routing shape: OpShape plus the build's kernel availability.
/// @trap Do not shadow OpShape's `transA` or `is_gpu`: resolve_route() slices this
///       struct to OpShape, so a shadowing member is dropped from every coverage row.
/// @ingroup dispatch
struct GemvShape : OpShape {
    bool direct_available = false;   ///< Direct kernel linked; not linked means unsupported, not unimplemented
    bool cta_available = false;      ///< CTA kernel linked

    /// Sub-group size 32 is available; enumerated from `sub_group_sizes`, because
    /// `MAX_SUB_GROUP_SIZE` reports `sub_group_sizes()[0]`.
    bool has_sg32 = false;

    /// @brief Output length; predicates must use this, since which of m and n it is swaps with transA.
    int64_t out_len() const { return transA == Transpose::NoTrans ? m : n; }
    /// @brief Reduction length; the counterpart of out_len().
    int64_t red_len() const { return transA == Transpose::NoTrans ? n : m; }
};

/// @brief GEMV walk order: a capability ladder, tighter first, not a preference list.
/// @ingroup dispatch
inline constexpr Route kGemvOrder[] = {
    {Origin::Native, Algorithm::CTA},
    {Origin::Native, Algorithm::Direct},
    {Origin::Vendor, Algorithm::Auto},
};

/// @brief GEMV routes: `{Native, CTA}`, `{Native, Direct}` and the vendor.
///
/// preferred() is a single window, complex<double> transposed CTA; everything
/// else prefers the vendor. Evidence: @ref md_docs_2perf_2gemv "docs/perf/gemv.md".
/// @ingroup dispatch
template <typename T>
struct RouteTable<Op::gemv, T> {
    static bool supports(Route r, const GemvShape& s) {
        if (is_vendor(r)) return true;   // vendor serves everything
        if (!is_native(r)) return false;

        // Correctness, not speed: gemv has no heterogeneous-batch walker.
        if (s.heterogeneous_batch) return false;

        if (s.m < 0 || s.n < 0 || s.batch < 1) return false;

        switch (r.algo) {
            case Algorithm::Direct:
                // No GPU gate, deliberately: vendor-free builds need it on native_cpu.
                return s.direct_available;

            case Algorithm::CTA:
                // Body 3 needs sub-group 32 (else the launch aborts); no NoTrans body.
                return s.cta_available && s.is_gpu && s.has_sg32 &&
                       s.transA != Transpose::NoTrans;

            default:
                return false;   // including Auto: a bare "native" names neither arm
        }
    }

    // Window: complex<double> + CTA + transposed, 64 <= red_len() <= 352,
    // out_len() >= 256, batch >= 320. evidence: docs/perf/gemv.md#the-cdouble-window-boundaries
    static bool preferred(Route r, const GemvShape& s) {
        if (!is_native(r) || r.algo != Algorithm::CTA) return false;

        if constexpr (std::is_same_v<T, std::complex<double>>) {
            if (s.transA == Transpose::NoTrans) return false;

            const int64_t red = s.red_len();   // == A.rows() under Trans
            const int64_t out = s.out_len();   // == A.cols() under Trans
            return red >= 64 && red <= 352 && out >= 256 && s.batch >= 320;
        }
        return false;
    }

    static constexpr const Route* order_begin() { return kGemvOrder; }
    static constexpr const Route* order_end() {
        return kGemvOrder + (sizeof(kGemvOrder) / sizeof(kGemvOrder[0]));
    }
};

/// @brief resolve_route() for gemv.
/// @ingroup dispatch
template <typename T>
inline Route resolve_gemv_route(Route forced, const GemvShape& s,
                                bool vendor_available = true) {
    return resolve_route<Op::gemv, T>(forced, s, vendor_available);
}

} // namespace batchlas::dispatch
