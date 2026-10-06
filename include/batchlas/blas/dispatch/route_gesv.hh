#pragma once

/// @file
/// @brief GESV's routing table: `{Native, Tiny}` is the fused kernel, `{Native, Blocked}` the `getrf; getrs` composition.
/// @ingroup dispatch

#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/dispatch/route_resolve.hh>

#include <complex>
#include <cstdint>
#include <type_traits>

namespace batchlas::dispatch {

/// @brief GESV routing shape: m is the order, n the number of right-hand sides.
/// @ingroup dispatch
struct GesvShape : OpShape {
    bool has_sg32 = false;   ///< the fused kernel carries reqd_sub_group_size(32)

    int64_t tiny_max_n = 0;      ///< fused kernel's largest order; 0 = absent
    int64_t tiny_max_nrhs = 0;   ///< fused kernel's largest nrhs; 0 = absent

    bool composed_available = false;   ///< the getrf + getrs composition is linked

    int64_t order() const { return m; }   ///< order of A
    int64_t nrhs() const { return n; }    ///< right-hand sides
};

/// @brief GESV walk order.
/// @invariant No vendor entry: no vendor ships a batched gesv, so `{Vendor, Auto}` is
///            only resolve_route()'s terminal "nothing serves this" answer.
/// @ingroup dispatch
inline constexpr Route kGesvOrder[] = {
    {Origin::Native, Algorithm::Tiny},
    {Origin::Native, Algorithm::Blocked},
};

/// @brief GESV routes: `{Native, Tiny}` (fused) and `{Native, Blocked}` (composition); no vendor.
///
/// preferred() is permanently all-false; resolve_gesv_route() passes
/// `vendor_available = false`, so native_tier_preferred() is the shipping window.
/// Evidence: @ref md_docs_2perf_2lu "docs/perf/lu.md".
/// @ingroup dispatch
template <typename T>
struct RouteTable<Op::gesv, T> {
    // Correctness only: a forced route bypasses preferred() but never supports().
    static bool supports(Route r, const GesvShape& s) {
        if (!is_native(r)) return false;

        if (s.order() < 1 || s.nrhs() < 1 || s.batch < 1) return false;
        if (s.heterogeneous_batch) return false;

        switch (r.algo) {
            case Algorithm::Tiny:
                // Pivot format, and the gate belongs HERE and not above the switch: the
                // GPU arms pack 1-based int32 into the int64 span's low half where netlib
                // writes true int64. The composed arm's legs share a backend and agree.
                if (s.backend == Backend::NETLIB) return false;
                if (!s.is_gpu || !s.has_sg32) return false;
                if (s.tiny_max_n <= 0 || s.tiny_max_nrhs <= 0) return false;
                if (s.order() > s.tiny_max_n) return false;
                if (s.nrhs() > s.tiny_max_nrhs) return false;
                return true;
            case Algorithm::Blocked:
                return s.composed_available;
            default:
                return false;
        }
    }

    // ALL FALSE, PERMANENTLY: not this op's shipping hook; see below.
    static bool preferred(Route r, const GesvShape& s) {
        static_cast<void>(r);
        static_cast<void>(s);
        return false;
    }

    // R8b, INVERTED here: vendor_available is always false, so preferred() above is
    // never reached and THIS is the shipping hook -- and the walk resolves the fit
    // first, so exactly one tier answers. evidence: docs/perf/lu.md#p2-the-measured-gesv-window
    static bool native_tier_preferred(Route r, const GesvShape& s) {
        switch (r.algo) {
            case Algorithm::Tiny:    return s.order() <= tiny_window_max_n();
            case Algorithm::Blocked: return true;
            default:                 return false;
        }
    }

    // 0 = no window; cfloat 17..32 is under the gate. evidence: docs/perf/lu.md#end-to-end-after-the-column-bucket
    static constexpr int64_t tiny_window_max_n() {
        if constexpr (std::is_same_v<T, float>) return 32;
        if constexpr (std::is_same_v<T, std::complex<float>>) return 16;
        return 0;
    }

    static constexpr const Route* order_begin() { return kGesvOrder; }
    static constexpr const Route* order_end() {
        return kGesvOrder + (sizeof(kGesvOrder) / sizeof(kGesvOrder[0]));
    }
};

/// @brief resolve_route() for gesv, always on the vendor-free walk.
///
/// `vendor_available` is not a parameter: passing true would let the automatic
/// walk answer `{Vendor, Auto}` for a shape the composed arm serves perfectly well.
/// @ingroup dispatch
template <typename T>
inline Route resolve_gesv_route(Route forced, const GesvShape& s) {
    return resolve_route<Op::gesv, T>(forced, s, /*vendor_available=*/false);
}

}  // namespace batchlas::dispatch
