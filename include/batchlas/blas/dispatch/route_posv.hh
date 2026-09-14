#pragma once

// POSV routing: {Native, Tiny} is the fused kernel, {Native, Blocked} the
// `potrf; trsm; trsm` composition. As in route_gesv.hh the order array carries NO
// vendor entry. evidence: docs/perf/potrf.md#the-fused-posv-tier

#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/dispatch/route_resolve.hh>

#include <complex>
#include <cstdint>
#include <type_traits>

namespace batchlas::dispatch {

struct PosvShape : OpShape {
    bool has_sg32 = false;

    int64_t tiny_max_n = 0;
    int64_t tiny_max_nrhs = 0;

    // potrf's own arms plus trsm; the composition cannot run without a potrf.
    bool composed_available = false;

    int64_t order() const { return m; }
    int64_t nrhs() const { return n; }
};

inline constexpr Route kPosvOrder[] = {
    {Origin::Native, Algorithm::Tiny},
    {Origin::Native, Algorithm::Blocked},
};

template <typename T>
struct RouteTable<Op::posv, T> {
    static bool supports(Route r, const PosvShape& s) {
        if (!is_native(r)) return false;

        if (s.order() < 1 || s.nrhs() < 1 || s.batch < 1) return false;
        if (s.heterogeneous_batch) return false;

        switch (r.algo) {
            case Algorithm::Tiny:
                // NETLIB is excluded by is_gpu, not by a backend test: unlike gesv
                // this op has no pivot span, so there is no int32/int64 hazard and
                // the host backend serves the composed arm normally.
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
    static bool preferred(Route r, const PosvShape& s) {
        static_cast<void>(r);
        static_cast<void>(s);
        return false;
    }

    // R8b: resolve_posv_route always passes vendor_available=false, so preferred()
    // above is never reached and THIS is the shipping hook; the walk tests
    // `supports && native_tier_preferred` in order, so the fit is resolved first.
    // evidence: docs/perf/potrf.md#p2-the-measured-posv-window
    static bool native_tier_preferred(Route r, const PosvShape& s) {
        switch (r.algo) {
            case Algorithm::Tiny:    return s.order() <= tiny_window_max_n();
            case Algorithm::Blocked: return true;
            default:                 return false;
        }
    }

    // cdouble stops at 16 because P1 instantiates no further (plan D3).
    static constexpr int64_t tiny_window_max_n() {
        if constexpr (std::is_same_v<T, std::complex<double>>) return 16;
        return 32;
    }

    static constexpr const Route* order_begin() { return kPosvOrder; }
    static constexpr const Route* order_end() {
        return kPosvOrder + (sizeof(kPosvOrder) / sizeof(kPosvOrder[0]));
    }
};

template <typename T>
inline Route resolve_posv_route(Route forced, const PosvShape& s) {
    return resolve_route<Op::posv, T>(forced, s, /*vendor_available=*/false);
}

}  // namespace batchlas::dispatch
