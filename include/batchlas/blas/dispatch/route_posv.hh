#pragma once

// POSV routing: Tiny is the fused kernel, CTA is potrf + one fused solve, Blocked is
// potrf + two trsm. evidence: docs/perf/potrf.md#the-fused-potrs-solve

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

    // The fused solve's device-queried capacity in n * nrhs elements; 0 = absent.
    int64_t fused_max_rhs_elems = 0;
    int64_t fused_max_nrhs = 0;

    int64_t order() const { return m; }
    int64_t nrhs() const { return n; }
};

inline constexpr Route kPosvOrder[] = {
    {Origin::Native, Algorithm::Tiny},
    {Origin::Native, Algorithm::CTA},
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
            case Algorithm::CTA:
                if (!s.is_gpu || !s.has_sg32 || !s.composed_available) return false;
                if (s.nrhs() > s.fused_max_nrhs) return false;
                return s.order() * s.nrhs() <= s.fused_max_rhs_elems;
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
            case Algorithm::Tiny:    return tiny_window(s);
            case Algorithm::CTA:     return !tiny_window(s);  // Blocked: capacity fallback
            case Algorithm::Blocked: return !tiny_window(s);
            default:                 return false;
        }
    }

    // cdouble stops at 16 because P1 instantiates no further (plan D3).
    static constexpr int64_t tiny_window_max_n() {
        if constexpr (std::is_same_v<T, std::complex<double>>) return 16;
        return 32;
    }

    // Measured against CTA for float and cfloat only; fp64 keeps the tier ceiling.
    // evidence: docs/perf/potrf.md#the-tiny-window-moved
    static bool tiny_window(const PosvShape& s) {
        if (s.order() < 1 || s.order() > tiny_window_max_n()) return false;
        if constexpr (std::is_same_v<T, float>) {
            return s.order() <= 16 || s.nrhs() >= 3;
        } else if constexpr (std::is_same_v<T, std::complex<float>>) {
            return s.order() <= 8 || (s.order() <= 16 && s.nrhs() >= 3);
        } else {
            return true;
        }
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
