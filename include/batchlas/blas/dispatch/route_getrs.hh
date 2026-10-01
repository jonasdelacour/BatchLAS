#pragma once

// GETRS routing table. {Native, CTA} is the fused narrow-RHS kernel (getrs_fused.cc);
// {Native, Blocked} is the composition (getrs_native.cc): laswp + 2 trsm.

#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/dispatch/route_resolve.hh>

#include <complex>
#include <cstdint>
#include <type_traits>

namespace batchlas::dispatch {

struct GetrsShape : OpShape {
    bool blocked_available = false;

    // From sub_group_sizes: the fused kernels carry reqd_sub_group_size(32), so a
    // {64}-only device cannot launch them.
    bool has_sg32 = false;

    // Bounds n*nrhs, not n: the fused kernel holds the whole RHS block in local memory.
    int64_t fused_max_elems = 0;

    // The widest nrhs the fused kernel is instantiated for: a build fact, not a device one.
    int64_t fused_max_nrhs = 0;

    // The register-resident tier's order ceiling (a build fact); 0 = absent. Any nrhs.
    int64_t tiny_max_n = 0;

    int64_t order() const { return m; }
    int64_t nrhs() const { return n; }
};

inline constexpr Route kGetrsOrder[] = {
    {Origin::Native, Algorithm::Tiny},
    {Origin::Native, Algorithm::CTA},
    {Origin::Native, Algorithm::Blocked},
    {Origin::Vendor, Algorithm::Auto},
};

template <typename T>
struct RouteTable<Op::getrs, T> {
    // Correctness only: a forced route bypasses preferred() but never supports(), so a
    // speed gate here would send a pinned `native:cta` to the vendor and pass green.
    static bool supports(Route r, const GetrsShape& s) {
        if (is_vendor(r)) return true;   // the vendor serves everything it is given
        if (!is_native(r)) return false;

        if (r.algo == Algorithm::Tiny) {
            if (s.tiny_max_n <= 0) return false;
        } else if (r.algo == Algorithm::CTA) {
            if (s.fused_max_elems <= 0 || s.fused_max_nrhs <= 0) return false;
        } else if (r.algo == Algorithm::Blocked) {
            if (!s.blocked_available) return false;
        }

        if (!s.is_gpu) return false;

        if (!s.has_sg32) return false;

        if (s.heterogeneous_batch) return false;

        if (s.order() < 1 || s.nrhs() < 1 || s.batch < 1) return false;

        // Pivot format: the GPU arms pack 1-based int32 into the low half of the int64
        // span, netlib writes true int64; mixing them is silently wrong (info stays 0).
        if (s.backend == Backend::NETLIB) return false;

        switch (r.algo) {
            case Algorithm::Tiny:
                return s.order() <= s.tiny_max_n;
            case Algorithm::CTA:
                return cta_fits(s);
            case Algorithm::Blocked:
                return true;
            default:
                return false;
        }
    }

    static bool cta_fits(const GetrsShape& s) {
        return s.fused_max_elems > 0 && s.fused_max_nrhs > 0 &&
               s.order() * s.nrhs() <= s.fused_max_elems && s.nrhs() <= s.fused_max_nrhs;
    }

    static bool tiny_fits(const GetrsShape& s) {
        return s.tiny_max_n > 0 && s.order() <= s.tiny_max_n;
    }

    // sm_120 float/cfloat only. evidence: docs/perf/blackwell.md#lu-getrs-tiny
    static bool tiny_native(const GetrsShape& s) {
        constexpr bool kSingle =
            std::is_same_v<T, float> || std::is_same_v<T, std::complex<float>>;
        return kSingle && is_sm120_family(s.cuda_cc) && tiny_fits(s);
    }

    // Every clause below is a window EDGE. evidence: docs/perf/lu.md#getrs-fused-window-evidence
    static bool preferred(Route r, const GetrsShape& s) {
        if (!is_native(r)) return false;

        if (r.algo == Algorithm::Tiny) return tiny_window(s);

        if (r.algo == Algorithm::Blocked) {
            // Deliberately conservative; it gives up measured wins below 128.
            // evidence: docs/perf/lu.md#getrs-composition-window-evidence
            if (s.batch < 128) return false;
            if (sm120_single(s)) return composition_sm120(s);
            if constexpr (std::is_same_v<T, float>)  return s.nrhs() >= 64;
            if constexpr (std::is_same_v<T, double>) return s.nrhs() >= 128;
            return false;   // cfloat and cdouble earn nothing at any width
        }

        if (r.algo != Algorithm::CTA) return false;

        // A defect fix, not a knob: A and B shipped unbounded on a grid whose smallest
        // order was 32. evidence: docs/perf/lu.md#getrs-order-floor-evidence
        if (s.order() < 32) return false;

        if (s.nrhs() <= 2) return true;                  // clause A

        if constexpr (std::is_same_v<T, float>) {        // clause B
            if (s.nrhs() <= 4) return true;
        }
        // evidence: docs/perf/blackwell.md#getrs-windows-after-the-trsm-fix
        if (is_sm120_family(s.cuda_cc)) {
            if constexpr (std::is_same_v<T, float>) return s.nrhs() <= 8 && s.order() >= 192;
            if constexpr (std::is_same_v<T, std::complex<float>>) {
                return (s.nrhs() <= 4 && s.order() >= 96) ||
                       (s.nrhs() <= 8 && s.order() >= 384);
            }
        }
        return false;
    }

    static bool sm120_single(const GetrsShape& s) {
        constexpr bool kSingle =
            std::is_same_v<T, float> || std::is_same_v<T, std::complex<float>>;
        return kSingle && is_sm120_family(s.cuda_cc);
    }

    // sm_120 composition window, re-bracketed after the trsm fix. cfloat dips below the
    // vendor at n 48..64, nrhs 16..24, batch 4096-8192, hence its second edge.
    // evidence: docs/perf/blackwell.md#getrs-windows-after-the-trsm-fix
    static bool composition_sm120(const GetrsShape& s) {
        if (s.order() < 32 || s.nrhs() < 16) return false;
        if constexpr (std::is_same_v<T, std::complex<float>>) {
            return s.nrhs() >= 32 || s.order() >= 96;
        }
        return true;
    }

    // Every measured sm_120 cell, n 2..32 x nrhs 1..64 x batch 64..32768, both types.
    static bool tiny_window(const GetrsShape& s) { return tiny_native(s); }

    // Vendor-free tie-break, EXHAUSTIVE: exactly one tier answers for every shape, so the
    // plain walk never picks Tiny (first in the order) where it was not measured. Off
    // sm_120 this is the pre-Tiny answer: CTA wherever it fits, Blocked elsewhere.
    static bool native_tier_preferred(Route r, const GetrsShape& s) {
        if (!is_native(r)) return true;
        switch (r.algo) {
            case Algorithm::Tiny:    return tiny_native(s);
            case Algorithm::CTA:     return !tiny_native(s) && cta_fits(s);
            case Algorithm::Blocked: return !tiny_native(s) && !cta_fits(s);
            default:                 return false;
        }
    }

    static constexpr const Route* order_begin() { return kGetrsOrder; }
    static constexpr const Route* order_end() {
        return kGetrsOrder + (sizeof(kGetrsOrder) / sizeof(kGetrsOrder[0]));
    }
};

template <typename T>
inline Route resolve_getrs_route(Route forced, const GetrsShape& s,
                                 bool vendor_available = true) {
    return resolve_route<Op::getrs, T>(forced, s, vendor_available);
}

} // namespace batchlas::dispatch
