#pragma once

// GEMV's routing table: pure predicates over GemvShape (device- and env-dependent
// facts are built in src/backends/gemv_route.hh). supports() is correctness only,
// preferred() the measured window; see docs/perf/gemv.md.

#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/dispatch/route_resolve.hh>

namespace batchlas::dispatch {

// Do not shadow OpShape's transA or is_gpu: resolve_route slices this struct to
// OpShape, so a shadowing member is dropped and every gemv coverage row is wrong.
struct GemvShape : OpShape {
    bool direct_available = false;   // not linked => unsupported, not unimplemented
    bool cta_available = false;

    // From sycl::info::device::sub_group_sizes, never
    // get_property(MAX_SUB_GROUP_SIZE), which reports sub_group_sizes()[0].
    bool has_sg32 = false;

    // Predicates must use these: which of m and n is which swaps with transA.
    int64_t out_len() const { return transA == Transpose::NoTrans ? m : n; }
    int64_t red_len() const { return transA == Transpose::NoTrans ? n : m; }
};

// A capability ladder, tighter first, not a preference: CTA serves only the
// transposed GPU shapes it supports, Direct serves everything else.
inline constexpr Route kGemvOrder[] = {
    {Origin::Native, Algorithm::CTA},
    {Origin::Native, Algorithm::Direct},
    {Origin::Vendor, Algorithm::Auto},
};

template <typename T>
struct RouteTable<Op::gemv, T> {
    static bool supports(Route r, const GemvShape& s) {
        if (is_vendor(r)) return true;   // vendor serves everything
        if (!is_native(r)) return false;

        // Correctness, not speed: one launch serves the batch from a single
        // (m, n, ld, stride) tuple and gemv has no heterogeneous walker.
        if (s.heterogeneous_batch) return false;

        if (s.m < 0 || s.n < 0 || s.batch < 1) return false;

        switch (r.algo) {
            case Algorithm::Direct:
                // No GPU gate, deliberately: both bodies are serial dot products
                // with no collective, and vendor-free builds need them on native_cpu.
                return s.direct_available;

            case Algorithm::CTA:
                // Body 3 carries [[sycl::reqd_sub_group_size(32)]] -- a device
                // not enumerating 32 aborts the launch -- and has no NoTrans body.
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

// Pure. resolve_route is also what records gemv in the coverage table.
template <typename T>
inline Route resolve_gemv_route(Route forced, const GemvShape& s,
                                bool vendor_available = true) {
    return resolve_route<Op::gemv, T>(forced, s, vendor_available);
}

} // namespace batchlas::dispatch
