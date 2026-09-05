#pragma once

// GEMM's routing table, pure -- the env read lives in route_env.hh. A speed
// cutoff in supports() would strand shapes with no route at all in a vendor-free
// build; preferred() is the measured window. docs/perf/dispatch.md

#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/dispatch/route_resolve.hh>

namespace batchlas::dispatch {

inline constexpr Route kGemmOrder[] = {
    {Origin::Native, Algorithm::RegisterTiled},
    {Origin::Vendor, Algorithm::Auto},
};

template <typename T>
struct RouteTable<Op::gemm, T> {
    static bool supports(Route r, const OpShape& s) {
        if (r.origin == Origin::Vendor) {
            return true;
        }
        if (r.origin != Origin::Native) {
            return false;
        }
        if (r.algo != Algorithm::RegisterTiled && r.algo != Algorithm::Auto) {
            return false;
        }
        if (s.precision != ComputePrecision::Default) return false;
        // Heterogeneous batch is correct here; preferred() excludes it, not supports().
        return s.m > 0 && s.n > 0 && s.k > 0;
    }

    static bool preferred(Route r, const OpShape& s) {
        if (r.origin != Origin::Native) return false;
        if (!supports(r, s)) return false;

        if (!s.is_gpu) return false;

        if (s.heterogeneous_batch) return false;

        // Complex has no register kernel. evidence: docs/perf/gemm.md#complex-is-refused
        if constexpr (is_std_complex_v<T>) {
            return false;
        } else {
            const int64_t max_dim = s.max_dim();

            if (s.batch < 64) return false;

            if constexpr (std::is_same_v<T, float>) {
                if (s.m != s.n || s.n != s.k) return false;
                // float: square NN only, max_dim <= 32.
                // evidence: docs/perf/gemm.md#float-nn-at-max_dim-32
                if (s.transA != Transpose::NoTrans || s.transB != Transpose::NoTrans) {
                    return false;
                }
                if (max_dim <= 32) return true;
                return false;
            } else if constexpr (std::is_same_v<T, double>) {
                // double: any transpose, any size, k >= 2 (k=1 rank-1 goes to the vendor).
                // evidence: docs/perf/gemm.md#double-the-only-fully-native-window
                return s.k >= 2;
            } else {
                return false;
            }
        }
    }

    static constexpr const Route* order_begin() { return kGemmOrder; }
    static constexpr const Route* order_end() {
        return kGemmOrder + (sizeof(kGemmOrder) / sizeof(kGemmOrder[0]));
    }
};

// A default-constructed `forced` means "no opinion"; see route_env.hh.
template <typename T>
inline Route resolve_gemm_route(Route forced, const OpShape& s,
                                bool vendor_available = true) {
    return resolve_route<Op::gemm, T>(forced, s, vendor_available);
}

} // namespace batchlas::dispatch
