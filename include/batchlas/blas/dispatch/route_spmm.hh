#pragma once

/// @file
/// @brief SPMM's routing table (sparse CSR times dense). Pure: device and environment
///        queries live in src/backends/spmm_route.hh.
/// @ingroup dispatch
// evidence: docs/perf/spmm.md

#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/dispatch/route_resolve.hh>
#include <batchlas/blas/enums.hh>

#include <complex>
#include <type_traits>

namespace batchlas::dispatch {

/// @brief SPMM routing shape: OpShape plus the sparse format and the linked kernel bodies.
///
/// No nnz field, deliberately: the honest per-item nnz(b) reads device memory, and
/// the shape builder also runs inside spmm_buffer_size, where that is a segfault.
/// @trap Never shadow an OpShape field: resolve_route() slices this struct to OpShape
///       for the coverage table, so a shadowing member is silently not copied.
/// @ingroup dispatch
struct SpmmShape : OpShape {
    MatrixFormat format = MatrixFormat::Dense;   ///< format of A; only CSR has a native route

    bool gather_available = false;   ///< the transA == NoTrans body is linked
    bool scatter_available = false;  ///< the transA != NoTrans bodies (scale + scatter) are linked

    int64_t nrhs() const { return n; }   ///< columns of B and C
    /// @brief Rows of C. out_rows/red_rows swap with transA, so predicates must never spell m or k.
    int64_t out_rows() const { return transA == Transpose::NoTrans ? m : k; }
    /// @brief Reduction length; the counterpart of out_rows().
    int64_t red_rows() const { return transA == Transpose::NoTrans ? k : m; }
};

/// @brief SPMM walk order: the native CSR kernel, then the vendor.
/// @ingroup dispatch
inline constexpr Route kSpmmOrder[] = {
    {Origin::Native, Algorithm::Direct},
    {Origin::Vendor, Algorithm::Auto},
};

/// @brief SPMM routes: `{Native, Direct}` (CSR gather / scatter) and the vendor.
///
/// preferred(): the native NoTrans gather everywhere, minus complex<float> with
/// transB. Evidence: @ref md_docs_2perf_2spmm "docs/perf/spmm.md".
/// @ingroup dispatch
template <typename T>
struct RouteTable<Op::spmm, T> {
    // Correctness only: a speed gate here drops the row from the vendor-free walk.
    // evidence: docs/perf/spmm.md#supports-and-what-is-deliberately-not-in-it
    static bool supports(Route r, const SpmmShape& s) {
        if (is_vendor(r)) return true;   // the vendor serves everything it is given
        if (!is_native(r)) return false;

        if (s.format != MatrixFormat::CSR) return false;

        if (s.heterogeneous_batch) return false;

        if (s.m < 0 || s.n < 0 || s.k < 0 || s.batch < 1) return false;

        switch (r.algo) {
            case Algorithm::Direct:
                // No is_gpu gate on purpose: gating strands the Backend::NETLIB rows.
                return (s.transA == Transpose::NoTrans) ? s.gather_available
                                                        : s.scatter_available;

            default:
                return false;
        }
    }

    // Window: the native CSR gather everywhere, minus complex<float> with transB.
    // evidence: docs/perf/spmm.md#spmm-the-preferred-window-as-implemented
    static bool preferred(Route r, const SpmmShape& s) {
        if (!is_native(r) || r.algo != Algorithm::Direct) return false;
        if (s.format != MatrixFormat::CSR) return false;

        if (s.transA != Transpose::NoTrans) return false;

        if constexpr (std::is_same_v<T, std::complex<float>>) {
            if (s.transB != Transpose::NoTrans) return false;
        }
        return true;
    }

    static constexpr const Route* order_begin() { return kSpmmOrder; }
    static constexpr const Route* order_end() {
        return kSpmmOrder + (sizeof(kSpmmOrder) / sizeof(kSpmmOrder[0]));
    }
};

/// @brief resolve_route() for spmm.
/// @ingroup dispatch
template <typename T>
inline Route resolve_spmm_route(Route forced, const SpmmShape& s,
                                bool vendor_available = true) {
    return resolve_route<Op::spmm, T>(forced, s, vendor_available);
}

} // namespace batchlas::dispatch
