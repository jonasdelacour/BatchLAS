#pragma once

// SPMM's routing table: the route order, the correctness gates and the measured
// preferred() window for sparse CSR times dense. Pure -- it reads only its
// arguments; anything that must ask the device, the environment or a kernel
// translation unit lives in src/backends/spmm_route.hh.
// evidence: docs/perf/spmm.md
//
// Pinning BATCHLAS_SPMM_ROUTE to a value supports() rejects, or misspelling it,
// does not fail or warn -- it falls through to automatic() and silently measures
// the vendor. Prove the arm from BATCHLAS_COVERAGE_OUT's `reached` rows.
// evidence: docs/perf/dispatch.md#the-environment-vocabulary

#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/dispatch/route_resolve.hh>
#include <batchlas/blas/enums.hh>

#include <complex>
#include <type_traits>

namespace batchlas::dispatch {

// Never shadow an OpShape field here: resolve_route slices this struct to
// OpShape for the coverage table, so a shadowing member is silently not copied.
struct SpmmShape : OpShape {
    MatrixFormat format = MatrixFormat::Dense;

    // False means "no such native body in this build", which makes the native
    // route unsupported rather than selectable-but-unimplemented.
    bool gather_available = false;   // the transA == NoTrans body
    bool scatter_available = false;  // the transA != NoTrans bodies (scale + scatter)

    // Which of m and k is the output extent swaps with transA, so a predicate
    // must spell these and never m or k directly.
    int64_t nrhs() const { return n; }
    int64_t out_rows() const { return transA == Transpose::NoTrans ? m : k; }
    int64_t red_rows() const { return transA == Transpose::NoTrans ? k : m; }
};

// No nnz field, deliberately: MatrixView<T, CSR>::nnz() is the per-item capacity
// (the batch maximum), and the honest per-item nnz(b) reads device memory -- the
// same shape builder runs inside spmm_buffer_size, where that is a segfault.

// {Native, Direct} names three bodies in src/sycl/spmm_native.cc; the launcher
// picks gather vs scale+scatter on transA. That is a decomposition, not an
// algorithm, so the Algorithm enum needs no new name.
inline constexpr Route kSpmmOrder[] = {
    {Origin::Native, Algorithm::Direct},
    {Origin::Vendor, Algorithm::Auto},
};

template <typename T>
struct RouteTable<Op::spmm, T> {
    // Correctness only, never a speed cutoff: a shape or speed gate here makes a
    // forced route silently measure a different arm and drops the row from the
    // vendor-free walk entirely.
    // evidence: docs/perf/spmm.md#supports-and-what-is-deliberately-not-in-it
    static bool supports(Route r, const SpmmShape& s) {
        if (is_vendor(r)) return true;   // the vendor serves everything it is given
        if (!is_native(r)) return false;

        if (s.format != MatrixFormat::CSR) return false;

        // One launch covers the batch with a single (ld, stride) tuple per dense
        // operand, so per-item B/C extents would be read at the wrong addresses.
        if (s.heterogeneous_batch) return false;

        // m == 0 and n == 0 are absent on purpose: the launcher quick-returns on
        // those. Only a negative extent or an empty batch has no launch geometry.
        if (s.m < 0 || s.n < 0 || s.k < 0 || s.batch < 1) return false;

        switch (r.algo) {
            case Algorithm::Direct:
                // Deliberately no is_gpu gate: every body is a plain loop with no
                // local memory and no group collective, and gating it would strand
                // the Backend::NETLIB rows the vendor-free burn-down needs.
                return (s.transA == Transpose::NoTrans) ? s.gather_available
                                                        : s.scatter_available;

            default:
                // Including Auto: a bare "native" names no body.
                return false;
        }
    }

    // The window: the native CSR gather at every extent and batch, minus
    // complex<float> with a transposed dense operand. This moves the default in
    // the vendor-present build too, not only in the vendor-free one.
    // evidence: docs/perf/spmm.md#the-preferred-window-as-implemented
    static bool preferred(Route r, const SpmmShape& s) {
        if (!is_native(r) || r.algo != Algorithm::Direct) return false;
        if (s.format != MatrixFormat::CSR) return false;

        // The scatter arm was measured on the same grid and loses; no shippable
        // transposed window exists.
        // evidence: docs/perf/spmm.md#the-transposed-refusal
        if (s.transA != Transpose::NoTrans) return false;

        // evidence: docs/perf/spmm.md#the-cfloat-transb-exclusion
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

template <typename T>
inline Route resolve_spmm_route(Route forced, const SpmmShape& s,
                                bool vendor_available = true) {
    return resolve_route<Op::spmm, T>(forced, s, vendor_available);
}

} // namespace batchlas::dispatch
