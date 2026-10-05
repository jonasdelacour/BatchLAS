#pragma once

// geqrf's can_run (rule R3), in a header so a host test can call it with a synthetic
// select::Device: no test device here lacks a GPU, sub-group size 32 or one CTA element of SLM.

#include <batchlas/blas/matrix.hh>

#include "choice.hh"
#include "../../select/select.hh"
#include "../../extensions/geqrf_native.hh"

#include <cstddef>
#include <cstdint>
#include <variant>

namespace batchlas::ops::geqrf {

template <class... F>
struct overloaded : F... { using F::operator()...; };
template <class... F>
overloaded(F...) -> overloaded<F...>;

template <class T>
using MV = MatrixView<T, MatrixFormat::Dense>;

// Correctness only (R3): each clause is the argument check at the top of that driver's
// *_dispatch. One launch covers the batch with a single (m, n, ld, stride), and on a wide view
// the trailing update runs off the panel, so no native family takes either.
template <class T>
bool can_run(const GeqrfChoice& c, const select::Device& d, const MV<T>& A) {
    const std::int64_t m = A.rows(), n = A.cols();
    const bool native = d.is_gpu && d.has_sg32 && !A.is_heterogeneous() && m >= n && n >= 1 && A.batch_size() >= 1;
    const auto budget = static_cast<std::size_t>(d.slm_budget);
    return std::visit(overloaded{
        [&](Tiny) { return native && m == n && n <= sycl_geqrf::geqrf_tiny_max_n_for_slm<T>(budget); },
        [&](Cta) { return native && sycl_geqrf::geqrf_cta_fits<T>(static_cast<int>(m), static_cast<int>(n), budget); },
        // Blocked's panel leaf IS Cta's device function: it needs the tier present, not the fit.
        [&](Blocked) {
            return native && sycl_geqrf::geqrf_blocked_available<T>() &&
                   sycl_geqrf::geqrf_cta_max_elems_for_slm<T>(budget) >= 1;
        },
        [&](Vendor) { return d.has_vendor_solver; },
    }, c);
}

}  // namespace batchlas::ops::geqrf
