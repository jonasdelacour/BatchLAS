// geqrf (flat-kernel-selection.md §4.3, R1; docs/design/flat-kernel-selection.md#phase-5-geqrf):
// select::run takes the first entry of the nearest tuned/geqrf.<dtype>.<device>.txt row that can_run()
// (can_run.hh) admits. Tiny factors a square n <= 32 in registers, Cta holds the whole panel in local
// memory, and Blocked factors nb-wide panels with Cta's device function and updates the trailing matrix
// with the public gemm, which picks its own kernel.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/functions/geqrf.hh>

#include "can_run.hh"
#include "choice.hh"
#include "geqrf.hh"
#include "../../select/select.hh"
#include "../../extensions/geqrf_native.hh"
#include "../../util/template-instantiations.hh"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <variant>

namespace batchlas {
namespace ops::geqrf {

template <class T>
select::Key key_of(const MV<T>& A) {
    const std::int64_t m = A.rows(), n = A.cols();
    return {{"form", m == n ? "sq" : (m > n ? "tall" : "wide")}, {"n", n}, {"aspect", aspect_of(m, n)}};
}

template <Backend B, class T>
Event launch(Queue& q, const GeqrfChoice& c, const MV<T>& A, Span<T> tau, Span<std::byte> ws) {
    return std::visit(overloaded{
        [&](Tiny) { return sycl_geqrf::geqrf_tiny_dispatch<T>(q, A, tau, ws); },
        [&](Cta) { return sycl_geqrf::geqrf_cta_dispatch<T>(q, A, tau, ws); },
        [&](Blocked) {
            // The public gemm, not gemm_custom: the trailing updates are TN/NN panel products the
            // native GEMM loses on some shapes, and the gemm table knows which.
            return sycl_geqrf::geqrf_blocked_dispatch<T>(
                q, A, tau, ws,
                [](Queue& c2, const MV<T>& ga, const MV<T>& gb, const MV<T>& gc, T galpha, T gbeta, Transpose gta,
                   Transpose gtb, ComputePrecision gp) {
                    return gemm<B, T>(c2, ga, gb, gc, galpha, gbeta, gta, gtb, gp);
                });
        },
        [&](Vendor) -> Event {
            if constexpr (select::has_library<B>(spec.vendor))
                return backend::geqrf_vendor<B, T>(q, A, tau, ws);
            else select::no_vendor<B, T>(spec);
        },
    }, c);
}

// Exactly the chosen family's need (R5). Never dereferences A or tau: band_reduction.cc sizes
// through a null-data view (see geqrf_buffer_size_bound).
template <Backend B, class T>
std::size_t workspace(Queue& q, const GeqrfChoice& c, const MV<T>& A, Span<T> tau) {
    return std::visit(overloaded{
        [&](Tiny) { return sycl_geqrf::geqrf_tiny_buffer_size<T>(q, A); },
        [&](Cta) { return sycl_geqrf::geqrf_cta_buffer_size<T>(q, A); },
        [&](Blocked) { return sycl_geqrf::geqrf_blocked_buffer_size<T>(q, A); },
        [&](Vendor) -> std::size_t {
            if constexpr (select::has_library<B>(spec.vendor))
                return backend::geqrf_vendor_buffer_size<B, T>(q, A, tau);
            else select::no_vendor<B, T>(spec);
        },
    }, c);
}

}  // namespace ops::geqrf

template <Backend Back, typename T>
Event geqrf(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Span<T> tau, Span<std::byte> work_space) {
    namespace o = ops::geqrf;
    geqrf_validate_params<T>(A);
    // The coverage row's key: m, n and k = min(m, n), as the old shape builder set them.
    const coverage::Shape shape{.m = A.rows(), .n = A.cols(), .k = std::min(A.rows(), A.cols()),
                                .batch = A.batch_size()};
    const select::Key key = o::key_of<T>(A);
    return select::run<Back, T>(
        o::spec, ctx, key, o::candidates<T>(), [&](const auto& c, const auto& d) { return o::can_run<T>(c, d, A); },
        shape, key, [&](const auto& c) { return o::launch<Back, T>(ctx, c, A, tau, work_space); });
}

template <Backend Back, typename T>
size_t geqrf_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Span<T> tau) {
    namespace o = ops::geqrf;
    geqrf_validate_params<T>(A);
    const auto c = select::pick<Back, T>(o::spec, ctx, o::key_of<T>(A), o::candidates<T>(),
                                         [&](const auto& k, const auto& d) { return o::can_run<T>(k, d, A); });
    return o::workspace<Back, T>(ctx, c, A, tau);
}

template <Backend Back, typename T>
std::size_t geqrf_buffer_size_bound(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Span<T> tau) {
    geqrf_validate_params<T>(A);
    const select::Device& d = select::device_of<Back>(ctx, ops::geqrf::spec.vendor);
    std::size_t need = 0;
    for (const auto& c : ops::geqrf::candidates<T>()) {
        const bool vendor = select::family_of(c) == "vendor";
        if (vendor ? d.has_vendor : d.is_gpu) need = std::max(need, ops::geqrf::workspace<Back, T>(ctx, c, A, tau));
    }
    return need;
}

#define GEQRF_INSTANTIATE(B_, fp)                                                                              \
    BATCHLAS_INSTANTIATE_OP(B_, fp, geqrf) BATCHLAS_INSTANTIATE_OP(B_, fp, geqrf_buffer_size)                \
    template std::size_t geqrf_buffer_size_bound<B_, BATCHLAS_UNPAREN fp>(                                     \
        Queue&, const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, Span<BATCHLAS_UNPAREN fp>);
BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(GEQRF_INSTANTIATE)
#undef GEQRF_INSTANTIATE

}  // namespace batchlas
