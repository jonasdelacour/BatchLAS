// SPDX-License-Identifier: MIT
// geqrf (flat-kernel-selection.md §4.3, R1; docs/design/flat-kernel-selection.md#phase-5-geqrf):
// select::run takes the first entry of the nearest tuned/geqrf.<device>.txt row that can_run()
// (can_run.hh) admits. Tiny factors a square n <= 32 in registers, Cta holds the whole panel in local
// memory, and Blocked factors nb-wide panels with Cta's device function and updates the trailing matrix
// with the public gemm, which picks its own kernel.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/functions/geqrf.hh>

#include <src/ops/geqrf/can_run.hh>
#include <src/ops/geqrf/choice.hh>
#include <src/ops/geqrf/geqrf.hh>
#include <src/select/select.hh>
#include <src/ops/geqrf/geqrf_native.hh>
#include <src/util/template-instantiations.hh>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <variant>

namespace batchlas {
namespace ops::geqrf {

template <typename Scalar>
select::Key key_of(const MV<Scalar>& A) {
    const std::int64_t m = A.rows(), n = A.cols();
    return {{"form", m == n ? "sq" : (m > n ? "tall" : "wide")}, {"n", n}, {"aspect", aspect_of(m, n)}};
}

template <Backend B, typename Scalar>
Event launch(Queue& q, const GeqrfChoice& c, const MV<Scalar>& A, Span<Scalar> tau, Span<std::byte> ws) {
    return std::visit(overloaded{
        [&](Tiny) { return ops::geqrf::geqrf_tiny_dispatch<Scalar>(q, A, tau, ws); },
        [&](Cta) { return ops::geqrf::geqrf_cta_dispatch<Scalar>(q, A, tau, ws); },
        [&](Blocked) {
            // The public gemm (table-routed): the trailing updates are TN/NN panel products the
            // native GEMM loses on some shapes, and the gemm table knows which.
            return ops::geqrf::geqrf_blocked_dispatch<Scalar>(
                q, A, tau, ws,
                [](Queue& c2, const MV<Scalar>& ga, const MV<Scalar>& gb, const MV<Scalar>& gc, Scalar galpha,
                   Scalar gbeta, Transpose gta,
                   Transpose gtb, ComputePrecision gp) {
                    return gemm<B, Scalar>(c2, ga, gb, gc, galpha, gbeta, gta, gtb, gp);
                });
        },
        [&](Vendor) -> Event {
            if constexpr (select::has_library<B>(spec.vendor))
                return backend::geqrf_vendor<B, Scalar>(q, A, tau, ws);
            else select::no_vendor<B, Scalar>(spec);
        },
    }, c);
}

// Exactly the chosen family's need (R5). Never dereferences A or tau: band_reduction.cc sizes
// through a null-data view (see geqrf_buffer_size_bound).
template <Backend B, typename Scalar>
std::size_t workspace(Queue& q, const GeqrfChoice& c, const MV<Scalar>& A, Span<Scalar> tau) {
    return std::visit(overloaded{
        [&](Tiny) { return ops::geqrf::geqrf_tiny_buffer_size<Scalar>(q, A); },
        [&](Cta) { return ops::geqrf::geqrf_cta_buffer_size<Scalar>(q, A); },
        [&](Blocked) { return ops::geqrf::geqrf_blocked_buffer_size<Scalar>(q, A); },
        [&](Vendor) -> std::size_t {
            if constexpr (select::has_library<B>(spec.vendor))
                return backend::geqrf_vendor_buffer_size<B, Scalar>(q, A, tau);
            else select::no_vendor<B, Scalar>(spec);
        },
    }, c);
}

}  // namespace ops::geqrf

template <Backend Back, typename Scalar>
Event geqrf(Queue& ctx, const MatrixView<Scalar, MatrixFormat::Dense>& A, Span<Scalar> tau, Span<std::byte> workspace) {
    geqrf_validate_params<Scalar>(A);
    (void)detail::call_mask("geqrf", A);
    // An empty batch is a no-op under any pin, as gemm's is: it never reaches selection.
    if (A.batch_size() == 0) return ctx.create_event_after_external_work();
    // The coverage row's key: m, n and k = min(m, n).
    const coverage::Shape shape{.m = A.rows(), .n = A.cols(), .k = std::min(A.rows(), A.cols()),
                                .batch = A.batch_size()};
    const select::Key key = ops::geqrf::key_of<Scalar>(A);
    return select::run<Back, Scalar>(
        ops::geqrf::spec, ctx, key, ops::geqrf::candidates<Scalar>(),
        [&](const auto& c, const auto& d) { return ops::geqrf::can_run<Back, Scalar>(c, d, A); }, shape, key,
        [&](const auto& c) { return ops::geqrf::launch<Back, Scalar>(ctx, c, A, tau, workspace); });
}

template <Backend Back, typename Scalar>
size_t geqrf_buffer_size(Queue& ctx, const MatrixView<Scalar, MatrixFormat::Dense>& A, Span<Scalar> tau) {
    geqrf_validate_params<Scalar>(A);
    (void)detail::call_mask("geqrf_buffer_size", A);
    if (A.batch_size() == 0) return 0;
    const auto c = select::pick<Back, Scalar>(
        ops::geqrf::spec, ctx, ops::geqrf::key_of<Scalar>(A), ops::geqrf::candidates<Scalar>(),
        [&](const auto& k, const auto& d) { return ops::geqrf::can_run<Back, Scalar>(k, d, A); });
    return ops::geqrf::workspace<Back, Scalar>(ctx, c, A, tau);
}

template <Backend Back, typename Scalar>
std::size_t geqrf_buffer_size_bound(Queue& ctx, const MatrixView<Scalar, MatrixFormat::Dense>& A, Span<Scalar> tau) {
    geqrf_validate_params<Scalar>(A);
    (void)detail::call_mask("geqrf_buffer_size", A);
    if (A.batch_size() == 0) return 0;
    const select::Device& d = select::device_of<Back>(ctx, ops::geqrf::spec.vendor);
    std::size_t need = 0;
    for (const auto& c : ops::geqrf::candidates<Scalar>()) {
        const bool vendor = select::family_of(c) == "vendor";
        if (vendor ? d.has_vendor : d.is_gpu) need = std::max(need, ops::geqrf::workspace<Back,
                                                              Scalar>(ctx, c, A, tau));
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
