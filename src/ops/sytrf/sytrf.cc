// SPDX-License-Identifier: MIT
// sytrf (flat-kernel-selection.md §4.3): select::run takes the first entry of the nearest
// tuned/sytrf.<device>.txt row that can_run() admits. sytrs is routed on its own (src/ops/sytrs/).
// evidence: docs/perf/sytrf.md

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/functions/sytrf.hh>

#include <src/ops/sytrf/choice.hh>
#include <src/ops/sytrf/sytrf_native.hh>
#include <src/select/select.hh>
#include <src/util/batch_mask.hh>
#include <src/select/vendor.hh>
#include <src/util/template-instantiations.hh>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <variant>

namespace batchlas {
namespace ops::sytrf {

using select::overloaded;

template <class Scalar>
using MV = MatrixView<Scalar, MatrixFormat::Dense>;

// The vendor arm exists for cuSOLVER and LAPACKE only; rocSOLVER's sytrf is not wired.
template <Backend B>
inline constexpr bool vendor_linked =
    (B == Backend::CUDA && select::kHasCudaVendor) || (B == Backend::NETLIB && select::kHasNetlib);

template <class Scalar>
select::Key key_of(const MV<Scalar>& A) {
    return {{"n", A.rows()}, {"batch", A.batch_size()}};
}

// Correctness only (R3): each clause is the argument check at the top of that driver's *_dispatch.
// The vendor loops run every item at the top-level order, so no family takes a heterogeneous batch.
// Every native family and LAPACKE's host loop honour a batch mask; cuSOLVER's per-item calls refuse it.
template <Backend B, class Scalar>
bool can_run(const SytrfChoice& c, const select::Device& d, const MV<Scalar>& A) {
    const std::int64_t n = A.rows();
    const bool native = B != Backend::NETLIB && d.is_gpu && d.has_sg32 && !A.is_heterogeneous() && n >= 1 &&
                        A.batch_size() >= 1 && d.max_wg >= ops::sytrf::kSytrfCtaWg;
    return std::visit(overloaded{
        [&](Cta) { return native && n <= ops::sytrf::kSytrfCtaMaxN; },
        [&](Wg) { return native; },
        [&](Blocked) { return native; },  // its gemm selects for itself (and honours the mask, W1a)
        [&](Vendor) {
            return vendor_linked<B> && d.has_vendor && !A.is_heterogeneous() &&
                   (B == Backend::NETLIB || !detail::any_batch_mask(A));
        },
    }, c);
}

template <Backend B, class Scalar>
Event launch(Queue& q, const SytrfChoice& c, const MV<Scalar>& A, Uplo uplo, Span<int64_t> pivots, Span<std::byte> ws,
             Span<int32_t> info) {
    return std::visit(overloaded{
        [&](Cta) { return ops::sytrf::sytrf_cta_dispatch<Scalar>(q, A, uplo, pivots, ws, info); },
        [&](Wg) { return ops::sytrf::sytrf_wg_dispatch<Scalar>(q, A, uplo, pivots, ws, info); },
        [&](Blocked) {
            // The public gemm (table-routed): its operands are sub-views carrying the parent's ld.
            return ops::sytrf::sytrf_blocked_dispatch<Scalar>(
                q, A, uplo, pivots, ws, info,
                [](Queue& c2, const MV<Scalar>& ga, const MV<Scalar>& gb, const MV<Scalar>& gc, Scalar galpha,
                   Scalar gbeta, Transpose gta,
                   Transpose gtb, ComputePrecision gp) { return gemm<B, Scalar>(c2, ga, gb, gc, galpha, gbeta, gta, gtb, gp); });
        },
        [&](Vendor) -> Event {
            if constexpr (vendor_linked<B>) return backend::sytrf_vendor<B, Scalar>(q, A, uplo, pivots, ws, info);
            else select::no_vendor<B, Scalar>(spec);
        },
    }, c);
}

template <Backend B, class Scalar>
std::size_t workspace(Queue& q, const SytrfChoice& c, const MV<Scalar>& A, Uplo uplo) {
    return std::visit(overloaded{
        [&](Cta) { return ops::sytrf::sytrf_native_buffer_size<Scalar>(q, A); },
        [&](Wg) { return ops::sytrf::sytrf_native_buffer_size<Scalar>(q, A); },
        [&](Blocked) { return ops::sytrf::sytrf_blocked_buffer_size<Scalar>(q, A); },
        [&](Vendor) -> std::size_t {
            if constexpr (vendor_linked<B>) return backend::sytrf_vendor_buffer_size<B, Scalar>(q, A, uplo);
            else select::no_vendor<B, Scalar>(spec);
        },
    }, c);
}

}  // namespace ops::sytrf

template <Backend B, typename Scalar>
Event sytrf(Queue& ctx, const MatrixView<Scalar, MatrixFormat::Dense>& A, Uplo uplo, Span<int64_t> pivots,
            Span<std::byte> work_space, Span<int32_t> info) {
    sytrf_validate_params<Scalar>(A, uplo);
    (void)detail::call_mask("sytrf", A);
    // An empty problem is a no-op under any pin: it never reaches selection.
    if (A.batch_size() == 0 || A.rows() == 0) return ctx.create_event_after_external_work();
    const coverage::Shape shape{.m = A.rows(), .n = A.rows(), .k = A.rows(), .batch = A.batch_size(), .uplo = uplo};
    const select::Key key = ops::sytrf::key_of<Scalar>(A);
    return select::run<B, Scalar>(
        ops::sytrf::spec, ctx, key, ops::sytrf::candidates<Scalar>(),
        [&](const auto& c, const auto& d) { return ops::sytrf::can_run<B, Scalar>(c, d, A); }, shape, key,
        [&](const auto& c) { return ops::sytrf::launch<B, Scalar>(ctx, c, A, uplo, pivots, work_space, info); });
}

template <Backend B, typename Scalar>
size_t sytrf_buffer_size(Queue& ctx, const MatrixView<Scalar, MatrixFormat::Dense>& A, Uplo uplo) {
    sytrf_validate_params<Scalar>(A, uplo);
    (void)detail::call_mask("sytrf", A);
    if (A.batch_size() == 0 || A.rows() == 0) return 0;
    const auto c = select::pick<B, Scalar>(
        ops::sytrf::spec, ctx, ops::sytrf::key_of<Scalar>(A), ops::sytrf::candidates<Scalar>(),
        [&](const auto& k, const auto& d) { return ops::sytrf::can_run<B, Scalar>(k, d, A); });
    return ops::sytrf::workspace<B, Scalar>(ctx, c, A, uplo);
}

#define SYTRF_INSTANTIATE(B_, fp) \
    BATCHLAS_INSTANTIATE_OP(B_, fp, sytrf) BATCHLAS_INSTANTIATE_OP(B_, fp, sytrf_buffer_size)
BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(SYTRF_INSTANTIATE)
#undef SYTRF_INSTANTIATE

}  // namespace batchlas
