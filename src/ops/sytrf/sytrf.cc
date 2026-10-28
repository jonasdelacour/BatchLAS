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

template <class T>
using MV = MatrixView<T, MatrixFormat::Dense>;

// The vendor arm exists for cuSOLVER and LAPACKE only; rocSOLVER's sytrf is not wired.
template <Backend B>
inline constexpr bool vendor_linked =
    (B == Backend::CUDA && select::kHasCudaVendor) || (B == Backend::NETLIB && select::kHasNetlib);

template <class T>
select::Key key_of(const MV<T>& A) {
    return {{"n", A.rows()}, {"batch", A.batch_size()}};
}

// Correctness only (R3): each clause is the argument check at the top of that driver's *_dispatch.
// The vendor loops run every item at the top-level order, so no family takes a heterogeneous batch.
template <Backend B, class T>
bool can_run(const SytrfChoice& c, const select::Device& d, const MV<T>& A) {
    const std::int64_t n = A.rows();
    const bool native = B != Backend::NETLIB && d.is_gpu && d.has_sg32 && !A.is_heterogeneous() && !detail::any_batch_mask(A) && n >= 1 &&
                        A.batch_size() >= 1 && d.max_wg >= sycl_sytrf::kSytrfCtaWg;
    return std::visit(overloaded{
        [&](Cta) { return native && n <= sycl_sytrf::kSytrfCtaMaxN; },
        [&](Wg) { return native; },
        [&](Blocked) { return native; },  // its gemm selects for itself; the public gemm has a last resort
        [&](Vendor) { return vendor_linked<B> && d.has_vendor && !A.is_heterogeneous() && !detail::any_batch_mask(A); },
    }, c);
}

template <Backend B, class T>
Event launch(Queue& q, const SytrfChoice& c, const MV<T>& A, Uplo uplo, Span<int64_t> pivots, Span<std::byte> ws,
             Span<int32_t> info) {
    return std::visit(overloaded{
        [&](Cta) { return sycl_sytrf::sytrf_cta_dispatch<T>(q, A, uplo, pivots, ws, info); },
        [&](Wg) { return sycl_sytrf::sytrf_wg_dispatch<T>(q, A, uplo, pivots, ws, info); },
        [&](Blocked) {
            // The public gemm (table-routed): its operands are sub-views carrying the parent's ld.
            return sycl_sytrf::sytrf_blocked_dispatch<T>(
                q, A, uplo, pivots, ws, info,
                [](Queue& c2, const MV<T>& ga, const MV<T>& gb, const MV<T>& gc, T galpha, T gbeta, Transpose gta,
                   Transpose gtb, ComputePrecision gp) { return gemm<B, T>(c2, ga, gb, gc, galpha, gbeta, gta, gtb, gp); });
        },
        [&](Vendor) -> Event {
            if constexpr (vendor_linked<B>) return backend::sytrf_vendor<B, T>(q, A, uplo, pivots, ws, info);
            else select::no_vendor<B, T>(spec);
        },
    }, c);
}

template <Backend B, class T>
std::size_t workspace(Queue& q, const SytrfChoice& c, const MV<T>& A, Uplo uplo) {
    return std::visit(overloaded{
        [&](Cta) { return sycl_sytrf::sytrf_native_buffer_size<T>(q, A); },
        [&](Wg) { return sycl_sytrf::sytrf_native_buffer_size<T>(q, A); },
        [&](Blocked) { return sycl_sytrf::sytrf_blocked_buffer_size<T>(q, A); },
        [&](Vendor) -> std::size_t {
            if constexpr (vendor_linked<B>) return backend::sytrf_vendor_buffer_size<B, T>(q, A, uplo);
            else select::no_vendor<B, T>(spec);
        },
    }, c);
}

}  // namespace ops::sytrf

template <Backend B, typename T>
Event sytrf(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo, Span<int64_t> pivots,
            Span<std::byte> work_space, Span<int32_t> info) {
    sytrf_validate_params<T>(A, uplo);
    (void)detail::call_mask("sytrf", A);
    // An empty problem is a no-op under any pin: it never reaches selection.
    if (A.batch_size() == 0 || A.rows() == 0) return ctx.create_event_after_external_work();
    const coverage::Shape shape{.m = A.rows(), .n = A.rows(), .k = A.rows(), .batch = A.batch_size(), .uplo = uplo};
    const select::Key key = ops::sytrf::key_of<T>(A);
    return select::run<B, T>(
        ops::sytrf::spec, ctx, key, ops::sytrf::candidates<T>(),
        [&](const auto& c, const auto& d) { return ops::sytrf::can_run<B, T>(c, d, A); }, shape, key,
        [&](const auto& c) { return ops::sytrf::launch<B, T>(ctx, c, A, uplo, pivots, work_space, info); });
}

template <Backend B, typename T>
size_t sytrf_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo) {
    sytrf_validate_params<T>(A, uplo);
    (void)detail::call_mask("sytrf", A);
    if (A.batch_size() == 0 || A.rows() == 0) return 0;
    const auto c = select::pick<B, T>(
        ops::sytrf::spec, ctx, ops::sytrf::key_of<T>(A), ops::sytrf::candidates<T>(),
        [&](const auto& k, const auto& d) { return ops::sytrf::can_run<B, T>(k, d, A); });
    return ops::sytrf::workspace<B, T>(ctx, c, A, uplo);
}

#define SYTRF_INSTANTIATE(B_, fp) \
    BATCHLAS_INSTANTIATE_OP(B_, fp, sytrf) BATCHLAS_INSTANTIATE_OP(B_, fp, sytrf_buffer_size)
BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(SYTRF_INSTANTIATE)
#undef SYTRF_INSTANTIATE

}  // namespace batchlas
