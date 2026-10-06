// potrf (flat-kernel-selection.md §4.3, R1): select::run takes the first entry of the nearest
// tuned/potrf.<dtype>.<device>.txt row that can_run() admits. Kernel bodies live in
// src/extensions/potrf_{tiny,cta,lpanel,blocked}.cc.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/functions/potrf.hh>
#include <batchlas/blas/functions/trsm.hh>

#include "choice.hh"
#include "../../select/select.hh"
#include "../../extensions/potrf_native.hh"
#include "../../util/template-instantiations.hh"

#include <complex>
#include <cstddef>
#include <variant>

namespace batchlas {
namespace ops::potrf {

using select::overloaded;

template <class T>
select::Key key_of(const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo) {
    return {{"uplo", uplo == Uplo::Lower ? "L" : "U"}, {"n", A.rows()}, {"batch", A.batch_size()}};
}

// Correctness only (R3): false means the driver would throw or answer wrongly. Each clause
// is the argument check at the top of that driver's *_dispatch.
template <class T>
bool can_run(const PotrfChoice& c, const select::Device& d, const MatrixView<T, MatrixFormat::Dense>& A,
             Uplo uplo) {
    const int64_t n = A.rows();
    const auto budget = static_cast<std::size_t>(d.slm_budget);
    const bool native = d.is_gpu && d.has_sg32 && !A.is_heterogeneous() && n >= 1 && A.batch_size() >= 1;
    return std::visit(overloaded{
        [&](Tiny) {
            return native && n <= sycl_potrf::potrf_tiny_max_n<T>() &&
                   d.max_wg >= sycl_potrf::kPotrfTinyWgSize;
        },
        [&](Cta) { return native && n <= sycl_potrf::potrf_cta_max_n_for_slm<T>(budget); },
        [&](const Lpanel& l) {
            // 0 for a panel width this scalar type does not instantiate.
            return native && uplo == Uplo::Lower &&
                   n <= sycl_potrf::potrf_lpanel_max_n_for_slm<T>(budget, d.max_wg,
                                                                  resident::kMinBlocksPerSm, l.panel);
        },
        [&](Blocked) {
            // Lower-only; its leaf is the CTA kernel, so CTA must hold at least order 1.
            return native && uplo == Uplo::Lower && sycl_potrf::potrf_blocked_available<T>() &&
                   sycl_potrf::potrf_cta_max_n_for_slm<T>(budget) >= 1;
        },
        [&](Vendor) { return d.has_vendor; },
    }, c);
}

template <Backend B, class T>
Event launch(Queue& q, const PotrfChoice& c, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo,
             Span<std::byte> ws, Span<int32_t> info) {
    using MV = MatrixView<T, MatrixFormat::Dense>;
    return std::visit(overloaded{
        [&](Tiny) { return sycl_potrf::potrf_tiny_dispatch<T>(q, A, uplo, ws, info); },
        [&](Cta) { return sycl_potrf::potrf_cta_dispatch<T>(q, A, uplo, ws, info); },
        [&](const Lpanel& l) {
            return sycl_potrf::potrf_lpanel_dispatch<T>(q, A, uplo, ws, info, resident::kMinBlocksPerSm,
                                                        l.panel);
        },
        [&](Blocked) {
            // The public gemm and trsm: each child decides its own kernel.
            return sycl_potrf::potrf_blocked_dispatch<T>(
                q, A, uplo, ws, info,
                [](Queue& c, const MV& a, const MV& b, const MV& r, T alpha, T beta, Transpose ta,
                   Transpose tb, ComputePrecision p) { return gemm<B, T>(c, a, b, r, alpha, beta, ta, tb, p); },
                [](Queue& c, const MV& a, const MV& b, T alpha, Side s, Uplo u, Transpose t, Diag dg) {
                    return trsm<B, T>(c, a, b, alpha, s, u, t, dg);
                });
        },
        [&](Vendor) -> Event {
            if constexpr (select::has_library<B>(spec.vendor))
                return backend::potrf_vendor<B, T>(q, A, uplo, ws, info);
            else select::no_vendor<B, T>(spec);
        },
    }, c);
}

// Exactly the chosen family's need (R5). Pure: no launch, no data read.
template <Backend B, class T>
std::size_t workspace(Queue& q, const PotrfChoice& c, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo) {
    return std::visit(overloaded{
        [&](Tiny) { return sycl_potrf::potrf_tiny_buffer_size<T>(q, A); },
        [&](Cta) { return sycl_potrf::potrf_cta_buffer_size<T>(q, A); },
        [&](const Lpanel&) { return sycl_potrf::potrf_lpanel_buffer_size<T>(q, A); },
        [&](Blocked) { return sycl_potrf::potrf_blocked_buffer_size<T>(q, A, uplo); },
        [&](Vendor) -> std::size_t {
            if constexpr (select::has_library<B>(spec.vendor))
                return backend::potrf_vendor_buffer_size<B, T>(q, A, uplo);
            else select::no_vendor<B, T>(spec);
        },
    }, c);
}

}  // namespace ops::potrf

template <Backend B, typename T>
Event potrf(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo, Span<std::byte> workspace,
            Span<int32_t> info) {
    namespace o = ops::potrf;
    potrf_validate_params<T>(A, uplo);
    // uplo is part of the coverage key; never inferred.
    const coverage::Shape shape{.m = A.rows(), .n = A.rows(), .k = A.rows(), .batch = A.batch_size(), .uplo = uplo};
    return select::run<B, T>(
        o::spec, ctx, o::key_of(A, uplo), o::candidates<T>(),
        [&](const auto& c, const auto& d) { return o::can_run<T>(c, d, A, uplo); }, shape, {},
        [&](const auto& c) { return o::launch<B, T>(ctx, c, A, uplo, workspace, info); });
}

template <Backend B, typename T>
size_t potrf_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo) {
    namespace o = ops::potrf;
    potrf_validate_params<T>(A, uplo);
    const auto c = select::pick<B, T>(o::spec, ctx, o::key_of(A, uplo), o::candidates<T>(),
                                      [&](const auto& k, const auto& d) { return o::can_run<T>(k, d, A, uplo); });
    return o::workspace<B, T>(ctx, c, A, uplo);
}

#define POTRF_INSTANTIATE(B_, fp) \
    BATCHLAS_INSTANTIATE_OP(B_, fp, potrf) BATCHLAS_INSTANTIATE_OP(B_, fp, potrf_buffer_size)
BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(POTRF_INSTANTIATE)
#undef POTRF_INSTANTIATE

}  // namespace batchlas
