// SPDX-License-Identifier: MIT
// herk (flat-kernel-selection.md §4.3, §12): select::run takes the first entry of the nearest
// tuned/herk.<device>.txt row that can_run() admits. Gram is syrk's single-tile kernel with the ^H
// conjugating whichever operand carries it (n <= 128); Fold is one gemm over the whole n x n product
// into scratch folded into the uplo triangle; Vendor is the per-item cublas?herk / cblas_?herk loop.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/herk.hh>
#include <batchlas/error.hh>

#include <src/backends/level3_shape.hh>
#include <src/expansion_budget.hh>
#include <src/ops/herk/choice.hh>
#include <src/ops/level3/batched.hh>
#include <src/ops/syrk/kernels/syrk_gram_tiles.hh>
#include <src/ops/syrk/kernels/syrk_triangular_tiles.hh>
#include <src/select/select.hh>
#include <src/util/batch_mask.hh>
#include <src/util/template-instantiations.hh>

#include <complex>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <variant>

namespace batchlas {
namespace ops::herk {

using select::overloaded;

template <typename Scalar>
using MV = MatrixView<Scalar, MatrixFormat::Dense>;

template <typename Scalar>
std::int64_t inner(const MV<Scalar>& A, Transpose transA) {
    return transA == Transpose::NoTrans ? A.cols() : A.rows();
}

template <typename Scalar>
select::Key key_of(const MV<Scalar>& A, const MV<Scalar>& C, Transpose transA) {
    return {{"trans", transA == Transpose::NoTrans ? "N" : "C"}, {"n", C.rows()}, {"k", inner<Scalar>(A, transA)},
            {"batch", C.batch_size()}};
}

// Correctness only (R3). Every family runs each item at the top-level (n, k), so none takes a
// heterogeneous operand. The natives are chunked over the batch (grid z), Fold also over its
// scratch budget, so Fold refuses only an item whose own n x n product does not fit. Gram is
// portable SYCL wired for CUDA, SYCL and a vendor-free NETLIB; Fold runs over the public gemm, so
// ROCm too.
template <Backend B, typename Scalar>
bool can_run(const HerkChoice& c, const select::Device& d, Queue& q, const MV<Scalar>& A, const MV<Scalar>& C,
             Transpose transA) {
    // Gram, Triangular and Fold honour a batch mask; the vendor's per-item loop cannot skip an item.
    const bool masked = batchlas::detail::any_batch_mask(A, C);
    const std::int64_t n = C.rows(), k = inner<Scalar>(A, transA);
    const bool homogeneous = !A.is_heterogeneous() && !C.is_heterogeneous();
    // A GPU, or a vendor-free host (natives_run_on); each arm keeps its compile gate.
    const bool runs = select::natives_run_on(d) && homogeneous && n >= 1 && C.batch_size() >= 1;
    return std::visit(overloaded{
        [&](Gram) {
            return select::native_backend<B> && runs && k >= 1 &&
                   backend::detail::syrk_gram_supported<Scalar>(A, C, transA, /*conjugated=*/true) &&
                   d.max_wg >= gram_threads(n) && d.slm_budget >= gram_slm_bytes<Scalar>(n);
        },
        [&](Triangular) {
            const std::int64_t side = (n + backend::detail::kTriangularTile - 1) / backend::detail::kTriangularTile;
            return select::native_backend<B> && std::is_same_v<Scalar, std::complex<float>> && runs && k >= 1 &&
                   d.max_wg >= kTriangularWg && d.slm_budget >= triangular_slm_bytes<Scalar>() &&
                   side * (side + 1) / 2 <= d.max_groups_yz;
        },
        [&](Fold) {
            return (select::native_backend<B> || B == Backend::ROCM) && runs && d.max_wg >= kFoldWg &&
                   backend::detail::expansion_chunk<Scalar>(q, int(n), 1, d.max_groups_yz) >= 1;
        },
        [&](Vendor) { return B != Backend::ROCM && d.has_vendor && homogeneous && !masked; },
    }, c);
}

template <Backend B, typename Scalar>
Event launch(Queue& q, const HerkChoice& c, const MV<Scalar>& A, const MV<Scalar>& C, float_t<Scalar> alpha,
             float_t<Scalar> beta,
             Uplo uplo, Transpose transA) {
    return std::visit(overloaded{
        [&](Gram) -> Event {
            if constexpr (select::native_backend<B>)
                return level3::for_each_chunk(q, C.batch_size(), level3::grid_chunk<B>(q), [&](int first, int count) {
                    return backend::detail::syrk_gram_tiles<Scalar, /*Conjugate=*/true>(
                        q, level3::batch_range(A, first, count), level3::batch_range(C, first, count), Scalar(alpha),
                        Scalar(beta), uplo, transA);
                });
            else
                throw batchlas::internal_error("herk: gram is wired for native backends only");  // can_run refuses it
        },
        [&](Triangular) -> Event {
            if constexpr (select::native_backend<B> && std::is_same_v<Scalar, std::complex<float>>)
                return level3::for_each_chunk(q, C.batch_size(), level3::grid_chunk<B>(q), [&](int first, int count) {
                    return backend::detail::syrk_triangular_tiles<Scalar, /*Conjugate=*/true>(
                        q, level3::batch_range(A, first, count), level3::batch_range(C, first, count), Scalar(alpha),
                        Scalar(beta), uplo, transA);
                });
            else
                throw batchlas::internal_error("herk: triangular is a complex<float> native kernel");  // can_run refuses
        },
        [&](Fold) -> Event {
            if constexpr (select::native_backend<B> || B == Backend::ROCM)
                return level3::gemm_fold<B, Scalar, /*TwoSided=*/false>(q, A, A, C, Scalar(alpha), beta, uplo, transA);
            else
                throw batchlas::internal_error("herk: fold is wired for native backends and ROCm only");  // can_run refuses it
        },
        [&](Vendor) -> Event {
            if constexpr (B != Backend::ROCM && select::has_library<B>(spec.vendor))
                return backend::herk_vendor<B, Scalar>(q, A, C, alpha, beta, uplo, transA);
            else select::no_vendor<B, Scalar>(spec);
        },
    }, c);
}

}  // namespace ops::herk

// herk takes no caller workspace; Fold leases its scratch from the queue's arena.
template <Backend Back, ComplexScalar T>
Event herk(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, const MatrixView<T, MatrixFormat::Dense>& C,
           float_t<T> alpha, float_t<T> beta, Uplo uplo, Transpose transA) {
    (void)batchlas::detail::call_mask("herk", A, C);
    backend::shape::validate_rank_k<batchlas::invalid_argument>("herk", A, C, transA, /*hermitian=*/true);
    if (C.batch_size() == 0 || C.rows() == 0) return ctx.create_event_after_external_work();
    const coverage::Shape shape{.m = C.rows(), .n = C.rows(), .k = ops::herk::inner<T>(A, transA),
                                .batch = C.batch_size(), .transA = transA, .uplo = uplo};
    const select::Key key = ops::herk::key_of<T>(A, C, transA);
    return select::run<Back, T>(
        ops::herk::spec, ctx, key, ops::herk::candidates<T>(),
        [&](const auto& c, const auto& d) { return ops::herk::can_run<Back, T>(c, d, ctx, A, C, transA); }, shape,
        key, [&](const auto& c) { return ops::herk::launch<Back, T>(ctx, c, A, C, alpha, beta, uplo, transA); });
}

#define HERK_INSTANTIATE(B_, fp) BATCHLAS_INSTANTIATE_OP(B_, fp, herk)
BATCHLAS_IF_CUDA(BATCHLAS_FOR_EACH_COMPLEX_TYPE_1(HERK_INSTANTIATE, Backend::CUDA))
BATCHLAS_IF_SYCL(BATCHLAS_FOR_EACH_DEVICE_COMPLEX_TYPE_1(HERK_INSTANTIATE, Backend::SYCL))
BATCHLAS_IF_ROCM(BATCHLAS_FOR_EACH_COMPLEX_TYPE_1(HERK_INSTANTIATE, Backend::ROCM))
BATCHLAS_IF_HOST(BATCHLAS_FOR_EACH_COMPLEX_TYPE_1(HERK_INSTANTIATE, Backend::NETLIB))
#undef HERK_INSTANTIATE

}  // namespace batchlas
