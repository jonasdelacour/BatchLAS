// SPDX-License-Identifier: MIT
// herk (flat-kernel-selection.md §4.3, §12): select::run takes the first entry of the nearest
// tuned/herk.<device>.txt row that can_run() admits. Gram is syrk's single-tile kernel with the ^H
// conjugating whichever operand carries it (n <= 128); Fold is one gemm over the whole n x n product
// into scratch folded into the uplo triangle; Vendor is the per-item cublas?herk / cblas_?herk loop.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/herk.hh>

#include <src/backends/level3_shape.hh>
#include <src/expansion_budget.hh>
#include <src/ops/herk/choice.hh>
#include <src/ops/level3/batched.hh>
#include <src/ops/syrk/kernels/syrk_gram_tiles.hh>
#include <src/ops/syrk/kernels/syrk_triangular_tiles.hh>
#include <src/select/select.hh>
#include <src/util/template-instantiations.hh>

#include <complex>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <variant>

namespace batchlas {
namespace ops::herk {

using select::overloaded;

template <typename T>
using MV = MatrixView<T, MatrixFormat::Dense>;

template <typename T>
std::int64_t inner(const MV<T>& A, Transpose transA) {
    return transA == Transpose::NoTrans ? A.cols() : A.rows();
}

template <typename T>
select::Key key_of(const MV<T>& A, const MV<T>& C, Transpose transA) {
    return {{"trans", transA == Transpose::NoTrans ? "N" : "C"}, {"n", C.rows()}, {"k", inner<T>(A, transA)},
            {"batch", C.batch_size()}};
}

// Correctness only (R3). Every family runs each item at the top-level (n, k), so none takes a
// heterogeneous operand. The natives are chunked over the batch (grid z), Fold also over its
// scratch budget, so Fold refuses only an item whose own n x n product does not fit. Gram is
// portable SYCL wired for CUDA and SYCL; Fold runs over the public gemm, so ROCm too.
template <Backend B, typename T>
bool can_run(const HerkChoice& c, const select::Device& d, Queue& q, const MV<T>& A, const MV<T>& C,
             Transpose transA) {
    const std::int64_t n = C.rows(), k = inner<T>(A, transA);
    const bool homogeneous = !A.is_heterogeneous() && !C.is_heterogeneous();
    const bool gpu = d.is_gpu && homogeneous && n >= 1 && C.batch_size() >= 1;
    return std::visit(overloaded{
        [&](Gram) {
            return select::native_gpu_backend<B> && gpu && k >= 1 &&
                   backend::detail::syrk_gram_supported<T>(A, C, transA, /*conjugated=*/true) &&
                   d.max_wg >= gram_threads(n) && d.slm_budget >= gram_slm_bytes<T>(n);
        },
        [&](Triangular) {
            const std::int64_t side = (n + backend::detail::kTriangularTile - 1) / backend::detail::kTriangularTile;
            return select::native_gpu_backend<B> && std::is_same_v<T, std::complex<float>> && gpu && k >= 1 &&
                   d.max_wg >= kTriangularWg && d.slm_budget >= triangular_slm_bytes<T>() &&
                   side * (side + 1) / 2 <= d.max_groups_yz;
        },
        [&](Fold) {
            return (select::native_gpu_backend<B> || B == Backend::ROCM) && gpu && d.max_wg >= kFoldWg &&
                   backend::detail::expansion_chunk<T>(q, int(n), 1, d.max_groups_yz) >= 1;
        },
        [&](Vendor) { return B != Backend::ROCM && d.has_vendor && homogeneous; },
    }, c);
}

template <Backend B, typename T>
Event launch(Queue& q, const HerkChoice& c, const MV<T>& A, const MV<T>& C, float_t<T> alpha, float_t<T> beta,
             Uplo uplo, Transpose transA) {
    return std::visit(overloaded{
        [&](Gram) -> Event {
            if constexpr (select::native_gpu_backend<B>)
                return level3::for_each_chunk(q, C.batch_size(), level3::grid_chunk<B>(q), [&](int first, int count) {
                    return backend::detail::syrk_gram_tiles<T, /*Conjugate=*/true>(
                        q, level3::batch_range(A, first, count), level3::batch_range(C, first, count), T(alpha),
                        T(beta), uplo, transA);
                });
            else
                throw std::logic_error("herk: gram is wired for native GPU backends only");  // can_run refuses it
        },
        [&](Triangular) -> Event {
            if constexpr (select::native_gpu_backend<B> && std::is_same_v<T, std::complex<float>>)
                return level3::for_each_chunk(q, C.batch_size(), level3::grid_chunk<B>(q), [&](int first, int count) {
                    return backend::detail::syrk_triangular_tiles<T, /*Conjugate=*/true>(
                        q, level3::batch_range(A, first, count), level3::batch_range(C, first, count), T(alpha),
                        T(beta), uplo, transA);
                });
            else
                throw std::logic_error("herk: triangular is a complex<float> native-GPU kernel");  // can_run refuses
        },
        [&](Fold) -> Event {
            if constexpr (select::native_gpu_backend<B> || B == Backend::ROCM)
                return level3::gemm_fold<B, T, /*TwoSided=*/false>(q, A, A, C, T(alpha), beta, uplo, transA);
            else
                throw std::logic_error("herk: fold is wired for GPU backends only");  // can_run refuses it
        },
        [&](Vendor) -> Event {
            if constexpr (B != Backend::ROCM && select::has_library<B>(spec.vendor))
                return backend::herk_vendor<B, T>(q, A, C, alpha, beta, uplo, transA);
            else select::no_vendor<B, T>(spec);
        },
    }, c);
}

}  // namespace ops::herk

// herk takes no caller workspace; Fold leases its scratch from the queue's arena.
template <Backend Back, ComplexScalar T>
Event herk(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, const MatrixView<T, MatrixFormat::Dense>& C,
           float_t<T> alpha, float_t<T> beta, Uplo uplo, Transpose transA) {
    backend::shape::validate_rank_k<std::invalid_argument>("HERK", A, C, transA, /*hermitian=*/true);
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
