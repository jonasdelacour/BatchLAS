// SPDX-License-Identifier: MIT
// hemm (flat-kernel-selection.md §4.3, §12): select::run takes the first entry of the nearest
// tuned/hemm.<device>.txt row that can_run() admits. Mirror is symm's (src/ops/symm/kernels/symm_mirror.cc)
// with the mirrored element conjugated and the diagonal real. Expand writes the Hermitian matrix that A's
// referenced triangle stands for into scratch and hands it to the public gemm (one strided-batched
// call per chunk); Vendor is the per-item cublas?hemm / cblas_?hemm loop.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/hemm.hh>
#include <batchlas/error.hh>

#include <src/backends/level3_shape.hh>
#include <src/expansion_budget.hh>
#include <src/ops/hemm/choice.hh>
#include <src/ops/symm/kernels/symm_mirror.hh>
#include <src/ops/level3/batched.hh>
#include <src/select/select.hh>
#include <src/util/batch_mask.hh>
#include <src/util/template-instantiations.hh>

#include <complex>
#include <stdexcept>
#include <variant>

namespace batchlas {
namespace ops::hemm {

using select::overloaded;

template <typename T>
using MV = MatrixView<T, MatrixFormat::Dense>;

template <typename T>
select::Key key_of(const MV<T>& C) {
    return {{"m", C.rows()}, {"n", C.cols()}, {"batch", C.batch_size()}};
}

// Correctness only (R3). The expansion reads A at one order and stride, and the gemm it feeds
// rejects a heterogeneous B or C against the homogeneous scratch; every vendor loop runs each item
// at the top-level extents. So no family takes a heterogeneous operand. Expand is chunked over the
// batch (grid z, scratch budget), so it refuses only an item whose own scratch does not fit.
// rocBLAS has no ?hemm wrapper, so on ROCm only Mirror and Expand (over rocBLAS gemm) run. Mirror's
// terms are symm's: its launch, chunked over the batch.
template <Backend B, typename T>
bool can_run(const HemmChoice& c, const select::Device& d, Queue& q, const MV<T>& A, const MV<T>& Bm,
             const MV<T>& C) {
    // A masked call is refused by every family until its kernels honour the mask.
    if (batchlas::detail::any_batch_mask(A, Bm, C)) return false;
    constexpr bool kWired = select::native_gpu_backend<B> || B == Backend::ROCM;
    const bool homogeneous = !A.is_heterogeneous() && !Bm.is_heterogeneous() && !C.is_heterogeneous();
    return std::visit(overloaded{
        [&](Mirror) {
            return kWired && d.is_gpu && homogeneous && d.max_wg >= mirror::kWg && C.rows() >= 1 && C.cols() >= 1 &&
                   A.batch_size() >= 1 && d.slm_budget_for(mirror::kWg) >= mirror::slm_bytes<T>() &&
                   (C.rows() + mirror::tile<T>() - 1) / mirror::tile<T>() <= d.max_groups_yz;
        },
        [&](Expand) {
            return kWired && d.is_gpu && homogeneous && d.max_wg >= kExpandWg && C.rows() >= 1 && C.cols() >= 1 &&
                   A.batch_size() >= 1 &&
                   backend::detail::expansion_chunk<T>(q, A.rows(), 1, d.max_groups_yz) >= 1;
        },
        [&](Vendor) { return B != Backend::ROCM && d.has_vendor && homogeneous; },
    }, c);
}

template <Backend B, typename T>
Event launch(Queue& q, const HemmChoice& c, const MV<T>& A, const MV<T>& Bm, const MV<T>& C, T alpha, T beta,
             Side side, Uplo uplo) {
    return std::visit(overloaded{
        [&](Mirror) -> Event {
            if constexpr (select::native_gpu_backend<B> || B == Backend::ROCM)
                return sycl_symm::symm_mirror<T, /*Herm=*/true>(q, A, Bm, C, alpha, beta, side, uplo,
                                                                level3::grid_chunk<B>(q));
            else
                throw batchlas::internal_error("hemm: mirror is wired for GPU backends only");  // can_run refuses it
        },
        [&](Expand) -> Event {
            if constexpr (select::native_gpu_backend<B> || B == Backend::ROCM)
                return level3::expand_gemm<B, T, /*Conjugate=*/true>(q, A, Bm, C, alpha, beta, side, uplo);
            else
                throw batchlas::internal_error("hemm: expand is wired for GPU backends only");  // can_run refuses it
        },
        [&](Vendor) -> Event {
            if constexpr (B != Backend::ROCM && select::has_library<B>(spec.vendor))
                return backend::hemm_vendor<B, T>(q, A, Bm, C, alpha, beta, side, uplo);
            else select::no_vendor<B, T>(spec);
        },
    }, c);
}

}  // namespace ops::hemm

// hemm takes no caller workspace; Expand leases its scratch from the queue's arena.
template <Backend Back, ComplexScalar T>
Event hemm(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, const MatrixView<T, MatrixFormat::Dense>& B,
           const MatrixView<T, MatrixFormat::Dense>& C, T alpha, T beta, Side side, Uplo uplo) {
    (void)batchlas::detail::call_mask("hemm", A, B, C);
    backend::shape::validate_product<batchlas::invalid_argument>("hemm", A, B, C, side);
    // An empty problem is a no-op in every build and under any pin, as symm's is.
    if (A.batch_size() == 0 || C.rows() == 0 || C.cols() == 0) return ctx.create_event_after_external_work();
    const coverage::Shape shape{.m = C.rows(), .n = C.cols(), .k = A.rows(), .batch = C.batch_size(), .uplo = uplo,
                                .side = side};
    const select::Key key = ops::hemm::key_of<T>(C);
    return select::run<Back, T>(
        ops::hemm::spec, ctx, key, ops::hemm::candidates<T>(),
        [&](const auto& c, const auto& d) { return ops::hemm::can_run<Back, T>(c, d, ctx, A, B, C); }, shape, key,
        [&](const auto& c) { return ops::hemm::launch<Back, T>(ctx, c, A, B, C, alpha, beta, side, uplo); });
}

#define HEMM_INSTANTIATE(B_, fp) BATCHLAS_INSTANTIATE_OP(B_, fp, hemm)
BATCHLAS_IF_CUDA(BATCHLAS_FOR_EACH_COMPLEX_TYPE_1(HEMM_INSTANTIATE, Backend::CUDA))
BATCHLAS_IF_SYCL(BATCHLAS_FOR_EACH_DEVICE_COMPLEX_TYPE_1(HEMM_INSTANTIATE, Backend::SYCL))
BATCHLAS_IF_ROCM(BATCHLAS_FOR_EACH_COMPLEX_TYPE_1(HEMM_INSTANTIATE, Backend::ROCM))
BATCHLAS_IF_HOST(BATCHLAS_FOR_EACH_COMPLEX_TYPE_1(HEMM_INSTANTIATE, Backend::NETLIB))
#undef HEMM_INSTANTIATE

}  // namespace batchlas
