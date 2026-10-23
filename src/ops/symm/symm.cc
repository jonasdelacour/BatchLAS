// symm (flat-kernel-selection.md §4.3, R1; §12): select::run takes the first entry of the nearest
// tuned/symm.<device>.txt row that can_run() admits. Expand mirrors A's referenced triangle into a
// dense scratch copy and hands it to the public gemm, which picks its own kernel; Vendor is the per-item
// cublas?symm / cblas_?symm loop.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/functions/symm.hh>
#include <batchlas/util/mempool.hh>

#include <src/ops/level3/batched.hh>
#include <src/ops/symm/choice.hh>
#include <src/backends/level3_shape.hh>
#include <src/kernels/common/triangular_expand.hh>
#include <src/expansion_budget.hh>
#include <src/select/select.hh>
#include <src/util/kernel-trace.hh>
#include <src/util/template-instantiations.hh>

#include <complex>
#include <cstddef>
#include <type_traits>
#include <cstdint>
#include <stdexcept>
#include <variant>

namespace batchlas {
namespace ops::symm {

using select::overloaded;

template <class T>
using MV = MatrixView<T, MatrixFormat::Dense>;

template <Backend B>
inline constexpr bool kExpandWired = select::native_gpu_backend<B> || B == Backend::ROCM;
// ROCm's vendor arm symmetrizes a real copy (MatrixView::symmetrize is real-only).
template <Backend B, class T>
inline constexpr bool kVendorWired =
    select::has_library<B>(spec.vendor) && !(B == Backend::ROCM && !std::is_floating_point_v<T>);

template <class T>
select::Key key_of(const MV<T>& C) {
    return {{"form", form_of(C.rows(), C.cols())}, {"m", C.rows()}, {"n", C.cols()}, {"batch", C.batch_size()}};
}

// Correctness only (R3). The expansion reads A at its storage order and stride, and the gemm it
// feeds rejects a heterogeneous B or C against the homogeneous scratch; every vendor loop
// (cuBLAS, netlib) runs each item at the top-level (m, n). So no family takes a heterogeneous
// operand. The expansion's scratch must fit (expansion_fits: the int-linearised grid and the
// BATCHLAS_EXPAND_MAX_BYTES / quarter-of-memory budget) and grid z holds its batch, so the launch
// chunks the batch and only an item whose own scratch does not fit is refused.
template <Backend B, class T>
bool can_run(const SymmChoice& c, const select::Device& d, Queue& q, const MV<T>& A, const MV<T>& Bm,
             const MV<T>& C) {
    constexpr bool kWired = kExpandWired<B>;
    const bool homogeneous = !A.is_heterogeneous() && !Bm.is_heterogeneous() && !C.is_heterogeneous();
    const bool native = kWired && d.is_gpu && homogeneous && d.max_wg >= kExpandWg && C.rows() >= 1 &&
                        C.cols() >= 1 && A.batch_size() >= 1;
    return std::visit(overloaded{
        [&](Expand) { return native && backend::detail::expansion_chunk<T>(q, A.rows(), 1, d.max_groups_yz) >= 1; },
        [&](Vendor) { return kVendorWired<B, T> && d.has_vendor && homogeneous; },
    }, c);
}

// rocBLAS has no symm wrapper: ROCm's vendor arm symmetrizes a copy of A and calls the public
// gemm, with none of expand's batch or scratch limits. The copy dies here, so the call waits.
template <Backend B, class T>
Event symmetrize_gemm(Queue& q, const MV<T>& A, const MV<T>& Bm, const MV<T>& C, T alpha, T beta, Side side,
                      Uplo uplo) {
    Matrix<T, MatrixFormat::Dense> full(A.rows(), A.cols(), A.batch_size());
    MV<T>::copy(q, full, A).wait();
    full.symmetrize(q, uplo).wait();
    if (side == Side::Left)
        (void)gemm<B, T>(q, full, Bm, C, alpha, beta, Transpose::NoTrans, Transpose::NoTrans,
                         ComputePrecision::Default);
    else
        (void)gemm<B, T>(q, Bm, full, C, alpha, beta, Transpose::NoTrans, Transpose::NoTrans,
                         ComputePrecision::Default);
    q.wait();
    return q.create_event_after_external_work();
}

template <Backend B, class T>
Event launch(Queue& q, const SymmChoice& c, const MV<T>& A, const MV<T>& Bm, const MV<T>& C, T alpha, T beta,
             Side side, Uplo uplo) {
    return std::visit(overloaded{
        [&](Expand) -> Event {
            if constexpr (kExpandWired<B>)
                return level3::expand_gemm<B, T, /*Conjugate=*/false>(q, A, Bm, C, alpha, beta, side, uplo);
            else
                throw std::logic_error("symm: expand is wired for GPU backends only");  // can_run refuses it
        },
        [&](Vendor) -> Event {
            if constexpr (B == Backend::ROCM && kVendorWired<B, T>)
                return symmetrize_gemm<B, T>(q, A, Bm, C, alpha, beta, side, uplo);
            else if constexpr (B != Backend::ROCM && kVendorWired<B, T>)
                return backend::symm_vendor<B, T>(q, A, Bm, C, alpha, beta, side, uplo);
            else select::no_vendor<B, T>(spec);
        },
    }, c);
}

}  // namespace ops::symm

// symm takes no caller workspace (no symm_buffer_size); Expand leases its scratch from the
// queue's arena, so R5 has nothing to size.
template <Backend Back, FloatingOrComplexScalar T>
Event symm(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, const MatrixView<T, MatrixFormat::Dense>& B,
           const MatrixView<T, MatrixFormat::Dense>& C, T alpha, T beta, Side side, Uplo uplo) {
    backend::shape::validate_product<std::invalid_argument>("SYMM", A, B, C, side);
    // An empty problem is a no-op under any pin, as syrk's and trmm's are (the cuBLAS loop faults on one).
    if (A.batch_size() == 0 || C.rows() == 0 || C.cols() == 0) return ctx.create_event_after_external_work();
    // The coverage row's key: C's extents and A's order.
    const coverage::Shape shape{.m = C.rows(), .n = C.cols(), .k = A.rows(), .batch = C.batch_size(), .uplo = uplo,
                                .side = side};
    const select::Key key = ops::symm::key_of<T>(C);
    return select::run<Back, T>(
        ops::symm::spec, ctx, key, ops::symm::candidates<T>(),
        [&](const auto& c, const auto& d) { return ops::symm::can_run<Back, T>(c, d, ctx, A, B, C); }, shape, key,
        [&](const auto& c) { return ops::symm::launch<Back, T>(ctx, c, A, B, C, alpha, beta, side, uplo); });
}

#define SYMM_INSTANTIATE(B_, fp) BATCHLAS_INSTANTIATE_OP(B_, fp, symm)
BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(SYMM_INSTANTIATE)
#undef SYMM_INSTANTIATE

}  // namespace batchlas
