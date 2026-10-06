// hemm (flat-kernel-selection.md §4.3, R1; §12 "Hermitian three"): select::run takes the first entry
// of the nearest tuned/hemm.<dtype>.<device>.txt row that can_run() admits. Expand mirrors A's triangle
// (conjugated) into scratch for the public gemm; Vendor is the per-item cublas?hemm / cblas_?hemm loop.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/functions/hemm.hh>
#include <batchlas/util/mempool.hh>

#include "choice.hh"
#include "../../backends/level3_shape.hh"
#include "../../backends/triangular_expand.hh"
#include "../../expansion_budget.hh"
#include "../../select/select.hh"
#include "../../util/kernel-trace.hh"
#include "../../util/template-instantiations.hh"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <variant>

namespace batchlas {
namespace ops::hemm {

using select::overloaded;

template <class T>
using MV = MatrixView<T, MatrixFormat::Dense>;

template <class T>
select::Key key_of(const MV<T>& A, const MV<T>& C, Side side) {
    return {{"order", A.rows()}, {"q", side == Side::Left ? C.cols() : C.rows()}, {"batch", C.batch_size()}};
}

// Correctness only (R3). The expansion reads A at its storage order and stride, and the gemm it
// feeds rejects a heterogeneous B or C against the homogeneous scratch; every vendor loop
// (cuBLAS, netlib) runs each item at the top-level (m, n). So no family takes a heterogeneous
// operand. The expansion's scratch must fit (expansion_fits: the int-linearised grid and the
// BATCHLAS_EXPAND_MAX_BYTES / quarter-of-memory budget), and its batch sits in grid z.
template <Backend B, class T>
bool can_run(const HemmChoice& c, const select::Device& d, Queue& q, const MV<T>& A, const MV<T>& Bm,
             const MV<T>& C) {
    constexpr bool kWired = B == Backend::CUDA;  // the old reach: the host stays vendor
    const std::int64_t k = A.rows(), batch = A.batch_size();
    const bool homogeneous = !A.is_heterogeneous() && !Bm.is_heterogeneous() && !C.is_heterogeneous();
    const bool native = kWired && d.is_gpu && homogeneous && d.max_wg >= kExpandWg && C.rows() >= 1 &&
                        C.cols() >= 1 && batch >= 1 && batch <= kMaxGridBatch;
    return std::visit(overloaded{
        [&](Expand) {
            return native && backend::detail::expansion_fits(
                                 q, int(k), int(batch),
                                 backend::detail::expanded_workspace_bytes<T>(q, int(k), int(batch)));
        },
        [&](Vendor) { return d.has_vendor && homogeneous; },
    }, c);
}

// The scratch is a lease on the queue's arena, not a Matrix: a managed allocation migrates on
// first touch and would be freed while the kernels reading it are only enqueued. The gemm cannot
// read the caller's A: its other triangle and the diagonal's imaginary part are not the operand.
template <Backend B, class T>
Event expand_gemm(Queue& q, const MV<T>& A, const MV<T>& Bm, const MV<T>& C, T alpha, T beta, Side side,
                  Uplo uplo) {
    const int n = A.rows(), batch = A.batch_size();
    const int ld = backend::detail::expanded_ld<T>(n);
    auto ws = q.workspace(backend::detail::expanded_workspace_bytes<T>(q, n, batch));
    BumpAllocator pool(ws.span());
    auto storage = pool.allocate<T>(q, std::size_t(ld) * std::size_t(n) * std::size_t(batch));
    const MV<T> full(storage.data(), n, n, ld, ld * n, batch);
    // BLAS: beta = 0 makes C output only, but the native direct and tiled gemms read it (0 * NaN;
    // known-defects.md #11), and a vendor-free gemm lands on them at small shapes.
    if (beta == T(0)) (void)C.fill(q, T(0));
    {
        BATCHLAS_KERNEL_TRACE_SCOPE("hemm.expand");
        (void)backend::detail::expand_mirrored<T, /*Conjugate=*/true>(q, full, A, uplo);
    }
    // An out-of-order queue orders nothing between the fill, the expansion and a vendor gemm's stream.
    if (!q.in_order()) q.wait();
    if (side == Side::Left)
        return gemm<B, T>(q, full, Bm, C, alpha, beta, Transpose::NoTrans, Transpose::NoTrans,
                          ComputePrecision::Default);
    return gemm<B, T>(q, Bm, full, C, alpha, beta, Transpose::NoTrans, Transpose::NoTrans,
                      ComputePrecision::Default);
}

template <Backend B, class T>
Event launch(Queue& q, const HemmChoice& c, const MV<T>& A, const MV<T>& Bm, const MV<T>& C, T alpha, T beta,
             Side side, Uplo uplo) {
    return std::visit(overloaded{
        [&](Expand) { return expand_gemm<B, T>(q, A, Bm, C, alpha, beta, side, uplo); },
        [&](Vendor) -> Event {
            if constexpr (select::has_library<B>(spec.vendor))
                return backend::hemm_vendor<B, T>(q, A, Bm, C, alpha, beta, side, uplo);
            else select::no_vendor<B, T>(spec);
        },
    }, c);
}

}  // namespace ops::hemm

// hemm takes no caller workspace (no hemm_buffer_size); Expand leases its scratch from the
// queue's arena, so R5 has nothing to size.
template <Backend Back, ComplexScalar T>
Event hemm(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, const MatrixView<T, MatrixFormat::Dense>& B,
           const MatrixView<T, MatrixFormat::Dense>& C, T alpha, T beta, Side side, Uplo uplo) {
    backend::shape::validate_product<std::invalid_argument>("HEMM", A, B, C, side);
    // An empty problem is a no-op under any pin, as symm's is (the cuBLAS loop faults on one).
    if (A.batch_size() == 0 || C.rows() == 0 || C.cols() == 0) return ctx.create_event_after_external_work();
    // The coverage row's key, as symm's: C's extents and A's order.
    const coverage::Shape shape{.m = C.rows(), .n = C.cols(), .k = A.rows(), .batch = C.batch_size(), .uplo = uplo,
                                .side = side};
    const select::Key key = ops::hemm::key_of<T>(A, C, side);
    return select::run<Back, T>(
        ops::hemm::spec, ctx, key, ops::hemm::candidates<T>(),
        [&](const auto& c, const auto& d) { return ops::hemm::can_run<Back, T>(c, d, ctx, A, B, C); }, shape, key,
        [&](const auto& c) { return ops::hemm::launch<Back, T>(ctx, c, A, B, C, alpha, beta, side, uplo); });
}

// Keyed on the device family, not the vendor library: without the library the Vendor arm
// compiles to a throw. rocblas.cc has no hemm wrapper, so ROCm instantiates none (as before).
#define HEMM_INSTANTIATE(B_, fp) BATCHLAS_INSTANTIATE_OP(B_, fp, hemm)
BATCHLAS_IF_CUDA(BATCHLAS_FOR_EACH_COMPLEX_TYPE_1(HEMM_INSTANTIATE, Backend::CUDA))
BATCHLAS_IF_HOST(BATCHLAS_FOR_EACH_COMPLEX_TYPE_1(HEMM_INSTANTIATE, Backend::NETLIB))
#undef HEMM_INSTANTIATE

}  // namespace batchlas
