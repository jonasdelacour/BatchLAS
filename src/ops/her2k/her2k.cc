// her2k (flat-kernel-selection.md §4.3, R1; §12 "Hermitian three"): select::run takes the first entry
// of the nearest tuned/her2k.<dtype>.<device>.txt row that can_run() admits. Fold runs one public gemm
// whose product alpha * op(A) op(B)^H carries both terms (the second is its conjugate transpose) and
// folds it into C's referenced triangle; Vendor is the per-item cublas?her2k / cblas_?her2k loop.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/functions/her2k.hh>
#include <batchlas/util/mempool.hh>

#include "choice.hh"
#include "../../backends/accumulate_hermitian.hh"
#include "../../backends/level3_shape.hh"
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
namespace ops::her2k {

using select::overloaded;

template <class T>
using MV = MatrixView<T, MatrixFormat::Dense>;

template <class T>
std::int64_t inner(const MV<T>& A, Transpose transA) {
    return transA == Transpose::NoTrans ? A.cols() : A.rows();
}

template <class T>
select::Key key_of(const MV<T>& A, const MV<T>& C, Transpose transA) {
    return {{"n", C.rows()}, {"k", inner<T>(A, transA)}, {"batch", C.batch_size()}};
}

// Correctness only (R3); shapes were validated at entry (transA is NoTrans or ConjTrans). The
// fold takes one (n, k, ld, stride) per launch, so no heterogeneous operand, puts the batch in
// grid z (65535), needs its n x n x batch scratch to fit (expansion_fits) and its padded launch
// range an int (expand_grid_fits). Every vendor loop (cuBLAS, netlib) runs each item at the
// top-level (n, k), a wrong answer for a heterogeneous batch, so no family takes one.
template <Backend B, class T>
bool can_run(const Her2kChoice& c, const select::Device& d, Queue& q, const MV<T>& A, const MV<T>& Bm,
             const MV<T>& C, Transpose transA) {
    const std::int64_t n = C.rows(), k = inner<T>(A, transA), batch = C.batch_size();
    const bool homogeneous = !A.is_heterogeneous() && !Bm.is_heterogeneous() && !C.is_heterogeneous();
    return std::visit(overloaded{
        [&](Fold) {
            return B == Backend::CUDA && d.is_gpu && homogeneous && d.max_wg >= kFoldWg && n >= 1 && k >= 1 &&
                   batch >= 1 && batch <= kMaxGridBatch && backend::detail::expand_grid_fits(int(n), int(batch)) &&
                   backend::detail::expansion_fits(
                       q, int(n), int(batch), backend::detail::expanded_workspace_bytes<T>(q, int(n), int(batch)));
        },
        [&](Vendor) { return d.has_vendor && homogeneous; },
    }, c);
}

template <Backend Bk, class T>
bool fold_chosen(Queue& q, const MV<T>& A, const MV<T>& Bm, const MV<T>& C, Transpose transA) {
    const Her2kChoice c = select::pick<Bk, T>(
        spec, q, key_of<T>(A, C, transA), candidates<T>(),
        [&](const auto& k, const auto& d) { return can_run<Bk, T>(k, d, q, A, Bm, C, transA); });
    return std::holds_alternative<Fold>(c);
}

// The gemm writes both triangles of the n x n product, so it cannot write C; the fold reads the
// mirrored element for the second term. The scratch is a lease on the queue's arena.
template <Backend B, class T>
Event fold(Queue& q, const MV<T>& A, const MV<T>& Bm, const MV<T>& C, T alpha, float_t<T> beta, Uplo uplo,
           Transpose transA) {
    const int n = C.rows(), batch = C.batch_size();
    const int ld = backend::detail::expanded_ld<T>(n);
    auto ws = q.workspace(backend::detail::expanded_workspace_bytes<T>(q, n, batch));
    BumpAllocator pool(ws.span());
    auto storage = pool.allocate<T>(q, std::size_t(ld) * std::size_t(n) * std::size_t(batch));
    const MV<T> product(storage.data(), n, n, ld, ld * n, batch);
    // The lease is never zeroed and the native direct/tiled gemms read C at beta = 0
    // (known-defects.md #11): stale NaN in the arena would reach C.
    q->memset(storage.data(), 0, storage.size() * sizeof(T));
    if (!q.in_order()) q.wait();
    (void)gemm<B, T>(q, A, Bm, product, alpha, T(0), transA,
                     transA == Transpose::NoTrans ? Transpose::ConjTrans : Transpose::NoTrans,
                     ComputePrecision::Default);
    // An out-of-order queue orders nothing between a vendor gemm's stream and the fold.
    if (!q.in_order()) q.wait();
    BATCHLAS_KERNEL_TRACE_SCOPE("her2k.fold");
    return backend::detail::accumulate_hermitian<T, /*TwoSided=*/true>(q, C, product, beta, uplo);
}

template <Backend B, class T>
Event launch(Queue& q, const Her2kChoice& c, const MV<T>& A, const MV<T>& Bm, const MV<T>& C, T alpha,
             float_t<T> beta, Uplo uplo, Transpose transA) {
    return std::visit(overloaded{
        [&](Fold) { return fold<B, T>(q, A, Bm, C, alpha, beta, uplo, transA); },
        [&](Vendor) -> Event {
            if constexpr (select::has_library<B>(spec.vendor))
                return backend::her2k_vendor<B, T>(q, A, Bm, C, alpha, beta, uplo, transA);
            else select::no_vendor<B, T>(spec);
        },
    }, c);
}

}  // namespace ops::her2k

// her2k takes no caller workspace (no her2k_buffer_size); Fold leases its scratch from the
// queue's arena, so R5 has nothing to size.
template <Backend Back, ComplexScalar T>
Event her2k(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, const MatrixView<T, MatrixFormat::Dense>& B,
            const MatrixView<T, MatrixFormat::Dense>& C, T alpha, float_t<T> beta, Uplo uplo, Transpose transA) {
    backend::shape::validate_rank_2k<std::invalid_argument>("HER2K", A, B, C, transA, /*hermitian=*/true);
    // An empty problem launches nothing under any pin, as syr2k's does (the cuBLAS loop faults on one).
    if (C.batch_size() == 0 || C.rows() == 0) return ctx.create_event_after_external_work();
    // The coverage key, as syr2k's: m = n = C's order, k = op(A)'s inner extent.
    const coverage::Shape shape{.m = C.rows(), .n = C.rows(), .k = ops::her2k::inner<T>(A, transA),
                                .batch = C.batch_size(), .transA = transA, .uplo = uplo};
    const select::Key key = ops::her2k::key_of<T>(A, C, transA);
    return select::run<Back, T>(
        ops::her2k::spec, ctx, key, ops::her2k::candidates<T>(),
        [&](const auto& c, const auto& d) { return ops::her2k::can_run<Back, T>(c, d, ctx, A, B, C, transA); },
        shape, key,
        [&](const auto& c) { return ops::her2k::launch<Back, T>(ctx, c, A, B, C, alpha, beta, uplo, transA); });
}

// Keyed on the device family, not the vendor library: without the library the Vendor arm
// compiles to a throw. rocblas.cc has no her2k wrapper, so ROCm instantiates none (as before).
#define HER2K_INSTANTIATE(B_, fp)                                                                         \
    BATCHLAS_INSTANTIATE_OP(B_, fp, her2k)                                                                \
    template bool ops::her2k::fold_chosen<B_, BATCHLAS_UNPAREN fp>(                                       \
        Queue&, const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&,                              \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&,                                      \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, Transpose);
BATCHLAS_IF_CUDA(BATCHLAS_FOR_EACH_COMPLEX_TYPE_1(HER2K_INSTANTIATE, Backend::CUDA))
BATCHLAS_IF_HOST(BATCHLAS_FOR_EACH_COMPLEX_TYPE_1(HER2K_INSTANTIATE, Backend::NETLIB))
#undef HER2K_INSTANTIATE

}  // namespace batchlas
