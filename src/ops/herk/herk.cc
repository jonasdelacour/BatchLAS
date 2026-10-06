// herk (flat-kernel-selection.md §4.3, R1; §12 "Hermitian three"): select::run takes the first entry
// of the nearest tuned/herk.<dtype>.<device>.txt row that can_run() admits. Fold runs one
// strided-batched public gemm over the whole n x n product into scratch and folds the referenced
// triangle into C; Gram is syrk's single-tile kernel with the ^H conjugating (n <= 128); Vendor is the
// per-item loop.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/functions/herk.hh>
#include <batchlas/util/mempool.hh>

#include "choice.hh"
#include "../../backends/accumulate_hermitian.hh"
#include "../../backends/level3_shape.hh"
#include "../../backends/syrk_gram_tiles.hh"
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
namespace ops::herk {

using select::overloaded;

template <class T>
using MV = MatrixView<T, MatrixFormat::Dense>;

// op(A)'s inner extent: the reduction length of C = op(A) op(A)^H.
template <class T>
std::int64_t inner(const MV<T>& A, Transpose transA) {
    return transA == Transpose::NoTrans ? A.cols() : A.rows();
}

template <class T>
select::Key key_of(const MV<T>& A, const MV<T>& C, Transpose transA) {
    return {{"n", C.rows()}, {"k", inner<T>(A, transA)}, {"batch", C.batch_size()}};
}

// Correctness only (R3); shapes were validated at entry (transA is NoTrans or ConjTrans). Both
// natives take one (n, k, ld, stride) per launch, so no heterogeneous operand, and put the batch
// in a grid dimension capped at 65535. The fold's n x n x batch scratch must fit
// (expansion_fits) and its padded launch range an int (expand_grid_fits); gram's one tile must
// cover C (n <= 128) with its thread count and SLM tile. Every vendor loop (cuBLAS, netlib) runs
// each item at the top-level (n, k), a wrong answer for a heterogeneous batch, so no family takes one.
template <Backend B, class T>
bool can_run(const HerkChoice& c, const select::Device& d, Queue& q, const MV<T>& A, const MV<T>& C,
             Transpose transA) {
    const std::int64_t n = C.rows(), k = inner<T>(A, transA), batch = C.batch_size();
    const bool homogeneous = !A.is_heterogeneous() && !C.is_heterogeneous();
    const bool native = B == Backend::CUDA && d.is_gpu && homogeneous && n >= 1 && k >= 1 && batch >= 1 &&
                        batch <= kMaxGridBatch;
    return std::visit(overloaded{
        [&](Fold) {
            return native && d.max_wg >= kFoldWg && backend::detail::expand_grid_fits(int(n), int(batch)) &&
                   backend::detail::expansion_fits(
                       q, int(n), int(batch), backend::detail::expanded_workspace_bytes<T>(q, int(n), int(batch)));
        },
        [&](Gram) {
            return native && backend::detail::syrk_gram_supported<T>(A, C, transA, /*conjugated=*/true) &&
                   d.max_wg >= gram_threads(n) && d.slm_budget >= gram_slm_bytes<T>(n);
        },
        [&](Vendor) { return d.has_vendor && homogeneous; },
    }, c);
}

// The gemm computes both triangles of the n x n product and keeps one, so it cannot write C. The
// scratch is a lease on the queue's arena (see symm.cc for why not a Matrix).
template <Backend B, class T>
Event fold(Queue& q, const MV<T>& A, const MV<T>& C, float_t<T> alpha, float_t<T> beta, Uplo uplo,
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
    (void)gemm<B, T>(q, A, A, product, T(alpha), T(0), transA,
                     transA == Transpose::NoTrans ? Transpose::ConjTrans : Transpose::NoTrans,
                     ComputePrecision::Default);
    // An out-of-order queue orders nothing between a vendor gemm's stream and the fold.
    if (!q.in_order()) q.wait();
    BATCHLAS_KERNEL_TRACE_SCOPE("herk.fold");
    return backend::detail::accumulate_hermitian<T, /*TwoSided=*/false>(q, C, product, beta, uplo);
}

template <Backend B, class T>
Event launch(Queue& q, const HerkChoice& c, const MV<T>& A, const MV<T>& C, float_t<T> alpha, float_t<T> beta,
             Uplo uplo, Transpose transA) {
    return std::visit(overloaded{
        [&](Fold) { return fold<B, T>(q, A, C, alpha, beta, uplo, transA); },
        [&](Gram) -> Event {
            if constexpr (B == Backend::CUDA)
                return backend::detail::syrk_gram_tiles<T, /*Conjugate=*/true>(q, A, C, T(alpha), T(beta), uplo,
                                                                              transA);
            else
                throw std::logic_error("herk: gram is wired for CUDA only");
        },
        [&](Vendor) -> Event {
            if constexpr (select::has_library<B>(spec.vendor))
                return backend::herk_vendor<B, T>(q, A, C, alpha, beta, uplo, transA);
            else select::no_vendor<B, T>(spec);
        },
    }, c);
}

}  // namespace ops::herk

// herk takes no caller workspace (no herk_buffer_size); Fold leases its scratch from the
// queue's arena, so R5 has nothing to size.
template <Backend Back, ComplexScalar T>
Event herk(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, const MatrixView<T, MatrixFormat::Dense>& C,
           float_t<T> alpha, float_t<T> beta, Uplo uplo, Transpose transA) {
    (void)backend::shape::validate_rank_k<std::invalid_argument>("HERK", A, C, transA, /*hermitian=*/true);
    // An empty problem launches nothing under any pin, as syrk's does.
    if (C.batch_size() == 0 || C.rows() == 0) return ctx.create_event_after_external_work();
    // The coverage row's key, as syrk's: m = n = C's order, k = op(A)'s inner extent.
    const coverage::Shape shape{.m = C.rows(), .n = C.rows(), .k = ops::herk::inner<T>(A, transA),
                                .batch = C.batch_size(), .transA = transA, .uplo = uplo};
    const select::Key key = ops::herk::key_of<T>(A, C, transA);
    return select::run<Back, T>(
        ops::herk::spec, ctx, key, ops::herk::candidates<T>(),
        [&](const auto& c, const auto& d) { return ops::herk::can_run<Back, T>(c, d, ctx, A, C, transA); }, shape,
        key, [&](const auto& c) { return ops::herk::launch<Back, T>(ctx, c, A, C, alpha, beta, uplo, transA); });
}

// Keyed on the device family, not the vendor library: without the library the Vendor arm
// compiles to a throw. rocblas.cc has no herk wrapper, so ROCm instantiates none (as before).
#define HERK_INSTANTIATE(B_, fp) BATCHLAS_INSTANTIATE_OP(B_, fp, herk)
BATCHLAS_IF_CUDA(BATCHLAS_FOR_EACH_COMPLEX_TYPE_1(HERK_INSTANTIATE, Backend::CUDA))
BATCHLAS_IF_HOST(BATCHLAS_FOR_EACH_COMPLEX_TYPE_1(HERK_INSTANTIATE, Backend::NETLIB))
#undef HERK_INSTANTIATE

}  // namespace batchlas
