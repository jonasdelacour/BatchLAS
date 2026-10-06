// herk: the whole selection path (docs/design/flat-kernel-selection.md §4.3, rule R1;
// §12 "Hermitian three"). public herk() -> choose() -> std::visit -> launch. The kernel for a
// shape is the first runnable entry of the nearest row in tuned/herk.<dtype>.<device>.txt;
// can_run() below only removes entries that cannot run. Fold runs one strided-batched public gemm
// over the whole n x n product into scratch and folds the referenced triangle into C; Gram is
// syrk's single-tile kernel with the ^H conjugating (n <= 128); Vendor is the per-item loop.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/functions/herk.hh>
#include <batchlas/no_route.hh>
#include <batchlas/util/mempool.hh>
#include "../../select/vendor.hh"

#include "choice.hh"
#include "../../backends/accumulate_hermitian.hh"
#include "../../backends/level3_shape.hh"
#include "../../backends/syrk_gram_tiles.hh"
#include "../../expansion_budget.hh"
#include "../../select/select.hh"
#include "../../util/kernel-trace.hh"
#include "../../util/template-instantiations.hh"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <variant>

namespace batchlas {
namespace ops::herk {

template <class... F>
struct overloaded : F... { using F::operator()...; };
template <class... F>
overloaded(F...) -> overloaded<F...>;

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
// (expansion_fits); gram's one tile must cover C (n <= 128) with its thread count and SLM tile.
// Every vendor loop (cuBLAS, netlib) runs each item at the top-level (n, k), a wrong answer for a
// heterogeneous batch, so no family takes one.
template <Backend B, class T>
bool can_run(const HerkChoice& c, const select::Device& d, Queue& q, const MV<T>& A, const MV<T>& C,
             Transpose transA) {
    const std::int64_t n = C.rows(), k = inner<T>(A, transA), batch = C.batch_size();
    const bool homogeneous = !A.is_heterogeneous() && !C.is_heterogeneous();
    const bool native = B == Backend::CUDA && d.is_gpu && homogeneous && n >= 1 && k >= 1 && batch >= 1 &&
                        batch <= kMaxGridBatch;
    return std::visit(overloaded{
        [&](Fold) {
            return native && d.max_wg >= kFoldWg &&
                   backend::detail::expansion_fits(
                       q, int(n), int(batch), backend::detail::expanded_workspace_bytes<T>(q, int(n), int(batch)));
        },
        [&](Gram) {
            return native && backend::detail::syrk_gram_supported<T>(A, C, transA, /*conjugated=*/true) &&
                   d.max_wg >= gram_threads(n) && d.slm_budget >= gram_slm_bytes<T>(n);
        },
        [&](Vendor) { return d.has_vendor_blas && homogeneous; },
    }, c);
}

template <Backend B, class T>
HerkChoice choose(Queue& q, const MV<T>& A, const MV<T>& C, Transpose transA) {
    const select::Device& d = select::device_of<B>(q);
    auto ok = [&](const HerkChoice& c) { return can_run<B, T>(c, d, q, A, C, transA); };
    try {
        return select::choose("herk", select::dtype_name<T>(), d, key_of<T>(A, C, transA), candidates<T>(), ok,
                              rules);
    } catch (const std::runtime_error&) {
        // Vendor-free burn-down reads coverage `miss` rows; a plain runtime_error records none.
        const auto all = candidates<T>();
        if (!select::level3_vendor_available<B> && std::none_of(all.begin(), all.end(), ok))
            select::throw_no_vendor_route<T>(Op::herk, B, select::kLevel3Library<B>);
        throw;
    }
}

// The coverage row's native flags (§5.6): computed only when coverage records a row.
template <Backend B, class T>
select::NativeFacts native_facts(Queue& q, const MV<T>& A, const MV<T>& C, Transpose transA) {
    if (!coverage::dynamic_enabled()) return {};
    const select::Device& d = select::device_of<B>(q);
    return select::native_facts(candidates<T>(),
                                [&](const HerkChoice& c) { return can_run<B, T>(c, d, q, A, C, transA); });
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
            // Instantiated only where can_run can admit it, so no other backend compiles the kernel.
            if constexpr (B == Backend::CUDA)
                return backend::detail::syrk_gram_tiles<T, /*Conjugate=*/true>(q, A, C, T(alpha), T(beta), uplo,
                                                                              transA);
            else
                throw std::logic_error("herk: gram is wired for CUDA only");
        },
        [&](Vendor) -> Event {
            if constexpr (select::level3_vendor_available<B>)
                return backend::herk_vendor<B, T>(q, A, C, alpha, beta, uplo, transA);
            else
                select::throw_no_vendor_route<T>(Op::herk, B, select::kLevel3Library<B>);
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
    const auto c = ops::herk::choose<Back, T>(ctx, A, C, transA);
    // The coverage row's key, as syrk's: m = n = C's order, k = op(A)'s inner extent.
    auto shape = select::square_shape<Back, T>(C.rows(), C.batch_size());
    shape.k = ops::herk::inner<T>(A, transA);
    shape.uplo = uplo;
    shape.transA = transA;
    const select::Key trace_key = ops::herk::key_of<T>(A, C, transA);
    select::TraceScope trace("herk", c, shape, ops::herk::native_facts<Back, T>(ctx, A, C, transA), trace_key);
    return ops::herk::launch<Back, T>(ctx, c, A, C, alpha, beta, uplo, transA);
}

#define HERK_INSTANTIATE(B_, fp) BATCHLAS_INSTANTIATE(sig::herk<fp>, herk, B_, fp)

// Keyed on the device family, not the vendor library: without the library the Vendor arm
// compiles to a throw. rocblas.cc has no herk wrapper, so ROCm instantiates none (as before).
#if BATCHLAS_HAS_CUDA_BACKEND
HERK_INSTANTIATE(Backend::CUDA, std::complex<float>)
HERK_INSTANTIATE(Backend::CUDA, std::complex<double>)
#endif
#if BATCHLAS_HAS_HOST_BACKEND
HERK_INSTANTIATE(Backend::NETLIB, std::complex<float>)
HERK_INSTANTIATE(Backend::NETLIB, std::complex<double>)
#endif

#undef HERK_INSTANTIATE

}  // namespace batchlas
