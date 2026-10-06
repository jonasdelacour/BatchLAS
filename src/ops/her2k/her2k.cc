// her2k: the whole selection path (docs/design/flat-kernel-selection.md §4.3, rule R1;
// §12 "Hermitian three"). public her2k() -> choose() -> std::visit -> launch. The kernel for a
// shape is the first runnable entry of the nearest row in tuned/her2k.<dtype>.<device>.txt;
// can_run() below only removes entries that cannot run. Fold runs one public gemm whose product
// alpha * op(A) op(B)^H carries both terms (the second is its conjugate transpose) and folds it
// into C's referenced triangle; Vendor is the per-item cublas?her2k / cblas_?her2k loop.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/functions/her2k.hh>
#include <batchlas/no_route.hh>
#include <batchlas/util/mempool.hh>
#include "../../select/vendor.hh"

#include "choice.hh"
#include "../../backends/accumulate_hermitian.hh"
#include "../../backends/level3_shape.hh"
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
namespace ops::her2k {

template <class... F>
struct overloaded : F... { using F::operator()...; };
template <class... F>
overloaded(F...) -> overloaded<F...>;

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
        [&](Vendor) { return d.has_vendor_blas && homogeneous; },
    }, c);
}

template <Backend B, class T>
Her2kChoice choose(Queue& q, const MV<T>& A, const MV<T>& Bm, const MV<T>& C, Transpose transA) {
    const select::Device& d = select::device_of<B>(q);
    auto ok = [&](const Her2kChoice& c) { return can_run<B, T>(c, d, q, A, Bm, C, transA); };
    try {
        return select::choose("her2k", select::dtype_name<T>(), d, key_of<T>(A, C, transA), candidates<T>(), ok,
                              rules);
    } catch (const std::runtime_error&) {
        // Vendor-free burn-down reads coverage `miss` rows; a plain runtime_error records none.
        const auto all = candidates<T>();
        if (!select::level3_vendor_available<B> && std::none_of(all.begin(), all.end(), ok))
            select::throw_no_vendor_route<T>(Op::her2k, B, select::kLevel3Library<B>);
        throw;
    }
}

template <Backend Bk, class T>
bool fold_chosen(Queue& q, const MV<T>& A, const MV<T>& Bm, const MV<T>& C, Transpose transA) {
    return std::holds_alternative<Fold>(choose<Bk, T>(q, A, Bm, C, transA));
}

// The coverage row's native flags (§5.6): computed only when coverage records a row.
template <Backend B, class T>
select::NativeFacts native_facts(Queue& q, const MV<T>& A, const MV<T>& Bm, const MV<T>& C, Transpose transA) {
    if (!coverage::dynamic_enabled()) return {};
    const select::Device& d = select::device_of<B>(q);
    return select::native_facts(candidates<T>(),
                                [&](const Her2kChoice& c) { return can_run<B, T>(c, d, q, A, Bm, C, transA); });
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
            if constexpr (select::level3_vendor_available<B>)
                return backend::her2k_vendor<B, T>(q, A, Bm, C, alpha, beta, uplo, transA);
            else
                select::throw_no_vendor_route<T>(Op::her2k, B, select::kLevel3Library<B>);
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
    const auto c = ops::her2k::choose<Back, T>(ctx, A, B, C, transA);
    // The coverage key, as syr2k's: m = n = C's order, k = op(A)'s inner extent.
    auto shape = select::square_shape<Back, T>(C.rows(), C.batch_size());
    shape.k = ops::her2k::inner<T>(A, transA);
    shape.uplo = uplo;
    shape.side = Side::Left;
    shape.diag = Diag::NonUnit;
    shape.transA = transA;
    const select::Key trace_key = ops::her2k::key_of<T>(A, C, transA);
    select::TraceScope trace("her2k", c, shape, ops::her2k::native_facts<Back, T>(ctx, A, B, C, transA), trace_key);
    return ops::her2k::launch<Back, T>(ctx, c, A, B, C, alpha, beta, uplo, transA);
}

#define HER2K_INSTANTIATE(B_, fp)                                                         \
    BATCHLAS_INSTANTIATE(sig::her2k<fp>, her2k, B_, fp)                                    \
    template bool ops::her2k::fold_chosen<B_, fp>(Queue&, const MatrixView<fp, MatrixFormat::Dense>&, \
                                                 const MatrixView<fp, MatrixFormat::Dense>&,          \
                                                 const MatrixView<fp, MatrixFormat::Dense>&, Transpose);

// Keyed on the device family, not the vendor library: without the library the Vendor arm
// compiles to a throw. rocblas.cc has no her2k wrapper, so ROCm instantiates none (as before).
#if BATCHLAS_HAS_CUDA_BACKEND
HER2K_INSTANTIATE(Backend::CUDA, std::complex<float>)
HER2K_INSTANTIATE(Backend::CUDA, std::complex<double>)
#endif
#if BATCHLAS_HAS_HOST_BACKEND
HER2K_INSTANTIATE(Backend::NETLIB, std::complex<float>)
HER2K_INSTANTIATE(Backend::NETLIB, std::complex<double>)
#endif

#undef HER2K_INSTANTIATE

}  // namespace batchlas
