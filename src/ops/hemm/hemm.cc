// hemm: the whole selection path (docs/design/flat-kernel-selection.md §4.3, rule R1;
// §12 "Hermitian three"). public hemm() -> choose() -> std::visit -> launch. The kernel for a
// shape is the first runnable entry of the nearest row in tuned/hemm.<dtype>.<device>.txt;
// can_run() below only removes entries that cannot run. Expand mirrors A's referenced triangle
// (conjugated, real diagonal) into a dense scratch copy and hands it to the public gemm, which
// picks its own kernel; Vendor is the per-item cublas?hemm / cblas_?hemm loop.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/functions/hemm.hh>
#include <batchlas/no_route.hh>
#include <batchlas/util/mempool.hh>
#include "../../select/vendor.hh"

#include "choice.hh"
#include "../../backends/level3_shape.hh"
#include "../../backends/triangular_expand.hh"
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
namespace ops::hemm {

template <class... F>
struct overloaded : F... { using F::operator()...; };
template <class... F>
overloaded(F...) -> overloaded<F...>;

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
        [&](Vendor) { return d.has_vendor_blas && homogeneous; },
    }, c);
}

template <Backend B, class T>
HemmChoice choose(Queue& q, const MV<T>& A, const MV<T>& Bm, const MV<T>& C, Side side) {
    const select::Device& d = select::device_of<B>(q);
    auto ok = [&](const HemmChoice& c) { return can_run<B, T>(c, d, q, A, Bm, C); };
    try {
        return select::choose("hemm", select::dtype_name<T>(), d, key_of<T>(A, C, side), candidates<T>(), ok,
                              rules);
    } catch (const std::runtime_error&) {
        // Vendor-free burn-down reads coverage `miss` rows; a plain runtime_error records none.
        const auto all = candidates<T>();
        if (!select::level3_vendor_available<B> && std::none_of(all.begin(), all.end(), ok))
            select::throw_no_vendor_route<T>(Op::hemm, B, select::kLevel3Library<B>);
        throw;
    }
}

// The coverage row's native flags (§5.6): computed only when coverage records a row.
template <Backend B, class T>
select::NativeFacts native_facts(Queue& q, const MV<T>& A, const MV<T>& Bm, const MV<T>& C) {
    if (!coverage::dynamic_enabled()) return {};
    const select::Device& d = select::device_of<B>(q);
    return select::native_facts(candidates<T>(),
                                [&](const HemmChoice& c) { return can_run<B, T>(c, d, q, A, Bm, C); });
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
    if (beta == T(0)) C.fill(q, T(0));
    {
        BATCHLAS_KERNEL_TRACE_SCOPE("hemm.expand");
        (void)backend::detail::expand_mirrored<T, /*Conjugate=*/true>(q, full, A, uplo);
    }
    // An out-of-order queue orders nothing between the expansion and a vendor gemm's stream.
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
            if constexpr (select::level3_vendor_available<B>)
                return backend::hemm_vendor<B, T>(q, A, Bm, C, alpha, beta, side, uplo);
            else
                select::throw_no_vendor_route<T>(Op::hemm, B, select::kLevel3Library<B>);
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
    const auto c = ops::hemm::choose<Back, T>(ctx, A, B, C, side);
    // The coverage row's key, as symm's: C's extents and A's order.
    auto shape = select::square_shape<Back, T>(C.rows(), C.batch_size());
    shape.n = C.cols();
    shape.k = A.rows();
    shape.side = side;
    shape.uplo = uplo;
    const select::Key trace_key = ops::hemm::key_of<T>(A, C, side);
    select::TraceScope trace("hemm", c, shape, ops::hemm::native_facts<Back, T>(ctx, A, B, C), trace_key);
    return ops::hemm::launch<Back, T>(ctx, c, A, B, C, alpha, beta, side, uplo);
}

#define HEMM_INSTANTIATE(B_, fp) BATCHLAS_INSTANTIATE(sig::hemm<fp>, hemm, B_, fp)

// Keyed on the device family, not the vendor library: without the library the Vendor arm
// compiles to a throw. rocblas.cc has no hemm wrapper, so ROCm instantiates none (as before).
#if BATCHLAS_HAS_CUDA_BACKEND
HEMM_INSTANTIATE(Backend::CUDA, std::complex<float>)
HEMM_INSTANTIATE(Backend::CUDA, std::complex<double>)
#endif
#if BATCHLAS_HAS_HOST_BACKEND
HEMM_INSTANTIATE(Backend::NETLIB, std::complex<float>)
HEMM_INSTANTIATE(Backend::NETLIB, std::complex<double>)
#endif

#undef HEMM_INSTANTIATE

}  // namespace batchlas
