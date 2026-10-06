// symm: the whole selection path (docs/design/flat-kernel-selection.md §4.3, rule R1;
// docs/design/flat-kernel-selection.md §12). public symm() -> choose() -> std::visit -> launch.
// The kernel for a shape is the first runnable entry of the nearest row in
// tuned/symm.<dtype>.<device>.txt; can_run() below only removes entries that cannot run.
// Expand mirrors A's referenced triangle into a dense scratch copy and hands it to the
// public gemm, which picks its own kernel; Vendor is the per-item cublas?symm / cblas_?symm loop.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/functions/symm.hh>
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
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <variant>

namespace batchlas {
namespace ops::symm {

template <class... F>
struct overloaded : F... { using F::operator()...; };
template <class... F>
overloaded(F...) -> overloaded<F...>;

template <class T>
using MV = MatrixView<T, MatrixFormat::Dense>;

template <class T>
select::Key key_of(const MV<T>& C) {
    return {{"form", form_of(C.rows(), C.cols())}, {"m", C.rows()}, {"n", C.cols()}, {"batch", C.batch_size()}};
}

// Correctness only (R3). The expansion reads A at its storage order and stride, and the gemm it
// feeds rejects a heterogeneous B or C against the homogeneous scratch, so no operand may be
// heterogeneous. Its scratch must fit (expansion_fits: the int-linearised grid and the
// BATCHLAS_EXPAND_MAX_BYTES / quarter-of-memory budget), and its batch sits in grid z.
template <Backend B, class T>
bool can_run(const SymmChoice& c, const select::Device& d, Queue& q, const MV<T>& A, const MV<T>& Bm,
             const MV<T>& C) {
    constexpr bool kWired = B == Backend::CUDA;  // the old reach: ROCm and the host stay vendor
    const std::int64_t k = A.rows(), batch = A.batch_size();
    const bool native = kWired && d.is_gpu && !A.is_heterogeneous() && !Bm.is_heterogeneous() &&
                        !C.is_heterogeneous() && C.rows() >= 1 && C.cols() >= 1 && batch >= 1 &&
                        batch <= kMaxGridBatch;
    return std::visit(overloaded{
        [&](Expand) {
            return native && backend::detail::expansion_fits(
                                 q, int(k), int(batch),
                                 backend::detail::expanded_workspace_bytes<T>(q, int(k), int(batch)));
        },
        [&](Vendor) { return d.has_vendor_blas; },
    }, c);
}

template <Backend B, class T>
SymmChoice choose(Queue& q, const MV<T>& A, const MV<T>& Bm, const MV<T>& C) {
    const select::Device& d = select::device_of<B>(q);
    auto ok = [&](const SymmChoice& c) { return can_run<B, T>(c, d, q, A, Bm, C); };
    try {
        return select::choose("symm", select::dtype_name<T>(), d, key_of<T>(C), candidates<T>(), ok, rules);
    } catch (const std::runtime_error&) {
        // Vendor-free burn-down reads coverage `miss` rows; a plain runtime_error records none.
        const auto all = candidates<T>();
        if (!select::level3_vendor_available<B> && std::none_of(all.begin(), all.end(), ok))
            select::throw_no_vendor_route<T>(Op::symm, B, select::kLevel3Library<B>);
        throw;
    }
}

// The coverage row's native flags (§5.6): computed only when coverage records a row.
template <Backend B, class T>
select::NativeFacts native_facts(Queue& q, const MV<T>& A, const MV<T>& Bm, const MV<T>& C) {
    if (!coverage::dynamic_enabled()) return {};
    const select::Device& d = select::device_of<B>(q);
    return select::native_facts(candidates<T>(),
                                [&](const SymmChoice& c) { return can_run<B, T>(c, d, q, A, Bm, C); });
}

// The scratch is a lease on the queue's arena, not a Matrix: a managed allocation migrates on
// first touch (an order of magnitude over the gemm at n=512 batch=512) and would be freed while
// the kernels reading it are only enqueued.
template <Backend B, class T>
Event expand_gemm(Queue& q, const MV<T>& A, const MV<T>& Bm, const MV<T>& C, T alpha, T beta, Side side,
                  Uplo uplo) {
    const int n = A.rows(), batch = A.batch_size();
    const int ld = backend::detail::expanded_ld<T>(n);
    auto ws = q.workspace(backend::detail::expanded_workspace_bytes<T>(q, n, batch));
    BumpAllocator pool(ws.span());
    auto storage = pool.allocate<T>(q, std::size_t(ld) * std::size_t(n) * std::size_t(batch));
    const MV<T> full(storage.data(), n, n, ld, ld * n, batch);
    Event expansion;
    {
        BATCHLAS_KERNEL_TRACE_SCOPE("symm.expand");
        expansion = backend::detail::expand_mirrored<T, /*Conjugate=*/false>(q, full, A, uplo);
    }
    // An out-of-order queue orders nothing between the expansion and a vendor gemm's stream.
    if (!q.in_order()) expansion.wait();
    if (side == Side::Left)
        return gemm<B, T>(q, full, Bm, C, alpha, beta, Transpose::NoTrans, Transpose::NoTrans,
                          ComputePrecision::Default);
    return gemm<B, T>(q, Bm, full, C, alpha, beta, Transpose::NoTrans, Transpose::NoTrans,
                      ComputePrecision::Default);
}

template <Backend B, class T>
Event launch(Queue& q, const SymmChoice& c, const MV<T>& A, const MV<T>& Bm, const MV<T>& C, T alpha, T beta,
             Side side, Uplo uplo) {
    return std::visit(overloaded{
        [&](Expand) { return expand_gemm<B, T>(q, A, Bm, C, alpha, beta, side, uplo); },
        [&](Vendor) -> Event {
            if constexpr (select::level3_vendor_available<B>)
                return backend::symm_vendor<B, T>(q, A, Bm, C, alpha, beta, side, uplo);
            else
                select::throw_no_vendor_route<T>(Op::symm, B, select::kLevel3Library<B>);
        },
    }, c);
}

}  // namespace ops::symm

// symm takes no caller workspace (no symm_buffer_size); Expand leases its scratch from the
// queue's arena, so R5 has nothing to size.
template <Backend Back, RealScalar T>
Event symm(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, const MatrixView<T, MatrixFormat::Dense>& B,
           const MatrixView<T, MatrixFormat::Dense>& C, T alpha, T beta, Side side, Uplo uplo) {
    backend::shape::validate_product<std::invalid_argument>("SYMM", A, B, C, side);
    const auto c = ops::symm::choose<Back, T>(ctx, A, B, C);
    // The coverage row's key, as the old record_level3_route wrote it: C's extents and A's order.
    auto shape = select::square_shape<Back, T>(C.rows(), C.batch_size());
    shape.n = C.cols();
    shape.k = A.rows();
    shape.side = side;
    shape.uplo = uplo;
    const select::Key trace_key = ops::symm::key_of<T>(C);
    select::TraceScope trace("symm", c, shape, ops::symm::native_facts<Back, T>(ctx, A, B, C), trace_key);
    return ops::symm::launch<Back, T>(ctx, c, A, B, C, alpha, beta, side, uplo);
}

#define SYMM_INSTANTIATE(B_, fp) BATCHLAS_INSTANTIATE(sig::symm<fp>, symm, B_, fp)

// Keyed on the device family, not the vendor library: without the library the Vendor arm
// compiles to a throw. ROCm/MKL symm is src/extensions/symm.cc; this file instantiates CUDA and
// NETLIB only (as before).
#if BATCHLAS_HAS_CUDA_BACKEND
SYMM_INSTANTIATE(Backend::CUDA, float)
SYMM_INSTANTIATE(Backend::CUDA, double)
#endif
#if BATCHLAS_HAS_HOST_BACKEND
SYMM_INSTANTIATE(Backend::NETLIB, float)
SYMM_INSTANTIATE(Backend::NETLIB, double)
#endif

#undef SYMM_INSTANTIATE

}  // namespace batchlas
