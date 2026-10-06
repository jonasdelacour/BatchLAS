// trmm: the whole selection path (docs/design/flat-kernel-selection.md §4.3, rule R1).
// public trmm() -> choose() -> std::visit -> launch. The kernel for a shape is the first
// runnable entry of the nearest row in tuned/trmm.<dtype>.<device>.txt; can_run() below only
// removes entries that cannot run. Triangular is the Side::Left tile kernel that skips the zero
// half of A (trmm_triangular_tiles.hh); Expand materialises op(A)'s triangle into scratch and
// hands it to the public gemm; Vendor is cuBLAS / rocBLAS / netlib trmm.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/functions/trmm.hh>
#include <batchlas/no_route.hh>
#include <batchlas/util/mempool.hh>
#include "../../select/vendor.hh"

#include "choice.hh"
#include "../../backends/level3_shape.hh"
#include "../../backends/triangular_expand.hh"
#include "../../backends/trmm_triangular_tiles.hh"
#include "../../expansion_budget.hh"
#include "../../select/select.hh"
#include "../../util/template-instantiations.hh"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <variant>

namespace batchlas {
namespace ops::trmm {

template <class... F>
struct overloaded : F... { using F::operator()...; };
template <class... F>
overloaded(F...) -> overloaded<F...>;

template <class T>
using MV = MatrixView<T, MatrixFormat::Dense>;

// The extent of B the triangular order does not occupy.
template <class T>
std::int64_t rhs_count(const MV<T>& B, Side side) {
    return side == Side::Left ? B.cols() : B.rows();
}

template <class T>
select::Key key_of(const MV<T>& A, const MV<T>& B, Side side) {
    return {{"side", side == Side::Left ? "L" : "R"}, {"order", A.rows()}, {"q", rhs_count<T>(B, side)},
            {"batch", A.batch_size()}};
}

// Correctness only (R3); shapes were validated at entry. Both native launches address every item
// as base + b * stride at one (order, q), so neither takes a heterogeneous batch (the expansion's
// gemm throws on one), and both put the batch in grid z. Every vendor loop (cuBLAS, rocBLAS,
// netlib) runs each item at the top-level (m, n), a wrong answer for a heterogeneous batch, so
// the vendor is refused one on every backend too.
template <Backend B, class T>
bool can_run(const TrmmChoice& c, const select::Device& d, Queue& q, const MV<T>& A, const MV<T>& Bm,
             const MV<T>& C, Side side) {
    constexpr bool kWired = B == Backend::CUDA;  // D6: the kernels are portable, wired for CUDA only
    const bool homogeneous = !A.is_heterogeneous() && !Bm.is_heterogeneous() && !C.is_heterogeneous();
    const bool native = kWired && d.is_gpu && homogeneous && d.max_wg >= kNativeWg && C.rows() >= 1 &&
                        C.cols() >= 1 && A.batch_size() >= 1 && A.batch_size() <= kMaxGridBatch;
    return std::visit(overloaded{
        [&](Triangular) { return native && backend::detail::trmm_tiles_supported(A, Bm, C, side); },
        [&](Expand) {
            if (!native) return false;
            const int k = A.rows(), batch = A.batch_size();
            return backend::detail::expansion_fits(q, k, batch,
                                                   backend::detail::expanded_workspace_bytes<T>(q, k, batch));
        },
        [&](Vendor) { return d.has_vendor_blas && homogeneous; },
    }, c);
}

template <Backend B, class T>
TrmmChoice choose(Queue& q, const MV<T>& A, const MV<T>& Bm, const MV<T>& C, Side side) {
    const select::Device& d = select::device_of<B>(q);
    auto ok = [&](const TrmmChoice& c) { return can_run<B, T>(c, d, q, A, Bm, C, side); };
    try {
        return select::choose("trmm", select::dtype_name<T>(), d, key_of<T>(A, Bm, side), candidates<T>(), ok,
                              rules);
    } catch (const std::runtime_error&) {
        // Vendor-free burn-down reads coverage `miss` rows; a plain runtime_error records none.
        const auto all = candidates<T>();
        if (!select::level3_vendor_available<B> && std::none_of(all.begin(), all.end(), ok))
            select::throw_no_vendor_route<T>(Op::trmm, B, select::kLevel3Library<B>);
        throw;
    }
}

// The coverage row's native flags (§5.6): computed only when coverage records a row.
template <Backend B, class T>
select::NativeFacts native_facts(Queue& q, const MV<T>& A, const MV<T>& Bm, const MV<T>& C, Side side) {
    if (!coverage::dynamic_enabled()) return {};
    const select::Device& d = select::device_of<B>(q);
    return select::native_facts(candidates<T>(),
                                [&](const TrmmChoice& c) { return can_run<B, T>(c, d, q, A, Bm, C, side); });
}

// op(A)'s triangle written densely into a workspace lease (zeros opposite it, ones on a Unit
// diagonal), then one strided-batched gemm at beta = 0: the caller's A may hold anything in the
// storage TRMM must not read, so the gemm cannot be pointed at it.
template <Backend B, class T>
Event launch_expand(Queue& ctx, const MV<T>& A, const MV<T>& Bm, const MV<T>& C, T alpha, Side side, Uplo uplo,
                    Transpose transA, Diag diag) {
    const int k = A.rows(), batch = A.batch_size();
    const int ld = backend::detail::expanded_ld<T>(k);
    auto ws = ctx.workspace(backend::detail::expanded_workspace_bytes<T>(ctx, k, batch));
    BumpAllocator pool(ws.span());
    auto storage = pool.allocate<T>(ctx, static_cast<std::size_t>(ld) * static_cast<std::size_t>(k) *
                                             static_cast<std::size_t>(batch));
    MV<T> expanded(storage.data(), k, k, ld, ld * k, batch);
    Event expansion = backend::detail::expand_triangular<T>(ctx, expanded, A, uplo, diag);
    // An out-of-order queue orders nothing across the SYCL / vendor-stream boundary.
    if (!ctx.in_order()) expansion.wait();
    if (side == Side::Left)
        return gemm<B, T>(ctx, expanded, Bm, C, alpha, T(0), transA, Transpose::NoTrans, ComputePrecision::Default);
    return gemm<B, T>(ctx, Bm, expanded, C, alpha, T(0), Transpose::NoTrans, transA, ComputePrecision::Default);
}

template <Backend B, class T>
Event launch(Queue& q, const TrmmChoice& c, const MV<T>& A, const MV<T>& Bm, const MV<T>& C, T alpha, Side side,
             Uplo uplo, Transpose transA, Diag diag) {
    return std::visit(overloaded{
        [&](Triangular) -> Event {
            if constexpr (B == Backend::CUDA)
                return backend::detail::trmm_triangular_tiles<T>(q, A, Bm, C, alpha, uplo, transA, diag);
            else
                throw std::logic_error("trmm: triangular is wired for CUDA only");  // can_run refuses it
        },
        [&](Expand) -> Event {
            if constexpr (B == Backend::CUDA)
                return launch_expand<B, T>(q, A, Bm, C, alpha, side, uplo, transA, diag);
            else
                throw std::logic_error("trmm: expand is wired for CUDA only");  // can_run refuses it
        },
        [&](Vendor) -> Event {
            if constexpr (select::level3_vendor_available<B>)
                return backend::trmm_vendor<B, T>(q, A, Bm, C, alpha, side, uplo, transA, diag);
            else
                select::throw_no_vendor_route<T>(Op::trmm, B, select::kLevel3Library<B>);
        },
    }, c);
}

}  // namespace ops::trmm

// trmm takes no caller workspace (no trmm_buffer_size); Expand leases its scratch from the
// Queue's arena, so R5 has nothing to size.
template <Backend Back, typename T>
Event trmm(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, const MatrixView<T, MatrixFormat::Dense>& B,
           const MatrixView<T, MatrixFormat::Dense>& C, T alpha, Side side, Uplo uplo, Transpose transA, Diag diag) {
    (void)backend::shape::validate_product<std::invalid_argument>("TRMM", A, B, C, side);
    // An empty problem is a no-op under any pin: both native launches throw on an empty batch,
    // and the cuBLAS loop faults on one.
    if (A.batch_size() == 0 || C.rows() == 0 || C.cols() == 0) return ctx.create_event_after_external_work();
    const auto c = ops::trmm::choose<Back, T>(ctx, A, B, C, side);
    // The coverage row's key, as the old level-3 recorder wrote it: C's extents and A's order.
    auto shape = select::square_shape<Back, T>(A.rows(), A.batch_size());
    shape.m = C.rows();
    shape.n = C.cols();
    shape.uplo = uplo;
    shape.side = side;
    shape.diag = diag;
    shape.transA = transA;
    const select::Key trace_key = ops::trmm::key_of<T>(A, B, side);
    select::TraceScope trace("trmm", c, shape, ops::trmm::native_facts<Back, T>(ctx, A, B, C, side), trace_key);
    return ops::trmm::launch<Back, T>(ctx, c, A, B, C, alpha, side, uplo, transA, diag);
}

#define TRMM_INSTANTIATE(B_, fp) BATCHLAS_INSTANTIATE(sig::trmm<fp>, trmm, B_, fp)

#define TRMM_ALL(B_)                          \
    TRMM_INSTANTIATE(B_, float)               \
    TRMM_INSTANTIATE(B_, double)              \
    TRMM_INSTANTIATE(B_, std::complex<float>) \
    TRMM_INSTANTIATE(B_, std::complex<double>)

// Keyed on the device family, not the vendor library: without the library the Vendor arm
// compiles to a throw, so the symbol exists in every build with the device.
#if BATCHLAS_HAS_CUDA_BACKEND
TRMM_ALL(Backend::CUDA)
#endif
#if BATCHLAS_HAS_ROCM_BACKEND
TRMM_ALL(Backend::ROCM)
#endif
#if BATCHLAS_HAS_HOST_BACKEND
TRMM_ALL(Backend::NETLIB)
#endif

#undef TRMM_ALL
#undef TRMM_INSTANTIATE

}  // namespace batchlas
