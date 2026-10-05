// gesv: the whole selection path (docs/design/flat-kernel-selection.md §4.3, rule R1;
// docs/design/flat-select-p5/gesv.md). public gesv() -> choose() -> std::visit -> launch.
// The kernel for a shape is the first runnable entry of the nearest row in
// tuned/gesv.<dtype>.<device>.txt; can_run() below only removes entries that cannot run.
// Tiny is the fused LU factor-and-solve kernel; Blocked composes the public getrf and getrs,
// and each child picks its own kernel. No vendor ships a batched gesv, so there is no vendor arm.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/gesv.hh>
#include <batchlas/blas/functions/getrf.hh>
#include <batchlas/blas/functions/getrs.hh>

#include "choice.hh"
#include "../../select/select.hh"
#include "../../extensions/solve_native.hh"
#include "../../util/template-instantiations.hh"

#include <complex>
#include <cstddef>
#include <string>
#include <variant>

namespace batchlas {
namespace ops::gesv {

template <class... F>
struct overloaded : F... { using F::operator()...; };
template <class... F>
overloaded(F...) -> overloaded<F...>;

template <class T>
using MV = MatrixView<T, MatrixFormat::Dense>;

template <class T>
select::Key key_of(const MV<T>& A, const MV<T>& B) {
    return {{"n", A.rows()}, {"nrhs", B.cols()}};
}

// Correctness only (R3): false means the driver would throw. Tiny's clauses are the argument
// checks of gesv_tiny_dispatch, plus the old router's NETLIB refusal: the GPU kernel packs 1-based
// int32 pivots into the int64 span's low half where netlib writes true int64. No family takes a
// heterogeneous batch (the old router refused it on every route).
template <Backend Bk, class T>
bool can_run(const GesvChoice& c, const select::Device& d, const MV<T>& A, const MV<T>& B) {
    const int64_t n = A.rows(), nrhs = B.cols();
    const bool homogeneous = !A.is_heterogeneous() && !B.is_heterogeneous();
    return std::visit(overloaded{
        [&](Tiny) {
            return Bk != Backend::NETLIB && d.is_gpu && d.has_sg32 && homogeneous && n >= 1 && nrhs >= 1 &&
                   A.batch_size() >= 1 && n <= sycl_gesv::gesv_tiny_max_n<T>() &&
                   nrhs <= sycl_gesv::kGesvTinyMaxRhs && d.max_wg >= sycl_gesv::kGesvTinyWgSize;
        },
        [&](Blocked) { return homogeneous; },
    }, c);
}

template <Backend B, class T>
GesvChoice choose(Queue& q, const MV<T>& A, const MV<T>& Bm) {
    const select::Device& d = select::device_of<B>(q);
    return select::choose("gesv", select::dtype_name<T>(), d, key_of<T>(A, Bm), candidates<T>(),
                          [&](const GesvChoice& c) { return can_run<B, T>(c, d, A, Bm); }, rules);
}

// The coverage row's native flags (§5.6): computed only when coverage records a row.
template <Backend B, class T>
select::NativeFacts native_facts(Queue& q, const MV<T>& A, const MV<T>& Bm) {
    if (!dispatch::coverage::dynamic_enabled()) return {};
    const select::Device& d = select::device_of<B>(q);
    return select::native_facts(candidates<T>(), [&](const GesvChoice& c) { return can_run<B, T>(c, d, A, Bm); });
}

// Blocked cuts the caller's span at getrf's reported size: BumpAllocator's sizing results are
// alignment multiples and a real pool's base is device-aligned, so the tail is aligned too.
template <Backend B, class T>
Event launch(Queue& q, const GesvChoice& c, const MV<T>& A, const MV<T>& Bm, Span<int64_t> pivots,
             Span<std::byte> ws, Span<int32_t> info) {
    return std::visit(overloaded{
        [&](Tiny) { return sycl_gesv::gesv_tiny_dispatch<T>(q, A, Bm, pivots, ws, info); },
        [&](Blocked) {
            const std::size_t split = getrf_buffer_size<B, T>(q, A);
            if (ws.size() < split)
                throw batchlas::workspace_error("gesv: workspace is shorter than the blocked choice's getrf needs");
            (void)getrf<B, T>(q, A, pivots, ws.subspan(0, split), info);
            return getrs<B, T>(q, A, Bm, Transpose::NoTrans, pivots, ws.subspan(split));
        },
    }, c);
}

// Exactly the chosen family's need (R5). Blocked: a SUM, the two legs hold disjoint halves.
template <Backend B, class T>
std::size_t workspace(Queue& q, const GesvChoice& c, const MV<T>& A, const MV<T>& Bm) {
    return std::visit(overloaded{
        [&](Tiny) { return sycl_gesv::gesv_tiny_buffer_size<T>(q, A, Bm); },
        [&](Blocked) {
            return getrf_buffer_size<B, T>(q, A) + getrs_buffer_size<B, T>(q, A, Bm, Transpose::NoTrans);
        },
    }, c);
}

// An empty problem (n, nrhs or batch 0) or a heterogeneous batch has no kernel; the old router
// refused both the same way. Thrown before choose() so a pin cannot take them either.
template <class T>
void throw_if_unservable(const MV<T>& A, const MV<T>& Bm, const char* who) {
    if (A.is_heterogeneous() || Bm.is_heterogeneous())
        throw batchlas::internal_error(std::string(who) + ": a heterogeneous batch (per-item active dims on " +
                                       (A.is_heterogeneous() ? "A" : "B") + "), which no gesv kernel serves.");
    if (A.rows() >= 1 && Bm.cols() >= 1 && A.batch_size() >= 1) return;
    throw batchlas::internal_error(std::string(who) + ": n = " + std::to_string(A.rows()) + ", nrhs = " +
                                   std::to_string(Bm.cols()) + ", batch = " + std::to_string(A.batch_size()) +
                                   " is an empty problem, which no gesv kernel serves.");
}

}  // namespace ops::gesv

template <Backend Back, typename T>
Event gesv(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, const MatrixView<T, MatrixFormat::Dense>& B,
           Span<int64_t> pivots, Span<std::byte> work_space, Span<int32_t> info) {
    gesv_validate_params<T>(A, B);
    ops::gesv::throw_if_unservable<T>(A, B, "gesv");
    const auto c = ops::gesv::choose<Back, T>(ctx, A, B);
    // The coverage row's key: m = k = order, n = nrhs (the old GesvShape's spelling).
    auto shape = select::square_shape<Back, T>(A.rows(), A.batch_size());
    shape.n = B.cols();
    const select::Key trace_key{{"n", A.rows()}, {"nrhs", B.cols()}, {"batch", A.batch_size()}};
    select::TraceScope trace("gesv", c, shape, ops::gesv::native_facts<Back, T>(ctx, A, B), trace_key);
    return ops::gesv::launch<Back, T>(ctx, c, A, B, pivots, work_space, info);
}

template <Backend Back, typename T>
size_t gesv_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                        const MatrixView<T, MatrixFormat::Dense>& B) {
    gesv_validate_params<T>(A, B);
    ops::gesv::throw_if_unservable<T>(A, B, "gesv_buffer_size");
    return ops::gesv::workspace<Back, T>(ctx, ops::gesv::choose<Back, T>(ctx, A, B), A, B);
}

#define GESV_INSTANTIATE(B_, fp)                                    \
    BATCHLAS_INSTANTIATE(sig::gesv<fp>, gesv, B_, fp)               \
    BATCHLAS_INSTANTIATE(sig::gesv_buffer_size<fp>, gesv_buffer_size, B_, fp)

#define GESV_ALL(B_)                          \
    GESV_INSTANTIATE(B_, float)               \
    GESV_INSTANTIATE(B_, double)              \
    GESV_INSTANTIATE(B_, std::complex<float>) \
    GESV_INSTANTIATE(B_, std::complex<double>)

// Keyed on the device family: gesv has no vendor arm, so every build with the device has it.
#if BATCHLAS_HAS_CUDA_BACKEND
GESV_ALL(Backend::CUDA)
#endif
#if BATCHLAS_HAS_ROCM_BACKEND
GESV_ALL(Backend::ROCM)
#endif
#if BATCHLAS_HAS_HOST_BACKEND
GESV_ALL(Backend::NETLIB)
#endif

#undef GESV_ALL
#undef GESV_INSTANTIATE

}  // namespace batchlas
