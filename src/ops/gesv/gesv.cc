// SPDX-License-Identifier: MIT
// gesv (flat-kernel-selection.md §4.3, R1; docs/design/flat-kernel-selection.md#phase-5-gesv): select::run
// takes the first entry of the nearest tuned/gesv.<device>.txt row that can_run() admits. Tiny is
// the fused LU factor-and-solve kernel; Blocked composes the public getrf and getrs, and each child picks
// its own kernel. No vendor ships a batched gesv, so there is no vendor arm.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/gesv.hh>
#include <batchlas/blas/functions/getrf.hh>
#include <batchlas/blas/functions/getrs.hh>

#include <src/ops/gesv/choice.hh>
#include <src/select/select.hh>
#include <src/util/batch_mask.hh>
#include <src/ops/gesv/gesv_native.hh>
#include <src/util/template-instantiations.hh>

#include <complex>
#include <cstddef>
#include <string>
#include <utility>
#include <variant>

namespace batchlas {
namespace ops::gesv {

using select::overloaded;

template <typename Scalar>
using MV = MatrixView<Scalar, MatrixFormat::Dense>;

template <typename Scalar>
select::Key key_of(const MV<Scalar>& A, const MV<Scalar>& B) {
    return {{"n", A.rows()}, {"nrhs", B.cols()}};
}

// Correctness only (R3): false means the driver would throw. Tiny's clauses are the argument
// checks of gesv_tiny_dispatch, plus a NETLIB refusal: the GPU kernel packs 1-based
// int32 pivots into the int64 span's low half where netlib writes true int64. No family takes a
// heterogeneous batch. Both families honour a batch mask: Tiny skips an inactive item, and
// Blocked's children (getrf, getrs) honour it or refuse it themselves.
template <Backend Bk, typename Scalar>
bool can_run(const GesvChoice& c, const select::Device& d, const MV<Scalar>& A, const MV<Scalar>& B) {
    const int64_t n = A.rows(), nrhs = B.cols();
    const bool homogeneous = !A.is_heterogeneous() && !B.is_heterogeneous();
    return std::visit(overloaded{
        [&](Tiny) {
            return Bk != Backend::NETLIB && d.is_gpu && d.has_sg32 && homogeneous && n >= 1 && nrhs >= 1 &&
                   A.batch_size() >= 1 && n <= ops::gesv::gesv_tiny_max_n<Scalar>() &&
                   nrhs <= ops::gesv::kGesvTinyMaxRhs && d.max_wg >= resident::kTinyWgSize;
        },
        [&](Blocked) { return homogeneous; },
    }, c);
}

// Blocked cuts the caller's span at getrf's reported size: BumpAllocator's sizing results are
// alignment multiples and a real pool's base is device-aligned, so the tail is aligned too.
template <Backend B, typename Scalar>
Event launch(Queue& q, const GesvChoice& c, const MV<Scalar>& A, const MV<Scalar>& Bm, Span<int64_t> pivots,
             Span<std::byte> ws, Span<int32_t> info) {
    return std::visit(overloaded{
        [&](Tiny) { return ops::gesv::gesv_tiny_dispatch<Scalar>(q, A, Bm, pivots, ws, info); },
        [&](Blocked) {
            const std::size_t split = getrf_buffer_size<B, Scalar>(q, A);
            if (ws.size() < split)
                throw batchlas::workspace_error("gesv: workspace is shorter than the blocked choice's getrf needs");
            (void)getrf<B, Scalar>(q, A, pivots, ws.subspan(0, split), info);
            return getrs<B, Scalar>(q, A, Bm, Transpose::NoTrans, pivots, ws.subspan(split));
        },
    }, c);
}

// Exactly the chosen family's need (R5). Blocked: a SUM, the two legs hold disjoint halves.
template <Backend B, typename Scalar>
std::size_t workspace(Queue& q, const GesvChoice& c, const MV<Scalar>& A, const MV<Scalar>& Bm) {
    return std::visit(overloaded{
        [&](Tiny) { return ops::gesv::gesv_tiny_buffer_size<Scalar>(q, A, Bm); },
        [&](Blocked) {
            return getrf_buffer_size<B, Scalar>(q, A) + getrs_buffer_size<B, Scalar>(q, A, Bm, Transpose::NoTrans);
        },
    }, c);
}

// n or nrhs 0 (batch 0 returned earlier, a no-op) or a heterogeneous batch has no kernel, so
// both throw before choose(): neither Auto nor a pin can take them.
template <typename Scalar>
void throw_if_unservable(const MV<Scalar>& A, const MV<Scalar>& Bm, const char* who) {
    if (A.is_heterogeneous() || Bm.is_heterogeneous())
        throw batchlas::invalid_argument(std::string(who) + ": a heterogeneous batch (per-item active dims on " +
                                       (A.is_heterogeneous() ? "A" : "B") + "), which no gesv kernel serves.");
    if (A.rows() >= 1 && Bm.cols() >= 1 && A.batch_size() >= 1) return;
    throw batchlas::invalid_argument(std::string(who) + ": n = " + std::to_string(A.rows()) + ", nrhs = " +
                                   std::to_string(Bm.cols()) + ", batch = " + std::to_string(A.batch_size()) +
                                   " is an empty problem, which no gesv kernel serves.");
}

// Both operands carry the call's mask whichever brought it: Blocked's getrf sees A alone.
template <typename Scalar>
std::pair<MV<Scalar>, MV<Scalar>> adopt_mask(const MV<Scalar>& A, const MV<Scalar>& B, const char* who) {
    const auto mask = detail::call_mask(who, A, B);
    if (mask.empty() || A.batch_size() != B.batch_size()) return {A, B};
    return {A.with_batch_mask(mask), B.with_batch_mask(mask)};
}

}  // namespace ops::gesv

template <Backend Back, typename Scalar>
Event gesv(Queue& ctx, const MatrixView<Scalar, MatrixFormat::Dense>& A_, const MatrixView<Scalar,
           MatrixFormat::Dense>& B_,
           Span<int64_t> pivots, Span<std::byte> workspace, Span<int32_t> info) {
    gesv_validate_params<Scalar>(A_, B_);
    const auto views = ops::gesv::adopt_mask<Scalar>(A_, B_, "gesv");
    const auto& A = views.first;
    const auto& B = views.second;
    // An empty batch is a no-op under any pin, as gemm's is: it never reaches selection.
    if (A.batch_size() == 0) return ctx.create_event_after_external_work();
    ops::gesv::throw_if_unservable<Scalar>(A, B, "gesv");
    // The coverage row's key: m = k = order, n = nrhs.
    const coverage::Shape shape{.m = A.rows(), .n = B.cols(), .k = A.rows(), .batch = A.batch_size()};
    return select::run<Back, Scalar>(
        ops::gesv::spec, ctx, ops::gesv::key_of<Scalar>(A, B), ops::gesv::candidates<Scalar>(),
        [&](const auto& c, const auto& d) { return ops::gesv::can_run<Back, Scalar>(c, d, A, B); }, shape,
        {{"n", A.rows()}, {"nrhs", B.cols()}, {"batch", A.batch_size()}},
        [&](const auto& c) { return ops::gesv::launch<Back, Scalar>(ctx, c, A, B, pivots, workspace, info); });
}

template <Backend Back, typename Scalar>
size_t gesv_buffer_size(Queue& ctx, const MatrixView<Scalar, MatrixFormat::Dense>& A_,
                        const MatrixView<Scalar, MatrixFormat::Dense>& B_) {
    gesv_validate_params<Scalar>(A_, B_);
    const auto views = ops::gesv::adopt_mask<Scalar>(A_, B_, "gesv_buffer_size");
    const auto& A = views.first;
    const auto& B = views.second;
    if (A.batch_size() == 0) return 0;
    ops::gesv::throw_if_unservable<Scalar>(A, B, "gesv_buffer_size");
    const auto c = select::pick<Back, Scalar>(
        ops::gesv::spec, ctx, ops::gesv::key_of<Scalar>(A, B), ops::gesv::candidates<Scalar>(),
        [&](const auto& k, const auto& d) { return ops::gesv::can_run<Back, Scalar>(k, d, A, B); });
    return ops::gesv::workspace<Back, Scalar>(ctx, c, A, B);
}

// Keyed on the device family: gesv has no vendor arm, so every build with the device has it.
#define GESV_INSTANTIATE(B_, fp) BATCHLAS_INSTANTIATE_OP(B_, fp, gesv) BATCHLAS_INSTANTIATE_OP(B_, fp, gesv_buffer_size)
BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(GESV_INSTANTIATE)
#undef GESV_INSTANTIATE

}  // namespace batchlas
