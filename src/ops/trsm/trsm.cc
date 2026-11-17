// SPDX-License-Identifier: MIT
// trsm (flat-kernel-selection.md §4.3, R1; flat-kernel-selection.md#phase-3-trsm): select::run takes
// the first entry of the nearest tuned/trsm.<device>.txt row that can_run() admits. Cta and SgLeft
// are register-resident solvers for order <= 32; Blocked solves 32-wide diagonal blocks with Cta and
// updates the rest with the public gemm, which picks its own kernel.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/functions/trsm.hh>

#include <src/ops/trsm/choice.hh>
#include <src/select/select.hh>
#include <src/util/batch_mask.hh>
#include <src/ops/trsm/trsm_native.hh>
#include <src/util/template-instantiations.hh>

#include <complex>
#include <cstdint>
#include <variant>

namespace batchlas {
namespace ops::trsm {

using select::overloaded;

template <typename Scalar>
using MV = MatrixView<Scalar, MatrixFormat::Dense>;

// The right-hand-side count: the extent of B the triangular order does not occupy.
template <typename Scalar>
std::int64_t rhs_count(const MV<Scalar>& B, Side side) {
    return side == Side::Left ? B.cols() : B.rows();
}

// ConjTrans folds to T: the kernels differ from Trans only in a conjugation (trsm_canonical.hh).
template <typename Scalar>
select::Key key_of(const MV<Scalar>& A, const MV<Scalar>& B, Side side, Transpose transA) {
    return {{"side", side == Side::Left ? "L" : "R"}, {"trans", transA == Transpose::NoTrans ? "N" : "T"},
            {"order", A.rows()}, {"q", rhs_count<Scalar>(B, side)}, {"batch", A.batch_size()}};
}

// Correctness only (R3): false means the driver would throw or answer wrongly. One launch covers
// the batch with a single (order, q, ld, stride), so no native family takes a heterogeneous
// batch, and neither does the vendor, which runs at the full storage order (known-defects #12).
template <typename Scalar>
bool can_run(const TrsmChoice& c, const select::Device& d, const MV<Scalar>& A, const MV<Scalar>& B, Side side) {
    // Every native family honours a batch mask; the vendor's strided call cannot skip an item.
    const bool masked = batchlas::detail::any_batch_mask(A, B);
    const std::int64_t order = A.rows(), q = rhs_count<Scalar>(B, side);
    const bool native = d.is_gpu && !A.is_heterogeneous() && !B.is_heterogeneous() && order >= 1 && q >= 1 &&
                        A.batch_size() >= 1;
    const int cta_max = ops::trsm::trsm_cta_max_n<Scalar>();
    // V1's work-group ladder bottoms out at 32 lanes whatever max_wg is (trsm_v1_ladder_wg).
    const bool v1_fits = d.max_wg >= 32;
    return std::visit(overloaded{
        [&](Cta) { return native && v1_fits && order <= cta_max; },
        [&](SgLeft) {
            return native && side == Side::Left && d.has_sg32 && order <= ops::trsm::kTrsmSgLeftMaxN &&
                   d.max_wg >= ops::trsm::kTrsmSgLeftWgSize;
        },
        [&](Blocked) { return native && v1_fits && cta_max >= 1; },
        [&](Vendor) { return d.has_vendor && !A.is_heterogeneous() && !B.is_heterogeneous() && !masked; },
    }, c);
}

template <Backend B, typename Scalar>
Event launch(Queue& q, const TrsmChoice& c, const MV<Scalar>& A, const MV<Scalar>& Bm, Scalar alpha, Side side,
             Uplo uplo,
             Transpose transA, Diag diag) {
    return std::visit(overloaded{
        [&](Cta) { return ops::trsm::trsm_native_v1_dispatch<Scalar>(q, A, Bm, alpha, side, uplo, transA, diag); },
        [&](SgLeft) { return ops::trsm::trsm_native_sg_left_dispatch<Scalar>(q, A, Bm, alpha, uplo, transA, diag); },
        [&](Blocked) {
            // The public gemm (table-routed): trsm's operands are sub-views carrying the
            // parent's ld, where the native GEMM collapses.
            // evidence: docs/perf/gemm.md#the-strided-ld-defect-and-the-routing-fix
            return ops::trsm::trsm_native_blocked<Scalar>(
                q, A, Bm, alpha, side, uplo, transA, diag,
                [](Queue& c2, const MV<Scalar>& ga, const MV<Scalar>& gb, const MV<Scalar>& gc, Scalar galpha,
                   Scalar gbeta, Transpose gta,
                   Transpose gtb, ComputePrecision gp) {
                    return gemm<B, Scalar>(c2, ga, gb, gc, galpha, gbeta, gta, gtb, gp);
                });
        },
        [&](Vendor) -> Event {
            if constexpr (select::has_library<B>(spec.vendor))
                return backend::trsm_vendor<B, Scalar>(q, A, Bm, side, uplo, transA, diag, alpha);
            else select::no_vendor<B, Scalar>(spec);
        },
    }, c);
}

}  // namespace ops::trsm

// trsm takes no workspace (no trsm_buffer_size), so R5 has nothing to size.
template <Backend Back, typename Scalar>
Event trsm(Queue& ctx, const MatrixView<Scalar, MatrixFormat::Dense>& A, const MatrixView<Scalar,
           MatrixFormat::Dense>& B,
           Scalar alpha, Side side, Uplo uplo, Transpose transA, Diag diag) {
    const auto mask = batchlas::detail::call_mask("trsm", A, B);
    trsm_validate_params(A, B, side, uplo, transA, diag);
    // An empty batch is a no-op under any pin, as gemm's is: it never reaches selection.
    if (A.batch_size() == 0) return ctx.create_event_after_external_work();
    // alpha == 0 is B := 0 reading neither A nor B (reference BLAS): one masked fill, no solve, since
    // the kernels form alpha * B. A refused mask still refuses; a heterogeneous batch has no route.
    if (alpha == Scalar(0) && !A.is_heterogeneous() && !B.is_heterogeneous()) {
        if (!mask.empty() && !select::device_of<Back>(ctx, ops::trsm::spec.vendor).is_gpu)
            batchlas::detail::refuse_batch_mask<Scalar>(Op::trsm, "trsm", Back, A, B);
        if (B.rows() == 0 || B.cols() == 0) return ctx.create_event_after_external_work();
        return (mask.empty() ? B : B.with_batch_mask(mask)).fill_zeros(ctx);
    }
    // The coverage row's key: m, n are B's extents and k the triangular order, as before.
    const coverage::Shape shape{.m = B.rows(), .n = B.cols(), .k = A.rows(), .batch = A.batch_size(),
                                .transA = transA, .uplo = uplo, .side = side, .diag = diag};
    const select::Key key = ops::trsm::key_of<Scalar>(A, B, side, transA);
    return select::run<Back, Scalar>(
        ops::trsm::spec, ctx, key, ops::trsm::candidates<Scalar>(),
        [&](const auto& c, const auto& d) { return ops::trsm::can_run<Scalar>(c, d, A, B, side); }, shape, key,
        [&](const auto& c) { return ops::trsm::launch<Back, Scalar>(ctx, c, A, B, alpha, side, uplo, transA, diag); });
}

#define TRSM_INSTANTIATE(B_, fp) BATCHLAS_INSTANTIATE_OP(B_, fp, trsm)
BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(TRSM_INSTANTIATE)
#undef TRSM_INSTANTIATE

}  // namespace batchlas
