// SPDX-License-Identifier: MIT
// potrs: B := A^-1 B from a Cholesky factor. select::run takes the first entry of the nearest
// tuned/potrs.<device>.txt row that can_run() admits: `fused` solves both triangles in one launch with
// B resident in local memory, `trsm` calls the public trsm twice (each picks its own kernel). Both are
// ops::potrs::solve, the solve posv's cta and blocked and potri pin. evidence: docs/perf/potrs.md#potrs-selection

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/potrs.hh>

#include <src/ops/potrs/choice.hh>
#include <src/ops/potrs/potrs.hh>
#include <src/select/select.hh>
#include <src/util/batch_mask.hh>
#include <src/util/template-instantiations.hh>

#include <string>
#include <variant>

namespace batchlas {
namespace ops::potrs {

using select::overloaded;

template <class T>
using MV = MatrixView<T, MatrixFormat::Dense>;

template <class T>
select::Key key_of(const MV<T>& A, const MV<T>& B, Uplo uplo) {
    return {{"uplo", uplo == Uplo::Lower ? "L" : "U"}, {"n", A.rows()}, {"nrhs", B.cols()}, {"batch", A.batch_size()}};
}

// Correctness only: false means the launch would throw. The trsm pair is its children's business
// (trsm refuses what it cannot take); a heterogeneous batch was refused before selection.
template <class T>
bool can_run(const PotrsChoice& c, const select::Device& d, const MV<T>& A, const MV<T>& B) {
    // Both honour a batch mask: the fused kernel skips an inactive item (I1), trsm does itself.
    return std::visit(overloaded{
        [&](Fused) { return fused_fits<T>(d, A.rows(), B.cols()); },
        [&](Trsm) { return true; },
    }, c);
}

// n or nrhs below 1 (batch 0 returned earlier, a no-op) or a heterogeneous batch has no kernel, so
// both throw before selection: neither Auto nor a pin can take them.
template <class T>
void throw_if_unservable(const MV<T>& A, const MV<T>& B) {
    // The kernels have no per-item extents: a heterogeneous batch would be solved at the storage order.
    if (A.is_heterogeneous() || B.is_heterogeneous())
        throw batchlas::internal_error("potrs: a heterogeneous batch (per-item active dims on " +
                                       std::string(A.is_heterogeneous() ? "A" : "B") + "), which no potrs kernel serves.");
    if (A.rows() < 1 || B.cols() < 1)
        throw batchlas::internal_error("potrs: n = " + std::to_string(A.rows()) + ", nrhs = " +
                                       std::to_string(B.cols()) + " is an empty problem, which no potrs kernel serves.");
}

}  // namespace ops::potrs

template <Backend Back, typename T>
Event potrs(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, const MatrixView<T, MatrixFormat::Dense>& B,
            Uplo uplo) {
    potrs_validate_params<T>(A, B, uplo);
    (void)detail::call_mask("potrs", A, B);
    if (A.batch_size() == 0) return ctx.create_event_after_external_work();
    ops::potrs::throw_if_unservable<T>(A, B);
    // The coverage row's key: m = k = order, n = nrhs, and uplo (the field separating rows).
    const coverage::Shape shape{.m = A.rows(), .n = B.cols(), .k = A.rows(), .batch = A.batch_size(), .uplo = uplo};
    return select::run<Back, T>(
        ops::potrs::spec, ctx, ops::potrs::key_of<T>(A, B, uplo), ops::potrs::candidates<T>(),
        [&](const auto& c, const auto& d) { return ops::potrs::can_run<T>(c, d, A, B); }, shape,
        {{"n", A.rows()}, {"nrhs", B.cols()}, {"batch", A.batch_size()}},
        [&](const auto& c) {
            const bool fused = std::holds_alternative<ops::potrs::Fused>(c);
            // The two solves of `trsm` chain, so they need an in-order queue.
            return select::on_in_order_queue(ctx, [&](Queue& q) { return ops::potrs::solve<Back, T>(q, A, B, uplo, fused); });
        });
}

#define POTRS_INSTANTIATE(B_, fp) BATCHLAS_INSTANTIATE_OP(B_, fp, potrs)
BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(POTRS_INSTANTIATE)
#undef POTRS_INSTANTIATE

}  // namespace batchlas
