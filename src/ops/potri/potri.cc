// SPDX-License-Identifier: MIT
// potri: the inverse from a Cholesky factor by LAPACK's algorithm, trtri then lauum, in place on
// A's uplo triangle. select::run takes the first entry of the nearest tuned/potri.<device>.txt row
// that can_run() admits: cta (both steps in one local-memory-resident launch) or blocked:nb (the
// CTA kernel on nb x nb diagonal blocks, public trmm, gemm and herk/syrk between them). cuSOLVER
// has no batched potri, so there is no vendor family. evidence: docs/perf/potri.md#potri-selection

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/potri.hh>

#include <batchlas/util/mempool.hh>

#include <src/ops/potri/choice.hh>
#include <src/ops/potri/potri_native.hh>
#include <src/select/select.hh>
#include <src/util/batch_mask.hh>
#include <src/util/template-instantiations.hh>

#include <algorithm>
#include <cstddef>
#include <string>
#include <variant>

namespace batchlas {
namespace ops::potri {

using select::overloaded;

template <class Scalar>
using MV = MatrixView<Scalar, MatrixFormat::Dense>;

template <class Scalar>
select::Key key_of(const MV<Scalar>& A) {
    return {{"n", A.rows()}, {"batch", A.batch_size()}};
}

template <class Scalar>
int cta_ceiling(const select::Device& d) {
    return ops::potri::cta_max_n<Scalar>(static_cast<std::size_t>(d.slm_budget), d.max_wg);
}

// Correctness only: false means the launch would throw. Both families need a homogeneous batch and
// a CTA block (the whole matrix, or one nb x nb diagonal block) within the device's local memory.
template <class Scalar>
bool can_run(const PotriChoice& c, const select::Device& d, const MV<Scalar>& A) {
    // cta honours a batch mask; blocked runs its children (trmm, syrk/herk) on the masked views.
    const bool homogeneous = !A.is_heterogeneous() && A.rows() >= 1 && A.batch_size() >= 1 && d.max_wg >= 1;
    const int n = A.rows();
    return std::visit(overloaded{
        [&](Cta) { return homogeneous && n <= cta_ceiling<Scalar>(d); },
        [&](Blocked b) { return homogeneous && b.nb >= 1 && std::min(b.nb, n) <= cta_ceiling<Scalar>(d); },
    }, c);
}

// Exactly the chosen family's need: cta none, blocked one part's W and slots plus the batch's flags.
template <class Scalar>
std::size_t workspace(Queue& q, const PotriChoice& c, const MV<Scalar>& A) {
    return std::visit(overloaded{
        [&](Cta) -> std::size_t { return 0; },
        [&](Blocked b) { return blocked::workspace<Scalar>(q, A, b.nb); },
    }, c);
}

template <Backend B, class Scalar>
Event launch(Queue& q, const PotriChoice& c, const MV<Scalar>& A, Uplo uplo, bool full, Span<std::byte> ws,
             Span<int32_t> info) {
    if (ws.size() < workspace<Scalar>(q, c, A))
        throw batchlas::workspace_error("potri: workspace is shorter than potri_buffer_size()");
    return std::visit(overloaded{
        [&](Cta) { return ops::potri::potri_cta<Scalar>(q, A, uplo, ops::potri::CtaMode::Potri, A.rows(), full,
                                                        info); },
        [&](Blocked b) { return blocked::run<B, Scalar>(q, A, uplo, b.nb, full, ws, info); },
    }, c);
}

// n below 1 (batch 0 returned earlier, a no-op) or a heterogeneous batch has no kernel.
template <class Scalar>
void throw_if_unservable(const MV<Scalar>& A, const char* who) {
    if (A.is_heterogeneous())
        throw batchlas::internal_error(std::string(who) +
                                       ": a heterogeneous batch (per-item active dims), which no potri kernel serves.");
    if (A.rows() >= 1) return;
    throw batchlas::internal_error(std::string(who) + ": n = " + std::to_string(A.rows()) +
                                   " is an empty problem, which no potri kernel serves.");
}

}  // namespace ops::potri

template <Backend Back, typename Scalar>
Event potri(Queue& ctx, const MatrixView<Scalar, MatrixFormat::Dense>& A, Uplo uplo, bool full,
            Span<std::byte> work_space,
            Span<int32_t> info) {
    potri_validate_params<Scalar>(A, uplo);
    (void)detail::call_mask("potri", A);
    if (A.batch_size() == 0) return ctx.create_event_after_external_work();
    ops::potri::throw_if_unservable<Scalar>(A, "potri");
    if (!info.empty() && info.size() < static_cast<std::size_t>(A.batch_size()))
        throw batchlas::invalid_argument("potri: info has " + std::to_string(info.size()) + " entries for a batch of " +
                                         std::to_string(A.batch_size()));
    const coverage::Shape shape{.m = A.rows(), .n = A.rows(), .k = A.rows(), .batch = A.batch_size(), .uplo = uplo};
    return select::run<Back, Scalar>(
        ops::potri::spec, ctx, ops::potri::key_of<Scalar>(A), ops::potri::candidates<Scalar>(),
        [&](const auto& c, const auto& d) { return ops::potri::can_run<Scalar>(c, d, A); }, shape,
        {{"n", A.rows()}, {"batch", A.batch_size()}},
        [&](const auto& c) {
            // blocked's children chain through A and W, so they need an in-order queue.
            return select::on_in_order_queue(ctx, [&](Queue& q) {
                return ops::potri::launch<Back, Scalar>(q, c, A, uplo, full, work_space, info);
            });
        });
}

template <Backend Back, typename Scalar>
size_t potri_buffer_size(Queue& ctx, const MatrixView<Scalar, MatrixFormat::Dense>& A, Uplo uplo) {
    potri_validate_params<Scalar>(A, uplo);
    (void)detail::call_mask("potri_buffer_size", A);
    if (A.batch_size() == 0) return 0;
    ops::potri::throw_if_unservable<Scalar>(A, "potri_buffer_size");
    const auto c = select::pick<Back, Scalar>(
        ops::potri::spec, ctx, ops::potri::key_of<Scalar>(A), ops::potri::candidates<Scalar>(),
        [&](const auto& k, const auto& d) { return ops::potri::can_run<Scalar>(k, d, A); });
    return ops::potri::workspace<Scalar>(ctx, c, A);
}

#define POTRI_INSTANTIATE(B_, fp) BATCHLAS_INSTANTIATE_OP(B_, fp, potri) BATCHLAS_INSTANTIATE_OP(B_, fp, potri_buffer_size)
BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(POTRI_INSTANTIATE)
#undef POTRI_INSTANTIATE

}  // namespace batchlas
