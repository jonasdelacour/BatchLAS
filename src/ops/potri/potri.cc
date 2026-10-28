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
#include <src/util/template-instantiations.hh>

#include <algorithm>
#include <cstddef>
#include <string>
#include <variant>

namespace batchlas {
namespace ops::potri {

using select::overloaded;

template <class T>
using MV = MatrixView<T, MatrixFormat::Dense>;

template <class T>
select::Key key_of(const MV<T>& A) {
    return {{"n", A.rows()}, {"batch", A.batch_size()}};
}

template <class T>
int cta_ceiling(const select::Device& d) {
    return sycl_potri::cta_max_n<T>(static_cast<std::size_t>(d.slm_budget), d.max_wg);
}

// Correctness only: false means the launch would throw. Both families need a homogeneous batch and
// a CTA block (the whole matrix, or one nb x nb diagonal block) within the device's local memory.
template <class T>
bool can_run(const PotriChoice& c, const select::Device& d, const MV<T>& A) {
    const bool homogeneous = !A.is_heterogeneous() && A.rows() >= 1 && A.batch_size() >= 1 && d.max_wg >= 1;
    const int n = A.rows();
    return std::visit(overloaded{
        [&](Cta) { return homogeneous && n <= cta_ceiling<T>(d); },
        [&](Blocked b) { return homogeneous && b.nb >= 1 && std::min(b.nb, n) <= cta_ceiling<T>(d); },
    }, c);
}

// Exactly the chosen family's need: cta none, blocked one part's W and slots plus the batch's flags.
template <class T>
std::size_t workspace(Queue& q, const PotriChoice& c, const MV<T>& A) {
    return std::visit(overloaded{
        [&](Cta) -> std::size_t { return 0; },
        [&](Blocked b) { return blocked::workspace<T>(q, A, b.nb); },
    }, c);
}

template <Backend B, class T>
Event launch(Queue& q, const PotriChoice& c, const MV<T>& A, Uplo uplo, bool full, Span<std::byte> ws,
             Span<int32_t> info) {
    if (ws.size() < workspace<T>(q, c, A))
        throw batchlas::workspace_error("potri: workspace is shorter than potri_buffer_size()");
    return std::visit(overloaded{
        [&](Cta) { return sycl_potri::potri_cta<T>(q, A, uplo, sycl_potri::CtaMode::Potri, A.rows(), full, info); },
        [&](Blocked b) { return blocked::run<B, T>(q, A, uplo, b.nb, full, ws, info); },
    }, c);
}

// n below 1 (batch 0 returned earlier, a no-op) or a heterogeneous batch has no kernel.
template <class T>
void throw_if_unservable(const MV<T>& A, const char* who) {
    if (A.is_heterogeneous())
        throw batchlas::internal_error(std::string(who) +
                                       ": a heterogeneous batch (per-item active dims), which no potri kernel serves.");
    if (A.rows() >= 1) return;
    throw batchlas::internal_error(std::string(who) + ": n = " + std::to_string(A.rows()) +
                                   " is an empty problem, which no potri kernel serves.");
}

}  // namespace ops::potri

template <Backend Back, typename T>
Event potri(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo, bool full, Span<std::byte> work_space,
            Span<int32_t> info) {
    potri_validate_params<T>(A, uplo);
    if (A.batch_size() == 0) return ctx.create_event_after_external_work();
    ops::potri::throw_if_unservable<T>(A, "potri");
    if (!info.empty() && info.size() < static_cast<std::size_t>(A.batch_size()))
        throw batchlas::invalid_argument("potri: info has " + std::to_string(info.size()) + " entries for a batch of " +
                                         std::to_string(A.batch_size()));
    const coverage::Shape shape{.m = A.rows(), .n = A.rows(), .k = A.rows(), .batch = A.batch_size(), .uplo = uplo};
    return select::run<Back, T>(
        ops::potri::spec, ctx, ops::potri::key_of<T>(A), ops::potri::candidates<T>(),
        [&](const auto& c, const auto& d) { return ops::potri::can_run<T>(c, d, A); }, shape,
        {{"n", A.rows()}, {"batch", A.batch_size()}},
        [&](const auto& c) {
            // blocked's children chain through A and W, so they need an in-order queue.
            return select::on_in_order_queue(ctx, [&](Queue& q) {
                return ops::potri::launch<Back, T>(q, c, A, uplo, full, work_space, info);
            });
        });
}

template <Backend Back, typename T>
size_t potri_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo) {
    potri_validate_params<T>(A, uplo);
    if (A.batch_size() == 0) return 0;
    ops::potri::throw_if_unservable<T>(A, "potri_buffer_size");
    const auto c = select::pick<Back, T>(
        ops::potri::spec, ctx, ops::potri::key_of<T>(A), ops::potri::candidates<T>(),
        [&](const auto& k, const auto& d) { return ops::potri::can_run<T>(k, d, A); });
    return ops::potri::workspace<T>(ctx, c, A);
}

#define POTRI_INSTANTIATE(B_, fp) BATCHLAS_INSTANTIATE_OP(B_, fp, potri) BATCHLAS_INSTANTIATE_OP(B_, fp, potri_buffer_size)
BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(POTRI_INSTANTIATE)
#undef POTRI_INSTANTIATE

}  // namespace batchlas
