// potri: the inverse from a Cholesky factor. select::run takes the first entry of the nearest
// tuned/potri.<device>.txt row that can_run() admits. Both families solve A X = I with the factor
// (ops::potrs::solve, the same solve as posv's cta and blocked) on an identity held in the workspace,
// then store X into A's triangle. cuSOLVER has no batched potri, so there is no vendor family.
// evidence: docs/perf/potri.md#potri-selection

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/potri.hh>

#include <batchlas/util/mempool.hh>

#include <src/ops/potri/choice.hh>
#include <src/ops/potri/potri_native.hh>
#include <src/ops/potrs/potrs.hh>
#include <src/select/select.hh>
#include <src/util/template-instantiations.hh>

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

// Correctness only: false means the launch would throw. Blocked is the children's own business
// (trsm refuses what it cannot take), except a heterogeneous batch, which no family serves.
template <class T>
bool can_run(const PotriChoice& c, const select::Device& d, const MV<T>& A) {
    const bool homogeneous = !A.is_heterogeneous() && A.rows() >= 1 && A.batch_size() >= 1;
    return std::visit(overloaded{
        [&](Fused) { return homogeneous && potrs::fused_fits<T>(d, A.rows(), A.rows()); },
        [&](Blocked) { return homogeneous; },
    }, c);
}

// n x n identity per item. Sized with the measuring pool, never by hand (BumpAllocator's alignment trap).
template <class T>
std::size_t workspace(Queue& q, const MV<T>& A) {
    auto pool = BumpAllocator::measuring();
    (void)pool.allocate<T>(q, static_cast<std::size_t>(A.rows()) * A.rows() * A.batch_size());
    (void)pool.allocate<T*>(q, static_cast<std::size_t>(A.batch_size()));
    return pool.required_bytes();
}

template <Backend B, class T>
Event launch(Queue& q, const PotriChoice& c, const MV<T>& A, Uplo uplo, bool full, Span<std::byte> ws,
             Span<int32_t> info) {
    const int n = static_cast<int>(A.rows());
    const int batch = static_cast<int>(A.batch_size());
    BumpAllocator pool(ws);
    const Span<T> buf = pool.allocate<T>(q, static_cast<std::size_t>(n) * n * batch);
    // The vendor trsm loops read the batch's pointer array: X owns one in the same workspace.
    const Span<T*> xptr = pool.allocate<T*>(q, static_cast<std::size_t>(batch));
    const MV<T> X(buf.data(), n, n, n, n * n, batch, xptr.data());
    // info reads the factor before the store overwrites it, so it is enqueued first.
    if (info.size() >= static_cast<std::size_t>(batch)) (void)sycl_potri::potri_diag_info<T>(q, A, info);
    (void)X.fill_identity(q);
    (void)std::visit(overloaded{
        [&](Fused) { return potrs::solve<B, T>(q, A, X, uplo, true); },
        [&](Blocked) { return potrs::solve<B, T>(q, A, X, uplo, false); },
    }, c);
    return sycl_potri::potri_store<T>(q, X, A, uplo, full);
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
    if (work_space.size() < ops::potri::workspace<T>(ctx, A))
        throw batchlas::workspace_error("potri: workspace is shorter than potri_buffer_size()");
    const coverage::Shape shape{.m = A.rows(), .n = A.rows(), .k = A.rows(), .batch = A.batch_size(), .uplo = uplo};
    return select::run<Back, T>(
        ops::potri::spec, ctx, ops::potri::key_of<T>(A), ops::potri::candidates<T>(),
        [&](const auto& c, const auto& d) { return ops::potri::can_run<T>(c, d, A); }, shape,
        {{"n", A.rows()}, {"batch", A.batch_size()}},
        [&](const auto& c) {
            // The identity fill, both solves and the store chain, so they need an in-order queue.
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
    return ops::potri::workspace<T>(ctx, A);
}

#define POTRI_INSTANTIATE(B_, fp) BATCHLAS_INSTANTIATE_OP(B_, fp, potri) BATCHLAS_INSTANTIATE_OP(B_, fp, potri_buffer_size)
BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(POTRI_INSTANTIATE)
#undef POTRI_INSTANTIATE

}  // namespace batchlas
