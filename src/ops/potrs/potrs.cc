// potrs: B := A^-1 B from a Cholesky factor. No Op of its own (include/batchlas/blas/functions/potrs.hh):
// the fused solve where it fits, else two public trsm calls, each of which selects its own kernel.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/potrs.hh>

#include <src/ops/potrs/potrs.hh>
#include <src/select/select.hh>
#include <src/util/template-instantiations.hh>

#include <string>

namespace batchlas {

template <Backend Back, typename T>
Event potrs(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, const MatrixView<T, MatrixFormat::Dense>& B,
            Uplo uplo) {
    potrs_validate_params<T>(A, B, uplo);
    if (A.batch_size() == 0) return ctx.create_event_after_external_work();
    // The kernels have no per-item extents: a heterogeneous batch would be solved at the storage order.
    if (A.is_heterogeneous() || B.is_heterogeneous())
        throw batchlas::internal_error("potrs: a heterogeneous batch (per-item active dims on " +
                                       std::string(A.is_heterogeneous() ? "A" : "B") + "), which no potrs kernel serves.");
    if (A.rows() < 1 || B.cols() < 1)
        throw batchlas::internal_error("potrs: n = " + std::to_string(A.rows()) + ", nrhs = " +
                                       std::to_string(B.cols()) + " is an empty problem, which no potrs kernel serves.");
    const bool fused = ops::potrs::fused_fits<T>(select::describe(ctx.device(), Back, false), A.rows(), B.cols());
    // The two solves chain, so they need an in-order queue.
    return select::on_in_order_queue(ctx, [&](Queue& q) { return ops::potrs::solve<Back, T>(q, A, B, uplo, fused); });
}

#define POTRS_INSTANTIATE(B_, fp) BATCHLAS_INSTANTIATE_OP(B_, fp, potrs)
BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(POTRS_INSTANTIATE)
#undef POTRS_INSTANTIATE

}  // namespace batchlas
