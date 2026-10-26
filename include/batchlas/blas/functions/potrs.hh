#pragma once

/// @file
/// @brief Batched solve with a Cholesky factor (potrs).
/// @ingroup api_factorizations

// No Op and no BATCHLAS_POTRS_ROUTE: the two families are the fused one-launch solve and two
// routed trsm calls, and the tie between them is a capacity predicate (src/ops/potrs/potrs.hh),
// the same one posv's `cta` and `blocked` choices pin. evidence: docs/perf/potri.md#potrs-and-potri

#include <batchlas/export.hh>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/queue-dispatch.hh>

namespace batchlas {

// Signature alias for explicit instantiation; see BATCHLAS_INSTANTIATE in
// src/util/template-instantiations.hh. Keep in sync with the declaration below.
namespace sig {
template <typename T>
using potrs = Event(Queue&,
                    const MatrixView<T, MatrixFormat::Dense>&,
                    const MatrixView<T, MatrixFormat::Dense>&,
                    Uplo);
}  // namespace sig

/// @brief Validates the arguments of potrs().
///
/// Checks non-negative extents, a square A, `B.rows() == A.rows()`, equal batch sizes and a
/// valid @p uplo.
/// @throws batchlas::invalid_argument if any check fails
/// @ingroup api_factorizations_lowlevel
template <typename T>
inline void potrs_validate_params(const MatrixView<T, MatrixFormat::Dense>& A,
                                  const MatrixView<T, MatrixFormat::Dense>& B,
                                  Uplo uplo) {
    if (A.rows() < 0 || A.cols() < 0 || B.rows() < 0 || B.cols() < 0) {
        throw batchlas::invalid_argument(
            "potrs: Matrix dimensions cannot be negative (A: rows=" + std::to_string(A.rows()) +
            ", cols=" + std::to_string(A.cols()) + "; B: rows=" + std::to_string(B.rows()) +
            ", cols=" + std::to_string(B.cols()) + ")");
    }
    if (A.rows() != A.cols()) {
        throw batchlas::invalid_argument(
            "potrs: A must be square, got " + std::to_string(A.rows()) + "x" + std::to_string(A.cols()));
    }
    if (A.rows() != B.rows()) {
        throw batchlas::invalid_argument(
            "potrs: B must have A.rows() rows, got " + std::to_string(B.rows()) + " against " +
            std::to_string(A.rows()));
    }
    if (A.batch_size() != B.batch_size()) {
        throw batchlas::invalid_argument(
            "potrs: A and B must share a batch size, got " + std::to_string(A.batch_size()) + " and " +
            std::to_string(B.batch_size()));
    }
    if (uplo != Uplo::Lower && uplo != Uplo::Upper) {
        throw batchlas::invalid_argument("potrs: Invalid uplo parameter: " + std::to_string(static_cast<int>(uplo)));
    }
}

/// @brief Batched solve of \f$ A X = B \f$ from the Cholesky factor of a Hermitian positive-definite A.
///
/// @p A holds the factor written by potrf() or posv() in the @p uplo triangle
/// (\f$ A = L L^H \f$ for Lower, \f$ U^H U \f$ for Upper); the other triangle is not read. On
/// return B holds X and A is unchanged. A factor is reusable across calls, so a caller with many
/// right-hand-side sets factors once and calls this per set. The result is bit-identical to the
/// solve half of posv() on its `cta` choice where the fused solve fits and on its `blocked`
/// choice elsewhere. No workspace and no per-item status: an item whose factor came from a
/// failed potrf() has a meaningless X, and the other items are unaffected.
///
/// Asynchronous: B is readable after the returned event is waited on.
/// @tparam Back  backend; the backend-deducing overload takes it from `ctx.backend()`
/// @tparam T     scalar type (float, double, std::complex<float>, std::complex<double>)
/// @param ctx   queue the kernels are enqueued on
/// @param A     batch of n x n Cholesky factors, read-only
/// @param B     batch of n x nrhs right-hand sides; overwritten with X
/// @param uplo  triangle of A that holds the factor
/// @return event of the last enqueued kernel
/// @throws batchlas::invalid_argument if potrs_validate_params() rejects the call
/// @throws batchlas::internal_error for n, nrhs or batch below 1 (batch 0 is a no-op) and for a heterogeneous batch
/// @ingroup api_factorizations
template <Backend Back, typename T>
BATCHLAS_API Event potrs(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         const MatrixView<T, MatrixFormat::Dense>& B,
                         Uplo uplo);

BATCHLAS_DISPATCH_ON_QUEUE(potrs)

}  // namespace batchlas
