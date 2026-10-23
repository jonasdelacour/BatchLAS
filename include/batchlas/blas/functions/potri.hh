#pragma once

/// @file
/// @brief Batched inverse of a Hermitian positive-definite matrix from its Cholesky factor (potri).
/// @ingroup api_factorizations

// No vendor family: cuSOLVER and rocSOLVER have no batched potri, so every route is native and
// vendor-free. evidence: docs/perf/potri.md#potrs-and-potri

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

// Signature aliases for explicit instantiation; see BATCHLAS_INSTANTIATE in
// src/util/template-instantiations.hh. Keep in sync with the declarations below.
namespace sig {
template <typename T>
using potri = Event(Queue&,
                    const MatrixView<T, MatrixFormat::Dense>&,
                    Uplo, bool, Span<std::byte>, Span<int32_t>);

template <typename T>
using potri_buffer_size = size_t(Queue&,
                                 const MatrixView<T, MatrixFormat::Dense>&,
                                 Uplo);
}  // namespace sig

/// @brief Validates the arguments of potri() and potri_buffer_size().
///
/// Checks non-negative extents, a square A and a valid @p uplo.
/// @throws batchlas::invalid_argument if any check fails
/// @ingroup api_factorizations_lowlevel
template <typename T>
inline void potri_validate_params(const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo) {
    if (A.rows() < 0 || A.cols() < 0) {
        throw batchlas::invalid_argument(
            "POTRI: Matrix dimensions cannot be negative (A: rows=" + std::to_string(A.rows()) +
            ", cols=" + std::to_string(A.cols()) + ")");
    }
    if (A.rows() != A.cols()) {
        throw batchlas::invalid_argument(
            "POTRI: A must be square, got " + std::to_string(A.rows()) + "x" + std::to_string(A.cols()));
    }
    if (uplo != Uplo::Lower && uplo != Uplo::Upper) {
        throw batchlas::invalid_argument("POTRI: Invalid uplo parameter: " + std::to_string(static_cast<int>(uplo)));
    }
}

/// @brief Batched inverse \f$ A^{-1} \f$ of a Hermitian positive-definite A from its Cholesky factor.
///
/// LAPACK `?potri` semantics: on entry the @p uplo triangle of A holds the factor written by
/// potrf() or posv(); on return it holds the @p uplo triangle of \f$ A^{-1} \f$. With
/// @p full = false the other triangle is neither read nor written; with @p full = true it
/// receives the conjugate transpose, so the result is exactly Hermitian (a real diagonal is not
/// forced: it carries what the solve computed). The inverse is formed by solving the factored
/// system against an identity held in the workspace, so the cost is that of a posv with
/// nrhs = n minus the factorization.
///
/// Asynchronous: A and @p info are readable after the returned event is waited on.
/// @tparam Back  backend; the backend-deducing overload takes it from `ctx.backend()`
/// @tparam T     scalar type (float, double, std::complex<float>, std::complex<double>)
/// @param ctx         queue the kernels are enqueued on
/// @param A           batch of n x n Cholesky factors; overwritten with the inverse
/// @param uplo        triangle of A that holds the factor and receives the inverse
/// @param full        also fill the other triangle with the conjugate transpose
/// @param work_space  device-accessible scratch of at least potri_buffer_size() bytes
/// @param info        per-item status as for LAPACK `?potri`: 0 on success, i > 0 if the factor's
///                    diagonal entry i is exactly zero (that item's inverse is meaningless, the
///                    others are unaffected). Empty span = not requested.
/// @return event of the last enqueued kernel
/// @throws batchlas::invalid_argument if potri_validate_params() rejects the call
/// @throws batchlas::internal_error for n below 1 (batch 0 is a no-op) and for a heterogeneous batch
/// @throws batchlas::workspace_error if @p work_space is smaller than potri_buffer_size()
/// @ingroup api_factorizations
template <Backend Back, typename T>
BATCHLAS_API Event potri(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         Uplo uplo,
                         bool full,
                         Span<std::byte> work_space,
                         Span<int32_t> info);

/// @brief Workspace, in bytes, that potri() needs for these operands: one n x n identity per item.
///
/// @throws batchlas::invalid_argument if potri_validate_params() rejects the operands
/// @ingroup api_factorizations_lowlevel
template <Backend Back, typename T>
BATCHLAS_API size_t potri_buffer_size(Queue& ctx,
                                      const MatrixView<T, MatrixFormat::Dense>& A,
                                      Uplo uplo);

/// @brief potri() writing only the @p uplo triangle.
/// @ingroup api_factorizations
// Not a defaulted `full`: the sig:: aliases are function types (see potrf.hh).
template <Backend Back, typename T>
inline Event potri(Queue& ctx,
                   const MatrixView<T, MatrixFormat::Dense>& A,
                   Uplo uplo,
                   Span<std::byte> work_space,
                   Span<int32_t> info) {
    return potri<Back, T>(ctx, A, uplo, false, work_space, info);
}

/// @brief potri() writing only the @p uplo triangle, without per-item status.
/// @ingroup api_factorizations
template <Backend Back, typename T>
inline Event potri(Queue& ctx,
                   const MatrixView<T, MatrixFormat::Dense>& A,
                   Uplo uplo,
                   Span<std::byte> work_space) {
    return potri<Back, T>(ctx, A, uplo, false, work_space, Span<int32_t>{});
}

BATCHLAS_DISPATCH_ON_QUEUE(potri)
BATCHLAS_DISPATCH_ON_QUEUE(potri_buffer_size)

}  // namespace batchlas
