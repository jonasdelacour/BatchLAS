#pragma once

/// @file
/// @brief Batched Hermitian positive-definite solve (posv) by Cholesky.
/// @ingroup factorizations

// No vendor arm and no `potrs` op: the composed arm is potrf then two routed
// trsm calls, and the documented contract below is exactly that composition.
// evidence: docs/perf/potrf.md#the-fused-posv-tier

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
using posv = Event(Queue&,
                   const MatrixView<T, MatrixFormat::Dense>&,
                   const MatrixView<T, MatrixFormat::Dense>&,
                   Uplo, Span<std::byte>, Span<int32_t>);

template <typename T>
using posv_buffer_size = size_t(Queue&,
                                const MatrixView<T, MatrixFormat::Dense>&,
                                const MatrixView<T, MatrixFormat::Dense>&,
                                Uplo);
}  // namespace sig

/// @brief Validates the arguments of posv() and posv_buffer_size().
///
/// Checks non-negative extents, a square A, `B.rows() == A.rows()`, equal batch
/// sizes and a valid @p uplo.
/// @throws batchlas::invalid_argument if any check fails
/// @ingroup factorizations
// Stricter than potrf's validator on purpose: with no vendor arm, a
// non-conforming pair would otherwise surface as a misleading "no route" error.
template <typename T>
inline void posv_validate_params(const MatrixView<T, MatrixFormat::Dense>& A,
                                 const MatrixView<T, MatrixFormat::Dense>& B,
                                 Uplo uplo) {
    if (A.rows() < 0 || A.cols() < 0 || B.rows() < 0 || B.cols() < 0) {
        throw batchlas::invalid_argument(
            "POSV: Matrix dimensions cannot be negative (A: rows=" +
            std::to_string(A.rows()) + ", cols=" + std::to_string(A.cols()) +
            "; B: rows=" + std::to_string(B.rows()) +
            ", cols=" + std::to_string(B.cols()) + ")");
    }
    if (A.rows() != A.cols()) {
        throw batchlas::invalid_argument(
            "POSV: A must be square, got " + std::to_string(A.rows()) + "x" +
            std::to_string(A.cols()));
    }
    if (A.rows() != B.rows()) {
        throw batchlas::invalid_argument(
            "POSV: B must have A.rows() rows, got " + std::to_string(B.rows()) +
            " against " + std::to_string(A.rows()));
    }
    if (A.batch_size() != B.batch_size()) {
        throw batchlas::invalid_argument(
            "POSV: A and B must share a batch size, got " +
            std::to_string(A.batch_size()) + " and " + std::to_string(B.batch_size()));
    }
    if (uplo != Uplo::Lower && uplo != Uplo::Upper) {
        throw batchlas::invalid_argument(
            "POSV: Invalid uplo parameter: " + std::to_string(static_cast<int>(uplo)));
    }
}

/// @brief Batched solve of \f$ A X = B \f$ for Hermitian positive-definite A.
///
/// Semantically `potrf(A, uplo)` followed by the two triangular solves with the
/// factor: on return A holds the Cholesky factor in the @p uplo triangle (the
/// other triangle is not read, and, as for potrf(), a vendor potrf leg may
/// overwrite it), B holds X, and @p info carries potrf's status. There is
/// no vendor library call behind this op on any backend; it is served by native
/// fused tiers or by the routed potrf + trsm composition.
///
/// Asynchronous: A, B and @p info are readable after the returned event is waited on.
/// @tparam Back  backend; the backend-deducing overload takes it from `ctx.backend()`
/// @tparam T     scalar type (float, double, std::complex<float>, std::complex<double>)
/// @param ctx         queue the kernels are enqueued on
/// @param A           batch of n x n matrices; overwritten with the Cholesky factor
/// @param B           batch of n x nrhs right-hand sides; overwritten with X
/// @param uplo        triangle of A that is read and receives the factor
/// @param work_space  device-accessible scratch of at least posv_buffer_size() bytes
/// @param info        per-item status as for potrf(): 0 on success, i > 0 if the
///                    leading minor of order i is not positive definite (that
///                    item's X is meaningless). Empty span = not requested.
/// @return event of the last enqueued kernel
/// @throws batchlas::invalid_argument if posv_validate_params() rejects the call
/// @ingroup factorizations
template <Backend Back, typename T>
BATCHLAS_API Event posv(Queue& ctx,
                        const MatrixView<T, MatrixFormat::Dense>& A,
                        const MatrixView<T, MatrixFormat::Dense>& B,
                        Uplo uplo,
                        Span<std::byte> work_space,
                        Span<int32_t> info);

/// @brief Workspace, in bytes, that posv() needs for these operands on this queue.
///
/// Takes the same operands as the call, so both resolve the same route.
/// @throws batchlas::invalid_argument if posv_validate_params() rejects the operands
/// @ingroup factorizations
template <Backend Back, typename T>
BATCHLAS_API size_t posv_buffer_size(Queue& ctx,
                                     const MatrixView<T, MatrixFormat::Dense>& A,
                                     const MatrixView<T, MatrixFormat::Dense>& B,
                                     Uplo uplo);

/// @brief posv() without per-item status (`info` not requested).
/// @ingroup factorizations
// Not a defaulted `info`: the sig:: aliases are function types (see potrf.hh).
template <Backend Back, typename T>
inline Event posv(Queue& ctx,
                  const MatrixView<T, MatrixFormat::Dense>& A,
                  const MatrixView<T, MatrixFormat::Dense>& B,
                  Uplo uplo,
                  Span<std::byte> work_space) {
    return posv<Back, T>(ctx, A, B, uplo, work_space, Span<int32_t>{});
}

BATCHLAS_ACCEPT_OWNING(posv)
BATCHLAS_ACCEPT_OWNING(posv_buffer_size)

BATCHLAS_DISPATCH_ON_QUEUE(posv)
BATCHLAS_DISPATCH_ON_QUEUE(posv_buffer_size)

}  // namespace batchlas
