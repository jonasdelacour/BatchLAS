#pragma once

// POSV: solve A X = B for Hermitian positive-definite A, by Cholesky, as one op.
//
// NO VENDOR ARM, and no `potrs` op in this library either, so the composed arm is
// `potrf` then two routed `trsm` calls -- and THAT IS THE CONTRACT: A comes back
// holding the Cholesky factor in the triangle `uplo` names, the other triangle
// neither read nor written, B holding X, `info` carrying potrf's leading-minor
// status. evidence: docs/perf/potrf.md#the-fused-posv-tier

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

// As in gesv.hh, this validator rejects rather than routes: with no vendor arm a
// non-conforming pair would otherwise reach throw_no_vendor_route and report the
// wrong cause.
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

// A is overwritten by its Cholesky factor, B by the solution X. An EMPTY `info`
// span means "not requested".
template <Backend Back, typename T>
BATCHLAS_API Event posv(Queue& ctx,
                        const MatrixView<T, MatrixFormat::Dense>& A,
                        const MatrixView<T, MatrixFormat::Dense>& B,
                        Uplo uplo,
                        Span<std::byte> work_space,
                        Span<int32_t> info);

template <Backend Back, typename T>
BATCHLAS_API size_t posv_buffer_size(Queue& ctx,
                                     const MatrixView<T, MatrixFormat::Dense>& A,
                                     const MatrixView<T, MatrixFormat::Dense>& B,
                                     Uplo uplo);

// Old-arity forwarder; see the note in gesv.hh on why `info` cannot be defaulted.
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
