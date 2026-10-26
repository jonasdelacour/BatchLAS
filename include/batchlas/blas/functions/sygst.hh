// SPDX-License-Identifier: MIT
#pragma once

/// @file
/// @brief Batched reduction of a generalized Hermitian-definite eigenproblem to standard form (sygst / hegst).
/// @ingroup api_eigen

// No cuSOLVER family: cuSOLVER 13.2 has no public ?sygst. The vendor family is LAPACKE on the host.
// evidence: docs/perf/sygvd.md#sygvd-routes

#include <batchlas/export.hh>
#include <cstdint>
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
using sygst = Event(Queue&, const MatrixView<T, MatrixFormat::Dense>&, const MatrixView<T, MatrixFormat::Dense>&,
                    int, Uplo, Span<std::byte>);
template <typename T>
using sygst_buffer_size = size_t(Queue&, const MatrixView<T, MatrixFormat::Dense>&,
                                 const MatrixView<T, MatrixFormat::Dense>&, int, Uplo);
template <typename T>
using sygst_vendor = sygst<T>;
}  // namespace sig

/// @brief Validates the arguments of sygst() and sygst_buffer_size().
///
/// Checks square A and B of the same order and batch size, `itype` in {1, 2, 3} and a valid @p uplo.
/// @throws batchlas::invalid_argument naming the first violated requirement
/// @ingroup api_eigen_lowlevel
template <typename T>
inline void sygst_validate_params(const char* who, const MatrixView<T, MatrixFormat::Dense>& A,
                                  const MatrixView<T, MatrixFormat::Dense>& B, int itype, Uplo uplo) {
    const std::string w(who);
    if (A.rows() < 0 || A.rows() != A.cols())
        throw batchlas::invalid_argument(w + ": A must be square, got " + std::to_string(A.rows()) + "x" +
                                         std::to_string(A.cols()));
    if (B.rows() != A.rows() || B.cols() != A.rows())
        throw batchlas::invalid_argument(w + ": B must be " + std::to_string(A.rows()) + "x" +
                                         std::to_string(A.rows()) + ", got " + std::to_string(B.rows()) + "x" +
                                         std::to_string(B.cols()));
    if (B.batch_size() != A.batch_size())
        throw batchlas::invalid_argument(w + ": A and B batch sizes differ (" + std::to_string(A.batch_size()) +
                                         " vs " + std::to_string(B.batch_size()) + ")");
    if (itype < 1 || itype > 3)
        throw batchlas::invalid_argument(w + ": itype must be 1, 2 or 3, got " + std::to_string(itype));
    if (uplo != Uplo::Lower && uplo != Uplo::Upper)
        throw batchlas::invalid_argument(w + ": invalid uplo " + std::to_string(static_cast<int>(uplo)));
}

/// @brief Batched reduction of \f$ A x = \lambda B x \f$ (and its two relatives) to a standard
///        Hermitian eigenproblem, from the Cholesky factor of B (LAPACK `?sygst` / `?hegst`).
///
/// On entry the @p uplo triangle of @p A holds the Hermitian A (the other triangle is not read) and
/// the @p uplo triangle of @p B holds potrf()'s factor of B (\f$ B = L L^H \f$ or \f$ U^H U \f$).
/// On return the @p uplo triangle of A holds
///   - itype 1: \f$ L^{-1} A L^{-H} \f$ (Lower) or \f$ U^{-H} A U^{-1} \f$ (Upper);
///   - itype 2 and 3: \f$ L^H A L \f$ (Lower) or \f$ U A U^H \f$ (Upper).
/// The other triangle of A is overwritten with unspecified values; B is read-only.
///
/// Asynchronous: A is readable after the returned event is waited on.
/// @tparam Back  backend; the backend-deducing overload takes it from `ctx.backend()`
/// @tparam T     float, double, std::complex<float> or std::complex<double>
/// @param ctx         queue the work is enqueued on
/// @param A           batch of n x n Hermitian matrices, overwritten with the reduced matrix
/// @param B           batch of n x n Cholesky factors (any `ld >= n`, any batch stride)
/// @param itype       1: \f$ A x = \lambda B x \f$; 2: \f$ A B x = \lambda x \f$; 3: \f$ B A x = \lambda x \f$
/// @param uplo        triangle of A and B that holds the data
/// @param work_space  device-accessible scratch of at least sygst_buffer_size() bytes
/// @return event of the last enqueued kernel
/// @throws batchlas::invalid_argument if sygst_validate_params() rejects the call
/// @throws batchlas::workspace_error if @p work_space is shorter than sygst_buffer_size()
/// @throws batchlas::NoRouteError when no family can run the call in this build
/// @see sygvd(), @ref perf_sygvd
/// @ingroup api_eigen
template <Backend Back, typename T>
BATCHLAS_API Event sygst(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                         const MatrixView<T, MatrixFormat::Dense>& B, int itype, Uplo uplo,
                         Span<std::byte> work_space);

/// @brief Workspace, in bytes, that sygst() needs for these operands (0 for itype 1 on the
///        native family; one n x n product per item for itypes 2 and 3).
/// @ingroup api_eigen_lowlevel
template <Backend Back, typename T>
BATCHLAS_API size_t sygst_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                                      const MatrixView<T, MatrixFormat::Dense>& B, int itype, Uplo uplo);

/// @brief The complex-Hermitian spelling of sygst() (LAPACK `?hegst`); identical semantics, and the same
///        backend-deducing overload `hegst(ctx, ...)`.
/// @ingroup api_eigen
template <Backend Back, typename T>
inline Event hegst(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                   const MatrixView<T, MatrixFormat::Dense>& B, int itype, Uplo uplo, Span<std::byte> work_space) {
    return sygst<Back, T>(ctx, A, B, itype, uplo, work_space);
}

BATCHLAS_DISPATCH_ON_QUEUE(sygst)
BATCHLAS_DISPATCH_ON_QUEUE(sygst_buffer_size)
BATCHLAS_DISPATCH_ON_QUEUE(hegst)

}  // namespace batchlas

namespace batchlas::backend {

/// @brief LAPACKE `?sygst` / `?hegst` per item, sygst()'s `vendor` family on the host backend.
/// @ingroup api_dispatch
template <Backend B, typename T>
BATCHLAS_API Event sygst_vendor(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                                const MatrixView<T, MatrixFormat::Dense>& Bm, int itype, Uplo uplo,
                                Span<std::byte> work_space);

}  // namespace batchlas::backend
