// SPDX-License-Identifier: MIT
#pragma once

/// @file
/// @brief Batched generalized Hermitian-definite eigensolver (sygvd / hegvd), and the entry that reuses a
///        factored B.
/// @ingroup api_eigen

#include <batchlas/export.hh>
#include <cstdint>

#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/queue-dispatch.hh>
#include <batchlas/blas/functions/sygst.hh>

namespace batchlas {

// Signature aliases for explicit instantiation; see BATCHLAS_INSTANTIATE in
// src/util/template-instantiations.hh. Keep in sync with the declarations below.
namespace sig {
template <typename T>
using sygvd = Event(Queue&, const MatrixView<T, MatrixFormat::Dense>&, const MatrixView<T, MatrixFormat::Dense>&,
                    Span<typename base_type<T>::type>, int, JobType, Uplo, Span<std::byte>, Span<int32_t>);
template <typename T>
using sygvd_buffer_size = size_t(Queue&, const MatrixView<T, MatrixFormat::Dense>&,
                                 const MatrixView<T, MatrixFormat::Dense>&, Span<typename base_type<T>::type>, int,
                                 JobType, Uplo);
template <typename T> using sygvd_factored = sygvd<T>;
template <typename T> using sygvd_factored_buffer_size = sygvd_buffer_size<T>;
template <typename T> using sygvd_vendor = sygvd<T>;
template <typename T> using sygvd_vendor_buffer_size = sygvd_buffer_size<T>;
}  // namespace sig

/// @brief Eigenvalues, and optionally eigenvectors, of a batch of generalized Hermitian-definite
///        problems (LAPACK `?sygvd` / `?hegvd`).
///
/// Per item solves, with B Hermitian positive definite,
///   - itype 1: \f$ A x = \lambda B x \f$, eigenvectors normalized \f$ X^H B X = I \f$;
///   - itype 2: \f$ A B x = \lambda x \f$, \f$ X^H B X = I \f$;
///   - itype 3: \f$ B A x = \lambda x \f$, \f$ X^H B^{-1} X = I \f$.
/// Only the @p uplo triangles of A and B are read. The family is the first entry of the nearest row of
/// `tuned/sygvd.<device>.txt` that can run the call: `compose` (potrf of B, sygst(), syev(), and a
/// trsm or trmm back-transform, each child selecting its own kernel) or `vendor` (cuSOLVER `?sygvd`
/// per item, LAPACKE `?sygvd` on the host). Pin with `BATCHLAS_SYGVD_ROUTE`.
///
/// Asynchronous: A, B, @p eigenvalues and @p info are readable after the returned event is waited on.
/// @param ctx          queue the work is enqueued on
/// @param A            batch of n x n Hermitian matrices; with `JobType::EigenVectors` overwritten by the
///                     eigenvectors (column j pairs with eigenvalue j), otherwise destroyed
/// @param B            batch of n x n Hermitian positive-definite matrices; overwritten by the Cholesky
///                     factor in its @p uplo triangle (the other triangle may be overwritten too)
/// @param eigenvalues  n real eigenvalues per item, packed, ascending
/// @param itype        1, 2 or 3 as above
/// @param jobz         `EigenVectors` or `NoEigenVectors`
/// @param uplo         triangle of A and B that holds the data
/// @param work_space   at least sygvd_buffer_size() bytes for the same arguments
/// @param info         per item, LAPACK's convention: 0 success; i in 1..n syev did not converge;
///                     n + i the leading minor of order i of B is not positive definite (A, B and the
///                     eigenvalues of that item are then unspecified; the other items are unaffected).
///                     An empty span means "not requested".
/// @return event of the last enqueued kernel
/// @throws batchlas::invalid_argument if sygst_validate_params() rejects A, B, itype or uplo
/// @throws batchlas::workspace_error if @p work_space is shorter than sygvd_buffer_size()
/// @throws batchlas::NoRouteError when no family can run the call in this build
/// @see sygvd_factored(), sygst(), syev(), @ref perf_sygvd
/// @ingroup api_eigen
template <Backend Back, typename T>
BATCHLAS_API Event sygvd(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                         const MatrixView<T, MatrixFormat::Dense>& B, Span<typename base_type<T>::type> eigenvalues,
                         int itype, JobType jobz, Uplo uplo, Span<std::byte> work_space, Span<int32_t> info);

/// @brief Workspace, in bytes, that sygvd() needs for the same arguments (it makes the same choices).
/// @ingroup api_eigen_lowlevel
template <Backend Back, typename T>
BATCHLAS_API size_t sygvd_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                                      const MatrixView<T, MatrixFormat::Dense>& B,
                                      Span<typename base_type<T>::type> eigenvalues, int itype, JobType jobz,
                                      Uplo uplo);

/// @brief sygvd() with B already factored: the @p uplo triangle of @p B holds potrf()'s Cholesky factor,
///        which is read and never written.
///
/// For a B that is fixed across many solves (an overlap matrix per geometry, solved again every SCC
/// iteration): factor it once with potrf() and call this each time. Always the `compose` family (no
/// vendor library takes a factored B): sygst(), syev() and the back-transform. `info` is syev()'s
/// (0, or i in 1..n); potrf()'s own `info` is where a non-positive-definite B shows.
/// @ingroup api_eigen
template <Backend Back, typename T>
BATCHLAS_API Event sygvd_factored(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                                  const MatrixView<T, MatrixFormat::Dense>& B,
                                  Span<typename base_type<T>::type> eigenvalues, int itype, JobType jobz, Uplo uplo,
                                  Span<std::byte> work_space, Span<int32_t> info);

/// @brief Workspace, in bytes, that sygvd_factored() needs for the same arguments.
/// @ingroup api_eigen_lowlevel
template <Backend Back, typename T>
BATCHLAS_API size_t sygvd_factored_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                                               const MatrixView<T, MatrixFormat::Dense>& B,
                                               Span<typename base_type<T>::type> eigenvalues, int itype,
                                               JobType jobz, Uplo uplo);

/// @brief The complex-Hermitian spelling of sygvd() (LAPACK `?hegvd`); identical semantics, and the same
///        backend-deducing overload `hegvd(ctx, ...)`.
/// @ingroup api_eigen
template <Backend Back, typename T>
inline Event hegvd(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                   const MatrixView<T, MatrixFormat::Dense>& B, Span<typename base_type<T>::type> eigenvalues,
                   int itype, JobType jobz, Uplo uplo, Span<std::byte> work_space, Span<int32_t> info) {
    return sygvd<Back, T>(ctx, A, B, eigenvalues, itype, jobz, uplo, work_space, info);
}

///        are not touched (bit-identical on return). The active items are gathered into packed copies in
///        the workspace, solved by sygvd() on that k-item batch (which selects for k items), and scattered back.
/// @throws batchlas::workspace_error if @p work_space is shorter than sygvd_masked_buffer_size()
/// @brief Workspace, in bytes, that sygvd_masked() needs for these operands under every mask (the largest
///        need over every count of active items; the mask's contents do not change it). Query once per
///        batch; costs one sygvd_buffer_size() per item count (host only).
/// @ingroup api_eigen_lowlevel
/// @brief Workspace, in bytes, that sygvd_factored_masked() needs for these operands under every mask, as
///        sygvd_masked_buffer_size().
/// @ingroup api_eigen_lowlevel
BATCHLAS_DISPATCH_ON_QUEUE(sygvd)
BATCHLAS_DISPATCH_ON_QUEUE(sygvd_buffer_size)
BATCHLAS_DISPATCH_ON_QUEUE(sygvd_factored)
BATCHLAS_DISPATCH_ON_QUEUE(sygvd_factored_buffer_size)
BATCHLAS_DISPATCH_ON_QUEUE(hegvd)

}  // namespace batchlas

namespace batchlas::backend {

/// @brief The vendor solver's sygvd (cuSOLVER `?sygvd` per item, LAPACKE `?sygvd` per item), sygvd()'s
///        `vendor` family; same contract as sygvd().
/// @ingroup api_dispatch
template <Backend B, typename T>
BATCHLAS_API Event sygvd_vendor(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                                const MatrixView<T, MatrixFormat::Dense>& Bm,
                                Span<typename base_type<T>::type> eigenvalues, int itype, JobType jobz, Uplo uplo,
                                Span<std::byte> work_space, Span<int32_t> info);

/// @brief Workspace, in bytes, for backend::sygvd_vendor(). @ingroup api_dispatch
template <Backend B, typename T>
BATCHLAS_API size_t sygvd_vendor_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                                             const MatrixView<T, MatrixFormat::Dense>& Bm,
                                             Span<typename base_type<T>::type> eigenvalues, int itype, JobType jobz,
                                             Uplo uplo);

}  // namespace batchlas::backend
