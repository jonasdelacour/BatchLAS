// SPDX-License-Identifier: MIT
#pragma once

#include <batchlas/export.hh>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <string_view>

#include <batchlas/settings.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/blas/matrix.hh>

#include <batchlas/blas/linalg.hh>
#include <batchlas/blas/extensions.hh>

#include <batchlas/blas/queue-dispatch.hh>

namespace batchlas {

// Signature aliases for explicit instantiation (BATCHLAS_INSTANTIATE, in
// src/util/template-instantiations.hh); keep in sync with the declarations below.
namespace sig {
template <typename T>
using syev = Event(Queue&,
                   const MatrixView<T, MatrixFormat::Dense>&,
                   Span<typename base_type<T>::type>,
                   JobType, Uplo, Span<std::byte>, Span<int32_t>);

template <typename T>
using syev_buffer_size = size_t(Queue&,
                                const MatrixView<T, MatrixFormat::Dense>&,
                                Span<typename base_type<T>::type>,
                                JobType, Uplo);

// backend::syev_vendor / syev_vendor_buffer_size share these signatures.
template <typename T> using syev_vendor = syev<T>;
template <typename T> using syev_vendor_buffer_size = syev_buffer_size<T>;
}  // namespace sig


/**
 * @brief Eigenvalues, and optionally eigenvectors, of a batch of symmetric/Hermitian
 *        matrices (LAPACK `?syev` / `?heev`).
 *
 * Computes \f$ A = Q \Lambda Q^H \f$ for every batch item from its `uplo` triangle.
 * The kernel family is the first entry of the nearest row of the device's tuned table
 * (`tuned/syev.<device>.txt`) that can run the call: on a GPU queue a native
 * tier (the \f$n \le 32\f$ sub-group solvers `cta`, `cta_fused` and `jacobi`, which need
 * sub-group size 32; `blocked` or `two_stage` at any \f$n\f$), otherwise `vendor`, the
 * solver library. Pin a family with `BATCHLAS_SYEV_ROUTE` (e.g. `two_stage`, `vendor`).
 * Asynchronous: returns once the work is enqueued.
 *
 * @tparam B  backend (NETLIB always runs the vendor LAPACKE path)
 * @tparam T  float, double, std::complex<float> or std::complex<double>
 * @param[in]     ctx          queue the work is enqueued on
 * @param[in,out] descrA       batch of \f$n \times n\f$ symmetric/Hermitian matrices; with
 *                             `JobType::EigenVectors` overwritten by the orthonormal eigenvectors
 *                             (column \f$j\f$ pairs with eigenvalue \f$j\f$), otherwise its contents
 *                             are destroyed
 * @param[out]    eigenvalues  \f$n\f$ real eigenvalues per batch item, packed
 *                             (\f$n \cdot \mathit{batch}\f$ elements), ascending
 * @param[in]     jobtype      `EigenVectors` or `NoEigenVectors`
 * @param[in]     uplo         which triangle of `descrA` holds the matrix; only it is read (the
 *                             diagonal's imaginary part is taken as zero), the other may be overwritten
 * @param[out]    workspace    scratch of at least syev_buffer_size() bytes for the same arguments;
 *                             contents undefined on return
 * @param[out]    info         per-item convergence status: 0 converged, \f$> 0\f$ LAPACK-like (the
 *                             number of off-diagonals that failed to converge, or 1 where the tier
 *                             only tracks failure). An EMPTY span means "not requested" and costs
 *                             nothing; syev_buffer_size() is the same either way.
 * @return event of the last enqueued kernel
 * @throws batchlas::invalid_argument if `descrA` is not square
 * @throws batchlas::workspace_error if the workspace span is smaller than the chosen kernel needs
 * @throws batchlas::NoRouteError when no native kernel can run (e.g. a CPU queue) in a
 *         build without the solver library
 * @throws std::invalid_argument if `BATCHLAS_SYEV_ROUTE` names a family that is not
 *         compiled or cannot run this call (the words `native` and `vendor` instead
 *         fall back to the tuned choice with a warning)
 * @see @ref selection_tables (which family ranks first where), @ref perf_syev,
 *      @ref md_docs_2cpp-api (convergence status)
 * @ingroup api_eigen
 */
template <Backend B, typename T>
BATCHLAS_API Event syev(Queue& ctx,
                        const MatrixView<T, MatrixFormat::Dense>& descrA, // A is overwritten with eigenvectors
                        Span<typename base_type<T>::type> eigenvalues,
                        JobType jobtype,
                        Uplo uplo,
                        Span<std::byte> workspace,
                        Span<int32_t> info);

// Old-arity forwarder, not a defaulted `info`: sig::syev is a function TYPE and cannot
// carry a default, and BATCHLAS_INSTANTIATE needs alias and declaration identical.
// evidence: docs/design/vendor-independence.md#info-spans-on-syev-gesvd-and-steqr-forwarder-or-default
/**
 * @brief syev() without the convergence status (`info` empty).
 * @param[in]     ctx          queue the work is enqueued on
 * @param[in,out] descrA       batch of \f$n \times n\f$ symmetric/Hermitian matrices; overwritten by
 *                             the eigenvectors or destroyed, as in syev()
 * @param[out]    eigenvalues  \f$n\f$ real eigenvalues per batch item, packed, ascending
 * @param[in]     jobtype      `EigenVectors` or `NoEigenVectors`
 * @param[in]     uplo         which triangle of `descrA` holds the matrix
 * @param[out]    workspace    scratch of at least syev_buffer_size() bytes; contents undefined on return
 * @return event of the last enqueued kernel
 * @ingroup api_eigen
 */
template <Backend B, typename T>
inline Event syev(Queue& ctx,
           const MatrixView<T, MatrixFormat::Dense>& descrA,
           Span<typename base_type<T>::type> eigenvalues,
           JobType jobtype,
           Uplo uplo,
           Span<std::byte> workspace) {
    return syev<B, T>(ctx, descrA, eigenvalues, jobtype, uplo, workspace, Span<int32_t>{});
}

/**
 * @brief Workspace, in bytes, that syev() needs for the same arguments.
 *
 * Makes the same kernel choice as the call, so the size is for the tier that will
 * run. `info` does not affect it.
 * @param[in] ctx          queue the syev() call will run on (selection reads its device)
 * @param[in] A            batch of \f$n \times n\f$ matrices the call will get; only the shape is read
 * @param[in] eigenvalues  the call's eigenvalue span; its contents are not read
 * @param[in] jobtype      the call's `JobType`
 * @param[in] uplo         the call's triangle
 * @return bytes to pass as the `workspace` span of syev()
 * @ingroup api_eigen_lowlevel
 */
template <Backend B, typename T>
BATCHLAS_API size_t syev_buffer_size(Queue& ctx,
                                     const MatrixView<T, MatrixFormat::Dense>& A,
                                     Span<typename base_type<T>::type> eigenvalues,
                                     JobType jobtype,
                                     Uplo uplo);

} // namespace batchlas

namespace batchlas::backend {

// Defined per backend TU (cuSOLVER / rocSOLVER / LAPACKE). `info_out` is DEFAULTED, unlike
// syev's forwarder: a default is not part of the function type, so sig::syev_vendor still
// matches and six-argument callers (norm.cc, cond.cc, syevx_lobpcg.cc) need no forwarder.
// evidence: docs/design/vendor-independence.md#info-spans-on-syev-gesvd-and-steqr-forwarder-or-default
/**
 * @brief The vendor solver's syev (cuSOLVER, rocSOLVER or a LAPACKE loop), as syev()'s
 *        `vendor` family calls it; same contract as syev(), with `info_out` as `info`.
 * @param[in]     ctx          queue the work is enqueued on
 * @param[in,out] descrA       batch of \f$n \times n\f$ symmetric/Hermitian matrices; overwritten by
 *                             the eigenvectors or destroyed, as in syev()
 * @param[out]    eigenvalues  \f$n\f$ real eigenvalues per batch item, packed, ascending
 * @param[in]     jobtype      `EigenVectors` or `NoEigenVectors`
 * @param[in]     uplo         which triangle of `descrA` holds the matrix
 * @param[out]    workspace    scratch of at least syev_vendor_buffer_size() bytes; contents undefined
 *                             on return
 * @param[out]    info_out     per-item convergence status as syev()'s `info`; empty (the default)
 *                             means "not requested"
 * @return event of the last enqueued kernel
 * @ingroup api_dispatch
 */
template <Backend B, typename T>
BATCHLAS_API Event syev_vendor(Queue& ctx,
                               const MatrixView<T, MatrixFormat::Dense>& descrA,
                               Span<typename base_type<T>::type> eigenvalues,
                               JobType jobtype,
                               Uplo uplo,
                               Span<std::byte> workspace,
                               Span<int32_t> info_out = Span<int32_t>());

/**
 * @brief Workspace, in bytes, for backend::syev_vendor().
 * @param[in] ctx          queue the call will run on
 * @param[in] descrA       batch of \f$n \times n\f$ matrices the call will get; only the shape is read
 * @param[in] eigenvalues  the call's eigenvalue span; its contents are not read
 * @param[in] jobtype      the call's `JobType`
 * @param[in] uplo         the call's triangle
 * @return bytes to pass as the `workspace` span of backend::syev_vendor()
 * @ingroup api_dispatch
 */
template <Backend B, typename T>
BATCHLAS_API size_t syev_vendor_buffer_size(Queue& ctx,
                                            const MatrixView<T, MatrixFormat::Dense>& descrA,
                                            Span<typename base_type<T>::type> eigenvalues,
                                            JobType jobtype,
                                            Uplo uplo);

} // namespace batchlas::backend

namespace batchlas::ops::syev::detail {

/// @brief Capability introspection for the Python binding: whether the `cta` kernel can run
///        @p A on this queue's device. Asks syev's own can_run (src/ops/syev/syev.cc), so it
///        cannot drift from what selection does.
/// @tparam T  scalar type (float, double, std::complex<float>, std::complex<double>)
/// @param[in] ctx  queue whose device is asked
/// @param[in] A    batch of \f$n \times n\f$ matrices; only the shape is read
/// @return true when the `cta` kernel can run @p A
template <typename T>
BATCHLAS_API bool syev_supports_cta(const Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A);
/// @brief As syev_supports_cta(), for the `blocked` kernel.
/// @tparam T  scalar type (float, double, std::complex<float>, std::complex<double>)
/// @param[in] ctx   queue whose device is asked
/// @param[in] A     batch of \f$n \times n\f$ matrices; only the shape is read
/// @param[in] uplo  accepted and ignored: the kernel mirrors Upper into Lower
/// @return true when the `blocked` kernel can run @p A
template <typename T>
BATCHLAS_API bool syev_supports_blocked(const Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo);
/// @brief As syev_supports_cta(), for the `two_stage` kernel.
/// @tparam T  scalar type (float, double, std::complex<float>, std::complex<double>)
/// @param[in] ctx   queue whose device is asked
/// @param[in] A     batch of \f$n \times n\f$ matrices; only the shape is read
/// @param[in] uplo  accepted and ignored: the kernel mirrors Upper into Lower
/// @return true when the `two_stage` kernel can run @p A
template <typename T>
BATCHLAS_API bool syev_supports_two_stage(const Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo);

} // namespace batchlas::ops::syev::detail

namespace batchlas {

// Backend-deducing overloads: `f(ctx, ...)` uses ctx.backend(). See
// BATCHLAS_DISPATCH_ON_QUEUE in blas/queue-dispatch.hh.


BATCHLAS_DISPATCH_ON_QUEUE(syev)
BATCHLAS_DISPATCH_ON_QUEUE(syev_buffer_size)

}  // namespace batchlas
