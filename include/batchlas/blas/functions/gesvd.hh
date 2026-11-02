// SPDX-License-Identifier: MIT
#pragma once

#include <batchlas/export.hh>
#include <stdexcept>
#include <optional>
#include <type_traits>
#include <vector>

#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/extensions.hh>

#include <batchlas/backend_config.h>

#include <batchlas/blas/queue-dispatch.hh>

namespace batchlas {

// Signature aliases for explicit instantiation; see BATCHLAS_INSTANTIATE in
// src/util/template-instantiations.hh. Keep in sync with the declarations below.
namespace sig {
template <typename T>
using gesvd_vendor = Event(Queue&,
                           const MatrixView<T, MatrixFormat::Dense>&,
                           Span<typename base_type<T>::type>,
                           const MatrixView<T, MatrixFormat::Dense>&,
                           const MatrixView<T, MatrixFormat::Dense>&,
                           SvdVectors, SvdVectors, Span<std::byte>, Span<int32_t>);

template <typename T>
using gesvd_vendor_buffer_size = size_t(Queue&,
                                        const MatrixView<T, MatrixFormat::Dense>&,
                                        Span<typename base_type<T>::type>,
                                        const MatrixView<T, MatrixFormat::Dense>&,
                                        const MatrixView<T, MatrixFormat::Dense>&,
                                        SvdVectors, SvdVectors);

// The public entry points (src/ops/gesvd/gesvd.cc). `_hermitian` adds the Uplo parameter.
template <typename T>
using gesvd = Event(Queue&, const MatrixView<T, MatrixFormat::Dense>&, Span<typename base_type<T>::type>,
                    const MatrixView<T, MatrixFormat::Dense>&, const MatrixView<T, MatrixFormat::Dense>&,
                    SvdVectors, SvdVectors, Span<std::byte>, Span<int32_t>);
template <typename T>
using gesvd_hermitian = Event(Queue&, const MatrixView<T, MatrixFormat::Dense>&,
                              Span<typename base_type<T>::type>, const MatrixView<T, MatrixFormat::Dense>&,
                              const MatrixView<T, MatrixFormat::Dense>&, SvdVectors, SvdVectors, Uplo,
                              Span<std::byte>, Span<int32_t>);
template <typename T>
using gesvd_buffer_size = size_t(Queue&, const MatrixView<T, MatrixFormat::Dense>&,
                                 Span<typename base_type<T>::type>, const MatrixView<T, MatrixFormat::Dense>&,
                                 const MatrixView<T, MatrixFormat::Dense>&, SvdVectors, SvdVectors);
template <typename T>
using gesvd_buffer_size_hermitian = size_t(Queue&, const MatrixView<T, MatrixFormat::Dense>&,
                                           Span<typename base_type<T>::type>,
                                           const MatrixView<T, MatrixFormat::Dense>&,
                                           const MatrixView<T, MatrixFormat::Dense>&, SvdVectors, SvdVectors,
                                           Uplo);
}  // namespace sig

/**
 * @brief Singular value decomposition of a batch of matrices (LAPACK `?gesvd`).
 *
 * Computes \f$ A = U \Sigma V^H \f$ for every \f$m \times n\f$ batch item, with the
 * \f$ k = \min(m, n) \f$ singular values in descending order. The kernel family is
 * the first entry of the nearest row of the device's tuned table
 * (`tuned/gesvd.<device>.txt`) that can run the call: `jacobi` (one-sided
 * Jacobi, \f$\max(m, n) \le 64\f$, or \f$\le 32\f$ for `complex<double>` with vectors), `cta`
 * (\f$\max(m, n) \le 32\f$; no `Thin` request that canonicalisation leaves thin),
 * `blocked` (bidiagonalisation for real input; square Hermitian `Lower` input of
 * any type runs syev_blocked()), or
 * `vendor`, the solver library. Complex general input therefore runs `jacobi` or
 * `vendor`. Pin a family with `BATCHLAS_GESVD_ROUTE`
 * (e.g. `jacobi`, `vendor`). Asynchronous: returns once the work is enqueued.
 *
 * @tparam B  backend the call is compiled for
 * @tparam T  scalar type (float, double, std::complex<float>, std::complex<double>)
 * @param[in] ctx              queue the work is enqueued on
 * @param[in,out] A            batch of \f$m \times n\f$ matrices; overwritten (destroyed)
 * @param[out] singular_values \f$k\f$ real singular values per batch item, packed, descending
 * @param[out] U               left singular vectors: \f$m \times m\f$ for `SvdVectors::All`,
 *                             \f$m \times k\f$ for `Thin`; unused for `None`
 * @param[out] Vh              \f$ V^H \f$: \f$n \times n\f$ for `All`, \f$k \times n\f$ for `Thin`;
 *                             unused for `None`
 * @param[in] jobu             which columns of `U` to compute
 * @param[in] jobvh            which rows of `Vh` to compute
 * @param[out] workspace       scratch of at least gesvd_buffer_size() bytes for the same
 *                             arguments; contents undefined on return
 * @param[out] info            per-item convergence status: 0 converged, \f$> 0\f$ LAPACK-like (the
 *                             number of off-diagonals that failed to converge, or 1 where the
 *                             kernel that ran tracks only the fact of failure). An
 *                             EMPTY span means "not requested" and costs nothing;
 *                             gesvd_buffer_size() is the same either way.
 * @return event of the last enqueued kernel
 * @throws batchlas::workspace_error if the workspace span is smaller than the chosen kernel needs
 * @throws batchlas::NoRouteError when no native kernel can run the call in a build
 *         without the solver library
 * @throws batchlas::unsupported if the `vendor` family runs a shape its binding cannot
 *         take: on CUDA (`gesvdjBatched`) \f$\max(m, n) > 32\f$, a non-packed batch stride or
 *         a thin request; on ROCm always (no binding). Complex general input above
 *         `jacobi`'s ceiling therefore has no kernel on those backends.
 * @throws batchlas::convergence_error on NETLIB when an item fails to converge and
 *         `info` is empty (with `info` given, the status is written instead)
 * @throws std::invalid_argument if `BATCHLAS_GESVD_ROUTE` names a family that is not
 *         compiled or cannot run this call (the words `native` and `vendor` instead
 *         fall back to the tuned choice with a warning)
 * @see @ref selection_tables, @ref perf_gesvd, @ref design_gesvd,
 *      @ref md_docs_2cpp-api (convergence status)
 * @ingroup api_svd
 */
template <Backend B, typename T>
BATCHLAS_API Event gesvd(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         Span<typename base_type<T>::type> singular_values,
                         const MatrixView<T, MatrixFormat::Dense>& U,
                         const MatrixView<T, MatrixFormat::Dense>& Vh,
                         SvdVectors jobu,
                         SvdVectors jobvh,
                         Span<std::byte> workspace,
                         Span<int32_t> info);

/**
 * @brief gesvd() of square Hermitian input. Other parameters as for the general form.
 *
 * The native kernels solve it as an eigenproblem (syev_cta() or syev_blocked()) and read
 * only the `hermitian_uplo` triangle (`blocked` takes `Lower` only). The `vendor`
 * family has no triangle argument and reads all of `A`, so keep both triangles valid
 * wherever that family can be chosen, e.g. `Upper` with \f$n > 32\f$.
 * @pre `A` is square
 * @tparam B  backend the call is compiled for
 * @tparam T  scalar type
 * @param[in] ctx              queue the work is enqueued on
 * @param[in,out] A            batch of \f$n \times n\f$ Hermitian matrices; overwritten (destroyed)
 * @param[out] singular_values \f$n\f$ real singular values per batch item, packed, descending
 * @param[out] U               left singular vectors (\f$n \times n\f$); unused for `None`
 * @param[out] Vh              \f$ V^H \f$ (\f$n \times n\f$); unused for `None`
 * @param[in] jobu             which columns of `U` to compute
 * @param[in] jobvh            which rows of `Vh` to compute
 * @param[in] hermitian_uplo   triangle of `A` the native kernels read
 * @param[out] workspace       scratch of at least gesvd_buffer_size() bytes for the same
 *                             arguments; contents undefined on return
 * @param[out] info            per-item convergence status as for the general form; empty
 *                             span = not requested
 * @return event of the last enqueued kernel
 * @ingroup api_svd
 */
template <Backend B, typename T>
BATCHLAS_API Event gesvd(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         Span<typename base_type<T>::type> singular_values,
                         const MatrixView<T, MatrixFormat::Dense>& U,
                         const MatrixView<T, MatrixFormat::Dense>& Vh,
                         SvdVectors jobu,
                         SvdVectors jobvh,
                         Uplo hermitian_uplo,
                         Span<std::byte> workspace,
                         Span<int32_t> info);

// Old-arity forwarders, not a defaulted `info`, as in functions/syev.hh: the sig:: alias
// is a function TYPE and cannot carry a default. Arity plus the Uplo/Span type at
// parameter 8 keeps all four overloads unambiguous.
// evidence: docs/design/vendor-independence.md#info-spans-on-syev-gesvd-and-steqr-forwarder-or-default
/**
 * @brief gesvd() without the convergence status (`info` empty).
 * @tparam B  backend the call is compiled for
 * @tparam T  scalar type
 * @param[in] ctx              queue the work is enqueued on
 * @param[in,out] A            batch of \f$m \times n\f$ matrices; overwritten (destroyed)
 * @param[out] singular_values \f$k = \min(m, n)\f$ real singular values per batch item, descending
 * @param[out] U               left singular vectors (\f$m \times m\f$ or \f$m \times k\f$); unused for `None`
 * @param[out] Vh              \f$ V^H \f$ (\f$n \times n\f$ or \f$k \times n\f$); unused for `None`
 * @param[in] jobu             which columns of `U` to compute
 * @param[in] jobvh            which rows of `Vh` to compute
 * @param[out] workspace       scratch of at least gesvd_buffer_size() bytes; contents undefined on return
 * @return event of the last enqueued kernel
 * @ingroup api_svd
 */
template <Backend B, typename T>
inline Event gesvd(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& A,
            Span<typename base_type<T>::type> singular_values,
            const MatrixView<T, MatrixFormat::Dense>& U,
            const MatrixView<T, MatrixFormat::Dense>& Vh,
            SvdVectors jobu,
            SvdVectors jobvh,
            Span<std::byte> workspace) {
    return gesvd<B, T>(ctx, A, singular_values, U, Vh, jobu, jobvh, workspace, Span<int32_t>{});
}

/**
 * @brief Hermitian gesvd() without the convergence status (`info` empty).
 * @tparam B  backend the call is compiled for
 * @tparam T  scalar type
 * @param[in] ctx              queue the work is enqueued on
 * @param[in,out] A            batch of \f$n \times n\f$ Hermitian matrices; overwritten (destroyed)
 * @param[out] singular_values \f$n\f$ real singular values per batch item, descending
 * @param[out] U               left singular vectors (\f$n \times n\f$); unused for `None`
 * @param[out] Vh              \f$ V^H \f$ (\f$n \times n\f$); unused for `None`
 * @param[in] jobu             which columns of `U` to compute
 * @param[in] jobvh            which rows of `Vh` to compute
 * @param[in] hermitian_uplo   triangle of `A` the native kernels read
 * @param[out] workspace       scratch of at least gesvd_buffer_size() bytes; contents undefined on return
 * @return event of the last enqueued kernel
 * @ingroup api_svd
 */
template <Backend B, typename T>
inline Event gesvd(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& A,
            Span<typename base_type<T>::type> singular_values,
            const MatrixView<T, MatrixFormat::Dense>& U,
            const MatrixView<T, MatrixFormat::Dense>& Vh,
            SvdVectors jobu,
            SvdVectors jobvh,
            Uplo hermitian_uplo,
            Span<std::byte> workspace) {
    return gesvd<B, T>(ctx, A, singular_values, U, Vh, jobu, jobvh, hermitian_uplo, workspace,
                       Span<int32_t>{});
}

/**
 * @brief Workspace, in bytes, that gesvd() needs for the same arguments.
 *
 * Canonicalises `jobu`/`jobvh` and makes the same kernel choice as the call does, so
 * the size is for the kernel that will run. `info` does not affect it.
 * @tparam B  backend the call is compiled for
 * @tparam T  scalar type
 * @param[in] ctx              queue the call will run on
 * @param[in] A                batch of \f$m \times n\f$ matrices that will be passed; not modified
 * @param[in] singular_values  singular-value span that will be passed; not modified
 * @param[in] U                left-vector matrix that will be passed; not modified
 * @param[in] Vh               \f$ V^H \f$ matrix that will be passed; not modified
 * @param[in] jobu             which columns of `U` will be computed
 * @param[in] jobvh            which rows of `Vh` will be computed
 * @return required workspace size in bytes
 * @ingroup api_svd_lowlevel
 */
template <Backend B, typename T>
BATCHLAS_API size_t gesvd_buffer_size(Queue& ctx,
                                      const MatrixView<T, MatrixFormat::Dense>& A,
                                      Span<typename base_type<T>::type> singular_values,
                                      const MatrixView<T, MatrixFormat::Dense>& U,
                                      const MatrixView<T, MatrixFormat::Dense>& Vh,
                                      SvdVectors jobu,
                                      SvdVectors jobvh);

/**
 * @brief Workspace, in bytes, for the Hermitian gesvd().
 * @tparam B  backend the call is compiled for
 * @tparam T  scalar type
 * @param[in] ctx              queue the call will run on
 * @param[in] A                batch of \f$n \times n\f$ Hermitian matrices that will be passed; not modified
 * @param[in] singular_values  singular-value span that will be passed; not modified
 * @param[in] U                left-vector matrix that will be passed; not modified
 * @param[in] Vh               \f$ V^H \f$ matrix that will be passed; not modified
 * @param[in] jobu             which columns of `U` will be computed
 * @param[in] jobvh            which rows of `Vh` will be computed
 * @param[in] hermitian_uplo   triangle of `A` the native kernels will read
 * @return required workspace size in bytes
 * @ingroup api_svd_lowlevel
 */
template <Backend B, typename T>
BATCHLAS_API size_t gesvd_buffer_size(Queue& ctx,
                                      const MatrixView<T, MatrixFormat::Dense>& A,
                                      Span<typename base_type<T>::type> singular_values,
                                      const MatrixView<T, MatrixFormat::Dense>& U,
                                      const MatrixView<T, MatrixFormat::Dense>& Vh,
                                      SvdVectors jobu,
                                      SvdVectors jobvh,
                                      Uplo hermitian_uplo);

} // namespace batchlas

namespace batchlas::backend {

// DECLARATION ONLY: each backend wrapper TU (cuSOLVER / rocSOLVER / LAPACKE) defines and
// instantiates it for its own Backend. A definition here makes theirs a redefinition error.
// evidence: docs/design/gesvd.md#gesvd-design-vendor-binding-and-dispatch
// `info_out` (caller's status span, or empty) is defaulted, as in syev_vendor, so
// sig::gesvd_vendor still names the full nine-parameter signature.
/**
 * @brief The vendor solver's SVD (cuSOLVER `gesvdjBatched` or a LAPACKE loop), as
 *        gesvd()'s `vendor` kernel family calls it; same contract as the general gesvd().
 * @tparam B  backend the call is compiled for
 * @tparam T  scalar type
 * @param[in] ctx              queue the work is enqueued on
 * @param[in,out] A            batch of \f$m \times n\f$ matrices; overwritten (destroyed)
 * @param[out] singular_values \f$k = \min(m, n)\f$ real singular values per batch item, descending
 * @param[out] U               left singular vectors (\f$m \times m\f$ or \f$m \times k\f$); unused for `None`
 * @param[out] Vh              \f$ V^H \f$ (\f$n \times n\f$ or \f$k \times n\f$); unused for `None`
 * @param[in] jobu             which columns of `U` to compute
 * @param[in] jobvh            which rows of `Vh` to compute
 * @param[out] workspace       scratch of at least gesvd_vendor_buffer_size() bytes; contents
 *                             undefined on return
 * @param[out] info_out        per-item convergence status as for gesvd(); empty span = not requested
 * @return event of the last enqueued kernel
 * @throws batchlas::unsupported on CUDA for \f$\max(m, n) > 32\f$, a non-packed batch stride
 *         or `Thin` vectors (`gesvdjBatched` has no `econ` flag); on ROCm always
 * @throws batchlas::convergence_error on NETLIB for a non-converged item when
 *         `info_out` is empty
 * @ingroup api_dispatch
 */
template <Backend B, typename T>
BATCHLAS_API Event gesvd_vendor(Queue& ctx,
                                const MatrixView<T, MatrixFormat::Dense>& A,
                                Span<typename base_type<T>::type> singular_values,
                                const MatrixView<T, MatrixFormat::Dense>& U,
                                const MatrixView<T, MatrixFormat::Dense>& Vh,
                                SvdVectors jobu,
                                SvdVectors jobvh,
                                Span<std::byte> workspace,
                                Span<int32_t> info_out = Span<int32_t>());

/**
 * @brief Workspace, in bytes, for backend::gesvd_vendor().
 * @tparam B  backend the call is compiled for
 * @tparam T  scalar type
 * @param[in] ctx              queue the call will run on
 * @param[in] A                batch of \f$m \times n\f$ matrices that will be passed; not modified
 * @param[in] singular_values  singular-value span that will be passed; not modified
 * @param[in] U                left-vector matrix that will be passed; not modified
 * @param[in] Vh               \f$ V^H \f$ matrix that will be passed; not modified
 * @param[in] jobu             which columns of `U` will be computed
 * @param[in] jobvh            which rows of `Vh` will be computed
 * @return required workspace size in bytes
 * @ingroup api_dispatch
 */
template <Backend B, typename T>
BATCHLAS_API size_t gesvd_vendor_buffer_size(Queue& ctx,
                                             const MatrixView<T, MatrixFormat::Dense>& A,
                                             Span<typename base_type<T>::type> singular_values,
                                             const MatrixView<T, MatrixFormat::Dense>& U,
                                             const MatrixView<T, MatrixFormat::Dense>& Vh,
                                             SvdVectors jobu,
                                             SvdVectors jobvh);

} // namespace batchlas::backend


namespace batchlas {

// Backend-deducing overloads: `f(ctx, ...)` uses ctx.backend(). See
// BATCHLAS_DISPATCH_ON_QUEUE in blas/queue-dispatch.hh.


BATCHLAS_DISPATCH_ON_QUEUE(gesvd)
BATCHLAS_DISPATCH_ON_QUEUE(gesvd_buffer_size)

}  // namespace batchlas
