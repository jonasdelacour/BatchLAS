#pragma once

/// @file
/// @brief Batched symmetric indefinite solve (sysv) by Bunch-Kaufman \f$ L D L^{T} \f$.
/// @ingroup api_factorizations

// No vendor arm: the composed arm is the routed sytrf then sytrs, and the contract below is exactly
// that composition. evidence: docs/perf/sytrf.md

#include <batchlas/export.hh>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/queue-dispatch.hh>
#include <batchlas/blas/functions/sytrf.hh>

namespace batchlas {

namespace sig {
template <typename T>
using sysv = Event(Queue&, const MatrixView<T, MatrixFormat::Dense>&, const MatrixView<T, MatrixFormat::Dense>&,
                   Uplo, Span<int64_t>, Span<std::byte>, Span<int32_t>);

template <typename T>
using sysv_buffer_size = size_t(Queue&, const MatrixView<T, MatrixFormat::Dense>&,
                                const MatrixView<T, MatrixFormat::Dense>&, Uplo);
}  // namespace sig

/// @brief Batched solve of \f$ A X = B \f$ for symmetric indefinite \f$A\f$, LAPACK ?sysv semantics.
///
/// sytrf() followed by sytrs(): on return the @p uplo triangle of \f$A\f$ holds \f$D\f$ and \f$L\f$
/// (or \f$U\f$), @p pivots holds sytrf()'s ipiv (same packing and \f$2 \times 2\f$ encoding), \f$B\f$
/// holds \f$X\f$ and @p info carries sytrf()'s status. An item with `info > 0` keeps its \f$B\f$, as
/// LAPACK ?sysv does; the other items are solved. A complex \f$A\f$ is symmetric
/// (\f$A = A^{T}\f$), not Hermitian (LAPACK csysv).
///
/// Asynchronous: \f$A\f$, \f$B\f$, @p pivots and @p info are readable after the returned event is waited on.
/// @tparam Back  backend; the backend-deducing overload takes it from `ctx.backend()`
/// @tparam T     scalar type (float, double, std::complex<float>, std::complex<double>)
/// @param[in]     ctx         queue the kernels are enqueued on
/// @param[in,out] A           batch of \f$n \times n\f$ symmetric matrices; only the @p uplo triangle is
///                            read, and it is overwritten with \f$D\f$ and \f$L\f$ (or \f$U\f$)
/// @param[in,out] B           batch of \f$n \times n_{\mathrm{rhs}}\f$ right-hand sides; overwritten with
///                            the solution \f$X\f$ (kept unchanged for an item with `info > 0`)
/// @param[in]     uplo        triangle of \f$A\f$ that is read and receives the factor
/// @param[out]    pivots      pivots written by sytrf(), at least `n * batch` int64 entries (int32
///                            packed, as sytrf())
/// @param[out]    work_space  device-accessible scratch of at least sysv_buffer_size() bytes; contents
///                            undefined on return
/// @param[out]    info        per-item status as for sytrf(): 0 on success, \f$i > 0\f$ if
///                            \f$D_{i,i}\f$ (1-based) is exactly zero. A span with fewer than `batch`
///                            entries (including an empty one) = not requested.
/// @return event of the last enqueued kernel
/// @throws batchlas::invalid_argument if sytrs_validate_params() rejects the call, or if @p A and
///         @p B carry different batch masks (or a broadcast operand is masked)
/// @throws batchlas::internal_error for a heterogeneous batch
/// @ingroup api_factorizations
template <Backend Back, typename T>
BATCHLAS_API Event sysv(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                        const MatrixView<T, MatrixFormat::Dense>& B, Uplo uplo, Span<int64_t> pivots,
                        Span<std::byte> work_space, Span<int32_t> info);

/// @brief Workspace, in bytes, that sysv() needs for these operands on this queue.
///
/// Takes the same operands as the call, so both resolve the same route.
/// @tparam Back  backend; the backend-deducing overload takes it from `ctx.backend()`
/// @tparam T     scalar type
/// @param[in] ctx   queue the call will run on
/// @param[in] A     batch of \f$n \times n\f$ matrices sysv() will factor; only its shape and batch mask are used
/// @param[in] B     batch of \f$n \times n_{\mathrm{rhs}}\f$ right-hand sides sysv() will solve; only its shape and
///                  batch mask are used
/// @param[in] uplo  triangle of \f$A\f$ that sysv() will factor
/// @return the workspace size in bytes for the `work_space` argument of sysv()
/// @throws batchlas::invalid_argument if sytrs_validate_params() rejects the operands or their batch masks differ
/// @throws batchlas::internal_error for a heterogeneous batch
/// @ingroup api_factorizations_lowlevel
template <Backend Back, typename T>
BATCHLAS_API size_t sysv_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                                     const MatrixView<T, MatrixFormat::Dense>& B, Uplo uplo);

/// @brief sysv() without per-item status (`info` not requested).
/// @tparam Back  backend; the backend-deducing overload takes it from `ctx.backend()`
/// @tparam T     scalar type
/// @param[in]     ctx         queue the kernels are enqueued on
/// @param[in,out] A           batch of \f$n \times n\f$ symmetric matrices; the @p uplo triangle is
///                            overwritten with \f$D\f$ and \f$L\f$ (or \f$U\f$)
/// @param[in,out] B           batch of \f$n \times n_{\mathrm{rhs}}\f$ right-hand sides; overwritten with
///                            the solution \f$X\f$ (kept unchanged for an item whose \f$D\f$ is singular)
/// @param[in]     uplo        triangle of \f$A\f$ that is read and receives the factor
/// @param[out]    pivots      pivots written by sytrf(), at least `n * batch` int64 entries (int32
///                            packed, as sytrf())
/// @param[out]    work_space  device-accessible scratch of at least sysv_buffer_size() bytes; contents
///                            undefined on return
/// @return event of the last enqueued kernel
/// @ingroup api_factorizations
template <Backend Back, typename T>
inline Event sysv(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                  const MatrixView<T, MatrixFormat::Dense>& B, Uplo uplo, Span<int64_t> pivots,
                  Span<std::byte> work_space) {
    return sysv<Back, T>(ctx, A, B, uplo, pivots, work_space, Span<int32_t>{});
}

BATCHLAS_DISPATCH_ON_QUEUE(sysv)
BATCHLAS_DISPATCH_ON_QUEUE(sysv_buffer_size)

}  // namespace batchlas
