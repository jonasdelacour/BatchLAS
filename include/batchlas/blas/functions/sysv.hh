#pragma once

/// @file
/// @brief Batched symmetric indefinite solve (sysv) by Bunch-Kaufman LDL^T.
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

/// @brief Batched solve of \f$ A X = B \f$ for symmetric indefinite A, LAPACK ?sysv semantics.
///
/// sytrf() followed by sytrs(): on return the @p uplo triangle of A holds D and L (or U), @p pivots
/// holds sytrf()'s ipiv (same packing and 2 x 2 encoding), B holds X and @p info carries sytrf()'s
/// status. An item with `info > 0` keeps its B, as LAPACK ?sysv does; the other items are solved.
/// A complex A is symmetric, not Hermitian (LAPACK csysv).
///
/// Asynchronous: A, B, @p pivots and @p info are readable after the returned event is waited on.
/// @tparam Back  backend; the backend-deducing overload takes it from `ctx.backend()`
/// @tparam T     scalar type (float, double, std::complex<float>, std::complex<double>)
/// @param ctx         queue the kernels are enqueued on
/// @param A           batch of n x n symmetric matrices; the @p uplo triangle is overwritten with the factor
/// @param B           batch of n x nrhs right-hand sides; overwritten with X
/// @param uplo        triangle of A that is read and receives the factor
/// @param pivots      pivot output, at least `n * batch` int64 entries (int32 packed, as sytrf())
/// @param work_space  device-accessible scratch of at least sysv_buffer_size() bytes
/// @param info        per-item status as for sytrf(). Empty span = not requested.
/// @return event of the last enqueued kernel
/// @throws batchlas::invalid_argument if sytrs_validate_params() rejects the call
/// @ingroup api_factorizations
template <Backend Back, typename T>
BATCHLAS_API Event sysv(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                        const MatrixView<T, MatrixFormat::Dense>& B, Uplo uplo, Span<int64_t> pivots,
                        Span<std::byte> work_space, Span<int32_t> info);

/// @brief Workspace, in bytes, that sysv() needs for these operands on this queue.
///
/// Takes the same operands as the call, so both resolve the same route.
/// @ingroup api_factorizations_lowlevel
template <Backend Back, typename T>
BATCHLAS_API size_t sysv_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                                     const MatrixView<T, MatrixFormat::Dense>& B, Uplo uplo);

/// @brief sysv() without per-item status (`info` not requested).
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
