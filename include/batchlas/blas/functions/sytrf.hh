#pragma once

/// @file
/// @brief Batched symmetric indefinite factorization (sytrf, Bunch-Kaufman LDL^T) and solve (sytrs).
/// @ingroup api_factorizations

// sytrf and sytrs are routed ops (src/ops/sytrf/, src/ops/sytrs/; BATCHLAS_SYTRF_ROUTE,
// BATCHLAS_SYTRS_ROUTE). evidence: docs/perf/sytrf.md

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
using sytrf = Event(Queue&, const MatrixView<T, MatrixFormat::Dense>&, Uplo, Span<int64_t>, Span<std::byte>,
                    Span<int32_t>);

template <typename T>
using sytrf_buffer_size = size_t(Queue&, const MatrixView<T, MatrixFormat::Dense>&, Uplo);

template <typename T>
using sytrs = Event(Queue&, const MatrixView<T, MatrixFormat::Dense>&, const MatrixView<T, MatrixFormat::Dense>&,
                    Uplo, Span<int64_t>, Span<std::byte>);

template <typename T>
using sytrs_buffer_size = size_t(Queue&, const MatrixView<T, MatrixFormat::Dense>&,
                                 const MatrixView<T, MatrixFormat::Dense>&, Uplo);

template <typename T>
using sytrf_vendor = sytrf<T>;

template <typename T>
using sytrf_vendor_buffer_size = sytrf_buffer_size<T>;

template <typename T>
using sytrs_vendor = Event(Queue&, const MatrixView<T, MatrixFormat::Dense>&, const MatrixView<T, MatrixFormat::Dense>&,
                           Uplo, Span<int64_t>, const int32_t*);
}  // namespace sig

/// @brief Validates the arguments of sytrf() and sytrf_buffer_size().
///
/// Checks non-negative extents, a square A and a valid @p uplo.
/// @throws batchlas::invalid_argument if any check fails
/// @ingroup api_factorizations_lowlevel
template <typename T>
inline void sytrf_validate_params(const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo) {
    if (A.rows() < 0 || A.cols() < 0) {
        throw batchlas::invalid_argument("SYTRF: Matrix dimensions cannot be negative (A: rows=" +
                                         std::to_string(A.rows()) + ", cols=" + std::to_string(A.cols()) + ")");
    }
    if (A.rows() != A.cols()) {
        throw batchlas::invalid_argument("SYTRF: A must be square, got " + std::to_string(A.rows()) + "x" +
                                         std::to_string(A.cols()));
    }
    if (uplo != Uplo::Lower && uplo != Uplo::Upper) {
        throw batchlas::invalid_argument("SYTRF: Invalid uplo parameter: " + std::to_string(static_cast<int>(uplo)));
    }
}

/// @brief Validates the arguments of sytrs(), sytrs_buffer_size(), sysv() and sysv_buffer_size().
///
/// sytrf_validate_params() on A, plus `B.rows() == A.rows()`, `B.cols() >= 0` and equal batch sizes.
/// @throws batchlas::invalid_argument if any check fails
/// @ingroup api_factorizations_lowlevel
template <typename T>
inline void sytrs_validate_params(const MatrixView<T, MatrixFormat::Dense>& A,
                                  const MatrixView<T, MatrixFormat::Dense>& B, Uplo uplo) {
    sytrf_validate_params<T>(A, uplo);
    if (B.rows() != A.rows() || B.cols() < 0) {
        throw batchlas::invalid_argument("SYTRS: B must be n x nrhs with n = " + std::to_string(A.rows()) + ", got " +
                                         std::to_string(B.rows()) + "x" + std::to_string(B.cols()));
    }
    if (A.batch_size() != B.batch_size()) {
        throw batchlas::invalid_argument("SYTRS: A and B must share a batch size, got " +
                                         std::to_string(A.batch_size()) + " and " + std::to_string(B.batch_size()));
    }
}

/// @brief Batched Bunch-Kaufman factorization \f$ A = L D L^T \f$ (Lower) or \f$ A = U D U^T \f$ (Upper)
///        of symmetric indefinite matrices, LAPACK ?sytrf semantics.
///
/// A is symmetric, not Hermitian: a complex A is factored without conjugation (LAPACK csytrf). Only
/// the @p uplo triangle is read and written; the other triangle is left untouched. On return that
/// triangle holds D (diagonal plus, for a 2 x 2 block, the one off-diagonal entry inside the
/// triangle) and the multipliers of L (or U), and @p pivots holds LAPACK's ipiv, 1-based:
/// - `ipiv[k] > 0`: D(k,k) is a 1 x 1 block and rows/columns k and `ipiv[k] - 1` were interchanged;
/// - Lower, `ipiv[k] = ipiv[k+1] = -p < 0`: D(k:k+1, k:k+1) is a 2 x 2 block and rows/columns k+1 and
///   p - 1 were interchanged;
/// - Upper, `ipiv[k-1] = ipiv[k] = -p < 0`: D(k-1:k, k-1:k) is a 2 x 2 block and rows/columns k-1 and
///   p - 1 were interchanged.
///
/// Pivots are int32, packed into the first half of the int64 span as getrf()'s are: item b's
/// entries are `pivots.as_span<int>()[b * n + k]`.
///
/// Asynchronous: A, @p pivots and @p info are readable after the returned event is waited on.
/// @tparam Back  backend; the backend-deducing overload takes it from `ctx.backend()`
/// @tparam T     scalar type (float, double, std::complex<float>, std::complex<double>)
/// @param ctx         queue the kernels are enqueued on
/// @param A           batch of n x n symmetric matrices; the @p uplo triangle is overwritten with D and L (or U)
/// @param uplo        triangle of A that is read and receives the factor
/// @param pivots      pivot output, at least `n * batch` int64 entries (int32 packed, see above)
/// @param work_space  device-accessible scratch of at least sytrf_buffer_size() bytes
/// @param info        per-item status: 0 on success, i > 0 if D(i,i) (1-based) is exactly zero; the
///                    factorization completes, but that item's D is singular. Empty span = not requested.
/// @return event of the last enqueued kernel
/// @throws batchlas::invalid_argument if sytrf_validate_params() rejects the call
/// @throws batchlas::NoRouteError if no native kernel can run the shape and the vendor library was not built in
/// @ingroup api_factorizations
template <Backend Back, typename T>
BATCHLAS_API Event sytrf(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo, Span<int64_t> pivots,
                         Span<std::byte> work_space, Span<int32_t> info);

/// @brief Workspace, in bytes, that sytrf() needs for these operands on this queue.
/// @ingroup api_factorizations_lowlevel
template <Backend Back, typename T>
BATCHLAS_API size_t sytrf_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo);

/// @brief sytrf() without per-item status (`info` not requested).
/// @ingroup api_factorizations
template <Backend Back, typename T>
inline Event sytrf(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo, Span<int64_t> pivots,
                   Span<std::byte> work_space) {
    return sytrf<Back, T>(ctx, A, uplo, pivots, work_space, Span<int32_t>{});
}

/// @brief Batched solve of \f$ A X = B \f$ with the factors from sytrf(), LAPACK ?sytrs semantics.
///
/// A, @p uplo and @p pivots must be exactly what sytrf() produced on the same backend. B is
/// overwritten with X. No singularity check is made: an item whose sytrf() `info` was positive
/// produces infinities or NaNs.
///
/// Asynchronous: B is readable after the returned event is waited on.
/// @param ctx         queue the kernels are enqueued on
/// @param A           batch of n x n factors from sytrf(); not modified
/// @param B           batch of n x nrhs right-hand sides; overwritten with X
/// @param uplo        the triangle sytrf() was called with
/// @param pivots      pivots from sytrf(), `n * batch` entries
/// @param work_space  device-accessible scratch of at least sytrs_buffer_size() bytes (the `blocked`
///                    route's converted factor and gathered B, per chunk of items; zero for the others)
/// @return event of the last enqueued kernel
/// @throws batchlas::invalid_argument if sytrs_validate_params() rejects the call
/// @throws batchlas::NoRouteError if no kernel can run the call (NETLIB without LAPACKE)
/// @ingroup api_factorizations
template <Backend Back, typename T>
BATCHLAS_API Event sytrs(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                         const MatrixView<T, MatrixFormat::Dense>& B, Uplo uplo, Span<int64_t> pivots,
                         Span<std::byte> work_space);

/// @brief Workspace, in bytes, that sytrs() needs for these operands on this queue.
/// @ingroup api_factorizations_lowlevel
template <Backend Back, typename T>
BATCHLAS_API size_t sytrs_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                                      const MatrixView<T, MatrixFormat::Dense>& B, Uplo uplo);

}  // namespace batchlas

namespace batchlas::backend {

/// @brief Vendor arm of sytrf() (cuSOLVER ?sytrf per item, LAPACKE ?sytrf); called by sytrf() when it
///        selects the `vendor` kernel family, not by users.
/// @ingroup api_dispatch
template <Backend Back, typename T>
BATCHLAS_API Event sytrf_vendor(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo,
                                Span<int64_t> pivots, Span<std::byte> work_space, Span<int32_t> info);

/// @brief Workspace query of the vendor arm of sytrf().
/// @ingroup api_dispatch
template <Backend Back, typename T>
BATCHLAS_API size_t sytrf_vendor_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo);

/// @brief The NETLIB backend's sytrs() (LAPACKE ?sytrs per item); not called by users. An item whose
///        @p skip_info entry is nonzero keeps its B (sysv() after a singular factor); null skips none.
/// @ingroup api_dispatch
template <Backend Back, typename T>
BATCHLAS_API Event sytrs_vendor(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                                const MatrixView<T, MatrixFormat::Dense>& B, Uplo uplo, Span<int64_t> pivots,
                                const int32_t* skip_info);

}  // namespace batchlas::backend

namespace batchlas {

// Owning-container and backend-deducing overloads; see blas/queue-dispatch.hh.

BATCHLAS_DISPATCH_ON_QUEUE(sytrf)
BATCHLAS_DISPATCH_ON_QUEUE(sytrf_buffer_size)
BATCHLAS_DISPATCH_ON_QUEUE(sytrs)
BATCHLAS_DISPATCH_ON_QUEUE(sytrs_buffer_size)

}  // namespace batchlas
