#pragma once

/// @file
/// @brief Batched general linear solve (gesv) by LU with partial pivoting.
/// @ingroup api_factorizations

// No vendor ships a batched gesv, so there is no gesv_vendor; the fallback is the
// routed composition getrf; getrs, and the documented contract is exactly that.
// evidence: docs/perf/lu.md#the-fused-gesv-tier

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
using gesv = Event(Queue&,
                   const MatrixView<T, MatrixFormat::Dense>&,
                   const MatrixView<T, MatrixFormat::Dense>&,
                   Span<int64_t>, Span<std::byte>, Span<int32_t>);

template <typename T>
using gesv_buffer_size = size_t(Queue&,
                                const MatrixView<T, MatrixFormat::Dense>&,
                                const MatrixView<T, MatrixFormat::Dense>&);
}  // namespace sig

/// @brief Validates the arguments of gesv() and gesv_buffer_size().
///
/// Checks non-negative extents, a square A, `B.rows() == A.rows()` and equal
/// batch sizes.
/// @throws batchlas::invalid_argument if any check fails
/// @ingroup api_factorizations
// Stricter than getrs_validate_params on purpose: getrs routes a non-conforming
// pair to the vendor, but gesv has none, so it would report the wrong cause.
template <typename T>
inline void gesv_validate_params(const MatrixView<T, MatrixFormat::Dense>& A,
                                 const MatrixView<T, MatrixFormat::Dense>& B) {
    if (A.rows() < 0 || A.cols() < 0 || B.rows() < 0 || B.cols() < 0) {
        throw batchlas::invalid_argument(
            "GESV: Matrix dimensions cannot be negative (A: rows=" +
            std::to_string(A.rows()) + ", cols=" + std::to_string(A.cols()) +
            "; B: rows=" + std::to_string(B.rows()) +
            ", cols=" + std::to_string(B.cols()) + ")");
    }
    if (A.rows() != A.cols()) {
        throw batchlas::invalid_argument(
            "GESV: A must be square, got " + std::to_string(A.rows()) + "x" +
            std::to_string(A.cols()));
    }
    if (A.rows() != B.rows()) {
        throw batchlas::invalid_argument(
            "GESV: B must have A.rows() rows, got " + std::to_string(B.rows()) +
            " against " + std::to_string(A.rows()));
    }
    if (A.batch_size() != B.batch_size()) {
        throw batchlas::invalid_argument(
            "GESV: A and B must share a batch size, got " +
            std::to_string(A.batch_size()) + " and " + std::to_string(B.batch_size()));
    }
}

/// @brief Batched solve of \f$ A X = B \f$ by LU with partial pivoting.
///
/// Semantically getrf() followed by getrs() with `Transpose::NoTrans`: on return
/// A holds L and U as getrf() leaves them, @p pivots the same packed 1-based
/// interchange list getrf() writes (so the factors can be reused with getrs()
/// or getri()), B holds X and @p info carries getrf's per-item status. No
/// vendor library call sits behind this op; it is served by a native fused
/// tier or by the routed getrf + getrs composition.
///
/// Asynchronous: A, B, @p pivots and @p info are readable after the returned
/// event is waited on.
/// @tparam Back  backend; the backend-deducing overload takes it from `ctx.backend()`
/// @tparam T     scalar type (float, double, std::complex<float>, std::complex<double>)
/// @param ctx         queue the kernels are enqueued on
/// @param A           batch of n x n matrices; overwritten with L and U
/// @param B           batch of n x nrhs right-hand sides; overwritten with X
/// @param pivots      pivot output, `n * batch` entries in getrf()'s format
/// @param work_space  device-accessible scratch of at least gesv_buffer_size() bytes
/// @param info        per-item status as for getrf(): 0 on success, i > 0 if
///                    U(i,i) is exactly zero (that item's X is meaningless).
///                    Empty span = not requested.
/// @return event of the last enqueued kernel
/// @throws batchlas::invalid_argument if gesv_validate_params() rejects the call
/// @throws batchlas::workspace_error if the composed getrf + getrs route is
///         chosen and @p work_space is shorter than its getrf leg needs
/// @ingroup api_factorizations
template <Backend Back, typename T>
BATCHLAS_API Event gesv(Queue& ctx,
                        const MatrixView<T, MatrixFormat::Dense>& A,
                        const MatrixView<T, MatrixFormat::Dense>& B,
                        Span<int64_t> pivots,
                        Span<std::byte> work_space,
                        Span<int32_t> info);

/// @brief Workspace, in bytes, that gesv() needs for these operands on this queue.
///
/// Takes the same operands as the call, so both resolve the same route.
/// @throws batchlas::invalid_argument if gesv_validate_params() rejects the operands
/// @ingroup api_factorizations
template <Backend Back, typename T>
BATCHLAS_API size_t gesv_buffer_size(Queue& ctx,
                                     const MatrixView<T, MatrixFormat::Dense>& A,
                                     const MatrixView<T, MatrixFormat::Dense>& B);

/// @brief gesv() without per-item status (`info` not requested).
/// @ingroup api_factorizations
// Not a defaulted `info`: the sig:: aliases are function types (see potrf.hh).
template <Backend Back, typename T>
inline Event gesv(Queue& ctx,
                  const MatrixView<T, MatrixFormat::Dense>& A,
                  const MatrixView<T, MatrixFormat::Dense>& B,
                  Span<int64_t> pivots,
                  Span<std::byte> work_space) {
    return gesv<Back, T>(ctx, A, B, pivots, work_space, Span<int32_t>{});
}

BATCHLAS_ACCEPT_OWNING(gesv)
BATCHLAS_ACCEPT_OWNING(gesv_buffer_size)

BATCHLAS_DISPATCH_ON_QUEUE(gesv)
BATCHLAS_DISPATCH_ON_QUEUE(gesv_buffer_size)

}  // namespace batchlas
