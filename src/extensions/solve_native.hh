#pragma once

// Native FUSED factor-and-solve, declarations only: gesv (LU) and posv (Cholesky) for
// order n <= 32 and nrhs <= 4, one matrix per SubGroupPartition<N> with both A and the
// RHS in registers. evidence: docs/perf/lu.md#the-fused-gesv-tier
//
// CONTRACT: the kernel consumes the UNFACTORED A and leaves BOTH outputs behind -- A
// overwritten by its factors and `pivots` filled exactly as getrf would, B by X -- so a
// caller can swap `getrf; getrs` for `gesv` and keep every reader of the factor working.

#include "../util/internal-api.hh"

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>

#include <cstddef>
#include <cstdint>

namespace batchlas::sycl_gesv {

inline constexpr int kGesvTinyMaxRhs = 4;

// The ONE place the order ceiling is spelled; there is no vendor arm to absorb a fork.
template <typename T>
BATCHLAS_INTERNAL_API int gesv_tiny_max_n();

// NOT zero: a short or empty `info` span means "not requested" and draws pool scratch.
template <typename T>
BATCHLAS_INTERNAL_API std::size_t gesv_tiny_buffer_size(
    Queue& ctx,
    const MatrixView<T, MatrixFormat::Dense>& A,
    const MatrixView<T, MatrixFormat::Dense>& B);

template <typename T>
BATCHLAS_INTERNAL_API Event gesv_tiny_dispatch(Queue& ctx,
                                               const MatrixView<T, MatrixFormat::Dense>& A,
                                               const MatrixView<T, MatrixFormat::Dense>& B,
                                               Span<int64_t> pivots,
                                               Span<std::byte> workspace,
                                               Span<int32_t> info);

}  // namespace batchlas::sycl_gesv

namespace batchlas::sycl_posv {

inline constexpr int kPosvTinyMaxRhs = 4;

template <typename T>
BATCHLAS_INTERNAL_API int posv_tiny_max_n();

template <typename T>
BATCHLAS_INTERNAL_API std::size_t posv_tiny_buffer_size(
    Queue& ctx,
    const MatrixView<T, MatrixFormat::Dense>& A,
    const MatrixView<T, MatrixFormat::Dense>& B);

template <typename T>
BATCHLAS_INTERNAL_API Event posv_tiny_dispatch(Queue& ctx,
                                               const MatrixView<T, MatrixFormat::Dense>& A,
                                               const MatrixView<T, MatrixFormat::Dense>& B,
                                               Uplo uplo,
                                               Span<std::byte> workspace,
                                               Span<int32_t> info);

}  // namespace batchlas::sycl_posv
