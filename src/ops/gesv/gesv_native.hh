// SPDX-License-Identifier: MIT
#pragma once

// Native fused gesv (n <= 32, nrhs <= 4, A and the RHS in registers). CONTRACT: A is overwritten by its
// factors and `pivots` filled exactly as getrf would, B by X, so `getrf; getrs` -> `gesv` keeps every
// reader of the factor working. evidence: docs/perf/lu.md#the-fused-gesv-tier

#include <src/util/internal-api.hh>
#include <src/util/resident_capacity.hh>

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>

#include <cstddef>
#include <cstdint>

namespace batchlas::ops::gesv {

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

}  // namespace batchlas::ops::gesv
