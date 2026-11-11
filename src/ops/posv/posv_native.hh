// SPDX-License-Identifier: MIT
#pragma once

// Native fused posv (n <= 32, nrhs <= 4, A and the RHS in registers, as gesv_native.hh). CONTRACT: A is
// overwritten by its Cholesky factor exactly as potrf would and B by X, so `potrf; potrs` -> `posv`.

#include <src/util/internal-api.hh>
#include <src/util/resident_capacity.hh>

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>

#include <cstddef>
#include <cstdint>

namespace batchlas::ops::posv {

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

}  // namespace batchlas::ops::posv
