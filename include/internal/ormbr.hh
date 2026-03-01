#pragma once

#include <blas/enums.hh>
#include <blas/matrix.hh>
#include <util/sycl-device-queue.hh>
#include <util/sycl-span.hh>

namespace batchlas {

template <Backend B, typename T>
Event ormbr(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& a,
            const VectorView<T>& tau,
            const MatrixView<T, MatrixFormat::Dense>& c,
            char vect,
            Side side,
            Transpose trans,
            const Span<std::byte>& ws,
            int32_t block_size);

template <Backend B, typename T>
size_t ormbr_buffer_size(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& a,
                         const VectorView<T>& tau,
                         const MatrixView<T, MatrixFormat::Dense>& c,
                         char vect,
                         Side side,
                         Transpose trans,
                         int32_t block_size);

} // namespace batchlas
