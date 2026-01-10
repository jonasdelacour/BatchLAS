#pragma once

#include <blas/matrix.hh>
#include <util/sycl-span.hh>

#include <cstddef>
#include <cstdint>

namespace batchlas {

// Blocked symmetric/Hermitian tridiagonal reduction (SYTRD/HETRD-style).
//
// Intended for medium/large matrices (n > 32). Produces the same outputs/layout
// as `sytrd_cta`:
// - Overwrites A with reflector storage (SYTD2-style) and later restores the
//   tridiagonal entries on the first off-diagonal.
// - Outputs diagonal/off-diagonal in (d,e) and reflector scalars in tau.
//
// Notes:
// - Currently optimized for in-order queues (required for correct sequencing
//   across SYCL kernels and backend BLAS calls).
// - `ws` is required (W workspace).

template <Backend B, typename T>
size_t sytrd_blocked_buffer_size(Queue& ctx,
                                 const MatrixView<T, MatrixFormat::Dense>& a,
                                 const VectorView<T>& d,
                                 const VectorView<T>& e,
                                 const VectorView<T>& tau,
                                 Uplo uplo,
                                 int32_t block_size);

template <Backend B, typename T>
Event sytrd_blocked(Queue& ctx,
                    const MatrixView<T, MatrixFormat::Dense>& a,
                    const VectorView<T>& d,
                    const VectorView<T>& e,
                    const VectorView<T>& tau,
                    Uplo uplo,
                    Span<std::byte> ws,
                    int32_t block_size);

} // namespace batchlas
