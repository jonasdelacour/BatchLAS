#pragma once

/// @file
/// @brief Blocked Hermitian tridiagonal reduction (sytrd/hetrd). Kernel helper, not API.
///
/// Installed with the rest of `include/batchlas` (the install rule copies the
/// tree wholesale); no public header includes it. Not a stable interface.
/// sytrd_blocked_buffer_size() repeats the documented declaration in
/// `batchlas/blas/extensions.hh` without its default `block_size`.
/// @ingroup api_internal_helpers

#include <batchlas/export.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/tuning_params.hh>
#include <batchlas/util/sycl-span.hh>

#include <cstddef>
#include <cstdint>

namespace batchlas {

template <Backend B, typename T>
BATCHLAS_API size_t sytrd_blocked_buffer_size(Queue& ctx,
                                              const MatrixView<T, MatrixFormat::Dense>& a,
                                              const VectorView<T>& d,
                                              const VectorView<T>& e,
                                              const VectorView<T>& tau,
                                              Uplo uplo,
                                              int32_t block_size);

/// @brief Blocked reduction \f$ Q^H A Q = T \f$ to real symmetric tridiagonal form.
///
/// Meant for n > 32, where the CTA kernel does not apply; same outputs and
/// layout as `sytrd_cta`. A is overwritten with the reflector storage (LAPACK
/// `sytd2` layout, with the tridiagonal entries restored on the first
/// off-diagonal), the diagonal goes to @p d, the off-diagonal to @p e and the
/// reflector scalars to @p tau.
/// @pre @p ctx is in-order (SYCL kernels and backend BLAS calls are not
///      separately ordered)
/// @pre @p ws holds at least sytrd_blocked_buffer_size() bytes (the W panel)
/// @throws batchlas::invalid_argument on a non-square A, short d/e/tau, a batch
///         mismatch or an out-of-order @p ctx
/// @throws batchlas::unsupported for `Uplo::Upper` (only Lower is implemented)
/// @ingroup api_internal_helpers
template <Backend B, typename T>
BATCHLAS_API Event sytrd_blocked(Queue& ctx,
                                 const MatrixView<T, MatrixFormat::Dense>& a,
                                 const VectorView<T>& d,
                                 const VectorView<T>& e,
                                 const VectorView<T>& tau,
                                 Uplo uplo,
                                 Span<std::byte> ws,
                                 int32_t block_size);

} // namespace batchlas
