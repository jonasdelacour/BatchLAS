#pragma once

/// @file
/// @brief Blocked (WY) application of Q from geqrf reflectors. Kernel helper, not API.
///
/// Installed only because `batchlas/blas/functions/ormqr.hh` includes it; not a
/// stable interface.
/// @ingroup internal_helpers

#include <batchlas/export.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/tuning_params.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>

namespace batchlas {

/// @brief Blocked native arm of ormqr(): applies Q with the compact WY form.
///
/// Groups the geqrf() reflectors (forward, columnwise, unit-lower V) into panels
/// of @p block_size and applies each as \f$ I - V T V^H \f$ (LAPACK
/// `larft`/`larfb`) with level-3 GEMMs. Same result as ormqr(); batched through
/// strided-batch views. Meant for medium sizes, where the CTA kernels do not apply.
/// @pre @p ctx is in-order (the pack / larft / GEMM steps are not separately ordered)
/// @pre @p workspace holds at least ormqr_blocked_buffer_size() bytes (V, T and W)
/// @throws batchlas::invalid_argument on mismatched batch or order, a short @p tau,
///         or an out-of-order @p ctx
/// @throws batchlas::unsupported for `Transpose::Trans` with complex T (use `ConjTrans`)
/// @ingroup internal_helpers
template <Backend B, typename T>
BATCHLAS_API Event ormqr_blocked(Queue& ctx,
                                 const MatrixView<T, MatrixFormat::Dense>& a,
                                 const MatrixView<T, MatrixFormat::Dense>& c,
                                 Side side,
                                 Transpose trans,
                                 Span<T> tau,
                                 Span<std::byte> workspace,
                                 int32_t block_size = tuning::ORMQR_BLOCK_SIZE_MEDIUM);

/// @brief Workspace, in bytes, that ormqr_blocked() needs for this @p block_size.
/// @ingroup internal_helpers
template <Backend B, typename T>
BATCHLAS_API size_t ormqr_blocked_buffer_size(Queue& ctx,
                                              const MatrixView<T, MatrixFormat::Dense>& a,
                                              const MatrixView<T, MatrixFormat::Dense>& c,
                                              Side side,
                                              Transpose trans,
                                              Span<T> tau,
                                              int32_t block_size = tuning::ORMQR_BLOCK_SIZE_MEDIUM);

} // namespace batchlas
