// SPDX-License-Identifier: MIT
#pragma once

/// @file
/// @brief Blocked (WY) application of Q from geqrf reflectors. Kernel helper, not API.
///
/// Installed only because `batchlas/blas/functions/ormqr.hh` includes it; not a
/// stable interface.
/// @ingroup api_internal_helpers

#include <batchlas/export.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>

namespace batchlas {

/// @brief Blocked native arm of ormqr(): applies Q with the compact WY form.
///
/// Groups the geqrf() reflectors (forward, columnwise, unit-lower V) into panels
/// of @p block_size and applies each as \f$ I - V T V^H \f$ (LAPACK
/// `larft`/`larfb`) with level-3 GEMMs. Same result as ormqr(); batched through
/// strided-batch views. Meant for medium sizes, where the CTA kernels do not apply.
/// @tparam B  backend the call is compiled for
/// @tparam T  scalar type of the operands
/// @param[in]     ctx         in-order queue the kernels are enqueued on
/// @param[in]     a           batch of geqrf() outputs holding \f$k\f$ reflectors; \f$n_q\f$ rows, the
///                            order of Q (\f$m\f$ for `Side::Left`, \f$n\f$ for `Side::Right`); not modified
/// @param[in,out] c           batch of \f$m \times n\f$ matrices; overwritten with
///                            \f$\mathrm{op}(Q)\,C\f$ (Left) or \f$C\,\mathrm{op}(Q)\f$ (Right)
/// @param[in]     side        apply Q from the left or the right
/// @param[in]     trans       apply Q (`NoTrans`) or \f$Q^H\f$ (`ConjTrans`; `Trans` for real T)
/// @param[in]     tau         reflector scalars from geqrf(), at least \f$k \cdot \mathit{batch\_size}\f$ elements,
///                            \f$k = \min(\text{rows}, \text{cols})\f$ of @p a
/// @param[out]    workspace   device-accessible scratch for V, T and W; contents undefined on return
/// @param[in]     block_size  panel width; 0 (the default) selects `tuning::ORMQR_BLOCK_SIZE_MEDIUM`
///                            (`batchlas/tuning_params.hh`), and a negative value is treated as 1. Pass the same
///                            value to ormqr_blocked_buffer_size().
/// @return event of the last enqueued kernel; @p c is valid once it completes
/// @pre @p ctx is in-order (the pack / larft / GEMM steps are not separately ordered)
/// @pre @p workspace holds at least ormqr_blocked_buffer_size() bytes (V, T and W)
/// @throws batchlas::invalid_argument on mismatched batch or order, a batch size below 1, a short @p tau,
///         or an out-of-order @p ctx
/// @throws batchlas::unsupported for `Transpose::Trans` with complex T (use `ConjTrans`)
/// @ingroup api_internal_helpers
template <Backend B, typename T>
BATCHLAS_API Event ormqr_blocked(Queue& ctx,
                                 const MatrixView<T, MatrixFormat::Dense>& a,
                                 const MatrixView<T, MatrixFormat::Dense>& c,
                                 Side side,
                                 Transpose trans,
                                 Span<T> tau,
                                 Span<std::byte> workspace,
                                 int32_t block_size = 0);

/// @brief Workspace, in bytes, that ormqr_blocked() needs for this @p block_size.
/// @tparam B  backend the call is compiled for
/// @tparam T  scalar type of the operands
/// @param[in] ctx         queue the call will run on (sizes the batch chunks)
/// @param[in] a           reflector batch, as for ormqr_blocked()
/// @param[in] c           batch of \f$m \times n\f$ matrices to multiply, as for ormqr_blocked()
/// @param[in] side        apply Q from the left or the right
/// @param[in] trans       apply Q or its (conjugate) transpose; does not change the size
/// @param[in] tau         reflector scalars, as for ormqr_blocked(); only its size is checked
/// @param[in] block_size  as for ormqr_blocked(); 0 selects `tuning::ORMQR_BLOCK_SIZE_MEDIUM`
/// @return the workspace size in bytes
/// @throws batchlas::invalid_argument on mismatched batch or order, a batch size below 1, or a short @p tau
/// @ingroup api_internal_helpers
template <Backend B, typename T>
BATCHLAS_API size_t ormqr_blocked_buffer_size(Queue& ctx,
                                              const MatrixView<T, MatrixFormat::Dense>& a,
                                              const MatrixView<T, MatrixFormat::Dense>& c,
                                              Side side,
                                              Transpose trans,
                                              Span<T> tau,
                                              int32_t block_size = 0);

} // namespace batchlas
