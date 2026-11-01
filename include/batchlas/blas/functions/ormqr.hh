// SPDX-License-Identifier: MIT
#pragma once

/// @file
/// @brief Batched application of Q from geqrf reflectors (ormqr/unmqr) and its workspace query.
/// @ingroup api_qr

#include <batchlas/export.hh>
#include <algorithm>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <complex>

#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/enums.hh>

#include <batchlas/internal/ormqr_blocked.hh>

#include <batchlas/blas/queue-dispatch.hh>

namespace batchlas {

// Signature aliases for explicit instantiation; see BATCHLAS_INSTANTIATE in
// src/util/template-instantiations.hh. Keep in sync with the declarations below.
namespace sig {
template <typename T>
using ormqr = Event(Queue&,
                    const MatrixView<T, MatrixFormat::Dense>&,
                    const MatrixView<T, MatrixFormat::Dense>&,
                    Side, Transpose, Span<T>, Span<std::byte>,
                    int32_t);

template <typename T>
using ormqr_buffer_size = size_t(Queue&,
                                 const MatrixView<T, MatrixFormat::Dense>&,
                                 const MatrixView<T, MatrixFormat::Dense>&,
                                 Side, Transpose, Span<T>,
                                 int32_t);

// The vendor signatures deliberately drop the block-size hint (a WY panel width
// means nothing to a vendor kernel), so they are spelled out, not aliased.
template <typename T>
using ormqr_vendor = Event(Queue&,
                           const MatrixView<T, MatrixFormat::Dense>&,
                           const MatrixView<T, MatrixFormat::Dense>&,
                           Side, Transpose, Span<T>, Span<std::byte>);

template <typename T>
using ormqr_vendor_buffer_size = size_t(Queue&,
                                        const MatrixView<T, MatrixFormat::Dense>&,
                                        const MatrixView<T, MatrixFormat::Dense>&,
                                        Side, Transpose, Span<T>);
}  // namespace sig


/// @brief Batched multiplication by the orthogonal/unitary Q from geqrf().
///
/// As LAPACK `?ormqr` / `?unmqr`. A (\f$m \times n\f$) holds, below its diagonal, the
/// \f$ k = \min(m, n) \f$ reflectors that geqrf() wrote, with
/// \f$ Q = H_1 \cdots H_k \f$ of order \f$m\f$ (`A.rows()`). For every batch item C is
/// overwritten with
/// \f$ \mathrm{op}(Q)\, C \f$ (`Side::Left`) or \f$ C\, \mathrm{op}(Q) \f$
/// (`Side::Right`), where \f$ \mathrm{op}(Q) \f$ is Q for `NoTrans` and
/// \f$ Q^H \f$ for `ConjTrans` (\f$ Q^T \f$ for real T, where `Trans` means the
/// same). Q is never formed.
///
/// The kernel (the blocked WY apply on a GPU queue, or the vendor library) is
/// chosen per call from the device's tuned table (src/ops/ormqr/ormqr.cc) and
/// can be pinned with `BATCHLAS_ORMQR_ROUTE`.
///
/// Asynchronous: C is readable after the returned event is waited on. On an
/// out-of-order @p ctx the work runs on an internal in-order queue that first
/// waits for everything already submitted to @p ctx.
/// @tparam B  backend; the backend-deducing overload takes it from `ctx.backend()`
/// @tparam T  scalar type (float, double, std::complex<float>, std::complex<double>)
/// @param[in]     ctx              queue the kernels are enqueued on
/// @param[in]     A                batch of \f$m \times n\f$ geqrf() outputs holding the reflectors;
///                                 not modified
/// @param[in,out] C                batch of matrices to multiply (\f$m\f$ rows for `Side::Left`,
///                                 \f$m\f$ columns for `Side::Right`); overwritten with the product
/// @param[in]     side             apply Q from the left or the right
/// @param[in]     trans            apply Q (`NoTrans`) or its (conjugate) transpose
/// @param[in]     tau              reflector scalars from geqrf(), `k * batch` elements
/// @param[out]    workspace        device-accessible scratch of at least ormqr_buffer_size() bytes,
///                                 sized with the same @p block_size_hint; contents undefined on return
/// @param[in]     block_size_hint  WY panel width for the blocked kernel; 0 lets the tuning
///                                 table choose. Clamped to \f$[1, k]\f$; ignored by the vendor kernel.
/// @return event of the last enqueued kernel
/// @pre `A.rows() == C.rows()` for `Side::Left`, `A.rows() == C.cols()` for
///      `Side::Right`, and A and C share a batch size
/// @pre `tau.size() >= k * batch` (checked by the option overloads)
/// @throws batchlas::workspace_error if @p workspace is smaller than the chosen kernel needs
/// @throws batchlas::invalid_argument for `Transpose::Trans` with complex T (use
///         `ConjTrans`), and on non-conforming operands (blocked kernel)
/// @throws batchlas::NoRouteError if no native kernel can run (e.g. a CPU queue) and
///         the vendor library was not built in
/// @see OrmqrOptions
/// @ingroup api_qr
// Defined in src/ops/ormqr/ormqr.cc, which picks the kernel (flat selection).
template <Backend B, typename T>
BATCHLAS_API Event ormqr(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         const MatrixView<T, MatrixFormat::Dense>& C,
                         Side side,
                         Transpose trans,
                         Span<T> tau,
                         Span<std::byte> workspace,
                         int32_t block_size_hint = 0);

/// @brief Workspace, in bytes, that ormqr() needs for these operands on this queue.
///
/// Makes the same kernel choice and block width as ormqr() from the same inputs, so
/// the result is valid for a call with the same arguments and @p block_size_hint.
/// `evidence: docs/perf/qr.md#ormqr-one-route-resolution-for-the-call-and-its-size-query`
/// @param[in] ctx              queue the ormqr() call will run on (kernel selection reads its device)
/// @param[in] A                batch of \f$m \times n\f$ reflectors the call will get; only the shape is read
/// @param[in] C                batch of matrices the call will multiply; only the shape is read
/// @param[in] side             the call's `Side`
/// @param[in] trans            the call's `Transpose`
/// @param[in] tau              the call's reflector scalars; contents not read
/// @param[in] block_size_hint  the call's WY panel width hint (0 = tuning table)
/// @return bytes to pass as the `workspace` span of ormqr()
/// @ingroup api_qr_lowlevel
template <Backend B, typename T>
BATCHLAS_API size_t ormqr_buffer_size(Queue& ctx,
                                      const MatrixView<T, MatrixFormat::Dense>& A,
                                      const MatrixView<T, MatrixFormat::Dense>& C,
                                      Side side,
                                      Transpose trans,
                                      Span<T> tau,
                                      int32_t block_size_hint = 0);

} // namespace batchlas

namespace batchlas::backend {

/// @brief Vendor arm of ormqr() (cuSOLVER / rocSOLVER / LAPACKE); not for direct use.
/// @param[in]     ctx        queue the kernels are enqueued on
/// @param[in]     A          batch of \f$m \times n\f$ geqrf() outputs holding the reflectors
/// @param[in,out] C          batch of matrices to multiply; overwritten with the product
/// @param[in]     side       apply Q from the left or the right
/// @param[in]     trans      apply Q (`NoTrans`) or its (conjugate) transpose
/// @param[in]     tau        reflector scalars from geqrf(), `k * batch` elements
/// @param[out]    workspace  scratch of at least ormqr_vendor_buffer_size() bytes; contents undefined
///                           on return
/// @return event of the last enqueued kernel
/// @ingroup api_dispatch
template <Backend B, typename T>
BATCHLAS_API Event ormqr_vendor(Queue& ctx,
                                const MatrixView<T, MatrixFormat::Dense>& A,
                                const MatrixView<T, MatrixFormat::Dense>& C,
                                Side side,
                                Transpose trans,
                                Span<T> tau,
                                Span<std::byte> workspace);

/// @brief Workspace query of the vendor arm of ormqr().
/// @param[in] ctx    queue the call will run on
/// @param[in] A      batch of \f$m \times n\f$ reflectors the call will get; only the shape is read
/// @param[in] C      batch of matrices the call will multiply; only the shape is read
/// @param[in] side   the call's `Side`
/// @param[in] trans  the call's `Transpose`
/// @param[in] tau    the call's reflector scalars; contents not read
/// @return bytes to pass as the `workspace` span of backend::ormqr_vendor()
/// @ingroup api_dispatch
template <Backend B, typename T>
BATCHLAS_API size_t ormqr_vendor_buffer_size(Queue& ctx,
                                             const MatrixView<T, MatrixFormat::Dense>& A,
                                             const MatrixView<T, MatrixFormat::Dense>& C,
                                             Side side,
                                             Transpose trans,
                                             Span<T> tau);

} // namespace batchlas::backend


namespace batchlas {

// Owning-container and backend-deducing overloads; see blas/queue-dispatch.hh.

BATCHLAS_DISPATCH_ON_QUEUE(ormqr)
BATCHLAS_DISPATCH_ON_QUEUE(ormqr_buffer_size)

}  // namespace batchlas
