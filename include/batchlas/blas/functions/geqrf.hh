// SPDX-License-Identifier: MIT
#pragma once

/// @file
/// @brief Batched Householder QR factorization (geqrf) and its workspace query.
/// @ingroup api_qr

#include <batchlas/export.hh>
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
using geqrf = Event(Queue&,
                    const MatrixView<T, MatrixFormat::Dense>&,
                    Span<T>, Span<std::byte>);

template <typename T>
using geqrf_buffer_size = size_t(Queue&,
                                 const MatrixView<T, MatrixFormat::Dense>&,
                                 Span<T>);

// Vendor signatures are spelled out, not aliased: a vendor parameter list can differ.
template <typename T>
using geqrf_vendor = Event(Queue&,
                           const MatrixView<T,MatrixFormat::Dense>&,
                           Span<T>,
                           Span<std::byte>);

template <typename T>
using geqrf_vendor_buffer_size = size_t(Queue&,
                                        const MatrixView<T,MatrixFormat::Dense>&,
                                        Span<T>);
}  // namespace sig

/// @brief Validates the arguments of the positional geqrf() entry point.
///
/// Checks only non-negative extents. Rectangular A of either orientation is
/// valid; the length of @p tau is checked by the option overloads.
/// @tparam T  scalar type of the matrix
/// @param[in] A  batch of \f$m \times n\f$ matrices whose extents are checked; not modified
/// @throws batchlas::invalid_argument on negative extents
/// @ingroup api_qr_lowlevel
// Runs in src/ops/geqrf/geqrf.cc before the selection key reads A.rows()/A.cols().
// Deliberately no squareness check (rectangular A is the point of geqrf), no
// m >= n check (the native cta and blocked kernels and the vendors serve m < n) and no
// tau-length check.
// evidence: docs/design/vendor-independence.md#positional-validators-reject-only-what-no-route-can-serve
template <typename T>
inline void geqrf_validate_params(const MatrixView<T, MatrixFormat::Dense>& A) {
    if (A.rows() < 0 || A.cols() < 0) {
        throw batchlas::invalid_argument(
            "geqrf: Matrix dimensions cannot be negative (rows=" +
            std::to_string(A.rows()) + ", cols=" + std::to_string(A.cols()) + ")");
    }
}


/// @brief Batched Householder QR factorization \f$ A = Q R \f$.
///
/// For every \f$m \times n\f$ batch item, as LAPACK `?geqrf`: on return the upper
/// triangle (upper trapezoid if \f$m < n\f$) of A holds \f$R\f$, and the part below the
/// diagonal holds the Householder vectors \f$ v_i \f$ with an implied unit
/// leading entry. With \f$ k = \min(m, n) \f$,
/// \f$ Q = H_1 H_2 \cdots H_k \f$, \f$ H_i = I - \tau_i v_i v_i^H \f$.
/// Pass A and @p tau unchanged to orgqr() to form \f$Q\f$ or to ormqr() to apply it.
///
/// Asynchronous: A and @p tau are readable after the returned event is waited on.
/// @tparam B  backend; the backend-deducing overload takes it from `ctx.backend()`
/// @tparam T  scalar type (float, double, std::complex<float>, std::complex<double>)
/// @param[in]     ctx        queue the kernels are enqueued on
/// @param[in,out] A          batch of \f$m \times n\f$ matrices; overwritten with \f$R\f$ and the
///                           reflectors \f$v_i\f$
/// @param[out]    tau        reflector scalars, `k * batch` elements; item \f$b\f$'s
///                           \f$\tau_i\f$ is `tau[b * k + i]`
/// @param[out]    workspace  device-accessible scratch of at least geqrf_buffer_size() bytes;
///                           contents undefined on return
/// @return event of the last enqueued kernel
/// @pre `tau.size() >= min(m, n) * A.batch_size()` (checked by the option overloads only)
/// @throws batchlas::invalid_argument on negative extents
/// @throws batchlas::NoRouteError if no native kernel can run the call (the native
///         kernels need a GPU queue with 32-wide sub-groups, a homogeneous batch and
///         \f$m, n \ge 1\f$) and the vendor library was not built in
/// @ingroup api_qr
template <Backend B, typename T>
BATCHLAS_API Event geqrf(Queue& ctx,
                         const MatrixView<T,MatrixFormat::Dense>& A,
                         Span<T> tau,
                         Span<std::byte> workspace);

/// @brief Workspace, in bytes, that geqrf() needs for this shape on this queue.
/// @tparam B  backend; the backend-deducing overload takes it from `ctx.backend()`
/// @tparam T  scalar type (float, double, std::complex<float>, std::complex<double>)
/// @param[in] ctx  queue the factorization will run on
/// @param[in] A    batch of \f$m \times n\f$ matrices to be factorized; only its shape is read
/// @param[in] tau  the reflector-scalar span that will be passed to geqrf(); not written
/// @return the minimum `workspace` size in bytes for geqrf() with these arguments
/// @ingroup api_qr_lowlevel
template <Backend B, typename T>
BATCHLAS_API size_t geqrf_buffer_size(Queue& ctx,
                                      const MatrixView<T,MatrixFormat::Dense>& A,
                                      Span<T> tau);

}  // namespace batchlas


namespace batchlas::backend {

/// @brief Vendor arm of geqrf(); called by geqrf() when it selects the `vendor`
///        kernel family, not by users.
/// @tparam B  backend whose vendor library runs the call
/// @tparam T  scalar type (float, double, std::complex<float>, std::complex<double>)
/// @param[in]     ctx        queue the vendor calls are enqueued on
/// @param[in,out] A          batch of \f$m \times n\f$ matrices; overwritten with \f$R\f$ and the
///                           reflectors \f$v_i\f$
/// @param[out]    tau        reflector scalars, \f$\min(m, n) \cdot \mathrm{batch}\f$ elements
/// @param[out]    workspace  scratch of at least geqrf_vendor_buffer_size() bytes; contents
///                           undefined on return
/// @return event of the last enqueued vendor call
/// @ingroup api_dispatch
// DECLARATION ONLY: the public geqrf is defined in src/ops/geqrf/geqrf.cc.
// evidence: docs/design/vendor-independence.md#the-entry-point-facade
template <Backend B, typename T>
BATCHLAS_API Event geqrf_vendor(Queue& ctx,
                                const MatrixView<T,MatrixFormat::Dense>& A,
                                Span<T> tau,
                                Span<std::byte> workspace);


/// @brief Workspace query of the vendor arm of geqrf().
/// @tparam B  backend whose vendor library runs the call
/// @tparam T  scalar type (float, double, std::complex<float>, std::complex<double>)
/// @param[in] ctx  queue the vendor arm will run on
/// @param[in] A    batch of \f$m \times n\f$ matrices to be factorized; only its shape is read
/// @param[in] tau  the reflector-scalar span that will be passed to geqrf_vendor(); not written
/// @return the minimum `workspace` size in bytes for geqrf_vendor()
/// @ingroup api_dispatch
template <Backend B, typename T>
BATCHLAS_API size_t geqrf_vendor_buffer_size(Queue& ctx,
                                             const MatrixView<T,MatrixFormat::Dense>& A,
                                             Span<T> tau);

}  // namespace batchlas::backend

namespace batchlas {

// Owning-container and backend-deducing overloads; see blas/queue-dispatch.hh.

BATCHLAS_DISPATCH_ON_QUEUE(geqrf)
BATCHLAS_DISPATCH_ON_QUEUE(geqrf_buffer_size)

}  // namespace batchlas
