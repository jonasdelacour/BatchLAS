#pragma once

/// @file
/// @brief Batched formation of Q from geqrf reflectors (orgqr/ungqr) and its workspace query.
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
using orgqr = Event(Queue&,
                    const MatrixView<T, MatrixFormat::Dense>&,
                    Span<T>, Span<std::byte>);

template <typename T>
using orgqr_buffer_size = size_t(Queue&,
                                 const MatrixView<T, MatrixFormat::Dense>&,
                                 Span<T>);

// Vendor signatures are spelled out, not aliased: a vendor parameter list can differ.
template <typename T>
using orgqr_vendor = Event(Queue&,
                           const MatrixView<T, MatrixFormat::Dense>&,
                           Span<T>,
                           Span<std::byte>);

template <typename T>
using orgqr_vendor_buffer_size = size_t(Queue&,
                                        const MatrixView<T, MatrixFormat::Dense>&,
                                        Span<T>);
}  // namespace sig

/// @brief Validates the arguments of the positional orgqr() entry point.
///
/// Checks only non-negative extents. `n <= m` is not checked: an n > m view is
/// served by the vendor kernel, the only one whose `can_run` admits it
/// (src/ops/orgqr/orgqr.cc). The length of `tau` is checked by the option overloads.
/// @throws batchlas::invalid_argument on negative extents
/// @ingroup api_qr_lowlevel
// Deliberately no n <= m check: rejecting such a view is a user-visible behaviour change.
// evidence: docs/design/vendor-independence.md#positional-validators-reject-only-what-no-route-can-serve
template <typename T>
inline void orgqr_validate_params(const MatrixView<T, MatrixFormat::Dense>& A) {
    if (A.rows() < 0 || A.cols() < 0) {
        throw batchlas::invalid_argument(
            "ORGQR: Matrix dimensions cannot be negative (rows=" +
            std::to_string(A.rows()) + ", cols=" + std::to_string(A.cols()) + ")");
    }
}


/// @brief Batched formation of the first n columns of Q from geqrf() output.
///
/// As LAPACK `?orgqr` / `?ungqr` with \f$ k = n \f$: A holds, below its
/// diagonal, the reflectors written by geqrf() on an m x n matrix, and on
/// return A holds the m x n matrix
/// \f$ Q = H_1 H_2 \cdots H_n \f$ restricted to its first n columns, with
/// orthonormal columns (\f$ Q^H Q = I \f$).
///
/// Asynchronous: A is readable after the returned event is waited on.
/// @tparam B  backend; the backend-deducing overload takes it from `ctx.backend()`
/// @tparam T  scalar type (float, double, std::complex<float>, std::complex<double>)
/// @param ctx        queue the kernels are enqueued on
/// @param A          batch of m x n geqrf() outputs, m >= n; overwritten with Q
/// @param tau        reflector scalars from geqrf(), `n * batch` elements
/// @param workspace  device-accessible scratch of at least orgqr_buffer_size() bytes
/// @return event of the last enqueued kernel
/// @pre `A.cols() <= A.rows()`; not checked by any overload
/// @pre `tau.size() >= min(m, n) * batch` (checked by the option overloads only)
/// @throws batchlas::invalid_argument on negative extents
/// @throws batchlas::NoRouteError if no native kernel can run the shape (the
///         blocked kernel needs a GPU queue, a homogeneous batch and n <= m)
///         and the vendor library was not built in
/// @ingroup api_qr
template <Backend B, typename T>
BATCHLAS_API Event orgqr(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         Span<T> tau,
                         Span<std::byte> workspace);

/// @brief Workspace, in bytes, that orgqr() needs for this shape on this queue.
/// @ingroup api_qr_lowlevel
template <Backend B, typename T>
BATCHLAS_API size_t orgqr_buffer_size(Queue& ctx,
                                      const MatrixView<T, MatrixFormat::Dense>& A,
                                      Span<T> tau);

}  // namespace batchlas


namespace batchlas::backend {

/// @brief Vendor arm of orgqr(); called by the public orgqr(), not by users.
/// @ingroup api_dispatch
// Declaration only: the public orgqr lives in src/ops/orgqr/orgqr.cc.
// evidence: docs/design/vendor-independence.md#the-entry-point-facade
template <Backend B, typename T>
BATCHLAS_API Event orgqr_vendor(Queue& ctx,
                                const MatrixView<T, MatrixFormat::Dense>& A,
                                Span<T> tau,
                                Span<std::byte> workspace);


/// @brief Workspace query of the vendor arm of orgqr().
/// @ingroup api_dispatch
template <Backend B, typename T>
BATCHLAS_API size_t orgqr_vendor_buffer_size(Queue& ctx,
                                             const MatrixView<T, MatrixFormat::Dense>& A,
                                             Span<T> tau);

}  // namespace batchlas::backend

namespace batchlas {

// Owning-container and backend-deducing overloads; see blas/queue-dispatch.hh.
BATCHLAS_ACCEPT_OWNING(orgqr)
BATCHLAS_ACCEPT_OWNING(orgqr_buffer_size)

BATCHLAS_DISPATCH_ON_QUEUE(orgqr)
BATCHLAS_DISPATCH_ON_QUEUE(orgqr_buffer_size)

}  // namespace batchlas
