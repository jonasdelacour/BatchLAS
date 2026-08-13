#pragma once

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

// backend::orgqr_vendor's signature, spelled out from the definition rather than
// aliased to sig::orgqr: a vendor parameter list can differ from the public one.
template <typename T>
using orgqr_vendor = Event(Queue&,
                           const MatrixView<T, MatrixFormat::Dense>&,
                           Span<T>,
                           Span<std::byte>);

// backend::orgqr_vendor_buffer_size's signature, spelled out from the definition rather than
// aliased to sig::orgqr_buffer_size: a vendor parameter list can differ from the public one.
template <typename T>
using orgqr_vendor_buffer_size = size_t(Queue&,
                                        const MatrixView<T, MatrixFormat::Dense>&,
                                        Span<T>);
}  // namespace sig


template <Backend B, typename T>
Event orgqr(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& A,
            Span<T> tau,
            Span<std::byte> workspace);

template <Backend B, typename T>
inline Event orgqr(Queue& ctx,
                        const Matrix<T, MatrixFormat::Dense>& A,
                        Span<T> tau,
                        Span<std::byte> workspace) {
        return orgqr<B,T>(ctx, MatrixView<T, MatrixFormat::Dense>(A), tau, workspace);
}

template <Backend B, typename T>
size_t orgqr_buffer_size(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         Span<T> tau);

template <Backend B, typename T>
inline size_t orgqr_buffer_size(Queue& ctx,
                                                 const Matrix<T, MatrixFormat::Dense>& A,
                                                 Span<T> tau) {
        return orgqr_buffer_size<B,T>(ctx, MatrixView<T, MatrixFormat::Dense>(A), tau);
}

}  // namespace batchlas


namespace batchlas::backend {

// The vendor path for orgqr.
//
// DECLARATION ONLY -- see the note on gemm_vendor in gemm.hh. The public
// `orgqr` used to be defined inside each vendor TU, so dropping a vendor library
// dropped the public entry point with it; WP0 S5 moves that definition to
// src/dispatch/entry_points/factorization.cc and leaves the vendor
// implementation here, named as such.
template <Backend B, typename T>
Event orgqr_vendor(Queue& ctx,
                   const MatrixView<T, MatrixFormat::Dense>& A,
                   Span<T> tau,
                   Span<std::byte> workspace);


template <Backend B, typename T>
size_t orgqr_vendor_buffer_size(Queue& ctx,
                                const MatrixView<T, MatrixFormat::Dense>& A,
                                Span<T> tau);

}  // namespace batchlas::backend

namespace batchlas {

// Backend-deducing overloads: `f(ctx, ...)` uses ctx.backend().
// See BATCHLAS_DISPATCH_ON_QUEUE in blas/queue-dispatch.hh.

BATCHLAS_DISPATCH_ON_QUEUE(orgqr)
BATCHLAS_DISPATCH_ON_QUEUE(orgqr_buffer_size)

}  // namespace batchlas
