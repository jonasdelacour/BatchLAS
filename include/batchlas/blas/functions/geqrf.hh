#pragma once

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

// backend::geqrf_vendor's signature, spelled out from the definition rather than
// aliased to sig::geqrf: a vendor parameter list can differ from the public one.
template <typename T>
using geqrf_vendor = Event(Queue&,
                           const MatrixView<T,MatrixFormat::Dense>&,
                           Span<T>,
                           Span<std::byte>);

// backend::geqrf_vendor_buffer_size's signature, spelled out from the definition rather than
// aliased to sig::geqrf_buffer_size: a vendor parameter list can differ from the public one.
template <typename T>
using geqrf_vendor_buffer_size = size_t(Queue&,
                                        const MatrixView<T,MatrixFormat::Dense>&,
                                        Span<T>);
}  // namespace sig

// Validation for the POSITIONAL entry point, which had none.
//
// It runs in the facade (src/dispatch/entry_points/factorization.cc), AHEAD of
// the shape builder, because the builder reads A.rows()/A.cols() and must not
// describe a non-conforming view. Same hoist, and same reason, as potrf's
// (potrf.hh:66-84) and trsm's (entry_points/level3.cc:167-174).
//
// SCOPE IS DELIBERATELY MINIMAL -- EXACTLY WHAT THE SHAPE BUILDER NEEDS, and for
// geqrf that is one line. Three things it deliberately does NOT check, each for a
// stated reason:
//
//   * NO SQUARENESS CHECK. Rectangular A is the entire point of geqrf
//     (options.hh:727-730), and the library's own callers pass tall panels
//     (band_reduction.cc:595, sytrd_sy2sb.cc:504). Copying potrf.hh:76's
//     `A.rows() != A.cols()` here would be a wrong edit.
//
//   * NO `m >= n` CHECK, even though RouteTable<Op::geqrf,T>::supports() carries
//     one. That gate says "the native drivers cannot serve a wide view", which
//     routes it to the vendor; it does not say the CALL is invalid, and the
//     vendor serves it. A validator that threw would turn a working call into an
//     error.
//
//   * NO tau LENGTH CHECK. options.hh:718-719 already does
//     require_span_at_least on the arena spellings; turning a currently-tolerated
//     short span into a throw on the positional one is a user-visible behaviour
//     change and belongs in its own commit with its own test. potrf.hh:59-65
//     states the rule.
template <typename T>
inline void geqrf_validate_params(const MatrixView<T, MatrixFormat::Dense>& A) {
    if (A.rows() < 0 || A.cols() < 0) {
        throw std::invalid_argument(
            "GEQRF: Matrix dimensions cannot be negative (rows=" +
            std::to_string(A.rows()) + ", cols=" + std::to_string(A.cols()) + ")");
    }
}


template <Backend B, typename T>
Event geqrf(Queue& ctx,
            const MatrixView<T,MatrixFormat::Dense>& A,
            Span<T> tau,
            Span<std::byte> work_space);

template <Backend B, typename T>
inline Event geqrf(Queue& ctx,
                        const Matrix<T,MatrixFormat::Dense>& A,
                        Span<T> tau,
                        Span<std::byte> work_space) {
        return geqrf<B,T>(ctx, MatrixView<T, MatrixFormat::Dense>(A), tau, work_space);
}

template <Backend B, typename T>
size_t geqrf_buffer_size(Queue& ctx,
                         const MatrixView<T,MatrixFormat::Dense>& A,
                         Span<T> tau);

template <Backend B, typename T>
inline size_t geqrf_buffer_size(Queue& ctx,
                                                 const Matrix<T,MatrixFormat::Dense>& A,
                                                 Span<T> tau) {
        return geqrf_buffer_size<B,T>(ctx, MatrixView<T, MatrixFormat::Dense>(A), tau);
}

}  // namespace batchlas


namespace batchlas::backend {

// The vendor path for geqrf.
//
// DECLARATION ONLY -- see the note on gemm_vendor in gemm.hh. The public
// `geqrf` used to be defined inside each vendor TU, so dropping a vendor library
// dropped the public entry point with it; WP0 S5 moves that definition to
// src/dispatch/entry_points/factorization.cc and leaves the vendor
// implementation here, named as such.
template <Backend B, typename T>
Event geqrf_vendor(Queue& ctx,
                   const MatrixView<T,MatrixFormat::Dense>& A,
                   Span<T> tau,
                   Span<std::byte> work_space);


template <Backend B, typename T>
size_t geqrf_vendor_buffer_size(Queue& ctx,
                                const MatrixView<T,MatrixFormat::Dense>& A,
                                Span<T> tau);

}  // namespace batchlas::backend

namespace batchlas {

// Backend-deducing overloads: `f(ctx, ...)` uses ctx.backend().
// See BATCHLAS_DISPATCH_ON_QUEUE in blas/queue-dispatch.hh.

BATCHLAS_DISPATCH_ON_QUEUE(geqrf)
BATCHLAS_DISPATCH_ON_QUEUE(geqrf_buffer_size)

}  // namespace batchlas
