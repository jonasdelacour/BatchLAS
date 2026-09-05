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
using getrs = Event(Queue&,
                    const MatrixView<T, MatrixFormat::Dense>&,
                    const MatrixView<T, MatrixFormat::Dense>&,
                    Transpose, Span<int64_t>, Span<std::byte>);

template <typename T>
using getrs_buffer_size = size_t(Queue&,
                                 const MatrixView<T, MatrixFormat::Dense>&,
                                 const MatrixView<T, MatrixFormat::Dense>&,
                                 Transpose);

// backend::getrs_vendor's signature, spelled out from the definition rather than
// aliased to sig::getrs: a vendor parameter list can differ from the public one.
template <typename T>
using getrs_vendor = Event(Queue&,
                           const MatrixView<T,MatrixFormat::Dense>&,
                           const MatrixView<T,MatrixFormat::Dense>&,
                           Transpose,
                           Span<int64_t>,
                           Span<std::byte>);

// backend::getrs_vendor_buffer_size's signature, spelled out from the definition rather than
// aliased to sig::getrs_buffer_size: a vendor parameter list can differ from the public one.
template <typename T>
using getrs_vendor_buffer_size = size_t(Queue&,
                                        const MatrixView<T,MatrixFormat::Dense>&,
                                        const MatrixView<T,MatrixFormat::Dense>&,
                                        Transpose);
}  // namespace sig


// WP6: the one thing that is invalid for EVERY route, checked once, hoisted above
// the shape builder in src/dispatch/entry_points/factorization.cc because the
// builder reads A.rows()/B.cols(). Modelled on geqrf_validate_params
// (geqrf.hh:71-77) and it obeys geqrf.hh:55-70's rule: validate only what no route
// could serve.
//
// WHAT IT DELIBERATELY DOES NOT CHECK: squareness of A, A.rows() == B.rows(),
// equal batch, and the pivot span's length. All four ARE checked on the arena
// spellings (options.hh:646-650) and the first three make
// backend::getrs_op_shape return nullopt, which routes the call to the vendor.
// Routing a call away from the native arms is not the same as rejecting it, and a
// validator that threw would turn a currently-working positional call into an
// error -- the behaviour change potrf.hh:59-65 rules out of scope for a
// scaffolding step.
template <typename T>
inline void getrs_validate_params(const MatrixView<T, MatrixFormat::Dense>& A,
                                  const MatrixView<T, MatrixFormat::Dense>& B) {
    if (A.rows() < 0 || A.cols() < 0 || B.rows() < 0 || B.cols() < 0) {
        throw std::invalid_argument(
            "GETRS: Matrix dimensions cannot be negative (A: rows=" +
            std::to_string(A.rows()) + ", cols=" + std::to_string(A.cols()) +
            "; B: rows=" + std::to_string(B.rows()) +
            ", cols=" + std::to_string(B.cols()) + ")");
    }
}


template <Backend Back, typename T>
Event getrs(Queue& ctx,
           const MatrixView<T, MatrixFormat::Dense>& A,
           const MatrixView<T, MatrixFormat::Dense>& B,
           Transpose transA,
           Span<int64_t> pivots,
           Span<std::byte> work_space);

template <Backend Back, typename T>
inline Event getrs(Queue& ctx,
                   const Matrix<T, MatrixFormat::Dense>& A,
                   const Matrix<T, MatrixFormat::Dense>& Bmat,
                   Transpose transA,
                   Span<int64_t> pivots,
                   Span<std::byte> work_space) {
        return getrs<Back,T>(ctx, MatrixView<T, MatrixFormat::Dense>(A), MatrixView<T, MatrixFormat::Dense>(Bmat), transA, pivots, work_space);
}

template <Backend Back, typename T>
size_t getrs_buffer_size(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         const MatrixView<T, MatrixFormat::Dense>& B,
                         Transpose transA);

template <Backend Back, typename T>
inline size_t getrs_buffer_size(Queue& ctx,
                                                 const Matrix<T, MatrixFormat::Dense>& A,
                                                 const Matrix<T, MatrixFormat::Dense>& Bmat,
                                                 Transpose transA) {
        return getrs_buffer_size<Back,T>(ctx, MatrixView<T, MatrixFormat::Dense>(A), MatrixView<T, MatrixFormat::Dense>(Bmat), transA);
}

}  // namespace batchlas


namespace batchlas::backend {

// The vendor path for getrs.
//
// DECLARATION ONLY -- see the note on gemm_vendor in gemm.hh. The public
// `getrs` used to be defined inside each vendor TU, so dropping a vendor library
// dropped the public entry point with it; WP0 S5 moves that definition to
// src/dispatch/entry_points/factorization.cc and leaves the vendor
// implementation here, named as such.
template <Backend Back, typename T>
Event getrs_vendor(Queue& ctx,
                   const MatrixView<T,MatrixFormat::Dense>& A,
                   const MatrixView<T,MatrixFormat::Dense>& B,
                   Transpose transA,
                   Span<int64_t> pivots,
                   Span<std::byte> work_space);


template <Backend Back, typename T>
size_t getrs_vendor_buffer_size(Queue& ctx,
                                const MatrixView<T,MatrixFormat::Dense>& A,
                                const MatrixView<T,MatrixFormat::Dense>& B,
                                Transpose transA);

}  // namespace batchlas::backend

namespace batchlas {

// Backend-deducing overloads: `f(ctx, ...)` uses ctx.backend().
// See BATCHLAS_DISPATCH_ON_QUEUE in blas/queue-dispatch.hh.

BATCHLAS_DISPATCH_ON_QUEUE(getrs)
BATCHLAS_DISPATCH_ON_QUEUE(getrs_buffer_size)

}  // namespace batchlas
