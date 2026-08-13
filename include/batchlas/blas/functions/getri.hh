#pragma once

#include <cstdint>

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
using getri = Event(Queue&,
                    const MatrixView<T, MatrixFormat::Dense>&,
                    const MatrixView<T, MatrixFormat::Dense>&,
                    Span<int64_t>, Span<std::byte>, Span<int32_t>);

template <typename T>
using getri_buffer_size = size_t(Queue&,
                                 const MatrixView<T, MatrixFormat::Dense>&);

// backend::getri_vendor's signature, spelled out from the definition rather than
// aliased to sig::getri: a vendor parameter list can differ from the public one.
template <typename T>
using getri_vendor = Event(Queue&,
                           const MatrixView<T, MatrixFormat::Dense>&,
                           const MatrixView<T, MatrixFormat::Dense>&,
                           Span<int64_t>,
                           Span<std::byte>,
                           Span<int32_t>);

// backend::getri_vendor_buffer_size's signature, spelled out from the definition rather than
// aliased to sig::getri_buffer_size: a vendor parameter list can differ from the public one.
template <typename T>
using getri_vendor_buffer_size = size_t(Queue&,
                                        const MatrixView<T, MatrixFormat::Dense>&);
}  // namespace sig


// `info` is the LAPACK per-item status: one int32 per batch item, 0 on success
// and >0 for the diagonal of U that was exactly zero, so the item has no
// inverse. The vendor call already writes it and every backend discarded it
// (see issue #73), leaving the caller to consume a matrix of infinities.
//
// An EMPTY span means "not requested" and is exactly today's behaviour: the
// backend falls back to its own scratch allocation. The workspace size is
// deliberately the same either way, so getri_buffer_size stays correct whether
// or not a caller asks for status.
template <Backend B, typename T>
Event getri(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& A,
            const MatrixView<T, MatrixFormat::Dense>& C,
            Span<int64_t> pivots,
            Span<std::byte> work_space,
            Span<int32_t> info);

// Old-arity forwarder. `info` cannot be a defaulted trailing parameter: the
// sig:: alias above is a function *type* and function types cannot carry
// default arguments (see src/util/template-instantiations.hh), so a default
// would not be part of the instantiated signature. A separate overload keeps
// every existing five-argument call site compiling unchanged.
template <Backend B, typename T>
inline Event getri(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& A,
            const MatrixView<T, MatrixFormat::Dense>& C,
            Span<int64_t> pivots,
            Span<std::byte> work_space) {
        return getri<B,T>(ctx, A, C, pivots, work_space, Span<int32_t>{});
}

template <Backend B, typename T>
inline Event getri(Queue& ctx,
                        const Matrix<T, MatrixFormat::Dense>& A,
                        const Matrix<T, MatrixFormat::Dense>& Cmat,
                        Span<int64_t> pivots,
                        Span<std::byte> work_space,
                        Span<int32_t> info) {
        return getri<B,T>(ctx, MatrixView<T, MatrixFormat::Dense>(A), MatrixView<T, MatrixFormat::Dense>(Cmat), pivots, work_space, info);
}

template <Backend B, typename T>
inline Event getri(Queue& ctx,
                        const Matrix<T, MatrixFormat::Dense>& A,
                        const Matrix<T, MatrixFormat::Dense>& Cmat,
                        Span<int64_t> pivots,
                        Span<std::byte> work_space) {
        return getri<B,T>(ctx, MatrixView<T, MatrixFormat::Dense>(A), MatrixView<T, MatrixFormat::Dense>(Cmat), pivots, work_space, Span<int32_t>{});
}

template <Backend B, typename T>
size_t getri_buffer_size(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A);

template <Backend B, typename T>
inline size_t getri_buffer_size(Queue& ctx,
                                                 const Matrix<T, MatrixFormat::Dense>& A) {
        return getri_buffer_size<B,T>(ctx, MatrixView<T, MatrixFormat::Dense>(A));
}

}  // namespace batchlas


namespace batchlas::backend {

// The vendor path for getri.
//
// DECLARATION ONLY -- see the note on gemm_vendor in gemm.hh. The public
// `getri` used to be defined inside each vendor TU, so dropping a vendor library
// dropped the public entry point with it; WP0 S5 moves that definition to
// src/dispatch/entry_points/factorization.cc and leaves the vendor
// implementation here, named as such.
template <Backend B, typename T>
Event getri_vendor(Queue& ctx,
                   const MatrixView<T, MatrixFormat::Dense>& A,
                   const MatrixView<T, MatrixFormat::Dense>& C,
                   Span<int64_t> pivots,
                   Span<std::byte> work_space,
                   Span<int32_t> info_out);


template <Backend B, typename T>
size_t getri_vendor_buffer_size(Queue& ctx,
                                const MatrixView<T, MatrixFormat::Dense>& A);

}  // namespace batchlas::backend

namespace batchlas {

// Backend-deducing overloads: `f(ctx, ...)` uses ctx.backend().
// See BATCHLAS_DISPATCH_ON_QUEUE in blas/queue-dispatch.hh.

BATCHLAS_DISPATCH_ON_QUEUE(getri)
BATCHLAS_DISPATCH_ON_QUEUE(getri_buffer_size)

}  // namespace batchlas
