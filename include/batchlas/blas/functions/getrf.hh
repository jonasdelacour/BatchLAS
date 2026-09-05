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
using getrf = Event(Queue&,
                    const MatrixView<T, MatrixFormat::Dense>&,
                    Span<int64_t>, Span<std::byte>, Span<int32_t>);

template <typename T>
using getrf_buffer_size = size_t(Queue&,
                                 const MatrixView<T, MatrixFormat::Dense>&);

// backend::getrf_vendor's signature, spelled out from the definition rather than
// aliased to sig::getrf: a vendor parameter list can differ from the public one.
template <typename T>
using getrf_vendor = Event(Queue&,
                           const MatrixView<T, MatrixFormat::Dense>&,
                           Span<int64_t>,
                           Span<std::byte>,
                           Span<int32_t>);

// backend::getrf_vendor_buffer_size's signature, spelled out from the definition rather than
// aliased to sig::getrf_buffer_size: a vendor parameter list can differ from the public one.
template <typename T>
using getrf_vendor_buffer_size = size_t(Queue&,
                                        const MatrixView<T, MatrixFormat::Dense>&);
}  // namespace sig


// `info` is the LAPACK per-item status: one int32 per batch item, 0 on success
// and >0 for the column at which U became exactly singular. Every backend
// already allocates that array for the vendor call and then throws it away, so
// a caller could not tell "the batch factorised" from "item 37 is singular and
// every getrs/getri downstream of it is noise" (see issue #73).
//
// An EMPTY span means "not requested" and is exactly today's behaviour: the
// backend falls back to its own scratch allocation. The workspace size is
// deliberately the same either way, so getrf_buffer_size stays correct whether
// or not a caller asks for status.
template <Backend B, typename T>
Event getrf(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& A,
            Span<int64_t> pivots,
            Span<std::byte> work_space,
            Span<int32_t> info);

// Old-arity forwarder. `info` cannot be a defaulted trailing parameter: the
// sig:: alias above is a function *type* and function types cannot carry
// default arguments (see src/util/template-instantiations.hh), so a default
// would not be part of the instantiated signature. A separate overload keeps
// every existing four-argument call site compiling unchanged.
template <Backend B, typename T>
inline Event getrf(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& A,
            Span<int64_t> pivots,
            Span<std::byte> work_space) {
        return getrf<B,T>(ctx, A, pivots, work_space, Span<int32_t>{});
}

template <Backend B, typename T>
inline Event getrf(Queue& ctx,
                        const Matrix<T, MatrixFormat::Dense>& A,
                        Span<int64_t> pivots,
                        Span<std::byte> work_space,
                        Span<int32_t> info) {
        return getrf<B,T>(ctx, MatrixView<T, MatrixFormat::Dense>(A), pivots, work_space, info);
}

template <Backend B, typename T>
inline Event getrf(Queue& ctx,
                        const Matrix<T, MatrixFormat::Dense>& A,
                        Span<int64_t> pivots,
                        Span<std::byte> work_space) {
        return getrf<B,T>(ctx, MatrixView<T, MatrixFormat::Dense>(A), pivots, work_space, Span<int32_t>{});
}

template <Backend B, typename T>
size_t getrf_buffer_size(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A);

template <Backend B, typename T>
inline size_t getrf_buffer_size(Queue& ctx,
                                                 const Matrix<T, MatrixFormat::Dense>& A) {
        return getrf_buffer_size<B,T>(ctx, MatrixView<T, MatrixFormat::Dense>(A));
}

}  // namespace batchlas


namespace batchlas::backend {

// The vendor path for getrf.
//
// DECLARATION ONLY -- see the note on gemm_vendor in gemm.hh. The public
// `getrf` used to be defined inside each vendor TU, so dropping a vendor library
// dropped the public entry point with it; WP0 S5 moves that definition to
// src/dispatch/entry_points/factorization.cc and leaves the vendor
// implementation here, named as such.
template <Backend B, typename T>
Event getrf_vendor(Queue& ctx,
                   const MatrixView<T, MatrixFormat::Dense>& A,
                   Span<int64_t> pivots,
                   Span<std::byte> work_space,
                   Span<int32_t> info_out);


template <Backend B, typename T>
size_t getrf_vendor_buffer_size(Queue& ctx,
                                const MatrixView<T, MatrixFormat::Dense>& A);

}  // namespace batchlas::backend

namespace batchlas {

// Backend-deducing overloads: `f(ctx, ...)` uses ctx.backend().
// See BATCHLAS_DISPATCH_ON_QUEUE in blas/queue-dispatch.hh.

BATCHLAS_DISPATCH_ON_QUEUE(getrf)
BATCHLAS_DISPATCH_ON_QUEUE(getrf_buffer_size)

}  // namespace batchlas
