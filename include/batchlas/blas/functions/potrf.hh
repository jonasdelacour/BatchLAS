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
using potrf = Event(Queue&,
                    const MatrixView<T, MatrixFormat::Dense>&,
                    Uplo, Span<std::byte>, Span<int32_t>);

template <typename T>
using potrf_buffer_size = size_t(Queue&,
                                 const MatrixView<T, MatrixFormat::Dense>&,
                                 Uplo);

// backend::potrf_vendor's signature, spelled out from the definition rather than
// aliased to sig::potrf: a vendor parameter list can differ from the public one.
template <typename T>
using potrf_vendor = Event(Queue&,
                           const MatrixView<T, MatrixFormat::Dense>&,
                           Uplo,
                           Span<std::byte>,
                           Span<int32_t>);

// backend::potrf_vendor_buffer_size's signature, spelled out from the definition rather than
// aliased to sig::potrf_buffer_size: a vendor parameter list can differ from the public one.
template <typename T>
using potrf_vendor_buffer_size = size_t(Queue&,
                                        const MatrixView<T,MatrixFormat::Dense>&,
                                        Uplo);
}  // namespace sig


template <Backend B, typename T>
size_t potrf_buffer_size(Queue& ctx,
                    const MatrixView<T, MatrixFormat::Dense>& A,
                    Uplo uplo);

// `info` is the LAPACK per-item status: one int32 per batch item, 0 on success
// and >0 for the leading minor at which the item stopped being positive
// definite. It used to be unreachable -- every backend allocated the array the
// vendor call needs, passed it, and dropped it -- so a caller could not tell a
// batch that factorised from one where item 37 is rank-deficient and everything
// downstream is noise (see issue #73).
//
// An EMPTY span means "not requested" and is exactly today's behaviour: the
// backend falls back to its own scratch allocation. The workspace size is
// deliberately the same either way, so potrf_buffer_size stays correct whether
// or not a caller asks for status.
template <Backend B, typename T>
Event potrf(Queue& ctx,
        const MatrixView<T, MatrixFormat::Dense>& descrA,
        Uplo uplo,
        Span<std::byte> workspace,
        Span<int32_t> info);

// Old-arity forwarder. `info` cannot be a defaulted trailing parameter: the
// sig:: aliases above are function *types* and function types cannot carry
// default arguments (see src/util/template-instantiations.hh), so a default
// would not be part of the instantiated signature. A separate overload keeps
// every existing four-argument call site compiling unchanged.
template <Backend B, typename T>
inline Event potrf(Queue& ctx,
        const MatrixView<T, MatrixFormat::Dense>& descrA,
        Uplo uplo,
        Span<std::byte> workspace) {
        return potrf<B,T>(ctx, descrA, uplo, workspace, Span<int32_t>{});
}

template <Backend B, typename T>
inline size_t potrf_buffer_size(Queue& ctx,
                                        const Matrix<T, MatrixFormat::Dense>& A,
                                        Uplo uplo) {
        return potrf_buffer_size<B,T>(ctx, MatrixView<T, MatrixFormat::Dense>(A), uplo);
}

template <Backend B, typename T>
inline Event potrf(Queue& ctx,
                const Matrix<T, MatrixFormat::Dense>& descrA,
                Uplo uplo,
                Span<std::byte> workspace,
                Span<int32_t> info) {
        return potrf<B,T>(ctx, MatrixView<T, MatrixFormat::Dense>(descrA), uplo, workspace, info);
}

template <Backend B, typename T>
inline Event potrf(Queue& ctx,
                const Matrix<T, MatrixFormat::Dense>& descrA,
                Uplo uplo,
                Span<std::byte> workspace) {
        return potrf<B,T>(ctx, MatrixView<T, MatrixFormat::Dense>(descrA), uplo, workspace, Span<int32_t>{});
}

}  // namespace batchlas


namespace batchlas::backend {

// The vendor path for potrf.
//
// DECLARATION ONLY -- see the note on gemm_vendor in gemm.hh. The public
// `potrf` used to be defined inside each vendor TU, so dropping a vendor library
// dropped the public entry point with it; WP0 S5 moves that definition to
// src/dispatch/entry_points/factorization.cc and leaves the vendor
// implementation here, named as such.
template <Backend B, typename T>
Event potrf_vendor(Queue& ctx,
                   const MatrixView<T, MatrixFormat::Dense>& descrA,
                   Uplo uplo,
                   Span<std::byte> workspace,
                   Span<int32_t> info_out);


template <Backend B, typename T>
size_t potrf_vendor_buffer_size(Queue& ctx,
                                const MatrixView<T,MatrixFormat::Dense>& A,
                                Uplo uplo);

}  // namespace batchlas::backend

namespace batchlas {

// Backend-deducing overloads: `f(ctx, ...)` uses ctx.backend().
// See BATCHLAS_DISPATCH_ON_QUEUE in blas/queue-dispatch.hh.

BATCHLAS_DISPATCH_ON_QUEUE(potrf)
BATCHLAS_DISPATCH_ON_QUEUE(potrf_buffer_size)

}  // namespace batchlas
