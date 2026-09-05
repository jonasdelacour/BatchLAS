#pragma once

#include <cstdint>
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

// Validation for the POSITIONAL entry point, which had none.
//
// There was no potrf_validate_params anywhere in the tree -- the analogous
// functions/trsm.hh:39 exists, potrf's did not. require_square /
// require_info_span are attached only to the OPTION overloads
// (options.hh:548-549, :557-558, :565-566); the workspace-taking <Backend B>
// overload at :539-543 -- the spelling src/extensions/ortho.cc:200 uses -- has
// neither. So a non-square view reached the backend and cuSOLVER factorised
// A.rows() x A.rows() out of it.
//
// It runs in the facade, ahead of the shape builder, because the builder reads
// A.rows()/A.cols() and must not describe a non-conforming view. Same hoist as
// trsm's (entry_points/level3.cc:167-174).
//
// SCOPE IS DELIBERATELY MINIMAL: exactly what the shape builder needs. In
// particular this does NOT check the length of a non-empty `info` span. A short
// non-empty span silently becomes pool scratch today
// (src/linalg-impl.hh:763-771, whose fallback is `>= count`), documented as by
// design; turning that into a throw is a user-visible behaviour change and
// belongs to its own change with its own test, not to a step whose gate is
// "zero behaviour change".
template <typename T>
inline void potrf_validate_params(const MatrixView<T, MatrixFormat::Dense>& A,
                                  Uplo uplo) {
    if (A.rows() < 0 || A.cols() < 0) {
        throw std::invalid_argument(
            "POTRF: Matrix dimensions cannot be negative (rows=" +
            std::to_string(A.rows()) + ", cols=" + std::to_string(A.cols()) + ")");
    }
    if (A.rows() != A.cols()) {
        throw std::invalid_argument(
            "POTRF: A must be square, got " + std::to_string(A.rows()) + "x" +
            std::to_string(A.cols()));
    }
    if (uplo != Uplo::Lower && uplo != Uplo::Upper) {
        throw std::invalid_argument(
            "POTRF: Invalid uplo parameter: " +
            std::to_string(static_cast<int>(uplo)));
    }
}


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
