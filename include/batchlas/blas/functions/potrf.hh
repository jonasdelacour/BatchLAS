#pragma once

/// @file
/// @brief Batched Cholesky factorization (potrf) and its workspace query.
/// @ingroup api_factorizations

#include <batchlas/export.hh>
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

// Vendor signatures are spelled out, not aliased: a vendor parameter list can differ.
template <typename T>
using potrf_vendor = Event(Queue&,
                           const MatrixView<T, MatrixFormat::Dense>&,
                           Uplo,
                           Span<std::byte>,
                           Span<int32_t>);

template <typename T>
using potrf_vendor_buffer_size = size_t(Queue&,
                                        const MatrixView<T,MatrixFormat::Dense>&,
                                        Uplo);
}  // namespace sig

/// @brief Validates the arguments of the positional potrf() entry point.
///
/// Checks only what no kernel can serve: non-negative extents, a square A and a
/// valid @p uplo. It does not check the length of a non-empty `info` span.
/// Called by the public potrf() before the selection key is built.
/// @throws batchlas::invalid_argument on negative extents, a non-square A or an
///         invalid @p uplo.
/// @ingroup api_factorizations_lowlevel
// evidence: docs/design/vendor-independence.md#positional-validators-reject-only-what-no-route-can-serve
template <typename T>
inline void potrf_validate_params(const MatrixView<T, MatrixFormat::Dense>& A,
                                  Uplo uplo) {
    if (A.rows() < 0 || A.cols() < 0) {
        throw batchlas::invalid_argument(
            "POTRF: Matrix dimensions cannot be negative (rows=" +
            std::to_string(A.rows()) + ", cols=" + std::to_string(A.cols()) + ")");
    }
    if (A.rows() != A.cols()) {
        throw batchlas::invalid_argument(
            "POTRF: A must be square, got " + std::to_string(A.rows()) + "x" +
            std::to_string(A.cols()));
    }
    if (uplo != Uplo::Lower && uplo != Uplo::Upper) {
        throw batchlas::invalid_argument(
            "POTRF: Invalid uplo parameter: " +
            std::to_string(static_cast<int>(uplo)));
    }
}


/// @brief Workspace, in bytes, that potrf() needs for this shape on this queue.
///
/// The size does not depend on whether a caller passes an `info` span, so the
/// same value serves both potrf() overloads.
/// @tparam B  backend; the backend-deducing overload takes it from `ctx.backend()`
/// @tparam T  scalar type (float, double, std::complex<float>, std::complex<double>)
/// @param ctx   queue the factorization will run on (kernel selection reads its device)
/// @param A     batch of n x n matrices to be factorized
/// @param uplo  triangle that will be factorized
/// @return bytes to pass as the `workspace` span of potrf()
/// @ingroup api_factorizations_lowlevel
template <Backend B, typename T>
BATCHLAS_API size_t potrf_buffer_size(Queue& ctx,
                                 const MatrixView<T, MatrixFormat::Dense>& A,
                                 Uplo uplo);

/// @brief Batched Cholesky factorization of Hermitian positive-definite matrices.
///
/// For every batch item computes \f$ A = L L^H \f$ (`Uplo::Lower`) or
/// \f$ A = U^H U \f$ (`Uplo::Upper`) and overwrites the @p uplo triangle of A
/// with the factor. Only that triangle is read. The opposite triangle is not
/// part of the result: some vendor paths (cuSOLVER's Upper potrf) overwrite it.
///
/// The call is asynchronous: it enqueues on @p ctx and returns; A and @p info
/// are readable only after the returned event (or the queue) has been waited on.
/// @tparam B  backend; the backend-deducing overload takes it from `ctx.backend()`
/// @tparam T  scalar type (float, double, std::complex<float>, std::complex<double>)
/// @param ctx        queue the kernels are enqueued on
/// @param descrA     batch of n x n matrices (any `ld >= n`, any batch stride);
///                   overwritten with the Cholesky factor
/// @param uplo       triangle of A that holds the input and receives the factor
/// @param workspace  device-accessible scratch of at least potrf_buffer_size() bytes
/// @param info       per-item LAPACK status, one int32 per batch item: 0 on
///                   success, i > 0 if the leading minor of order i is not
///                   positive definite (that item's factor is incomplete).
///                   An empty span means "not requested".
/// @return event of the last enqueued kernel
/// @pre `descrA.rows() == descrA.cols()`
/// @pre a non-empty @p info holds at least `descrA.batch_size()` elements; a
///      shorter non-empty span is silently ignored by this overload (the
///      checked option overloads throw).
/// @throws batchlas::invalid_argument on negative extents, a non-square A or an
///         invalid @p uplo
/// @throws batchlas::NoRouteError if no native kernel can run the shape and the
///         vendor library was not built in
/// @see PotrfOptions for the option-struct spelling; its checked overloads
///      validate the info span, and its arena overloads lease the workspace.
/// @ingroup api_factorizations
// evidence: docs/design/vendor-independence.md#per-item-info-spans-for-potrf-getrf-and-getri
template <Backend B, typename T>
BATCHLAS_API Event potrf(Queue& ctx,
                     const MatrixView<T, MatrixFormat::Dense>& descrA,
                     Uplo uplo,
                     Span<std::byte> workspace,
                     Span<int32_t> info);

/// @brief potrf() without per-item status (`info` not requested).
/// @ingroup api_factorizations
// Not a defaulted `info`: the sig:: aliases are function types, which cannot
// carry default arguments, so a default would not be part of the instantiation.
template <Backend B, typename T>
inline Event potrf(Queue& ctx,
        const MatrixView<T, MatrixFormat::Dense>& descrA,
        Uplo uplo,
        Span<std::byte> workspace) {
        return potrf<B,T>(ctx, descrA, uplo, workspace, Span<int32_t>{});
}

}  // namespace batchlas


namespace batchlas::backend {

/// @brief Vendor arm of potrf(); called by the public potrf(), not by users.
/// @ingroup api_dispatch
// Declaration only: the public potrf lives in src/ops/potrf/potrf.cc.
// evidence: docs/design/vendor-independence.md#the-entry-point-facade
template <Backend B, typename T>
BATCHLAS_API Event potrf_vendor(Queue& ctx,
                                const MatrixView<T, MatrixFormat::Dense>& descrA,
                                Uplo uplo,
                                Span<std::byte> workspace,
                                Span<int32_t> info_out);


/// @brief Workspace query of the vendor arm of potrf().
/// @ingroup api_dispatch
template <Backend B, typename T>
BATCHLAS_API size_t potrf_vendor_buffer_size(Queue& ctx,
                                             const MatrixView<T,MatrixFormat::Dense>& A,
                                             Uplo uplo);

}  // namespace batchlas::backend

namespace batchlas {

// Owning-container and backend-deducing overloads; see blas/queue-dispatch.hh.
BATCHLAS_ACCEPT_OWNING(potrf)
BATCHLAS_ACCEPT_OWNING(potrf_buffer_size)

BATCHLAS_DISPATCH_ON_QUEUE(potrf)
BATCHLAS_DISPATCH_ON_QUEUE(potrf_buffer_size)

}  // namespace batchlas
