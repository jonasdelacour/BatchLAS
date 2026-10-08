#pragma once

/// @file
/// @brief Batched matrix inverse from LU factors (getri) and its workspace query.
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
using getri = Event(Queue&,
                    const MatrixView<T, MatrixFormat::Dense>&,
                    const MatrixView<T, MatrixFormat::Dense>&,
                    Span<int64_t>, Span<std::byte>, Span<int32_t>);

template <typename T>
using getri_buffer_size = size_t(Queue&,
                                 const MatrixView<T, MatrixFormat::Dense>&);

// Vendor signatures are spelled out, not aliased: a vendor parameter list can differ.
template <typename T>
using getri_vendor = Event(Queue&,
                           const MatrixView<T, MatrixFormat::Dense>&,
                           const MatrixView<T, MatrixFormat::Dense>&,
                           Span<int64_t>,
                           Span<std::byte>,
                           Span<int32_t>);

template <typename T>
using getri_vendor_buffer_size = size_t(Queue&,
                                        const MatrixView<T, MatrixFormat::Dense>&);
}  // namespace sig


/// @brief Validates A for getri_buffer_size(): non-negative extents only.
/// @throws batchlas::invalid_argument on negative extents
/// @ingroup api_factorizations
// Runs before choose() in src/ops/getri/getri.cc, because the key reads A.rows()/A.cols().
// Two arities because the query takes A alone and getri's key and can_run are
// functions of A alone; neither checks squareness, agreement of A and C, or the
// pivots length (a non-square A fails can_run(Blocked) and goes to the vendor).
// evidence: docs/design/vendor-independence.md#positional-validators-reject-only-what-no-route-can-serve
template <typename T>
inline void getri_validate_params(const MatrixView<T, MatrixFormat::Dense>& A) {
    if (A.rows() < 0 || A.cols() < 0) {
        throw batchlas::invalid_argument(
            "GETRI: Matrix dimensions cannot be negative (A: rows=" +
            std::to_string(A.rows()) + ", cols=" + std::to_string(A.cols()) + ")");
    }
}

/// @brief Validates A and C for the positional getri(): non-negative extents only.
/// @throws batchlas::invalid_argument on negative extents of A or C
/// @ingroup api_factorizations
template <typename T>
inline void getri_validate_params(const MatrixView<T, MatrixFormat::Dense>& A,
                                  const MatrixView<T, MatrixFormat::Dense>& C) {
    getri_validate_params<T>(A);
    if (C.rows() < 0 || C.cols() < 0) {
        throw batchlas::invalid_argument(
            "GETRI: Matrix dimensions cannot be negative (C: rows=" +
            std::to_string(C.rows()) + ", cols=" + std::to_string(C.cols()) + ")");
    }
}


/// @brief Batched inverse \f$ C = A^{-1} \f$ from the LU factors produced by getrf().
///
/// A and @p pivots must be exactly what getrf() produced on the same backend
/// (see getrf() for the packed 1-based int32 pivot format). The inverse is
/// written to C, out of place; A is not part of the output.
///
/// Asynchronous: C and @p info are readable after the returned event is waited on.
/// @tparam B  backend; the backend-deducing overload takes it from `ctx.backend()`
/// @tparam T  scalar type (float, double, std::complex<float>, std::complex<double>)
/// @param ctx         queue the kernels are enqueued on
/// @param A           batch of n x n LU factors from getrf()
/// @param C           batch of n x n outputs, receives \f$ A^{-1} \f$
/// @param pivots      pivots from getrf(), `n * batch` entries
/// @param work_space  device-accessible scratch of at least getri_buffer_size() bytes
/// @param info        per-item LAPACK status, one int32 per batch item: 0 on
///                    success, i > 0 if U(i,i) is exactly zero, so the item has
///                    no inverse (its C is not meaningful). Empty span = not requested.
/// @return event of the last enqueued kernel
/// @pre A and C square of the same order and batch, and
///      `pivots.size() >= n * batch` (checked by the option overloads only)
/// @throws batchlas::invalid_argument on negative extents
/// @throws batchlas::NoRouteError if no native kernel can run the shape and the
///         vendor library was not built in
/// @note The workspace size does not depend on whether @p info is requested.
/// @ingroup api_factorizations
// evidence: docs/design/vendor-independence.md#per-item-info-spans-for-potrf-getrf-and-getri
template <Backend B, typename T>
BATCHLAS_API Event getri(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         const MatrixView<T, MatrixFormat::Dense>& C,
                         Span<int64_t> pivots,
                         Span<std::byte> work_space,
                         Span<int32_t> info);

/// @brief getri() without per-item status (`info` not requested).
/// @ingroup api_factorizations
// Not a defaulted `info`: the sig:: aliases are function types (see potrf.hh).
template <Backend B, typename T>
inline Event getri(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& A,
            const MatrixView<T, MatrixFormat::Dense>& C,
            Span<int64_t> pivots,
            Span<std::byte> work_space) {
        return getri<B,T>(ctx, A, C, pivots, work_space, Span<int32_t>{});
}

/// @brief Workspace, in bytes, that getri() needs for A on this queue.
///
/// Takes A alone: the kernel choice, and therefore the size, depends only on A.
/// @ingroup api_factorizations
template <Backend B, typename T>
BATCHLAS_API size_t getri_buffer_size(Queue& ctx,
                                      const MatrixView<T, MatrixFormat::Dense>& A);

}  // namespace batchlas


namespace batchlas::backend {

/// @brief Vendor arm of getri(); called by getri() when it selects the `vendor`
///        kernel family, not by users.
/// @ingroup api_dispatch
// DECLARATION ONLY: the public getri is defined in src/ops/getri/getri.cc.
// evidence: docs/design/vendor-independence.md#the-entry-point-facade
template <Backend B, typename T>
BATCHLAS_API Event getri_vendor(Queue& ctx,
                                const MatrixView<T, MatrixFormat::Dense>& A,
                                const MatrixView<T, MatrixFormat::Dense>& C,
                                Span<int64_t> pivots,
                                Span<std::byte> work_space,
                                Span<int32_t> info_out);


/// @brief Workspace query of the vendor arm of getri().
/// @ingroup api_dispatch
template <Backend B, typename T>
BATCHLAS_API size_t getri_vendor_buffer_size(Queue& ctx,
                                             const MatrixView<T, MatrixFormat::Dense>& A);

}  // namespace batchlas::backend

namespace batchlas {

// Owning-container and backend-deducing overloads; see blas/queue-dispatch.hh.
BATCHLAS_ACCEPT_OWNING(getri)
BATCHLAS_ACCEPT_OWNING(getri_buffer_size)

BATCHLAS_DISPATCH_ON_QUEUE(getri)
BATCHLAS_DISPATCH_ON_QUEUE(getri_buffer_size)

}  // namespace batchlas
