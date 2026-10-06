#pragma once

/// @file
/// @brief Batched LU factorization with partial pivoting (getrf) and its workspace query.
/// @ingroup factorizations

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
using getrf = Event(Queue&,
                    const MatrixView<T, MatrixFormat::Dense>&,
                    Span<int64_t>, Span<std::byte>, Span<int32_t>);

template <typename T>
using getrf_buffer_size = size_t(Queue&,
                                 const MatrixView<T, MatrixFormat::Dense>&);

// Vendor signatures are spelled out, not aliased: a vendor parameter list can differ.
template <typename T>
using getrf_vendor = Event(Queue&,
                           const MatrixView<T, MatrixFormat::Dense>&,
                           Span<int64_t>,
                           Span<std::byte>,
                           Span<int32_t>);

template <typename T>
using getrf_vendor_buffer_size = size_t(Queue&,
                                        const MatrixView<T, MatrixFormat::Dense>&);
}  // namespace sig


/// @brief Validates the arguments of the positional getrf() entry point.
///
/// Checks only non-negative extents. Squareness and the pivot span's length are
/// checked by the option overloads, not here: a non-square view is routed to
/// the vendor rather than rejected.
/// @throws batchlas::invalid_argument on negative extents
/// @ingroup factorizations
// Deliberately no squareness or pivots-length check; adding either is a
// user-visible behaviour change.
// evidence: docs/design/vendor-independence.md#positional-validators-reject-only-what-no-route-can-serve
template <typename T>
inline void getrf_validate_params(const MatrixView<T, MatrixFormat::Dense>& A) {
    if (A.rows() < 0 || A.cols() < 0) {
        throw batchlas::invalid_argument(
            "GETRF: Matrix dimensions cannot be negative (rows=" +
            std::to_string(A.rows()) + ", cols=" + std::to_string(A.cols()) + ")");
    }
}


/// @brief Batched LU factorization with partial pivoting.
///
/// For every batch item computes \f$ A = P L U \f$ with L unit lower triangular
/// and U upper triangular, and overwrites A with L (below the diagonal, unit
/// diagonal implied) and U (on and above it), as LAPACK `?getrf`.
///
/// **Pivot format.** @p pivots is typed `int64_t`, but on the GPU backends and
/// every native kernel it holds *packed 1-based int32* indices: item b's row
/// interchange i (row i was swapped with row `ipiv[i]`) is the int32 at index
/// `b * n + i` of the span's bytes reinterpreted as `int32_t`. The NETLIB
/// backend writes genuine 1-based int64 instead. Pass the span unchanged to
/// getrs() / getri() on the same backend; do not read or compare the entries
/// across backends. Complex pivots are chosen on `|Re| + |Im|` natively and on
/// the modulus by cuBLAS, so pivot sequences can differ between routes while
/// both factorizations are valid.
///
/// Asynchronous: A, @p pivots and @p info are readable after the returned event
/// is waited on.
/// @tparam B  backend; the backend-deducing overload takes it from `ctx.backend()`
/// @tparam T  scalar type (float, double, std::complex<float>, std::complex<double>)
/// @param ctx         queue the kernels are enqueued on
/// @param A           batch of n x n matrices; overwritten with L and U
/// @param pivots      pivot output, at least `n * batch` `int64_t` elements
/// @param work_space  device-accessible scratch of at least getrf_buffer_size() bytes
/// @param info        per-item LAPACK status, one int32 per batch item: 0 on
///                    success, i > 0 if U(i,i) is exactly zero (1-based; the
///                    factorization completed but U is singular). Empty span =
///                    not requested.
/// @return event of the last enqueued kernel
/// @pre `A.rows() == A.cols()` (checked by the option overloads only)
/// @pre `pivots.size() >= A.rows() * A.batch_size()` (checked by the option overloads only)
/// @pre a non-empty @p info holds at least `A.batch_size()` elements; a shorter
///      one is silently ignored by this overload
/// @throws batchlas::invalid_argument on negative extents
/// @throws batchlas::dispatch::NoRouteError if no native route supports the
///         shape and the vendor library was not built in
/// @note The workspace size does not depend on whether @p info is requested.
/// @ingroup factorizations
// evidence: docs/design/vendor-independence.md#per-item-info-spans-for-potrf-getrf-and-getri
template <Backend B, typename T>
BATCHLAS_API Event getrf(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         Span<int64_t> pivots,
                         Span<std::byte> work_space,
                         Span<int32_t> info);

/// @brief getrf() without per-item status (`info` not requested).
/// @ingroup factorizations
// Not a defaulted `info`: the sig:: aliases are function types (see potrf.hh).
template <Backend B, typename T>
inline Event getrf(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& A,
            Span<int64_t> pivots,
            Span<std::byte> work_space) {
        return getrf<B,T>(ctx, A, pivots, work_space, Span<int32_t>{});
}

/// @brief Workspace, in bytes, that getrf() needs for this shape on this queue.
/// @param ctx  queue the factorization will run on (routing reads its device)
/// @param A    batch of n x n matrices to be factorized
/// @return bytes to pass as the `work_space` span of getrf()
/// @ingroup factorizations
template <Backend B, typename T>
BATCHLAS_API size_t getrf_buffer_size(Queue& ctx,
                                      const MatrixView<T, MatrixFormat::Dense>& A);

}  // namespace batchlas


namespace batchlas::backend {

/// @brief Vendor arm of getrf(); called by the entry-point facade, not by users.
/// @ingroup dispatch
// Declaration only: the public getrf lives in src/dispatch/entry_points/factorization.cc.
// evidence: docs/design/vendor-independence.md#the-entry-point-facade
template <Backend B, typename T>
BATCHLAS_API Event getrf_vendor(Queue& ctx,
                                const MatrixView<T, MatrixFormat::Dense>& A,
                                Span<int64_t> pivots,
                                Span<std::byte> work_space,
                                Span<int32_t> info_out);


/// @brief Workspace query of the vendor arm of getrf().
/// @ingroup dispatch
template <Backend B, typename T>
BATCHLAS_API size_t getrf_vendor_buffer_size(Queue& ctx,
                                             const MatrixView<T, MatrixFormat::Dense>& A);

}  // namespace batchlas::backend

namespace batchlas {

// Owning-container and backend-deducing overloads; see blas/queue-dispatch.hh.
BATCHLAS_ACCEPT_OWNING(getrf)
BATCHLAS_ACCEPT_OWNING(getrf_buffer_size)

BATCHLAS_DISPATCH_ON_QUEUE(getrf)
BATCHLAS_DISPATCH_ON_QUEUE(getrf_buffer_size)

}  // namespace batchlas
