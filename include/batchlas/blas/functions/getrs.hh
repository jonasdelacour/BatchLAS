#pragma once

/// @file
/// @brief Batched solve with LU factors from getrf (getrs) and its workspace query.
/// @ingroup factorizations

#include <batchlas/export.hh>
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

// Vendor signatures are spelled out, not aliased: a vendor parameter list can differ.
template <typename T>
using getrs_vendor = Event(Queue&,
                           const MatrixView<T,MatrixFormat::Dense>&,
                           const MatrixView<T,MatrixFormat::Dense>&,
                           Transpose,
                           Span<int64_t>,
                           Span<std::byte>);

template <typename T>
using getrs_vendor_buffer_size = size_t(Queue&,
                                        const MatrixView<T,MatrixFormat::Dense>&,
                                        const MatrixView<T,MatrixFormat::Dense>&,
                                        Transpose);
}  // namespace sig


/// @brief Validates the arguments of the positional getrs() entry point.
///
/// Checks only non-negative extents. Squareness of A, `A.rows() == B.rows()`,
/// equal batch sizes and the pivot span's length are checked by the option
/// overloads; on this path a non-conforming pair fails every native can_run and
/// goes to the vendor.
/// @throws batchlas::invalid_argument on negative extents
/// @ingroup factorizations
// Runs before selection in src/ops/getrs/getrs.cc reads A.rows()/B.cols().
// Deliberately minimal; rejecting more would change a working call into an error.
// evidence: docs/design/vendor-independence.md#positional-validators-reject-only-what-no-route-can-serve
template <typename T>
inline void getrs_validate_params(const MatrixView<T, MatrixFormat::Dense>& A,
                                  const MatrixView<T, MatrixFormat::Dense>& B) {
    if (A.rows() < 0 || A.cols() < 0 || B.rows() < 0 || B.cols() < 0) {
        throw batchlas::invalid_argument(
            "GETRS: Matrix dimensions cannot be negative (A: rows=" +
            std::to_string(A.rows()) + ", cols=" + std::to_string(A.cols()) +
            "; B: rows=" + std::to_string(B.rows()) +
            ", cols=" + std::to_string(B.cols()) + ")");
    }
}


/// @brief Batched solve of \f$ \mathrm{op}(A) X = B \f$ using the LU factors from getrf().
///
/// A and @p pivots must be exactly what getrf() produced on the same backend
/// (see getrf() for the packed 1-based int32 pivot format). \f$ \mathrm{op}(A) \f$
/// is A, \f$ A^T \f$ or \f$ A^H \f$ for `NoTrans`, `Trans`, `ConjTrans`.
/// B is overwritten with X. No singularity check is made: a zero U(i,i)
/// (reported by getrf's `info`) produces infinities or NaNs in that item.
///
/// Asynchronous: B is readable after the returned event is waited on.
/// @tparam Back  backend; the backend-deducing overload takes it from `ctx.backend()`
/// @tparam T     scalar type (float, double, std::complex<float>, std::complex<double>)
/// @param ctx         queue the kernels are enqueued on
/// @param A           batch of n x n LU factors from getrf(); not modified
/// @param B           batch of n x nrhs right-hand sides; overwritten with X
/// @param transA      which of A, A^T, A^H to solve with
/// @param pivots      pivots from getrf(), `n * batch` entries
/// @param work_space  device-accessible scratch of at least getrs_buffer_size() bytes
/// @return event of the last enqueued kernel
/// @pre `A.rows() == A.cols() == B.rows()`, equal batch sizes, and
///      `pivots.size() >= n * batch` (checked by the option overloads only)
/// @throws batchlas::invalid_argument on negative extents
/// @throws batchlas::NoRouteError if no native kernel can run the shape and the
///         vendor library was not built in
/// @see GetrsOptions
/// @ingroup factorizations
template <Backend Back, typename T>
BATCHLAS_API Event getrs(Queue& ctx,
                        const MatrixView<T, MatrixFormat::Dense>& A,
                        const MatrixView<T, MatrixFormat::Dense>& B,
                        Transpose transA,
                        Span<int64_t> pivots,
                        Span<std::byte> work_space);

/// @brief Workspace, in bytes, that getrs() needs for these operands on this queue.
/// @ingroup factorizations
template <Backend Back, typename T>
BATCHLAS_API size_t getrs_buffer_size(Queue& ctx,
                                      const MatrixView<T, MatrixFormat::Dense>& A,
                                      const MatrixView<T, MatrixFormat::Dense>& B,
                                      Transpose transA);

}  // namespace batchlas


namespace batchlas::backend {

/// @brief Vendor arm of getrs(); called by getrs() when it selects the `vendor`
///        kernel family, not by users.
/// @ingroup dispatch
// DECLARATION ONLY: the public getrs is defined in src/ops/getrs/getrs.cc.
// evidence: docs/design/vendor-independence.md#the-entry-point-facade
template <Backend Back, typename T>
BATCHLAS_API Event getrs_vendor(Queue& ctx,
                                const MatrixView<T,MatrixFormat::Dense>& A,
                                const MatrixView<T,MatrixFormat::Dense>& B,
                                Transpose transA,
                                Span<int64_t> pivots,
                                Span<std::byte> work_space);


/// @brief Workspace query of the vendor arm of getrs().
/// @ingroup dispatch
template <Backend Back, typename T>
BATCHLAS_API size_t getrs_vendor_buffer_size(Queue& ctx,
                                             const MatrixView<T,MatrixFormat::Dense>& A,
                                             const MatrixView<T,MatrixFormat::Dense>& B,
                                             Transpose transA);

}  // namespace batchlas::backend

namespace batchlas {

// Owning-container and backend-deducing overloads; see blas/queue-dispatch.hh.
BATCHLAS_ACCEPT_OWNING(getrs)
BATCHLAS_ACCEPT_OWNING(getrs_buffer_size)

BATCHLAS_DISPATCH_ON_QUEUE(getrs)
BATCHLAS_DISPATCH_ON_QUEUE(getrs_buffer_size)

}  // namespace batchlas
