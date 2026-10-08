#pragma once

#include <batchlas/export.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/queue-dispatch.hh>

namespace batchlas {

// Signature aliases for explicit instantiation; see BATCHLAS_INSTANTIATE in
// src/util/template-instantiations.hh. Keep in sync with the declarations below.
namespace sig {
template <typename T, MatrixFormat F>
using spmm = Event(Queue&,
                   const MatrixView<T, F>&,
                   const MatrixView<T, MatrixFormat::Dense>&,
                   const MatrixView<T, MatrixFormat::Dense>&,
                   T, T, Transpose, Transpose, Span<std::byte>);

template <typename T, MatrixFormat F>
using spmm_buffer_size = size_t(Queue&,
                                const MatrixView<T, F>&,
                                const MatrixView<T, MatrixFormat::Dense>&,
                                const MatrixView<T, MatrixFormat::Dense>&,
                                T, T, Transpose, Transpose);

// backend::spmm_vendor / _vendor_buffer_size share the public signatures.
template <typename T, MatrixFormat F>
using spmm_vendor = spmm<T, F>;
template <typename T, MatrixFormat F>
using spmm_vendor_buffer_size = spmm_buffer_size<T, F>;
}  // namespace sig


/// @brief Batched sparse-times-dense matrix multiply.
///
/// For every batch item computes
/// \f[ C := \alpha \, \mathrm{op}(A) \, \mathrm{op}(B) + \beta \, C \f]
/// with `A` sparse (CSR) and `B`, `C` dense; \f$\mathrm{op}(X)\f$ is
/// \f$X\f$, \f$X^T\f$ or \f$X^H\f$ according to `transA` / `transB`. With
/// \f$\mathrm{op}(A)\f$ m x k and \f$\mathrm{op}(B)\f$ k x n, `C` is m x n.
///
/// Takes a caller-supplied workspace, sized by spmm_buffer_size with the same
/// arguments. Lease it from the queue's arena with `ctx.workspace(bytes)`. Also
/// callable with owning `Matrix` arguments and without `B` (taken from
/// `ctx.backend()`); spell a partial explicit call `spmm<Backend::CUDA>(...)`
/// and let `T` and `MFormat` deduce.
///
/// @tparam B        backend the call is compiled for; must match `ctx`'s device
/// @tparam T        scalar type: `float`, `double`, `std::complex<float>` or `std::complex<double>`
/// @tparam MFormat  storage format of `A`; only `MatrixFormat::CSR` is instantiated
/// @param ctx        queue the work is enqueued on
/// @param A          batch of sparse matrices; not modified
/// @param descrB     batch of dense k x n (NoTrans) or n x k matrices; not modified
/// @param descrC     batch of dense m x n matrices; input scaled by `beta`, overwritten with the result
/// @param alpha      scale of the product
/// @param beta       scale of the input `C`
/// @param transA     op() applied to `A`
/// @param transB     op() applied to `B`
/// @param workspace  device-accessible scratch of at least spmm_buffer_size bytes
/// @return event of the last enqueued kernel; `C` is valid once it completes
/// @pre All operands have the same batch size and conforming shapes per item.
/// @throws batchlas::dispatch::NoRouteError in a build without the vendor
///         sparse library for `B` when the native kernel does not support the call
///         (a heterogeneous batch, or negative extents).
/// @note No argument validation is done up front: a shape the native kernel
///       refuses goes to the vendor library, which reports it.
/// @note On `Backend::ROCM` a real `T` with `ConjTrans` is suspected to give wrong
///       results (known defect 4, @ref md_docs_2design_2known-defects); use `Trans`.
/// @see spmm_buffer_size, @ref md_docs_2perf_2spmm
/// @ingroup api_sparse
template <Backend B, typename T, MatrixFormat MFormat>
BATCHLAS_API Event spmm(Queue& ctx,
                 const MatrixView<T, MFormat>& A,
                 const MatrixView<T, MatrixFormat::Dense>& descrB,
                 const MatrixView<T, MatrixFormat::Dense>& descrC,
                 T alpha,
                 T beta,
                 Transpose transA,
                 Transpose transB,
                 Span<std::byte> workspace);

/// @brief Bytes of workspace spmm needs for these arguments.
///
/// Resolves the same route as spmm and returns that route's need; a natively
/// routed call needs zero bytes and is sized without touching device memory.
/// Pass exactly the arguments the spmm call will get.
/// @tparam B        backend the call is compiled for
/// @tparam T        scalar type
/// @tparam MFormat  storage format of `A`; only `MatrixFormat::CSR` is instantiated
/// @param ctx     queue the call will run on
/// @param A       the sparse operand of the spmm call
/// @param B_mat   the dense operand of the spmm call
/// @param C       the output of the spmm call
/// @param alpha   the spmm call's `alpha`
/// @param beta    the spmm call's `beta`
/// @param transA  the spmm call's `transA`
/// @param transB  the spmm call's `transB`
/// @return required workspace size in bytes (may be 0)
/// @throws batchlas::dispatch::NoRouteError under the same conditions as spmm
/// @ingroup api_sparse_lowlevel
template <Backend B, typename T, MatrixFormat MFormat>
BATCHLAS_API size_t spmm_buffer_size(Queue& ctx,
                                     const MatrixView<T, MFormat>& A,
                                     const MatrixView<T, MatrixFormat::Dense>& B_mat,
                                     const MatrixView<T, MatrixFormat::Dense>& C,
                                     T alpha,
                                     T beta,
                                     Transpose transA,
                                     Transpose transB);

}  // namespace batchlas


namespace batchlas::backend {

// Declaration only (see gemm_vendor). spmm carries a MatrixFormat parameter, so
// its instantiations are hand-written in each vendor TU.
// evidence: docs/design/vendor-independence.md#the-entry-point-facade
/// @brief Vendor-library implementation of spmm (cuSPARSE, rocSPARSE, host).
///
/// Not an entry point: batchlas::spmm calls it. Same arguments and semantics.
/// @ingroup api_dispatch
template <Backend B, typename T, MatrixFormat MFormat>
BATCHLAS_API Event spmm_vendor(Queue& ctx,
                               const MatrixView<T, MFormat>& A,
                               const MatrixView<T, MatrixFormat::Dense>& B_mat,
                               const MatrixView<T, MatrixFormat::Dense>& C,
                               T alpha,
                               T beta,
                               Transpose transA,
                               Transpose transB,
                               Span<std::byte> workspace);

/// @brief Workspace bytes spmm_vendor needs; called by batchlas::spmm_buffer_size.
/// @ingroup api_dispatch
template <Backend B, typename T, MatrixFormat MFormat>
BATCHLAS_API size_t spmm_vendor_buffer_size(Queue& ctx,
                                            const MatrixView<T, MFormat>& A,
                                            const MatrixView<T, MatrixFormat::Dense>& B_mat,
                                            const MatrixView<T, MatrixFormat::Dense>& C,
                                            T alpha,
                                            T beta,
                                            Transpose transA,
                                            Transpose transB);

}  // namespace batchlas::backend

namespace batchlas {

// Owning-argument (`f(ctx, Matrix, ...)`) and backend-deducing (`f(ctx, ...)`)
// overloads; see blas/queue-dispatch.hh.

BATCHLAS_ACCEPT_OWNING(spmm)
BATCHLAS_ACCEPT_OWNING(spmm_buffer_size)

BATCHLAS_DISPATCH_ON_QUEUE(spmm)
BATCHLAS_DISPATCH_ON_QUEUE(spmm_buffer_size)

}  // namespace batchlas
