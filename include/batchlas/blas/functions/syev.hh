// SPDX-License-Identifier: MIT
#pragma once

#include <batchlas/export.hh>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <string_view>

#include <batchlas/settings.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/blas/matrix.hh>

#include <batchlas/blas/linalg.hh>
#include <batchlas/blas/extensions.hh>

#include <batchlas/blas/queue-dispatch.hh>

namespace batchlas {

// Signature aliases for explicit instantiation (BATCHLAS_INSTANTIATE, in
// src/util/template-instantiations.hh); keep in sync with the declarations below.
namespace sig {
template <typename T>
using syev = Event(Queue&,
                   const MatrixView<T, MatrixFormat::Dense>&,
                   Span<typename base_type<T>::type>,
                   JobType, Uplo, Span<std::byte>, Span<int32_t>);

template <typename T>
using syev_buffer_size = size_t(Queue&,
                                const MatrixView<T, MatrixFormat::Dense>&,
                                Span<typename base_type<T>::type>,
                                JobType, Uplo);

template <typename T>
using syev_masked = Event(Queue&, const MatrixView<T, MatrixFormat::Dense>&, Span<typename base_type<T>::type>,
                          JobType, Uplo, Span<std::byte>, Span<int32_t>, Span<const uint8_t>);
template <typename T>
using syev_masked_buffer_size = size_t(Queue&, const MatrixView<T, MatrixFormat::Dense>&,
                                       Span<typename base_type<T>::type>, JobType, Uplo, Span<const uint8_t>);

// backend::syev_vendor / syev_vendor_buffer_size share these signatures.
template <typename T> using syev_vendor = syev<T>;
template <typename T> using syev_vendor_buffer_size = syev_buffer_size<T>;
}  // namespace sig


/**
 * @brief Eigenvalues, and optionally eigenvectors, of a batch of symmetric/Hermitian
 *        matrices (LAPACK `?syev` / `?heev`).
 *
 * Computes \f$ A = Q \Lambda Q^H \f$ for every batch item from its `uplo` triangle.
 * The kernel family is the first entry of the nearest row of the device's tuned table
 * (`tuned/syev.<device>.txt`) that can run the call: on a GPU queue a native
 * tier (the n <= 32 sub-group solvers `cta`, `cta_fused` and `jacobi`, which need
 * sub-group size 32; `blocked` or `two_stage` at any n), otherwise `vendor`, the
 * solver library. Pin a family with `BATCHLAS_SYEV_ROUTE` (e.g. `two_stage`, `vendor`).
 * Asynchronous: returns once the work is enqueued.
 *
 * @tparam B  backend (NETLIB always runs the vendor LAPACKE path)
 * @tparam T  float, double, std::complex<float> or std::complex<double>
 * @param ctx         queue the work is enqueued on
 * @param descrA      batch of n x n matrices; with `JobType::EigenVectors` overwritten by
 *                    the orthonormal eigenvectors (column j pairs with eigenvalue j),
 *                    otherwise its contents are destroyed
 * @param eigenvalues n real eigenvalues per batch item, packed, ascending
 * @param jobtype     `EigenVectors` or `NoEigenVectors`
 * @param uplo        which triangle of `descrA` holds the matrix; only it is read (the
 *                    diagonal's imaginary part is taken as zero), the other may be overwritten
 * @param workspace   at least syev_buffer_size() bytes for the same arguments
 * @param info        per-item convergence status: 0 converged, > 0 LAPACK-like (the
 *                    number of off-diagonals that failed to converge, or 1 where the tier
 *                    only tracks failure). An EMPTY span means "not requested" and costs
 *                    nothing; syev_buffer_size() is the same either way.
 * @return event of the last enqueued kernel
 * @throws batchlas::invalid_argument if `descrA` is not square
 * @throws batchlas::workspace_error if the workspace span is smaller than the chosen kernel needs
 * @throws batchlas::NoRouteError when no native kernel can run (e.g. a CPU queue) in a
 *         build without the solver library
 * @throws std::invalid_argument if `BATCHLAS_SYEV_ROUTE` names a family that is not
 *         compiled or cannot run this call (the words `native` and `vendor` instead
 *         fall back to the tuned choice with a warning)
 * @see @ref selection_tables (which family ranks first where), @ref perf_syev,
 *      @ref md_docs_2cpp-api (convergence status)
 * @ingroup api_eigen
 */
template <Backend B, typename T>
BATCHLAS_API Event syev(Queue& ctx,
                        const MatrixView<T, MatrixFormat::Dense>& descrA, // A is overwritten with eigenvectors
                        Span<typename base_type<T>::type> eigenvalues,
                        JobType jobtype,
                        Uplo uplo,
                        Span<std::byte> workspace,
                        Span<int32_t> info);

// Old-arity forwarder, not a defaulted `info`: sig::syev is a function TYPE and cannot
// carry a default, and BATCHLAS_INSTANTIATE needs alias and declaration identical.
// evidence: docs/design/vendor-independence.md#info-spans-on-syev-gesvd-and-steqr-forwarder-or-default
/**
 * @brief syev() without the convergence status (`info` empty).
 * @ingroup api_eigen
 */
template <Backend B, typename T>
inline Event syev(Queue& ctx,
           const MatrixView<T, MatrixFormat::Dense>& descrA,
           Span<typename base_type<T>::type> eigenvalues,
           JobType jobtype,
           Uplo uplo,
           Span<std::byte> workspace) {
    return syev<B, T>(ctx, descrA, eigenvalues, jobtype, uplo, workspace, Span<int32_t>{});
}

/**
 * @brief Workspace, in bytes, that syev() needs for the same arguments.
 *
 * Makes the same kernel choice as the call, so the size is for the tier that will
 * run. `info` does not affect it.
 * @ingroup api_eigen_lowlevel
 */
template <Backend B, typename T>
BATCHLAS_API size_t syev_buffer_size(Queue& ctx,
                                     const MatrixView<T, MatrixFormat::Dense>& A,
                                     Span<typename base_type<T>::type> eigenvalues,
                                     JobType jobtype,
                                     Uplo uplo);

/**
 * @brief syev() on the active items of the batch only: item b is solved where `mask[b] != 0`; an inactive
 *        item's A, eigenvalues and info are not touched (bit-identical on return).
 *
 * For a batch whose items finish at different times (an SCC loop's molecules converge at different
 * iterations): re-solve the rest without compacting the batch yourself. The active items are gathered
 * into a packed batch in the workspace (same `ld` and stride), solved by syev() on that k-item batch
 * (which selects its family for k items, pins included), and scattered back, so an inactive item costs no
 * kernel work. With every item active this is syev(); with none it is a no-op.
 * @param mask  one entry per item, nonzero = active; host-readable (USM shared or host memory), and the
 *              work that wrote it complete: it is read on the host before anything is enqueued
 * @throws batchlas::invalid_argument if @p mask holds fewer than `descrA.batch_size()` entries, if
 *         @p eigenvalues holds fewer than n * batch values, and as syev()
 * @throws batchlas::workspace_error if @p workspace is shorter than syev_masked_buffer_size()
 * @see @ref design_masked_batches
 * @ingroup api_eigen
 */
template <Backend B, typename T>
BATCHLAS_API Event syev_masked(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& descrA,
                               Span<typename base_type<T>::type> eigenvalues, JobType jobtype, Uplo uplo,
                               Span<std::byte> workspace, Span<int32_t> info, Span<const uint8_t> mask);

/**
 * @brief Workspace, in bytes, that syev_masked() needs for these operands under *every* mask: the largest
 *        need over every count of active items (each count sized as the packed call would select it).
 *
 * Query it once per batch and reuse it while items drop out. The mask's contents do not change the result;
 * a mask shorter than the batch is refused as by syev_masked(). Costs one syev_buffer_size() per item
 * count (host only).
 * @ingroup api_eigen_lowlevel
 */
template <Backend B, typename T>
BATCHLAS_API size_t syev_masked_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                                            Span<typename base_type<T>::type> eigenvalues, JobType jobtype, Uplo uplo,
                                            Span<const uint8_t> mask);

} // namespace batchlas

namespace batchlas::backend {

// Defined per backend TU (cuSOLVER / rocSOLVER / LAPACKE). `info_out` is DEFAULTED, unlike
// syev's forwarder: a default is not part of the function type, so sig::syev_vendor still
// matches and six-argument callers (norm.cc, cond.cc, syevx_lobpcg.cc) need no forwarder.
// evidence: docs/design/vendor-independence.md#info-spans-on-syev-gesvd-and-steqr-forwarder-or-default
/**
 * @brief The vendor solver's syev (cuSOLVER, rocSOLVER or a LAPACKE loop), as syev()'s
 *        `vendor` family calls it; same contract as syev(), with `info_out` as `info`.
 * @ingroup api_dispatch
 */
template <Backend B, typename T>
BATCHLAS_API Event syev_vendor(Queue& ctx,
                               const MatrixView<T, MatrixFormat::Dense>& descrA,
                               Span<typename base_type<T>::type> eigenvalues,
                               JobType jobtype,
                               Uplo uplo,
                               Span<std::byte> workspace,
                               Span<int32_t> info_out = Span<int32_t>());

/** @brief Workspace, in bytes, for backend::syev_vendor(). @ingroup api_dispatch */
template <Backend B, typename T>
BATCHLAS_API size_t syev_vendor_buffer_size(Queue& ctx,
                                            const MatrixView<T, MatrixFormat::Dense>& descrA,
                                            Span<typename base_type<T>::type> eigenvalues,
                                            JobType jobtype,
                                            Uplo uplo);

} // namespace batchlas::backend

namespace batchlas::ops::syev::detail {

// Capability introspection for the Python binding: whether the cta / blocked / two_stage
// kernel can run A on this queue's device. They ask syev's own can_run (src/ops/syev/syev.cc),
// so they cannot drift from what selection does. `uplo` is accepted and ignored: both
// large-n kernels mirror Upper into Lower.
template <typename T>
BATCHLAS_API bool syev_supports_cta(const Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A);
template <typename T>
BATCHLAS_API bool syev_supports_blocked(const Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo);
template <typename T>
BATCHLAS_API bool syev_supports_two_stage(const Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo);

} // namespace batchlas::ops::syev::detail

namespace batchlas {

// Backend-deducing overloads: `f(ctx, ...)` uses ctx.backend(). See
// BATCHLAS_DISPATCH_ON_QUEUE in blas/queue-dispatch.hh.


BATCHLAS_DISPATCH_ON_QUEUE(syev)
BATCHLAS_DISPATCH_ON_QUEUE(syev_buffer_size)
BATCHLAS_DISPATCH_ON_QUEUE(syev_masked)
BATCHLAS_DISPATCH_ON_QUEUE(syev_masked_buffer_size)

}  // namespace batchlas
