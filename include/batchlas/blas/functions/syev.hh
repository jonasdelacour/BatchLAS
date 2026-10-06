#pragma once

#include <batchlas/export.hh>
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

// backend::syev_vendor / syev_vendor_buffer_size share these signatures.
template <typename T> using syev_vendor = syev<T>;
template <typename T> using syev_vendor_buffer_size = syev_buffer_size<T>;
}  // namespace sig


// `info` is the per-item convergence status: one int32 per batch item, 0 when the
// item converged and > 0 LAPACK-like (the number of off-diagonal elements that
// failed to converge, or 1 where the tier that ran tracks only the fact of
// failure). syev is exactly the routine where LAPACK returns info > 0, and until
// now a non-converged item at batch 16384 was invisible -- the call returned,
// ctx.wait() returned, and the caller read eigenvalues that were simply wrong for
// that item with nothing anywhere saying so.
//
// An EMPTY span means "not requested" and costs nothing: `info` is the CALLER's
// USM, written in place by whichever kernel already knows the answer, so no tier
// needs workspace for it and syev_buffer_size is the same either way.
template <Backend B, typename T>
BATCHLAS_API Event syev(Queue& ctx,
                        const MatrixView<T, MatrixFormat::Dense>& descrA, // A is overwritten with eigenvectors
                        Span<typename base_type<T>::type> eigenvalues,
                        JobType jobtype,
                        Uplo uplo,
                        Span<std::byte> workspace,
                        Span<int32_t> info);

// Old-arity forwarder rather than a defaulted trailing parameter, mirroring
// potrf.hh:110.
//
// What forces the shape is sig::syev above: it is a function *type*, and function
// types cannot carry default arguments, so `info` has to be spelled out there
// whichever way the declaration is written (src/util/template-instantiations.hh).
// Leaving the declaration default-free too keeps the two parameter-for-parameter
// identical, which is the invariant BATCHLAS_INSTANTIATE reads; this inline
// overload is then what keeps every existing six-argument call site -- the
// SyevOptions spellings in blas/options.hh among them -- compiling unchanged.
template <Backend B, typename T>
inline Event syev(Queue& ctx,
           const MatrixView<T, MatrixFormat::Dense>& descrA,
           Span<typename base_type<T>::type> eigenvalues,
           JobType jobtype,
           Uplo uplo,
           Span<std::byte> workspace) {
    return syev<B, T>(ctx, descrA, eigenvalues, jobtype, uplo, workspace, Span<int32_t>{});
}

template <Backend B, typename T>
BATCHLAS_API size_t syev_buffer_size(Queue& ctx,
                                     const MatrixView<T, MatrixFormat::Dense>& A,
                                     Span<typename base_type<T>::type> eigenvalues,
                                     JobType jobtype,
                                     Uplo uplo);

} // namespace batchlas

namespace batchlas::backend {

// Implemented by backend wrapper TUs (e.g. cuSOLVER / rocSOLVER / LAPACKE).
// `info_out` is the caller's per-item status span, or empty. Every vendor already
// allocates this array because the vendor call demands somewhere to write; before
// this it was pool scratch that nobody read.
//
// Defaulted rather than forwarded, unlike the public `syev` above. A default
// argument is a property of the declaration and not of the function type, so
// sig::syev_vendor still names the full seven-parameter signature and the
// explicit instantiations in the vendor TUs still match. That default is what
// keeps the six-argument call sites in src/extra/norm.cc, src/extra/cond.cc and
// src/extensions/syevx_lobpcg.cc compiling with no extra overload -- none of them
// is public API, so none needs a forwarder of its own.
template <Backend B, typename T>
BATCHLAS_API Event syev_vendor(Queue& ctx,
                               const MatrixView<T, MatrixFormat::Dense>& descrA,
                               Span<typename base_type<T>::type> eigenvalues,
                               JobType jobtype,
                               Uplo uplo,
                               Span<std::byte> workspace,
                               Span<int32_t> info_out = Span<int32_t>());

template <Backend B, typename T>
BATCHLAS_API size_t syev_vendor_buffer_size(Queue& ctx,
                                            const MatrixView<T, MatrixFormat::Dense>& descrA,
                                            Span<typename base_type<T>::type> eigenvalues,
                                            JobType jobtype,
                                            Uplo uplo);

} // namespace batchlas::backend

namespace batchlas::blas::dispatch::detail {

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

} // namespace batchlas::blas::dispatch::detail

namespace batchlas {

// Owning-argument and backend-deducing overloads: `f(ctx, Matrix, ...)` accepts
// owning containers where the primary takes views, and `f(ctx, ...)` uses
// ctx.backend(). See BATCHLAS_ACCEPT_OWNING and BATCHLAS_DISPATCH_ON_QUEUE in
// blas/queue-dispatch.hh.

BATCHLAS_ACCEPT_OWNING(syev)
BATCHLAS_ACCEPT_OWNING(syev_buffer_size)

BATCHLAS_DISPATCH_ON_QUEUE(syev)
BATCHLAS_DISPATCH_ON_QUEUE(syev_buffer_size)

}  // namespace batchlas
