#pragma once

#include <batchlas/export.hh>
#include <stdexcept>
#include <optional>
#include <type_traits>
#include <vector>

#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/extensions.hh>

#include <batchlas/backend_config.h>

#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/dispatch/no_route.hh>
#include <batchlas/blas/dispatch/vendor_available.hh>
#include <batchlas/blas/dispatch/route_env.hh>
#include <batchlas/blas/queue-dispatch.hh>

namespace batchlas {

// Signature aliases for explicit instantiation; see BATCHLAS_INSTANTIATE in
// src/util/template-instantiations.hh. Keep in sync with the declarations below.
namespace sig {
template <typename T>
using gesvd_vendor = Event(Queue&,
                           const MatrixView<T, MatrixFormat::Dense>&,
                           Span<typename base_type<T>::type>,
                           const MatrixView<T, MatrixFormat::Dense>&,
                           const MatrixView<T, MatrixFormat::Dense>&,
                           SvdVectors, SvdVectors, Span<std::byte>, Span<int32_t>);

template <typename T>
using gesvd_vendor_buffer_size = size_t(Queue&,
                                        const MatrixView<T, MatrixFormat::Dense>&,
                                        Span<typename base_type<T>::type>,
                                        const MatrixView<T, MatrixFormat::Dense>&,
                                        const MatrixView<T, MatrixFormat::Dense>&,
                                        SvdVectors, SvdVectors);

// The public entry points (src/ops/gesvd/gesvd.cc). `_hermitian` adds the Uplo parameter.
template <typename T>
using gesvd = Event(Queue&, const MatrixView<T, MatrixFormat::Dense>&, Span<typename base_type<T>::type>,
                    const MatrixView<T, MatrixFormat::Dense>&, const MatrixView<T, MatrixFormat::Dense>&,
                    SvdVectors, SvdVectors, Span<std::byte>, Span<int32_t>);
template <typename T>
using gesvd_hermitian = Event(Queue&, const MatrixView<T, MatrixFormat::Dense>&,
                              Span<typename base_type<T>::type>, const MatrixView<T, MatrixFormat::Dense>&,
                              const MatrixView<T, MatrixFormat::Dense>&, SvdVectors, SvdVectors, Uplo,
                              Span<std::byte>, Span<int32_t>);
template <typename T>
using gesvd_buffer_size = size_t(Queue&, const MatrixView<T, MatrixFormat::Dense>&,
                                 Span<typename base_type<T>::type>, const MatrixView<T, MatrixFormat::Dense>&,
                                 const MatrixView<T, MatrixFormat::Dense>&, SvdVectors, SvdVectors);
template <typename T>
using gesvd_buffer_size_hermitian = size_t(Queue&, const MatrixView<T, MatrixFormat::Dense>&,
                                           Span<typename base_type<T>::type>,
                                           const MatrixView<T, MatrixFormat::Dense>&,
                                           const MatrixView<T, MatrixFormat::Dense>&, SvdVectors, SvdVectors,
                                           Uplo);
}  // namespace sig

// A is overwritten during factorization. General real-matrix support accepts
// rectangular inputs with full-vector outputs (U and V^H). Hermitian overloads
// remain square-only.
// `info` is the per-item convergence status: one int32 per batch item, 0 when the
// item converged and > 0 LAPACK-like (the number of off-diagonal elements that
// failed to converge, or 1 where the tier that ran tracks only the fact of
// failure). gesvd is one of the routines where LAPACK returns info > 0, and until
// now a non-converged item in a large batch was invisible -- the call returned,
// ctx.wait() returned, and the caller read singular values that were simply wrong.
//
// An EMPTY span means "not requested" and costs nothing: `info` is the CALLER's
// USM, written in place by whichever kernel already knows the answer, so no tier
// needs workspace for it and gesvd_buffer_size is the same either way.
template <Backend B, typename T>
BATCHLAS_API Event gesvd(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         Span<typename base_type<T>::type> singular_values,
                         const MatrixView<T, MatrixFormat::Dense>& U,
                         const MatrixView<T, MatrixFormat::Dense>& Vh,
                         SvdVectors jobu,
                         SvdVectors jobvh,
                         Span<std::byte> workspace,
                         Span<int32_t> info);

template <Backend B, typename T>
BATCHLAS_API Event gesvd(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         Span<typename base_type<T>::type> singular_values,
                         const MatrixView<T, MatrixFormat::Dense>& U,
                         const MatrixView<T, MatrixFormat::Dense>& Vh,
                         SvdVectors jobu,
                         SvdVectors jobvh,
                         Uplo hermitian_uplo,
                         Span<std::byte> workspace,
                         Span<int32_t> info);

// Old-arity forwarders, one per overload, rather than a defaulted trailing
// parameter -- the same shape as potrf.hh:110 and functions/syev.hh, and for the
// same reason: sig::gesvd_vendor below is a function *type* and cannot carry a
// default, so leaving the declarations default-free too keeps alias and
// declaration parameter-for-parameter identical. The two forwarders keep every
// existing eight- and nine-argument call site -- the GesvdOptions spellings in
// blas/options.hh among them -- compiling unchanged. Arity plus the Uplo/Span
// type difference at parameter 8 keeps all four overloads unambiguous.
template <Backend B, typename T>
inline Event gesvd(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& A,
            Span<typename base_type<T>::type> singular_values,
            const MatrixView<T, MatrixFormat::Dense>& U,
            const MatrixView<T, MatrixFormat::Dense>& Vh,
            SvdVectors jobu,
            SvdVectors jobvh,
            Span<std::byte> workspace) {
    return gesvd<B, T>(ctx, A, singular_values, U, Vh, jobu, jobvh, workspace, Span<int32_t>{});
}

template <Backend B, typename T>
inline Event gesvd(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& A,
            Span<typename base_type<T>::type> singular_values,
            const MatrixView<T, MatrixFormat::Dense>& U,
            const MatrixView<T, MatrixFormat::Dense>& Vh,
            SvdVectors jobu,
            SvdVectors jobvh,
            Uplo hermitian_uplo,
            Span<std::byte> workspace) {
    return gesvd<B, T>(ctx, A, singular_values, U, Vh, jobu, jobvh, hermitian_uplo, workspace,
                       Span<int32_t>{});
}

template <Backend B, typename T>
BATCHLAS_API size_t gesvd_buffer_size(Queue& ctx,
                                      const MatrixView<T, MatrixFormat::Dense>& A,
                                      Span<typename base_type<T>::type> singular_values,
                                      const MatrixView<T, MatrixFormat::Dense>& U,
                                      const MatrixView<T, MatrixFormat::Dense>& Vh,
                                      SvdVectors jobu,
                                      SvdVectors jobvh);

template <Backend B, typename T>
BATCHLAS_API size_t gesvd_buffer_size(Queue& ctx,
                                      const MatrixView<T, MatrixFormat::Dense>& A,
                                      Span<typename base_type<T>::type> singular_values,
                                      const MatrixView<T, MatrixFormat::Dense>& U,
                                      const MatrixView<T, MatrixFormat::Dense>& Vh,
                                      SvdVectors jobu,
                                      SvdVectors jobvh,
                                      Uplo hermitian_uplo);

} // namespace batchlas

namespace batchlas::backend {

// Vendor path for gesvd.
//
// DECLARATION ONLY. Each backend wrapper TU (cuSOLVER / rocSOLVER / LAPACKE)
// defines this primary template for its own Backend value and explicitly
// instantiates it there -- the same mechanism syev_vendor (functions/syev.hh)
// and ormqr_vendor (functions/ormqr.hh) use.
//
// It used to be *defined* here: a NETLIB LAPACKE loop plus a throw for every
// other backend. That made a CUDA definition in src/backends/cusolver.cc a
// redefinition error rather than an override, which is why there was never a
// cuSOLVER SVD binding. The LAPACKE body now lives in
// src/backends/netlib_lapack.cc.
// `info_out` is the caller's per-item status span, or empty. cuSOLVER's
// gesvdjBatched already returns an info array and this library dropped it; netlib
// captured LAPACKE's scalar info per item and threw it away in a batch-wide
// exception. Defaulted rather than forwarded, exactly as syev_vendor's is and for
// the reason spelled out there (functions/syev.hh): a default is a property of
// the declaration, not of the function type, so sig::gesvd_vendor still names the
// full nine-parameter signature that the vendor TUs instantiate.
template <Backend B, typename T>
BATCHLAS_API Event gesvd_vendor(Queue& ctx,
                                const MatrixView<T, MatrixFormat::Dense>& A,
                                Span<typename base_type<T>::type> singular_values,
                                const MatrixView<T, MatrixFormat::Dense>& U,
                                const MatrixView<T, MatrixFormat::Dense>& Vh,
                                SvdVectors jobu,
                                SvdVectors jobvh,
                                Span<std::byte> workspace,
                                Span<int32_t> info_out = Span<int32_t>());

template <Backend B, typename T>
BATCHLAS_API size_t gesvd_vendor_buffer_size(Queue& ctx,
                                             const MatrixView<T, MatrixFormat::Dense>& A,
                                             Span<typename base_type<T>::type> singular_values,
                                             const MatrixView<T, MatrixFormat::Dense>& U,
                                             const MatrixView<T, MatrixFormat::Dense>& Vh,
                                             SvdVectors jobu,
                                             SvdVectors jobvh);

} // namespace batchlas::backend


namespace batchlas {

// Owning-argument and backend-deducing overloads: `f(ctx, Matrix, ...)` accepts
// owning containers where the primary takes views, and `f(ctx, ...)` uses
// ctx.backend(). See BATCHLAS_ACCEPT_OWNING and BATCHLAS_DISPATCH_ON_QUEUE in
// blas/queue-dispatch.hh.

BATCHLAS_ACCEPT_OWNING(gesvd)
BATCHLAS_ACCEPT_OWNING(gesvd_buffer_size)

BATCHLAS_DISPATCH_ON_QUEUE(gesvd)
BATCHLAS_DISPATCH_ON_QUEUE(gesvd_buffer_size)

}  // namespace batchlas
