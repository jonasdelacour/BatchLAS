#pragma once

#include <stdexcept>
#include <optional>
#include <type_traits>
#include <vector>

#include <util/sycl-device-queue.hh>
#include <util/sycl-span.hh>
#include <blas/matrix.hh>
#include <blas/enums.hh>
#include <blas/extensions.hh>

#include <batchlas/backend_config.h>

#if BATCHLAS_HAS_HOST_BACKEND
#include <lapacke.h>
#endif

#include <blas/dispatch/context.hh>
#include <blas/dispatch/env.hh>
#include <blas/dispatch/provider.hh>
#include <blas/queue-dispatch.hh>

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
                           SvdVectors, SvdVectors, Span<std::byte>);

template <typename T>
using gesvd_vendor_buffer_size = size_t(Queue&,
                                        const MatrixView<T, MatrixFormat::Dense>&,
                                        Span<typename base_type<T>::type>,
                                        const MatrixView<T, MatrixFormat::Dense>&,
                                        const MatrixView<T, MatrixFormat::Dense>&,
                                        SvdVectors, SvdVectors);
}  // namespace sig

// A is overwritten during factorization. General real-matrix support accepts
// rectangular inputs with full-vector outputs (U and V^H). Hermitian overloads
// remain square-only.
template <Backend B, typename T>
Event gesvd(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& A,
            Span<typename base_type<T>::type> singular_values,
            const MatrixView<T, MatrixFormat::Dense>& U,
            const MatrixView<T, MatrixFormat::Dense>& Vh,
            SvdVectors jobu,
            SvdVectors jobvh,
            Span<std::byte> workspace);

template <Backend B, typename T>
Event gesvd(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& A,
            Span<typename base_type<T>::type> singular_values,
            const MatrixView<T, MatrixFormat::Dense>& U,
            const MatrixView<T, MatrixFormat::Dense>& Vh,
            SvdVectors jobu,
            SvdVectors jobvh,
            Uplo hermitian_uplo,
            Span<std::byte> workspace);

template <Backend B, typename T>
size_t gesvd_buffer_size(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         Span<typename base_type<T>::type> singular_values,
                         const MatrixView<T, MatrixFormat::Dense>& U,
                         const MatrixView<T, MatrixFormat::Dense>& Vh,
                         SvdVectors jobu,
                         SvdVectors jobvh);

template <Backend B, typename T>
size_t gesvd_buffer_size(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         Span<typename base_type<T>::type> singular_values,
                         const MatrixView<T, MatrixFormat::Dense>& U,
                         const MatrixView<T, MatrixFormat::Dense>& Vh,
                         SvdVectors jobu,
                         SvdVectors jobvh,
                         Uplo hermitian_uplo);

template <Backend B, typename T>
inline Event gesvd(Queue& ctx,
                   const Matrix<T, MatrixFormat::Dense>& A,
                   Span<typename base_type<T>::type> singular_values,
                   const Matrix<T, MatrixFormat::Dense>& U,
                   const Matrix<T, MatrixFormat::Dense>& Vh,
                   SvdVectors jobu,
                   SvdVectors jobvh,
                   Span<std::byte> workspace) {
    return gesvd<B, T>(ctx,
                       MatrixView<T, MatrixFormat::Dense>(A),
                       singular_values,
                       MatrixView<T, MatrixFormat::Dense>(U),
                       MatrixView<T, MatrixFormat::Dense>(Vh),
                       jobu,
                       jobvh,
                       workspace);
}

template <Backend B, typename T>
inline Event gesvd(Queue& ctx,
                   const Matrix<T, MatrixFormat::Dense>& A,
                   Span<typename base_type<T>::type> singular_values,
                   const Matrix<T, MatrixFormat::Dense>& U,
                   const Matrix<T, MatrixFormat::Dense>& Vh,
                   SvdVectors jobu,
                   SvdVectors jobvh,
                   Uplo hermitian_uplo,
                   Span<std::byte> workspace) {
    return gesvd<B, T>(ctx,
                       MatrixView<T, MatrixFormat::Dense>(A),
                       singular_values,
                       MatrixView<T, MatrixFormat::Dense>(U),
                       MatrixView<T, MatrixFormat::Dense>(Vh),
                       jobu,
                       jobvh,
                       hermitian_uplo,
                       workspace);
}

template <Backend B, typename T>
inline size_t gesvd_buffer_size(Queue& ctx,
                                const Matrix<T, MatrixFormat::Dense>& A,
                                Span<typename base_type<T>::type> singular_values,
                                const Matrix<T, MatrixFormat::Dense>& U,
                                const Matrix<T, MatrixFormat::Dense>& Vh,
                                SvdVectors jobu,
                                SvdVectors jobvh) {
    return gesvd_buffer_size<B, T>(ctx,
                                   MatrixView<T, MatrixFormat::Dense>(A),
                                   singular_values,
                                   MatrixView<T, MatrixFormat::Dense>(U),
                                   MatrixView<T, MatrixFormat::Dense>(Vh),
                                   jobu,
                                   jobvh);
}

template <Backend B, typename T>
inline size_t gesvd_buffer_size(Queue& ctx,
                                const Matrix<T, MatrixFormat::Dense>& A,
                                Span<typename base_type<T>::type> singular_values,
                                const Matrix<T, MatrixFormat::Dense>& U,
                                const Matrix<T, MatrixFormat::Dense>& Vh,
                                SvdVectors jobu,
                                SvdVectors jobvh,
                                Uplo hermitian_uplo) {
    return gesvd_buffer_size<B, T>(ctx,
                                   MatrixView<T, MatrixFormat::Dense>(A),
                                   singular_values,
                                   MatrixView<T, MatrixFormat::Dense>(U),
                                   MatrixView<T, MatrixFormat::Dense>(Vh),
                                   jobu,
                                   jobvh,
                                   hermitian_uplo);
}

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
template <Backend B, typename T>
Event gesvd_vendor(Queue& ctx,
                   const MatrixView<T, MatrixFormat::Dense>& A,
                   Span<typename base_type<T>::type> singular_values,
                   const MatrixView<T, MatrixFormat::Dense>& U,
                   const MatrixView<T, MatrixFormat::Dense>& Vh,
                   SvdVectors jobu,
                   SvdVectors jobvh,
                   Span<std::byte> workspace);

template <Backend B, typename T>
size_t gesvd_vendor_buffer_size(Queue& ctx,
                                const MatrixView<T, MatrixFormat::Dense>& A,
                                Span<typename base_type<T>::type> singular_values,
                                const MatrixView<T, MatrixFormat::Dense>& U,
                                const MatrixView<T, MatrixFormat::Dense>& Vh,
                                SvdVectors jobu,
                                SvdVectors jobvh);

} // namespace batchlas::backend

namespace batchlas::blas::dispatch {

namespace detail {

inline Provider normalize_gesvd_vendor_like(Provider p) {
    if (p == Provider::Netlib) return Provider::Vendor;
    return p;
}

template <typename T>
inline bool gesvd_supports_cta(const DeviceCaps& caps,
                               const MatrixView<T, MatrixFormat::Dense>& A,
                               SvdVectors jobu,
                               SvdVectors jobvh,
                               std::optional<Uplo> hermitian_uplo = std::nullopt) {
    if (!caps.is_gpu) return false;
    if (caps.max_sub_group < 32) return false;
    if (A.rows() < 1 || A.cols() < 1 || A.batch_size() < 1) return false;
    if (std::max(A.rows(), A.cols()) > 32) return false;
    if (jobu != SvdVectors::None && jobu != SvdVectors::All) return false;
    if (jobvh != SvdVectors::None && jobvh != SvdVectors::All) return false;
    if (hermitian_uplo.has_value()) {
        if (A.rows() != A.cols()) return false;
        return *hermitian_uplo == Uplo::Lower || *hermitian_uplo == Uplo::Upper;
    }
    if constexpr (!RealScalar<T>) {
        return false;
    }
    return true;
}

template <typename T>
inline bool gesvd_supports_blocked(const DeviceCaps& caps,
                                   const MatrixView<T, MatrixFormat::Dense>& A,
                                   SvdVectors jobu,
                                   SvdVectors jobvh,
                                   std::optional<Uplo> hermitian_uplo = std::nullopt) {
    // Current native path supports real matrices with optional full U and/or
    // V^H backtransforms via ORMBR. Hermitian support remains square-only.
    if (!caps.is_gpu) return false;
    if (A.rows() < 1 || A.cols() < 1 || A.batch_size() < 1) return false;
    if (jobu != SvdVectors::None && jobu != SvdVectors::All) return false;
    if (jobvh != SvdVectors::None && jobvh != SvdVectors::All) return false;
    if (hermitian_uplo.has_value()) {
        if (A.rows() != A.cols()) return false;
        return *hermitian_uplo == Uplo::Lower;
    }
    if constexpr (!RealScalar<T>) {
        return false;
    }
    return true;
}

template <typename T>
inline Provider choose_gesvd_provider(const DispatchPolicy& policy,
                                      const DeviceCaps& caps,
                                      const MatrixView<T, MatrixFormat::Dense>& A,
                                      SvdVectors jobu,
                                      SvdVectors jobvh,
                                      std::optional<Uplo> hermitian_uplo = std::nullopt) {
    Provider chosen = normalize_gesvd_vendor_like(policy.forced);
    if (chosen != Provider::Auto) {
        if (chosen == Provider::BatchLAS_CTA && gesvd_supports_cta(caps, A, jobu, jobvh, hermitian_uplo)) {
            return chosen;
        }
        if (chosen == Provider::BatchLAS_Blocked && gesvd_supports_blocked(caps, A, jobu, jobvh, hermitian_uplo)) {
            return chosen;
        }
        if (chosen == Provider::Vendor) return Provider::Vendor;
        chosen = Provider::Auto;
    }

    for (Provider p : policy.order) {
        p = normalize_gesvd_vendor_like(p);
        if (p == Provider::BatchLAS_CTA && gesvd_supports_cta(caps, A, jobu, jobvh, hermitian_uplo)) {
            return p;
        }
        if (p == Provider::BatchLAS_Blocked && gesvd_supports_blocked(caps, A, jobu, jobvh, hermitian_uplo)) {
            return p;
        }
        if (p == Provider::Vendor) return Provider::Vendor;
    }

    return Provider::Vendor;
}

} // namespace detail

template <Backend B, typename T>
inline Event gesvd_dispatch(Queue& ctx,
                            const MatrixView<T, MatrixFormat::Dense>& A,
                            Span<typename base_type<T>::type> singular_values,
                            const MatrixView<T, MatrixFormat::Dense>& U,
                            const MatrixView<T, MatrixFormat::Dense>& Vh,
                            SvdVectors jobu,
                            SvdVectors jobvh,
                            std::optional<Uplo> hermitian_uplo,
                            Span<std::byte> workspace) {
    const DeviceCaps caps = query_caps(ctx);
    const DispatchPolicy policy = policy_from_env("GESVD");
    Provider chosen = detail::choose_gesvd_provider(policy, caps, A, jobu, jobvh, hermitian_uplo);

    if constexpr (B == Backend::NETLIB) {
        chosen = Provider::Vendor;
    }

    size_t need_ws = 0;
    if (chosen == Provider::Vendor) {
        need_ws = backend::gesvd_vendor_buffer_size<B, T>(ctx, A, singular_values, U, Vh, jobu, jobvh);
    } else if (chosen == Provider::BatchLAS_CTA) {
        need_ws = hermitian_uplo.has_value()
            ? gesvd_cta_buffer_size<B, T>(ctx, A, singular_values, U, Vh, jobu, jobvh, *hermitian_uplo)
            : gesvd_cta_buffer_size<B, T>(ctx, A, singular_values, U, Vh, jobu, jobvh);
    } else {
        need_ws = hermitian_uplo.has_value()
            ? gesvd_blocked_buffer_size<B, T>(ctx, A, singular_values, U, Vh, jobu, jobvh, *hermitian_uplo)
            : gesvd_blocked_buffer_size<B, T>(ctx, A, singular_values, U, Vh, jobu, jobvh);
    }

    if (workspace.size() < need_ws) {
        throw std::runtime_error("gesvd: insufficient workspace for chosen provider");
    }

    // std::optional, not a plain `Queue`: the default Queue constructor is not inert, it
    // builds a real sycl::queue on Device::default_device(). A by-value declaration here
    // would pay that construction (and, on a multi-GPU box, touch device 0) on every gesvd
    // call, including the common in-order path that never looks at it. It also cannot be
    // sunk into the if-block -- run_q escapes to the calls below, so the queue has to
    // outlive the branch.
    Queue* run_q = &ctx;
    std::optional<Queue> in_order_q;
    if (!ctx.in_order()) {
        in_order_q.emplace(ctx, true);
        Event dep = ctx.get_event();
        in_order_q->enqueue(dep);
        run_q = &*in_order_q;
    }

    if (chosen == Provider::Vendor) {
        return backend::gesvd_vendor<B, T>(*run_q, A, singular_values, U, Vh, jobu, jobvh, workspace);
    }

    if (chosen == Provider::BatchLAS_CTA) {
        return hermitian_uplo.has_value()
            ? gesvd_cta<B, T>(*run_q, A, singular_values, U, Vh, jobu, jobvh, *hermitian_uplo, workspace)
            : gesvd_cta<B, T>(*run_q, A, singular_values, U, Vh, jobu, jobvh, workspace);
    }

    return hermitian_uplo.has_value()
        ? gesvd_blocked<B, T>(*run_q, A, singular_values, U, Vh, jobu, jobvh, *hermitian_uplo, workspace)
        : gesvd_blocked<B, T>(*run_q, A, singular_values, U, Vh, jobu, jobvh, workspace);
}

template <Backend B, typename T>
inline size_t gesvd_buffer_size_dispatch(Queue& ctx,
                                         const MatrixView<T, MatrixFormat::Dense>& A,
                                         Span<typename base_type<T>::type> singular_values,
                                         const MatrixView<T, MatrixFormat::Dense>& U,
                                         const MatrixView<T, MatrixFormat::Dense>& Vh,
                                         SvdVectors jobu,
                                         SvdVectors jobvh,
                                         std::optional<Uplo> hermitian_uplo) {
    const DeviceCaps caps = query_caps(ctx);
    const DispatchPolicy policy = policy_from_env("GESVD");
    Provider chosen = detail::choose_gesvd_provider(policy, caps, A, jobu, jobvh, hermitian_uplo);

    if constexpr (B == Backend::NETLIB) {
        chosen = Provider::Vendor;
    }

    if (chosen == Provider::Vendor) {
        return backend::gesvd_vendor_buffer_size<B, T>(ctx, A, singular_values, U, Vh, jobu, jobvh);
    }

    if (chosen == Provider::BatchLAS_CTA) {
        return hermitian_uplo.has_value()
            ? gesvd_cta_buffer_size<B, T>(ctx, A, singular_values, U, Vh, jobu, jobvh, *hermitian_uplo)
            : gesvd_cta_buffer_size<B, T>(ctx, A, singular_values, U, Vh, jobu, jobvh);
    }

    return hermitian_uplo.has_value()
        ? gesvd_blocked_buffer_size<B, T>(ctx, A, singular_values, U, Vh, jobu, jobvh, *hermitian_uplo)
        : gesvd_blocked_buffer_size<B, T>(ctx, A, singular_values, U, Vh, jobu, jobvh);
}

} // namespace batchlas::blas::dispatch

namespace batchlas {

template <Backend B, typename T>
inline Event gesvd(Queue& ctx,
                   const MatrixView<T, MatrixFormat::Dense>& A,
                   Span<typename base_type<T>::type> singular_values,
                   const MatrixView<T, MatrixFormat::Dense>& U,
                   const MatrixView<T, MatrixFormat::Dense>& Vh,
                   SvdVectors jobu,
                   SvdVectors jobvh,
                   Span<std::byte> workspace) {
    return blas::dispatch::gesvd_dispatch<B, T>(ctx, A, singular_values, U, Vh, jobu, jobvh, std::nullopt, workspace);
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
    return blas::dispatch::gesvd_dispatch<B, T>(ctx, A, singular_values, U, Vh, jobu, jobvh, hermitian_uplo, workspace);
}

template <Backend B, typename T>
inline size_t gesvd_buffer_size(Queue& ctx,
                                const MatrixView<T, MatrixFormat::Dense>& A,
                                Span<typename base_type<T>::type> singular_values,
                                const MatrixView<T, MatrixFormat::Dense>& U,
                                const MatrixView<T, MatrixFormat::Dense>& Vh,
                                SvdVectors jobu,
                                SvdVectors jobvh) {
    return blas::dispatch::gesvd_buffer_size_dispatch<B, T>(ctx, A, singular_values, U, Vh, jobu, jobvh, std::nullopt);
}

template <Backend B, typename T>
inline size_t gesvd_buffer_size(Queue& ctx,
                                const MatrixView<T, MatrixFormat::Dense>& A,
                                Span<typename base_type<T>::type> singular_values,
                                const MatrixView<T, MatrixFormat::Dense>& U,
                                const MatrixView<T, MatrixFormat::Dense>& Vh,
                                SvdVectors jobu,
                                SvdVectors jobvh,
                                Uplo hermitian_uplo) {
    return blas::dispatch::gesvd_buffer_size_dispatch<B, T>(ctx, A, singular_values, U, Vh, jobu, jobvh, hermitian_uplo);
}

} // namespace batchlas

namespace batchlas {

// Backend-deducing overloads: `f(ctx, ...)` uses ctx.backend().
// See BATCHLAS_DISPATCH_ON_QUEUE in blas/queue-dispatch.hh.

BATCHLAS_DISPATCH_ON_QUEUE(gesvd)
BATCHLAS_DISPATCH_ON_QUEUE(gesvd_buffer_size)

}  // namespace batchlas
