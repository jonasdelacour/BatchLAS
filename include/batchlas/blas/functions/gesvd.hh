#pragma once

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
#include <batchlas/blas/dispatch/route_gesvd.hh>
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
Event gesvd(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& A,
            Span<typename base_type<T>::type> singular_values,
            const MatrixView<T, MatrixFormat::Dense>& U,
            const MatrixView<T, MatrixFormat::Dense>& Vh,
            SvdVectors jobu,
            SvdVectors jobvh,
            Span<std::byte> workspace,
            Span<int32_t> info);

template <Backend B, typename T>
Event gesvd(Queue& ctx,
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
Event gesvd_vendor(Queue& ctx,
                   const MatrixView<T, MatrixFormat::Dense>& A,
                   Span<typename base_type<T>::type> singular_values,
                   const MatrixView<T, MatrixFormat::Dense>& U,
                   const MatrixView<T, MatrixFormat::Dense>& Vh,
                   SvdVectors jobu,
                   SvdVectors jobvh,
                   Span<std::byte> workspace,
                   Span<int32_t> info_out = Span<int32_t>());

template <Backend B, typename T>
size_t gesvd_vendor_buffer_size(Queue& ctx,
                                const MatrixView<T, MatrixFormat::Dense>& A,
                                Span<typename base_type<T>::type> singular_values,
                                const MatrixView<T, MatrixFormat::Dense>& U,
                                const MatrixView<T, MatrixFormat::Dense>& Vh,
                                SvdVectors jobu,
                                SvdVectors jobvh);

} // namespace batchlas::backend


namespace batchlas::blas::dispatch::detail {

// The vendor call, gated on the vendor actually being compiled in.
//
// Without this, a build with no cuSOLVER / rocSOLVER / netlib library leaves backend::gesvd_vendor<B, T> undefined and the LINK fails -- which is
// the state WP0 exists to remove. Being `if constexpr`, the vendor call is not
// compiled at all when the library is absent, so there is no symbol to satisfy.
template <Backend B, typename T, typename... Args>
Event gesvd_vendor_or_throw(Args&&... args) {
    if constexpr (!batchlas::dispatch::solver_vendor_available<B>) {
        batchlas::dispatch::throw_no_vendor_route<T>(
            batchlas::dispatch::Op::gesvd, B, batchlas::dispatch::kSolverLibrary<B>);
    } else {
        return batchlas::backend::gesvd_vendor<B, T>(std::forward<Args>(args)...);
    }
}

template <Backend B, typename T, typename... Args>
size_t gesvd_vendor_buffer_size_or_throw(Args&&... args) {
    if constexpr (!batchlas::dispatch::solver_vendor_available<B>) {
        batchlas::dispatch::throw_no_vendor_route<T>(
            batchlas::dispatch::Op::gesvd, B, batchlas::dispatch::kSolverLibrary<B>);
    } else {
        return batchlas::backend::gesvd_vendor_buffer_size<B, T>(std::forward<Args>(args)...);
    }
}

} // namespace batchlas::blas::dispatch::detail

namespace batchlas::blas::dispatch {

namespace detail {

// The routing inputs, in one place so the call and its buffer-size query cannot
// build different ones.
//
// `jobu`/`jobvh` must already be canonicalised -- both entry points do that
// first, and the old predicates re-canonicalised internally precisely because
// one that disagreed with the caller about what "Thin" means would reject
// shapes it can serve. Doing it once, before the shape exists, removes the
// possibility of disagreement rather than papering over it.
template <typename T>
inline batchlas::dispatch::GesvdShape gesvd_op_shape(const Queue& ctx,
                                                     const MatrixView<T, MatrixFormat::Dense>& A,
                                                     SvdVectors jobu,
                                                     SvdVectors jobvh,
                                                     std::optional<Uplo> hermitian_uplo) {
    batchlas::dispatch::GesvdShape s;
    s.op = batchlas::dispatch::Op::gesvd;
    s.scalar = batchlas::dispatch::scalar_kind_of<T>;
    s.m = A.rows();
    s.n = A.cols();
    s.k = std::min<int64_t>(A.rows(), A.cols());
    s.batch = A.batch_size();
    s.jobu = jobu;
    s.jobvh = jobvh;
    s.hermitian_uplo = hermitian_uplo;
    try {
        s.is_gpu = ctx.device().type == DeviceType::GPU;
    } catch (...) {
        // query_caps was best-effort and never threw; keep that contract.
    }
    try {
        s.max_sub_group =
            static_cast<int>(ctx.device().get_property(DeviceProperty::MAX_SUB_GROUP_SIZE));
    } catch (...) {
        // leave default
    }
    return s;
}

// One resolution per call, shared by gesvd_dispatch and its buffer-size query.
// Replaces choose_gesvd_provider; the wide-band Jacobi rule it carried is now
// `preferred` in route_gesvd.hh, so it can no longer make a route ineligible.
template <typename T>
inline batchlas::dispatch::Route gesvd_route(const Queue& ctx,
                                             const MatrixView<T, MatrixFormat::Dense>& A,
                                             SvdVectors jobu,
                                             SvdVectors jobvh,
                                             std::optional<Uplo> hermitian_uplo) {
    namespace d = batchlas::dispatch;
    const auto parsed = d::parse_route_env(d::Op::gesvd);
    const d::Route forced = parsed.found ? parsed.route : d::legacy_unset_default(d::Op::gesvd);
    return d::resolve_gesvd_route<T>(
        forced, gesvd_op_shape<T>(ctx, A, jobu, jobvh, hermitian_uplo));
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
                            Span<std::byte> workspace,
                            Span<int32_t> info) {
    // Canonicalise before anything else, and identically to
    // gesvd_buffer_size_dispatch below: these two independently repeat the
    // provider choice, and a divergence in what they think "Thin" means sizes
    // the workspace for a different computation than the one that runs.
    {
        const int64_t k = std::min<int64_t>(A.rows(), A.cols());
        jobu = canonical_jobu(jobu, A.rows(), k);
        jobvh = canonical_jobvh(jobvh, A.cols(), k);
    }

    namespace d = batchlas::dispatch;
    // NETLIB has no native gesvd route at all, so the resolution is skipped
    // rather than overridden after the fact.
    const d::Route chosen = (B == Backend::NETLIB)
        ? d::Route{d::Origin::Vendor, d::Algorithm::Auto}
        : detail::gesvd_route<T>(ctx, A, jobu, jobvh, hermitian_uplo);

    size_t need_ws = 0;
    if (d::is_vendor(chosen)) {
        need_ws = detail::gesvd_vendor_buffer_size_or_throw<B, T>(ctx, A, singular_values, U, Vh, jobu, jobvh);
    } else if (chosen.algo == d::Algorithm::Jacobi) {
        need_ws = gesvdj_cta_buffer_size<B, T>(ctx, A, singular_values, U, Vh, jobu, jobvh);
    } else if (chosen.algo == d::Algorithm::CTA) {
        need_ws = hermitian_uplo.has_value()
            ? gesvd_cta_buffer_size<B, T>(ctx, A, singular_values, U, Vh, jobu, jobvh, *hermitian_uplo)
            : gesvd_cta_buffer_size<B, T>(ctx, A, singular_values, U, Vh, jobu, jobvh);
    } else {
        need_ws = hermitian_uplo.has_value()
            ? gesvd_blocked_buffer_size<B, T>(ctx, A, singular_values, U, Vh, jobu, jobvh, *hermitian_uplo)
            : gesvd_blocked_buffer_size<B, T>(ctx, A, singular_values, U, Vh, jobu, jobvh);
    }

    if (workspace.size() < need_ws) {
        throw batchlas::workspace_error("gesvd: insufficient workspace for chosen provider");
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

    if (d::is_vendor(chosen)) {
        // Every arm clears `info` itself, and exactly one arm runs, so
        // gesvd_dispatch adds no clear of its own.
        return detail::gesvd_vendor_or_throw<B, T>(*run_q, A, singular_values, U, Vh, jobu, jobvh, workspace, info);
    }

    // The explicit branch is not optional: the tail of this function is an
    // unguarded `return gesvd_blocked(...)`, so a provider without its own
    // branch silently executes the blocked normal-equation path -- the exact
    // defect this kernel exists to remove -- while every label says otherwise.
    if (chosen.algo == d::Algorithm::Jacobi) {
        return gesvdj_cta<B, T>(*run_q, A, singular_values, U, Vh, jobu, jobvh, workspace,
                                GesvdjParams<T>(), info);
    }

    if (chosen.algo == d::Algorithm::CTA) {
        return hermitian_uplo.has_value()
            ? gesvd_cta<B, T>(*run_q, A, singular_values, U, Vh, jobu, jobvh, *hermitian_uplo, workspace, info)
            : gesvd_cta<B, T>(*run_q, A, singular_values, U, Vh, jobu, jobvh, workspace, info);
    }

    return hermitian_uplo.has_value()
        ? gesvd_blocked<B, T>(*run_q, A, singular_values, U, Vh, jobu, jobvh, *hermitian_uplo, workspace, info)
        : gesvd_blocked<B, T>(*run_q, A, singular_values, U, Vh, jobu, jobvh, workspace, info);
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
    // Must match gesvd_dispatch's canonicalisation exactly -- see the note there.
    {
        const int64_t k = std::min<int64_t>(A.rows(), A.cols());
        jobu = canonical_jobu(jobu, A.rows(), k);
        jobvh = canonical_jobvh(jobvh, A.cols(), k);
    }

    namespace d = batchlas::dispatch;
    // NETLIB has no native gesvd route at all, so the resolution is skipped
    // rather than overridden after the fact.
    const d::Route chosen = (B == Backend::NETLIB)
        ? d::Route{d::Origin::Vendor, d::Algorithm::Auto}
        : detail::gesvd_route<T>(ctx, A, jobu, jobvh, hermitian_uplo);

    if (d::is_vendor(chosen)) {
        return detail::gesvd_vendor_buffer_size_or_throw<B, T>(ctx, A, singular_values, U, Vh, jobu, jobvh);
    }

    if (chosen.algo == d::Algorithm::Jacobi) {
        return gesvdj_cta_buffer_size<B, T>(ctx, A, singular_values, U, Vh, jobu, jobvh);
    }

    if (chosen.algo == d::Algorithm::CTA) {
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
                   Span<std::byte> workspace,
                   Span<int32_t> info) {
    return blas::dispatch::gesvd_dispatch<B, T>(ctx, A, singular_values, U, Vh, jobu, jobvh, std::nullopt, workspace, info);
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
                   Span<std::byte> workspace,
                   Span<int32_t> info) {
    return blas::dispatch::gesvd_dispatch<B, T>(ctx, A, singular_values, U, Vh, jobu, jobvh, hermitian_uplo, workspace, info);
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

// Owning-argument and backend-deducing overloads: `f(ctx, Matrix, ...)` accepts
// owning containers where the primary takes views, and `f(ctx, ...)` uses
// ctx.backend(). See BATCHLAS_ACCEPT_OWNING and BATCHLAS_DISPATCH_ON_QUEUE in
// blas/queue-dispatch.hh.

BATCHLAS_ACCEPT_OWNING(gesvd)
BATCHLAS_ACCEPT_OWNING(gesvd_buffer_size)

BATCHLAS_DISPATCH_ON_QUEUE(gesvd)
BATCHLAS_DISPATCH_ON_QUEUE(gesvd_buffer_size)

}  // namespace batchlas
