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

/**
 * @brief Singular value decomposition of a batch of matrices (LAPACK `?gesvd`).
 *
 * Computes \f$ A = U \Sigma V^H \f$ for every m x n batch item, with the
 * \f$ k = \min(m, n) \f$ singular values in descending order. The route (native
 * one-sided Jacobi, CTA or blocked bidiagonal; or the vendor solver) is chosen by
 * `RouteTable<Op::gesvd, T>` and can be pinned with `BATCHLAS_GESVD_ROUTE`.
 * Asynchronous: returns once the work is enqueued.
 *
 * @param ctx             queue the work is enqueued on
 * @param A               batch of m x n matrices; overwritten (destroyed)
 * @param singular_values k real singular values per batch item, packed, descending
 * @param U               left singular vectors: m x m for `SvdVectors::All`, m x k for
 *                        `Thin`; unused for `None`
 * @param Vh              \f$ V^H \f$: n x n for `All`, k x n for `Thin`; unused for `None`
 * @param jobu            which columns of `U` to compute
 * @param jobvh           which rows of `Vh` to compute
 * @param workspace       at least gesvd_buffer_size() bytes for the same arguments
 * @param info            per-item convergence status: 0 converged, > 0 LAPACK-like. An
 *                        EMPTY span means "not requested" and costs nothing;
 *                        gesvd_buffer_size() is the same either way.
 * @return event of the last enqueued kernel
 * @throws batchlas::workspace_error if `workspace` is smaller than the chosen route needs
 * @throws batchlas::dispatch::NoRouteError when the vendor route is chosen in a build
 *         without the solver library
 * @see @ref perf_gesvd, @ref design_gesvd, @ref md_docs_2cpp-api (convergence status)
 * @ingroup svd
 */
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

/**
 * @brief gesvd() of Hermitian input: only the `hermitian_uplo` triangle of the square
 *        `A` is read. Other parameters as for the general form.
 * @ingroup svd
 */
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

// Old-arity forwarders, not a defaulted `info`, as in functions/syev.hh: the sig:: alias
// is a function TYPE and cannot carry a default. Arity plus the Uplo/Span type at
// parameter 8 keeps all four overloads unambiguous.
// evidence: docs/design/vendor-independence.md#info-spans-on-syev-gesvd-and-steqr-forwarder-or-default
/** @brief gesvd() without the convergence status (`info` empty). @ingroup svd */
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

/** @brief Hermitian gesvd() without the convergence status (`info` empty). @ingroup svd */
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

/**
 * @brief Workspace, in bytes, that gesvd() needs for the same arguments.
 *
 * Canonicalises `jobu`/`jobvh` and resolves the route exactly as the call does, so the
 * size is for the tier that will run. `info` does not affect it.
 * @ingroup svd
 */
template <Backend B, typename T>
BATCHLAS_API size_t gesvd_buffer_size(Queue& ctx,
                                      const MatrixView<T, MatrixFormat::Dense>& A,
                                      Span<typename base_type<T>::type> singular_values,
                                      const MatrixView<T, MatrixFormat::Dense>& U,
                                      const MatrixView<T, MatrixFormat::Dense>& Vh,
                                      SvdVectors jobu,
                                      SvdVectors jobvh);

/** @brief Workspace, in bytes, for the Hermitian gesvd(). @ingroup svd */
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

// DECLARATION ONLY: each backend wrapper TU (cuSOLVER / rocSOLVER / LAPACKE) defines and
// instantiates it for its own Backend. A definition here makes theirs a redefinition error.
// evidence: docs/design/gesvd.md#gesvd-design-vendor-binding-and-dispatch
// `info_out` (caller's status span, or empty) is defaulted, as in syev_vendor, so
// sig::gesvd_vendor still names the full nine-parameter signature.
/**
 * @brief The vendor solver's SVD (cuSOLVER `gesvdjBatched` or a LAPACKE loop), as
 *        gesvd()'s vendor route calls it; same contract as the general gesvd().
 * @ingroup dispatch
 */
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

/** @brief Workspace, in bytes, for backend::gesvd_vendor(). @ingroup dispatch */
template <Backend B, typename T>
BATCHLAS_API size_t gesvd_vendor_buffer_size(Queue& ctx,
                                             const MatrixView<T, MatrixFormat::Dense>& A,
                                             Span<typename base_type<T>::type> singular_values,
                                             const MatrixView<T, MatrixFormat::Dense>& U,
                                             const MatrixView<T, MatrixFormat::Dense>& Vh,
                                             SvdVectors jobu,
                                             SvdVectors jobvh);

} // namespace batchlas::backend


namespace batchlas::blas::dispatch::detail {

// `if constexpr`, not a runtime check: with the vendor absent the call is never
// compiled, so there is no undefined backend::gesvd_vendor symbol to fail the link.
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
// build different ones. `jobu`/`jobvh` must already be canonicalised (both entry
// points do it first): a predicate that disagreed with the caller about what "Thin"
// means would reject shapes it can serve.
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
        // best-effort, never throws; leave default
    }
    try {
        s.max_sub_group =
            static_cast<int>(ctx.device().get_property(DeviceProperty::MAX_SUB_GROUP_SIZE));
    } catch (...) {
        // leave default
    }
    return s;
}

// One resolution per call, shared by gesvd_dispatch and its buffer-size query. The
// wide-band Jacobi rule is in `preferred` (route_gesvd.hh), so it cannot make a route
// ineligible. evidence: docs/perf/gesvd.md#gesvd-the-wide-band-33-to-64
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
    // NETLIB has no native gesvd route; skip resolution rather than override it.
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

    // std::optional, not a plain `Queue`: the default Queue constructor builds a real
    // sycl::queue on device 0, which a by-value declaration would pay on every call.
    // It cannot be sunk into the if-block either -- run_q escapes to the calls below.
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

    // The explicit branch is not optional: the tail is an unguarded
    // `return gesvd_blocked(...)`, so a route without its own branch silently runs
    // the blocked path under another label.
    // evidence: docs/design/gesvd.md#gesvd-design-vendor-binding-and-dispatch
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
    // NETLIB has no native gesvd route; skip resolution rather than override it.
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
