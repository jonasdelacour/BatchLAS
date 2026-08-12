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

#include <batchlas/blas/dispatch/context.hh>
#include <batchlas/blas/dispatch/env.hh>
#include <batchlas/blas/dispatch/provider.hh>
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
    // Canonicalise here too, not only at the dispatch entry point: these
    // predicates are the contract, and one that disagreed with the caller about
    // what "Thin" means would reject shapes it can serve.
    {
        const int64_t k = std::min<int64_t>(A.rows(), A.cols());
        jobu = canonical_jobu(jobu, A.rows(), k);
        jobvh = canonical_jobvh(jobvh, A.cols(), k);
    }
    // A genuinely thin factor is out of reach for this route: mode CTA always
    // takes the normal-equations branch, whose patch_zero_left_vectors writes m
    // columns of U unconditionally.
    if (jobu == SvdVectors::Thin || jobvh == SvdVectors::Thin) return false;
    if (hermitian_uplo.has_value()) {
        if (A.rows() != A.cols()) return false;
        return *hermitian_uplo == Uplo::Lower || *hermitian_uplo == Uplo::Upper;
    }
    if constexpr (!RealScalar<T>) {
        return false;
    }
    return true;
}

// Largest max(m, n) gesvdj_cta accepts, per scalar type. Mirrors
// gesvdj_cta_max_dim in src/extensions/gesvdj_cta.cc.
//
// The kernel keeps P = 32 lanes above n = 32 and grows the tile capacity C to
// 64, so each lane owns two rows. The limit is local memory: per problem with
// the V tile resident, C=64 costs 37,952 B for float, 71,744 B for double and
// complex<float>, and 138,816 B for complex<double>, against a measured device
// limit of 101,376 B. Values-only drops the V tile and halves it.
template <typename T>
inline constexpr int64_t gesvd_jacobi_max_dim(bool want_vectors) {
    if constexpr (std::is_same_v<T, std::complex<double>>) {
        return want_vectors ? 32 : 64;
    } else {
        return 64;
    }
}

// gesvdj_cta supports complex GENERAL input natively, unlike the two predicates
// below, which both return false for non-real T outside the Hermitian branch.
// That is the Tier 4 coverage gap: complex general SVD on GPU used to fall
// through to Vendor and throw. Do NOT copy the RealScalar gate here.
template <typename T>
inline bool gesvd_supports_jacobi(const DeviceCaps& caps,
                                  const MatrixView<T, MatrixFormat::Dense>& A,
                                  SvdVectors jobu,
                                  SvdVectors jobvh,
                                  std::optional<Uplo> hermitian_uplo = std::nullopt) {
    if (hermitian_uplo.has_value()) return false;   // no Hermitian shortcut
    if (!caps.is_gpu) return false;
    if (caps.max_sub_group < 32) return false;
    if (A.rows() < 1 || A.cols() < 1 || A.batch_size() < 1) return false;
    const bool want_vectors = (jobu != SvdVectors::None) || (jobvh != SvdVectors::None);
    if (std::max(A.rows(), A.cols()) > gesvd_jacobi_max_dim<T>(want_vectors)) return false;
    // Every job combination is served, Thin included: one-sided Jacobi produces
    // the thin U natively -- it IS the rotated, normalised A -- and the full-U
    // columns are the extra work, manufactured by an in-kernel Gram-Schmidt
    // that a Thin request skips outright.
    return true;
}

template <typename T>
inline bool gesvd_supports_blocked(const DeviceCaps& caps,
                                   const MatrixView<T, MatrixFormat::Dense>& A,
                                   SvdVectors jobu,
                                   SvdVectors jobvh,
                                   std::optional<Uplo> hermitian_uplo = std::nullopt) {
    // Current native path supports real matrices with optional full, thin, or
    // absent U and/or V^H backtransforms via ORMBR. Hermitian support remains
    // square-only, where Thin canonicalises to All anyway.
    if (!caps.is_gpu) return false;
    if (A.rows() < 1 || A.cols() < 1 || A.batch_size() < 1) return false;
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
        if (chosen == Provider::BatchLAS_Jacobi && gesvd_supports_jacobi(caps, A, jobu, jobvh, hermitian_uplo)) {
            return chosen;
        }
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
        if (p == Provider::BatchLAS_Jacobi && gesvd_supports_jacobi(caps, A, jobu, jobvh, hermitian_uplo)) {
            // The 33..64 band is served by Jacobi only where the alternative
            // cannot serve it at all -- that is, complex GENERAL input, which
            // gesvd_supports_blocked declines, leaving Vendor and a throw.
            //
            // For REAL input in that band the blocked path is the better
            // default, and this is the one place the two disagree. Measured at
            // n=64, batch=4096, float, full vectors: blocked 4.86 us/matrix
            // against Jacobi's 7.25, and at low conditioning blocked is also
            // the more accurate of the two (kappa=1e1: 1.1e-6 vs 1.2e-5).
            //
            // Jacobi wins decisively at HIGH conditioning -- kappa=1e6, n=64:
            // singular-value relative error 6.2e-3 vs 0.526, orthogonality
            // 3.7e-5 vs 0.144, i.e. the blocked path returns no correct digits
            // and a U that is not a basis. But which regime a caller is in is
            // not knowable from the shape, and unlike the n <= 32 case (where
            // the CTA path was worse from kappa=1e2 up) blocked is genuinely
            // better below ~1e4. So the default stays blocked and the accurate
            // route is opt-in via BATCHLAS_GESVD_PROVIDER=jacobi, which is
            // checked ahead of this loop and so is unaffected by this rule.
            const bool wide_band = std::max(A.rows(), A.cols()) > 32;
            if constexpr (RealScalar<T>) {
                if (!wide_band) return p;
            } else {
                return p;
            }
        }
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
    // Canonicalise before anything else, and identically to
    // gesvd_buffer_size_dispatch below: these two independently repeat the
    // provider choice, and a divergence in what they think "Thin" means sizes
    // the workspace for a different computation than the one that runs.
    {
        const int64_t k = std::min<int64_t>(A.rows(), A.cols());
        jobu = canonical_jobu(jobu, A.rows(), k);
        jobvh = canonical_jobvh(jobvh, A.cols(), k);
    }

    const DeviceCaps caps = query_caps(ctx);
    const DispatchPolicy policy = policy_from_env("GESVD");
    Provider chosen = detail::choose_gesvd_provider(policy, caps, A, jobu, jobvh, hermitian_uplo);

    if constexpr (B == Backend::NETLIB) {
        chosen = Provider::Vendor;
    }

    size_t need_ws = 0;
    if (chosen == Provider::Vendor) {
        need_ws = backend::gesvd_vendor_buffer_size<B, T>(ctx, A, singular_values, U, Vh, jobu, jobvh);
    } else if (chosen == Provider::BatchLAS_Jacobi) {
        need_ws = gesvdj_cta_buffer_size<B, T>(ctx, A, singular_values, U, Vh, jobu, jobvh);
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

    // The explicit branch is not optional: the tail of this function is an
    // unguarded `return gesvd_blocked(...)`, so a provider without its own
    // branch silently executes the blocked normal-equation path -- the exact
    // defect this kernel exists to remove -- while every label says otherwise.
    if (chosen == Provider::BatchLAS_Jacobi) {
        return gesvdj_cta<B, T>(*run_q, A, singular_values, U, Vh, jobu, jobvh, workspace);
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
    // Must match gesvd_dispatch's canonicalisation exactly -- see the note there.
    {
        const int64_t k = std::min<int64_t>(A.rows(), A.cols());
        jobu = canonical_jobu(jobu, A.rows(), k);
        jobvh = canonical_jobvh(jobvh, A.cols(), k);
    }

    const DeviceCaps caps = query_caps(ctx);
    const DispatchPolicy policy = policy_from_env("GESVD");
    Provider chosen = detail::choose_gesvd_provider(policy, caps, A, jobu, jobvh, hermitian_uplo);

    if constexpr (B == Backend::NETLIB) {
        chosen = Provider::Vendor;
    }

    if (chosen == Provider::Vendor) {
        return backend::gesvd_vendor_buffer_size<B, T>(ctx, A, singular_values, U, Vh, jobu, jobvh);
    }

    if (chosen == Provider::BatchLAS_Jacobi) {
        return gesvdj_cta_buffer_size<B, T>(ctx, A, singular_values, U, Vh, jobu, jobvh);
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
