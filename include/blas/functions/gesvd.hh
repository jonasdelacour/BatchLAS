#pragma once

#include <stdexcept>
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

namespace batchlas {

// A is overwritten during factorization. Initial scope supports square matrices
// and full-vector outputs (U and V^H).
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
size_t gesvd_buffer_size(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         Span<typename base_type<T>::type> singular_values,
                         const MatrixView<T, MatrixFormat::Dense>& U,
                         const MatrixView<T, MatrixFormat::Dense>& Vh,
                         SvdVectors jobu,
                         SvdVectors jobvh);

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

} // namespace batchlas

namespace batchlas::backend {

// Vendor path for gesvd. Implementations are provided incrementally by backend TUs.
template <Backend B, typename T>
Event gesvd_vendor(Queue& ctx,
                   const MatrixView<T, MatrixFormat::Dense>& A,
                   Span<typename base_type<T>::type> singular_values,
                   const MatrixView<T, MatrixFormat::Dense>& U,
                   const MatrixView<T, MatrixFormat::Dense>& Vh,
                   SvdVectors jobu,
                   SvdVectors jobvh,
                   Span<std::byte> workspace) {
    static_cast<void>(workspace);

    if constexpr (B != Backend::NETLIB) {
        static_cast<void>(ctx);
        static_cast<void>(A);
        static_cast<void>(singular_values);
        static_cast<void>(U);
        static_cast<void>(Vh);
        static_cast<void>(jobu);
        static_cast<void>(jobvh);
        throw std::runtime_error("gesvd_vendor: backend implementation not available yet");
    } else {
#if BATCHLAS_HAS_HOST_BACKEND
        if (A.rows() != A.cols()) {
            throw std::invalid_argument("gesvd_vendor (NETLIB): square matrices only in current implementation");
        }
        if (A.batch_size() < 1 || A.rows() < 1) {
            throw std::invalid_argument("gesvd_vendor (NETLIB): invalid matrix shape or batch size");
        }

        const int n = static_cast<int>(A.rows());
        const int batch = static_cast<int>(A.batch_size());
        const std::size_t need_s = static_cast<std::size_t>(n) * static_cast<std::size_t>(batch);
        if (singular_values.size() < need_s) {
            throw std::invalid_argument("gesvd_vendor (NETLIB): singular_values span too small");
        }

        const char lapack_jobu = (jobu == SvdVectors::All) ? 'A' : 'N';
        const char lapack_jobvt = (jobvh == SvdVectors::All) ? 'A' : 'N';

        if (jobu == SvdVectors::All) {
            if (U.rows() != n || U.cols() != n || U.batch_size() != batch) {
                throw std::invalid_argument("gesvd_vendor (NETLIB): U must be (n x n) with matching batch");
            }
        }
        if (jobvh == SvdVectors::All) {
            if (Vh.rows() != n || Vh.cols() != n || Vh.batch_size() != batch) {
                throw std::invalid_argument("gesvd_vendor (NETLIB): Vh must be (n x n) with matching batch");
            }
        }

        ctx.wait();

        std::vector<typename base_type<T>::type> superb(static_cast<std::size_t>(std::max(0, n - 1)));
        auto& A_mut = const_cast<MatrixView<T, MatrixFormat::Dense>&>(A);
        for (int b = 0; b < batch; ++b) {
            auto Ab = A_mut.batch_item(b);
            auto Ub = U.batch_item(b);
            auto Vhb = Vh.batch_item(b);
            typename base_type<T>::type* sb = singular_values.data() + static_cast<std::size_t>(b) * static_cast<std::size_t>(n);

            lapack_int info = 0;
            if constexpr (std::is_same_v<T, float>) {
                info = LAPACKE_sgesvd(LAPACK_COL_MAJOR,
                                      lapack_jobu,
                                      lapack_jobvt,
                                      n,
                                      n,
                                      Ab.data_ptr(),
                                      Ab.ld(),
                                      sb,
                                      (jobu == SvdVectors::All) ? Ub.data_ptr() : nullptr,
                                      (jobu == SvdVectors::All) ? Ub.ld() : 1,
                                      (jobvh == SvdVectors::All) ? Vhb.data_ptr() : nullptr,
                                      (jobvh == SvdVectors::All) ? Vhb.ld() : 1,
                                      superb.data());
            } else if constexpr (std::is_same_v<T, double>) {
                info = LAPACKE_dgesvd(LAPACK_COL_MAJOR,
                                      lapack_jobu,
                                      lapack_jobvt,
                                      n,
                                      n,
                                      Ab.data_ptr(),
                                      Ab.ld(),
                                      sb,
                                      (jobu == SvdVectors::All) ? Ub.data_ptr() : nullptr,
                                      (jobu == SvdVectors::All) ? Ub.ld() : 1,
                                      (jobvh == SvdVectors::All) ? Vhb.data_ptr() : nullptr,
                                      (jobvh == SvdVectors::All) ? Vhb.ld() : 1,
                                      superb.data());
            } else if constexpr (std::is_same_v<T, std::complex<float>>) {
                info = LAPACKE_cgesvd(LAPACK_COL_MAJOR,
                                      lapack_jobu,
                                      lapack_jobvt,
                                      n,
                                      n,
                                      reinterpret_cast<lapack_complex_float*>(Ab.data_ptr()),
                                      Ab.ld(),
                                      sb,
                                      (jobu == SvdVectors::All) ? reinterpret_cast<lapack_complex_float*>(Ub.data_ptr()) : nullptr,
                                      (jobu == SvdVectors::All) ? Ub.ld() : 1,
                                      (jobvh == SvdVectors::All) ? reinterpret_cast<lapack_complex_float*>(Vhb.data_ptr()) : nullptr,
                                      (jobvh == SvdVectors::All) ? Vhb.ld() : 1,
                                      superb.data());
            } else if constexpr (std::is_same_v<T, std::complex<double>>) {
                info = LAPACKE_zgesvd(LAPACK_COL_MAJOR,
                                      lapack_jobu,
                                      lapack_jobvt,
                                      n,
                                      n,
                                      reinterpret_cast<lapack_complex_double*>(Ab.data_ptr()),
                                      Ab.ld(),
                                      sb,
                                      (jobu == SvdVectors::All) ? reinterpret_cast<lapack_complex_double*>(Ub.data_ptr()) : nullptr,
                                      (jobu == SvdVectors::All) ? Ub.ld() : 1,
                                      (jobvh == SvdVectors::All) ? reinterpret_cast<lapack_complex_double*>(Vhb.data_ptr()) : nullptr,
                                      (jobvh == SvdVectors::All) ? Vhb.ld() : 1,
                                      superb.data());
            } else {
                throw std::runtime_error("gesvd_vendor (NETLIB): unsupported scalar type");
            }

            if (info != 0) {
                throw std::runtime_error("gesvd_vendor (NETLIB): LAPACKE gesvd failed");
            }
        }

        return ctx.create_event_after_external_work();
#else
        static_cast<void>(ctx);
        static_cast<void>(A);
        static_cast<void>(singular_values);
        static_cast<void>(U);
        static_cast<void>(Vh);
        static_cast<void>(jobu);
        static_cast<void>(jobvh);
        throw std::runtime_error("gesvd_vendor (NETLIB): host backend not enabled");
#endif
    }
}

template <Backend B, typename T>
size_t gesvd_vendor_buffer_size(Queue& ctx,
                                const MatrixView<T, MatrixFormat::Dense>& A,
                                Span<typename base_type<T>::type> singular_values,
                                const MatrixView<T, MatrixFormat::Dense>& U,
                                const MatrixView<T, MatrixFormat::Dense>& Vh,
                                SvdVectors jobu,
                                SvdVectors jobvh) {
    static_cast<void>(ctx);
    static_cast<void>(A);
    static_cast<void>(singular_values);
    static_cast<void>(U);
    static_cast<void>(Vh);
    static_cast<void>(jobu);
    static_cast<void>(jobvh);
    return 0;
}

} // namespace batchlas::backend

namespace batchlas::blas::dispatch {

namespace detail {

inline Provider normalize_gesvd_vendor_like(Provider p) {
    if (p == Provider::Netlib) return Provider::Vendor;
    return p;
}

template <typename T>
inline bool gesvd_supports_blocked(const DeviceCaps& caps,
                                   const MatrixView<T, MatrixFormat::Dense>& A,
                                   SvdVectors jobu,
                                   SvdVectors jobvh) {
    // Current native path supports square real matrices with optional full
    // U and/or V^H backtransforms via ORMBR.
    if (!caps.is_gpu) return false;
    if (A.rows() != A.cols()) return false;
    if (A.rows() < 1 || A.batch_size() < 1) return false;
    if constexpr (!std::is_same_v<T, typename base_type<T>::type>) {
        return false;
    }
    if (jobu != SvdVectors::None && jobu != SvdVectors::All) return false;
    if (jobvh != SvdVectors::None && jobvh != SvdVectors::All) return false;
    return true;
}

template <typename T>
inline Provider choose_gesvd_provider(const DispatchPolicy& policy,
                                      const DeviceCaps& caps,
                                      const MatrixView<T, MatrixFormat::Dense>& A,
                                      SvdVectors jobu,
                                      SvdVectors jobvh) {
    Provider chosen = normalize_gesvd_vendor_like(policy.forced);
    if (chosen != Provider::Auto) {
        if (chosen == Provider::BatchLAS_Blocked && gesvd_supports_blocked(caps, A, jobu, jobvh)) {
            return chosen;
        }
        if (chosen == Provider::Vendor) return Provider::Vendor;
        chosen = Provider::Auto;
    }

    for (Provider p : policy.order) {
        p = normalize_gesvd_vendor_like(p);
        if (p == Provider::BatchLAS_Blocked && gesvd_supports_blocked(caps, A, jobu, jobvh)) {
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
                            Span<std::byte> workspace) {
    const DeviceCaps caps = query_caps(ctx);
    const DispatchPolicy policy = policy_from_env("GESVD");
    Provider chosen = detail::choose_gesvd_provider(policy, caps, A, jobu, jobvh);

    if constexpr (B == Backend::NETLIB) {
        chosen = Provider::Vendor;
    }

    size_t need_ws = 0;
    if (chosen == Provider::Vendor) {
        need_ws = backend::gesvd_vendor_buffer_size<B, T>(ctx, A, singular_values, U, Vh, jobu, jobvh);
    } else {
        need_ws = gesvd_blocked_buffer_size<B, T>(ctx, A, singular_values, U, Vh, jobu, jobvh);
    }

    if (workspace.size() < need_ws) {
        throw std::runtime_error("gesvd: insufficient workspace for chosen provider");
    }

    Queue* run_q = &ctx;
    Queue in_order_q;
    if (!ctx.in_order()) {
        in_order_q = Queue(ctx, true);
        Event dep = ctx.get_event();
        in_order_q.enqueue(dep);
        run_q = &in_order_q;
    }

    if (chosen == Provider::Vendor) {
        return backend::gesvd_vendor<B, T>(*run_q, A, singular_values, U, Vh, jobu, jobvh, workspace);
    }

    return gesvd_blocked<B, T>(*run_q, A, singular_values, U, Vh, jobu, jobvh, workspace);
}

template <Backend B, typename T>
inline size_t gesvd_buffer_size_dispatch(Queue& ctx,
                                         const MatrixView<T, MatrixFormat::Dense>& A,
                                         Span<typename base_type<T>::type> singular_values,
                                         const MatrixView<T, MatrixFormat::Dense>& U,
                                         const MatrixView<T, MatrixFormat::Dense>& Vh,
                                         SvdVectors jobu,
                                         SvdVectors jobvh) {
    const DeviceCaps caps = query_caps(ctx);
    const DispatchPolicy policy = policy_from_env("GESVD");
    Provider chosen = detail::choose_gesvd_provider(policy, caps, A, jobu, jobvh);

    if constexpr (B == Backend::NETLIB) {
        chosen = Provider::Vendor;
    }

    if (chosen == Provider::Vendor) {
        return backend::gesvd_vendor_buffer_size<B, T>(ctx, A, singular_values, U, Vh, jobu, jobvh);
    }

    return gesvd_blocked_buffer_size<B, T>(ctx, A, singular_values, U, Vh, jobu, jobvh);
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
    return blas::dispatch::gesvd_dispatch<B, T>(ctx, A, singular_values, U, Vh, jobu, jobvh, workspace);
}

template <Backend B, typename T>
inline size_t gesvd_buffer_size(Queue& ctx,
                                const MatrixView<T, MatrixFormat::Dense>& A,
                                Span<typename base_type<T>::type> singular_values,
                                const MatrixView<T, MatrixFormat::Dense>& U,
                                const MatrixView<T, MatrixFormat::Dense>& Vh,
                                SvdVectors jobu,
                                SvdVectors jobvh) {
    return blas::dispatch::gesvd_buffer_size_dispatch<B, T>(ctx, A, singular_values, U, Vh, jobu, jobvh);
}

} // namespace batchlas
