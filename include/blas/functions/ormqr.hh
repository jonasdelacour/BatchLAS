#pragma once

#include <stdexcept>
#include <type_traits>
#include <complex>

#include <util/sycl-device-queue.hh>
#include <util/sycl-span.hh>
#include <blas/matrix.hh>
#include <blas/enums.hh>

#include <internal/ormqr_blocked.hh>

#include <blas/dispatch/context.hh>
#include <blas/dispatch/env.hh>
#include <blas/dispatch/provider.hh>

namespace batchlas {

// Public API
template <Backend B, typename T>
Event ormqr(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& A,
            const MatrixView<T, MatrixFormat::Dense>& C,
            Side side,
            Transpose trans,
            Span<T> tau,
            Span<std::byte> workspace);

template <Backend B, typename T>
size_t ormqr_buffer_size(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         const MatrixView<T, MatrixFormat::Dense>& C,
                         Side side,
                         Transpose trans,
                         Span<T> tau);

template <Backend B, typename T>
inline Event ormqr(Queue& ctx,
                   const Matrix<T, MatrixFormat::Dense>& A,
                   const Matrix<T, MatrixFormat::Dense>& Cmat,
                   Side side,
                   Transpose trans,
                   Span<T> tau,
                   Span<std::byte> workspace) {
    return ormqr<B, T>(ctx,
                       MatrixView<T, MatrixFormat::Dense>(A),
                       MatrixView<T, MatrixFormat::Dense>(Cmat),
                       side,
                       trans,
                       tau,
                       workspace);
}

template <Backend B, typename T>
inline size_t ormqr_buffer_size(Queue& ctx,
                                const Matrix<T, MatrixFormat::Dense>& A,
                                const Matrix<T, MatrixFormat::Dense>& Cmat,
                                Side side,
                                Transpose trans,
                                Span<T> tau) {
    return ormqr_buffer_size<B, T>(ctx,
                                  MatrixView<T, MatrixFormat::Dense>(A),
                                  MatrixView<T, MatrixFormat::Dense>(Cmat),
                                  side,
                                  trans,
                                  tau);
}

} // namespace batchlas

namespace batchlas::backend {

// Implemented by backend wrapper TUs (e.g. cuSOLVER / rocSOLVER / LAPACKE).
template <Backend B, typename T>
Event ormqr_vendor(Queue& ctx,
                   const MatrixView<T, MatrixFormat::Dense>& A,
                   const MatrixView<T, MatrixFormat::Dense>& C,
                   Side side,
                   Transpose trans,
                   Span<T> tau,
                   Span<std::byte> workspace);

template <Backend B, typename T>
size_t ormqr_vendor_buffer_size(Queue& ctx,
                                const MatrixView<T, MatrixFormat::Dense>& A,
                                const MatrixView<T, MatrixFormat::Dense>& C,
                                Side side,
                                Transpose trans,
                                Span<T> tau);

} // namespace batchlas::backend

namespace batchlas::blas::dispatch {

namespace detail {

template <typename U>
struct is_std_complex : std::false_type {};

template <typename U>
struct is_std_complex<std::complex<U>> : std::true_type {};

inline Provider normalize_ormqr_vendor_like(Provider p) {
    if (p == Provider::Netlib) return Provider::Vendor;
    return p;
}

template <typename T>
inline bool ormqr_supports_blocked(const DeviceCaps& caps,
                                  Side /*side*/,
                                  Transpose trans) {
    if (!caps.is_gpu) return false;

    if constexpr (is_std_complex<T>::value) {
        if (trans == Transpose::Trans) return false;
    }

    return true;
}

template <typename T>
inline Provider choose_ormqr_provider(const DispatchPolicy& policy,
                                     const DeviceCaps& caps,
                                     Side side,
                                     Transpose trans) {
    Provider chosen = normalize_ormqr_vendor_like(policy.forced);
    if (chosen != Provider::Auto) return chosen;

    for (Provider p : policy.order) {
        p = normalize_ormqr_vendor_like(p);
        if (p == Provider::BatchLAS_Blocked && ormqr_supports_blocked<T>(caps, side, trans)) return p;
        if (p == Provider::Vendor) return Provider::Vendor;
    }

    return Provider::Vendor;
}

} // namespace detail

template <Backend B, typename T>
inline Event ormqr_dispatch(Queue& ctx,
                           const MatrixView<T, MatrixFormat::Dense>& A,
                           const MatrixView<T, MatrixFormat::Dense>& C,
                           Side side,
                           Transpose trans,
                           Span<T> tau,
                           Span<std::byte> workspace) {
    const DeviceCaps caps = query_caps(ctx);
    const DispatchPolicy policy = policy_from_env("ORMQR");
    Provider chosen = detail::choose_ormqr_provider<T>(policy, caps, side, trans);

    constexpr int32_t block_size = 64;

    size_t need_ws = 0;
    if (chosen == Provider::Vendor) {
        need_ws = backend::ormqr_vendor_buffer_size<B, T>(ctx, A, C, side, trans, tau);
    } else if (chosen == Provider::BatchLAS_Blocked) {
        need_ws = ormqr_blocked_buffer_size<B, T>(ctx, A, C, side, trans, tau, block_size);
    } else {
        chosen = Provider::Vendor;
        need_ws = backend::ormqr_vendor_buffer_size<B, T>(ctx, A, C, side, trans, tau);
    }

    if (workspace.size() < need_ws) {
        throw std::runtime_error("ormqr: insufficient workspace for chosen provider");
    }

    Queue* run_q = &ctx;
    Queue in_order_q;
    if (!ctx.in_order()) {
        in_order_q = Queue(ctx, true);
        Event dep = ctx.get_event();
        in_order_q.enqueue(dep);
        run_q = &in_order_q;
    }

    Event e;
    if (chosen == Provider::Vendor) {
        e = backend::ormqr_vendor<B, T>(*run_q, A, C, side, trans, tau, workspace);
    } else {
        e = ormqr_blocked<B, T>(*run_q, A, C, side, trans, tau, workspace, block_size);
    }

    return e;
}

template <Backend B, typename T>
inline size_t ormqr_buffer_size_dispatch(Queue& ctx,
                                        const MatrixView<T, MatrixFormat::Dense>& A,
                                        const MatrixView<T, MatrixFormat::Dense>& C,
                                        Side side,
                                        Transpose trans,
                                        Span<T> tau) {
    const DeviceCaps caps = query_caps(ctx);
    const DispatchPolicy policy = policy_from_env("ORMQR");
    const Provider chosen = detail::choose_ormqr_provider<T>(policy, caps, side, trans);

    constexpr int32_t block_size = 64;

    if (chosen == Provider::Vendor) {
        return backend::ormqr_vendor_buffer_size<B, T>(ctx, A, C, side, trans, tau);
    }

    return ormqr_blocked_buffer_size<B, T>(ctx, A, C, side, trans, tau, block_size);
}

} // namespace batchlas::blas::dispatch

namespace batchlas {

template <Backend B, typename T>
inline Event ormqr(Queue& ctx,
                   const MatrixView<T, MatrixFormat::Dense>& A,
                   const MatrixView<T, MatrixFormat::Dense>& C,
                   Side side,
                   Transpose trans,
                   Span<T> tau,
                   Span<std::byte> workspace) {
    return blas::dispatch::ormqr_dispatch<B, T>(ctx, A, C, side, trans, tau, workspace);
}

template <Backend B, typename T>
inline size_t ormqr_buffer_size(Queue& ctx,
                                const MatrixView<T, MatrixFormat::Dense>& A,
                                const MatrixView<T, MatrixFormat::Dense>& C,
                                Side side,
                                Transpose trans,
                                Span<T> tau) {
    return blas::dispatch::ormqr_buffer_size_dispatch<B, T>(ctx, A, C, side, trans, tau);
}

} // namespace batchlas
