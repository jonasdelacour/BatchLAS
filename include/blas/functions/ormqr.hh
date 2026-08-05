#pragma once

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <complex>

#include <util/sycl-device-queue.hh>
#include <util/sycl-span.hh>
#include <blas/matrix.hh>
#include <blas/enums.hh>
#include <batchlas/tuning_params.hh>

#include <internal/ormqr_blocked.hh>

#include <blas/dispatch/context.hh>
#include <blas/dispatch/env.hh>
#include <blas/dispatch/provider.hh>
#include <blas/queue-dispatch.hh>

namespace batchlas {

// Signature aliases for explicit instantiation; see BATCHLAS_INSTANTIATE in
// src/util/template-instantiations.hh. Keep in sync with the declarations below.
namespace sig {
template <typename T>
using ormqr = Event(Queue&,
                    const MatrixView<T, MatrixFormat::Dense>&,
                    const MatrixView<T, MatrixFormat::Dense>&,
                    Side, Transpose, Span<T>, Span<std::byte>,
                    int32_t);

template <typename T>
using ormqr_buffer_size = size_t(Queue&,
                                 const MatrixView<T, MatrixFormat::Dense>&,
                                 const MatrixView<T, MatrixFormat::Dense>&,
                                 Side, Transpose, Span<T>,
                                 int32_t);

// The vendor entry points deliberately do NOT take the block-size hint: it
// selects a WY panel width in the blocked implementation and means nothing to a
// vendor kernel. So these are spelled out rather than aliased to the two above.
template <typename T>
using ormqr_vendor = Event(Queue&,
                           const MatrixView<T, MatrixFormat::Dense>&,
                           const MatrixView<T, MatrixFormat::Dense>&,
                           Side, Transpose, Span<T>, Span<std::byte>);

template <typename T>
using ormqr_vendor_buffer_size = size_t(Queue&,
                                        const MatrixView<T, MatrixFormat::Dense>&,
                                        const MatrixView<T, MatrixFormat::Dense>&,
                                        Side, Transpose, Span<T>);
}  // namespace sig


// Public API
template <Backend B, typename T>
Event ormqr(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& A,
            const MatrixView<T, MatrixFormat::Dense>& C,
            Side side,
            Transpose trans,
            Span<T> tau,
            Span<std::byte> workspace,
            int32_t block_size_hint = 0);

template <Backend B, typename T>
size_t ormqr_buffer_size(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         const MatrixView<T, MatrixFormat::Dense>& C,
                         Side side,
                         Transpose trans,
                         Span<T> tau,
                         int32_t block_size_hint = 0);

template <Backend B, typename T>
inline Event ormqr(Queue& ctx,
                   const Matrix<T, MatrixFormat::Dense>& A,
                   const Matrix<T, MatrixFormat::Dense>& Cmat,
                   Side side,
                   Transpose trans,
                   Span<T> tau,
                   Span<std::byte> workspace,
                   int32_t block_size_hint = 0) {
    return ormqr<B, T>(ctx,
                       MatrixView<T, MatrixFormat::Dense>(A),
                       MatrixView<T, MatrixFormat::Dense>(Cmat),
                       side,
                       trans,
                       tau,
                       workspace,
                       block_size_hint);
}

template <Backend B, typename T>
inline size_t ormqr_buffer_size(Queue& ctx,
                                const Matrix<T, MatrixFormat::Dense>& A,
                                const Matrix<T, MatrixFormat::Dense>& Cmat,
                                Side side,
                                Transpose trans,
                                Span<T> tau,
                                int32_t block_size_hint = 0) {
    return ormqr_buffer_size<B, T>(ctx,
                                  MatrixView<T, MatrixFormat::Dense>(A),
                                  MatrixView<T, MatrixFormat::Dense>(Cmat),
                                  side,
                                  trans,
                                  tau,
                                  block_size_hint);
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

inline Provider normalize_ormqr_vendor_like(Provider p) {
    if (p == Provider::Netlib) return Provider::Vendor;
    return p;
}

template <typename T>
inline bool ormqr_supports_blocked(const DeviceCaps& caps,
                                  Side /*side*/,
                                  Transpose trans) {
    if (!caps.is_gpu) return false;

    if constexpr (is_std_complex_v<T>) {
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

// Resolve the WY block width used by the blocked provider.
//
// `block_size_hint > 0` lets a caller that knows the *reflector count* k pick the
// width; the tuning table is keyed on A.rows() (the panel height), which for a
// tall skinny panel is the wrong dimension entirely. Clamped to k = min(rows,cols)
// so the hint can never exceed the number of reflectors, and computed from A alone
// so the buffer-size query and the call always agree.
template <typename T>
inline int32_t resolve_ormqr_block_size(const MatrixView<T, MatrixFormat::Dense>& A,
                                        int32_t block_size_hint) {
    const int32_t k = static_cast<int32_t>(std::min(A.rows(), A.cols()));
    if (block_size_hint > 0) {
        return std::max<int32_t>(1, std::min<int32_t>(block_size_hint, std::max<int32_t>(1, k)));
    }
    return batchlas::tuning::ormqr_block_size_for_n(static_cast<int32_t>(A.rows()));
}

} // namespace detail

template <Backend B, typename T>
inline Event ormqr_dispatch(Queue& ctx,
                           const MatrixView<T, MatrixFormat::Dense>& A,
                           const MatrixView<T, MatrixFormat::Dense>& C,
                           Side side,
                           Transpose trans,
                           Span<T> tau,
                           Span<std::byte> workspace,
                           int32_t block_size_hint = 0) {
    const DeviceCaps caps = query_caps(ctx);
    const DispatchPolicy policy = policy_from_env("ORMQR");
    Provider chosen = detail::choose_ormqr_provider<T>(policy, caps, side, trans);

    const int32_t block_size = detail::resolve_ormqr_block_size<T>(A, block_size_hint);

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

    // std::optional, not a plain `Queue`: the default Queue constructor is not inert, it
    // builds a real sycl::queue on Device::default_device(). A by-value declaration here
    // would pay that construction (and, on a multi-GPU box, touch device 0) on every ormqr
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
                                        Span<T> tau,
                                        int32_t block_size_hint = 0) {
    const DeviceCaps caps = query_caps(ctx);
    const DispatchPolicy policy = policy_from_env("ORMQR");
    const Provider chosen = detail::choose_ormqr_provider<T>(policy, caps, side, trans);

    const int32_t block_size = detail::resolve_ormqr_block_size<T>(A, block_size_hint);

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
                   Span<std::byte> workspace,
                   int32_t block_size_hint) {
    return blas::dispatch::ormqr_dispatch<B, T>(ctx, A, C, side, trans, tau, workspace, block_size_hint);
}

template <Backend B, typename T>
inline size_t ormqr_buffer_size(Queue& ctx,
                                const MatrixView<T, MatrixFormat::Dense>& A,
                                const MatrixView<T, MatrixFormat::Dense>& C,
                                Side side,
                                Transpose trans,
                                Span<T> tau,
                                int32_t block_size_hint) {
    return blas::dispatch::ormqr_buffer_size_dispatch<B, T>(ctx, A, C, side, trans, tau, block_size_hint);
}

} // namespace batchlas

namespace batchlas {

// Backend-deducing overloads: `f(ctx, ...)` uses ctx.backend().
// See BATCHLAS_DISPATCH_ON_QUEUE in blas/queue-dispatch.hh.

BATCHLAS_DISPATCH_ON_QUEUE(ormqr)
BATCHLAS_DISPATCH_ON_QUEUE(ormqr_buffer_size)

}  // namespace batchlas
