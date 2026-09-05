#pragma once

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <complex>

#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/tuning_params.hh>

#include <batchlas/internal/ormqr_blocked.hh>

#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/dispatch/no_route.hh>
#include <batchlas/blas/dispatch/vendor_available.hh>
#include <batchlas/blas/dispatch/route_env.hh>
#include <batchlas/blas/dispatch/route_ormqr.hh>
#include <batchlas/blas/queue-dispatch.hh>

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


namespace batchlas::blas::dispatch::detail {

// The vendor call, gated on the vendor actually being compiled in.
//
// Without this, a build with no cuBLAS / rocSOLVER / netlib library leaves backend::ormqr_vendor<B, T> undefined and the LINK fails -- which is
// the state WP0 exists to remove. Being `if constexpr`, the vendor call is not
// compiled at all when the library is absent, so there is no symbol to satisfy.
template <Backend B, typename T, typename... Args>
Event ormqr_vendor_or_throw(Args&&... args) {
    if constexpr (!batchlas::dispatch::factorization_vendor_available<B>) {
        batchlas::dispatch::throw_no_vendor_route<T>(
            batchlas::dispatch::Op::ormqr, B, batchlas::dispatch::kFactorizationLibrary<B>);
    } else {
        return batchlas::backend::ormqr_vendor<B, T>(std::forward<Args>(args)...);
    }
}

template <Backend B, typename T, typename... Args>
size_t ormqr_vendor_buffer_size_or_throw(Args&&... args) {
    if constexpr (!batchlas::dispatch::factorization_vendor_available<B>) {
        batchlas::dispatch::throw_no_vendor_route<T>(
            batchlas::dispatch::Op::ormqr, B, batchlas::dispatch::kFactorizationLibrary<B>);
    } else {
        return batchlas::backend::ormqr_vendor_buffer_size<B, T>(std::forward<Args>(args)...);
    }
}

} // namespace batchlas::blas::dispatch::detail

namespace batchlas::blas::dispatch {

namespace detail {

// The routing inputs, in one place so the call and its buffer-size query cannot
// build different ones. `side` is not read: ormqr_supports_blocked ignored it
// too, and it is carried only so the shape describes the call faithfully.
template <typename T>
inline batchlas::dispatch::OpShape ormqr_op_shape(const Queue& ctx,
                                                  const MatrixView<T, MatrixFormat::Dense>& A,
                                                  Side side,
                                                  Transpose trans) {
    batchlas::dispatch::OpShape s;
    s.op = batchlas::dispatch::Op::ormqr;
    s.scalar = batchlas::dispatch::scalar_kind_of<T>;
    s.m = A.rows();
    s.n = A.cols();
    s.k = std::min(A.rows(), A.cols());
    s.batch = A.batch_size();
    s.side = side;
    s.transA = trans;
    s.is_gpu = ctx.device().type == DeviceType::GPU;
    return s;
}

// One resolution per call, shared by ormqr_dispatch and its buffer-size query.
//
// This replaces choose_ormqr_provider, which returned a forced provider without
// checking it against ormqr_supports_blocked -- see route_ormqr.hh for the two
// defects that followed. The unset default for ormqr is Auto, unlike GEMM's
// Vendor.
template <typename T>
inline batchlas::dispatch::Route ormqr_route(const Queue& ctx,
                                             const MatrixView<T, MatrixFormat::Dense>& A,
                                             Side side,
                                             Transpose trans) {
    namespace d = batchlas::dispatch;
    const auto parsed = d::parse_route_env(d::Op::ormqr);
    const d::Route forced = parsed.found ? parsed.route : d::legacy_unset_default(d::Op::ormqr);
    return d::resolve_ormqr_route<T>(forced, ormqr_op_shape<T>(ctx, A, side, trans));
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
    const batchlas::dispatch::Route chosen = detail::ormqr_route<T>(ctx, A, side, trans);
    const bool use_vendor = batchlas::dispatch::is_vendor(chosen);

    const int32_t block_size = detail::resolve_ormqr_block_size<T>(A, block_size_hint);

    // No third arm. The resolver returns either a vendor route or a supported
    // native one, so the old `else { chosen = Vendor; ... }` branch -- the one
    // that disagreed with ormqr_buffer_size -- has nothing left to catch.
    const size_t need_ws = use_vendor
        ? detail::ormqr_vendor_buffer_size_or_throw<B, T>(ctx, A, C, side, trans, tau)
        : ormqr_blocked_buffer_size<B, T>(ctx, A, C, side, trans, tau, block_size);

    if (workspace.size() < need_ws) {
        throw std::invalid_argument("ormqr: insufficient workspace for chosen provider");
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
    if (use_vendor) {
        e = detail::ormqr_vendor_or_throw<B, T>(*run_q, A, C, side, trans, tau, workspace);
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
    // The SAME resolution ormqr_dispatch performs, from the same pure inputs.
    // Previously these two disagreed: a forced provider that was neither Vendor
    // nor Blocked reached ormqr_dispatch's `else` arm and ran on the vendor,
    // while this function fell past its single `if` and returned the BLOCKED
    // size -- so sizing a workspace here and passing it there could throw
    // "insufficient workspace for chosen provider".
    const batchlas::dispatch::Route chosen = detail::ormqr_route<T>(ctx, A, side, trans);

    const int32_t block_size = detail::resolve_ormqr_block_size<T>(A, block_size_hint);

    if (batchlas::dispatch::is_vendor(chosen)) {
        return detail::ormqr_vendor_buffer_size_or_throw<B, T>(ctx, A, C, side, trans, tau);
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
