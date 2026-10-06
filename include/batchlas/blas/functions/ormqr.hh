#pragma once

/// @file
/// @brief Batched application of Q from geqrf reflectors (ormqr/unmqr) and its workspace query.
/// @ingroup qr

#include <batchlas/export.hh>
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

// The vendor signatures deliberately drop the block-size hint (a WY panel width
// means nothing to a vendor kernel), so they are spelled out, not aliased.
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


/// @brief Batched multiplication by the orthogonal/unitary Q from geqrf().
///
/// As LAPACK `?ormqr` / `?unmqr`. A holds, below its diagonal, the
/// \f$ k = \min(A_{rows}, A_{cols}) \f$ reflectors that geqrf() wrote, with
/// \f$ Q = H_1 \cdots H_k \f$ of order `A.rows()`. For every batch item C is
/// overwritten with
/// \f$ \mathrm{op}(Q)\, C \f$ (`Side::Left`) or \f$ C\, \mathrm{op}(Q) \f$
/// (`Side::Right`), where \f$ \mathrm{op}(Q) \f$ is Q for `NoTrans` and
/// \f$ Q^H \f$ (\f$ Q^T \f$ for real T) otherwise. Q is never formed.
///
/// Asynchronous: C is readable after the returned event is waited on. On an
/// out-of-order @p ctx the work runs on an internal in-order queue that first
/// waits for everything already submitted to @p ctx.
/// @tparam B  backend; the backend-deducing overload takes it from `ctx.backend()`
/// @tparam T  scalar type (float, double, std::complex<float>, std::complex<double>)
/// @param ctx              queue the kernels are enqueued on
/// @param A                batch of geqrf() outputs holding the reflectors; not modified
/// @param C                batch of matrices to multiply; overwritten with the product
/// @param side             apply Q from the left or the right
/// @param trans            apply Q (`NoTrans`) or its (conjugate) transpose
/// @param tau              reflector scalars from geqrf(), `k * batch` elements
/// @param workspace        device-accessible scratch of at least ormqr_buffer_size() bytes,
///                         sized with the same @p block_size_hint
/// @param block_size_hint  WY panel width for the blocked route; 0 lets the tuning
///                         table choose. Clamped to [1, k]; ignored by the vendor route.
/// @return event of the last enqueued kernel
/// @pre `A.rows() == C.rows()` for `Side::Left`, `A.rows() == C.cols()` for
///      `Side::Right`, and A and C share a batch size
/// @pre `tau.size() >= k * batch` (checked by the option overloads)
/// @throws batchlas::workspace_error if @p workspace is smaller than the chosen route needs
/// @throws batchlas::invalid_argument on non-conforming operands (blocked route)
/// @throws batchlas::dispatch::NoRouteError if the vendor route is chosen and the
///         vendor library was not built in
/// @see OrmqrOptions
/// @ingroup qr
template <Backend B, typename T>
BATCHLAS_API Event ormqr(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         const MatrixView<T, MatrixFormat::Dense>& C,
                         Side side,
                         Transpose trans,
                         Span<T> tau,
                         Span<std::byte> workspace,
                         int32_t block_size_hint = 0);

/// @brief Workspace, in bytes, that ormqr() needs for these operands on this queue.
///
/// Resolves the same route and block width as ormqr() from the same inputs, so
/// the result is valid for a call with the same arguments and @p block_size_hint.
/// @ingroup qr
template <Backend B, typename T>
BATCHLAS_API size_t ormqr_buffer_size(Queue& ctx,
                                      const MatrixView<T, MatrixFormat::Dense>& A,
                                      const MatrixView<T, MatrixFormat::Dense>& C,
                                      Side side,
                                      Transpose trans,
                                      Span<T> tau,
                                      int32_t block_size_hint = 0);

} // namespace batchlas

namespace batchlas::backend {

/// @brief Vendor arm of ormqr() (cuSOLVER / rocSOLVER / LAPACKE); not for direct use.
/// @ingroup dispatch
template <Backend B, typename T>
BATCHLAS_API Event ormqr_vendor(Queue& ctx,
                                const MatrixView<T, MatrixFormat::Dense>& A,
                                const MatrixView<T, MatrixFormat::Dense>& C,
                                Side side,
                                Transpose trans,
                                Span<T> tau,
                                Span<std::byte> workspace);

/// @brief Workspace query of the vendor arm of ormqr().
/// @ingroup dispatch
template <Backend B, typename T>
BATCHLAS_API size_t ormqr_vendor_buffer_size(Queue& ctx,
                                             const MatrixView<T, MatrixFormat::Dense>& A,
                                             const MatrixView<T, MatrixFormat::Dense>& C,
                                             Side side,
                                             Transpose trans,
                                             Span<T> tau);

} // namespace batchlas::backend


namespace batchlas::blas::dispatch::detail {

/// @brief Calls ormqr_vendor(), or throws NoRouteError when no vendor library is built in.
/// @ingroup dispatch
// Must stay `if constexpr`: a vendor-free build has no ormqr_vendor symbol to link.
// evidence: docs/design/vendor-independence.md#the-vendor-gate
template <Backend B, typename T, typename... Args>
Event ormqr_vendor_or_throw(Args&&... args) {
    if constexpr (!batchlas::dispatch::factorization_vendor_available<B>) {
        batchlas::dispatch::throw_no_vendor_route<T>(
            batchlas::dispatch::Op::ormqr, B, batchlas::dispatch::kFactorizationLibrary<B>);
    } else {
        return batchlas::backend::ormqr_vendor<B, T>(std::forward<Args>(args)...);
    }
}

/// @brief Calls ormqr_vendor_buffer_size(), or throws NoRouteError when no vendor library is built in.
/// @ingroup dispatch
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

/// @brief Routing shape of an ormqr() call; shared by the call and its size query.
/// @ingroup dispatch
// `side` is carried for fidelity but no predicate reads it.
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

/// @brief Resolves the ormqr() route from `BATCHLAS_ORMQR_ROUTE` and the call's shape.
///
/// The one resolution shared by ormqr_dispatch() and ormqr_buffer_size_dispatch().
/// With the variable unset the request is Auto (unlike gemm, whose default is Vendor).
/// @ingroup dispatch
// evidence: docs/perf/qr.md#ormqr-one-route-resolution-for-the-call-and-its-size-query
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

/// @brief WY block width for the blocked ormqr route.
///
/// A positive @p block_size_hint is clamped to [1, k], k = min(rows, cols);
/// otherwise the tuning table picks a width from `A.rows()`.
/// @ingroup dispatch
// Computed from A alone so the size query and the call agree. The hint exists
// because the table is keyed on panel height, the wrong axis for tall panels.
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

/// @brief Routed implementation behind ormqr(); see ormqr() for the contract.
/// @ingroup dispatch
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

    // Two arms only: the resolver returns a vendor route or a supported native one.
    const size_t need_ws = use_vendor
        ? detail::ormqr_vendor_buffer_size_or_throw<B, T>(ctx, A, C, side, trans, tau)
        : ormqr_blocked_buffer_size<B, T>(ctx, A, C, side, trans, tau, block_size);

    if (workspace.size() < need_ws) {
        throw batchlas::workspace_error("ormqr: insufficient workspace for chosen provider");
    }

    // std::optional, not a plain Queue: the default Queue constructor builds a real
    // sycl::queue on Device::default_device(), which every call (even in-order ones)
    // would pay. It must outlive the if-block because run_q escapes.
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

/// @brief Routed implementation behind ormqr_buffer_size().
/// @ingroup dispatch
template <Backend B, typename T>
inline size_t ormqr_buffer_size_dispatch(Queue& ctx,
                                        const MatrixView<T, MatrixFormat::Dense>& A,
                                        const MatrixView<T, MatrixFormat::Dense>& C,
                                        Side side,
                                        Transpose trans,
                                        Span<T> tau,
                                        int32_t block_size_hint = 0) {
    // Must perform exactly ormqr_dispatch's resolution, from the same inputs.
    // evidence: docs/perf/qr.md#ormqr-one-route-resolution-for-the-call-and-its-size-query
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

// Owning-container and backend-deducing overloads; see blas/queue-dispatch.hh.
BATCHLAS_ACCEPT_OWNING(ormqr)
BATCHLAS_ACCEPT_OWNING(ormqr_buffer_size)

BATCHLAS_DISPATCH_ON_QUEUE(ormqr)
BATCHLAS_DISPATCH_ON_QUEUE(ormqr_buffer_size)

}  // namespace batchlas
