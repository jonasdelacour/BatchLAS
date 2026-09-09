#pragma once

// The GETRI shape builder and route resolution; the routed window itself lives in
// route_getri.hh (evidence: docs/perf/lu.md#getri-window-evidence). This header must
// not gain src/queue.hh or <sycl/sycl.hpp> -- the vendor-free facade includes it.
// The shape is built from A alone because getri_buffer_size has no C, and both call
// sites must build it identically to reach the same route.

#include <batchlas/blas/dispatch/route_env.hh>
#include <batchlas/blas/dispatch/route_getri.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>

#include "../extensions/getri_native.hh"

#include <optional>

namespace batchlas::backend {

// nullopt means "this view does not describe one GETRI"; that caller takes the
// vendor. Nothing below may dereference A.data_ptr(): getri_buffer_size resolves a
// route from inside a layout function under BumpAllocator::measuring().
template <Backend B, typename T>
inline std::optional<dispatch::GetriShape> getri_op_shape(
    const Queue& ctx,
    const MatrixView<T, MatrixFormat::Dense>& A) {

    if (A.rows() != A.cols()) return std::nullopt;
    if (A.rows() < 0) return std::nullopt;

    dispatch::GetriShape s;
    s.op = dispatch::Op::getri;
    s.scalar = dispatch::scalar_kind_of<T>;

    s.backend = B;

    s.m = A.rows();
    s.n = A.cols();
    s.k = A.rows();
    s.batch = A.batch_size();

    s.is_gpu = (ctx.device().type == DeviceType::GPU);

    // Enumerated, not `max_sub_group >= 32`. See GetrfShape::has_sg32.
    s.has_sg32 = ctx.device().supports_sub_group_size(32);

    s.heterogeneous_batch = A.is_heterogeneous();

    s.blocked_available = sycl_getri::getri_blocked_available<T>();
    return s;
}

template <Backend B, typename T>
inline dispatch::Route getri_route(
    const Queue& ctx,
    const MatrixView<T, MatrixFormat::Dense>& A,
    bool vendor_available) {

    const auto shape = getri_op_shape<B, T>(ctx, A);
    if (!shape) {
        return dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::Auto};
    }
    const auto parsed = dispatch::parse_route_env(dispatch::Op::getri);
    const dispatch::Route forced =
        parsed.found ? parsed.route : dispatch::legacy_unset_default(dispatch::Op::getri);
    return dispatch::resolve_getri_route<T>(forced, *shape, vendor_available);
}

} // namespace batchlas::backend
