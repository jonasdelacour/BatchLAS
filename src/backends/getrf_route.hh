#pragma once

// GETRF shape builder and route resolution. Must not gain src/queue.hh or
// <sycl/sycl.hpp>: the vendor-free facade includes this header.
// Windows and evidence: docs/perf/lu.md#getrf-window-evidence

#include <batchlas/blas/dispatch/route_env.hh>
#include <batchlas/blas/dispatch/route_getrf.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>

#include "../extensions/getrf_native.hh"

#include <cstddef>
#include <optional>

namespace batchlas::backend {

// Nothing below may dereference A.data_ptr(): getrf_buffer_size reaches this from
// a layout function under BumpAllocator::measuring(), where a data read segfaults.
template <Backend B, typename T>
inline std::optional<dispatch::GetrfShape> getrf_op_shape(
    const Queue& ctx,
    const MatrixView<T, MatrixFormat::Dense>& A) {

    if (A.rows() != A.cols()) return std::nullopt;
    if (A.rows() < 0) return std::nullopt;

    dispatch::GetrfShape s;
    s.op = dispatch::Op::getrf;
    s.scalar = dispatch::scalar_kind_of<T>;
    s.backend = B;

    s.m = A.rows();
    s.n = A.cols();
    s.k = A.rows();
    s.batch = A.batch_size();

    s.is_gpu = (ctx.device().type == DeviceType::GPU);

    // Enumerated: the MAX_SUB_GROUP_SIZE property is wrong in both directions.
    s.has_sg32 = ctx.device().supports_sub_group_size(32);

    s.heterogeneous_batch = A.is_heterogeneous();

    // Ask this device: device_limits.hh's hardcoded 49152 would claim an unlaunchable route.
    const std::size_t local_mem =
        static_cast<std::size_t>(ctx.device().get_property(DeviceProperty::LOCAL_MEM_SIZE));
    const std::size_t budget = local_mem > 4096 ? local_mem - 4096 : 0;
    s.cta_max_n = sycl_getrf::getrf_cta_max_n_for_slm<T>(budget);
    s.blocked_available = sycl_getrf::getrf_blocked_available<T>();
    return s;
}

// The only environment read on this path. getrf and getrf_buffer_size must call it
// with identical arguments, or the sizing query and the call can resolve differently.
template <Backend B, typename T>
inline dispatch::Route getrf_route(
    const Queue& ctx,
    const MatrixView<T, MatrixFormat::Dense>& A,
    bool vendor_available) {

    const auto shape = getrf_op_shape<B, T>(ctx, A);
    if (!shape) {
        return dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::Auto};
    }
    const auto parsed = dispatch::parse_route_env(dispatch::Op::getrf);
    const dispatch::Route forced =
        parsed.found ? parsed.route : dispatch::legacy_unset_default(dispatch::Op::getrf);
    return dispatch::resolve_getrf_route<T>(forced, *shape, vendor_available);
}

} // namespace batchlas::backend
