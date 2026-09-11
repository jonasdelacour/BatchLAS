#pragma once

// GEQRF shape builder: device and environment queries live here so the route table sees
// a plain struct. Do not add src/queue.hh or <sycl/sycl.hpp> -- the vendor-free facade
// includes this header. evidence: docs/perf/qr.md#route-arms

#include <batchlas/blas/dispatch/route_env.hh>
#include <batchlas/blas/dispatch/route_geqrf.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>

#include "../extensions/geqrf_native.hh"

#include <algorithm>
#include <cstddef>
#include <optional>

namespace batchlas::backend {

// No squareness test by design (the operand is rectangular), and NOTHING here may
// dereference A.data_ptr(): band_reduction sizes sytrd through a null-data MatrixView.
template <Backend B, typename T>
inline std::optional<dispatch::GeqrfShape> geqrf_op_shape(
    const Queue& ctx,
    const MatrixView<T, MatrixFormat::Dense>& A) {

    if (A.rows() < 0 || A.cols() < 0) return std::nullopt;

    dispatch::GeqrfShape s;
    s.op = dispatch::Op::geqrf;
    s.scalar = dispatch::scalar_kind_of<T>;

    s.backend = B;

    s.m = A.rows();
    s.n = A.cols();
    s.k = std::min<int64_t>(A.rows(), A.cols());
    s.batch = A.batch_size();

    s.is_gpu = (ctx.device().type == DeviceType::GPU);

    s.has_sg32 = ctx.device().supports_sub_group_size(32);

    s.heterogeneous_batch = A.is_heterogeneous();

    // Ask THIS device (local_mem_size less the 4 KiB reserve): device_limits.hh's
    // hardcoded 49152 admits unlaunchable routes. evidence: docs/perf/qr.md#the-48-kib-launch-hole
    const std::size_t budget = resident::device_slm_budget(
        static_cast<std::size_t>(ctx.device().get_property(DeviceProperty::LOCAL_MEM_SIZE)));
    s.cta_max_m = sycl_geqrf::geqrf_cta_max_m_for_slm<T>(budget);
    s.cta_max_elems = sycl_geqrf::geqrf_cta_max_elems_for_slm<T>(budget);
    // The SAME predicate the tiny launcher applies, from the SAME budget: a second
    // spelling here lets supports() advertise an order geqrf_tiny_dispatch refuses.
    s.tiny_max_n = sycl_geqrf::geqrf_tiny_max_n_for_slm<T>(budget);
    s.blocked_available = sycl_geqrf::geqrf_blocked_available<T>();
    return s;
}

// geqrf and geqrf_buffer_size must pass IDENTICAL arguments, or they resolve apart.
template <Backend B, typename T>
inline dispatch::Route geqrf_route(
    const Queue& ctx,
    const MatrixView<T, MatrixFormat::Dense>& A,
    bool vendor_available) {

    const auto shape = geqrf_op_shape<B, T>(ctx, A);
    if (!shape) {
        return dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::Auto};
    }
    const auto parsed = dispatch::parse_route_env(dispatch::Op::geqrf);
    const dispatch::Route forced =
        parsed.found ? parsed.route : dispatch::legacy_unset_default(dispatch::Op::geqrf);
    return dispatch::resolve_geqrf_route<T>(forced, *shape, vendor_available);
}

} // namespace batchlas::backend
