#pragma once

// The POTRF shape builder and route resolution: device and environment queries
// live here so the route table reads only its arguments. Do not add src/queue.hh
// or <sycl/sycl.hpp>; the vendor-free facade includes this. docs/perf/potrf.md

#include <batchlas/blas/dispatch/route_env.hh>
#include <batchlas/blas/dispatch/route_potrf.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>

#include "../extensions/potrf_native.hh"

#include <cstddef>
#include <optional>

namespace batchlas::backend {

template <Backend B, typename T>
inline std::optional<dispatch::PotrfShape> potrf_op_shape(
    const Queue& ctx,
    const MatrixView<T, MatrixFormat::Dense>& A,
    Uplo uplo) {

    if (A.rows() != A.cols()) return std::nullopt;

    dispatch::PotrfShape s;
    s.op = dispatch::Op::potrf;
    s.scalar = dispatch::scalar_kind_of<T>;
    s.backend = B;

    // m and n stay separate so the `m == n` gate is representable; k is the order.
    s.m = A.rows();
    s.n = A.cols();
    s.k = A.rows();
    s.batch = A.batch_size();
    s.uplo = uplo;

    s.is_gpu = (ctx.device().type == DeviceType::GPU);

    // Enumerated support, not `>= 32`.
    s.has_sg32 = ctx.device().supports_sub_group_size(32);

    // Not dead code: this becomes a correctness gate the moment the CTA kernel lands.
    s.heterogeneous_batch = A.is_heterogeneous();

    // Query THIS device: a hardcoded budget makes supports() admit a route that
    // cannot launch. evidence: docs/perf/potrf.md#the-slm-budget-and-the-fit-ceilings
    const std::size_t local_mem =
        static_cast<std::size_t>(ctx.device().get_property(DeviceProperty::LOCAL_MEM_SIZE));
    s.cta_max_n = sycl_potrf::potrf_cta_max_n_for_slm<T>(local_mem > 4096 ? local_mem - 4096 : 0);
    s.blocked_available = sycl_potrf::potrf_blocked_available<T>();
    return s;
}

// The only env read on the potrf path; potrf and potrf_buffer_size must call it
// with identical arguments or the reported buffer size will not match the call.
template <Backend B, typename T>
inline dispatch::Route potrf_route(
    const Queue& ctx,
    const MatrixView<T, MatrixFormat::Dense>& A,
    Uplo uplo,
    bool vendor_available) {

    const auto shape = potrf_op_shape<B, T>(ctx, A, uplo);
    if (!shape) {
        return dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::Auto};
    }
    const auto parsed = dispatch::parse_route_env(dispatch::Op::potrf);
    const dispatch::Route forced =
        parsed.found ? parsed.route : dispatch::legacy_unset_default(dispatch::Op::potrf);
    return dispatch::resolve_potrf_route<T>(forced, *shape, vendor_available);
}

} // namespace batchlas::backend
