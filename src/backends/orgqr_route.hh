#pragma once

// The ORGQR shape builder and route resolution: every getenv and device query lives here so
// the table reads only a plain struct (route_resolve.hh). Included by the vendor-free facade:
// no src/queue.hh, no <sycl/sycl.hpp>.

#include <batchlas/blas/dispatch/route_env.hh>
#include <batchlas/blas/dispatch/route_orgqr.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>

#include "../extensions/orgqr_native.hh"

#include <algorithm>
#include <cstddef>
#include <optional>

namespace batchlas::backend {

// nullopt only for negative extents. n > m is well-formed with no native route: supports()
// says false, so the coverage row still records the call.
// NOTHING HERE DEREFERENCES A.data_ptr() OR tau.data() (shared sizing path in the facade).
template <Backend B, typename T>
inline std::optional<dispatch::OrgqrShape> orgqr_op_shape(
    const Queue& ctx,
    const MatrixView<T, MatrixFormat::Dense>& A) {

    if (A.rows() < 0 || A.cols() < 0) return std::nullopt;

    dispatch::OrgqrShape s;
    s.op = dispatch::Op::orgqr;
    s.scalar = dispatch::scalar_kind_of<T>;

    // SET, unlike ormqr's builder (whose coverage rows therefore all read AUTO).
    s.backend = B;

    s.m = A.rows();
    s.n = A.cols();
    s.k = std::min<int64_t>(A.rows(), A.cols());   // reflectors consumed
    s.batch = A.batch_size();

    // Q = H_1 ... H_k I is ormqr(Left, NoTrans). Recorded so route_orgqr.hh's inherited
    // complex-Trans gate reads a field that is actually set: a future Q^H changes one line here.
    s.side = Side::Left;
    s.transA = Transpose::NoTrans;

    s.is_gpu = (ctx.device().type == DeviceType::GPU);

    // The gate and its writer land together (potrf_route.hh); ormqr's table has neither.
    s.heterogeneous_batch = A.is_heterogeneous();

    // NO has_sg32 AND NO SLM CAPACITY, deliberately: ormqr_blocked requires no sub-group size
    // and holds nothing resident, so either field would be a decorative input (route_orgqr.hh).
    // TRUE for all four types. preferred() is true to n = 512, so FALSE would also send the
    // vendor-present default back to the per-item cusolverDnXorgqr loop. evidence: docs/perf/small-n-baseline.md#orgqr
    s.blocked_available = sycl_orgqr::orgqr_blocked_available<T>();
    return s;
}

// Resolve a route for one call. THE ENV READ IS HERE AND ONLY HERE: BATCHLAS_ORGQR_ROUTE,
// with no legacy spelling. TRAP: TWO VARIABLES GOVERN A NATIVE ORGQR -- its native arm
// re-enters the ROUTED ormqr, which reads BATCHLAS_ORMQR_ROUTE itself; pin both when timing.
// Called from orgqr and orgqr_buffer_size with the same arguments.
template <Backend B, typename T>
inline dispatch::Route orgqr_route(
    const Queue& ctx,
    const MatrixView<T, MatrixFormat::Dense>& A,
    bool vendor_available) {

    const auto shape = orgqr_op_shape<B, T>(ctx, A);
    if (!shape) {
        return dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::Auto};
    }
    const auto parsed = dispatch::parse_route_env(dispatch::Op::orgqr);
    const dispatch::Route forced =
        parsed.found ? parsed.route : dispatch::legacy_unset_default(dispatch::Op::orgqr);
    return dispatch::resolve_orgqr_route<T>(forced, *shape, vendor_available);
}

} // namespace batchlas::backend
