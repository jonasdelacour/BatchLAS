#pragma once

// The ORGQR shape builder and route resolution.
//
// Same split, and same reason, as src/backends/geqrf_route.hh and
// potrf_route.hh: route_resolve.hh:19-20 requires the table to read ONLY its
// arguments, so every getenv and every SYCL query lives here and the table sees
// a plain struct.
//
// The include set is public headers plus one private kernel header. No
// src/queue.hh, no <sycl/sycl.hpp> -- this header is included by the vendor-free
// facade (gemm_variant.hh:1-9).

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

// nullopt means "this view does not describe one ORGQR" -- the gemm_op_shape
// pattern (gemm_variant.hh:189-197). Only negative extents qualify: n > m is a
// well-formed view that simply has no native route, and it is reported by
// supports() returning false rather than by withholding the shape, so the
// coverage row still records that a call arrived.
//
// NOTHING HERE DEREFERENCES A.data_ptr() OR tau.data(). orgqr is not sized
// against a null view in this tree the way geqrf is (band_reduction.cc:1041-1044),
// but the two queries now share a code path in the facade and the rule costs
// nothing to keep.
template <Backend B, typename T>
inline std::optional<dispatch::OrgqrShape> orgqr_op_shape(
    const Queue& ctx,
    const MatrixView<T, MatrixFormat::Dense>& A) {

    if (A.rows() < 0 || A.cols() < 0) return std::nullopt;

    dispatch::OrgqrShape s;
    s.op = dispatch::Op::orgqr;
    s.scalar = dispatch::scalar_kind_of<T>;

    // SET. See the same note in geqrf_route.hh: ormqr's builder
    // (ormqr.hh:182-192) never assigns it, which is why every ormqr coverage row
    // reads Backend::AUTO. orgqr delegates to ormqr but must not inherit that.
    s.backend = B;

    s.m = A.rows();
    s.n = A.cols();
    s.k = std::min<int64_t>(A.rows(), A.cols());   // reflectors consumed
    s.batch = A.batch_size();

    // THE APPLY IS FIXED AT (Left, NoTrans), AND RECORDING IT HERE IS WHAT MAKES
    // route_orgqr.hh's INHERITED complex-Trans gate honest rather than dead.
    // Q = H_1 H_2 ... H_k I is ormqr(A, I, Side::Left, Transpose::NoTrans), so
    // the gate transcribed from route_ormqr.hh:63-66 cannot fire -- but it is
    // written against a field this builder actually sets, not against a distant
    // invariant, so a future Q^H spelling changes one line here and the gate
    // starts working.
    s.side = Side::Left;
    s.transA = Transpose::NoTrans;

    s.is_gpu = (ctx.device().type == DeviceType::GPU);

    // THE GATE AND ITS WRITER LAND TOGETHER (potrf_route.hh:83-96). ormqr's table
    // has no heterogeneous_batch gate and its builder never sets the field, so
    // ormqr's routing is blind to per-item extents today; orgqr's is not.
    s.heterogeneous_batch = A.is_heterogeneous();

    // NO has_sg32 AND NO SLM CAPACITY. Deliberate, and the reason is in
    // route_orgqr.hh: ormqr_blocked carries no [[sycl::reqd_sub_group_size(32)]]
    // and holds nothing resident, so a sub-group field or a capacity here would
    // be a DECORATIVE input -- the state route_potrf.hh:83-96 criticises trsm
    // for. They arrive with the arm that needs them.
    //
    // FALSE today: no native driver is linked, so the native arm is unsupported
    // for every shape and resolve_orgqr_route always returns {Vendor, Auto}.
    s.blocked_available = sycl_orgqr::orgqr_blocked_available<T>();
    return s;
}

// Resolve a route for one call. Reads the environment.
//
// THE ENV READ IS HERE AND ONLY HERE. parse_route_env(Op::orgqr) synthesises
// "BATCHLAS_ORGQR_ROUTE" (route_env.hh:214-217) and legacy_variable_for(Op::orgqr)
// correctly returns empty (route_env.hh:119) -- no legacy orgqr variable ever
// shipped, and adding a case would invent one.
//
// TWO VARIABLES GOVERN A NATIVE ORGQR, NOT ONE. This call decides whether orgqr
// takes its native arm at all; that arm then re-enters the ROUTED ormqr, which
// reads BATCHLAS_ORMQR_ROUTE (or its legacy BATCHLAS_ORMQR_PROVIDER,
// route_env.hh:118) for itself. Pinning one and not the other is a way to end up
// measuring something other than what was intended.
//
// CALLED FROM EXACTLY TWO PLACES -- orgqr and orgqr_buffer_size -- with the same
// arguments (factorization.cc:8-10).
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
