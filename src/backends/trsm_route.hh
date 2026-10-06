#pragma once

// The TRSM shape builder and route resolution: every device and environment
// query, so the route table stays pure.
// evidence: docs/perf/trsm.md#the-shape-builder-and-the-field-mapping

#include <batchlas/blas/dispatch/route_env.hh>
#include <batchlas/blas/dispatch/route_trsm.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>

#include "../sycl/trsm_native.hh"

#include <optional>

namespace batchlas::backend {

// nullopt = "these views do not describe one TRSM" -> the vendor. The batch
// check is the only one in the tree: trsm_validate_params does not compare them.
template <typename T>
inline std::optional<dispatch::TrsmShape> trsm_op_shape(
    const Queue& ctx,
    const MatrixView<T, MatrixFormat::Dense>& A,
    const MatrixView<T, MatrixFormat::Dense>& B,
    Side side, Uplo uplo, Transpose transA, Diag diag) {

    if (A.batch_size() != B.batch_size()) return std::nullopt;
    if (A.rows() != A.cols()) return std::nullopt;

    dispatch::TrsmShape s;
    s.op = dispatch::Op::trsm;
    s.scalar = dispatch::scalar_kind_of<T>;
    // m, n are B's extents; k is the TRIANGULAR ORDER (read via tri_order()).
    s.m = B.rows();
    s.n = B.cols();
    s.k = A.rows();
    s.batch = A.batch_size();
    // Without this, supports()'s heterogeneous refusal can never fire.
    s.heterogeneous_batch = A.is_heterogeneous() || B.is_heterogeneous();
    s.side = side;
    s.uplo = uplo;
    s.transA = transA;
    s.diag = diag;
    s.is_gpu = (ctx.device().type == DeviceType::GPU);
    s.cta_max_n = sycl_trsm::trsm_cta_max_n<T>();
    s.blocked_available = sycl_trsm::trsm_blocked_available<T>();
    return s;
}

// Resolve a route for one call; the only env read.
template <typename T>
inline dispatch::Route trsm_route(
    const Queue& ctx,
    const MatrixView<T, MatrixFormat::Dense>& A,
    const MatrixView<T, MatrixFormat::Dense>& B,
    Side side, Uplo uplo, Transpose transA, Diag diag,
    bool vendor_available) {

    const auto shape = trsm_op_shape<T>(ctx, A, B, side, uplo, transA, diag);
    if (!shape) {
        return dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::Auto};
    }
    const auto parsed = dispatch::parse_route_env(dispatch::Op::trsm);
    const dispatch::Route forced =
        parsed.found ? parsed.route : dispatch::legacy_unset_default(dispatch::Op::trsm);
    return dispatch::resolve_trsm_route<T>(forced, *shape, vendor_available);
}

} // namespace batchlas::backend
