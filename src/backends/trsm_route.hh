#pragma once

// The TRSM shape builder and route resolution.
//
// This lives in src/ rather than in the route table header for one reason:
// route_resolve.hh:19-20 requires the table to read ONLY its arguments -- no
// getenv, no SYCL query -- so everything that has to ask the device or the
// environment happens here, and the table sees a plain struct.

#include <batchlas/blas/dispatch/route_env.hh>
#include <batchlas/blas/dispatch/route_trsm.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>

#include "../sycl/trsm_native.hh"

#include <optional>

namespace batchlas::backend {

// nullopt means "these two views do not describe one TRSM". OpShape is a POD of
// scalars and cannot represent disagreement, so absence is the honest encoding;
// a caller with no shape takes the vendor. Same pattern as gemm_op_shape
// (src/backends/gemm_variant.hh).
//
// NOTE the batch check. trsm_validate_params (functions/trsm.hh:39) does NOT
// compare A.batch_size() to B.batch_size(), so this is the only place that
// disagreement is caught before a kernel would index off the end of one of them.
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
    // m, n are B's extents; k is the TRIANGULAR ORDER. s.n is NOT the triangular
    // order, which is why the table only ever reads tri_order() / rhs_count().
    s.m = B.rows();
    s.n = B.cols();
    s.k = A.rows();
    s.batch = A.batch_size();
    s.side = side;
    s.uplo = uplo;
    s.transA = transA;
    s.diag = diag;
    s.is_gpu = (ctx.device().type == DeviceType::GPU);
    s.cta_max_n = sycl_trsm::trsm_cta_max_n<T>();
    s.blocked_available = sycl_trsm::trsm_blocked_available<T>();
    return s;
}

// Resolve a route for one call. Reads the environment; everything shape-derived
// comes from the builder above.
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
