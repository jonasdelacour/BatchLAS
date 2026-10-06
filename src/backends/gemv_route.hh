#pragma once

// GEMV shape builder + route resolution: all device/env queries, so the table stays
// pure. Included by the vendor-free facade: never add src/queue.hh or <sycl/sycl.hpp>.
// evidence: docs/perf/gemv.md#gemv-the-shape-builder-contract

#include <batchlas/blas/dispatch/route_env.hh>
#include <batchlas/blas/dispatch/route_gemv.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>

#include "../sycl/gemv_native.hh"

#include <optional>

namespace batchlas::backend {

// nullopt -> the vendor. These are the ONLY agreement checks in the tree, and one
// live caller fails them. evidence: docs/perf/gemv.md#the-known-bad-caller
// NEVER dereference data_ptr() here: this runs in sizing paths.
template <Backend B, typename T>
inline std::optional<dispatch::GemvShape> gemv_op_shape(
    const Queue& ctx,
    const MatrixView<T, MatrixFormat::Dense>& A,
    const VectorView<T>& X,
    const VectorView<T>& Y,
    Transpose transA) {

    const int m = A.rows();
    const int n = A.cols();
    if (m < 0 || n < 0) return std::nullopt;

    if (A.batch_size() != X.batch_size()) return std::nullopt;
    if (A.batch_size() != Y.batch_size()) return std::nullopt;

    const int red_len = (transA == Transpose::NoTrans) ? n : m;
    const int out_len = (transA == Transpose::NoTrans) ? m : n;
    if (X.size() != red_len) return std::nullopt;
    if (Y.size() != out_len) return std::nullopt;

    dispatch::GemvShape s;
    s.op = dispatch::Op::gemv;
    s.scalar = dispatch::scalar_kind_of<T>;

    // Set, or the coverage row reads Backend::AUTO.
    s.backend = B;

    // A's extents AS STORED; k repeats m so max_dim()/min_dim() never see a zero.
    s.m = m;
    s.n = n;
    s.k = m;
    s.batch = A.batch_size();

    // Load-bearing: separates body 1 from bodies 2/3 in the coverage variant_key.
    s.transA = transA;

    s.is_gpu = (ctx.device().type == DeviceType::GPU);

    // ENUMERATED, never MAX_SUB_GROUP_SIZE: a reqd_sub_group_size(32) launch aborts.
    s.has_sg32 = ctx.device().supports_sub_group_size(32);

    // Only A can be heterogeneous: VectorView has no active size.
    s.heterogeneous_batch = A.is_heterogeneous();

    s.direct_available = sycl_gemv::gemv_direct_available<T>();
    s.cta_available = sycl_gemv::gemv_cta_available<T>();
    return s;
}

// The only env read. No legacy gemv variable ever shipped; do not invent one.
template <Backend B, typename T>
inline dispatch::Route gemv_route(
    const Queue& ctx,
    const MatrixView<T, MatrixFormat::Dense>& A,
    const VectorView<T>& X,
    const VectorView<T>& Y,
    Transpose transA,
    bool vendor_available) {

    const auto shape = gemv_op_shape<B, T>(ctx, A, X, Y, transA);
    if (!shape) {
        return dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::Auto};
    }
    const auto parsed = dispatch::parse_route_env(dispatch::Op::gemv);
    const dispatch::Route forced =
        parsed.found ? parsed.route : dispatch::legacy_unset_default(dispatch::Op::gemv);
    return dispatch::resolve_gemv_route<T>(forced, *shape, vendor_available);
}

} // namespace batchlas::backend
