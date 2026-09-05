#pragma once

// SPMM shape building and route resolution -- the routing work that must ask the
// device, the environment or a kernel TU. Do not add src/queue.hh or <sycl/sycl.hpp>
// here: the vendor-free facade (dispatch/entry_points/sparse.cc) includes this
// header. evidence: docs/perf/spmm.md

#include <batchlas/blas/dispatch/route_env.hh>
#include <batchlas/blas/dispatch/route_spmm.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>

#include "../sycl/spmm_native.hh"

#include <optional>

namespace batchlas::backend {

// nullopt means "these views do not describe one SPMM"; such a call takes the
// vendor. These checks are the only validation in the tree. Nothing below may
// dereference data_ptr(), row_offsets() or col_indices(): the memory may be
// device-only, and spmm_buffer_size calls this same builder.
template <Backend B, typename T, MatrixFormat MF>
inline std::optional<dispatch::SpmmShape> spmm_op_shape(
    const Queue& ctx,
    const MatrixView<T, MF>& A,
    const MatrixView<T, MatrixFormat::Dense>& B_mat,
    const MatrixView<T, MatrixFormat::Dense>& C,
    Transpose transA,
    Transpose transB) {

    const int m = A.rows();
    const int kA = A.cols();
    if (m < 0 || kA < 0) return std::nullopt;

    const int opA_rows = (transA == Transpose::NoTrans) ? m : kA;
    const int opA_cols = (transA == Transpose::NoTrans) ? kA : m;
    const int opB_rows = (transB == Transpose::NoTrans) ? B_mat.rows() : B_mat.cols();
    const int opB_cols = (transB == Transpose::NoTrans) ? B_mat.cols() : B_mat.rows();

    if (opA_cols != opB_rows) return std::nullopt;
    if (C.rows() != opA_rows || C.cols() != opB_cols) return std::nullopt;

    if (A.batch_size() != B_mat.batch_size()) return std::nullopt;
    if (A.batch_size() != C.batch_size()) return std::nullopt;

    if (B_mat.ld() <= 0 || C.ld() <= 0) return std::nullopt;

    if constexpr (MF == MatrixFormat::CSR) {
        // Bodies read ro[i + 1], so an offset stride under m + 1 walks into the next item.
        if (A.offset_stride() < m + 1) return std::nullopt;
        if (A.matrix_stride() < 0) return std::nullopt;
    }

    dispatch::SpmmShape s;
    s.op = dispatch::Op::spmm;
    s.scalar = dispatch::scalar_kind_of<T>;

    s.backend = B;

    // A's extents AS STORED; the table derives out_rows()/red_rows() from these and transA.
    s.m = m;
    s.k = kA;
    s.n = C.cols();
    s.batch = A.batch_size();

    s.transA = transA;
    s.transB = transB;

    s.format = MF;

    // Recorded for coverage; supports() deliberately ignores it -- the bodies have no GPU gate.
    s.is_gpu = (ctx.device().type == DeviceType::GPU);

    // Only the dense operands can be heterogeneous; a CSR view varies per item only through nnz.
    s.heterogeneous_batch = B_mat.is_heterogeneous() || C.is_heterogeneous();

    s.gather_available = sycl_spmm::spmm_gather_available<T>();
    s.scatter_available = sycl_spmm::spmm_scatter_available<T>();
    return s;
}

// spmm and spmm_buffer_size must call this with the SAME arguments, so the sizing
// query and the call resolve to one route by construction.
template <Backend B, typename T, MatrixFormat MF>
inline dispatch::Route spmm_route(
    const Queue& ctx,
    const MatrixView<T, MF>& A,
    const MatrixView<T, MatrixFormat::Dense>& B_mat,
    const MatrixView<T, MatrixFormat::Dense>& C,
    Transpose transA,
    Transpose transB,
    bool vendor_available) {

    const auto shape = spmm_op_shape<B, T, MF>(ctx, A, B_mat, C, transA, transB);
    if (!shape) {
        return dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::Auto};
    }
    const auto parsed = dispatch::parse_route_env(dispatch::Op::spmm);
    const dispatch::Route forced =
        parsed.found ? parsed.route : dispatch::legacy_unset_default(dispatch::Op::spmm);
    return dispatch::resolve_spmm_route<T>(forced, *shape, vendor_available);
}

} // namespace batchlas::backend
