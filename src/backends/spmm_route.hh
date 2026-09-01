#pragma once

// The SPMM shape builder and route resolution.
//
// Lives in src/ rather than in the route table header for route_resolve.hh:19-21's
// reason: the table must read ONLY its arguments -- no getenv, no SYCL query -- so
// everything that has to ask the device, the environment or a kernel translation
// unit happens here.
//
// The include set is public headers plus one private kernel header, and must NOT
// gain src/queue.hh or <sycl/sycl.hpp>: gemm_variant.hh:1-9 records that dropping
// the last such include is what made the routing adapters includable from the
// vendor-free facade, and this header is included by that facade
// (src/dispatch/entry_points/sparse.cc).

#include <batchlas/blas/dispatch/route_env.hh>
#include <batchlas/blas/dispatch/route_spmm.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>

#include "../sycl/spmm_native.hh"

#include <optional>

namespace batchlas::backend {

// nullopt means "these three views do not describe one SPMM". OpShape is a POD
// of scalars and holds ONE batch and ONE shape, so it cannot represent
// disagreement between A, B and C -- absence is the honest encoding, and a
// caller with no shape takes the vendor. Same pattern as gemm_op_shape
// (gemm_variant.hh:189-197), getrs_op_shape and gemv_op_shape
// (gemv_route.hh:26-33).
//
// THE AGREEMENT CHECKS ARE MADE HERE AND NOWHERE ELSE. There is no
// spmm_validate_params in this tree: the public entry has never validated
// anything, and WP8 deliberately does not add a throw -- that would turn today's
// silent bugs into crashes in live paths and would make WP8 unattributable for
// them. So the checks below are not duplicated safety; they are the only thing
// standing between a non-conforming call and a native kernel indexing off the
// end of a buffer, and their answer is "hand it to the vendor", which is
// precisely what happens today for every spmm call in the tree. Today's
// behaviour is therefore preserved bit for bit on every call the vendor
// currently accepts.
//
// NOTHING BELOW MAY DEREFERENCE data_ptr(), row_offsets()[k] OR col_indices()[k].
// rows()/cols()/batch_size()/ld()/stride()/matrix_stride()/offset_stride()/
// nnz()/is_heterogeneous() are plain int members (matrix.hh:1023-1098) and are
// safe. A.nnz(b) is NOT: it reads row_offsets, and matrix.hh:1081-1086 states
// that a MatrixView over sycl::malloc_device memory is not host-reachable.
// spmm_buffer_size calls this SAME builder, and a data read in a sizing path is
// an immediate segfault rather than a wrong route.
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

    // EXTENTS OF THE OPERANDS AS THE PRODUCT SEES THEM. Which of m and kA is the
    // output extent SWAPS with transA; see SpmmShape::out_rows()/red_rows().
    const int opA_rows = (transA == Transpose::NoTrans) ? m : kA;
    const int opA_cols = (transA == Transpose::NoTrans) ? kA : m;
    const int opB_rows = (transB == Transpose::NoTrans) ? B_mat.rows() : B_mat.cols();
    const int opB_cols = (transB == Transpose::NoTrans) ? B_mat.cols() : B_mat.rows();

    // THE INNER DIMENSIONS MUST MEET, AND C MUST BE THE PRODUCT'S SHAPE.
    if (opA_cols != opB_rows) return std::nullopt;
    if (C.rows() != opA_rows || C.cols() != opB_cols) return std::nullopt;

    // BATCH. The kernels walk A.batch_size() items out of all three views using
    // each view's own stride, so a disagreement is a buffer overrun -- but the
    // vendor is where it goes today, and moving that failure onto a new kernel
    // would make WP8 own it.
    if (A.batch_size() != B_mat.batch_size()) return std::nullopt;
    if (A.batch_size() != C.batch_size()) return std::nullopt;

    // LEADING DIMENSIONS. A non-positive ld makes every dense index degenerate.
    if (B_mat.ld() <= 0 || C.ld() <= 0) return std::nullopt;

    if constexpr (MF == MatrixFormat::CSR) {
        // A CSR item's row-offset array is m + 1 entries long and the bodies read
        // ro[i + 1]; an offset stride shorter than that walks into the next
        // item's offsets. matrix_stride() is the per-item value/index slab, and a
        // negative one has no meaning.
        if (A.offset_stride() < m + 1) return std::nullopt;
        if (A.matrix_stride() < 0) return std::nullopt;
    }

    dispatch::SpmmShape s;
    s.op = dispatch::Op::spmm;
    s.scalar = dispatch::scalar_kind_of<T>;

    // SET. trsm's builder does not, which is why every trsm coverage row reads
    // Backend::AUTO and its burn-down is unreadable (gemv_route.hh:82-86).
    // resolve_route slices this straight into the coverage table
    // (route_resolve.hh:212).
    s.backend = B;

    // FIELD MAPPING. m and k are A's extents AS STORED, not as transposed -- the
    // table derives out_rows()/red_rows() from them and transA. n is the dense
    // width, i.e. nrhs.
    s.m = m;
    s.k = kA;
    s.n = C.cols();
    s.batch = A.batch_size();

    // BOTH TRANSPOSES, AND OMITTING EITHER WOULD BE SILENT. coverage.cc:52-58's
    // variant_key packs transA and transB. For spmm transA is not a flag on one
    // kernel: it is the gather body versus the scale+scatter pair -- different
    // access patterns and different atomics. transB selects a different B index
    // in every body and is the ~7x layout lever this work package measures.
    // Dropping either collapses them into ONE first-writer-wins coverage row and
    // makes scripts/route_diff.sh blind to the distinction.
    s.transA = transA;
    s.transB = transB;

    // MatrixFormat is a runtime FIELD on SpmmShape, not a RouteTable template
    // parameter; see the note above SpmmShape.
    s.format = MF;

    // RECORDED FOR THE COVERAGE ROW, AND DELIBERATELY NOT READ BY supports().
    // The native spmm bodies carry NO GPU GATE -- that is half the WP8
    // deliverable, and the Backend::NETLIB rows on a native_cpu queue are what
    // it buys. See the Direct arm of RouteTable<Op::spmm,T>::supports().
    s.is_gpu = (ctx.device().type == DeviceType::GPU);

    // THE GATE AND ITS WRITER LAND TOGETHER (potrf_route.hh:83-96). Only the
    // DENSE operands can be heterogeneous: active_rows_/active_cols_ are
    // Dense-only (matrix.hh:1036-1042), so a CSR view's is_heterogeneous() is
    // always false and per-item variation in A is expressible only as nnz(b),
    // which the bodies handle exactly through the row offsets.
    s.heterogeneous_batch = B_mat.is_heterogeneous() || C.is_heterogeneous();

    // The capabilities, asked of the kernel TU so the table describes the BUILD
    // and not the design (route_trsm.hh:62-97). Two flags because the NoTrans
    // gather and the transposed scale+scatter pair are independent bodies.
    s.gather_available = sycl_spmm::spmm_gather_available<T>();
    s.scatter_available = sycl_spmm::spmm_scatter_available<T>();
    return s;
}

// Resolve a route for one call. Reads the environment; everything shape-derived
// comes from the builder above.
//
// THE ENV READ IS HERE AND ONLY HERE. parse_route_env(Op::spmm) synthesises
// "BATCHLAS_SPMM_ROUTE" from op_env_stem (route_env.hh:214-217) -- no registry
// entry exists or is needed, and legacy_variable_for(Op::spmm) correctly falls
// to `default: return {}` (route_env.hh:109-121) because no legacy spmm variable
// ever shipped. Adding a case there would INVENT a legacy spelling.
//
// spmm and spmm_buffer_size must call this with the SAME arguments, so that the
// query and the call reach the same route by construction rather than by a
// comment asking them to.
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
