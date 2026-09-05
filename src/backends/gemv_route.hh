#pragma once

// The GEMV shape builder and route resolution.
//
// Lives in src/ rather than in the route table header for route_resolve.hh:19-21's
// reason: the table must read ONLY its arguments -- no getenv, no SYCL query -- so
// everything that has to ask the device or the environment happens here.
//
// The include set is public headers plus one private kernel header, and must NOT
// gain src/queue.hh or <sycl/sycl.hpp>: gemm_variant.hh:1-9 records that dropping
// the last such include is what made the routing adapters includable from the
// vendor-free facade, and this header is included by that facade.

#include <batchlas/blas/dispatch/route_env.hh>
#include <batchlas/blas/dispatch/route_gemv.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>

#include "../sycl/gemv_native.hh"

#include <optional>

namespace batchlas::backend {

// nullopt means "these three views do not describe one GEMV". OpShape is a POD
// of scalars and holds ONE batch and ONE shape, so it cannot represent
// disagreement between A, x and y -- absence is the honest encoding, and a
// caller with no shape takes the vendor. Same pattern as gemm_op_shape
// (gemm_variant.hh:189-197) and getrs_op_shape.
//
// THE AGREEMENT CHECKS ARE MADE HERE AND NOWHERE ELSE. There is no
// gemv_validate_params in this tree: the public entry has never validated
// anything, and WP7 deliberately does not add a throw (that would turn today's
// silent bugs into crashes in live paths, and would make WP7 unattributable for
// them). So the checks below are not duplicated safety -- they are the only
// thing standing between a non-conforming call and a native kernel indexing off
// the end of a buffer, and their answer is "hand it to the vendor", which is
// precisely what happens today for every gemv call in the tree.
//
// ONE SUCH NON-CONFORMING CALL IS LIVE AND KNOWN. ortho.cc:217-224's
// transA = Trans branch builds A_i as (i x m) and then passes A(Slice(), i) --
// a column of length A.rows() -- as x, so x's length does not match the
// reduction length. It is structurally wrong TODAY, under the vendor, and WP7
// does not fix it and does not throw on it. What the length check below DOES
// guarantee is that it keeps going wherever it goes today rather than becoming
// a native out-of-bounds read.
//
// NOTHING BELOW MAY DEREFERENCE data_ptr(). rows()/cols()/size()/inc()/
// batch_size()/is_heterogeneous() are metadata and are safe; a data read is an
// immediate segfault in a sizing path.
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

    // BATCH. cuBLAS's strided-batched call reads A.batch_size() items out of
    // all three views using each view's own stride, so a disagreement is a
    // buffer overrun in the vendor too -- but the vendor is where it goes
    // today, and moving that failure onto a new kernel would make WP7 own it.
    if (A.batch_size() != X.batch_size()) return std::nullopt;
    if (A.batch_size() != Y.batch_size()) return std::nullopt;

    // LENGTHS. Which of m and n is x's and which is y's SWAPS with transA; see
    // GemvShape::out_len()/red_len().
    const int red_len = (transA == Transpose::NoTrans) ? n : m;
    const int out_len = (transA == Transpose::NoTrans) ? m : n;
    if (X.size() != red_len) return std::nullopt;
    if (Y.size() != out_len) return std::nullopt;

    dispatch::GemvShape s;
    s.op = dispatch::Op::gemv;
    s.scalar = dispatch::scalar_kind_of<T>;

    // SET. trsm's builder (trsm_route.hh:40-56) does not, which is why every
    // trsm coverage row reads Backend::AUTO and the burn-down is unreadable for
    // it. resolve_route slices this straight into the coverage table
    // (route_resolve.hh:190-192).
    s.backend = B;

    // FIELD MAPPING. m and n are A's extents as STORED, not as transposed --
    // the table derives out_len/red_len from them and transA. k repeats m so
    // max_dim()/min_dim() range over the two real extents rather than over a
    // zero.
    s.m = m;
    s.n = n;
    s.k = m;
    s.batch = A.batch_size();

    // THE FIELD THAT SEPARATES GEMV'S TWO KERNELS, and the one whose omission
    // would be silent. coverage.cc:47-58's variant_key carries transA; the two
    // transA values here are not a flag on one kernel, they are body 1 versus
    // bodies 2/3 -- different access patterns, different routes, different
    // measured behaviour (the one cuBLAS slow region in the whole baseline is
    // Trans-only). Dropping this line collapses them into ONE first-writer-wins
    // coverage row and makes route_diff blind to the distinction.
    s.transA = transA;

    s.is_gpu = (ctx.device().type == DeviceType::GPU);

    // ENUMERATED, not `max_sub_group >= 32`. Device::supports_sub_group_size
    // (sycl-device-queue.hh:178-190) walks sycl::info::device::sub_group_sizes;
    // get_property(MAX_SUB_GROUP_SIZE) returns sub_group_sizes()[0], the FIRST
    // supported size, which is wrong in both directions -- and for a kernel
    // carrying [[sycl::reqd_sub_group_size(32)]] the "accepted although it has
    // no 32" direction is a launch abort.
    s.has_sg32 = ctx.device().supports_sub_group_size(32);

    // THE GATE AND ITS WRITER LAND TOGETHER (potrf_route.hh:83-96). Only A can
    // be heterogeneous -- VectorView has no active-size concept at all, which
    // is itself why gemv cannot have gemm's heterogeneous walker.
    s.heterogeneous_batch = A.is_heterogeneous();

    // The capabilities, asked of the kernel TU so the table describes the BUILD
    // and not the design (route_trsm.hh:62-84).
    s.direct_available = sycl_gemv::gemv_direct_available<T>();
    s.cta_available = sycl_gemv::gemv_cta_available<T>();
    return s;
}

// Resolve a route for one call. Reads the environment; everything shape-derived
// comes from the builder above.
//
// THE ENV READ IS HERE AND ONLY HERE. parse_route_env(Op::gemv) synthesises
// "BATCHLAS_GEMV_ROUTE" from op_env_stem (route_env.hh:214-217) -- no registry
// entry exists or is needed, and legacy_variable_for(Op::gemv) correctly falls
// to `default: return {}` (route_env.hh:119) because no legacy gemv variable
// ever shipped. Adding a case there would INVENT a legacy spelling.
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
