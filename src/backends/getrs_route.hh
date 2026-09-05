#pragma once

// The GETRS shape builder and route resolution.
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
#include <batchlas/blas/dispatch/route_getrs.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>

#include "../extensions/getrs_native.hh"

#include <optional>

namespace batchlas::backend {

// nullopt means "these views do not describe one GETRS". OpShape is a POD of
// scalars and holds ONE shape, so it cannot represent disagreement between A and
// B -- absence is the honest encoding, and a caller with no shape takes the
// vendor. Same pattern as gemm_op_shape (gemm_variant.hh:189-197).
//
// THE THREE STRUCTURAL AGREEMENTS ARE TESTED HERE AND NOWHERE ELSE IN THE ROUTING
// LAYER: A square, A.rows() == B.rows(), and equal batch. They duplicate
// options.hh:646-650's checks deliberately (the potrf_route.hh:43-47 rule): the
// builder must not describe a non-conforming pair even if a future caller reaches
// it without the arena spelling.
//
// NOTHING BELOW MAY DEREFERENCE data_ptr(). rows()/cols()/batch_size()/
// is_heterogeneous() are metadata and are safe; a data read is an immediate
// segfault in a sizing path.
template <Backend B, typename T>
inline std::optional<dispatch::GetrsShape> getrs_op_shape(
    const Queue& ctx,
    const MatrixView<T, MatrixFormat::Dense>& A,
    const MatrixView<T, MatrixFormat::Dense>& Bmat,
    Transpose transA) {

    if (A.rows() != A.cols()) return std::nullopt;
    if (A.rows() < 0 || Bmat.rows() < 0 || Bmat.cols() < 0) return std::nullopt;
    if (A.rows() != Bmat.rows()) return std::nullopt;
    if (A.batch_size() != Bmat.batch_size()) return std::nullopt;

    dispatch::GetrsShape s;
    s.op = dispatch::Op::getrs;
    s.scalar = dispatch::scalar_kind_of<T>;

    // SET. trsm's builder (trsm_route.hh:40-56) and ormqr's (ormqr.hh:182-192) do
    // not, which is why every trsm and every ormqr coverage row reads
    // Backend::AUTO and the burn-down is unreadable for them. resolve_route slices
    // this straight into the coverage table (route_resolve.hh:190-192).
    s.backend = B;

    // FIELD MAPPING -- getrs's own. m is the ORDER of the factored matrix, n is
    // nrhs, k is the order again so max_dim()/min_dim() behave sensibly.
    s.m = A.rows();
    s.n = Bmat.cols();
    s.k = A.rows();
    s.batch = A.batch_size();

    // THE ONE LU OP WITH A LIVE VARIANT, AND THE LINE THAT MAKES ITS COVERAGE ROWS
    // SEPARABLE. coverage.cc:52-58's variant_key carries uplo/side/diag/transA/
    // transB; getrf and getri set NONE of them, so their rows collapse to
    // shape_class alone (first-writer-wins, coverage.cc:284-292) and route_diff
    // cannot tell one LU call from another. transA is the only field in this family
    // that separates anything. Dropping this line would be silent.
    //
    // It is also a genuine algorithm fork, not just a label: NoTrans applies P
    // first and solves L then U, while Trans/ConjTrans solves U^T/U^H then L^T/L^H
    // and applies P^T LAST, on the output, in reverse. See route_getrs.hh.
    s.transA = transA;

    s.is_gpu = (ctx.device().type == DeviceType::GPU);

    // ENUMERATED, not `max_sub_group >= 32`. See GetrfShape::has_sg32 for why
    // MAX_SUB_GROUP_SIZE is wrong in both directions.
    s.has_sg32 = ctx.device().supports_sub_group_size(32);

    // THE GATE AND ITS WRITER LAND TOGETHER (potrf_route.hh:83-96). Both views
    // are asked, because either one being heterogeneous breaks the single-tuple
    // launch -- and OpShape has one flag, so the honest reduction is OR.
    s.heterogeneous_batch = A.is_heterogeneous() || Bmat.is_heterogeneous();

    // The capability. TRUE for all four scalar types -- the driver is linked
    // (src/extensions/getrs_native.cc), so supports() admits the native arm and a
    // vendor-free build takes it for every shape. A vendor-PRESENT build still
    // gets {Vendor, Auto} everywhere, because preferred() is all-false, not
    // because the arm is missing.
    s.blocked_available = sycl_getrs::getrs_blocked_available<T>();

    // THE FUSED TIER'S TWO CAPACITY NUMBERS, and the local-memory one is ASKED OF
    // THE DEVICE rather than taken from a constant -- route_potrf.hh:114-127's
    // rule, and getrf_route.hh does the same for cta_max_n. The 4096 B reserve is
    // the one cmake/BatchLASDetectSYCL.cmake:57-67 applies to every other
    // device-BLAS sizing decision in this library, and the formula behind the
    // number lives in src/extensions/getrs_fused.cc beside the launcher so the
    // ceiling this table advertises and the allocation that launcher makes cannot
    // disagree (route_trsm.hh:62-72).
    //
    // BOTH ARE ZERO WHEN THE KERNEL IS ABSENT, which correctly makes the CTA route
    // unsupported rather than selectable-but-unimplemented -- TrsmShape::cta_max_n's
    // convention.
    if (sycl_getrs::getrs_fused_available<T>()) {
        const std::size_t local_mem = ctx.device().get_property(DeviceProperty::LOCAL_MEM_SIZE);
        const std::size_t budget = (local_mem > 4096) ? (local_mem - 4096) : 0;
        s.fused_max_elems =
            static_cast<int64_t>(sycl_getrs::getrs_fused_max_rhs_elems<T>(budget));
        s.fused_max_nrhs = sycl_getrs::kGetrsFusedMaxRhs;
    }
    return s;
}

// Resolve a route for one call. Reads the environment; everything shape-derived
// comes from the builder above.
//
// THE ENV READ IS HERE AND ONLY HERE. parse_route_env(Op::getrs) synthesises
// "BATCHLAS_GETRS_ROUTE" from op_env_stem (route_env.hh:214-217) -- no registry
// entry exists or is needed, and legacy_variable_for(Op::getrs) correctly falls to
// `default: return {}` (route_env.hh:119) because no legacy getrs variable ever
// shipped. Adding a case there would INVENT a legacy spelling.
//
// CALLED FROM EXACTLY TWO PLACES -- getrs and getrs_buffer_size -- WITH THE SAME
// ARGUMENTS, which is what makes them reach the same route by construction rather
// than by a comment asking for it (factorization.cc:8-10). The double resolution
// is real: options.hh:651 sizes and :652 calls, two getenv reads inside one API
// call.
template <Backend B, typename T>
inline dispatch::Route getrs_route(
    const Queue& ctx,
    const MatrixView<T, MatrixFormat::Dense>& A,
    const MatrixView<T, MatrixFormat::Dense>& Bmat,
    Transpose transA,
    bool vendor_available) {

    const auto shape = getrs_op_shape<B, T>(ctx, A, Bmat, transA);
    if (!shape) {
        return dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::Auto};
    }
    const auto parsed = dispatch::parse_route_env(dispatch::Op::getrs);
    const dispatch::Route forced =
        parsed.found ? parsed.route : dispatch::legacy_unset_default(dispatch::Op::getrs);
    return dispatch::resolve_getrs_route<T>(forced, *shape, vendor_available);
}

} // namespace batchlas::backend
