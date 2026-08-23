#pragma once

// The GETRI shape builder and route resolution.
//
// Lives in src/ rather than in the route table header for route_resolve.hh:19-21's
// reason: the table must read ONLY its arguments -- no getenv, no SYCL query.
//
// The include set is public headers plus one private kernel header, and must NOT
// gain src/queue.hh or <sycl/sycl.hpp>: gemm_variant.hh:1-9 records that dropping
// the last such include is what made the routing adapters includable from the
// vendor-free facade, and this header is included by that facade.
//
// ===========================================================================
// THE BUILDER TAKES A AND NOT C, AND THAT IS FORCED, NOT A SIMPLIFICATION.
//
// The rule this family lives by is geqrf_route.hh:136-141's: the shape builder is
// "CALLED FROM EXACTLY TWO PLACES ... WITH THE SAME ARGUMENTS, which is what makes
// them reach the same route by construction rather than by a comment asking for
// it" -- the defect class being ormqr's, where the query said 2560 bytes and the
// call demanded 276480.
//
// getri's two places do not have the same arguments available:
//     the call  : getri(ctx, A, C, pivots, work_space, info)   (getri.hh:301-307)
//     the query : getri_buffer_size(ctx, A)                    (getri.hh:342-344)
// The query has NO C. So the route MUST be a function of A alone; a builder that
// read C could only be called from one of the two sites, which is exactly the
// split the rule forbids.
//
// CONSEQUENCE, STATED SO IT IS NOT MISTAKEN FOR AN OVERSIGHT: a structural
// disagreement between A and C -- non-square C, different order, different batch
// -- is NOT expressible in GetriShape and cannot gate the route. It is checked on
// the arena spellings (options.hh:687-690: require_square on both, require_same_rows,
// require_same_batch) and it must be re-checked by the DRIVER when one exists,
// because the positional overload reaches the facade without those checks. Do not
// "fix" this by giving the builder a C parameter; fix it in the driver, or by
// widening getri_buffer_size's signature in its own change with its own test.
// ===========================================================================

#include <batchlas/blas/dispatch/route_env.hh>
#include <batchlas/blas/dispatch/route_getri.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>

#include "../extensions/getri_native.hh"

#include <optional>

namespace batchlas::backend {

// nullopt means "this view does not describe one GETRI". OpShape is a POD of
// scalars and cannot represent disagreement, so absence is the honest encoding; a
// caller with no shape takes the vendor. Same pattern as potrf_op_shape
// (potrf_route.hh:49-54) and gemm_op_shape (gemm_variant.hh:189-197).
//
// The squareness test duplicates options.hh:687's require_square deliberately, the
// potrf_route.hh:43-47 rule: the builder must not describe a non-conforming view
// even if a future caller reaches it without the arena spelling. Note that the
// VALIDATOR must not throw on a non-square A -- the vendor serves what supports()
// refuses, and a validator that threw would turn a working call into an error --
// so the builder is the only place in this path that can say it.
//
// NOTHING BELOW MAY DEREFERENCE A.data_ptr(). getri_buffer_size is reached from
// inside a layout function under BumpAllocator::measuring()
// (src/extensions/inv.cc:35, from inv_buffer_size at :54-57), and that query
// resolves a route through this builder.
template <Backend B, typename T>
inline std::optional<dispatch::GetriShape> getri_op_shape(
    const Queue& ctx,
    const MatrixView<T, MatrixFormat::Dense>& A) {

    if (A.rows() != A.cols()) return std::nullopt;
    if (A.rows() < 0) return std::nullopt;

    dispatch::GetriShape s;
    s.op = dispatch::Op::getri;
    s.scalar = dispatch::scalar_kind_of<T>;

    // SET. trsm's and ormqr's builders do not, which is why their coverage rows
    // all read Backend::AUTO. resolve_route slices this straight into the coverage
    // table (route_resolve.hh:190-192).
    s.backend = B;

    // FIELD MAPPING -- potrf's: m and n are the two extents separately so
    // supports()' `m == n` gate is representable at all, k is the ORDER and the
    // table reads only order().
    s.m = A.rows();
    s.n = A.cols();
    s.k = A.rows();
    s.batch = A.batch_size();

    // uplo / side / diag / transA / transB stay at their defaults: getri takes
    // none of them. They still reach coverage.cc:52-58's variant_key, which is why
    // getri's coverage rows collapse to shape_class alone -- see the route_diff
    // warning in route_getrf.hh, which applies here verbatim.

    s.is_gpu = (ctx.device().type == DeviceType::GPU);

    // ENUMERATED, not `max_sub_group >= 32`. See GetrfShape::has_sg32.
    s.has_sg32 = ctx.device().supports_sub_group_size(32);

    // THE GATE AND ITS WRITER LAND TOGETHER (potrf_route.hh:83-96). Only A can be
    // asked here -- see the header note on why C is not a parameter -- so a
    // heterogeneous C would slip past this flag. The driver must re-check it.
    s.heterogeneous_batch = A.is_heterogeneous();

    // The capability. TRUE for all four scalar types -- the driver is linked
    // (src/extensions/getri_blocked.cc), so supports() admits the native arm and a
    // vendor-free build takes it for every shape; it is what closed inverse_tests.
    // A vendor-PRESENT build still gets {Vendor, Auto} everywhere, because
    // preferred() is all-false, not because the arm is missing.
    s.blocked_available = sycl_getri::getri_blocked_available<T>();
    return s;
}

// Resolve a route for one call. Reads the environment; everything shape-derived
// comes from the builder above.
//
// THE ENV READ IS HERE AND ONLY HERE. parse_route_env(Op::getri) synthesises
// "BATCHLAS_GETRI_ROUTE" from op_env_stem (route_env.hh:214-217) -- no registry
// entry exists or is needed, and legacy_variable_for(Op::getri) correctly falls to
// `default: return {}` (route_env.hh:119) because no legacy getri variable ever
// shipped. Adding a case there would INVENT a legacy spelling.
//
// CALLED FROM EXACTLY TWO PLACES -- getri and getri_buffer_size -- WITH THE SAME
// ARGUMENTS (factorization.cc:8-10). There are TWO double-resolution sites in the
// tree: options.hh:695 sizes and :696 calls, and src/extensions/inv.cc:35 sizes and
// :49 calls -- each a separate getenv read inside one API call.
template <Backend B, typename T>
inline dispatch::Route getri_route(
    const Queue& ctx,
    const MatrixView<T, MatrixFormat::Dense>& A,
    bool vendor_available) {

    const auto shape = getri_op_shape<B, T>(ctx, A);
    if (!shape) {
        return dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::Auto};
    }
    const auto parsed = dispatch::parse_route_env(dispatch::Op::getri);
    const dispatch::Route forced =
        parsed.found ? parsed.route : dispatch::legacy_unset_default(dispatch::Op::getri);
    return dispatch::resolve_getri_route<T>(forced, *shape, vendor_available);
}

} // namespace batchlas::backend
