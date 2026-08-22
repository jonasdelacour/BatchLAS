#pragma once

// The GEQRF shape builder and route resolution.
//
// This lives in src/ rather than in the route table header for one reason:
// route_resolve.hh:19-20 requires the table to read ONLY its arguments -- no
// getenv, no SYCL query -- so everything that has to ask the device or the
// environment happens here, and the table sees a plain struct. Modelled on
// src/backends/potrf_route.hh, whose header says the same at :5-8.
//
// The include set is deliberately identical in KIND to potrf_route.hh's: public
// headers plus one private kernel header. It must not gain src/queue.hh or
// <sycl/sycl.hpp> -- gemm_variant.hh:1-9 records that dropping the last such
// include is what made the routing adapters includable from the vendor-free
// facade, and this header is included by that facade.

#include <batchlas/blas/dispatch/route_env.hh>
#include <batchlas/blas/dispatch/route_geqrf.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>

#include "../extensions/geqrf_native.hh"

#include <algorithm>
#include <cstddef>
#include <optional>

namespace batchlas::backend {

// nullopt means "this view does not describe one GEQRF". OpShape is a POD of
// scalars and cannot represent disagreement, so absence is the honest encoding;
// a caller with no shape takes the vendor. Same pattern as gemm_op_shape
// (src/backends/gemm_variant.hh:189-197) and trsm_op_shape
// (src/backends/trsm_route.hh:31-38).
//
// THERE IS NO SQUARENESS TEST HERE, and its absence is the deliberate half of
// this builder. potrf_route.hh:52 rejects a non-square view because a Cholesky
// factor of one does not exist; geqrf's operand is rectangular BY DESIGN
// (options.hh:727-730), so the only structurally non-conforming views are ones
// with negative extents.
//
// NOTHING BELOW MAY DEREFERENCE A.data_ptr(). src/extensions/band_reduction.cc:
// 1041-1044 and :1185-1187 size sytrd's band reduction by calling
// geqrf_buffer_size with `MatrixView<T,Dense> dummyB(nullptr, m_max, nb_max, ...)`
// and `Span<T> dummyTau(nullptr, ...)`, and that query resolves a route through
// this builder. rows()/cols()/batch_size()/is_heterogeneous() are metadata and
// are safe; a data read is an immediate segfault in a sizing path.
template <Backend B, typename T>
inline std::optional<dispatch::GeqrfShape> geqrf_op_shape(
    const Queue& ctx,
    const MatrixView<T, MatrixFormat::Dense>& A) {

    if (A.rows() < 0 || A.cols() < 0) return std::nullopt;

    dispatch::GeqrfShape s;
    s.op = dispatch::Op::geqrf;
    s.scalar = dispatch::scalar_kind_of<T>;

    // SET, unlike trsm's builder (trsm_route.hh:40-56 never assigns s.backend)
    // and unlike ormqr's (ormqr.hh:182-192 does not either) -- which is why every
    // trsm and every ormqr coverage row records Backend::AUTO and the burn-down
    // is unreadable for them. syev sets it (syev.hh:772), potrf sets it
    // (potrf_route.hh:59). resolve_route slices this straight into the coverage
    // table (route_resolve.hh:145-147). Do not inherit ormqr's omission.
    s.backend = B;

    // THE FIELD MAPPING IS ormqr's, NOT potrf's. m and n are the two extents
    // separately because they genuinely differ, and k is the REFLECTOR COUNT
    // min(m,n) -- see the FIELD MAPPING note at the top of route_geqrf.hh for why
    // copying potrf's m == n == k == order here would strip geqrf of its main
    // use.
    s.m = A.rows();
    s.n = A.cols();
    s.k = std::min<int64_t>(A.rows(), A.cols());
    s.batch = A.batch_size();

    // uplo / side / diag / transA / transB are left at their defaults: geqrf
    // takes none of them, and a gate on one would be dead code reading as a live
    // invariant. They still reach coverage.cc:52-58's variant_key, which is part
    // of why a route_diff capture cannot separate geqrf's shape classes -- see
    // the warning in route_geqrf.hh.
    //
    // s.precision stays ComputePrecision::Default: there is no GeqrfOptions
    // (options.hh:597-600 records that geqrf deliberately has none) and no geqrf
    // overload takes a precision, so there is nothing to read.

    s.is_gpu = (ctx.device().type == DeviceType::GPU);

    // ENUMERATED, not `max_sub_group >= 32`. See Device::supports_sub_group_size
    // (sycl-device-queue.hh:180-190) and GeqrfShape::has_sg32 for why the
    // MAX_SUB_GROUP_SIZE property is wrong in both directions.
    s.has_sg32 = ctx.device().supports_sub_group_size(32);

    // THE GATE AND ITS WRITER LAND TOGETHER (potrf_route.hh:83-96).
    // OpShape::heterogeneous_batch (route.hh:236) has few writers in this tree,
    // and every table that gates on it without a builder that sets it carries a
    // DECORATIVE gate -- trsm's (route_trsm.hh:144-151) is exactly that today.
    // Writing it here is inert at merge (supports() is false anyway while the
    // capacities are 0) and is a hard correctness gate the moment a kernel lands.
    s.heterogeneous_batch = A.is_heterogeneous();

    // THE CAPACITIES ARE ASKED OF THIS DEVICE, not of a constant.
    //
    // The budget is the runtime local_mem_size minus the same 4096 B reserve
    // every other device-BLAS sizing decision in this library applies
    // (BatchLASDetectSYCL.cmake:57-67). It must NOT come from
    // build/include/batchlas/device_limits.hh, whose 49152 is hardcoded for any
    // nvidia_gpu_sm_* pattern and is 2.06x wrong on this box (WP4 finding W1) --
    // baking that in would make supports() claim an unlaunchable route on a
    // smaller device and leave a band of extents with no route at all in a
    // vendor-free build.
    //
    // Both answers are 0 today: no kernel is linked, so both native arms are
    // unsupported for every shape and resolve_geqrf_route always returns
    // {Vendor, Auto}. That is the merge state.
    const std::size_t local_mem =
        static_cast<std::size_t>(ctx.device().get_property(DeviceProperty::LOCAL_MEM_SIZE));
    const std::size_t budget = local_mem > 4096 ? local_mem - 4096 : 0;
    s.cta_max_m = sycl_geqrf::geqrf_cta_max_m_for_slm<T>(budget);
    s.cta_max_elems = sycl_geqrf::geqrf_cta_max_elems_for_slm<T>(budget);
    s.blocked_available = sycl_geqrf::geqrf_blocked_available<T>();
    return s;
}

// Resolve a route for one call. Reads the environment; everything shape-derived
// comes from the builder above.
//
// THE ENV READ IS HERE AND ONLY HERE. parse_route_env(Op::geqrf) synthesises
// "BATCHLAS_GEQRF_ROUTE" from op_env_stem (route_env.hh:214-217) -- no registry
// entry exists or is needed, and legacy_variable_for(Op::geqrf) correctly falls
// to `default: return {}` (route_env.hh:119) because no legacy geqrf variable
// ever shipped. Adding a case there would INVENT a legacy spelling;
// tests/route_vocabulary_tests.cc pins that it stays empty.
//
// CALLED FROM EXACTLY TWO PLACES -- geqrf and geqrf_buffer_size -- WITH THE SAME
// ARGUMENTS, which is what makes them reach the same route by construction
// rather than by a comment asking for it. entry_points/factorization.cc:8-10
// states why: "Splitting them would let the two resolve differently, which is
// the defect class S4d found in ormqr (buffer size 2560 bytes, call demanded
// 276480)."
template <Backend B, typename T>
inline dispatch::Route geqrf_route(
    const Queue& ctx,
    const MatrixView<T, MatrixFormat::Dense>& A,
    bool vendor_available) {

    const auto shape = geqrf_op_shape<B, T>(ctx, A);
    if (!shape) {
        return dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::Auto};
    }
    const auto parsed = dispatch::parse_route_env(dispatch::Op::geqrf);
    const dispatch::Route forced =
        parsed.found ? parsed.route : dispatch::legacy_unset_default(dispatch::Op::geqrf);
    return dispatch::resolve_geqrf_route<T>(forced, *shape, vendor_available);
}

} // namespace batchlas::backend
