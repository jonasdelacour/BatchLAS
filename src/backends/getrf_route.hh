#pragma once

// The GETRF shape builder and route resolution.
//
// This lives in src/ rather than in the route table header for one reason:
// route_resolve.hh:19-21 requires the table to read ONLY its arguments -- no
// getenv, no SYCL query -- so everything that has to ask the device or the
// environment happens here, and the table sees a plain struct. Modelled line for
// line on src/backends/geqrf_route.hh, itself modelled on potrf_route.hh.
//
// The include set is deliberately identical in KIND to geqrf_route.hh's: public
// headers plus one private kernel header. It must NOT gain src/queue.hh or
// <sycl/sycl.hpp> -- gemm_variant.hh:1-9 records that dropping the last such
// include is what made the routing adapters includable from the vendor-free
// facade, and this header is included by that facade.

#include <batchlas/blas/dispatch/route_env.hh>
#include <batchlas/blas/dispatch/route_getrf.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>

#include "../extensions/getrf_native.hh"

#include <cstddef>
#include <optional>

namespace batchlas::backend {

// nullopt means "this view does not describe one GETRF". OpShape is a POD of
// scalars and cannot represent disagreement, so absence is the honest encoding; a
// caller with no shape takes the vendor. Same pattern as gemm_op_shape
// (src/backends/gemm_variant.hh:189-197), trsm_op_shape (trsm_route.hh:31-38) and
// potrf_op_shape (potrf_route.hh:49-54).
//
// THE SQUARENESS TEST IS HERE, and unlike geqrf's builder -- which deliberately
// has none, because rectangular A is the entire point of that op -- it belongs.
// It duplicates getrf_validate_params' absence deliberately, exactly as
// potrf_route.hh:52 duplicates potrf_validate_params': the builder reads
// A.rows()/A.cols()/A.batch_size() and must not describe a non-conforming view
// even if a future caller reaches it without the facade. Note that the VALIDATOR
// must not throw on a non-square A (the vendor serves what supports() refuses, and
// a validator that threw would turn a working call into an error) -- so the
// builder is the only place this can be said.
//
// NOTHING BELOW MAY DEREFERENCE A.data_ptr(). getrf_buffer_size is reached from
// inside a layout function under BumpAllocator::measuring()
// (src/extensions/inv.cc:36, from inv_buffer_size at :54-57), and that query
// resolves a route through this builder. rows()/cols()/batch_size()/
// is_heterogeneous() are metadata and are safe; a data read is an immediate
// segfault in a sizing path.
template <Backend B, typename T>
inline std::optional<dispatch::GetrfShape> getrf_op_shape(
    const Queue& ctx,
    const MatrixView<T, MatrixFormat::Dense>& A) {

    if (A.rows() != A.cols()) return std::nullopt;
    if (A.rows() < 0) return std::nullopt;

    dispatch::GetrfShape s;
    s.op = dispatch::Op::getrf;
    s.scalar = dispatch::scalar_kind_of<T>;

    // SET, unlike trsm's builder (trsm_route.hh:40-56 never assigns s.backend) and
    // unlike ormqr's (ormqr.hh:182-192 does not either) -- which is why every trsm
    // and every ormqr coverage row records Backend::AUTO and the burn-down is
    // unreadable for them. syev sets it (syev.hh:772), potrf sets it
    // (potrf_route.hh:59), geqrf sets it (geqrf_route.hh:66). resolve_route slices
    // this straight into the coverage table (route_resolve.hh:190-192). Do not
    // inherit ormqr's omission.
    s.backend = B;

    // THE FIELD MAPPING IS potrf's, NOT geqrf's. m and n are the two extents
    // separately so that supports()' `m == n` gate is representable at all; k is
    // THE ORDER, and the table reads only order(). See the FIELD MAPPING note at
    // the top of route_getrf.hh for why getrf agrees with potrf here and geqrf
    // does not.
    s.m = A.rows();
    s.n = A.cols();
    s.k = A.rows();
    s.batch = A.batch_size();

    // uplo / side / diag / transA / transB are left at their defaults: getrf takes
    // none of them, and a gate on one would be dead code reading as a live
    // invariant. They still reach coverage.cc:52-58's variant_key, which is why a
    // route_diff capture cannot separate getrf's shape classes at all -- see the
    // warning in route_getrf.hh.
    //
    // s.precision stays ComputePrecision::Default: there is no GetrfOptions and no
    // getrf overload takes a precision, so there is nothing to read.

    s.is_gpu = (ctx.device().type == DeviceType::GPU);

    // ENUMERATED, not `max_sub_group >= 32`. See Device::supports_sub_group_size
    // (sycl-device-queue.hh:180-190) and GetrfShape::has_sg32 for why the
    // MAX_SUB_GROUP_SIZE property is wrong in both directions.
    s.has_sg32 = ctx.device().supports_sub_group_size(32);

    // THE GATE AND ITS WRITER LAND TOGETHER (potrf_route.hh:83-96).
    // OpShape::heterogeneous_batch (route.hh:236) has few writers in this tree, and
    // every table that gates on it without a builder that sets it carries a
    // DECORATIVE gate -- trsm's (route_trsm.hh:144-151) is exactly that today.
    // Writing it here is inert at merge (supports() is false anyway while the
    // capacity is 0) and is a hard correctness gate the moment a kernel lands.
    s.heterogeneous_batch = A.is_heterogeneous();

    // THE CAPACITY IS ASKED OF THIS DEVICE, not of a constant.
    //
    // The budget is the runtime local_mem_size minus the same 4096 B reserve every
    // other device-BLAS sizing decision in this library applies
    // (BatchLASDetectSYCL.cmake:57-67). It must NOT come from
    // build/include/batchlas/device_limits.hh, whose 49152 is hardcoded for any
    // nvidia_gpu_sm_* pattern and is 2.06x wrong on this box (WP4 finding W1) --
    // baking that in would make supports() claim an unlaunchable route on a smaller
    // device and leave a band of orders with no route at all in a vendor-free
    // build.
    //
    // BOTH ANSWERS ARE REAL. On an RTX 4090 (97,280 B budget) cta_max_n measures
    // 155/109/109/77 for float/double/cfloat/cdouble and blocked_available is true
    // for every type, so a vendor-free build takes a native arm for every square
    // shape and native_tier_preferred picks between the two. A vendor-PRESENT
    // build still resolves {Vendor, Auto} everywhere -- because preferred() is
    // all-false, not because a capability is zero.
    const std::size_t local_mem =
        static_cast<std::size_t>(ctx.device().get_property(DeviceProperty::LOCAL_MEM_SIZE));
    const std::size_t budget = local_mem > 4096 ? local_mem - 4096 : 0;
    s.cta_max_n = sycl_getrf::getrf_cta_max_n_for_slm<T>(budget);
    s.blocked_available = sycl_getrf::getrf_blocked_available<T>();
    return s;
}

// Resolve a route for one call. Reads the environment; everything shape-derived
// comes from the builder above.
//
// THE ENV READ IS HERE AND ONLY HERE. parse_route_env(Op::getrf) synthesises
// "BATCHLAS_GETRF_ROUTE" from op_env_stem (route_env.hh:214-217) -- no registry
// entry exists or is needed, and legacy_variable_for(Op::getrf) correctly falls to
// `default: return {}` (route_env.hh:119) because no legacy getrf variable ever
// shipped. Adding a case there would INVENT a legacy spelling;
// tests/route_vocabulary_tests.cc pins that it stays empty. (Op::ormqr DOES have
// one at :118 -- that is not a precedent for this op.)
//
// CALLED FROM EXACTLY TWO PLACES -- getrf and getrf_buffer_size -- WITH THE SAME
// ARGUMENTS, which is what makes them reach the same route by construction rather
// than by a comment asking for it. entry_points/factorization.cc:8-10 states why:
// "Splitting them would let the two resolve differently, which is the defect class
// S4d found in ormqr (buffer size 2560 bytes, call demanded 276480)." For getrf the
// double resolution is real and there are TWO of them in one API call:
// options.hh:619 sizes and :620 calls, and src/extensions/inv.cc:36 sizes and :48
// calls -- each a separate getenv read.
template <Backend B, typename T>
inline dispatch::Route getrf_route(
    const Queue& ctx,
    const MatrixView<T, MatrixFormat::Dense>& A,
    bool vendor_available) {

    const auto shape = getrf_op_shape<B, T>(ctx, A);
    if (!shape) {
        return dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::Auto};
    }
    const auto parsed = dispatch::parse_route_env(dispatch::Op::getrf);
    const dispatch::Route forced =
        parsed.found ? parsed.route : dispatch::legacy_unset_default(dispatch::Op::getrf);
    return dispatch::resolve_getrf_route<T>(forced, *shape, vendor_available);
}

} // namespace batchlas::backend
