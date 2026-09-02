#pragma once

// The POTRF shape builder and route resolution.
//
// This lives in src/ rather than in the route table header for one reason:
// route_resolve.hh:19-20 requires the table to read ONLY its arguments -- no
// getenv, no SYCL query -- so everything that has to ask the device or the
// environment happens here, and the table sees a plain struct. Modelled on
// src/backends/trsm_route.hh, whose header says the same at :5-8.
//
// docs/perf/potrf.md STEP 1.4 IS DELETED BY THIS FILE. spec:474 asks for
// "potrf_supports_cta returns true only under policy.forced == BatchLAS_CTA or
// BATCHLAS_POTRF_PROVIDER=cta". An env-gated supports() violates the purity
// contract AND breaks forcing: resolve_route never bypasses supports()
// (route_resolve.hh:8-10, :101), so making support conditional on the force
// makes the two mutually recursive in meaning.
//
// The include set is deliberately identical in KIND to trsm_route.hh's: public
// headers plus one private kernel header. It must not gain src/queue.hh or
// <sycl/sycl.hpp> -- gemm_variant.hh:1-9 records that dropping the last such
// include is what made the routing adapters includable from the vendor-free
// facade, and this header is included by that facade.

#include <batchlas/blas/dispatch/route_env.hh>
#include <batchlas/blas/dispatch/route_potrf.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>

#include "../extensions/potrf_native.hh"

#include <cstddef>
#include <optional>

namespace batchlas::backend {

// nullopt means "this view does not describe one POTRF". OpShape is a POD of
// scalars and cannot represent disagreement, so absence is the honest encoding;
// a caller with no shape takes the vendor. Same pattern as gemm_op_shape
// (src/backends/gemm_variant.hh:189-197) and trsm_op_shape
// (src/backends/trsm_route.hh:31-38).
//
// The squareness test duplicates potrf_validate_params' -- deliberately, and
// exactly as trsm_route.hh:38 duplicates trsm_validate_params'. The builder
// reads A.rows()/A.cols()/A.batch_size() and must not describe a non-conforming
// view even if a future caller reaches it without the facade.
template <Backend B, typename T>
inline std::optional<dispatch::PotrfShape> potrf_op_shape(
    const Queue& ctx,
    const MatrixView<T, MatrixFormat::Dense>& A,
    Uplo uplo) {

    if (A.rows() != A.cols()) return std::nullopt;

    dispatch::PotrfShape s;
    s.op = dispatch::Op::potrf;
    s.scalar = dispatch::scalar_kind_of<T>;

    // SET, unlike trsm's builder (trsm_route.hh:40-56 never assigns s.backend),
    // which is why every trsm coverage row records Backend::AUTO. syev sets it
    // (syev.hh:772). resolve_route slices this straight into the coverage table
    // (route_resolve.hh:145-147), so an unset backend makes the burn-down
    // unreadable.
    s.backend = B;

    // m and n are the two extents SEPARATELY so that supports()' `m == n` gate
    // is representable at all; k is the order, and the table reads only
    // order(). syev's convention (syev.hh:774-776).
    s.m = A.rows();
    s.n = A.cols();
    s.k = A.rows();
    s.batch = A.batch_size();
    s.uplo = uplo;

    // s.precision is left at ComputePrecision::Default: PotrfOptions
    // (options.hh:519-528) has no precision member and no potrf overload takes
    // one, so there is nothing to read and a gate on it would be dead code
    // reading as a live invariant.

    s.is_gpu = (ctx.device().type == DeviceType::GPU);

    // ENUMERATED, not `>= 32`. See Device::supports_sub_group_size and
    // PotrfShape::has_sg32.
    s.has_sg32 = ctx.device().supports_sub_group_size(32);

    // THE GATE AND ITS WRITER LAND TOGETHER. OpShape::heterogeneous_batch
    // (route.hh:236) has exactly ONE other writer in the tree
    // (gemm_variant.hh:209), and trsm_route.hh:40-56 never sets it -- so trsm's
    // own heterogeneous_batch gate (route_trsm.hh:144-151) is DECORATIVE today.
    // An implementer who copies trsm's builder verbatim inherits the dead gate.
    // Writing it here is inert at merge (supports() is false anyway while
    // cta_max_n == 0) and is a hard correctness gate the moment the kernel
    // lands.
    //
    // RECOMMENDED, OUT OF SCOPE HERE: add the same line to trsm_route.hh. It
    // can only move a route from native to vendor and only for heterogeneous
    // views, so it is a strict de-risking -- but it IS a route change and needs
    // its own scripts/route_diff.sh run.
    s.heterogeneous_batch = A.is_heterogeneous();

    // THE CEILING IS ASKED OF THIS DEVICE, not of a constant.
    //
    // potrf_cta_max_n<T>() answers for the 97,280 B budget this box measures,
    // and shipping that answer everywhere is what would make supports() claim an
    // unlaunchable route on a device with less local memory -- the exact class of
    // false positive supports() exists to exclude. The budget is the runtime
    // local_mem_size minus the same 4096 B reserve every other device-BLAS
    // sizing decision in this library applies (BatchLASDetectSYCL.cmake:57-67).
    // It must NOT come from build/include/batchlas/device_limits.hh, whose 49152
    // is hardcoded for any nvidia_gpu_sm_* pattern and is wrong here by 2.06x
    // (W1). The launcher recomputes the same number from the same query and
    // throws if it disagrees, so the two cannot silently drift.
    const std::size_t local_mem =
        static_cast<std::size_t>(ctx.device().get_property(DeviceProperty::LOCAL_MEM_SIZE));
    s.cta_max_n = sycl_potrf::potrf_cta_max_n_for_slm<T>(local_mem > 4096 ? local_mem - 4096 : 0);
    s.blocked_available = sycl_potrf::potrf_blocked_available<T>();
    return s;
}

// Resolve a route for one call. Reads the environment; everything shape-derived
// comes from the builder above.
//
// THE ENV READ IS HERE AND ONLY HERE. parse_route_env(Op::potrf) synthesises
// "BATCHLAS_POTRF_ROUTE" from op_env_stem (route_env.hh:214-217) -- no registry
// entry exists or is needed, and legacy_variable_for(Op::potrf) correctly falls
// to `default: return {}` (route_env.hh:119) because no legacy potrf variable
// ever shipped. Adding a case there would INVENT a legacy spelling. The spec's
// BATCHLAS_POTRF_PROVIDER is read by nothing in this tree; this call is what
// makes the canonical variable actually take effect, and
// tests/route_vocabulary_tests.cc pins it.
//
// CALLED FROM EXACTLY TWO PLACES -- potrf and potrf_buffer_size -- with the
// same arguments, which is what makes them reach the same route. That is the
// syev pattern (one detail::syev_route, two call sites), and
// entry_points/factorization.cc:8-10 states why: "Splitting them would let the
// two resolve differently, which is the defect class S4d found in ormqr (buffer
// size 2560 bytes, call demanded 276480)."
template <Backend B, typename T>
inline dispatch::Route potrf_route(
    const Queue& ctx,
    const MatrixView<T, MatrixFormat::Dense>& A,
    Uplo uplo,
    bool vendor_available) {

    const auto shape = potrf_op_shape<B, T>(ctx, A, uplo);
    if (!shape) {
        return dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::Auto};
    }
    const auto parsed = dispatch::parse_route_env(dispatch::Op::potrf);
    const dispatch::Route forced =
        parsed.found ? parsed.route : dispatch::legacy_unset_default(dispatch::Op::potrf);
    return dispatch::resolve_potrf_route<T>(forced, *shape, vendor_available);
}

} // namespace batchlas::backend
