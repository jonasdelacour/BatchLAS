#pragma once

// Record the route a level-3 dispatcher ACTUALLY took.
//
// WHY THIS EXISTS. WP1 changes where symm/syrk/syr2k/trmm terminate. The only
// acceptable outcome on a vendor-present box is that no decision moves, and
// establishing that needs an instrument, because:
//
//   * reading the diff cannot show it -- the whole point of a dispatcher is
//     that the decision is not visible at the call site;
//   * timing cannot show it -- an unsaturated benchmark's ratios are overhead,
//     and routing a shape to cuBLAS may well be FASTER, so a perf gate cannot
//     flag a wrong route;
//   * the kernel trace cannot show it -- src/util/kernel-trace.hh's Record
//     holds a sycl::event, so a vendor-to-vendor route change is invisible to
//     it, and so is any route whose kernel it does not scope.
//
// The dispatch coverage table can, and scripts/route_diff.sh diffs it. But it
// was blind to these four ops: dispatch::resolve_route records every op that
// goes through it, and symm/syrk/syr2k/trmm do not go through it. They have no
// RouteTable<Op, T> specialisation at all -- only gemm, gesvd, ormqr and syev
// do. WP0 gave these ops the Route VOCABULARY (parse_route_env, is_plain_vendor)
// but never the RESOLVER; their thresholds are still hand-rolled if-chains.
//
// So they are instrumented directly, at each terminal, reporting the branch
// actually taken. This changes no decision -- every call site is a statement
// added beside a `return`, never in place of one.
//
// GIVING THEM A RouteTable INSTEAD WOULD NOT BE EQUIVALENT, and is the reason
// this is a separate header rather than a resolver specialisation. The live
// thresholds are GATE-ONLY: syrk_cuda_custom's Auto arm takes
// syrk_triangular_tiles unconditionally once the gram test fails, with no
// second preference check. Transcribing those thresholds into `preferred()`
// makes resolve_route's automatic() reject the tile route for shapes it serves
// today -- for 129 <= n <= 383 at every batch, since
// triangular_tiles_per_side(256) == 2 fails a `>= 3` rule -- sending n=256 to a
// route that writes BOTH triangles. Measuring first, transcribing later.

#include <cstdint>

#include <batchlas/blas/dispatch/coverage.hh>
#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/enums.hh>

namespace batchlas::backend::detail {

// `native_supported` is whether a native kernel could have served THIS shape,
// which is what separates "the vendor was faster here" from "there was nothing
// else to choose". The four dispatchers already compute it as their
// *_problem_supported / *_triangular_supported predicate, so it is passed in
// rather than recomputed.
//
// It is a TRI-STATE, and the third value is load-bearing. When the gate in
// cublas.cc declines, `*_cuda_custom` is never entered and the caller has only
// one bit: the gate said no. That conflates "no native route serves this
// shape" with "one does, but the heuristic preferred the vendor" -- e.g.
// syrk_use_cuda_custom is false both when syrk_triangular_supported fails and
// when all three preference tests simply do not fire. Recording either of
// those as a definite `false` would be a claim the call site cannot support,
// so it records kUnknown and the column reads -1.
enum : int { kNativeUnsupported = 0, kNativeSupported = 1, kNativeUnknown = -1 };
//
// Hardcoded F32/CUDA because these four dispatchers are float-only and reached
// only from the CUDA backend -- every entry point in
// {symm,syrk,syr2k,trmm}_custom_dispatch.hh takes MatrixView<float>. If that
// stops being true, this signature has to grow, and a caller that forgets will
// not compile.
// uplo/side/diag/transA are part of the coverage KEY, not decoration: they
// select which triangle or which operand the op touches, so two calls that
// differ only in `uplo` must not collapse into one row with whichever ran
// first deciding what the table reports. See variant_key() in coverage.cc.
struct Level3Variant {
    Uplo uplo = Uplo::Lower;
    Side side = Side::Left;
    Diag diag = Diag::NonUnit;
    Transpose transA = Transpose::NoTrans;
};

inline void record_level3_route(dispatch::Op op,
                                dispatch::Route taken,
                                int64_t m, int64_t n, int64_t k, int64_t batch,
                                int native_supported,
                                Level3Variant v = {}) {
    if (!dispatch::coverage::dynamic_enabled()) {
        return;
    }
    dispatch::OpShape s;
    s.op      = op;
    s.scalar  = dispatch::ScalarKind::F32;
    s.backend = Backend::CUDA;
    s.m = m;
    s.n = n;
    s.k = k;
    s.batch = batch;
    s.uplo   = v.uplo;
    s.side   = v.side;
    s.diag   = v.diag;
    s.transA = v.transA;
    dispatch::coverage::record(op, s.scalar, s.backend, s, taken,
                               /*native_existed=*/true, native_supported);
}

} // namespace batchlas::backend::detail
