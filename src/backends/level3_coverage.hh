#pragma once

// Records the route a level-3 dispatcher ACTUALLY took; never a RouteTable (the
// thresholds are gate-only). Records sit beside a `return`, never in place of one.
// evidence: docs/perf/level3.md#level-3-why-the-four-ops-are-instrumented-rather-than-routed

#include <cstdint>

#include <batchlas/blas/dispatch/coverage.hh>
#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/enums.hh>

namespace batchlas::backend::detail {

// TRI-STATE: a declined gate cannot say whether a native route existed (-1).
// evidence: docs/perf/level3.md#the-coverage-instrument-itself
enum : int { kNativeUnsupported = 0, kNativeSupported = 1, kNativeUnknown = -1 };

// F32/CUDA hardcoded (all entry points take float). uplo/side/diag/transA are
// part of the coverage KEY (variant_key() in coverage.cc), not decoration.
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
