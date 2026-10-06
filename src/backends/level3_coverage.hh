#pragma once

// The coverage row for the route a level-3 dispatcher (symm, syrk, syr2k, trmm) took.
// They choose by rule, not select::choose, so each terminal records beside its return.
// evidence: docs/perf/level3.md#level-3-why-the-four-ops-are-instrumented-rather-than-routed

#include "../select/coverage.hh"

#include <batchlas/blas/enums.hh>

#include <cstdint>
#include <string_view>

namespace batchlas::backend::detail {

// Whether a native kernel could have served THIS shape; kNativeUnknown (-1) when the
// gate declined before the dispatcher ran, so "none" and "vendor preferred" look alike.
// evidence: docs/perf/level3.md#the-coverage-instrument-itself
enum : int { kNativeUnsupported = 0, kNativeSupported = 1, kNativeUnknown = -1 };

struct Level3Variant {  // part of the coverage key
    Uplo uplo = Uplo::Lower;
    Side side = Side::Left;
    Diag diag = Diag::NonUnit;
    Transpose transA = Transpose::NoTrans;
};

// `spelling` is the route taken: "vendor", "triangular", "gram", "expand" or
// "cublasdx". Float/CUDA: the four dispatchers take MatrixView<float> only.
inline void record_level3_route(Op op, const char* spelling,
                                int64_t m, int64_t n, int64_t k, int64_t batch,
                                int native_supported, Level3Variant v = {}) {
    if (!coverage::dynamic_enabled()) {
        return;
    }
    coverage::Shape s;
    s.op      = op;
    s.scalar  = ScalarKind::F32;
    s.backend = Backend::CUDA;
    s.m = m;
    s.n = n;
    s.k = k;
    s.batch = batch;
    s.uplo   = v.uplo;
    s.side   = v.side;
    s.diag   = v.diag;
    s.transA = v.transA;
    const std::string_view sp(spelling);
    const bool vendor = sp == "vendor" || sp == "cublasdx";  // MathDx counts as vendor
    coverage::record_choice(op, s.scalar, s.backend, s, vendor ? "vendor" : "native", spelling,
                            /*native_route_existed=*/true, native_supported);
}

} // namespace batchlas::backend::detail
