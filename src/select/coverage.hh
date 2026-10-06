#pragma once

// What this build links and what a run reached, as CSV in $BATCHLAS_COVERAGE_OUT.<pid>; the
// columns are a contract with scripts/, tools/tune and benchviz. The gate is a runtime bool in
// one TU. evidence: docs/perf/dispatch.md#the-coverage-instrument

#include <batchlas/export.hh>
#include <batchlas/no_route.hh>

#include <cstdint>
#include <string>

namespace batchlas::coverage {

struct Shape {  // the row key: different triangles or operands are different rows
    Op op = Op::COUNT;
    ScalarKind scalar = ScalarKind::F32;
    Backend backend = Backend::AUTO;
    int64_t m = 0, n = 0, k = 0;  // square ops set all three
    int64_t batch = 1;
    Transpose transA = Transpose::NoTrans;
    Transpose transB = Transpose::NoTrans;
    Uplo uplo = Uplo::Lower;
    Side side = Side::Left;
    Diag diag = Diag::NonUnit;

    int64_t max_dim() const { return m > n ? (m > k ? m : k) : (n > k ? n : k); }

    uint32_t shape_class() const {  // power-of-two buckets on max(m,n,k) and batch
        auto log2b = [](int64_t v) -> uint32_t {
            uint32_t r = 0;
            while (v > 1) { v >>= 1; ++r; }
            return r;
        };
        return (log2b(max_dim()) << 8) | log2b(batch);
    }
};

// A `reached` row: origin "native"/"vendor" and the choice spelling. native_route_supported
// is 1, 0, or -1 when the call site cannot tell.
BATCHLAS_API void record_choice(Op op, ScalarKind scalar, Backend backend, const Shape& shape,
                                const char* origin, const char* spelling, bool native_route_existed,
                                int native_route_supported);

BATCHLAS_API void record_miss(  // a call that found nothing to run; always recorded
    Op op, ScalarKind scalar, Backend backend, const char* library);

BATCHLAS_API std::string static_table();  // the `linked` rows: what this build contains

// Set once from $BATCHLAS_COVERAGE_OUT; tests toggle it across the DSO boundary.
BATCHLAS_API extern bool g_dynamic_enabled;

inline bool dynamic_enabled() { return g_dynamic_enabled; }

} // namespace batchlas::coverage
