// SPDX-License-Identifier: MIT
#pragma once

/// @file
/// @brief What this build links and what a run reached, as CSV in BATCHLAS_COVERAGE_OUT.pid; the
/// columns are a contract with scripts/, tools/tune and benchviz. @ingroup api_selection
// evidence: docs/perf/dispatch.md#dispatch-the-coverage-instrument

#include <batchlas/export.hh>
#include <batchlas/no_route.hh>

#include <src/util/internal-api.hh>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace batchlas::coverage {  /** @addtogroup api_selection */ /** @{ */

struct Shape {  // the row key: different triangles or operands are different rows
    Op op = Op::COUNT;
    ScalarKind scalar = ScalarKind::F32;
    Backend backend = Backend::Auto;
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

/// A `reached` row: origin "native"/"vendor" and the choice spelling. native_route_supported
/// is 1, 0, or -1 when the call site cannot tell.
BATCHLAS_API void record_choice(Op op, ScalarKind scalar, Backend backend, const Shape& shape,
                                const char* origin, const char* spelling, bool native_route_existed,
                                int native_route_supported);

BATCHLAS_API void record_miss(  // a call that found nothing to run; always recorded
    Op op, ScalarKind scalar, Backend backend, const char* library);

BATCHLAS_API std::string static_table();  // the `linked` rows: what this build contains

/// Set once from $BATCHLAS_COVERAGE_OUT; tests toggle it across the DSO boundary (a runtime bool in one TU).
BATCHLAS_API extern bool g_dynamic_enabled;

inline bool dynamic_enabled() { return g_dynamic_enabled; }

/// A test-only capture: snapshot (the tables moved aside; the run records into empty ones), run,
/// rows_since, restore (once).
struct Snapshot {
    struct Tables;
    std::shared_ptr<Tables> saved;
    bool enabled = false;
    bool keep = false;  // merge the capture's calls back on restore; snapshot() sets it to `enabled`
    Snapshot() = default;
    Snapshot(const Snapshot&) = delete;
    Snapshot& operator=(const Snapshot&) = delete;
    Snapshot(Snapshot&&) = default;
    Snapshot& operator=(Snapshot&&) = default;
};

BATCHLAS_INTERNAL_API Snapshot snapshot();
BATCHLAS_INTERNAL_API std::vector<std::string> rows_since(const Snapshot& s);  // emit()'s lines of the open capture
/// The saved tables and switch back; with `s.keep` the capture's calls merge into them (counts add,
/// new rows insert), so BATCHLAS_COVERAGE_OUT keeps them, else they are dropped.
BATCHLAS_INTERNAL_API void restore(Snapshot& s);

/** @} */ } // namespace batchlas::coverage
