#include <batchlas/blas/dispatch/coverage.hh>

#include <batchlas/blas/dispatch/route_compiled.hh>
#include <batchlas/blas/dispatch/vendor_available.hh>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>

#include <unistd.h>   // getpid; see emit()

namespace batchlas::dispatch::coverage {

// One definition in one TU, keyed on the same variable emit() reads.
bool g_dynamic_enabled = [] {
    const char* p = std::getenv("BATCHLAS_COVERAGE_OUT");
    return p != nullptr && *p != '\0';
}();

namespace {

struct Row {
    Op op{};
    ScalarKind scalar{};
    Backend backend{};
    OpShape shape{};
    Route chosen{};
    bool native_existed = false;
    int  native_supported = 0;   // tri-state; -1 = call site could not tell
    uint64_t calls = 0;
};

// Structural flags belong in the KEY; without them calls differing only in
// `uplo` collapse into one row. evidence: docs/perf/dispatch.md#instrument-defects
uint32_t variant_key(const OpShape& s) {
    return (static_cast<uint32_t>(s.uplo)   << 12) |
           (static_cast<uint32_t>(s.side)   <<  9) |
           (static_cast<uint32_t>(s.diag)   <<  6) |
           (static_cast<uint32_t>(s.transA) <<  3) |
           (static_cast<uint32_t>(s.transB));
}

// Keyed on shape_class, not the exact shape: it buckets max(m,n,k) and batch by
// power of two, so a 10,000-iteration test collapses to a handful of rows.
uint64_t key_of(Op op, ScalarKind s, Backend b, uint32_t shape_class,
                uint32_t variant = 0) {
    return (static_cast<uint64_t>(op) << 56) | (static_cast<uint64_t>(s) << 48) |
           (static_cast<uint64_t>(b) << 40) |
           (static_cast<uint64_t>(variant) << 24) | shape_class;
}

std::mutex& table_mutex() {
    static auto* m = new std::mutex();  // leaked; see table()
    return *m;
}

// DELIBERATELY LEAKED: emit() runs from atexit, and a function-local static
// constructed after the installer is destroyed before it runs -- which showed
// up not as a crash but as an output file with no `miss` rows at all.
std::unordered_map<uint64_t, Row>& table() {
    static auto* t = new std::unordered_map<uint64_t, Row>();
    return *t;
}

struct MissRow {
    Op op{};
    ScalarKind scalar{};
    Backend backend{};
    std::string library;
    uint64_t calls = 0;
};

std::unordered_map<uint64_t, MissRow>& misses() {
    static auto* t = new std::unordered_map<uint64_t, MissRow>();  // leaked; see table()
    return *t;
}

const char* backend_name(Backend b) {
    switch (b) {
        case Backend::CUDA:   return "CUDA";
        case Backend::ROCM:   return "ROCM";
        case Backend::NETLIB: return "NETLIB";
        case Backend::MKL:    return "MKL";
        // Expected, not a defect: adapters that build an OpShape leave `backend`
        // unset (it is a template parameter), so `reached` rows read AUTO.
        case Backend::AUTO:   return "AUTO";
        default:              return "?";
    }
}

void emit() {
    const char* path = std::getenv("BATCHLAS_COVERAGE_OUT");
    if (!path || !*path) {
        return;
    }

    // ONE FILE PER PROCESS: a shared path is truncated by the next process, and
    // appending tears lines under `ctest -j`. Merge with scripts/coverage_merge.sh.
    const std::string out = std::string(path) + "." + std::to_string(::getpid());

    // FILE* and no SYCL object: this runs from atexit, where any SYCL handle is
    // already a use-after-free risk.
    std::FILE* f = std::fopen(out.c_str(), "w");
    if (!f) {
        return;
    }

    std::fputs("kind,op,scalar,backend,shape_class,m,n,k,batch,"
               "chosen_origin,chosen_algo,calls,native_route_existed,"
               "native_route_supported,library,uplo,side,diag,transA,transB\n", f);

    std::lock_guard<std::mutex> lock(table_mutex());
    for (const auto& [k, r] : table()) {
        std::fprintf(f, "reached,%s,%s,%s,%u,%lld,%lld,%lld,%lld,%s,%s,%llu,%d,%d,,%d,%d,%d,%d,%d\n",
                     std::string(op_name(r.op)).c_str(),
                     std::string(to_string(r.scalar)).c_str(),
                     backend_name(r.backend),
                     r.shape.shape_class(),
                     static_cast<long long>(r.shape.m), static_cast<long long>(r.shape.n),
                     static_cast<long long>(r.shape.k), static_cast<long long>(r.shape.batch),
                     std::string(to_string(r.chosen.origin)).c_str(),
                     std::string(to_string(r.chosen.algo)).c_str(),
                     static_cast<unsigned long long>(r.calls),
                     r.native_existed ? 1 : 0, r.native_supported,
                     static_cast<int>(r.shape.uplo), static_cast<int>(r.shape.side),
                     static_cast<int>(r.shape.diag), static_cast<int>(r.shape.transA),
                     static_cast<int>(r.shape.transB));
    }
    for (const auto& [k, m] : misses()) {
        std::fprintf(f, "miss,%s,%s,%s,,,,,,,,%llu,0,0,%s\n",
                     std::string(op_name(m.op)).c_str(),
                     std::string(to_string(m.scalar)).c_str(),
                     backend_name(m.backend),
                     static_cast<unsigned long long>(m.calls),
                     m.library.c_str());
    }

    std::fputs(static_table().c_str(), f);
    std::fclose(f);
}

struct AtExitInstaller {
    AtExitInstaller() { std::atexit(&emit); }
};
AtExitInstaller installer;

// --- the static half --------------------------------------------------------

template <Backend B>
void append_static_rows(std::ostringstream& out) {
    struct Entry {
        const char* op;
        bool vendor;
        bool native;
    };
    // `native` means a kernel is LINKED in this build, not that traffic reaches
    // it; the `reached` rows answer that. Float is reported because tile-route
    // availability is per (backend, scalar).
    // evidence: docs/perf/dispatch.md#the-coverage-instrument
    const bool tiles_f32 = level3_tile_route_available<B, float>;
    const Entry entries[] = {
        {"gemm",  level3_vendor_available<B>,        true},
        {"gemv",  level3_vendor_available<B>,        true},
        {"trsm",  level3_vendor_available<B>,        false},
        {"trmm",  level3_vendor_available<B>,        tiles_f32},
        // symm has no tile kernel; its portable path is the mirrored expansion
        // (triangular_expand.hh) feeding a GEMM, which must itself be native.
        {"symm",  level3_vendor_available<B>,        tiles_f32},
        {"syrk",  level3_vendor_available<B>,        tiles_f32},
        {"syr2k", level3_vendor_available<B>,        tiles_f32},
        {"hemm",  level3_vendor_available<B>,        false},
        {"herk",  level3_vendor_available<B>,        false},
        {"her2k", level3_vendor_available<B>,        false},
        {"geqrf", factorization_vendor_available<B>, true},   // geqrf_cta + geqrf_blocked
        {"orgqr", factorization_vendor_available<B>, true},   // orgqr_blocked (ormqr on I)
        {"getrf", factorization_vendor_available<B>, true},   // getrf_cta + getrf_blocked
        {"getrs", factorization_vendor_available<B>, true},   // getrs_native (laswp + 2 routed trsm)
        {"getri", factorization_vendor_available<B>, true},   // getri_blocked (P into C + 2 routed trsm)
        {"ormqr", factorization_vendor_available<B>, true},   // ormqr_blocked
        {"potrf", solver_vendor_available<B>,        true},   // potrf_cta + potrf_blocked
        {"syev",  solver_vendor_available<B>,        true},   // cta/blocked/two_stage
        {"gesvd", solver_vendor_available<B>,        true},   // jacobi/cta/blocked
        {"spmm",  sparse_vendor_available<B>,        true},   // spmm_native_csr (gather + atomic scatter)
    };
    for (const auto& e : entries) {
        out << "linked," << e.op << ",float,"
            << backend_name(B) << ",,,,,,"
            << (e.native ? "native" : "-") << ","
            << (e.vendor ? "vendor" : "-") << ",,"
            << (e.native ? 1 : 0) << "," << (e.vendor ? 1 : 0) << ",\n";
    }
}

} // namespace

void record(Op op, ScalarKind scalar, Backend backend, const OpShape& shape,
            Route chosen, bool native_existed, int native_supported) {
    std::lock_guard<std::mutex> lock(table_mutex());
    auto& row = table()[key_of(op, scalar, backend, shape.shape_class(),
                               variant_key(shape))];
    if (row.calls == 0) {
        row.op = op;
        row.scalar = scalar;
        row.backend = backend;
        row.shape = shape;
        row.chosen = chosen;
        row.native_existed = native_existed;
        row.native_supported = native_supported;
    }
    ++row.calls;
}

void record_miss(Op op, ScalarKind scalar, Backend backend, const char* library) {
    std::lock_guard<std::mutex> lock(table_mutex());
    auto& row = misses()[key_of(op, scalar, backend, 0)];
    if (row.calls == 0) {
        row.op = op;
        row.scalar = scalar;
        row.backend = backend;
        row.library = library ? library : "";
    }
    ++row.calls;
}

std::string static_table() {
    std::ostringstream out;
#if BATCHLAS_HAS_CUDA_BACKEND
    append_static_rows<Backend::CUDA>(out);
#endif
#if BATCHLAS_HAS_ROCM_BACKEND
    append_static_rows<Backend::ROCM>(out);
#endif
#if BATCHLAS_HAS_HOST_BACKEND
    append_static_rows<Backend::NETLIB>(out);
#endif
    return out.str();
}

} // namespace batchlas::dispatch::coverage
