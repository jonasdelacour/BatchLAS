// SPDX-License-Identifier: MIT
#include <src/select/coverage.hh>

#include <src/select/vendor.hh>

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <unistd.h>   // getpid; see emit()
#include <batchlas/settings.hh>

namespace batchlas::coverage {

// Runs at static init, before any configure() call; settings() is safe there.
bool g_dynamic_enabled = [] {
    const char* p = batchlas::settings().diagnostics.coverage_out.get();
    return p != nullptr && *p != '\0';
}();

namespace {

struct Row {
    Op op{};
    ScalarKind scalar{};
    Backend backend{};
    Shape shape{};
    bool native_existed = false;
    int  native_supported = 0;   // tri-state; -1 = call site could not tell
    uint64_t calls = 0;
    std::string origin;
    std::string algo;
};

// Structural flags belong in the KEY; without them calls differing only in
// `uplo` collapse into one row. evidence: docs/perf/dispatch.md#instrument-defects
uint32_t variant_key(const Shape& s) {
    return (static_cast<uint32_t>(s.uplo)   << 12) |
           (static_cast<uint32_t>(s.side)   <<  9) |
           (static_cast<uint32_t>(s.diag)   <<  6) |
           (static_cast<uint32_t>(s.transA) <<  3) |
           (static_cast<uint32_t>(s.transB));
}

uint64_t key_of(Op op, ScalarKind s, Backend b, uint32_t shape_class,
                uint32_t variant = 0) {
    return (static_cast<uint64_t>(op) << 56) | (static_cast<uint64_t>(s) << 48) |
           (static_cast<uint64_t>(b) << 40) |
           (static_cast<uint64_t>(variant) << 24) | shape_class;
}

std::mutex& table_mutex() {
    static auto* m = new std::mutex();  // leaked; see choice_table()
    return *m;
}

// DELIBERATELY LEAKED: emit() runs from atexit, and a function-local static
// constructed after the installer is destroyed before it runs -- which showed
// up not as a crash but as an output file with no `miss` rows at all.
// Keyed by text too: one shape class can reach several spellings of one family.
std::unordered_map<std::string, Row>& choice_table() {
    static auto* t = new std::unordered_map<std::string, Row>();
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
    static auto* t = new std::unordered_map<uint64_t, MissRow>();  // leaked; see choice_table()
    return *t;
}

const char* backend_name(Backend b) {
    switch (b) {
        case Backend::CUDA:   return "CUDA";
        case Backend::ROCM:   return "ROCM";
        case Backend::SYCL:   return "SYCL";
        case Backend::NETLIB: return "NETLIB";
        case Backend::MKL:    return "MKL";
        case Backend::Auto:   return "AUTO";
        default:              return "?";
    }
}

// One `reached` CSV line; emit() and rows_since() share it, so a capture reads what a file holds.
std::string reached_line(const Row& r, uint64_t calls) {
    char buf[512];
    std::snprintf(buf, sizeof buf, "reached,%s,%s,%s,%u,%lld,%lld,%lld,%lld,%s,%s,%llu,%d,%d,,%d,%d,%d,%d,%d\n",
                  std::string(op_name(r.op)).c_str(),
                  std::string(to_string(r.scalar)).c_str(),
                  backend_name(r.backend),
                  r.shape.shape_class(),
                  static_cast<long long>(r.shape.m), static_cast<long long>(r.shape.n),
                  static_cast<long long>(r.shape.k), static_cast<long long>(r.shape.batch),
                  r.origin.c_str(), r.algo.c_str(),
                  static_cast<unsigned long long>(calls),
                  r.native_existed ? 1 : 0, r.native_supported,
                  static_cast<int>(r.shape.uplo), static_cast<int>(r.shape.side),
                  static_cast<int>(r.shape.diag), static_cast<int>(r.shape.transA),
                  static_cast<int>(r.shape.transB));
    return buf;
}

std::string miss_line(const MissRow& m, uint64_t calls) {
    char buf[256];
    std::snprintf(buf, sizeof buf, "miss,%s,%s,%s,,,,,,,,%llu,0,0,%s\n", std::string(op_name(m.op)).c_str(),
                  std::string(to_string(m.scalar)).c_str(), backend_name(m.backend),
                  static_cast<unsigned long long>(calls), m.library.c_str());
    return buf;
}

void emit() {
    const char* path = batchlas::settings().diagnostics.coverage_out.get();
    if (!path || !*path) {
        return;
    }

    // ONE FILE PER PROCESS: a shared path is truncated by the next process, and
    // appending tears lines under `ctest -j`. Merge with scripts/coverage_merge.sh.
    const std::string out = std::string(path) + "." + std::to_string(::getpid());

    // FILE* and no SYCL object: this runs from atexit.
    std::FILE* f = std::fopen(out.c_str(), "w");
    if (!f) {
        return;
    }

    std::fputs("kind,op,scalar,backend,shape_class,m,n,k,batch,"
               "chosen_origin,chosen_algo,calls,native_route_existed,"
               "native_route_supported,library,uplo,side,diag,transA,transB\n", f);

    std::lock_guard<std::mutex> lock(table_mutex());
    for (const auto& [k, r] : choice_table()) std::fputs(reached_line(r, r.calls).c_str(), f);
    for (const auto& [k, m] : misses()) std::fputs(miss_line(m, m.calls).c_str(), f);

    std::fputs(static_table().c_str(), f);
    std::fclose(f);
}

struct AtExitInstaller {
    AtExitInstaller() { std::atexit(&emit); }
};
AtExitInstaller installer;

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
    // evidence: docs/perf/dispatch.md#dispatch-the-coverage-instrument
    using namespace batchlas::select;
    const bool tiles_f32 = level3_tile_substitution_enabled<B, float>;
    const bool gemm_folds = native_gpu_backend<B> || B == Backend::ROCM;
    const bool hermitian_vendor = level3_vendor_available<B> && B != Backend::ROCM;  // rocblas.cc wraps none
    const Entry entries[] = {
        {"gemm",  level3_vendor_available<B>,        true},
        {"gemv",  level3_vendor_available<B>,        true},
        {"trsm",  level3_vendor_available<B>,        true},
        {"trmm",  level3_vendor_available<B>,        tiles_f32},
        // symm's portable path is the mirrored expansion feeding the public gemm.
        {"symm",  level3_vendor_available<B>,        tiles_f32},
        {"syrk",  level3_vendor_available<B>,        tiles_f32},
        {"syr2k", level3_vendor_available<B>,        tiles_f32},
        // hemm's expand and herk's / her2k's fold run over the public gemm, so on ROCm too.
        {"hemm",  hermitian_vendor,                  gemm_folds},
        {"herk",  hermitian_vendor,                  gemm_folds},
        {"her2k", hermitian_vendor,                  gemm_folds},
        {"geqrf", factorization_vendor_available<B>, true},
        {"orgqr", factorization_vendor_available<B>, true},
        {"getrf", factorization_vendor_available<B>, true},
        {"getrs", factorization_vendor_available<B>, true},
        {"getri", factorization_vendor_available<B>, true},
        {"ormqr", factorization_vendor_available<B>, true},
        {"potrf", solver_vendor_available<B>,        true},
        {"syev",  solver_vendor_available<B>,        true},
        {"gesvd", solver_vendor_available<B>,        true},
        {"spmm",  sparse_vendor_available<B>,        true},
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

void record_choice(Op op, ScalarKind scalar, Backend backend, const Shape& shape,
                   const char* origin, const char* spelling, bool native_existed, int native_supported) {
    const std::string o = origin ? origin : "";
    const std::string a = spelling ? spelling : "";
    const std::string key = std::to_string(key_of(op, scalar, backend, shape.shape_class(),
                                                  variant_key(shape))) + "|" + o + "|" + a;
    std::lock_guard<std::mutex> lock(table_mutex());
    auto& row = choice_table()[key];
    if (row.calls == 0) {
        row.op = op;
        row.scalar = scalar;
        row.backend = backend;
        row.shape = shape;
        row.native_existed = native_existed;
        row.native_supported = native_supported;
        row.origin = o;
        row.algo = a;
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

Snapshot snapshot() {
    Snapshot s;
    s.enabled = g_dynamic_enabled;
    std::lock_guard<std::mutex> lock(table_mutex());
    for (const auto& [k, r] : choice_table()) s.calls.emplace(k, r.calls);
    for (const auto& [k, m] : misses()) s.misses.emplace(k, m.calls);
    return s;
}

std::vector<std::string> rows_since(const Snapshot& s) {
    std::vector<std::string> out;
    std::lock_guard<std::mutex> lock(table_mutex());
    for (const auto& [k, r] : choice_table()) {
        const auto it = s.calls.find(k);
        const uint64_t before = it == s.calls.end() ? 0 : it->second;
        if (r.calls > before) out.push_back(reached_line(r, r.calls - before));
    }
    for (const auto& [k, m] : misses()) {
        const auto it = s.misses.find(k);
        const uint64_t before = it == s.misses.end() ? 0 : it->second;
        if (m.calls > before) out.push_back(miss_line(m, m.calls - before));
    }
    return out;
}

namespace {
template <typename Table, typename Calls>
void rewind(Table& t, const Calls& was_calls) {
    for (auto it = t.begin(); it != t.end();) {
        const auto was = was_calls.find(it->first);
        if (was == was_calls.end()) {
            it = t.erase(it);
        } else {
            it->second.calls = was->second;
            ++it;
        }
    }
}
}  // namespace

void restore(const Snapshot& s) {
    {
        std::lock_guard<std::mutex> lock(table_mutex());
        rewind(choice_table(), s.calls);
        rewind(misses(), s.misses);
    }
    g_dynamic_enabled = s.enabled;
}

std::string static_table() {
    std::ostringstream out;
#if BATCHLAS_HAS_CUDA_BACKEND
    append_static_rows<Backend::CUDA>(out);
#endif
#if BATCHLAS_HAS_ROCM_BACKEND
    append_static_rows<Backend::ROCM>(out);
#endif
#if BATCHLAS_HAS_SYCL_BACKEND
    append_static_rows<Backend::SYCL>(out);
#endif
#if BATCHLAS_HAS_HOST_BACKEND
    append_static_rows<Backend::NETLIB>(out);
#endif
    return out.str();
}

} // namespace batchlas::coverage
