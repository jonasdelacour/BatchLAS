#include <batchlas/blas/dispatch/coverage.hh>

#include <batchlas/blas/dispatch/route_compiled.hh>
#include <batchlas/blas/dispatch/vendor_available.hh>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <sstream>
#include <unordered_map>

namespace batchlas::dispatch::coverage {

namespace {

struct Row {
    Op op{};
    ScalarKind scalar{};
    Backend backend{};
    OpShape shape{};
    Route chosen{};
    bool native_existed = false;
    bool native_supported = false;
    uint64_t calls = 0;
};

// Keyed on shape_class, not on the exact shape: shape_class() buckets
// max(m,n,k) and batch by power of two, so a 10,000-iteration test collapses to
// a handful of rows instead of 10,000.
uint64_t key_of(Op op, ScalarKind s, Backend b, uint32_t shape_class) {
    return (static_cast<uint64_t>(op) << 48) | (static_cast<uint64_t>(s) << 40) |
           (static_cast<uint64_t>(b) << 32) | shape_class;
}

std::mutex& table_mutex() {
    static auto* m = new std::mutex();  // leaked; see table()
    return *m;
}

// DELIBERATELY LEAKED, and this is not laziness.
//
// The atexit handler below reads these maps. atexit handlers and static
// destructors are interleaved in reverse registration order, so a
// function-local static map constructed AFTER the installer is DESTROYED
// BEFORE emit() runs -- and emit() then walks a dead container. That is
// undefined behaviour, and the way it actually presented was worse than a
// crash: the file was written, with a correct header and correct static rows,
// and simply no `miss` rows at all. A silently empty measurement is the one
// failure mode a coverage tool must not have.
//
// Never destroying them removes the ordering question entirely. The memory is
// reclaimed by process exit, which is when this runs anyway.
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
        default:              return "?";
    }
}

void emit() {
    const char* path = std::getenv("BATCHLAS_COVERAGE_OUT");
    if (!path || !*path) {
        return;
    }
    // Deliberately FILE* and no SYCL object: this runs from atexit, where any
    // SYCL handle is already a use-after-free risk. See the standing rule in
    // this tree about static destruction.
    std::FILE* f = std::fopen(path, "w");
    if (!f) {
        return;
    }

    std::fputs("kind,op,scalar,backend,shape_class,m,n,k,batch,"
               "chosen_origin,chosen_algo,calls,native_route_existed,"
               "native_route_supported,library\n", f);

    std::lock_guard<std::mutex> lock(table_mutex());
    for (const auto& [k, r] : table()) {
        std::fprintf(f, "reached,%s,%s,%s,%u,%lld,%lld,%lld,%lld,%s,%s,%llu,%d,%d,\n",
                     std::string(op_name(r.op)).c_str(),
                     std::string(to_string(r.scalar)).c_str(),
                     backend_name(r.backend),
                     r.shape.shape_class(),
                     static_cast<long long>(r.shape.m), static_cast<long long>(r.shape.n),
                     static_cast<long long>(r.shape.k), static_cast<long long>(r.shape.batch),
                     std::string(to_string(r.chosen.origin)).c_str(),
                     std::string(to_string(r.chosen.algo)).c_str(),
                     static_cast<unsigned long long>(r.calls),
                     r.native_existed ? 1 : 0, r.native_supported ? 1 : 0);
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
    // `native` here is "BatchLAS has a kernel for this op that is LINKED in this
    // build", which is what the burn-down needs to know. gemm's register-tiled
    // family lives in the vendor-free batchlas_sycl component, so it is always
    // present; the four level-3 tile routes are still gated with cuBLAS (WP1);
    // everything else has no native kernel yet at all.
    const bool tiles = level3_tile_kernels_compiled<B>;
    const Entry entries[] = {
        {"gemm",  level3_vendor_available<B>,        true},
        {"gemv",  level3_vendor_available<B>,        false},
        {"trsm",  level3_vendor_available<B>,        false},
        {"trmm",  level3_vendor_available<B>,        tiles},
        {"symm",  level3_vendor_available<B>,        tiles},
        {"syrk",  level3_vendor_available<B>,        tiles},
        {"syr2k", level3_vendor_available<B>,        tiles},
        {"hemm",  level3_vendor_available<B>,        false},
        {"herk",  level3_vendor_available<B>,        false},
        {"her2k", level3_vendor_available<B>,        false},
        {"geqrf", factorization_vendor_available<B>, false},
        {"orgqr", factorization_vendor_available<B>, false},
        {"getrf", factorization_vendor_available<B>, false},
        {"getrs", factorization_vendor_available<B>, false},
        {"getri", factorization_vendor_available<B>, false},
        {"ormqr", factorization_vendor_available<B>, true},   // ormqr_blocked
        {"potrf", solver_vendor_available<B>,        false},
        {"syev",  solver_vendor_available<B>,        true},   // cta/blocked/two_stage
        {"gesvd", solver_vendor_available<B>,        true},   // jacobi/cta/blocked
        {"spmm",  sparse_vendor_available<B>,        false},
    };
    for (const auto& e : entries) {
        out << "linked," << e.op << ",,"
            << backend_name(B) << ",,,,,,"
            << (e.native ? "native" : "-") << ","
            << (e.vendor ? "vendor" : "-") << ",,"
            << (e.native ? 1 : 0) << "," << (e.vendor ? 1 : 0) << ",\n";
    }
}

} // namespace

void record(Op op, ScalarKind scalar, Backend backend, const OpShape& shape,
            Route chosen, bool native_existed, bool native_supported) {
    std::lock_guard<std::mutex> lock(table_mutex());
    auto& row = table()[key_of(op, scalar, backend, shape.shape_class())];
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
