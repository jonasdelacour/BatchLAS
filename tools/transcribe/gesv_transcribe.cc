// gesv transcriber (flat-kernel-selection-phase3-plan.md §2 option D). Evaluates the OLD router,
// RouteTable<Op::gesv,T> + resolve_route, at every grid cell of src/ops/gesv/choice.hh and prints
// its preference order as a transcriber CSV for `scripts/sweep_to_table.py --transcribe`.
//
// Host-only, built against the headers of a tree that still has route_gesv.hh (424a45bc, the
// sha in the tables' headers; OLD = that checkout, configured once for build/include):
//   g++ -std=c++20 -I$OLD/include -I$OLD/build/include tools/transcribe/gesv_transcribe.cc -o /tmp/gt
//   /tmp/gt sm_89 > tuned/transcribed/gesv.sm_89.csv
//   /tmp/gt sm_120 > tuned/transcribed/gesv.sm_120.csv
// The old predicates read no architecture, so both devices get the same rows.
//
// The order is DERIVED, not written down: resolve the cell, record the winner, mark it unsupported
// and resolve again, until the walk returns no native route. Capacities (the tiny order and nrhs
// ceilings) are unlimited because can_run re-applies them at run time; the device is a CUDA GPU
// with sub-group 32. The list stops after `blocked`, which can_run never refuses.
//
// `--points FILE` (lines "dtype,n,nrhs") instead prints the old router's FIRST choice with the
// real capacities applied, for the off-grid data gate (scripts/gesv_offgrid_gate.py).

#include <batchlas/blas/dispatch/route_gesv.hh>

#include <array>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

namespace batchlas::dispatch {

inline constexpr Op kExcluding = static_cast<Op>(250);  // a private Op value, never a real op
inline thread_local unsigned g_excluded = 0;            // bit per kGesvOrder index

inline unsigned bit_of(Route r) {
    for (unsigned i = 0; i < std::size(kGesvOrder); ++i)
        if (kGesvOrder[i] == r) return 1u << i;
    return 0;
}

// The real table with the already-ranked routes removed from supports().
template <typename T>
struct RouteTable<kExcluding, T> {
    using Real = RouteTable<Op::gesv, T>;
    static bool supports(Route r, const GesvShape& s) { return !(g_excluded & bit_of(r)) && Real::supports(r, s); }
    static bool preferred(Route r, const GesvShape& s) { return Real::preferred(r, s); }
    static bool native_tier_preferred(Route r, const GesvShape& s) { return Real::native_tier_preferred(r, s); }
    static constexpr const Route* order_begin() { return Real::order_begin(); }
    static constexpr const Route* order_end() { return Real::order_end(); }
};

}  // namespace batchlas::dispatch

namespace {

using namespace batchlas;
using namespace batchlas::dispatch;

// src/ops/gesv/choice.hh's grid: both sides of the old window edges (16|17 cfloat, 32|33 float)
// and of the tiny ceilings (16|17 cdouble, 32|33 otherwise; nrhs 4|5), log-spaced elsewhere.
constexpr std::array<int, 29> kGridN{1, 2, 3, 4, 6, 8, 12, 16, 17, 20, 24, 28, 32, 33, 40, 48, 64,
                                     96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096};
constexpr std::array<int, 8> kGridNrhs{1, 2, 4, 5, 8, 16, 64, 256};

// The tiny tier's real ceilings (gesv_tiny.cc tiny_cap, solve_native.hh kGesvTinyMaxRhs).
template <typename T>
constexpr int64_t tiny_cap() {
    return std::is_same_v<T, std::complex<double>> ? 16 : 32;
}

std::string spelling(Route r) {
    switch (r.algo) {
        case Algorithm::Tiny: return "tiny";
        case Algorithm::Blocked: return "blocked";
        default: return "?" + std::string(to_string(r.algo));
    }
}

template <typename T>
GesvShape shape(int n, int nrhs, bool real_caps) {
    GesvShape s;
    s.op = Op::gesv;
    s.scalar = scalar_kind_of<T>;
    s.backend = Backend::CUDA;
    s.m = s.k = n;
    s.n = nrhs;
    s.batch = 1024;
    s.is_gpu = true;
    s.has_sg32 = true;
    s.tiny_max_n = real_caps ? tiny_cap<T>() : INT64_MAX / 4;
    s.tiny_max_nrhs = real_caps ? 4 : INT64_MAX / 4;
    s.composed_available = true;
    return s;
}

template <typename T>
std::string ranked(int n, int nrhs) {
    const GesvShape s = shape<T>(n, nrhs, false);
    std::string out;
    g_excluded = 0;
    for (;;) {
        const Route r = resolve_route_uninstrumented<kExcluding, T>(Route{Origin::Auto, Algorithm::Auto}, s, false);
        if (!is_native(r)) break;
        out += (out.empty() ? "" : "|") + spelling(r);
        if (r.algo == Algorithm::Blocked) break;
        g_excluded |= bit_of(r);
    }
    return out;
}

template <typename T>
void emit(const char* dtype, const char* device) {
    for (int n : kGridN)
        for (int nrhs : kGridNrhs)
            std::printf("gesv,%s,%s,%d,%d,%s\n", dtype, device, n, nrhs, ranked<T>(n, nrhs).c_str());
}

template <typename T>
std::string first_choice(int n, int nrhs) {
    // resolve_gesv_route minus its coverage hook, which would need the library.
    return spelling(resolve_route_uninstrumented<Op::gesv, T>(Route{Origin::Auto, Algorithm::Auto},
                                                              shape<T>(n, nrhs, true), false));
}

int points(const char* path) {
    FILE* f = std::fopen(path, "r");
    if (!f) return 2;
    char dt[16];
    int n = 0, nrhs = 0;
    std::printf("dtype,n,nrhs,old\n");
    while (std::fscanf(f, " %15[a-z],%d,%d", dt, &n, &nrhs) == 3) {
        std::string c;
        if (!std::strcmp(dt, "float")) c = first_choice<float>(n, nrhs);
        else if (!std::strcmp(dt, "double")) c = first_choice<double>(n, nrhs);
        else if (!std::strcmp(dt, "cfloat")) c = first_choice<std::complex<float>>(n, nrhs);
        else if (!std::strcmp(dt, "cdouble")) c = first_choice<std::complex<double>>(n, nrhs);
        else return 3;
        std::printf("%s,%d,%d,%s\n", dt, n, nrhs, c.c_str());
    }
    std::fclose(f);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 2 && !std::strcmp(argv[1], "--points")) return points(argv[2]);
    const char* device = argc > 1 ? argv[1] : "sm_89";
    std::printf("op,dtype,device,n,nrhs,ranked\n");
    emit<float>("float", device);
    emit<double>("double", device);
    emit<std::complex<float>>("cfloat", device);
    emit<std::complex<double>>("cdouble", device);
    return 0;
}
