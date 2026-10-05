// getrs transcriber (flat-kernel-selection.md §13: tables are the transcribed old routing).
// Evaluates the OLD router, RouteTable<Op::getrs,T> + resolve_route, at every grid cell of
// src/ops/getrs/choice.hh and prints its preference order as a transcriber CSV for
// `scripts/sweep_to_table.py --transcribe`.
//
// Host-only, built against the headers of a tree that still has route_getrs.hh (424a45bc, the
// sha in the tables' headers; OLD = that checkout, configured once for build/include):
//   g++ -std=c++20 -I$OLD/include -I$OLD/build/include tools/transcribe/getrs_transcribe.cc -o /tmp/gt
//   /tmp/gt sm_89 sm_120 > tuned/transcribed/getrs.csv
//   /tmp/gt --replay < points   # old Auto decisions for the off-grid data gate (replay() below)
//
// The order is DERIVED, not written down: resolve the cell with the vendor present, record the
// winner, mark it unsupported and resolve again. Once the walk returns the vendor, the remaining
// natives are ranked by the vendor-free walk. Both capacities are unlimited: the fused tier's
// n * nrhs ceiling is a device fact and its nrhs ceiling (kGetrsFusedMaxRhs) a build fact, and
// both stay in can_run at run time. The old predicates read no architecture, so every device
// argument gets the same rows.

#include <batchlas/blas/dispatch/route_getrs.hh>

#include <array>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

namespace batchlas::dispatch {

inline constexpr Op kExcluding = static_cast<Op>(250);  // a private Op value, never a real op
inline thread_local unsigned g_excluded = 0;            // bit per kGetrsOrder index

inline unsigned bit_of(Route r) {
    for (unsigned i = 0; i < std::size(kGetrsOrder); ++i)
        if (kGetrsOrder[i] == r) return 1u << i;
    return 0;
}

// The real table with the already-ranked routes removed from supports().
template <typename T>
struct RouteTable<kExcluding, T> {
    using Real = RouteTable<Op::getrs, T>;
    static bool supports(Route r, const GetrsShape& s) { return !(g_excluded & bit_of(r)) && Real::supports(r, s); }
    static bool preferred(Route r, const GetrsShape& s) { return Real::preferred(r, s); }
    static bool native_tier_preferred(Route r, const GetrsShape& s) { return Real::native_tier_preferred(r, s); }
    static constexpr const Route* order_begin() { return Real::order_begin(); }
    static constexpr const Route* order_end() { return Real::order_end(); }
};

}  // namespace batchlas::dispatch

namespace {

using namespace batchlas;
using namespace batchlas::dispatch;

// src/ops/getrs/choice.hh's grid.
constexpr std::array<int, 18> kGridN{1, 2, 4, 8, 16, 24, 31, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024};
constexpr std::array<int, 15> kGridNrhs{1, 2, 3, 4, 5, 8, 16, 32, 63, 64, 127, 128, 256, 512, 1024};
constexpr std::array<int, 8> kGridBatch{1, 16, 127, 128, 512, 2048, 8192, 32768};

std::string spelling(Route r) {
    if (is_vendor(r)) return "vendor";
    switch (r.algo) {
        case Algorithm::CTA: return "cta";
        case Algorithm::Blocked: return "blocked";
        default: return "?" + std::string(to_string(r.algo));
    }
}

template <typename T>
GetrsShape shape_of(std::int64_t n, std::int64_t nrhs, std::int64_t batch) {
    GetrsShape s;
    s.op = Op::getrs;
    s.scalar = scalar_kind_of<T>;
    s.backend = Backend::CUDA;
    s.m = n;  // as getrs_op_shape built it: m = k = order, n = nrhs
    s.n = nrhs;
    s.k = n;
    s.batch = batch;
    s.transA = Transpose::NoTrans;
    s.is_gpu = true;
    s.has_sg32 = true;
    s.blocked_available = true;
    s.fused_max_elems = std::numeric_limits<std::int64_t>::max();
    s.fused_max_nrhs = std::numeric_limits<std::int64_t>::max();
    return s;
}

template <typename T>
std::string ranked(std::int64_t n, std::int64_t nrhs, std::int64_t batch) {
    const GetrsShape s = shape_of<T>(n, nrhs, batch);
    std::string out;
    bool vendor_ranked = false;
    g_excluded = 0;
    for (;;) {
        const Route r =
            resolve_route_uninstrumented<kExcluding, T>(Route{Origin::Auto, Algorithm::Auto}, s, !vendor_ranked);
        if (is_vendor(r) && vendor_ranked) break;
        out += (out.empty() ? "" : "|") + spelling(r);
        if (is_vendor(r)) vendor_ranked = true;
        else g_excluded |= bit_of(r);
    }
    return out;
}

// The old Auto decision for one cell with the real capacities applied (the replay oracle).
template <typename T>
std::string decide(std::int64_t n, std::int64_t nrhs, std::int64_t batch, bool vendor, std::int64_t max_elems,
                   std::int64_t max_nrhs) {
    GetrsShape s = shape_of<T>(n, nrhs, batch);
    s.fused_max_elems = max_elems;
    s.fused_max_nrhs = max_nrhs;
    g_excluded = 0;
    return spelling(resolve_route_uninstrumented<Op::getrs, T>(Route{Origin::Auto, Algorithm::Auto}, s, vendor));
}

template <typename T>
void emit(const char* dtype, const char* device) {
    for (int n : kGridN)
        for (int nrhs : kGridNrhs)
            for (int batch : kGridBatch)
                std::printf("getrs,%s,%s,%d,%d,%d,%s\n", dtype, device, n, nrhs, batch,
                            ranked<T>(n, nrhs, batch).c_str());
}

// stdin lines "<dtype> <n> <nrhs> <batch> <vendor 0|1> <max_elems> <max_nrhs>" -> one old Auto
// decision per line on stdout.
int replay() {
    char dt[16];
    long long n, nrhs, batch, vendor, me, mr;
    while (std::scanf("%15s %lld %lld %lld %lld %lld %lld", dt, &n, &nrhs, &batch, &vendor, &me, &mr) == 7) {
        const std::string d = dt;
        std::string r;
        if (d == "float") r = decide<float>(n, nrhs, batch, vendor != 0, me, mr);
        else if (d == "double") r = decide<double>(n, nrhs, batch, vendor != 0, me, mr);
        else if (d == "cfloat") r = decide<std::complex<float>>(n, nrhs, batch, vendor != 0, me, mr);
        else if (d == "cdouble") r = decide<std::complex<double>>(n, nrhs, batch, vendor != 0, me, mr);
        else return 2;
        std::printf("%s\n", r.c_str());
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--replay") == 0) return replay();
    std::printf("op,dtype,device,n,nrhs,batch,ranked\n");
    for (int i = 1; i < argc; ++i) {
        emit<float>("float", argv[i]);
        emit<double>("double", argv[i]);
        emit<std::complex<float>>("cfloat", argv[i]);
        emit<std::complex<double>>("cdouble", argv[i]);
    }
    return 0;
}
