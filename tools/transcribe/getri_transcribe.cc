// getri transcriber (flat-kernel-selection-phase3-plan.md §2 option D). Evaluates the OLD router,
// RouteTable<Op::getri,T> + resolve_route, at every grid cell of src/ops/getri/choice.hh and prints
// its preference order as a transcriber CSV for `scripts/sweep_to_table.py --transcribe`.
//
// Host-only, built against the headers of a tree that still has route_getri.hh (424a45bc, the
// sha in the tables' headers; OLD = that checkout, configured once for build/include):
//   g++ -std=c++20 -I$OLD/include -I$OLD/build/include tools/transcribe/getri_transcribe.cc -o /tmp/gt
//   /tmp/gt sm_89 > tuned/transcribed/getri.sm_89.csv
//   /tmp/gt sm_120 > tuned/transcribed/getri.sm_120.csv
//   /tmp/gt --offgrid 4000 7 > points.csv   # the data gate's random off-grid probes
//
// The old predicates read no architecture, so both devices get the same rows. The order is
// DERIVED, not written down: resolve with the vendor present, record the winner, mark it
// unsupported and resolve again; once the vendor is ranked, the remaining native is ranked by the
// vendor-free walk. getri has no capacity term (supports() reads only structure), so nothing is
// capped here. preferred() reads the order alone (float >= 128, cfloat >= 256), so the grid puts
// a point on both sides of each threshold; batch is a key for the retune, the old router never
// read it beyond batch >= 1.

#include <batchlas/blas/dispatch/route_getri.hh>

#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>

namespace batchlas::dispatch {

inline constexpr Op kExcluding = static_cast<Op>(250);  // a private Op value, never a real op
inline thread_local unsigned g_excluded = 0;            // bit per kGetriOrder index

inline unsigned bit_of(Route r) {
    for (unsigned i = 0; i < std::size(kGetriOrder); ++i)
        if (kGetriOrder[i] == r) return 1u << i;
    return 0;
}

// The real table with the already-ranked routes removed from supports().
template <typename T>
struct RouteTable<kExcluding, T> {
    using Real = RouteTable<Op::getri, T>;
    static bool supports(Route r, const GetriShape& s) { return !(g_excluded & bit_of(r)) && Real::supports(r, s); }
    static bool preferred(Route r, const GetriShape& s) { return Real::preferred(r, s); }
    static constexpr const Route* order_begin() { return Real::order_begin(); }
    static constexpr const Route* order_end() { return Real::order_end(); }
};

}  // namespace batchlas::dispatch

namespace {

using namespace batchlas;
using namespace batchlas::dispatch;

// src/ops/getri/choice.hh's grid.
constexpr std::array<int, 24> kGridN{1,   2,   3,   4,   6,   8,   12,  16,  24,  32,   48,   64,
                                     96,  127, 128, 192, 255, 256, 384, 512, 768, 1024, 2048, 4096};
constexpr std::array<int, 5> kGridBatch{128, 512, 2048, 8192, 32768};

std::string spelling(Route r) {
    if (is_vendor(r)) return "vendor";
    if (r.algo == Algorithm::Blocked) return "blocked";
    return "?" + std::string(to_string(r.algo));
}

template <typename T>
std::string ranked(int n, int batch) {
    GetriShape s;
    s.op = Op::getri;
    s.scalar = scalar_kind_of<T>;
    s.backend = Backend::CUDA;
    s.m = s.n = s.k = n;  // as getri_op_shape builds it
    s.batch = batch;
    s.is_gpu = true;
    s.has_sg32 = true;
    s.blocked_available = true;
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

template <typename T>
void emit(const char* dtype, const char* device) {
    for (int n : kGridN)
        for (int batch : kGridBatch)
            std::printf("getri,%s,%s,%d,%d,%s\n", dtype, device, n, batch, ranked<T>(n, batch).c_str());
}

bool on_grid(int n, int batch) {
    bool gn = false, gb = false;
    for (int v : kGridN) gn = gn || v == n;
    for (int v : kGridBatch) gb = gb || v == batch;
    return gn && gb;
}

// Log-uniform n in [1, 8192] and batch in [1, 65536], grid cells excluded; the old ranking per dtype.
void offgrid(int count, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> ln(0.0, 13.0), lb(0.0, 16.0);
    std::printf("n,batch,float,double,cfloat,cdouble\n");
    for (int i = 0; i < count;) {
        const int n = static_cast<int>(std::lround(std::exp2(ln(gen))));
        const int batch = static_cast<int>(std::lround(std::exp2(lb(gen))));
        if (on_grid(n, batch)) continue;
        std::printf("%d,%d,%s,%s,%s,%s\n", n, batch, ranked<float>(n, batch).c_str(), ranked<double>(n, batch).c_str(),
                    ranked<std::complex<float>>(n, batch).c_str(), ranked<std::complex<double>>(n, batch).c_str());
        ++i;
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--offgrid") {
        offgrid(argc > 2 ? std::atoi(argv[2]) : 2000, argc > 3 ? static_cast<unsigned>(std::atoi(argv[3])) : 1u);
        return 0;
    }
    const char* device = argc > 1 ? argv[1] : "sm_89";
    std::printf("op,dtype,device,n,batch,ranked\n");
    emit<float>("float", device);
    emit<double>("double", device);
    emit<std::complex<float>>("cfloat", device);
    emit<std::complex<double>>("cdouble", device);
    return 0;
}
