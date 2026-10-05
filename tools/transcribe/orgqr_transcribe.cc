// orgqr transcriber (flat-kernel-selection-phase3-plan.md §2 option D). Evaluates the OLD router,
// RouteTable<Op::orgqr,T> + resolve_route, at every grid cell of src/ops/orgqr/choice.hh and
// prints its preference order as a transcriber CSV for `scripts/sweep_to_table.py --transcribe`.
//
// Host-only, built against the headers of a tree that still has route_orgqr.hh (424a45bc, the
// sha in the tables' headers; OLD = that checkout, configured once for build/include):
//   g++ -std=c++20 -I$OLD/include -I$OLD/build/include tools/transcribe/orgqr_transcribe.cc -o /tmp/ot
//   /tmp/ot sm_89 sm_120 > tuned/transcribed/orgqr.csv
//   /tmp/ot --random 2500 7 [lo hi] > points.csv   # the data gate: old order at random off-grid
//                                                 # (m, n), log2 extents uniform in [lo, hi) (0, 14)
//
// The order is DERIVED, not written down: resolve the cell with the vendor present, record the
// winner, mark it unsupported and resolve again; once the walk returns the vendor, the remaining
// natives come from the vendor-free walk. The old predicates read no architecture and no batch,
// so one transcription serves every device. The grid holds n <= m only: above the diagonal the
// native driver cannot run, which can_run re-applies at run time.

#include <batchlas/blas/dispatch/route_orgqr.hh>

#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>

namespace batchlas::dispatch {

inline constexpr Op kExcluding = static_cast<Op>(250);  // a private Op value, never a real op
inline thread_local unsigned g_excluded = 0;            // bit per kOrgqrOrder index

inline unsigned bit_of(Route r) {
    for (unsigned i = 0; i < std::size(kOrgqrOrder); ++i)
        if (kOrgqrOrder[i] == r) return 1u << i;
    return 0;
}

// The real table with the already-ranked routes removed from supports().
template <typename T>
struct RouteTable<kExcluding, T> {
    using Real = RouteTable<Op::orgqr, T>;
    static bool supports(Route r, const OrgqrShape& s) { return !(g_excluded & bit_of(r)) && Real::supports(r, s); }
    static bool preferred(Route r, const OrgqrShape& s) { return Real::preferred(r, s); }
    static constexpr const Route* order_begin() { return Real::order_begin(); }
    static constexpr const Route* order_end() { return Real::order_end(); }
};

}  // namespace batchlas::dispatch

namespace {

using namespace batchlas;
using namespace batchlas::dispatch;

// src/ops/orgqr/choice.hh's grid: both sides of the old 512 ceiling, a coarse log grid elsewhere.
constexpr std::array<int, 17> kGrid{1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 513, 768, 1024, 2048, 4096, 8192};

std::string spelling(Route r) {
    if (is_vendor(r)) return "vendor";
    if (r.algo == Algorithm::Blocked) return "blocked";
    return "?" + std::string(to_string(r.algo));
}

// orgqr_op_shape's fields for a homogeneous GPU call; batch is read by no predicate.
template <typename T>
std::string ranked(long m, long n) {
    OrgqrShape s;
    s.op = Op::orgqr;
    s.scalar = scalar_kind_of<T>;
    s.backend = Backend::CUDA;
    s.m = m;
    s.n = n;
    s.k = m < n ? m : n;
    s.batch = 128;
    s.side = Side::Left;
    s.transA = Transpose::NoTrans;
    s.is_gpu = true;
    s.heterogeneous_batch = false;
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
    for (int m : kGrid)
        for (int n : kGrid)
            if (n <= m) std::printf("orgqr,%s,%s,%d,%d,%s\n", dtype, device, m, n, ranked<T>(m, n).c_str());
}

// Log-uniform (m, n) over [1, 16384]^2, off the grid, both triangles; the gate's input.
template <typename T>
void random_points(const char* dtype, int count, unsigned seed, double lo, double hi) {
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<double> e(lo, hi);
    auto on_grid = [](long v) {
        for (int g : kGrid)
            if (g == v) return true;
        return false;
    };
    for (int i = 0; i < count;) {
        const long m = static_cast<long>(std::exp2(e(gen))), n = static_cast<long>(std::exp2(e(gen)));
        if (on_grid(m) && on_grid(n)) continue;
        std::printf("orgqr,%s,any,%ld,%ld,%s\n", dtype, m, n, ranked<T>(m, n).c_str());
        ++i;
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::printf("op,dtype,device,m,n,ranked\n");
    if (argc > 1 && std::strcmp(argv[1], "--random") == 0) {
        const int count = argc > 2 ? std::atoi(argv[2]) : 2000;
        const unsigned seed = argc > 3 ? static_cast<unsigned>(std::atoi(argv[3])) : 1u;
        const double lo = argc > 4 ? std::atof(argv[4]) : 0.0, hi = argc > 5 ? std::atof(argv[5]) : 14.0;
        random_points<float>("float", count, seed, lo, hi);
        random_points<double>("double", count, seed + 1, lo, hi);
        random_points<std::complex<float>>("cfloat", count, seed + 2, lo, hi);
        random_points<std::complex<double>>("cdouble", count, seed + 3, lo, hi);
        return 0;
    }
    for (int i = 1; i < (argc > 1 ? argc : 2); ++i) {
        const char* device = argc > 1 ? argv[i] : "sm_89";
        emit<float>("float", device);
        emit<double>("double", device);
        emit<std::complex<float>>("cfloat", device);
        emit<std::complex<double>>("cdouble", device);
    }
    return 0;
}
