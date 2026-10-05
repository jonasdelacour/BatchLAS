// spmm transcriber (docs/design/flat-select-p5/spmm.md). Evaluates the OLD router,
// RouteTable<Op::spmm,T> + resolve_route, at every grid cell of src/ops/spmm/choice.hh and
// prints its preference order as a transcriber CSV for `scripts/sweep_to_table.py --transcribe`.
//
// Host-only, built against the headers of a tree that still has route_spmm.hh (424a45bc, the
// sha in the tables' headers; OLD = that checkout, configured once for build/include):
//   g++ -std=c++20 -I$OLD/include -I$OLD/build/include tools/transcribe/spmm_transcribe.cc -o /tmp/st
//   /tmp/st sm_89 > tuned/transcribed/spmm.sm_89.csv        (likewise sm_120 and cpu)
//   /tmp/st --random 4000 7 > points.csv                    (off-grid gate points, any transA/B)
//
// The order is DERIVED, not written down: resolve the cell with the vendor present, record the
// winner, mark it unsupported and resolve again; once the vendor is ranked, the remaining natives
// are ranked by the vendor-free walk. The old predicates read no device fact (not even is_gpu),
// so one transcription serves every device. Capacities are the build's: both bodies compiled.

#include <batchlas/blas/dispatch/route_spmm.hh>

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
inline thread_local unsigned g_excluded = 0;            // bit per kSpmmOrder index

inline unsigned bit_of(Route r) {
    for (unsigned i = 0; i < std::size(kSpmmOrder); ++i)
        if (kSpmmOrder[i] == r) return 1u << i;
    return 0;
}

// The real table with the already-ranked routes removed from supports().
template <typename T>
struct RouteTable<kExcluding, T> {
    using Real = RouteTable<Op::spmm, T>;
    static bool supports(Route r, const SpmmShape& s) { return !(g_excluded & bit_of(r)) && Real::supports(r, s); }
    static bool preferred(Route r, const SpmmShape& s) { return Real::preferred(r, s); }
    static constexpr const Route* order_begin() { return Real::order_begin(); }
    static constexpr const Route* order_end() { return Real::order_end(); }
};

}  // namespace batchlas::dispatch

namespace {

using namespace batchlas;
using namespace batchlas::dispatch;

// src/ops/spmm/choice.hh's grid.
constexpr std::array<int, 5> kGridM{1, 16, 256, 4096, 65536};
constexpr std::array<int, 5> kGridNrhs{1, 2, 4, 16, 64};
constexpr std::array<int, 5> kGridBatch{1, 8, 128, 1024, 16384};

std::string spelling(Route r) {
    if (is_vendor(r)) return "vendor";
    if (r.algo == Algorithm::Direct) return "direct";
    return "?" + std::string(to_string(r.algo));
}

const char* letter(Transpose t) { return t == Transpose::NoTrans ? "N" : (t == Transpose::Trans ? "T" : "C"); }

template <typename T>
std::string ranked(Transpose ta, Transpose tb, std::int64_t m, std::int64_t k, std::int64_t nrhs,
                   std::int64_t batch) {
    SpmmShape s;  // as spmm_op_shape built it for a valid CSR call
    s.op = Op::spmm;
    s.scalar = scalar_kind_of<T>;
    s.backend = Backend::CUDA;
    s.m = m;
    s.k = k;
    s.n = nrhs;
    s.batch = batch;
    s.transA = ta;
    s.transB = tb;
    s.format = MatrixFormat::CSR;
    s.is_gpu = true;
    s.gather_available = true;
    s.scatter_available = true;
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
    for (Transpose ta : {Transpose::NoTrans, Transpose::Trans})
        for (Transpose tb : {Transpose::NoTrans, Transpose::Trans})
            for (int m : kGridM)
                for (int nrhs : kGridNrhs)
                    for (int batch : kGridBatch)
                        std::printf("spmm,%s,%s,%s,%s,%d,%d,%d,%s\n", dtype, device, letter(ta), letter(tb), m, nrhs,
                                    batch, ranked<T>(ta, tb, m, m, nrhs, batch).c_str());
}

// Off-grid points for the data gate: log-uniform extents (k independent of m), all three
// spellings of each transpose, so the C -> T fold is exercised too.
template <typename T>
void emit_random(const char* dtype, int count, std::mt19937_64& gen) {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    auto logu = [&](double lo, double hi) {
        return static_cast<std::int64_t>(std::exp(std::log(lo) + u(gen) * (std::log(hi) - std::log(lo))));
    };
    const Transpose all[3] = {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans};
    for (int i = 0; i < count; ++i) {
        const Transpose ta = all[gen() % 3], tb = all[gen() % 3];
        const std::int64_t m = logu(1, 300000), k = logu(1, 300000), nrhs = logu(1, 256), batch = logu(1, 65536);
        std::printf("%s,%s,%s,%lld,%lld,%lld,%lld,%s\n", dtype, letter(ta), letter(tb), static_cast<long long>(m),
                    static_cast<long long>(k), static_cast<long long>(nrhs), static_cast<long long>(batch),
                    ranked<T>(ta, tb, m, k, nrhs, batch).c_str());
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--random") {
        const int count = argc > 2 ? std::atoi(argv[2]) : 2000;
        std::mt19937_64 gen(argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 1);
        std::printf("dtype,transA,transB,m,k,nrhs,batch,ranked\n");
        emit_random<float>("float", count, gen);
        emit_random<double>("double", count, gen);
        emit_random<std::complex<float>>("cfloat", count, gen);
        emit_random<std::complex<double>>("cdouble", count, gen);
        return 0;
    }
    const char* device = argc > 1 ? argv[1] : "sm_89";
    std::printf("op,dtype,device,transA,transB,m,nrhs,batch,ranked\n");
    emit<float>("float", device);
    emit<double>("double", device);
    emit<std::complex<float>>("cfloat", device);
    emit<std::complex<double>>("cdouble", device);
    return 0;
}
