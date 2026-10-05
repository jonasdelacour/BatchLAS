// ormqr transcriber (flat-kernel-selection.md §13). Evaluates the OLD router,
// RouteTable<Op::ormqr,T> + resolve_route, at every grid cell of src/ops/ormqr/choice.hh and
// prints its preference order as a transcriber CSV for `scripts/sweep_to_table.py --transcribe`.
//
// Host-only, built against the headers of a tree that still has route_ormqr.hh (424a45bc, the
// sha in the tables' headers; OLD = that checkout, configured once for build/include):
//   g++ -std=c++20 -I$OLD/include -I$OLD/build/include tools/transcribe/ormqr_transcribe.cc -o /tmp/ot
//   /tmp/ot sm_89 sm_120 > tuned/transcribed/ormqr.csv
//
// The order is DERIVED, not written down: resolve the cell with the vendor present, record the
// winner, mark it unsupported and resolve again; once the walk returns the vendor, the remaining
// natives are ranked by the vendor-free walk. The old predicates read no architecture and no
// capacity, so one evaluation serves every device named on the command line (a GPU queue).
// With `--points` it reads "dtype side trans m k q batch" lines on stdin and prints each
// cell's ranking (the off-grid data gate, docs/design/flat-select-p5/ormqr.md).

#include <batchlas/blas/dispatch/route_ormqr.hh>

#include <array>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace batchlas::dispatch {

inline constexpr Op kExcluding = static_cast<Op>(250);  // a private Op value, never a real op
inline thread_local unsigned g_excluded = 0;            // bit per kOrmqrOrder index

inline unsigned bit_of(Route r) {
    for (unsigned i = 0; i < std::size(kOrmqrOrder); ++i)
        if (kOrmqrOrder[i] == r) return 1u << i;
    return 0;
}

// The real table with the already-ranked routes removed from supports().
template <typename T>
struct RouteTable<kExcluding, T> {
    using Real = RouteTable<Op::ormqr, T>;
    static bool supports(Route r, const OpShape& s) { return !(g_excluded & bit_of(r)) && Real::supports(r, s); }
    static bool preferred(Route r, const OpShape& s) { return Real::preferred(r, s); }
    static constexpr const Route* order_begin() { return Real::order_begin(); }
    static constexpr const Route* order_end() { return Real::order_end(); }
};

}  // namespace batchlas::dispatch

namespace {

using namespace batchlas;
using namespace batchlas::dispatch;

// src/ops/ormqr/choice.hh's grid; k runs over grid_m up to m.
constexpr std::array<int, 4> kGridM{1, 8, 64, 512};
constexpr std::array<int, 3> kGridQ{1, 32, 1024};
constexpr std::array<int, 3> kGridBatch{128, 2048, 32768};

std::string spelling(Route r) {
    if (is_vendor(r)) return "vendor";
    if (r.algo == Algorithm::Blocked) return "blocked";
    return "?" + std::string(to_string(r.algo));
}

Transpose trans_of(char t) { return t == 'N' ? Transpose::NoTrans : (t == 'T' ? Transpose::Trans : Transpose::ConjTrans); }

// The shape ormqr_op_shape built (functions/ormqr.hh @ 424a45bc): A is m x k here, so
// k = min(rows, cols) = k; q is not part of it (and is read by nothing).
template <typename T>
std::string ranked(Side side, Transpose trans, int m, int k, int batch) {
    OpShape s;
    s.op = Op::ormqr;
    s.scalar = scalar_kind_of<T>;
    s.m = m;
    s.n = k;
    s.k = k;
    s.batch = batch;
    s.side = side;
    s.transA = trans;
    s.is_gpu = true;
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
    for (Side side : {Side::Left, Side::Right})
        for (char t : {'N', 'T', 'C'})
            for (int m : kGridM)
                for (int k : kGridM) {
                    if (k > m) continue;
                    for (int q : kGridQ)
                        for (int batch : kGridBatch)
                            std::printf("ormqr,%s,%s,%s,%c,%d,%d,%d,%d,%s\n", dtype, device,
                                        side == Side::Left ? "L" : "R", t, m, k, q, batch,
                                        ranked<T>(side, trans_of(t), m, k, batch).c_str());
                }
}

template <typename T>
std::string point(char side, char t, int m, int k, int batch) {
    return ranked<T>(side == 'L' ? Side::Left : Side::Right, trans_of(t), m, k, batch);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--points") == 0) {
        char dt[16], side, t;
        int m, k, q, batch;
        while (std::scanf("%15s %c %c %d %d %d %d", dt, &side, &t, &m, &k, &q, &batch) == 7) {
            const std::string d = dt;
            const std::string r = d == "float"    ? point<float>(side, t, m, k, batch)
                                  : d == "double" ? point<double>(side, t, m, k, batch)
                                  : d == "cfloat" ? point<std::complex<float>>(side, t, m, k, batch)
                                                  : point<std::complex<double>>(side, t, m, k, batch);
            std::printf("%s\n", r.c_str());
        }
        return 0;
    }
    std::vector<const char*> devices(argv + 1, argv + argc);
    if (devices.empty()) devices.push_back("sm_89");
    std::printf("op,dtype,device,side,trans,m,k,q,batch,ranked\n");
    for (const char* device : devices) {
        emit<float>("float", device);
        emit<double>("double", device);
        emit<std::complex<float>>("cfloat", device);
        emit<std::complex<double>>("cdouble", device);
    }
    return 0;
}
