// trsm transcriber (flat-kernel-selection-phase3-plan.md §2 option D, §3 trsm grid). Evaluates
// the OLD router, RouteTable<Op::trsm,T> + resolve_route, at every grid cell of
// src/ops/trsm/choice.hh and prints its preference order as a transcriber CSV for
// `scripts/sweep_to_table.py --transcribe`.
//
// Host-only, built against the headers of a tree that still has route_trsm.hh (8b9adeb3, the
// sha in the tables' headers; OLD = that checkout, configured once for build/include):
//   g++ -std=c++20 -I$OLD/include -I$OLD/build/include tools/transcribe/trsm_transcribe.cc -o /tmp/tt
//   /tmp/tt sm_89 > tuned/transcribed/trsm.sm_89.csv
//
// The order is DERIVED, not written down: resolve the cell with the vendor present, record the
// winner, mark it unsupported and resolve again. Once the walk returns the vendor, the remaining
// natives are ranked by the vendor-free walk (what the old router ran without a vendor). The
// CTA capacity is trsm_cta_max_n<T>() == 32, a build constant identical on every device
// (trsm_native.cc), so it is applied here and cta is not listed above order 32. The old batch
// floor (batch < 8 -> vendor) lies below the grid's smallest batch, 128, and is not transcribed.
// sg_left is new in P3.2b; the old router never chose it, so no transcribed row names it.

#include <batchlas/blas/dispatch/route_trsm.hh>

#include <array>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <string>

namespace batchlas::dispatch {

inline constexpr Op kExcluding = static_cast<Op>(250);  // a private Op value, never a real op
inline thread_local unsigned g_excluded = 0;            // bit per kTrsmOrder index

inline unsigned bit_of(Route r) {
    for (unsigned i = 0; i < std::size(kTrsmOrder); ++i)
        if (kTrsmOrder[i] == r) return 1u << i;
    return 0;
}

// The real table with the already-ranked routes removed from supports().
template <typename T>
struct RouteTable<kExcluding, T> {
    using Real = RouteTable<Op::trsm, T>;
    static bool supports(Route r, const TrsmShape& s) { return !(g_excluded & bit_of(r)) && Real::supports(r, s); }
    static bool preferred(Route r, const TrsmShape& s) { return Real::preferred(r, s); }
    static constexpr const Route* order_begin() { return Real::order_begin(); }
    static constexpr const Route* order_end() { return Real::order_end(); }
};

}  // namespace batchlas::dispatch

namespace {

using namespace batchlas;
using namespace batchlas::dispatch;

// src/ops/trsm/choice.hh's grid.
constexpr std::array<int, 18> kGridOrder{1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024};
constexpr std::array<int, 12> kGridQ{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096};
constexpr std::array<int, 5> kGridBatch{128, 512, 2048, 8192, 32768};
constexpr int kCtaMaxN = 32;  // trsm_cta_max_n<T>() for all four types

std::string spelling(Route r) {
    if (is_vendor(r)) return "vendor";
    switch (r.algo) {
        case Algorithm::CTA: return "cta";
        case Algorithm::Blocked: return "blocked";
        default: return "?" + std::string(to_string(r.algo));
    }
}

template <typename T>
std::string ranked(Side side, Transpose trans, int order, int q, int batch) {
    TrsmShape s;
    s.op = Op::trsm;
    s.scalar = scalar_kind_of<T>;
    s.backend = Backend::CUDA;
    s.m = side == Side::Left ? order : q;  // B's extents, as trsm_op_shape builds them
    s.n = side == Side::Left ? q : order;
    s.k = order;
    s.batch = batch;
    s.side = side;
    s.uplo = Uplo::Lower;
    s.transA = trans;
    s.diag = Diag::NonUnit;
    s.is_gpu = true;
    s.cta_max_n = kCtaMaxN;
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
    for (Side side : {Side::Left, Side::Right})
        for (Transpose trans : {Transpose::NoTrans, Transpose::Trans})
            for (int order : kGridOrder)
                for (int q : kGridQ)
                    for (int batch : kGridBatch)
                        std::printf("trsm,%s,%s,%s,%s,%d,%d,%d,%s\n", dtype, device, side == Side::Left ? "L" : "R",
                                    trans == Transpose::NoTrans ? "N" : "T", order, q, batch,
                                    ranked<T>(side, trans, order, q, batch).c_str());
}

}  // namespace

int main(int argc, char** argv) {
    const char* device = argc > 1 ? argv[1] : "sm_89";
    std::printf("op,dtype,device,side,trans,order,q,batch,ranked\n");
    emit<float>("float", device);
    emit<double>("double", device);
    emit<std::complex<float>>("cfloat", device);
    emit<std::complex<double>>("cdouble", device);
    return 0;
}
