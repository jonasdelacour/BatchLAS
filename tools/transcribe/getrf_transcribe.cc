// getrf transcriber (flat-kernel-selection-phase3-plan.md §2 option D). Evaluates the OLD router,
// RouteTable<Op::getrf,T> + resolve_route, at every grid cell of src/ops/getrf/choice.hh and
// prints its preference order as a transcriber CSV for `scripts/sweep_to_table.py --transcribe`.
//
// Host-only, built against the headers of a tree that still has route_getrf.hh (424a45bc, the
// sha in the tables' headers; OLD = that checkout, configured once for build/include):
//   g++ -std=c++20 -I$OLD/include -I$OLD/build/include tools/transcribe/getrf_transcribe.cc -o /tmp/gt
//   /tmp/gt sm_89 sm_120 > tuned/transcribed/getrf.csv
//
// The order is DERIVED, not written down: resolve the cell with the vendor present, record the
// winner, mark it unsupported and resolve again. Once the walk returns the vendor, the remaining
// natives are ranked by the vendor-free walk (native_tier_preferred, then the plain walk), which
// is what the old router ran without a vendor or with a tier unable to serve the shape. The old
// predicates read no architecture, so every device gets the same rows. The CTA capacity is
// unlimited (can_run re-applies the device's SLM ceiling at run time); the tiny ceiling is
// getrf_tiny_max_n<T>(), a build constant identical on every device (32, cdouble 16), applied.
//
// `--eval CTA_MAX` instead reads "dtype n batch" lines on stdin and prints, per line, the old
// router's Auto choice with the vendor present and vendor-free at that CTA capacity, followed by
// the full ranked list at unlimited capacity: the data gate's oracle (docs/design/flat-select-p5/getrf.md).

#include <batchlas/blas/dispatch/route_getrf.hh>

#include <array>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

namespace batchlas::dispatch {

inline constexpr Op kExcluding = static_cast<Op>(250);  // a private Op value, never a real op
inline thread_local unsigned g_excluded = 0;            // bit per kGetrfOrder index

inline unsigned bit_of(Route r) {
    for (unsigned i = 0; i < std::size(kGetrfOrder); ++i)
        if (kGetrfOrder[i] == r) return 1u << i;
    return 0;
}

// The real table with the already-ranked routes removed from supports().
template <typename T>
struct RouteTable<kExcluding, T> {
    using Real = RouteTable<Op::getrf, T>;
    static bool supports(Route r, const GetrfShape& s) { return !(g_excluded & bit_of(r)) && Real::supports(r, s); }
    static bool preferred(Route r, const GetrfShape& s) { return Real::preferred(r, s); }
    static bool native_tier_preferred(Route r, const GetrfShape& s) { return Real::native_tier_preferred(r, s); }
    static constexpr const Route* order_begin() { return Real::order_begin(); }
    static constexpr const Route* order_end() { return Real::order_end(); }
};

}  // namespace batchlas::dispatch

namespace {

using namespace batchlas;
using namespace batchlas::dispatch;

// src/ops/getrf/choice.hh's grid: both sides of every old threshold, a coarse log grid elsewhere.
constexpr std::array<int, 28> kGridN{1,  2,  3,  4,   5,   6,   7,   8,   9,   12,  16,  17,  24,  25,
                                     32, 33, 48, 64,  96,  128, 192, 255, 256, 384, 511, 512, 768, 1024};
constexpr std::array<int, 10> kGridBatch{1, 8, 64, 128, 255, 256, 512, 2048, 8192, 32768};
constexpr int kUnlimited = 1 << 30;

template <typename T>
constexpr int tiny_max_n() {  // getrf_tiny.cc tiny_cap<T>()
    return std::is_same_v<T, std::complex<double>> ? 16 : 32;
}

std::string spelling(Route r) {
    if (is_vendor(r)) return "vendor";
    switch (r.algo) {
        case Algorithm::Tiny: return "tiny";
        case Algorithm::CTA: return "cta";
        case Algorithm::Blocked: return "blocked";
        default: return "?" + std::string(to_string(r.algo));
    }
}

template <typename T>
GetrfShape shape_of(int n, int batch, int cta_max) {
    GetrfShape s;
    s.op = Op::getrf;
    s.scalar = scalar_kind_of<T>;
    s.backend = Backend::CUDA;
    s.m = s.n = s.k = n;
    s.batch = batch;
    s.is_gpu = true;
    s.has_sg32 = true;
    s.cta_max_n = cta_max;
    s.blocked_available = true;
    s.tiny_max_n = tiny_max_n<T>();
    return s;
}

template <typename T>
std::string ranked(int n, int batch, int cta_max = kUnlimited) {
    const GetrfShape s = shape_of<T>(n, batch, cta_max);
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
    g_excluded = 0;
    return out;
}

template <typename T>
std::string old_auto(int n, int batch, int cta_max, bool vendor) {
    return spelling(resolve_route_uninstrumented<Op::getrf, T>(Route{Origin::Auto, Algorithm::Auto},
                                                              shape_of<T>(n, batch, cta_max), vendor));
}

template <typename T>
void emit(const char* dtype, const char* device) {
    for (int n : kGridN)
        for (int batch : kGridBatch)
            std::printf("getrf,%s,%s,%d,%d,%s\n", dtype, device, n, batch, ranked<T>(n, batch).c_str());
}

template <typename T>
void eval_line(int n, int batch, int cta_max) {
    std::printf("%s %s %s\n", old_auto<T>(n, batch, cta_max, true).c_str(),
                old_auto<T>(n, batch, cta_max, false).c_str(), ranked<T>(n, batch).c_str());
}

int eval(int cta_max) {
    std::string dtype;
    int n = 0, batch = 0;
    while (std::cin >> dtype >> n >> batch) {
        if (dtype == "float") eval_line<float>(n, batch, cta_max);
        else if (dtype == "double") eval_line<double>(n, batch, cta_max);
        else if (dtype == "cfloat") eval_line<std::complex<float>>(n, batch, cta_max);
        else if (dtype == "cdouble") eval_line<std::complex<double>>(n, batch, cta_max);
        else return std::fprintf(stderr, "bad dtype %s\n", dtype.c_str()), 2;
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 3 && std::strcmp(argv[1], "--eval") == 0) return eval(std::atoi(argv[2]));
    std::printf("op,dtype,device,n,batch,ranked\n");
    for (int i = 1; i < argc; ++i) {
        emit<float>("float", argv[i]);
        emit<double>("double", argv[i]);
        emit<std::complex<float>>("cfloat", argv[i]);
        emit<std::complex<double>>("cdouble", argv[i]);
    }
    return 0;
}
