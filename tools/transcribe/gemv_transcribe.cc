// gemv transcriber (flat-kernel-selection.md §13; phase-5 maintainer decision: transcribed old
// routing for sm_89 AND sm_120). Evaluates the OLD router, RouteTable<Op::gemv,T> + resolve_route,
// at every grid cell of src/ops/gemv/choice.hh and prints its preference order as a transcriber
// CSV for `scripts/sweep_to_table.py --transcribe`.
//
// Host-only, built against the headers of a tree that still has route_gemv.hh (424a45bc, the sha
// in the tables' headers; OLD = that checkout, configured once for build/include):
//   g++ -std=c++20 -I$OLD/include -I$OLD/build/include tools/transcribe/gemv_transcribe.cc -o /tmp/gt
//   /tmp/gt sm_89  > tuned/transcribed/gemv.sm_89.csv
//   /tmp/gt sm_120 > tuned/transcribed/gemv.sm_120.csv
//   /tmp/gt --points < cells.txt    # "dtype trans out red batch" per line -> the ranked list
//   /tmp/gt --choices < cells.txt   # "... sg32 vendor" (0/1) -> the old Auto choice there
//
// The order is DERIVED, not written down: resolve the cell with the vendor present, record the
// winner, mark it unsupported and resolve again. Once the walk returns the vendor, the remaining
// natives are ranked by the vendor-free walk. The old predicates read no architecture and gemv
// has no capacity limit, so both devices get the same rows; the device is a sub-group-32 GPU
// with both kernels compiled. --points is the off-grid oracle of the data gate.

#include <batchlas/blas/dispatch/route_gemv.hh>

#include <array>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

namespace batchlas::dispatch {

inline constexpr Op kExcluding = static_cast<Op>(250);  // a private Op value, never a real op
inline thread_local unsigned g_excluded = 0;            // bit per kGemvOrder index

inline unsigned bit_of(Route r) {
    for (unsigned i = 0; i < std::size(kGemvOrder); ++i)
        if (kGemvOrder[i] == r) return 1u << i;
    return 0;
}

// The real table with the already-ranked routes removed from supports().
template <typename T>
struct RouteTable<kExcluding, T> {
    using Real = RouteTable<Op::gemv, T>;
    static bool supports(Route r, const GemvShape& s) { return !(g_excluded & bit_of(r)) && Real::supports(r, s); }
    static bool preferred(Route r, const GemvShape& s) { return Real::preferred(r, s); }
    static constexpr const Route* order_begin() { return Real::order_begin(); }
    static constexpr const Route* order_end() { return Real::order_end(); }
};

}  // namespace batchlas::dispatch

namespace {

using namespace batchlas;
using namespace batchlas::dispatch;

// src/ops/gemv/choice.hh's grid: both sides of every old threshold (red 63/64 and 352/353,
// out 255/256, batch 319/320) plus a coarse log grid.
constexpr std::array<int, 8> kGridOut{1, 8, 64, 255, 256, 1024, 4096, 32768};
constexpr std::array<int, 11> kGridRed{1, 8, 32, 63, 64, 128, 352, 353, 1024, 4096, 32768};
constexpr std::array<int, 8> kGridBatch{1, 16, 128, 319, 320, 1024, 8192, 32768};

std::string spelling(Route r) {
    if (is_vendor(r)) return "vendor";
    switch (r.algo) {
        case Algorithm::CTA: return "cta";
        case Algorithm::Direct: return "direct";
        default: return "?" + std::string(to_string(r.algo));
    }
}

// m and n are A's stored extents: out = m, red = n under NoTrans, swapped otherwise.
template <typename T>
std::string ranked(Transpose trans, long out, long red, long batch) {
    GemvShape s;
    s.op = Op::gemv;
    s.scalar = scalar_kind_of<T>;
    s.backend = Backend::CUDA;
    s.transA = trans;
    s.m = trans == Transpose::NoTrans ? out : red;
    s.n = trans == Transpose::NoTrans ? red : out;
    s.k = s.m;
    s.batch = batch;
    s.is_gpu = true;
    s.has_sg32 = true;
    s.direct_available = true;
    s.cta_available = true;
    std::string res;
    bool vendor_ranked = false;
    g_excluded = 0;
    for (;;) {
        const Route r =
            resolve_route_uninstrumented<kExcluding, T>(Route{Origin::Auto, Algorithm::Auto}, s, !vendor_ranked);
        if (is_vendor(r) && vendor_ranked) break;
        res += (res.empty() ? "" : "|") + spelling(r);
        if (is_vendor(r)) vendor_ranked = true;
        else g_excluded |= bit_of(r);
    }
    return res;
}

template <typename T>
void emit(const char* dtype, const char* device) {
    for (Transpose trans : {Transpose::NoTrans, Transpose::Trans})
        for (int out : kGridOut)
            for (int red : kGridRed)
                for (int batch : kGridBatch)
                    std::printf("gemv,%s,%s,%s,%d,%d,%d,%s\n", dtype, device, trans == Transpose::NoTrans ? "N" : "T",
                                out, red, batch, ranked<T>(trans, out, red, batch).c_str());
}

std::string ranked_any(const char* dtype, Transpose t, long out, long red, long batch) {
    if (!std::strcmp(dtype, "float")) return ranked<float>(t, out, red, batch);
    if (!std::strcmp(dtype, "double")) return ranked<double>(t, out, red, batch);
    if (!std::strcmp(dtype, "cfloat")) return ranked<std::complex<float>>(t, out, red, batch);
    if (!std::strcmp(dtype, "cdouble")) return ranked<std::complex<double>>(t, out, red, batch);
    return "?dtype";
}

int points() {
    char dtype[16], tr[4];
    long out, red, batch;
    while (std::scanf("%15s %3s %ld %ld %ld", dtype, tr, &out, &red, &batch) == 5) {
        const Transpose t = tr[0] == 'N' ? Transpose::NoTrans : (tr[0] == 'C' ? Transpose::ConjTrans : Transpose::Trans);
        std::printf("%s %s %ld %ld %ld %s\n", dtype, tr, out, red, batch, ranked_any(dtype, t, out, red, batch).c_str());
    }
    return 0;
}

// The old router's single Auto answer for one device scenario (sub-group 32, vendor present).
template <typename T>
std::string old_choice(Transpose trans, long out, long red, long batch, bool sg32, bool vendor) {
    GemvShape s;
    s.op = Op::gemv;
    s.scalar = scalar_kind_of<T>;
    s.transA = trans;
    s.m = trans == Transpose::NoTrans ? out : red;
    s.n = trans == Transpose::NoTrans ? red : out;
    s.k = s.m;
    s.batch = batch;
    s.is_gpu = true;
    s.has_sg32 = sg32;
    s.direct_available = s.cta_available = true;
    return spelling(resolve_route_uninstrumented<Op::gemv, T>(Route{Origin::Auto, Algorithm::Auto}, s, vendor));
}

int choices() {
    char dtype[16], tr[4];
    long out, red, batch;
    int sg32, vendor;
    while (std::scanf("%15s %3s %ld %ld %ld %d %d", dtype, tr, &out, &red, &batch, &sg32, &vendor) == 7) {
        const Transpose t = tr[0] == 'N' ? Transpose::NoTrans : (tr[0] == 'C' ? Transpose::ConjTrans : Transpose::Trans);
        std::string c = "?dtype";
        if (!std::strcmp(dtype, "float")) c = old_choice<float>(t, out, red, batch, sg32, vendor);
        if (!std::strcmp(dtype, "double")) c = old_choice<double>(t, out, red, batch, sg32, vendor);
        if (!std::strcmp(dtype, "cfloat")) c = old_choice<std::complex<float>>(t, out, red, batch, sg32, vendor);
        if (!std::strcmp(dtype, "cdouble")) c = old_choice<std::complex<double>>(t, out, red, batch, sg32, vendor);
        std::printf("%s %s %ld %ld %ld %d %d %s\n", dtype, tr, out, red, batch, sg32, vendor, c.c_str());
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && !std::strcmp(argv[1], "--points")) return points();
    if (argc > 1 && !std::strcmp(argv[1], "--choices")) return choices();
    const char* device = argc > 1 ? argv[1] : "sm_89";
    std::printf("op,dtype,device,trans,out,red,batch,ranked\n");
    emit<float>("float", device);
    emit<double>("double", device);
    emit<std::complex<float>>("cfloat", device);
    emit<std::complex<double>>("cdouble", device);
    return 0;
}
