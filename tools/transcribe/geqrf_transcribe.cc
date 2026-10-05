// geqrf transcriber (flat-kernel-selection-phase3-plan.md §2 option D). Evaluates the OLD router,
// RouteTable<Op::geqrf,T> + resolve_route, at every grid cell of src/ops/geqrf/choice.hh and
// prints its preference order as a transcriber CSV for `scripts/sweep_to_table.py --transcribe`.
//
// Host-only, built against the headers of a tree that still has route_geqrf.hh (424a45bc, the
// sha in the tables' headers; OLD = that checkout, configured once for build/include):
//   g++ -std=c++20 -I$OLD/include -I$OLD/build/include tools/transcribe/geqrf_transcribe.cc -o /tmp/gt
//   /tmp/gt sm_89 sm_120 > tuned/transcribed/geqrf.csv
//
// The old predicates read no architecture, so every device gets the same rows.
//
// The order is DERIVED, not written down: resolve the cell, record the winner, make it unable to
// run and resolve again. Once the walk returns the vendor, the remaining natives are ranked by the
// vendor-free walk. Capacities start unlimited, because can_run re-applies them at run time, and
// "unable to run" is spelled as the capacity that refuses the tier: tiny_max_n = 0 for tiny, a
// 1-element CTA tile for cta (Blocked keeps its leaf), blocked_available = false for blocked. So the
// next entry is what the old router picked on a device where the earlier entries do not fit: its
// preferred() asks best_native_tier, which reads those capacities, not the excluded list.
//
// Keys (choice.hh): form (sq | tall | wide), n = cols, aspect = max(m,n) / min(m,n) (integer
// division). The old tall-panel clause `m >= 128 && n >= 32 && m >= A*n` is `n >= 32 && aspect >= A`
// exactly (A >= 4 makes m >= 128 follow), so every threshold the old predicates use is an
// axis-aligned step in (n, aspect), and the grid holds a point on each side of each step.
//
// --offgrid COUNT SEED: random off-grid (m, n) points instead, with the old Auto choice at the
// capacities given as `dtype:tiny_max_n:cta_max_m:cta_max_elems` arguments, vendor present and
// vendor-free. scripts/geqrf_offgrid_gate.py replays them against the tables.

#include <batchlas/blas/dispatch/route_geqrf.hh>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace batchlas::dispatch {

inline constexpr Op kExcluding = static_cast<Op>(250);  // a private Op value, never a real op
inline thread_local unsigned g_excluded = 0;            // bit per kGeqrfOrder index

inline unsigned bit_of(Route r) {
    for (unsigned i = 0; i < std::size(kGeqrfOrder); ++i)
        if (kGeqrfOrder[i] == r) return 1u << i;
    return 0;
}

// The real table with the already-ranked routes removed from supports(); the shape's capacities
// carry the same exclusions into preferred()'s best_native_tier.
template <typename T>
struct RouteTable<kExcluding, T> {
    using Real = RouteTable<Op::geqrf, T>;
    static bool supports(Route r, const GeqrfShape& s) { return !(g_excluded & bit_of(r)) && Real::supports(r, s); }
    static bool preferred(Route r, const GeqrfShape& s) { return Real::preferred(r, s); }
    static bool native_tier_preferred(Route r, const GeqrfShape& s) { return Real::native_tier_preferred(r, s); }
    static constexpr const Route* order_begin() { return Real::order_begin(); }
    static constexpr const Route* order_end() { return Real::order_end(); }
};

}  // namespace batchlas::dispatch

namespace {

using namespace batchlas;
using namespace batchlas::dispatch;

// src/ops/geqrf/choice.hh's grid.
constexpr std::array<int, 52> kGridN{1,  2,  3,  4,  5,  6,  8,   9,   10,  11,  12,  14,  16,  17,  20,  21,  22,  23,
                                     24, 28, 31, 32, 33, 40, 47,  48,  49,  56,  63,  64,  75,  76,  80,  96,  97,  112,
                                     128, 160, 192, 224, 255, 256, 288, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 8192};
constexpr std::array<int, 11> kGridAspect{1, 2, 3, 4, 5, 7, 8, 12, 16, 64, 256};
constexpr std::array<int, 4> kGridWideN{2, 64, 1024, 8192};
constexpr std::array<int, 3> kGridWideAspect{1, 4, 64};

constexpr int kUnlimited = std::numeric_limits<int>::max();

std::string spelling(Route r) {
    if (is_vendor(r)) return "vendor";
    switch (r.algo) {
        case Algorithm::Tiny: return "tiny";
        case Algorithm::CTA: return "cta";
        case Algorithm::Blocked: return "blocked";
        default: return "?" + std::string(to_string(r.algo));
    }
}

struct Caps {
    int tiny_max_n = kUnlimited;
    int cta_max_m = kUnlimited;
    int64_t cta_max_elems = std::numeric_limits<int64_t>::max() / 4;
};

template <typename T>
GeqrfShape shape_of(int64_t m, int64_t n, const Caps& c) {
    GeqrfShape s;
    s.op = Op::geqrf;
    s.scalar = scalar_kind_of<T>;
    s.backend = Backend::CUDA;
    s.m = m;
    s.n = n;
    s.k = m < n ? m : n;
    s.batch = 1024;  // the old predicates read batch only as batch >= 1
    s.is_gpu = true;
    s.has_sg32 = true;
    s.cta_max_m = c.cta_max_m;
    s.cta_max_elems = c.cta_max_elems;
    s.tiny_max_n = c.tiny_max_n;
    s.blocked_available = true;
    return s;
}

// The excluded routes spelled as capacities: tiny_max_n 0 refuses tiny, a 1-element tile refuses
// cta (above 1x1; the bit refuses it there) and keeps blocked's leaf gate, blocked_available
// refuses blocked.
template <typename T>
GeqrfShape excluded_shape(int64_t m, int64_t n) {
    Caps c;
    if (g_excluded & bit_of({Origin::Native, Algorithm::Tiny})) c.tiny_max_n = 0;
    if (g_excluded & bit_of({Origin::Native, Algorithm::CTA})) c.cta_max_m = 1, c.cta_max_elems = 1;
    GeqrfShape s = shape_of<T>(m, n, c);
    if (g_excluded & bit_of({Origin::Native, Algorithm::Blocked})) s.blocked_available = false;
    return s;
}

template <typename T>
std::string ranked(int64_t m, int64_t n) {
    std::string out;
    bool vendor_ranked = false;
    g_excluded = 0;
    for (;;) {
        const Route r = resolve_route_uninstrumented<kExcluding, T>(Route{Origin::Auto, Algorithm::Auto},
                                                                    excluded_shape<T>(m, n), !vendor_ranked);
        if (is_vendor(r) && vendor_ranked) break;
        out += (out.empty() ? "" : "|") + spelling(r);
        if (is_vendor(r)) vendor_ranked = true;
        else g_excluded |= bit_of(r);
    }
    return out;
}

// A representative (m, n) for a key. With unlimited capacities the old predicates read m only
// through the form and the aspect, so any m with that form and aspect gives the same row.
void cell(const char* dtype_name, const char* device, const char* form, int n, int aspect,
          std::string (*rank)(int64_t, int64_t)) {
    int64_t m = n;
    if (std::strcmp(form, "tall") == 0) m = aspect == 1 ? n + 1 : static_cast<int64_t>(n) * aspect;
    if (std::strcmp(form, "wide") == 0) m = n - 1;  // every native refuses m < n
    std::printf("geqrf,%s,%s,%s,%d,%d,%s\n", dtype_name, device, form, n, aspect, rank(m, n).c_str());
}

template <typename T>
void emit(const char* dtype_name, const char* device) {
    for (int n : kGridN) cell(dtype_name, device, "sq", n, 1, &ranked<T>);
    for (int n : kGridN)
        for (int a : kGridAspect) cell(dtype_name, device, "tall", n, a, &ranked<T>);
    for (int n : kGridWideN)
        for (int a : kGridWideAspect) cell(dtype_name, device, "wide", n, a, &ranked<T>);
}

// ---- --offgrid ----------------------------------------------------------------------------------

template <typename T>
std::string old_auto(int64_t m, int64_t n, const Caps& c, bool vendor) {
    return spelling(resolve_route_uninstrumented<Op::geqrf, T>(Route{Origin::Auto, Algorithm::Auto},
                                                              shape_of<T>(m, n, c), vendor));
}

template <typename T>
void offgrid(const char* dtype_name, const Caps& c, int count, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> logn(0.0, 13.0);  // n, m in [1, 8192]
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (int i = 0; i < count; ++i) {
        const int64_t n = static_cast<int64_t>(std::exp2(logn(rng)));
        int64_t m = n;
        const double p = u(rng);
        // A third square, a half tall (half of those within aspect 16), the rest wide.
        if (p >= 1.0 / 3 && p < 5.0 / 6) {
            const double top = u(rng) < 0.5 ? 4.0 : 13.0;
            const double r = std::exp2(std::uniform_real_distribution<double>(0.0, top)(rng));
            m = std::max<int64_t>(n + 1, std::llround(static_cast<double>(n) * r));
        } else if (p >= 5.0 / 6) {
            if (n < 2) continue;
            m = 1 + static_cast<int64_t>(u(rng) * static_cast<double>(n - 1));
        }
        std::printf("%s,%lld,%lld,%d,%d,%lld,%s,%s\n", dtype_name, static_cast<long long>(m),
                    static_cast<long long>(n), c.tiny_max_n, c.cta_max_m, static_cast<long long>(c.cta_max_elems),
                    old_auto<T>(m, n, c, true).c_str(), old_auto<T>(m, n, c, false).c_str());
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--offgrid") == 0) {
        if (argc < 5) {
            std::fprintf(stderr, "usage: %s --offgrid COUNT SEED dtype:tiny:cta_m:cta_elems...\n", argv[0]);
            return 2;
        }
        const int count = std::atoi(argv[2]);
        std::mt19937_64 rng(std::strtoull(argv[3], nullptr, 10));
        std::printf("dtype,m,n,tiny_max_n,cta_max_m,cta_max_elems,old_vendor,old_vendor_free\n");
        for (int i = 4; i < argc; ++i) {
            char dt[16] = {};
            Caps c;
            long long e = 0;
            if (std::sscanf(argv[i], "%15[a-z]:%d:%d:%lld", dt, &c.tiny_max_n, &c.cta_max_m, &e) != 4) return 2;
            c.cta_max_elems = e;
            const std::string d = dt;
            if (d == "float") offgrid<float>("float", c, count, rng);
            else if (d == "double") offgrid<double>("double", c, count, rng);
            else if (d == "cfloat") offgrid<std::complex<float>>("cfloat", c, count, rng);
            else if (d == "cdouble") offgrid<std::complex<double>>("cdouble", c, count, rng);
            else return 2;
        }
        return 0;
    }
    std::printf("op,dtype,device,form,n,aspect,ranked\n");
    std::vector<const char*> devices;
    for (int i = 1; i < argc; ++i) devices.push_back(argv[i]);
    if (devices.empty()) devices.push_back("sm_89");
    for (const char* device : devices) {
        emit<float>("float", device);
        emit<double>("double", device);
        emit<std::complex<float>>("cfloat", device);
        emit<std::complex<double>>("cdouble", device);
    }
    return 0;
}
