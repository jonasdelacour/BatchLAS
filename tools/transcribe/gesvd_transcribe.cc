// gesvd transcriber (flat-kernel-selection-phase3-plan.md §2 option D). Evaluates the OLD
// router, RouteTable<Op::gesvd,T> + resolve_route, at every grid cell of
// src/ops/gesvd/choice.hh and prints its preference order as a transcriber CSV for
// `scripts/sweep_to_table.py --transcribe`.
//
// Host-only, built against the headers of a tree that still has route_gesvd.hh (424a45bc, the
// sha in the tables' headers; OLD = that checkout, configured once for build/include):
//   g++ -std=c++20 -I$OLD/include -I$OLD/build/include tools/transcribe/gesvd_transcribe.cc -o /tmp/gt
//   /tmp/gt grid sm_89 sm_120 > tuned/transcribed/gesvd.csv
//   /tmp/gt points FILE      # one "dtype herm jobu jobvh m n" per line -> the ranked order
//
// The old predicates read no architecture, so every device gets the same rows. The order is
// DERIVED: resolve the cell with the vendor present, record the winner, mark it unsupported
// and resolve again; once the walk returns the vendor, the remaining natives are ranked by the
// vendor-free walk. The device is a sub-group-32 GPU. Jacobi's max(m, n) cap is a build
// constant (gesvd_jacobi_max_dim), the same on every device, so it is applied.
// The keys: herm = N (general) | L | U; vec = none | all | thin from the canonical jobs (thin
// when either side stays Thin), the only two job facts the old predicates read; m and n.

#include <batchlas/blas/dispatch/route_gesvd.hh>

#include <array>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>

namespace batchlas::dispatch {

inline constexpr Op kExcluding = static_cast<Op>(251);  // a private Op value, never a real op
inline thread_local unsigned g_excluded = 0;            // bit per kGesvdOrder index

inline unsigned bit_of(Route r) {
    for (unsigned i = 0; i < std::size(kGesvdOrder); ++i)
        if (kGesvdOrder[i] == r) return 1u << i;
    return 0;
}

// The real table with the already-ranked routes removed from supports().
template <typename T>
struct RouteTable<kExcluding, T> {
    using Real = RouteTable<Op::gesvd, T>;
    static bool supports(Route r, const GesvdShape& s) { return !(g_excluded & bit_of(r)) && Real::supports(r, s); }
    static bool preferred(Route r, const GesvdShape& s) { return Real::preferred(r, s); }
    static constexpr const Route* order_begin() { return Real::order_begin(); }
    static constexpr const Route* order_end() { return Real::order_end(); }
};

}  // namespace batchlas::dispatch

namespace {

using namespace batchlas;
using namespace batchlas::dispatch;

// src/ops/gesvd/choice.hh's grid: both sides of every threshold the old predicates use
// (max(m, n) <= 32 for cta and the real Jacobi preference, <= 64 for the Jacobi cap).
constexpr std::array<int, 15> kGridMN{1, 2, 4, 8, 16, 24, 32, 33, 48, 64, 65, 128, 256, 512, 1024};
constexpr std::array<const char*, 3> kHerm{"N", "L", "U"};
constexpr std::array<const char*, 3> kVec{"none", "all", "thin"};

std::string spelling(Route r) {
    if (is_vendor(r)) return "vendor";
    switch (r.algo) {
        case Algorithm::Jacobi: return "jacobi";
        case Algorithm::CTA: return "cta";
        case Algorithm::Blocked: return "blocked";
        default: return "?" + std::string(to_string(r.algo));
    }
}

std::optional<Uplo> herm_of(const std::string& h) {
    if (h == "L") return Uplo::Lower;
    if (h == "U") return Uplo::Upper;
    return std::nullopt;
}

// jobu/jobvh as the entry point hands them to the old shape (already canonical).
template <typename T>
std::string ranked(std::optional<Uplo> herm, SvdVectors jobu, SvdVectors jobvh, int m, int n) {
    GesvdShape s;
    s.op = Op::gesvd;
    s.scalar = scalar_kind_of<T>;
    s.backend = Backend::CUDA;
    s.m = m;
    s.n = n;
    s.k = m < n ? m : n;
    s.batch = 128;  // read only as batch >= 1
    s.jobu = jobu;
    s.jobvh = jobvh;
    s.hermitian_uplo = herm;
    s.is_gpu = true;
    s.max_sub_group = 32;
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

// A grid cell's vec value as literal jobs: thin rows keep Thin on both sides even where m == n
// (a thin query near the diagonal maps there; the old predicates only ask "is either Thin").
void jobs_of(const std::string& vec, SvdVectors& ju, SvdVectors& jv) {
    ju = jv = vec == "none" ? SvdVectors::None : (vec == "all" ? SvdVectors::All : SvdVectors::Thin);
}

template <typename T>
void emit(const char* dtype, const char* device) {
    for (const char* h : kHerm)
        for (const char* v : kVec)
            for (int m : kGridMN)
                for (int n : kGridMN) {
                    SvdVectors ju, jv;
                    jobs_of(v, ju, jv);
                    std::printf("gesvd,%s,%s,%s,%s,%d,%d,%s\n", dtype, device, h, v, m, n,
                                ranked<T>(herm_of(h), ju, jv, m, n).c_str());
                }
}

SvdVectors job_of(const std::string& j) {
    return j == "N" ? SvdVectors::None : (j == "A" ? SvdVectors::All : SvdVectors::Thin);
}

// Off-grid replay for the data gate: canonicalise exactly as the entry point does.
int points(const char* path) {
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        std::string dtype, h, ju, jv;
        int m = 0, n = 0;
        if (!(ss >> dtype >> h >> ju >> jv >> m >> n)) continue;
        const int64_t k = m < n ? m : n;
        const SvdVectors cu = canonical_jobu(job_of(ju), m, k), cv = canonical_jobvh(job_of(jv), n, k);
        std::string r;
        if (dtype == "float") r = ranked<float>(herm_of(h), cu, cv, m, n);
        else if (dtype == "double") r = ranked<double>(herm_of(h), cu, cv, m, n);
        else if (dtype == "cfloat") r = ranked<std::complex<float>>(herm_of(h), cu, cv, m, n);
        else r = ranked<std::complex<double>>(herm_of(h), cu, cv, m, n);
        std::printf("%s %s %s %s %d %d %s\n", dtype.c_str(), h.c_str(), ju.c_str(), jv.c_str(), m, n, r.c_str());
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "grid";
    if (mode == "points") return argc > 2 ? points(argv[2]) : 2;
    std::printf("op,dtype,device,herm,vec,m,n,ranked\n");
    for (int i = 2; i < (argc > 2 ? argc : 3); ++i) {
        const char* device = argc > 2 ? argv[i] : "sm_89";
        emit<float>("float", device);
        emit<double>("double", device);
        emit<std::complex<float>>("cfloat", device);
        emit<std::complex<double>>("cdouble", device);
    }
    return 0;
}
