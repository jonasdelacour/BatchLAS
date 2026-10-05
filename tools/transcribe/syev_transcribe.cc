// syev transcriber (docs/design/flat-select-p5/syev.md). Evaluates the OLD router,
// RouteTable<Op::syev,T> + resolve_route, at every grid cell of src/ops/syev/choice.hh and prints
// its preference order as a transcriber CSV for `scripts/sweep_to_table.py --transcribe`.
//
// Host-only, built against the headers of a tree that still has the RouteTable in
// include/batchlas/blas/functions/syev.hh (424a45bc, the sha in the tables' headers; OLD = that
// checkout, built once), linked to OLD's util/core libraries for settings() and MatrixView. The
// compiler is DPC++'s clang++ (no -fsycl): g++ mangles concept-constrained constructors differently.
//   clang++ -std=c++20 -I$OLD/include -I$OLD/build/include tools/transcribe/syev_transcribe.cc \
//       -L$OLD/build/src -lbatchlas_util -lbatchlas_core -Wl,--allow-shlib-undefined -o /tmp/st
//   env -u BATCHLAS_SYEV_CTA_MAX_N -u BATCHLAS_SYEV_SMALL_KERNEL /tmp/st sm_89 > tuned/transcribed/syev.sm_89.csv
//   /tmp/st --points P.csv   # gate mode: one old ranking per "dtype,jobz,n,batch" line of P.csv
//
// The order is DERIVED: resolve the cell with the vendor present, record the winner, mark it
// unsupported and resolve again; once the walk returns the vendor, the remaining natives are
// ranked by the vendor-free walk. The old CTA route ran one of three small-n drivers, picked by
// syev_choose_small_kernel<T>(A) from the type and n; that pick is the CTA slot's spelling.
// The old predicates read no architecture (only backend == CUDA, is_gpu and a sub-group of 32),
// so the same rows serve sm_89 and sm_120. Capacities are the old supports(): CTA n <= 32.

#include <batchlas/blas/functions/syev.hh>

#include <array>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

namespace batchlas::dispatch {

inline constexpr Op kExcluding = static_cast<Op>(250);  // a private Op value, never a real op
inline thread_local unsigned g_excluded = 0;            // bit per kSyevOrder index

inline unsigned bit_of(Route r) {
    for (unsigned i = 0; i < std::size(kSyevOrder); ++i)
        if (kSyevOrder[i] == r) return 1u << i;
    return 0;
}

// The real table with the already-ranked routes removed from supports().
template <typename T>
struct RouteTable<kExcluding, T> {
    using Real = RouteTable<Op::syev, T>;
    using Shape = typename Real::Shape;
    static bool supports(Route r, const Shape& s) { return !(g_excluded & bit_of(r)) && Real::supports(r, s); }
    static bool preferred(Route r, const Shape& s) { return Real::preferred(r, s); }
    static constexpr const Route* order_begin() { return Real::order_begin(); }
    static constexpr const Route* order_end() { return Real::order_end(); }
};

}  // namespace batchlas::dispatch

namespace {

using namespace batchlas;
using namespace batchlas::dispatch;
namespace det = batchlas::blas::dispatch::detail;

// src/ops/syev/choice.hh's grid: both sides of every old threshold (n = 8|9 small kernel,
// 24|25 cdouble CTA-vs-vendor, 32|33 CTA cap, 256|257, 320|321, 448|449, 512|513, 1024|1025
// windows) plus a coarse log grid. batch is read by no live predicate.
constexpr std::array<int, 37> kGridN{1,   2,   3,   4,   6,   8,   9,   12,  16,  20,  24,   25,   28,
                                     32,  33,  40,  48,  64,  96,  128, 192, 256, 257, 320, 321,  384,
                                     448, 449, 512, 513, 640, 768, 1024, 1025, 1536, 2048, 4096};
constexpr std::array<int, 5> kGridBatch{128, 512, 2048, 8192, 32768};

template <typename T>
std::string spelling(Route r, int n) {
    if (is_vendor(r)) return "vendor";
    switch (r.algo) {
        case Algorithm::CTA: {
            const MatrixView<T, MatrixFormat::Dense> A(nullptr, n, n, n, n * n, 1);
            switch (det::syev_choose_small_kernel<T>(A)) {
                case det::SyevSmallKernel::Jacobi: return "jacobi";
                case det::SyevSmallKernel::CtaFused: return "cta_fused";
                default: return "cta";
            }
        }
        case Algorithm::Blocked: return "blocked";
        case Algorithm::TwoStage: return "two_stage";
        default: return "?" + std::string(to_string(r.algo));
    }
}

template <typename T>
std::string ranked(JobType jobz, int n, int batch) {
    det::SyevShape s;
    s.op = Op::syev;
    s.scalar = scalar_kind_of<T>;
    s.backend = Backend::CUDA;
    s.m = s.n = s.k = n;
    s.batch = batch;
    s.uplo = Uplo::Lower;
    s.jobtype = jobz;
    s.is_gpu = true;
    s.max_sub_group = 32;
    std::string out;
    bool vendor_ranked = false;
    g_excluded = 0;
    for (;;) {
        const Route r =
            resolve_route_uninstrumented<kExcluding, T>(Route{Origin::Auto, Algorithm::Auto}, s, !vendor_ranked);
        if (is_vendor(r) && vendor_ranked) break;
        out += (out.empty() ? "" : "|") + spelling<T>(r, n);
        if (is_vendor(r)) vendor_ranked = true;
        else g_excluded |= bit_of(r);
    }
    return out;
}

template <typename T>
void emit(const char* dtype, const char* device) {
    for (JobType jobz : {JobType::NoEigenVectors, JobType::EigenVectors})
        for (int n : kGridN)
            for (int batch : kGridBatch)
                std::printf("syev,%s,%s,%s,%d,%d,%s\n", dtype, device, jobz == JobType::EigenVectors ? "V" : "N", n,
                            batch, ranked<T>(jobz, n, batch).c_str());
}

std::string ranked_any(const std::string& dtype, JobType jobz, int n, int batch) {
    if (dtype == "float") return ranked<float>(jobz, n, batch);
    if (dtype == "double") return ranked<double>(jobz, n, batch);
    if (dtype == "cfloat") return ranked<std::complex<float>>(jobz, n, batch);
    return ranked<std::complex<double>>(jobz, n, batch);
}

// Gate mode: "dtype,jobz,n,batch" per line in, the same line plus ",<ranked>" out.
int points(const char* path) {
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        std::stringstream ss(line);
        std::string dtype, jobz, n, batch;
        if (!std::getline(ss, dtype, ',') || !std::getline(ss, jobz, ',') || !std::getline(ss, n, ',') ||
            !std::getline(ss, batch, ','))
            continue;
        const JobType j = jobz == "V" ? JobType::EigenVectors : JobType::NoEigenVectors;
        std::printf("%s,%s\n", line.c_str(), ranked_any(dtype, j, std::stoi(n), std::stoi(batch)).c_str());
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 2 && std::strcmp(argv[1], "--points") == 0) return points(argv[2]);
    const char* device = argc > 1 ? argv[1] : "sm_89";
    std::printf("op,dtype,device,jobz,n,batch,ranked\n");
    emit<float>("float", device);
    emit<double>("double", device);
    emit<std::complex<float>>("cfloat", device);
    emit<std::complex<double>>("cdouble", device);
    return 0;
}
