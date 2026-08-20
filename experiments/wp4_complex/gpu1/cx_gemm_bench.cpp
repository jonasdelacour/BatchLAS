// Complex GEMM cost-of-vendor-freedom harness.
//
// WHY NOT benchmarks/gemm_benchmark.cc: that binary is NN-only (it passes
// Transpose::NoTrans twice, gemm_benchmark.cc:100-101) and its shapes come from
// registered size sets. The complex demand this campaign measured is 58% of
// calls TRANSPOSED (CN/NC), so an NN-only harness structurally cannot measure
// the majority of it.
//
// Everything else is copied from that file deliberately: the same xorshift fill
// for padded operands (so a pad-0-vs-pad-N ratio is not also a data change), the
// same alpha=1, the same beta from BATCHLAS_BENCH_BETA.
//
//   cx_gemm_bench <type> <m> <n> <k> <batch> <transA> <transB> [reps]
//     type   : cfloat | cdouble | float | double
//     trans  : N | T | C
//   env: BATCHLAS_BENCH_BETA (default 1), BATCHLAS_BENCH_LD_PAD[_A|_B|_C],
//        BATCHLAS_GEMM_ROUTE (native|vendor), BATCHLAS_GEMM_SYCL_KERNEL
//
// Prints one CSV row: type,m,n,k,batch,tA,tB,beta,padA,padB,padC,reps,
//                     median_ms,min_ms,rel_sd,gflops
#include <batchlas/blas/linalg.hh>
#include <batchlas/backend_config.h>



#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace batchlas;

static int env_int(const char* v, int dflt) {
    if (const char* p = std::getenv(v)) return std::atoi(p);
    return dflt;
}
static double env_dbl(const char* v, double dflt) {
    if (const char* p = std::getenv(v)) return std::atof(p);
    return dflt;
}
static int pad_for(const char* v) { return env_int(v, env_int("BATCHLAS_BENCH_LD_PAD", 0)); }

static Transpose parse_trans(const std::string& s) {
    if (s == "N") return Transpose::NoTrans;
    if (s == "T") return Transpose::Trans;
    if (s == "C") return Transpose::ConjTrans;
    std::fprintf(stderr, "bad transpose '%s'\n", s.c_str());
    std::exit(2);
}

template <typename T>
static Matrix<T> make_mat(size_t rows, size_t cols, size_t batch, int pad) {
    if (pad == 0) return Matrix<T>::Random(rows, cols, false, batch);
    Matrix<T> M(static_cast<int>(rows), static_cast<int>(cols), static_cast<int>(batch),
                static_cast<int>(rows) + pad);
    auto host = M.data();
    uint32_t s = 0x9E3779B9u;
    for (size_t i = 0; i < host.size(); ++i) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        const double u = static_cast<double>(s) / 4294967296.0 - 0.5;
        host[i] = static_cast<T>(u);
    }
    return M;
}

template <typename T>
static int run(const char* tname, size_t m, size_t n, size_t k, size_t batch,
               Transpose tA, Transpose tB, int reps) {
    const double beta_d = env_dbl("BATCHLAS_BENCH_BETA", 1.0);
    const int pa = pad_for("BATCHLAS_BENCH_LD_PAD_A");
    const int pb = pad_for("BATCHLAS_BENCH_LD_PAD_B");
    const int pc = pad_for("BATCHLAS_BENCH_LD_PAD_C");

    const size_t Ar = (tA == Transpose::NoTrans) ? m : k;
    const size_t Ac = (tA == Transpose::NoTrans) ? k : m;
    const size_t Br = (tB == Transpose::NoTrans) ? k : n;
    const size_t Bc = (tB == Transpose::NoTrans) ? n : k;

    auto A = make_mat<T>(Ar, Ac, batch, pa);
    auto B = make_mat<T>(Br, Bc, batch, pb);
    auto C = make_mat<T>(m, n, batch, pc);

    auto q = std::make_shared<Queue>(Device("gpu"), Backend::CUDA);
    const T alpha = T(1);
    const T beta = static_cast<T>(beta_d);

    // JIT warm-up AND clock warm-up, both mandatory here.
    //  * a first-run SYCL JIT has fabricated a 3.7x regression in this tree;
    //  * gpu_guard reports the SM clock at start and on an idle 4090 it is
    //    210 MHz. A 0.4 s process never leaves the idle clock, and two kernels
    //    with different arithmetic intensity do not scale with clock the same
    //    way, so a cold ratio is not the warm ratio.
    // Run for at least WARM_S seconds before the first timed rep.
    const double warm_s = env_dbl("BATCHLAS_BENCH_WARM_S", 1.5);
    {
        const auto w0 = std::chrono::steady_clock::now();
        do {
            for (int i = 0; i < 5; ++i) gemm(*q, A, B, C, alpha, beta, tA, tB);
            q->wait();
        } while (std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count() < warm_s);
    }

    std::vector<double> ms;
    ms.reserve(reps);
    for (int r = 0; r < reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        const int inner = 5;
        for (int i = 0; i < inner; ++i) gemm(*q, A, B, C, alpha, beta, tA, tB);
        q->wait();
        const auto t1 = std::chrono::steady_clock::now();
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count() / inner);
    }
    std::vector<double> sorted = ms;
    std::sort(sorted.begin(), sorted.end());
    const double med = sorted[sorted.size() / 2];
    const double mn = sorted.front();
    double mean = 0.0;
    for (double v : ms) mean += v;
    mean /= ms.size();
    double var = 0.0;
    for (double v : ms) var += (v - mean) * (v - mean);
    const double sd = std::sqrt(var / ms.size());
    const double flop_scale = (std::string(tname).front() == 'c') ? 8.0 : 2.0;
    const double gflops = flop_scale * double(m) * n * k * batch / (med * 1e6);

    auto tc = [](Transpose t) { return t == Transpose::NoTrans ? 'N' : (t == Transpose::Trans ? 'T' : 'C'); };
    std::printf("%s,%zu,%zu,%zu,%zu,%c,%c,%g,%d,%d,%d,%d,%.6f,%.6f,%.4f,%.2f\n",
                tname, m, n, k, batch, tc(tA), tc(tB), beta_d, pa, pb, pc, reps,
                med, mn, sd / mean, gflops);
    std::fflush(stdout);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 8) {
        std::fprintf(stderr,
                     "usage: %s <cfloat|cdouble|float|double> <m> <n> <k> <batch> <tA> <tB> [reps]\n",
                     argv[0]);
        return 2;
    }
    const std::string type = argv[1];
    const size_t m = std::strtoull(argv[2], nullptr, 10);
    const size_t n = std::strtoull(argv[3], nullptr, 10);
    const size_t k = std::strtoull(argv[4], nullptr, 10);
    const size_t batch = std::strtoull(argv[5], nullptr, 10);
    const Transpose tA = parse_trans(argv[6]);
    const Transpose tB = parse_trans(argv[7]);
    const int reps = (argc > 8) ? std::atoi(argv[8]) : 9;

    if (type == "cfloat") return run<std::complex<float>>("cfloat", m, n, k, batch, tA, tB, reps);
    if (type == "cdouble") return run<std::complex<double>>("cdouble", m, n, k, batch, tA, tB, reps);
    if (type == "float") return run<float>("float", m, n, k, batch, tA, tB, reps);
    if (type == "double") return run<double>("double", m, n, k, batch, tA, tB, reps);
    std::fprintf(stderr, "unknown type %s\n", type.c_str());
    return 2;
}
