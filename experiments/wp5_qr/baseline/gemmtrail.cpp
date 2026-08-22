// Time geqrf's WY trailing-update GEMM PAIR on real SUB-VIEWS of an N x N
// parent, at the parent ld and the parent batch stride.
//
// WHY NOT benchmarks/gemm_benchmark: it allocates its own operands at
// ld == rows. Every operand a blocked geqrf hands gemm is a sub-view carrying
// the PARENT ld, and the native register kernels are known to be sensitive to
// that (see the "native GEMM collapses on strided ld" note). The sub-views here
// are built EXPLICITLY with the parent ld, stride and batch -- NOT with
// operator()(Slice,Slice), which propagates the parent pointer array
// (matrix.hh:1140, a known open bug).
//
//   G1   W   = V^H A22    m=ib, n=n2, k=m1   transA = Trans (real) / ConjTrans
//   G3   A22 = A22 - V W  m=m1, n=n2, k=ib   NN, alpha=-1, beta=1
//
// Built twice, against build/ and build-novendor/, so the vendor/native
// comparison is a BUILD difference and not a forced route.
#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace batchlas;
static constexpr Backend BE = Backend::CUDA;

template <class T> struct IsC { static constexpr bool v = false; };
template <class R> struct IsC<std::complex<R>> { static constexpr bool v = true; };

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 6364136223846793005ULL + 1442695040888963407ULL) {}
    double next() { s = s * 6364136223846793005ULL + 1442695040888963407ULL;
                    return double(int32_t(uint32_t(s >> 32))) / 2147483648.0; }
};
template <class T> static inline T mk(double re, double im);
template <> inline float mk<float>(double re, double) { return float(re); }
template <> inline double mk<double>(double re, double) { return re; }
template <> inline std::complex<float> mk<std::complex<float>>(double re, double im) { return {float(re), float(im)}; }
template <> inline std::complex<double> mk<std::complex<double>>(double re, double im) { return {re, im}; }

template <typename T>
static int run(const char* tn, int N, int nb, int j0, int batch, const std::string& which, int reps) {
    auto q = std::make_shared<Queue>(Device("gpu"), BE);
    const int ib = std::min(nb, N - j0);
    const int m1 = N - j0;
    const int n2 = N - j0 - ib;
    if (n2 <= 0) { std::printf("%s,%s,%d,%d,%d,%d,SKIP_n2_le_0\n", which.c_str(), tn, N, nb, j0, batch); return 0; }

    const size_t st = size_t(N) * N;
    UnifiedVector<T> A(st * batch);
    UnifiedVector<T> W(size_t(ib) * n2 * batch);
    { Rng rg(4242); for (size_t i = 0; i < A.size(); ++i) A[i] = mk<T>(rg.next(), rg.next()); }
    { Rng rg(2424); for (size_t i = 0; i < W.size(); ++i) W[i] = mk<T>(rg.next(), rg.next()); }

    // EXPLICIT sub-views: parent ld N, parent stride st, parent batch, own ptr array.
    UnifiedVector<T*> pV(batch), pA22(batch), pW(batch);
    MatrixView<T, MatrixFormat::Dense> V(A.data() + size_t(j0) * N + size_t(j0),
                                         m1, ib, N, int(st), batch, pV.data());
    MatrixView<T, MatrixFormat::Dense> A22(A.data() + size_t(j0 + ib) * N + size_t(j0),
                                           m1, n2, N, int(st), batch, pA22.data());
    MatrixView<T, MatrixFormat::Dense> Wv(W.data(), ib, n2, ib, ib * n2, batch, pW.data());

    const Transpose tA = IsC<T>::v ? Transpose::ConjTrans : Transpose::Trans;
    auto call = [&] {
        if (which == "G1")
            gemm<BE, T>(*q, V, A22, Wv, T(1), T(0), tA, Transpose::NoTrans);
        else
            gemm<BE, T>(*q, V, Wv, A22, T(-1), T(1), Transpose::NoTrans, Transpose::NoTrans);
    };

    const double ws = std::getenv("WARM_S") ? std::atof(std::getenv("WARM_S")) : 1.0;
    const auto w0 = std::chrono::steady_clock::now();
    do { call(); q->wait(); }
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count() < ws);

    std::vector<double> ms;
    for (int r = 0; r < reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        call(); q->wait();
        ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(ms.begin(), ms.end());
    const double med = ms[ms.size() / 2];
    double mean = 0; for (double v : ms) mean += v; mean /= double(ms.size());
    double sd = 0; for (double v : ms) sd += (v - mean) * (v - mean); sd = std::sqrt(sd / double(ms.size()));

    const double dm = (which == "G1") ? ib : m1;
    const double dn = n2;
    const double dk = (which == "G1") ? m1 : ib;
    const double fl = double(batch) * (IsC<T>::v ? 8.0 : 2.0) * dm * dn * dk;
    std::printf("%s,%s,%d,%d,%d,%d,%d,%d,%d,%.4f,%.4f,%.2f\n", which.c_str(), tn, N, nb, j0,
                batch, int(dm), int(dn), int(dk), med, mean > 0 ? sd / mean : 0.0, fl / (med * 1e6));
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 8) {
        std::fprintf(stderr, "usage: gemmtrail <type> <N> <nb> <j0> <batch> <G1|G3> <reps>\n"
                             "prints: which,type,N,nb,j0,batch,m,n,k,med_ms,rel_sd,GFLOPs\n");
        return 2;
    }
    const std::string t = argv[1];
    const int N = std::atoi(argv[2]), nb = std::atoi(argv[3]), j0 = std::atoi(argv[4]);
    const int b = std::atoi(argv[5]); const std::string w = argv[6]; const int r = std::atoi(argv[7]);
    try {
        if (t == "float")   return run<float>("float", N, nb, j0, b, w, r);
        if (t == "double")  return run<double>("double", N, nb, j0, b, w, r);
        if (t == "cfloat")  return run<std::complex<float>>("cfloat", N, nb, j0, b, w, r);
        if (t == "cdouble") return run<std::complex<double>>("cdouble", N, nb, j0, b, w, r);
    } catch (const std::exception& e) {
        std::printf("%s,%s,%d,%d,%d,%d,THREW,%s\n", w.c_str(), t.c_str(), N, nb, j0, b, e.what());
        return 0;
    }
    return 2;
}
