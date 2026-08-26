// WP8/I3 -- THE GATE-B A/B: body 5 (GemvSegTKernel<T,W>) against THE ARM IT
// REPLACES, which is body 3 (GemvCtaTKernel<T>) and NOT cuBLAS.
//
// WHY A DEDICATED BINARY RATHER THAN TWO RUNS OF experiments/wp7_gemv/ab.
// Body 5 and body 3 are two DRIVER SPELLINGS of ONE route -- both resolve to
// `native:cta` -- so BATCHLAS_GEMV_ROUTE cannot separate them and the route
// column cannot report which ran. Two things follow, and both are campaign
// traps this file exists to close:
//
//   * INTERLEAVING. GATE-B wants the two arms alternated inside ONE session so
//     a clock drift or a foreign process has to hit both. Here they alternate
//     REP BY REP inside one process, on the same buffers, at the same clocks.
//     src/extensions/getrs_native.cc's BATCHLAS_GETRS_LASWP A/B is the
//     precedent (experiments/wp8_getrs/getrsab.cpp).
//
//   * "LINKED IS NOT REACHABLE" (trap 4). If the gate declines -- wrong type,
//     red_len above the gate, no enumerated sub-group 32 -- body 5 SILENTLY
//     FALLS THROUGH to body 3 and the A/B reports a flat 1.00x that looks like
//     an honest negative. So each arm reads its resolved kernel back from
//     sycl_gemv::gemv_seg_trans_width_debug, THE SAME function the launcher's
//     gate calls, and both resolved widths are printed as columns. A row whose
//     wA equals its wB is comparing an arm with itself and must be refused.
//
// THE TWO ARMS DO NOT PRODUCE BIT-IDENTICAL ANSWERS and must not be asserted to.
// They sum the same red_len products in a different ORDER -- body 3 folds 32
// partials, body 5 folds L = 32/W -- so they differ in the last ulp. Both are
// checked against the SAME in-process host oracle instead, and both relerrs are
// printed; a fast wrong answer cannot enter the record from either side.
//
// Buffer construction, the Matrix/VectorView traps and the warm-up are
// experiments/wp7_gemv/ab/gemvab.cpp's, unchanged.
#include <batchlas/blas/functions/gemv.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/sycl_interop.hh>

#include "src/sycl/gemv_native.hh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

using namespace batchlas;

static inline double gen(long long a, long long b, long long c) {
    long long h = (a * 7 + b * 13 + c * 3) % 17;
    if (h < 0) h += 17;
    return double(h) * 0.0625;
}
template <typename T> static inline T mk(double v) { return static_cast<T>(v); }
template <> inline std::complex<float>  mk<std::complex<float>>(double v)  { return {float(v), float(v * 0.5)}; }
template <> inline std::complex<double> mk<std::complex<double>>(double v) { return {v, v * 0.5}; }
template <typename T> static inline std::complex<double> cd(const T& v) {
    if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>)
        return std::complex<double>(double(v.real()), double(v.imag()));
    else
        return std::complex<double>(double(v), 0.0);
}

static double med_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}
static double relsd_of(const std::vector<double>& v) {
    double mean = 0; for (double x : v) mean += x; mean /= double(v.size());
    double sd = 0; for (double x : v) sd += (x - mean) * (x - mean);
    sd = std::sqrt(sd / double(v.size()));
    return mean > 0 ? sd / mean : 0.0;
}

template <typename T>
static int run(const char* tn, int m, int n, int batch, char trc, int reps, const char* warm) {
    auto q = std::make_shared<Queue>(Device("gpu"), Backend::CUDA);
    sycl::queue& sq = sycl_queue(*q);

    const Transpose tr = (trc == 'C') ? Transpose::ConjTrans : Transpose::Trans;
    const bool conj = (tr == Transpose::ConjTrans);
    const int lx = m, ly = n;                 // Trans: red_len = m, out_len = n
    const int ld = std::max(m, std::getenv("LD") ? std::atoi(std::getenv("LD")) : 0);
    const size_t stA = size_t(ld) * size_t(n);

    UnifiedVector<T> A(stA * size_t(batch));
    UnifiedVector<T> X(size_t(lx) * size_t(batch));
    UnifiedVector<T> Ya(size_t(ly) * size_t(batch));
    UnifiedVector<T> Yb(size_t(ly) * size_t(batch));

    T* pA = A.data(); T* pX = X.data(); T* pYa = Ya.data(); T* pYb = Yb.data();
    const size_t nA = A.size(), nX = X.size(), nY = Ya.size();
    const size_t ldz = size_t(ld), lxx = size_t(lx);
    sq.parallel_for(sycl::range<1>(nA), [=](sycl::id<1> i) {
        const size_t k = i[0], b = k / stA, r = k % stA;
        pA[k] = mk<T>(gen((long long)(r % ldz), (long long)(r / ldz), (long long)b));
    });
    sq.parallel_for(sycl::range<1>(nX), [=](sycl::id<1> i) {
        const size_t k = i[0];
        pX[k] = mk<T>(gen((long long)(k % lxx), 5, (long long)(k / lxx)));
    });
    sq.parallel_for(sycl::range<1>(nY), [=](sycl::id<1> i) { pYa[i[0]] = T(0); pYb[i[0]] = T(0); });
    sq.wait();

    const size_t nptr = size_t(batch);
    UnifiedVector<T*> pa(nptr), pb(nptr);      // trap 5: each view needs its OWN array
    MatrixView<T, MatrixFormat::Dense> Av(pA, m, n, ld, int(stA), batch, pa.data());
    VectorView<T> Xv(pX, lx, batch, Inc{1}, Stride{lx});
    VectorView<T> Yav(pYa, ly, batch, Inc{1}, Stride{ly});
    VectorView<T> Ybv(pYb, ly, batch, Inc{1}, Stride{ly});
    static_cast<void>(pb);

    const T alpha = static_cast<T>(1), beta = static_cast<T>(0);

    // The pin that makes BOTH arms reach the CTA route at all. Set once; the
    // ARM is then chosen by BATCHLAS_GEMV_SEGT alone.
    ::setenv("BATCHLAS_GEMV_ROUTE", "native:cta", 1);

    // THE RESOLVED KERNEL PER ARM, from the launcher's own gate function.
    ::setenv("BATCHLAS_GEMV_SEGT", warm, 1);
    const int64_t kItems = static_cast<int64_t>(n) * batch;
    const int wA = sycl_gemv::gemv_seg_trans_width_debug<T>(*q, m, kItems);
    ::setenv("BATCHLAS_GEMV_SEGT", "off", 1);
    const int wB = sycl_gemv::gemv_seg_trans_width_debug<T>(*q, m, kItems);

    const double warm_s = std::getenv("WARM_S") ? std::atof(std::getenv("WARM_S")) : 1.0;
    const auto w0 = std::chrono::steady_clock::now();
    do {
        ::setenv("BATCHLAS_GEMV_SEGT", warm, 1);
        gemv<Backend::CUDA, T>(*q, Av, Xv, Yav, alpha, beta, tr);
        ::setenv("BATCHLAS_GEMV_SEGT", "off", 1);
        gemv<Backend::CUDA, T>(*q, Av, Xv, Ybv, alpha, beta, tr);
        q->wait();
    } while (std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count() < warm_s);

    std::vector<double> ma, mb;
    for (int r = 0; r < reps; ++r) {
        // ALTERNATED REP BY REP, and the order flipped on odd reps so a
        // systematic first-of-pair effect cannot favour one arm.
        for (int half = 0; half < 2; ++half) {
            const bool doA = (r % 2 == 0) ? (half == 0) : (half == 1);
            ::setenv("BATCHLAS_GEMV_SEGT", doA ? warm : "off", 1);
            auto& Y = doA ? Yav : Ybv;
            const auto t0 = std::chrono::steady_clock::now();
            gemv<Backend::CUDA, T>(*q, Av, Xv, Y, alpha, beta, tr);
            q->wait();
            const auto t1 = std::chrono::steady_clock::now();
            (doA ? ma : mb).push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
    }

    double ea = 0, eb = 0;
    for (int b : {0, batch - 1}) {
        double na = 0, nb = 0, den = 0;
        for (int o = 0; o < ly; ++o) {
            std::complex<double> acc(0, 0);
            for (int k = 0; k < lx; ++k) {
                std::complex<double> av = cd<T>(A[size_t(b) * stA + size_t(k) + size_t(o) * size_t(ld)]);
                if (conj) av = std::conj(av);
                acc += av * cd<T>(X[size_t(b) * size_t(lx) + size_t(k)]);
            }
            na = std::max(na, std::abs(cd<T>(Ya[size_t(b) * size_t(ly) + size_t(o)]) - acc));
            nb = std::max(nb, std::abs(cd<T>(Yb[size_t(b) * size_t(ly) + size_t(o)]) - acc));
            den = std::max(den, std::abs(acc));
        }
        if (den > 0) { ea = std::max(ea, na / den); eb = std::max(eb, nb / den); }
    }

    const double bytes = (double(m) * double(n) + double(lx) + double(ly))
                       * double(sizeof(T)) * double(batch);
    const double va = med_of(ma), vb = med_of(mb);
    std::printf("%s,%d,%d,%d,%c,%s,%d,%d,%.5f,%.5f,%.4f,%.4f,%.1f,%.1f,%.4f,%.2e,%.2e,%d\n",
                tn, m, n, batch, trc, warm, wA, wB, va, vb,
                relsd_of(ma), relsd_of(mb),
                bytes / (va * 1e-3) / 1e9, bytes / (vb * 1e-3) / 1e9,
                vb / va, ea, eb, ld);
    std::fflush(stdout);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 7) {
        std::fprintf(stderr,
          "usage: gemvsegab <type> <m> <n> <batch> <T|C> <reps> [W=auto|2|4|8]\n"
          "  m is red_len and n is out_len (transposed only).\n"
          "prints: type,m,n,batch,transA,arm,wA,wB,med_a_ms,med_b_ms,relsd_a,relsd_b,"
          "GBs_a,GBs_b,ratio,relerr_a,relerr_b,ld\n"
          "  arm A = body 5 at the named W; arm B = body 3 (SEGT=off).\n"
          "  ratio = med_b/med_a, so > 1 means BODY 5 IS FASTER.\n"
          "  wA == wB means the gate declined and the row compares an arm with itself.\n");
        return 2;
    }
    const std::string t = argv[1];
    const int m = std::atoi(argv[2]), n = std::atoi(argv[3]), b = std::atoi(argv[4]);
    char tr = argv[5][0];
    if (tr == 't') tr = 'T';
    if (tr == 'c') tr = 'C';
    const int r = std::atoi(argv[6]);
    const char* w = (argc > 7) ? argv[7] : "auto";
    if (t == "float")   return run<float>("float", m, n, b, tr, r, w);
    if (t == "double")  return run<double>("double", m, n, b, tr, r, w);
    if (t == "cfloat")  return run<std::complex<float>>("cfloat", m, n, b, tr, r, w);
    if (t == "cdouble") return run<std::complex<double>>("cdouble", m, n, b, tr, r, w);
    std::fprintf(stderr, "unknown type %s\n", t.c_str());
    return 2;
}
