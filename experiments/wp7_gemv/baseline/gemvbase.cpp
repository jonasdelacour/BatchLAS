// WP7 R4 -- time the SHIPPED public gemv API against the vendor build.
//
// Recipe copied from experiments/wp4_potrf/phase2_ab/realpotrf.cpp, which records
// the two traps this file must not step on:
//   * Matrix(const T*, ...) COPIES into its own storage, so a Matrix built that
//     way is not the buffer this program checks. Use a VIEW over our own buffer.
//   * every MatrixView needs its OWN pointer array or data_ptrs(ctx) throws
//     "data_ptrs target is null".
// and the matrix.hh:46 trap:
//   * Vector(size,batch,stride,inc) vs VectorView(data,size,batch,inc,stride) --
//     positions 3 and 4 are swapped. This file uses the TAGGED spelling
//     VectorView(p, size, batch, Inc{...}, Stride{...}) so the compiler holds it.
//
// LAYOUT. Column major, A(i,j) at i + j*ld, ld = m, stride = m*n.
//   transA = NoTrans : y(m) = alpha * A(m,n) * x(n) + beta*y   -- row walk, stride ld
//   transA = Trans   : y(n) = alpha * A(m,n)^T * x(m) + beta*y -- column walk, contiguous
//
// Buffers are filled ON DEVICE so that a multi-GB shared allocation is resident
// on the GPU when the timed region starts; the host only reads A after the last
// timed rep, for the correctness check.
#include <batchlas/blas/functions/gemv.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/sycl_interop.hh>

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

// Deterministic and exactly representable in float: k/16 for k in [0,16].
static inline double gen(long long a, long long b, long long c) {
    long long h = (a * 7 + b * 13 + c * 3) % 17;
    if (h < 0) h += 17;
    return double(h) * 0.0625;
}

// The value at a slot. For complex the imaginary part is half the real part, so
// a wrong-half kernel cannot pass by accident.
template <typename T> static inline T mk(double v) { return static_cast<T>(v); }
template <> inline std::complex<float>  mk<std::complex<float>>(double v)  { return {float(v), float(v * 0.5)}; }
template <> inline std::complex<double> mk<std::complex<double>>(double v) { return {v, v * 0.5}; }

// One reference type for the host check, whatever the device scalar is.
template <typename T> static inline std::complex<double> cd(const T& v) {
    if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>)
        return std::complex<double>(double(v.real()), double(v.imag()));
    else
        return std::complex<double>(double(v), 0.0);
}

template <typename T>
static int run(const char* tn, int m, int n, int batch, bool trans, int reps) {
    auto q = std::make_shared<Queue>(Device("gpu"), Backend::CUDA);
    sycl::queue& sq = sycl_queue(*q);

    const Transpose tr = trans ? Transpose::Trans : Transpose::NoTrans;
    const int lx = trans ? m : n;          // length of x
    const int ly = trans ? n : m;          // length of y
    // LD lets a probe break the ld == m coincidence without changing anything
    // else: the SHAPE, the traffic and the reference are all still (m, n).
    // Padding ld is the cheap fix if a slow cell turns out to be an ld effect,
    // and a kernel is the expensive fix if it is not.
    const int ld = std::max(m, std::getenv("LD") ? std::atoi(std::getenv("LD")) : 0);
    const size_t stA = size_t(ld) * size_t(n);

    UnifiedVector<T> A(stA * size_t(batch));
    UnifiedVector<T> X(size_t(lx) * size_t(batch));
    UnifiedVector<T> Y(size_t(ly) * size_t(batch));

    T* pA = A.data(); T* pX = X.data(); T* pY = Y.data();
    const size_t nA = A.size(), nX = X.size(), nY = Y.size();
    const size_t ldz = size_t(ld), lxx = size_t(lx);
    sq.parallel_for(sycl::range<1>(nA), [=](sycl::id<1> i) {
        const size_t k = i[0], b = k / stA, r = k % stA;
        pA[k] = mk<T>(gen((long long)(r % ldz), (long long)(r / ldz), (long long)b));
    });
    sq.parallel_for(sycl::range<1>(nX), [=](sycl::id<1> i) {
        const size_t k = i[0];
        pX[k] = mk<T>(gen((long long)(k % lxx), 5, (long long)(k / lxx)));
    });
    sq.parallel_for(sycl::range<1>(nY), [=](sycl::id<1> i) { pY[i[0]] = T(0); });
    sq.wait();

    const size_t nptr = size_t(batch);   // named: `pa(size_t(batch))` is a function declaration
    UnifiedVector<T*> pa(nptr);
    MatrixView<T, MatrixFormat::Dense> Av(pA, m, n, ld, int(stA), batch, pa.data());
    VectorView<T> Xv(pX, lx, batch, Inc{1}, Stride{lx});
    VectorView<T> Yv(pY, ly, batch, Inc{1}, Stride{ly});

    const T alpha = static_cast<T>(1), beta = static_cast<T>(0);

    // Warm the JIT, the clocks, and the first-touch migration of a multi-GB
    // shared allocation. A cold first run has fabricated a 3.7x result here.
    const double warm_s = std::getenv("WARM_S") ? std::atof(std::getenv("WARM_S")) : 1.0;
    const auto w0 = std::chrono::steady_clock::now();
    do {
        gemv<Backend::CUDA, T>(*q, Av, Xv, Yv, alpha, beta, tr);
        q->wait();
    } while (std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count() < warm_s);

    std::vector<double> ms;
    for (int r = 0; r < reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        gemv<Backend::CUDA, T>(*q, Av, Xv, Yv, alpha, beta, tr);
        q->wait();
        const auto t1 = std::chrono::steady_clock::now();
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(ms.begin(), ms.end());
    const double med = ms[ms.size() / 2];
    double mean = 0; for (double v : ms) mean += v; mean /= double(ms.size());
    double sd = 0; for (double v : ms) sd += (v - mean) * (v - mean);
    sd = std::sqrt(sd / double(ms.size()));

    // Correctness, in the same process: a fast WRONG answer must not become the
    // baseline. Items 0 and batch-1; item 0 alone is blind to a bad stride.
    double relerr = 0;
    for (int b : {0, batch - 1}) {
        double num = 0, den = 0;
        for (int o = 0; o < ly; ++o) {
            std::complex<double> acc(0, 0);
            for (int k = 0; k < lx; ++k) {
                const size_t ai = trans ? (size_t(k) + size_t(o) * size_t(ld))
                                        : (size_t(o) + size_t(k) * size_t(ld));
                acc += cd<T>(A[size_t(b) * stA + ai]) * cd<T>(X[size_t(b) * size_t(lx) + size_t(k)]);
            }
            const std::complex<double> got = cd<T>(Y[size_t(b) * size_t(ly) + size_t(o)]);
            num = std::max(num, std::abs(got - acc));
            den = std::max(den, std::abs(acc));
        }
        if (den > 0) relerr = std::max(relerr, num / den);
    }

    // Bytes: A dominates. x read once per item, y written once (beta = 0).
    // Only the m*n LIVE elements count as useful traffic; ld padding is not read.
    const double bytes = (double(m) * double(n) + double(lx) + double(ly))
                       * double(sizeof(T)) * double(batch);
    const double gbs = bytes / (med * 1e-3) / 1e9;
    std::printf("%s,%d,%d,%d,%s,%.5f,%.5f,%.4f,%.1f,%.3f,%.2e,%d\n",
                tn, m, n, batch, trans ? "Trans" : "NoTrans",
                med, mean, mean > 0 ? sd / mean : 0.0, gbs, gbs / 900.0, relerr, ld);
    std::fflush(stdout);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 7) {
        std::fprintf(stderr,
          "usage: gemvbase <type> <m> <n> <batch> <N|T> <reps>\n"
          "prints: type,m,n,batch,transA,median_ms,mean_ms,rel_sd,GBs,frac_of_900,relerr,ld\n"
          "env: WARM_S (seconds of warm-up), LD (override the leading dimension)\n");
        return 2;
    }
    const std::string t = argv[1];
    const int m = std::atoi(argv[2]), n = std::atoi(argv[3]), b = std::atoi(argv[4]);
    const bool tr = (argv[5][0] == 'T' || argv[5][0] == 't');
    const int r = std::atoi(argv[6]);
    if (t == "float")   return run<float>("float", m, n, b, tr, r);
    if (t == "double")  return run<double>("double", m, n, b, tr, r);
    if (t == "cfloat")  return run<std::complex<float>>("cfloat", m, n, b, tr, r);
    if (t == "cdouble") return run<std::complex<double>>("cdouble", m, n, b, tr, r);
    std::fprintf(stderr, "unknown type %s\n", t.c_str());
    return 2;
}
