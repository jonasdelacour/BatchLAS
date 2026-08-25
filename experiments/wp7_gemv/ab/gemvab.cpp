// WP7 -- the NATIVE vs VENDOR A/B for gemv, through the SHIPPED public API.
//
// Derived from experiments/wp7_gemv/baseline/gemvbase.cpp (the recon harness),
// with three additions and one deletion:
//
//   +  transA takes N | T | C. ConjTrans was never measured by recon and is the
//      LIVE production path -- ortho.cc selects it for all four complex types.
//   +  the RESOLVED ROUTE is printed as a column, resolved through
//      src/backends/gemv_route.hh over the same pure RouteTable<Op::gemv,T> the
//      library uses. A kernel being LINKED is not evidence it RAN; the route
//      column is. (It resolves in THIS TU, which is why this binary must be
//      rebuilt after any preferred() change -- campaign trap 2.)
//   +  the host reference conjugates when transA == ConjTrans.
//   -  nothing.
//
// Pin the arm with BATCHLAS_GEMV_ROUTE. ALWAYS spell it out -- a bare
// `native` resolves to the FIRST supported native route in kGemvOrder, which is
// CTA, so `native` and `native:cta` are the same thing and `native:direct`
// is the only way to reach the Direct body on a GPU with sub-group 32.
//
// ORIGINAL HEADER FOLLOWS.
//
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

// The route adapter, so the harness can PRINT the route the library will pick
// rather than assert it. Reached by relative path because src/ is not installed.
#include "src/backends/gemv_route.hh"

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
static int run(const char* tn, int m, int n, int batch, char trc, int reps) {
    auto q = std::make_shared<Queue>(Device("gpu"), Backend::CUDA);
    sycl::queue& sq = sycl_queue(*q);

    const Transpose tr = (trc == 'N') ? Transpose::NoTrans
                       : (trc == 'C') ? Transpose::ConjTrans
                                      : Transpose::Trans;
    const bool trans = (tr != Transpose::NoTrans);
    const bool conj = (tr == Transpose::ConjTrans);
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

    // THE ROUTE THAT WILL ACTUALLY RUN, resolved from the same pure table the
    // library consults, with vendor_available = true because this binary links
    // the vendor build. Printed, not assumed.
    const auto rt = backend::gemv_route<Backend::CUDA, T>(*q, Av, Xv, Yv, tr, true);
    char route[64];
    std::snprintf(route, sizeof(route), "%s:%s",
                  std::string(dispatch::to_string(rt.origin)).c_str(),
                  std::string(dispatch::to_string(rt.algo)).c_str());

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
                std::complex<double> av = cd<T>(A[size_t(b) * stA + ai]);
                if (conj) av = std::conj(av);
                acc += av * cd<T>(X[size_t(b) * size_t(lx) + size_t(k)]);
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
    std::printf("%s,%d,%d,%d,%c,%s,%.5f,%.5f,%.4f,%.1f,%.3f,%.2e,%d\n",
                tn, m, n, batch, trc, route,
                med, mean, mean > 0 ? sd / mean : 0.0, gbs, gbs / 950.0, relerr, ld);
    std::fflush(stdout);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 7) {
        std::fprintf(stderr,
          "usage: gemvab <type> <m> <n> <batch> <N|T|C> <reps>\n"
          "prints: type,m,n,batch,transA,route,median_ms,mean_ms,rel_sd,GBs,frac_of_950,relerr,ld\n"
          "env: WARM_S (seconds of warm-up), LD (override the leading dimension)\n");
        return 2;
    }
    const std::string t = argv[1];
    const int m = std::atoi(argv[2]), n = std::atoi(argv[3]), b = std::atoi(argv[4]);
    char tr = argv[5][0];
    if (tr == 't') tr = 'T';
    if (tr == 'n') tr = 'N';
    if (tr == 'c') tr = 'C';
    const int r = std::atoi(argv[6]);
    if (t == "float")   return run<float>("float", m, n, b, tr, r);
    if (t == "double")  return run<double>("double", m, n, b, tr, r);
    if (t == "cfloat")  return run<std::complex<float>>("cfloat", m, n, b, tr, r);
    if (t == "cdouble") return run<std::complex<double>>("cdouble", m, n, b, tr, r);
    std::fprintf(stderr, "unknown type %s\n", t.c_str());
    return 2;
}
