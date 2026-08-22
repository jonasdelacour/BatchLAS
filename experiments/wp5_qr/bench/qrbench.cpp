// WP5 MEASUREMENT: time the SHIPPED geqrf / orgqr PUBLIC API, A/B by BUILD.
//
// Derived from experiments/wp5_qr/baseline/wp5qr.cpp, which is itself
// experiments/wp4_potrf/phase2_ab/realpotrf.cpp's pattern: public entry points
// only, no harness-local re-implementation of the driver, correctness checked
// in the SAME PROCESS so a fast wrong answer cannot be reported as a win, and
// the same program linked once against build/ and once against
// build-novendor/ so "vendor-free" is the BUILD and not an environment
// variable inside a build that still links cuSOLVER.
//
// FIVE THINGS IT ADDS OVER wp5qr.cpp, each because the baseline could not
// answer a question this phase has to answer:
//
//   1. nanmax. wp5qr.cpp's probes use std::max(0.0, x); std::max returns the
//      FIRST argument when the comparison is false, so a NaN residual reads as
//      a PERFECT one. That defect is recorded in
//      experiments/wp5_qr/kernels/README.md section 4b (break K5) as still
//      present in wp5qr.cpp. Every probe here propagates NaN.
//   2. The RESOLVED ROUTE for geqrf and for orgqr is printed on every row, from
//      backend::geqrf_route / backend::orgqr_route -- the same functions the
//      facade calls. An unrecognised BATCHLAS_*_ROUTE value silently means
//      {Auto,Auto}, which with preferred() all-false is the VENDOR, so a pin
//      that did not take is invisible without this column.
//   3. RECTANGULAR m != n, because the library's own geqrf callers pass tall
//      panels (band_reduction.cc:595, sytrd_sy2sb.cc:504) and every cell in the
//      baseline table is square.
//   4. The native CTA capacity for T is printed, so a cell that claims to
//      exercise the CTA tier can be checked against the tier's own arithmetic.
//   5. A finiteness flag on every row: if any probe is not finite the row is
//      marked BAD and the timing must not be quoted.
#include <batchlas/blas/functions/geqrf.hh>
#include <batchlas/blas/functions/orgqr.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>

#include <batchlas/blas/dispatch/vendor_available.hh>
#include "src/backends/geqrf_route.hh"
#include "src/backends/orgqr_route.hh"
#include "src/extensions/geqrf_native.hh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

using namespace batchlas;
static constexpr Backend BE = Backend::CUDA;
static constexpr bool kVendor = dispatch::factorization_vendor_available<BE>;

// ---------------------------------------------------------------- promotion
template <class T> struct Prom { using type = double; };
template <class R> struct Prom<std::complex<R>> { using type = std::complex<double>; };

static inline double ab(double x) { return std::fabs(x); }
static inline double ab(std::complex<double> x) { return std::abs(x); }
static inline double cj(double x) { return x; }
static inline std::complex<double> cj(std::complex<double> x) { return std::conj(x); }

static inline double up(float x) { return double(x); }
static inline double up(double x) { return x; }
static inline std::complex<double> up(std::complex<float> x) { return {double(x.real()), double(x.imag())}; }
static inline std::complex<double> up(std::complex<double> x) { return x; }

// NAN-PROPAGATING max. std::max(a,b) returns a when b is NaN, so a poisoned
// probe reads as a perfect one -- the defect that made break K5 print
// qr=4.788e-07 over garbage.
static inline double nanmax(double a, double b) {
    if (std::isnan(a) || std::isnan(b)) return std::numeric_limits<double>::quiet_NaN();
    return a > b ? a : b;
}

template <class T> static inline T mk(double re, double im);
template <> inline float mk<float>(double re, double) { return float(re); }
template <> inline double mk<double>(double re, double) { return re; }
template <> inline std::complex<float> mk<std::complex<float>>(double re, double im) { return {float(re), float(im)}; }
template <> inline std::complex<double> mk<std::complex<double>>(double re, double im) { return {re, im}; }

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 6364136223846793005ULL + 1442695040888963407ULL) {}
    double next() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return double(int32_t(uint32_t(s >> 32))) / 2147483648.0;
    }
};

struct Stat { double med, mean, relsd, min; };
static Stat stat_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const double med = v[v.size() / 2];
    double mean = 0; for (double x : v) mean += x; mean /= double(v.size());
    double sd = 0; for (double x : v) sd += (x - mean) * (x - mean);
    sd = std::sqrt(sd / double(v.size()));
    return {med, mean, mean > 0 ? sd / mean : 0.0, v.front()};
}

static double warm_s() { const char* e = std::getenv("WARM_S"); return e ? std::atof(e) : 1.5; }
static int nprobe()    { const char* e = std::getenv("NPROBE"); return e ? std::atoi(e) : 3; }

static const char* origin_name(dispatch::Origin o) {
    switch (o) { case dispatch::Origin::Auto: return "auto";
                 case dispatch::Origin::Native: return "native";
                 case dispatch::Origin::Vendor: return "vendor";
                 default: return "?"; }
}
static const char* algo_name(dispatch::Algorithm a) {
    switch (a) { case dispatch::Algorithm::Auto: return "auto";
                 case dispatch::Algorithm::CTA: return "cta";
                 case dispatch::Algorithm::Blocked: return "blocked";
                 default: return "other"; }
}
static std::string route_str(dispatch::Route r) {
    return std::string(origin_name(r.origin)) + ":" + algo_name(r.algo);
}

// ---------------------------------------------------------------- checks
// || Q R x - A x ||inf / || A x ||inf from the PACKED factor. O(mn) per probe.
template <typename T>
static double geqrf_probe(const UnifiedVector<T>& F, const UnifiedVector<T>& A0,
                          const UnifiedVector<T>& tau, int m, int n, int batch, int np) {
    using D = typename Prom<T>::type;
    const int k = std::min(m, n);
    const size_t st = size_t(m) * n;
    double worst = 0;
    for (int b : {0, batch - 1}) {
        for (int p = 0; p < np; ++p) {
            Rng rg(uint64_t(b) * 977 + uint64_t(p) * 31 + 7);
            std::vector<D> x(n);
            for (int j = 0; j < n; ++j) x[j] = up(mk<T>(rg.next(), rg.next()));
            std::vector<D> ref(m, D(0)), y(m, D(0));
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < m; ++i)
                    ref[i] += up(A0[size_t(b) * st + size_t(j) * m + i]) * x[j];
            for (int i = 0; i < k; ++i) {
                D acc = D(0);
                for (int j = i; j < n; ++j) acc += up(F[size_t(b) * st + size_t(j) * m + i]) * x[j];
                y[i] = acc;
            }
            for (int step = 0; step < k; ++step) {
                const int i = k - 1 - step;
                const D t = up(tau[size_t(b) * size_t(k) + size_t(i)]);
                D s = y[i];
                for (int l = i + 1; l < m; ++l) s += cj(up(F[size_t(b) * st + size_t(i) * m + l])) * y[l];
                y[i] -= t * s;
                for (int l = i + 1; l < m; ++l) y[l] -= t * up(F[size_t(b) * st + size_t(i) * m + l]) * s;
            }
            double num = 0, den = 0;
            for (int i = 0; i < m; ++i) { num = nanmax(num, ab(y[i] - ref[i])); den = nanmax(den, ab(ref[i])); }
            worst = nanmax(worst, den > 0 ? num / den : num);
        }
    }
    return worst;
}

// || Q^H Q x - x ||inf / ||x||inf on an EXPLICIT Q (m x ncols, k columns used).
template <typename T>
static double ortho_probe(const UnifiedVector<T>& Q, int m, int ncols, int k, int batch, int np) {
    using D = typename Prom<T>::type;
    const size_t st = size_t(m) * ncols;
    double worst = 0;
    for (int b : {0, batch - 1}) {
        for (int p = 0; p < np; ++p) {
            Rng rg(uint64_t(b) * 613 + uint64_t(p) * 17 + 3);
            std::vector<D> x(k);
            for (int j = 0; j < k; ++j) x[j] = up(mk<T>(rg.next(), rg.next()));
            std::vector<D> z(m, D(0)), w(k, D(0));
            for (int j = 0; j < k; ++j)
                for (int i = 0; i < m; ++i) z[i] += up(Q[size_t(b) * st + size_t(j) * m + i]) * x[j];
            for (int j = 0; j < k; ++j) {
                D acc = D(0);
                for (int i = 0; i < m; ++i) acc += cj(up(Q[size_t(b) * st + size_t(j) * m + i])) * z[i];
                w[j] = acc;
            }
            double num = 0, den = 0;
            for (int j = 0; j < k; ++j) { num = nanmax(num, ab(w[j] - x[j])); den = nanmax(den, ab(x[j])); }
            worst = nanmax(worst, den > 0 ? num / den : num);
        }
    }
    return worst;
}

// || Q R x - A x || with the EXPLICIT Q -- catches a Q orthonormal but not A's.
template <typename T>
static double qr_probe(const UnifiedVector<T>& Q, const UnifiedVector<T>& F,
                       const UnifiedVector<T>& A0, int m, int n, int batch, int np) {
    using D = typename Prom<T>::type;
    const int k = std::min(m, n);
    const size_t st = size_t(m) * n;
    double worst = 0;
    for (int b : {0, batch - 1}) {
        for (int p = 0; p < np; ++p) {
            Rng rg(uint64_t(b) * 401 + uint64_t(p) * 53 + 11);
            std::vector<D> x(n);
            for (int j = 0; j < n; ++j) x[j] = up(mk<T>(rg.next(), rg.next()));
            std::vector<D> ref(m, D(0)), y(k, D(0)), z(m, D(0));
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < m; ++i) ref[i] += up(A0[size_t(b) * st + size_t(j) * m + i]) * x[j];
            for (int i = 0; i < k; ++i) {
                D acc = D(0);
                for (int j = i; j < n; ++j) acc += up(F[size_t(b) * st + size_t(j) * m + i]) * x[j];
                y[i] = acc;
            }
            for (int j = 0; j < k; ++j)
                for (int i = 0; i < m; ++i) z[i] += up(Q[size_t(b) * st + size_t(j) * m + i]) * y[j];
            double num = 0, den = 0;
            for (int i = 0; i < m; ++i) { num = nanmax(num, ab(z[i] - ref[i])); den = nanmax(den, ab(ref[i])); }
            worst = nanmax(worst, den > 0 ? num / den : num);
        }
    }
    return worst;
}

// ---------------------------------------------------------------- driver
template <typename T>
static int run(const std::string& mode, const char* tn, int m, int n, int batch, int reps) {
    auto q = std::make_shared<Queue>(Device("gpu"), BE);
    const int k = std::min(m, n);
    const size_t st = size_t(m) * n;

    UnifiedVector<T> A0(st * batch), A(st * batch);
    UnifiedVector<T> tau(size_t(batch) * k);
    {
        Rng rg(12345);
        for (size_t i = 0; i < A0.size(); ++i) A0[i] = mk<T>(rg.next(), rg.next());
    }
    UnifiedVector<T*> pA0(batch), pA(batch);
    MatrixView<T, MatrixFormat::Dense> A0v(A0.data(), m, n, m, int(st), batch, pA0.data());
    MatrixView<T, MatrixFormat::Dense> Av(A.data(), m, n, m, int(st), batch, pA.data());

    auto reset_A = [&] { MatrixView<T, MatrixFormat::Dense>::copy(*q, Av, A0v); q->wait(); };

    const std::string groute = route_str(backend::geqrf_route<BE, T>(*q, Av, kVendor));
    const int64_t cta_elems = sycl_geqrf::geqrf_cta_max_elems<T>();

    const size_t g_ws = geqrf_buffer_size<BE, T>(*q, Av, tau.to_span());
    UnifiedVector<std::byte> gws(g_ws ? g_ws : 1);

    if (mode == "geqrf") {
        const auto w0 = std::chrono::steady_clock::now();
        do { reset_A(); geqrf<BE, T>(*q, Av, tau.to_span(), gws.to_span()); q->wait(); }
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count() < warm_s());
        std::vector<double> ms;
        for (int r = 0; r < reps; ++r) {
            reset_A();
            const auto t0 = std::chrono::steady_clock::now();
            geqrf<BE, T>(*q, Av, tau.to_span(), gws.to_span());
            q->wait();
            ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        const Stat s = stat_of(ms);
        const double res = geqrf_probe<T>(A, A0, tau, m, n, batch, nprobe());
        const double dm = m, dn = n;
        const double fl = double(batch) * (2.0 * dm * dn * dn - (2.0 / 3.0) * dn * dn * dn);
        std::printf("geqrf,%s,%d,%d,%d,%.4f,%.4f,%.4f,%.2f,%.3e,-1,-1,%zu,%s,%lld,%s\n",
                    tn, m, n, batch, s.med, s.mean, s.relsd, fl / (s.med * 1e6), res,
                    g_ws, groute.c_str(), (long long)cta_elems,
                    std::isfinite(res) ? "ok" : "BAD");
        return 0;
    }

    // orgqr: build the factor once, UNTIMED, with this build's own geqrf.
    //
    // SYNTH=1 REPLACES THAT WITH SYNTHETIC REFLECTORS, and it exists for ONE
    // purpose: an nsys capture of orgqr that is not contaminated by the setup.
    // The untimed geqrf call issues its own panel, larft, pack_v and THREE GEMM
    // launches per panel, and the gemm rows of cuda_gpu_kern_sum carry no tag
    // saying which caller they came from -- so a profile of `orgqr` on a real
    // factor attributes ~40% of its GEMM time to a call that was never timed.
    // Timing is unaffected by the substitution: ormqr's cost is a function of
    // the SHAPE, not of the values (wp5qr.cpp says the same for its synthI mode).
    //
    // The strict lower triangle of F holds v_i with an implicit 1 at row i, and
    // tau_i = 2/(v_i^H v_i) makes each H_i = I - tau v v^H both Hermitian and
    // unitary, so ortho= is still a real check on the result. recon= is not
    // computable against A0 and is reported as -1.
    const bool synth = std::getenv("SYNTH") != nullptr;
    UnifiedVector<T> F(st * batch);
    if (synth) {
        Rng rg(98765);
        for (size_t i = 0; i < F.size(); ++i) F[i] = mk<T>(rg.next(), rg.next());
        for (int b = 0; b < batch; ++b)
            for (int i = 0; i < k; ++i) {
                double nrm = 1.0;
                for (int l = i + 1; l < m; ++l) {
                    const double a = ab(up(F[size_t(b) * st + size_t(i) * m + l]));
                    nrm += a * a;
                }
                tau[size_t(b) * size_t(k) + size_t(i)] = mk<T>(2.0 / nrm, 0.0);
            }
    } else {
        reset_A();
        geqrf<BE, T>(*q, Av, tau.to_span(), gws.to_span());
        q->wait();
        std::memcpy(F.data(), A.data(), F.size() * sizeof(T));
    }
    const double fres = synth ? 0.0 : geqrf_probe<T>(F, A0, tau, m, n, batch, nprobe());

    UnifiedVector<T*> pF(batch);
    MatrixView<T, MatrixFormat::Dense> Fv(F.data(), m, n, m, int(st), batch, pF.data());
    auto reset_from_F = [&] { MatrixView<T, MatrixFormat::Dense>::copy(*q, Av, Fv); q->wait(); };

    if (mode == "orgqr") {
        const std::string oroute = route_str(backend::orgqr_route<BE, T>(*q, Av, kVendor));
        const size_t o_ws = orgqr_buffer_size<BE, T>(*q, Av, tau.to_span());
        UnifiedVector<std::byte> ows(o_ws ? o_ws : 1);
        const auto w0 = std::chrono::steady_clock::now();
        do { reset_from_F(); orgqr<BE, T>(*q, Av, tau.to_span(), ows.to_span()); q->wait(); }
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count() < warm_s());
        std::vector<double> ms;
        for (int r = 0; r < reps; ++r) {
            reset_from_F();
            const auto t0 = std::chrono::steady_clock::now();
            orgqr<BE, T>(*q, Av, tau.to_span(), ows.to_span());
            q->wait();
            ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        const Stat s = stat_of(ms);
        const double orth = ortho_probe<T>(A, m, n, k, batch, nprobe());
        const double rec = synth ? -1.0 : qr_probe<T>(A, F, A0, m, n, batch, nprobe());
        const double dm = m, dn = n, dk = k;
        const double fl = double(batch) * (4.0 * dm * dn * dk - 2.0 * (dm + dn) * dk * dk + (4.0 / 3.0) * dk * dk * dk);
        const bool good = std::isfinite(orth) && std::isfinite(rec) && std::isfinite(fres);
        std::printf("orgqr,%s,%d,%d,%d,%.4f,%.4f,%.4f,%.2f,%.3e,%.3e,%.3e,%zu,%s,%lld,%s\n",
                    tn, m, n, batch, s.med, s.mean, s.relsd, fl / (s.med * 1e6),
                    fres, orth, rec, o_ws, oroute.c_str(), (long long)cta_elems,
                    good ? "ok" : "BAD");
        return 0;
    }
    std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 2;
}

int main(int argc, char** argv) {
    if (argc < 7) {
        std::fprintf(stderr,
            "usage: qrbench <mode> <type> <m> <n> <batch> <reps>\n"
            "modes : geqrf | orgqr\n"
            "types : float double cfloat cdouble\n"
            "cols  : op,type,m,n,batch,med_ms,mean_ms,relsd,GFLOPs,geqrf_res,ortho,recon,ws_bytes,route,cta_max_elems,flag\n");
        return 2;
    }
    const std::string mode = argv[1], t = argv[2];
    const int m = std::atoi(argv[3]), n = std::atoi(argv[4]);
    const int b = std::atoi(argv[5]), r = std::atoi(argv[6]);
    try {
        if (t == "float")   return run<float>(mode, "float", m, n, b, r);
        if (t == "double")  return run<double>(mode, "double", m, n, b, r);
        if (t == "cfloat")  return run<std::complex<float>>(mode, "cfloat", m, n, b, r);
        if (t == "cdouble") return run<std::complex<double>>(mode, "cdouble", m, n, b, r);
    } catch (const std::exception& e) {
        std::printf("%s,%s,%d,%d,%d,THREW,%s\n", mode.c_str(), t.c_str(), m, n, b, e.what());
        return 0;
    }
    std::fprintf(stderr, "unknown type %s\n", t.c_str());
    return 2;
}
