// WP8-I1 A/B: the blocked getrf driver's LEFT-HAND INTERCHANGE, two spellings,
// INTERLEAVED INSIDE ONE PROCESS.
//
// WHY A NEW HARNESS RATHER THAN TWO RUNS OF lubench6. GATE-B asks for the two
// arms interleaved within one session. Two getrf DRIVER spellings are neither
// two routes nor two builds, so no pin and no library swap can separate them --
// but BATCHLAS_GETRF_LASWP is re-read on every call once it is present in the
// environment (src/extensions/getrf_blocked.cc), so setenv() between reps is a
// real per-rep interleave. Everything else -- the diagonally dominant then
// row-permuted matrix, the host LAPACKE pivot oracle, the NaN-propagating solve
// probe, the resolved-route column -- is experiments/wp6_lu/bench/lubench6.cpp's,
// copied rather than re-derived.
//
// THE EXTRA ORACLE THIS HARNESS HAS AND lubench6 DOES NOT: the two arms compute
// the SAME composition, so their factors must be BIT-IDENTICAL. That is a
// stronger statement than either residual and it is the one the deferral
// identity actually claims. It is checked elementwise over the whole batch.
//
// ANTI-VACUITY: the resolved left-hand spelling is READ BACK from
// getrf_blocked_debug_params' bits 24+ on every arm and printed. If the
// environment read had already latched (the flag is a function-local static)
// the two arms would print the same mode and every row would be marked BAD.

#include <batchlas/blas/functions/getrf.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>

#include <batchlas/blas/dispatch/vendor_available.hh>
#include "src/backends/getrf_route.hh"
#include "src/extensions/getrf_native.hh"

#include <lapacke.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

using namespace batchlas;
static constexpr Backend BE = Backend::CUDA;
static constexpr bool kVendorF = dispatch::factorization_vendor_available<BE>;

template <class T> struct Prom { using type = double; };
template <class R> struct Prom<std::complex<R>> { using type = std::complex<double>; };
static inline double ab(double x) { return std::fabs(x); }
static inline double ab(std::complex<double> x) { return std::abs(x); }
static inline double up(float x) { return double(x); }
static inline double up(double x) { return x; }
static inline std::complex<double> up(std::complex<float> x) { return {double(x.real()), double(x.imag())}; }
static inline std::complex<double> up(std::complex<double> x) { return x; }

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

struct Stat { double med, mean, relsd; };
static Stat stat_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const double med = v[v.size() / 2];
    double mean = 0; for (double x : v) mean += x; mean /= double(v.size());
    double sd = 0; for (double x : v) sd += (x - mean) * (x - mean);
    sd = std::sqrt(sd / double(v.size()));
    return {med, mean, mean > 0 ? sd / mean : 0.0};
}

template <typename T> struct Tol;
template <> struct Tol<float>                { static constexpr double v = 1e-4; };
template <> struct Tol<std::complex<float>>  { static constexpr double v = 1e-4; };
template <> struct Tol<double>               { static constexpr double v = 1e-11; };
template <> struct Tol<std::complex<double>> { static constexpr double v = 1e-11; };

static double warm_s() { const char* e = std::getenv("WARM_S"); return e ? std::atof(e) : 1.0; }

static const char* orig_name(dispatch::Origin o) {
    switch (o) { case dispatch::Origin::Auto: return "auto";
                 case dispatch::Origin::Native: return "native";
                 case dispatch::Origin::Vendor: return "vendor"; }
    return "?";
}
static const char* alg_name(dispatch::Algorithm a) {
    switch (a) { case dispatch::Algorithm::Auto: return "auto";
                 case dispatch::Algorithm::Direct: return "direct";
                 case dispatch::Algorithm::CTA: return "cta";
                 case dispatch::Algorithm::Blocked: return "blocked";
                 default: return "other"; }
}
static std::string rstr(dispatch::Route r) {
    return std::string(orig_name(r.origin)) + ":" + alg_name(r.algo);
}

static int host_getrf(int n, float* a, int lda, int* ip) { return LAPACKE_sgetrf(LAPACK_COL_MAJOR, n, n, a, lda, ip); }
static int host_getrf(int n, double* a, int lda, int* ip) { return LAPACKE_dgetrf(LAPACK_COL_MAJOR, n, n, a, lda, ip); }
static int host_getrf(int n, std::complex<float>* a, int lda, int* ip) {
    return LAPACKE_cgetrf(LAPACK_COL_MAJOR, n, n, reinterpret_cast<lapack_complex_float*>(a), lda, ip);
}
static int host_getrf(int n, std::complex<double>* a, int lda, int* ip) {
    return LAPACKE_zgetrf(LAPACK_COL_MAJOR, n, n, reinterpret_cast<lapack_complex_double*>(a), lda, ip);
}

// || (P A0) x - L (U x) ||inf / || A0 x ||inf, P rebuilt from the DEVICE pivots.
template <typename T>
static double getrf_probe(const UnifiedVector<T>& F, const UnifiedVector<T>& A0,
                          const int* piv, int piv_stride, int n, int batch) {
    using D = typename Prom<T>::type;
    const size_t st = size_t(n) * n;
    double worst = 0;
    for (int b : {0, batch - 1}) {
        Rng rg(uint64_t(b) * 977 + 7);
        std::vector<D> x(n);
        for (int j = 0; j < n; ++j) x[j] = up(mk<T>(rg.next(), rg.next()));
        std::vector<D> ref(n, D(0));
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                ref[i] += up(A0[size_t(b) * st + size_t(j) * n + i]) * x[j];
        double den = 0; for (int i = 0; i < n; ++i) den = std::max(den, ab(ref[i]));
        const int* pv = piv + size_t(b) * size_t(piv_stride);
        for (int k = 0; k < n; ++k) {
            const int ip = pv[k] - 1;
            if (ip < 0 || ip >= n) return std::numeric_limits<double>::quiet_NaN();
            if (ip != k) std::swap(ref[k], ref[ip]);
        }
        std::vector<D> y(n, D(0)), z(n, D(0));
        for (int i = 0; i < n; ++i) {
            D acc = D(0);
            for (int j = i; j < n; ++j) acc += up(F[size_t(b) * st + size_t(j) * n + i]) * x[j];
            y[i] = acc;
        }
        for (int i = 0; i < n; ++i) {
            D acc = y[i];
            for (int j = 0; j < i; ++j) acc += up(F[size_t(b) * st + size_t(j) * n + i]) * y[j];
            z[i] = acc;
        }
        double num = 0;
        for (int i = 0; i < n; ++i) {
            const double d = ab(ref[i] - z[i]);
            if (std::isnan(d)) return std::numeric_limits<double>::quiet_NaN();
            num = std::max(num, d);
        }
        worst = std::max(worst, den > 0 ? num / den : num);
    }
    return worst;
}

static int nontrivial_pivots(const int* piv, int n) {
    int c = 0; for (int k = 0; k < n; ++k) if (piv[k] != k + 1) ++c; return c;
}

template <typename T>
static void fill_A0(UnifiedVector<T>& A0, int n, int batch, uint64_t seed) {
    const size_t sa = size_t(n) * n;
    std::vector<T> col(n);
    for (int b = 0; b < batch; ++b) {
        Rng rg(seed + uint64_t(b) * 7919);
        for (size_t i = 0; i < sa; ++i) A0[size_t(b) * sa + i] = mk<T>(rg.next(), rg.next());
        for (int i = 0; i < n; ++i)
            A0[size_t(b) * sa + size_t(i) * n + i] =
                A0[size_t(b) * sa + size_t(i) * n + i] + mk<T>(double(n), 0.0);
        std::vector<int> perm(n);
        for (int i = 0; i < n; ++i) perm[i] = i;
        for (int i = n - 1; i > 0; --i) {
            const int j = int((rg.next() * 0.5 + 0.5) * double(i + 1)) % (i + 1);
            std::swap(perm[i], perm[j]);
        }
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) col[i] = A0[size_t(b) * sa + size_t(j) * n + size_t(perm[i])];
            for (int i = 0; i < n; ++i) A0[size_t(b) * sa + size_t(j) * n + i] = col[i];
        }
    }
}

template <typename T>
static int run(const char* tn, int n, int batch, int reps,
               const std::string& armA, const std::string& armB) {
    // The knob must be PRESENT before the first getrf call: the presence test in
    // getrf_blocked.cc is a function-local static.
    setenv("BATCHLAS_GETRF_LASWP", armA.c_str(), 1);

    auto q = std::make_shared<Queue>(Device("gpu"), BE);
    const size_t sa = size_t(n) * n;

    UnifiedVector<T> A0(sa * batch), A(sa * batch), FA(sa * batch);
    UnifiedVector<int64_t> piv(size_t(n) * batch), pivA(size_t(n) * batch);
    UnifiedVector<int32_t> info(batch);
    fill_A0<T>(A0, n, batch, 12345);

    UnifiedVector<T*> pA0(batch), pA(batch);
    MatrixView<T, MatrixFormat::Dense> A0v(A0.data(), n, n, n, int(sa), batch, pA0.data());
    MatrixView<T, MatrixFormat::Dense> Av(A.data(), n, n, n, int(sa), batch, pA.data());
    auto reset_A = [&] { MatrixView<T, MatrixFormat::Dense>::copy(*q, Av, A0v); q->wait(); };

    const size_t f_ws = getrf_buffer_size<BE, T>(*q, Av);
    UnifiedVector<std::byte> fws(f_ws ? f_ws : 1);
    const int* pivi = reinterpret_cast<const int*>(piv.data());
    const int pstride = n;
    const std::string rf = rstr(backend::getrf_route<BE, T>(*q, Av, kVendorF));

    auto arm = [&](const std::string& a) { setenv("BATCHLAS_GETRF_LASWP", a.c_str(), 1); };
    auto mode_of = [&](const std::string& a) {
        arm(a);
        return (sycl_getrf::getrf_blocked_debug_params<T>(*q, n) >> 24) & 0xffu;
    };
    const unsigned mA = mode_of(armA), mB = mode_of(armB);

    auto once = [&]() {
        reset_A();
        const auto t0 = std::chrono::steady_clock::now();
        getrf<BE, T>(*q, Av, piv.to_span(), fws.to_span(), info.to_span());
        q->wait();
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    };

    // WARM both arms: JIT, clocks, first touch. Untimed and discarded.
    const auto w0 = std::chrono::steady_clock::now();
    do { arm(armA); once(); arm(armB); once(); }
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count() < warm_s());

    // ARM A once more, and KEEP its factor and pivots for the bit-identity check.
    arm(armA); once();
    std::memcpy(FA.data(), A.data(), sa * batch * sizeof(T));
    std::memcpy(pivA.data(), piv.data(), size_t(n) * batch * sizeof(int64_t));
    const double resA = getrf_probe<T>(A, A0, pivi, pstride, n, batch);
    const int ntp = nontrivial_pivots(pivi, n);

    int pivmm = 0, infomm = 0;
    {
        std::vector<T> h(sa); std::vector<int> hp(n);
        for (int b : {0, batch - 1}) {
            std::memcpy(h.data(), A0.data() + size_t(b) * sa, sa * sizeof(T));
            const int hi = host_getrf(n, h.data(), n, hp.data());
            for (int k = 0; k < n; ++k) if (hp[k] != pivi[size_t(b) * pstride + k]) ++pivmm;
            if (hi != info[b]) ++infomm;
        }
    }

    // ARM B once, and compare BIT FOR BIT against arm A over the WHOLE batch.
    arm(armB); once();
    size_t diff = 0;
    {
        const unsigned char* p = reinterpret_cast<const unsigned char*>(FA.data());
        const unsigned char* r = reinterpret_cast<const unsigned char*>(A.data());
        if (std::memcmp(p, r, sa * batch * sizeof(T)) != 0) {
            for (size_t i = 0; i < sa * batch; ++i)
                if (std::memcmp(p + i * sizeof(T), r + i * sizeof(T), sizeof(T)) != 0) ++diff;
        }
        if (std::memcmp(pivA.data(), piv.data(), size_t(n) * batch * sizeof(int64_t)) != 0) diff += 1;
    }
    const double resB = getrf_probe<T>(A, A0, pivi, pstride, n, batch);

    // INTERLEAVED reps.
    std::vector<double> msA, msB;
    for (int r = 0; r < reps; ++r) {
        arm(armA); msA.push_back(once());
        arm(armB); msB.push_back(once());
    }
    const Stat sA = stat_of(msA), sB = stat_of(msB);

    int bad = 0; for (int b = 0; b < batch; ++b) if (info[b] != 0) ++bad;
    const bool okc = std::isfinite(resA) && resA <= Tol<T>::v &&
                     std::isfinite(resB) && resB <= Tol<T>::v &&
                     bad == 0 && ntp > 0 && pivmm == 0 && infomm == 0 &&
                     diff == 0 && mA != mB;

    std::printf("%s,%d,%d,%s,%s,%u,%u,%.4f,%.4f,%.4f,%.4f,%.4f,%.3e,%.3e,%zu,%s,%d,%s\n",
                tn, n, batch, armA.c_str(), armB.c_str(), mA, mB,
                sA.med, sB.med, sA.med / sB.med, sA.relsd, sB.relsd,
                resA, resB, diff, rf.c_str(), ntp, okc ? "ok" : "BAD");
    return okc ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc < 6) {
        std::fprintf(stderr, "usage: getrfab <type> <n> <batch> <reps> <armA> <armB>\n");
        return 2;
    }
    const std::string t = argv[1];
    const int n = std::atoi(argv[2]), batch = std::atoi(argv[3]), reps = std::atoi(argv[4]);
    const std::string a = argv[5], b = (argc > 6) ? argv[6] : "defer_gather";
    if (t == "float")   return run<float>(t.c_str(), n, batch, reps, a, b);
    if (t == "double")  return run<double>(t.c_str(), n, batch, reps, a, b);
    if (t == "cfloat")  return run<std::complex<float>>(t.c_str(), n, batch, reps, a, b);
    if (t == "cdouble") return run<std::complex<double>>(t.c_str(), n, batch, reps, a, b);
    std::fprintf(stderr, "unknown type %s\n", t.c_str());
    return 2;
}
