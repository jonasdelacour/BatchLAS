// WP6 MEASURE PHASE. This file is experiments/wp6_lu/kernels/luverify.cpp with
// ONE change: the getrs row printed a hard `size_t(0)` in the workspace column,
// so no getrs workspace number existed anywhere. It now prints the NoTrans
// getrs_buffer_size. Everything else -- the host oracle, the elementwise pivot
// comparison, the row-permuted test matrix, the Tol<T> bounds, the resolved-route
// column -- is unchanged, deliberately: re-deriving a harness is how WP4's
// measurement ended up 2x off its own shipped numbers.
//
// WP6 KERNEL VERIFICATION AND A/B: the PUBLIC getrf / getrs / getri, one
// program, linked once against build/ and once against build-novendor/ so that
// "vendor-free" means the BUILD and not a forced route.
//
// Everything here follows experiments/wp6_lu/baseline/lubench.cpp, which follows
// experiments/wp5_qr/bench/qrbench.cpp and wp4_potrf/phase2_ab/realpotrf.cpp:
// public entry points only, correctness checked IN THE SAME PROCESS so a fast
// wrong answer cannot be reported as a win, NaN-propagating probes, and the
// RESOLVED ROUTE printed on every row so a pin that did not take is visible
// (route_resolve.hh:165 falls through to automatic() at :175, so an unsupported
// forced route silently becomes the vendor).
//
// THE ORACLE IS THE HOST, NEVER THE VENDOR, and for getrf it is stronger than a
// residual: the PIVOT SEQUENCE is compared ELEMENTWISE against LAPACKE_?getrf on
// the same matrix. A residual bound is satisfied by ANY valid pivot choice, so a
// kernel that pivots on |z| instead of LAPACK's cabs1, or breaks ties the other
// way, passes every residual test in existence. `pivmm` below is that check.
//
// THE TEST MATRIX IS DIAGONALLY DOMINANT AND THEN ROW-PERMUTED, and that is a
// recorded break rather than a precaution: on the dominant matrix alone, partial
// pivoting selects the diagonal at every step, ipiv is the identity, and both
// BREAK=piv and BREAK=laswp left the baseline residual BIT-IDENTICAL. `ntpiv`
// (non-diagonal pivots on item 0) is the anti-vacuity assertion on the
// CONFIGURATION -- necessary, and not sufficient.

#include <batchlas/blas/functions/getrf.hh>
#include <batchlas/blas/functions/getri.hh>
#include <batchlas/blas/functions/getrs.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>

#include <batchlas/blas/dispatch/vendor_available.hh>
#include "src/backends/getrf_route.hh"
#include "src/backends/getrs_route.hh"
#include "src/backends/getri_route.hh"

#include "src/extensions/getrf_native.hh"
#include "src/extensions/getrs_native.hh"

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

// conj on the PROMOTED scalar: identity for real, std::conj for complex.
static inline double dconj(double x) { return x; }
static inline std::complex<double> dconj(std::complex<double> x) { return std::conj(x); }
static inline double ab(double x) { return std::fabs(x); }
static inline double ab(std::complex<double> x) { return std::abs(x); }
static inline double up(float x) { return double(x); }
static inline double up(double x) { return x; }
static inline std::complex<double> up(std::complex<float> x) { return {double(x.real()), double(x.imag())}; }
static inline std::complex<double> up(std::complex<double> x) { return x; }

// NAN-PROPAGATING max. std::max(a,b) returns a when b is NaN, so a poisoned
// probe reads as a perfect one (WP5 break K5).
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

struct Stat { double med, mean, relsd; };
static Stat stat_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const double med = v[v.size() / 2];
    double mean = 0; for (double x : v) mean += x; mean /= double(v.size());
    double sd = 0; for (double x : v) sd += (x - mean) * (x - mean);
    sd = std::sqrt(sd / double(v.size()));
    return {med, mean, mean > 0 ? sd / mean : 0.0};
}

// THE RESIDUAL THRESHOLD, and it exists because its ABSENCE was measured.
//
// The first version of this harness computed every residual correctly and then
// gated `ok` on isfinite() alone. BREAK=laswp_left (the blocked driver drops the
// row interchange on the already-factorised columns) drove the getrf residual
// from 1.5e-07 to 1.2e-01 -- and the row still printed "ok", with FAILS=0. That
// is this repository's blind-guard class exactly: a probe that computes the right
// number and then does not assert on it. Every pass criterion below now carries a
// bound.
//
// The matrix is diagonally dominant (then row-permuted), so its condition number
// is O(1) and a few hundred eps is a generous bound rather than a tuned one: the
// measured spread across every green cell is 1.5e-07..7e-07 for float/cfloat and
// 4.7e-16..5.4e-15 for double/cdouble.
template <typename T> struct Tol;
template <> struct Tol<float>                { static constexpr double v = 1e-4; };
template <> struct Tol<std::complex<float>>  { static constexpr double v = 1e-4; };
template <> struct Tol<double>               { static constexpr double v = 1e-11; };
template <> struct Tol<std::complex<double>> { static constexpr double v = 1e-11; };

static double warm_s() { const char* e = std::getenv("WARM_S"); return e ? std::atof(e) : 1.0; }
static int nprobe()    { const char* e = std::getenv("NPROBE"); return e ? std::atoi(e) : 2; }

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

// --------------------------------------------------------------- host LAPACK
// The independent factorisation the pivot sequence is compared against.
static int host_getrf(int n, float* a, int lda, int* ip) {
    return LAPACKE_sgetrf(LAPACK_COL_MAJOR, n, n, a, lda, ip);
}
static int host_getrf(int n, double* a, int lda, int* ip) {
    return LAPACKE_dgetrf(LAPACK_COL_MAJOR, n, n, a, lda, ip);
}
static int host_getrf(int n, std::complex<float>* a, int lda, int* ip) {
    return LAPACKE_cgetrf(LAPACK_COL_MAJOR, n, n,
                          reinterpret_cast<lapack_complex_float*>(a), lda, ip);
}
static int host_getrf(int n, std::complex<double>* a, int lda, int* ip) {
    return LAPACKE_zgetrf(LAPACK_COL_MAJOR, n, n,
                          reinterpret_cast<lapack_complex_double*>(a), lda, ip);
}

// --------------------------------------------------------------- probes
// || (P A0) x - L (U x) ||inf / || A0 x ||inf, P rebuilt from the DEVICE pivots.
// Independent of who factored: it reads A0, which no device call ever writes.
template <typename T>
static double getrf_probe(const UnifiedVector<T>& F, const UnifiedVector<T>& A0,
                          const int* piv, int piv_stride, int n, int batch, int np) {
    using D = typename Prom<T>::type;
    const size_t st = size_t(n) * n;
    double worst = 0;
    for (int b : {0, batch - 1}) {
        for (int p = 0; p < np; ++p) {
            Rng rg(uint64_t(b) * 977 + uint64_t(p) * 31 + 7);
            std::vector<D> x(n);
            for (int j = 0; j < n; ++j) x[j] = up(mk<T>(rg.next(), rg.next()));
            std::vector<D> ref(n, D(0));
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                    ref[i] += up(A0[size_t(b) * st + size_t(j) * n + i]) * x[j];
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
                D acc = y[i];                       // unit diagonal
                for (int j = 0; j < i; ++j) acc += up(F[size_t(b) * st + size_t(j) * n + i]) * y[j];
                z[i] = acc;
            }
            double num = 0, den = 0;
            for (int i = 0; i < n; ++i) { num = nanmax(num, ab(z[i] - ref[i])); den = nanmax(den, ab(ref[i])); }
            worst = nanmax(worst, den > 0 ? num / den : num);
        }
    }
    return worst;
}

// || op(A0) X[:,j] - B0[:,j] ||inf / || B0[:,j] ||inf, on a few RHS columns.
// `tr` selects which system the residual is formed against, which is the whole
// point of testing the transposed modes at all.
template <typename T>
static double solve_probe(const UnifiedVector<T>& X, const UnifiedVector<T>& B0,
                          const UnifiedVector<T>& A0, int n, int nrhs, int batch,
                          int np, Transpose tr) {
    using D = typename Prom<T>::type;
    const size_t sa = size_t(n) * n, sb = size_t(n) * nrhs;
    double worst = 0;
    for (int b : {0, batch - 1}) {
        for (int p = 0; p < np; ++p) {
            const int j = (p * 7) % nrhs;
            std::vector<D> r(n, D(0));
            for (int i = 0; i < n; ++i) {
                D acc = D(0);
                for (int c = 0; c < n; ++c) {
                    // A0 is column-major: element (row, col) at col*n + row.
                    D aij;
                    if (tr == Transpose::NoTrans)        aij = up(A0[size_t(b) * sa + size_t(c) * n + i]);
                    else if (tr == Transpose::Trans)     aij = up(A0[size_t(b) * sa + size_t(i) * n + c]);
                    else                                 aij = dconj(up(A0[size_t(b) * sa + size_t(i) * n + c]));
                    acc += aij * up(X[size_t(b) * sb + size_t(j) * n + c]);
                }
                r[i] = acc;
            }
            double num = 0, den = 0;
            for (int i = 0; i < n; ++i) {
                const D bi = up(B0[size_t(b) * sb + size_t(j) * n + i]);
                num = nanmax(num, ab(r[i] - bi)); den = nanmax(den, ab(bi));
            }
            worst = nanmax(worst, den > 0 ? num / den : num);
        }
    }
    return worst;
}

// || A0 (C e) - e ||inf / ||e||inf.
template <typename T>
static double inv_probe(const UnifiedVector<T>& C, const UnifiedVector<T>& A0,
                        int n, int batch, int np) {
    using D = typename Prom<T>::type;
    const size_t st = size_t(n) * n;
    double worst = 0;
    for (int b : {0, batch - 1}) {
        for (int p = 0; p < np; ++p) {
            Rng rg(uint64_t(b) * 613 + uint64_t(p) * 17 + 3);
            std::vector<D> e(n);
            for (int j = 0; j < n; ++j) e[j] = up(mk<T>(rg.next(), rg.next()));
            std::vector<D> z(n, D(0)), w(n, D(0));
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i) z[i] += up(C[size_t(b) * st + size_t(j) * n + i]) * e[j];
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i) w[i] += up(A0[size_t(b) * st + size_t(j) * n + i]) * z[j];
            double num = 0, den = 0;
            for (int i = 0; i < n; ++i) { num = nanmax(num, ab(w[i] - e[i])); den = nanmax(den, ab(e[i])); }
            worst = nanmax(worst, den > 0 ? num / den : num);
        }
    }
    return worst;
}

static int nontrivial_pivots(const int* piv, int n) {
    int c = 0;
    for (int k = 0; k < n; ++k) if (piv[k] != k + 1) ++c;
    return c;
}

// --------------------------------------------------------------- setup
// Diagonally dominant, then ROW-PERMUTED per item. See the file header.
template <typename T>
static void fill_A0(UnifiedVector<T>& A0, int n, int batch, uint64_t seed) {
    const size_t sa = size_t(n) * n;
    Rng rg(seed);
    for (size_t i = 0; i < A0.size(); ++i) A0[i] = mk<T>(rg.next(), rg.next());
    std::vector<T> col(n);
    for (int b = 0; b < batch; ++b) {
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

// ---- the getrs A/B body; spliced onto lubench6.cpp's scaffolding by build.py.
//
// WP8-I2 A/B: the composed getrs driver's ROW PERMUTATION, two spellings,
// INTERLEAVED INSIDE ONE PROCESS.
//
// WHY A NEW HARNESS RATHER THAN TWO RUNS OF lubench6. GATE-B asks for the two
// arms interleaved within one session. Two DRIVER SPELLINGS are neither two
// routes nor two builds, so no pin and no library swap can separate them -- but
// BATCHLAS_GETRS_LASWP is re-read on every call once it is PRESENT in the
// environment (src/extensions/getrs_native.cc's perm_spelling(); presence
// latches, the value does not), so setenv() between reps is a real per-rep
// interleave. Everything above this line is lubench6.cpp's, copied by script
// rather than re-derived: the diagonally dominant then row-permuted matrix, the
// host LAPACKE factorisation, the NaN-propagating solve probe, Tol<T>, the
// resolved-route column.
//
// THE EXTRA ORACLE THIS HARNESS HAS AND lubench6 DOES NOT: the two arms compute
// the SAME permutation of the same buffer followed by the SAME two trsm calls,
// so their SOLUTIONS must be BIT-IDENTICAL. That is a stronger statement than
// either residual, and it is exactly what the collapse claims.
//
// ANTI-VACUITY, and it is not optional here for TWO independent reasons:
//   * the env presence flag latches, so a mis-ordered harness would run one arm
//     twice and report a flat 1.00x;
//   * the gather FALLS BACK to the walk when the tile does not fit, silently.
// Both are read back from sycl_getrs::getrs_perm_spelling_debug<T>, which
// resolves through the driver's own functions, and a row whose two arms report
// the SAME spelling is marked BAD.

static double warm_s2() { const char* e = std::getenv("WARM_S"); return e ? std::atof(e) : 1.0; }
static int nprobe2()    { const char* e = std::getenv("NPROBE"); return e ? std::atoi(e) : 1; }

template <typename T>
static int run_ab(const char* tn, int n, int nrhs, int batch, int reps,
                  const std::string& armA, const std::string& armB, Transpose tr) {
    // The knob must be PRESENT before the first getrs call.
    setenv("BATCHLAS_GETRS_LASWP", armA.c_str(), 1);

    auto q = std::make_shared<Queue>(Device("gpu"), BE);
    const size_t sa = size_t(n) * n, sb = size_t(n) * nrhs;

    UnifiedVector<T> A0(sa * batch), A(sa * batch);
    UnifiedVector<int64_t> piv(size_t(n) * batch);
    UnifiedVector<int32_t> info(batch);
    fill_A0<T>(A0, n, batch, 12345);

    UnifiedVector<T*> pA0(batch), pA(batch);
    MatrixView<T, MatrixFormat::Dense> A0v(A0.data(), n, n, n, int(sa), batch, pA0.data());
    MatrixView<T, MatrixFormat::Dense> Av(A.data(), n, n, n, int(sa), batch, pA.data());
    MatrixView<T, MatrixFormat::Dense>::copy(*q, Av, A0v);
    q->wait();

    const size_t f_ws = getrf_buffer_size<BE, T>(*q, Av);
    UnifiedVector<std::byte> fws(f_ws ? f_ws : 1);
    getrf<BE, T>(*q, Av, piv.to_span(), fws.to_span(), info.to_span());
    q->wait();
    const int* pivi = reinterpret_cast<const int*>(piv.data());
    const int ntp = nontrivial_pivots(pivi, n);
    const double fres = getrf_probe<T>(A, A0, pivi, n, n, batch, nprobe2());
    const std::string rf = rstr(backend::getrf_route<BE, T>(*q, Av, kVendorF));

    // B0 is the right-hand side; X is the buffer solved in place. XA keeps arm
    // A's answer for the bit-identity check.
    UnifiedVector<T> B0(sb * batch), X(sb * batch), XA(sb * batch);
    { Rng rg(777); for (size_t i = 0; i < B0.size(); ++i) B0[i] = mk<T>(rg.next(), rg.next()); }
    UnifiedVector<T*> pB0(batch), pX(batch);
    MatrixView<T, MatrixFormat::Dense> B0v(B0.data(), n, nrhs, n, int(sb), batch, pB0.data());
    MatrixView<T, MatrixFormat::Dense> Xv(X.data(), n, nrhs, n, int(sb), batch, pX.data());

    const size_t s_ws = getrs_buffer_size<BE, T>(*q, Av, Xv, tr);
    UnifiedVector<std::byte> sws(s_ws ? s_ws : 1);
    const std::string rs = rstr(backend::getrs_route<BE, T>(*q, Av, Xv, tr, kVendorF));

    auto arm = [&](const std::string& a) { setenv("BATCHLAS_GETRS_LASWP", a.c_str(), 1); };
    auto spell = [&](const std::string& a) {
        arm(a);
        return sycl_getrs::getrs_perm_spelling_debug<T>(*q, n, nrhs);
    };
    const int sA = spell(armA), sB = spell(armB);

    // The RHS copy is OUTSIDE the timed region -- the same discipline lubench6's
    // getrs mode uses, because the copy is not part of the op.
    auto once = [&]() {
        MatrixView<T, MatrixFormat::Dense>::copy(*q, Xv, B0v);
        q->wait();
        const auto t0 = std::chrono::steady_clock::now();
        getrs<BE, T>(*q, Av, Xv, tr, piv.to_span(), sws.to_span());
        q->wait();
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    };

    // WARM BOTH ARMS: JIT, clocks, first touch. Untimed and discarded.
    const auto w0 = std::chrono::steady_clock::now();
    do { arm(armA); once(); arm(armB); once(); }
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count() < warm_s2());

    arm(armA); once();
    std::memcpy(XA.data(), X.data(), sb * batch * sizeof(T));
    const double resA = solve_probe<T>(X, B0, A0, n, nrhs, batch, nprobe2(), tr);

    arm(armB); once();
    size_t diff = 0;
    if (std::memcmp(XA.data(), X.data(), sb * batch * sizeof(T)) != 0) {
        const unsigned char* p = reinterpret_cast<const unsigned char*>(XA.data());
        const unsigned char* r = reinterpret_cast<const unsigned char*>(X.data());
        for (size_t i = 0; i < sb * batch; ++i)
            if (std::memcmp(p + i * sizeof(T), r + i * sizeof(T), sizeof(T)) != 0) ++diff;
    }
    const double resB = solve_probe<T>(X, B0, A0, n, nrhs, batch, nprobe2(), tr);

    // INTERLEAVED reps, arm by arm, inside the one process.
    std::vector<double> msA, msB;
    for (int r = 0; r < reps; ++r) {
        arm(armA); msA.push_back(once());
        arm(armB); msB.push_back(once());
    }
    const Stat stA = stat_of(msA), stB = stat_of(msB);

    const bool okc = std::isfinite(resA) && resA <= Tol<T>::v &&
                     std::isfinite(resB) && resB <= Tol<T>::v &&
                     std::isfinite(fres) && fres <= Tol<T>::v &&
                     ntp > 0 && diff == 0 && sA != sB;

    // ratio > 1 means arm B (the second name, by convention the NEW arm) is faster.
    std::printf("%s,%d,%d,%d,%s,%s,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.3e,%.3e,%zu,%zu,%s,%d,%s\n",
                tn, n, nrhs, batch, armA.c_str(), armB.c_str(), sA, sB,
                stA.med, stB.med, stA.med / stB.med, stA.relsd, stB.relsd,
                resA, resB, diff, s_ws, (rf + "|" + rs).c_str(), ntp,
                okc ? "ok" : "BAD");
    return okc ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc < 6) {
        std::fprintf(stderr,
            "usage: getrsab <type> <n> <nrhs> <batch> <reps> [armA=walk] [armB=gather] [N|T|C]\n"
            "cols  : type,n,nrhs,batch,armA,armB,spellA,spellB,medA,medB,ratio,"
            "relsdA,relsdB,resA,resB,bitdiff,ws,route,ntpiv,flag\n");
        return 2;
    }
    const std::string t = argv[1];
    const int n = std::atoi(argv[2]), nrhs = std::atoi(argv[3]);
    const int batch = std::atoi(argv[4]), reps = std::atoi(argv[5]);
    const std::string a = (argc > 6) ? argv[6] : "walk";
    const std::string b = (argc > 7) ? argv[7] : "gather";
    const char tc = (argc > 8) ? argv[8][0] : 'N';
    const Transpose tr = (tc == 'T') ? Transpose::Trans
                       : (tc == 'C') ? Transpose::ConjTrans
                                     : Transpose::NoTrans;
    if (t == "float")   return run_ab<float>(t.c_str(), n, nrhs, batch, reps, a, b, tr);
    if (t == "double")  return run_ab<double>(t.c_str(), n, nrhs, batch, reps, a, b, tr);
    if (t == "cfloat")  return run_ab<std::complex<float>>(t.c_str(), n, nrhs, batch, reps, a, b, tr);
    if (t == "cdouble") return run_ab<std::complex<double>>(t.c_str(), n, nrhs, batch, reps, a, b, tr);
    std::fprintf(stderr, "unknown type %s\n", t.c_str());
    return 2;
}
