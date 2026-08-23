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

// --------------------------------------------------------------- driver
template <typename T>
static int run(const std::string& mode, const char* tn, int n, int nrhs, int batch, int reps) {
    auto q = std::make_shared<Queue>(Device("gpu"), BE);
    const size_t sa = size_t(n) * n;
    const size_t sb = size_t(n) * nrhs;

    UnifiedVector<T> A0(sa * batch), A(sa * batch);
    UnifiedVector<int64_t> piv(size_t(n) * batch);
    UnifiedVector<int32_t> info(batch);
    fill_A0<T>(A0, n, batch, 12345);

    UnifiedVector<T*> pA0(batch), pA(batch);
    MatrixView<T, MatrixFormat::Dense> A0v(A0.data(), n, n, n, int(sa), batch, pA0.data());
    MatrixView<T, MatrixFormat::Dense> Av(A.data(), n, n, n, int(sa), batch, pA.data());
    auto reset_A = [&] { MatrixView<T, MatrixFormat::Dense>::copy(*q, Av, A0v); q->wait(); };

    const size_t f_ws = getrf_buffer_size<BE, T>(*q, Av);
    UnifiedVector<std::byte> fws(f_ws ? f_ws : 1);
    const int* pivi = reinterpret_cast<const int*>(piv.data());
    const int pstride = n;   // packed int32, n per item, at the FRONT of the int64 buffer

    const std::string rf = rstr(backend::getrf_route<BE, T>(*q, Av, kVendorF));
    const double dn = double(n);

    if (mode == "getrf") {
        // WARM the JIT and the clocks; a cold run once fabricated a 3.7x loss.
        const auto w0 = std::chrono::steady_clock::now();
        do { reset_A(); getrf<BE, T>(*q, Av, piv.to_span(), fws.to_span(), info.to_span()); q->wait(); }
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count() < warm_s());
        std::vector<double> ms;
        for (int r = 0; r < reps; ++r) {
            reset_A();
            const auto t0 = std::chrono::steady_clock::now();
            getrf<BE, T>(*q, Av, piv.to_span(), fws.to_span(), info.to_span());
            q->wait();
            ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        const Stat s = stat_of(ms);
        int bad = 0; for (int b = 0; b < batch; ++b) if (info[b] != 0) ++bad;
        const double res = getrf_probe<T>(A, A0, pivi, pstride, n, batch, nprobe());
        const int ntp = nontrivial_pivots(pivi, n);

        // THE DISCRIMINATING ORACLE: the pivot sequence, elementwise, against an
        // independent host xGETRF, on the FIRST and LAST batch items.
        int pivmm = 0, infomm = 0;
        {
            std::vector<T> h(sa);
            std::vector<int> hp(n);
            for (int b : {0, batch - 1}) {
                std::memcpy(h.data(), A0.data() + size_t(b) * sa, sa * sizeof(T));
                const int hi = host_getrf(n, h.data(), n, hp.data());
                for (int k = 0; k < n; ++k)
                    if (hp[k] != pivi[size_t(b) * pstride + k]) ++pivmm;
                if (hi != info[b]) ++infomm;
            }
        }
        const bool ok = std::isfinite(res) && res <= Tol<T>::v && bad == 0 && ntp > 0 &&
                        pivmm == 0 && infomm == 0;
        std::printf("getrf,%s,%d,%d,%d,%.4f,%.4f,%.4f,%.2f,%.3e,%zu,%s,%d,%d,%d,%d,%s\n",
                    tn, n, nrhs, batch, s.med, s.mean, s.relsd,
                    double(batch) * (2.0 / 3.0) * dn * dn * dn / (s.med * 1e6),
                    res, f_ws, rf.c_str(), bad, ntp, pivmm, infomm, ok ? "ok" : "BAD");
        return ok ? 0 : 1;
    }

    // Every remaining mode needs a FACTORED A. Produce it once, UNTIMED, through
    // the SAME public getrf (and therefore the same pinned route), so a native
    // getrs/getri is tested against the pivot buffer a native getrf wrote.
    reset_A();
    getrf<BE, T>(*q, Av, piv.to_span(), fws.to_span(), info.to_span());
    q->wait();
    const double fres = getrf_probe<T>(A, A0, pivi, pstride, n, batch, nprobe());
    const int ntp = nontrivial_pivots(pivi, n);

    if (mode == "getri") {
        UnifiedVector<T> C(sa * batch);
        UnifiedVector<T*> pC(batch);
        MatrixView<T, MatrixFormat::Dense> Cv(C.data(), n, n, n, int(sa), batch, pC.data());
        const size_t i_ws = getri_buffer_size<BE, T>(*q, Av);
        UnifiedVector<std::byte> iws(i_ws ? i_ws : 1);
        const std::string ri = rstr(backend::getri_route<BE, T>(*q, Av, kVendorF));

        // A MUST NOT BE WRITTEN (cuBLAS takes `const T* const A[]`; measured
        // max|A_after - A_factored| == 0). Snapshot it and diff after.
        std::vector<T> Asnap(sa * batch);
        std::memcpy(Asnap.data(), A.data(), sa * batch * sizeof(T));

        const auto w0 = std::chrono::steady_clock::now();
        do { getri<BE, T>(*q, Av, Cv, piv.to_span(), iws.to_span(), info.to_span()); q->wait(); }
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count() < warm_s());
        std::vector<double> ms;
        for (int r = 0; r < reps; ++r) {
            const auto t0 = std::chrono::steady_clock::now();
            getri<BE, T>(*q, Av, Cv, piv.to_span(), iws.to_span(), info.to_span());
            q->wait();
            ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        const Stat s = stat_of(ms);
        double adiff = 0;
        for (size_t i = 0; i < sa * size_t(batch); ++i)
            adiff = nanmax(adiff, ab(up(A[i]) - up(Asnap[i])));
        const double res = inv_probe<T>(C, A0, n, batch, nprobe());
        int bad = 0; for (int b = 0; b < batch; ++b) if (info[b] != 0) ++bad;
        const bool ok = std::isfinite(res) && res <= Tol<T>::v &&
                        std::isfinite(fres) && fres <= Tol<T>::v &&
                        ntp > 0 && adiff == 0.0 && bad == 0;
        std::printf("getri,%s,%d,%d,%d,%.4f,%.4f,%.4f,%.2f,%.3e,%zu,%s,%.3e,%d,%.1e,%s\n",
                    tn, n, nrhs, batch, s.med, s.mean, s.relsd,
                    double(batch) * (4.0 / 3.0) * dn * dn * dn / (s.med * 1e6),
                    res, i_ws, (rf + "|" + ri).c_str(), fres, ntp, adiff, ok ? "ok" : "BAD");
        return ok ? 0 : 1;
    }

    if (mode == "getrs") {
        // ALL THREE transA MODES in one row. The scaffolding's break B8 measured
        // that no test in the suite issues a Trans getrs at all, and the
        // transposed case is where the permutation moves to the OUTPUT, in
        // REVERSE -- a silently wrong answer no NoTrans test can see.
        UnifiedVector<T> B0(sb * batch), X(sb * batch);
        { Rng rg(777); for (size_t i = 0; i < B0.size(); ++i) B0[i] = mk<T>(rg.next(), rg.next()); }
        UnifiedVector<T*> pB0(batch), pX(batch);
        MatrixView<T, MatrixFormat::Dense> B0v(B0.data(), n, nrhs, n, int(sb), batch, pB0.data());
        MatrixView<T, MatrixFormat::Dense> Xv(X.data(), n, nrhs, n, int(sb), batch, pX.data());

        const Transpose trs[3] = {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans};
        const char* trn[3] = {"N", "T", "C"};
        // NTRANS=1 runs only NoTrans. The A/B grid uses it because the transposed
        // modes are a CORRECTNESS question (the permutation moves to the output,
        // in reverse) and not a timing one -- they cost the same two solves -- and
        // paying three warm-ups per cell in a timing run buys nothing. The
        // correctness sweep leaves it at 3.
        const int ntrans = [] { const char* e = std::getenv("NTRANS");
                                const int v = e ? std::atoi(e) : 3;
                                return (v >= 1 && v <= 3) ? v : 3; }();
        double resmax = 0;
        std::string rs;
        size_t s_ws_report = 0;   // WP6 bench: the getrs workspace, which the
                                  // kernels/ harness printed as a hard 0.
        double med = 0, mean = 0, relsd = 0;
        for (int ti = 0; ti < ntrans; ++ti) {
            const Transpose tr = trs[ti];
            const size_t s_ws = getrs_buffer_size<BE, T>(*q, Av, Xv, tr);
            UnifiedVector<std::byte> sws(s_ws ? s_ws : 1);
            if (ti == 0) rs = rstr(backend::getrs_route<BE, T>(*q, Av, Xv, tr, kVendorF));
            if (ti == 0) s_ws_report = s_ws;

            auto once = [&] {
                MatrixView<T, MatrixFormat::Dense>::copy(*q, Xv, B0v);
                q->wait();
                getrs<BE, T>(*q, Av, Xv, tr, piv.to_span(), sws.to_span());
                q->wait();
            };
            const auto w0 = std::chrono::steady_clock::now();
            do { once(); }
            while (std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count() < warm_s());
            std::vector<double> ms;
            for (int r = 0; r < reps; ++r) {
                MatrixView<T, MatrixFormat::Dense>::copy(*q, Xv, B0v);
                q->wait();
                const auto t0 = std::chrono::steady_clock::now();
                getrs<BE, T>(*q, Av, Xv, tr, piv.to_span(), sws.to_span());
                q->wait();
                ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            }
            const Stat s = stat_of(ms);
            if (ti == 0) { med = s.med; mean = s.mean; relsd = s.relsd; }
            const double r = solve_probe<T>(X, B0, A0, n, nrhs, batch, nprobe(), tr);
            resmax = nanmax(resmax, r);
            std::fprintf(stderr, "    getrs %s %s n=%d nrhs=%d batch=%d : resid %.3e\n",
                         tn, trn[ti], n, nrhs, batch, r);
        }
        const bool ok = std::isfinite(resmax) && resmax <= Tol<T>::v &&
                        std::isfinite(fres) && fres <= Tol<T>::v && ntp > 0;
        std::printf("getrs,%s,%d,%d,%d,%.4f,%.4f,%.4f,%.2f,%.3e,%zu,%s,%.3e,%d,%s\n",
                    tn, n, nrhs, batch, med, mean, relsd,
                    double(batch) * 2.0 * dn * dn * double(nrhs) / (med * 1e6),
                    resmax, s_ws_report, (rf + "|" + rs).c_str(), fres, ntp, ok ? "ok" : "BAD");
        return ok ? 0 : 1;
    }

    std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 2;
}

// --------------------------------------------------------------- singular
// EXACT-ZERO info semantics, per item, 1-based, GLOBAL column index, and the
// failed item must stay FINITE. Item 1's row 1 is exactly 2x its row 0, so step
// 2 cancels to a true binary zero -- the same construction the interface-contract
// probe used to measure cuBLAS reporting info == 2 where the host reports 2.
template <typename T>
static int run_singular(const char* tn, int n, int batch) {
    auto q = std::make_shared<Queue>(Device("gpu"), BE);
    const size_t sa = size_t(n) * n;
    UnifiedVector<T> A0(sa * batch), A(sa * batch);
    UnifiedVector<int64_t> piv(size_t(n) * batch);
    UnifiedVector<int32_t> info(batch);
    fill_A0<T>(A0, n, batch, 4242);
    // Row 1 := 2 * row 0, on item 1 only.
    for (int j = 0; j < n; ++j)
        A0[sa * 1 + size_t(j) * n + 1] = A0[sa * 1 + size_t(j) * n + 0] + A0[sa * 1 + size_t(j) * n + 0];

    UnifiedVector<T*> pA0(batch), pA(batch);
    MatrixView<T, MatrixFormat::Dense> A0v(A0.data(), n, n, n, int(sa), batch, pA0.data());
    MatrixView<T, MatrixFormat::Dense> Av(A.data(), n, n, n, int(sa), batch, pA.data());
    MatrixView<T, MatrixFormat::Dense>::copy(*q, Av, A0v);
    q->wait();

    const size_t f_ws = getrf_buffer_size<BE, T>(*q, Av);
    UnifiedVector<std::byte> fws(f_ws ? f_ws : 1);
    // Poison info so a kernel that never writes it cannot pass.
    for (int b = 0; b < batch; ++b) info[b] = -12345;
    const std::string rf = rstr(backend::getrf_route<BE, T>(*q, Av, kVendorF));
    getrf<BE, T>(*q, Av, piv.to_span(), fws.to_span(), info.to_span());
    q->wait();

    int mism = 0, nonfinite = 0;
    std::vector<int> hinfo(batch);
    std::vector<T> h(sa); std::vector<int> hp(n);
    for (int b = 0; b < batch; ++b) {
        std::memcpy(h.data(), A0.data() + size_t(b) * sa, sa * sizeof(T));
        hinfo[b] = host_getrf(n, h.data(), n, hp.data());
        if (hinfo[b] != info[b]) ++mism;
        for (size_t i = 0; i < sa; ++i)
            if (!std::isfinite(ab(up(A[size_t(b) * sa + i])))) ++nonfinite;
    }
    // The smallest |U(i,i)| on the singular item, both sides, so an
    // exact-zero DISAGREEMENT can be told apart from a wrong answer: the
    // predicate "U(i,i) == 0" is not stable across two different orders of the
    // same arithmetic, and the vendor diverges from the host on it too
    // (measured: cuBLAS produced U(3,3) = -1.375e-08 and info = 0 where LAPACKE
    // got a true 0.0 and reported 3).
    double dmin = 1e300;
    for (int i = 0; i < n; ++i)
        dmin = std::min(dmin, ab(up(A[sa * 1 + size_t(i) * n + i])));
    std::printf("singular,%s,%d,-,%d,-,-,-,-,-,%zu,%s,dev_info=[", tn, n, batch, f_ws, rf.c_str());
    for (int b = 0; b < batch; ++b) std::printf("%d%s", info[b], b + 1 < batch ? " " : "");
    std::printf("],host_info=[");
    for (int b = 0; b < batch; ++b) std::printf("%d%s", hinfo[b], b + 1 < batch ? " " : "");
    // THE PASS CRITERION IS STRUCTURAL, NOT "AGREES WITH THE HOST", AND THAT IS
    // A MEASURED DECISION RATHER THAN A WEAKENING.
    //
    // Exact-zero is not a stable predicate across two different orders of the
    // same arithmetic, and this probe measured all three implementations
    // disagreeing with each other on it for COMPLEX types on the same matrix:
    //     cfloat  : native |U(6,6)| = 0 -> info 6 ; host LAPACKE 0 ; cuBLAS 9.78e-10 -> 0
    //     cdouble : native |U(6,6)| = 0 -> info 6 ; host 6         ; cuBLAS 2.93e-18 -> 0
    // i.e. cuBLAS itself mismatches the host oracle at cdouble. So "device info
    // == host info" cannot be the gate; what CAN be, and is, is that
    //   * info is EXACT-ZERO semantics and not a tolerance -- non-zero exactly
    //     when |U(i,i)| is a true binary zero;
    //   * the failed item stays FINITE (LAPACK skips the reciprocal scale rather
    //     than dividing by zero);
    //   * the non-singular items report 0, from a span PRE-POISONED with -12345,
    //     so a kernel that never writes info cannot pass;
    //   * info is 1-BASED and a GLOBAL column index, which the cross-tier
    //     agreement between native:cta and native:blocked pins.
    // The host and vendor values are printed rather than asserted.
    const bool zero_iff_info = (dmin == 0.0) == (info[1] != 0);
    bool others_clean = true;
    for (int b = 0; b < batch; ++b) if (b != 1 && info[b] != 0) others_clean = false;
    const bool ok = (nonfinite == 0) && zero_iff_info && others_clean;
    std::printf("],min_absUii_item1=%.3e,host_mismatch=%d,nonfinite=%d,%s\n", dmin, mism, nonfinite,
                ok ? "ok" : "BAD");
    return ok ? 0 : 1;
}

// THE PIVOT METRIC, on a matrix built so that cabs1 and the modulus DISAGREE.
//
// This exists because a break measured the ordinary oracle to be blind to it. On
// the random test matrix, replacing LAPACK's cabs1 (|Re| + |Im|) with the true
// modulus changed NOTHING: every pivot still matched host LAPACKE elementwise,
// for both complex types, so BREAK=pivot_metric turned no getrf row red. The two
// rules agree almost everywhere on random data -- separating them needs a matrix
// that puts the two candidates on opposite sides of the two functionals.
//
// Column 0 of every item is (3 + 0i) in row 0 and (2 + 2i) in row 1:
//     cabs1   : 3 vs 4       -> I?AMAX (and cuBLAS, and this kernel) pick ROW 1
//     modulus : 3 vs 2.828   -> a modulus-based argmax picks ROW 0
// so ipiv[0] is 2 under the LAPACK rule and 1 under the other. Everything else is
// a well-conditioned diagonal, so the item is non-singular and the residual stays
// small under EITHER choice -- which is the point: only the pivot sequence can
// see this, and this is the case where it can.
template <typename T>
static int run_pivmetric(const char* tn, int n, int batch) {
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
        std::printf("pivmetric,%s,%d,-,%d,-,-,-,-,-,0,-,skipped-real,0,ok\n", tn, n, batch);
        return 0;
    } else {
        auto q = std::make_shared<Queue>(Device("gpu"), BE);
        const size_t sa = size_t(n) * n;
        UnifiedVector<T> A0(sa * batch), A(sa * batch);
        UnifiedVector<int64_t> piv(size_t(n) * batch);
        UnifiedVector<int32_t> info(batch);
        for (size_t i = 0; i < A0.size(); ++i) A0[i] = mk<T>(0.0, 0.0);
        for (int b = 0; b < batch; ++b) {
            for (int i = 0; i < n; ++i)
                A0[size_t(b) * sa + size_t(i) * n + i] = mk<T>(1.0 + 0.25 * i, 0.0);
            A0[size_t(b) * sa + 0] = mk<T>(3.0, 0.0);            // (0,0) = 3 + 0i
            A0[size_t(b) * sa + 1] = mk<T>(2.0, 2.0);            // (1,0) = 2 + 2i
            A0[size_t(b) * sa + size_t(1) * n + 0] = mk<T>(0.5, 0.25);
        }
        UnifiedVector<T*> pA0(batch), pA(batch);
        MatrixView<T, MatrixFormat::Dense> A0v(A0.data(), n, n, n, int(sa), batch, pA0.data());
        MatrixView<T, MatrixFormat::Dense> Av(A.data(), n, n, n, int(sa), batch, pA.data());
        MatrixView<T, MatrixFormat::Dense>::copy(*q, Av, A0v);
        q->wait();
        const size_t f_ws = getrf_buffer_size<BE, T>(*q, Av);
        UnifiedVector<std::byte> fws(f_ws ? f_ws : 1);
        const std::string rf = rstr(backend::getrf_route<BE, T>(*q, Av, kVendorF));
        getrf<BE, T>(*q, Av, piv.to_span(), fws.to_span(), info.to_span());
        q->wait();
        const int* pivi = reinterpret_cast<const int*>(piv.data());

        int mm = 0;
        std::vector<T> h(sa); std::vector<int> hp(n);
        for (int b : {0, batch - 1}) {
            std::memcpy(h.data(), A0.data() + size_t(b) * sa, sa * sizeof(T));
            (void)host_getrf(n, h.data(), n, hp.data());
            for (int k = 0; k < n; ++k) if (hp[k] != pivi[size_t(b) * size_t(n) + k]) ++mm;
        }
        // ANTI-VACUITY ON THE CONFIGURATION: the first pivot must actually be the
        // one the two metrics disagree about, i.e. row 1 (1-based 2). If it is
        // not, this probe is testing nothing and says so.
        const bool discriminating = (pivi[0] == 2);
        std::printf("pivmetric,%s,%d,-,%d,-,-,-,-,-,%zu,%s,dev_ipiv0=%d,host_ipiv0=%d,mismatch=%d,%s\n",
                    tn, n, batch, f_ws, rf.c_str(), pivi[0], hp[0], mm,
                    (mm == 0 && discriminating) ? "ok" : "BAD");
        return (mm == 0 && discriminating) ? 0 : 1;
    }
}

// The launch geometry the driver would actually use, asked through the SAME pure
// functions the driver calls -- so a test that must straddle a block boundary can
// see where the boundary is, and one that must exercise the GLOBAL panel leaf can
// see that it did. potrf_native.hh:246-266 records the failure this prevents: a
// test that hardcodes the width keeps passing after any of its inputs moves while
// silently no longer testing a short final panel.
template <typename T>
static int run_params(const char* tn, int n) {
    auto q = std::make_shared<Queue>(Device("gpu"), BE);
    const std::size_t lm = q->device().get_property(DeviceProperty::LOCAL_MEM_SIZE);
    const std::size_t budget = lm > 4096 ? lm - 4096 : 0;
    const unsigned w = sycl_getrf::getrf_blocked_debug_params<T>(*q, n);
    std::printf("params,%s,n=%d,local_mem=%zu,budget=%zu,cta_max_n=%d,cta_fits=%d,nb=%u,leading_leaf=%s\n",
                tn, n, lm, budget,
                sycl_getrf::getrf_cta_max_n_for_slm<T>(budget),
                sycl_getrf::getrf_cta_fits<T>(n, budget) ? 1 : 0,
                w & 0xffffu,
                (w >> 16) == 1u ? "resident" : ((w >> 16) == 2u ? "global" : "absent"));
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 7) {
        std::fprintf(stderr,
            "usage: luverify <mode> <type> <n> <nrhs> <batch> <reps>\n"
            "modes : getrf | getrs | getri | singular\n"
            "types : float double cfloat cdouble\n"
            "cols  : op,type,n,nrhs,batch,med_ms,mean_ms,relsd,GFLOPs,resid,ws,route,...,flag\n");
        return 2;
    }
    const std::string mode = argv[1], t = argv[2];
    const int n = std::atoi(argv[3]), nrhs = std::atoi(argv[4]);
    const int b = std::atoi(argv[5]), r = std::atoi(argv[6]);
    try {
        if (mode == "pivmetric") {
            if (t == "float")   return run_pivmetric<float>("float", n, b);
            if (t == "double")  return run_pivmetric<double>("double", n, b);
            if (t == "cfloat")  return run_pivmetric<std::complex<float>>("cfloat", n, b);
            if (t == "cdouble") return run_pivmetric<std::complex<double>>("cdouble", n, b);
        }
        if (mode == "params") {
            if (t == "float")   return run_params<float>("float", n);
            if (t == "double")  return run_params<double>("double", n);
            if (t == "cfloat")  return run_params<std::complex<float>>("cfloat", n);
            if (t == "cdouble") return run_params<std::complex<double>>("cdouble", n);
        }
        if (mode == "singular") {
            if (t == "float")   return run_singular<float>("float", n, b);
            if (t == "double")  return run_singular<double>("double", n, b);
            if (t == "cfloat")  return run_singular<std::complex<float>>("cfloat", n, b);
            if (t == "cdouble") return run_singular<std::complex<double>>("cdouble", n, b);
        }
        if (t == "float")   return run<float>(mode, "float", n, nrhs, b, r);
        if (t == "double")  return run<double>(mode, "double", n, nrhs, b, r);
        if (t == "cfloat")  return run<std::complex<float>>(mode, "cfloat", n, nrhs, b, r);
        if (t == "cdouble") return run<std::complex<double>>(mode, "cdouble", n, nrhs, b, r);
    } catch (const std::exception& e) {
        std::printf("%s,%s,%d,%d,%d,THREW,%s\n", mode.c_str(), t.c_str(), n, nrhs, b, e.what());
        return 3;
    }
    std::fprintf(stderr, "unknown type %s\n", t.c_str());
    return 2;
}
