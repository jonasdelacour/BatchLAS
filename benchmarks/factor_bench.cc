// P0 of docs/design/small-n-factorization-plan.md: the in-tree, correct,
// saturation-aware A/B harness for the batched factorizations
// (potrf / getrf / getrs / geqrf / orgqr) at n = 4..512.
//
// It exists because there was no potrf, getrf or getrs harness in benchmarks/
// at all: every Cholesky and LU number in docs/perf came from standalone
// programs under experiments/ that now live only at the tag
// perf-evidence/vendor-independence. This file ports the proven pieces of three
// of them -- wp4_potrf/phase2_ab/realpotrf.cpp, wp6_lu/bench/lubench6.cpp and
// wp5_qr/bench/qrbench.cpp -- into the tree, so P1..P7 are gated on a harness
// that is reviewed and versioned rather than on a scratch file.
//
// ONE CELL PER PROCESS. The binary takes exactly one (op, type, shape, batch)
// and never loops over shapes internally. That is not a style choice: the SLM
// carve-out attribute is sticky per CUfunction, so an earlier, larger launch in
// the same process can make a later launch that should have failed succeed, and
// the result then depends on iteration order. run_factor_grid.sh forks per cell.
//
// WHAT IT REFUSES TO DO. It never reports a timing it has not checked in the
// same process. Every timed arm is followed by an untimed run of the same route
// whose output is verified on the host, in double (or complex<double>)
// promotion, on items 0 AND batch-1 -- item 0 alone is blind to a wrong batch
// stride. A row that fails any gate carries bad=1 and a reason, and the caller
// is expected to drop it rather than quote it.
//
//   usage: factor_bench <op> <type> <m> <n> <nrhs> <batch> <reps>
//                       [--route=<pin>] [--csv=<path>] [--arms=vendor,native]
//   env:   WARM_S (seconds, default 1.5), LD_PAD (ld = m + LD_PAD)
//
#include <batchlas/blas/functions/geqrf.hh>
#include <batchlas/blas/functions/getrf.hh>
#include <batchlas/blas/functions/getrs.hh>
#include <batchlas/blas/functions/orgqr.hh>
#include <batchlas/blas/functions/potrf.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>

#include <batchlas/blas/dispatch/route_env.hh>

#include <lapacke.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

using namespace batchlas;
static constexpr Backend BE = Backend::CUDA;

// ------------------------------------------------------------- promotion
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

// NAN-PROPAGATING max. std::max(a, b) returns `a` when the comparison against a
// NaN `b` is false, so a poisoned probe reads as a perfect one -- the exact
// defect (WP5 break K5) that once printed 4.788e-07 over garbage. Every
// worst-case accumulation below goes through this, never through std::max.
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
    // uniform in [-1, 1)
    double next() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return double(int32_t(uint32_t(s >> 32))) / 2147483648.0;
    }
    double uni01() { const double u = next() * 0.5 + 0.5; return u <= 0.0 ? 1e-12 : u; }
    double gauss() { return std::sqrt(-2.0 * std::log(uni01())) * std::cos(6.283185307179586 * uni01()); }
};

// THE RESIDUAL BOUND, and it exists because its ABSENCE was measured: an earlier
// LU harness computed every residual correctly and then gated only on
// isfinite(), so a driver that dropped a row interchange drove the residual from
// 1.5e-07 to 1.2e-01 and the row still printed "ok". Every number below carries
// a bound. The inputs are conditioned O(1) by construction, so a few hundred eps
// is generous rather than tuned.
template <typename T> struct Tol;
template <> struct Tol<float>                { static constexpr double v = 1e-4; };
template <> struct Tol<std::complex<float>>  { static constexpr double v = 1e-4; };
template <> struct Tol<double>               { static constexpr double v = 1e-11; };
template <> struct Tol<std::complex<double>> { static constexpr double v = 1e-11; };

static double warm_s() { const char* e = std::getenv("WARM_S"); return e ? std::atof(e) : 1.5; }
static int ld_pad()    { const char* e = std::getenv("LD_PAD"); return e ? std::atoi(e) : 0; }

struct Stat { double med = 0, mean = 0, relsd = 0; };
static Stat stat_of(std::vector<double> v) {
    if (v.empty()) return {};
    std::sort(v.begin(), v.end());
    Stat s;
    s.med = v[v.size() / 2];
    for (double x : v) s.mean += x;
    s.mean /= double(v.size());
    double sd = 0;
    for (double x : v) sd += (x - s.mean) * (x - s.mean);
    sd = std::sqrt(sd / double(v.size()));
    s.relsd = s.mean > 0 ? sd / s.mean : 0.0;
    return s;
}

// ------------------------------------------------------------- host LAPACK
// The independent factorisation getrf's PIVOT SEQUENCE is compared against.
// A residual bound is satisfied by ANY valid pivot choice, so a kernel that
// pivots on |z| instead of LAPACK's cabs1, or breaks ties the other way, passes
// every residual test in existence. This is the check that does not.
static int host_getrf(int m, int n, float* a, int lda, int* ip) {
    return LAPACKE_sgetrf(LAPACK_COL_MAJOR, m, n, a, lda, ip);
}
static int host_getrf(int m, int n, double* a, int lda, int* ip) {
    return LAPACKE_dgetrf(LAPACK_COL_MAJOR, m, n, a, lda, ip);
}
static int host_getrf(int m, int n, std::complex<float>* a, int lda, int* ip) {
    return LAPACKE_cgetrf(LAPACK_COL_MAJOR, m, n,
                          reinterpret_cast<lapack_complex_float*>(a), lda, ip);
}
static int host_getrf(int m, int n, std::complex<double>* a, int lda, int* ip) {
    return LAPACKE_zgetrf(LAPACK_COL_MAJOR, m, n,
                          reinterpret_cast<lapack_complex_double*>(a), lda, ip);
}

// ------------------------------------------------------------- route pins
enum class OpKind { potrf, getrf, getrs, geqrf, orgqr };

static const char* pin_variable(OpKind k) {
    switch (k) {
        case OpKind::potrf: return "BATCHLAS_POTRF_ROUTE";
        case OpKind::getrf: return "BATCHLAS_GETRF_ROUTE";
        case OpKind::getrs: return "BATCHLAS_GETRS_ROUTE";
        case OpKind::geqrf: return "BATCHLAS_GEQRF_ROUTE";
        case OpKind::orgqr: return "BATCHLAS_ORGQR_ROUTE";
    }
    return "";
}
static dispatch::Op dispatch_op(OpKind k) {
    switch (k) {
        case OpKind::potrf: return dispatch::Op::potrf;
        case OpKind::getrf: return dispatch::Op::getrf;
        case OpKind::getrs: return dispatch::Op::getrs;
        case OpKind::geqrf: return dispatch::Op::geqrf;
        case OpKind::orgqr: return dispatch::Op::orgqr;
    }
    return dispatch::Op::COUNT;
}
static const char* op_text(OpKind k) {
    switch (k) {
        case OpKind::potrf: return "potrf";
        case OpKind::getrf: return "getrf";
        case OpKind::getrs: return "getrs";
        case OpKind::geqrf: return "geqrf";
        case OpKind::orgqr: return "orgqr";
    }
    return "?";
}

// PIN_PARSED SAYS THE VALUE WAS UNDERSTOOD, NOT THAT THE ROUTE TOOK. Route
// resolution falls through to automatic() when a forced route does not support
// the shape, so `--route=cta` on an order the CTA tier cannot hold reports
// pin_parsed=1 and then silently runs whatever automatic() picks -- in a vendor
// build, the vendor. The RESOLVED route is a separate readback:
// run_factor_grid.sh re-runs each cell once, untimed, with BATCHLAS_COVERAGE_OUT
// set and greps the `reached,` row. Never read this column as "the route ran".
//
// It is queried INSIDE the ScopedEnvVar scope on purpose: settings() is a
// pre-main snapshot, so a raw ::setenv is invisible to it and only
// ScopedEnvVar's reload_settings() makes the pin readable at all.
static bool pin_parsed_now(OpKind k) {
    const auto p = dispatch::parse_route_env(dispatch_op(k));
    return p.found && !p.unparsed;
}

// ------------------------------------------------------------- inputs
// potrf: diagonally dominant SPD, condition number close to 1, EXACTLY as the
// archived realpotrf.cpp built it -- so any nonzero info is the implementation
// and not the input.
template <typename T>
static void fill_spd(UnifiedVector<T>& A0, int n, int ld, size_t stride, int batch) {
    for (int b = 0; b < batch; ++b)
        for (int c = 0; c < n; ++c)
            for (int r = 0; r < n; ++r) {
                const double v = (r == c) ? double(n) + 1.0 : 0.5 / (1.0 + std::abs(r - c));
                A0[size_t(b) * stride + size_t(c) * size_t(ld) + size_t(r)] = mk<T>(v, 0.0);
            }
}

// getrf/getrs: diagonally dominant, THEN ROW-PERMUTED per item. The permutation
// is a recorded break rather than a precaution: on the dominant matrix alone
// partial pivoting picks the diagonal at every step, ipiv is the identity, and
// both a broken pivot search and a dropped row interchange left the residual
// BIT-IDENTICAL. `nontrivial_pivots` below is the anti-vacuity check on the
// configuration -- necessary, and not sufficient.
template <typename T>
static void fill_lu(UnifiedVector<T>& A0, int n, int ld, size_t stride, int batch, uint64_t seed) {
    Rng rg(seed);
    const size_t nn = size_t(n);
    std::vector<T> col(nn);
    std::vector<int> perm(nn);
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < n; ++c)
            for (int r = 0; r < n; ++r)
                A0[size_t(b) * stride + size_t(c) * size_t(ld) + size_t(r)] = mk<T>(rg.next(), rg.next());
        for (int i = 0; i < n; ++i) {
            const size_t d = size_t(b) * stride + size_t(i) * size_t(ld) + size_t(i);
            A0[d] = A0[d] + mk<T>(double(n), 0.0);
        }
        for (int i = 0; i < n; ++i) perm[size_t(i)] = i;
        for (int i = n - 1; i > 0; --i) {
            const int j = int((rg.next() * 0.5 + 0.5) * double(i + 1)) % (i + 1);
            std::swap(perm[size_t(i)], perm[size_t(j)]);
        }
        for (int c = 0; c < n; ++c) {
            for (int i = 0; i < n; ++i)
                col[size_t(i)] = A0[size_t(b) * stride + size_t(c) * size_t(ld) + size_t(perm[size_t(i)])];
            for (int i = 0; i < n; ++i)
                A0[size_t(b) * stride + size_t(c) * size_t(ld) + size_t(i)] = col[size_t(i)];
        }
    }
}

// geqrf/orgqr: random Gaussian, fixed seed.
template <typename T>
static void fill_gauss(UnifiedVector<T>& A0, int m, int n, int ld, size_t stride, int batch, uint64_t seed) {
    Rng rg(seed);
    for (int b = 0; b < batch; ++b)
        for (int c = 0; c < n; ++c)
            for (int r = 0; r < m; ++r)
                A0[size_t(b) * stride + size_t(c) * size_t(ld) + size_t(r)] = mk<T>(rg.gauss(), rg.gauss());
}

// ------------------------------------------------------------- host probes
// All of them: Frobenius norms, double promotion, items 0 AND batch-1.

// || A0 - L L^H ||_F / || A0 ||_F over the FACTORED (lower) triangle.
template <typename T>
static double potrf_residual(const UnifiedVector<T>& F, const UnifiedVector<T>& A0,
                             int n, int ld, size_t stride, int batch) {
    using D = typename Prom<T>::type;
    double worst = 0;
    for (int b : {0, batch - 1}) {
        const size_t o = size_t(b) * stride;
        double num = 0, den = 0;
        for (int j = 0; j < n; ++j)
            for (int i = j; i < n; ++i) {
                D acc = D(0);
                for (int k = 0; k <= j; ++k)
                    acc += up(F[o + size_t(k) * size_t(ld) + size_t(i)]) *
                           cj(up(F[o + size_t(k) * size_t(ld) + size_t(j)]));
                const D a = up(A0[o + size_t(j) * size_t(ld) + size_t(i)]);
                const double d = ab(acc - a), r = ab(a);
                num += d * d;
                den += r * r;
            }
        if (std::isnan(num) || std::isnan(den)) return std::numeric_limits<double>::quiet_NaN();
        worst = nanmax(worst, den > 0 ? std::sqrt(num) / std::sqrt(den) : std::sqrt(num));
    }
    return worst;
}

// || P A0 - L U ||_F / || A0 ||_F, P rebuilt from the DEVICE pivots.
template <typename T>
static double getrf_residual(const UnifiedVector<T>& F, const UnifiedVector<T>& A0,
                             const int* piv, int pstride, int n, int ld, size_t stride, int batch) {
    using D = typename Prom<T>::type;
    double worst = 0;
    const size_t npa = size_t(n) * size_t(n);
    std::vector<D> PA(npa);
    for (int b : {0, batch - 1}) {
        const size_t o = size_t(b) * stride;
        for (int c = 0; c < n; ++c)
            for (int r = 0; r < n; ++r)
                PA[size_t(c) * size_t(n) + size_t(r)] = up(A0[o + size_t(c) * size_t(ld) + size_t(r)]);
        const int* pv = piv + size_t(b) * size_t(pstride);
        for (int k = 0; k < n; ++k) {
            const int ip = pv[k] - 1;                       // LAPACK 1-based
            if (ip < 0 || ip >= n) return std::numeric_limits<double>::quiet_NaN();
            if (ip != k)
                for (int c = 0; c < n; ++c)
                    std::swap(PA[size_t(c) * size_t(n) + size_t(k)],
                              PA[size_t(c) * size_t(n) + size_t(ip)]);
        }
        double num = 0, den = 0;
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                D acc = D(0);
                const int kmax = std::min(i, j);
                for (int k = 0; k <= kmax; ++k) {
                    const D l = (k == i) ? D(1) : up(F[o + size_t(k) * size_t(ld) + size_t(i)]);
                    acc += l * up(F[o + size_t(j) * size_t(ld) + size_t(k)]);
                }
                const D a = PA[size_t(j) * size_t(n) + size_t(i)];
                const double d = ab(acc - a), r = ab(a);
                num += d * d;
                den += r * r;
            }
        if (std::isnan(num) || std::isnan(den)) return std::numeric_limits<double>::quiet_NaN();
        worst = nanmax(worst, den > 0 ? std::sqrt(num) / std::sqrt(den) : std::sqrt(num));
    }
    return worst;
}

static int nontrivial_pivots(const int* piv, int n) {
    int c = 0;
    for (int k = 0; k < n; ++k) if (piv[k] != k + 1) ++c;
    return c;
}

// || A0 X - B0 ||_F / (|| A0 ||_F || X ||_F)
template <typename T>
static double getrs_residual(const UnifiedVector<T>& X, const UnifiedVector<T>& B0,
                             const UnifiedVector<T>& A0, int n, int nrhs, int lda, size_t sa,
                             int ldb, size_t sb, int batch) {
    using D = typename Prom<T>::type;
    double worst = 0;
    for (int b : {0, batch - 1}) {
        const size_t oa = size_t(b) * sa, ob = size_t(b) * sb;
        double na = 0, nx = 0, num = 0;
        for (int c = 0; c < n; ++c)
            for (int r = 0; r < n; ++r) {
                const double v = ab(up(A0[oa + size_t(c) * size_t(lda) + size_t(r)]));
                na += v * v;
            }
        for (int c = 0; c < nrhs; ++c)
            for (int r = 0; r < n; ++r) {
                const double v = ab(up(X[ob + size_t(c) * size_t(ldb) + size_t(r)]));
                nx += v * v;
            }
        for (int c = 0; c < nrhs; ++c)
            for (int r = 0; r < n; ++r) {
                D acc = D(0);
                for (int k = 0; k < n; ++k)
                    acc += up(A0[oa + size_t(k) * size_t(lda) + size_t(r)]) *
                           up(X[ob + size_t(c) * size_t(ldb) + size_t(k)]);
                const double d = ab(acc - up(B0[ob + size_t(c) * size_t(ldb) + size_t(r)]));
                num += d * d;
            }
        if (std::isnan(num) || std::isnan(na) || std::isnan(nx))
            return std::numeric_limits<double>::quiet_NaN();
        const double den = std::sqrt(na) * std::sqrt(nx);
        worst = nanmax(worst, den > 0 ? std::sqrt(num) / den : std::sqrt(num));
    }
    return worst;
}

// The explicit Q (m x k) for one item, from the packed reflectors:
// Q = H_0 H_1 ... H_{k-1} applied to the first k columns of I_m.
template <typename T>
static void form_Q(const UnifiedVector<T>& F, const UnifiedVector<T>& tau,
                   int b, int m, int k, int ld, size_t stride, size_t taustride,
                   std::vector<typename Prom<T>::type>& Q) {
    using D = typename Prom<T>::type;
    const size_t o = size_t(b) * stride;
    Q.assign(size_t(m) * size_t(k), D(0));
    for (int j = 0; j < k; ++j) Q[size_t(j) * size_t(m) + size_t(j)] = D(1);
    for (int step = 0; step < k; ++step) {
        const int i = k - 1 - step;
        const D t = up(tau[taustride * size_t(b) + size_t(i)]);
        for (int c = 0; c < k; ++c) {
            D s = Q[size_t(c) * size_t(m) + size_t(i)];      // v(i) == 1 implicitly
            for (int r = i + 1; r < m; ++r)
                s += cj(up(F[o + size_t(i) * size_t(ld) + size_t(r)])) *
                     Q[size_t(c) * size_t(m) + size_t(r)];
            Q[size_t(c) * size_t(m) + size_t(i)] -= t * s;
            for (int r = i + 1; r < m; ++r)
                Q[size_t(c) * size_t(m) + size_t(r)] -=
                    t * up(F[o + size_t(i) * size_t(ld) + size_t(r)]) * s;
        }
    }
}

// || Q^H Q - I ||_F over the k columns of an explicit column-major Q.
template <class D>
static double ortho_norm(const D* Q, int m, int k, int lq) {
    double num = 0;
    for (int a = 0; a < k; ++a)
        for (int c = 0; c < k; ++c) {
            D acc = D(0);
            for (int r = 0; r < m; ++r)
                acc += cj(Q[size_t(a) * size_t(lq) + size_t(r)]) * Q[size_t(c) * size_t(lq) + size_t(r)];
            const double d = ab(acc - D(a == c ? 1 : 0));
            num += d * d;
        }
    return std::sqrt(num);
}

// ------------------------------------------------------------- one arm
struct Arm {
    std::string name;      // "vendor" | "native"
    std::string pin;       // the value actually written to BATCHLAS_<OP>_ROUTE
    bool pin_parsed = false;
    Stat st;
    double residual = std::numeric_limits<double>::quiet_NaN();
    double extra = 0;
    int info_nonzero = 0;
    int bad = 0;
    std::string reason;
};

static void flag(Arm& a, const char* why) {
    a.bad = 1;
    if (!a.reason.empty()) a.reason += "+";
    a.reason += why;
}

static void gate(Arm& a, double tol, int reps) {
    if (!std::isfinite(a.residual)) flag(a, "residual_nonfinite");
    else if (a.residual > tol) flag(a, "residual");
    if (!std::isfinite(a.extra)) flag(a, "extra_nonfinite");
    if (a.info_nonzero != 0) flag(a, "info");
    if (reps > 1 && a.st.relsd > 0.10) flag(a, "relsd");
    if (a.bad == 0) a.reason = "ok";
}

struct Cfg {
    OpKind op = OpKind::potrf;
    std::string type;
    int m = 0, n = 0, nrhs = 0, batch = 0, reps = 0;
    std::string route_pin;      // --route=, applied to the NATIVE arm only
    std::string csv;
    bool want_vendor = true, want_native = true;
};

static void emit(const Cfg& c, const Arm& a, std::FILE* csv) {
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "%s,%s,%d,%d,%d,%d,%s,%s,%d,%.6f,%.6f,%.4f,%d,%.3e,%.3e,%d,%d,%s\n",
                  op_text(c.op), c.type.c_str(), c.m, c.n, c.nrhs, c.batch,
                  a.name.c_str(), a.pin.c_str(), a.pin_parsed ? 1 : 0,
                  a.st.med, a.st.mean, a.st.relsd, c.reps,
                  a.residual, a.extra, a.info_nonzero, a.bad, a.reason.c_str());
    std::fputs(buf, stdout);
    if (csv) std::fputs(buf, csv);
}

// ------------------------------------------------------------- driver
template <typename T>
static int run(const Cfg& c) {
    auto q = std::make_shared<Queue>(Device("gpu"), BE);
    using MV = MatrixView<T, MatrixFormat::Dense>;

    const int m = c.m, n = c.n, nrhs = c.nrhs, batch = c.batch;
    const int pad = ld_pad();
    const int lda = m + pad;                       // the strided-ld trap, honoured
    const size_t sa = size_t(lda) * size_t(n);
    const int ldb = n + pad;
    const size_t sb = size_t(ldb) * size_t(nrhs > 0 ? nrhs : 1);
    const int kmin = std::min(m, n);
    const bool needs_factor = (c.op == OpKind::getrs || c.op == OpKind::orgqr);

    const size_t nbatch = size_t(batch);
    UnifiedVector<T> A0(sa * nbatch), A(sa * nbatch);
    // THESE ARE VIEWS OVER THE CALLER'S BUFFERS, never Matrix: Matrix's
    // (const T*, ...) constructor COPIES into its own storage, so factorising a
    // Matrix built that way leaves the array this program checks untouched --
    // which reads as "info == 0 with a huge residual", a wrong-answer report for
    // a correct kernel.
    // And EVERY VIEW GETS ITS OWN POINTER ARRAY: the vendor batched paths call
    // data_ptrs(ctx) and throw "data_ptrs target is null" on a view built
    // without one, and sharing a single array between two views is a recorded
    // aliasing trap.
    UnifiedVector<T*> pA0(nbatch), pA(nbatch);
    MV A0v(A0.data(), m, n, lda, int(sa), batch, pA0.data());
    MV Av(A.data(), m, n, lda, int(sa), batch, pA.data());

    UnifiedVector<int32_t> info(nbatch, 0);
    UnifiedVector<int64_t> piv(size_t(n) * nbatch, 0);
    // getrf packs 1-based int32 pivots in the FIRST HALF of the int64 span
    // (src/extensions/getrf_native.hh), n per item.
    const int* pivi = reinterpret_cast<const int*>(piv.data());
    const int pstride = n;
    UnifiedVector<T> tau(size_t(kmin) * nbatch);

    const size_t bsz = (c.op == OpKind::getrs) ? sb * nbatch : size_t(1);
    UnifiedVector<T> B0(bsz), X(bsz);
    UnifiedVector<T*> pB0(nbatch), pX(nbatch);
    MV B0v, Xv;
    if (c.op == OpKind::getrs) {
        B0v = MV(B0.data(), n, nrhs, ldb, int(sb), batch, pB0.data());
        Xv  = MV(X.data(),  n, nrhs, ldb, int(sb), batch, pX.data());
        Rng rg(777);
        for (int b = 0; b < batch; ++b)
            for (int cc = 0; cc < nrhs; ++cc)
                for (int r = 0; r < n; ++r)
                    B0[size_t(b) * sb + size_t(cc) * size_t(ldb) + size_t(r)] = mk<T>(rg.next(), rg.next());
    }

    switch (c.op) {
        case OpKind::potrf: fill_spd<T>(A0, n, lda, sa, batch); break;
        case OpKind::getrf:
        case OpKind::getrs: fill_lu<T>(A0, n, lda, sa, batch, 12345); break;
        case OpKind::geqrf:
        case OpKind::orgqr: fill_gauss<T>(A0, m, n, lda, sa, batch, 12345); break;
    }

    auto reset_A = [&] { (void)MV::copy(*q, Av, A0v); q->wait(); };

    // getrs and orgqr both consume a factorisation. Produce it ONCE, untimed,
    // through the public entry point under the AUTOMATIC route, and keep a host
    // copy, so both arms are handed byte-identical input every rep.
    const size_t fsz = needs_factor ? sa * nbatch : size_t(1);
    UnifiedVector<T> F(fsz);
    if (needs_factor) {
        reset_A();
        if (c.op == OpKind::getrs) {
            const size_t fw = getrf_buffer_size<BE, T>(*q, Av);
            const size_t fwsz = fw ? fw : size_t(1);
            UnifiedVector<std::byte> fws(fwsz);
            (void)getrf<BE, T>(*q, Av, piv.to_span(), fws.to_span(), info.to_span());
        } else {
            const size_t gw = geqrf_buffer_size<BE, T>(*q, Av, tau.to_span());
            const size_t gwsz = gw ? gw : size_t(1);
            UnifiedVector<std::byte> gws(gwsz);
            (void)geqrf<BE, T>(*q, Av, tau.to_span(), gws.to_span());
        }
        q->wait();
        std::memcpy(F.data(), A.data(), sa * nbatch * sizeof(T));
    }

    // FACTORIZATION IS DESTRUCTIVE, so the input is restored before EVERY rep.
    // For getrs the factored A must survive instead, so the reset restores the
    // right-hand side; for orgqr it restores the packed reflectors.
    auto reset = [&] {
        if (c.op == OpKind::getrs) { (void)MV::copy(*q, Xv, B0v); q->wait(); }
        else if (c.op == OpKind::orgqr) { std::memcpy(A.data(), F.data(), sa * nbatch * sizeof(T)); }
        else reset_A();
    };

    std::vector<Arm> arms;
    if (c.want_vendor) { Arm a; a.name = "vendor"; a.pin = "vendor"; arms.push_back(a); }
    if (c.want_native) { Arm a; a.name = "native"; a.pin = c.route_pin.empty() ? "native" : c.route_pin; arms.push_back(a); }
    if (arms.empty()) { std::fprintf(stderr, "factor_bench: no arms selected\n"); return 2; }

    const char* var = pin_variable(c.op);

    // The workspace is sized per arm INSIDE that arm's pin and the MAXIMUM is
    // allocated: the two routes do not agree on how much they need, and handing
    // a native tier a vendor-sized workspace is the recorded "108x too small"
    // defect. Over-allocating is safe; under-allocating is not.
    size_t wneed = 0;
    for (size_t i = 0; i < arms.size(); ++i) {
        ScopedEnvVar pinned(var, arms[i].pin.c_str());
        arms[i].pin_parsed = pin_parsed_now(c.op);
        size_t need = 0;
        switch (c.op) {
            case OpKind::potrf: need = potrf_buffer_size<BE, T>(*q, Av, Uplo::Lower); break;
            case OpKind::getrf: need = getrf_buffer_size<BE, T>(*q, Av); break;
            case OpKind::getrs: need = getrs_buffer_size<BE, T>(*q, Av, Xv, Transpose::NoTrans); break;
            case OpKind::geqrf: need = geqrf_buffer_size<BE, T>(*q, Av, tau.to_span()); break;
            case OpKind::orgqr: need = orgqr_buffer_size<BE, T>(*q, Av, tau.to_span()); break;
        }
        wneed = std::max(wneed, need);
    }
    const size_t wsz = wneed ? wneed : size_t(1);
    UnifiedVector<std::byte> ws(wsz);

    auto call = [&] {
        switch (c.op) {
            case OpKind::potrf: (void)potrf<BE, T>(*q, Av, Uplo::Lower, ws.to_span(), info.to_span()); break;
            case OpKind::getrf: (void)getrf<BE, T>(*q, Av, piv.to_span(), ws.to_span(), info.to_span()); break;
            case OpKind::getrs: (void)getrs<BE, T>(*q, Av, Xv, Transpose::NoTrans, piv.to_span(), ws.to_span()); break;
            case OpKind::geqrf: (void)geqrf<BE, T>(*q, Av, tau.to_span(), ws.to_span()); break;
            case OpKind::orgqr: (void)orgqr<BE, T>(*q, Av, tau.to_span(), ws.to_span()); break;
        }
        q->wait();
    };

    // TIME-BASED WARM-UP, DISCARDED. A cold first run -- SYCL JIT plus cold
    // clocks -- has fabricated a 3.7x result in this repository.
    //
    // IT IS INTERLEAVED, in the same arm order the timed loop uses, and that is
    // measured rather than stylistic. With a per-arm warm-up (all of arm 0, then
    // all of arm 1) the first TIMED rep of arm 0 is the only one in the run not
    // preceded by an arm-1 call, and it came in 2.2x slow every time -- 0.0567
    // ms against a steady 0.0261 at potrf float n=8 -- which alone pushed the
    // vendor arm's rel_sd to 0.27 and tripped the gate on a cell whose median
    // was perfectly stable. Warming in the timed loop's own order removes it.
    // WARM_S is per arm, so the loop runs for WARM_S x arms.
    {
        const double budget = warm_s() * double(arms.size());
        const auto w0 = std::chrono::steady_clock::now();
        do {
            for (size_t i = 0; i < arms.size(); ++i) {
                ScopedEnvVar pinned(var, arms[i].pin.c_str());
                reset();
                call();
            }
        } while (std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count() < budget);
    }

    // INTERLEAVED A/B IN ONE PROCESS: every rep runs every arm back to back, so
    // clock drift hits both arms equally. Medians are taken per arm afterwards.
    std::vector<std::vector<double>> ms(arms.size());
    for (int r = 0; r < c.reps; ++r) {
        for (size_t i = 0; i < arms.size(); ++i) {
            ScopedEnvVar pinned(var, arms[i].pin.c_str());
            reset();
            const auto t0 = std::chrono::steady_clock::now();
            call();
            const auto t1 = std::chrono::steady_clock::now();
            ms[i].push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
    }

    // DUMP_MS prints the raw per-rep times. rel_sd is a gate, so it has to be
    // auditable: a cell rejected for rel_sd is either a noisy neighbour, a
    // single cold outlier, or a genuinely bimodal route, and the median alone
    // cannot tell those apart.
    if (std::getenv("DUMP_MS")) {
        for (size_t i = 0; i < arms.size(); ++i) {
            std::fprintf(stderr, "  raw %s:", arms[i].name.c_str());
            for (double v : ms[i]) std::fprintf(stderr, " %.4f", v);
            std::fprintf(stderr, "\n");
        }
    }

    // CORRECTNESS IN THE SAME PROCESS, per arm, from one extra UNTIMED run of
    // that arm's route -- so a fast wrong answer cannot be reported as a win.
    for (size_t i = 0; i < arms.size(); ++i) {
        Arm& a = arms[i];
        a.st = stat_of(ms[i]);
        {
            ScopedEnvVar pinned(var, a.pin.c_str());
            for (int b = 0; b < batch; ++b) info[b] = 0;
            reset();
            call();
        }
        switch (c.op) {
            case OpKind::potrf:
                a.residual = potrf_residual<T>(A, A0, n, lda, sa, batch);
                break;
            case OpKind::getrf: {
                a.residual = getrf_residual<T>(A, A0, pivi, pstride, n, lda, sa, batch);
                // The DISCRIMINATING oracle: the pivot sequence, elementwise,
                // against an independent host xGETRF on the same input.
                int mism = 0;
                const size_t nn = size_t(n);
                std::vector<T> h(nn * nn);
                std::vector<int> hp(nn);
                for (int b : {0, batch - 1}) {
                    for (int cc = 0; cc < n; ++cc)
                        std::memcpy(h.data() + size_t(cc) * size_t(n),
                                    A0.data() + size_t(b) * sa + size_t(cc) * size_t(lda),
                                    size_t(n) * sizeof(T));
                    host_getrf(n, n, h.data(), n, hp.data());
                    for (int k = 0; k < n; ++k)
                        if (hp[size_t(k)] != pivi[size_t(b) * size_t(pstride) + size_t(k)]) ++mism;
                }
                a.extra = double(mism);
                if (mism != 0) flag(a, "pivot_mismatch");
                if (nontrivial_pivots(pivi, n) == 0) flag(a, "vacuous_pivots");
                break;
            }
            case OpKind::getrs:
                a.residual = getrs_residual<T>(X, B0, A0, n, nrhs, lda, sa, ldb, sb, batch);
                break;
            case OpKind::geqrf: {
                using D = typename Prom<T>::type;
                std::vector<D> Q;
                double worst = 0, orth = 0;
                for (int b : {0, batch - 1}) {
                    form_Q<T>(A, tau, b, m, kmin, lda, sa, size_t(kmin), Q);
                    orth = nanmax(orth, ortho_norm<D>(Q.data(), m, kmin, m));
                    const size_t o = size_t(b) * sa;
                    double num = 0, den = 0;
                    for (int j = 0; j < n; ++j)
                        for (int r = 0; r < m; ++r) {
                            D acc = D(0);
                            const int kk = std::min(kmin, j + 1);
                            for (int k = 0; k < kk; ++k)
                                acc += Q[size_t(k) * size_t(m) + size_t(r)] *
                                       up(A[o + size_t(j) * size_t(lda) + size_t(k)]);
                            const D a0 = up(A0[o + size_t(j) * size_t(lda) + size_t(r)]);
                            const double d = ab(acc - a0), rr = ab(a0);
                            num += d * d;
                            den += rr * rr;
                        }
                    if (std::isnan(num) || std::isnan(den)) {
                        worst = std::numeric_limits<double>::quiet_NaN();
                        break;
                    }
                    worst = nanmax(worst, den > 0 ? std::sqrt(num) / std::sqrt(den) : std::sqrt(num));
                }
                a.residual = worst;
                a.extra = orth;
                if (!(orth <= Tol<T>::v)) flag(a, "orthogonality");
                break;
            }
            case OpKind::orgqr: {
                using D = typename Prom<T>::type;
                double orth = 0;
                const size_t qsz = size_t(m) * size_t(n);
                std::vector<D> Q(qsz);
                for (int b : {0, batch - 1}) {
                    const size_t o = size_t(b) * sa;
                    for (int cc = 0; cc < n; ++cc)
                        for (int r = 0; r < m; ++r)
                            Q[size_t(cc) * size_t(m) + size_t(r)] =
                                up(A[o + size_t(cc) * size_t(lda) + size_t(r)]);
                    orth = nanmax(orth, ortho_norm<D>(Q.data(), m, n, m));
                }
                a.residual = orth;   // orgqr has no factorisation residual of its own
                a.extra = orth;
                break;
            }
        }
        if (c.op == OpKind::potrf || c.op == OpKind::getrf)
            for (int b = 0; b < batch; ++b) if (info[b] != 0) ++a.info_nonzero;
        gate(a, Tol<T>::v, c.reps);
    }

    std::FILE* csv = nullptr;
    if (!c.csv.empty()) {
        std::FILE* probe = std::fopen(c.csv.c_str(), "r");
        const bool fresh = (probe == nullptr);
        if (probe) std::fclose(probe);
        csv = std::fopen(c.csv.c_str(), "a");
        if (csv && fresh)
            std::fputs("op,type,m,n,nrhs,batch,arm,pin,pin_parsed,median_ms,mean_ms,"
                       "rel_sd,reps,residual,extra_check,info_nonzero,bad,reason\n", csv);
    }
    int rc = 0;
    for (const Arm& a : arms) { emit(c, a, csv); rc |= a.bad; }
    if (csv) std::fclose(csv);
    return rc;
}

// ------------------------------------------------------------- main
int main(int argc, char** argv) {
    if (argc < 8) {
        std::fprintf(stderr,
            "usage: factor_bench <op> <type> <m> <n> <nrhs> <batch> <reps>\n"
            "                    [--route=<pin>] [--csv=<path>] [--arms=vendor,native]\n"
            "  op   : potrf getrf getrs geqrf orgqr\n"
            "  type : float double cfloat cdouble\n"
            "  env  : WARM_S (seconds, default 1.5), LD_PAD (ld = m + LD_PAD)\n"
            "prints: op,type,m,n,nrhs,batch,arm,pin,pin_parsed,median_ms,mean_ms,"
            "rel_sd,reps,residual,extra_check,info_nonzero,bad,reason\n");
        return 2;
    }
    Cfg c;
    const std::string opn = argv[1];
    if      (opn == "potrf") c.op = OpKind::potrf;
    else if (opn == "getrf") c.op = OpKind::getrf;
    else if (opn == "getrs") c.op = OpKind::getrs;
    else if (opn == "geqrf") c.op = OpKind::geqrf;
    else if (opn == "orgqr") c.op = OpKind::orgqr;
    else { std::fprintf(stderr, "factor_bench: unknown op %s\n", opn.c_str()); return 2; }

    c.type  = argv[2];
    c.m     = std::atoi(argv[3]);
    c.n     = std::atoi(argv[4]);
    c.nrhs  = std::atoi(argv[5]);
    c.batch = std::atoi(argv[6]);
    c.reps  = std::atoi(argv[7]);
    for (int i = 8; i < argc; ++i) {
        const std::string a = argv[i];
        if (a.rfind("--route=", 0) == 0) c.route_pin = a.substr(8);
        else if (a.rfind("--csv=", 0) == 0) c.csv = a.substr(6);
        else if (a.rfind("--arms=", 0) == 0) {
            const std::string v = a.substr(7);
            c.want_vendor = v.find("vendor") != std::string::npos;
            c.want_native = v.find("native") != std::string::npos;
        } else { std::fprintf(stderr, "factor_bench: unknown flag %s\n", a.c_str()); return 2; }
    }
    if (c.m <= 0 || c.n <= 0 || c.batch <= 0 || c.reps <= 0) {
        std::fprintf(stderr, "factor_bench: need m>0 n>0 batch>0 reps>0\n");
        return 2;
    }
    if (c.op == OpKind::getrs && c.nrhs <= 0) {
        std::fprintf(stderr, "factor_bench: getrs needs nrhs > 0\n");
        return 2;
    }
    if ((c.op == OpKind::potrf || c.op == OpKind::getrf || c.op == OpKind::getrs) && c.m != c.n) {
        std::fprintf(stderr, "factor_bench: %s is square; m must equal n\n", opn.c_str());
        return 2;
    }
    if (c.n > c.m) {
        std::fprintf(stderr, "factor_bench: this harness assumes n <= m\n");
        return 2;
    }

    if (c.type == "float")   return run<float>(c);
    if (c.type == "double")  return run<double>(c);
    if (c.type == "cfloat")  return run<std::complex<float>>(c);
    if (c.type == "cdouble") return run<std::complex<double>>(c);
    std::fprintf(stderr, "factor_bench: unknown type %s\n", c.type.c_str());
    return 2;
}
