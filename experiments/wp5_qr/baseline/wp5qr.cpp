// WP5 QR baseline: time the SHIPPED geqrf / orgqr / ormqr PUBLIC API.
//
// Modelled on experiments/wp4_potrf/phase2_ab/realpotrf.cpp: public entry
// points only, no forced routes, correctness checked in the SAME PROCESS so a
// fast wrong answer cannot be reported as a win. Built twice -- once against
// build/ (vendor present) and once against build-novendor/ -- so "vendor-free"
// means the vendor-free BUILD, not an environment variable inside a build that
// still has cuSOLVER linked in.
//
// Why not benchmarks/geqrf_benchmark.cc: it is registered with
// BATCHLAS_REGISTER_BENCHMARK (float/double only, no complex), it never checks
// the answer, and its GFLOP count is 2mn^2 + (2/3)n^3, which is the wrong sign
// on the second term (LAPACK's geqrf is 2mn^2 - (2/3)n^3).
#include <batchlas/blas/functions/geqrf.hh>
#include <batchlas/blas/functions/orgqr.hh>
#include <batchlas/blas/functions/ormqr.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace batchlas;
static constexpr Backend BE = Backend::CUDA;

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

template <class T> static inline T mk(double re, double im);
template <> inline float mk<float>(double re, double) { return float(re); }
template <> inline double mk<double>(double re, double) { return re; }
template <> inline std::complex<float> mk<std::complex<float>>(double re, double im) { return {float(re), float(im)}; }
template <> inline std::complex<double> mk<std::complex<double>>(double re, double im) { return {re, im}; }

// deterministic LCG in [-1,1)
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 6364136223846793005ULL + 1442695040888963407ULL) {}
    double next() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return double(int32_t(uint32_t(s >> 32))) / 2147483648.0;
    }
};

// ---------------------------------------------------------------- timing
struct Stat { double med, mean, relsd; };
static Stat stat_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const double med = v[v.size() / 2];
    double mean = 0; for (double x : v) mean += x; mean /= double(v.size());
    double sd = 0; for (double x : v) sd += (x - mean) * (x - mean);
    sd = std::sqrt(sd / double(v.size()));
    return {med, mean, mean > 0 ? sd / mean : 0.0};
}

static double warm_s() {
    const char* e = std::getenv("WARM_S");
    return e ? std::atof(e) : 1.5;
}

// BREAK=<n> damages the CHECKER'S REFERENCE, so that a green control and a red
// break together prove the residual can discriminate. This repo has shipped
// five tests that could not fail by construction; a residual that is never
// shown failing is the same defect in a harness.
//   1  drop the LAST reflector             (the sy2sb short-final-panel class)
//   2  apply the reflectors in WY order     (reversed reflector order)
//   3  drop the last COLUMN of the explicit Q
//   4  conjugate tau                        (the complex phase-convention class)
static int break_mode() {
    const char* e = std::getenv("BREAK");
    return e ? std::atoi(e) : 0;
}

// ---------------------------------------------------------------- checks
// Probe-vector residual for the geqrf FACTOR (reflectors + R still packed in F).
// || Q R x - A x ||inf / || A x ||inf, worst over `nprobe` random x and the
// two batch items 0 and batch-1. O(n^2) per probe, so it is affordable at
// n = 2048 where a dense host reconstruction is not.
template <typename T>
static double geqrf_probe(const UnifiedVector<T>& F, const UnifiedVector<T>& A0,
                          const UnifiedVector<T>& tau, int m, int n, int batch,
                          int nprobe) {
    using D = typename Prom<T>::type;
    const int k = std::min(m, n);
    const size_t st = size_t(m) * n;
    double worst = 0;
    for (int b : {0, batch - 1}) {
        for (int p = 0; p < nprobe; ++p) {
            Rng rg(uint64_t(b) * 977 + uint64_t(p) * 31 + 7);
            std::vector<D> x(n);
            for (int j = 0; j < n; ++j) x[j] = up(mk<T>(rg.next(), rg.next()));
            std::vector<D> ref(m, D(0)), y(m, D(0));
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < m; ++i)
                    ref[i] += up(A0[size_t(b) * st + size_t(j) * m + i]) * x[j];
            // y = R x  (R = upper triangle of F, k x n)
            for (int i = 0; i < k; ++i) {
                D acc = D(0);
                for (int j = i; j < n; ++j) acc += up(F[size_t(b) * st + size_t(j) * m + i]) * x[j];
                y[i] = acc;
            }
            // y = H_1 H_2 ... H_k y, applied right-to-left
            const int bm = break_mode();
            const int lo = (bm == 1) ? 1 : 0;         // BREAK 1: drop the LAST reflector (H_k)
            const int skip = (bm == 5) ? k / 2 : -1;  // BREAK 5: drop a MIDDLE reflector
            for (int step = lo; step < k; ++step) {
                const int i = (bm == 2) ? step : (k - 1 - step);   // BREAK 2: reversed order
                if (i == skip) continue;
                const D t0 = up(tau[size_t(b) * size_t(k) + size_t(i)]);
                const D t = (bm == 4) ? cj(t0) : t0;              // BREAK 4: conjugated tau
                D s = y[i];                       // v_i = 1 implicit
                for (int l = i + 1; l < m; ++l) s += cj(up(F[size_t(b) * st + size_t(i) * m + l])) * y[l];
                y[i] -= t * s;
                for (int l = i + 1; l < m; ++l) y[l] -= t * up(F[size_t(b) * st + size_t(i) * m + l]) * s;
            }
            double num = 0, den = 0;
            for (int i = 0; i < m; ++i) { num = std::max(num, ab(y[i] - ref[i])); den = std::max(den, ab(ref[i])); }
            worst = std::max(worst, den > 0 ? num / den : num);
        }
    }
    return worst;
}

// Orthonormality of an explicit Q (m x k): || Q^H Q x - x ||inf / ||x||inf.
template <typename T>
static double ortho_probe(const UnifiedVector<T>& Q, int m, int n, int k, int batch, int nprobe) {
    using D = typename Prom<T>::type;
    const size_t st = size_t(m) * n;
    double worst = 0;
    for (int b : {0, batch - 1}) {
        for (int p = 0; p < nprobe; ++p) {
            Rng rg(uint64_t(b) * 613 + uint64_t(p) * 17 + 3);
            std::vector<D> x(k);
            for (int j = 0; j < k; ++j) x[j] = up(mk<T>(rg.next(), rg.next()));
            std::vector<D> z(m, D(0)), w(k, D(0));
            const int kk = (break_mode() == 3) ? k - 1 : k;   // BREAK 3: drop Q's last column
            for (int j = 0; j < kk; ++j)
                for (int i = 0; i < m; ++i) z[i] += up(Q[size_t(b) * st + size_t(j) * m + i]) * x[j];
            for (int j = 0; j < k; ++j) {
                D acc = D(0);
                for (int i = 0; i < m; ++i) acc += cj(up(Q[size_t(b) * st + size_t(j) * m + i])) * z[i];
                w[j] = acc;
            }
            double num = 0, den = 0;
            for (int j = 0; j < k; ++j) { num = std::max(num, ab(w[j] - x[j])); den = std::max(den, ab(x[j])); }
            worst = std::max(worst, den > 0 ? num / den : num);
        }
    }
    return worst;
}

// || Q R x - A x ||inf / || A x ||inf with an EXPLICIT Q -- catches a Q that is
// orthonormal but is not the Q of this A.
template <typename T>
static double qr_probe(const UnifiedVector<T>& Q, const UnifiedVector<T>& F,
                       const UnifiedVector<T>& A0, int m, int n, int batch, int nprobe) {
    using D = typename Prom<T>::type;
    const int k = std::min(m, n);
    const size_t st = size_t(m) * n;
    double worst = 0;
    for (int b : {0, batch - 1}) {
        for (int p = 0; p < nprobe; ++p) {
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
            const int kk = (break_mode() == 3) ? k - 1 : k;   // BREAK 3: drop Q's last column
            for (int j = 0; j < kk; ++j)
                for (int i = 0; i < m; ++i) z[i] += up(Q[size_t(b) * st + size_t(j) * m + i]) * y[j];
            double num = 0, den = 0;
            for (int i = 0; i < m; ++i) { num = std::max(num, ab(z[i] - ref[i])); den = std::max(den, ab(ref[i])); }
            worst = std::max(worst, den > 0 ? num / den : num);
        }
    }
    return worst;
}

// ---------------------------------------------------------------- synthetic
// SYNTHETIC REFLECTORS, so the orgqr-via-ormqr candidate can be timed in the
// VENDOR-FREE BUILD, where there is no geqrf to produce real ones.
//
// The strict lower triangle of F holds v_i (with the implicit 1 at row i), and
// tau_i = 2 / (v_i^H v_i) makes each H_i = I - tau v v^H Hermitian AND unitary,
// so the product is unitary and ortho_probe is a real check on the result. The
// same trick the WP5 ormqr ground brief used. ormqr's cost does not depend on
// the VALUES of the reflectors, only on the shape, so this times the same work
// the real thing would.
template <typename T>
static void synth_reflectors(UnifiedVector<T>& F, UnifiedVector<T>& tau,
                             int m, int n, int batch) {
    using D = typename Prom<T>::type;
    const int k = std::min(m, n);
    const size_t st = size_t(m) * n;
    Rng rg(98765);
    for (size_t i = 0; i < F.size(); ++i) F[i] = mk<T>(rg.next(), rg.next());
    for (int b = 0; b < batch; ++b)
        for (int i = 0; i < k; ++i) {
            double nrm = 1.0;   // the implicit v_i(i) = 1
            for (int l = i + 1; l < m; ++l) {
                const double a = ab(up(F[size_t(b) * st + size_t(i) * m + l]));
                nrm += a * a;
            }
            tau[size_t(b) * size_t(k) + size_t(i)] = mk<T>(2.0 / nrm, 0.0);
        }
    (void)sizeof(D);
}

// ---------------------------------------------------------------- driver
template <typename T>
static int run(const std::string& mode, const char* tn, int n, int batch, int reps) {
    auto q = std::make_shared<Queue>(Device("gpu"), BE);
    const int m = n;
    const int k = n;
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

    // synthI never calls geqrf, so it must not QUERY geqrf either: the query is
    // the same facade and throws the same "no route" in a vendor-free build.
    const bool need_geqrf = (mode != "synthI");
    const size_t g_ws = need_geqrf ? geqrf_buffer_size<BE, T>(*q, Av, tau.to_span()) : 0;
    UnifiedVector<std::byte> gws(g_ws ? g_ws : 1);

    auto do_geqrf = [&] { geqrf<BE, T>(*q, Av, tau.to_span(), gws.to_span()); q->wait(); };

    // ---------------- geqrf ----------------
    if (mode == "geqrf") {
        const auto w0 = std::chrono::steady_clock::now();
        do { reset_A(); do_geqrf(); }
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
        const double res = geqrf_probe<T>(A, A0, tau, m, n, batch, 3);
        const double fl = double(batch) * (2.0 * m * double(n) * n - (2.0 / 3.0) * double(n) * n * n);
        std::printf("geqrf,%s,%d,%d,%.4f,%.4f,%.4f,%.2f,%.3e,%zu\n", tn, n, batch,
                    s.med, s.mean, s.relsd, fl / (s.med * 1e6), res, g_ws);
        return 0;
    }

    // Everything below needs the factor, computed once, untimed.
    UnifiedVector<T> F(st * batch);
    const bool synth = (mode == "synthI");
    if (synth) {
        synth_reflectors<T>(F, tau, m, n, batch);
    } else {
        reset_A(); do_geqrf();
        std::memcpy(F.data(), A.data(), F.size() * sizeof(T));
    }
    const double fres = synth ? -1.0 : geqrf_probe<T>(F, A0, tau, m, n, batch, 3);
    if (std::getenv("SHOW_TAU")) {
        std::fprintf(stderr, "SHOW_TAU %s n=%d: |tau[k-1]|=%.6e |tau[k-2]|=%.6e |tau[0]|=%.6e\n",
                     tn, n, ab(up(tau[size_t(k) - 1])), ab(up(tau[size_t(k) - 2])), ab(up(tau[0])));
    }

    UnifiedVector<T*> pF(batch);
    MatrixView<T, MatrixFormat::Dense> Fv(F.data(), m, n, m, int(st), batch, pF.data());
    auto reset_from_F = [&] { MatrixView<T, MatrixFormat::Dense>::copy(*q, Av, Fv); q->wait(); };

    // ---------------- orgqr (vendor: cuSOLVER, per-batch-item loop) ----------------
    UnifiedVector<T> Qref;
    if (mode == "orgqr" || mode == "qcheck") {
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
        const double orth = ortho_probe<T>(A, m, n, k, batch, 3);
        const double rec = qr_probe<T>(A, F, A0, m, n, batch, 3);
        // orgqr flops: 4mnk - 2(m+n)k^2 + 4k^3/3
        const double dm = m, dn = n, dk = k;
        const double fl = double(batch) * (4.0 * dm * dn * dk - 2.0 * (dm + dn) * dk * dk + (4.0 / 3.0) * dk * dk * dk);
        std::printf("orgqr,%s,%d,%d,%.4f,%.4f,%.4f,%.2f,%.3e,%.3e,%.3e,%zu\n", tn, n, batch,
                    s.med, s.mean, s.relsd, fl / (s.med * 1e6), fres, orth, rec, o_ws);
        if (mode == "orgqr") return 0;
        Qref.resize(st * batch);
        std::memcpy(Qref.data(), A.data(), Qref.size() * sizeof(T));
    }

    // ---------------- orgqr-via-ormqr: Q = ormqr(F, I, Left, NoTrans) ----------
    if (mode == "ormqrI" || mode == "qcheck" || mode == "synthI") {
        UnifiedVector<T> C(st * batch), C0(st * batch);
        for (size_t i = 0; i < C0.size(); ++i) C0[i] = mk<T>(0.0, 0.0);
        for (int b = 0; b < batch; ++b)
            for (int i = 0; i < m; ++i) C0[size_t(b) * st + size_t(i) * m + size_t(i)] = mk<T>(1.0, 0.0);
        UnifiedVector<T*> pC(batch), pC0(batch);
        MatrixView<T, MatrixFormat::Dense> Cv(C.data(), m, m, m, int(st), batch, pC.data());
        MatrixView<T, MatrixFormat::Dense> C0v(C0.data(), m, m, m, int(st), batch, pC0.data());
        auto reset_C = [&] { MatrixView<T, MatrixFormat::Dense>::copy(*q, Cv, C0v); q->wait(); };

        const auto chosen = blas::dispatch::detail::ormqr_route<T>(*q, Fv, Side::Left, Transpose::NoTrans);
        const int32_t nb = blas::dispatch::detail::resolve_ormqr_block_size<T>(Fv, 0);
        const size_t m_ws = ormqr_buffer_size<BE, T>(*q, Fv, Cv, Side::Left, Transpose::NoTrans, tau.to_span());
        UnifiedVector<std::byte> mws(m_ws ? m_ws : 1);

        const auto w0 = std::chrono::steady_clock::now();
        do { reset_C(); ormqr<BE, T>(*q, Fv, Cv, Side::Left, Transpose::NoTrans, tau.to_span(), mws.to_span()); q->wait(); }
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count() < warm_s());
        std::vector<double> ms;
        for (int r = 0; r < reps; ++r) {
            reset_C();
            const auto t0 = std::chrono::steady_clock::now();
            ormqr<BE, T>(*q, Fv, Cv, Side::Left, Transpose::NoTrans, tau.to_span(), mws.to_span());
            q->wait();
            ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        const Stat s = stat_of(ms);
        const double orth = ortho_probe<T>(C, m, m, k, batch, 3);
        const double rec = synth ? -1.0 : qr_probe<T>(C, F, A0, m, n, batch, 3);
        double dq = -1.0;
        if (mode == "qcheck") {
            double num = 0, den = 0;
            for (int b : {0, batch - 1})
                for (size_t i = 0; i < st; ++i) {
                    num = std::max(num, ab(up(C[size_t(b) * st + i]) - up(Qref[size_t(b) * st + i])));
                    den = std::max(den, ab(up(Qref[size_t(b) * st + i])));
                }
            dq = den > 0 ? num / den : num;
        }
        const double dm = m, dn = m, dk = k;   // C is m x m
        const double fl = double(batch) * (4.0 * dm * dn * dk - 2.0 * dn * dk * dk);
        std::printf("%s,%s,%d,%d,%.4f,%.4f,%.4f,%.2f,%.3e,%.3e,%.3e,%zu,route=%d:%d,nb=%d,dQ=%.3e\n",
                    mode.c_str(), tn, n, batch, s.med, s.mean, s.relsd, fl / (s.med * 1e6), fres, orth, rec, m_ws,
                    int(chosen.origin), int(chosen.algo), nb, dq);
        return 0;
    }
    std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 2;
}

int main(int argc, char** argv) {
    if (argc < 6) {
        std::fprintf(stderr,
            "usage: wp5qr <mode> <type> <n> <batch> <reps>\n"
            "modes : geqrf | orgqr | ormqrI | qcheck\n"
            "types : float double cfloat cdouble\n");
        return 2;
    }
    const std::string mode = argv[1], t = argv[2];
    const int n = std::atoi(argv[3]), b = std::atoi(argv[4]), r = std::atoi(argv[5]);
    try {
        if (t == "float")   return run<float>(mode, "float", n, b, r);
        if (t == "double")  return run<double>(mode, "double", n, b, r);
        if (t == "cfloat")  return run<std::complex<float>>(mode, "cfloat", n, b, r);
        if (t == "cdouble") return run<std::complex<double>>(mode, "cdouble", n, b, r);
    } catch (const std::exception& e) {
        std::printf("%s,%s,%d,%d,THREW,%s\n", mode.c_str(), t.c_str(), n, b, e.what());
        return 0;
    }
    std::fprintf(stderr, "unknown type %s\n", t.c_str());
    return 2;
}
