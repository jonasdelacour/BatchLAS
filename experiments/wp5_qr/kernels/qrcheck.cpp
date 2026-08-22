// WP5 native geqrf / orgqr -- the CORRECTNESS harness.
//
// It proves five things, and each is a separate column of the output:
//
//   route=      which route the FACADE resolves for this shape and this
//               BATCHLAS_GEQRF_ROUTE. Printed, not assumed: an unrecognised env
//               value silently means {Auto, Auto}, which with preferred() all
//               false is the VENDOR -- so a "native" run that looks like the
//               vendor probably IS the vendor.
//   leaf=       which panel leaf the blocked driver's LEADING panel takes
//               (1 = local-memory resident, 2 = global), from
//               geqrf_blocked_debug_params. Two code paths that a test cannot
//               tell apart is the blind-guard shape.
//   qr=         || Q R x - A x ||inf / || A x ||inf with Q applied from the
//               PACKED reflectors -- the factorisation itself.
//   orth=       || Q^H Q x - x ||inf / ||x||inf on the EXPLICIT Q that native
//               orgqr produced.
//   qrQ=        || Q R x - A x ||inf with that explicit Q -- catches a Q that is
//               orthonormal but is not this A's Q.
//   dF=, dtau=  ELEMENTWISE max relative difference against the VENDOR's own
//               geqrf output. This is the drop-in test, and it is only meaningful
//               because the native larfg follows LAPACK's REAL-beta convention
//               rather than internal::larfg's phase-preserving one. If someone
//               changes that convention, dF and dtau go red for complex while qr
//               and orth stay green -- which is exactly the distinction that
//               needs to be visible.
//
// THE NATIVE CALLS ARE THE DIRECT ENTRY POINTS, not the facade.
// route_resolve.hh:101 falls through to automatic() when a forced route is
// unsupported, so a test that pins BATCHLAS_GEQRF_ROUTE and gets one gate wrong
// silently runs cuSOLVER and passes GREEN over a kernel nothing executed
// (tests/potrf_tests.cc:6-25). geqrf_cta_dispatch / geqrf_blocked_dispatch cannot
// be served by a vendor. The facade route is printed ALONGSIDE, so the pin is
// checked rather than trusted.
//
// BREAK=<n> damages the CHECKER'S reference so a green control and a red break
// together prove the probes can discriminate. See break_mode() below.

#include <batchlas/blas/functions/geqrf.hh>
#include <batchlas/blas/functions/orgqr.hh>
#include <batchlas/blas/functions/ormqr.hh>
#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/dispatch/vendor_available.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>

#include "src/extensions/geqrf_native.hh"
#include "src/extensions/orgqr_native.hh"
#include "src/backends/geqrf_route.hh"
#include "src/backends/orgqr_route.hh"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// BREAK=<n> damages the REFERENCE, never the kernel:
//   1  drop the LAST reflector    (the sy2sb short-final-panel class -- and the
//                                  one WP5's baseline measured VACUOUS on a
//                                  square real matrix)
//   2  apply the reflectors in reversed order
//   3  drop the last COLUMN of the explicit Q
//   4  conjugate tau              (the complex phase-convention class; expected
//                                  to be a NULL result for float and double)
//   5  drop a MIDDLE reflector
static int break_mode() {
    const char* e = std::getenv("BREAK");
    return e ? std::atoi(e) : 0;
}

// ---------------------------------------------------------------- probes
// NaN-PROPAGATING MAX, AND IT IS NOT A DETAIL.
//
// std::max(0.0, NaN) returns 0.0 -- `a < b` is false for any NaN, so the
// accumulator wins and a NaN residual reads as a PERFECT one. That is not
// hypothetical here: kernel break K5 (a wrong tau batch stride) drove tau to
// -12345 for most batch items, the probe overflowed to NaN, and every residual
// column printed 0.000e+00 / 4.788e-07 -- GREEN -- while the factorisation was
// garbage. A checker that reports NaN as a pass is the blind-guard shape this
// repository keeps shipping, and the probes below were inherited from
// experiments/wp5_qr/baseline/wp5qr.cpp, which has it too.
//
// `!(v <= acc)` is false only when v is genuinely no larger; a NaN makes it true
// and the NaN becomes the accumulator, so it propagates all the way out.
static inline void nanmax(double& acc, double v) {
    if (!(v <= acc)) acc = v;
}

template <typename T>
static double geqrf_probe(const UnifiedVector<T>& F, const UnifiedVector<T>& A0,
                          const UnifiedVector<T>& tau, int m, int n, int batch, int nprobe) {
    using D = typename Prom<T>::type;
    const int k = std::min(m, n);
    const size_t st = size_t(m) * n;
    double worst = 0;
    for (int b : {0, batch / 2, batch - 1}) {
        for (int p = 0; p < nprobe; ++p) {
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
            const int bm = break_mode();
            const int lo = (bm == 1) ? 1 : 0;
            const int skip = (bm == 5) ? k / 2 : -1;
            for (int step = lo; step < k; ++step) {
                const int i = (bm == 2) ? step : (k - 1 - step);
                if (i == skip) continue;
                const D t0 = up(tau[size_t(b) * size_t(k) + size_t(i)]);
                const D t = (bm == 4) ? cj(t0) : t0;
                D s = y[i];
                for (int l = i + 1; l < m; ++l) s += cj(up(F[size_t(b) * st + size_t(i) * m + l])) * y[l];
                y[i] -= t * s;
                for (int l = i + 1; l < m; ++l) y[l] -= t * up(F[size_t(b) * st + size_t(i) * m + l]) * s;
            }
            double num = 0, den = 0;
            for (int i = 0; i < m; ++i) { nanmax(num, ab(y[i] - ref[i])); nanmax(den, ab(ref[i])); }
            if (std::getenv("DUMPTAU") && p == 0)
                std::fprintf(stderr, "  probe b=%d num=%.3e den=%.3e tau[b*k]=%.4e\n", b, num, den,
                             ab(up(tau[size_t(b) * size_t(k)])));
            nanmax(worst, den > 0 ? num / den : num);
        }
    }
    return worst;
}

template <typename T>
static double ortho_probe(const UnifiedVector<T>& Q, int m, int n, int k, int batch, int nprobe) {
    using D = typename Prom<T>::type;
    const size_t st = size_t(m) * n;
    double worst = 0;
    for (int b : {0, batch / 2, batch - 1}) {
        for (int p = 0; p < nprobe; ++p) {
            Rng rg(uint64_t(b) * 613 + uint64_t(p) * 17 + 3);
            std::vector<D> x(k);
            for (int j = 0; j < k; ++j) x[j] = up(mk<T>(rg.next(), rg.next()));
            std::vector<D> z(m, D(0)), w(k, D(0));
            const int kk = (break_mode() == 3) ? k - 1 : k;
            for (int j = 0; j < kk; ++j)
                for (int i = 0; i < m; ++i) z[i] += up(Q[size_t(b) * st + size_t(j) * m + i]) * x[j];
            for (int j = 0; j < k; ++j) {
                D acc = D(0);
                for (int i = 0; i < m; ++i) acc += cj(up(Q[size_t(b) * st + size_t(j) * m + i])) * z[i];
                w[j] = acc;
            }
            double num = 0, den = 0;
            for (int j = 0; j < k; ++j) { nanmax(num, ab(w[j] - x[j])); nanmax(den, ab(x[j])); }
            nanmax(worst, den > 0 ? num / den : num);
        }
    }
    return worst;
}

template <typename T>
static double qr_probe(const UnifiedVector<T>& Q, const UnifiedVector<T>& F,
                       const UnifiedVector<T>& A0, int m, int n, int batch, int nprobe) {
    using D = typename Prom<T>::type;
    const int k = std::min(m, n);
    const size_t st = size_t(m) * n;
    double worst = 0;
    for (int b : {0, batch / 2, batch - 1}) {
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
            const int kk = (break_mode() == 3) ? k - 1 : k;
            for (int j = 0; j < kk; ++j)
                for (int i = 0; i < m; ++i) z[i] += up(Q[size_t(b) * st + size_t(j) * m + i]) * y[j];
            double num = 0, den = 0;
            for (int i = 0; i < m; ++i) { nanmax(num, ab(z[i] - ref[i])); nanmax(den, ab(ref[i])); }
            nanmax(worst, den > 0 ? num / den : num);
        }
    }
    return worst;
}

// R MUST BE UPPER TRIANGULAR ON THE FIRST k ROWS -- and, because geqrf packs the
// reflectors into the strict lower triangle, the only thing that can be checked
// structurally is that rows k..m-1 of the first k columns hold reflector data and
// not R. What IS checkable and is checked here: for a COMPLEX factorisation the
// LAPACK convention makes R's DIAGONAL REAL. internal::larfg's phase-preserving
// convention does not. This is the cheapest possible detector for a silent
// convention change, and it is free.
template <typename T>
static double diag_imag_max(const UnifiedVector<T>& F, int m, int n, int batch) {
    using D = typename Prom<T>::type;
    const int k = std::min(m, n);
    const size_t st = size_t(m) * n;
    double worst = 0, scale = 0;
    for (int b = 0; b < batch; ++b)
        for (int i = 0; i < k; ++i) {
            const D d = up(F[size_t(b) * st + size_t(i) * m + i]);
            nanmax(worst, std::fabs(std::imag(std::complex<double>(d))));
            nanmax(scale, ab(d));
        }
    return scale > 0 ? worst / scale : worst;
}

template <typename T>
static double elem_rel_diff(const UnifiedVector<T>& X, const UnifiedVector<T>& Y) {
    double num = 0, den = 0;
    for (size_t i = 0; i < X.size(); ++i) {
        nanmax(num, ab(up(X[i]) - up(Y[i])));
        nanmax(den, ab(up(X[i])));
    }
    return den > 0 ? num / den : num;
}

// ---------------------------------------------------------------- one cell
template <typename T>
static int cell(const char* tn, int m, int n, int batch) {
    auto q = std::make_shared<Queue>(Device("gpu"), BE);
    const int k = std::min(m, n);
    const size_t st = size_t(m) * n;

    UnifiedVector<T> A0(st * batch), A(st * batch);
    UnifiedVector<T> tau(size_t(batch) * k);
    {
        Rng rg(12345 + uint64_t(m) * 7 + uint64_t(n));
        for (size_t i = 0; i < A0.size(); ++i) A0[i] = mk<T>(rg.next(), rg.next());
    }
    UnifiedVector<T*> pA0(batch), pA(batch);
    MatrixView<T, MatrixFormat::Dense> A0v(A0.data(), m, n, m, int(st), batch, pA0.data());
    MatrixView<T, MatrixFormat::Dense> Av(A.data(), m, n, m, int(st), batch, pA.data());
    auto reset_A = [&] { MatrixView<T, MatrixFormat::Dense>::copy(*q, Av, A0v); q->wait(); };

    // WHAT THE FACADE WOULD DO WITH THIS SHAPE AND THIS ENVIRONMENT. Printed, not
    // assumed. It is the same builder + resolver the facade calls, with the same
    // arguments.
    const auto route = backend::geqrf_route<BE, T>(*q, A0v,
                                                   dispatch::factorization_vendor_available<BE>);
    char rbuf[64];
    std::snprintf(rbuf, sizeof rbuf, "%s:%s",
                  std::string(dispatch::to_string(route.origin)).c_str(),
                  std::string(dispatch::to_string(route.algo)).c_str());

    const std::size_t slm_budget = [&] {
        const std::size_t lm = q->device().get_property(DeviceProperty::LOCAL_MEM_SIZE);
        return lm > 4096 ? lm - 4096 : std::size_t(0);
    }();
    const bool cta_fits = sycl_geqrf::geqrf_cta_fits<T>(m, n, slm_budget);
    const unsigned dbg = sycl_geqrf::geqrf_blocked_debug_params<T>(*q, m, n);
    const unsigned nb = dbg & 0xffffu;
    const unsigned leaf = dbg >> 16;

    // ------------------------------------------------ the VENDOR reference
    UnifiedVector<T> Fv_(st * batch), tauv(size_t(batch) * k);
    bool have_vendor = false;
    if constexpr (kVendor) {
        reset_A();
        const size_t ws = backend::geqrf_vendor_buffer_size<BE, T>(*q, Av, tau.to_span());
        UnifiedVector<std::byte> w(ws ? ws : 1);
        backend::geqrf_vendor<BE, T>(*q, Av, tau.to_span(), w.to_span());
        q->wait();
        std::memcpy(Fv_.data(), A.data(), Fv_.size() * sizeof(T));
        std::memcpy(tauv.data(), tau.data(), tauv.size() * sizeof(T));
        have_vendor = true;
    }

    // ------------------------------------------------ the NATIVE variants
    struct Variant { const char* name; bool run; };
    const Variant vs[] = {{"cta", cta_fits}, {"blocked", true}};

    int bad = 0;
    for (const Variant& v : vs) {
        if (!v.run) continue;
        reset_A();
        std::fill(tau.begin(), tau.end(), mk<T>(-12345.0, -12345.0));
        try {
            if (std::string(v.name) == "cta") {
                const std::size_t ws = sycl_geqrf::geqrf_cta_buffer_size<T>(*q, Av);
                UnifiedVector<std::byte> w(ws ? ws : 1);
                sycl_geqrf::geqrf_cta_dispatch<T>(*q, Av, tau.to_span(), w.to_span());
            } else {
                const std::size_t ws = sycl_geqrf::geqrf_blocked_buffer_size<T>(*q, Av);
                UnifiedVector<std::byte> w(ws ? ws : 1);
                // THE ROUTED gemm, injected exactly as the facade injects it, so
                // this harness measures the same trailing update the shipped path
                // does rather than the native kernel unconditionally.
                sycl_geqrf::geqrf_blocked_dispatch<T>(
                    *q, Av, tau.to_span(), w.to_span(),
                    [](Queue& c,
                       const MatrixView<T, MatrixFormat::Dense>& ga,
                       const MatrixView<T, MatrixFormat::Dense>& gb,
                       const MatrixView<T, MatrixFormat::Dense>& gc,
                       T galpha, T gbeta, Transpose gta, Transpose gtb,
                       ComputePrecision gp) {
                        return gemm<BE, T>(c, ga, gb, gc, galpha, gbeta, gta, gtb, gp);
                    });
            }
            q->wait();
        } catch (const std::exception& e) {
            std::printf("%s,%s,%d,%d,%d,route=%s,leaf=%u,nb=%u,THREW,%s\n",
                        v.name, tn, m, n, batch, rbuf, leaf, nb, e.what());
            ++bad;
            continue;
        }

        UnifiedVector<T> F(st * batch), tn_(size_t(batch) * k);
        std::memcpy(F.data(), A.data(), F.size() * sizeof(T));
        std::memcpy(tn_.data(), tau.data(), tn_.size() * sizeof(T));

        if (std::getenv("DUMPTAU")) {
            size_t poison = 0;
            for (size_t i = 0; i < tn_.size(); ++i)
                if (ab(up(tn_[i]) - up(mk<T>(-12345.0, -12345.0))) < 1e-3) ++poison;
            std::fprintf(stderr, "DUMPTAU %s %s %dx%d b=%d poison=%zu/%zu tau[0]=%.4e tau[b/2*k]=%.4e\n",
                         v.name, tn, m, n, batch, poison, tn_.size(),
                         ab(up(tn_[0])), ab(up(tn_[size_t(batch / 2) * size_t(k)])));
        }
        const double qr = geqrf_probe<T>(F, A0, tn_, m, n, batch, 3);
        const double dimag = diag_imag_max<T>(F, m, n, batch);
        const double dF = have_vendor ? elem_rel_diff<T>(Fv_, F) : -1.0;
        const double dtau = have_vendor ? elem_rel_diff<T>(tauv, tn_) : -1.0;

        // ---------------------------------------- native orgqr on this factor
        double orth = -1.0, qrQ = -1.0;
        const char* oerr = nullptr;
        UnifiedVector<T> Qm(st * batch);
        {
            std::memcpy(Qm.data(), F.data(), Qm.size() * sizeof(T));
            UnifiedVector<T*> pQ(batch);
            MatrixView<T, MatrixFormat::Dense> Qv(Qm.data(), m, n, m, int(st), batch, pQ.data());
            auto apply = [](Queue& c,
                            const MatrixView<T, MatrixFormat::Dense>& oa,
                            const MatrixView<T, MatrixFormat::Dense>& oc,
                            Side os, Transpose ot, Span<T> ot2,
                            Span<std::byte> ows, int32_t obs) {
                return ormqr<BE, T>(c, oa, oc, os, ot, ot2, ows, obs);
            };
            auto applybs = [](Queue& c,
                              const MatrixView<T, MatrixFormat::Dense>& oa,
                              const MatrixView<T, MatrixFormat::Dense>& oc,
                              Side os, Transpose ot, Span<T> ot2, int32_t obs) {
                return ormqr_buffer_size<BE, T>(c, oa, oc, os, ot, ot2, obs);
            };
            try {
                const std::size_t ws =
                    sycl_orgqr::orgqr_blocked_buffer_size<T>(*q, Qv, tn_.to_span(), applybs);
                UnifiedVector<std::byte> w(ws ? ws : 1);
                sycl_orgqr::orgqr_blocked_dispatch<T>(*q, Qv, tn_.to_span(), w.to_span(),
                                                      apply, applybs);
                q->wait();
                orth = ortho_probe<T>(Qm, m, n, k, batch, 3);
                qrQ = qr_probe<T>(Qm, F, A0, m, n, batch, 3);
            } catch (const std::exception& e) {
                oerr = e.what();
            }
        }

        std::printf("%s,%s,%d,%d,%d,route=%s,leaf=%u,nb=%u,qr=%.3e,orth=%.3e,qrQ=%.3e,"
                    "dF=%.3e,dtau=%.3e,dimag=%.3e%s%s\n",
                    v.name, tn, m, n, batch, rbuf, leaf, nb, qr, orth, qrQ, dF, dtau, dimag,
                    oerr ? ",ORGQR_THREW=" : "", oerr ? oerr : "");
        std::fflush(stdout);
    }
    return bad;
}

// ---------------------------------------------------------------- main
static bool want(const char* t, int argc, char** argv) {
    if (argc < 2) return true;
    for (int i = 1; i < argc; ++i) if (std::string(argv[i]) == t) return true;
    return false;
}

int main(int argc, char** argv) {
    struct Shape { int m, n, batch; };
    // The size list, and why each row is here.
    //   64x64    : fits the CTA tier for every type; n % nb == 0 for both widths
    //   65x65    : n % 32 == 1 and n % 16 == 1 -- the residue ec1a178 was GREEN at
    //   66x66    : n % 32 == 2 and n % 16 == 2 -- the residue it FAILED at
    //   100x64   : m > n, so the last panel's reflectors outrun its own width
    //   129x33   : tall, odd, n % 32 == 1
    //   200x200  : blocked, several panels, n % 32 = 8 and n % 16 = 8
    //   256x256  : blocked, n an exact multiple of BOTH block widths
    //   300x200  : blocked, m > n, short final panel AND middle panels
    //   512x128  : blocked, tall; the leading panel is 512x32 = too big to be
    //              resident for cdouble, so the two leaves are both exercised
    const Shape shapes[] = {
        {64, 64, 8},   {65, 65, 8},   {66, 66, 8},   {100, 64, 8},
        {129, 33, 8},  {200, 200, 4}, {256, 256, 4}, {300, 200, 4},
        {512, 128, 4}, {128, 128, 64},
    };

    std::printf("variant,type,m,n,batch,route,leaf,nb,qr,orth,qrQ,dF,dtau,dimag\n");
    int bad = 0;
    for (const Shape& s : shapes) {
        if (want("float", argc, argv))   bad += cell<float>("float", s.m, s.n, s.batch);
        if (want("double", argc, argv))  bad += cell<double>("double", s.m, s.n, s.batch);
        if (want("cfloat", argc, argv))  bad += cell<std::complex<float>>("cfloat", s.m, s.n, s.batch);
        if (want("cdouble", argc, argv)) bad += cell<std::complex<double>>("cdouble", s.m, s.n, s.batch);
    }
    std::fprintf(stderr, "threw=%d vendor_build=%d\n", bad, int(kVendor));
    return bad ? 1 : 0;
}
