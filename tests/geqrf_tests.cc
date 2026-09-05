// Native batched GEQRF and the ORGQR that consumes its output.
//
// A forced route that supports() rejects falls through to the vendor silently, so every
// numerical test calls the native entry points directly against a host reference in
// double. Residuals are blind to the tau/beta convention -- geqrf's real contract --
// which is why the convention tests exist. evidence: docs/perf/qr.md
#include <gtest/gtest.h>

#include <batchlas/blas/functions/geqrf.hh>
#include <batchlas/blas/functions/orgqr.hh>
#include <batchlas/blas/functions/ormqr.hh>
#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/dispatch/vendor_available.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>

#include "test_utils.hh"

#include "../src/extensions/geqrf_native.hh"
#include "../src/extensions/orgqr_native.hh"
#include "../src/backends/geqrf_route.hh"
#include "../src/backends/orgqr_route.hh"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <limits>
#include <string>
#include <utility>
#include <vector>

using namespace batchlas;

namespace {

template <typename T>
using RealOf = typename batchlas::base_type<T>::type;

// References accumulate in double, so a float residual measures the KERNEL's error.
template <class T> struct Prom { using type = double; };
template <class R> struct Prom<std::complex<R>> { using type = std::complex<double>; };

inline double up(float x) { return double(x); }
inline double up(double x) { return x; }
inline std::complex<double> up(std::complex<float> x) { return {double(x.real()), double(x.imag())}; }
inline std::complex<double> up(std::complex<double> x) { return x; }

inline double hconj(double x) { return x; }
inline std::complex<double> hconj(std::complex<double> x) { return std::conj(x); }
inline double habs(double x) { return std::fabs(x); }
inline double habs(std::complex<double> x) { return std::abs(x); }
inline double hreal(double x) { return x; }
inline double hreal(std::complex<double> x) { return x.real(); }
inline double himag(double) { return 0.0; }
inline double himag(std::complex<double> x) { return x.imag(); }

template <class T> inline T mk(double re, double im);
template <> inline float mk<float>(double re, double) { return float(re); }
template <> inline double mk<double>(double re, double) { return re; }
template <> inline std::complex<float> mk<std::complex<float>>(double re, double im) {
    return {float(re), float(im)};
}
template <> inline std::complex<double> mk<std::complex<double>>(double re, double im) {
    return {re, im};
}

// A deterministic LCG rather than <random>: tests below assert that two batch items
// DIFFER, and that assertion has to be about the data, not luck.
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 6364136223846793005ULL + 1442695040888963407ULL) {}
    double next() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return double(int32_t(uint32_t(s >> 32))) / 2147483648.0;
    }
};

// Q is built by APPLYING the packed reflectors to the first k columns of I_m, in
// LAPACK's order Q = H_0 H_1 ... H_{k-1}, H_i = I - tau_i v_i v_i^H with
// v_i = [0 .. 0, 1, F(i+1:m-1, i)].
template <typename T>
std::vector<typename Prom<T>::type>
host_form_Q(const T* F, const T* tau, int m, int k, int ld) {
    using D = typename Prom<T>::type;
    std::vector<D> Q(static_cast<size_t>(m) * k, D(0));
    for (int j = 0; j < k; ++j) Q[static_cast<size_t>(j) * m + j] = D(1);
    for (int i = k - 1; i >= 0; --i) {
        const D t = up(tau[i]);
        if (habs(t) == 0.0) continue;
        for (int c = 0; c < k; ++c) {
            // w = v_i^H Q(:, c), with the implicit v_i(i) = 1.
            D w = Q[static_cast<size_t>(c) * m + i];
            for (int r = i + 1; r < m; ++r) {
                w += hconj(up(F[static_cast<size_t>(i) * ld + r])) *
                     Q[static_cast<size_t>(c) * m + r];
            }
            const D f = t * w;
            Q[static_cast<size_t>(c) * m + i] -= f;
            for (int r = i + 1; r < m; ++r) {
                Q[static_cast<size_t>(c) * m + r] -= f * up(F[static_cast<size_t>(i) * ld + r]);
            }
        }
    }
    return Q;
}

// ||Q R - A||_F / ||A||_F, with Q supplied (m x k, column-major, tight) and R
// read out of F's upper triangle.
template <typename T>
double qr_residual(const std::vector<typename Prom<T>::type>& Q,
                   const T* F, const T* A0, int m, int n, int k, int ld, int ld0) {
    using D = typename Prom<T>::type;
    double num = 0.0, den = 0.0;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            D acc(0);
            const int top = std::min(k - 1, j);
            for (int p = 0; p <= top; ++p) {
                acc += Q[static_cast<size_t>(p) * m + i] * up(F[static_cast<size_t>(j) * ld + p]);
            }
            const D a = up(A0[static_cast<size_t>(j) * ld0 + i]);
            const D d = acc - a;
            num += hreal(d) * hreal(d) + himag(d) * himag(d);
            den += hreal(a) * hreal(a) + himag(a) * himag(a);
        }
    }
    return den > 0.0 ? std::sqrt(num) / std::sqrt(den) : std::sqrt(num);
}

// ||Q^H Q - I||_F / sqrt(k), for a Q held tight in double.
template <typename D>
double orth_of_promoted(const std::vector<D>& Q, int m, int k) {
    double num = 0.0;
    for (int a = 0; a < k; ++a) {
        for (int b = 0; b < k; ++b) {
            D acc(0);
            for (int r = 0; r < m; ++r)
                acc += hconj(Q[static_cast<size_t>(a) * m + r]) * Q[static_cast<size_t>(b) * m + r];
            const D d = acc - D(a == b ? 1 : 0);
            num += hreal(d) * hreal(d) + himag(d) * himag(d);
        }
    }
    return std::sqrt(num) / std::sqrt(double(k));
}

// Copy a device Q (ld) into the tight m x k double buffer the probes want.
template <typename T>
std::vector<typename Prom<T>::type> promote_Q(const T* Q, int m, int k, int ld) {
    using D = typename Prom<T>::type;
    std::vector<D> out(static_cast<size_t>(m) * k);
    for (int j = 0; j < k; ++j)
        for (int i = 0; i < m; ++i)
            out[static_cast<size_t>(j) * m + i] = up(Q[static_cast<size_t>(j) * ld + i]);
    return out;
}

// Householder QR's backward error is O(m k) eps ||A||; this constant is tight, not slack.
template <typename T>
double residual_tol(int m, int k) {
    return 0.5 * double(m + k) * double(std::numeric_limits<RealOf<T>>::epsilon());
}

template <typename T>
double orth_tol(int m, int k) {
    return 0.5 * double(m + k) * double(std::numeric_limits<RealOf<T>>::epsilon());
}

inline bool verbose() { return std::getenv("GEQRF_TESTS_VERBOSE") != nullptr; }

template <typename T, Backend B>
struct GeqrfConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

// DISTINCT matrices per item at ld != rows and stride != ld*cols, so a view built on the
// stride-defaulting constructor is falsifiable by DEFAULT. The pad carries a large POISON,
// so an out-of-window read is a wrong answer rather than a near one.
template <typename T>
struct Problem {
    int m = 0, n = 0, k = 0, batch = 0, ld = 0, stride = 0;
    UnifiedVector<T> buf;      // the working copy, overwritten by geqrf
    std::vector<T> a0;         // the pristine input, same ld/stride
    UnifiedVector<T> tau;
    // Required, not decoration: the vendor geqrf/orgqr are pointer-array APIs, and a
    // view built without one throws "data_ptrs target is null" at the vendor boundary.
    UnifiedVector<T*> ptrs;
};

template <typename T>
Problem<T> make_problem(int m, int n, int batch, unsigned seed,
                        int ld_pad = 5, int stride_pad = 11) {
    Problem<T> p;
    p.m = m; p.n = n; p.k = std::min(m, n); p.batch = batch;
    p.ld = m + ld_pad;
    p.stride = p.ld * n + stride_pad;
    p.buf = UnifiedVector<T>(static_cast<size_t>(p.stride) * batch, mk<T>(-9.75e3, 4.5e3));
    p.tau = UnifiedVector<T>(static_cast<size_t>(p.k) * batch, mk<T>(-12345.0, -12345.0));
    p.ptrs = UnifiedVector<T*>(static_cast<size_t>(batch), nullptr);
    Rng rg(seed);
    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                p.buf[static_cast<size_t>(b) * p.stride + static_cast<size_t>(j) * p.ld + i] =
                    mk<T>(rg.next(), rg.next());
            }
        }
    }
    p.a0.assign(p.buf.begin(), p.buf.end());
    return p;
}

template <typename T>
MatrixView<T, MatrixFormat::Dense> view_of(Problem<T>& p) {
    return MatrixView<T, MatrixFormat::Dense>(p.buf.data(), p.m, p.n, p.ld, p.stride, p.batch,
                                             p.ptrs.data());
}

// Restore the pristine input, poison and all, and re-poison tau so a slot the
// kernel never writes is visible as -12345 rather than as a plausible leftover.
template <typename T>
void reset(Problem<T>& p) {
    std::copy(p.a0.begin(), p.a0.end(), p.buf.begin());
    std::fill(p.tau.begin(), p.tau.end(), mk<T>(-12345.0, -12345.0));
}

template <typename Config>
class GeqrfTest : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    using R = RealOf<T>;
    using D = typename Prom<T>::type;
    static constexpr Backend BackendType = Config::BackendVal;

    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        if (this->HasFatalFailure() || ::testing::Test::IsSkipped()) return;
        if (!this->ctx) GTEST_SKIP() << "no queue";
        if (this->ctx->device().type != DeviceType::GPU) {
            GTEST_SKIP() << "the native geqrf kernels are GPU-only (route_geqrf.hh gate 3)";
        }
        if (!this->ctx->device().supports_sub_group_size(32)) {
            GTEST_SKIP() << "device does not offer sub-group size 32 (route_geqrf.hh gate 4)";
        }
    }

    // The DEVICE's local-memory budget, spelled as src/backends/geqrf_route.hh spells
    // it -- NOT device_limits.hh's hardcoded 49152.
    std::size_t budget() const {
        const std::size_t lm = static_cast<std::size_t>(
            this->ctx->device().get_property(DeviceProperty::LOCAL_MEM_SIZE));
        return lm > 4096 ? lm - 4096 : std::size_t(0);
    }

    bool cta_fits(int m, int n) const {
        return sycl_geqrf::geqrf_cta_fits<T>(m, n, budget());
    }

    // Queried, never hardcoded: a hardcoded width silently stops straddling when it moves.
    int nb(int m, int n) const {
        return static_cast<int>(
            sycl_geqrf::geqrf_blocked_debug_params<T>(*this->ctx, m, n) & 0xffffu);
    }
    unsigned leaf(int m, int n) const {
        return sycl_geqrf::geqrf_blocked_debug_params<T>(*this->ctx, m, n) >> 16;
    }
};

// Both residuals plus finiteness and batch-distinctness on EVERY batch item. Item 0 sits
// at offset 0, so only the distinctness check can fail a kernel that broadcast item 0.
template <typename T>
void check_one(const Problem<T>& p, const char* what) {
    for (int b = 0; b < p.batch; ++b) {
        const T* F = p.buf.data() + static_cast<size_t>(b) * p.stride;
        const T* A0 = p.a0.data() + static_cast<size_t>(b) * p.stride;
        const T* tau = p.tau.data() + static_cast<size_t>(b) * p.k;

        for (int i = 0; i < p.k; ++i) {
            ASSERT_TRUE(std::isfinite(hreal(up(tau[i]))) && std::isfinite(himag(up(tau[i]))))
                << what << ": tau[" << i << "] is not finite at b=" << b;
        }
        const auto Q = host_form_Q<T>(F, tau, p.m, p.k, p.ld);
        const double res = qr_residual<T>(Q, F, A0, p.m, p.n, p.k, p.ld, p.ld);
        const double orth = orth_of_promoted(Q, p.m, p.k);
        if (verbose()) {
            std::printf("[verbose] %-28s m=%4d n=%4d b=%d/%d  qr=%.4e orth=%.4e  tol=%.4e\n",
                        what, p.m, p.n, b, p.batch, res, orth, residual_tol<T>(p.m, p.k));
            std::fflush(stdout);
        }
        EXPECT_LE(res, residual_tol<T>(p.m, p.k))
            << what << ": ||QR-A||_F/||A||_F too large at b=" << b
            << " (m=" << p.m << " n=" << p.n << ")";
        EXPECT_LE(orth, orth_tol<T>(p.m, p.k))
            << what << ": the packed reflectors are not orthonormal at b=" << b
            << " (m=" << p.m << " n=" << p.n << ")";
    }

    if (p.batch > 1) {
        const T* f0 = p.buf.data();
        const T* fl = p.buf.data() + static_cast<size_t>(p.batch - 1) * p.stride;
        bool differ = false;
        for (int j = 0; j < p.n && !differ; ++j)
            for (int i = 0; i < p.m && !differ; ++i)
                if (habs(up(f0[static_cast<size_t>(j) * p.ld + i]) -
                         up(fl[static_cast<size_t>(j) * p.ld + i])) > 0.0)
                    differ = true;
        EXPECT_TRUE(differ) << what << ": the first and last batch items' factors are identical, "
                               "so this shape cannot see a batch-stride defect";
    }
}

struct GeqrfEnvGuard {
    std::string saved;
    bool had = false;
    explicit GeqrfEnvGuard(const char* v) {
        if (const char* s = std::getenv("BATCHLAS_GEQRF_ROUTE")) { saved = s; had = true; }
        ::setenv("BATCHLAS_GEQRF_ROUTE", v, 1);
    }
    ~GeqrfEnvGuard() {
        if (had) ::setenv("BATCHLAS_GEQRF_ROUTE", saved.c_str(), 1);
        else ::unsetenv("BATCHLAS_GEQRF_ROUTE");
    }
};

struct OrgqrEnvGuard {
    std::string saved;
    bool had = false;
    explicit OrgqrEnvGuard(const char* v) {
        if (const char* s = std::getenv("BATCHLAS_ORGQR_ROUTE")) { saved = s; had = true; }
        ::setenv("BATCHLAS_ORGQR_ROUTE", v, 1);
    }
    ~OrgqrEnvGuard() {
        if (had) ::setenv("BATCHLAS_ORGQR_ROUTE", saved.c_str(), 1);
        else ::unsetenv("BATCHLAS_ORGQR_ROUTE");
    }
};

using GeqrfTestTypes = typename test_utils::backend_types<GeqrfConfig>::type;

}  // namespace

TYPED_TEST_SUITE(GeqrfTest, GeqrfTestTypes);

// G0. THE 48 KB LAUNCH HOLE. DECLARED FIRST ON PURPOSE: the shared-memory cap the CUDA
// adapter raises is STICKY PER CUfunction, so an earlier resident-leaf launch of a larger
// panel raises it for the rest of the process and this guard can never fail again.
// GoogleTest runs a suite in declaration order -- do not move this test, and do not add a
// resident-leaf launch above it. evidence: docs/perf/qr.md#the-48-kib-launch-hole
TYPED_TEST(GeqrfTest, ResidentLeafLaunchHoleAt48KiB) {
    using T = typename TestFixture::T;

    // Byte sizes as element counts for THIS scalar type, as an m x n panel with
    // m >= n (supports() gate 2).
    struct S { std::size_t bytes; int m, n; };
    const std::size_t sz = sizeof(T);
    const S rows[] = {
        {48896, static_cast<int>(48896 / sz / 16), 16},
        {49152, static_cast<int>(49152 / sz / 16), 16},
        {49664, static_cast<int>(49664 / sz / 16), 16},
    };

    for (const S& r : rows) {
        ASSERT_EQ(static_cast<std::size_t>(r.m) * static_cast<std::size_t>(r.n) * sz, r.bytes)
            << "this row does not ask for " << r.bytes << " B";
        ASSERT_GE(r.m, r.n);
        ASSERT_TRUE(this->cta_fits(r.m, r.n))
            << "a " << r.bytes << " B tile is not admissible to the CTA tier at this budget; "
               "the hole row is unreachable and this test proves nothing";

        auto p = make_problem<T>(r.m, r.n, 2, 271u + unsigned(r.bytes % 1000));
        auto V = view_of(p);
        UnifiedVector<std::byte> wb(std::max<std::size_t>(
            1, sycl_geqrf::geqrf_cta_buffer_size<T>(*this->ctx, V)));
        ASSERT_NO_THROW(
            sycl_geqrf::geqrf_cta_dispatch<T>(*this->ctx, V, p.tau.to_span(), wb.to_span()))
            << "the resident leaf could not be launched with a " << r.bytes
            << " B tile (" << r.m << "x" << r.n << ")";
        this->ctx->wait();
        check_one(p, "cta/launch-hole");
        if (this->HasFailure()) return;
    }
}

// G1 / G2. The factorisation itself, on both native tiers, over a residue ladder.
TYPED_TEST(GeqrfTest, CtaResidualAndOrthogonality) {
    using T = typename TestFixture::T;
    struct S { int m, n, b; };
    // n%nb residues 0/1/2/8 against both shipped widths (32 and 16), m == n and m > n.
    const S shapes[] = {{32, 32, 5}, {33, 17, 5}, {48, 48, 5}, {49, 49, 4},
                        {50, 34, 4}, {64, 40, 4}, {17, 9, 3},  {96, 24, 3}};
    int ran = 0;
    for (const S& s : shapes) {
        if (!this->cta_fits(s.m, s.n)) continue;
        ++ran;
        auto p = make_problem<T>(s.m, s.n, s.b, 5150u + 17u * unsigned(s.m) + unsigned(s.n));
        auto V = view_of(p);
        const std::size_t ws = sycl_geqrf::geqrf_cta_buffer_size<T>(*this->ctx, V);
        UnifiedVector<std::byte> w(ws ? ws : 1);
        ASSERT_NO_THROW(
            sycl_geqrf::geqrf_cta_dispatch<T>(*this->ctx, V, p.tau.to_span(), w.to_span()));
        this->ctx->wait();
        check_one(p, "cta");
        if (this->HasFailure()) return;
    }
    ASSERT_GE(ran, 6) << "only " << ran << " of the CTA shapes fit; this test has stopped "
                         "covering the tier it names";
}

TYPED_TEST(GeqrfTest, BlockedResidualAndOrthogonality) {
    using T = typename TestFixture::T;
    struct S { int m, n, b; };
    // 96x96's leading cdouble panel is exactly the 49,152 B G0 guards, but this is NOT a
    // substitute for G0: an earlier row has already raised the sticky per-CUfunction cap.
    const S shapes[] = {{64, 64, 4},  {65, 65, 4},   {66, 66, 4}, {100, 64, 3},
                        {129, 33, 3}, {130, 130, 2}, {96, 96, 2}, {112, 80, 3}};
    for (const S& s : shapes) {
        auto p = make_problem<T>(s.m, s.n, s.b, 909u + 31u * unsigned(s.m) + unsigned(s.n));
        auto V = view_of(p);
        const std::size_t ws = sycl_geqrf::geqrf_blocked_buffer_size<T>(*this->ctx, V);
        UnifiedVector<std::byte> w(ws ? ws : 1);
        ASSERT_NO_THROW(
            sycl_geqrf::geqrf_blocked_dispatch<T>(*this->ctx, V, p.tau.to_span(), w.to_span()));
        this->ctx->wait();
        check_one(p, "blocked");
        if (this->HasFailure()) return;
    }
}

// G3. The block boundary, straddled on purpose and ASSERTED to be straddled. The rows
// are m > n deliberately: on a square REAL matrix larfg returns tau = 0 for the final
// 1x1 reflector, so dropping it changes nothing and the test guards nothing.
TYPED_TEST(GeqrfTest, ShortFinalPanelStraddlesTheBlockWidth) {
    using T = typename TestFixture::T;

    // Ask at a shape large enough that the answer is the type's own width, not min(nb, k).
    const int w = this->nb(512, 512);
    ASSERT_GE(w, 2) << "a block width of " << w << " cannot straddle anything";

    const int n_mult = 3 * w;              // exact multiple: every panel is full
    const int n_odd  = 3 * w + w / 2;      // NOT a multiple: the last panel is short
    ASSERT_EQ(n_mult % w, 0);
    ASSERT_NE(n_odd % w, 0);
    ASSERT_GE((n_mult + w - 1) / w, 2) << "the multiple case has only one panel";
    ASSERT_GE((n_odd + w - 1) / w, 2) << "the odd case has only one panel";
    ASSERT_LT(n_odd % w, w) << "the odd case's final panel is not short";

    struct S { int m, n; };
    const S shapes[] = {{n_mult + 23, n_mult}, {n_odd + 23, n_odd},
                        {n_odd, n_odd},        {n_mult + 1, n_mult}};
    for (const S& s : shapes) {
        ASSERT_EQ(this->nb(s.m, s.n), w)
            << "the block width moved at m=" << s.m << " n=" << s.n
            << "; this test's straddle assertions no longer describe the driver";
        auto p = make_problem<T>(s.m, s.n, 3, 3131u + 7u * unsigned(s.n) + unsigned(s.m));
        auto V = view_of(p);
        const std::size_t ws = sycl_geqrf::geqrf_blocked_buffer_size<T>(*this->ctx, V);
        UnifiedVector<std::byte> wbuf(ws ? ws : 1);
        ASSERT_NO_THROW(sycl_geqrf::geqrf_blocked_dispatch<T>(*this->ctx, V, p.tau.to_span(),
                                                              wbuf.to_span()));
        this->ctx->wait();
        check_one(p, (s.n % w) ? "blocked/short-final-panel" : "blocked/exact-multiple");
        if (this->HasFailure()) return;
    }
}

// G4. Both panel leaves. The blocked driver picks a local_accessor tile or a raw global
// pointer per panel, so this test ASSERTS which one it got before believing any residual.
TYPED_TEST(GeqrfTest, BothPanelLeavesFactoriseCorrectly) {
    using T = typename TestFixture::T;
    const int w = this->nb(4096, 64);
    ASSERT_GE(w, 2);

    // Resident: a short matrix whose leading m x nb panel certainly fits.
    {
        const int m = 96, n = std::min(64, 2 * w);
        ASSERT_EQ(this->leaf(m, n), 1u) << "the short shape did not take the resident leaf";
        auto p = make_problem<T>(m, n, 3, 777u);
        auto V = view_of(p);
        UnifiedVector<std::byte> wb(std::max<std::size_t>(
            1, sycl_geqrf::geqrf_blocked_buffer_size<T>(*this->ctx, V)));
        ASSERT_NO_THROW(sycl_geqrf::geqrf_blocked_dispatch<T>(*this->ctx, V, p.tau.to_span(),
                                                              wb.to_span()));
        this->ctx->wait();
        check_one(p, "blocked/resident-leaf");
        if (this->HasFailure()) return;
    }

    // Global: grow m until the leading panel no longer fits.
    {
        int m = 256;
        while (m <= 32768 && this->leaf(m, 64) == 1u) m *= 2;
        ASSERT_LE(m, 32768) << "no height pushes the leading panel off the resident leaf; "
                               "the global leaf is unreachable from this test";
        ASSERT_EQ(this->leaf(m, 64), 2u);
        auto p = make_problem<T>(m, 64, 2, 4242u);
        auto V = view_of(p);
        UnifiedVector<std::byte> wb(std::max<std::size_t>(
            1, sycl_geqrf::geqrf_blocked_buffer_size<T>(*this->ctx, V)));
        ASSERT_NO_THROW(sycl_geqrf::geqrf_blocked_dispatch<T>(*this->ctx, V, p.tau.to_span(),
                                                              wb.to_span()));
        this->ctx->wait();
        check_one(p, "blocked/global-leaf");
    }
}

// G5. A rank-deficient input: one exactly-zero column and one exact duplicate. The zero
// column drives larfg's tau = 0 identity branch, whose work-group uniformity the kernel's
// `continue` depends on.
TYPED_TEST(GeqrfTest, RankDeficientColumnsStillFactorise) {
    using T = typename TestFixture::T;
    for (int pass = 0; pass < 2; ++pass) {
        const int m = pass ? 100 : 40;         // blocked, then CTA-sized
        const int n = pass ? 70 : 24;
        auto p = make_problem<T>(m, n, 3, 6161u + unsigned(pass));
        for (int b = 0; b < p.batch; ++b) {
            const size_t base = static_cast<size_t>(b) * p.stride;
            for (int i = 0; i < m; ++i) {
                p.buf[base + static_cast<size_t>(5) * p.ld + i] = mk<T>(0, 0);   // zero column
                p.buf[base + static_cast<size_t>(9) * p.ld + i] =
                    p.buf[base + static_cast<size_t>(3) * p.ld + i];             // duplicate
            }
        }
        p.a0.assign(p.buf.begin(), p.buf.end());

        auto V = view_of(p);
        UnifiedVector<std::byte> wb(std::max<std::size_t>(
            1, pass ? sycl_geqrf::geqrf_blocked_buffer_size<T>(*this->ctx, V)
                    : sycl_geqrf::geqrf_cta_buffer_size<T>(*this->ctx, V)));
        if (pass) {
            ASSERT_NO_THROW(sycl_geqrf::geqrf_blocked_dispatch<T>(*this->ctx, V, p.tau.to_span(),
                                                                  wb.to_span()));
        } else {
            ASSERT_TRUE(this->cta_fits(m, n));
            ASSERT_NO_THROW(sycl_geqrf::geqrf_cta_dispatch<T>(*this->ctx, V, p.tau.to_span(),
                                                              wb.to_span()));
        }
        this->ctx->wait();
        check_one(p, pass ? "blocked/rank-deficient" : "cta/rank-deficient");
        if (this->HasFailure()) return;

        // The identity branch was TAKEN, not merely survived.
        for (int b = 0; b < p.batch; ++b) {
            const T* F = p.buf.data() + static_cast<size_t>(b) * p.stride;
            EXPECT_LE(habs(up(F[static_cast<size_t>(5) * p.ld + 5])), 1e-5)
                << "R(5,5) is not ~0 for a rank-deficient column, b=" << b;
        }
    }
}

// G6. The complex convention no residual can see: LAPACK's clarfg/zlarfg return a REAL
// beta, internal::larfg is phase-preserving, and swapping the two leaves qr, orth and the
// explicit-Q residual GREEN for every type.
TYPED_TEST(GeqrfTest, ComplexRDiagonalIsExactlyReal) {
    using T = typename TestFixture::T;
    if constexpr (!test_utils::is_complex<T>::value) {
        GTEST_SKIP() << "the diagonal is trivially real for a real scalar type";
    } else {
        struct S { int m, n; bool blocked; };
        const S shapes[] = {{40, 24, false}, {100, 70, true}, {96, 96, true}};
        for (const S& s : shapes) {
            auto p = make_problem<T>(s.m, s.n, 3, 8181u + unsigned(s.n));
            auto V = view_of(p);
            UnifiedVector<std::byte> wb(std::max<std::size_t>(
                1, s.blocked ? sycl_geqrf::geqrf_blocked_buffer_size<T>(*this->ctx, V)
                             : sycl_geqrf::geqrf_cta_buffer_size<T>(*this->ctx, V)));
            if (s.blocked)
                sycl_geqrf::geqrf_blocked_dispatch<T>(*this->ctx, V, p.tau.to_span(), wb.to_span());
            else
                sycl_geqrf::geqrf_cta_dispatch<T>(*this->ctx, V, p.tau.to_span(), wb.to_span());
            this->ctx->wait();
            for (int b = 0; b < p.batch; ++b) {
                const T* F = p.buf.data() + static_cast<size_t>(b) * p.stride;
                for (int i = 0; i < p.k; ++i) {
                    // EXACTLY zero, not "small": the LAPACK convention WRITES a real scalar into
                    // A(j,j); it does not merely rotate the imaginary part down to rounding.
                    ASSERT_EQ(himag(up(F[static_cast<size_t>(i) * p.ld + i])), 0.0)
                        << "imag(R(" << i << "," << i << ")) != 0 at m=" << s.m << " n=" << s.n
                        << " b=" << b << " -- the larfg phase convention is not LAPACK's, and "
                           "every residual test in this file is blind to that";
                }
            }
        }
    }
}

// G7. The interface contract, in both directions: geqrf's output is a CONVENTION that
// ormqr, orgqr, ormbr and sy2sb consume. A -- NATIVE geqrf -> ROUTED ormqr.
TYPED_TEST(GeqrfTest, TauConventionSurvivesTheRoutedOrmqr) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::BackendType;
    const int m = 64, n = 40, batch = 3;

    auto p = make_problem<T>(m, n, batch, 2468u);
    auto V = view_of(p);
    UnifiedVector<std::byte> wb(std::max<std::size_t>(
        1, sycl_geqrf::geqrf_blocked_buffer_size<T>(*this->ctx, V)));
    sycl_geqrf::geqrf_blocked_dispatch<T>(*this->ctx, V, p.tau.to_span(), wb.to_span());
    this->ctx->wait();

    // C = the first n columns of I_m; then C <- Q C, so Q's first n columns come
    // out. C carries its own padded ld and non-default stride.
    const int cld = m + 3, cstride = cld * n + 7;
    UnifiedVector<T> C(static_cast<size_t>(cstride) * batch, mk<T>(0, 0));
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < n; ++j)
            C[static_cast<size_t>(b) * cstride + static_cast<size_t>(j) * cld + j] = mk<T>(1, 0);
    UnifiedVector<T*> cptrs(static_cast<size_t>(batch), nullptr);
    MatrixView<T, MatrixFormat::Dense> Cv(C.data(), m, n, cld, cstride, batch, cptrs.data());

    const std::size_t ows = ormqr_buffer_size<B, T>(*this->ctx, V, Cv, Side::Left,
                                                    Transpose::NoTrans, p.tau.to_span(), 0);
    UnifiedVector<std::byte> ow(ows ? ows : 1);
    ormqr<B, T>(*this->ctx, V, Cv, Side::Left, Transpose::NoTrans, p.tau.to_span(),
                ow.to_span(), 0);
    this->ctx->wait();

    for (int b = 0; b < batch; ++b) {
        const T* Q = C.data() + static_cast<size_t>(b) * cstride;
        const T* F = p.buf.data() + static_cast<size_t>(b) * p.stride;
        const T* A0 = p.a0.data() + static_cast<size_t>(b) * p.stride;
        const auto Qp = promote_Q<T>(Q, m, n, cld);
        EXPECT_LE(orth_of_promoted(Qp, m, n), orth_tol<T>(m, n)) << "b=" << b;
        EXPECT_LE((qr_residual<T>(Qp, F, A0, m, n, n, p.ld, p.ld)), residual_tol<T>(m, n))
            << "the routed ormqr does not reproduce A from the NATIVE geqrf's reflectors at b="
            << b << " -- the tau/reflector convention disagrees";
    }
}

// Direction B -- the VENDOR's geqrf -> the NATIVE orgqr. The `if constexpr` over a
// dependent expression discards the body, which keeps this compiling with no cuBLAS.
TYPED_TEST(GeqrfTest, VendorFactorFeedsTheNativeOrgqr) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::BackendType;
    if constexpr (!dispatch::factorization_vendor_available<B>) {
        GTEST_SKIP() << "no factorization vendor in this build";
    } else {
        const int m = 64, n = 40, batch = 3;
        auto p = make_problem<T>(m, n, batch, 1357u);
        auto V = view_of(p);
        const std::size_t vws =
            backend::geqrf_vendor_buffer_size<B, T>(*this->ctx, V, p.tau.to_span());
        UnifiedVector<std::byte> vw(vws ? vws : 1);
        backend::geqrf_vendor<B, T>(*this->ctx, V, p.tau.to_span(), vw.to_span());
        this->ctx->wait();

        // The native orgqr overwrites its A with Q, so keep the factor.
        const std::vector<T> F(p.buf.begin(), p.buf.end());

        auto apply = [](Queue& c, const MatrixView<T, MatrixFormat::Dense>& oa,
                        const MatrixView<T, MatrixFormat::Dense>& oc, Side os, Transpose ot,
                        Span<T> ot2, Span<std::byte> ows, int32_t obs) {
            return ormqr<B, T>(c, oa, oc, os, ot, ot2, ows, obs);
        };
        auto applybs = [](Queue& c, const MatrixView<T, MatrixFormat::Dense>& oa,
                          const MatrixView<T, MatrixFormat::Dense>& oc, Side os, Transpose ot,
                          Span<T> ot2, int32_t obs) {
            return ormqr_buffer_size<B, T>(c, oa, oc, os, ot, ot2, obs);
        };
        const std::size_t ows =
            sycl_orgqr::orgqr_blocked_buffer_size<T>(*this->ctx, V, p.tau.to_span(), applybs);
        UnifiedVector<std::byte> ow(ows ? ows : 1);
        ASSERT_NO_THROW(sycl_orgqr::orgqr_blocked_dispatch<T>(
            *this->ctx, V, p.tau.to_span(), ow.to_span(), apply, applybs));
        this->ctx->wait();

        for (int b = 0; b < batch; ++b) {
            const T* Q = p.buf.data() + static_cast<size_t>(b) * p.stride;
            const T* Fb = F.data() + static_cast<size_t>(b) * p.stride;
            const T* A0 = p.a0.data() + static_cast<size_t>(b) * p.stride;
            const auto Qp = promote_Q<T>(Q, m, n, p.ld);
            EXPECT_LE(orth_of_promoted(Qp, m, n), orth_tol<T>(m, n)) << "b=" << b;
            EXPECT_LE((qr_residual<T>(Qp, Fb, A0, m, n, n, p.ld, p.ld)), residual_tol<T>(m, n))
                << "the native orgqr does not reproduce A from the VENDOR's reflectors at b="
                << b << " -- the tau/reflector convention disagrees";
        }
    }
}

// G8b. THE CONVENTION, GUARDED WITHOUT A VENDOR. Flipping LAPACK's sign choice leaves qr,
// orth and qrQ green for every type, so the oracle is an independent host xGEQR2 in double
// (beta = -SIGN(||x||, Re(alpha)), tau = (beta - alpha)/beta).
// evidence: docs/perf/qr.md#a-residual-test-cannot-guard-a-convention
TYPED_TEST(GeqrfTest, ConventionMatchesReferenceLapackWithoutAVendor) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;
    using D = typename TestFixture::D;

    // An independent, unblocked host QR in double: LAPACK xGEQR2 + xLARFG.
    auto host_geqr2 = [](const T* A0, int m, int n, int ld,
                         std::vector<D>& rdiag, std::vector<D>& taus) {
        const int k = std::min(m, n);
        std::vector<D> W(static_cast<size_t>(m) * n);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i)
                W[static_cast<size_t>(j) * m + i] = up(A0[static_cast<size_t>(j) * ld + i]);
        rdiag.assign(k, D(0));
        taus.assign(k, D(0));

        for (int j = 0; j < k; ++j) {
            // xLARFG on W(j:m-1, j).
            const D alpha = W[static_cast<size_t>(j) * m + j];
            double xnorm2 = 0.0;
            for (int i = j + 1; i < m; ++i) {
                const double a = habs(W[static_cast<size_t>(j) * m + i]);
                xnorm2 += a * a;
            }
            if (xnorm2 == 0.0 && himag(alpha) == 0.0) {
                // H = I. tau = 0, the column below the diagonal is already zero.
                rdiag[j] = alpha;
                taus[j] = D(0);
                continue;
            }
            const double anorm =
                std::sqrt(hreal(alpha) * hreal(alpha) + himag(alpha) * himag(alpha) + xnorm2);
            // beta = -SIGN(||[alpha; x]||, Re(alpha)); Fortran SIGN treats +0 as +.
            const double beta_r = (hreal(alpha) >= 0.0) ? -anorm : anorm;
            const D beta = D(beta_r);           // REAL beta, the LAPACK convention
            const D tau = (beta - alpha) / beta;
            const D d = alpha - beta;
            // v = [1; x/d]; overwrite the column with v below the diagonal.
            for (int i = j + 1; i < m; ++i) W[static_cast<size_t>(j) * m + i] /= d;
            W[static_cast<size_t>(j) * m + j] = beta;
            rdiag[j] = beta;
            taus[j] = tau;

            // Apply H = I - tau v v^H to the trailing columns.
            for (int c = j + 1; c < n; ++c) {
                D w = W[static_cast<size_t>(c) * m + j];   // v(j) == 1
                for (int i = j + 1; i < m; ++i)
                    w += hconj(W[static_cast<size_t>(j) * m + i]) *
                         W[static_cast<size_t>(c) * m + i];
                // conj(TAU), NOT TAU: zgeqr2 forms the reflector with zlarfg and applies it with
                // DCONJG(TAU), because reducing A from the left applies H^H. hconj is the identity
                // for a real T, so half the type list cannot see this line.
                const D f = hconj(tau) * w;
                W[static_cast<size_t>(c) * m + j] -= f;
                for (int i = j + 1; i < m; ++i)
                    W[static_cast<size_t>(c) * m + i] -= f * W[static_cast<size_t>(j) * m + i];
            }
        }
    };

    // Both leaves and both drivers: CTA-resident, an exact multiple of nb, and not.
    struct S { int m, n; bool blocked; };
    const S shapes[] = {{40, 24, false}, {96, 96, true}, {100, 70, true}};

    for (const S& s : shapes) {
        if (!s.blocked && !this->cta_fits(s.m, s.n)) continue;
        auto p = make_problem<T>(s.m, s.n, 3, 91237u + unsigned(s.n));
        auto V = view_of(p);
        UnifiedVector<std::byte> wb(std::max<std::size_t>(
            1, s.blocked ? sycl_geqrf::geqrf_blocked_buffer_size<T>(*this->ctx, V)
                         : sycl_geqrf::geqrf_cta_buffer_size<T>(*this->ctx, V)));
        if (s.blocked) {
            sycl_geqrf::geqrf_blocked_dispatch<T>(*this->ctx, V, p.tau.to_span(), wb.to_span(), {});
        } else {
            sycl_geqrf::geqrf_cta_dispatch<T>(*this->ctx, V, p.tau.to_span(), wb.to_span());
        }
        this->ctx->wait();

        for (int b = 0; b < p.batch; ++b) {
            const T* F = p.buf.data() + static_cast<size_t>(b) * p.stride;
            const T* A0 = p.a0.data() + static_cast<size_t>(b) * p.stride;
            const T* tau = p.tau.data() + static_cast<size_t>(b) * p.k;

            std::vector<D> rdiag, taus;
            host_geqr2(A0, s.m, s.n, p.ld, rdiag, taus);

            // The reference is unblocked and the kernel is not. The sign check below has no tolerance.
            const double tol = 512.0 * double(std::numeric_limits<R>::epsilon()) *
                               double(s.m + s.n);

            for (int j = 0; j < p.k; ++j) {
                const D rj = up(F[static_cast<size_t>(j) * p.ld + j]);
                const D tj = up(tau[j]);

                if constexpr (!test_utils::is_complex<T>::value) {
                    // Nothing else in this file notices a sign flip; a zero diagonal is the
                    // tau == 0 case and has no sign.
                    if (habs(rdiag[j]) > 0.0 && habs(rj) > 0.0) {
                        EXPECT_EQ(hreal(rj) < 0.0, hreal(rdiag[j]) < 0.0)
                            << "R(" << j << "," << j << ") has the WRONG SIGN: got "
                            << hreal(rj) << ", reference LAPACK xGEQR2 gives "
                            << hreal(rdiag[j]) << " (m=" << s.m << " n=" << s.n
                            << " b=" << b << "). geqrf's beta convention is its contract "
                               "with ormqr/orgqr/ormbr/sy2sb, and no residual test sees this.";
                    }
                }

                EXPECT_LE(habs(tj - taus[j]), tol * std::max(1.0, habs(taus[j])))
                    << "tau[" << j << "] disagrees with reference LAPACK xGEQR2 (m=" << s.m
                    << " n=" << s.n << " b=" << b << ")";

                EXPECT_LE(std::fabs(habs(rj) - habs(rdiag[j])),
                          tol * std::max(1.0, habs(rdiag[j])))
                    << "|R(" << j << "," << j << ")| disagrees with reference LAPACK xGEQR2"
                    << " (m=" << s.m << " n=" << s.n << " b=" << b << ")";
            }
        }
    }
}

// G8c. geqrf_larfg_scalars' division arm, on data that reaches it. That arm is taken ONLY
// on SUBNORMAL input, so a "simplify this to a plain reciprocal-multiply" edit would fill
// every such column's v with infinities and nothing would go red. Hence FINITENESS and a
// loose, scale-invariant orthogonality rather than a residual: ||A||_F^2 underflows to
// zero at this magnitude and the oracle returns nan. The `!dev_is_zero(r)` half is
// uncovered: docs/perf/qr.md#open-debts
TYPED_TEST(GeqrfTest, SubnormalScaleColumnsTakeTheDivisionPath) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;

    // Below the smallest NORMAL of R, so that 1/(alpha - beta) overflows.
    const double scale = std::is_same_v<R, float> ? 1e-41 : 1e-311;

    struct S { int m, n; bool blocked; };
    const S shapes[] = {{64, 8, false}, {96, 96, true}};

    for (const S& s : shapes) {
        if (!s.blocked && !this->cta_fits(s.m, s.n)) continue;
        auto p = make_problem<T>(s.m, s.n, 2, 4441u + unsigned(s.n));

        // Rescale the WINDOW only: the pad keeps its poison, far out of window at this scale.
        for (int b = 0; b < p.batch; ++b)
            for (int j = 0; j < s.n; ++j)
                for (int i = 0; i < s.m; ++i) {
                    const size_t o = static_cast<size_t>(b) * p.stride +
                                     static_cast<size_t>(j) * p.ld + i;
                    p.buf[o] = mk<T>(hreal(up(p.buf[o])) * scale, himag(up(p.buf[o])) * scale);
                }
        p.a0.assign(p.buf.begin(), p.buf.end());

        double biggest = 0.0;
        for (int j = 0; j < s.n; ++j)
            for (int i = 0; i < s.m; ++i)
                biggest = std::max(biggest, habs(up(p.buf[static_cast<size_t>(j) * p.ld + i])));
        ASSERT_GT(biggest, 0.0)
            << "the rescaled problem flushed to all zeros; this test would prove nothing";
        ASSERT_FALSE(std::isfinite(R(1) / static_cast<R>(biggest * 16.0)))
            << "1/(" << biggest * 16.0 << ") is finite in this scalar type, so "
            << "geqrf_larfg_scalars keeps use_mul == true and the DIVISION path is never "
               "reached. This test is vacuous as configured.";

        auto V = view_of(p);
        UnifiedVector<std::byte> wb(std::max<std::size_t>(
            1, s.blocked ? sycl_geqrf::geqrf_blocked_buffer_size<T>(*this->ctx, V)
                         : sycl_geqrf::geqrf_cta_buffer_size<T>(*this->ctx, V)));
        if (s.blocked) {
            sycl_geqrf::geqrf_blocked_dispatch<T>(*this->ctx, V, p.tau.to_span(), wb.to_span(), {});
        } else {
            sycl_geqrf::geqrf_cta_dispatch<T>(*this->ctx, V, p.tau.to_span(), wb.to_span());
        }
        this->ctx->wait();

        const char* what = s.blocked ? "subnormal-scale blocked" : "subnormal-scale cta";
        for (int b = 0; b < p.batch; ++b) {
            const T* F = p.buf.data() + static_cast<size_t>(b) * p.stride;
            const T* tau = p.tau.data() + static_cast<size_t>(b) * p.k;

            // (1) Finiteness -- the property the division arm exists to preserve.
            for (int i = 0; i < p.k; ++i) {
                ASSERT_TRUE(std::isfinite(hreal(up(tau[i]))) && std::isfinite(himag(up(tau[i]))))
                    << what << ": tau[" << i << "] is not finite at b=" << b
                    << " -- the reciprocal guard in geqrf_larfg_scalars is not doing its job";
            }
            for (int j = 0; j < s.n; ++j)
                for (int i = 0; i < s.m; ++i) {
                    const auto v = up(F[static_cast<size_t>(j) * p.ld + i]);
                    ASSERT_TRUE(std::isfinite(hreal(v)) && std::isfinite(himag(v)))
                        << what << ": factor element (" << i << "," << j << ") is not finite at b="
                        << b << " -- 1/(alpha-beta) overflowed and was used anyway";
                }

            // (2) Orthogonality at a tolerance set by the INPUT's precision, not the kernel's:
            //     v is scale-invariant, and a subnormal carries ~5 bits for float, ~35 for double.
            const auto Q = host_form_Q<T>(F, tau, p.m, p.k, p.ld);
            const double orth = orth_of_promoted(Q, p.m, p.k);
            if (verbose()) {
                std::printf("[verbose] %-28s m=%4d n=%4d b=%d scale=%.1e orth=%.4e\n", what, p.m,
                            p.n, b, scale, orth);
                std::fflush(stdout);
            }
            ASSERT_TRUE(std::isfinite(orth))
                << what << ": ||Q^H Q - I|| is not finite at b=" << b;
            EXPECT_LE(orth, 1e-2)
                << what << ": the reflectors from subnormal-scale columns are not orthonormal at b="
                << b << " (orth=" << orth << "). v is scale-invariant, so this does not depend on "
                   "the input magnitude -- it means the divisor was wrong.";
        }
    }
}

// The drop-in property itself: the native factor is the vendor's, ELEMENTWISE,
// tau included -- what makes geqrf substitutable underneath sy2sb and band_reduction.
TYPED_TEST(GeqrfTest, NativeFactorMatchesTheVendorElementwise) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::BackendType;
    if constexpr (!dispatch::factorization_vendor_available<B>) {
        GTEST_SKIP() << "no factorization vendor in this build";
    } else {
        struct S { int m, n; bool blocked; };
        const S shapes[] = {{40, 24, false}, {96, 96, true}, {100, 70, true}};
        for (const S& s : shapes) {
            auto p = make_problem<T>(s.m, s.n, 3, 5309u + unsigned(s.n));
            auto V = view_of(p);
            const std::size_t vws =
                backend::geqrf_vendor_buffer_size<B, T>(*this->ctx, V, p.tau.to_span());
            UnifiedVector<std::byte> vw(vws ? vws : 1);
            backend::geqrf_vendor<B, T>(*this->ctx, V, p.tau.to_span(), vw.to_span());
            this->ctx->wait();
            const std::vector<T> Fv(p.buf.begin(), p.buf.end());
            const std::vector<T> tv(p.tau.begin(), p.tau.end());

            reset(p);
            UnifiedVector<std::byte> wb(std::max<std::size_t>(
                1, s.blocked ? sycl_geqrf::geqrf_blocked_buffer_size<T>(*this->ctx, V)
                             : sycl_geqrf::geqrf_cta_buffer_size<T>(*this->ctx, V)));
            if (s.blocked)
                sycl_geqrf::geqrf_blocked_dispatch<T>(*this->ctx, V, p.tau.to_span(), wb.to_span());
            else
                sycl_geqrf::geqrf_cta_dispatch<T>(*this->ctx, V, p.tau.to_span(), wb.to_span());
            this->ctx->wait();

            // A RELATIVE elementwise bound: the two do not share a reduction order.
            double scale = 0.0, worst = 0.0, tworst = 0.0, tscale = 0.0;
            for (size_t i = 0; i < Fv.size(); ++i) scale = std::max(scale, habs(up(Fv[i])));
            for (size_t i = 0; i < Fv.size(); ++i)
                worst = std::max(worst, habs(up(Fv[i]) - up(p.buf[i])));
            for (size_t i = 0; i < tv.size(); ++i) tscale = std::max(tscale, habs(up(tv[i])));
            for (size_t i = 0; i < tv.size(); ++i)
                tworst = std::max(tworst, habs(up(tv[i]) - up(p.tau[i])));
            const double dF = scale > 0 ? worst / scale : worst;
            const double dtau = tscale > 0 ? tworst / tscale : tworst;
            const double tol = 64.0 * double(std::numeric_limits<RealOf<T>>::epsilon());
            if (verbose()) {
                std::printf("[verbose] dropin m=%4d n=%4d dF=%.4e dtau=%.4e tol=%.4e\n", s.m,
                            s.n, dF, dtau, tol);
                std::fflush(stdout);
            }
            EXPECT_LE(dF, tol) << "the native factor is not the vendor's, m=" << s.m
                               << " n=" << s.n;
            EXPECT_LE(dtau, tol) << "the native tau is not the vendor's, m=" << s.m
                                 << " n=" << s.n;
        }
    }
}

// G8. m < n, m == n, m > n. route_geqrf.hh gate 2 refuses m < n as a CORRECTNESS gate:
// a wide view walks the trailing update past the bottom of the panel. The direct entry
// points are reachable without the table, so both are checked.
TYPED_TEST(GeqrfTest, WideIsRefusedAndTallAndSquareAreNot) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::BackendType;
    using Tbl = dispatch::RouteTable<dispatch::Op::geqrf, T>;
    const dispatch::Route cta{dispatch::Origin::Native, dispatch::Algorithm::CTA};
    const dispatch::Route blk{dispatch::Origin::Native, dispatch::Algorithm::Blocked};

    // (a) the TABLE refuses m < n on both native arms, and accepts m == n and m > n.
    {
        auto p = make_problem<T>(24, 40, 2, 11u);           // WIDE
        auto V = view_of(p);
        const auto s = backend::geqrf_op_shape<B, T>(*this->ctx, V);
        ASSERT_TRUE(s.has_value());
        EXPECT_FALSE(Tbl::supports(cta, *s));
        EXPECT_FALSE(Tbl::supports(blk, *s));
    }
    const std::pair<int, int> ok[] = {{40, 40}, {64, 40}};
    for (const auto& dims : ok) {
        auto p = make_problem<T>(dims.first, dims.second, 2, 12u);
        auto V = view_of(p);
        const auto s = backend::geqrf_op_shape<B, T>(*this->ctx, V);
        ASSERT_TRUE(s.has_value());
        EXPECT_TRUE(Tbl::supports(blk, *s)) << dims.first << "x" << dims.second;
        EXPECT_EQ(Tbl::supports(cta, *s), this->cta_fits(dims.first, dims.second));
    }

    // (b) the KERNELS refuse it too, rather than returning wrong numbers.
    {
        auto p = make_problem<T>(24, 40, 2, 13u);
        auto V = view_of(p);
        UnifiedVector<std::byte> wb(4096);
        EXPECT_THROW(
            sycl_geqrf::geqrf_cta_dispatch<T>(*this->ctx, V, p.tau.to_span(), wb.to_span()),
            std::invalid_argument);
        EXPECT_THROW(
            sycl_geqrf::geqrf_blocked_dispatch<T>(*this->ctx, V, p.tau.to_span(), wb.to_span()),
            std::invalid_argument);
    }
}

// The remaining direct-entry-point gates. Dropping the heterogeneous one is not an error
// but a SILENT WRONG ANSWER: one launch covers the batch with a single (m, n, ld, stride)
// tuple and reads at the CAPACITY extents.
TYPED_TEST(GeqrfTest, DirectEntryPointsRefuseWhatSupportsRefuses) {
    using T = typename TestFixture::T;
    auto p = make_problem<T>(32, 16, 4, 14u);
    auto V = view_of(p);
    UnifiedVector<std::byte> wb(std::max<std::size_t>(
        1, sycl_geqrf::geqrf_blocked_buffer_size<T>(*this->ctx, V)));

    // (a) heterogeneous batch.
    UnifiedVector<int> ar(4), ac(4);
    for (int b = 0; b < 4; ++b) { ar[b] = 32 - b; ac[b] = 16 - b; }
    auto H = V.with_active_dims(ar.to_span(), ac.to_span());
    ASSERT_TRUE(H.is_heterogeneous())
        << "the view is not actually heterogeneous; this case would prove nothing";
    EXPECT_THROW(sycl_geqrf::geqrf_cta_dispatch<T>(*this->ctx, H, p.tau.to_span(), wb.to_span()),
                 std::invalid_argument);
    EXPECT_THROW(
        sycl_geqrf::geqrf_blocked_dispatch<T>(*this->ctx, H, p.tau.to_span(), wb.to_span()),
        std::invalid_argument);

    // (b) a tau span shorter than k * batch. tau is packed per matrix with stride k OF
    // THE WHOLE MATRIX, so a short span is an out-of-bounds write, not a smaller problem.
    Span<T> shortTau(p.tau.data(), p.tau.size() - 1);
    EXPECT_THROW(sycl_geqrf::geqrf_cta_dispatch<T>(*this->ctx, V, shortTau, wb.to_span()),
                 std::invalid_argument);
    EXPECT_THROW(sycl_geqrf::geqrf_blocked_dispatch<T>(*this->ctx, V, shortTau, wb.to_span()),
                 std::invalid_argument);
}

// G9. The route table and the vendor-free fallback, through the SHAPE BUILDER against
// a real device -- the half tests/route_vocabulary_tests.cc cannot see.
TYPED_TEST(GeqrfTest, RouteTableAndTheVendorFreeFallback) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::BackendType;

    auto p = make_problem<T>(96, 96, 2, 15u);
    auto V = view_of(p);

    // With NO vendor a supported native route must be handed over (route_resolve.hh:60-63).
    const auto free_route =
        backend::geqrf_route<B, T>(*this->ctx, V, /*vendor_available=*/false);
    ASSERT_TRUE(dispatch::is_native(free_route))
        << "a vendor-free build has no geqrf route for 96x96; the fallback is broken";

    // preferred() is false everywhere for geqrf, so Auto must take the vendor.
    // evidence: docs/perf/qr.md#route-arms
    if (!std::getenv("BATCHLAS_GEQRF_ROUTE")) {
        const auto auto_route =
            backend::geqrf_route<B, T>(*this->ctx, V, /*vendor_available=*/true);
        EXPECT_TRUE(dispatch::is_vendor(auto_route))
            << "preferred() is documented as false everywhere for geqrf; it is not";
    }

    // The same fallback for orgqr.
    const auto ofree = backend::orgqr_route<B, T>(*this->ctx, V, /*vendor_available=*/false);
    ASSERT_TRUE(dispatch::is_native(ofree))
        << "a vendor-free build has no orgqr route for 96x96";
    if (!std::getenv("BATCHLAS_ORGQR_ROUTE")) {
        const auto oauto = backend::orgqr_route<B, T>(*this->ctx, V, /*vendor_available=*/true);
        EXPECT_TRUE(dispatch::is_vendor(oauto))
            << "preferred() is documented as false everywhere for orgqr; it is not";
    }
}

// G9b. The native-vs-native tie-break lives in RouteTable::native_tier_preferred, NOT in
// supports(): both arms must stay supports()-true on both sides of the crossover, or a
// pinned `cta` falls through to automatic() and measures something else, and the same
// window in preferred() would flip the vendor-present answer too.
// evidence: docs/perf/qr.md#cta-vs-blocked-crossover
TYPED_TEST(GeqrfTest, NativeTierTieBreakPicksTheFasterNativeVendorFree) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;
    static constexpr Backend B = TestFixture::BackendType;

    // The crossover route_geqrf.hh declares; both complex types stay on CTA to the top of
    // their capacity, so for them only the supports() half runs.
    const bool has_crossover =
        std::is_same_v<T, float> || (std::is_same_v<T, double> && !test_utils::is_complex<T>::value);
    const int nc = std::is_same_v<R, float> ? 96 : 48;   // last n that prefers CTA
    const int above = std::is_same_v<R, float> ? 128 : 64;  // first n that prefers Blocked

    if (!has_crossover) {
        GTEST_SKIP() << "no measured native crossover for this type (route_geqrf.hh: both "
                        "complex types stay on CTA to the top of their SLM capacity)";
    }

    // Both shapes must be CTA-ELIGIBLE, or "blocked was chosen" proves nothing.
    ASSERT_TRUE(this->cta_fits(nc, nc))
        << "the below-crossover shape " << nc << "x" << nc << " does not fit the CTA tile on this "
        << "device, so this test cannot see the tie-break";
    ASSERT_TRUE(this->cta_fits(above, above))
        << "the above-crossover shape " << above << "x" << above << " does not fit the CTA tile "
        << "on this device, so 'blocked was chosen' would prove nothing -- CTA was never a "
           "candidate. The tie-break is untested here.";

    auto p_lo = make_problem<T>(nc, nc, 2, 771u);
    auto p_hi = make_problem<T>(above, above, 2, 773u);
    auto V_lo = view_of(p_lo);
    auto V_hi = view_of(p_hi);

    // (1) The tie-break itself, through the real shape builder.
    const auto lo_free = backend::geqrf_route<B, T>(*this->ctx, V_lo, /*vendor_available=*/false);
    const auto hi_free = backend::geqrf_route<B, T>(*this->ctx, V_hi, /*vendor_available=*/false);
    if (!std::getenv("BATCHLAS_GEQRF_ROUTE")) {
        EXPECT_EQ(lo_free.algo, dispatch::Algorithm::CTA)
            << "vendor-free at n=" << nc << " should take the CTA tier (tier_summary.txt has it "
            << "ahead there); got " << dispatch::to_string(lo_free.algo);
        EXPECT_EQ(hi_free.algo, dispatch::Algorithm::Blocked)
            << "vendor-free at n=" << above << " should take the BLOCKED tier -- CTA is measured "
            << "1.37x-1.43x slower there and both are linked into this build; got "
            << dispatch::to_string(hi_free.algo);
    }

    // (2) Neither arm may have lost supports() anywhere -- the half that keeps a pin honest.
    {
        const auto sh_lo = backend::geqrf_op_shape<B, T>(*this->ctx, V_lo);
        const auto sh_hi = backend::geqrf_op_shape<B, T>(*this->ctx, V_hi);
        ASSERT_TRUE(sh_lo.has_value() && sh_hi.has_value());
        using Tbl = dispatch::RouteTable<dispatch::Op::geqrf, T>;
        const dispatch::Route cta{dispatch::Origin::Native, dispatch::Algorithm::CTA};
        const dispatch::Route blk{dispatch::Origin::Native, dispatch::Algorithm::Blocked};
        EXPECT_TRUE(Tbl::supports(cta, *sh_hi))
            << "the CTA arm lost supports() ABOVE the crossover -- the tier window was moved into "
               "supports(), which makes a forced `cta` fall through to automatic() "
               "(route_resolve.hh:101) and measure something else. It belongs in "
               "native_tier_preferred().";
        EXPECT_TRUE(Tbl::supports(blk, *sh_lo))
            << "the blocked arm lost supports() BELOW the crossover -- same defect, other "
               "direction (route_geqrf.hh's 'NO LOWER BOUND ON THE EXTENTS' note).";
    }

    // (3) The vendor-present answer must not have moved: native_tier_preferred is consulted
    //     only on the vendor-free walk, preferred() always.
    if (!std::getenv("BATCHLAS_GEQRF_ROUTE")) {
        EXPECT_TRUE(dispatch::is_vendor(
            backend::geqrf_route<B, T>(*this->ctx, V_lo, /*vendor_available=*/true)))
            << "the native tier tie-break leaked into the vendor-present decision at n=" << nc;
        EXPECT_TRUE(dispatch::is_vendor(
            backend::geqrf_route<B, T>(*this->ctx, V_hi, /*vendor_available=*/true)))
            << "the native tier tie-break leaked into the vendor-present decision at n=" << above;
    }
}

// G10. THE FACADE REACHES THE KERNEL, guarded by BIT-EXACTNESS against the direct entry
// point, not a residual: a residual bound is satisfied by either implementation. The
// blocked arm must inject the SAME routed gemm the facade injects.
TYPED_TEST(GeqrfTest, FacadeReachesTheCtaKernel) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::BackendType;
    const int m = 40, n = 24, batch = 3;
    ASSERT_TRUE(this->cta_fits(m, n));

    GeqrfEnvGuard guard("cta");
    auto p = make_problem<T>(m, n, batch, 999u);
    auto V = view_of(p);

    // Localises a failure; it is NOT the guard.
    const auto route = backend::geqrf_route<B, T>(*this->ctx, V, /*vendor_available=*/true);
    ASSERT_TRUE(dispatch::is_native(route))
        << "BATCHLAS_GEQRF_ROUTE=cta did not resolve to a native route";
    ASSERT_EQ(route.algo, dispatch::Algorithm::CTA);

    UnifiedVector<std::byte> ws(std::max<std::size_t>(
        1, geqrf_buffer_size<B, T>(*this->ctx, V, p.tau.to_span())));
    geqrf<B, T>(*this->ctx, V, p.tau.to_span(), ws.to_span());
    this->ctx->wait();
    const std::vector<T> facade(p.buf.begin(), p.buf.end());
    const std::vector<T> ftau(p.tau.begin(), p.tau.end());

    reset(p);
    UnifiedVector<std::byte> dws(std::max<std::size_t>(
        1, sycl_geqrf::geqrf_cta_buffer_size<T>(*this->ctx, V)));
    sycl_geqrf::geqrf_cta_dispatch<T>(*this->ctx, V, p.tau.to_span(), dws.to_span());
    this->ctx->wait();

    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                const size_t o =
                    static_cast<size_t>(b) * p.stride + static_cast<size_t>(j) * p.ld + i;
                ASSERT_EQ(hreal(up(facade[o])), hreal(up(p.buf[o])))
                    << "the facade did not run the CTA kernel: its answer differs from "
                       "geqrf_cta_dispatch's at (" << i << "," << j << ") b=" << b;
                ASSERT_EQ(himag(up(facade[o])), himag(up(p.buf[o])));
            }
        }
        for (int i = 0; i < p.k; ++i) {
            const size_t o = static_cast<size_t>(b) * p.k + i;
            ASSERT_EQ(hreal(up(ftau[o])), hreal(up(p.tau[o]))) << "tau differs at " << i;
            ASSERT_EQ(himag(up(ftau[o])), himag(up(p.tau[o])));
        }
    }
    check_one(p, "cta/facade");
}

TYPED_TEST(GeqrfTest, FacadeReachesTheBlockedDriver) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::BackendType;
    const int m = 100, n = 70, batch = 3;

    GeqrfEnvGuard guard("blocked");
    auto p = make_problem<T>(m, n, batch, 998u);
    auto V = view_of(p);

    const auto route = backend::geqrf_route<B, T>(*this->ctx, V, /*vendor_available=*/true);
    ASSERT_TRUE(dispatch::is_native(route))
        << "BATCHLAS_GEQRF_ROUTE=blocked did not resolve to a native route";
    ASSERT_EQ(route.algo, dispatch::Algorithm::Blocked);

    UnifiedVector<std::byte> ws(std::max<std::size_t>(
        1, geqrf_buffer_size<B, T>(*this->ctx, V, p.tau.to_span())));
    geqrf<B, T>(*this->ctx, V, p.tau.to_span(), ws.to_span());
    this->ctx->wait();
    const std::vector<T> facade(p.buf.begin(), p.buf.end());
    const std::vector<T> ftau(p.tau.begin(), p.tau.end());

    reset(p);
    UnifiedVector<std::byte> dws(std::max<std::size_t>(
        1, sycl_geqrf::geqrf_blocked_buffer_size<T>(*this->ctx, V)));
    sycl_geqrf::geqrf_blocked_dispatch<T>(
        *this->ctx, V, p.tau.to_span(), dws.to_span(),
        [](Queue& c, const MatrixView<T, MatrixFormat::Dense>& ga,
           const MatrixView<T, MatrixFormat::Dense>& gb,
           const MatrixView<T, MatrixFormat::Dense>& gc, T ga2, T gb2, Transpose gta,
           Transpose gtb, ComputePrecision gp) {
            return gemm<B, T>(c, ga, gb, gc, ga2, gb2, gta, gtb, gp);
        });
    this->ctx->wait();

    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                const size_t o =
                    static_cast<size_t>(b) * p.stride + static_cast<size_t>(j) * p.ld + i;
                ASSERT_EQ(hreal(up(facade[o])), hreal(up(p.buf[o])))
                    << "the facade did not run the blocked driver: its answer differs from "
                       "geqrf_blocked_dispatch's at (" << i << "," << j << ") b=" << b;
                ASSERT_EQ(himag(up(facade[o])), himag(up(p.buf[o])));
            }
        }
        for (int i = 0; i < p.k; ++i) {
            const size_t o = static_cast<size_t>(b) * p.k + i;
            ASSERT_EQ(hreal(up(ftau[o])), hreal(up(p.tau[o]))) << "tau differs at " << i;
        }
    }
    check_one(p, "blocked/facade");
}

// The facade's ORGQR arm, same discipline: the native driver is ormqr on an identity,
// so a facade that fell through to cuSOLVER would still produce an orthonormal Q.
TYPED_TEST(GeqrfTest, FacadeReachesTheNativeOrgqr) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::BackendType;
    const int m = 64, n = 40, batch = 3;

    auto p = make_problem<T>(m, n, batch, 606u);
    auto V = view_of(p);
    UnifiedVector<std::byte> gws(std::max<std::size_t>(
        1, sycl_geqrf::geqrf_blocked_buffer_size<T>(*this->ctx, V)));
    sycl_geqrf::geqrf_blocked_dispatch<T>(*this->ctx, V, p.tau.to_span(), gws.to_span());
    this->ctx->wait();
    const std::vector<T> F(p.buf.begin(), p.buf.end());

    OrgqrEnvGuard oguard("blocked");
    const auto route = backend::orgqr_route<B, T>(*this->ctx, V, /*vendor_available=*/true);
    ASSERT_TRUE(dispatch::is_native(route))
        << "BATCHLAS_ORGQR_ROUTE=blocked did not resolve to a native route";
    ASSERT_EQ(route.algo, dispatch::Algorithm::Blocked);

    UnifiedVector<std::byte> ows(std::max<std::size_t>(
        1, orgqr_buffer_size<B, T>(*this->ctx, V, p.tau.to_span())));
    orgqr<B, T>(*this->ctx, V, p.tau.to_span(), ows.to_span());
    this->ctx->wait();
    const std::vector<T> facade(p.buf.begin(), p.buf.end());

    std::copy(F.begin(), F.end(), p.buf.begin());
    auto apply = [](Queue& c, const MatrixView<T, MatrixFormat::Dense>& oa,
                    const MatrixView<T, MatrixFormat::Dense>& oc, Side os, Transpose ot,
                    Span<T> ot2, Span<std::byte> ws2, int32_t obs) {
        return ormqr<B, T>(c, oa, oc, os, ot, ot2, ws2, obs);
    };
    auto applybs = [](Queue& c, const MatrixView<T, MatrixFormat::Dense>& oa,
                      const MatrixView<T, MatrixFormat::Dense>& oc, Side os, Transpose ot,
                      Span<T> ot2, int32_t obs) {
        return ormqr_buffer_size<B, T>(c, oa, oc, os, ot, ot2, obs);
    };
    const std::size_t dwsz =
        sycl_orgqr::orgqr_blocked_buffer_size<T>(*this->ctx, V, p.tau.to_span(), applybs);
    UnifiedVector<std::byte> dw(dwsz ? dwsz : 1);
    sycl_orgqr::orgqr_blocked_dispatch<T>(*this->ctx, V, p.tau.to_span(), dw.to_span(), apply,
                                          applybs);
    this->ctx->wait();

    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                const size_t o =
                    static_cast<size_t>(b) * p.stride + static_cast<size_t>(j) * p.ld + i;
                ASSERT_EQ(hreal(up(facade[o])), hreal(up(p.buf[o])))
                    << "the facade did not run the native orgqr at (" << i << "," << j
                    << ") b=" << b;
                ASSERT_EQ(himag(up(facade[o])), himag(up(p.buf[o])));
            }
        }
        // And the thing it produced really is Q for THIS A.
        const T* Q = p.buf.data() + static_cast<size_t>(b) * p.stride;
        const T* Fb = F.data() + static_cast<size_t>(b) * p.stride;
        const T* A0 = p.a0.data() + static_cast<size_t>(b) * p.stride;
        const auto Qp = promote_Q<T>(Q, m, n, p.ld);
        EXPECT_LE(orth_of_promoted(Qp, m, n), orth_tol<T>(m, n)) << "b=" << b;
        EXPECT_LE((qr_residual<T>(Qp, Fb, A0, m, n, n, p.ld, p.ld)), residual_tol<T>(m, n))
            << "b=" << b;
    }
}

// G11. Two workspace contracts. (a) Neither query may dereference A.data_ptr() or
// tau.data(): band_reduction sizes against a MatrixView built on nullptr, so a read there
// segfaults. (b) The sizes must be MONOTONE in (rows, cols, batch): band_reduction queries
// at (m_max x nb_max) and calls at an m x r sub-view, so a non-monotone query silently
// under-allocates sytrd.
TYPED_TEST(GeqrfTest, BufferSizeIsMonotoneAndNeverDereferencesTheData) {
    using T = typename TestFixture::T;
    const int ms[] = {16, 32, 64, 128, 257};
    const int ns[] = {8, 16, 32, 64, 96};
    const int bs[] = {1, 3, 16};

    auto q = [&](int m, int n, int b) {
        // NULL data and NULL tau, exactly as band_reduction.cc does it.
        MatrixView<T, MatrixFormat::Dense> dummy(nullptr, m, n, m, m * n, b);  // NULL data
        return std::pair<std::size_t, std::size_t>{
            sycl_geqrf::geqrf_cta_buffer_size<T>(*this->ctx, dummy),
            sycl_geqrf::geqrf_blocked_buffer_size<T>(*this->ctx, dummy)};
    };

    for (int b : bs) {
        for (size_t i = 0; i + 1 < std::size(ms); ++i) {
            EXPECT_LE(q(ms[i], 32, b).second, q(ms[i + 1], 32, b).second)
                << "blocked buffer size decreased as rows grew (" << ms[i] << "->" << ms[i + 1]
                << ", b=" << b << ")";
            EXPECT_LE(q(ms[i], 32, b).first, q(ms[i + 1], 32, b).first);
        }
        for (size_t j = 0; j + 1 < std::size(ns); ++j) {
            EXPECT_LE(q(257, ns[j], b).second, q(257, ns[j + 1], b).second)
                << "blocked buffer size decreased as cols grew (" << ns[j] << "->" << ns[j + 1]
                << ", b=" << b << ")";
        }
    }
    for (size_t i = 0; i + 1 < std::size(bs); ++i) {
        EXPECT_LE(q(257, 96, bs[i]).second, q(257, 96, bs[i + 1]).second)
            << "blocked buffer size decreased as the batch grew";
    }
}

// The facade's query must cover EVERY supported native tier, not the one this resolution
// chose: a chosen-only size turns a query/call disagreement into an UNDER-allocation.
TYPED_TEST(GeqrfTest, BufferSizeCoversEverySupportedNativeTier) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::BackendType;
    const int m = 40, n = 24, batch = 4;
    ASSERT_TRUE(this->cta_fits(m, n)) << "this shape must be servable by BOTH tiers";

    auto p = make_problem<T>(m, n, batch, 17u);
    auto V = view_of(p);
    for (const char* pin : {"cta", "blocked"}) {
        GeqrfEnvGuard guard(pin);
        const std::size_t facade = geqrf_buffer_size<B, T>(*this->ctx, V, p.tau.to_span());
        EXPECT_GE(facade, sycl_geqrf::geqrf_cta_buffer_size<T>(*this->ctx, V)) << pin;
        EXPECT_GE(facade, sycl_geqrf::geqrf_blocked_buffer_size<T>(*this->ctx, V)) << pin;
        // And the pinned call actually runs inside it.
        reset(p);
        UnifiedVector<std::byte> ws(facade ? facade : 1);
        EXPECT_NO_THROW((geqrf<B, T>(*this->ctx, V, p.tau.to_span(), ws.to_span()))) << pin;
        this->ctx->wait();
        check_one(p, pin);
    }
}

// Break-sweep evidence for these tests: docs/perf/qr.md#break-sweeps

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
