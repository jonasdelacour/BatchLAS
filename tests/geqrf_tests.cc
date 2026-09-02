// Native batched GEQRF (+ the ORGQR that consumes its output) -- the WP5 tests.
//
// ---------------------------------------------------------------------------
// THE ORACLE IS NEVER THE VENDOR, AND THE ROUTE IS NEVER TRUSTED
// ---------------------------------------------------------------------------
// Both rules are inherited verbatim from tests/potrf_tests.cc:1-25, and both bite
// harder here.
//
//   * A VENDOR REFERENCE IS INERT IN A VENDOR-FREE BUILD. resolve_route falls
//     back to a supported NATIVE route when no vendor exists -- which is the code
//     under test -- so a "compare against cuSOLVER" test compares the kernel with
//     itself in exactly the build this campaign exists for.
//
//   * A FORCED ROUTE THAT supports() REJECTS SILENTLY BECOMES THE VENDOR.
//     route_resolve.hh:101 tests `if (Table::supports(forced, s)) return forced;`
//     and falls through to automatic() at :111, so a test that sets
//     BATCHLAS_GEQRF_ROUTE=cta and gets one gate wrong runs cuSOLVER and passes
//     GREEN over a kernel nothing executed.
//
// So every numerical test below calls sycl_geqrf::geqrf_cta_dispatch<T> /
// geqrf_blocked_dispatch<T> / sycl_orgqr::orgqr_blocked_dispatch<T> DIRECTLY --
// calls a vendor cannot serve -- and checks a HOST reference built here from the
// input this file generated: Q is formed on the host from the packed reflectors
// and tau, R is read out of the upper triangle, and the two residuals are
//
//     ||Q R - A||_F / ||A||_F        and        ||Q^H Q - I||_F / sqrt(k)
//
// in DOUBLE arithmetic regardless of T. That oracle depends on no other
// implementation in this tree. Three tests are about routing (FacadeReaches*) and
// each asserts BIT-EXACT agreement with the direct entry point before believing
// anything -- see the note on those.
//
// ---------------------------------------------------------------------------
// WHAT A RESIDUAL TEST CANNOT SEE, AND WHY THIS FILE HAS THREE MORE ORACLES
// ---------------------------------------------------------------------------
// WP5's kernel break K3 (docs/perf/qr.md#break-sweeps) replaced
// LAPACK's real-beta larfg convention with internal::larfg's phase-preserving
// one. THE QR, ORTHOGONALITY AND EXPLICIT-Q RESIDUALS ALL STAYED GREEN for every
// type: a phase-preserving factorisation is a perfectly good QR, it is just not
// the one ormqr/orgqr/sy2sb/netlib/cuSOLVER all agree on. A residual-only suite
// is BLIND to the drop-in property, which for geqrf IS the contract.
//
// Hence:
//   * ComplexRDiagonalIsExactlyReal   -- the LAPACK convention, checkable with no
//                                        second implementation, in EITHER build.
//   * NativeFactorMatchesTheVendorElementwise -- the drop-in property itself,
//                                        vendor builds only.
//   * TauConventionSurvivesTheRoutedOrmqr / VendorFactorFeedsTheNativeOrgqr --
//                                        the interface contract in both
//                                        directions, which is where a convention
//                                        error actually hurts a caller.
//
// ---------------------------------------------------------------------------
// EVERY TEST IN THIS FILE WAS BROKEN ON PURPOSE, AND THE RESULTS ARE RECORDED
// AT THE BOTTOM OF THIS FILE -- INCLUDING THE BREAKS THAT TURNED NOTHING RED.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Host arithmetic. EVERY reference computation below promotes to double (or
// complex<double>) before it accumulates, so a float residual measures the
// KERNEL's error and not the reference's.
// ---------------------------------------------------------------------------
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

// A deterministic LCG rather than <random>: the same stream must come out on
// every platform and every rerun, because several tests below assert that two
// batch items DIFFER and that assertion has to be about the data, not luck.
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 6364136223846793005ULL + 1442695040888963407ULL) {}
    double next() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return double(int32_t(uint32_t(s >> 32))) / 2147483648.0;
    }
};

// ---------------------------------------------------------------------------
// THE REFERENCE. Q is built by APPLYING the packed reflectors to the first k
// columns of I_m, in LAPACK's order Q = H_0 H_1 ... H_{k-1} with
// H_i = I - tau_i v_i v_i^H and v_i = [0 .. 0, 1, F(i+1:m-1, i)].
//
// It is deliberately the textbook definition rather than a clever rearrangement:
// this function IS the specification geqrf is being held to, so it has to be
// readable against LAPACK's own dgeqrf documentation line by line.
// ---------------------------------------------------------------------------
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

// Copy a device Q (ld) into the tight m x k double buffer the two probes want,
// so an EXPLICIT Q can be checked with exactly the same oracle.
template <typename T>
std::vector<typename Prom<T>::type> promote_Q(const T* Q, int m, int k, int ld) {
    using D = typename Prom<T>::type;
    std::vector<D> out(static_cast<size_t>(m) * k);
    for (int j = 0; j < k; ++j)
        for (int i = 0; i < m; ++i)
            out[static_cast<size_t>(j) * m + i] = up(Q[static_cast<size_t>(j) * ld + i]);
    return out;
}

// THE RESIDUAL BOUND, and it is a MEASURED one rather than a comfortable one.
//
// Householder QR's backward error is O(m k) eps ||A||. The constant was set by
// running the whole file with GEQRF_TESTS_VERBOSE=1 and taking the worst measured
// value over every shape, type and tier, then rounding up by one binary order --
// the record is at the bottom of this file. potrf_tests.cc:280-300 records what
// the alternative costs: a bound with 40-200x of slack that "an accuracy defect
// would have to be catastrophic to be seen" through.
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

// ---------------------------------------------------------------------------
// A batch of DISTINCT random matrices in a buffer with a padded ld and a stride
// that is not ld*cols.
//
// The padding is not decoration. trsm_native.cc:590-599 records the failure it
// guards: the 6-arg MatrixView constructor DEFAULTS stride to ld*cols when 0 is
// passed, after which every batch item but the first reads the wrong matrix, and
// there is a standing memory entry for the GEMM twin ("Native GEMM collapses on
// strided ld"). Every shape test in this file therefore runs at ld != rows and
// stride != ld*cols, so the two most consequential lines in each launcher are
// falsifiable by DEFAULT rather than in one dedicated test.
//
// The pad is filled with a large POISON so that reading outside the window is a
// wrong answer rather than merely a different one.
// ---------------------------------------------------------------------------
template <typename T>
struct Problem {
    int m = 0, n = 0, k = 0, batch = 0, ld = 0, stride = 0;
    UnifiedVector<T> buf;      // the working copy, overwritten by geqrf
    std::vector<T> a0;         // the pristine input, same ld/stride
    UnifiedVector<T> tau;
    // THE PER-ITEM POINTER ARRAY. Not optional and not decoration: the vendor
    // geqrf/orgqr are pointer-array APIs (cublas.cc), and a view built without
    // one throws "data_ptrs target is null" the moment a test crosses to the
    // vendor -- which is exactly what the drop-in tests below do.
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

// Restore the working buffer to the pristine input, poison and all, and re-poison
// tau so a tau slot the kernel never writes is visible as -12345 rather than as
// a plausible leftover.
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
        // supports()' own correctness gates, not a convenience: route_geqrf.hh
        // gate 3 (GPU only) and gate 4 (sub-group 32) reject everything else, and
        // both direct entry points re-apply them.
        if (this->ctx->device().type != DeviceType::GPU) {
            GTEST_SKIP() << "the native geqrf kernels are GPU-only (route_geqrf.hh gate 3)";
        }
        if (!this->ctx->device().supports_sub_group_size(32)) {
            GTEST_SKIP() << "device does not offer sub-group size 32 (route_geqrf.hh gate 4)";
        }
    }

    // The DEVICE's local-memory budget, spelled exactly as
    // src/backends/geqrf_route.hh:117-119 spells it. NOT device_limits.hh's
    // 49152, which cmake/BatchLASDetectSYCL.cmake:44-45 hardcodes for any
    // nvidia_gpu_sm_* pattern and which is 2.06x wrong on this box (WP4's W1).
    std::size_t budget() const {
        const std::size_t lm = static_cast<std::size_t>(
            this->ctx->device().get_property(DeviceProperty::LOCAL_MEM_SIZE));
        return lm > 4096 ? lm - 4096 : std::size_t(0);
    }

    bool cta_fits(int m, int n) const {
        return sycl_geqrf::geqrf_cta_fits<T>(m, n, budget());
    }

    // The blocked driver's OWN block width and OWN leading-panel leaf choice.
    // Queried, never hardcoded: potrf_native.hh:246-266 records why a test that
    // must straddle a block boundary and cannot see where the boundary is stops
    // testing anything the moment the width moves.
    int nb(int m, int n) const {
        return static_cast<int>(
            sycl_geqrf::geqrf_blocked_debug_params<T>(*this->ctx, m, n) & 0xffffu);
    }
    unsigned leaf(int m, int n) const {
        return sycl_geqrf::geqrf_blocked_debug_params<T>(*this->ctx, m, n) >> 16;
    }
};

// The two residuals plus the finiteness and batch-distinctness checks, run on
// EVERY batch item.
//
// EVERY BATCH ITEM IS CHECKED, NOT ITEM 0. Item 0 sits at offset 0, so a wrong
// batch stride cannot move it and a suite that checks only item 0 is blind to the
// entire class -- which is not hypothetical: WP5's kernel break K5 (a tau batch
// stride of ib instead of k) leaves item 0 perfect. The distinctness assertion at
// the end is what makes "the kernel broadcast item 0 over the batch" a failure
// rather than a pass.
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

// ===========================================================================
// G0. THE 48 KB LAUNCH HOLE. DECLARED FIRST ON PURPOSE.
//
// A resident-leaf launch that asks for EXACTLY 49,152 B of local memory is
// refused by the CUDA backend -- too big for the non-opt-in 48 KB limit once the
// kernel's static shared is added, not big enough for the UR adapter to raise
// MaxDynamicSharedMemorySize. WP4 found the band and padded potrf over it
// (potrf_cta.cc:258-296) and wrote down the condition that reopens it: a
// reduce_over_group in the body. geqr2_panel_device has two per reflector, so
// WP5's leaf was in the hole and shipped without the pad.
//
// MEASURED COLD, one process per point, before the pad was added:
//     48,896 B PASS   49,152 B FAIL   49,664 B PASS
// for cdouble (96x32 and 192x16), cfloat (192x32), double (384x16) and float
// (384x32) alike -- a BYTE threshold, not a shape or a type.
//
// WHY THIS TEST IS DECLARED FIRST, AND WHY THAT IS LOAD-BEARING RATHER THAN
// TIDINESS. The attribute the adapter sets is STICKY PER CUfunction, and one
// instantiation serves every panel shape of a type. Any earlier launch of a
// LARGER panel raises the cap for the rest of the process and this test can
// never fail again. That is exactly how the defect escaped: this file's
// BlockedResidualAndOrthogonality reaches 100x32 (51,200 B) before it reaches
// 96x32 (49,152 B) and is green with or without the pad, and it took
// orgqr_tests asking for cdouble 96x96 as the FIRST blocked shape in its
// process to expose it. GoogleTest runs a suite's tests in declaration order, so
// being first in this file is what keeps the guard cold. DO NOT MOVE IT, and do
// not add a resident-leaf launch above it.
//
// The cold check by hand, which is what the pad was verified with:
//     ./build/tests/geqrf_tests --gtest_filter='GeqrfTest/7.LaunchHole*'
// ===========================================================================
TYPED_TEST(GeqrfTest, ResidentLeafLaunchHoleAt48KiB) {
    using T = typename TestFixture::T;

    // The three byte sizes, expressed as element counts for THIS scalar type, and
    // split into an m x n panel with m >= n (supports() gate 2). 49,920 is the pad
    // target, included so the row above the band is exercised too.
    struct S { std::size_t bytes; int m, n; };
    const std::size_t sz = sizeof(T);
    const S rows[] = {
        {48896, static_cast<int>(48896 / sz / 16), 16},
        {49152, static_cast<int>(49152 / sz / 16), 16},
        {49664, static_cast<int>(49664 / sz / 16), 16},
    };

    for (const S& r : rows) {
        // ANTI-VACUITY: the row must actually ask for the byte count it names,
        // and the tile must actually be admissible to the resident leaf. Without
        // both, this test is a launch of some other size that proves nothing.
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

// ===========================================================================
// G1 / G2. THE FACTORISATION ITSELF, on both native tiers, over a residue ladder.
// ===========================================================================
TYPED_TEST(GeqrfTest, CtaResidualAndOrthogonality) {
    using T = typename TestFixture::T;
    struct S { int m, n, b; };
    // n%nb residues 0/1/2/8 against BOTH shipped widths (32 and 16), m == n and
    // m > n, and one shape whose k is smaller than either width.
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
    // Anti-vacuity: a capacity change that silently emptied the ladder would
    // otherwise leave this test green having exercised nothing.
    ASSERT_GE(ran, 6) << "only " << ran << " of the CTA shapes fit; this test has stopped "
                         "covering the tier it names";
}

TYPED_TEST(GeqrfTest, BlockedResidualAndOrthogonality) {
    using T = typename TestFixture::T;
    struct S { int m, n, b; };
    // 96x96 is here for a second reason: its leading cdouble panel is 96x32, i.e.
    // exactly 49,152 B, the byte size of the 48 KB launch hole G0 guards. It is
    // NOT a substitute for G0 -- by the time this test reaches it, 100x32
    // (51,200 B) has already raised the sticky per-CUfunction cap and the hole is
    // invisible here. That is precisely how the defect shipped.
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

// ===========================================================================
// G3. THE BLOCK BOUNDARY, STRADDLED ON PURPOSE AND ASSERTED TO BE STRADDLED.
//
// This is the sy2sb stage-1 SHORT FINAL PANEL class (memory: "sy2sb
// trailing-panel bug"), which produced wrong numbers with a green suite. The
// width comes from geqrf_blocked_debug_params, so this test follows the width if
// it ever moves; and the properties it depends on -- a multiple, a non-multiple,
// more than one panel, and a final panel strictly narrower than nb -- are
// ASSERTED rather than assumed, because a test that silently stops straddling is
// this repository's recorded blind-guard shape.
//
// ONE THING WP5's BASELINE MEASURED THAT INVALIDATES THE OBVIOUS VERSION OF THIS
// TEST: on a SQUARE REAL matrix, dropping the LAST reflector changes NOTHING --
// LAPACK's larfg returns tau = 0 for a 1x1 real trailing reflector. A short-final-
// panel test written on a square real matrix guards nothing. Hence the m > n rows.
// ===========================================================================
TYPED_TEST(GeqrfTest, ShortFinalPanelStraddlesTheBlockWidth) {
    using T = typename TestFixture::T;

    // Ask at a shape large enough that the answer is the type's own width and not
    // the min(nb, k) clamp, then assert the clamp does not move it at the shapes
    // actually used.
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

// ===========================================================================
// G4. BOTH PANEL LEAVES.
//
// The blocked driver's panel leaf has TWO residencies -- a local_accessor tile
// and a raw global pointer -- chosen per panel by geqrf_cta_fits. Two code paths
// a test cannot tell apart is the blind-guard shape, so this test ASSERTS which
// one it got from geqrf_blocked_debug_params' high half before it believes any
// residual, and it SEARCHES for the height that pushes the leading panel off the
// resident path rather than hardcoding one -- a hardcoded height silently stops
// exercising the global leaf the day the budget or the width changes.
// ===========================================================================
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

// ===========================================================================
// G5. A RANK-DEFICIENT INPUT.
//
// One exactly-zero column and one exact duplicate of an earlier column. The zero
// column makes larfg take its tau = 0 identity branch -- the branch whose
// work-group uniformity the kernel's `continue` depends on -- and the duplicate
// drives a trailing R diagonal to ~0. Neither may produce a NaN, and QR = A must
// still hold: a rank-deficient A has a perfectly well-defined QR.
// ===========================================================================
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

        // The zero column really did produce a null reflector at its own position
        // -- i.e. the identity branch was TAKEN rather than merely survived.
        for (int b = 0; b < p.batch; ++b) {
            const T* F = p.buf.data() + static_cast<size_t>(b) * p.stride;
            EXPECT_LE(habs(up(F[static_cast<size_t>(5) * p.ld + 5])), 1e-5)
                << "R(5,5) is not ~0 for a rank-deficient column, b=" << b;
        }
    }
}

// ===========================================================================
// G6. THE COMPLEX CONVENTION, WHICH NO RESIDUAL CAN SEE.
//
// LAPACK's clarfg/zlarfg -- and cuSOLVER, rocSOLVER and netlib with them --
// return a REAL beta, so R's diagonal is real. internal::larfg in this tree
// returns a phase-preserving complex one. WP5's kernel break K3 swapped the two:
// qr, orth and the explicit-Q residual ALL STAYED GREEN for every type, because a
// phase-preserving factorisation is a valid QR -- it is simply not the one every
// consumer of geqrf's output in this tree assumes. This is the cheapest possible
// detector, it needs no second implementation, and it works in a vendor-free
// build.
// ===========================================================================
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
                    // EXACTLY zero, not "small": the LAPACK convention WRITES a
                    // real scalar into A(j,j); it does not merely rotate the
                    // imaginary part down to rounding.
                    ASSERT_EQ(himag(up(F[static_cast<size_t>(i) * p.ld + i])), 0.0)
                        << "imag(R(" << i << "," << i << ")) != 0 at m=" << s.m << " n=" << s.n
                        << " b=" << b << " -- the larfg phase convention is not LAPACK's, and "
                           "every residual test in this file is blind to that";
                }
            }
        }
    }
}

// ===========================================================================
// G7. THE INTERFACE CONTRACT, IN BOTH DIRECTIONS.
//
// This is the single most likely silent failure in WP5: geqrf's output is not a
// number, it is a CONVENTION consumed by ormqr, orgqr, ormbr, sy2sb and
// band_reduction. Feeding it across an implementation boundary is the only test
// that can see a convention error at all, and G6 above sees only its complex half.
//
// Direction A -- NATIVE geqrf -> the ROUTED ormqr. In a vendor build that is
// cuBLAS/cuSOLVER's ormqr; in a vendor-free build it is the native ormqr. Either
// way it is a DIFFERENT implementation from the one that produced the reflectors.
// ===========================================================================
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

// Direction B -- the VENDOR's geqrf -> the NATIVE orgqr. Only a vendor build can
// run it; `if constexpr` over a dependent expression discards the body otherwise,
// which is what keeps this file compiling with no cuBLAS in the link.
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

// The drop-in property itself: the native factor is the vendor's, ELEMENTWISE,
// tau included. This is the strongest statement WP5 can make, and it is the one
// that makes geqrf substitutable underneath sy2sb and band_reduction. Vendor
// builds only, by construction.
// ===========================================================================
// G8b. THE CONVENTION, GUARDED WITHOUT A VENDOR.
//
// WHY THIS EXISTS. NativeFactorMatchesTheVendorElementwise below is the only
// test in this file that compares against a DIFFERENT IMPLEMENTATION, and it
// opens with GTEST_SKIP in a vendor-free build. So until this test, the build
// this whole work package exists for -- -DBATCHLAS_ENABLE_VENDOR_BLAS=OFF --
// had NO guard at all on geqrf's tau/beta convention for the real scalar types.
// ComplexRDiagonalIsExactlyReal covers the complex half vendor-free; the real
// half had nothing.
//
// AND A RESIDUAL TEST CANNOT COVER IT. This is measured, not asserted: kernel
// break K3 (docs/perf/qr.md#break-sweeps 4b) flipped LAPACK's sign
// choice in geqrf_larfg_scalars to `beta_s = (alphr >= 0) ? nrm : -nrm` and
// left qr, orth and qrQ GREEN FOR EVERY TYPE. The factorisation stays
// mathematically exact -- it is a different, equally valid QR -- so every
// ||QR-A|| and ||Q^H Q-I|| probe in check_one is blind to it, and so is
// TauConventionSurvivesTheRoutedOrmqr, because ormqr just applies
// I - tau v v^H to whatever tau and v it is handed. The convention is not a
// nicety: it is geqrf's CONTRACT with ormqr, orgqr, ormbr, sy2sb and
// band_reduction, and with any caller that expected a cuSOLVER drop-in.
//
// THE ORACLE IS AN INDEPENDENT HOST xGEQR2, written from the LAPACK reference
// (xLARFG's beta = -SIGN(||x||, Re(alpha)), tau = (beta - alpha)/beta) and
// evaluated in double regardless of T. It is a genuinely separate
// implementation -- unblocked, right-looking, no panels, no scratch, no
// device -- so a shared misunderstanding would have to be a misunderstanding of
// LAPACK itself, which is what the citation above is for.
//
// WHAT IS COMPARED, and why each is here:
//   * sign(Re(R(j,j))) for the REAL types. EXACT, and it is precisely the bit
//     K3 flips. No tolerance, so it cannot be loosened into vacuity.
//   * tau, elementwise. Catches the same flip for the COMPLEX types, where the
//     sign lives in a ratio rather than in a sign bit: flipping beta takes tau
//     from ~1 + |a|/|b| to ~1 - |a|/|b|, far outside any sane tolerance.
//   * |R(j,j)|, elementwise. Guards the magnitude half of the diagonal
//     independently of its sign.
// ===========================================================================
TYPED_TEST(GeqrfTest, ConventionMatchesReferenceLapackWithoutAVendor) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;
    using D = typename TestFixture::D;

    // An independent, unblocked host QR in double, LAPACK xGEQR2 + xLARFG.
    // Fills the R diagonal and tau, both promoted.
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
                // conj(TAU), NOT TAU, AND THIS IS LAPACK'S OWN ASYMMETRY.
                // zgeqr2 forms the reflector with zlarfg and then applies it
                // with `CALL ZLARF( 'Left', ..., DCONJG( TAU( I ) ), ... )` --
                // because reducing A from the left applies H^H, not H. Writing
                // `tau * w` here made this reference disagree with the kernel by
                // 1-4% for cfloat and cdouble while leaving float and double
                // exact, which is the same signature as the implementer's kernel
                // break KE (conj(tau) -> tau: RED for the complex types only,
                // GREEN for the real ones). For a real T hconj is the identity,
                // which is why that break -- and this line -- are invisible to
                // half the type list.
                const D f = hconj(tau) * w;
                W[static_cast<size_t>(c) * m + j] -= f;
                for (int i = j + 1; i < m; ++i)
                    W[static_cast<size_t>(c) * m + i] -= f * W[static_cast<size_t>(j) * m + i];
            }
        }
    };

    // Both leaves and both drivers: a CTA-resident shape, a blocked shape that is
    // an exact multiple of the block width, and one that is not.
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

        // EVERY BATCH ITEM, for the reason check_one gives.
        for (int b = 0; b < p.batch; ++b) {
            const T* F = p.buf.data() + static_cast<size_t>(b) * p.stride;
            const T* A0 = p.a0.data() + static_cast<size_t>(b) * p.stride;
            const T* tau = p.tau.data() + static_cast<size_t>(b) * p.k;

            std::vector<D> rdiag, taus;
            host_geqr2(A0, s.m, s.n, p.ld, rdiag, taus);

            // The reference is unblocked and the kernel is not, so the two sum in
            // different orders. This tolerance is a NUMERICAL one; the sign check
            // below carries no tolerance at all.
            const double tol = 512.0 * double(std::numeric_limits<R>::epsilon()) *
                               double(s.m + s.n);

            for (int j = 0; j < p.k; ++j) {
                const D rj = up(F[static_cast<size_t>(j) * p.ld + j]);
                const D tj = up(tau[j]);

                if constexpr (!test_utils::is_complex<T>::value) {
                    // THE EXACT BIT. K3 flips this and nothing else in the file
                    // notices. A zero diagonal is the tau == 0 case (a null
                    // column or the final 1x1 reflector) and has no sign.
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

// ===========================================================================
// G8c. THE RECIPROCAL GUARD IN geqrf_larfg_scalars, ON DATA THAT ACTUALLY
// REACHES IT.
//
// WHAT IS UNGUARDED WITHOUT THIS. geqrf_cta_device.hh carries ~30 lines arguing
// that this file does NOT inherit the tree-wide overflow gap at
// math-helpers.hh:411 / gebrd.cc:52 / sytrd_cta_device.hh:133, and the whole
// argument comes down to two lines:
//
//     out.use_mul = dev_isfinite(r) && !dev_is_zero(r);   // r = 1/(alpha-beta)
//     out.vfactor = out.use_mul ? r : d;   // multiply by 1/d, or DIVIDE by d
//
// The `false` arm -- per-element division instead of a reciprocal-multiply --
// had never executed in any test, harness or benchmark in this tree. Every data
// generator here draws from ~[-1, 1], and the arm needs 1/(alpha - beta) to stop
// being finite. A future "simplify this to a plain reciprocal-multiply" edit
// would turn every such column's v into infinities, and nothing would go red.
//
// HOW THIS REACHES IT, and the constraint that shapes the whole test.
// |alpha - beta| >= s, the column's scale, and |alpha - beta| ~ s*(1 + sqrt(m)).
// So 1/(alpha - beta) overflows only when s is BELOW the smallest normal of the
// scalar type -- 1.18e-38 for float, 2.23e-308 for double. THE BRANCH IS
// REACHABLE ONLY ON SUBNORMAL INPUT. That is not a defect, it is what the branch
// is for, but it dictates what this test can assert:
//
//   * NOT a tight residual. A subnormal float near 1e-41 carries about five
//     significant bits, so ||QR-A||/||A|| is limited by the INPUT's precision,
//     not the kernel's, and any tolerance tight enough to be interesting would
//     be measuring the wrong thing. Worse, ||A||_F^2 itself underflows to zero
//     at this magnitude and the existing oracle returns nan -- which is exactly
//     what the first version of this test did, and why it is not written that
//     way.
//   * FINITENESS, which is precisely what the branch buys and what deleting it
//     destroys, and ORTHOGONALITY of Q at a LOOSE and explicitly-justified
//     tolerance. Q is built from v and tau, both scale-invariant, so a broken
//     divisor shows up as inf/nan (caught by the finiteness sweep) or as a v
//     that is zero or wildly wrong (caught by orthogonality: with v = [1;0...]
//     the reflector I - tau v v^H is not unitary unless tau came from that same
//     v).
//
// THE ANTI-VACUITY WORK IS THE CONFIGURATION ASSERTION, and it is necessary and
// NOT sufficient -- this test was also confirmed to go RED with the division arm
// removed. It checks the thing the branch keys on: that the reciprocal of a
// number of the column's own magnitude really does overflow in type R.
//
// WHAT IS DELIBERATELY NOT TESTED: the `!dev_is_zero(r)` half. r == 0 needs
// |alpha - beta| > 2e323, i.e. alpha - beta must itself have overflowed to inf,
// which needs input at ~1e308. There vfactor becomes inf and v becomes exactly
// zero -- finite, but not a correct reflector. That is an open question recorded
// in docs/perf/qr.md#open-debts, not a property this test claims to hold.
// ===========================================================================
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

        // Rescale the WINDOW only. The pad keeps its +/-9.75e3 poison, which at
        // this magnitude is an astronomically out-of-window value -- so an
        // out-of-bounds read stays a wrong answer rather than a small one.
        for (int b = 0; b < p.batch; ++b)
            for (int j = 0; j < s.n; ++j)
                for (int i = 0; i < s.m; ++i) {
                    const size_t o = static_cast<size_t>(b) * p.stride +
                                     static_cast<size_t>(j) * p.ld + i;
                    p.buf[o] = mk<T>(hreal(up(p.buf[o])) * scale, himag(up(p.buf[o])) * scale);
                }
        p.a0.assign(p.buf.begin(), p.buf.end());

        // ANTI-VACUITY ON THE CONFIGURATION.
        double biggest = 0.0;
        for (int j = 0; j < s.n; ++j)
            for (int i = 0; i < s.m; ++i)
                biggest = std::max(biggest, habs(up(p.buf[static_cast<size_t>(j) * p.ld + i])));
        ASSERT_GT(biggest, 0.0)
            << "the rescaled problem flushed to all zeros; this test would prove nothing";
        // The branch keys on 1/(alpha - beta) overflowing in R, and
        // |alpha - beta| is within a small factor of the column scale. If THIS
        // reciprocal is finite, the kernel never leaves the multiply path and the
        // assertions below are guarding nothing.
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

            // (1) FINITENESS. This is the property the division arm exists to
            //     preserve; a plain reciprocal-multiply produces inf here.
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

            // (2) ORTHOGONALITY, at a tolerance set by the INPUT's precision, not
            //     the kernel's. A subnormal at this magnitude carries roughly
            //     log2(scale / denorm_min) bits, which is ~5 for float at 1e-41
            //     and ~35 for double at 1e-311. 1e-2 is far looser than either
            //     needs and still orders of magnitude tighter than the inf, nan
            //     or all-zero v that a missing division arm produces.
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

            // A RELATIVE elementwise bound scaled by the vendor factor's own
            // largest entry: the two implementations do not share a reduction
            // order, so bit-exactness is not the claim -- agreement to the
            // rounding of the SAME algorithm is.
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

// ===========================================================================
// G8. m < n, m == n, m > n -- per what supports() CLAIMS.
//
// route_geqrf.hh gate 2 refuses m < n as a CORRECTNESS gate, not a speed one: a
// wide view walks the trailing update past the bottom of the panel. The table
// saying so and the kernel enforcing it are two different facts, and the direct
// entry points are reachable WITHOUT the table -- every numerical test in this
// file reaches them that way -- so both are checked here.
// ===========================================================================
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

// The remaining direct-entry-point gates. The heterogeneous one is the one that
// matters: deleting it does not produce an error, it produces a SILENT WRONG
// ANSWER, because one launch covers the batch with a single (m, n, ld, stride)
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

    // (b) a tau span shorter than k * batch. geqrf's contract packs tau per matrix
    // with stride k OF THE WHOLE MATRIX; a short span is an out-of-bounds write,
    // not a smaller problem.
    Span<T> shortTau(p.tau.data(), p.tau.size() - 1);
    EXPECT_THROW(sycl_geqrf::geqrf_cta_dispatch<T>(*this->ctx, V, shortTau, wb.to_span()),
                 std::invalid_argument);
    EXPECT_THROW(sycl_geqrf::geqrf_blocked_dispatch<T>(*this->ctx, V, shortTau, wb.to_span()),
                 std::invalid_argument);
}

// ===========================================================================
// G9. THE ROUTE TABLE AND THE VENDOR-FREE FALLBACK.
//
// The pure table has its own unit tests (tests/route_vocabulary_tests.cc), which
// build their own shapes. This one goes through the SHAPE BUILDER against a real
// device, which is the half those cannot see: a builder that forgot to fill
// cta_max_elems, or read the wrong local-memory property, produces a table answer
// that is perfectly self-consistent and wrong about this machine.
// ===========================================================================
TYPED_TEST(GeqrfTest, RouteTableAndTheVendorFreeFallback) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::BackendType;

    auto p = make_problem<T>(96, 96, 2, 15u);
    auto V = view_of(p);

    // With NO vendor, a supported native route must be handed over. This is
    // route_resolve.hh:60-63, and it is the entire point of the campaign.
    const auto free_route =
        backend::geqrf_route<B, T>(*this->ctx, V, /*vendor_available=*/false);
    ASSERT_TRUE(dispatch::is_native(free_route))
        << "a vendor-free build has no geqrf route for 96x96; the fallback is broken";

    // preferred() is still false everywhere, so WITH a vendor Origin::Auto must
    // take the vendor. If that ever flips it is a deliberate act, and this line is
    // where it gets noticed.
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

// ===========================================================================
// G9b. THE NATIVE-VS-NATIVE TIE-BREAK, AND THAT IT IS NOT A supports() GATE.
//
// WHAT BROKE BEFORE THIS EXISTED. kGeqrfOrder lists {Native, CTA} first, and the
// vendor-free walk used to return the FIRST supported native route, full stop.
// supports() admits CTA anywhere the tile fits SLM -- square n <= 155 for float,
// n <= 110 for double on this box -- while the tier sweep
// (docs/perf/qr.md#cta-vs-blocked-crossover) measures the blocked driver AHEAD
// of CTA from n ~= 104 (float) and n ~= 48 (double). So a vendor-free build,
// with nothing pinned, took a route 1.37x-1.43x slower than the other native
// tier in the same build. RouteTable::native_tier_preferred is the fix and this
// test is what holds it in place.
//
// THE SECOND HALF IS THE IMPORTANT HALF. Both arms must remain fully
// supports()-true on both sides of the crossover. Putting a speed threshold in
// supports() is this repository's recorded four-times-shipped defect: a forced
// route bypasses preferred() but NEVER supports() (route_resolve.hh:101), so a
// pinned `cta` at n=128 would fall through to automatic() and the test that
// pinned it would silently measure the blocked driver -- or, in a vendor build,
// cuSOLVER. Deleting the supports() assertions below and moving the window into
// supports() is exactly the change this test exists to turn red.
//
// AND THE VENDOR-PRESENT ANSWER MUST NOT MOVE. native_tier_preferred is
// consulted ONLY on the vendor-free walk. If a future edit reaches for
// preferred() to express the same window, the vendor-present resolution flips
// too -- including at shapes where cuSOLVER beats both natives -- and the last
// block here fails.
// ===========================================================================
TYPED_TEST(GeqrfTest, NativeTierTieBreakPicksTheFasterNativeVendorFree) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;
    static constexpr Backend B = TestFixture::BackendType;

    // The crossover this type's table declares, from route_geqrf.hh. Only float
    // and double have a measured one; both complex types stay on CTA to the top
    // of their capacity, so for them there is no boundary to straddle and the
    // test degrades to the supports()-neutrality half.
    const bool has_crossover =
        std::is_same_v<T, float> || (std::is_same_v<T, double> && !test_utils::is_complex<T>::value);
    const int nc = std::is_same_v<R, float> ? 96 : 48;   // last n that prefers CTA
    const int above = std::is_same_v<R, float> ? 128 : 64;  // first n that prefers Blocked

    if (!has_crossover) {
        GTEST_SKIP() << "no measured native crossover for this type (route_geqrf.hh: both "
                        "complex types stay on CTA to the top of their SLM capacity)";
    }

    // Both shapes must be CTA-ELIGIBLE, or the test is vacuous in the direction
    // that matters: if the CTA arm cannot serve `above` at all, "blocked was
    // chosen" proves nothing about the tie-break.
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

    // (1) THE TIE-BREAK ITSELF, through the real shape builder.
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

    // (2) NEITHER ARM MAY HAVE LOST supports() ANYWHERE. This is the half that
    //     keeps a pin honest.
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

    // (3) THE VENDOR-PRESENT ANSWER MUST NOT HAVE MOVED. native_tier_preferred is
    //     consulted only on the vendor-free walk; preferred() is consulted always.
    //     Expressing this window in preferred() instead would flip both builds.
    if (!std::getenv("BATCHLAS_GEQRF_ROUTE")) {
        EXPECT_TRUE(dispatch::is_vendor(
            backend::geqrf_route<B, T>(*this->ctx, V_lo, /*vendor_available=*/true)))
            << "the native tier tie-break leaked into the vendor-present decision at n=" << nc;
        EXPECT_TRUE(dispatch::is_vendor(
            backend::geqrf_route<B, T>(*this->ctx, V_hi, /*vendor_available=*/true)))
            << "the native tier tie-break leaked into the vendor-present decision at n=" << above;
    }
}

// ===========================================================================
// G10. THE FACADE REACHES THE KERNEL.
//
// THE GUARD IS BIT-EXACTNESS AGAINST THE DIRECT ENTRY POINT, NOT A RESIDUAL, and
// that distinction is this repository's FIFTH recorded blind guard
// (tests/potrf_tests.cc:895-908): a route-assertion-plus-residual test "stayed
// GREEN across all four scalar types while every number in it came from
// cuSOLVER", because a residual bound is satisfied by either implementation. The
// vendor does not reproduce this kernel's reduction order, so an element-by-
// element comparison discriminates and a residual does not.
//
// The blocked arm INJECTS THE SAME ROUTED gemm THE FACADE INJECTS. Handing the
// direct call an empty function would make it use sycl_gemm::gemm_custom while
// the facade used the routed gemm; in a vendor build those differ in the low bits
// and the test would fail for a reason that is not a defect.
// ===========================================================================
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

// The facade's ORGQR arm, same discipline. The native driver is ormqr on an
// identity, so a facade that fell through to cuSOLVER would still produce an
// orthonormal Q -- only a bit-exact comparison says which code ran.
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

    // Rebuild the factor and run the direct entry point on it.
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

// ===========================================================================
// G11. THE WORKSPACE CONTRACTS.
//
// Two of them, and both are geqrf-specific rather than inherited from potrf.
//
//   (a) NEITHER QUERY MAY DEREFERENCE A.data_ptr() OR tau.data().
//       band_reduction.cc:1041-1044 sizes sytrd's band reduction against a
//       MatrixView built on nullptr. Any read there is an immediate segfault in
//       sytrd's sizing path -- and a segfault in a test binary is a red test, so
//       this one is self-enforcing.
//
//   (b) THE SIZES MUST BE MONOTONE NON-DECREASING IN (rows, cols, batch), because
//       the query and the call are made against DIFFERENT SHAPES: the same
//       band_reduction sizes at (m_max x nb_max) and calls at an m x r sub-view.
//       max() over ROUTES at one shape says nothing about that, so a non-monotone
//       native query silently under-allocates sytrd.
// ===========================================================================
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

// The facade's query must cover EVERY supported native tier, not the tier this
// resolution happened to choose. A chosen-only size turns a query/call
// disagreement into an UNDER-allocation, which is the recorded ormqr failure
// (buffer size 2560 bytes, call demanded 276480).
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

// ===========================================================================
// THE BREAK RECORD
// ===========================================================================
//
// This repository has shipped FIVE tests that could not fail by construction,
// the most recent written in the same change as the fix it guarded. So every
// test above was run against a deliberately damaged library: the damage was
// applied to the SOURCE, the whole .so was REBUILT (the device link, not just
// this TU), and both this suite and orgqr_tests were re-run. Nine breaks, all
// measured, none reasoned about. Restored and re-run green afterwards.
//
// FIRST, THE BOUND ITSELF. B0: residual_tol/orth_tol tightened from 0.5 to 0.05
// (m+k) eps and the drop-in bound from 64 to 8 eps. 16 rows RED, including
// CtaResidualAndOrthogonality for every type. So the shipped bound CAN fail: the
// worst measured residual over the whole file is 0.096 (m+k) eps and the worst
// elementwise drop-in difference is 28.7 eps, i.e. the shipped constants carry
// 5.2x and 2.2x of margin -- not the 40-200x that potrf_tests.cc:280-300 records
// as being wide enough to hide an accuracy defect.
//
//  KERNEL BREAK                                          WHAT TURNED RED
//  ---------------------------------------------------------------------------
//  KA  the 48 KB hole pad removed from the resident       ResidentLeafLaunchHole,
//      leaf's local_accessor (geqrf_cta.cc)               all 4 types, and ONLY
//                                                        that test. Cold-filtered
//                                                        per type: 4/4 FAIL.
//  KB  the blocked driver's sub-view built with `nr`      33 rows / 9 tests, all
//      instead of the parent `ld` (geqrf_blocked.cc)      4 types
//  KC  panel loop `j0 < k` -> `j0 + nb <= k`, i.e. the    62 rows / 8 tests, all
//      SHORT FINAL PANEL dropped                          4 types, incl.
//                                                        ShortFinalPanel* and
//                                                        ComplexRDiagonal
//  KD  the panel leaf handed `nb` instead of `ib` --      58 rows / 8 tests, all
//      the block width where the PANEL width belongs      4 types
//  KE  the apply's conj(tau) -> tau                       48 rows / 12 tests,
//                                                        COMPLEX ONLY. float and
//                                                        double stayed green,
//                                                        which is the correct
//                                                        null: conj is the
//                                                        identity there.
//  KF  the LAST WY trailing update skipped                70 rows / 9 tests, all
//      (`n2 <= 0` -> `n2 <= 0 || n2 <= nb`)               4 types
//  KG  tau batch stride `k` -> `ib` in the leaf call      68 rows / 8 tests, all
//                                                        4 types. Item 0 is
//                                                        unaffected by this
//                                                        defect, which is why
//                                                        check_one walks EVERY
//                                                        batch item.
//  KH  the CTA leaf reading `b * (ld*n)` instead of       100 rows / 12 tests,
//      `b * A.stride()`                                   all 4 types
//
// THE REPAIR PASS ADDED FOUR MORE, AGAINST THE THREE TESTS IT ADDED:
//
//  BR1 LAPACK's beta sign choice dropped in              ConventionMatchesReference-
//      geqrf_larfg_scalars: `(alphr >= 0) ? -nrm : nrm`   Lapack..., ALL 4 TYPES,
//      -> `? nrm : -nrm`. This IS break K3.               in BOTH builds.
//      ---------------------------------------------------------------------
//      THE POINT OF THAT TEST. Before it, the only check that could see this
//      was NativeFactorMatchesTheVendorElementwise, which GTEST_SKIPs in a
//      vendor-free build -- so in build-novendor the real-scalar half of
//      geqrf's drop-in contract had NO guard at all. K3's recorded outcome was
//      "qr, orth, qrQ ALL GREEN for every type".
//      TWO SECONDARY RESULTS, both worth keeping:
//        * this time a FEW residual tests did go red (BlockedResidual float and
//          double, ShortFinalPanel double) because dropping the sign choice
//          causes cancellation in alpha - beta on some data. But the CTA tier
//          stayed green and BOTH COMPLEX TYPES stayed green. A residual test
//          catches this break SOMETIMES. Do not rely on it.
//        * zgeqr2 applies conj(TAU), not TAU -- reducing from the left applies
//          H^H. The first version of the host reference used tau and disagreed
//          with the kernel by 1-4% for cfloat/cdouble while being EXACT for
//          float/double: the same signature as KE above. For a real T the
//          conjugate is the identity, so this entire defect class is invisible
//          to half the type list by construction.
//
//  BR3 the division arm of the reciprocal guard          SubnormalScaleColumns...,
//      deleted: `vfactor = use_mul ? r : d` -> always     all 4 types, and NOTHING
//      `r`, `use_mul = true`                             ELSE.
//
//  BR4 RouteTable<Op::geqrf>::native_tier_preferred      NativeTierTieBreak...,
//      removed entirely (renamed so the `requires`        float and double -- the
//      detection misses it and the default `true`         two types with a
//      applies)                                          measured crossover --
//                                                        and nothing else.
//
//  BR4b the SAME window moved INTO supports(), which     NativeTierTieBreak...,
//      is this repository's four-times-shipped defect     and on the INTENDED
//                                                        assertion:
//                                                        supports(cta, sh_hi)
//                                                        was false. That is what
//                                                        proves the test's
//                                                        second half is not
//                                                        vacuous -- part 1 still
//                                                        passed under it.
//
// TWO BREAKS THAT TURNED NOTHING RED, AND THEY ARE THE USEFUL PART:
//
//  N1  `ib` -> `nb` in the larft/pack-V calls of the WY update. VACUOUS BY
//      CONSTRUCTION, and it is worth knowing why rather than adding a test for
//      it: supports() requires m >= n, so k == n, so a short final panel has
//      j0 + ib == k and therefore n2 == 0 and the driver breaks out BEFORE the
//      WY update. larft is never handed a short panel at all. The short-final-
//      panel error class exists in this driver only at the LEAF (KD), not in the
//      trailing update -- which is the opposite of where sy2sb's stage-1 bug was,
//      and is why this file's straddle test drives the leaf.
//
//  N2  EVERY kernel break above left tests/orgqr_tests.cc GREEN in the
//      VENDOR build. That suite pins no route, so its facade geqrf/orgqr resolve
//      to cuSOLVER and no native kernel runs: as a guard on WP5's kernels it is a
//      NULL in a vendor-present build and only discriminates in build-novendor.
//      The tests in THIS file call the direct entry points precisely so they do
//      not have that property.
//
// AND ONE ORDERING HAZARD THAT IS DELIBERATELY NOT GUARDED BY AN ASSERTION:
// ResidentLeafLaunchHoleAt48KiB is only discriminating while it is the first
// resident-leaf launch of its scalar type in the process (see its own comment).
// Nothing in GoogleTest can assert that; the guard is declaration order plus the
// comment, and the cold-filter command is written down there.
// ===========================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
