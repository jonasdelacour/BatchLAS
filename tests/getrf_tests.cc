// Native batched LU -- getrf, getrs and getri: both getrf tiers (CTA and
// blocked), both getrs arms (composed and fused narrow-RHS), and getri.
// Every numerical test drives the native dispatch entry points DIRECTLY against a
// HOST reference; the vendor is never a pivot oracle, because
// cublas{C,Z}getrfBatched pivots on the modulus where this library and LAPACK
// pivot on cabs1 = |Re| + |Im|.
// evidence: docs/perf/lu.md
#include <gtest/gtest.h>

#include <batchlas/blas/functions/getrf.hh>
#include <batchlas/blas/functions/getrs.hh>
#include <batchlas/blas/functions/getri.hh>
#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/functions/trsm.hh>
#include <batchlas/blas/dispatch/vendor_available.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>

#include "test_utils.hh"

#include "../src/extensions/getrf_native.hh"
#include "../src/extensions/getrs_native.hh"
#include "../src/extensions/getri_native.hh"
#include "../src/backends/getrf_route.hh"
#include "../src/backends/getrs_route.hh"
#include "../src/backends/getri_route.hh"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

using namespace batchlas;

namespace {

template <typename T>
using RealOf = typename batchlas::base_type<T>::type;

// Host arithmetic: every reference promotes to double (or complex<double>)
// before it accumulates, so a float residual measures the KERNEL's error.
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
// cabs1 = |Re| + |Im|: the metric ?GETRF's I?AMAX pivots on, not the modulus.
inline double hcabs1(double x) { return std::fabs(x); }
inline double hcabs1(std::complex<double> x) { return std::fabs(x.real()) + std::fabs(x.imag()); }
inline bool hfinite(double x) { return std::isfinite(x); }
inline bool hfinite(std::complex<double> x) { return std::isfinite(x.real()) && std::isfinite(x.imag()); }

template <class T> inline T mk(double re, double im);
template <> inline float mk<float>(double re, double) { return float(re); }
template <> inline double mk<double>(double re, double) { return re; }
template <> inline std::complex<float> mk<std::complex<float>>(double re, double im) {
    return {float(re), float(im)};
}
template <> inline std::complex<double> mk<std::complex<double>>(double re, double im) {
    return {re, im};
}

// Scale by a real factor without naming .real()/.imag() on a type without them.
template <class T> inline T scale(T v, double f) { return T(v * static_cast<RealOf<T>>(f)); }
template <class R> inline std::complex<R> scale(std::complex<R> v, double f) {
    return std::complex<R>(v.real() * static_cast<R>(f), v.imag() * static_cast<R>(f));
}

template <typename T>
constexpr double eps_of() {
    if constexpr (std::is_same_v<RealOf<T>, float>) return 1.1920929e-7;
    else return 2.220446049250313e-16;
}

// LU with partial pivoting is backward stable, so these bounds scale with n * eps
// and not with conditioning; only the inverse residual carries cond(A).
template <typename T> double lu_tol(int n)    { return 200.0 * double(n) * eps_of<T>(); }
template <typename T> double solve_tol(int n) { return 400.0 * double(n) * eps_of<T>(); }
template <typename T> double inv_tol(int n)   { return 800.0 * double(n) * eps_of<T>(); }

bool verbose() {
    static const bool v = (std::getenv("BATCHLAS_TEST_VERBOSE") != nullptr);
    return v;
}

// A deterministic LCG rather than <random>: several tests below assert that two
// batch items DIFFER, and that must be a fact about the data, not about luck.
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 6364136223846793005ULL + 1442695040888963407ULL) {}
    double next() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return double(int32_t(uint32_t(s >> 32))) / 2147483648.0;
    }
};

template <typename T, Backend B>
struct LuConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

// A batch of DISTINCT matrices with a PADDED ld and a stride that is NOT ld*cols,
// with the pad POISONED, so a launcher that lets MatrixView default the stride to
// ld*cols is falsifiable by default. The pointer array is not optional either:
// every vendor batched call dereferences data_ptrs_ and throws when it is empty.
template <typename T>
struct Lu {
    int n = 0, batch = 0, ld = 0, stride = 0;
    UnifiedVector<T> buf;        // working copy, overwritten by getrf
    std::vector<T> a0;           // the pristine input, same ld/stride
    UnifiedVector<T*> ptrs;
    UnifiedVector<int64_t> piv;  // the PUBLIC int64 span; CUDA/ROCm pack int32 into it
    UnifiedVector<int32_t> info;
    // The interchange list this matrix MUST produce, when it is known exactly
    // (the dominant-permuted construction). Empty when it is not.
    std::vector<int> expect_piv;
};

template <typename T>
void poison(Lu<T>& p) {
    std::copy(p.a0.begin(), p.a0.end(), p.buf.begin());
    std::fill(p.piv.begin(), p.piv.end(), int64_t(0x0BADBEEF0BADBEEFLL));
    std::fill(p.info.begin(), p.info.end(), int32_t(-12345));
}

template <typename T>
void alloc(Lu<T>& p, int n, int batch, int ld_pad, int stride_pad) {
    p.n = n; p.batch = batch;
    p.ld = n + ld_pad;
    p.stride = p.ld * n + stride_pad;
    p.buf = UnifiedVector<T>(static_cast<size_t>(p.stride) * batch, mk<T>(-9.75e3, 4.5e3));
    p.ptrs = UnifiedVector<T*>(static_cast<size_t>(batch), nullptr);
    p.piv = UnifiedVector<int64_t>(static_cast<size_t>(n) * batch, int64_t(0));
    p.info = UnifiedVector<int32_t>(static_cast<size_t>(batch), int32_t(0));
}

// A RANDOM matrix: the pivot sequence is data-dependent, so only the residual and
// pivot-ratio oracles apply.
template <typename T>
Lu<T> make_random(int n, int batch, unsigned seed, int ld_pad = 5, int stride_pad = 11) {
    Lu<T> p;
    alloc(p, n, batch, ld_pad, stride_pad);
    Rng rg(seed);
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                p.buf[size_t(b) * p.stride + size_t(j) * p.ld + i] = mk<T>(rg.next(), rg.next());
    p.a0.assign(p.buf.begin(), p.buf.end());
    poison(p);
    return p;
}

// THE MATRIX WITH A KNOWN-EXACT PIVOT SEQUENCE. A strictly column-diagonally-
// dominant B (|B(k,k)| = 4n, |B(i,k)| <= 1) whose rows are then permuted: column
// dominance survives elimination, so the winner at step k is known exactly and the
// expected interchange list is pure integer bookkeeping. cond(A) is O(1), which is
// why every getrs and getri residual runs on it too.
template <typename T>
Lu<T> make_dominant_permuted(int n, int batch, unsigned seed,
                             int ld_pad = 5, int stride_pad = 11) {
    Lu<T> p;
    alloc(p, n, batch, ld_pad, stride_pad);
    Rng rg(seed);

    // sigma: B's row r ends up at position sigma[r]. A CYCLIC SHIFT and not a
    // reversal: a reversal is its own inverse, so every test of a permutation
    // DIRECTION would be satisfied by the wrong direction too.
    std::vector<int> sigma(n);
    for (int r = 0; r < n; ++r) sigma[r] = (r + 1) % n;

    for (int b = 0; b < batch; ++b) {
        for (int r = 0; r < n; ++r) {
            const int dst = sigma[r];
            for (int j = 0; j < n; ++j) {
                const double re = rg.next();
                const double im = rg.next();
                T v = mk<T>(re, im);
                if (j == r) v = mk<T>(4.0 * double(n) * (re >= 0 ? 1.0 : -1.0), 0.0);
                p.buf[size_t(b) * p.stride + size_t(j) * p.ld + dst] = v;
            }
        }
    }
    p.a0.assign(p.buf.begin(), p.buf.end());

    // The expected interchange list, by simulating the SELECTION only.
    // home[i] = which B-row currently sits at position i.
    std::vector<int> home(n);
    for (int r = 0; r < n; ++r) home[sigma[r]] = r;
    p.expect_piv.resize(n);
    for (int k = 0; k < n; ++k) {
        int q = -1;
        for (int i = k; i < n; ++i) if (home[i] == k) { q = i; break; }
        p.expect_piv[k] = q + 1;              // 1-BASED, LAPACK ipiv
        std::swap(home[k], home[q]);
    }
    poison(p);
    return p;
}

template <typename T>
MatrixView<T, MatrixFormat::Dense> view_of(Lu<T>& p) {
    return MatrixView<T, MatrixFormat::Dense>(p.buf.data(), p.n, p.n, p.ld, p.stride, p.batch,
                                              p.ptrs.data());
}

// A right-hand-side / output block, same padding and poison discipline.
template <typename T>
struct Rhs {
    int n = 0, nrhs = 0, batch = 0, ld = 0, stride = 0;
    UnifiedVector<T> buf;
    std::vector<T> b0;
    UnifiedVector<T*> ptrs;
};

template <typename T>
Rhs<T> make_rhs(int n, int nrhs, int batch, unsigned seed,
                int ld_pad = 3, int stride_pad = 7) {
    Rhs<T> r;
    r.n = n; r.nrhs = nrhs; r.batch = batch;
    r.ld = n + ld_pad;
    r.stride = r.ld * nrhs + stride_pad;
    r.buf = UnifiedVector<T>(size_t(r.stride) * batch, mk<T>(-9.75e3, 4.5e3));
    r.ptrs = UnifiedVector<T*>(size_t(batch), nullptr);
    Rng rg(seed);
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < nrhs; ++j)
            for (int i = 0; i < n; ++i)
                r.buf[size_t(b) * r.stride + size_t(j) * r.ld + i] = mk<T>(rg.next(), rg.next());
    r.b0.assign(r.buf.begin(), r.buf.end());
    return r;
}

template <typename T>
void reset_rhs(Rhs<T>& r) { std::copy(r.b0.begin(), r.b0.end(), r.buf.begin()); }

template <typename T>
MatrixView<T, MatrixFormat::Dense> view_of(Rhs<T>& r) {
    return MatrixView<T, MatrixFormat::Dense>(r.buf.data(), r.n, r.nrhs, r.ld, r.stride, r.batch,
                                              r.ptrs.data());
}

// The PACKED int32 view of the public int64 pivot span, for ONE batch item. This
// spelling -- not a widening read -- IS the pivot contract on CUDA and ROCm.
template <typename T>
const int* piv_item(const Lu<T>& p, int b) {
    return reinterpret_cast<const int*>(p.piv.data()) + size_t(b) * p.n;
}

// ORACLE 1: ||P A - L U||_F / ||A||_F, rectangular-capable (m >= n). P is rebuilt
// here from a 1-based INTERCHANGE LIST applied FORWARDS, as the contract claims.
template <typename T>
double pa_lu_residual(const T* A0, const T* F, const int* ipiv,
                      int m, int n, int ld) {
    using D = typename Prom<T>::type;
    const int k = std::min(m, n);
    std::vector<D> PA(size_t(m) * n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            PA[size_t(j) * m + i] = up(A0[size_t(j) * ld + i]);

    for (int s = 0; s < k; ++s) {
        const int q = ipiv[s] - 1;            // 1-BASED on the wire
        if (q == s) continue;
        for (int j = 0; j < n; ++j) std::swap(PA[size_t(j) * m + s], PA[size_t(j) * m + q]);
    }

    double num = 0.0, den = 0.0;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            D acc = D(0);
            const int tmax = std::min(std::min(i, j), k - 1);
            for (int t = 0; t <= tmax; ++t) {
                const D l = (i == t) ? D(1) : up(F[size_t(t) * ld + i]);
                const D u = up(F[size_t(j) * ld + t]);
                acc += l * u;
            }
            const D d = PA[size_t(j) * m + i] - acc;
            num += habs(d) * habs(d);
            den += habs(PA[size_t(j) * m + i]) * habs(PA[size_t(j) * m + i]);
        }
    }
    return (den > 0.0) ? std::sqrt(num / den) : std::sqrt(num);
}

// ORACLE 3: the partial-pivoting property, in the metric the library pivots on.
// "max |L(i,j)| <= 1" is WRONG for complex (cabs1(z) <= sqrt(2)|z|); the exact
// statement is cabs1(L(i,k) U(k,k)) <= cabs1(U(k,k)), which a kernel pivoting on
// the modulus violates. Returns the worst ratio; 1 is the bound.
template <typename T>
double worst_pivot_ratio(const T* F, int m, int n, int ld) {
    const int k = std::min(m, n);
    double worst = 0.0;
    for (int j = 0; j < k; ++j) {
        const auto ukk = up(F[size_t(j) * ld + j]);
        const double den = hcabs1(ukk);
        if (den == 0.0) continue;                 // a singular column says nothing here
        for (int i = j + 1; i < m; ++i)
            worst = std::max(worst, hcabs1(up(F[size_t(j) * ld + i]) * ukk) / den);
    }
    return worst;
}

// ORACLE 4a: ||op(A) X - B||_F / (||A||_F ||X||_F).
template <typename T>
double solve_residual(const T* A0, const T* X, const T* B0,
                      int n, int nrhs, int lda, int ldb, Transpose op) {
    using D = typename Prom<T>::type;
    double num = 0.0, na = 0.0, nx = 0.0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            const double a = habs(up(A0[size_t(j) * lda + i]));
            na += a * a;
        }
    for (int j = 0; j < nrhs; ++j)
        for (int i = 0; i < n; ++i) {
            const double x = habs(up(X[size_t(j) * ldb + i]));
            nx += x * x;
        }
    for (int j = 0; j < nrhs; ++j) {
        for (int i = 0; i < n; ++i) {
            D acc = D(0);
            for (int t = 0; t < n; ++t) {
                D a = up(A0[size_t(t) * lda + i]);                  // A(i,t)
                if (op == Transpose::Trans)     a = up(A0[size_t(i) * lda + t]);
                if (op == Transpose::ConjTrans) a = hconj(up(A0[size_t(i) * lda + t]));
                acc += a * up(X[size_t(j) * ldb + t]);
            }
            const D d = acc - up(B0[size_t(j) * ldb + i]);
            num += habs(d) * habs(d);
        }
    }
    const double scale = std::sqrt(na) * std::sqrt(nx);
    return (scale > 0.0) ? std::sqrt(num) / scale : std::sqrt(num);
}

// ORACLE 4b: ||A C - I||_F / n.
template <typename T>
double inverse_residual(const T* A0, const T* C, int n, int lda, int ldc) {
    using D = typename Prom<T>::type;
    double num = 0.0;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            D acc = D(0);
            for (int t = 0; t < n; ++t)
                acc += up(A0[size_t(t) * lda + i]) * up(C[size_t(j) * ldc + t]);
            if (i == j) acc -= D(1);
            num += habs(acc) * habs(acc);
        }
    }
    return std::sqrt(num) / double(n);
}

// The fixture.
template <typename Config>
class LuTest : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    static constexpr Backend BackendType = Config::BackendVal;

    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        if (this->HasFatalFailure() || ::testing::Test::IsSkipped()) return;
        if (!this->ctx) GTEST_SKIP() << "no queue";
        if (this->ctx->device().type != DeviceType::GPU)
            GTEST_SKIP() << "the native LU kernels are GPU-only (route_getrf.hh gate 2)";
        if (!this->ctx->device().supports_sub_group_size(32))
            GTEST_SKIP() << "device does not offer sub-group size 32 (route_getrf.hh gate 3)";
    }

    // The DEVICE's local-memory budget, spelled exactly as
    // src/backends/getrf_route.hh spells it, NOT device_limits.hh's hardcoded 49152.
    std::size_t budget() const {
        const std::size_t lm = static_cast<std::size_t>(
            this->ctx->device().get_property(DeviceProperty::LOCAL_MEM_SIZE));
        return lm > 4096 ? lm - 4096 : std::size_t(0);
    }
    int cta_max_n() const { return sycl_getrf::getrf_cta_max_n_for_slm<T>(budget()); }
    bool leaf_fits(int m, int n) const { return sycl_getrf::getrf_leaf_fits<T>(m, n, budget()); }

    // The blocked driver's OWN block width and leading-panel leaf choice, QUERIED and
    // never hardcoded: a straddle test that cannot see the boundary tests nothing.
    int nb(int n) const {
        return int(sycl_getrf::getrf_blocked_debug_params<T>(*this->ctx, n) & 0xffffu);
    }
    unsigned leaf(int n) const {
        return (sycl_getrf::getrf_blocked_debug_params<T>(*this->ctx, n) >> 16) & 0xffu;
    }
    // Bits 24+: the deferred left-hand interchange spelling (0 in-loop, 1 deferred
    // walk, 2 deferred gather). MASKED, so a field added above cannot move leaf().
    unsigned left_mode(int n) const {
        return (sycl_getrf::getrf_blocked_debug_params<T>(*this->ctx, n) >> 24) & 0xffu;
    }

    // The ROUTED gemm and trsm, exactly as the factorization entry points inject them.
    // A direct caller MUST inject them: the blocked driver throws on an empty seam.
    sycl_getrf::GetrfTrailingGemm<T> gemm_seam() const {
        return [](Queue& c, const MatrixView<T, MatrixFormat::Dense>& ga,
                  const MatrixView<T, MatrixFormat::Dense>& gb,
                  const MatrixView<T, MatrixFormat::Dense>& gc,
                  T al, T be, Transpose ta, Transpose tb, ComputePrecision pr) {
            return gemm<BackendType, T>(c, ga, gb, gc, al, be, ta, tb, pr);
        };
    }
    sycl_getrf::GetrfPanelSolveTrsm<T> trsm_seam() const {
        return [](Queue& c, const MatrixView<T, MatrixFormat::Dense>& ta,
                  const MatrixView<T, MatrixFormat::Dense>& tb,
                  T al, Side sd, Uplo ul, Transpose tr, Diag dg) {
            return trsm<BackendType, T>(c, ta, tb, al, sd, ul, tr, dg);
        };
    }
    sycl_getrs::GetrsSolveTrsm<T> getrs_seam() const {
        return [](Queue& c, const MatrixView<T, MatrixFormat::Dense>& ta,
                  const MatrixView<T, MatrixFormat::Dense>& tb,
                  T al, Side sd, Uplo ul, Transpose tr, Diag dg) {
            return trsm<BackendType, T>(c, ta, tb, al, sd, ul, tr, dg);
        };
    }
    sycl_getri::GetriSolveTrsm<T> getri_seam() const {
        return [](Queue& c, const MatrixView<T, MatrixFormat::Dense>& ta,
                  const MatrixView<T, MatrixFormat::Dense>& tb,
                  T al, Side sd, Uplo ul, Transpose tr, Diag dg) {
            return trsm<BackendType, T>(c, ta, tb, al, sd, ul, tr, dg);
        };
    }

    // Run one tier DIRECTLY. `pass_info` false exercises the info_target
    // fallback, of which src/extensions/inv.cc:48 is a real instance.
    void run_cta(Lu<T>& p, bool pass_info = true) {
        auto V = view_of(p);
        UnifiedVector<std::byte> ws(std::max<std::size_t>(
            1, sycl_getrf::getrf_cta_buffer_size<T>(*this->ctx, V)));
        sycl_getrf::getrf_cta_dispatch<T>(*this->ctx, V, p.piv.to_span(), ws.to_span(),
                                          pass_info ? p.info.to_span() : Span<int32_t>{});
        this->ctx->wait();
    }
    void run_blocked(Lu<T>& p, bool pass_info = true) {
        auto V = view_of(p);
        UnifiedVector<std::byte> ws(std::max<std::size_t>(
            1, sycl_getrf::getrf_blocked_buffer_size<T>(*this->ctx, V)));
        sycl_getrf::getrf_blocked_dispatch<T>(*this->ctx, V, p.piv.to_span(), ws.to_span(),
                                              pass_info ? p.info.to_span() : Span<int32_t>{},
                                              gemm_seam(), trsm_seam());
        this->ctx->wait();
    }
};

// EVERY BATCH ITEM IS CHECKED, NOT ITEM 0: item 0 sits at offset 0, so a wrong
// batch stride cannot move it. The distinctness assertion is what makes "the
// kernel broadcast item 0 over the batch" a failure rather than a pass.
template <typename T>
void check_factor(const Lu<T>& p, const char* what, bool check_L = true) {
    for (int b = 0; b < p.batch; ++b) {
        const T* F = p.buf.data() + size_t(b) * p.stride;
        const T* A0 = p.a0.data() + size_t(b) * p.stride;
        const int* ip = piv_item(p, b);

        for (int k = 0; k < p.n; ++k)
            ASSERT_TRUE(ip[k] >= k + 1 && ip[k] <= p.n)
                << what << ": ipiv[" << k << "] = " << ip[k]
                << " is outside [k+1, n] at b=" << b << " -- not a 1-based interchange list";

        for (int j = 0; j < p.n; ++j)
            for (int i = 0; i < p.n; ++i)
                ASSERT_TRUE(hfinite(up(F[size_t(j) * p.ld + i])))
                    << what << ": F(" << i << "," << j << ") is not finite at b=" << b;

        const double res = pa_lu_residual<T>(A0, F, ip, p.n, p.n, p.ld);
        if (verbose())
            std::printf("[verbose] %-34s n=%4d b=%d  ||PA-LU||=%.4e  tol=%.4e\n",
                        what, p.n, b, res, lu_tol<T>(p.n));
        EXPECT_LE(res, lu_tol<T>(p.n))
            << what << ": ||PA - LU||_F / ||A||_F too large at b=" << b << " (n=" << p.n << ")";

        if (check_L) {
            const double ratio = worst_pivot_ratio<T>(F, p.n, p.n, p.ld);
            EXPECT_LE(ratio, 1.0 + 32.0 * eps_of<T>())
                << what << ": cabs1(L(i,k) U(k,k)) / cabs1(U(k,k)) reached " << ratio
                << " > 1 at b=" << b
                << " -- a row with a LARGER cabs1 than the chosen pivot was left below it, so "
                   "this is not a cabs1 PARTIAL-pivoting factorization";
        }

        if (!p.expect_piv.empty()) {
            for (int k = 0; k < p.n; ++k)
                ASSERT_EQ(ip[k], p.expect_piv[k])
                    << what << ": ipiv[" << k << "] = " << ip[k] << ", expected "
                    << p.expect_piv[k] << " at b=" << b
                    << " -- the pivot base, direction or metric disagrees with LAPACK";
        }
    }

    if (p.batch > 1) {
        const T* f0 = p.buf.data();
        const T* fl = p.buf.data() + size_t(p.batch - 1) * p.stride;
        bool differ = false;
        for (int j = 0; j < p.n && !differ; ++j)
            for (int i = 0; i < p.n && !differ; ++i)
                if (habs(up(f0[size_t(j) * p.ld + i]) - up(fl[size_t(j) * p.ld + i])) > 0.0)
                    differ = true;
        EXPECT_TRUE(differ) << what << ": the first and last batch items' factors are identical, "
                               "so this shape cannot see a batch-stride defect";
    }
}

// ANTI-VACUITY FOR EVERY TEST OF A PERMUTATION *DIRECTION*: if the permutation the
// interchange list denotes is SELF-INVERSE, a backwards walk and a forwards walk
// produce the same answer and no residual can tell them apart.
inline bool interchange_is_involution(const std::vector<int>& ipiv) {
    const int n = int(ipiv.size());
    std::vector<int> p(n);
    for (int i = 0; i < n; ++i) p[i] = i;
    for (int k = 0; k < n; ++k) std::swap(p[k], p[ipiv[k] - 1]);
    for (int i = 0; i < n; ++i) if (p[p[i]] != i) return false;
    return true;
}

// Anti-vacuity for the pivot oracle: the construction must actually MOVE rows.
// On a plain diagonally dominant matrix partial pivoting picks the diagonal at
// every step and every pivot assertion in this file would be vacuous.
template <typename T>
int non_diagonal_pivots(const Lu<T>& p, int b) {
    const int* ip = piv_item(p, b);
    int c = 0;
    for (int k = 0; k < p.n; ++k) if (ip[k] != k + 1) ++c;
    return c;
}

struct EnvGuard {
    std::string name, saved;
    bool had = false;
    EnvGuard(const char* n, const char* v) : name(n) {
        if (const char* s = std::getenv(n)) { saved = s; had = true; }
        ::setenv(n, v, 1);
    }
    ~EnvGuard() {
        if (had) ::setenv(name.c_str(), saved.c_str(), 1);
        else ::unsetenv(name.c_str());
    }
};

// THE FUSED NARROW-RHS GETRS TIER -- SHARED SCAFFOLDING. getrs_fused.cc is a
// SECOND native getrs arm: one work-group per matrix, the interchange walk and
// BOTH substitutions in ONE kernel. All three ways in are exercised below.
// evidence: docs/perf/lu.md#the-fused-narrow-rhs-getrs

// The tier's local-memory request, PINNED below against the library's own capacity
// query. A request landing in the 48 KB launch hole -- (47104, 49664] BYTES -- is
// raised to 49920, so the pad is NOT monotone and its inversion is not obvious.
constexpr std::size_t kFusedHoleLo    = 47104;
constexpr std::size_t kFusedHoleHi    = 49664;
constexpr std::size_t kFusedHolePadTo = 49920;
constexpr int kFusedNbMax = 32;

inline std::size_t fused_pad(std::size_t bytes) {
    return (bytes > kFusedHoleLo && bytes <= kFusedHoleHi) ? kFusedHolePadTo : bytes;
}

// What the LAUNCHER will actually ask for, given the block width it will pick.
inline int fused_nb_for(int n) {
    const int nb = (n >= 1024) ? 32 : 16;
    return (nb > n) ? n : nb;
}
inline std::size_t fused_launch_bytes(int n, int nrhs, std::size_t sz) {
    const int nb = fused_nb_for(n);
    return fused_pad((std::size_t(n) * std::size_t(nrhs) +
                      std::size_t(nb) * std::size_t(nb + 1)) * sz);
}
// What the CAPACITY QUERY charges for a given element count (the largest nb).
inline std::size_t fused_capacity_bytes(std::size_t rhs_elems, std::size_t sz) {
    return fused_pad((rhs_elems + std::size_t(kFusedNbMax) * std::size_t(kFusedNbMax + 1)) * sz);
}

// The order n whose RAW (pre-pad) launch request is exactly `want` bytes at this
// nrhs, or -1. Solved rather than tabulated because nb itself depends on n.
inline int fused_order_for_raw_bytes(std::size_t want, int nrhs, std::size_t sz) {
    if (want % sz) return -1;
    const std::size_t total = want / sz;
    for (int nb : {16, 32}) {
        const std::size_t blk = std::size_t(nb) * std::size_t(nb + 1);
        if (total <= blk) continue;
        const std::size_t rhs = total - blk;
        if (rhs % std::size_t(nrhs)) continue;
        const std::size_t n = rhs / std::size_t(nrhs);
        if (n < 1 || n > (std::size_t(1) << 20)) continue;
        if (fused_nb_for(int(n)) != nb) continue;   // the launcher must agree
        return int(n);
    }
    return -1;
}

// A FABRICATED LU FACTOR, built on the host with NO getrf: the 48 KB ladder runs at
// orders of 334-1428, where a ||PA - LU|| oracle is O(n^3), and this makes an exact
// O(n^2) residual available instead. The pivot list is a genuine INTERCHANGE LIST
// -- ipiv[k] in [k+1, n], 1-BASED, PACKED int32 into the public int64 span.
template <typename T>
Lu<T> make_fabricated_factor(int n, int batch, unsigned seed,
                             int ld_pad = 5, int stride_pad = 11) {
    Lu<T> p;
    alloc(p, n, batch, ld_pad, stride_pad);
    Rng rg(seed);
    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                T v;
                if (i == j)      v = mk<T>(4.0 * double(n), 0.0);
                else if (i < j)  v = mk<T>(rg.next(), rg.next());                 // U
                else             v = scale(mk<T>(rg.next(), rg.next()), 0.25);    // L
                p.buf[size_t(b) * p.stride + size_t(j) * p.ld + i] = v;
            }
        int* ip = reinterpret_cast<int*>(p.piv.data()) + size_t(b) * n;
        for (int k = 0; k < n; ++k) {
            const int span = n - k;
            const int off = int(std::fabs(rg.next()) * double(span)) % span;
            ip[k] = k + off + 1;                       // 1-BASED, in [k+1, n]
        }
        if (b == 0) p.expect_piv.assign(ip, ip + n);
    }
    p.a0.assign(p.buf.begin(), p.buf.end());
    return p;
}

// THE O(n^2) GETRS RESIDUAL, straight from the contract:
//
//   NoTrans   b = F^{-1}( L ( U x ) )      F^{-1} = the list walked BACKWARDS
//   Trans/CT  b = op(U) ( op(L) ( F x ) )  F      = the list walked FORWARDS
//
// It never forms A, so it cannot see a flip of the CONVENTION itself.
template <typename T>
double fused_factor_residual(const T* F, const int* ipiv, const T* X, const T* B0,
                             int n, int nrhs, int ldf, int ldb, Transpose op) {
    using D = typename Prom<T>::type;
    const bool tr   = (op != Transpose::NoTrans);
    const bool conj = (op == Transpose::ConjTrans);

    double nu = 0.0, nx = 0.0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i <= j; ++i) {
            const double u = habs(up(F[size_t(j) * ldf + i]));
            nu += u * u;
        }
    for (int j = 0; j < nrhs; ++j)
        for (int i = 0; i < n; ++i) {
            const double x = habs(up(X[size_t(j) * ldb + i]));
            nx += x * x;
        }

    auto El = [&](int i, int j) { return up(F[size_t(j) * ldf + i]); };

    double num = 0.0;
    std::vector<D> v(n), w(n);
    for (int c = 0; c < nrhs; ++c) {
        for (int i = 0; i < n; ++i) v[i] = up(X[size_t(c) * ldb + i]);

        if (!tr) {
            for (int i = 0; i < n; ++i) {                    // w = U v
                D acc = D(0);
                for (int t = i; t < n; ++t) acc += El(i, t) * v[t];
                w[i] = acc;
            }
            for (int i = n - 1; i >= 0; --i) {               // v = L w, UNIT lower
                D acc = w[i];
                for (int t = 0; t < i; ++t) acc += El(i, t) * w[t];
                v[i] = acc;
            }
            for (int k = n - 1; k >= 0; --k) {               // F^{-1}, BACKWARDS
                const int q = ipiv[k] - 1;
                if (q != k) std::swap(v[k], v[q]);
            }
        } else {
            for (int k = 0; k < n; ++k) {                    // F, FORWARDS
                const int q = ipiv[k] - 1;
                if (q != k) std::swap(v[k], v[q]);
            }
            for (int i = 0; i < n; ++i) {                    // w = op(L) v, UNIT upper
                D acc = v[i];
                for (int t = i + 1; t < n; ++t) {
                    const D l = El(t, i);
                    acc += (conj ? hconj(l) : l) * v[t];
                }
                w[i] = acc;
            }
            for (int i = n - 1; i >= 0; --i) {               // v = op(U) w, LOWER
                D acc = D(0);
                for (int t = 0; t <= i; ++t) {
                    const D u = El(t, i);
                    acc += (conj ? hconj(u) : u) * w[t];
                }
                v[i] = acc;
            }
        }
        for (int i = 0; i < n; ++i) {
            const D d = v[i] - up(B0[size_t(c) * ldb + i]);
            num += habs(d) * habs(d);
        }
    }
    const double sc = std::sqrt(nu) * std::sqrt(nx);
    return (sc > 0.0) ? std::sqrt(num) / sc : std::sqrt(num);
}

// The RHS pad and the inter-item gap must come back BIT-IDENTICAL: the fused
// kernel writes B[i + c*ldb] for i < n and c < nrhs and nothing else.
template <typename T>
void check_rhs_pad_intact(const Rhs<T>& r, const char* what) {
    for (int b = 0; b < r.batch; ++b)
        for (int j = 0; j < r.stride; ++j) {
            const int col = j / r.ld, row = j % r.ld;
            const bool live = (col < r.nrhs) && (row < r.n);
            if (live) continue;
            const size_t k = size_t(b) * r.stride + j;
            ASSERT_EQ(habs(up(r.buf[k]) - up(r.b0[k])), 0.0)
                << what << ": the RHS PAD was written at b=" << b << " offset " << j
                << " (ld=" << r.ld << ", n=" << r.n << ", nrhs=" << r.nrhs
                << ", stride=" << r.stride << ")";
        }
}

// Every batch item, and the LAST one especially: item 0 sits at offset 0, so a
// wrong batch stride cannot move it.
template <typename T>
void check_items_differ(const Rhs<T>& r, const char* what) {
    if (r.batch < 2) return;
    bool differ = false;
    const T* x0 = r.buf.data();
    const T* xl = r.buf.data() + size_t(r.batch - 1) * r.stride;
    for (int c = 0; c < r.nrhs && !differ; ++c)
        for (int i = 0; i < r.n && !differ; ++i)
            if (habs(up(x0[size_t(c) * r.ld + i]) - up(xl[size_t(c) * r.ld + i])) > 0.0)
                differ = true;
    EXPECT_TRUE(differ) << what << ": the first and last batch items' solutions are identical, "
                           "so this shape cannot see a batch-stride defect";
}

using LuTestTypes = typename test_utils::backend_types<LuConfig>::type;

}  // namespace

TYPED_TEST_SUITE(LuTest, LuTestTypes);

// L0. THE 48 KB LAUNCH HOLE: a resident-leaf launch asking for a local-memory
// size in (47104, 49664] BYTES is refused, so the launcher pads it to 49,920 B.
//
// DECLARED FIRST ON PURPOSE: the raised cap is STICKY PER CUfunction and one
// GetrfPanelResidentKernel<T> serves every panel shape of a type, so any earlier
// launch of a LARGER panel warms the cap and this test can never fail again. DO
// NOT MOVE IT, and add no resident-leaf launch above it.
// evidence: docs/perf/lu.md#the-48-kb-launch-hole
TYPED_TEST(LuTest, ResidentLeafLaunchHoleAt48KiB) {
    using T = typename TestFixture::T;

    // getrf_cta.cc's getrf_scratch_bytes: 32 argmax slots, each a real plus an
    // int. Restated here and then PINNED against the library's own predicate.
    const std::size_t sz = sizeof(T);
    const std::size_t scratch = 32u * (sizeof(RealOf<T>) + sizeof(int));
    auto raw_bytes = [&](int m, int n) {
        return std::size_t(m | 1) * std::size_t(n) * sz + scratch;
    };
    // The smallest budget at which the LIBRARY says an m x n leaf fits.
    auto min_budget = [&](int m, int n) -> std::size_t {
        std::size_t lo = 0, hi = std::size_t(1) << 24;
        if (!sycl_getrf::getrf_leaf_fits<T>(m, n, hi)) return 0;
        while (lo + 1 < hi) {
            const std::size_t mid = lo + (hi - lo) / 2;
            if (sycl_getrf::getrf_leaf_fits<T>(m, n, mid)) hi = mid; else lo = mid;
        }
        return hi;
    };

    // ANTI-VACUITY 1: the byte formula must be the library's, checked below the band.
    ASSERT_EQ(min_budget(16, 8), raw_bytes(16, 8))
        << "this test's local-memory formula is not the library's; every byte "
           "count below names some other size and the ladder proves nothing";

    // The hole band's endpoints, from getrf_cta.cc:136-138.
    constexpr std::size_t kLo = 47104, kHi = 49664, kPadTo = 49920;

    struct Row { std::size_t bytes; int m, n; };
    std::vector<Row> rows;
    for (std::size_t target : {std::size_t(46080),   // below the band: no pad
                               std::size_t(48896),   // measured failure, double/cdouble
                               std::size_t(49152),   // measured failure, ALL FOUR types
                               std::size_t(49664),   // the band's upper edge
                               std::size_t(50176)}) {// above the band: no pad
        if (target <= scratch) continue;
        const std::size_t tile = target - scratch;
        // (m|1) is ODD by construction (getrf_tile_ld), so search the odd
        // factorisations of tile/sz with m >= n.
        Row found{0, 0, 0};
        for (int n = 1; n <= 1024 && !found.m; ++n) {
            const std::size_t denom = std::size_t(n) * sz;
            if (tile % denom) continue;
            const std::size_t q = tile / denom;
            if ((q & 1u) == 0) continue;                // must be an odd ld
            if (q < std::size_t(n)) continue;           // keep the panel tall
            if (q > 8192) continue;
            found = Row{target, int(q), n};
        }
        if (found.m) rows.push_back(found);
    }

    // ANTI-VACUITY 2: the row that fails for every scalar type must be present.
    bool has_49152 = false;
    for (const Row& r : rows) if (r.bytes == 49152) has_49152 = true;
    ASSERT_TRUE(has_49152) << "no (m, n) with a 49,152 B footprint was constructible for this "
                              "scalar type; the discriminating row is missing";

    for (const Row& r : rows) {
        ASSERT_EQ(raw_bytes(r.m, r.n), r.bytes) << "row does not ask for " << r.bytes << " B";
        const bool in_band = (r.bytes > kLo && r.bytes <= kHi);

        // (a) THE PAD ARITHMETIC. EXPECT and not ASSERT, deliberately: an ASSERT returns
        // from the whole test and would MASK (b), which is an independent claim.
        EXPECT_EQ(min_budget(r.m, r.n), in_band ? kPadTo : r.bytes)
            << "the " << r.bytes << " B leaf (" << r.m << "x" << r.n << ") "
            << (in_band ? "is inside the 48 KB hole band but is not padded over it"
                        : "is outside the band and must not be padded");

        // (b) THE LAUNCH.
        const std::size_t need = in_band ? kPadTo : r.bytes;
        if (this->budget() < need) continue;   // a smaller device; (a) still ran
        ASSERT_TRUE(this->leaf_fits(r.m, r.n));

        const int m = r.m, n = r.n, batch = 2, k = std::min(m, n);
        const int ld = m + 3;
        const int stride = ld * n + 5;
        UnifiedVector<T> buf(size_t(stride) * batch, mk<T>(-9.75e3, 4.5e3));
        Rng rg(unsigned(r.bytes % 9973) + 17u);
        for (int b = 0; b < batch; ++b)
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < m; ++i)
                    buf[size_t(b) * stride + size_t(j) * ld + i] = mk<T>(rg.next(), rg.next());
        std::vector<T> a0(buf.begin(), buf.end());
        UnifiedVector<int> piv(size_t(k) * batch, -12345);
        UnifiedVector<int32_t> info(size_t(batch), 0);

        bool resident = false;
        ASSERT_NO_THROW(
            sycl_getrf::getrf_panel_factorize<T>(*this->ctx, buf.data(), ld, stride,
                                                 m, n, batch, piv.data(), k, 0,
                                                 info.data(), &resident))
            << "the resident leaf could not be launched with a " << r.bytes << " B tile ("
            << m << "x" << n << ")";
        this->ctx->wait();
        EXPECT_TRUE(resident)
            << "the " << r.bytes << " B panel took the GLOBAL leaf, so this row did not "
               "exercise the local-memory launch at all";
        if (!resident) continue;

        for (int b = 0; b < batch; ++b) {
            const double res = pa_lu_residual<T>(a0.data() + size_t(b) * stride,
                                                 buf.data() + size_t(b) * stride,
                                                 piv.data() + size_t(b) * k, m, n, ld);
            EXPECT_LE(res, lu_tol<T>(std::max(m, n)))
                << "the " << r.bytes << " B panel launched but factorised incorrectly at b=" << b;
        }
    }
}

// L0b. THE 48 KB LAUNCH HOLE, FOR THE **FUSED GETRS** KERNELS.
//
// DECLARED AHEAD OF EVERY OTHER FUSED-GETRS TEST, for the sticky-per-CUfunction
// reason L0 states: the tier's kernels are templated on the compile-time
// accumulator width NR, every rung below runs at nrhs = 8, and any earlier getrs
// with 4 < nrhs <= 8 would warm them. DO NOT MOVE IT BELOW THE F-SERIES.
TYPED_TEST(LuTest, FusedGetrsLaunchHoleAt48KiB) {
    using T = typename TestFixture::T;
    const std::size_t sz = sizeof(T);

    // ---- layer (a): the capacity inversion --------------------------------
    // The advertised capacity, once a caller sizes by it, must still be launchable
    // within the budget it was asked about; getrs_hole_padded is NOT monotone.
    for (std::size_t budget = kFusedHoleLo - 2048; budget <= kFusedHolePadTo + 2048; ++budget) {
        const std::size_t cap = sycl_getrs::getrs_fused_max_rhs_elems<T>(budget);
        if (cap == 0) continue;
        ASSERT_LE(fused_capacity_bytes(cap, sz), budget)
            << "getrs_fused_max_rhs_elems advertised " << cap << " elements for a budget of "
            << budget << " B, but that capacity asks the runtime for "
            << fused_capacity_bytes(cap, sz)
            << " B once the 48 KB hole pad is applied -- an UNLAUNCHABLE capacity, which is "
               "route_getrs.hh's fused_max_elems and therefore a supports() that promises a "
               "route the facade cannot service";
    }
    // Coarse ladder past the band, including this device's own budget.
    for (std::size_t budget : {std::size_t(4096), std::size_t(16384), std::size_t(32768),
                               std::size_t(65536), std::size_t(98304), std::size_t(163840),
                               std::size_t(232448), this->budget()}) {
        const std::size_t cap = sycl_getrs::getrs_fused_max_rhs_elems<T>(budget);
        if (cap == 0) continue;
        ASSERT_LE(fused_capacity_bytes(cap, sz), budget) << "budget " << budget;
    }
    // ANTI-VACUITY for the sweep: the pad must actually fire somewhere inside it,
    // otherwise the loop above is a tautology over a function that never pads.
    {
        const std::size_t blk = std::size_t(kFusedNbMax) * std::size_t(kFusedNbMax + 1) * sz;
        bool padded_somewhere = false;
        for (std::size_t e = 0; e * sz + blk <= kFusedHolePadTo; ++e)
            if (fused_capacity_bytes(e, sz) == kFusedHolePadTo && e * sz + blk != kFusedHolePadTo)
                padded_somewhere = true;
        ASSERT_TRUE(padded_somewhere)
            << "no element count in range lands inside the (47104, 49664] band, so layer (a) "
               "cannot see a pad regression at all";
    }
    // And the device must advertise a usable capacity, or the tier is dead here.
    ASSERT_GT(sycl_getrs::getrs_fused_max_rhs_elems<T>(this->budget()), std::size_t(0))
        << "the fused tier reports zero capacity on this device";

    // ---- layer (b): the launch, across the band ---------------------------
    const int nrhs = int(sycl_getrs::kGetrsFusedMaxRhs);
    ASSERT_EQ(nrhs, 8) << "the ladder is solved at nrhs = 8; re-derive the orders if this moves";

    const std::size_t cap = sycl_getrs::getrs_fused_max_rhs_elems<T>(this->budget());
    int rungs_run = 0, rungs_over_capacity = 0;
    for (std::size_t want : {kFusedHoleLo, std::size_t(48896), std::size_t(49152),
                             kFusedHoleHi, kFusedHolePadTo}) {
        const int n = fused_order_for_raw_bytes(want, nrhs, sz);
        if (n < 0) {
            ADD_FAILURE() << "no order lands on a raw request of exactly " << want
                          << " B at nrhs=" << nrhs << " for a " << sz << "-byte scalar; the "
                             "ladder cannot cross the band and this test is vacuous";
            continue;
        }
        if (std::size_t(n) * std::size_t(nrhs) > cap) { ++rungs_over_capacity; continue; }

        // The rung must land where this file thinks it does, PAD INCLUDED.
        const std::size_t asked = fused_launch_bytes(n, nrhs, sz);
        const bool in_band = (want > kFusedHoleLo && want <= kFusedHoleHi);
        ASSERT_EQ(asked, in_band ? kFusedHolePadTo : want)
            << "rung " << want << " B (n=" << n << "): the launcher asks for " << asked;

        auto p = make_fabricated_factor<T>(n, 1, 2200u + unsigned(want % 1000u));
        ASSERT_FALSE(interchange_is_involution(p.expect_piv))
            << "rung " << want << ": the fabricated interchange list is SELF-INVERSE, so the "
               "transposed arm's backwards walk is indistinguishable from a forwards one";

        for (Transpose op : {Transpose::NoTrans, Transpose::Trans}) {
            auto rhs = make_rhs<T>(n, nrhs, 1,
                                   3300u + unsigned(want % 1000u) + unsigned(int(op)));
            auto A = view_of(p);
            auto Bv = view_of(rhs);
            UnifiedVector<std::byte> ws(std::max<std::size_t>(
                1, sycl_getrs::getrs_fused_buffer_size<T>(*this->ctx, A, Bv, op)));
            ASSERT_NO_THROW(sycl_getrs::getrs_fused_dispatch<T>(
                *this->ctx, A, Bv, op, p.piv.to_span(), ws.to_span()))
                << "rung " << want << " B (n=" << n << ", asked " << asked
                << " B, transA=" << int(op) << ") REFUSED TO LAUNCH";
            this->ctx->wait();

            const double res = fused_factor_residual<T>(
                p.buf.data(), reinterpret_cast<const int*>(p.piv.data()),
                rhs.buf.data(), rhs.b0.data(), n, nrhs, p.ld, rhs.ld, op);
            if (verbose())
                std::printf("[verbose] fused hole rung %6zu B n=%4d op=%d  res=%.4e tol=%.4e\n",
                            want, n, int(op), res, solve_tol<T>(n));
            EXPECT_LE(res, solve_tol<T>(n))
                << "rung " << want << " B (n=" << n << ", transA=" << int(op)
                << ") launched but did not solve";
            check_rhs_pad_intact(rhs, "fused/hole");
            if (this->HasFailure()) return;
        }
        ++rungs_run;
    }
    if (rungs_over_capacity > 0) {
        GTEST_SKIP() << rungs_over_capacity << " of 5 rungs are above this device's "
                     << "resident-RHS capacity (" << cap << " elements); the band was crossed "
                     << rungs_run << " times";
    }
    EXPECT_EQ(rungs_run, 5) << "the ladder did not cross the band from both sides";
}

// L1. THE CTA TIER: the residual, the partial-pivoting property, and the
// EXACT interchange list.
TYPED_TEST(LuTest, CtaFactorisesAndPivotsExactly) {
    using T = typename TestFixture::T;
    const int cap = this->cta_max_n();
    ASSERT_GE(cap, 32) << "the CTA tier advertises a capacity of " << cap
                       << ", so this test cannot reach it";

    int ran = 0;
    for (int n : {3, 8, 31, 32, 33, 64}) {
        if (n > cap) continue;
        ++ran;
        auto rnd = make_random<T>(n, 3, 991u + unsigned(n));
        this->run_cta(rnd);
        check_factor(rnd, "cta/random");
        for (int b = 0; b < rnd.batch; ++b) ASSERT_EQ(rnd.info[b], 0) << "n=" << n << " b=" << b;

        auto dom = make_dominant_permuted<T>(n, 3, 7717u + unsigned(n));
        this->run_cta(dom);
        // ANTI-VACUITY: the pivot assertion in check_factor is worthless if the
        // matrix pivots on its own diagonal at every step.
        ASSERT_GE(non_diagonal_pivots(dom, 0), n / 2)
            << "n=" << n << ": the dominant-permuted construction produced a near-identity "
               "pivot list, so the elementwise pivot oracle is vacuous";
        check_factor(dom, "cta/dominant-permuted");
        for (int b = 0; b < dom.batch; ++b) ASSERT_EQ(dom.info[b], 0);
        if (this->HasFailure()) return;
    }
    ASSERT_GT(ran, 0);
}

// L2. THE BLOCKED DRIVER, same two oracles, over orders that straddle its own
// block width in both directions.
TYPED_TEST(LuTest, BlockedFactorisesAndPivotsExactly) {
    using T = typename TestFixture::T;
    for (int n : {5, 31, 32, 33, 64, 96, 100, 129}) {
        auto rnd = make_random<T>(n, 3, 2311u + unsigned(n));
        this->run_blocked(rnd);
        check_factor(rnd, "blocked/random");
        for (int b = 0; b < rnd.batch; ++b) ASSERT_EQ(rnd.info[b], 0) << "n=" << n << " b=" << b;

        auto dom = make_dominant_permuted<T>(n, 3, 5501u + unsigned(n));
        this->run_blocked(dom);
        ASSERT_GE(non_diagonal_pivots(dom, 0), n / 2)
            << "n=" << n << ": the dominant-permuted construction produced a near-identity "
               "pivot list, so the elementwise pivot oracle is vacuous";
        check_factor(dom, "blocked/dominant-permuted");
        for (int b = 0; b < dom.batch; ++b) ASSERT_EQ(dom.info[b], 0);
        if (this->HasFailure()) return;
    }
}

// L2b. THE THREE SPELLINGS OF THE LEFT-HAND INTERCHANGE AGREE BIT FOR BIT -- the
// SAME transposition list in the SAME order, so the assertion is BITWISE and not
// "both residuals are small". n = 129 leaves a ONE-COLUMN final panel, where the
// deferred pass's extents must come from ib and never from nb.
// getrf_blocked.cc latches its environment read in a function-local static, so the
// file-scope object below is what makes the latch land on "present" before main.
// evidence: docs/perf/lu.md#getrf-deferred-left-gather
namespace {
struct LeftLaswpKnobPresent {
    LeftLaswpKnobPresent() { ::setenv("BATCHLAS_GETRF_LASWP", "defer_gather", /*overwrite=*/0); }
};
const LeftLaswpKnobPresent kLeftLaswpKnobPresent;
}  // namespace

TYPED_TEST(LuTest, LeftInterchangeSpellingsAgreeBitForBit) {
    using T = typename TestFixture::T;
    struct Arm { const char* env; unsigned mode; };
    const Arm arms[] = {{"inloop", 0u}, {"defer_walk", 1u}, {"defer_gather", 2u}};

    for (int n : {33, 64, 96, 129, 160}) {
        std::vector<std::vector<T>> facs;
        std::vector<std::vector<int>> pivs;
        for (const Arm& a : arms) {
            ::setenv("BATCHLAS_GETRF_LASWP", a.env, 1);
            ASSERT_EQ(this->left_mode(n), a.mode)
                << "n=" << n << ": the driver did not resolve the '" << a.env
                << "' spelling, so every comparison below would be between two copies of the "
                   "SAME arm. The environment read latched before this test ran.";

            auto p = make_random<T>(n, 3, 4441u + unsigned(n));
            this->run_blocked(p);
            check_factor(p, a.env);
            for (int b = 0; b < p.batch; ++b)
                ASSERT_EQ(p.info[b], 0) << a.env << " n=" << n << " b=" << b;

            facs.emplace_back(p.buf.data(), p.buf.data() + p.buf.size());
            std::vector<int> pv;
            for (int b = 0; b < p.batch; ++b) {
                const int* ip = piv_item(p, b);
                pv.insert(pv.end(), ip, ip + p.n);
            }
            pivs.push_back(std::move(pv));
            if (this->HasFailure()) { ::setenv("BATCHLAS_GETRF_LASWP", "defer_gather", 1); return; }
        }

        for (std::size_t a = 1; a < facs.size(); ++a) {
            ASSERT_EQ(facs[a].size(), facs[0].size());
            std::size_t diff = 0;
            for (std::size_t i = 0; i < facs[0].size(); ++i)
                if (std::memcmp(&facs[a][i], &facs[0][i], sizeof(T)) != 0) ++diff;
            EXPECT_EQ(diff, std::size_t(0))
                << "n=" << n << ": '" << arms[a].env << "' differs from '" << arms[0].env
                << "' in " << diff << " of " << facs[0].size()
                << " elements -- the deferred pass is not the same composition";
            EXPECT_EQ(pivs[a], pivs[0])
                << "n=" << n << ": '" << arms[a].env << "' produced a different interchange list";
        }
        ::setenv("BATCHLAS_GETRF_LASWP", "defer_gather", 1);
        if (this->HasFailure()) return;
    }
}

// L3. THE BLOCK BOUNDARY IS QUERIED, NOT ASSUMED. A straddle test that cannot
// see where the boundary is keeps passing after the width moves while silently
// no longer testing a short final panel.
TYPED_TEST(LuTest, BlockWidthStraddleIsQueriedNotAssumed) {
    using T = typename TestFixture::T;

    const int nb0 = this->nb(256);
    ASSERT_GE(nb0, 1) << "getrf_blocked_debug_params reports no blocking, so the blocked "
                         "driver is absent and this test cannot straddle anything";

    const int n_exact = 4 * nb0;          // an EXACT multiple: no short final panel
    const int n_short = 4 * nb0 + 1;      // one column over: the shortest possible final panel
    const int n_mid   = 3 * nb0 + nb0 / 2 + 1;

    // The straddle is ASSERTED against the driver's own width, at the ORDERS the
    // test actually runs, because nb is clamped to n (getrf_blocked_nb).
    ASSERT_EQ(this->nb(n_exact), nb0);
    ASSERT_EQ(this->nb(n_short), nb0);
    ASSERT_EQ(this->nb(n_mid), nb0);
    ASSERT_EQ(n_exact % nb0, 0) << "the 'exact multiple' order is not one";
    ASSERT_NE(n_short % nb0, 0) << "the 'short final panel' order has none";
    ASSERT_NE(n_mid % nb0, 0);
    ASSERT_EQ(n_short % nb0, 1) << "the short final panel is not the narrowest one available";

    // The leaf choice the query reports must be the one getrf_leaf_fits makes for the
    // leading panel.
    for (int n : {n_exact, n_short, n_mid}) {
        const unsigned lf = this->leaf(n);
        ASSERT_TRUE(lf == 1u || lf == 2u) << "n=" << n << ": leaf tag " << lf;
        EXPECT_EQ(lf == 1u, this->leaf_fits(n, std::min(nb0, n)))
            << "n=" << n << ": getrf_blocked_debug_params and getrf_leaf_fits disagree about "
               "the leading panel's residency";
    }

    for (int n : {n_exact, n_short, n_mid}) {
        auto dom = make_dominant_permuted<T>(n, 2, 313u + unsigned(n));
        this->run_blocked(dom);
        ASSERT_GE(non_diagonal_pivots(dom, 0), n / 2);
        check_factor(dom, "blocked/straddle");
        for (int b = 0; b < dom.batch; ++b) ASSERT_EQ(dom.info[b], 0) << "n=" << n;
        if (this->HasFailure()) return;
    }
}

// L4. BOTH PANEL RESIDENCIES FACTORISE CORRECTLY. getrf_panel_factorize is the ONE
// decision site between the local-memory leaf and the global one; the residency is
// ASSERTED from the launcher's own out-parameter.
TYPED_TEST(LuTest, BothPanelLeavesFactoriseCorrectly) {
    using T = typename TestFixture::T;
    const int nbw = this->nb(4096);
    ASSERT_GE(nbw, 1);

    // A panel that fits and one that provably cannot: grow m until the predicate
    // says no, rather than picking a number that stops being large enough.
    int m_small = 64, m_big = 128;
    while (m_big < (1 << 20) && this->leaf_fits(m_big, nbw)) m_big *= 2;
    ASSERT_FALSE(this->leaf_fits(m_big, nbw))
        << "no panel height was found that overflows local memory";
    ASSERT_TRUE(this->leaf_fits(m_small, nbw));

    for (int pass = 0; pass < 2; ++pass) {
        const int m = pass ? m_big : m_small;
        const int n = nbw, batch = 2, k = std::min(m, n);
        const int ld = m + 3, stride = ld * n + 5;
        UnifiedVector<T> buf(size_t(stride) * batch, mk<T>(-9.75e3, 4.5e3));
        Rng rg(4441u + unsigned(pass));
        for (int b = 0; b < batch; ++b)
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < m; ++i)
                    buf[size_t(b) * stride + size_t(j) * ld + i] = mk<T>(rg.next(), rg.next());
        std::vector<T> a0(buf.begin(), buf.end());
        UnifiedVector<int> piv(size_t(k) * batch, -12345);
        UnifiedVector<int32_t> info(size_t(batch), 0);

        bool resident = false;
        ASSERT_NO_THROW(
            sycl_getrf::getrf_panel_factorize<T>(*this->ctx, buf.data(), ld, stride,
                                                 m, n, batch, piv.data(), k, 0,
                                                 info.data(), &resident));
        this->ctx->wait();
        ASSERT_EQ(resident, pass == 0)
            << "panel " << m << "x" << n << " took the "
            << (resident ? "resident" : "global") << " leaf, which is not the one under test";

        for (int b = 0; b < batch; ++b) {
            ASSERT_EQ(info[b], 0);
            const int* ip = piv.data() + size_t(b) * k;
            for (int s = 0; s < k; ++s)
                ASSERT_TRUE(ip[s] >= s + 1 && ip[s] <= m) << "ipiv[" << s << "] = " << ip[s];
            EXPECT_LE(pa_lu_residual<T>(a0.data() + size_t(b) * stride,
                                        buf.data() + size_t(b) * stride, ip, m, n, ld),
                      lu_tol<T>(m))
                << (resident ? "resident" : "global") << " leaf, b=" << b;
            EXPECT_LE(worst_pivot_ratio<T>(buf.data() + size_t(b) * stride, m, n, ld),
                      1.0 + 32.0 * eps_of<T>())
                << (resident ? "resident" : "global") << " leaf, b=" << b;
        }
        if (this->HasFailure()) return;
    }
}

// L5. A SINGULAR MATRIX: `info` is EXACT-ZERO, 1-BASED, GLOBAL, per item and
// FIRST-FAILURE-WINS, with the other batch items unaffected and the failed item
// still FINITE (?GETF2 records the failure and SKIPS the reciprocal scale). The
// failure is planted as an EXACTLY ZERO COLUMN inside the SECOND AND THIRD PANELS:
// a block-local info offset reports the panel-relative column and passes every
// single-panel test.
TYPED_TEST(LuTest, SingularColumnGivesGlobalOneBasedInfoFirstFailureWins) {
    using T = typename TestFixture::T;
    const int nbw = this->nb(256);
    ASSERT_GE(nbw, 2);

    const int n = 3 * nbw + 7;
    const int c1 = nbw + 3;              // second panel: piv_base = nbw, so a block-local
    const int c2 = 2 * nbw + 5;          // offset would report 4 instead of c1 + 1
    ASSERT_GT(c1, nbw) << "the planted failure is inside the FIRST panel; a block-local info "
                          "offset would be indistinguishable from a global one";
    ASSERT_LT(c1, c2);
    ASSERT_LT(c2, n);

    auto p = make_dominant_permuted<T>(n, 4, 8123u);
    const int bad = 2;                   // NOT item 0: a wrong batch stride cannot move item 0
    for (int j : {c1, c2})
        for (int i = 0; i < n; ++i)
            p.a0[size_t(bad) * p.stride + size_t(j) * p.ld + i] = mk<T>(0.0, 0.0);
    p.expect_piv.clear();                // the zero columns change the sequence
    poison(p);

    this->run_blocked(p);

    for (int b = 0; b < p.batch; ++b) {
        if (b == bad) {
            EXPECT_EQ(p.info[b], c1 + 1)
                << "info must be the GLOBAL 1-based column of the FIRST zero pivot "
                   "(planted at global columns " << c1 << " and " << c2 << ", nb=" << nbw << ")";
        } else {
            EXPECT_EQ(p.info[b], 0) << "healthy item " << b << " reported a failure";
        }
        const T* F = p.buf.data() + size_t(b) * p.stride;
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                ASSERT_TRUE(hfinite(up(F[size_t(j) * p.ld + i])))
                    << "item " << b << " left F(" << i << "," << j << ") non-finite; a failed "
                       "item must stay finite, as LAPACK's and cuBLAS's do";
        // A failure in one item must not corrupt the others.
        if (b != bad) {
            const int* ip = piv_item(p, b);
            EXPECT_LE(pa_lu_residual<T>(p.a0.data() + size_t(b) * p.stride, F, ip, n, n, p.ld),
                      lu_tol<T>(n)) << "healthy item " << b;
        }
    }

    // The CTA tier tells the same story at an order it can hold, where piv_base is
    // 0 throughout -- so this half pins the 1-based-ness alone.
    if (this->cta_max_n() >= 40) {
        const int nc = 40, cc1 = 11, cc2 = 27;
        auto q = make_dominant_permuted<T>(nc, 3, 6611u);
        for (int j : {cc1, cc2})
            for (int i = 0; i < nc; ++i)
                q.a0[size_t(1) * q.stride + size_t(j) * q.ld + i] = mk<T>(0.0, 0.0);
        q.expect_piv.clear();
        poison(q);
        this->run_cta(q);
        EXPECT_EQ(q.info[0], 0);
        EXPECT_EQ(q.info[1], cc1 + 1);
        EXPECT_EQ(q.info[2], 0);
    }
}

// L5b. THE `info` ZERO PRE-PASS IS ORDERED AHEAD OF THE PANEL THAT READS IT, ON
// AN OUT-OF-ORDER QUEUE -- the only test here not on the fixture's queue.
//
// getf2_panel_device READS info[b] to implement first-failure-wins across the
// blocked driver's panels, so the fill is a true read-after-write dependence.
// Unordered, the panel loads the caller's pre-call garbage and never records the
// real failure. The batch is large on purpose: this is a race.
TYPED_TEST(LuTest, InfoFillIsOrderedAheadOfThePanelOnAnOutOfOrderQueue) {
    using T = typename TestFixture::T;
    // ONE SCALAR TYPE, DELIBERATELY: what is under test is a HOST-SIDE SUBMISSION
    // ORDER, identical for every scalar type and backend; the sweep's cost is not.
    if constexpr (!std::is_same_v<T, float>) {
        GTEST_SKIP() << "the submission-order defect is type-independent; float carries it";
    } else {
    if (this->cta_max_n() < 32)
        GTEST_SKIP() << "this device's CTA tile cannot hold order 32";

    Queue ooo(*this->ctx, /*in_order=*/false);
    ASSERT_FALSE(ooo.in_order())
        << "ANTI-VACUITY: an in-order queue orders the fill for free, so the whole "
           "test would pass over the defect it exists to catch";

    const int zc = 7;                   // the planted singular column, 0-based

    // THE LOOP SHAPE IS PART OF THE CALIBRATION: re-copying the matrix from the host
    // between repetitions -- the obvious thing to write -- touches managed memory hard
    // enough to serialise the queue and close the window, so the matrix is staged once
    // and re-factorised in place; the planted zero column survives the factorisation.
    auto seed_info = [](Lu<T>& q) {
        std::fill(q.info.begin(), q.info.end(), int32_t(-12345));
    };
    auto count_wrong = [&](Lu<T>& q, long& wrong, long& poisoned) {
        for (int b = 0; b < q.batch; ++b) {
            if (q.info[b] != zc + 1) ++wrong;
            if (q.info[b] == -12345) ++poisoned;
        }
    };

    // --- the CTA tier ---------------------------------------------------
    {
        const int n = 32, batch = 65536, reps = 25;
        auto p = make_random<T>(n, batch, 3u);
        for (int b = 0; b < batch; ++b)
            for (int i = 0; i < n; ++i)
                p.a0[size_t(b) * p.stride + size_t(zc) * p.ld + i] = mk<T>(0.0, 0.0);
        p.expect_piv.clear();
        poison(p);                                   // stage the matrix ONCE
        auto V = view_of(p);

        // THE IN-ORDER CONTROL, which makes the sweep's oracle legitimate: a miss is then
        // an ORDERING failure and not the oracle drifting.
        long cw = 0, cp = 0;
        for (int r = 0; r < 5; ++r) {
            seed_info(p);
            UnifiedVector<std::byte> ws(std::max<std::size_t>(
                1, sycl_getrf::getrf_cta_buffer_size<T>(*this->ctx, V)));
            sycl_getrf::getrf_cta_dispatch<T>(*this->ctx, V, p.piv.to_span(), ws.to_span(),
                                              p.info.to_span());
            this->ctx->wait();
            count_wrong(p, cw, cp);
        }
        ASSERT_EQ(cw, 0) << "CONTROL: the in-order queue itself did not report the planted "
                            "column, so the oracle is wrong and the sweep below means nothing";

        long wrong = 0, poisoned = 0;
        for (int r = 0; r < reps; ++r) {
            seed_info(p);
            UnifiedVector<std::byte> ws(std::max<std::size_t>(
                1, sycl_getrf::getrf_cta_buffer_size<T>(ooo, V)));
            sycl_getrf::getrf_cta_dispatch<T>(ooo, V, p.piv.to_span(), ws.to_span(),
                                              p.info.to_span());
            ooo.wait();
            count_wrong(p, wrong, poisoned);
        }
        EXPECT_EQ(wrong, 0)
            << "CTA tier: " << wrong << " of " << long(reps) * batch
            << " items did not report the planted singular column " << (zc + 1)
            << "; " << poisoned << " of them returned the CALLER's own -12345, which is "
               "the signature of the panel reading info before the fill landed";
    }

    // --- the blocked tier, whose panel loop reads info ACROSS panels -----
    {
        const int nbw = this->nb(256);
        ASSERT_GE(nbw, 2);
        const int n = 2 * nbw;              // at least two panels, so the READ matters
        const int batch = 32768, reps = 15;
        auto p = make_random<T>(n, batch, 17u);
        for (int b = 0; b < batch; ++b)
            for (int i = 0; i < n; ++i)
                p.a0[size_t(b) * p.stride + size_t(zc) * p.ld + i] = mk<T>(0.0, 0.0);
        p.expect_piv.clear();
        poison(p);
        auto V = view_of(p);

        long cw = 0, cp = 0;
        for (int r = 0; r < 3; ++r) {
            seed_info(p);
            UnifiedVector<std::byte> ws(std::max<std::size_t>(
                1, sycl_getrf::getrf_blocked_buffer_size<T>(*this->ctx, V)));
            sycl_getrf::getrf_blocked_dispatch<T>(*this->ctx, V, p.piv.to_span(), ws.to_span(),
                                                  p.info.to_span(),
                                                  this->gemm_seam(), this->trsm_seam());
            this->ctx->wait();
            count_wrong(p, cw, cp);
        }
        ASSERT_EQ(cw, 0) << "CONTROL (blocked): the in-order queue disagrees with the oracle";

        long wrong = 0, poisoned = 0;
        for (int r = 0; r < reps; ++r) {
            seed_info(p);
            UnifiedVector<std::byte> ws(std::max<std::size_t>(
                1, sycl_getrf::getrf_blocked_buffer_size<T>(ooo, V)));
            sycl_getrf::getrf_blocked_dispatch<T>(ooo, V, p.piv.to_span(), ws.to_span(),
                                                  p.info.to_span(),
                                                  this->gemm_seam(), this->trsm_seam());
            ooo.wait();
            count_wrong(p, wrong, poisoned);
        }
        EXPECT_EQ(wrong, 0)
            << "blocked tier (n=" << n << ", nb=" << nbw << "): " << wrong << " of "
            << long(reps) * batch << " items wrong, " << poisoned
            << " of them the caller's own -12345";
    }
    }
}

// L6. A NEARLY singular matrix is NOT flagged. The `info` predicate is a TRUE
// BINARY ZERO, never a tolerance, and that is a PUBLIC CONTRACT shared with LAPACK
// and cuBLAS: an epsilon floor would report a failure where neither of them does.
TYPED_TEST(LuTest, NearlySingularIsNotFlagged) {
    using T = typename TestFixture::T;
    const int n = 48, c = 19;
    auto p = make_dominant_permuted<T>(n, 2, 4242u);
    // Scale one whole column to ~1e-30: U(c,c) is then tiny but exactly representable
    // and NON-ZERO for all four types, and column scaling does not move an argmax.
    for (int b = 0; b < p.batch; ++b)
        for (int i = 0; i < n; ++i) {
            T& v = p.a0[size_t(b) * p.stride + size_t(c) * p.ld + i];
            v = scale(v, 1e-30);
        }
    poison(p);
    this->run_blocked(p);

    for (int b = 0; b < p.batch; ++b) {
        const T* F = p.buf.data() + size_t(b) * p.stride;
        const double diag = hcabs1(up(F[size_t(c) * p.ld + c]));
        // ANTI-VACUITY: the pivot really is tiny, or this is just "info == 0".
        ASSERT_GT(diag, 0.0) << "U(c,c) is exactly zero, so this is the SINGULAR case";
        ASSERT_LT(diag, 1e-20) << "U(c,c) = " << diag << " is not nearly singular at all";
        EXPECT_EQ(p.info[b], 0)
            << "info = " << p.info[b] << " at b=" << b << " with |U(c,c)| = " << diag
            << " -- a tolerance crept into the singularity predicate, which diverges from "
               "LAPACK and cuBLAS invisibly";
    }
    check_factor(p, "blocked/near-singular", /*check_L=*/true);
}

// L7. THE PIVOT METRIC IS cabs1, NOT THE MODULUS. cublas{C,Z}getrfBatched pivots
// on |z| while LAPACK and this kernel pivot on |Re| + |Im|; on the matrix below the
// two rules SELECT DIFFERENT ROWS, which they do not on random or dominant data.
// evidence: docs/perf/lu.md#correctness-findings
TYPED_TEST(LuTest, PivotSelectionUsesCabs1AndNotTheModulus) {
    using T = typename TestFixture::T;
    if constexpr (!test_utils::is_complex_type_v<T>) {
        GTEST_SKIP() << "cabs1 and the modulus coincide for a real scalar type";
    } else {
        const int n = 4, batch = 2;
        auto p = make_dominant_permuted<T>(n, batch, 99u);
        p.expect_piv.clear();
        for (int b = 0; b < batch; ++b) {
            T* A = p.a0.data() + size_t(b) * p.stride;
            // The per-item factor keeps the batch items DISTINCT -- without it check_factor's
            // batch-stride assertion is unsatisfiable -- and leaves column 0's two decisive
            // entries untouched.
            const double f = 1.0 + 0.25 * double(b);
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                    A[size_t(j) * p.ld + i] = (i == j) ? mk<T>(5.0 * f, 0.0)
                                                       : mk<T>(0.25 * f, -0.125 * f);
            // Column 0: cabs1 reads 3 vs 4 (row 1 wins); |z| reads 3 vs 2.828
            // (row 0 wins). Every other row of column 0 is far below both.
            A[0]        = mk<T>(3.0, 0.0);
            A[1]        = mk<T>(2.0, 2.0);
            for (int i = 2; i < n; ++i) A[size_t(i)] = mk<T>(0.1 * f, 0.1 * f);
        }
        poison(p);

        // ANTI-VACUITY: the two functionals must genuinely disagree on this data.
        const auto z0 = up(p.a0[0]);
        const auto z1 = up(p.a0[1]);
        ASSERT_LT(hcabs1(z0), hcabs1(z1)) << "cabs1 does not prefer row 1 on this matrix";
        ASSERT_GT(habs(z0), habs(z1))     << "the modulus does not prefer row 0 on this matrix";

        for (int tier = 0; tier < 2; ++tier) {
            poison(p);
            if (tier == 0) {
                if (this->cta_max_n() < n) continue;
                this->run_cta(p);
            } else {
                this->run_blocked(p);
            }
            for (int b = 0; b < batch; ++b) {
                EXPECT_EQ(piv_item(p, b)[0], 2)
                    << (tier ? "blocked" : "cta") << ": ipiv[0] = " << piv_item(p, b)[0]
                    << " at b=" << b << ". 2 is cabs1's answer (LAPACK's, netlib's); 1 is the "
                       "MODULUS's answer, which is cuBLAS's and is not this library's contract";
            }
            check_factor(p, tier ? "blocked/pivot-metric" : "cta/pivot-metric");
            if (this->HasFailure()) return;
        }
    }
}

// L8. GETRS, ALL THREE transA MODES. The permutation SIDE changes with the
// transpose, and getting it wrong is a silently wrong answer no NoTrans test can
// see:
//   NoTrans  : apply F to B, then solve L, then U.
//   Trans/CT : solve U^T then L^T, then apply F^{-1} -- the SAME list walked
//              BACKWARDS -- to the OUTPUT.
TYPED_TEST(LuTest, GetrsSolvesAllThreeTransposeModes) {
    using T = typename TestFixture::T;
    const int n = 96, batch = 3;

    for (int nrhs : {1, 5}) {
        auto p = make_dominant_permuted<T>(n, batch, 1777u + unsigned(nrhs));
        this->run_blocked(p);
        ASSERT_GE(non_diagonal_pivots(p, 0), n / 2);
        ASSERT_FALSE(interchange_is_involution(p.expect_piv))
            << "this matrix's permutation is SELF-INVERSE, so the transposed getrs's "
               "backwards walk is indistinguishable from a forwards one and the Trans and "
               "ConjTrans rows below prove nothing";
        check_factor(p, "getrs/factor");
        if (this->HasFailure()) return;

        auto rhs = make_rhs<T>(n, nrhs, batch, 909u + unsigned(nrhs));
        std::vector<std::vector<T>> solutions;

        for (Transpose op : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans}) {
            reset_rhs(rhs);
            auto A = view_of(p);
            auto Bv = view_of(rhs);
            UnifiedVector<std::byte> ws(std::max<std::size_t>(
                1, sycl_getrs::getrs_blocked_buffer_size<T>(*this->ctx, A, Bv, op)));
            ASSERT_NO_THROW(sycl_getrs::getrs_blocked_dispatch<T>(
                *this->ctx, A, Bv, op, p.piv.to_span(), ws.to_span(), this->getrs_seam()));
            this->ctx->wait();

            for (int b = 0; b < batch; ++b) {
                const double res = solve_residual<T>(p.a0.data() + size_t(b) * p.stride,
                                                     rhs.buf.data() + size_t(b) * rhs.stride,
                                                     rhs.b0.data() + size_t(b) * rhs.stride,
                                                     n, nrhs, p.ld, rhs.ld, op);
                if (verbose())
                    std::printf("[verbose] getrs op=%d nrhs=%d b=%d  res=%.4e tol=%.4e\n",
                                int(op), nrhs, b, res, solve_tol<T>(n));
                EXPECT_LE(res, solve_tol<T>(n))
                    << "getrs transA=" << int(op) << " nrhs=" << nrhs << " b=" << b;
            }
            solutions.emplace_back(rhs.buf.begin(), rhs.buf.end());
            if (this->HasFailure()) return;
        }

        // ANTI-VACUITY. NoTrans must differ from Trans, or the mode is not read at all,
        // and for a complex type ConjTrans must differ from Trans, or conj(A) is untested.
        EXPECT_NE(solutions[0], solutions[1])
            << "NoTrans and Trans produced identical solutions; transA is not being read";
        if constexpr (test_utils::is_complex_type_v<T>) {
            EXPECT_NE(solutions[1], solutions[2])
                << "Trans and ConjTrans produced identical solutions; the conjugation is "
                   "not being applied";
        }
    }
}

// L8b. GETRS, THE TWO PERMUTATION SPELLINGS, AGREE BIT FOR BIT. Which one runs is
// a SPEED decision and never a correctness one, so the two arms must agree BIT FOR
// BIT -- strictly stronger than the residual, which both arms pass with the SAME
// wrong permutation. The spelling is READ BACK per arm, because the gather FALLS
// BACK to the walk silently when the tile does not fit local memory.
// evidence: docs/perf/lu.md#getrs-collapsed-permutation
TYPED_TEST(LuTest, GetrsPermutationSpellingsAgreeBitForBit) {
    using T = typename TestFixture::T;
    const int batch = 3;

    for (int n : {96, 257}) {
        for (int nrhs : {5, 70}) {
            auto p = make_dominant_permuted<T>(n, batch, 4242u + unsigned(n + nrhs));
            this->run_blocked(p);
            ASSERT_GE(non_diagonal_pivots(p, 0), n / 4);
            ASSERT_FALSE(interchange_is_involution(p.expect_piv))
                << "this matrix's permutation is SELF-INVERSE, so the gather's REVERSED "
                   "index walk is indistinguishable from its forward one and the Trans and "
                   "ConjTrans rows below prove nothing";
            check_factor(p, "getrs/spellings/factor");
            if (this->HasFailure()) return;

            auto rhs = make_rhs<T>(n, nrhs, batch, 1313u + unsigned(n + nrhs));

            for (Transpose op : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans}) {
                std::vector<std::vector<T>> answer;
                for (const char* spelling : {"walk", "gather"}) {
                    setenv("BATCHLAS_GETRS_LASWP", spelling, 1);

                    // GUARD (1). The driver's own resolution, for THIS shape on
                    // THIS queue, so a fallback the caller cannot see is visible.
                    const int got =
                        sycl_getrs::getrs_perm_spelling_debug<T>(*this->ctx, n, nrhs);
                    const int want = (std::strcmp(spelling, "gather") == 0) ? 1 : 0;
                    ASSERT_EQ(got, want)
                        << "BATCHLAS_GETRS_LASWP=" << spelling << " at n=" << n
                        << " nrhs=" << nrhs << " resolved spelling " << got
                        << ". A gather that fell back to the walk would make the "
                           "bit-identity assertion below compare the walk with itself.";

                    reset_rhs(rhs);
                    auto A = view_of(p);
                    auto Bv = view_of(rhs);
                    // The query must stay 0 under BOTH spellings: the gather is in place.
                    UnifiedVector<std::byte> ws(std::max<std::size_t>(
                        1, sycl_getrs::getrs_blocked_buffer_size<T>(*this->ctx, A, Bv, op)));
                    ASSERT_NO_THROW(sycl_getrs::getrs_blocked_dispatch<T>(
                        *this->ctx, A, Bv, op, p.piv.to_span(), ws.to_span(),
                        this->getrs_seam()));
                    this->ctx->wait();

                    for (int b = 0; b < batch; ++b) {
                        const double res = solve_residual<T>(
                            p.a0.data() + size_t(b) * p.stride,
                            rhs.buf.data() + size_t(b) * rhs.stride,
                            rhs.b0.data() + size_t(b) * rhs.stride,
                            n, nrhs, p.ld, rhs.ld, op);
                        EXPECT_LE(res, solve_tol<T>(n))
                            << "getrs spelling=" << spelling << " transA=" << int(op)
                            << " n=" << n << " nrhs=" << nrhs << " b=" << b;
                    }
                    answer.emplace_back(rhs.buf.begin(), rhs.buf.end());
                    if (this->HasFailure()) { unsetenv("BATCHLAS_GETRS_LASWP"); return; }
                }
                unsetenv("BATCHLAS_GETRS_LASWP");

                // THE STRONG ASSERTION: the same permutation and the same two solves, so the
                // answers must be identical to the last bit.
                size_t diff = 0;
                for (size_t i = 0; i < answer[0].size(); ++i)
                    if (std::memcmp(&answer[0][i], &answer[1][i], sizeof(T)) != 0) ++diff;
                EXPECT_EQ(diff, size_t(0))
                    << "the walk and the collapsed gather disagree in " << diff
                    << " of " << answer[0].size() << " elements at transA=" << int(op)
                    << " n=" << n << " nrhs=" << nrhs
                    << ". They apply the SAME permutation to the SAME buffer and then run "
                       "the SAME two trsm calls, so any difference is a defect.";
                if (this->HasFailure()) return;
            }
        }
    }
}

// L8c. THE SPELLING DECISION SURFACE, WITHOUT RUNNING A KERNEL: the default nrhs
// boundary kGetrsPermGatherMinNrhs, and the CAPACITY REFUSAL above which the
// gather enqueues NOTHING and the driver re-schedules the walk. That fallback is
// silent by design and invisible to every other test in this suite.
TYPED_TEST(LuTest, GetrsPermSpellingDecisionSurface) {
    using T = typename TestFixture::T;
    if (this->ctx->device().type != DeviceType::GPU) GTEST_SKIP() << "the gather is GPU-only";

    unsetenv("BATCHLAS_GETRS_LASWP");
    constexpr int kMin = sycl_getrs::kGetrsPermGatherMinNrhs;
    ASSERT_GE(kMin, 1) << "a boundary below 1 would make the walk unreachable by default";

    // THE DEFAULT nrhs BOUNDARY, both sides.
    if (kMin > 1) {
        EXPECT_EQ(sycl_getrs::getrs_perm_spelling_debug<T>(*this->ctx, 128, kMin - 1), 0)
            << "nrhs just below kGetrsPermGatherMinNrhs must take the WALK by default";
    }
    EXPECT_EQ(sycl_getrs::getrs_perm_spelling_debug<T>(*this->ctx, 128, kMin), 1)
        << "nrhs at kGetrsPermGatherMinNrhs must take the GATHER by default";
    EXPECT_EQ(sycl_getrs::getrs_perm_spelling_debug<T>(*this->ctx, 128, 4 * kMin), 1);

    // linalg::solve issues getrs at nrhs = 1 and is the only caller in the tree;
    // it must keep the walk.
    EXPECT_EQ(sycl_getrs::getrs_perm_spelling_debug<T>(*this->ctx, 512, 1), kMin <= 1 ? 1 : 0);

    // THE OVERRIDES beat the boundary in both directions.
    setenv("BATCHLAS_GETRS_LASWP", "walk", 1);
    EXPECT_EQ(sycl_getrs::getrs_perm_spelling_debug<T>(*this->ctx, 128, 4 * kMin), 0)
        << "BATCHLAS_GETRS_LASWP=walk must force the walk above the boundary";
    setenv("BATCHLAS_GETRS_LASWP", "gather", 1);
    EXPECT_EQ(sycl_getrs::getrs_perm_spelling_debug<T>(*this->ctx, 128, 1), 1)
        << "BATCHLAS_GETRS_LASWP=gather must force the gather below the boundary";

    // THE CAPACITY REFUSAL, forced on, at an order no tile can hold. This is the
    // only assertion in the suite that the fallback branch is reachable at all.
    EXPECT_EQ(sycl_getrs::getrs_perm_spelling_debug<T>(*this->ctx, 1 << 20, 4 * kMin), 0)
        << "the gather must REFUSE (and fall back to the walk) at an order whose column "
           "cannot fit local memory, rather than launching a kernel that cannot run";

    // ...and it must NOT refuse at an order the suite reaches: a capacity that fires
    // early is a lever that never runs.
    EXPECT_EQ(sycl_getrs::getrs_perm_spelling_debug<T>(*this->ctx, 1024, 4 * kMin), 1)
        << "the gather must still fit at n = 1024, the largest order this pass measured";
    unsetenv("BATCHLAS_GETRS_LASWP");
}

// L8d. THE GATHER BUYS NO WORKSPACE, AT ANY WIDTH. The facade takes the workspace
// maximum over EVERY NATIVE TIER THAT supports() the shape, not over the tier the
// route named, so a gather that bought an out-of-place RHS here would bill every
// narrow call that routes to the FUSED tier and needs nothing.
TYPED_TEST(LuTest, GetrsPermGatherBuysNoWorkspace) {
    using T = typename TestFixture::T;
    const int n = 96, batch = 3;
    auto p = make_dominant_permuted<T>(n, batch, 606u);

    for (int nrhs : {1, 8, 128}) {
        auto rhs = make_rhs<T>(n, nrhs, batch, 707u + unsigned(nrhs));
        auto A = view_of(p);
        auto Bv = view_of(rhs);
        for (const char* spelling : {"walk", "gather"}) {
            setenv("BATCHLAS_GETRS_LASWP", spelling, 1);
            for (Transpose op : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans}) {
                EXPECT_EQ(sycl_getrs::getrs_blocked_buffer_size<T>(*this->ctx, A, Bv, op),
                          std::size_t(0))
                    << "the composed getrs must stay workspace-free at nrhs=" << nrhs
                    << " spelling=" << spelling << " transA=" << int(op)
                    << ". A buffer bought here is charged to every narrow call that "
                       "routes to the FUSED tier, because the facade maxes over every "
                       "SUPPORTED native tier and not over the routed one.";
            }
        }
    }
    unsetenv("BATCHLAS_GETRS_LASWP");
}

// L9. GETRI: the inverse, and the promise that A SURVIVES. cublas<t>getriBatched
// takes `const T* const A[]`, so a native arm that wrote through A would be a
// drop-in failure invisible to every residual; the survival is asserted BIT-EXACTLY.
TYPED_TEST(LuTest, GetriInvertsAndLeavesTheFactorUntouched) {
    using T = typename TestFixture::T;
    const int n = 80, batch = 3;

    auto p = make_dominant_permuted<T>(n, batch, 3131u);
    this->run_blocked(p);
    ASSERT_GE(non_diagonal_pivots(p, 0), n / 2);
    ASSERT_FALSE(interchange_is_involution(p.expect_piv))
        << "this matrix's permutation is SELF-INVERSE, so getri's BACKWARD trace through the "
           "interchange list is indistinguishable from a forward one";
    check_factor(p, "getri/factor");
    if (this->HasFailure()) return;
    const std::vector<T> factored(p.buf.begin(), p.buf.end());

    Lu<T> c;
    alloc(c, n, batch, 7, 13);
    std::fill(c.buf.begin(), c.buf.end(), mk<T>(-9.75e3, 4.5e3));
    UnifiedVector<int32_t> cinfo(size_t(batch), int32_t(-12345));

    auto A = view_of(p);
    auto C = view_of(c);
    UnifiedVector<std::byte> ws(std::max<std::size_t>(
        1, sycl_getri::getri_blocked_buffer_size<T>(*this->ctx, A)));
    ASSERT_NO_THROW(sycl_getri::getri_blocked_dispatch<T>(
        *this->ctx, A, C, p.piv.to_span(), ws.to_span(), cinfo.to_span(), this->getri_seam()));
    this->ctx->wait();

    for (int b = 0; b < batch; ++b) {
        EXPECT_EQ(cinfo[b], 0) << "b=" << b;
        const double res = inverse_residual<T>(p.a0.data() + size_t(b) * p.stride,
                                               c.buf.data() + size_t(b) * c.stride,
                                               n, p.ld, c.ld);
        if (verbose())
            std::printf("[verbose] getri b=%d  ||AC-I||/n=%.4e tol=%.4e\n",
                        b, res, inv_tol<T>(n));
        EXPECT_LE(res, inv_tol<T>(n)) << "||A C - I||_F / n at b=" << b;
    }

    for (size_t i = 0; i < factored.size(); ++i)
        ASSERT_EQ(habs(up(p.buf[i]) - up(factored[i])), 0.0)
            << "getri wrote through A at element " << i
            << "; cublas<t>getriBatched takes A as const and a caller may reuse it";

    // The LAST batch item differs from the first, so a wrong output stride cannot pass
    // by broadcasting item 0.
    bool differ = false;
    for (int j = 0; j < n && !differ; ++j)
        for (int i = 0; i < n && !differ; ++i)
            if (habs(up(c.buf[size_t(j) * c.ld + i]) -
                     up(c.buf[size_t(batch - 1) * c.stride + size_t(j) * c.ld + i])) > 0.0)
                differ = true;
    EXPECT_TRUE(differ) << "the first and last inverses are identical";
}

// L10 / L11. THE DROP-IN CONTRACT, BOTH DIRECTIONS. getrf, getrs and getri carry
// INDEPENDENT env variables and INDEPENDENT preferred() windows, so every mixture
// of native and vendor arms is reachable in a shipped build. The two getrf
// implementations are NOT required to agree on the PIVOTS they choose: cuBLAS
// pivots on the modulus for complex.
TYPED_TEST(LuTest, NativeFactorFeedsTheVendorSolvers) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::BackendType;
    if constexpr (!dispatch::factorization_vendor_available<B>) {
        GTEST_SKIP() << "no factorization vendor in this build";
    } else {
        const int n = 72, batch = 3, nrhs = 4;
        auto p = make_dominant_permuted<T>(n, batch, 6767u);
        this->run_blocked(p);
        check_factor(p, "dropin/native-factor");
        if (this->HasFailure()) return;

        auto A = view_of(p);
        {   // vendor getrs on the native factor
            auto rhs = make_rhs<T>(n, nrhs, batch, 4545u);
            auto Bv = view_of(rhs);
            UnifiedVector<std::byte> ws(std::max<std::size_t>(
                1, backend::getrs_vendor_buffer_size<B, T>(*this->ctx, A, Bv,
                                                           Transpose::NoTrans)));
            ASSERT_NO_THROW((backend::getrs_vendor<B, T>(*this->ctx, A, Bv, Transpose::NoTrans,
                                                         p.piv.to_span(), ws.to_span())));
            this->ctx->wait();
            for (int b = 0; b < batch; ++b)
                EXPECT_LE(solve_residual<T>(p.a0.data() + size_t(b) * p.stride,
                                            rhs.buf.data() + size_t(b) * rhs.stride,
                                            rhs.b0.data() + size_t(b) * rhs.stride,
                                            n, nrhs, p.ld, rhs.ld, Transpose::NoTrans),
                          solve_tol<T>(n))
                    << "the VENDOR getrs could not consume the NATIVE getrf's factor, b=" << b;
        }
        {   // vendor getri on the native factor
            Lu<T> c; alloc(c, n, batch, 7, 13);
            std::fill(c.buf.begin(), c.buf.end(), mk<T>(-9.75e3, 4.5e3));
            UnifiedVector<int32_t> ci(size_t(batch), int32_t(-12345));
            auto C = view_of(c);
            UnifiedVector<std::byte> ws(std::max<std::size_t>(
                1, backend::getri_vendor_buffer_size<B, T>(*this->ctx, A)));
            ASSERT_NO_THROW((backend::getri_vendor<B, T>(*this->ctx, A, C, p.piv.to_span(),
                                                         ws.to_span(), ci.to_span())));
            this->ctx->wait();
            for (int b = 0; b < batch; ++b) {
                EXPECT_EQ(ci[b], 0);
                EXPECT_LE(inverse_residual<T>(p.a0.data() + size_t(b) * p.stride,
                                              c.buf.data() + size_t(b) * c.stride, n, p.ld, c.ld),
                          inv_tol<T>(n))
                    << "the VENDOR getri could not consume the NATIVE getrf's factor, b=" << b;
            }
        }
    }
}

TYPED_TEST(LuTest, VendorFactorFeedsTheNativeSolvers) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::BackendType;
    if constexpr (!dispatch::factorization_vendor_available<B>) {
        GTEST_SKIP() << "no factorization vendor in this build";
    } else {
        const int n = 72, batch = 3, nrhs = 4;
        auto p = make_dominant_permuted<T>(n, batch, 8989u);
        {
            auto A = view_of(p);
            UnifiedVector<std::byte> ws(std::max<std::size_t>(
                1, backend::getrf_vendor_buffer_size<B, T>(*this->ctx, A)));
            ASSERT_NO_THROW((backend::getrf_vendor<B, T>(*this->ctx, A, p.piv.to_span(),
                                                         ws.to_span(), p.info.to_span())));
            this->ctx->wait();
            for (int b = 0; b < batch; ++b) ASSERT_EQ(p.info[b], 0);
            // The vendor's own factor must satisfy the same host reconstruction, which proves
            // the two agree on the pivot FORMAT even where they differ on the pivot CHOICE.
            check_factor(p, "dropin/vendor-factor", /*check_L=*/false);
            if (this->HasFailure()) return;
        }

        auto A = view_of(p);
        {   // native getrs on the vendor factor
            auto rhs = make_rhs<T>(n, nrhs, batch, 2323u);
            auto Bv = view_of(rhs);
            UnifiedVector<std::byte> ws(std::max<std::size_t>(
                1, sycl_getrs::getrs_blocked_buffer_size<T>(*this->ctx, A, Bv,
                                                            Transpose::NoTrans)));
            ASSERT_NO_THROW(sycl_getrs::getrs_blocked_dispatch<T>(
                *this->ctx, A, Bv, Transpose::NoTrans, p.piv.to_span(), ws.to_span(),
                this->getrs_seam()));
            this->ctx->wait();
            for (int b = 0; b < batch; ++b)
                EXPECT_LE(solve_residual<T>(p.a0.data() + size_t(b) * p.stride,
                                            rhs.buf.data() + size_t(b) * rhs.stride,
                                            rhs.b0.data() + size_t(b) * rhs.stride,
                                            n, nrhs, p.ld, rhs.ld, Transpose::NoTrans),
                          solve_tol<T>(n))
                    << "the NATIVE getrs could not consume the VENDOR getrf's factor, b=" << b;
        }
        {   // native getri on the vendor factor
            Lu<T> c; alloc(c, n, batch, 7, 13);
            std::fill(c.buf.begin(), c.buf.end(), mk<T>(-9.75e3, 4.5e3));
            UnifiedVector<int32_t> ci(size_t(batch), int32_t(-12345));
            auto C = view_of(c);
            UnifiedVector<std::byte> ws(std::max<std::size_t>(
                1, sycl_getri::getri_blocked_buffer_size<T>(*this->ctx, A)));
            ASSERT_NO_THROW(sycl_getri::getri_blocked_dispatch<T>(
                *this->ctx, A, C, p.piv.to_span(), ws.to_span(), ci.to_span(),
                this->getri_seam()));
            this->ctx->wait();
            for (int b = 0; b < batch; ++b) {
                EXPECT_EQ(ci[b], 0);
                EXPECT_LE(inverse_residual<T>(p.a0.data() + size_t(b) * p.stride,
                                              c.buf.data() + size_t(b) * c.stride, n, p.ld, c.ld),
                          inv_tol<T>(n))
                    << "the NATIVE getri could not consume the VENDOR getrf's factor, b=" << b;
            }
        }
    }
}

// L12. THE ROUTE TABLE AND THE VENDOR-FREE FALLBACK, asked of the REAL shape
// builder on the REAL device. route_vocabulary_tests.cc exercises the table
// against SYNTHETIC shapes; what it cannot see is whether the builder reports a
// capacity at all here -- an LU versus a NoRouteError in a vendor-free build.
TYPED_TEST(LuTest, RouteTableAndTheVendorFreeFallback) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::BackendType;

    auto small = make_dominant_permuted<T>(std::min(40, std::max(2, this->cta_max_n())), 2, 5u);
    auto large = make_dominant_permuted<T>(512, 2, 6u);
    auto Vs = view_of(small);
    auto Vl = view_of(large);

    const auto shape = backend::getrf_op_shape<B, T>(*this->ctx, Vs);
    ASSERT_TRUE(shape.has_value());
    EXPECT_EQ(shape->backend, B) << "the builder must SET s.backend or every coverage row for "
                                    "this op reads Backend::AUTO";
    EXPECT_GE(shape->cta_max_n, 1) << "the CTA capacity is 0 on this device, so the tier is "
                                      "advertised as absent";
    EXPECT_TRUE(shape->blocked_available);

    for (auto* V : {&Vs, &Vl}) {
        const auto free_r = backend::getrf_route<B, T>(*this->ctx, *V, /*vendor_available=*/false);
        EXPECT_TRUE(dispatch::is_native(free_r))
            << "getrf has NO route in a vendor-free build at n=" << V->rows();
        EXPECT_TRUE(free_r.algo == dispatch::Algorithm::CTA ||
                    free_r.algo == dispatch::Algorithm::Blocked);
    }
    {
        auto Bv = make_rhs<T>(large.n, 3, large.batch, 77u);
        auto Bview = view_of(Bv);
        for (Transpose op : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans}) {
            const auto r = backend::getrs_route<B, T>(*this->ctx, Vl, Bview, op, false);
            EXPECT_TRUE(dispatch::is_native(r)) << "getrs has no vendor-free route, transA="
                                                << int(op);
        }
        const auto ri = backend::getri_route<B, T>(*this->ctx, Vl, false);
        EXPECT_TRUE(dispatch::is_native(ri)) << "getri has no vendor-free route";
    }

    // THE SHIPPED WINDOWS, met here at BATCH 2; neither carries a batch term.
    //   getrf: native:blocked for float at order >= 256, cfloat at >= 512.
    //   getri: native:blocked for float at order >= 128, cfloat at >= 256.
    //   double and cdouble earn no window in either op, at any order.
    // evidence: docs/perf/lu.md#getrf-window-evidence
    //           docs/perf/lu.md#getri-window-evidence
    if constexpr (dispatch::factorization_vendor_available<B>) {
        constexpr bool kF  = std::is_same_v<T, float>;
        constexpr bool kCF = std::is_same_v<T, std::complex<float>>;

        // Vs (order <= 40) is below EVERY boundary of both windows, for every
        // type: this is what catches a window that forgets its lower bound.
        EXPECT_TRUE(dispatch::is_vendor(
            backend::getrf_route<B, T>(*this->ctx, Vs, /*vendor_available=*/true)))
            << "getrf routed NATIVE at n=" << Vs.rows()
            << ", below every measured boundary";
        EXPECT_TRUE(dispatch::is_vendor(
            backend::getri_route<B, T>(*this->ctx, Vs, /*vendor_available=*/true)));

        // Vl is n = 512: inside both windows for float and cfloat, outside for the doubles.
        const auto rf512 = backend::getrf_route<B, T>(*this->ctx, Vl, true);
        const auto ri512 = backend::getri_route<B, T>(*this->ctx, Vl, true);
        if constexpr (kF || kCF) {
            EXPECT_TRUE(dispatch::is_native(rf512) &&
                        rf512.algo == dispatch::Algorithm::Blocked)
                << "getrf n=512 batch=2 is inside the measured window for this type "
                   "and must resolve native:blocked";
            EXPECT_TRUE(dispatch::is_native(ri512) &&
                        ri512.algo == dispatch::Algorithm::Blocked)
                << "getri n=512 batch=2 is inside the measured window for this type";
        } else {
            EXPECT_TRUE(dispatch::is_vendor(rf512))
                << "double and cdouble earned NO getrf window: their best cell "
                   "anywhere is 1.067 and 1.012";
            EXPECT_TRUE(dispatch::is_vendor(ri512))
                << "double and cdouble earned NO getri window: cdouble n=512 LOSES "
                   "at 0.954 and double n=1024 is 1.155 and falling";
        }
    }

    // ---- GETRS'S MEASURED WINDOW, ASKED OF THE REAL SHAPE BUILDER ----------
    // route_vocabulary_tests.cc pins the same window against SYNTHETIC shapes. What it
    // cannot see is whether the BUILDER on THIS DEVICE reports capacities at all: one
    // returning 0 for fused_max_elems makes every window assertion there hold
    // vacuously. So the capacities first, then the window.
    // evidence: docs/perf/lu.md#getrs-fused-window-evidence
    {
        auto rhs1 = make_rhs<T>(large.n, 1, large.batch, 5151u);
        auto V1 = view_of(rhs1);
        const auto gs = backend::getrs_op_shape<B, T>(*this->ctx, Vl, V1,
                                                      Transpose::NoTrans);
        ASSERT_TRUE(gs.has_value());
        EXPECT_GT(gs->fused_max_elems, 0)
            << "the builder reports NO resident-RHS capacity on this device, so the "
               "fused tier is advertised as absent and every window assertion in "
               "route_vocabulary_tests.cc holds vacuously";
        EXPECT_EQ(gs->fused_max_nrhs, int64_t(sycl_getrs::kGetrsFusedMaxRhs));
        EXPECT_GE(gs->fused_max_elems, int64_t(large.n))
            << "n=" << large.n << " at nrhs=1 must fit, or the window below is about "
               "a route this device cannot take";

        if constexpr (dispatch::factorization_vendor_available<B>) {
            // INSIDE the window, all three transpose modes.
            for (Transpose op : {Transpose::NoTrans, Transpose::Trans,
                                 Transpose::ConjTrans}) {
                const auto r = backend::getrs_route<B, T>(*this->ctx, Vl, V1, op,
                                                          /*vendor_available=*/true);
                EXPECT_TRUE(dispatch::is_native(r) && r.algo == dispatch::Algorithm::CTA)
                    << "nrhs=1 with a vendor present, transA=" << int(op)
                    << ": the measured window is nrhs<=2 for every type";
            }
            // OUTSIDE it -- above the widest instantiation -- the vendor.
            auto rhsw = make_rhs<T>(large.n, int(sycl_getrs::kGetrsFusedMaxRhs) + 8,
                                    large.batch, 5252u);
            auto Vw2 = view_of(rhsw);
            EXPECT_TRUE(dispatch::is_vendor(backend::getrs_route<B, T>(
                *this->ctx, Vl, Vw2, Transpose::NoTrans, /*vendor_available=*/true)));
        }
    }

    // supports() is CORRECTNESS ONLY. A non-square view has no shape at all, and
    // a heterogeneous batch is refused outright.
    {
        UnifiedVector<T> wide(size_t(40) * 64, mk<T>(1.0, 0.0));
        UnifiedVector<T*> wp(1, nullptr);
        MatrixView<T, MatrixFormat::Dense> W(wide.data(), 40, 64, 40, 40 * 64, 1, wp.data());
        EXPECT_FALSE((backend::getrf_op_shape<B, T>(*this->ctx, W).has_value()))
            << "a non-square view must not describe a getrf";
    }
    {
        using Tbl = dispatch::RouteTable<dispatch::Op::getrf, T>;
        auto s = *shape;
        s.heterogeneous_batch = true;
        EXPECT_FALSE(Tbl::supports({dispatch::Origin::Native, dispatch::Algorithm::CTA}, s));
        EXPECT_FALSE(Tbl::supports({dispatch::Origin::Native, dispatch::Algorithm::Blocked}, s));
        s = *shape;
        s.is_gpu = false;
        EXPECT_FALSE(Tbl::supports({dispatch::Origin::Native, dispatch::Algorithm::Blocked}, s));
        s = *shape;
        s.batch = 0;
        EXPECT_FALSE(Tbl::supports({dispatch::Origin::Native, dispatch::Algorithm::CTA}, s));
    }
}

// L13. THE FACADE REACHES THE NATIVE KERNELS, ASSERTED BIT-EXACTLY. A route
// assertion plus a residual can stay GREEN while every number in it comes from the
// vendor, so the comparison here is BIT-EXACT against the direct entry point --
// factor AND pivots -- which no vendor can reproduce.
TYPED_TEST(LuTest, FacadeReachesTheNativeKernelsBitExactly) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 64, batch = 3;
    ASSERT_GE(this->cta_max_n(), n) << "the CTA pin cannot be exercised at n=" << n;

    for (const char* pin : {"cta", "blocked"}) {
        EnvGuard g("BATCHLAS_GETRF_ROUTE", pin);
        auto direct = make_dominant_permuted<T>(n, batch, 1234u);
        auto viafac = make_dominant_permuted<T>(n, batch, 1234u);

        // THE PIN IS VERIFIED, NEVER ASSUMED: an unrecognised value, or one supports()
        // refuses, silently resolves to the VENDOR and this test compares it with itself.
        auto Vf = view_of(viafac);
        const auto route = backend::getrf_route<B, T>(
            *this->ctx, Vf, dispatch::factorization_vendor_available<B>);
        ASSERT_TRUE(dispatch::is_native(route)) << "the '" << pin << "' pin did not take";
        ASSERT_EQ(route.algo, std::strcmp(pin, "cta") == 0 ? dispatch::Algorithm::CTA
                                                           : dispatch::Algorithm::Blocked)
            << "the '" << pin << "' pin resolved to the other native tier";

        if (std::strcmp(pin, "cta") == 0) this->run_cta(direct);
        else                              this->run_blocked(direct);

        UnifiedVector<std::byte> ws(std::max<std::size_t>(
            1, getrf_buffer_size<B, T>(*this->ctx, Vf)));
        ASSERT_NO_THROW((getrf<B, T>(*this->ctx, Vf, viafac.piv.to_span(), ws.to_span(),
                                     viafac.info.to_span())));
        this->ctx->wait();

        for (size_t i = 0; i < direct.buf.size(); ++i)
            ASSERT_EQ(habs(up(direct.buf[i]) - up(viafac.buf[i])), 0.0)
                << "pin=" << pin << ": the facade's factor differs from the direct entry "
                   "point's at element " << i << " -- something else served this call";
        for (int b = 0; b < batch; ++b)
            for (int k = 0; k < n; ++k)
                ASSERT_EQ(piv_item(direct, b)[k], piv_item(viafac, b)[k])
                    << "pin=" << pin << ": pivot " << k << " of item " << b << " differs";
        check_factor(viafac, "facade/getrf");
        if (this->HasFailure()) return;
    }

    // getrs and getri through the facade, against their direct entry points.
    {
        auto p = make_dominant_permuted<T>(n, batch, 4321u);
        this->run_blocked(p);
        auto A = view_of(p);

        EnvGuard g("BATCHLAS_GETRS_ROUTE", "blocked");
        auto r1 = make_rhs<T>(n, 3, batch, 88u);
        auto r2 = make_rhs<T>(n, 3, batch, 88u);
        auto V1 = view_of(r1);
        auto V2 = view_of(r2);
        const auto rr = backend::getrs_route<B, T>(*this->ctx, A, V2, Transpose::Trans,
                                                   dispatch::factorization_vendor_available<B>);
        ASSERT_TRUE(dispatch::is_native(rr)) << "the getrs pin did not take";

        UnifiedVector<std::byte> w1(std::max<std::size_t>(
            1, sycl_getrs::getrs_blocked_buffer_size<T>(*this->ctx, A, V1, Transpose::Trans)));
        sycl_getrs::getrs_blocked_dispatch<T>(*this->ctx, A, V1, Transpose::Trans,
                                              p.piv.to_span(), w1.to_span(), this->getrs_seam());
        this->ctx->wait();
        UnifiedVector<std::byte> w2(std::max<std::size_t>(
            1, getrs_buffer_size<B, T>(*this->ctx, A, V2, Transpose::Trans)));
        ASSERT_NO_THROW((getrs<B, T>(*this->ctx, A, V2, Transpose::Trans, p.piv.to_span(),
                                     w2.to_span())));
        this->ctx->wait();
        for (size_t i = 0; i < r1.buf.size(); ++i)
            ASSERT_EQ(habs(up(r1.buf[i]) - up(r2.buf[i])), 0.0)
                << "the facade's getrs differs from the direct driver at element " << i;
    }
    {
        auto p = make_dominant_permuted<T>(n, batch, 4321u);
        this->run_blocked(p);
        auto A = view_of(p);

        EnvGuard g("BATCHLAS_GETRI_ROUTE", "blocked");
        Lu<T> c1, c2;
        alloc(c1, n, batch, 7, 13);
        alloc(c2, n, batch, 7, 13);
        std::fill(c1.buf.begin(), c1.buf.end(), mk<T>(-9.75e3, 4.5e3));
        std::fill(c2.buf.begin(), c2.buf.end(), mk<T>(-9.75e3, 4.5e3));
        UnifiedVector<int32_t> i1(size_t(batch), -12345), i2(size_t(batch), -12345);
        auto C1 = view_of(c1);
        auto C2 = view_of(c2);
        const auto rr = backend::getri_route<B, T>(*this->ctx, A,
                                                   dispatch::factorization_vendor_available<B>);
        ASSERT_TRUE(dispatch::is_native(rr)) << "the getri pin did not take";

        UnifiedVector<std::byte> w1(std::max<std::size_t>(
            1, sycl_getri::getri_blocked_buffer_size<T>(*this->ctx, A)));
        sycl_getri::getri_blocked_dispatch<T>(*this->ctx, A, C1, p.piv.to_span(), w1.to_span(),
                                              i1.to_span(), this->getri_seam());
        this->ctx->wait();
        UnifiedVector<std::byte> w2(std::max<std::size_t>(
            1, getri_buffer_size<B, T>(*this->ctx, A)));
        ASSERT_NO_THROW((getri<B, T>(*this->ctx, A, C2, p.piv.to_span(), w2.to_span(),
                                     i2.to_span())));
        this->ctx->wait();
        for (size_t i = 0; i < c1.buf.size(); ++i)
            ASSERT_EQ(habs(up(c1.buf[i]) - up(c2.buf[i])), 0.0)
                << "the facade's getri differs from the direct driver at element " << i;
    }
}

// L14. THE DIRECT ENTRY POINTS REFUSE WHAT supports() REFUSES. They are reachable
// WITHOUT the table, so every gate has to be re-applied there or a pinned-route
// caller walks into an unlaunchable configuration.
TYPED_TEST(LuTest, DirectEntryPointsRefuseWhatSupportsRefuses) {
    using T = typename TestFixture::T;
    const int n = 24, batch = 2;
    auto p = make_dominant_permuted<T>(n, batch, 31u);
    auto A = view_of(p);
    UnifiedVector<std::byte> ws(4096);

    // A non-square view.
    {
        UnifiedVector<T> w(size_t(24) * 32, mk<T>(1.0, 0.0));
        UnifiedVector<T*> wp(1, nullptr);
        MatrixView<T, MatrixFormat::Dense> W(w.data(), 24, 32, 24, 24 * 32, 1, wp.data());
        UnifiedVector<int64_t> pv(64, 0);
        EXPECT_THROW(sycl_getrf::getrf_blocked_dispatch<T>(*this->ctx, W, pv.to_span(),
                                                           ws.to_span(), Span<int32_t>{},
                                                           this->gemm_seam(), this->trsm_seam()),
                     std::invalid_argument);
    }
    // A pivot span shorter than n * batch.
    {
        UnifiedVector<int64_t> shortpiv(size_t(n) * batch - 1, 0);
        EXPECT_THROW(sycl_getrf::getrf_blocked_dispatch<T>(*this->ctx, A, shortpiv.to_span(),
                                                           ws.to_span(), Span<int32_t>{},
                                                           this->gemm_seam(), this->trsm_seam()),
                     std::invalid_argument);
        EXPECT_THROW(sycl_getrf::getrf_cta_dispatch<T>(*this->ctx, A, shortpiv.to_span(),
                                                       ws.to_span(), Span<int32_t>{}),
                     std::invalid_argument);
    }
    // An empty panel-solve seam. NOT defaulted to a native trsm.
    EXPECT_THROW(sycl_getrf::getrf_blocked_dispatch<T>(*this->ctx, A, p.piv.to_span(),
                                                       ws.to_span(), Span<int32_t>{},
                                                       this->gemm_seam(),
                                                       sycl_getrf::GetrfPanelSolveTrsm<T>{}),
                 std::invalid_argument);
    // An order past the CTA tier's advertised capacity.
    {
        const int over = this->cta_max_n() + 1;
        auto big = make_dominant_permuted<T>(over, 1, 41u);
        auto Vb = view_of(big);
        EXPECT_THROW(sycl_getrf::getrf_cta_dispatch<T>(*this->ctx, Vb, big.piv.to_span(),
                                                       ws.to_span(), Span<int32_t>{}),
                     std::invalid_argument)
            << "getrf_cta_dispatch accepted order " << over << " with a capacity of "
            << this->cta_max_n() << "; the table would have promised a launch the device refuses";
    }
    // getrs / getri seam and shape refusals.
    {
        this->run_blocked(p);
        auto rhs = make_rhs<T>(n, 2, batch, 12u);
        auto Bv = view_of(rhs);
        EXPECT_THROW(sycl_getrs::getrs_blocked_dispatch<T>(*this->ctx, A, Bv, Transpose::NoTrans,
                                                           p.piv.to_span(), ws.to_span(),
                                                           sycl_getrs::GetrsSolveTrsm<T>{}),
                     std::invalid_argument);
        auto mismatched = make_rhs<T>(n + 1, 2, batch, 13u);
        auto Bm = view_of(mismatched);
        EXPECT_THROW(sycl_getrs::getrs_blocked_dispatch<T>(*this->ctx, A, Bm, Transpose::NoTrans,
                                                           p.piv.to_span(), ws.to_span(),
                                                           this->getrs_seam()),
                     std::invalid_argument);

        UnifiedVector<int32_t> ci(size_t(batch), 0);
        EXPECT_THROW(sycl_getri::getri_blocked_dispatch<T>(*this->ctx, A, A, p.piv.to_span(),
                                                           ws.to_span(), ci.to_span(),
                                                           this->getri_seam()),
                     std::invalid_argument)
            << "getri must refuse C aliasing A: C is zeroed before A's triangles are read";
        EXPECT_THROW(sycl_getri::getri_blocked_dispatch<T>(*this->ctx, A, A, p.piv.to_span(),
                                                           ws.to_span(), ci.to_span(),
                                                           sycl_getri::GetriSolveTrsm<T>{}),
                     std::invalid_argument);
    }
}

// L15. THE WORKSPACE QUERY COVERS EVERY SUPPORTED ROUTE, AND DEREFERENCES
// NOTHING: getrf_buffer_size and getri_buffer_size are reached from inside a
// layout function under BumpAllocator::measuring() (src/extensions/inv.cc), where
// A arrives with a NULL data pointer. The facade's figure is max(native, vendor).
TYPED_TEST(LuTest, BufferSizeCoversEveryRouteAndNeverDereferences) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 64, batch = 3;
    ASSERT_GE(this->cta_max_n(), n);

    // NULL data, exactly as a measuring pass presents it.
    {
        MatrixView<T, MatrixFormat::Dense> nullv(nullptr, n, n, n + 4, (n + 4) * n + 3, batch,
                                                 nullptr);
        EXPECT_NO_THROW((void)sycl_getrf::getrf_cta_buffer_size<T>(*this->ctx, nullv));
        EXPECT_NO_THROW((void)sycl_getrf::getrf_blocked_buffer_size<T>(*this->ctx, nullv));
        EXPECT_NO_THROW((void)sycl_getri::getri_blocked_buffer_size<T>(*this->ctx, nullv));
        EXPECT_NO_THROW(((void)getrf_buffer_size<B, T>(*this->ctx, nullv)));
        EXPECT_NO_THROW(((void)getri_buffer_size<B, T>(*this->ctx, nullv)));
    }

    for (const char* pin : {"cta", "blocked"}) {
        EnvGuard g("BATCHLAS_GETRF_ROUTE", pin);
        auto p = make_dominant_permuted<T>(n, batch, 2u);
        auto V = view_of(p);
        const auto route = backend::getrf_route<B, T>(
            *this->ctx, V, dispatch::factorization_vendor_available<B>);
        ASSERT_TRUE(dispatch::is_native(route)) << "pin '" << pin << "' did not take";

        const std::size_t need = getrf_buffer_size<B, T>(*this->ctx, V);
        const std::size_t native_need =
            (route.algo == dispatch::Algorithm::CTA)
                ? sycl_getrf::getrf_cta_buffer_size<T>(*this->ctx, V)
                : sycl_getrf::getrf_blocked_buffer_size<T>(*this->ctx, V);
        EXPECT_GE(need, native_need)
            << "pin '" << pin << "': the facade's figure is smaller than the arm it resolved to";

        // Serve EXACTLY that many bytes: a short workspace is a silent heap overflow.
        UnifiedVector<std::byte> ws(std::max<std::size_t>(1, need));
        ASSERT_NO_THROW((getrf<B, T>(*this->ctx, V, p.piv.to_span(), ws.to_span(),
                                     p.info.to_span())));
        this->ctx->wait();
        check_factor(p, "buffer-size/getrf");
        if (this->HasFailure()) return;
    }
}

// F1. THE FUSED TIER SOLVES ALL THREE transA MODES AT EVERY INSTANTIATED WIDTH.
//
// The accumulator width NR is a COMPILE-TIME template parameter chosen by a
// runtime ladder (nrhs <= 1 -> 1, <= 2 -> 2, <= 4 -> 4, else 8), so 1, 2, 4 and 8
// are four different kernels, and 3 and 5 are the shapes where the `if (c < nrhs)`
// guards inside a WIDER accumulator are all that keeps a lane out of a column that
// does not exist. n = 97 is six full nb = 16 blocks plus a FINAL BLOCK OF ONE.
TYPED_TEST(LuTest, FusedGetrsSolvesEveryTransposeAtEveryInstantiatedWidth) {
    using T = typename TestFixture::T;
    const int n = 97, batch = 3;

    auto p = make_dominant_permuted<T>(n, batch, 5150u);
    this->run_blocked(p);
    ASSERT_GE(non_diagonal_pivots(p, 0), n / 2);
    ASSERT_FALSE(interchange_is_involution(p.expect_piv))
        << "this matrix's permutation is SELF-INVERSE, so the transposed arm's backwards walk "
           "is indistinguishable from a forwards one and the Trans/ConjTrans rows prove nothing";
    check_factor(p, "fused/factor");
    if (this->HasFailure()) return;

    for (int nrhs : {1, 2, 3, 4, 5, 8}) {
        ASSERT_LE(int64_t(nrhs), sycl_getrs::kGetrsFusedMaxRhs);
        auto rhs = make_rhs<T>(n, nrhs, batch, 6160u + unsigned(nrhs));
        std::vector<std::vector<T>> solutions;

        for (Transpose op : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans}) {
            reset_rhs(rhs);
            auto A = view_of(p);
            auto Bv = view_of(rhs);
            UnifiedVector<std::byte> ws(std::max<std::size_t>(
                1, sycl_getrs::getrs_fused_buffer_size<T>(*this->ctx, A, Bv, op)));
            ASSERT_NO_THROW(sycl_getrs::getrs_fused_dispatch<T>(
                *this->ctx, A, Bv, op, p.piv.to_span(), ws.to_span()));
            this->ctx->wait();

            // THE ORACLE IS ||op(A) X - B|| AGAINST THE **ORIGINAL** A, and not the L/U-based
            // one the hole ladder uses: only this form is sensitive to the permutation
            // DIRECTION, because only this form knows what A was before it was factored.
            for (int b = 0; b < batch; ++b) {
                const double res = solve_residual<T>(p.a0.data() + size_t(b) * p.stride,
                                                     rhs.buf.data() + size_t(b) * rhs.stride,
                                                     rhs.b0.data() + size_t(b) * rhs.stride,
                                                     n, nrhs, p.ld, rhs.ld, op);
                if (verbose())
                    std::printf("[verbose] fused getrs op=%d nrhs=%d b=%d  res=%.4e tol=%.4e\n",
                                int(op), nrhs, b, res, solve_tol<T>(n));
                EXPECT_LE(res, solve_tol<T>(n))
                    << "fused getrs transA=" << int(op) << " nrhs=" << nrhs << " b=" << b;
            }
            check_rhs_pad_intact(rhs, "fused/window");
            check_items_differ(rhs, "fused/window");
            solutions.emplace_back(rhs.buf.begin(), rhs.buf.end());
            if (this->HasFailure()) return;
        }

        EXPECT_NE(solutions[0], solutions[1])
            << "nrhs=" << nrhs << ": NoTrans and Trans produced identical solutions; transA is "
               "not being read";
        if constexpr (test_utils::is_complex_type_v<T>) {
            EXPECT_NE(solutions[1], solutions[2])
                << "nrhs=" << nrhs << ": Trans and ConjTrans produced identical solutions; the "
                   "conjugation is not being applied";
        }
    }
}

// F2. ORDERS: ONE, THE BLOCK BOUNDARIES, AND THE nb SWITCH AT 1024. nb = 16 below
// order 1024 and 32 at or above it, then clamped to n, so 1023/1024/1025 straddle a
// change of BOTH the block width and the resident block's leading dimension, and
// jb == 1 on the final block disables the unit-diagonal recurrence entirely in two
// of the four substitutions. ORDERS 1 AND 2 SKIP THE INVOLUTION ASSERTION: an
// n-cycle IS self-inverse for n <= 2.
TYPED_TEST(LuTest, FusedGetrsAtBlockBoundariesAndTheNbSwitch) {
    using T = typename TestFixture::T;
    const int batch = 2;

    for (int n : {1, 2, 3, 15, 16, 17, 31, 32, 33, 48, 64, 65, 128, 1023, 1024, 1025}) {
        auto p = make_dominant_permuted<T>(n, batch, 7070u + unsigned(n));
        this->run_blocked(p);
        if (n > 2) {
            ASSERT_FALSE(interchange_is_involution(p.expect_piv)) << "n=" << n;
            ASSERT_GT(non_diagonal_pivots(p, batch - 1), 0) << "n=" << n;
        }
        // ||PA - LU|| is O(n^3); at 1023 and above the end-to-end solve residual
        // against the ORIGINAL A is the oracle, which is O(n^2 nrhs).
        if (n <= 128) {
            check_factor(p, "fused/orders");
            if (this->HasFailure()) return;
        }

        for (int nrhs : {1, 3}) {
            const std::size_t cap = sycl_getrs::getrs_fused_max_rhs_elems<T>(this->budget());
            if (std::size_t(n) * std::size_t(nrhs) > cap) continue;
            auto rhs = make_rhs<T>(n, nrhs, batch, 8080u + unsigned(n) + unsigned(nrhs));
            for (Transpose op : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans}) {
                reset_rhs(rhs);
                auto A = view_of(p);
                auto Bv = view_of(rhs);
                UnifiedVector<std::byte> ws(std::max<std::size_t>(
                    1, sycl_getrs::getrs_fused_buffer_size<T>(*this->ctx, A, Bv, op)));
                ASSERT_NO_THROW(sycl_getrs::getrs_fused_dispatch<T>(
                    *this->ctx, A, Bv, op, p.piv.to_span(), ws.to_span()))
                    << "n=" << n << " nrhs=" << nrhs << " transA=" << int(op);
                this->ctx->wait();
                for (int b = 0; b < batch; ++b) {
                    const double res = solve_residual<T>(p.a0.data() + size_t(b) * p.stride,
                                                         rhs.buf.data() + size_t(b) * rhs.stride,
                                                         rhs.b0.data() + size_t(b) * rhs.stride,
                                                         n, nrhs, p.ld, rhs.ld, op);
                    EXPECT_LE(res, solve_tol<T>(n))
                        << "fused getrs n=" << n << " nrhs=" << nrhs << " transA=" << int(op)
                        << " b=" << b;
                }
                check_rhs_pad_intact(rhs, "fused/orders");
                if (this->HasFailure()) return;
            }
        }
    }
}

// F3. THE TWO CEILINGS: THE WIDTH THE BUILD INSTANTIATED (kGetrsFusedMaxRhs = 8)
// AND THE DEVICE'S RESIDENT-RHS CAPACITY. BOTH MUST HAND BACK, NOT PRODUCE
// GARBAGE. Both live in supports() and never in preferred(), because above either
// the kernel does not launch -- and a SPEED threshold in supports() would make a
// pinned `native:cta` fall through to automatic() and measure the vendor instead.
TYPED_TEST(LuTest, FusedGetrsHandsBackAtBothCeilings) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::BackendType;
    using Tbl = dispatch::RouteTable<dispatch::Op::getrs, T>;
    const dispatch::Route cta{dispatch::Origin::Native, dispatch::Algorithm::CTA};
    const int maxr = int(sycl_getrs::kGetrsFusedMaxRhs);
    const int n = 40, batch = 2;

    auto p = make_dominant_permuted<T>(n, batch, 9090u);
    this->run_blocked(p);
    check_factor(p, "fused/ceiling");
    if (this->HasFailure()) return;
    auto A = view_of(p);

    // ---- ceiling 1: the instantiated width --------------------------------
    for (int nrhs : {maxr, maxr + 1}) {
        auto rhs = make_rhs<T>(n, nrhs, batch, 1010u + unsigned(nrhs));
        auto Bv = view_of(rhs);
        const auto shape = backend::getrs_op_shape<B, T>(*this->ctx, A, Bv, Transpose::NoTrans);
        ASSERT_TRUE(shape.has_value());
        EXPECT_EQ(Tbl::supports(cta, *shape), nrhs <= maxr)
            << "supports({Native, CTA}) at nrhs=" << nrhs << " with fused_max_nrhs="
            << shape->fused_max_nrhs;

        UnifiedVector<std::byte> ws(std::max<std::size_t>(
            1, sycl_getrs::getrs_fused_buffer_size<T>(*this->ctx, A, Bv, Transpose::NoTrans)));
        if (nrhs <= maxr) {
            ASSERT_NO_THROW(sycl_getrs::getrs_fused_dispatch<T>(
                *this->ctx, A, Bv, Transpose::NoTrans, p.piv.to_span(), ws.to_span()));
        } else {
            EXPECT_THROW(sycl_getrs::getrs_fused_dispatch<T>(
                             *this->ctx, A, Bv, Transpose::NoTrans, p.piv.to_span(),
                             ws.to_span()),
                         std::invalid_argument)
                << "the direct entry point accepted nrhs=" << nrhs << " with only " << maxr
                << " instantiated; the table would then promise a route with no kernel behind it";
        }
        this->ctx->wait();
    }

    // ONE PAST THE WIDTH, THROUGH THE FACADE, MUST STILL BE RIGHT: the route has to
    // fall to a tier that can serve it and the answer has to survive the handover.
    {
        const int nrhs = maxr + 1;
        auto rhs = make_rhs<T>(n, nrhs, batch, 1212u);
        auto Bv = view_of(rhs);
        for (Transpose op : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans}) {
            reset_rhs(rhs);
            const auto r = backend::getrs_route<B, T>(
                *this->ctx, A, Bv, op, dispatch::factorization_vendor_available<B>);
            EXPECT_FALSE(dispatch::is_native(r) && r.algo == dispatch::Algorithm::CTA)
                << "nrhs=" << nrhs << " routed to the fused tier, which is not instantiated "
                   "that wide";
            UnifiedVector<std::byte> ws(std::max<std::size_t>(
                1, getrs_buffer_size<B, T>(*this->ctx, A, Bv, op)));
            ASSERT_NO_THROW((getrs<B, T>(*this->ctx, A, Bv, op, p.piv.to_span(), ws.to_span())));
            this->ctx->wait();
            for (int b = 0; b < batch; ++b)
                EXPECT_LE(solve_residual<T>(p.a0.data() + size_t(b) * p.stride,
                                            rhs.buf.data() + size_t(b) * rhs.stride,
                                            rhs.b0.data() + size_t(b) * rhs.stride,
                                            n, nrhs, p.ld, rhs.ld, op),
                          solve_tol<T>(n))
                    << "one width past the fused tier the facade returned a wrong answer, "
                       "transA=" << int(op) << " b=" << b;
        }
    }

    // ---- ceiling 2: the device's resident-RHS capacity ---------------------
    {
        const std::size_t cap = sycl_getrs::getrs_fused_max_rhs_elems<T>(this->budget());
        ASSERT_GT(cap, std::size_t(maxr));
        const int nrhs = maxr;
        // The largest order that still fits, and the first that does not.
        const int fit  = int(cap / std::size_t(nrhs));
        const int over = fit + 1;
        UnifiedVector<int64_t> piv(size_t(over) * 2, int64_t(1));
        for (int b = 0; b < 2; ++b) {
            int* ip = reinterpret_cast<int*>(piv.data()) + size_t(b) * over;
            for (int k = 0; k < over; ++k) ip[k] = k + 1;
        }
        for (int order : {fit, over}) {
            MatrixView<T, MatrixFormat::Dense> An(nullptr, order, order, order,
                                                  int64_t(order) * order, 2, nullptr);
            MatrixView<T, MatrixFormat::Dense> Bn(nullptr, order, nrhs, order,
                                                  int64_t(order) * nrhs, 2, nullptr);
            const auto shape = backend::getrs_op_shape<B, T>(*this->ctx, An, Bn,
                                                             Transpose::NoTrans);
            ASSERT_TRUE(shape.has_value());
            EXPECT_EQ(Tbl::supports(cta, *shape), order == fit)
                << "supports({Native, CTA}) at n=" << order << " nrhs=" << nrhs
                << " against fused_max_elems=" << shape->fused_max_elems;
            if (order == over) {
                UnifiedVector<std::byte> ws(1);
                EXPECT_THROW(sycl_getrs::getrs_fused_dispatch<T>(
                                 *this->ctx, An, Bn, Transpose::NoTrans, piv.to_span(),
                                 ws.to_span()),
                             std::invalid_argument)
                    << "the direct entry point accepted n*nrhs = "
                    << std::size_t(order) * std::size_t(nrhs) << " against a capacity of " << cap;
            }
        }
    }
}

// F4. THE DROP-IN CONTRACT FOR THE FUSED TIER, BOTH DIRECTIONS AND BOTH PRODUCERS.
// The fused tier reads the pivot buffer DIRECTLY -- pivots.as_span<int>(), packed
// 1-BASED int32, an INTERCHANGE LIST -- and re-derives the walk in its own kernel
// rather than delegating to the shared laswp, so it is the arm most exposed to a
// format disagreement.
TYPED_TEST(LuTest, FusedGetrsConsumesEveryFactorProducer) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 40, nrhs = 3, batch = 3;
    ASSERT_GE(this->cta_max_n(), n);

    auto solve_and_check = [&](Lu<T>& p, const char* who) {
        auto A = view_of(p);
        for (Transpose op : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans}) {
            auto rhs = make_rhs<T>(n, nrhs, batch, 2424u + unsigned(int(op)));
            auto Bv = view_of(rhs);
            UnifiedVector<std::byte> ws(std::max<std::size_t>(
                1, sycl_getrs::getrs_fused_buffer_size<T>(*this->ctx, A, Bv, op)));
            ASSERT_NO_THROW(sycl_getrs::getrs_fused_dispatch<T>(
                *this->ctx, A, Bv, op, p.piv.to_span(), ws.to_span()));
            this->ctx->wait();
            for (int b = 0; b < batch; ++b)
                EXPECT_LE(solve_residual<T>(p.a0.data() + size_t(b) * p.stride,
                                            rhs.buf.data() + size_t(b) * rhs.stride,
                                            rhs.b0.data() + size_t(b) * rhs.stride,
                                            n, nrhs, p.ld, rhs.ld, op),
                          solve_tol<T>(n))
                    << "the FUSED getrs could not consume the " << who << " factor, transA="
                    << int(op) << " b=" << b;
            check_rhs_pad_intact(rhs, who);
        }
    };

    {   // native getrf, BLOCKED tier
        auto p = make_dominant_permuted<T>(n, batch, 3535u);
        this->run_blocked(p);
        check_factor(p, "dropin/fused/native-blocked");
        if (this->HasFailure()) return;
        solve_and_check(p, "NATIVE BLOCKED getrf's");
    }
    {   // native getrf, CTA tier
        auto p = make_dominant_permuted<T>(n, batch, 3636u);
        this->run_cta(p);
        check_factor(p, "dropin/fused/native-cta");
        if (this->HasFailure()) return;
        solve_and_check(p, "NATIVE CTA getrf's");
    }
    if constexpr (dispatch::factorization_vendor_available<B>) {
        // VENDOR getrf. Its pivot CHOICE differs from ours for complex types, which is why
        // the oracle is a residual against the ORIGINAL A and not a factor comparison.
        auto p = make_dominant_permuted<T>(n, batch, 3737u);
        {
            auto A = view_of(p);
            UnifiedVector<std::byte> ws(std::max<std::size_t>(
                1, backend::getrf_vendor_buffer_size<B, T>(*this->ctx, A)));
            ASSERT_NO_THROW((backend::getrf_vendor<B, T>(*this->ctx, A, p.piv.to_span(),
                                                         ws.to_span(), p.info.to_span())));
            this->ctx->wait();
            for (int b = 0; b < batch; ++b) ASSERT_EQ(p.info[b], 0);
            check_factor(p, "dropin/fused/vendor-factor", /*check_L=*/false);
            if (this->HasFailure()) return;
        }
        solve_and_check(p, "VENDOR getrf's");

        // AND THE OTHER DIRECTION, on the same factor: the vendor getrs must still consume
        // it, which is what makes the pivot FORMAT a shared fact and not our convention.
        auto A = view_of(p);
        auto rhs = make_rhs<T>(n, nrhs, batch, 2626u);
        auto Bv = view_of(rhs);
        UnifiedVector<std::byte> ws(std::max<std::size_t>(
            1, backend::getrs_vendor_buffer_size<B, T>(*this->ctx, A, Bv, Transpose::NoTrans)));
        ASSERT_NO_THROW((backend::getrs_vendor<B, T>(*this->ctx, A, Bv, Transpose::NoTrans,
                                                     p.piv.to_span(), ws.to_span())));
        this->ctx->wait();
        for (int b = 0; b < batch; ++b)
            EXPECT_LE(solve_residual<T>(p.a0.data() + size_t(b) * p.stride,
                                        rhs.buf.data() + size_t(b) * rhs.stride,
                                        rhs.b0.data() + size_t(b) * rhs.stride,
                                        n, nrhs, p.ld, rhs.ld, Transpose::NoTrans),
                      solve_tol<T>(n))
                << "the VENDOR getrs could not consume the factor the fused tier just read";
    }
}

// F5. A SINGULAR AND A NEARLY-SINGULAR FACTOR. getrs has no info output and no
// singularity contract: ?GETRS divides by U(k,k) unconditionally. So NEARLY
// singular must still be SOLVED to a backward-error bound, and EXACTLY singular
// must PROPAGATE: a finite answer means an EPSILON FLOOR or a SKIPPED division
// returned a plausible-looking wrong number. k = 0 and k = n-1 are where an
// off-by-one in the reverse loop lands.
TYPED_TEST(LuTest, FusedGetrsOnSingularAndNearlySingularFactors) {
    using T = typename TestFixture::T;
    const int n = 64, nrhs = 2, batch = 2;
    const double tiny = std::is_same_v<RealOf<T>, float> ? 1e-6 : 1e-12;

    for (int kz : {0, n / 2, n - 1}) {
        // ---- nearly singular ------------------------------------------------
        {
            auto p = make_dominant_permuted<T>(n, batch, 4747u + unsigned(kz));
            this->run_blocked(p);
            if (this->HasFailure()) return;
            for (int b = 0; b < batch; ++b) {
                T& d = p.buf[size_t(b) * p.stride + size_t(kz) * p.ld + kz];
                d = scale(d, tiny);
            }
            auto A = view_of(p);
            for (Transpose op : {Transpose::NoTrans, Transpose::Trans}) {
                auto rhs = make_rhs<T>(n, nrhs, batch, 4848u + unsigned(kz) + unsigned(int(op)));
                auto Bv = view_of(rhs);
                UnifiedVector<std::byte> ws(std::max<std::size_t>(
                    1, sycl_getrs::getrs_fused_buffer_size<T>(*this->ctx, A, Bv, op)));
                ASSERT_NO_THROW(sycl_getrs::getrs_fused_dispatch<T>(
                    *this->ctx, A, Bv, op, p.piv.to_span(), ws.to_span()));
                this->ctx->wait();
                for (int b = 0; b < batch; ++b) {
                    for (int c = 0; c < nrhs; ++c)
                        for (int i = 0; i < n; ++i)
                            ASSERT_TRUE(hfinite(up(
                                rhs.buf[size_t(b) * rhs.stride + size_t(c) * rhs.ld + i])))
                                << "a NEARLY singular factor produced a non-finite answer at kz="
                                << kz << " transA=" << int(op) << " b=" << b;
                    const double res = fused_factor_residual<T>(
                        p.buf.data() + size_t(b) * p.stride,
                        reinterpret_cast<const int*>(p.piv.data()) + size_t(b) * n,
                        rhs.buf.data() + size_t(b) * rhs.stride,
                        rhs.b0.data() + size_t(b) * rhs.stride, n, nrhs, p.ld, rhs.ld, op);
                    EXPECT_LE(res, solve_tol<T>(n))
                        << "a NEARLY singular factor was not solved to a backward-error bound "
                           "at kz=" << kz << " transA=" << int(op) << " b=" << b
                        << " -- an epsilon floor or a skipped division would look like this";
                }
                if (this->HasFailure()) return;
            }
        }
        // ---- exactly singular -----------------------------------------------
        {
            auto p = make_dominant_permuted<T>(n, batch, 4949u + unsigned(kz));
            this->run_blocked(p);
            if (this->HasFailure()) return;
            for (int b = 0; b < batch; ++b)
                p.buf[size_t(b) * p.stride + size_t(kz) * p.ld + kz] = mk<T>(0.0, 0.0);
            auto A = view_of(p);
            for (Transpose op : {Transpose::NoTrans, Transpose::Trans}) {
                auto rhs = make_rhs<T>(n, nrhs, batch, 5050u + unsigned(kz) + unsigned(int(op)));
                auto Bv = view_of(rhs);
                UnifiedVector<std::byte> ws(std::max<std::size_t>(
                    1, sycl_getrs::getrs_fused_buffer_size<T>(*this->ctx, A, Bv, op)));
                ASSERT_NO_THROW(sycl_getrs::getrs_fused_dispatch<T>(
                    *this->ctx, A, Bv, op, p.piv.to_span(), ws.to_span()))
                    << "an exactly singular factor must not make the launch throw or hang";
                this->ctx->wait();
                for (int b = 0; b < batch; ++b) {
                    bool nonfinite = false;
                    for (int c = 0; c < nrhs && !nonfinite; ++c)
                        for (int i = 0; i < n && !nonfinite; ++i)
                            if (!hfinite(up(rhs.buf[size_t(b) * rhs.stride +
                                                    size_t(c) * rhs.ld + i])))
                                nonfinite = true;
                    EXPECT_TRUE(nonfinite)
                        << "a factor with U(" << kz << "," << kz << ") == 0 produced an entirely "
                           "FINITE answer at transA=" << int(op) << " b=" << b
                        << " -- the division by the zero pivot was floored or skipped, which is "
                           "the silently-plausible-wrong-answer failure mode";
                }
                check_rhs_pad_intact(rhs, "fused/singular");
                if (this->HasFailure()) return;
            }
        }
    }
}

// F6. THE FACADE REACHES THE FUSED KERNEL, ASSERTED BIT-EXACTLY, AND THE
// VENDOR-FREE DEFAULT LANDS ON IT. The comparison is BIT-EXACT against the direct
// entry point, which no vendor and no other native tier can reproduce; the route
// half also pins that a pinned `blocked` still reaches the COMPOSED tier.
// evidence: docs/perf/lu.md#getrs-fused-window-evidence
TYPED_TEST(LuTest, FacadeReachesTheFusedGetrsBitExactly) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 64, nrhs = 3, batch = 3;

    auto p = make_dominant_permuted<T>(n, batch, 6161u);
    this->run_blocked(p);
    check_factor(p, "facade/fused/factor");
    if (this->HasFailure()) return;
    auto A = view_of(p);

    for (Transpose op : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans}) {
        EnvGuard g("BATCHLAS_GETRS_ROUTE", "cta");
        auto r1 = make_rhs<T>(n, nrhs, batch, 7171u);
        auto r2 = make_rhs<T>(n, nrhs, batch, 7171u);
        auto V1 = view_of(r1);
        auto V2 = view_of(r2);

        // THE PIN IS VERIFIED, NEVER ASSUMED.
        const auto route = backend::getrs_route<B, T>(
            *this->ctx, A, V2, op, dispatch::factorization_vendor_available<B>);
        ASSERT_TRUE(dispatch::is_native(route)) << "the 'cta' getrs pin did not take";
        ASSERT_EQ(route.algo, dispatch::Algorithm::CTA)
            << "the 'cta' getrs pin resolved to the other native tier";

        UnifiedVector<std::byte> w1(std::max<std::size_t>(
            1, sycl_getrs::getrs_fused_buffer_size<T>(*this->ctx, A, V1, op)));
        sycl_getrs::getrs_fused_dispatch<T>(*this->ctx, A, V1, op, p.piv.to_span(), w1.to_span());
        this->ctx->wait();

        UnifiedVector<std::byte> w2(std::max<std::size_t>(
            1, getrs_buffer_size<B, T>(*this->ctx, A, V2, op)));
        ASSERT_NO_THROW((getrs<B, T>(*this->ctx, A, V2, op, p.piv.to_span(), w2.to_span())));
        this->ctx->wait();

        for (size_t i = 0; i < r1.buf.size(); ++i)
            ASSERT_EQ(habs(up(r1.buf[i]) - up(r2.buf[i])), 0.0)
                << "transA=" << int(op) << ": the facade's getrs differs from the FUSED direct "
                   "entry point at element " << i << " -- something else served this call";
    }

    // The other pin must still reach the COMPOSED tier, not the fused route ahead of
    // it in kGetrsOrder.
    {
        EnvGuard g("BATCHLAS_GETRS_ROUTE", "blocked");
        auto rhs = make_rhs<T>(n, nrhs, batch, 7272u);
        auto Bv = view_of(rhs);
        const auto route = backend::getrs_route<B, T>(
            *this->ctx, A, Bv, Transpose::NoTrans, dispatch::factorization_vendor_available<B>);
        ASSERT_TRUE(dispatch::is_native(route));
        EXPECT_EQ(route.algo, dispatch::Algorithm::Blocked)
            << "the 'blocked' getrs pin resolved to the FUSED tier; the composed arm is now "
               "unreachable and every test that pins it measures something else";
    }

    // The AUTOMATIC route, on both sides of the fused tier's capability.
    {
        auto narrow = make_rhs<T>(n, 1, batch, 7373u);
        auto wide   = make_rhs<T>(n, int(sycl_getrs::kGetrsFusedMaxRhs) + 1, batch, 7474u);
        auto Vn = view_of(narrow);
        auto Vw = view_of(wide);
        const auto rn = backend::getrs_route<B, T>(*this->ctx, A, Vn, Transpose::NoTrans, false);
        const auto rw = backend::getrs_route<B, T>(*this->ctx, A, Vw, Transpose::NoTrans, false);
        EXPECT_TRUE(dispatch::is_native(rn));
        EXPECT_EQ(rn.algo, dispatch::Algorithm::CTA)
            << "a vendor-free build did not take the fused tier at nrhs=1, which is the entire "
               "point of native_tier_preferred";
        EXPECT_TRUE(dispatch::is_native(rw));
        EXPECT_EQ(rw.algo, dispatch::Algorithm::Blocked)
            << "a vendor-free build routed nrhs=" << (sycl_getrs::kGetrsFusedMaxRhs + 1)
            << " to the fused tier, which is not instantiated that wide";

        // THE VENDOR-PRESENT ROUTE, BOTH sides of the window: an assertion that only pins
        // the inside cannot fail when someone widens the clause.
        // evidence: docs/perf/lu.md#getrs-fused-window-evidence
        if constexpr (dispatch::factorization_vendor_available<B>) {
            const auto rv = backend::getrs_route<B, T>(
                *this->ctx, A, Vn, Transpose::NoTrans, /*vendor_available=*/true);
            EXPECT_TRUE(dispatch::is_native(rv) && rv.algo == dispatch::Algorithm::CTA)
                << "nrhs=1 is INSIDE the measured window (geomean 2.26x over 111 cells, "
                   "min 1.24x, flat across every batch ladder at seven orders) and must "
                   "route to the fused tier even with cuBLAS present";

            // OUTSIDE the window, the vendor still takes it: nrhs = 8 is inside the tier's
            // CAPABILITY and outside its measured WINDOW, a pair supports() must not merge.
            auto w8 = make_rhs<T>(n, int(sycl_getrs::kGetrsFusedMaxRhs), batch, 7575u);
            auto V8 = view_of(w8);
            EXPECT_TRUE(dispatch::is_vendor(backend::getrs_route<B, T>(
                *this->ctx, A, V8, Transpose::NoTrans, /*vendor_available=*/true)))
                << "nrhs=" << sycl_getrs::kGetrsFusedMaxRhs << " is OUTSIDE the window "
                   "(geomean 0.819x over 24 cells, 13 losses) and must take the vendor";

            // ... and clause B is FLOAT ONLY. nrhs = 4 splits by type, the half of the window
            // most likely to be widened by someone who drops the type test.
            auto w4 = make_rhs<T>(n, 4, batch, 7676u);
            auto V4 = view_of(w4);
            const auto r4 = backend::getrs_route<B, T>(
                *this->ctx, A, V4, Transpose::NoTrans, /*vendor_available=*/true);
            if constexpr (std::is_same_v<T, float>) {
                EXPECT_TRUE(dispatch::is_native(r4) && r4.algo == dispatch::Algorithm::CTA)
                    << "float nrhs=4 is clause B: full batch ladders at seven orders, "
                       "every one a flat win, min 1.13x";
            } else {
                EXPECT_TRUE(dispatch::is_vendor(r4))
                    << "only FLOAT is in the window at nrhs=4; double dips to 0.940x at "
                       "n=128 batch 2048, cfloat to 0.976x at n=1024 batch 16, cdouble to "
                       "0.577x at n=32 -- and a mid-ladder dip cannot be gated away";
            }
        }
    }
}

// F7. THE FUSED DIRECT ENTRY POINT REFUSES WHAT supports() REFUSES, AND ITS
// WORKSPACE QUERY DEREFERENCES NOTHING. The workspace is ZERO in every mode for
// this tier by design (the RHS is permuted and solved in LOCAL memory, in place),
// and the facade's figure is a max over BOTH native tiers; the query and the call
// resolve INDEPENDENTLY, so a lease sized for one tier is one the other overruns.
TYPED_TEST(LuTest, FusedGetrsDirectEntryPointRefusesWhatSupportsRefuses) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 24, nrhs = 2, batch = 2;
    auto p = make_dominant_permuted<T>(n, batch, 8181u);
    this->run_blocked(p);
    auto A = view_of(p);
    auto rhs = make_rhs<T>(n, nrhs, batch, 8282u);
    auto Bv = view_of(rhs);
    UnifiedVector<std::byte> ws(4096);

    // A non-square A.
    {
        UnifiedVector<T> w(size_t(24) * 32, mk<T>(1.0, 0.0));
        UnifiedVector<T*> wp(1, nullptr);
        MatrixView<T, MatrixFormat::Dense> W(w.data(), 24, 32, 24, 24 * 32, 1, wp.data());
        EXPECT_THROW(sycl_getrs::getrs_fused_dispatch<T>(*this->ctx, W, Bv, Transpose::NoTrans,
                                                         p.piv.to_span(), ws.to_span()),
                     std::invalid_argument);
    }
    // B with the wrong number of rows.
    {
        auto mismatched = make_rhs<T>(n + 1, nrhs, batch, 8383u);
        auto Bm = view_of(mismatched);
        EXPECT_THROW(sycl_getrs::getrs_fused_dispatch<T>(*this->ctx, A, Bm, Transpose::NoTrans,
                                                         p.piv.to_span(), ws.to_span()),
                     std::invalid_argument);
    }
    // A and B disagreeing on the batch size.
    {
        auto other = make_rhs<T>(n, nrhs, batch + 1, 8484u);
        auto Bo = view_of(other);
        EXPECT_THROW(sycl_getrs::getrs_fused_dispatch<T>(*this->ctx, A, Bo, Transpose::NoTrans,
                                                         p.piv.to_span(), ws.to_span()),
                     std::invalid_argument);
    }
    // A pivot span shorter than n * batch.
    {
        UnifiedVector<int64_t> shortpiv(size_t(n) * batch - 1, 0);
        EXPECT_THROW(sycl_getrs::getrs_fused_dispatch<T>(*this->ctx, A, Bv, Transpose::NoTrans,
                                                         shortpiv.to_span(), ws.to_span()),
                     std::invalid_argument);
    }

    // ---- the workspace query ----------------------------------------------
    {
        // NULL data, exactly as a measuring pass presents it.
        MatrixView<T, MatrixFormat::Dense> nullA(nullptr, n, n, n + 4, (n + 4) * n + 3, batch,
                                                 nullptr);
        MatrixView<T, MatrixFormat::Dense> nullB(nullptr, n, nrhs, n + 2,
                                                 (n + 2) * nrhs + 5, batch, nullptr);
        for (Transpose op : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans}) {
            EXPECT_NO_THROW((void)sycl_getrs::getrs_fused_buffer_size<T>(*this->ctx, nullA,
                                                                        nullB, op));
            EXPECT_EQ(sycl_getrs::getrs_fused_buffer_size<T>(*this->ctx, nullA, nullB, op),
                      std::size_t(0))
                << "the fused tier claims a workspace; it has none, and the facade's max() over "
                   "the two native tiers is what would have to carry it";
            EXPECT_NO_THROW(((void)getrs_buffer_size<B, T>(*this->ctx, nullA, nullB, op)));
        }
    }
    // Serve EXACTLY the facade's figure under the CTA pin. A short workspace is a
    // silent heap overflow, not a throw.
    {
        EnvGuard g("BATCHLAS_GETRS_ROUTE", "cta");
        const auto route = backend::getrs_route<B, T>(
            *this->ctx, A, Bv, Transpose::NoTrans, dispatch::factorization_vendor_available<B>);
        ASSERT_TRUE(dispatch::is_native(route) && route.algo == dispatch::Algorithm::CTA)
            << "the 'cta' pin did not take, so this sizing check measures another route";
        const std::size_t need = getrs_buffer_size<B, T>(*this->ctx, A, Bv, Transpose::NoTrans);
        EXPECT_GE(need, sycl_getrs::getrs_fused_buffer_size<T>(*this->ctx, A, Bv,
                                                               Transpose::NoTrans));
        UnifiedVector<std::byte> exact(std::max<std::size_t>(1, need));
        ASSERT_NO_THROW((getrs<B, T>(*this->ctx, A, Bv, Transpose::NoTrans, p.piv.to_span(),
                                     exact.to_span())));
        this->ctx->wait();
        for (int b = 0; b < batch; ++b)
            EXPECT_LE(solve_residual<T>(p.a0.data() + size_t(b) * p.stride,
                                        rhs.buf.data() + size_t(b) * rhs.stride,
                                        rhs.b0.data() + size_t(b) * rhs.stride,
                                        n, nrhs, p.ld, rhs.ld, Transpose::NoTrans),
                      solve_tol<T>(n));
    }
}

// The break record for every guarded property, including the breaks that turned
// nothing red: docs/perf/lu.md#blind-guards-and-what-made-them-blind

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
