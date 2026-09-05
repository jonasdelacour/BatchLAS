// Native batched LU -- GETRF, GETRS and GETRI. The WP6 tests.
//
// ---------------------------------------------------------------------------
// THE ORACLE IS NEVER THE VENDOR, AND THE ROUTE IS NEVER TRUSTED
// ---------------------------------------------------------------------------
// The first two rules are inherited verbatim from tests/potrf_tests.cc:1-25 and
// tests/geqrf_tests.cc:1-31. The third is specific to LU and was MEASURED during
// WP6.
//
//   * A VENDOR REFERENCE IS INERT IN A VENDOR-FREE BUILD. resolve_route falls
//     back to a supported NATIVE route when no vendor exists -- which is the code
//     under test -- so a "compare against cuBLAS" test compares the kernel with
//     itself in exactly the build this campaign exists for.
//
//   * A FORCED ROUTE THAT supports() REJECTS SILENTLY BECOMES THE VENDOR.
//     route_resolve.hh:165 tests `if (Table::supports(forced, s)) return forced;`
//     and falls through to automatic() at :175, so a test that sets
//     BATCHLAS_GETRF_ROUTE=cta and gets one gate wrong runs cuBLAS and passes
//     GREEN over a kernel nothing executed.
//
//   * THE VENDOR IS NOT A VALID PIVOT ORACLE EVEN WHERE IT EXISTS.
//     cublas{C,Z}getrfBatched selects its pivot on the MODULUS; LAPACK, netlib
//     and this kernel select on cabs1 = |Re| + |Im| (WP6's measured finding,
//     experiments/wp6_lu/kernels/README.md section 3). On a matrix built to
//     separate the two rules the vendor and the host DISAGREE, for both complex
//     types. An elementwise native-vs-vendor pivot comparison is therefore a
//     WRONG test, and PivotSelectionUsesCabs1AndNotTheModulus below is what pins
//     the rule this library actually implements.
//
// So every numerical test in this file calls sycl_getrf::getrf_cta_dispatch<T> /
// getrf_blocked_dispatch<T> / sycl_getrs::getrs_blocked_dispatch<T> /
// sycl_getri::getri_blocked_dispatch<T> DIRECTLY -- calls no vendor can serve --
// and checks a HOST reference built here from the input this file generated.
//
// ---------------------------------------------------------------------------
// THE FOUR HOST ORACLES, AND WHY A RESIDUAL ALONE IS NOT ENOUGH
// ---------------------------------------------------------------------------
// 1. ||P A - L U||_F / ||A||_F, with P RECONSTRUCTED ON THE HOST from the
//    returned pivot array in the base and direction the contract claims
//    (1-based, an INTERCHANGE LIST applied FORWARDS: for k = 0..n-1 swap rows k
//    and ipiv[k]-1). Getting that reconstruction right IS the pivot-contract
//    test: a 0-based array, a permutation vector, or a backwards walk each make
//    this residual O(1).
//
// 2. THE PIVOT SEQUENCE ELEMENTWISE, against a sequence known EXACTLY and
//    without arithmetic. A random matrix cannot support that check across the
//    blocked tier, because a blocked LU rounds its trailing update in a different
//    order from any host getf2 and a near-tie can legitimately flip. So the
//    pivot-equality tests run on a STRICTLY COLUMN-DIAGONALLY-DOMINANT matrix
//    that has then been ROW-PERMUTED: dominance is preserved by elimination, so
//    at step k the winner is the row carrying the dominant entry, ahead by a
//    factor of order 4n, and the expected interchange list follows from integer
//    bookkeeping alone. That sequence is stable under ANY rounding and identical
//    for both tiers and all four scalar types.
//
// 3. |L(i,j)| <= 1. The defining property of PARTIAL pivoting and the one thing
//    a residual cannot see: an unpivoted (Doolittle) factorization satisfies
//    ||PA - LU|| perfectly with ipiv = identity, and WP6's baseline measured that
//    on the obvious diagonally dominant matrix DROPPING THE INTERCHANGE ENTIRELY
//    leaves the residual bit-identical. This one runs on the RANDOM matrix, where
//    it has something to say.
//
// 4. THE SOLVE AND THE INVERSE against the ORIGINAL A: ||op(A) X - B|| and
//    ||A C - I||, in double regardless of T.
//
// ---------------------------------------------------------------------------
// EVERY TEST IN THIS FILE WAS BROKEN ON PURPOSE, AND THE RESULTS ARE RECORDED
// AT THE BOTTOM OF THIS FILE -- INCLUDING THE BREAKS THAT TURNED NOTHING RED.
// ---------------------------------------------------------------------------
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
// cabs1, LAPACK's |Re| + |Im|. THE metric ?GETRF's I?AMAX pivots on, and NOT the
// modulus -- see the note at the top of this file.
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

// Scale a scalar of either kind by a real factor, without naming .real()/.imag()
// on a type that has neither.
template <class T> inline T scale(T v, double f) { return T(v * static_cast<RealOf<T>>(f)); }
template <class R> inline std::complex<R> scale(std::complex<R> v, double f) {
    return std::complex<R>(v.real() * static_cast<R>(f), v.imag() * static_cast<R>(f));
}

template <typename T>
constexpr double eps_of() {
    if constexpr (std::is_same_v<RealOf<T>, float>) return 1.1920929e-7;
    else return 2.220446049250313e-16;
}

// Backward-error bounds. LU with partial pivoting is backward stable, so the
// factorization and solve bounds scale with n * eps and NOT with conditioning.
// The inverse residual DOES carry cond(A), which is why every getri and getrs
// test runs on the dominant matrix, whose condition number is O(1).
template <typename T> double lu_tol(int n)    { return 200.0 * double(n) * eps_of<T>(); }
template <typename T> double solve_tol(int n) { return 400.0 * double(n) * eps_of<T>(); }
template <typename T> double inv_tol(int n)   { return 800.0 * double(n) * eps_of<T>(); }

bool verbose() {
    static const bool v = (std::getenv("BATCHLAS_TEST_VERBOSE") != nullptr);
    return v;
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

template <typename T, Backend B>
struct LuConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

// ---------------------------------------------------------------------------
// A batch of DISTINCT square matrices in a buffer with a PADDED ld and a stride
// that is NOT ld*cols, with the pad POISONED.
//
// None of that is decoration. trsm_native.cc:590-599 records the failure it
// guards: the 6-arg MatrixView constructor DEFAULTS stride to ld*cols when 0 is
// passed, after which every batch item but the first reads the wrong matrix;
// there is a standing memory entry for the GEMM twin ("Native GEMM collapses on
// strided ld"). Every shape test in this file therefore runs at ld != n and
// stride != ld*n, so the two most consequential lines in each launcher are
// falsifiable BY DEFAULT rather than in one dedicated test.
//
// THE POINTER ARRAY IS NOT OPTIONAL EITHER. getrf_blocked.cc:152-165 records the
// measurement: a MatrixView built by the 6-arg constructor has an EMPTY
// data_ptrs_ span, and every vendor batched call dereferences it and throws
// "data_ptrs target is null". The drop-in tests below cross to the vendor, and
// the blocked driver's own routed gemm reaches the vendor for float and double.
// ---------------------------------------------------------------------------
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

// A RANDOM matrix. Well-scaled but structureless, so the pivot sequence is
// data-dependent and only the residual and |L| <= 1 oracles apply.
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

// ---------------------------------------------------------------------------
// THE MATRIX WITH A KNOWN-EXACT PIVOT SEQUENCE.
//
// Build a STRICTLY COLUMN-DIAGONALLY-DOMINANT B (|B(k,k)| = 4n, |B(i,k)| <= 1)
// and then move B's row r to position sigma(r). Two facts make the interchange
// list exact and rounding-independent:
//
//   * column diagonal dominance is PRESERVED by Gaussian elimination, so at step
//     k the largest cabs1 in column k over rows k..n-1 is the one carrying B's
//     row k, ahead of every other by a factor of order 4n/3 -- far outside any
//     rounding of any of the four scalar types;
//   * therefore the pivot at step k is simply "wherever B's row k currently is",
//     which is pure integer bookkeeping.
//
// It is also the matrix every getrs/getri residual runs on, because its
// condition number is O(1) and ||A C - I|| carries cond(A) where ||PA - LU||
// does not.
// ---------------------------------------------------------------------------
template <typename T>
Lu<T> make_dominant_permuted(int n, int batch, unsigned seed,
                             int ld_pad = 5, int stride_pad = 11) {
    Lu<T> p;
    alloc(p, n, batch, ld_pad, stride_pad);
    Rng rg(seed);

    // sigma: B's row r ends up at position sigma[r]. A CYCLIC SHIFT, and the
    // choice is load-bearing rather than arbitrary.
    //
    // THIS WAS A REVERSAL, AND THE REVERSAL WAS A BLIND GUARD. A reversal is its
    // OWN INVERSE, so the permutation the interchange list composes to satisfies
    // F = F^{-1} -- and every test of a DIRECTION (getrs's transposed walk,
    // getri's backward trace) is then satisfied by the wrong direction too.
    // Measured: break `getrs_forward`, which walks the transposed permutation
    // forwards instead of backwards, turned NOTHING red on the reversal (62
    // passed, 0 failed) and turns the transposed getrs red on this shift.
    //
    // A cyclic shift composes to an n-cycle, which is self-inverse only for
    // n <= 2. interchange_is_involution() below asserts that at every use.
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

// The PACKED int32 view of the public int64 pivot span, for ONE batch item.
// This spelling -- not a widening read of the int64 values -- IS the pivot
// contract on CUDA and ROCm: cublas.cc:1508 and rocsolver.cc:227 both do
// pivots.as_span<int>(), and the native kernels write the same layout.
template <typename T>
const int* piv_item(const Lu<T>& p, int b) {
    return reinterpret_cast<const int*>(p.piv.data()) + size_t(b) * p.n;
}

// ---------------------------------------------------------------------------
// ORACLE 1: ||P A - L U||_F / ||A||_F, rectangular-capable (m >= n).
//
// P is reconstructed HERE, on the host, in the base and direction the contract
// claims: a 1-based INTERCHANGE LIST applied FORWARDS.
// ---------------------------------------------------------------------------
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
            // (L U)(i,j) = sum_t L(i,t) U(t,j) with L unit-lower and U upper, so
            // t runs to min(i, j, k-1) and the t == i term carries L = 1.
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

// ORACLE 3: THE PARTIAL-PIVOTING PROPERTY ITSELF, IN THE METRIC THE LIBRARY
// PIVOTS ON, recovered from the factor alone.
//
// The naive form of this oracle is "max |L(i,j)| <= 1", which is what a real
// ?GETRF guarantees. IT IS WRONG FOR COMPLEX: LAPACK selects on cabs1 =
// |Re| + |Im|, and cabs1(z) <= sqrt(2)|z|, so a perfectly correct zgetrf can and
// does return |L| up to sqrt(2) (measured here at 1.051 on the first random
// cfloat matrix this file generated). A test written to the real-only bound is
// red on half the type list for no defect at all.
//
// The exact statement is recoverable instead. L(i,k) = a_ik / a_kk with a_kk the
// CHOSEN PIVOT, and a_kk survives in the factor as U(k,k) -- so the selection
// rule "cabs1(a_ik) <= cabs1(a_kk) for every i > k" is
//
//     cabs1( L(i,k) * U(k,k) ) <= cabs1( U(k,k) )
//
// which is checkable elementwise, is uniform over the four scalar types, and --
// unlike the |L| <= 1 form -- IS SENSITIVE TO THE METRIC: a kernel that pivoted
// on the modulus satisfies |a_ik| <= |a_kk| but can violate the line above by up
// to sqrt(2), and does so on ordinary random complex data.
//
// Returns the worst ratio; 1 is the bound, an unpivoted factorization is
// unbounded.
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

// ---------------------------------------------------------------------------
// THE FIXTURE.
// ---------------------------------------------------------------------------
template <typename Config>
class LuTest : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    static constexpr Backend BackendType = Config::BackendVal;

    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        if (this->HasFatalFailure() || ::testing::Test::IsSkipped()) return;
        if (!this->ctx) GTEST_SKIP() << "no queue";
        // supports()' own correctness gates, not a convenience: route_getrf.hh
        // gate 2 (GPU only) and gate 3 (sub-group 32) reject everything else, and
        // every direct entry point re-applies them.
        if (this->ctx->device().type != DeviceType::GPU)
            GTEST_SKIP() << "the native LU kernels are GPU-only (route_getrf.hh gate 2)";
        if (!this->ctx->device().supports_sub_group_size(32))
            GTEST_SKIP() << "device does not offer sub-group size 32 (route_getrf.hh gate 3)";
    }

    // The DEVICE's local-memory budget, spelled exactly as
    // src/backends/getrf_route.hh spells it. NOT device_limits.hh's 49152, which
    // cmake/BatchLASDetectSYCL.cmake:44-45 hardcodes for any nvidia_gpu_sm_*
    // pattern and which is 2.06x wrong on this box (WP4's finding W1).
    std::size_t budget() const {
        const std::size_t lm = static_cast<std::size_t>(
            this->ctx->device().get_property(DeviceProperty::LOCAL_MEM_SIZE));
        return lm > 4096 ? lm - 4096 : std::size_t(0);
    }
    int cta_max_n() const { return sycl_getrf::getrf_cta_max_n_for_slm<T>(budget()); }
    bool leaf_fits(int m, int n) const { return sycl_getrf::getrf_leaf_fits<T>(m, n, budget()); }

    // The blocked driver's OWN block width and OWN leading-panel leaf choice.
    // QUERIED, never hardcoded: potrf_native.hh:246-266 records why a test that
    // must straddle a block boundary and cannot see where the boundary is stops
    // testing anything the moment the width moves.
    int nb(int n) const {
        return int(sycl_getrf::getrf_blocked_debug_params<T>(*this->ctx, n) & 0xffffu);
    }
    unsigned leaf(int n) const {
        return sycl_getrf::getrf_blocked_debug_params<T>(*this->ctx, n) >> 16;
    }

    // The ROUTED gemm and trsm, exactly as src/dispatch/entry_points/
    // factorization.cc injects them. A direct caller MUST inject them itself --
    // the blocked driver throws on an empty trsm seam rather than reaching for a
    // native kernel, which is WP3 step 16's defect refused by construction.
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

// ---------------------------------------------------------------------------
// EVERY BATCH ITEM IS CHECKED, NOT ITEM 0. Item 0 sits at offset 0, so a wrong
// batch stride cannot move it and a suite that checks only item 0 is blind to
// the whole class. The distinctness assertion is what makes "the kernel
// broadcast item 0 over the batch" a failure rather than a pass.
// ---------------------------------------------------------------------------
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

// ANTI-VACUITY FOR EVERY TEST OF A PERMUTATION *DIRECTION*.
//
// Compose the interchange list into the permutation it denotes and ask whether
// that permutation is SELF-INVERSE. If it is, a backwards walk and a forwards
// walk produce the same answer and no residual can tell them apart -- which is
// exactly what this file's first test matrix did, and what the `getrs_forward`
// break exposed by turning nothing red.
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
// every step and every pivot assertion in this file would be vacuous -- a
// recorded WP6 baseline finding.
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

using LuTestTypes = typename test_utils::backend_types<LuConfig>::type;

}  // namespace

TYPED_TEST_SUITE(LuTest, LuTestTypes);

// ===========================================================================
// L0. THE 48 KB LAUNCH HOLE. DECLARED FIRST ON PURPOSE.
//
// A resident-leaf launch that asks for EXACTLY 49,152 B of local memory is
// refused by the CUDA backend -- too big for the non-opt-in 48 KB limit once the
// kernel's static shared is added, not big enough for the UR adapter to raise
// MaxDynamicSharedMemorySize. WP4 found the band and padded potrf over it
// (potrf_cta.cc:258-296); WP5 walked into it anyway; WP6 re-measured it from
// scratch with a PAD= knob holding kernel, shape and work-group fixed:
//     49,024 B PASS   49,152 B FAIL   49,280 B PASS
// 5/5 deterministic across five processes and independent of work-group width,
// with 48,896 B additionally failing for double and cdouble. A BYTE threshold,
// not a shape and not a type.
//
// WHY THIS TEST IS DECLARED FIRST, AND WHY THAT IS LOAD-BEARING RATHER THAN
// TIDINESS. The attribute the adapter sets is STICKY PER CUfunction, and one
// instantiation of GetrfPanelResidentKernel<T> serves every panel shape of a
// type. Any earlier launch of a LARGER panel raises the cap for the rest of the
// process and this test can never fail again. GoogleTest runs a suite's tests in
// declaration order, so being first in this file is what keeps the guard cold.
// DO NOT MOVE IT, and do not add a resident-leaf launch above it.
//
// The cold check by hand:
//     ./build/tests/getrf_tests --gtest_filter='*LaunchHole*'
//
// MEASURED RESULT OF THE `hole_pad` BREAK ON THIS BOX, recorded because it
// changes what this test is worth. Removing the pad turns layer (a) RED for
// every in-band row and every scalar type -- and leaves layer (b) GREEN: the
// 49,152 B resident launch SUCCEEDS here without the pad. That is consistent
// with getrf_cta.cc:124-129's own reading (WP6 attributed the hole to
// sycl::reduce_over_group alone, and this kernel uses no group collective, only
// permute_group_by_xor). So on this device the pad is DEFENSIVE and the
// arithmetic assertion is the one carrying the guard. Layer (b) is kept because
// it is the layer that would fire if a group algorithm were ever added to the
// body -- which is precisely the condition WP4 wrote down and WP5 walked into.
//
// WHAT IT ASSERTS, IN TWO INDEPENDENT LAYERS:
//   (a) the PAD ARITHMETIC, through the library's own getrf_leaf_fits: a shape
//       whose raw footprint lands in (47104, 49664] must need a budget of
//       49,920 B, not its raw size. This is a pure-function assertion and does
//       not depend on the device refusing anything.
//   (b) the LAUNCH ITSELF, through getrf_panel_factorize, with the resident leaf
//       ASSERTED (used_resident_out) so a silent fall-through to the global leaf
//       cannot pass, plus the full host factorization oracle on the result.
// ===========================================================================
TYPED_TEST(LuTest, ResidentLeafLaunchHoleAt48KiB) {
    using T = typename TestFixture::T;

    // getrf_cta.cc's getrf_scratch_bytes: kLuRedSlots (32) argmax slots, each a
    // real plus an int. Restated here and then PINNED against the library's own
    // predicate below rather than trusted.
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

    // ANTI-VACUITY 1: the byte formula above must be the library's. Checked at a
    // shape far BELOW the hole band, where no pad can apply, so the answer is the
    // raw footprint itself.
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

    // ANTI-VACUITY 2: the ONE byte count measured to fail for every scalar type
    // must be represented. Without it this test is a ladder that steps over the
    // hole -- which is exactly how an n-ladder misses the defect entirely.
    bool has_49152 = false;
    for (const Row& r : rows) if (r.bytes == 49152) has_49152 = true;
    ASSERT_TRUE(has_49152) << "no (m, n) with a 49,152 B footprint was constructible for this "
                              "scalar type; the discriminating row is missing";

    for (const Row& r : rows) {
        ASSERT_EQ(raw_bytes(r.m, r.n), r.bytes) << "row does not ask for " << r.bytes << " B";
        const bool in_band = (r.bytes > kLo && r.bytes <= kHi);

        // (a) THE PAD ARITHMETIC. EXPECT and not ASSERT, deliberately: ASSERT
        // returns from the whole test, so a failure here would MASK (b) and the
        // break record could not say which of the two halves is carrying the
        // guard. They are independent claims and both must be reachable.
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

// ===========================================================================
// L1. THE CTA TIER: the residual, the partial-pivoting property, and the
// EXACT interchange list.
// ===========================================================================
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

// ===========================================================================
// L2. THE BLOCKED DRIVER, same two oracles, over orders that straddle its own
// block width in both directions.
// ===========================================================================
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

// ===========================================================================
// L3. THE BLOCK BOUNDARY IS QUERIED, NOT ASSUMED.
//
// potrf_native.hh:246-266 records the failure this guards: a test that must
// straddle a block boundary and cannot see where the boundary is keeps passing
// after the width moves while silently no longer testing a short final panel.
// This family has produced exactly that failure before (the sy2sb stage-1
// short-final-panel bug: wrong numbers, green suite).
//
// It also pins the debug query itself to the predicate the driver uses, so the
// query cannot report a leaf choice the call does not make.
// ===========================================================================
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

    // The leaf choice the query reports must be the one getrf_leaf_fits makes for
    // the leading panel -- the potrf_cta_launch_params discipline.
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

// ===========================================================================
// L4. BOTH PANEL RESIDENCIES FACTORISE CORRECTLY.
//
// getrf_panel_factorize is the ONE decision site between the local-memory leaf
// and the global-memory one, and a build in which only one of them is ever
// exercised is a build in which half the panel code is untested. Driving it
// directly with a TALL panel is what makes the global leaf reachable for every
// scalar type: through the blocked driver the crossover is at n ~ 760 for float,
// where the host oracle alone would cost seconds.
//
// The residency is ASSERTED from the launcher's own out-parameter, not inferred.
// ===========================================================================
TYPED_TEST(LuTest, BothPanelLeavesFactoriseCorrectly) {
    using T = typename TestFixture::T;
    const int nbw = this->nb(4096);
    ASSERT_GE(nbw, 1);

    // A panel that fits, and one that provably cannot: grow m until the predicate
    // says no, rather than picking a number that stops being large enough when a
    // constant moves.
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

// ===========================================================================
// L5. A SINGULAR MATRIX: `info` is EXACT-ZERO, 1-BASED, GLOBAL, per item, and
// FIRST-FAILURE-WINS -- with the other batch items unaffected and the failed
// item still FINITE.
//
// The failure is planted as an EXACTLY ZERO COLUMN, which is the only
// construction whose zero survives every rounding: column c is zero on input, so
// U(t,c) = 0 for every t and the column is never updated away from zero. The
// pivot search over it returns cabs1 = 0 at the lowest index, i.e. ipiv[c] = c+1
// (I?AMAX's convention), and ?GETF2 records the failure and SKIPS the reciprocal
// scale -- which is what keeps the item finite instead of Inf/NaN.
//
// THE COLUMNS ARE CHOSEN INSIDE THE SECOND AND THIRD PANELS on purpose. A
// block-local info offset reports the panel-relative column and passes every
// single-panel test; only a failure planted beyond the first panel can see it.
// ===========================================================================
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
        // The healthy items must still be a correct factorization -- a failure in
        // one item must not corrupt the others.
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

// ===========================================================================
// L5b. THE `info` ZERO PRE-PASS IS ORDERED AHEAD OF THE PANEL THAT READS IT, ON
// AN OUT-OF-ORDER QUEUE.
//
// It is the ONLY test in this file that does not run on the fixture's queue, and
// that is the whole point: every other case here -- and every call anywhere in
// src/ and tests/ -- uses the DEFAULT in-order queue
// (sycl-device-queue.hh:254), which orders the fill ahead of the kernel for
// free. Out-of-order queues are nonetheless public API
// (sycl-device-queue.hh:258, `Queue(const Queue& base, bool in_order)`), and on
// one of them the fill and the first panel were concurrent in both native tiers.
//
// WHY IT IS A WRONG ANSWER AND NOT A TIDINESS QUESTION: getf2_panel_device READS
// info[b] (getrf_cta_device.hh) to implement first-failure-wins across the
// blocked driver's panels, so the fill is a true read-after-write dependence
// rather than a pure output. Unordered, the panel loads the CALLER's pre-call
// garbage, concludes an earlier panel already failed, never records the real
// failure, and writes the garbage back. Measured before the fix: 6,979 items of
// 1,638,400 came back holding the caller's own -12345 sentinel and NONE reported
// the real singular column; the blocked tier, 3,743 of 983,040. A caller testing
// `info[b] != 0` therefore saw a FALSE singularity, and a caller whose garbage
// happened to be 0 would have seen a masked one.
//
// THE BATCH IS LARGE ON PURPOSE. This is a race, so the test can only be as
// falsifiable as the window it opens; the counts above are the calibration. The
// assertion is on EVERY item of EVERY repetition, so a single unordered read
// anywhere in the sweep fails it.
// ===========================================================================
TYPED_TEST(LuTest, InfoFillIsOrderedAheadOfThePanelOnAnOutOfOrderQueue) {
    using T = typename TestFixture::T;
    // ONE SCALAR TYPE, DELIBERATELY, AND IT IS NOT A SHORTCUT. What is under test
    // is a HOST-SIDE SUBMISSION ORDER -- whether the fill is ordered ahead of the
    // launch that reads its result -- which is identical for every scalar type and
    // every backend; the driver code is one line, shared. What DOES vary with the
    // type is the cost: the sweep below is 1.6M factorisations, and running it four
    // times over would add minutes to a suite that otherwise finishes in under a
    // second while re-testing the same line.
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

    // THE LOOP SHAPE IS PART OF THE CALIBRATION, not incidental. Only `info` is
    // re-seeded between repetitions: re-copying the 300 MB matrix from the host
    // instead -- the obvious thing to write -- touches managed memory hard enough
    // to serialise the queue, and MEASURED, it closes the window completely (the
    // first draft of this test did exactly that and stayed GREEN with both guards
    // deleted). The matrix is therefore staged once and re-factorised in place,
    // which is stable: the planted zero column survives the factorisation, so
    // every repetition must report the same info, and the in-order control below
    // asserts precisely that rather than assuming it.
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

        // THE IN-ORDER CONTROL, and it is what makes the sweep's oracle legitimate:
        // it establishes that repeated in-place factorisation reports zc+1 every
        // time, so a miss in the sweep is an ORDERING failure and not the oracle
        // drifting. 5 x 65536 items.
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

// ===========================================================================
// L6. A NEARLY singular matrix is NOT flagged. The `info` predicate is a TRUE
// BINARY ZERO, never a tolerance, and that is a PUBLIC CONTRACT shared with
// LAPACK and cuBLAS rather than an internal choice: an epsilon floor would make
// this library report a failure where both of the implementations a caller might
// swap in report success. WP6 declined to implement the floor the ground brief
// suggested for exactly this reason, and this is the test that pins the
// decision.
// ===========================================================================
TYPED_TEST(LuTest, NearlySingularIsNotFlagged) {
    using T = typename TestFixture::T;
    const int n = 48, c = 19;
    auto p = make_dominant_permuted<T>(n, 2, 4242u);
    // Scale one whole column to ~1e-30, which makes U(c,c) tiny but exactly
    // representable and NON-ZERO for all four scalar types. The pivot list is
    // unchanged (column scaling does not move an argmax), so the exact pivot
    // oracle still applies.
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
        // ANTI-VACUITY: the pivot really is tiny. Without this the test is
        // "info == 0 on an ordinary matrix", which every test above already says.
        ASSERT_GT(diag, 0.0) << "U(c,c) is exactly zero, so this is the SINGULAR case";
        ASSERT_LT(diag, 1e-20) << "U(c,c) = " << diag << " is not nearly singular at all";
        EXPECT_EQ(p.info[b], 0)
            << "info = " << p.info[b] << " at b=" << b << " with |U(c,c)| = " << diag
            << " -- a tolerance crept into the singularity predicate, which diverges from "
               "LAPACK and cuBLAS invisibly";
    }
    check_factor(p, "blocked/near-singular", /*check_L=*/true);
}

// ===========================================================================
// L7. THE PIVOT METRIC IS cabs1, NOT THE MODULUS.
//
// WP6's measured finding: cublas{C,Z}getrfBatched pivots on |z| while LAPACK,
// netlib and this kernel pivot on |Re| + |Im|. On the matrix below the two rules
// SELECT DIFFERENT ROWS, and substituting the modulus into the kernel reproduces
// cuBLAS's answer exactly.
//
// This test exists because the ordinary sweep is BLIND to it: WP6's own
// `pivot_metric` break (cabs1 -> modulus) turned NOTHING red on random and
// dominant matrices, where the two rules agree at every step. An oracle can be
// correct, necessary, and still blind; only a break says which.
// ===========================================================================
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
            // The per-item factor keeps the batch items DISTINCT -- without it the
            // batch-stride assertion in check_factor is unsatisfiable -- while
            // leaving column 0's two decisive entries untouched below.
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

        // ANTI-VACUITY: the two functionals must genuinely disagree on the data
        // this test actually put in the buffer.
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

// ===========================================================================
// L8. GETRS, ALL THREE transA MODES.
//
// The permutation SIDE changes with the transpose, and getting it wrong is a
// silently wrong answer no NoTrans test can see:
//   NoTrans   : apply F to B, then solve L, then U.
//   Trans/CT  : solve U^T then L^T, then apply F^{-1} -- the SAME list walked
//               BACKWARDS -- to the OUTPUT.
// WP6's scaffolding measured that no test in this suite issued a Trans getrs at
// all before this one.
// ===========================================================================
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

        // ANTI-VACUITY. NoTrans must differ from Trans (otherwise the mode is not
        // being read at all), and for a complex type ConjTrans must differ from
        // Trans -- without which conj(A) is untested on half the type list, which
        // is precisely the class WP5's zgeqr2 tau defect belonged to.
        EXPECT_NE(solutions[0], solutions[1])
            << "NoTrans and Trans produced identical solutions; transA is not being read";
        if constexpr (test_utils::is_complex_type_v<T>) {
            EXPECT_NE(solutions[1], solutions[2])
                << "Trans and ConjTrans produced identical solutions; the conjugation is "
                   "not being applied";
        }
    }
}

// ===========================================================================
// L9. GETRI: the inverse, and the promise that A SURVIVES.
//
// cublas<t>getriBatched takes `const T* const A[]` (cublas_api.h:5568-5576) and
// WP6's interface probe measured max |A_after - A_factored| = 0.0 for all four
// types. A native arm that wrote through A would be a drop-in failure invisible
// to every residual, so the survival is asserted BIT-EXACTLY.
// ===========================================================================
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

    // The LAST batch item is distinct from the first, so a wrong output stride
    // cannot pass by broadcasting item 0.
    bool differ = false;
    for (int j = 0; j < n && !differ; ++j)
        for (int i = 0; i < n && !differ; ++i)
            if (habs(up(c.buf[size_t(j) * c.ld + i]) -
                     up(c.buf[size_t(batch - 1) * c.stride + size_t(j) * c.ld + i])) > 0.0)
                differ = true;
    EXPECT_TRUE(differ) << "the first and last inverses are identical";
}

// ===========================================================================
// L10 / L11. THE DROP-IN CONTRACT, BOTH DIRECTIONS.
//
// This is the highest-value test in the package and the direct analogue of WP5's
// tau cross-check. getrf, getrs and getri carry INDEPENDENT env variables and
// INDEPENDENT preferred() windows, so every mixture of native and vendor arms is
// reachable in a shipped build. A factor and a pivot list produced by one must be
// consumable by the other, in both directions.
//
// Note what is NOT asserted: that the two getrf implementations produce the same
// pivots. They do not, and must not be required to -- cuBLAS pivots on the
// modulus for complex types. What must hold is that each consumer works with the
// factor its producer actually returned, which is what a residual against the
// ORIGINAL A measures.
// ===========================================================================
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
            // The vendor's own factor must satisfy the same host reconstruction --
            // which is what proves the two implementations agree on the pivot
            // FORMAT (packed 1-based int32 interchange list) even where they
            // disagree on the pivot CHOICE.
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

// ===========================================================================
// L12. THE ROUTE TABLE AND THE VENDOR-FREE FALLBACK, asked of the REAL shape
// builder on the REAL device.
//
// tests/route_vocabulary_tests.cc already exercises the table against SYNTHETIC
// shapes with cta_max_n handed in. What it cannot see is whether the builder
// reports a capacity at all on this device -- which is the difference between a
// vendor-free build having an LU and throwing NoRouteError.
// ===========================================================================
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

    // preferred() is FALSE everywhere, deliberately (WP6 ships route-neutral), so
    // a vendor-present build must still take the vendor for every shape. This is
    // the assertion that would catch a preferred() window landing without the
    // measured grid that justifies it.
    if constexpr (dispatch::factorization_vendor_available<B>) {
        for (auto* V : {&Vs, &Vl})
            EXPECT_TRUE(dispatch::is_vendor(
                backend::getrf_route<B, T>(*this->ctx, *V, /*vendor_available=*/true)))
                << "getrf routed NATIVE with a vendor present at n=" << V->rows()
                << "; preferred() moved without a measurement";
        EXPECT_TRUE(dispatch::is_vendor(
            backend::getri_route<B, T>(*this->ctx, Vl, /*vendor_available=*/true)));
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

// ===========================================================================
// L13. THE FACADE REACHES THE NATIVE KERNELS, ASSERTED BIT-EXACTLY.
//
// tests/potrf_tests.cc:895-908 records this repository's fifth blind guard: a
// route-assertion-plus-residual test "stayed GREEN across all four scalar types
// while every number in it came from cuSOLVER", because a residual bound is
// satisfied by either implementation. So the comparison here is BIT-EXACT
// against the direct entry point -- factor AND pivots -- which no vendor can
// reproduce (and, for complex, provably does not: cuBLAS pivots on the modulus).
// ===========================================================================
TYPED_TEST(LuTest, FacadeReachesTheNativeKernelsBitExactly) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 64, batch = 3;
    ASSERT_GE(this->cta_max_n(), n) << "the CTA pin cannot be exercised at n=" << n;

    for (const char* pin : {"cta", "blocked"}) {
        EnvGuard g("BATCHLAS_GETRF_ROUTE", pin);
        auto direct = make_dominant_permuted<T>(n, batch, 1234u);
        auto viafac = make_dominant_permuted<T>(n, batch, 1234u);

        // THE PIN IS VERIFIED, NEVER ASSUMED: an unrecognised value, or one
        // supports() refuses, silently resolves to the VENDOR (route_resolve.hh:
        // 165 -> :175) and this test would then compare cuBLAS with itself.
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

// ===========================================================================
// L14. THE DIRECT ENTRY POINTS REFUSE WHAT supports() REFUSES.
//
// They are reachable WITHOUT the table, so every gate has to be re-applied there
// or a pinned-route caller walks straight into an unlaunchable configuration.
// The injected-seam refusals are here too: an empty trsm seam must THROW rather
// than reach for a native kernel, which is WP3 step 16's defect refused by
// construction.
// ===========================================================================
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

// ===========================================================================
// L15. THE WORKSPACE QUERY COVERS EVERY SUPPORTED ROUTE, AND DEREFERENCES
// NOTHING.
//
// getrf_buffer_size and getri_buffer_size are reached from INSIDE a layout
// function under BumpAllocator::measuring() (src/extensions/inv.cc:35-36, from
// inv_buffer_size at :54-57), where A arrives with a NULL data pointer. A query
// that reads the data is an immediate segfault in a sizing path -- and it is the
// path tests/inverse_tests.cc actually takes.
//
// The facade's figure is max(native, vendor), so a call served with exactly that
// many bytes must succeed for EVERY pin: the ormqr defect (query 2560 bytes, call
// demanded 276480) is what this pins shut.
// ===========================================================================
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

        // Serve EXACTLY that many bytes. A short workspace is a silent heap
        // overflow, not a throw.
        UnifiedVector<std::byte> ws(std::max<std::size_t>(1, need));
        ASSERT_NO_THROW((getrf<B, T>(*this->ctx, V, p.piv.to_span(), ws.to_span(),
                                     p.info.to_span())));
        this->ctx->wait();
        check_factor(p, "buffer-size/getrf");
        if (this->HasFailure()) return;
    }
}

// ===========================================================================
// THE BREAK RECORD.
//
// Fourteen guarded properties, each corrupted at its source, the .so REBUILT,
// and tests/getrf_tests run ONCE PER SCALAR TYPE -- four filtered runs, not one,
// because a corrupted kernel can take the process down and a single run then
// reports nothing about the three types that never executed (`short_final` and
// `piv_stride_nb` both abort). Tooling and raw output:
// experiments/wp6_lu/tests/break.py, run_break.sh, breaks.txt, break_*.txt.
//
// | break              | property corrupted                          | outcome
// |--------------------|---------------------------------------------|--------
// | piv_base_zero      | ipiv written 0-based instead of 1-based      | RED, 12-13 of 16 per type
// | getrs_forward      | transposed permutation walked forwards       | RED, GetrsSolvesAllThreeTransposeModes, all 4 types -- BUT SEE BELOW
// | info_block_local   | info offset made panel-local, not global     | RED, SingularColumn..., all 4
// | short_final        | panel loop stops at the last FULL panel      | RED + SIGSEGV (exit 139), all 4
// | subview_ld         | sub-view built with rows, not the parent ld  | RED, 9 of 16 per type
// | getrs_perm_first   | transposed permutation moved to the INPUT    | RED, GetrsSolves..., all 4
// | hole_pad           | the 48 KB pad removed                        | RED, ResidentLeafLaunchHole..., all 4 -- ARITHMETIC HALF ONLY, see below
// | pivot_metric       | cabs1 -> the modulus (cuBLAS's rule)         | RED for cfloat (5) and cdouble (4); NOTHING for float/double, correctly
// | laswp_left         | interchanges not applied to columns [0, j0)  | RED, 9 of 16 per type
// | getri_forward      | getri's backward trace run forwards          | RED, Getri... and VendorFactorFeedsTheNative..., all 4
// | leaf_swap_right    | leaf row exchange restricted to columns >= k | RED, 12 of 16 per type
// | info_epsilon_floor | an epsilon floor in the singularity test     | RED, NearlySingularIsNotFlagged, all 4
// | piv_stride_nb      | pivot stride nb instead of the matrix order  | RED + SIGABRT (OUT_OF_RESOURCES), all 4
// | getri_perm_t       | F written transposed into C                  | RED, Getri... and VendorFactorFeedsTheNative..., all 4
//
// ---------------------------------------------------------------------------
// THE TWO RESULTS THAT ARE FINDINGS RATHER THAN CONFIRMATIONS
// ---------------------------------------------------------------------------
//
// 1. `getrs_forward` TURNED NOTHING RED ON THE FIRST VERSION OF THIS FILE --
//    62 passed, 0 failed -- AND THE CAUSE WAS THIS FILE'S TEST MATRIX.
//
//    make_dominant_permuted originally permuted rows by a REVERSAL. A reversal
//    is its own inverse, so the permutation the interchange list composes to
//    satisfies F = F^{-1}, and getrs's transposed arm -- whose whole content is
//    "the SAME list walked BACKWARDS" -- gives the identical answer walked
//    forwards. Three tests of a permutation DIRECTION (getrs Trans, getrs
//    ConjTrans, getri's backward trace) were unfalsifiable, on every scalar
//    type, while looking like the strongest tests in the file.
//
//    Fixed by permuting with a CYCLIC SHIFT, which composes to an n-cycle, plus
//    interchange_is_involution() asserted at every direction-sensitive use so
//    the property cannot silently regress. On the shift the break is RED for all
//    four types, and so are getrs_perm_first, getri_forward and getri_perm_t.
//    This is the sixth-plus instance of this repository's blind-guard class and
//    the first one caught in the test file rather than in the kernel.
//
// 2. `hole_pad` GOES RED ON THE ARITHMETIC HALF AND GREEN ON THE LAUNCH HALF.
//    Removing the pad makes getrf_leaf_fits admit a 49,152 B tile at a 49,152 B
//    budget -- caught -- and the resulting resident launch then SUCCEEDS on this
//    box. So on this device, for this kernel, the 48 KB hole does not reproduce,
//    which agrees with getrf_cta.cc:124-129's own reading: WP6 attributed the
//    hole to sycl::reduce_over_group alone and this body uses no group
//    collective. The pad is DEFENSIVE and layer (a) is the guard that has teeth;
//    layer (b) is kept for the day a group algorithm enters the body, which is
//    the condition WP4 wrote down and WP5 walked into anyway.
//
// ---------------------------------------------------------------------------
// THREE THINGS THE BREAKS ALSO SETTLED, worth stating because each was a guess
// before it was measured:
//
//   * ORACLE 3 IS METRIC-SENSITIVE. `pivot_metric` turns the ordinary complex
//     sweeps red -- CtaFactorises, BlockedFactorises, BothPanelLeaves -- and not
//     just the dedicated probe, because the pivot-ratio form of the oracle asks
//     the question in cabs1. WP6's kernel-side campaign recorded that this break
//     "turned NOTHING red on the ordinary sweep" against a |L| <= 1 oracle. It
//     does now.
//   * `pivot_metric` turns NOTHING red for float and double, and that is
//     correct, not a gap: cabs1 and the modulus are the same function there and
//     PivotSelectionUsesCabs1AndNotTheModulus SKIPs.
//   * `subview_ld` and `laswp_left` leave CtaFactorises and BothPanelLeaves
//     GREEN, correctly: neither the CTA tier nor the panel leaf builds a
//     sub-view or issues an interchange outside its own tile.
// ===========================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
