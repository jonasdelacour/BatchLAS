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
//
// THERE ARE TWO SUCH RECORDS AT THE BOTTOM, in this order: the FUSED-GETRS one
// (L0b plus the F-series, fourteen breaks) and then the original WP6 one (L0
// through L15, fourteen breaks). They are separate because they corrupt separate
// files -- src/extensions/getrs_fused.cc against src/extensions/getrf_*.cc -- and
// because their tooling is separate: experiments/wp6_getrs/tests/ against
// experiments/wp6_lu/tests/.
//
// THE FUSED NARROW-RHS GETRS TIER (src/extensions/getrs_fused.cc) IS THE SECOND
// NATIVE getrs ARM and it is what a vendor-free build now takes at every width
// its right-hand side is resident for. It is covered by L0b (the 48 KB ladder,
// declared early ON PURPOSE) and by F1-F7 near the end of this file; the composed
// tier's own tests (L8, L10, L11) are unchanged and still pin the arm the fused
// one hands back to.
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
        return (sycl_getrf::getrf_blocked_debug_params<T>(*this->ctx, n) >> 16) & 0xffu;
    }
    // Bits 24+: WHICH SPELLING of the deferred left-hand interchange the driver
    // resolved for this call (0 in-loop, 1 deferred walk, 2 deferred gather).
    // MASKED, not shifted bare, so adding a field above the leaf cannot silently
    // turn `leaf()` into a different number.
    unsigned left_mode(int n) const {
        return (sycl_getrf::getrf_blocked_debug_params<T>(*this->ctx, n) >> 24) & 0xffu;
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

// ===========================================================================
// THE FUSED NARROW-RHS GETRS TIER -- SHARED SCAFFOLDING.
//
// src/extensions/getrs_fused.cc is a SECOND native getrs arm: one work-group per
// matrix, the interchange walk and BOTH substitutions in ONE kernel, no GEMM and
// no separate laswp. It is reachable three ways and every one of them is
// exercised below: the direct entry point getrs_fused_dispatch, the facade under
// BATCHLAS_GETRS_ROUTE=cta, and -- in a vendor-free build -- the AUTOMATIC route,
// because native_tier_preferred puts {Native, CTA} ahead of {Native, Blocked}
// wherever supports() admits it.
// ===========================================================================

// The tier's local-memory request, RESTATED here and then PINNED against the
// library's own capacity query rather than trusted. Both halves are copied from
// src/extensions/getrs_fused.cc:
//
//   * the request is (n * nrhs + nb * (nb + 1)) scalars, and the CAPACITY QUERY
//     charges the LARGEST nb the tier ever uses (32) rather than the nb this
//     call would pick, so a caller can compare against one conservative ceiling;
//   * a request landing in the 48 KB launch hole -- (47104, 49664] BYTES -- is
//     raised to 49920, which is why the inversion in getrs_fused_max_rhs_elems
//     is NOT the obvious one: getrs_hole_padded is not monotone.
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
// nrhs, or -1 when no such order exists. Solved rather than tabulated because nb
// itself depends on n, so both candidate block widths have to be tried and only
// the one the launcher would actually choose kept.
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

// ---------------------------------------------------------------------------
// A FABRICATED LU FACTOR, built directly on the host, with NO getrf.
//
// The 48 KB ladder below runs at orders of 334-1428 to hit the byte thresholds
// exactly, and a ||PA - LU|| oracle there is O(n^3) -- seconds per rung per type.
// Fabricating the factor removes getrf from that test entirely AND makes an
// EXACT O(n^2) residual available (see fused_factor_residual): the getrs contract
// is stated in terms of L, U and the interchange list, so a reference that never
// forms A can still check the whole of it.
//
// U is strongly diagonally dominant (|U(k,k)| = 4n against off-diagonals <= 1)
// and L is near-identity (|L(i,k)| <= 1/4), so the solve is well conditioned and
// a backward-stable answer is a small residual.
//
// The pivot list is a genuine INTERCHANGE LIST -- ipiv[k] in [k+1, n], 1-BASED,
// PACKED int32 into the public int64 span -- and interchange_is_involution is
// asserted false at the use site, so the transposed arm's backwards walk is
// falsifiable here too.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// THE O(n^2) GETRS RESIDUAL, straight from the contract.
//
//   NoTrans   b = F^{-1}( L ( U x ) )          F^{-1} = the list walked BACKWARDS
//   Trans/CT  b = op(U) ( op(L) ( F x ) )      F      = the list walked FORWARDS
//
// Both are two triangular products and one permutation walk, so this is O(n^2)
// per right-hand side where forming A and multiplying is O(n^3). Returns
// ||b_ref - b|| / (||U||_F ||x||_F).
//
// NOTE what this oracle can and cannot see, MEASURED rather than assumed. It is
// exact for the arithmetic and the pivot base, and the break record at the
// bottom of this file records it going RED for `trans_perm_forward`,
// `perm_wrong_side`, `swap_solves`, `piv_base` and `rhs_ld` -- so a direction
// flip IN THE KERNEL is caught here, contrary to the first reading of this
// comment. What it cannot see is a flip of the CONVENTION ITSELF: it never forms
// A, so if the library and this reference both changed "forwards" to
// "backwards" they would still agree. That last property is what the tests
// running against a real getrf factor and the ||op(A)X - B|| oracle carry.
// ---------------------------------------------------------------------------
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

// The RHS pad and the inter-item gap must come back BIT-IDENTICAL. The fused
// kernel writes B[i + c*ldb] for i < n and c < nrhs and nothing else, so a wrong
// ld, a wrong stride, or a loop that runs to ld instead of n shows up here and in
// nothing else -- a residual over the live block cannot see it.
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
// L0b. THE 48 KB LAUNCH HOLE, FOR THE **FUSED GETRS** KERNELS.
//
// DECLARED HERE, SECOND IN THE FILE AND AHEAD OF EVERY OTHER FUSED-GETRS TEST,
// AND THAT IS LOAD-BEARING RATHER THAN TIDINESS -- the same rule L0 above states
// for GetrfPanelResidentKernel, applied to a different pair of CUfunctions.
// MaxDynamicSharedMemorySize is raised STICKILY PER CUfunction, and the fused
// tier's kernels are GetrsFusedNKernel<T, NR> / GetrsFusedTKernel<T, NR> with NR
// the compile-time accumulator width (1, 2, 4 or 8, chosen by nrhs). Every rung
// below runs at nrhs = 8, i.e. NR = 8, so ANY earlier test issuing a getrs with
// 4 < nrhs <= 8 would warm those two functions and this guard could never fail
// again. DO NOT MOVE IT BELOW THE F-SERIES.
//
// The cold check by hand:
//     ./build/tests/getrf_tests --gtest_filter='*FusedGetrsLaunchHole*'
//
// TWO INDEPENDENT LAYERS, as L0 has:
//
//   (a) THE CAPACITY INVERSION, a pure-function assertion that needs no device.
//       getrs_fused_max_rhs_elems<T>(budget) answers "how many RHS elements fit",
//       and its inverse is NOT the obvious one because getrs_hole_padded is NOT
//       MONOTONE: a request landing in (47104, 49664] is RAISED to 49920, so a
//       naive floor division can advertise a capacity whose launch then asks for
//       MORE than the budget. The sweep below is dense over the whole band and
//       past it, one byte at a time, and asserts the only thing that matters:
//       THE ADVERTISED CAPACITY MUST BE LAUNCHABLE WITHIN THE BUDGET IT WAS
//       ASKED ABOUT. This layer found a real defect -- see the break record.
//
//   (b) THE LAUNCH ITSELF, on a ladder of orders whose RAW request is exactly
//       47104 / 48896 / 49152 / 49664 / 49920 bytes, crossing the band from both
//       sides, in BOTH kernels (NoTrans and Trans are separate CUfunctions).
//       Each rung is checked against an EXACT O(n^2) host residual built from
//       the contract itself, so a rung that launches but computes nothing
//       sensible is still red.
//
// The orders are SOLVED for, not tabulated, because the block width the launcher
// picks depends on n. On this box they come out as, at nrhs = 8:
//     float             1340 1396 1404 1420 1428   (nb = 32)
//     double / cfloat    702  730  734  742  746   (nb = 16)
//     cdouble            334  348  350  354  356   (nb = 16)
// ===========================================================================
TYPED_TEST(LuTest, FusedGetrsLaunchHoleAt48KiB) {
    using T = typename TestFixture::T;
    const std::size_t sz = sizeof(T);

    // ---- layer (a): the capacity inversion --------------------------------
    //
    // Every budget from below the band to well past the pad target, one byte at
    // a time. `cap` is what the library advertises; fused_capacity_bytes is this
    // file's restatement of what a caller sized by that capacity would then ask
    // the runtime for. The second must never exceed the budget the first was
    // asked about.
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
    // Coarse ladder past the band, including this device's own budget, so a
    // regression that only shows up at realistic sizes is caught too.
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
// L2b. THE THREE SPELLINGS OF THE LEFT-HAND INTERCHANGE AGREE BIT FOR BIT.
//
// WHAT IS BEING ASSERTED. lu_laswp.hh's deferral identity says column block r
// receives the SAME transposition list in the SAME order whether it is applied
// one panel at a time inside the block loop (LAPACK's schedule) or as one suffix
// after it. "The same composition" is a stronger claim than "both residuals are
// small", and a residual bound cannot distinguish the two: any valid pivot
// sequence satisfies it. So the assertion here is BITWISE EQUALITY of the whole
// factor and of the whole interchange list, over the whole batch -- the exact
// statement the deferral makes.
//
// IT ALSO COVERS THE FALLBACK. `defer_walk` is not a third algorithm: it is the
// branch the driver takes when the SLM-staged gather refuses the shape, spelled
// with the ordinary walk. Without this arm that branch is only reachable at
// orders above ~6,000 (float) and would be shipped untested -- the "capacity
// with no test" hazard route_potrf.hh:442-454 records.
//
// ANTI-VACUITY, and it is the point of `left_mode`. The environment read in
// getrf_blocked.cc latches its PRESENCE in a function-local static, so a test
// that merely setenv()s and hopes could silently compare an arm against itself.
// The resolved spelling is read back from the driver's own debug query and
// asserted before every arm runs; a latched flag makes this test RED, not green.
// The file-scope object below is what makes the latch land on "present" in the
// first place -- it runs before main and therefore before any getrf call.
//
// THE ORDERS straddle the block width in both directions and include n = 129,
// where the last panel is ONE column wide: the deferred pass's block extents
// must come from ib, never from nb, and this family has shipped exactly that
// bug once already with a green suite (sy2sb stage 1).
// ===========================================================================
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
// L8b. GETRS, THE TWO PERMUTATION SPELLINGS.
//
// WP8-I2 gave the composed getrs driver a SECOND spelling of the row
// permutation: instead of walking the interchange list down every column of B
// (lu_laswp.hh's lu_laswp_launch), it applies the list ONCE to an identity index
// array in local memory and then gathers, dst[i] = src[idxs[i]], contiguously.
// Which one runs is a SPEED decision -- BATCHLAS_GETRS_LASWP, or B.cols()
// against kGetrsPermGatherMinNrhs -- and never a correctness one.
//
// So the two arms must agree BIT FOR BIT: the permutation is a pure data move
// and the two trsm calls that follow it are identical, so any difference at all
// is a defect and not rounding. That is a strictly stronger assertion than the
// residual, which both arms would pass with the SAME wrong permutation.
//
// THREE ANTI-VACUITY GUARDS, and none is decorative:
//
//  (1) THE SPELLING IS READ BACK, per arm, from getrs_perm_spelling_debug --
//      the driver's OWN resolution, not a re-derivation. Two things it catches:
//      an environment read that did not take, and the gather's SILENT FALLBACK
//      to the walk when the tile does not fit local memory. Without it, a test
//      that believes it is exercising the gather can be running the walk twice
//      and asserting that the walk equals itself.
//  (2) THE PERMUTATION MUST NOT BE AN INVOLUTION. If it were, the forward and
//      reversed index walks would coincide and the Trans/ConjTrans rows -- the
//      only place the reversed direction is exercised -- would prove nothing.
//  (3) nrhs = 70 EXCEEDS THE TILE WIDTH for every scalar type on this device, so
//      the multi-chunk loop and its partial last chunk both run. At nrhs = 5 the
//      first chunk is already partial. A single mid-size nrhs would test neither.
//
// n = 96 and n = 257 are both below and above the 256-wide work-group, which
// selects different branches of the (column, row) flattening; 257 is odd, so the
// odd-ld padding is a no-op there and a real pad at 96.
// ===========================================================================
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

                    // GUARD (1). The driver's own resolution, for THIS shape, on
                    // THIS queue -- so a fallback the caller cannot see is visible
                    // here.
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
                    // The query must stay 0 under BOTH spellings: the gather is in
                    // place. See GetrsPermGatherBuysNoWorkspace below.
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

                // THE STRONG ASSERTION. Same permutation, same two solves, so the
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

// ===========================================================================
// L8c. THE SPELLING DECISION SURFACE, WITHOUT RUNNING A KERNEL.
//
// getrs_perm_spelling_debug resolves through the driver's own perm_spelling()
// and its own capacity arithmetic, so this test pins the two boundaries the
// driver actually has:
//
//   THE nrhs BOUNDARY, kGetrsPermGatherMinNrhs, which is the default policy.
//   The constant is TRANSCRIBED from the header rather than written out, so a
//   later retune moves the test with the code -- but the CELLS ON EITHER SIDE
//   are asserted, which is what a wrongly-inverted comparison would break.
//
//   THE CAPACITY REFUSAL. The gather needs 2*n ints plus one column of B in
//   local memory; above that it enqueues NOTHING and the driver re-schedules the
//   identical composition with the walk. That fallback is silent by design --
//   RouteTable<Op::getrs,T> has no field to advertise a laswp capacity -- and it
//   is therefore invisible to every other test in this suite. Asserting it here
//   costs no kernel launch, because the query takes n as an integer.
// ===========================================================================
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

    // linalg::solve issues getrs at nrhs = 1 and is the only caller in the tree.
    // It must keep the walk, which is what makes "the boundary buys nothing at
    // the narrow end" a property of the shipped library and not of a comment.
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

    // ...and it must NOT refuse at an order the suite and the benchmarks reach.
    // A capacity that fires early is a lever that never runs -- 'linked is not
    // reachable', with the sign flipped.
    EXPECT_EQ(sycl_getrs::getrs_perm_spelling_debug<T>(*this->ctx, 1024, 4 * kMin), 1)
        << "the gather must still fit at n = 1024, the largest order this pass measured";
    unsetenv("BATCHLAS_GETRS_LASWP");
}

// ===========================================================================
// L8d. THE GATHER BUYS NO WORKSPACE, AT ANY WIDTH.
//
// THE HAZARD THIS GUARDS, stated exactly, because nothing else in the suite can
// see it. src/dispatch/entry_points/factorization.cc:846-866 takes the workspace
// maximum over EVERY NATIVE TIER THAT supports() THE SHAPE, not over the tier
// the route named -- and at nrhs <= kGetrsFusedMaxRhs BOTH tiers supports(). So
// a gather implemented the way the WP6 plan budgets for it -- an out-of-place
// RHS plus an int32[n] per item, bought in getrs_blocked_buffer_size -- would
// bill every nrhs = 1 call that routes to the FUSED tier and needs nothing:
// 1,310,720 B at cdouble n=512 batch=128, on linalg::solve's hot path.
//
// The shipped gather permutes IN LOCAL MEMORY, in place, so there is nothing to
// bill. This test is what keeps that true: it asserts ZERO at a wide nrhs under
// BOTH spellings, so a later out-of-place strategy cannot arrive silently.
// ===========================================================================
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

    // getrf AND getri ARE STILL ROUTE-NEUTRAL. preferred() is false everywhere for
    // both (WP6 ships them that way), so a vendor-present build must still take the
    // vendor at every shape. This is the assertion that catches a window landing on
    // either of them without the measured grid that justifies it -- and it is
    // deliberately left as it was, because WP6-PERF moved getrs ALONE. If a getrf
    // window ever lands, this is the test to rewrite, the way the getrs half below
    // was rewritten rather than deleted.
    // ---- WP8 ROUTING PASS: BOTH WINDOWS HAVE NOW LANDED, SO THIS ASSERTION IS
    // REWRITTEN RATHER THAN DELETED, exactly as its own note said it should be.
    //
    // Vs is n = min(40, cta_max_n) and Vl is n = 512, both at BATCH 2. Neither
    // window carries a batch term -- for getri that is measured (batch 1..32 is
    // where the native driver's advantage is LARGEST, 1.7x-28x, because cuBLAS's
    // batched getri is a per-item loop there), and for getrf the clause is on
    // order alone -- so what this asks of the REAL shape builder on the REAL
    // device is the per-type order boundary, at a batch the perf grids never
    // measured. That is the point: it is the one place in the suite where the
    // window meets a batch of 2.
    if constexpr (dispatch::factorization_vendor_available<B>) {
        constexpr bool kF  = std::is_same_v<T, float>;
        constexpr bool kCF = std::is_same_v<T, std::complex<float>>;

        // Vs (order <= 40) is below EVERY boundary of both windows, for every
        // type. This half is unchanged in meaning and is what catches a window
        // that forgets its lower bound.
        EXPECT_TRUE(dispatch::is_vendor(
            backend::getrf_route<B, T>(*this->ctx, Vs, /*vendor_available=*/true)))
            << "getrf routed NATIVE at n=" << Vs.rows()
            << ", below every measured boundary";
        EXPECT_TRUE(dispatch::is_vendor(
            backend::getri_route<B, T>(*this->ctx, Vs, /*vendor_available=*/true)));

        // Vl is n = 512.
        //   getrf window: float order >= 256, cfloat order >= 512  -> both IN.
        //   getri window: float order >= 128, cfloat order >= 256  -> both IN.
        // double and cdouble earn nothing in either op at any order.
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
    //
    // route_vocabulary_tests.cc pins the same window against SYNTHETIC shapes with
    // the two fused capacities handed in. What it cannot see -- and what made its
    // getrs assertions blind guards until this pass -- is whether the BUILDER on
    // THIS DEVICE reports capacities at all. A builder that returns 0 for
    // fused_max_elems turns supports({Native, CTA}) false everywhere, and then
    // every window assertion in the pure suite passes for a reason that has
    // nothing to do with the window. So: the capacities first, then the window.
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
// F1. THE FUSED TIER SOLVES ALL THREE transA MODES AT EVERY INSTANTIATED WIDTH.
//
// The permutation SIDE changes with the transpose and the two substitutions SWAP
// ORDER, and getting either wrong is a silently wrong answer no NoTrans test can
// see. Measured on this tier (break record at the bottom of this file): B1
// (transposed permutation walked forwards) and B4 (transposed output permutation
// dropped) turn ONLY Trans and ConjTrans red; B3 (NoTrans interchange walk
// dropped) turns ONLY NoTrans red; B6 (ConjTrans stops conjugating) turns ONLY
// ConjTrans red AND only by 1e-1, far subtler than the rest.
//
// EVERY nrhs THE TIER SERVES, not a sample: the accumulator width NR is a
// COMPILE-TIME template parameter chosen by a runtime ladder (nrhs <= 1 -> 1,
// <= 2 -> 2, <= 4 -> 4, else 8), so 1, 2, 4 and 8 are four different kernels and
// 3 and 5 are the shapes where the `if (c < nrhs)` guards inside a WIDER
// accumulator are the only thing keeping a lane from writing a column that does
// not exist.
//
// n = 97 is 6 full nb = 16 blocks plus a FINAL BLOCK OF ONE, which is the shape
// that exercises the jb == 1 guards in both substitutions and both kernels.
// ===========================================================================
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

            // THE ORACLE IS ||op(A) X - B|| AGAINST THE **ORIGINAL** A, in double
            // regardless of T, and NOT the L/U-based one used by the hole ladder:
            // only this form is sensitive to the permutation DIRECTION, because
            // only this form knows what A was before it was factored.
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

// ===========================================================================
// F2. ORDERS: ONE, THE BLOCK BOUNDARIES, AND THE nb SWITCH AT 1024.
//
// The fused launcher's geometry is a set of thresholds and every one of them is
// an off-by-one waiting to happen:
//   * nb = 16 below order 1024 and 32 at or above it, then clamped to n, so
//     1023 / 1024 / 1025 straddle a change of BOTH the block width and the
//     resident block's leading dimension;
//   * jb = n - j on the FINAL block, so 15 / 17 / 31 / 33 / 65 / 97 each leave a
//     short tail, and jb == 1 disables the unit-diagonal recurrence entirely
//     (`sgid == 0 && jb > 1`) in two of the four substitutions;
//   * n = 1 is the whole op reduced to one division, with every trailing-update
//     loop empty.
//
// ORDER 1 AND 2 SKIP THE INVOLUTION ASSERTION, and that is a statement about the
// construction rather than an exemption: make_dominant_permuted's cyclic shift
// is an n-cycle, and an n-cycle IS self-inverse for n <= 2. There is no
// permutation of one or two rows that can distinguish a forwards walk from a
// backwards one, so those two orders are testing the arithmetic and the tails,
// not the direction.
// ===========================================================================
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

// ===========================================================================
// F3. THE TWO CEILINGS: THE WIDTH THE BUILD INSTANTIATED, AND THE DEVICE'S
// RESIDENT-RHS CAPACITY. BOTH MUST HAND BACK, NOT PRODUCE GARBAGE.
//
// They are DIFFERENT KINDS of ceiling and route_getrs.hh keeps them separate for
// that reason: fused_max_nrhs is what this BUILD compiled (kGetrsFusedMaxRhs = 8;
// above it no instantiation exists), fused_max_elems is what THIS DEVICE's local
// memory holds. Both live in supports(), never in preferred(), because above
// either of them the kernel does not launch -- and a speed threshold in
// supports() would make a pinned `native:cta` fall through to automatic()
// (route_resolve.hh:165 -> :175) and the test that pinned it would measure the
// vendor and pass green.
//
// WHAT IS CHECKED AT EACH CEILING:
//   * supports() says yes AT the boundary and no ONE PAST it;
//   * the DIRECT entry point throws one past it rather than launching;
//   * the FACADE one past it still returns the RIGHT ANSWER, from some other
//     route, and that route is NOT {Native, CTA}.
//
// The capacity half uses NULL-DATA views: the entry point refuses on metadata
// alone, before it dereferences anything, so proving that costs no allocation at
// orders of ~3000.
// ===========================================================================
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

    // ONE PAST THE WIDTH, THROUGH THE FACADE, MUST STILL BE RIGHT. This is the
    // half a supports() test cannot reach: the route has to fall to a tier that
    // can serve it and the answer has to survive the handover.
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

// ===========================================================================
// F4. THE DROP-IN CONTRACT FOR THE FUSED TIER, BOTH DIRECTIONS AND BOTH
// PRODUCERS.
//
// getrf and getrs carry INDEPENDENT env variables and INDEPENDENT preferred()
// windows, so a vendor getrf feeding a native getrs is a shipped configuration
// and not a hypothetical. The fused tier reads the pivot buffer DIRECTLY --
// pivots.as_span<int>(), packed 1-BASED int32, an INTERCHANGE LIST -- so it is
// the arm most exposed to a format disagreement, and it re-derives the walk in
// its own kernel rather than delegating to the shared laswp the composed tier
// uses.
//
// BOTH NATIVE PRODUCERS ARE USED, not just one: getrf's CTA tier and its blocked
// tier write the pivot list from different code, and n = 40 is inside the CTA
// tier's capacity on this device.
// ===========================================================================
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
        // VENDOR getrf. Its pivot CHOICE differs from ours for complex types
        // (cuBLAS pivots on the modulus, this library on cabs1), which is exactly
        // why the oracle is a residual against the ORIGINAL A and never an
        // elementwise comparison of the two factors.
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

        // AND THE OTHER DIRECTION, on the same factor: the vendor getrs must
        // still consume it. That is what makes the pivot FORMAT -- as opposed to
        // the pivot CHOICE -- a shared fact rather than an internal convention.
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

// ===========================================================================
// F5. A SINGULAR AND A NEARLY-SINGULAR FACTOR.
//
// getrs has no info output and no singularity contract: LAPACK's ?GETRS divides
// by U(k,k) unconditionally and the caller is expected to have looked at getrf's
// info. What must NOT happen is the two failure modes this repository has already
// shipped once elsewhere -- an EPSILON FLOOR that silently perturbs the answer
// (getrf's `info_epsilon_floor` break) and a guard that SKIPS the division and
// returns a plausible-looking wrong number.
//
// So the assertions are:
//   * NEARLY singular (a diagonal scaled down by 1e-6 / 1e-12) must still be
//     SOLVED, to a BACKWARD-error bound. The residual used here normalises by
//     ||X||, so conditioning cancels and the bound is legitimately tight.
//   * EXACTLY singular must PROPAGATE: the answer must contain a non-finite
//     value. A finite answer means something floored the division.
// Both diagonals probed are boundaries in their own right: k = 0 is the first
// step of the back substitution's LAST block and k = n-1 is the last row of its
// FIRST, which is where an off-by-one in the reverse loop lands.
// ===========================================================================
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

// ===========================================================================
// F6. THE FACADE REACHES THE FUSED KERNEL, ASSERTED BIT-EXACTLY, AND THE
// VENDOR-FREE DEFAULT LANDS ON IT.
//
// tests/potrf_tests.cc:895-908 records this repository's fifth blind guard: a
// route-assertion-plus-residual test "stayed GREEN across all four scalar types
// while every number in it came from cuSOLVER", because a residual bound is
// satisfied by either implementation. So the comparison is BIT-EXACT against the
// direct entry point, which no vendor and no other native tier can reproduce.
//
// The route half is what pins the ROUTING change this tier landed with:
//   * pinned `cta`   -> {Native, CTA};
//   * pinned `blocked` -> {Native, Blocked} and NOT CTA, which is what keeps the
//     composed tier reachable for its own tests (route_resolve.hh:165 -> :175
//     would otherwise hand it to automatic() and it would measure the other arm);
//   * UNPINNED in a VENDOR-FREE build -> {Native, CTA} inside the capability and
//     {Native, Blocked} outside it, which is native_tier_preferred's only job;
//   * UNPINNED in a VENDOR-PRESENT build -> the VENDOR, because preferred() is
//     still all-false and this tier did not move it.
// ===========================================================================
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

    // The other pin must still reach the COMPOSED tier, not be swallowed by the
    // new route sitting ahead of it in kGetrsOrder.
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

        // THE VENDOR-PRESENT ROUTE, AND THIS ASSERTION IS THE INVERSE OF THE ONE
        // IT REPLACES. It used to read is_vendor at nrhs = 1, and it was right to:
        // preferred() was all-false, WP6 shipped route-neutral, and the assertion
        // existed to catch a window landing without a grid. The grid landed
        // (experiments/wp6_perf/bench/, 488 cells over six sweeps), so the window
        // did, and the guard is rewritten around the window rather than deleted --
        // BOTH sides of it, because an assertion that only pins the inside cannot
        // fail when someone widens the clause.
        if constexpr (dispatch::factorization_vendor_available<B>) {
            const auto rv = backend::getrs_route<B, T>(
                *this->ctx, A, Vn, Transpose::NoTrans, /*vendor_available=*/true);
            EXPECT_TRUE(dispatch::is_native(rv) && rv.algo == dispatch::Algorithm::CTA)
                << "nrhs=1 is INSIDE the measured window (geomean 2.26x over 111 cells, "
                   "min 1.24x, flat across every batch ladder at seven orders) and must "
                   "route to the fused tier even with cuBLAS present";

            // OUTSIDE the window, the vendor still takes it. nrhs = 8 is inside the
            // tier's CAPABILITY and outside its measured WINDOW, which is exactly
            // the pair of facts a speed test in supports() would destroy.
            auto w8 = make_rhs<T>(n, int(sycl_getrs::kGetrsFusedMaxRhs), batch, 7575u);
            auto V8 = view_of(w8);
            EXPECT_TRUE(dispatch::is_vendor(backend::getrs_route<B, T>(
                *this->ctx, A, V8, Transpose::NoTrans, /*vendor_available=*/true)))
                << "nrhs=" << sycl_getrs::kGetrsFusedMaxRhs << " is OUTSIDE the window "
                   "(geomean 0.819x over 24 cells, 13 losses) and must take the vendor";

            // ... and clause B is FLOAT ONLY. nrhs = 4 splits by type, which is the
            // half of the window most likely to be widened by someone who reads
            // "nrhs <= 4" and drops the type test.
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

// ===========================================================================
// F7. THE FUSED DIRECT ENTRY POINT REFUSES WHAT supports() REFUSES, AND ITS
// WORKSPACE QUERY DEREFERENCES NOTHING.
//
// It is reachable WITHOUT the table -- that is why it exists (potrf_native.hh:
// 126-141) -- so every gate has to be re-applied there or a pinned-route caller
// walks straight into an unlaunchable configuration.
//
// The workspace half is ZERO in every mode for this tier, and that is a
// consequence of the design rather than a coincidence: the RHS is permuted and
// solved in LOCAL memory and written back in place. The facade's figure is a max
// over BOTH native tiers, and it is asserted here because the query and the call
// resolve INDEPENDENTLY (options.hh:651 sizes, :652 calls, two getenv reads
// inside one API call), so a lease sized for one tier is a lease the other can
// overrun.
// ===========================================================================
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

// ===========================================================================
// THE FUSED-GETRS BREAK RECORD (the F-series and L0b).
//
// SIXTEEN guarded properties, each corrupted at its source, the .so REBUILT,
// and the whole binary re-run. FIFTEEN of the sixteen went RED; the one that did
// not is a finding rather than a gap. Tooling:
// experiments/wp6_getrs/tests/ (break.py, break2.py, runbreak.sh).
//
// COUNT THE ROWS, NOT THIS SENTENCE. An earlier version of this paragraph said
// "fourteen ... thirteen of the fourteen" over a table that already had sixteen
// rows and fifteen REDs, and a reader who trusted the prose would have concluded
// that exactly one property is unguarded when in fact TWO breaks turned nothing
// red -- `cap_band` here and `B5` (the +1 bank-conflict pad) recorded in
// src/extensions/getrs_fused.cc, in a different file. The table below is the
// record; this sentence is a summary of it and has been wrong once already. This
// is the same misreading class the campaign records for WP5's "zero suites
// closed", which was three.
//
// TYPE INDICES BELOW ARE THE GPU SUITES: 4 = float, 5 = double, 6 = cfloat,
// 7 = cdouble. Suites 0-3 are the NETLIB rows and SKIP (CPU queue, gate 2).
//
// | break               | property corrupted                              | outcome (of 95)
// |---------------------|-------------------------------------------------|-----------------
// | piv_base            | ipiv read 0-based instead of 1-based            | RED, 28, 7 of 8 F-tests x 4 types
// | rhs_ld              | the RHS write-back uses n instead of ldb        | RED, 24, 6 F-tests x 4 types
// | unit_u              | a UNIT-DIAGONAL assumption on U, both kernels   | RED, 24, 6 F-tests x 4 types
// | trans_perm_forward  | the transposed output permutation walked FORWARDS | RED, 20, 5 F-tests x 4 types
// | perm_wrong_side     | the transposed permutation moved to the INPUT   | RED, 20, 5 F-tests x 4 types
// | swap_solves         | the two NoTrans substitutions SWAPPED (U before L) | RED, 19, all 4 types
// | last_row            | off-by-one at the LAST ROW of the trailing update | RED, 13, all 4 types
// | conj                | ConjTrans stops conjugating                     | RED, 6, cfloat and cdouble ONLY
// | cap_inversion       | the capacity-inversion repair reverted          | RED, 4, LaunchHole layer (a), all 4 types
// | hole_pad            | the 48 KB pad removed from the launcher         | RED, 4, LaunchHole layer (a) ONLY
// | reg_cap             | the register cap on the work-group width removed | RED, 1, LaunchHole, float ONLY -- a LAUNCH ABORT (RE-VERIFIED, see below)
// | facade_arm          | the facade's CTA arm routed to the composed tier | RED, 4, FacadeReaches..., all 4 types
// | tier_pref           | native_tier_preferred inverted                  | RED, 4, FacadeReaches..., all 4 types
// | supports_gates      | supports() stops checking the two CTA ceilings  | RED, 8, HandsBack... + FacadeReaches..., all 4
// | dispatch_gates      | the direct entry point stops re-checking them   | RED as a PROCESS ABORT inside HandsBack...
// | cap_band            | the hole band dropped from the capacity query   | NOTHING RED -- see below
//
// ---------------------------------------------------------------------------
// THE WP6-PERF WINDOW BREAK RECORD -- five more, run when preferred() landed.
//
// The window is a claim spanning TWO files (this one, on the real device, and
// route_vocabulary_tests.cc, on synthetic shapes), so each break is reported
// against BOTH. Same method: patch the source, rebuild the .so, re-run the whole
// binary.
//
// | break     | what was corrupted                  | getrf_tests | route_vocabulary_tests
// |-----------|-------------------------------------|-------------|----------------------
// | W1        | clause A (nrhs <= 2) switched off    | RED, 6      | RED, 1
// | W2        | clause B widened to EVERY type      | RED, 3      | RED, 2
// | W3        | the COMPOSITION also made preferred | green       | RED, 2
// | V1        | the fused capacities in route_vocabulary_tests' getrs_shape()
// |           | helper returned to 0, i.e. the blind-guard state
// |           |                                     | n/a         | RED, 3
// | R1        | the register cap removed entirely   | RED, 1      | n/a
//
// FOUR OF THE FIVE OUTCOMES ARE THEMSELVES FINDINGS:
//
// * W1 LEAVES FLOAT GREEN, and that is correct rather than a hole: with clause A
//   off, float nrhs = 1 is still inside clause B (nrhs <= 4 for float), so the
//   float route does not move. Only double, cfloat and cdouble go red. A reader
//   who expects "all four types" from a window break would mis-read this.
//
// * W3 TURNS NOTHING RED IN THIS FILE, and that is what the pure suite is for.
//   Making the composition preferred does not change any RESOLVED route, because
//   CTA is listed first in kGetrsOrder and automatic() returns the first route
//   that is both supported and preferred. Only a direct assertion on preferred()
//   itself can see it, and route_vocabulary_tests carries two.
//
// * V1 IS THE BLIND GUARD MADE VISIBLE. Before this pass, getrs_shape() set
//   neither fused capacity, so supports({Native, CTA}) was false on every shape
//   in route_vocabulary_tests and every getrs routing assertion in it held no
//   matter what the table said -- 78/78 through the flip and through its
//   inverse. Re-arming that (V1) turns three tests red, which is the proof that
//   the repair is load-bearing and that the file is no longer blind here.
//
// * R1 REPRODUCES A HARD LAUNCH ABORT, not a wrong answer:
//       "Exceeded the number of registers available on the hardware.
//        The kernel uses 68 registers per work-item for a total of 1024
//        work-items per work-group."
//   in LuTest/4 (float) FusedGetrsLaunchHoleAt48KiB. A review of this change
//   asserted that NO test in the suite reaches a shape where the cap can bite;
//   it does -- the 48 KB ladder's top rung is n = 1428 at nrhs = 8 and picks
//   wg = 1024, and it runs transA = Trans. The review had looked only at the two
//   tests with "Width"/"Boundaries" in their names.
//
// ---------------------------------------------------------------------------
// THE RESULTS THAT ARE FINDINGS RATHER THAN CONFIRMATIONS
// ---------------------------------------------------------------------------
//
// 1. `cap_inversion` IS A REAL DEFECT THIS TEST FOUND, not a re-confirmation.
//    getrs_fused_max_rhs_elems originally answered a budget with a plain floor
//    division. getrs_hole_padded is NOT MONOTONE, so for a budget a few bytes
//    above kGetrsHoleHi the division rounds the implied request back DOWN INTO
//    the band, where it is then RAISED to 49,920 and no longer fits: at 49,665 B
//    the query advertised a capacity needing 49,920 B, for all four scalar
//    types. That is a supports() promising a route whose launch the runtime
//    refuses -- exactly what potrf_cta.cc:445-470's `break` exists to prevent.
//    The window is only sizeof(T) bytes wide and needs a device with 53,761 B of
//    local memory, so it is UNREACHABLE on this box and no launch test could
//    ever have found it; the byte-by-byte sweep of a PURE FUNCTION could, and
//    did. Repaired in src/extensions/getrs_fused.cc; the break re-confirms the
//    repair is load-bearing.
//
// 2. `cap_band` TURNED NOTHING RED, AND THAT IS CORRECT: the two mechanisms are
//    REDUNDANT after the repair above. Clamping `admissible` to kGetrsHoleLo and
//    re-checking the implied request against the budget close the same hole from
//    opposite ends, so removing either one alone leaves the other doing the
//    whole job. Verified by hand across the sweep: with the clamp gone, a budget
//    of 49,152 B still yields 10,720 float elements, the same answer. It is a
//    genuinely unfalsifiable-in-isolation line and is kept as the cheap common
//    case, not as the guard.
//
// 3. `hole_pad` GOES RED ON LAYER (a) AND GREEN ON LAYER (b) -- the same split
//    L0's own `hole_pad` produced for getrf, and for the same reason: the
//    49,920 B launch this device is asked for succeeds without the pad too, so
//    the 48 KB hole does not reproduce here for these kernels either. Note that
//    layer (a) catches it only INDIRECTLY -- this file keeps its own restatement
//    of the pad, so removing the library's makes the two disagree. That is still
//    a guard, but it is a "the two copies of the rule diverged" guard rather
//    than a device-behaviour one, and the launch half remains the layer that
//    would fire if the hole ever reproduced.
//
// 4. `reg_cap` IS A LAUNCH ABORT, NOT A WRONG ANSWER, and only ONE cell in the
//    whole binary reaches it: float, transA=Trans, nrhs=8, n=1428 -- the top
//    rung of the 48 KB ladder, which picks wg = 1024 against a 68-register
//    kernel for 69,632 registers per work-group against a 65,536 limit. The
//    exception text is quoted verbatim by the runtime. Nothing else in this file
//    is wide enough AND deep enough at once to hit it, which is worth knowing
//    before anyone trims the ladder for runtime.
//
// 5. THE THREE BREAKS THAT SEPARATE THE TRANSPOSED ARM FROM THE NoTrans ONE --
//    `trans_perm_forward`, `perm_wrong_side` and `conj` -- leave
//    FusedGetrsHandsBackAtBothCeilings GREEN, correctly: that test asserts
//    routing and refusals, and checks numbers only on the path that has already
//    handed BACK to another tier.
//
// 6. `last_row` and `swap_solves` LEAVE FusedGetrsSolvesEveryTransposeAtEvery-
//    InstantiatedWidth GREEN FOR FLOAT and are caught only by
//    FusedGetrsAtBlockBoundariesAndTheNbSwitch. At n = 97 in float the dropped
//    last row sits under 400 n eps; at the small orders it does not. The order
//    sweep is not decoration -- for two of these breaks it is the only thing
//    that fires on all four types.
//
// 7. `unit_u` LEAVES FacadeReachesTheFusedGetrsBitExactly GREEN, correctly and
//    by construction: both sides of that comparison are the same corrupted
//    kernel. A bit-exactness test pins WHICH CODE RAN and can never pin what
//    that code computes -- which is why it sits beside the residual tests and
//    not instead of them.
//
// 8. `dispatch_gates` IS THE ONLY BREAK IN THIS SET THAT TAKES THE PROCESS DOWN
//    (SIGABRT, exit 134, inside FusedGetrsHandsBackAtBothCeilings on the FIRST
//    GPU type). That is the point of the ceiling test's null-data half: with the
//    re-check gone, the entry point launches an order of ~2900 whose resident RHS
//    the device cannot hold, and there is no graceful failure mode left. Because
//    it aborts, the three later types never run -- so if this break is ever
//    repeated, run it FILTERED, once per scalar type, the way L0's break record
//    says `short_final` and `piv_stride_nb` have to be run.
//
// 9. `supports_gates` TURNS **TWO** TESTS RED, AND THE SECOND ONE IS THE POINT.
//    FusedGetrsHandsBackAtBothCeilings goes red on the supports() assertions, as
//    designed. FacadeReachesTheFusedGetrsBitExactly ALSO goes red -- because with
//    the ceilings gone, the nrhs = 9 shape now resolves to {Native, CTA} and the
//    facade reaches a kernel that does not exist at that width. That is the
//    supports()/preferred() confusion route_getrs.hh's CTA clause warns about,
//    caught from the route side rather than the numeric side.
// ===========================================================================

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
