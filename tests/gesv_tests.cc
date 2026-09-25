// P2: the fused factor-and-solve tier for GESV, and the routing that (deliberately)
// does not yet reach it. Every numerical test drives the native dispatch entry point
// DIRECTLY against a host reference; the vendor is never an oracle here, and could not
// be -- there is no batched vendor gesv on any backend.
// evidence: docs/perf/lu.md#the-fused-gesv-tier
#include <gtest/gtest.h>

#include <batchlas/blas/functions/gesv.hh>
#include <batchlas/blas/functions/getrf.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>

#include "test_utils.hh"

#include "../src/extensions/solve_native.hh"
#include "../src/extensions/getrf_native.hh"
#include "../src/backends/gesv_route.hh"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using namespace batchlas;

namespace {

template <typename T>
using RealOf = typename batchlas::base_type<T>::type;

// Host arithmetic promotes to double before it accumulates, so a float residual
// measures the KERNEL's error and not the reference's.
inline double up(float x) { return double(x); }
inline double up(double x) { return x; }
inline std::complex<double> up(std::complex<float> x) { return {double(x.real()), double(x.imag())}; }
inline std::complex<double> up(std::complex<double> x) { return x; }

inline double habs(double x) { return std::fabs(x); }
inline double habs(std::complex<double> x) { return std::abs(x); }
inline bool hfinite(double x) { return std::isfinite(x); }
inline bool hfinite(std::complex<double> x) {
    return std::isfinite(x.real()) && std::isfinite(x.imag());
}

template <class T> inline T mk(double re, double im);
template <> inline float mk<float>(double re, double) { return float(re); }
template <> inline double mk<double>(double re, double) { return re; }
template <> inline std::complex<float> mk<std::complex<float>>(double re, double im) {
    return {float(re), float(im)};
}
template <> inline std::complex<double> mk<std::complex<double>>(double re, double im) {
    return {re, im};
}

template <typename T>
constexpr double eps_of() {
    if constexpr (std::is_same_v<RealOf<T>, float>) return 1.1920929e-7;
    else return 2.220446049250313e-16;
}

// LU with partial pivoting is backward stable, so the bound scales with n * eps and
// not with conditioning -- which is why every residual below runs on a
// diagonally-dominant matrix whose cond(A) is O(1).
template <typename T> double solve_tol(int n) { return 400.0 * double(n) * eps_of<T>(); }

// A deterministic LCG rather than <random>: two runs of this file must build the
// same matrices, because several tests compare two kernels element by element.
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 6364136223846793005ULL + 1442695040888963407ULL) {}
    double next() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return double(int32_t(uint32_t(s >> 32))) / 2147483648.0;
    }
};

template <typename T, Backend B>
struct GesvConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

// A batch of DISTINCT systems with a PADDED ld and a stride that is NOT ld*cols, the
// pad POISONED. A launcher that lets MatrixView default the stride, or that writes
// `c < N` instead of `c < n`, is falsifiable by construction.
template <typename T>
struct Sys {
    int n = 0, nrhs = 0, batch = 0;
    int lda = 0, stra = 0, ldb = 0, strb = 0;
    UnifiedVector<T> a;          // overwritten by the factor
    UnifiedVector<T> b;          // overwritten by X
    std::vector<T> a0, b0;       // pristine inputs, same ld/stride
    UnifiedVector<T*> aptr, bptr;
    UnifiedVector<int64_t> piv;
    UnifiedVector<int32_t> info;
};

template <typename T>
void reset(Sys<T>& p) {
    std::copy(p.a0.begin(), p.a0.end(), p.a.begin());
    std::copy(p.b0.begin(), p.b0.end(), p.b.begin());
    std::fill(p.piv.begin(), p.piv.end(), int64_t(0x0BADBEEF0BADBEEFLL));
    std::fill(p.info.begin(), p.info.end(), int32_t(-12345));
}

// Strictly column-diagonally-dominant, so cond(A) is O(1) and the residual bound above
// is the right one. The diagonal magnitude is 4n against off-diagonals bounded by 1.
template <typename T>
Sys<T> make_system(int n, int nrhs, int batch, unsigned seed,
                   int ld_pad = 5, int stride_pad = 11, bool singular_col = false) {
    Sys<T> p;
    p.n = n; p.nrhs = nrhs; p.batch = batch;
    p.lda = n + ld_pad;  p.stra = p.lda * n + stride_pad;
    p.ldb = n + ld_pad;  p.strb = p.ldb * nrhs + stride_pad;

    const T poison = mk<T>(-9.75e3, 4.5e3);
    p.a = UnifiedVector<T>(static_cast<size_t>(p.stra) * batch, poison);
    p.b = UnifiedVector<T>(static_cast<size_t>(p.strb) * batch, poison);
    p.aptr = UnifiedVector<T*>(static_cast<size_t>(batch), nullptr);
    p.bptr = UnifiedVector<T*>(static_cast<size_t>(batch), nullptr);
    p.piv = UnifiedVector<int64_t>(static_cast<size_t>(n) * batch, int64_t(0));
    p.info = UnifiedVector<int32_t>(static_cast<size_t>(batch), int32_t(0));

    Rng rg(seed);
    for (int bi = 0; bi < batch; ++bi) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                const double re = rg.next(), im = rg.next();
                T v = mk<T>(re, im);
                if (i == j) v = mk<T>(4.0 * double(n) * (re >= 0 ? 1.0 : -1.0), 0.0);
                // A whole zero COLUMN, not a zero diagonal entry: partial pivoting
                // finds a nonzero pivot for a merely small diagonal, so only a zero
                // column makes U exactly singular at a predictable step.
                if (singular_col && j == (n / 2)) v = T{};
                p.a[size_t(bi) * p.stra + size_t(j) * p.lda + i] = v;
            }
        }
        for (int k = 0; k < nrhs; ++k)
            for (int i = 0; i < n; ++i)
                p.b[size_t(bi) * p.strb + size_t(k) * p.ldb + i] = mk<T>(rg.next(), rg.next());
    }
    p.a0.assign(p.a.begin(), p.a.end());
    p.b0.assign(p.b.begin(), p.b.end());
    reset(p);
    return p;
}

template <typename T>
MatrixView<T, MatrixFormat::Dense> a_view(Sys<T>& p) {
    return MatrixView<T, MatrixFormat::Dense>(p.a.data(), p.n, p.n, p.lda, p.stra, p.batch,
                                              p.aptr.data());
}
template <typename T>
MatrixView<T, MatrixFormat::Dense> b_view(Sys<T>& p) {
    return MatrixView<T, MatrixFormat::Dense>(p.b.data(), p.n, p.nrhs, p.ldb, p.strb, p.batch,
                                              p.bptr.data());
}

// ||A x - b||_F / (||A||_F ||x||_F) for ONE batch item, in double.
template <typename T>
double solve_residual(const Sys<T>& p, int item) {
    using D = std::complex<double>;
    const T* A0 = p.a0.data() + size_t(item) * p.stra;
    const T* B0 = p.b0.data() + size_t(item) * p.strb;
    const T* X = p.b.data() + size_t(item) * p.strb;

    double num = 0.0, an = 0.0, xn = 0.0;
    for (int j = 0; j < p.n; ++j)
        for (int i = 0; i < p.n; ++i) {
            const double m = habs(up(A0[size_t(j) * p.lda + i]));
            an += m * m;
        }
    for (int k = 0; k < p.nrhs; ++k)
        for (int i = 0; i < p.n; ++i) {
            const double m = habs(up(X[size_t(k) * p.ldb + i]));
            xn += m * m;
        }
    for (int k = 0; k < p.nrhs; ++k) {
        for (int i = 0; i < p.n; ++i) {
            D acc = up(B0[size_t(k) * p.ldb + i]);
            acc = D(-acc.real(), -acc.imag());
            for (int t = 0; t < p.n; ++t)
                acc += up(A0[size_t(t) * p.lda + i]) * up(X[size_t(k) * p.ldb + t]);
            num += habs(acc) * habs(acc);
        }
    }
    an = std::sqrt(an); xn = std::sqrt(xn);
    if (an == 0.0 || xn == 0.0) return std::sqrt(num);
    return std::sqrt(num) / (an * xn);
}

// Every element of the working buffer that no correct kernel may touch: the ld pad
// rows, the stride pad, and (for A) nothing else. Returns the first offender.
template <typename T>
bool pad_intact(const UnifiedVector<T>& live, const std::vector<T>& pristine,
                int n, int cols, int ld, int stride, int batch, size_t* where) {
    for (int bi = 0; bi < batch; ++bi) {
        for (size_t o = 0; o < size_t(stride); ++o) {
            const int col = int(o) / ld;
            const int row = int(o) % ld;
            const bool owned = (col < cols) && (row < n) && (int(o) < ld * cols);
            if (owned) continue;
            const size_t idx = size_t(bi) * stride + o;
            if (std::memcmp(&live[idx], &pristine[idx], sizeof(T)) != 0) {
                *where = idx;
                return false;
            }
        }
    }
    return true;
}

template <typename Config>
class GesvTest : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    static constexpr Backend BackendType = Config::BackendVal;

    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        if (this->HasFatalFailure() || ::testing::Test::IsSkipped()) return;
        if (!this->ctx) GTEST_SKIP() << "no queue";
        if (this->ctx->device().type != DeviceType::GPU)
            GTEST_SKIP() << "the fused gesv tier is GPU-only (route_gesv.hh)";
        if (!this->ctx->device().supports_sub_group_size(32))
            GTEST_SKIP() << "device does not offer sub-group size 32";
    }

    int cap() const { return sycl_gesv::gesv_tiny_max_n<T>(); }

    // Run the tiny tier on one prepared system, waiting for it.
    void run_tiny(Sys<T>& p) {
        auto A = a_view(p);
        auto B = b_view(p);
        const size_t need = sycl_gesv::gesv_tiny_buffer_size<T>(*this->ctx, A, B);
        UnifiedVector<std::byte> ws(need > 0 ? need : size_t(1));
        (void)sycl_gesv::gesv_tiny_dispatch<T>(*this->ctx, A, B, p.piv.to_span(),
                                         Span<std::byte>(ws.data(), need),
                                         p.info.to_span());
        this->ctx->wait();
    }
};

using GesvTestTypes = typename test_utils::backend_types<GesvConfig>::type;

}  // namespace

TYPED_TEST_SUITE(GesvTest, GesvTestTypes);

// G1. The residual over the whole order ladder including the padded buckets (9, 17,
// 24) and every instantiated nrhs, on items 0 AND batch-1.
// ARMED BREAK (R9): in gesv_tiny.cc step 6 change `tiny_select(rowid < i, upd, rB[k])`
// to `rowid < i - 1`. EXPECTED RED at every n >= 2, residual of order 1 rather than
// n*eps, with G2 still GREEN -- which is the point of having both.
TYPED_TEST(GesvTest, TinySolveResidualMatchesHostReference) {
    using T = typename TestFixture::T;
    const int cap = this->cap();
    for (int n : {1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 24, 31, 32}) {
        if (n > cap) continue;
        for (int nrhs : {1, 2, 3, 4}) {
            auto p = make_system<T>(n, nrhs, 9, 1234u + unsigned(n * 7 + nrhs));
            this->run_tiny(p);
            for (int item : {0, p.batch - 1}) {
                EXPECT_EQ(p.info[item], 0) << "n=" << n << " nrhs=" << nrhs;
                const double r = solve_residual(p, item);
                EXPECT_LT(r, solve_tol<T>(n))
                    << "n=" << n << " nrhs=" << nrhs << " item=" << item;
            }
        }
    }
}

// G2. THE CONTRACT solve_native.hh states: gesv leaves behind exactly what getrf
// would, bit-for-bit, because both run the same recurrence in the same order.
// ARMED BREAK (R9): in gesv_tiny.cc step 5 drop the `if (act)` guard so the pivot
// lane updates its own RHS. EXPECTED: G1 RED, THIS test GREEN -- the pair is what
// pins down which half broke.
TYPED_TEST(GesvTest, TinyFactorAndPivotsAreBitIdenticalToGetrf) {
    using T = typename TestFixture::T;
    const int cap = this->cap();
    for (int n : {4, 8, 9, 16, 17, 32}) {
        if (n > sycl_getrf::getrf_tiny_max_n<T>() || n > cap) continue;
        auto p = make_system<T>(n, 2, 5, 777u + unsigned(n));
        auto q = make_system<T>(n, 2, 5, 777u + unsigned(n));   // same seed, same data

        this->run_tiny(p);

        auto Aq = a_view(q);
        const size_t need = sycl_getrf::getrf_tiny_buffer_size<T>(*this->ctx, Aq);
        UnifiedVector<std::byte> ws(need > 0 ? need : size_t(1));
        (void)sycl_getrf::getrf_tiny_dispatch<T>(*this->ctx, Aq, q.piv.to_span(),
                                           Span<std::byte>(ws.data(), need),
                                           q.info.to_span());
        this->ctx->wait();

        for (size_t i = 0; i < p.a.size(); ++i) {
            ASSERT_EQ(0, std::memcmp(&p.a[i], &q.a[i], sizeof(T)))
                << "factor differs at element " << i << " for n=" << n;
        }
        for (size_t i = 0; i < p.piv.size(); ++i) {
            ASSERT_EQ(p.piv[i], q.piv[i]) << "ipiv differs at " << i << " for n=" << n;
        }
        for (int bi = 0; bi < p.batch; ++bi) ASSERT_EQ(p.info[bi], q.info[bi]);
    }
}

// G3. `info` on an exactly singular U, and the rule that X stays FINITE: LAPACK
// leaves X undefined when info > 0, and this tier writes zeros rather than infinities.
// ARMED BREAK (R9): in gesv_tiny.cc step 6 replace `tiny_select(zero, D{}, ...)` with
// the bare division. EXPECTED: finiteness RED, info GREEN.
TYPED_TEST(GesvTest, TinyInfoReportsExactSingularityAndLeavesXFinite) {
    using T = typename TestFixture::T;
    const int cap = this->cap();
    for (int n : {4, 8, 16}) {
        if (n > cap) continue;
        auto p = make_system<T>(n, 1, 4, 31u + unsigned(n), 5, 11, /*singular_col=*/true);
        this->run_tiny(p);
        for (int bi = 0; bi < p.batch; ++bi) {
            EXPECT_EQ(p.info[bi], n / 2 + 1)
                << "n=" << n << ": the zero column is " << (n / 2) << ", 1-based " << (n / 2 + 1);
        }
        for (size_t i = 0; i < p.b.size(); ++i) {
            ASSERT_TRUE(hfinite(up(p.b[i]))) << "X is not finite at element " << i;
        }
    }
}

// G4. The ld pad, the stride pad and the nrhs pad: `ld = n + 5` with a stride that is
// not ld*cols means an off-by-one in either store lands in a poisoned cell.
// ARMED BREAK (R9): in gesv_tiny.cc's store change the RHS loop's `if (k >= nrhs)` to
// `if (k >= NR)`. EXPECTED RED at nrhs = 3 (it runs in the NR = 4 instantiation) and
// GREEN at 1 and 4, which is why the nrhs = 3 row must be present.
TYPED_TEST(GesvTest, TinyWritesNothingOutsideItsOwnExtents) {
    using T = typename TestFixture::T;
    const int cap = this->cap();
    for (int n : {5, 8, 17}) {
        if (n > cap) continue;
        for (int nrhs : {1, 3, 4}) {
            auto p = make_system<T>(n, nrhs, 6, 55u + unsigned(n * 5 + nrhs));
            this->run_tiny(p);
            size_t where = 0;
            EXPECT_TRUE(pad_intact(p.a, p.a0, n, n, p.lda, p.stra, p.batch, &where))
                << "A pad written at " << where << " (n=" << n << ")";
            EXPECT_TRUE(pad_intact(p.b, p.b0, n, nrhs, p.ldb, p.strb, p.batch, &where))
                << "B pad written at " << where << " (n=" << n << " nrhs=" << nrhs << ")";
        }
    }
}

// G5. PACKED LAUNCHES MUST NOT BLEED: 32/N matrices per sub-group and two sub-groups
// per work-group, so a batch that is not a multiple of the packing exercises the
// clamped tail lanes and the neighbour partitions together.
// ARMED BREAK (R9): drop the `sg_id *` term in tiny_device.hh's tiny_partition_id.
// EXPECTED RED at N = 8, half the batch never touched and its `info` still holding
// the -12345 poison -- which is why this asserts on info and not only the residual.
TYPED_TEST(GesvTest, TinyPackedLaunchCoversEveryBatchItem) {
    using T = typename TestFixture::T;
    const int cap = this->cap();
    for (int n : {8, 16}) {
        if (n > cap) continue;
        for (int batch : {1, 3, 7, 9, 17, 33}) {
            auto p = make_system<T>(n, 1, batch, 909u + unsigned(n * 100 + batch));
            this->run_tiny(p);
            for (int bi = 0; bi < batch; ++bi) {
                ASSERT_EQ(p.info[bi], 0)
                    << "item " << bi << " of " << batch << " was not written (n=" << n << ")";
                EXPECT_LT(solve_residual(p, bi), solve_tol<T>(n))
                    << "item " << bi << " of " << batch;
            }
        }
    }
}

// G6. The launcher re-applies every supports() gate, because there is no vendor to
// fall through to: an unservable shape must throw, never silently solve a leading
// submatrix or a truncated RHS.
// ARMED BREAK (R9): delete the `rbucket < 1` test in gesv_tiny_dispatch. EXPECTED:
// the nrhs = 5 case RED, throwing "no instantiation" instead of `unsupported`.
TYPED_TEST(GesvTest, TinyRefusesShapesAboveItsCeilings) {
    using T = typename TestFixture::T;
    const int cap = this->cap();

    auto wide = make_system<T>(8, 5, 2, 3u);
    auto Aw = a_view(wide);
    auto Bw = b_view(wide);
    UnifiedVector<std::byte> ws(size_t(1024));
    EXPECT_THROW((void)sycl_gesv::gesv_tiny_dispatch<T>(*this->ctx, Aw, Bw, wide.piv.to_span(),
                                                  ws.to_span(), wide.info.to_span()),
                 batchlas::unsupported);

    auto big = make_system<T>(cap + 1, 1, 2, 4u);
    auto Ab = a_view(big);
    auto Bb = b_view(big);
    EXPECT_THROW((void)sycl_gesv::gesv_tiny_dispatch<T>(*this->ctx, Ab, Bb, big.piv.to_span(),
                                                  ws.to_span(), big.info.to_span()),
                 batchlas::unsupported);
}

// G7. THE ROUTE, pinned to the MEASURED window and to nothing else. float takes the
// fused tier over the whole ladder it fits; cfloat stops at 16 because n = 17 is a
// measured LOSS (0.45-0.84x of the composed arm); double and cdouble have no window.
// The n = 64 row is the structural bracket at the top: the tier does not fit there at
// any type, so the composed arm must answer. evidence: docs/perf/lu.md#p2-the-measured-gesv-window
//
// preferred() is asserted all-false DELIBERATELY and permanently: this op passes
// vendor_available=false, so resolve_route never reaches preferred() -- the window
// lives in native_tier_preferred. A window written into preferred() here would be the
// R8b defect with no compensating effect.
//
// ARMED BREAK (R9): widen route_gesv.hh's tiny_window_max_n for cfloat from 16 to 32.
// EXPECTED RED at n = 17 and 32 for cfloat only, reporting tiny where blocked is
// expected, with float and the n = 64 row GREEN.
TYPED_TEST(GesvTest, AutoTakesTheMeasuredWindow) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::BackendType;
    using Tbl = dispatch::RouteTable<dispatch::Op::gesv, T>;

    // The window as the grid measured it, restated here rather than read back from the
    // header: a test that asks the header what the header says cannot fail.
    const int win = std::is_same_v<T, float>                ? 32
                  : std::is_same_v<T, std::complex<float>>  ? 16
                                                            : 0;

    for (int n : {4, 8, 16, 17, 32, 64}) {
        auto p = make_system<T>(n, 1, 4, 11u + unsigned(n));
        auto A = a_view(p);
        auto Bv = b_view(p);
        const auto shape = backend::gesv_op_shape<B, T>(*this->ctx, A, Bv);
        ASSERT_TRUE(shape.has_value()) << "n=" << n;

        EXPECT_FALSE(Tbl::preferred({dispatch::Origin::Native, dispatch::Algorithm::Tiny},
                                    *shape))
            << "n=" << n << ": preferred() is not this op's shipping hook";
        EXPECT_FALSE(Tbl::preferred({dispatch::Origin::Native, dispatch::Algorithm::Blocked},
                                    *shape));

        const bool fits = (n <= sycl_gesv::gesv_tiny_max_n<T>());
        const auto want = (fits && n <= win) ? dispatch::Algorithm::Tiny
                                             : dispatch::Algorithm::Blocked;
        const auto r = backend::gesv_route<B, T>(*this->ctx, A, Bv);
        EXPECT_EQ(r.algo, want)
            << "n=" << n << ": Auto resolved to "
            << std::string(dispatch::to_string(r.algo));
        EXPECT_EQ(r.origin, dispatch::Origin::Native);
    }
}

// G8. supports() is a CORRECTNESS predicate: it must admit exactly the shapes the
// launcher accepts, or a forced route reaches an entry point that throws.
TYPED_TEST(GesvTest, SupportsAgreesWithTheLauncherCeilings) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::BackendType;
    using Tbl = dispatch::RouteTable<dispatch::Op::gesv, T>;
    const dispatch::Route tiny{dispatch::Origin::Native, dispatch::Algorithm::Tiny};
    const int cap = this->cap();

    for (int n : {1, 8, 16, 32, 33, 64}) {
        for (int nrhs : {1, 4, 5}) {
            auto p = make_system<T>(n, nrhs, 2, 21u);
            auto A = a_view(p);
            auto Bv = b_view(p);
            const auto shape = backend::gesv_op_shape<B, T>(*this->ctx, A, Bv);
            ASSERT_TRUE(shape.has_value());
            const bool want = (n <= cap) && (nrhs <= sycl_gesv::kGesvTinyMaxRhs);
            EXPECT_EQ(Tbl::supports(tiny, *shape), want)
                << "n=" << n << " nrhs=" << nrhs;
        }
    }
}

// G9. THE PUBLIC OP end to end against the same residual bound and the same pivot
// list as the tiny tier on the same data.
//
// THE COMPOSED ARM IS PINNED. Since the window landed, Auto sends float and cfloat at
// these orders to the fused tier, so an unpinned public call would be compared against
// itself -- the guard would still be green with the composition arbitrarily wrong.
// Pinning makes this a COMPOSED-vs-FUSED agreement test, which is what it was for.
TYPED_TEST(GesvTest, PublicGesvSolvesAndMatchesTheTinyTier) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::BackendType;
    const int cap = this->cap();
    const ScopedEnvVar pin("BATCHLAS_GESV_ROUTE", "blocked");

    for (int n : {4, 16, 32}) {
        if (n > cap) continue;
        auto p = make_system<T>(n, 2, 6, 4242u + unsigned(n));
        auto q = make_system<T>(n, 2, 6, 4242u + unsigned(n));

        auto A = a_view(p);
        auto Bv = b_view(p);
        const size_t need = gesv_buffer_size<B, T>(*this->ctx, A, Bv);
        UnifiedVector<std::byte> ws(need > 0 ? need : size_t(1));
        (void)gesv<B, T>(*this->ctx, A, Bv, p.piv.to_span(), Span<std::byte>(ws.data(), need),
                   p.info.to_span());
        this->ctx->wait();

        this->run_tiny(q);

        for (int item : {0, p.batch - 1}) {
            EXPECT_EQ(p.info[item], 0);
            EXPECT_LT(solve_residual(p, item), solve_tol<T>(n)) << "public, n=" << n;
            EXPECT_LT(solve_residual(q, item), solve_tol<T>(n)) << "tiny, n=" << n;
        }
        // The pivot lists must agree exactly: both arms run the same getrf recurrence.
        for (size_t i = 0; i < p.piv.size(); ++i) ASSERT_EQ(p.piv[i], q.piv[i]) << i;
    }
}

// G10. The validator rejects rather than routes: with no vendor arm, a non-conforming
// pair would otherwise surface as "no vendor library for this op".
TYPED_TEST(GesvTest, PublicGesvRejectsNonConformingViews) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::BackendType;

    auto p = make_system<T>(8, 1, 3, 6u);
    auto A = a_view(p);

    // B with the wrong number of rows.
    UnifiedVector<T> bb(size_t(4 * 1 * 3), T{});
    UnifiedVector<T*> bp(size_t(3), nullptr);
    MatrixView<T, MatrixFormat::Dense> Bad(bb.data(), 4, 1, 4, 4, 3, bp.data());
    EXPECT_THROW((gesv_buffer_size<B, T>(*this->ctx, A, Bad)), batchlas::invalid_argument);

    // A non-square.
    MatrixView<T, MatrixFormat::Dense> NotSq(p.a.data(), 8, 4, p.lda, p.stra, 3, p.aptr.data());
    auto Bv = b_view(p);
    EXPECT_THROW((gesv_buffer_size<B, T>(*this->ctx, NotSq, Bv)), batchlas::invalid_argument);
}
