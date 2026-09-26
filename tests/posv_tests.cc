// P2: the fused factor-and-solve tier for POSV, and the routing that (deliberately)
// does not yet reach it. There is no batched vendor posv and no `potrs` op, so the
// only oracles here are a host residual and potrf itself.
// evidence: docs/perf/potrf.md#the-fused-posv-tier
#include <gtest/gtest.h>

#include <batchlas/blas/functions/posv.hh>
#include <batchlas/blas/functions/potrf.hh>
#include <batchlas/blas/linalg-ops.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>

#include "test_utils.hh"

#include "../src/extensions/solve_native.hh"
#include "../src/extensions/potrf_native.hh"
#include "../src/backends/posv_route.hh"
#include "../src/backends/potrf_route.hh"

#include <batchlas/blas/dispatch/vendor_available.hh>

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

inline double up(float x) { return double(x); }
inline double up(double x) { return x; }
inline std::complex<double> up(std::complex<float> x) { return {double(x.real()), double(x.imag())}; }
inline std::complex<double> up(std::complex<double> x) { return x; }

inline double hconj(double x) { return x; }
inline std::complex<double> hconj(std::complex<double> x) { return std::conj(x); }
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

// Cholesky is backward stable and the matrices below are diagonally dominant with
// cond(A) = O(1), so the bound scales with n * eps.
template <typename T> double solve_tol(int n) { return 400.0 * double(n) * eps_of<T>(); }

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 6364136223846793005ULL + 1442695040888963407ULL) {}
    double next() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return double(int32_t(uint32_t(s >> 32))) / 2147483648.0;
    }
};

template <typename T, Backend B>
struct PosvConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

template <typename T>
struct Sys {
    int n = 0, nrhs = 0, batch = 0;
    int lda = 0, stra = 0, ldb = 0, strb = 0;
    UnifiedVector<T> a, b;
    std::vector<T> a0, b0;
    UnifiedVector<T*> aptr, bptr;
    UnifiedVector<int32_t> info;
};

// A Hermitian positive-definite A with a padded ld, a stride that is not ld*cols and
// a POISONED pad. THE UNREFERENCED TRIANGLE IS POISONED TOO, which is what makes
// "the other triangle is neither read nor written" falsifiable: if the kernel read
// it, every residual would blow up; if it wrote it, the pad check catches it.
template <typename T>
Sys<T> make_spd(int n, int nrhs, int batch, Uplo uplo, unsigned seed,
                bool indefinite = false, bool zero_minor = false) {
    Sys<T> p;
    p.n = n; p.nrhs = nrhs; p.batch = batch;
    p.lda = n + 5;  p.stra = p.lda * n + 11;
    p.ldb = n + 5;  p.strb = p.ldb * nrhs + 11;

    const T poison = mk<T>(-9.75e3, 4.5e3);
    p.a = UnifiedVector<T>(static_cast<size_t>(p.stra) * batch, poison);
    p.b = UnifiedVector<T>(static_cast<size_t>(p.strb) * batch, poison);
    p.aptr = UnifiedVector<T*>(static_cast<size_t>(batch), nullptr);
    p.bptr = UnifiedVector<T*>(static_cast<size_t>(batch), nullptr);
    p.info = UnifiedVector<int32_t>(static_cast<size_t>(batch), int32_t(-12345));

    Rng rg(seed);
    for (int bi = 0; bi < batch; ++bi) {
        // Build the full Hermitian matrix first, then copy only the owned triangle
        // in: the two triangles must be exact conjugates or the "Upper == Lower"
        // comparison below would be measuring the data, not the kernel.
        std::vector<std::complex<double>> full(size_t(n) * n);
        for (int j = 0; j < n; ++j)
            for (int i = j; i < n; ++i) {
                std::complex<double> v(rg.next(), (i == j) ? 0.0 : rg.next());
                if (i == j) v = std::complex<double>(4.0 * double(n), 0.0);
                full[size_t(j) * n + i] = v;
                full[size_t(i) * n + j] = std::conj(v);
            }
        // A single negative diagonal entry makes the leading minor at that step
        // non-positive-definite at a PREDICTABLE step, which is what `info` must name.
        if (indefinite) full[size_t(n / 2) * n + (n / 2)] = std::complex<double>(-1.0, 0.0);
        // The OTHER failure shape, and the only one that reaches the kernel's
        // divide-by-zero guard: zeroing row and column n/2 entirely leaves the
        // leading minor of order n/2 positive definite and makes the updated
        // diagonal at step n/2 EXACTLY zero (every L(n/2, c<n/2) is zero too), so
        // `dev_is_zero` fires where a merely negative entry never does.
        if (zero_minor)
            for (int t = 0; t < n; ++t) {
                full[size_t(n / 2) * n + t] = std::complex<double>(0.0, 0.0);
                full[size_t(t) * n + (n / 2)] = std::complex<double>(0.0, 0.0);
            }

        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                const bool owned = (uplo == Uplo::Lower) ? (i >= j) : (i <= j);
                if (!owned) continue;
                const auto v = full[size_t(j) * n + i];
                p.a[size_t(bi) * p.stra + size_t(j) * p.lda + i] = mk<T>(v.real(), v.imag());
            }
        for (int k = 0; k < nrhs; ++k)
            for (int i = 0; i < n; ++i)
                p.b[size_t(bi) * p.strb + size_t(k) * p.ldb + i] = mk<T>(rg.next(), rg.next());
    }
    p.a0.assign(p.a.begin(), p.a.end());
    p.b0.assign(p.b.begin(), p.b.end());
    return p;
}

template <typename T>
void reset(Sys<T>& p) {
    std::copy(p.a0.begin(), p.a0.end(), p.a.begin());
    std::copy(p.b0.begin(), p.b0.end(), p.b.begin());
    std::fill(p.info.begin(), p.info.end(), int32_t(-12345));
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

// ||A x - b|| / (||A|| ||x||), with A reconstructed HERMITIAN from the owned triangle
// of the pristine input.
template <typename T>
double solve_residual(const Sys<T>& p, int item, Uplo uplo) {
    using D = std::complex<double>;
    const T* A0 = p.a0.data() + size_t(item) * p.stra;
    const T* B0 = p.b0.data() + size_t(item) * p.strb;
    const T* X = p.b.data() + size_t(item) * p.strb;

    auto at = [&](int i, int j) -> D {
        const bool owned = (uplo == Uplo::Lower) ? (i >= j) : (i <= j);
        if (owned) return up(A0[size_t(j) * p.lda + i]);
        return hconj(up(A0[size_t(i) * p.lda + j]));
    };

    double num = 0.0, an = 0.0, xn = 0.0;
    for (int j = 0; j < p.n; ++j)
        for (int i = 0; i < p.n; ++i) { const double m = habs(at(i, j)); an += m * m; }
    for (int k = 0; k < p.nrhs; ++k)
        for (int i = 0; i < p.n; ++i) {
            const double m = habs(up(X[size_t(k) * p.ldb + i])); xn += m * m;
        }
    for (int k = 0; k < p.nrhs; ++k)
        for (int i = 0; i < p.n; ++i) {
            D acc = up(B0[size_t(k) * p.ldb + i]);
            acc = D(-acc.real(), -acc.imag());
            for (int t = 0; t < p.n; ++t) acc += at(i, t) * up(X[size_t(k) * p.ldb + t]);
            num += habs(acc) * habs(acc);
        }
    an = std::sqrt(an); xn = std::sqrt(xn);
    if (an == 0.0 || xn == 0.0) return std::sqrt(num);
    return std::sqrt(num) / (an * xn);
}

// Everything the kernel may not write: the ld pad, the stride pad, AND the triangle
// the caller did not give it.
template <typename T>
bool untouched_outside_triangle(const Sys<T>& p, Uplo uplo, size_t* where) {
    for (int bi = 0; bi < p.batch; ++bi)
        for (size_t o = 0; o < size_t(p.stra); ++o) {
            const int col = int(o) / p.lda;
            const int row = int(o) % p.lda;
            const bool in_matrix = (col < p.n) && (row < p.n) && (int(o) < p.lda * p.n);
            const bool owned = in_matrix &&
                               ((uplo == Uplo::Lower) ? (row >= col) : (row <= col));
            if (owned) continue;
            const size_t idx = size_t(bi) * p.stra + o;
            if (std::memcmp(&p.a[idx], &p.a0[idx], sizeof(T)) != 0) { *where = idx; return false; }
        }
    return true;
}

template <typename Config>
class PosvTest : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    static constexpr Backend BackendType = Config::BackendVal;

    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        if (this->HasFatalFailure() || ::testing::Test::IsSkipped()) return;
        if (!this->ctx) GTEST_SKIP() << "no queue";
        if (this->ctx->device().type != DeviceType::GPU)
            GTEST_SKIP() << "the fused posv tier is GPU-only (route_posv.hh)";
        if (!this->ctx->device().supports_sub_group_size(32))
            GTEST_SKIP() << "device does not offer sub-group size 32";
    }

    int cap() const { return sycl_posv::posv_tiny_max_n<T>(); }

    void run_tiny(Sys<T>& p, Uplo uplo) {
        auto A = a_view(p);
        auto B = b_view(p);
        const size_t need = sycl_posv::posv_tiny_buffer_size<T>(*this->ctx, A, B);
        UnifiedVector<std::byte> ws(need > 0 ? need : size_t(1));
        (void)sycl_posv::posv_tiny_dispatch<T>(*this->ctx, A, B, uplo,
                                         Span<std::byte>(ws.data(), need), p.info.to_span());
        this->ctx->wait();
    }
};

using PosvTestTypes = typename test_utils::backend_types<PosvConfig>::type;

}  // namespace

TYPED_TEST_SUITE(PosvTest, PosvTestTypes);

// P1. The residual, both triangles, the whole order ladder including the padded
// buckets, every instantiated nrhs, items 0 AND batch-1.
//
// ARMED BREAK (R9): in posv_tiny.cc step 3, bound the transpose-read loop by
// `c >= i - 1` instead of `c >= i`, dropping the last column of the backward
// update. EXPECTED: RED at every n >= 2 with a residual of order 1, and P2 (the
// factor comparison) GREEN -- which localises the break to the solve.
TYPED_TEST(PosvTest, TinySolveResidualMatchesHostReference) {
    using T = typename TestFixture::T;
    const int cap = this->cap();
    for (Uplo uplo : {Uplo::Lower, Uplo::Upper}) {
        for (int n : {1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 24, 31, 32}) {
            if (n > cap) continue;
            for (int nrhs : {1, 2, 3, 4}) {
                auto p = make_spd<T>(n, nrhs, 9, uplo, 555u + unsigned(n * 7 + nrhs));
                this->run_tiny(p, uplo);
                for (int item : {0, p.batch - 1}) {
                    EXPECT_EQ(p.info[item], 0) << "n=" << n << " nrhs=" << nrhs;
                    EXPECT_LT(solve_residual(p, item, uplo), solve_tol<T>(n))
                        << "n=" << n << " nrhs=" << nrhs << " item=" << item
                        << " uplo=" << (uplo == Uplo::Lower ? "L" : "U");
                }
            }
        }
    }
}

// P2. The factor posv leaves behind must be bit-for-bit what potrf leaves behind:
// posv_tiny.cc's step 1 is a verbatim copy of potrf_tiny.cc's recurrence, so any
// difference means one of them reordered the arithmetic.
//
// ARMED BREAK (R9): in posv_tiny.cc step 1 change `rinv = R(1) / dkk` to
// `sycl::rsqrt(akk)`. EXPECTED: RED here (the factors differ in the last bits) while
// P1 stays GREEN, because rsqrt.approx is still accurate enough for the residual
// bound -- which is exactly why the bit-exact comparison has to exist.
TYPED_TEST(PosvTest, TinyFactorIsBitIdenticalToPotrf) {
    using T = typename TestFixture::T;
    const int cap = this->cap();
    for (Uplo uplo : {Uplo::Lower, Uplo::Upper}) {
        for (int n : {4, 8, 9, 16, 17, 32}) {
            if (n > cap || n > sycl_potrf::potrf_tiny_max_n<T>()) continue;
            auto p = make_spd<T>(n, 2, 5, uplo, 888u + unsigned(n));
            auto q = make_spd<T>(n, 2, 5, uplo, 888u + unsigned(n));

            this->run_tiny(p, uplo);

            auto Aq = a_view(q);
            const size_t need = sycl_potrf::potrf_tiny_buffer_size<T>(*this->ctx, Aq);
            UnifiedVector<std::byte> ws(need > 0 ? need : size_t(1));
            (void)sycl_potrf::potrf_tiny_dispatch<T>(*this->ctx, Aq, uplo,
                                               Span<std::byte>(ws.data(), need),
                                               q.info.to_span());
            this->ctx->wait();

            for (size_t i = 0; i < p.a.size(); ++i)
                ASSERT_EQ(0, std::memcmp(&p.a[i], &q.a[i], sizeof(T)))
                    << "factor differs at " << i << " for n=" << n;
            for (int bi = 0; bi < p.batch; ++bi) ASSERT_EQ(p.info[bi], q.info[bi]);
        }
    }
}

// P3. `info` on a non-positive-definite leading minor, and X stays FINITE. TWO
// failure shapes: a NEGATIVE diagonal leaves a non-zero divisor, so only the zeroed
// row and column reaches the kernel's divide-by-zero guard at all.
// ARMED BREAK (R9): replace either solve's `tiny_select(zero, D{}, ...)` with the bare
// division. EXPECTED: finiteness RED on the zero_minor rows only, info GREEN.
TYPED_TEST(PosvTest, TinyInfoReportsTheLeadingMinorAndLeavesXFinite) {
    using T = typename TestFixture::T;
    const int cap = this->cap();
    for (int shape = 0; shape < 2; ++shape) {
        const bool zero_minor = (shape == 1);
        for (int n : {4, 8, 16}) {
            if (n > cap) continue;
            auto p = make_spd<T>(n, 1, 4, Uplo::Lower, 17u + unsigned(n),
                                 /*indefinite=*/!zero_minor, zero_minor);
            this->run_tiny(p, Uplo::Lower);
            for (int bi = 0; bi < p.batch; ++bi)
                EXPECT_EQ(p.info[bi], n / 2 + 1) << "n=" << n << " zero_minor=" << zero_minor;
            for (size_t i = 0; i < p.b.size(); ++i)
                ASSERT_TRUE(hfinite(up(p.b[i])))
                    << "X is not finite at " << i << " (n=" << n
                    << " zero_minor=" << zero_minor << ")";
        }
    }
}

// P4. THE OTHER TRIANGLE IS NEITHER READ NOR WRITTEN (ortho.cc's contract), plus the
// ld and stride pads; the unreferenced triangle is poisoned, so a read of it would
// also destroy P1's residual.
// ARMED BREAK (R9): in tiny_device.hh's tiny_store_lower change `c <= lane` to `c < N`.
// EXPECTED: RED for Uplo::Lower at every n.
TYPED_TEST(PosvTest, TinyWritesOnlyItsOwnTriangle) {
    using T = typename TestFixture::T;
    const int cap = this->cap();
    for (Uplo uplo : {Uplo::Lower, Uplo::Upper}) {
        for (int n : {5, 8, 17}) {
            if (n > cap) continue;
            auto p = make_spd<T>(n, 3, 6, uplo, 66u + unsigned(n));
            this->run_tiny(p, uplo);
            size_t where = 0;
            EXPECT_TRUE(untouched_outside_triangle(p, uplo, &where))
                << "A written outside its triangle at " << where << " (n=" << n << ")";
        }
    }
}

// P5. Packed launches must cover every batch item; see gesv_tests.cc G5 for the
// break this arms (tiny_device.hh's tiny_partition_id).
TYPED_TEST(PosvTest, TinyPackedLaunchCoversEveryBatchItem) {
    using T = typename TestFixture::T;
    const int cap = this->cap();
    for (int n : {8, 16}) {
        if (n > cap) continue;
        for (int batch : {1, 3, 7, 9, 17, 33}) {
            auto p = make_spd<T>(n, 1, batch, Uplo::Lower, 303u + unsigned(n * 100 + batch));
            this->run_tiny(p, Uplo::Lower);
            for (int bi = 0; bi < batch; ++bi) {
                ASSERT_EQ(p.info[bi], 0)
                    << "item " << bi << " of " << batch << " was not written";
                EXPECT_LT(solve_residual(p, bi, Uplo::Lower), solve_tol<T>(n))
                    << "item " << bi << " of " << batch;
            }
        }
    }
}

// P6. Every supports() gate is re-applied at the launcher, because there is no
// vendor to fall through to. See gesv_tests.cc G6 for the armed break.
TYPED_TEST(PosvTest, TinyRefusesShapesAboveItsCeilings) {
    using T = typename TestFixture::T;
    const int cap = this->cap();
    UnifiedVector<std::byte> ws(size_t(1024));

    auto wide = make_spd<T>(8, 5, 2, Uplo::Lower, 3u);
    auto Aw = a_view(wide); auto Bw = b_view(wide);
    EXPECT_THROW((void)sycl_posv::posv_tiny_dispatch<T>(*this->ctx, Aw, Bw, Uplo::Lower,
                                                  ws.to_span(), wide.info.to_span()),
                 batchlas::unsupported);

    auto big = make_spd<T>(cap + 1, 1, 2, Uplo::Lower, 4u);
    auto Ab = a_view(big); auto Bb = b_view(big);
    EXPECT_THROW((void)sycl_posv::posv_tiny_dispatch<T>(*this->ctx, Ab, Bb, Uplo::Lower,
                                                  ws.to_span(), big.info.to_span()),
                 batchlas::unsupported);
}

// P7. THE ROUTE, pinned to the MEASURED window; preferred() is asserted all-false
// permanently because this op passes vendor_available=false and the window therefore
// lives in native_tier_preferred. Outside the tiny window the fused-solve CTA arm takes
// every shape it can hold. evidence: docs/perf/potrf.md#the-fused-potrs-solve
// ARMED BREAK (R9): make route_posv.hh's cfloat tiny_window return `order() <= 16`.
// EXPECTED: RED for cfloat at n = 9 and 16 with nrhs = 1 only.
TYPED_TEST(PosvTest, AutoTakesTheMeasuredWindow) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::BackendType;
    using Tbl = dispatch::RouteTable<dispatch::Op::posv, T>;

    // Restated, not read back from the header: a test that asks the header what the
    // header says cannot fail. The tiny window's ceiling for narrow and for wide RHS.
    constexpr bool kF = std::is_same_v<T, float>;
    constexpr bool kC = std::is_same_v<T, std::complex<float>>;
    constexpr bool kZ = std::is_same_v<T, std::complex<double>>;
    const int narrow = kF ? 16 : kC ? 8 : kZ ? 16 : 32;
    const int wide = kF ? 32 : kC ? 16 : kZ ? 16 : 32;

    for (int nrhs : {1, 4}) {
        for (int n : {4, 8, 9, 16, 17, 32, 64}) {
            auto p = make_spd<T>(n, nrhs, 4, Uplo::Lower, 12u + unsigned(n));
            auto A = a_view(p); auto Bv = b_view(p);
            const auto shape = backend::posv_op_shape<B, T>(*this->ctx, A, Bv, Uplo::Lower);
            ASSERT_TRUE(shape.has_value()) << "n=" << n;

            EXPECT_FALSE(Tbl::preferred({dispatch::Origin::Native, dispatch::Algorithm::Tiny},
                                        *shape))
                << "n=" << n << ": preferred() is not this op's shipping hook";

            const bool fits = (n <= sycl_posv::posv_tiny_max_n<T>());
            const int win = (nrhs >= 3) ? wide : narrow;
            const auto want = (fits && n <= win) ? dispatch::Algorithm::Tiny
                                                 : dispatch::Algorithm::CTA;
            const auto r = backend::posv_route<B, T>(*this->ctx, A, Bv, Uplo::Lower);
            EXPECT_EQ(r.algo, want)
                << "n=" << n << " nrhs=" << nrhs << ": Auto resolved to "
                << std::string(dispatch::to_string(r.algo));
            EXPECT_EQ(r.origin, dispatch::Origin::Native);
        }
    }
}

// P7b. The fused-solve CTA arm, pinned, on both triangles. n = 17 and 100 leave a short
// final nb block, n = 33 and 64 are whole blocks; nrhs covers every accumulator bucket.
// Batch 96 gives a missing barrier many resident work-groups to race against.
// potrf is pinned NATIVE: cuSOLVER's Upper potrf overwrites the unreferenced lower
// triangle, and with the poison gone a wrong-triangle read in the solve passes green.
// ARMED BREAK (R9): swap ld_a/ld_h in potrs_fused_launch's backward staging.
// EXPECTED: RED for every n > 1 on Lower, and on Upper wherever potrf ran native.
TYPED_TEST(PosvTest, FusedSolveArmSolvesOnBothTriangles) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::BackendType;
    const ScopedEnvVar pin("BATCHLAS_POSV_ROUTE", "cta");
    const ScopedEnvVar pin_potrf("BATCHLAS_POTRF_ROUTE", "native");

    for (Uplo uplo : {Uplo::Lower, Uplo::Upper}) {
        for (int nrhs : {1, 2, 3, 8}) {
            for (int n : {1, 17, 33, 64, 100}) {
                auto p = make_spd<T>(n, nrhs, 96, uplo, 4242u + unsigned(n * 16 + nrhs));
                auto A = a_view(p); auto Bv = b_view(p);
                const auto r = backend::posv_route<B, T>(*this->ctx, A, Bv, uplo);
                ASSERT_EQ(r.algo, dispatch::Algorithm::CTA)
                    << "the pin fell through at n=" << n << " nrhs=" << nrhs;
                const size_t need = posv_buffer_size<B, T>(*this->ctx, A, Bv, uplo);
                UnifiedVector<std::byte> ws(need > 0 ? need : size_t(1));
                (void)posv<B, T>(*this->ctx, A, Bv, uplo, Span<std::byte>(ws.data(), need),
                           p.info.to_span());
                this->ctx->wait();

                for (int item : {0, p.batch / 2, p.batch - 1}) {
                    EXPECT_EQ(p.info[item], 0) << "n=" << n;
                    EXPECT_LT(solve_residual(p, item, uplo), solve_tol<T>(n))
                        << "n=" << n << " nrhs=" << nrhs << " item=" << item
                        << " uplo=" << (uplo == Uplo::Lower ? "L" : "U");
                }
                // The poison in the other triangle is what makes a wrong-triangle READ
                // in the solve visible, so a native potrf must leave it in place.
                const auto pr = backend::potrf_route<B, T>(
                    *this->ctx, A, uplo, dispatch::factorization_vendor_available<B>);
                if (dispatch::is_native(pr)) {
                    size_t where = 0;
                    EXPECT_TRUE(untouched_outside_triangle(p, uplo, &where))
                        << "n=" << n << " uplo=" << (uplo == Uplo::Lower ? "L" : "U")
                        << ": first changed element at " << where;
                }
            }
        }
    }
}

// P8. The public op on the COMPOSED arm, both triangles: the only test that exercises
// the Upper composition's transpose arguments. THE ROUTE IS PINNED because Auto now
// sends n <= 32 to the fused tier, which would leave this guard covering n = 33 alone.
// ARMED BREAK (R9): swap factorization.cc's two Upper trsm calls.
// EXPECTED: RED for Uplo::Upper only, GREEN for Lower.
TYPED_TEST(PosvTest, PublicPosvSolvesOnBothTriangles) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::BackendType;
    const ScopedEnvVar pin("BATCHLAS_POSV_ROUTE", "blocked");

    for (Uplo uplo : {Uplo::Lower, Uplo::Upper}) {
        for (int n : {4, 16, 33}) {
            auto p = make_spd<T>(n, 2, 6, uplo, 2626u + unsigned(n));
            auto A = a_view(p); auto Bv = b_view(p);
            const size_t need = posv_buffer_size<B, T>(*this->ctx, A, Bv, uplo);
            UnifiedVector<std::byte> ws(need > 0 ? need : size_t(1));
            (void)posv<B, T>(*this->ctx, A, Bv, uplo, Span<std::byte>(ws.data(), need),
                       p.info.to_span());
            this->ctx->wait();

            for (int item : {0, p.batch - 1}) {
                EXPECT_EQ(p.info[item], 0) << "n=" << n;
                EXPECT_LT(solve_residual(p, item, uplo), solve_tol<T>(n))
                    << "n=" << n << " uplo=" << (uplo == Uplo::Lower ? "L" : "U");
            }
        }
    }
}

// P9. linalg::solve_spd copies its inputs, so neither A nor B may come back changed.
// A value-returning wrapper that leaked its scratch, or that factored in place, is
// the defect docs/design pattern "linalg scratch lifetime" records.
TYPED_TEST(PosvTest, LinalgSolveSpdLeavesItsInputsAlone) {
    using T = typename TestFixture::T;
    const int n = 16;
    auto p = make_spd<T>(n, 2, 4, Uplo::Lower, 1919u);
    auto A = a_view(p); auto Bv = b_view(p);

    auto X = linalg::solve_spd<T>(*this->ctx, A, Bv, Uplo::Lower);
    this->ctx->wait();

    for (size_t i = 0; i < p.a.size(); ++i)
        ASSERT_EQ(0, std::memcmp(&p.a[i], &p.a0[i], sizeof(T))) << "A changed at " << i;
    for (size_t i = 0; i < p.b.size(); ++i)
        ASSERT_EQ(0, std::memcmp(&p.b[i], &p.b0[i], sizeof(T))) << "B changed at " << i;

    // And the answer is right: copy X back into the working buffer and reuse the
    // residual above, which reads p.b as the solution.
    for (int bi = 0; bi < p.batch; ++bi)
        for (int k = 0; k < p.nrhs; ++k)
            for (int i = 0; i < n; ++i)
                p.b[size_t(bi) * p.strb + size_t(k) * p.ldb + i] =
                    X.view().data_ptr()[size_t(bi) * X.view().stride() +
                                        size_t(k) * X.view().ld() + i];
    for (int item : {0, p.batch - 1})
        EXPECT_LT(solve_residual(p, item, Uplo::Lower), solve_tol<T>(n));
}
