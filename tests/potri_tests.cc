// potrs (solve with a Cholesky factor) and potri (inverse from one): residuals through
// batchlas::verify, bit-for-bit agreement of potrs with posv on the same route, per-item info
// with the other items undisturbed, padded ld and stride with a poisoned unreferenced triangle,
// and a saturating batch. Both ops are native and vendor-free, so no oracle is a vendor library.
// evidence: docs/perf/potri.md#potrs-and-potri
#include <gtest/gtest.h>

#include <batchlas/blas/functions/posv.hh>
#include <batchlas/blas/functions/potrf.hh>
#include <batchlas/blas/functions/potri.hh>
#include <batchlas/blas/functions/potrs.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>

#include <tests/test_utils.hh>
#include <batchlas/verify/tolerance.hh>

#include <src/ops/posv/choice.hh>
#include <src/ops/potri/choice.hh>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstring>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

using namespace batchlas;

namespace {

template <typename T>
using MV = MatrixView<T, MatrixFormat::Dense>;
template <typename T>
using RealOf = typename batchlas::base_type<T>::type;

template <typename T, Backend B>
struct PotriConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

template <typename T>
bool same_bits(const T& a, const T& b) {
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}

// One Hermitian positive-definite item, strictly diagonally dominant (condition below 8), every
// off-diagonal entry with a nonzero imaginary part. Column-major n x n.
template <typename T>
std::vector<T> make_hpd(int n, std::mt19937& gen) {
    using R = RealOf<T>;
    std::uniform_real_distribution<R> d(R(0.1), R(1));
    std::vector<T> A(static_cast<size_t>(n) * n);
    for (int j = 0; j < n; ++j) {
        A[j + static_cast<size_t>(j) * n] = MV<T>::make(R(2.5) + R(0.5) * d(gen), R(0));
        for (int i = j + 1; i < n; ++i) {
            const R x = (gen() & 1) ? d(gen) : -d(gen);
            const R y = d(gen);
            A[i + static_cast<size_t>(j) * n] = MV<T>::make(x / R(n), y / R(n));
            A[j + static_cast<size_t>(i) * n] = MV<T>::make(x / R(n), -y / R(n));
        }
    }
    return A;
}

// A and B at padded ld and a stride that is not ld*cols. Everything the op may not touch (the
// other triangle of A, the ld pads, the inter-item gaps) holds a large finite value a kernel
// would happily consume: reading it wrecks a residual, writing it is caught by memcmp.
template <typename T>
struct Sys {
    int n = 0, nrhs = 0, batch = 0, lda = 0, stra = 0, ldb = 0, strb = 0;
    Uplo uplo = Uplo::Lower;
    UnifiedVector<T> a, b;
    UnifiedVector<T*> aptr, bptr;
    std::vector<std::vector<T>> full, rhs;  // the whole Hermitian A and the n x nrhs B per item

    MV<T> A() { return {a.data(), n, n, lda, stra, batch, aptr.data()}; }
    MV<T> B() { return {b.data(), n, nrhs, ldb, strb, batch, bptr.data()}; }
    bool owned(int i, int j) const { return uplo == Uplo::Lower ? i >= j : i <= j; }
};

// identical: every item is item 0's data (the saturating-batch shape).
template <typename T>
Sys<T> make_sys(int n, int nrhs, int batch, Uplo uplo, unsigned seed, bool identical = false) {
    using R = RealOf<T>;
    Sys<T> p;
    p.n = n, p.nrhs = std::max(nrhs, 1), p.batch = batch, p.uplo = uplo;
    p.lda = n + 3, p.stra = p.lda * n + 5, p.ldb = n + 2, p.strb = p.ldb * p.nrhs + 7;
    const T poison = MV<T>::make(R(-999), R(777));
    p.a = UnifiedVector<T>(static_cast<size_t>(p.stra) * batch, poison);
    p.b = UnifiedVector<T>(static_cast<size_t>(p.strb) * batch, poison);
    p.aptr = UnifiedVector<T*>(batch, nullptr);
    p.bptr = UnifiedVector<T*>(batch, nullptr);
    p.full.resize(batch);
    p.rhs.resize(batch);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<R> d(R(-1), R(1));
    for (int it = 0; it < batch; ++it) {
        if (identical && it > 0) {
            p.full[it] = p.full[0];
            p.rhs[it] = p.rhs[0];
        } else {
            p.full[it] = make_hpd<T>(n, gen);
            p.rhs[it].resize(static_cast<size_t>(n) * p.nrhs);
            for (auto& v : p.rhs[it]) v = MV<T>::make(d(gen), d(gen));
        }
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                if (p.owned(i, j))
                    p.a[static_cast<size_t>(it) * p.stra + static_cast<size_t>(j) * p.lda + i] =
                        p.full[it][i + static_cast<size_t>(j) * n];
        for (int k = 0; k < p.nrhs; ++k)
            for (int i = 0; i < n; ++i)
                p.b[static_cast<size_t>(it) * p.strb + static_cast<size_t>(k) * p.ldb + i] =
                    p.rhs[it][i + static_cast<size_t>(k) * n];
    }
    return p;
}

template <typename Config>
class PotriTest : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    static constexpr Backend BK = Config::BackendVal;

    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        if (this->HasFatalFailure() || ::testing::Test::IsSkipped()) return;
        if (!this->ctx) GTEST_SKIP() << "no queue";
        if (this->ctx->device().type != DeviceType::GPU) GTEST_SKIP() << "the native potri families are GPU-only here";
        if (!this->ctx->device().supports_sub_group_size(32)) GTEST_SKIP() << "device does not offer sub-group size 32";
    }

    Queue& q() { return *this->ctx; }

    // The Cholesky factor in A's triangle, through the public potrf; returns its info.
    void factor(Sys<T>& p, UnifiedVector<int32_t>& info) {
        info = UnifiedVector<int32_t>(static_cast<size_t>(p.batch), int32_t(-12345));
        const size_t need = potrf_buffer_size<BK, T>(q(), p.A(), p.uplo);
        UnifiedVector<std::byte> ws(std::max<size_t>(need, 1));
        (void)potrf<BK, T>(q(), p.A(), p.uplo, Span<std::byte>(ws.data(), need), info.to_span());
        q().wait();
    }

    Event run_potri(Sys<T>& p, bool full, UnifiedVector<int32_t>* info = nullptr) {
        const size_t need = potri_buffer_size<BK, T>(q(), p.A(), p.uplo);
        ws_ = UnifiedVector<std::byte>(std::max<size_t>(need, 1));
        Span<int32_t> s = info ? info->to_span() : Span<int32_t>{};
        auto ev = potri<BK, T>(q(), p.A(), p.uplo, full, Span<std::byte>(ws_.data(), need), s);
        ev.wait();
        q().wait();
        return ev;
    }

    // max over items of ||A0 X - I|| / (||A0|| ||X||), X read from item @p x_of(item) of a packed
    // n x n batch, A0 the original full Hermitian matrices.
    double inverse_residual(const Sys<T>& p, const std::vector<std::vector<T>>& inv) {
        Matrix<T> A0(p.n, p.n, p.batch), X(p.n, p.n, p.batch);
        for (int it = 0; it < p.batch; ++it)
            for (int j = 0; j < p.n; ++j)
                for (int i = 0; i < p.n; ++i) {
                    A0(i, j, it) = p.full[it][i + static_cast<size_t>(j) * p.n];
                    X(i, j, it) = inv[it][i + static_cast<size_t>(j) * p.n];
                }
        return MV<T>::solve_residual(q(), A0, X, MV<T>{});
    }

    // Item @p it of A as a packed n x n matrix, the triangle mirrored (Hermitian) onto the rest.
    std::vector<T> mirrored(const Sys<T>& p, int it) const {
        std::vector<T> out(static_cast<size_t>(p.n) * p.n);
        for (int j = 0; j < p.n; ++j)
            for (int i = 0; i < p.n; ++i) {
                const T v = p.owned(i, j) ? p.a[static_cast<size_t>(it) * p.stra + static_cast<size_t>(j) * p.lda + i]
                                          : p.a[static_cast<size_t>(it) * p.stra + static_cast<size_t>(i) * p.lda + j];
                out[i + static_cast<size_t>(j) * p.n] = p.owned(i, j) ? v : MV<T>::make(double(std::real(v)), -double(std::imag(v)));
            }
        return out;
    }

    double solve_res(const Sys<T>& p) {
        Matrix<T> A0(p.n, p.n, p.batch), X(p.n, p.nrhs, p.batch), B0(p.n, p.nrhs, p.batch);
        for (int it = 0; it < p.batch; ++it) {
            for (int j = 0; j < p.n; ++j)
                for (int i = 0; i < p.n; ++i) A0(i, j, it) = p.full[it][i + static_cast<size_t>(j) * p.n];
            for (int k = 0; k < p.nrhs; ++k)
                for (int i = 0; i < p.n; ++i) {
                    X(i, k, it) = p.b[static_cast<size_t>(it) * p.strb + static_cast<size_t>(k) * p.ldb + i];
                    B0(i, k, it) = p.rhs[it][i + static_cast<size_t>(k) * p.n];
                }
        }
        return MV<T>::solve_residual(q(), A0, X, B0);
    }

    bool outside_a_untouched(const Sys<T>& p, const UnifiedVector<T>& before, bool full) const {
        for (int it = 0; it < p.batch; ++it)
            for (size_t o = 0; o < static_cast<size_t>(p.stra); ++o) {
                const int col = static_cast<int>(o) / p.lda, row = static_cast<int>(o) % p.lda;
                const bool in = col < p.n && row < p.n && static_cast<int>(o) < p.lda * p.n;
                if (in && (p.owned(row, col) || full)) continue;
                const size_t idx = static_cast<size_t>(it) * p.stra + o;
                if (!same_bits(p.a[idx], before[idx])) return false;
            }
        return true;
    }

    UnifiedVector<std::byte> ws_;
};

using PotriTestTypes = typename test_utils::backend_types<PotriConfig>::type;

}  // namespace

TYPED_TEST_SUITE(PotriTest, PotriTestTypes);

// potrs on the factor of potrf, Auto route, both triangles, an order ladder through both sides of
// the fused ceiling (nrhs <= 8 and n * nrhs within SLM), padded ld and stride. The factor must be
// read-only, a second right-hand-side set must reuse it, and B's pads must survive.
// ARMED BREAK: in src/ops/potrs/potrs.hh solve(), swap the two Upper trsm transposes. EXPECTED: RED
// for Uplo::Upper in the residual tests below (here and the posv comparison), Lower GREEN.
TYPED_TEST(PotriTest, PotrsSolvesAndReusesTheFactor) {
    using T = typename TestFixture::T;
    for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
        for (int n : {1, 3, 8, 9, 20, 33, 70})
            for (int nrhs : {1, 3, 8, 9, 20}) {
                auto p = make_sys<T>(n, nrhs, 5, uplo, 100u + n * 31 + nrhs);
                UnifiedVector<int32_t> info;
                this->factor(p, info);
                const UnifiedVector<T> factor_bits = p.a;
                (void)potrs<TestFixture::BK, T>(this->q(), p.A(), p.B(), uplo).wait();
                this->q().wait();
                for (int it = 0; it < p.batch; ++it) ASSERT_EQ(info[it], 0);
                EXPECT_VERIFY(T, verify::Check::solve, n, this->solve_res(p))
                    << "n=" << n << " nrhs=" << nrhs << " uplo=" << (uplo == Uplo::Lower ? "L" : "U");
                for (size_t i = 0; i < p.a.size(); ++i) ASSERT_TRUE(same_bits(p.a[i], factor_bits[i])) << "A written at " << i;
                for (int it = 0; it < p.batch; ++it)
                    for (size_t o = 0; o < static_cast<size_t>(p.strb); ++o) {
                        const int col = static_cast<int>(o) / p.ldb, row = static_cast<int>(o) % p.ldb;
                        if (col < p.nrhs && row < n) continue;
                        ASSERT_TRUE(same_bits(p.b[static_cast<size_t>(it) * p.strb + o], MV<T>::make(RealOf<T>(-999), RealOf<T>(777))))
                            << "B pad written at item " << it << " offset " << o;
                    }
                // The same factor, a second right-hand-side set.
                auto p2 = make_sys<T>(n, nrhs, 5, uplo, 100u + n * 31 + nrhs);
                p2.a = p.a;
                for (int it = 0; it < p2.batch; ++it)
                    for (int k = 0; k < p2.nrhs; ++k)
                        for (int i = 0; i < n; ++i) {
                            p2.rhs[it][i + static_cast<size_t>(k) * n] = MV<T>::make(double(i + 1), double(-k));
                            p2.b[static_cast<size_t>(it) * p2.strb + static_cast<size_t>(k) * p2.ldb + i] = p2.rhs[it][i + static_cast<size_t>(k) * n];
                        }
                (void)potrs<TestFixture::BK, T>(this->q(), p2.A(), p2.B(), uplo).wait();
                this->q().wait();
                EXPECT_VERIFY(T, verify::Check::solve, n, this->solve_res(p2)) << "second right-hand side n=" << n;
            }
}

// potrs is the solve half of posv, bit for bit, on each posv route that shares its family: `cta`
// where the fused solve fits (n = 8, nrhs = 4), `blocked` where it does not (nrhs = 16).
// ARMED BREAK: make potrs::solve() ignore `fused` and always take the trsm pair. EXPECTED: RED
// for the cta case only (the fused kernel and two trsm round differently), blocked case GREEN.
TYPED_TEST(PotriTest, PotrsIsBitIdenticalToPosvOnTheSameRoute) {
    using T = typename TestFixture::T;
    using namespace ops::posv;
    for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
        for (auto [n, nrhs, blocked] : {std::tuple{8, 4, false}, std::tuple{20, 16, true}, std::tuple{9, 2, false}}) {
            auto p = make_sys<T>(n, nrhs, 7, uplo, 7u + n, false);
            auto q2 = make_sys<T>(n, nrhs, 7, uplo, 7u + n, false);
            UnifiedVector<int32_t> info;
            {
                const size_t need = posv_buffer_size<TestFixture::BK, T>(this->q(), p.A(), p.B(), uplo);
                UnifiedVector<std::byte> ws(std::max<size_t>(need, 1));
                const select::ScopedPin<PosvChoice> pin("posv", blocked ? PosvChoice{Blocked{}} : PosvChoice{Cta{}});
                info = UnifiedVector<int32_t>(7, int32_t(-1));
                (void)posv<TestFixture::BK, T>(this->q(), p.A(), p.B(), uplo, Span<std::byte>(ws.data(), need), info.to_span()).wait();
            }
            UnifiedVector<int32_t> info2;
            this->factor(q2, info2);
            (void)potrs<TestFixture::BK, T>(this->q(), q2.A(), q2.B(), uplo).wait();
            this->q().wait();
            for (size_t i = 0; i < p.a.size(); ++i) ASSERT_TRUE(same_bits(p.a[i], q2.a[i])) << "factor differs at " << i;
            for (size_t i = 0; i < p.b.size(); ++i) ASSERT_TRUE(same_bits(p.b[i], q2.b[i])) << "X differs at " << i << " n=" << n << " nrhs=" << nrhs;
        }
}

// posv composes potrf and potrs, so a non-positive-definite item is flagged in `info` at the minor
// potrf names, and every other item's X is bit-identical to a batch without the bad item.
// ARMED BREAK: in potrs::solve() skip the Lower NoTrans trsm. EXPECTED: RED here and in the posv
// comparison above; the info assertions alone stay GREEN.
TYPED_TEST(PotriTest, NonSpdItemIsFlaggedAndTheOthersAreUndisturbed) {
    using T = typename TestFixture::T;
    using namespace ops::posv;
    for (bool blocked : {false, true})
        for (int n : {8, 20}) {
            const int nrhs = blocked ? 16 : 4, bad = 2, batch = 5;
            auto clean = make_sys<T>(n, nrhs, batch, Uplo::Lower, 41u + n);
            auto dirty = make_sys<T>(n, nrhs, batch, Uplo::Lower, 41u + n);
            for (int j = 0; j < n; ++j)  // item 2 loses positive-definiteness at minor n/2 + 1
                if (j == n / 2) dirty.a[static_cast<size_t>(bad) * dirty.stra + static_cast<size_t>(j) * dirty.lda + j] = MV<T>::make(-1.0, 0.0);
            UnifiedVector<int32_t> ic(batch, -1), id(batch, -1);
            for (Sys<T>* s : {&clean, &dirty}) {
                const size_t need = posv_buffer_size<TestFixture::BK, T>(this->q(), s->A(), s->B(), Uplo::Lower);
                UnifiedVector<std::byte> ws(std::max<size_t>(need, 1));
                const select::ScopedPin<PosvChoice> pin("posv", blocked ? PosvChoice{Blocked{}} : PosvChoice{Cta{}});
                (void)posv<TestFixture::BK, T>(this->q(), s->A(), s->B(), Uplo::Lower, Span<std::byte>(ws.data(), need),
                                               (s == &clean ? ic : id).to_span()).wait();
            }
            this->q().wait();
            for (int it = 0; it < batch; ++it) {
                EXPECT_EQ(ic[it], 0);
                EXPECT_EQ(id[it], it == bad ? n / 2 + 1 : 0) << "item " << it;
                if (it == bad) continue;
                for (size_t o = 0; o < static_cast<size_t>(dirty.strb); ++o)
                    ASSERT_TRUE(same_bits(dirty.b[static_cast<size_t>(it) * dirty.strb + o], clean.b[static_cast<size_t>(it) * clean.strb + o]))
                        << "item " << it << " disturbed by item " << bad << " at " << o;
            }
        }
}

// potri: A * A^-1 = I through verify, both triangles, Auto and each pin, an order ladder through the
// fused ceiling (8), padded ld and stride, the unreferenced triangle poisoned and written by nobody.
// ARMED BREAK: in potri.cc launch() drop the fill_identity. EXPECTED: RED in every potri test that
// reads the inverse, the potrs tests GREEN.
TYPED_TEST(PotriTest, PotriInvertsAndWritesOnlyItsTriangle) {
    using T = typename TestFixture::T;
    using namespace ops::potri;
    for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
        for (int n : {1, 2, 5, 8, 9, 16, 31, 64, 100})
            for (int route = 0; route < 3; ++route) {  // 0 auto, 1 fused, 2 blocked
                if (route == 1 && n > 8) continue;
                auto p = make_sys<T>(n, 1, 4, uplo, 900u + n);
                UnifiedVector<int32_t> info;
                this->factor(p, info);
                const UnifiedVector<T> before = p.a;
                UnifiedVector<int32_t> pinfo(p.batch, int32_t(-7));
                {
                    std::optional<select::ScopedPin<PotriChoice>> pin;
                    if (route == 1) pin.emplace("potri", PotriChoice{Fused{}});
                    if (route == 2) pin.emplace("potri", PotriChoice{Blocked{}});
                    (void)this->run_potri(p, false, &pinfo);
                }
                for (int it = 0; it < p.batch; ++it) ASSERT_EQ(pinfo[it], 0);
                std::vector<std::vector<T>> inv;
                for (int it = 0; it < p.batch; ++it) inv.push_back(this->mirrored(p, it));
                EXPECT_VERIFY(T, verify::Check::solve, n, this->inverse_residual(p, inv))
                    << "n=" << n << " route=" << route << " uplo=" << (uplo == Uplo::Lower ? "L" : "U");
                EXPECT_TRUE(this->outside_a_untouched(p, before, false)) << "n=" << n << " route=" << route;
            }
}

// With full = true the other triangle is exactly the conjugate transpose (bit for bit: a complex
// inverse with nonzero imaginary parts is Hermitian only if the mirror conjugates), and the whole
// matrix is the inverse with no mirroring on the caller's side.
// ARMED BREAK: in potri_native.cc potri_store drop the dev_conj. EXPECTED: RED for complex types
// only, here; real types and every full = false test GREEN.
TYPED_TEST(PotriTest, PotriFullFillsTheOtherTriangleWithTheConjugateTranspose) {
    using T = typename TestFixture::T;
    for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
        for (int n : {3, 8, 17, 40}) {
            auto p = make_sys<T>(n, 1, 3, uplo, 31u + n);
            UnifiedVector<int32_t> info;
            this->factor(p, info);
            const UnifiedVector<T> before = p.a;
            (void)this->run_potri(p, true);
            std::vector<std::vector<T>> inv;
            for (int it = 0; it < p.batch; ++it) {
                std::vector<T> whole(static_cast<size_t>(n) * n);
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < n; ++i) whole[i + static_cast<size_t>(j) * n] = p.a[static_cast<size_t>(it) * p.stra + static_cast<size_t>(j) * p.lda + i];
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < n; ++i) {
                        const T a = whole[i + static_cast<size_t>(j) * n], b = whole[j + static_cast<size_t>(i) * n];
                        // Diagonals: a real value (a bitwise +0 against -0 is not a defect).
                        if (i == j) ASSERT_EQ(std::imag(a), 0) << "(" << i << "," << j << ")";
                        else ASSERT_TRUE(same_bits(a, MV<T>::make(double(std::real(b)), -double(std::imag(b))))) << "(" << i << "," << j << ")";
                    }
                inv.push_back(whole);
            }
            EXPECT_VERIFY(T, verify::Check::solve, n, this->inverse_residual(p, inv)) << "n=" << n;
            EXPECT_TRUE(this->outside_a_untouched(p, before, true));
        }
}

// info follows ?potri: the first exactly-zero factor diagonal, per item, and the other items'
// inverses are bit-identical to a clean batch's.
// ARMED BREAK: in potri_native.cc potri_diag_info start the scan at i = 1. EXPECTED: RED for the
// zero at index 0 only.
TYPED_TEST(PotriTest, PotriInfoNamesTheZeroDiagonalAndLeavesOtherItemsAlone) {
    using T = typename TestFixture::T;
    for (int n : {5, 8, 20})
        for (int zero_at : {0, n / 2, n - 1}) {
            const int bad = 1, batch = 4;
            auto clean = make_sys<T>(n, 1, batch, Uplo::Lower, 5u + n);
            UnifiedVector<int32_t> info;
            this->factor(clean, info);
            auto dirty = make_sys<T>(n, 1, batch, Uplo::Lower, 5u + n);
            dirty.a = clean.a;
            dirty.a[static_cast<size_t>(bad) * dirty.stra + static_cast<size_t>(zero_at) * dirty.lda + zero_at] = MV<T>::make(0.0, 0.0);
            UnifiedVector<int32_t> ic(batch, -1), id(batch, -1);
            (void)this->run_potri(clean, false, &ic);
            (void)this->run_potri(dirty, false, &id);
            for (int it = 0; it < batch; ++it) {
                EXPECT_EQ(ic[it], 0);
                EXPECT_EQ(id[it], it == bad ? zero_at + 1 : 0) << "item " << it;
                if (it == bad) continue;
                for (size_t o = 0; o < static_cast<size_t>(dirty.stra); ++o)
                    ASSERT_TRUE(same_bits(dirty.a[static_cast<size_t>(it) * dirty.stra + o], clean.a[static_cast<size_t>(it) * clean.stra + o]))
                        << "item " << it << " disturbed at " << o;
            }
        }
}

// The workspace is one n x n identity per item: sized by potri_buffer_size, enforced, and the
// empty-batch, empty-problem and heterogeneous cases are refused or no-ops, not wrong answers.
TYPED_TEST(PotriTest, PotriWorkspaceAndDegenerateCalls) {
    using T = typename TestFixture::T;
    auto p = make_sys<T>(12, 1, 3, Uplo::Lower, 3u);
    UnifiedVector<int32_t> info;
    this->factor(p, info);
    const size_t need = potri_buffer_size<TestFixture::BK, T>(this->q(), p.A(), p.uplo);
    EXPECT_GE(need, sizeof(T) * 12 * 12 * 3);
    UnifiedVector<std::byte> small(need / 2);
    EXPECT_THROW(([&] { (void)potri<TestFixture::BK, T>(this->q(), p.A(), p.uplo, false, Span<std::byte>(small.data(), small.size()), Span<int32_t>{}); }()), batchlas::workspace_error);
    UnifiedVector<std::byte> ws(need);
    MV<T> empty_batch(p.a.data(), 12, 12, p.lda, p.stra, 0, nullptr);
    EXPECT_NO_THROW(([&] { (void)potri<TestFixture::BK, T>(this->q(), empty_batch, p.uplo, false, Span<std::byte>(ws.data(), need), Span<int32_t>{}).wait(); }()));
    EXPECT_EQ((potri_buffer_size<TestFixture::BK, T>(this->q(), empty_batch, p.uplo)), 0u);
    MV<T> nonsquare(p.a.data(), 12, 11, p.lda, p.stra, 3, nullptr);
    EXPECT_THROW(([&] { (void)potri_buffer_size<TestFixture::BK, T>(this->q(), nonsquare, p.uplo); }()), batchlas::invalid_argument);
    MV<T> zero_n(p.a.data(), 0, 0, 1, 1, 3, nullptr);
    EXPECT_THROW(([&] { (void)potri_buffer_size<TestFixture::BK, T>(this->q(), zero_n, p.uplo); }()), batchlas::internal_error);
}

// The route straddle: the fused pin runs at n = 8 and is refused at n = 9, Auto serves both sides,
// and an unknown pin throws rather than falling back.
TYPED_TEST(PotriTest, PotriFusedPinStraddlesItsCeiling) {
    using T = typename TestFixture::T;
    using namespace ops::potri;
    for (int n : {8, 9}) {
        auto p = make_sys<T>(n, 1, 3, Uplo::Lower, 77u + n);
        UnifiedVector<int32_t> info;
        this->factor(p, info);
        const size_t need = potri_buffer_size<TestFixture::BK, T>(this->q(), p.A(), p.uplo);
        UnifiedVector<std::byte> ws(std::max<size_t>(need, 1));
        auto call = [&] { (void)potri<TestFixture::BK, T>(this->q(), p.A(), p.uplo, false, Span<std::byte>(ws.data(), need), Span<int32_t>{}).wait(); };
        {
            const select::ScopedPin<PotriChoice> pin("potri", PotriChoice{Fused{}});
            if (n == 8) EXPECT_NO_THROW(call());
            else EXPECT_THROW(call(), std::invalid_argument);
        }
        {
            const select::ScopedPin<PotriChoice> pin("potri", std::string_view("lpanel:panel=8"));
            EXPECT_THROW(call(), std::invalid_argument);
        }
        if (n == 9) EXPECT_NO_THROW(call());  // Auto
    }
}

// A saturating batch: 1024 identical items, every result bit-identical to item 0 (a race between
// items or work-groups shows as a differing item; small batches cannot race). Both families, and
// potrs on the same factor.
// ARMED BREAK: in potri_native.cc potri_store index X with item 0's stride. EXPECTED: RED here
// only (the identical items hide it from every smaller test, whose items differ).
TYPED_TEST(PotriTest, SaturatingBatchOfIdenticalItemsIsBitIdentical) {
    using T = typename TestFixture::T;
    for (int n : {8, 16}) {
        auto p = make_sys<T>(n, 3, 1024, Uplo::Lower, 13u + n, /*identical=*/true);
        UnifiedVector<int32_t> info;
        this->factor(p, info);
        auto s = make_sys<T>(n, 3, 1024, Uplo::Lower, 13u + n, true);
        s.a = p.a;
        (void)potrs<TestFixture::BK, T>(this->q(), s.A(), s.B(), Uplo::Lower).wait();
        (void)this->run_potri(p, true);
        this->q().wait();
        for (int it = 1; it < p.batch; ++it) {
            for (size_t o = 0; o < static_cast<size_t>(p.stra); ++o)
                ASSERT_TRUE(same_bits(p.a[static_cast<size_t>(it) * p.stra + o], p.a[o])) << "potri item " << it << " offset " << o;
            for (size_t o = 0; o < static_cast<size_t>(s.strb); ++o)
                ASSERT_TRUE(same_bits(s.b[static_cast<size_t>(it) * s.strb + o], s.b[o])) << "potrs item " << it << " offset " << o;
        }
    }
}
