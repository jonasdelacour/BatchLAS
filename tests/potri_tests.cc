// potrs (solve with a Cholesky factor) and potri (inverse from one): residuals through
// batchlas::verify, bit-for-bit agreement of potrs with posv on the same route, per-item info
// with the other items undisturbed, padded ld and stride with a poisoned unreferenced triangle,
// and a saturating batch. Both ops are native and vendor-free, so no oracle is a vendor library.
// evidence: docs/perf/potri.md#potrs-and-potri
#include <gtest/gtest.h>

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/posv.hh>
#include <batchlas/blas/functions/potrf.hh>
#include <batchlas/blas/functions/potri.hh>
#include <batchlas/no_route.hh>
#include <batchlas/blas/functions/potrs.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>

#include <tests/test_utils.hh>
#include <batchlas/verify/tolerance.hh>

#include <src/ops/posv/choice.hh>
#include <src/ops/potri/choice.hh>
#include <src/ops/potri/potri_native.hh>
#include <src/select/coverage.hh>
#include <src/select/select.hh>
#include <batchlas/util/env.hh>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
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
        // NETLIB serves potrf and trsm through LAPACKE/CBLAS only (no CPU-native kernel), so a
        // vendor-free build has nothing to test on the host queue.
        if (BK == Backend::NETLIB && !(BATCHLAS_HAS_LAPACKE && BATCHLAS_HAS_CBLAS))
            GTEST_SKIP() << "NETLIB without LAPACKE/CBLAS has no potrf or trsm route";
    }

    Queue& q() { return *this->ctx; }
    // The fused solve (potri `fused`, posv `cta`) needs a GPU with sub-group size 32; `blocked` runs anywhere.
    bool fused_ok() { return q().device().type == DeviceType::GPU && q().device().supports_sub_group_size(32); }

    // The Cholesky factor in A's triangle, through the public potrf; returns its info.
    // False when the build has no potrf route for the shape (vendor-free: the native potrf does not cover it).
    bool factor(Sys<T>& p, UnifiedVector<int32_t>& info) {
        info = UnifiedVector<int32_t>(static_cast<size_t>(p.batch), int32_t(-12345));
        try {
            const size_t need = potrf_buffer_size<BK, T>(q(), p.A(), p.uplo);
            UnifiedVector<std::byte> ws(std::max<size_t>(need, 1));
            (void)potrf<BK, T>(q(), p.A(), p.uplo, Span<std::byte>(ws.data(), need), info.to_span());
        } catch (const NoRouteError&) {
            return false;
        }
        q().wait();
        return true;
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
// BREAK (run 2026-10-25): in src/ops/potrs/potrs.hh solve(), swap the two Upper trsm transposes. RED:
// exactly the tests with an Upper case (this one, PotriInverts, PotriFull, BatchBeyond, posv's
// PublicPosvSolvesOnBothTriangles), every type and backend; the posv comparison GREEN (shared solve).
TYPED_TEST(PotriTest, PotrsSolvesAndReusesTheFactor) {
    using T = typename TestFixture::T;
    for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
        for (int n : {1, 3, 8, 9, 20, 33, 70})
            for (int nrhs : {1, 3, 8, 9, 20}) {
                auto p = make_sys<T>(n, nrhs, 5, uplo, 100u + n * 31 + nrhs);
                UnifiedVector<int32_t> info;
                if (!this->factor(p, info)) continue;
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
            if (!blocked && !this->fused_ok()) continue;
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
            if (!this->factor(q2, info2)) continue;
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
            if (!blocked && !this->fused_ok()) continue;
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

// ---- potri ---------------------------------------------------------------------------------------
// Families: cta (sycl_potri::potri_cta, trtri + lauum in one local-memory launch) and blocked:nb
// (LAPACK's blocked trtri and lauum: the CTA kernel on nb x nb diagonal blocks, public trmm, gemm,
// herk/syrk). Every test pins each family; Auto is exercised beside them. The deliberate breaks and
// their red sets are in docs/perf/potri.md#potri-what-the-tests-pin.

namespace {

using ops::potri::PotriChoice;

// cta, then blocked at the two block orders every type compiles (nb = 32 crosses 33 and 65).
const std::vector<PotriChoice>& potri_pins() {
    static const std::vector<PotriChoice> v{ops::potri::Cta{}, ops::potri::Blocked{32}, ops::potri::Blocked{64}};
    return v;
}

std::string pin_name(const PotriChoice& c) { return select::to_string(c); }

// The order the cta pin can run on this queue (the can_run term, read from the same device facts).
template <Backend B, typename T>
int cta_ceiling(Queue& q) {
    const auto& d = select::device_of<B>(q);
    return sycl_potri::cta_max_n<T>(static_cast<std::size_t>(d.slm_budget), d.max_wg);
}

// A Hermitian positive-definite matrix Q diag(lam) Q^H with eigenvalues graded geometrically from 1 to
// 1 / cond and a random unitary Q (Gram-Schmidt twice, in double), column-major n x n.
template <typename T>
std::vector<T> make_graded_hpd(int n, double cond, std::mt19937& gen) {
    using C = std::complex<double>;
    constexpr bool kCx = !std::is_same_v<T, RealOf<T>>;
    std::normal_distribution<double> d(0.0, 1.0);
    std::vector<C> Q(static_cast<size_t>(n) * n);
    for (auto& v : Q) v = C(d(gen), kCx ? d(gen) : 0.0);
    auto col = [&](int j) { return Q.begin() + static_cast<std::ptrdiff_t>(j) * n; };
    for (int j = 0; j < n; ++j)
        for (int pass = 0; pass < 2; ++pass) {
            for (int k = 0; k < j; ++k) {
                C dot = 0;
                for (int i = 0; i < n; ++i) dot += std::conj(col(k)[i]) * col(j)[i];
                for (int i = 0; i < n; ++i) col(j)[i] -= dot * col(k)[i];
            }
            double nrm = 0;
            for (int i = 0; i < n; ++i) nrm += std::norm(col(j)[i]);
            for (int i = 0; i < n; ++i) col(j)[i] /= std::sqrt(nrm);
        }
    std::vector<T> A(static_cast<size_t>(n) * n);
    for (int j = 0; j < n; ++j)
        for (int i = j; i < n; ++i) {
            C s = 0;
            for (int k = 0; k < n; ++k)
                s += col(k)[i] * std::pow(cond, -double(k) / std::max(1, n - 1)) * std::conj(col(k)[j]);
            const double im = i == j ? 0.0 : s.imag();
            A[i + static_cast<size_t>(j) * n] = MV<T>::make(s.real(), im);
            A[j + static_cast<size_t>(i) * n] = MV<T>::make(s.real(), -im);
        }
    return A;
}

// make_sys's layout and poison, every item replaced by a graded matrix of condition @p cond.
template <typename T>
Sys<T> make_graded_sys(int n, int batch, Uplo uplo, unsigned seed, double cond) {
    Sys<T> p = make_sys<T>(n, 1, batch, uplo, seed);
    std::mt19937 gen(seed * 7u + 1u);
    for (int it = 0; it < batch; ++it) {
        p.full[it] = make_graded_hpd<T>(n, cond, gen);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                if (p.owned(i, j))
                    p.a[static_cast<size_t>(it) * p.stra + static_cast<size_t>(j) * p.lda + i] =
                        p.full[it][i + static_cast<size_t>(j) * n];
    }
    return p;
}

}  // namespace

// A * A^-1 = I through verify, both triangles, Auto and each pin, an order ladder through the tile
// width (4), the block orders (32, 64) and their ragged neighbours, padded ld and stride, the
// unreferenced triangle poisoned and written by nobody, info all zero.
TYPED_TEST(PotriTest, PotriInvertsAndWritesOnlyItsTriangle) {
    using T = typename TestFixture::T;
    const int ceiling = cta_ceiling<TestFixture::BK, T>(this->q());
    for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
        for (int n : {1, 2, 3, 4, 5, 8, 9, 16, 31, 32, 33, 64, 65, 100})
            for (int route = -1; route < static_cast<int>(potri_pins().size()); ++route) {
                if (route == 0 && n > ceiling) continue;
                auto p = make_sys<T>(n, 1, 4, uplo, 900u + n);
                UnifiedVector<int32_t> info;
                if (!this->factor(p, info)) continue;
                const UnifiedVector<T> before = p.a;
                UnifiedVector<int32_t> pinfo(p.batch, int32_t(-7));
                {
                    std::optional<select::ScopedPin<PotriChoice>> pin;
                    if (route >= 0) pin.emplace("potri", potri_pins()[route]);
                    (void)this->run_potri(p, false, &pinfo);
                }
                const std::string what = "n=" + std::to_string(n) + " route=" + (route < 0 ? "auto" : pin_name(potri_pins()[route])) +
                                         " uplo=" + (uplo == Uplo::Lower ? "L" : "U");
                for (int it = 0; it < p.batch; ++it) ASSERT_EQ(pinfo[it], 0) << what;
                std::vector<std::vector<T>> inv;
                for (int it = 0; it < p.batch; ++it) inv.push_back(this->mirrored(p, it));
                EXPECT_VERIFY(T, verify::Check::solve, n, this->inverse_residual(p, inv)) << what;
                EXPECT_TRUE(this->outside_a_untouched(p, before, false)) << what;
            }
}

// Graded spectra (condition 1e3 for float and cfloat, 1e6 for double and cdouble) on every family,
// both triangles: the diagonally dominant inputs above cannot see an unstable update order.
TYPED_TEST(PotriTest, PotriInvertsGradedIllConditionedInputs) {
    using T = typename TestFixture::T;
    const double cond = std::is_same_v<RealOf<T>, float> ? 1e3 : 1e6;
    const int ceiling = cta_ceiling<TestFixture::BK, T>(this->q());
    for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
        for (int n : {7, 33, 100})
            for (const PotriChoice& c : potri_pins()) {
                if (c.index() == 0 && n > ceiling) continue;
                auto p = make_graded_sys<T>(n, 3, uplo, 61u + n, cond);
                UnifiedVector<int32_t> info;
                if (!this->factor(p, info)) continue;
                for (int it = 0; it < p.batch; ++it) ASSERT_EQ(info[it], 0) << "potrf refused a graded input";
                {
                    const select::ScopedPin<PotriChoice> pin("potri", c);
                    (void)this->run_potri(p, false);
                }
                std::vector<std::vector<T>> inv;
                for (int it = 0; it < p.batch; ++it) inv.push_back(this->mirrored(p, it));
                EXPECT_VERIFY(T, verify::Check::solve, n, this->inverse_residual(p, inv))
                    << "n=" << n << " " << pin_name(c) << " uplo=" << (uplo == Uplo::Lower ? "L" : "U");
            }
}

// With full = true the other triangle is exactly the conjugate transpose (bit for bit: a complex
// inverse with nonzero imaginary parts is Hermitian only if the mirror conjugates), the diagonal is
// real, and the whole matrix is the inverse with no mirroring on the caller's side. Every family.
TYPED_TEST(PotriTest, PotriFullFillsTheOtherTriangleWithTheConjugateTranspose) {
    using T = typename TestFixture::T;
    for (const PotriChoice& c : potri_pins())
        for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
            for (int n : {3, 8, 17, 40, 70}) {
                auto p = make_sys<T>(n, 1, 3, uplo, 31u + n);
                UnifiedVector<int32_t> info;
                if (!this->factor(p, info)) continue;
                if (c.index() == 0 && n > cta_ceiling<TestFixture::BK, T>(this->q())) continue;
                const UnifiedVector<T> before = p.a;
                {
                    const select::ScopedPin<PotriChoice> pin("potri", c);
                    (void)this->run_potri(p, true);
                }
                std::vector<std::vector<T>> inv;
                for (int it = 0; it < p.batch; ++it) {
                    std::vector<T> whole(static_cast<size_t>(n) * n);
                    for (int j = 0; j < n; ++j)
                        for (int i = 0; i < n; ++i) whole[i + static_cast<size_t>(j) * n] = p.a[static_cast<size_t>(it) * p.stra + static_cast<size_t>(j) * p.lda + i];
                    for (int j = 0; j < n; ++j)
                        for (int i = 0; i < n; ++i) {
                            const T a = whole[i + static_cast<size_t>(j) * n], b = whole[j + static_cast<size_t>(i) * n];
                            // Diagonals: a real value (a bitwise +0 against -0 is not a defect).
                            if (i == j) ASSERT_EQ(std::imag(a), 0) << "(" << i << "," << j << ") " << pin_name(c);
                            else ASSERT_TRUE(same_bits(a, MV<T>::make(double(std::real(b)), -double(std::imag(b))))) << "(" << i << "," << j << ") " << pin_name(c);
                        }
                    inv.push_back(whole);
                }
                EXPECT_VERIFY(T, verify::Check::solve, n, this->inverse_residual(p, inv)) << "n=" << n << " " << pin_name(c);
                EXPECT_TRUE(this->outside_a_untouched(p, before, true)) << pin_name(c);
            }
}

// info follows ?potri: the first exactly-zero factor diagonal, per item, and the other items'
// inverses are bit-identical to a clean batch's, on every family.
TYPED_TEST(PotriTest, PotriInfoNamesTheZeroDiagonalAndLeavesOtherItemsAlone) {
    using T = typename TestFixture::T;
    for (const PotriChoice& c : potri_pins())
        for (int n : {5, 8, 20, 70})
            for (int zero_at : {0, n / 2, n - 1}) {
                if (c.index() == 0 && n > cta_ceiling<TestFixture::BK, T>(this->q())) continue;
                const int bad = 1, batch = 4;
                auto clean = make_sys<T>(n, 1, batch, Uplo::Lower, 5u + n);
                UnifiedVector<int32_t> info;
                if (!this->factor(clean, info)) continue;
                auto dirty = make_sys<T>(n, 1, batch, Uplo::Lower, 5u + n);
                dirty.a = clean.a;
                dirty.a[static_cast<size_t>(bad) * dirty.stra + static_cast<size_t>(zero_at) * dirty.lda + zero_at] = MV<T>::make(0.0, 0.0);
                UnifiedVector<int32_t> ic(batch, -1), id(batch, -1);
                {
                    const select::ScopedPin<PotriChoice> pin("potri", c);
                    (void)this->run_potri(clean, false, &ic);
                    (void)this->run_potri(dirty, false, &id);
                }
                for (int it = 0; it < batch; ++it) {
                    EXPECT_EQ(ic[it], 0) << pin_name(c);
                    EXPECT_EQ(id[it], it == bad ? zero_at + 1 : 0) << "item " << it << " " << pin_name(c);
                    if (it == bad) continue;
                    for (size_t o = 0; o < static_cast<size_t>(dirty.stra); ++o)
                        ASSERT_TRUE(same_bits(dirty.a[static_cast<size_t>(it) * dirty.stra + o], clean.a[static_cast<size_t>(it) * clean.stra + o]))
                            << "item " << it << " disturbed at " << o << " " << pin_name(c);
                }
            }
}

// The workspace is the chosen family's: none for cta, none for blocked at n <= nb, and for blocked
// above it one part's n x nb panel per item, never n x n per item; it is enforced, and the
// empty-batch, empty-problem and non-square cases are refused or no-ops, not wrong answers.
TYPED_TEST(PotriTest, PotriWorkspaceAndDegenerateCalls) {
    using T = typename TestFixture::T;
    constexpr Backend BK = TestFixture::BK;
    auto p = make_sys<T>(100, 1, 3, Uplo::Lower, 3u);
    UnifiedVector<int32_t> info;
    if (!this->factor(p, info)) GTEST_SKIP() << "no potrf route for this shape in this build";
    {
        const select::ScopedPin<PotriChoice> pin("potri", PotriChoice{ops::potri::Cta{}});
        if (100 <= cta_ceiling<BK, T>(this->q())) EXPECT_EQ((potri_buffer_size<BK, T>(this->q(), p.A(), p.uplo)), 0u);
    }
    const select::ScopedPin<PotriChoice> pin("potri", PotriChoice{ops::potri::Blocked{32}});
    const size_t need = potri_buffer_size<BK, T>(this->q(), p.A(), p.uplo);
    EXPECT_GE(need, sizeof(T) * 100 * 32 * 3);
    EXPECT_LT(need, sizeof(T) * 100 * 100 * 3);
    MV<T> small_n(p.a.data(), 32, 32, p.lda, p.stra, 3, nullptr);
    EXPECT_EQ((potri_buffer_size<BK, T>(this->q(), small_n, p.uplo)), 0u);
    UnifiedVector<std::byte> small(need / 2);
    EXPECT_THROW(([&] { (void)potri<BK, T>(this->q(), p.A(), p.uplo, false, Span<std::byte>(small.data(), small.size()), Span<int32_t>{}); }()), batchlas::workspace_error);
    UnifiedVector<std::byte> ws(need);
    MV<T> empty_batch(p.a.data(), 12, 12, p.lda, p.stra, 0, nullptr);
    EXPECT_NO_THROW(([&] { (void)potri<BK, T>(this->q(), empty_batch, p.uplo, false, Span<std::byte>(ws.data(), need), Span<int32_t>{}).wait(); }()));
    EXPECT_EQ((potri_buffer_size<BK, T>(this->q(), empty_batch, p.uplo)), 0u);
    MV<T> nonsquare(p.a.data(), 12, 11, p.lda, p.stra, 3, nullptr);
    EXPECT_THROW(([&] { (void)potri_buffer_size<BK, T>(this->q(), nonsquare, p.uplo); }()), batchlas::invalid_argument);
    MV<T> zero_n(p.a.data(), 0, 0, 1, 1, 3, nullptr);
    EXPECT_THROW(([&] { (void)potri_buffer_size<BK, T>(this->q(), zero_n, p.uplo); }()), batchlas::internal_error);
}

// blocked runs the batch in parts: with the chunk forced to 3 items, a 10-item batch is parts of
// 3, 3 and 4, every item is inverted (each verified, not a default sample), and the workspace is one part's.
TYPED_TEST(PotriTest, PotriBlockedPartsCoverEveryItem) {
    using T = typename TestFixture::T;
    constexpr Backend BK = TestFixture::BK;
    for (Uplo uplo : {Uplo::Lower, Uplo::Upper}) {
        auto p = make_sys<T>(70, 1, 10, uplo, 404u);
        UnifiedVector<int32_t> info;
        if (!this->factor(p, info)) GTEST_SKIP() << "no potrf route for this shape in this build";
        const select::ScopedPin<PotriChoice> pin("potri", PotriChoice{ops::potri::Blocked{32}});
        const size_t whole = potri_buffer_size<BK, T>(this->q(), p.A(), p.uplo);
        const ScopedEnvVar chunk("BATCHLAS_TUNE_L2_CHUNK_ITEMS", "3");
        const size_t part = potri_buffer_size<BK, T>(this->q(), p.A(), p.uplo);
        EXPECT_LT(part, whole);
        (void)this->run_potri(p, true);
        std::vector<int> all(p.batch);
        for (int it = 0; it < p.batch; ++it) all[it] = it;
        Matrix<T> A0(p.n, p.n, p.batch), X(p.n, p.n, p.batch);
        for (int it = 0; it < p.batch; ++it)
            for (int j = 0; j < p.n; ++j)
                for (int i = 0; i < p.n; ++i) {
                    A0(i, j, it) = p.full[it][i + static_cast<size_t>(j) * p.n];
                    X(i, j, it) = p.a[static_cast<size_t>(it) * p.stra + static_cast<size_t>(j) * p.lda + i];
                }
        EXPECT_VERIFY(T, verify::Check::solve, p.n, MV<T>::solve_residual(this->q(), A0, X, MV<T>{}, all));
    }
}

// The cta pin at its advertised ceiling launches and inverts; one past it is refused (a bad pin
// throws, never falls back) while Auto still serves it, and a foreign spelling throws.
TYPED_TEST(PotriTest, PotriCtaPinStraddlesItsCeiling) {
    using T = typename TestFixture::T;
    constexpr Backend BK = TestFixture::BK;
    const int ceiling = cta_ceiling<BK, T>(this->q());
    ASSERT_GE(ceiling, 32);
    for (int n : {ceiling, ceiling + 1}) {
        auto p = make_sys<T>(n, 1, 2, Uplo::Lower, 77u + n);
        UnifiedVector<int32_t> info;
        if (!this->factor(p, info)) continue;
        const auto factor = p.a;
        UnifiedVector<std::byte> ws(1);
        auto call = [&] { (void)potri<BK, T>(this->q(), p.A(), p.uplo, false, Span<std::byte>(ws.data(), std::size_t{0}), Span<int32_t>{}).wait(); };
        {
            const select::ScopedPin<PotriChoice> pin("potri", PotriChoice{ops::potri::Cta{}});
            if (n == ceiling) {
                EXPECT_NO_THROW(call());
                std::vector<std::vector<T>> inv;
                for (int it = 0; it < p.batch; ++it) inv.push_back(this->mirrored(p, it));
                EXPECT_VERIFY(T, verify::Check::solve, n, this->inverse_residual(p, inv)) << "at the ceiling " << n;
            } else {
                EXPECT_THROW(call(), std::invalid_argument);
            }
        }
        {
            const select::ScopedPin<PotriChoice> pin("potri", std::string_view("lpanel:panel=8"));
            EXPECT_THROW(call(), std::invalid_argument);
        }
        if (n == ceiling + 1) {
            p.a = factor;
            const size_t need = potri_buffer_size<BK, T>(this->q(), p.A(), p.uplo);
            ws = UnifiedVector<std::byte>(std::max<size_t>(need, 1));
            EXPECT_NO_THROW(([&] { (void)potri<BK, T>(this->q(), p.A(), p.uplo, false, Span<std::byte>(ws.data(), need), Span<int32_t>{}).wait(); }()));
        }
    }
}

// A saturating batch: 1024 identical items, every result bit-identical to item 0 (a race between
// work-items or work-groups shows as a differing item; small batches cannot race). Each family
// pinned, at orders whose CTA work-groups span several sub-groups (n = 37: 64 work-items, n = 100:
// 352) and, for blocked, several diagonal blocks per item. The host queue (no race to find, 77 s
// at n = 100) stops at 37.
TYPED_TEST(PotriTest, SaturatingBatchOfIdenticalItemsIsBitIdentical) {
    using T = typename TestFixture::T;
    const int ceiling = cta_ceiling<TestFixture::BK, T>(this->q());
    for (int n : {8, 37, 100})
        for (const PotriChoice& c : potri_pins()) {
            if ((c.index() == 0 && n > ceiling) || (TestFixture::BK == Backend::NETLIB && n > 37)) continue;
            auto p = make_sys<T>(n, 1, 1024, Uplo::Lower, 13u + n, /*identical=*/true);
            UnifiedVector<int32_t> info;
            if (!this->factor(p, info)) continue;
            {
                const select::ScopedPin<PotriChoice> pin("potri", c);
                (void)this->run_potri(p, true);
            }
            for (int it = 1; it < p.batch; ++it)
                for (size_t o = 0; o < static_cast<size_t>(p.stra); ++o)
                    ASSERT_TRUE(same_bits(p.a[static_cast<size_t>(it) * p.stra + o], p.a[o]))
                        << "potri item " << it << " offset " << o << " " << pin_name(c) << " n=" << n;
            auto one = make_sys<T>(n, 1, 1, Uplo::Lower, 13u + n, true);
            one.full.assign(1, p.full[0]);
            EXPECT_VERIFY(T, verify::Check::solve, n, this->inverse_residual(one, {this->mirrored(p, 0)}))
                << "n=" << n << " " << pin_name(c);
        }
}

// A coverage guard for batch > 65535 (a CUDA grid y/z limit), at n = 2: every item must equal item 0,
// and item 0 must be the inverse. Both families launch one work-group per item (blocked at n <= nb is
// the CTA kernel twice), so the item index past 65535 is exercised on each.
TYPED_TEST(PotriTest, BatchBeyondTheGridYLimitIsCovered) {
    using T = typename TestFixture::T;
    const int n = 2, batch = 70001;
    for (const PotriChoice& c : {PotriChoice{ops::potri::Cta{}}, PotriChoice{ops::potri::Blocked{32}}}) {
        auto p = make_sys<T>(n, 1, batch, Uplo::Upper, 4242u, /*identical=*/true);
        UnifiedVector<int32_t> info;
        if (!this->factor(p, info)) GTEST_SKIP() << "no potrf route for this shape in this build";
        UnifiedVector<int32_t> pinfo(batch, int32_t(-7));
        {
            const select::ScopedPin<PotriChoice> pin("potri", c);
            (void)this->run_potri(p, true, &pinfo);
        }
        for (int it = 0; it < batch; ++it) {
            ASSERT_EQ(pinfo[it], 0) << "item " << it;
            for (size_t o = 0; o < static_cast<size_t>(p.stra); ++o)
                ASSERT_TRUE(same_bits(p.a[static_cast<size_t>(it) * p.stra + o], p.a[o]))
                    << "item " << it << " offset " << o << " " << pin_name(c);
        }
        auto one = make_sys<T>(n, 1, 1, Uplo::Upper, 4242u, true);
        one.full.assign(1, p.full[0]);
        EXPECT_VERIFY(T, verify::Check::solve, n, this->inverse_residual(one, {this->mirrored(p, 0)})) << pin_name(c);
    }
}

// A non-empty info span shorter than the batch is refused, not silently ignored.
// BREAK (run 2026-10-25): drop the info-size check in potri.cc. RED: this test only.
TYPED_TEST(PotriTest, ShortInfoSpanIsRefused) {
    using T = typename TestFixture::T;
    auto p = make_sys<T>(4, 1, 3, Uplo::Lower, 8u);
    UnifiedVector<int32_t> info;
    if (!this->factor(p, info)) GTEST_SKIP() << "no potrf route for this shape in this build";
    UnifiedVector<int32_t> short_info(2, int32_t(-7));
    EXPECT_THROW((void)this->run_potri(p, false, &short_info), batchlas::invalid_argument);
}

// Linked is not reached: the coverage rows of a child process show each pinned family ran, and that
// blocked above nb reached its children (trmm, gemm, and herk or syrk) while cta reached none of them.
// threadsafe: the child re-executes the binary, so the device runtime starts fresh.
TYPED_TEST(PotriTest, PotriCoverageShowsEachFamilyAndTheBlockedChildren) {
    using T = typename TestFixture::T;
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    const std::string dir = ::testing::TempDir() + "potri_cov." + std::string(select::dtype_name<T>()) +
                            (TestFixture::BK == Backend::NETLIB ? ".netlib" : ".gpu");
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const std::string out = dir + "/cov";
    // Factored before coverage is switched on, so potrf's own children leave no rows.
    auto lo = make_sys<T>(16, 1, 3, Uplo::Lower, 5u), hi = make_sys<T>(100, 1, 3, Uplo::Lower, 6u);
    UnifiedVector<int32_t> info;
    if (!this->factor(lo, info) || !this->factor(hi, info)) GTEST_SKIP() << "no potrf route for this shape in this build";
    auto child = [&] {
        const ScopedEnvVar cov("BATCHLAS_COVERAGE_OUT", out.c_str());
        const ScopedEnvVar clear("BATCHLAS_POTRI_ROUTE", nullptr);
        batchlas::coverage::g_dynamic_enabled = true;
        {
            const select::ScopedPin<PotriChoice> pin("potri", PotriChoice{ops::potri::Cta{}});
            (void)this->run_potri(lo, false);
        }
        const select::ScopedPin<PotriChoice> pin("potri", PotriChoice{ops::potri::Blocked{32}});
        (void)this->run_potri(hi, false);
        std::exit(0);
    };
    EXPECT_EXIT(child(), ::testing::ExitedWithCode(0), "");
    std::map<std::string, std::set<std::string>> algos;  // op -> chosen spellings
    std::set<std::string> blocked_children;
    for (const auto& ent : std::filesystem::directory_iterator(dir)) {
        std::ifstream in(ent.path());
        for (std::string line; std::getline(in, line);) {
            if (line.rfind("reached,", 0) != 0) continue;
            std::vector<std::string> f;
            std::stringstream ss(line);
            for (std::string tok; std::getline(ss, tok, ',');) f.push_back(tok);
            ASSERT_GE(f.size(), 11u) << line;
            algos[f[1]].insert(f[10]);
            if (f[1] != "potri") blocked_children.insert(f[1]);
        }
    }
    std::filesystem::remove_all(dir);
    EXPECT_EQ(algos["potri"], (std::set<std::string>{"cta", "blocked:nb=32"}));
    constexpr bool kCx = !std::is_same_v<T, RealOf<T>>;
    // cta at n = 16 calls no child; blocked at n = 100 calls these, and nothing else.
    EXPECT_EQ(blocked_children, (std::set<std::string>{"trmm", "gemm", kCx ? "herk" : "syrk"}));
}
