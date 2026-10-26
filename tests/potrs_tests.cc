// potrs (solve with a Cholesky factor), a routed op with two families: `fused` (both substitutions in
// one launch, B in SLM) and `trsm` (two public trsm calls). Residuals through batchlas::verify on graded
// SPD inputs, each family pinned and checked reached (coverage rows), the fused ceilings straddled and
// launched at, Auto against synthetic and shipped tables, bit-for-bit agreement with posv on the same
// solve, padded ld and stride with a poisoned unreferenced triangle, and a saturating batch.
// evidence: docs/perf/potrs.md#potrs-selection
#include <gtest/gtest.h>

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/posv.hh>
#include <batchlas/blas/functions/potrf.hh>
#include <batchlas/blas/functions/potrs.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/no_route.hh>
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>

#include <tests/shipped_table_pick.hh>
#include <tests/test_utils.hh>
#include <batchlas/verify/tolerance.hh>

#include <src/ops/getrs/getrs_native.hh>
#include <src/ops/posv/choice.hh>
#include <src/ops/potrs/choice.hh>
#include <src/ops/potrs/potrs.hh>
#include <src/select/coverage.hh>
#include <src/select/select.hh>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

using namespace batchlas;

namespace {

namespace ps = ops::potrs;
using C = ps::PotrsChoice;
using Pin = select::ScopedPin<C>;

template <typename T>
using MV = MatrixView<T, MatrixFormat::Dense>;
template <typename T>
using RealOf = typename batchlas::base_type<T>::type;

template <typename T, Backend B>
struct PotrsConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

template <typename T>
bool same_bits(const T& a, const T& b) {
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}
template <typename T>
T poison() {
    return MV<T>::make(RealOf<T>(-999), RealOf<T>(777));
}
const char* uplo_s(Uplo u) { return u == Uplo::Lower ? "L" : "U"; }

// One Hermitian positive-definite item: D C D with C strictly diagonally dominant (condition below 8,
// every off-diagonal entry with a nonzero imaginary part) and D a diagonal graded geometrically up to
// sqrt(cond), its entries shuffled so large and small rows interleave. Column-major n x n.
template <typename T>
std::vector<T> make_hpd(int n, double cond, std::mt19937& gen) {
    using R = RealOf<T>;
    std::uniform_real_distribution<R> d(R(0.1), R(1));
    std::vector<double> g(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) g[i] = n > 1 ? std::pow(cond, 0.5 * double(i) / double(n - 1)) : 1.0;
    std::shuffle(g.begin(), g.end(), gen);
    std::vector<T> A(static_cast<size_t>(n) * n);
    for (int j = 0; j < n; ++j) {
        A[j + static_cast<size_t>(j) * n] = MV<T>::make(R((2.5 + 0.5 * d(gen)) * g[j] * g[j]), R(0));
        for (int i = j + 1; i < n; ++i) {
            const double x = ((gen() & 1) ? d(gen) : -d(gen)) * g[i] * g[j] / n, y = d(gen) * g[i] * g[j] / n;
            A[i + static_cast<size_t>(j) * n] = MV<T>::make(R(x), R(y));
            A[j + static_cast<size_t>(i) * n] = MV<T>::make(R(x), R(-y));
        }
    }
    return A;
}

// The graded condition the accuracy cases use: 1e6 in double, 1e3 in single precision.
template <typename T>
double graded_cond() {
    return sizeof(RealOf<T>) == 8 ? 1e6 : 1e3;
}

// A and B at padded ld and a stride that is not ld*cols. Everything potrs may not touch (the other
// triangle of A, the ld pads, the inter-item gaps) holds a large finite value a kernel would consume.
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
Sys<T> make_sys(int n, int nrhs, int batch, Uplo uplo, unsigned seed, double cond = 1.0, bool identical = false) {
    using R = RealOf<T>;
    Sys<T> p;
    p.n = n, p.nrhs = nrhs, p.batch = batch, p.uplo = uplo;
    p.lda = n + 3, p.stra = p.lda * n + 5, p.ldb = n + 2, p.strb = p.ldb * nrhs + 7;
    p.a = UnifiedVector<T>(static_cast<size_t>(p.stra) * batch, poison<T>());
    p.b = UnifiedVector<T>(static_cast<size_t>(p.strb) * batch, poison<T>());
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
            p.full[it] = make_hpd<T>(n, cond, gen);
            p.rhs[it].resize(static_cast<size_t>(n) * nrhs);
            for (auto& v : p.rhs[it]) v = MV<T>::make(d(gen), d(gen));
        }
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                if (p.owned(i, j))
                    p.a[static_cast<size_t>(it) * p.stra + static_cast<size_t>(j) * p.lda + i] = p.full[it][i + static_cast<size_t>(j) * n];
        for (int k = 0; k < nrhs; ++k)
            for (int i = 0; i < n; ++i)
                p.b[static_cast<size_t>(it) * p.strb + static_cast<size_t>(k) * p.ldb + i] = p.rhs[it][i + static_cast<size_t>(k) * n];
    }
    return p;
}

// The outermost potrs trace line for whatever `run` calls, and the choice it names.
template <typename F>
std::string traced_choice(F&& run) {
    const ScopedEnvVar trace("BATCHLAS_SELECT_TRACE", "1");
    ::testing::internal::CaptureStderr();
    try {
        run();
    } catch (...) {
        (void)::testing::internal::GetCapturedStderr();
        throw;
    }
    const std::string err = ::testing::internal::GetCapturedStderr();
    std::istringstream in(err);
    for (std::string line; std::getline(in, line);)
        if (line.rfind("potrs ", 0) == 0 && line.find(" -> ") != std::string::npos) {
            const std::string tail = line.substr(line.find(" -> ") + 4);
            return tail.substr(0, tail.find(' '));
        }
    return "<no potrs trace line in: " + err + ">";
}

struct TableGuard {
    ~TableGuard() { select::testing::use_embedded_tables(); }
};

template <typename Config>
class PotrsTest : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    static constexpr Backend BK = Config::BackendVal;

    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        if (this->HasFatalFailure() || ::testing::Test::IsSkipped()) return;
        if (!this->ctx) GTEST_SKIP() << "no queue";
        // NETLIB serves potrf and trsm through LAPACKE/CBLAS only, so a vendor-free build has nothing to test there.
        if (BK == Backend::NETLIB && !(BATCHLAS_HAS_LAPACKE && BATCHLAS_HAS_CBLAS))
            GTEST_SKIP() << "NETLIB without LAPACKE/CBLAS has no potrf or trsm route";
    }

    Queue& q() { return *this->ctx; }
    const select::Device& dev() { return select::device_of<BK>(q()); }
    bool fits(int n, int nrhs) { return ps::fused_fits<T>(dev(), n, nrhs); }
    // The largest n the fused solve takes at nrhs = 8, from the device's own SLM budget.
    int fused_ceiling_n() {
        return static_cast<int>(sycl_getrs::getrs_fused_max_rhs_elems<T>(static_cast<size_t>(dev().slm_budget)) / 8);
    }

    // The Cholesky factor in A's triangle, through the public potrf. False when the build has no potrf route.
    bool factor(Sys<T>& p) {
        UnifiedVector<int32_t> info(static_cast<size_t>(p.batch), int32_t(-12345));
        try {
            const size_t need = potrf_buffer_size<BK, T>(q(), p.A(), p.uplo);
            UnifiedVector<std::byte> ws(std::max<size_t>(need, 1));
            (void)potrf<BK, T>(q(), p.A(), p.uplo, Span<std::byte>(ws.data(), need), info.to_span());
        } catch (const NoRouteError&) {
            return false;
        }
        q().wait();
        for (int it = 0; it < p.batch; ++it) EXPECT_EQ(info[it], 0) << "potrf failed on item " << it;
        return true;
    }
    void solve(Sys<T>& p) {
        (void)potrs<BK, T>(q(), p.A(), p.B(), p.uplo).wait();
        q().wait();
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
    // B's ld pads and inter-item gaps still hold the poison.
    bool b_pads_intact(const Sys<T>& p) const {
        for (int it = 0; it < p.batch; ++it)
            for (size_t o = 0; o < static_cast<size_t>(p.strb); ++o) {
                const int col = static_cast<int>(o) / p.ldb, row = static_cast<int>(o) % p.ldb;
                if (col < p.nrhs && row < p.n) continue;
                if (!same_bits(p.b[static_cast<size_t>(it) * p.strb + o], poison<T>())) return false;
            }
        return true;
    }
};

using PotrsTestTypes = typename test_utils::backend_types<PotrsConfig>::type;

}  // namespace

TYPED_TEST_SUITE(PotrsTest, PotrsTestTypes);

// Each family pinned, and Auto, on graded SPD factors (condition 1e6 double, 1e3 single), both triangles,
// an order ladder through the fused block edge (nb = 16: 16/17, 33 = 2*16 + 1, 130) and the accumulator
// buckets (nrhs 1, 2, 3, 4, 5, 8; 9 and 20 past the fused width). The factor is read-only, B's pads survive.
// BREAK (run 2026-10-26): getrs_fused.cc's potrs body stages the Upper forward block from ld_h(j+s, j+t), the
// poisoned triangle. RED: the fused Upper cases of this test, ASecondRightHandSide and SaturatingBatch, every
// CUDA type; trsm, Lower and the posv comparison (same kernel) GREEN.
// BREAK (run 2026-10-26): the packed block solve gives every slot the group's first column (c = g * P). RED:
// exactly the fused cases with nrhs >= 3 (P >= 2) here, in ASecondRightHandSide, PinnedCandidatesStraddle and
// SaturatingBatch, every CUDA type; nrhs 1 and 2 (one slot) GREEN.
TYPED_TEST(PotrsTest, EachFamilySolvesGradedSystemsOnBothTriangles) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_POTRS_ROUTE", nullptr);
    for (const char* family : {"fused", "trsm", "auto"})
        for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
            for (int n : {1, 3, 8, 16, 17, 33, 130})
                for (int nrhs : {1, 2, 3, 4, 5, 8, 9, 20}) {
                    const bool fused = std::string_view(family) == "fused";
                    if (fused && !this->fits(n, nrhs)) continue;
                    auto p = make_sys<T>(n, nrhs, 3, uplo, 100u + n * 31 + nrhs, graded_cond<T>());
                    if (!this->factor(p)) continue;
                    const UnifiedVector<T> factor_bits = p.a;
                    {
                        const Pin pin("potrs", std::string_view(family));
                        this->solve(p);
                    }
                    const std::string what = std::string(family) + " " + uplo_s(uplo) + " n=" + std::to_string(n) +
                                             " nrhs=" + std::to_string(nrhs);
                    EXPECT_VERIFY(T, verify::Check::solve, n, this->solve_res(p)) << what;
                    for (size_t i = 0; i < p.a.size(); ++i) ASSERT_TRUE(same_bits(p.a[i], factor_bits[i])) << what << ": A written at " << i;
                    ASSERT_TRUE(this->b_pads_intact(p)) << what << ": a B pad was written";
                }
}

// The factor is reusable: a second right-hand-side set on the same factor, each family.
TYPED_TEST(PotrsTest, ASecondRightHandSideReusesTheFactor) {
    using T = typename TestFixture::T;
    for (const char* family : {"fused", "trsm"})
        for (int n : {8, 40}) {
            if (std::string_view(family) == "fused" && !this->fits(n, 4)) continue;
            auto p = make_sys<T>(n, 4, 5, Uplo::Upper, 7u + n);
            if (!this->factor(p)) continue;
            const Pin pin("potrs", std::string_view(family));
            this->solve(p);
            auto p2 = make_sys<T>(n, 4, 5, Uplo::Upper, 7u + n);
            p2.a = p.a;
            for (int it = 0; it < p2.batch; ++it)
                for (int k = 0; k < p2.nrhs; ++k)
                    for (int i = 0; i < n; ++i) {
                        p2.rhs[it][i + static_cast<size_t>(k) * n] = MV<T>::make(double(i + 1), double(-k));
                        p2.b[static_cast<size_t>(it) * p2.strb + static_cast<size_t>(k) * p2.ldb + i] = p2.rhs[it][i + static_cast<size_t>(k) * n];
                    }
            this->solve(p2);
            EXPECT_VERIFY(T, verify::Check::solve, n, this->solve_res(p2)) << family << " second right-hand side n=" << n;
        }
}

// potrs's `fused` is posv's `cta` solve and `trsm` is posv's `blocked` solve, bit for bit.
// BREAK (run 2026-10-26): potrs.cc's launch passes fused = false whatever the choice. RED: this test only,
// every CUDA type (the fused cases); every other potrs test GREEN, NETLIB GREEN.
TYPED_TEST(PotrsTest, FusedMatchesPosvCtaAndTrsmMatchesPosvBlockedBitForBit) {
    using T = typename TestFixture::T;
    namespace pv = ops::posv;
    for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
        for (auto [n, nrhs, fused] : {std::tuple{8, 4, true}, std::tuple{9, 2, true}, std::tuple{40, 8, true},
                                      std::tuple{8, 4, false}, std::tuple{20, 16, false}}) {
            if (fused && !this->fits(n, nrhs)) continue;
            auto p = make_sys<T>(n, nrhs, 7, uplo, 7u + n);
            auto q2 = make_sys<T>(n, nrhs, 7, uplo, 7u + n);
            {
                const size_t need = posv_buffer_size<TestFixture::BK, T>(this->q(), p.A(), p.B(), uplo);
                UnifiedVector<std::byte> ws(std::max<size_t>(need, 1));
                const select::ScopedPin<pv::PosvChoice> pin("posv", fused ? pv::PosvChoice{pv::Cta{}} : pv::PosvChoice{pv::Blocked{}});
                UnifiedVector<int32_t> info(7, int32_t(-1));
                (void)posv<TestFixture::BK, T>(this->q(), p.A(), p.B(), uplo, Span<std::byte>(ws.data(), need), info.to_span()).wait();
            }
            if (!this->factor(q2)) continue;
            {
                const Pin pin("potrs", fused ? C{ps::Fused{}} : C{ps::Trsm{}});
                this->solve(q2);
            }
            for (size_t i = 0; i < p.a.size(); ++i) ASSERT_TRUE(same_bits(p.a[i], q2.a[i])) << "factor differs at " << i;
            for (size_t i = 0; i < p.b.size(); ++i)
                ASSERT_TRUE(same_bits(p.b[i], q2.b[i])) << "X differs at " << i << " n=" << n << " nrhs=" << nrhs << " fused=" << fused;
        }
}

// posv composes potrf and the potrs solve, so a non-positive-definite item is flagged in `info` at the
// minor potrf names, and every other item's X is bit-identical to a batch without the bad item.
// ARMED BREAK: in potrs::solve() skip the Lower NoTrans trsm. EXPECTED: RED here (blocked) and in the
// posv comparison above; the info assertions alone stay GREEN.
TYPED_TEST(PotrsTest, NonSpdItemIsFlaggedAndTheOthersAreUndisturbed) {
    using T = typename TestFixture::T;
    using namespace ops::posv;
    for (bool blocked : {false, true})
        for (int n : {8, 20}) {
            const int nrhs = blocked ? 16 : 4, bad = 2, batch = 5;
            if (!blocked && !this->fits(n, nrhs)) continue;
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

// The fused pin on both sides of each ceiling: the accumulator width (nrhs 8 runs, 9 is refused) and the
// SLM capacity, LAUNCHED at the advertised ceiling (nrhs = 8, n = cap / 8 runs and solves; n + 1 is
// refused). trsm and Auto run on both sides; an unknown spelling throws. On a CPU queue fused never runs.
// BREAK (run 2026-10-26): fused_fits drops `nrhs <= kGetrsFusedMaxRhs`. RED: this test (the nrhs = 9 pin is
// refused by the kernel's entry check, not by selection) and EachFamilySolves (its fused pin at nrhs 9 and 20
// reaches the kernel and throws), every CUDA type; every other potrs test GREEN, NETLIB GREEN.
TYPED_TEST(PotrsTest, PinnedCandidatesStraddleTheirLimits) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_POTRS_ROUTE", nullptr);
    const bool gpu = this->fits(1, 1);
    const int ceil_n = gpu ? this->fused_ceiling_n() : 16;
    if (gpu) ASSERT_GT(ceil_n, 64) << "the fused SLM capacity reads implausibly small";
    struct Cell { int n, nrhs; bool fused_runs; };
    const Cell cells[] = {{24, 8, gpu}, {24, 9, false}, {ceil_n, 8, gpu}, {ceil_n + 1, 8, false}};
    for (const Cell& k : cells) {
        const int batch = k.n > 256 ? 2 : 3;
        for (const char* family : {"fused", "trsm", "auto", "no_such_family"}) {
            auto p = make_sys<T>(k.n, k.nrhs, batch, Uplo::Lower, 5u + k.n);
            if (!this->factor(p)) break;
            const std::string what = std::string(family) + " n=" + std::to_string(k.n) + " nrhs=" + std::to_string(k.nrhs);
            const std::string_view f(family);
            const Pin pin("potrs", f);
            if (f == "no_such_family" || (f == "fused" && !k.fused_runs)) {
                // Refused by selection itself (can_run), not by the kernel's own entry check.
                std::string err;
                try {
                    this->solve(p);
                } catch (const std::invalid_argument& e) {
                    err = e.what();
                }
                EXPECT_NE(err.find(f == "fused" ? "cannot run this shape" : "no_such_family"), std::string::npos) << what << ": " << err;
                continue;
            }
            EXPECT_NO_THROW(this->solve(p)) << what;
            EXPECT_VERIFY(T, verify::Check::solve, k.n, this->solve_res(p)) << what;
        }
    }
}

// Every pinned family records a `reached` coverage row under its own spelling, with potrs's key
// (m = k = order, n = nrhs) and uplo. threadsafe: the child re-executes the binary, so CUDA starts fresh.
TYPED_TEST(PotrsTest, CoverageRowNamesThePinnedFamily) {
    using T = typename TestFixture::T;
    if (!this->fits(16, 4)) GTEST_SKIP() << "the fused family cannot run on this queue";
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    const std::string dir = ::testing::TempDir() + "potrs_cov." + std::string(select::dtype_name<T>());
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const std::string out = dir + "/cov";
    auto child = [&] {
        const ScopedEnvVar cov("BATCHLAS_COVERAGE_OUT", out.c_str());
        const ScopedEnvVar clear("BATCHLAS_POTRS_ROUTE", nullptr);
        batchlas::coverage::g_dynamic_enabled = true;
        for (auto [n, nrhs, uplo, c] : {std::tuple{16, 4, Uplo::Lower, C{ps::Fused{}}}, std::tuple{24, 3, Uplo::Upper, C{ps::Trsm{}}}}) {
            auto p = make_sys<T>(n, nrhs, 2, uplo, 3u);
            if (!this->factor(p)) std::exit(3);
            const Pin pin("potrs", c);
            this->solve(p);
        }
        std::exit(0);
    };
    EXPECT_EXIT(child(), ::testing::ExitedWithCode(0), "");
    std::map<std::string, std::vector<std::string>> rows;  // "m n k" -> fields
    for (const auto& ent : std::filesystem::directory_iterator(dir)) {
        std::ifstream in(ent.path());
        for (std::string line; std::getline(in, line);) {
            if (line.rfind("reached,potrs,", 0) != 0) continue;
            std::vector<std::string> f;
            std::stringstream ss(line);
            for (std::string tok; std::getline(ss, tok, ',');) f.push_back(tok);
            ASSERT_GE(f.size(), 16u) << line;
            rows[f[5] + " " + f[6] + " " + f[7]] = f;
        }
    }
    std::filesystem::remove_all(dir);
    ASSERT_EQ(rows.size(), 2u);
    for (const auto& [key, algo, uplo] : {std::tuple{std::string("16 4 16"), "fused", Uplo::Lower},
                                          std::tuple{std::string("24 3 24"), "trsm", Uplo::Upper}}) {
        ASSERT_TRUE(rows.count(key)) << "no row " << key;
        EXPECT_EQ(rows[key][10], algo) << key;
        EXPECT_EQ(rows[key][15], std::to_string(static_cast<int>(uplo))) << key;
    }
}

// key_of's every field reaches choose(): a synthetic table whose winner changes with uplo, n, nrhs and
// batch alone. BREAK (run 2026-10-26): key_of spells nrhs as 1. RED: exactly the nrhs probe of this test
// and AutoReadsTheShippedTable (its nrhs-straddling cells), every CUDA type; every other potrs test GREEN.
TYPED_TEST(PotrsTest, AutoReadsEveryKeyField) {
    using T = typename TestFixture::T;
    if (!this->fits(64, 8)) GTEST_SKIP() << "the fused family cannot run on this queue";
    const ScopedEnvVar clear("BATCHLAS_POTRS_ROUTE", nullptr);
    const TableGuard restore;
    const std::string dtype(select::dtype_name<T>());
    const std::string devk = this->dev().key;
    std::vector<std::pair<std::string, std::string>> files;
    for (const auto& t : select::embedded_tables())
        if (t.name.rfind("potrs.", 0) != 0) files.emplace_back(std::string(t.name), std::string(t.text));
    files.emplace_back("potrs." + dtype + "." + devk + ".txt",
                       "# op=potrs dtype=" + dtype + " device=" + devk + " kernels=unknown\n"
                       "# keys: uplo:exact n:log:2 nrhs:log batch:log\n"
                       "uplo=L n=8 nrhs=2 batch=128 | fused 1 | trsm 2\n"
                       "uplo=L n=8 nrhs=2 batch=32768 | trsm 1 | fused 2\n"
                       "uplo=L n=8 nrhs=8 batch=128 | trsm 1 | fused 2\n"
                       "uplo=L n=64 nrhs=2 batch=128 | trsm 1 | fused 2\n"
                       "uplo=U n=8 nrhs=2 batch=128 | trsm 1 | fused 2\n");
    select::testing::set_builtin_tables(test_utils::dtype_keyed(std::move(files)));
    struct Probe { Uplo uplo; int n, nrhs, batch; const char* expect; const char* field; };
    const Probe probes[] = {{Uplo::Lower, 8, 2, 128, "fused", "base"},   {Uplo::Lower, 8, 2, 32768, "trsm", "batch"},
                            {Uplo::Lower, 8, 8, 128, "trsm", "nrhs"},    {Uplo::Lower, 64, 2, 128, "trsm", "n"},
                            {Uplo::Upper, 8, 2, 128, "trsm", "uplo"}};
    for (const auto& k : probes) {
        auto p = make_sys<T>(k.n, k.nrhs, k.batch, k.uplo, 43u);
        if (!this->factor(p)) continue;
        EXPECT_EQ(traced_choice([&] { this->solve(p); }), k.expect) << "the " << k.field << " row";
    }
}

// Auto against the shipped table of this device on both sides of the table's crossovers (nrhs 1/2 and
// 4/5 at n 16 and 64, the fused width 8/9, n 12/16, an off-grid batch), so a retune moves the expectation.
TYPED_TEST(PotrsTest, AutoReadsTheShippedTable) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_POTRS_ROUTE", nullptr);
    struct Cell { Uplo uplo; int n, nrhs, batch; };
    const Cell cells[] = {{Uplo::Lower, 16, 1, 128},  {Uplo::Lower, 16, 2, 128},  {Uplo::Lower, 16, 8, 2048},
                          {Uplo::Lower, 16, 9, 2048}, {Uplo::Lower, 64, 1, 512},  {Uplo::Lower, 64, 5, 512},
                          {Uplo::Upper, 64, 4, 512},  {Uplo::Upper, 12, 8, 300},  {Uplo::Lower, 256, 2, 128},
                          {Uplo::Upper, 256, 8, 128}};
    for (Cell k : cells) {
        if (TestFixture::BK == Backend::NETLIB) k.batch = std::min(k.batch, 16);  // no CPU table: the batch keys nothing
        const std::string label = std::string(uplo_s(k.uplo)) + " n=" + std::to_string(k.n) + " nrhs=" + std::to_string(k.nrhs) +
                                  " batch=" + std::to_string(k.batch);
        auto p = make_sys<T>(k.n, k.nrhs, k.batch, k.uplo, 41u + k.n);
        if (!this->factor(p)) continue;
        const std::string want = test_utils::shipped_table_pick<C>(
            "potrs", select::dtype_name<T>(), this->dev(),
            select::Key{{"uplo", uplo_s(k.uplo)}, {"n", k.n}, {"nrhs", k.nrhs}, {"batch", k.batch}},
            [&] {
                auto s = make_sys<T>(k.n, k.nrhs, 1, k.uplo, 3u);
                (void)this->factor(s);
                this->solve(s);
            });
        const std::string expect = want == test_utils::kNoTableEntryRuns ? "trsm" : want;
        EXPECT_EQ(traced_choice([&] { this->solve(p); }), expect) << label;
        EXPECT_VERIFY(T, verify::Check::solve, k.n, this->solve_res(p)) << "auto " << label;
    }
}

// A saturating batch: 1024 identical items, every result bit-identical to item 0 (a race between items or
// sub-groups shows as a differing item; small batches cannot race). Each family pinned, at one block
// (n = 16), several blocks (n = 40) and a 4-sub-group work-group (n = 130), both triangles.
// BREAK (run 2026-10-26): drop the forward-pass barrier between the block staging and the block solve in
// getrs_fused.cc's potrs body. RED: this test and EachFamilySolves (fused), every CUDA type, and
// PinnedCandidatesStraddle (double, cdouble); trsm and NETLIB GREEN.
TYPED_TEST(PotrsTest, SaturatingBatchOfIdenticalItemsIsBitIdentical) {
    using T = typename TestFixture::T;
    if (TestFixture::BK == Backend::NETLIB) GTEST_SKIP() << "no shared-local-memory kernel runs on the host queue";
    for (const char* family : {"fused", "trsm"})
        for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
            for (auto [n, nrhs] : {std::tuple{16, 8}, std::tuple{40, 5}, std::tuple{130, 8}}) {
                if (std::string_view(family) == "fused" && !this->fits(n, nrhs)) continue;
                auto p = make_sys<T>(n, nrhs, 1024, uplo, 13u + n, graded_cond<T>(), /*identical=*/true);
                if (!this->factor(p)) continue;
                {
                    const Pin pin("potrs", std::string_view(family));
                    this->solve(p);
                }
                for (int it = 1; it < p.batch; ++it)
                    for (size_t o = 0; o < static_cast<size_t>(p.strb); ++o)
                        ASSERT_TRUE(same_bits(p.b[static_cast<size_t>(it) * p.strb + o], p.b[o]))
                            << family << " " << uplo_s(uplo) << " n=" << n << " item " << it << " offset " << o;
                auto one = make_sys<T>(n, nrhs, 1, uplo, 13u + n, graded_cond<T>(), true);
                std::copy(p.b.data(), p.b.data() + one.strb, one.b.data());
                EXPECT_VERIFY(T, verify::Check::solve, n, this->solve_res(one)) << family << " n=" << n;
            }
}

// An empty batch is a no-op under any pin (it never reaches selection); an empty problem and a non-square
// A are refused before selection.
TYPED_TEST(PotrsTest, DegenerateCallsAreRefusedOrNoOps) {
    using T = typename TestFixture::T;
    auto p = make_sys<T>(12, 2, 3, Uplo::Lower, 3u);
    MV<T> empty_a(p.a.data(), 12, 12, p.lda, p.stra, 0, nullptr), empty_b(p.b.data(), 12, 2, p.ldb, p.strb, 0, nullptr);
    {
        const Pin pin("potrs", std::string_view("no_such_family"));
        EXPECT_NO_THROW(([&] { (void)potrs<TestFixture::BK, T>(this->q(), empty_a, empty_b, Uplo::Lower).wait(); }()));
    }
    MV<T> zero_rhs(p.b.data(), 12, 0, p.ldb, p.strb, 3, nullptr);
    EXPECT_THROW(([&] { (void)potrs<TestFixture::BK, T>(this->q(), p.A(), zero_rhs, Uplo::Lower); }()), batchlas::internal_error);
    MV<T> nonsquare(p.a.data(), 12, 11, p.lda, p.stra, 3, nullptr);
    EXPECT_THROW(([&] { (void)potrs<TestFixture::BK, T>(this->q(), nonsquare, p.B(), Uplo::Lower); }()), batchlas::invalid_argument);
}
