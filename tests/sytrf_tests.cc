// sytrf / sytrs / sysv: Bunch-Kaufman LDL^T with LAPACK ?sytrf semantics. Every route is pinned and
// checked against the full symmetric matrix through batchlas::verify; the operand layout is never
// natural (padded ld, padded batch stride, both pads and the unreferenced triangle poisoned with large
// finite values), complex data has nonzero imaginary parts and is symmetric, not Hermitian.
// evidence: docs/perf/sytrf.md
#include <gtest/gtest.h>

#include <batchlas/blas/functions/sysv.hh>
#include <batchlas/blas/functions/sytrf.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/verify/tolerance.hh>

#include <tests/test_utils.hh>

#include <src/ops/sysv/choice.hh>
#include <src/ops/sytrf/choice.hh>
#include <src/select/select.hh>
#include <src/select/vendor.hh>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstring>
#include <optional>
#include <random>
#include <string>
#include <vector>

using namespace batchlas;

namespace {

template <typename T>
using MV = MatrixView<T, MatrixFormat::Dense>;

using verify::Check;

// random: uniform [-1, 1], every pivot kind occurs; dominant: |diagonal| = n + 2 with alternating
// signs, so every pivot is 1 x 1 with no interchange; kkt: [[H, A^T], [A, 0]] with a small H, which
// forces 2 x 2 pivots; zero: the zero matrix (info > 0 at the first step).
enum class Gen { random, dominant, kkt, zero };

template <typename T>
struct Sys {
    int n = 0, nrhs = 0, batch = 0, lda = 0, stra = 0, ldb = 0, strb = 0;
    Uplo uplo = Uplo::Lower;
    UnifiedVector<T> a, b;
    std::vector<T> a_in, b_in;
    std::vector<T> full;  // the symmetric matrix the triangle stands for, n x n per item, ld n
    UnifiedVector<int64_t> piv;
    UnifiedVector<int32_t> info;
};

template <typename T>
T draw(std::mt19937& g, double scale) {
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    const double re = u(g), im = u(g);
    return MV<T>::make(scale * re, scale * im);
}

template <typename T>
std::vector<T> symmetric(int n, Gen gen, std::mt19937& g) {
    std::vector<T> m(size_t(n) * n, T{});
    if (gen == Gen::zero) return m;
    const int nc = gen == Gen::kkt ? n / 3 : 0;  // constraint rows, the zero block
    const int nh = n - nc;
    for (int j = 0; j < n; ++j)
        for (int i = j; i < n; ++i) {
            T v = draw<T>(g, 1.0);
            if (gen == Gen::kkt) {
                if (i < nh) v = draw<T>(g, 1e-2);          // H
                else if (j >= nh) v = T{};                 // the zero block
            }
            if (gen == Gen::dominant && i == j) v = MV<T>::make((j % 2 ? -1.0 : 1.0) * (n + 2), 0.5);
            m[size_t(j) * n + i] = v;
            m[size_t(i) * n + j] = v;
        }
    return m;
}

template <typename T>
bool referenced(Uplo uplo, int i, int j) {
    return uplo == Uplo::Lower ? i >= j : i <= j;
}

template <typename T>
Sys<T> make_sys(int n, int nrhs, int batch, Uplo uplo, Gen gen, unsigned seed, bool identical = false) {
    Sys<T> p;
    p.n = n; p.nrhs = nrhs; p.batch = batch; p.uplo = uplo;
    p.lda = n + 3; p.stra = p.lda * n + 7;
    p.ldb = n + 2; p.strb = p.ldb * nrhs + 5;
    const T pad = MV<T>::make(-9.75e3, 4.5e3);
    const T tri = MV<T>::make(7.25e5, -3.5e5);  // in range, large, finite: a kernel that reads it is wrong
    p.a = UnifiedVector<T>(size_t(p.stra) * batch, pad);
    p.b = UnifiedVector<T>(size_t(p.strb) * batch, pad);
    p.full.assign(size_t(n) * n * batch, T{});
    p.piv = UnifiedVector<int64_t>(size_t(n) * batch, int64_t(0x0BADBEEF0BADBEEFLL));
    p.info = UnifiedVector<int32_t>(size_t(batch), int32_t(-12345));
    std::mt19937 g(seed);
    for (int bi = 0; bi < batch; ++bi) {
        if (identical && bi > 0) g.seed(seed);
        const auto m = symmetric<T>(n, gen, g);
        std::copy(m.begin(), m.end(), p.full.begin() + size_t(bi) * n * n);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                p.a[size_t(bi) * p.stra + size_t(j) * p.lda + i] = referenced<T>(uplo, i, j) ? m[size_t(j) * n + i] : tri;
        for (int c = 0; c < nrhs; ++c)
            for (int i = 0; i < n; ++i) p.b[size_t(bi) * p.strb + size_t(c) * p.ldb + i] = draw<T>(g, 1.0);
    }
    p.a_in.assign(p.a.begin(), p.a.end());
    p.b_in.assign(p.b.begin(), p.b.end());
    return p;
}

template <typename T>
MV<T> a_view(Sys<T>& p) { return MV<T>(p.a.data(), p.n, p.n, p.lda, p.stra, p.batch); }
template <typename T>
MV<T> b_view(Sys<T>& p) { return MV<T>(p.b.data(), p.n, p.nrhs, p.ldb, p.strb, p.batch); }

template <typename T>
std::vector<int> all_items(const Sys<T>& p) {
    std::vector<int> v(size_t(p.batch));
    for (int i = 0; i < p.batch; ++i) v[size_t(i)] = i;
    return v;
}

template <typename T>
double residual(Queue& q, Sys<T>& p, const std::vector<int>& items) {
    return MV<T>::solve_residual(q, MV<T>(p.full.data(), p.n, p.n, p.n, p.n * p.n, p.batch), b_view(p),
                                 MV<T>(p.b_in.data(), p.n, p.nrhs, p.ldb, p.strb, p.batch), items);
}

template <typename T>
bool same(const T& x, const T& y) { return std::memcmp(&x, &y, sizeof(T)) == 0; }

// The unreferenced triangle and both pads must come back bit-unchanged: the first offender, or -1.
template <typename T>
long long first_touched(const Sys<T>& p) {
    for (int bi = 0; bi < p.batch; ++bi)
        for (int idx = 0; idx < p.stra; ++idx) {
            const int j = idx / p.lda, i = idx % p.lda;
            const bool live = j < p.n && i < p.n && referenced<T>(p.uplo, i, j);
            const size_t at = size_t(bi) * p.stra + idx;
            if (!live && !same(p.a[at], p.a_in[at])) return (long long)at;
        }
    for (int bi = 0; bi < p.batch; ++bi)
        for (int idx = 0; idx < p.strb; ++idx) {
            const int j = idx / p.ldb, i = idx % p.ldb;
            const size_t at = size_t(bi) * p.strb + idx;
            if (!(j < p.nrhs && i < p.n) && !same(p.b[at], p.b_in[at])) return -2 - (long long)at;
        }
    return -1;
}

// LAPACK's encoding, checked: |ipiv| in [1, n], a 2 x 2 block is two equal negative entries.
// Returns {1 x 1 interchanges, 2 x 2 blocks}, or {-1, -1} for a malformed list.
template <typename T>
std::pair<int, int> pivot_kinds(const Sys<T>& p, int item) {
    const int* ip = reinterpret_cast<const int*>(p.piv.data()) + size_t(item) * p.n;
    int swaps = 0, blocks = 0;
    for (int k = 0; k < p.n;) {
        const int v = ip[k];
        if (v == 0 || std::abs(v) > p.n) return {-1, -1};
        if (v > 0) { swaps += v != k + 1; ++k; continue; }
        if (k + 1 >= p.n || ip[k + 1] != v) return {-1, -1};
        ++blocks;
        k += 2;
    }
    return {swaps, blocks};
}

template <typename T, Backend B>
struct SytrfConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

template <typename Config>
class SytrfTest : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    static constexpr Backend Bk = Config::BackendVal;

    static std::vector<ops::sytrf::SytrfChoice> routes(int n) {
        std::vector<ops::sytrf::SytrfChoice> r;
        if (Bk != Backend::NETLIB) {
            if (n <= 32) r.push_back(ops::sytrf::Cta{});
            r.push_back(ops::sytrf::Wg{});
        }
        if ((Bk == Backend::CUDA && select::kHasCudaVendor) || (Bk == Backend::NETLIB && select::kHasNetlib))
            r.push_back(ops::sytrf::Vendor{});
        return r;
    }

    void factor(Sys<T>& p, const ops::sytrf::SytrfChoice& c) {
        const select::ScopedPin<ops::sytrf::SytrfChoice> pin("sytrf", c);
        factor_auto(p);
    }

    void factor_auto(Sys<T>& p) {
        auto A = a_view(p);
        const size_t need = sytrf_buffer_size<Bk, T>(*this->ctx, A, p.uplo);
        UnifiedVector<std::byte> ws(std::max<size_t>(need, 1));
        (void)sytrf<Bk, T>(*this->ctx, A, p.uplo, p.piv.to_span(), Span<std::byte>(ws.data(), need), p.info.to_span());
        this->ctx->wait();
    }

    void solve(Sys<T>& p) {
        auto A = a_view(p);
        auto Bv = b_view(p);
        (void)sytrs<Bk, T>(*this->ctx, A, Bv, p.uplo, p.piv.to_span(), Span<std::byte>{});
        this->ctx->wait();
    }

    void sysv_run(Sys<T>& p, const ops::sysv::SysvChoice* c) {
        std::optional<select::ScopedPin<ops::sysv::SysvChoice>> pin;
        if (c) pin.emplace("sysv", *c);
        auto A = a_view(p);
        auto Bv = b_view(p);
        const size_t need = sysv_buffer_size<Bk, T>(*this->ctx, A, Bv, p.uplo);
        UnifiedVector<std::byte> ws(std::max<size_t>(need, 1));
        (void)sysv<Bk, T>(*this->ctx, A, Bv, p.uplo, p.piv.to_span(), Span<std::byte>(ws.data(), need),
                          p.info.to_span());
        this->ctx->wait();
    }

    std::vector<ops::sysv::SysvChoice> sysv_routes(int n) const {
        std::vector<ops::sysv::SysvChoice> r;
        if (Bk != Backend::NETLIB && n <= 32) r.push_back(ops::sysv::Cta{});
        r.push_back(ops::sysv::Compose{});
        return r;
    }
};

using SytrfTestTypes = typename test_utils::backend_types<SytrfConfig>::type;

std::string tag(Uplo u) { return u == Uplo::Lower ? "L" : "U"; }

}  // namespace

TYPED_TEST_SUITE(SytrfTest, SytrfTestTypes);

// The test of the tests: each generator produces the pivot mix the suite relies on. random has
// interchanges and 2 x 2 blocks somewhere in the batch, dominant none of either, kkt a 2 x 2 block in
// every item. A generator that stopped forcing its case would turn every test below vacuous.
TYPED_TEST(SytrfTest, GeneratorsForceTheirPivotKinds) {
    using T = typename TestFixture::T;
    for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
        for (const auto& c : this->routes(24)) {
            auto r = make_sys<T>(24, 1, 8, uplo, Gen::random, 11u);
            this->factor(r, c);
            int swaps = 0, blocks = 0;
            for (int bi = 0; bi < r.batch; ++bi) {
                const auto [s, b] = pivot_kinds(r, bi);
                ASSERT_GE(s, 0) << "malformed ipiv, item " << bi << " " << select::to_string(c) << tag(uplo);
                swaps += s;
                blocks += b;
            }
            EXPECT_GT(swaps, 0) << select::to_string(c) << tag(uplo);
            EXPECT_GT(blocks, 0) << select::to_string(c) << tag(uplo);
            auto d = make_sys<T>(24, 1, 4, uplo, Gen::dominant, 12u);
            this->factor(d, c);
            for (int bi = 0; bi < d.batch; ++bi) EXPECT_EQ(pivot_kinds(d, bi), std::make_pair(0, 0));
            auto k = make_sys<T>(24, 1, 4, uplo, Gen::kkt, 13u);
            this->factor(k, c);
            for (int bi = 0; bi < k.batch; ++bi) EXPECT_GT(pivot_kinds(k, bi).second, 0) << select::to_string(c);
        }
}

// sytrf on every runnable route, then sytrs: the residual against the full symmetric matrix over every
// item, info = 0, and the unreferenced triangle and the pads bit-unchanged. Orders straddle the CTA
// ceiling (32 | 33) and the 256-wide work-group (256 | 257).
TYPED_TEST(SytrfTest, FactorThenSolveEveryRoute) {
    using T = typename TestFixture::T;
    for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
        for (Gen gen : {Gen::random, Gen::dominant, Gen::kkt})
            for (int n : {1, 2, 3, 5, 8, 17, 31, 32, 33, 64, 100, 257}) {
                if (gen == Gen::kkt && n < 3) continue;
                for (const auto& c : this->routes(n)) {
                    auto p = make_sys<T>(n, 3, 5, uplo, gen, 100u + unsigned(n));
                    this->factor(p, c);
                    this->solve(p);
                    const std::string what = select::to_string(c) + " " + tag(uplo) + " gen=" +
                                             std::to_string(int(gen)) + " n=" + std::to_string(n);
                    for (int bi = 0; bi < p.batch; ++bi) EXPECT_EQ(p.info[bi], 0) << what;
                    EXPECT_VERIFY(T, Check::solve, n, residual(*this->ctx, p, all_items(p))) << what;
                    EXPECT_EQ(first_touched(p), -1) << what;
                }
            }
}

// sysv on both routes, KKT systems included: the solve, ipiv's encoding and the untouched triangle.
TYPED_TEST(SytrfTest, SysvEveryRoute) {
    using T = typename TestFixture::T;
    for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
        for (Gen gen : {Gen::random, Gen::kkt})
            for (int n : {4, 9, 32, 33, 96}) {
                for (const auto& c : this->sysv_routes(n)) {
                    auto p = make_sys<T>(n, 4, 6, uplo, gen, 300u + unsigned(n));
                    this->sysv_run(p, &c);
                    const std::string what = select::to_string(c) + " " + tag(uplo) + " n=" + std::to_string(n);
                    for (int bi = 0; bi < p.batch; ++bi) {
                        EXPECT_EQ(p.info[bi], 0) << what;
                        EXPECT_GE(pivot_kinds(p, bi).first, 0) << what;
                        if (gen == Gen::kkt) EXPECT_GT(pivot_kinds(p, bi).second, 0) << what;
                    }
                    EXPECT_VERIFY(T, Check::solve, n, residual(*this->ctx, p, all_items(p))) << what;
                    EXPECT_EQ(first_touched(p), -1) << what;
                }
            }
}

// A singular item reports info > 0 and disturbs nobody: every other item's factor, ipiv, info and X is
// bit-identical to the same batch with a regular matrix in that slot. The zero matrix fails at the
// first step, which is physical column 1 (Lower) or n (Upper), as LAPACK numbers it. sysv keeps the
// singular item's B, as LAPACK ?sysv does.
TYPED_TEST(SytrfTest, SingularItemReportsInfoWithoutDisturbingTheOthers) {
    using T = typename TestFixture::T;
    const int n = 20, s = 2;
    for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
        for (const auto& c : this->sysv_routes(n)) {
            auto good = make_sys<T>(n, 2, 6, uplo, Gen::kkt, 77u);
            auto bad = make_sys<T>(n, 2, 6, uplo, Gen::kkt, 77u);
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                    if (referenced<T>(uplo, i, j)) bad.a[size_t(s) * bad.stra + size_t(j) * bad.lda + i] = T{};
            bad.a_in.assign(bad.a.begin(), bad.a.end());
            this->sysv_run(good, &c);
            this->sysv_run(bad, &c);
            const std::string what = select::to_string(c) + " " + tag(uplo);
            EXPECT_EQ(bad.info[s], uplo == Uplo::Lower ? 1 : n) << what;
            for (int bi = 0; bi < bad.batch; ++bi) {
                if (bi == s) continue;
                EXPECT_EQ(bad.info[bi], 0) << what;
                for (int e = 0; e < bad.stra; ++e)
                    ASSERT_TRUE(same(bad.a[size_t(bi) * bad.stra + e], good.a[size_t(bi) * good.stra + e])) << what;
                for (int e = 0; e < bad.strb; ++e)
                    ASSERT_TRUE(same(bad.b[size_t(bi) * bad.strb + e], good.b[size_t(bi) * good.strb + e])) << what;
                const int* gp = reinterpret_cast<const int*>(good.piv.data()) + size_t(bi) * n;
                const int* bp = reinterpret_cast<const int*>(bad.piv.data()) + size_t(bi) * n;
                for (int k = 0; k < n; ++k) ASSERT_EQ(gp[k], bp[k]) << what;
            }
            for (int e = 0; e < bad.strb; ++e)
                ASSERT_TRUE(same(bad.b[size_t(s) * bad.strb + e], bad.b_in[size_t(s) * bad.strb + e])) << what;
        }
}

// The same algorithm in local memory (cta) and in global memory (wg) does the same arithmetic in the
// same order: for n <= 32 the factor, ipiv and info agree bit for bit. sysv's fused cta then equals
// the cta factor plus sytrs (both reduce over one sub-group).
TYPED_TEST(SytrfTest, CtaAndWgAgreeBitForBit) {
    using T = typename TestFixture::T;
    if constexpr (TestFixture::Bk == Backend::NETLIB) GTEST_SKIP() << "no native route on NETLIB";
    for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
        for (int n : {7, 24, 32}) {
            auto p = make_sys<T>(n, 2, 9, uplo, Gen::random, 500u + unsigned(n));
            auto q = make_sys<T>(n, 2, 9, uplo, Gen::random, 500u + unsigned(n));
            auto f = make_sys<T>(n, 2, 9, uplo, Gen::random, 500u + unsigned(n));
            this->factor(p, ops::sytrf::Cta{});
            this->factor(q, ops::sytrf::Wg{});
            this->solve(p);
            const ops::sysv::SysvChoice cta = ops::sysv::Cta{};
            this->sysv_run(f, &cta);
            for (size_t e = 0; e < p.a.size(); ++e) {
                ASSERT_TRUE(same(p.a[e], q.a[e])) << "factor, n=" << n << " " << tag(uplo) << " at " << e;
                ASSERT_TRUE(same(p.a[e], f.a[e])) << "sysv factor, n=" << n << " " << tag(uplo) << " at " << e;
            }
            for (size_t e = 0; e < p.b.size(); ++e) ASSERT_TRUE(same(p.b[e], f.b[e])) << "X, n=" << n << " at " << e;
            for (size_t e = 0; e < p.piv.size(); ++e) {
                ASSERT_EQ(p.piv[e], q.piv[e]);
                ASSERT_EQ(p.piv[e], f.piv[e]);
            }
        }
}

// The vendor's factor (cuSOLVER ?sytrf) solved by the native sytrs: ipiv's convention and the stored
// D and L are LAPACK's on both sides, or this residual fails.
TYPED_TEST(SytrfTest, VendorFactorSolvesWithNativeSytrs) {
    using T = typename TestFixture::T;
    if constexpr (TestFixture::Bk != Backend::CUDA) GTEST_SKIP() << "the CUDA vendor arm only";
    else {
        if (!select::kHasCudaVendor) GTEST_SKIP() << "vendor-free build";
        for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
            for (int n : {6, 32, 65}) {
                auto p = make_sys<T>(n, 2, 4, uplo, Gen::kkt, 900u + unsigned(n));
                this->factor(p, ops::sytrf::Vendor{});
                this->solve(p);
                for (int bi = 0; bi < p.batch; ++bi) EXPECT_GT(pivot_kinds(p, bi).second, 0);
                EXPECT_VERIFY(T, Check::solve, n, residual(*this->ctx, p, all_items(p))) << "n=" << n << tag(uplo);
                EXPECT_EQ(first_touched(p), -1) << "vendor wrote outside the triangle, n=" << n << tag(uplo);
            }
    }
}

// A saturating batch of identical KKT matrices: every item's factor, ipiv, info and X must equal item
// 0's bit for bit, on the local-memory kernels (cta, sysv cta) and the work-group kernels (wg, sytrs).
TYPED_TEST(SytrfTest, SaturatingBatchIsBitIdenticalToItemZero) {
    using T = typename TestFixture::T;
    if constexpr (TestFixture::Bk == Backend::NETLIB) GTEST_SKIP() << "no native route on NETLIB";
    const int batch = 1024;
    for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
        for (int route = 0; route < 3; ++route) {
            const int n = route == 2 ? 70 : 32;
            auto p = make_sys<T>(n, 2, batch, uplo, Gen::kkt, 4242u, /*identical=*/true);
            if (route == 0) {
                const ops::sysv::SysvChoice cta = ops::sysv::Cta{};
                this->sysv_run(p, &cta);
            } else {
                this->factor(p, route == 1 ? ops::sytrf::SytrfChoice{ops::sytrf::Cta{}}
                                           : ops::sytrf::SytrfChoice{ops::sytrf::Wg{}});
                this->solve(p);
            }
            EXPECT_GT(pivot_kinds(p, 0).second, 0);
            EXPECT_VERIFY(T, Check::solve, n, residual(*this->ctx, p, {0, batch - 1})) << route;
            const int* ip = reinterpret_cast<const int*>(p.piv.data());
            for (int bi = 1; bi < batch; ++bi) {
                ASSERT_EQ(p.info[bi], p.info[0]) << "route " << route << " item " << bi;
                for (int e = 0; e < p.stra; ++e)
                    ASSERT_TRUE(same(p.a[size_t(bi) * p.stra + e], p.a[size_t(e)])) << "route " << route << " item " << bi;
                for (int e = 0; e < p.strb; ++e)
                    ASSERT_TRUE(same(p.b[size_t(bi) * p.strb + e], p.b[size_t(e)])) << "route " << route << " item " << bi;
                for (int k = 0; k < n; ++k) ASSERT_EQ(ip[size_t(bi) * n + k], ip[k]) << "route " << route;
            }
        }
}

// The CTA ceiling straddled both ways: a cta pin runs at 32 and throws at 33, and Auto under the
// shipped table runs cta at 32 and wg at 33 (bit-identical to the pinned run).
TYPED_TEST(SytrfTest, CtaCeilingStraddled) {
    using T = typename TestFixture::T;
    if constexpr (TestFixture::Bk == Backend::NETLIB) GTEST_SKIP() << "no native route on NETLIB";
    for (int n : {32, 33}) {
        auto pinned = make_sys<T>(n, 1, 128, Uplo::Lower, Gen::random, 31u);
        auto autop = make_sys<T>(n, 1, 128, Uplo::Lower, Gen::random, 31u);
        if (n == 33) {
            EXPECT_ANY_THROW(this->factor(pinned, ops::sytrf::Cta{}));
            pinned = make_sys<T>(n, 1, 128, Uplo::Lower, Gen::random, 31u);
        }
        this->factor(pinned, n <= 32 ? ops::sytrf::SytrfChoice{ops::sytrf::Cta{}}
                                     : ops::sytrf::SytrfChoice{ops::sytrf::Wg{}});
        this->factor_auto(autop);
        for (size_t e = 0; e < pinned.a.size(); ++e) ASSERT_TRUE(same(pinned.a[e], autop.a[e])) << "n=" << n;
    }
}

// Validation and the empty problem.
TYPED_TEST(SytrfTest, ArgumentChecks) {
    using T = typename TestFixture::T;
    UnifiedVector<T> buf(64, T{});
    UnifiedVector<int64_t> piv(64, 0);
    const MV<T> rect(buf.data(), 4, 3, 4, 12, 1);
    EXPECT_THROW(((void)sytrf<TestFixture::Bk, T>(*this->ctx, rect, Uplo::Lower, piv.to_span(), Span<std::byte>{},
                                                  Span<int32_t>{})),
                 batchlas::invalid_argument);
    const MV<T> sq(buf.data(), 4, 4, 4, 16, 1);
    const MV<T> b3(buf.data(), 3, 1, 3, 3, 1);
    EXPECT_THROW(((void)sytrs<TestFixture::Bk, T>(*this->ctx, sq, b3, Uplo::Lower, piv.to_span(), Span<std::byte>{})),
                 batchlas::invalid_argument);
    const MV<T> empty(buf.data(), 4, 4, 4, 16, 0);
    EXPECT_NO_THROW(((void)sytrf<TestFixture::Bk, T>(*this->ctx, empty, Uplo::Upper, piv.to_span(), Span<std::byte>{},
                                                     Span<int32_t>{})));
}
