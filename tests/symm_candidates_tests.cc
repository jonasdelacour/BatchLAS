// Every symm candidate, pinned: docs/design/flat-kernel-selection.md §8 and
// docs/design/flat-kernel-selection.md §12. Which kernel ran is read back from the select trace or a
// bit-for-bit comparison with the direct call, never assumed from the pin being accepted.
#include <gtest/gtest.h>

#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/functions/symm.hh>
#include <batchlas/no_route.hh>
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>
#include "../src/select/coverage.hh"
#include "../src/select/vendor.hh"

#include "test_utils.hh"
#include "shipped_table_pick.hh"

#include "../src/backends/triangular_expand.hh"
#include "../src/expansion_budget.hh"
#include "../src/ops/symm/choice.hh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

using namespace batchlas;

namespace {

namespace sy = batchlas::ops::symm;
using C = sy::SymmChoice;
using Pin = select::ScopedPin<C>;
template <typename T>
using MVof = MatrixView<T, MatrixFormat::Dense>;

template <typename T>
bool same_bits(T a, T b) {
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}

const char* side_s(Side s) { return s == Side::Left ? "L" : "R"; }
const char* uplo_s(Uplo u) { return u == Uplo::Lower ? "L" : "U"; }

struct Region {
    std::size_t off = 0;
    int rows = 0, cols = 0, ld = 0, stride = 0;
};

struct Spec {
    Side side = Side::Left;
    Uplo uplo = Uplo::Lower;
    int m = 8, n = 5, batch = 3;
    int period = 0;  // > 0: item it repeats item it % period
    unsigned seed = 1;
    int k() const { return side == Side::Left ? m : n; }
};

// A, B and C in one buffer of a large finite poison, at padded lds and strides that are not
// ld * cols. A's unreferenced triangle, the pads and the gaps hold the poison; symm may write
// only C's footprint.
template <typename T>
struct Prob {
    Spec s;
    T alpha = T(1.5), beta = T(-0.75);
    Region a, b, c;
    UnifiedVector<T> mem;
    std::vector<T> mem0;
    UnifiedVector<T*> aptr, bptr, cptr;

    MVof<T> A() { return {mem.data() + a.off, a.rows, a.cols, a.ld, a.stride, s.batch, aptr.data()}; }
    MVof<T> B() { return {mem.data() + b.off, b.rows, b.cols, b.ld, b.stride, s.batch, bptr.data()}; }
    MVof<T> Cv() { return {mem.data() + c.off, c.rows, c.cols, c.ld, c.stride, s.batch, cptr.data()}; }
    std::size_t at(const Region& r, int it, int i, int j) const {
        return r.off + std::size_t(it) * r.stride + std::size_t(j) * r.ld + i;
    }
    bool referenced(int i, int j) const { return s.uplo == Uplo::Lower ? i >= j : i <= j; }
};

template <typename T>
T poison() {
    return T(-997.0);
}

template <typename T>
Prob<T> make_prob(const Spec& s) {
    Prob<T> p;
    p.s = s;
    const int k = s.k();
    p.a = {0, k, k, k + 3, (k + 3) * k + 5};
    p.b = {std::size_t(p.a.stride) * s.batch + 11, s.m, s.n, s.m + 2, (s.m + 2) * s.n + 7};
    p.c = {p.b.off + std::size_t(p.b.stride) * s.batch + 13, s.m, s.n, s.m + 5, (s.m + 5) * s.n + 3};
    p.mem = UnifiedVector<T>(p.c.off + std::size_t(p.c.stride) * s.batch + 17, poison<T>());
    p.aptr = UnifiedVector<T*>(s.batch, nullptr);
    p.bptr = UnifiedVector<T*>(s.batch, nullptr);
    p.cptr = UnifiedVector<T*>(s.batch, nullptr);
    std::mt19937 gen(s.seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    const int reps = s.period > 0 ? std::min(s.period, s.batch) : s.batch;
    for (int it = 0; it < s.batch; ++it) {
        const int r = it % reps;
        for (int j = 0; j < k; ++j)
            for (int i = 0; i < k; ++i)
                if (p.referenced(i, j)) p.mem[p.at(p.a, it, i, j)] = it < reps ? T(u(gen)) : p.mem[p.at(p.a, r, i, j)];
        for (int j = 0; j < s.n; ++j)
            for (int i = 0; i < s.m; ++i) {
                p.mem[p.at(p.b, it, i, j)] = it < reps ? T(u(gen)) : p.mem[p.at(p.b, r, i, j)];
                p.mem[p.at(p.c, it, i, j)] = it < reps ? T(u(gen)) : p.mem[p.at(p.c, r, i, j)];
            }
    }
    p.mem0.assign(p.mem.begin(), p.mem.end());
    return p;
}

template <typename T>
double tol(int k) {
    return 64.0 * std::max(k, 1) * double(std::numeric_limits<T>::epsilon());
}

// Each checked item against a double host reference of the symmetric product (A mirrored from
// its referenced triangle), every element outside C bit for bit, and for a repeating batch
// every item bit-identical to its representative.
template <typename T>
void expect_symm(const Prob<T>& p, const std::string& what) {
    const Spec& s = p.s;
    const int k = s.k();
    std::vector<int> items;
    if (s.period > 0) {
        for (int it = 0; it < std::min(s.period, s.batch); ++it) items.push_back(it);
    } else if (s.batch <= 8) {
        for (int it = 0; it < s.batch; ++it) items.push_back(it);
    } else {
        items = {0, 1, s.batch / 2, s.batch - 1};
    }
    auto asym = [&](int it, int i, int j) {
        return double(p.referenced(i, j) ? p.mem0[p.at(p.a, it, i, j)] : p.mem0[p.at(p.a, it, j, i)]);
    };
    for (int it : items)
        for (int j = 0; j < s.n; ++j)
            for (int i = 0; i < s.m; ++i) {
                double acc = 0, mag = 0;
                for (int t = 0; t < k; ++t) {
                    const double x = s.side == Side::Left ? asym(it, i, t) * double(p.mem0[p.at(p.b, it, t, j)])
                                                          : double(p.mem0[p.at(p.b, it, i, t)]) * asym(it, t, j);
                    acc += x;
                    mag += std::abs(x);
                }
                const double c0 = double(p.mem0[p.at(p.c, it, i, j)]);
                const double want = double(p.alpha) * acc + double(p.beta) * c0;
                const double bound = tol<T>(k) * (std::abs(double(p.alpha)) * mag + std::abs(double(p.beta) * c0)) + 1e-30;
                const double got = double(p.mem[p.at(p.c, it, i, j)]);
                ASSERT_TRUE(std::isfinite(got) && std::abs(got - want) <= bound)
                    << what << " item " << it << " (" << i << "," << j << "): got " << got << " want " << want;
            }
    std::vector<char> inc(p.mem.size(), 0);
    for (int it = 0; it < s.batch; ++it)
        for (int j = 0; j < s.n; ++j)
            for (int i = 0; i < s.m; ++i) inc[p.at(p.c, it, i, j)] = 1;
    for (std::size_t e = 0; e < p.mem.size(); ++e)
        if (!inc[e]) ASSERT_TRUE(same_bits(p.mem[e], p.mem0[e])) << what << ": wrote outside C at element " << e;
    if (s.period > 0)
        for (int it = s.period; it < s.batch; ++it)
            for (int j = 0; j < s.n; ++j)
                for (int i = 0; i < s.m; ++i)
                    ASSERT_TRUE(same_bits(p.mem[p.at(p.c, it, i, j)], p.mem[p.at(p.c, it % s.period, i, j)]))
                        << what << ": item " << it << " differs from its representative at (" << i << "," << j << ")";
}

std::string label(const Spec& s) {
    return std::string(side_s(s.side)) + uplo_s(s.uplo) + " m=" + std::to_string(s.m) + " n=" + std::to_string(s.n) +
           " batch=" + std::to_string(s.batch);
}
std::string name(const C& c, const Spec& s) { return select::to_string(c) + " " + label(s); }

// The outermost symm trace line for whatever `run` calls (nested gemm lines are indented).
template <class F>
std::string traced_line(F&& run, std::string* all = nullptr) {
    const ScopedEnvVar trace("BATCHLAS_SELECT_TRACE", "1");
    ::testing::internal::CaptureStderr();
    try {
        run();
    } catch (...) {
        (void)::testing::internal::GetCapturedStderr();
        throw;
    }
    const std::string err = ::testing::internal::GetCapturedStderr();
    if (all) *all = err;
    std::istringstream in(err);
    for (std::string line; std::getline(in, line);)
        if (line.rfind("symm ", 0) == 0 && line.find(" -> ") != std::string::npos) return line;
    return "<no symm trace line in: " + err + ">";
}
template <class F>
std::string traced_choice(F&& run, std::string* all = nullptr) {
    const std::string line = traced_line(std::forward<F>(run), all);
    const auto arrow = line.find(" -> ");
    if (arrow == std::string::npos) return line;
    const std::string tail = line.substr(arrow + 4);
    return tail.substr(0, tail.find(' '));
}

struct TableGuard {
    ~TableGuard() { select::testing::use_embedded_tables(); }
};

// The old router (ff340fc6 symm_custom_dispatch.cc + triangular_expand.hh), restated: float,
// squareish C, and batch >= 4 or max(m, n) >= 256 took the expansion; everything else the vendor.
template <typename T>
std::string old_auto(int m, int n, int batch) {
    const bool sq = 2 * std::min(m, n) >= std::max(m, n);
    return std::is_same_v<T, float> && sq && (batch >= 4 || std::max(m, n) >= 256) ? "expand" : "vendor";
}

// A heterogeneous operand has no route on any backend: expand cannot honour per-item extents (it
// reads A at its storage order; the gemm it feeds rejects a heterogeneous B or C), and
// every vendor loop (cuBLAS, rocBLAS, netlib) runs each item at the top-level extents. A
// spelling pin throws invalid_argument (the vendor class word falls back to Auto), Auto throws
// runtime_error, or NoRouteError without a vendor library, and C is left untouched.
// On ff340fc6 the cuBLAS loop answered the consistent all-three case off by 2.56.
template <Backend B, class T>
void expect_heterogeneous_has_no_route(Queue& ctx) {
    const int n = 16, batch = 4;
    auto A = Matrix<T, MatrixFormat::Dense>::Random(n, n, false, batch, 3);
    auto Bm = Matrix<T, MatrixFormat::Dense>::Random(n, n, false, batch, 5);
    Matrix<T, MatrixFormat::Dense> Cm(n, n, batch);
    Cm.fill(T(0.25));
    UnifiedVector<int> act(batch), cols(batch);
    for (int b = 0; b < batch; ++b) act[b] = n - 2 * b, cols[b] = n - b;
    const auto hA = A.view().with_active_dims(act.to_span(), act.to_span());
    const auto hB = Bm.view().with_active_dims(act.to_span(), cols.to_span());
    const auto hC = Cm.view().with_active_dims(act.to_span(), cols.to_span());
    ASSERT_TRUE(hA.is_heterogeneous() && hB.is_heterogeneous() && hC.is_heterogeneous());
    struct Case { const char* what; MVof<T> a, b, c; };
    const Case cases[] = {{"A heterogeneous", hA, Bm.view(), Cm.view()},
                          {"B and C heterogeneous", A.view(), hB, hC},
                          {"all three, consistent per item", hA, hB, hC}};
    for (const auto& k : cases) {
        auto call = [&] {
            (void)symm<B, T>(ctx, k.a, k.b, k.c, T(1), T(0.5), Side::Left, Uplo::Lower);
            ctx.wait();
        };
        auto expect_no_route = [&](const std::string& what) {
            if constexpr (select::level3_vendor_available<B>) EXPECT_THROW(call(), std::runtime_error) << what;
            else EXPECT_THROW(call(), batchlas::NoRouteError) << what;
        };
        for (const C& c : sy::candidates<T>()) {
            const Pin pin("symm", c);
            if (std::holds_alternative<sy::Vendor>(c)) expect_no_route(std::string(k.what) + ": vendor pin");
            else EXPECT_THROW(call(), std::invalid_argument) << k.what << ": " << select::to_string(c);
        }
        const ScopedEnvVar clear("BATCHLAS_SYMM_ROUTE", nullptr);
        expect_no_route(std::string(k.what) + ": auto");
    }
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) ASSERT_EQ(Cm(i, j, b), T(0.25)) << "a refused call wrote C";
}

template <typename T, Backend B>
struct Cfg {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};
using Types = typename test_utils::backend_types_filtered<Cfg, false>::type;

template <typename Config>
class SymmCandidates : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    using MV = MatrixView<T, MatrixFormat::Dense>;
    static constexpr Backend B = Config::BackendVal;
    static constexpr bool kVendor = batchlas::select::level3_vendor_available<B>;

    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        if (this->HasFatalFailure() || ::testing::Test::IsSkipped()) return;
        if (!this->ctx) GTEST_SKIP() << "no queue";
        if (this->ctx->device().type != DeviceType::GPU) GTEST_SKIP() << "expand is a GPU kernel";
    }

    // The Vendor choice spells the class word `vendor`, which falls back to Auto where no vendor
    // can run instead of throwing (§5.3).
    static bool vendor_word_falls_back(const C& c) { return std::holds_alternative<sy::Vendor>(c) && !kVendor; }
    static bool expect_runs(const C& c) { return std::holds_alternative<sy::Expand>(c) || kVendor; }

    void run(Prob<T>& p) {
        (void)symm<B, T>(*this->ctx, p.A(), p.B(), p.Cv(), p.alpha, p.beta, p.s.side, p.s.uplo);
        this->ctx->wait();
    }
    void run_pinned(const C& c, Prob<T>& p) {
        const Pin pin("symm", c);
        run(p);
    }
    bool pin_accepted(const C& c, Prob<T>& p) {
        const Pin pin("symm", c);
        try {
            run(p);
            return true;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find("cannot run this shape") == std::string::npos) throw;
            return false;
        }
    }
    std::string auto_choice(Prob<T>& p) {
        const ScopedEnvVar clear("BATCHLAS_SYMM_ROUTE", nullptr);
        return traced_choice([&] { run(p); });
    }

    // The family's own code: Expand = expand_mirrored into an arena lease + the public gemm, with
    // the expanded ld the launcher derives; Vendor = the library loop.
    bool direct(const C& c, const MV& A, const MV& Bm, const MV& Cm, T alpha, T beta, Side side, Uplo uplo,
                std::string* why = nullptr) {
        Queue& q = *this->ctx;
        try {
            if (std::holds_alternative<sy::Expand>(c)) {
                const int k = A.rows(), batch = A.batch_size(), ld = backend::detail::expanded_ld<T>(k);
                UnifiedVector<T> scratch(std::size_t(ld) * k * std::max(batch, 1), T(0));
                const MV full(scratch.data(), k, k, ld, ld * k, batch);
                backend::detail::expand_mirrored<T, false>(q, full, A, uplo).wait();
                if (side == Side::Left) (void)gemm<B, T>(q, full, Bm, Cm, alpha, beta, Transpose::NoTrans, Transpose::NoTrans, ComputePrecision::Default);
                else (void)gemm<B, T>(q, Bm, full, Cm, alpha, beta, Transpose::NoTrans, Transpose::NoTrans, ComputePrecision::Default);
            } else {
                if constexpr (kVendor) {
                    (void)backend::symm_vendor<B, T>(q, A, Bm, Cm, alpha, beta, side, uplo);
                } else {
                    if (why) *why = "no vendor BLAS in this build";
                    return false;
                }
            }
            q.wait();
            return true;
        } catch (const std::exception& e) {
            if (why) *why = e.what();
            return false;
        }
    }
    bool direct(const C& c, Prob<T>& p, std::string* why = nullptr) {
        return direct(c, p.A(), p.B(), p.Cv(), p.alpha, p.beta, p.s.side, p.s.uplo, why);
    }
};

TYPED_TEST_SUITE(SymmCandidates, Types);

// §8.1: each candidate on extents straddling the expansion's 32-wide mirror tile (31/32/33, a
// ragged 77) and the old 255|256 threshold, both sides, both triangles, batch 1 and 5. Then the
// scratch budget straddled: BATCHLAS_EXPAND_MAX_BYTES at exactly the lease, and one byte short.
TYPED_TEST(SymmCandidates, PinnedCandidatesStraddleTheirLimits) {
    using T = typename TestFixture::T;
    int ran = 0;
    for (const C& c : sy::candidates<T>())
        for (Side side : {Side::Left, Side::Right})
            for (auto [m, n] : {std::pair{1, 1}, std::pair{31, 3}, std::pair{32, 32}, std::pair{33, 70},
                                std::pair{77, 1}, std::pair{255, 9}, std::pair{4, 256}})
                for (int batch : {1, 5}) {
                    Spec s{side, (m + n) % 2 ? Uplo::Lower : Uplo::Upper, m, n, batch};
                    s.seed = 100u + 7u * m + n;
                    auto p = make_prob<T>(s);
                    const std::string what = name(c, s);
                    if (TestFixture::vendor_word_falls_back(c)) {
                        auto q = make_prob<T>(s);
                        const std::string want = this->auto_choice(q);
                        EXPECT_EQ(traced_choice([&] { this->run_pinned(c, p); }), want) << what;
                        expect_symm(p, what + " (vendor-free: Auto)");
                        continue;
                    }
                    ASSERT_TRUE(this->pin_accepted(c, p)) << what << " was refused";
                    expect_symm(p, what);
                    ++ran;
                }
    EXPECT_GT(ran, 0);
    for (auto [k, batch] : {std::pair{33, 3}, std::pair{64, 2}}) {
        Spec s{Side::Left, Uplo::Lower, k, 5, batch};
        const std::size_t bytes = backend::detail::expanded_workspace_bytes<T>(*this->ctx, k, batch);
        for (std::size_t cap : {bytes, bytes - 1}) {
            const ScopedEnvVar budget("BATCHLAS_EXPAND_MAX_BYTES", std::to_string(cap).c_str());
            auto p = make_prob<T>(s);
            const bool ok = this->pin_accepted(C{sy::Expand{}}, p);
            EXPECT_EQ(ok, cap == bytes) << label(s) << " budget " << cap << " of " << bytes;
            if (ok) expect_symm(p, label(s) + " at the exact budget");
        }
    }
}

// Every (side, uplo) at three shapes, rows != cols, alpha != 1, beta != 0.
TYPED_TEST(SymmCandidates, EveryCombinationOnEveryCandidate) {
    using T = typename TestFixture::T;
    for (const C& c : sy::candidates<T>()) {
        if (!TestFixture::expect_runs(c)) continue;
        for (auto [m, n] : {std::pair{13, 7}, std::pair{40, 33}, std::pair{70, 129}})
            for (Side side : {Side::Left, Side::Right})
                for (Uplo uplo : {Uplo::Lower, Uplo::Upper}) {
                    Spec s{side, uplo, m, n, 3};
                    s.seed = 77u + m;
                    auto p = make_prob<T>(s);
                    this->run_pinned(c, p);
                    expect_symm(p, name(c, s));
                    if (::testing::Test::HasFatalFailure()) return;
                }
    }
}

// The pinned facade runs exactly that family's code: bit-identical to the direct call.
TYPED_TEST(SymmCandidates, PinnedRunIsTheDirectKernelBitForBit) {
    using T = typename TestFixture::T;
    int compared = 0;
    for (const C& c : sy::candidates<T>()) {
        if (!TestFixture::expect_runs(c)) continue;
        for (Side side : {Side::Left, Side::Right})
            for (auto [m, n] : {std::pair{13, 7}, std::pair{33, 40}, std::pair{96, 5}}) {
                Spec s{side, Uplo::Upper, m, n, 3};
                s.seed = 4242u + m;
                auto pinned = make_prob<T>(s);
                auto direct = make_prob<T>(s);
                const std::string what = name(c, s);
                this->run_pinned(c, pinned);
                std::string why;
                ASSERT_TRUE(this->direct(c, direct, &why)) << what << ": " << why;
                expect_symm(pinned, what);
                for (std::size_t e = 0; e < pinned.mem.size(); ++e)
                    ASSERT_TRUE(same_bits(pinned.mem[e], direct.mem[e]))
                        << what << ": the pinned facade did not run this family's code; element " << e;
                ++compared;
            }
    }
    EXPECT_GT(compared, 0);
}

// expand_mirrored stages 32x32 tiles in local memory: 1024 items repeating 7 distinct problems,
// each representative correct and every other item bit-identical to its representative.
TYPED_TEST(SymmCandidates, SaturatingBatchIsBitIdenticalToItsRepresentative) {
    using T = typename TestFixture::T;
    for (const C& c : sy::candidates<T>()) {
        if (!TestFixture::expect_runs(c)) continue;
        for (auto [side, m, n] : {std::tuple{Side::Left, 40, 9}, std::tuple{Side::Right, 6, 70}}) {
            Spec s{side, Uplo::Lower, m, n, 1024};
            s.period = 7;
            s.seed = 777u + m;
            auto p = make_prob<T>(s);
            this->run_pinned(c, p);
            expect_symm(p, name(c, s));
            if (::testing::Test::HasFatalFailure()) return;
        }
    }
}

// §8.2 (R3): a pin is accepted exactly when the family's own code launches, over shapes, the
// grid-z batch ceiling and a heterogeneous B/C (the gemm under expand rejects it).
TYPED_TEST(SymmCandidates, CanRunEqualsLaunch) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    int disagreements = 0;
    for (const C& c : sy::candidates<T>()) {
        if (TestFixture::vendor_word_falls_back(c)) continue;
        for (Side side : {Side::Left, Side::Right})
            for (auto [m, n] : {std::pair{1, 5}, std::pair{33, 2}, std::pair{64, 64}}) {
                Spec s{side, Uplo::Lower, m, n, 2};
                auto a = make_prob<T>(s);
                auto b = make_prob<T>(s);
                const bool pin = this->pin_accepted(c, a);
                std::string why;
                const bool run = this->direct(c, b, &why);
                EXPECT_EQ(pin, run) << name(c, s) << ": can_run says " << pin << ", the code " << (run ? "launches" : "refuses: " + why);
                disagreements += pin != run;
            }
    }
    const int n = 4, batch = 3;
    Matrix<T, MatrixFormat::Dense> A(n, n, batch), Bm(n, n, batch), Cm(n, n, batch);
    A.fill(T(1)), Bm.fill(T(1)), Cm.fill(T(0));
    UnifiedVector<int> act(batch);
    for (int i = 0; i < batch; ++i) act[i] = n - i;
    const auto hB = Bm.view().with_active_dims(act.to_span(), act.to_span());
    const auto hC = Cm.view().with_active_dims(act.to_span(), act.to_span());
    std::string why;
    EXPECT_FALSE(this->direct(C{sy::Expand{}}, A.view(), hB, hC, T(1), T(0), Side::Left, Uplo::Lower, &why))
        << "expand ran a heterogeneous B/C";
    const Pin pin("symm", C{sy::Expand{}});
    EXPECT_THROW(((void)symm<B, T>(*this->ctx, A.view(), hB, hC, T(1), T(0), Side::Left, Uplo::Lower)), std::invalid_argument);
    EXPECT_EQ(disagreements, 0);
}

// expand_mirrored and its gemm put the batch in grid z: 65535 items launch, 65536 do not, so
// the pin refuses 65536 and Auto takes the vendor (vendor-free: no route).
TYPED_TEST(SymmCandidates, GridBatchCeiling) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    const int ceiling = int(sy::kMaxGridBatch);
    for (int batch : {ceiling, ceiling + 1}) {
        Matrix<T, MatrixFormat::Dense> A(1, 1, batch), Bm(1, 2, batch), Cm(1, 2, batch);
        A.fill(T(2)), Bm.fill(T(3)), Cm.fill(T(1));
        auto call = [&] {
            (void)symm<B, T>(*this->ctx, A.view(), Bm.view(), Cm.view(), T(1), T(1), Side::Left, Uplo::Lower);
            this->ctx->wait();
        };
        std::string why;
        Matrix<T, MatrixFormat::Dense> Cd(1, 2, batch);
        Cd.fill(T(1));
        const bool launches = this->direct(C{sy::Expand{}}, A.view(), Bm.view(), Cd.view(), T(1), T(1), Side::Left, Uplo::Lower, &why);
        EXPECT_EQ(launches, batch <= ceiling) << "direct expand at batch " << batch << ": " << why;
        {
            const Pin pin("symm", C{sy::Expand{}});
            if (batch <= ceiling) {
                call();
                for (int it : {0, batch / 2, batch - 1}) EXPECT_EQ(double(Cm(0, 1, it)), 7.0) << "item " << it;
                continue;
            }
            try {
                call();
                ADD_FAILURE() << "expand accepted batch " << batch;
            } catch (const std::invalid_argument& e) {
                EXPECT_NE(std::string(e.what()).find("cannot run this shape"), std::string::npos) << e.what();
            }
        }
        const ScopedEnvVar clear("BATCHLAS_SYMM_ROUTE", nullptr);
        if constexpr (TestFixture::kVendor) {
            EXPECT_EQ(traced_choice(call), "vendor");
            EXPECT_EQ(double(Cm(0, 1, batch - 1)), 7.0);
        } else {
            EXPECT_THROW(call(), batchlas::NoRouteError);
        }
    }
}

// Shape and batch errors throw invalid_argument before choose(), under Auto and every pin, and
// write nothing.
TYPED_TEST(SymmCandidates, InvalidShapesThrowBeforeChoose) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    auto p = make_prob<T>(Spec{Side::Left, Uplo::Lower, 8, 5, 3});
    std::vector<std::optional<C>> pins{std::nullopt};
    for (const C& c : sy::candidates<T>()) pins.push_back(c);
    struct Bad { const char* what; MVof<T> a, b, c; Side side; };
    const Bad bads[] = {
        {"A not square", MVof<T>(p.mem.data() + p.a.off, 8, 7, p.a.ld, p.a.stride, 3), p.B(), p.Cv(), Side::Left},
        {"order is n on the right", p.A(), p.B(), p.Cv(), Side::Right},
        {"batch mismatch", MVof<T>(p.mem.data() + p.a.off, 8, 8, p.a.ld, p.a.stride, 2), p.B(), p.Cv(), Side::Left}};
    for (const auto& bad : bads)
        for (const auto& c : pins) {
            std::optional<Pin> pin;
            if (c) pin.emplace("symm", *c);
            EXPECT_THROW(((void)symm<B, T>(*this->ctx, bad.a, bad.b, bad.c, T(1), T(1), bad.side, Uplo::Lower)),
                         std::invalid_argument)
                << bad.what << " under " << (c ? select::to_string(*c) : std::string("auto"));
        }
    this->ctx->wait();
    for (std::size_t e = 0; e < p.mem.size(); ++e)
        ASSERT_TRUE(same_bits(p.mem[e], p.mem0[e])) << "a refused call wrote element " << e;
}

// R6: a spelling that names nothing compiled throws instead of meaning Auto: the removed
// level-3 words (cublasdx included), other ops' families, fields on a fieldless family.
TYPED_TEST(SymmCandidates, UnknownPinsThrow) {
    using T = typename TestFixture::T;
    for (const char* word : {"cublasdx", "tiles", "narrow", "gemm", "custom", "dx", "fused", "diag_full_gemm",
                             "triangular_tiles", "gram_tiles", "expand_gemm", "fused_device", "register_tiled",
                             "native:auto", "vendor:auto", "bogus", "triangular", "gram", "loop", "native:expand",
                             "expand:1", "expand:tile=32", "cta", "blocked", "direct", "tiled"}) {
        auto p = make_prob<T>(Spec{});
        const Pin pin("symm", std::string_view(word));
        EXPECT_THROW(this->run(p), std::invalid_argument) << word;
    }
}

// The named can_run-false cases for expand, each with its message.
TYPED_TEST(SymmCandidates, CanRunFalsePinsThrow) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    const int n = 6, batch = 3;
    Matrix<T, MatrixFormat::Dense> A(n, n, batch), Bm(n, n, batch), Cm(n, n, batch);
    A.fill(T(1)), Bm.fill(T(1)), Cm.fill(T(0));
    UnifiedVector<int> act(batch);
    for (int i = 0; i < batch; ++i) act[i] = n - i;
    const auto hA = A.view().with_active_dims(act.to_span(), act.to_span());
    const auto hB = Bm.view().with_active_dims(act.to_span(), act.to_span());
    const auto hC = Cm.view().with_active_dims(act.to_span(), act.to_span());
    struct Case { const char* what; MVof<T> a, b, c; const char* budget; };
    const Case cases[] = {{"A heterogeneous", hA, Bm.view(), Cm.view(), nullptr},
                          {"B heterogeneous", A.view(), hB, Cm.view(), nullptr},
                          {"C heterogeneous", A.view(), Bm.view(), hC, nullptr},
                          {"scratch over BATCHLAS_EXPAND_MAX_BYTES", A.view(), Bm.view(), Cm.view(), "0"}};
    for (const auto& k : cases) {
        const ScopedEnvVar budget("BATCHLAS_EXPAND_MAX_BYTES", k.budget);
        const Pin pin("symm", C{sy::Expand{}});
        try {
            (void)symm<B, T>(*this->ctx, k.a, k.b, k.c, T(1), T(0), Side::Left, Uplo::Lower);
            this->ctx->wait();
            ADD_FAILURE() << k.what << " was accepted";
        } catch (const std::invalid_argument& e) {
            EXPECT_NE(std::string(e.what()).find("cannot run this shape"), std::string::npos) << k.what << ": " << e.what();
        }
    }
}

// A heterogeneous operand on CUDA (expect_heterogeneous_has_no_route), plus the reason expand
// carries the term.
TYPED_TEST(SymmCandidates, HeterogeneousBatchHasNoRoute) {
    using T = typename TestFixture::T;
    expect_heterogeneous_has_no_route<TestFixture::B, T>(*this->ctx);
    const int n = 16, batch = 4;
    auto A = Matrix<T, MatrixFormat::Dense>::Random(n, n, false, batch, 3);
    auto Bm = Matrix<T, MatrixFormat::Dense>::Random(n, n, false, batch, 5);
    UnifiedVector<int> act(batch);
    for (int b = 0; b < batch; ++b) act[b] = n - 2 * b;
    const auto hA = A.view().with_active_dims(act.to_span(), act.to_span());
    // The A-only case is not a throw but a wrong answer: the expansion mirrors the full storage
    // order, so item 1 (active order 14) picks up rows 14..15 of B.
    Matrix<T, MatrixFormat::Dense> Cd(n, n, batch);
    Cd.fill(T(0));
    ASSERT_TRUE(this->direct(C{sy::Expand{}}, hA, Bm.view(), Cd.view(), T(1), T(0), Side::Left, Uplo::Lower));
    double off = 0;
    for (int i = 0; i < act[1]; ++i) {
        double want = 0;
        for (int t = 0; t < act[1]; ++t) want += double(i >= t ? A(i, t, 1) : A(t, i, 1)) * double(Bm(t, 0, 1));
        off = std::max(off, std::abs(double(Cd(i, 0, 1)) - want));
    }
    EXPECT_GT(off, 1e-3) << "expand honoured A's active order; the heterogeneity term may be revisited";
}

// An empty batch or an empty C is a no-op under Auto and every pin, vendor-free too (the cuBLAS
// loop threw CUBLAS error 7 on a batch of 0).
TYPED_TEST(SymmCandidates, EmptyProblemIsANoOp) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    UnifiedVector<T> a(256, T(1)), b(256, T(3)), c(256, T(2));
    struct Case { int m, n, batch; Side side; };
    const Case cases[] = {{8, 6, 0, Side::Left}, {8, 6, 0, Side::Right}, {8, 0, 2, Side::Left}, {0, 6, 2, Side::Left},
                          {8, 0, 2, Side::Right}};
    std::vector<std::optional<C>> pins{std::nullopt};
    for (const C& k : sy::candidates<T>()) pins.push_back(k);
    const ScopedEnvVar clear("BATCHLAS_SYMM_ROUTE", nullptr);
    for (const auto& k : cases) {
        const int ka = k.side == Side::Left ? k.m : k.n;
        const MVof<T> A(a.data(), ka, ka, std::max(ka, 1), 64, k.batch);
        const MVof<T> Bm(b.data(), k.m, k.n, std::max(k.m, 1), 64, k.batch);
        const MVof<T> Cm(c.data(), k.m, k.n, std::max(k.m, 1), 64, k.batch);
        for (const auto& p : pins) {
            std::optional<Pin> pin;
            if (p) pin.emplace("symm", *p);
            EXPECT_NO_THROW(((void)symm<B, T>(*this->ctx, A, Bm, Cm, T(1.5), T(-0.75), k.side, Uplo::Lower),
                             this->ctx->wait()))
                << (p ? select::to_string(*p) : std::string("auto")) << " m=" << k.m << " n=" << k.n
                << " batch=" << k.batch << " " << side_s(k.side);
        }
    }
    for (std::size_t e = 0; e < c.size(); ++e) ASSERT_EQ(c[e], T(2)) << "an empty problem wrote element " << e;
}

// §5.3: spellings (case-folded) and the class words, via ScopedPin and via the environment. Auto
// is the shipped table's pick (shipped_table_pick.hh).
TYPED_TEST(SymmCandidates, ClassWordsAndSpellings) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_SYMM_ROUTE", nullptr);
    const Spec s{Side::Right, Uplo::Upper, 20, 24, 2};
    std::string auto_pick;
    {
        auto p = make_prob<T>(s);
        auto_pick = traced_choice([&] { this->run(p); });
    }
    const select::Key key{{"form", sy::form_of(s.m, s.n)}, {"m", s.m}, {"n", s.n}, {"batch", s.batch}};
    EXPECT_EQ(auto_pick, test_utils::shipped_table_pick<C>("symm", select::dtype_name<T>(),
                                                           select::device_of<TestFixture::B>(*this->ctx), key, [&] {
                                                               auto q = make_prob<T>(s);
                                                               this->run(q);
                                                           }));
    const std::string vendor_pick = TestFixture::kVendor ? "vendor" : auto_pick;
    const std::pair<const char*, std::string> expect[] = {
        {"EXPAND", "expand"}, {" expand ", "expand"}, {"native", "expand"}, {"vendor", vendor_pick}, {"Auto", auto_pick}};
    for (const auto& [word, spelling] : expect)
        for (bool via_env : {false, true}) {
            auto p = make_prob<T>(s);
            select::testing::reset_warnings();
            std::string err;
            const std::string got = traced_choice([&] {
                const ScopedEnvVar env("BATCHLAS_SYMM_ROUTE", via_env ? word : nullptr);
                std::optional<Pin> pin;
                if (!via_env) pin.emplace("symm", std::string_view(word));
                this->run(p);
            }, &err);
            const std::string what = std::string("'") + word + (via_env ? "' via BATCHLAS_SYMM_ROUTE" : "' via ScopedPin");
            EXPECT_EQ(got, spelling) << what;
            const bool warns = std::string(word) == "vendor" && !TestFixture::kVendor;
            EXPECT_EQ(err.find("symm pinned \"vendor\", but no vendor candidate") != std::string::npos, warns) << what << ": " << err;
            expect_symm(p, what);
        }
}

// §5.3: a ScopedPin wins over BATCHLAS_SYMM_ROUTE, and nested pins restore the outer one.
TYPED_TEST(SymmCandidates, ScopedPinBeatsTheEnvironment) {
    using T = typename TestFixture::T;
    const Spec s{Side::Left, Uplo::Lower, 12, 7, 2};
    const ScopedEnvVar env("BATCHLAS_SYMM_ROUTE", "bogus");
    {
        auto p = make_prob<T>(s);
        EXPECT_THROW(this->run(p), std::invalid_argument) << "the environment pin was not read";
    }
    auto p = make_prob<T>(s);
    EXPECT_EQ(traced_choice([&] {
                  const Pin pin("symm", C{sy::Expand{}});
                  this->run(p);
              }),
              "expand");
    expect_symm(p, "expand over env bogus");
    const Pin outer("symm", std::string_view("bogus"));
    {
        const Pin inner("symm", C{sy::Expand{}});
        auto r = make_prob<T>(s);
        EXPECT_EQ(traced_choice([&] { this->run(r); }), "expand");
    }
    auto r = make_prob<T>(s);
    EXPECT_THROW(this->run(r), std::invalid_argument) << "the inner pin did not restore the outer";
}

// Auto against the shipped table of this device (shipped_table_pick.hh), at cells on both sides of
// every threshold the old router read (batch 3|4, max(m, n) 255|256, the squareish form), on and
// off the grid. On a transcribed table that is the old rule (SymmTranscribedTable checks the rows).
TYPED_TEST(SymmCandidates, AutoReadsTheShippedTable) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    const ScopedEnvVar clear("BATCHLAS_SYMM_ROUTE", nullptr);
    struct Cell { int m, n, batch; };
    const Cell cells[] = {{64, 64, 3},   {64, 64, 4},   {255, 255, 1}, {256, 256, 1}, {128, 255, 3}, {129, 256, 1},
                          {256, 100, 8}, {100, 256, 8}, {300, 200, 2}, {200, 150, 5}, {200, 150, 3}, {40, 19, 1000},
                          {19, 40, 1000}, {5, 9, 4},    {9, 5, 3},     {1, 1, 4},     {600, 290, 1}, {290, 600, 2}};
    for (const Cell& c : cells)
        for (Side side : {Side::Left, Side::Right}) {
            Spec s{side, Uplo::Lower, c.m, c.n, c.batch};
            s.seed = 41u + c.m;
            const select::Key key{{"form", sy::form_of(c.m, c.n)}, {"m", c.m}, {"n", c.n}, {"batch", c.batch}};
            const std::string want = test_utils::shipped_table_pick<C>("symm", select::dtype_name<T>(),
                                                                       select::device_of<B>(*this->ctx), key, [&] {
                                                                           auto q = make_prob<T>(s);
                                                                           this->run(q);
                                                                       });
            ASSERT_NE(want, test_utils::kNoTableEntryRuns) << label(s);
            if (!TestFixture::kVendor) EXPECT_EQ(want, "expand") << label(s);
            auto p = make_prob<T>(s);
            EXPECT_EQ(traced_choice([&] { this->run(p); }), want) << label(s);
            expect_symm(p, "auto " + label(s));
        }
}

// key_of's every field reaches choose(): a synthetic table whose winner changes with batch,
// m (and not n: a swapped m/n turns two probes red), and the tall and wide forms.
TYPED_TEST(SymmCandidates, AutoReadsEveryKeyField) {
    using T = typename TestFixture::T;
    if (!TestFixture::kVendor) GTEST_SKIP() << "only expand can run vendor-free, so no row can steer Auto";
    const ScopedEnvVar clear("BATCHLAS_SYMM_ROUTE", nullptr);
    const std::string dtype(select::dtype_name<T>());
    const std::string dev = select::device_of<TestFixture::B>(*this->ctx).key;
    std::vector<std::pair<std::string, std::string>> files;
    for (const auto& t : select::embedded_tables())
        if (t.name.rfind("symm.", 0) != 0) files.emplace_back(std::string(t.name), std::string(t.text));
    files.emplace_back("symm." + dtype + "." + dev + ".txt",
                       "# op=symm dtype=" + dtype + " device=" + dev + " kernels=unknown\n"
                       "# keys: form:exact m:log n:log batch:log\n"
                       "form=sq m=64 n=64 batch=128 | vendor 1 | expand 2\n"
                       "form=sq m=64 n=64 batch=32768 | expand 1 | vendor 2\n"
                       "form=sq m=64 n=32 batch=128 | expand 1 | vendor 2\n"
                       "form=tall m=160 n=64 batch=128 | expand 1 | vendor 2\n"
                       "form=tall m=64 n=160 batch=128 | vendor 1 | expand 2\n"
                       "form=wide m=64 n=160 batch=128 | expand 1 | vendor 2\n"
                       "form=wide m=160 n=64 batch=128 | vendor 1 | expand 2\n");
    const TableGuard restore;
    select::testing::set_builtin_tables(std::move(files));
    struct Probe { int m, n, batch; const char* expect; const char* field; };
    const Probe probes[] = {{64, 64, 128, "vendor", "base"},          {64, 64, 32768, "expand", "batch"},
                            {64, 32, 128, "expand", "m"},             {32, 64, 128, "vendor", "n (m and n swapped)"},
                            {160, 64, 128, "expand", "form (tall)"}, {64, 160, 128, "expand", "form (wide)"}};
    for (const auto& k : probes)
        for (Side side : {Side::Left, Side::Right}) {
            Spec s{side, Uplo::Upper, k.m, k.n, k.batch};
            s.seed = 43u;
            auto p = make_prob<T>(s);
            EXPECT_EQ(traced_choice([&] { this->run(p); }), k.expect) << "the " << k.field << " row, " << label(s);
            expect_symm(p, std::string("the ") + k.field + " row");
        }
}

// The trace key: form and C's extents (not A's order), whatever the side.
TYPED_TEST(SymmCandidates, TraceKeyIsCsShape) {
    using T = typename TestFixture::T;
    for (auto [side, m, n] : {std::tuple{Side::Left, 37, 12}, std::tuple{Side::Right, 12, 37}, std::tuple{Side::Left, 9, 8}}) {
        Spec s{side, Uplo::Upper, m, n, 5};
        auto p = make_prob<T>(s);
        const std::string line = traced_line([&] { this->run(p); });
        const std::string form = 2 * std::min(m, n) >= std::max(m, n) ? "sq" : (m > n ? "tall" : "wide");
        const std::string want = "form=" + form + " m=" + std::to_string(m) + " n=" + std::to_string(n) + " batch=5 ->";
        EXPECT_NE(line.find(want), std::string::npos) << line;
        expect_symm(p, line);
    }
}

// With a table naming only the vendor, a vendor-free build takes the last resort, expand.
TYPED_TEST(SymmCandidates, VendorFreeLastResortIsExpand) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_SYMM_ROUTE", nullptr);
    const std::string dtype(select::dtype_name<T>());
    const std::string dev = select::device_of<TestFixture::B>(*this->ctx).key;
    std::vector<std::pair<std::string, std::string>> files;
    for (const auto& t : select::embedded_tables())
        if (t.name.rfind("symm.", 0) != 0) files.emplace_back(std::string(t.name), std::string(t.text));
    files.emplace_back("symm." + dtype + "." + dev + ".txt",
                       "# op=symm dtype=" + dtype + " device=" + dev + " kernels=unknown\n"
                       "# keys: form:exact m:log n:log batch:log\n"
                       "form=sq m=32 n=32 batch=128 | vendor 1\n");
    const TableGuard restore;
    select::testing::set_builtin_tables(std::move(files));
    for (const Spec& s : {Spec{Side::Left, Uplo::Lower, 32, 32, 64}, Spec{Side::Right, Uplo::Upper, 3, 300, 1}}) {
        auto p = make_prob<T>(s);
        std::string all;
        EXPECT_EQ(traced_choice([&] { this->run(p); }, &all), TestFixture::kVendor ? "vendor" : "expand") << label(s);
        if (!TestFixture::kVendor) EXPECT_NE(all.find("last resort"), std::string::npos) << all;
        expect_symm(p, "vendor-only table, " + label(s));
    }
}

// The coverage row (§5.6): the real backend and scalar (the old recorder hard-coded CUDA/F32),
// the old key (m, n = C's extents, k = A's order, uplo, side) and the native flags.
// threadsafe: the child re-executes the binary, so CUDA is initialised fresh, never forked.
TYPED_TEST(SymmCandidates, CoverageRowCarriesBackendKeyAndNativeFlags) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    const std::string dir = ::testing::TempDir() + "symm_cov." + std::string(select::dtype_name<T>());
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const std::string out = dir + "/cov";
    const Spec lo{Side::Left, Uplo::Lower, 16, 5, 2};
    const Spec hi{Side::Right, Uplo::Upper, 7, 40, 2};
    auto child = [&] {
        const ScopedEnvVar cov("BATCHLAS_COVERAGE_OUT", out.c_str());
        const ScopedEnvVar clear("BATCHLAS_SYMM_ROUTE", nullptr);
        batchlas::coverage::g_dynamic_enabled = true;
        {
            const Pin pin("symm", C{sy::Expand{}});
            auto p = make_prob<T>(lo);
            this->run(p);
        }
        const Pin pin("symm", TestFixture::kVendor ? C{sy::Vendor{}} : C{sy::Expand{}});
        auto p = make_prob<T>(hi);
        this->run(p);
        std::exit(0);
    };
    EXPECT_EXIT(child(), ::testing::ExitedWithCode(0), "");
    std::map<std::string, std::vector<std::string>> rows;
    for (const auto& ent : std::filesystem::directory_iterator(dir)) {
        std::ifstream in(ent.path());
        for (std::string line; std::getline(in, line);) {
            if (line.rfind("reached,symm,", 0) != 0) continue;
            std::vector<std::string> f;
            std::stringstream ss(line);
            for (std::string tok; std::getline(ss, tok, ',');) f.push_back(tok);
            ASSERT_GE(f.size(), 19u) << line;
            rows[f[5] + " " + f[6] + " " + f[7]] = f;
        }
    }
    std::filesystem::remove_all(dir);
    ASSERT_EQ(rows.size(), 2u);
    const std::string klo = "16 5 16", khi = "7 40 40";
    ASSERT_TRUE(rows.count(klo)) << "no Left 16x5 row";
    ASSERT_TRUE(rows.count(khi)) << "no Right 7x40 row";
    const std::string backend = B == Backend::CUDA ? "CUDA" : (B == Backend::ROCM ? "ROCM" : "?");
    const std::string scalar = std::is_same_v<T, float> ? "float" : "double";
    auto as_int = [](auto e) { return std::to_string(static_cast<int>(e)); };
    const std::string hi_algo = TestFixture::kVendor ? "vendor" : "expand";
    for (const auto& [key, s, algo] : {std::tuple{klo, lo, std::string("expand")}, std::tuple{khi, hi, hi_algo}}) {
        const auto& f = rows[key];
        EXPECT_EQ(f[2], scalar) << key << ": the scalar column";
        EXPECT_EQ(f[3], backend) << key << ": the backend column";
        EXPECT_EQ(f[9], algo == "vendor" ? "vendor" : "native") << key;
        EXPECT_EQ(f[10], algo) << key;
        EXPECT_EQ(f[12], "1") << key;
        EXPECT_EQ(f[13], "1") << key;
        EXPECT_EQ(f[15], as_int(s.uplo)) << key;
        EXPECT_EQ(f[16], as_int(s.side)) << key;
        EXPECT_EQ(f[17], as_int(Diag::NonUnit)) << key;
        EXPECT_EQ(f[18], as_int(Transpose::NoTrans)) << key;
    }
}

// On a CPU queue no native family can run: expand pins throw, Auto is the vendor (or no route).
template <typename Config>
class SymmCandidatesCpu : public test_utils::BatchLASTest<Config> {};
TYPED_TEST_SUITE(SymmCandidatesCpu, Types);

TYPED_TEST(SymmCandidatesCpu, CpuQueueRunsNoNativeFamily) {
    using T = typename TypeParam::ScalarType;
    static constexpr Backend B = TypeParam::BackendVal;
    if (!this->ctx) GTEST_SKIP() << "no queue";
    if (this->ctx->device().type == DeviceType::GPU) GTEST_SKIP() << "a GPU queue";
    const ScopedEnvVar clear("BATCHLAS_SYMM_ROUTE", nullptr);
    const Spec s{Side::Left, Uplo::Lower, 8, 3, 4};
    {
        auto p = make_prob<T>(s);
        const Pin pin("symm", C{sy::Expand{}});
        EXPECT_THROW(((void)symm<B, T>(*this->ctx, p.A(), p.B(), p.Cv(), p.alpha, p.beta, s.side, s.uplo)),
                     std::invalid_argument);
    }
    auto p = make_prob<T>(s);
    auto call = [&] {
        (void)symm<B, T>(*this->ctx, p.A(), p.B(), p.Cv(), p.alpha, p.beta, s.side, s.uplo);
        this->ctx->wait();
    };
    if constexpr (batchlas::select::level3_vendor_available<B>) {
        EXPECT_EQ(traced_choice(call), "vendor");
        expect_symm(p, "cpu vendor");
    } else {
        EXPECT_THROW(call(), batchlas::NoRouteError);
    }
}

// The netlib loop shares the vendor loops' single top-level (m, n), so a heterogeneous batch has
// no route on a CPU queue either.
TYPED_TEST(SymmCandidatesCpu, HeterogeneousBatchHasNoRoute) {
    if (!this->ctx) GTEST_SKIP() << "no queue";
    if (this->ctx->device().type == DeviceType::GPU) GTEST_SKIP() << "a GPU queue";
    expect_heterogeneous_has_no_route<TypeParam::BackendVal, typename TypeParam::ScalarType>(*this->ctx);
}

// The sm_89 transcription (no GPU): the table holds exactly choice.hh's grid, every row ranks
// both candidates, and the float first entries restate the old rule at the cell's form
// representative (double: vendor everywhere). sm_120 is deep-measured since 2026-10-09.
TEST(SymmTranscribedTable, HoldsTheChoiceGridAndTheOldRule) {
    auto rep = [](const std::string& form, int m, int n) {
        if (std::string(sy::form_of(m, n)) == form) return std::pair{m, n};
        const int M = std::max(m, n), h = std::max(1, (M - 1) / 2);
        return form == "sq" ? std::pair{M, M} : (form == "tall" ? std::pair{M, h} : std::pair{h, M});
    };
    for (const char* dt : {"float", "double"}) {
        for (const char* dev : {"sm_89"}) {
            const auto tables = select::tables_in_borrow_order("symm", dt, select::device_from_key(dev));
            ASSERT_FALSE(tables.empty()) << dt << " " << dev;
            const select::Table& t = *tables.front();
            ASSERT_EQ(t.device, dev) << t.file;
            EXPECT_EQ(t.source, "transcribed:ff340fc6") << t.file;
            ASSERT_EQ(t.keys.size(), sy::key_names.size()) << t.file;
            for (std::size_t i = 0; i < t.keys.size(); ++i) {
                const std::string spec = t.keys[i].name + (t.keys[i].log ? ":log" : ":exact");
                EXPECT_EQ(spec, std::string(sy::key_names[i])) << t.file;
            }
            std::set<std::string> want, got;
            for (auto f : sy::grid_form)
                for (int m : sy::grid_mn)
                    for (int n : sy::grid_mn)
                        for (int b : sy::grid_batch)
                            want.insert(std::string(f) + " " + std::to_string(m) + " " + std::to_string(n) + " " + std::to_string(b));
            for (const auto& row : t.rows) {
                got.insert(row.keys[0] + " " + row.keys[1] + " " + row.keys[2] + " " + row.keys[3]);
                ASSERT_EQ(row.ranked.size(), 2u) << t.file << ":" << row.line;
                EXPECT_FALSE(row.timed) << t.file << ":" << row.line;
                const auto [rm, rn] = rep(row.keys[0], std::stoi(row.keys[1]), std::stoi(row.keys[2]));
                const std::string first = std::string(dt) == "float" ? old_auto<float>(rm, rn, std::stoi(row.keys[3]))
                                                                     : old_auto<double>(rm, rn, std::stoi(row.keys[3]));
                EXPECT_EQ(row.ranked.front().spelling, first) << t.file << ":" << row.line;
            }
            EXPECT_EQ(got, want) << t.file;
            EXPECT_EQ(t.rows.size(), want.size()) << t.file;
        }
    }
    struct Spot { const char* dt; const char* form; int m, n, batch; const char* first; };
    const Spot spots[] = {{"float", "sq", 64, 64, 3, "vendor"},   {"float", "sq", 64, 64, 4, "expand"},
                          {"float", "sq", 255, 200, 2, "vendor"}, {"float", "sq", 256, 200, 2, "expand"},
                          {"float", "tall", 900, 40, 9000, "vendor"}, {"float", "wide", 3, 2000, 50, "vendor"},
                          {"double", "sq", 512, 512, 4096, "vendor"}};
    for (const Spot& s : spots) {
        const select::Table& t = *select::tables_in_borrow_order("symm", s.dt, select::device_from_key("sm_89")).front();
        const select::Key key{{"form", s.form}, {"m", s.m}, {"n", s.n}, {"batch", s.batch}};
        const select::TableRow* row = t.nearest(key);
        ASSERT_NE(row, nullptr);
        EXPECT_EQ(row->ranked.front().spelling, s.first) << t.file << ":" << row->line;
    }
}

}  // namespace
