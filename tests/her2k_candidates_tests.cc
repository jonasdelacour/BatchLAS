// Every her2k candidate, pinned: docs/design/flat-kernel-selection.md §8 and §12 "Hermitian three",
// plus sytrd_blocked's predictor (ops::her2k::fold_chosen). Which kernel ran is read back from the
// select trace or a bit-for-bit comparison with the direct launch, never assumed from the pin.
#include <gtest/gtest.h>

#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/functions/her2k.hh>
#include <batchlas/no_route.hh>
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>
#include "../src/select/coverage.hh"
#include "../src/select/vendor.hh"

#include "test_utils.hh"

#include "../src/backends/accumulate_hermitian.hh"
#include "../src/expansion_budget.hh"
#include "../src/ops/her2k/choice.hh"

#include <algorithm>
#include <cmath>
#include <complex>
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

namespace h2 = batchlas::ops::her2k;
using C = h2::Her2kChoice;
using Pin = select::ScopedPin<C>;
template <typename T>
using MVof = MatrixView<T, MatrixFormat::Dense>;
using cd = std::complex<double>;

// A max that keeps a NaN: std::max(w, NaN) returns w, so a NaN result would pass.
double worse(double w, double x) { return std::isnan(x) || x > w ? x : w; }

// NaN bytes (0xff) in the queue's arena where the next lease starts: a released lease hands its
// bytes to the next one, so a scratch region the op never writes reads this.
void poison_arena(Queue& q, std::size_t bytes) {
    auto ws = q.workspace(bytes);
    const auto s = ws.span();
    q->memset(s.data(), 0xff, s.size());
    q.wait();
}

template <typename T>
bool same_bits(T a, T b) {
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}
template <typename T>
cd wide(T x) {
    return cd(double(x.real()), double(x.imag()));
}

const char* uplo_s(Uplo u) { return u == Uplo::Lower ? "L" : "U"; }
const char* trans_s(Transpose t) { return t == Transpose::NoTrans ? "N" : (t == Transpose::Trans ? "T" : "C"); }

struct Region {
    std::size_t off = 0;
    int rows = 0, cols = 0, ld = 0, stride = 0;
};

struct Spec {
    int n = 8, k = 4, batch = 3;
    Uplo uplo = Uplo::Lower;
    Transpose trans = Transpose::NoTrans;
    int period = 0;  // > 0: item it repeats item it % period
    unsigned seed = 1;
};

std::string label(const Spec& s) {
    return std::string(uplo_s(s.uplo)) + trans_s(s.trans) + " n=" + std::to_string(s.n) + " k=" +
           std::to_string(s.k) + " batch=" + std::to_string(s.batch);
}
std::string name(const C& c, const Spec& s) { return select::to_string(c) + " " + label(s); }

// A, B and C in one buffer of a large finite poison, at padded lds and strides that are not
// ld * cols and differ between A and B. C's other triangle, the pads and the gaps hold the
// poison, and C's diagonal carries an imaginary part (555) that is not an input.
template <typename T>
struct Prob {
    using R = typename T::value_type;
    Spec s;
    T alpha = T(1.25, -0.5);
    R beta = R(-0.75);
    Region a, b, c;
    UnifiedVector<T> mem;
    std::vector<T> mem0;
    UnifiedVector<T*> aptr, bptr, cptr;

    MVof<T> A() { return {mem.data() + a.off, a.rows, a.cols, a.ld, a.stride, s.batch, aptr.data()}; }
    MVof<T> B() { return {mem.data() + b.off, b.rows, b.cols, b.ld, b.stride, s.batch, bptr.data()}; }
    MVof<T> Cv() { return {mem.data() + c.off, s.n, s.n, c.ld, c.stride, s.batch, cptr.data()}; }
    std::size_t at(const Region& r, int it, int i, int j) const {
        return r.off + std::size_t(it) * r.stride + std::size_t(j) * r.ld + i;
    }
    bool ref(int i, int j) const { return s.uplo == Uplo::Lower ? i >= j : i <= j; }
    // op(X)(i, l): X, or X^H for ConjTrans.
    cd op(const Region& r, int it, int i, int l) const {
        return s.trans == Transpose::NoTrans ? wide(mem0[at(r, it, i, l)]) : std::conj(wide(mem0[at(r, it, l, i)]));
    }
};

template <typename T>
T poison() {
    return T(4096.5, -2048.25);
}

template <typename T>
Prob<T> make_prob(const Spec& s) {
    using R = typename T::value_type;
    Prob<T> p;
    p.s = s;
    const int ar = s.trans == Transpose::NoTrans ? s.n : s.k, ac = s.trans == Transpose::NoTrans ? s.k : s.n;
    p.a = {0, ar, ac, ar + 3, (ar + 3) * ac + 5};
    p.b = {std::size_t(p.a.stride) * s.batch + 7, ar, ac, ar + 1, (ar + 1) * ac + 9};
    p.c = {p.b.off + std::size_t(p.b.stride) * s.batch + 11, s.n, s.n, s.n + 2, (s.n + 2) * s.n + 7};
    p.mem = UnifiedVector<T>(p.c.off + std::size_t(p.c.stride) * s.batch + 13, poison<T>());
    p.aptr = UnifiedVector<T*>(s.batch, nullptr);
    p.bptr = UnifiedVector<T*>(s.batch, nullptr);
    p.cptr = UnifiedVector<T*>(s.batch, nullptr);
    std::mt19937 gen(s.seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    auto z = [&] { return T(R(u(gen)), R(u(gen))); };
    const int reps = s.period > 0 ? std::min(s.period, s.batch) : s.batch;
    for (int it = 0; it < s.batch; ++it) {
        const int r = it % reps;
        for (int j = 0; j < ac; ++j)
            for (int i = 0; i < ar; ++i) {
                p.mem[p.at(p.a, it, i, j)] = it < reps ? z() : p.mem[p.at(p.a, r, i, j)];
                p.mem[p.at(p.b, it, i, j)] = it < reps ? z() : p.mem[p.at(p.b, r, i, j)];
            }
        for (int j = 0; j < s.n; ++j)
            for (int i = 0; i < s.n; ++i)
                if (p.ref(i, j))
                    p.mem[p.at(p.c, it, i, j)] = it >= reps ? p.mem[p.at(p.c, r, i, j)]
                                                            : (i == j ? T(R(u(gen)), R(555)) : z());
    }
    p.mem0.assign(p.mem.begin(), p.mem.end());
    return p;
}

// Max over C's triangle of the error against alpha op(A) op(B)^H + conj(alpha) op(B) op(A)^H +
// beta C0 in double (C0's diagonal imaginary part ignored); inf if a diagonal leaves nonzero imag.
template <typename T>
double rel_error(const Prob<T>& p, int it) {
    const cd alpha = wide(p.alpha);
    double worst = 0;
    for (int j = 0; j < p.s.n; ++j)
        for (int i = 0; i < p.s.n; ++i) {
            if (!p.ref(i, j)) continue;
            cd acc = 0;
            double mag = 0;
            for (int l = 0; l < p.s.k; ++l) {
                const cd x = alpha * p.op(p.a, it, i, l) * std::conj(p.op(p.b, it, j, l));
                const cd y = std::conj(alpha) * p.op(p.b, it, i, l) * std::conj(p.op(p.a, it, j, l));
                acc += x + y;
                mag += std::abs(x) + std::abs(y);
            }
            cd c0 = wide(p.mem0[p.at(p.c, it, i, j)]);
            if (i == j) c0 = cd(c0.real(), 0.0);
            if (p.beta == 0) c0 = 0;  // BLAS: beta = 0 makes C output only
            cd want = acc + double(p.beta) * c0;
            if (i == j) want = cd(want.real(), 0.0);
            const cd got = wide(p.mem[p.at(p.c, it, i, j)]);
            if (i == j && got.imag() != 0.0) return std::numeric_limits<double>::infinity();
            const double den = mag + std::abs(double(p.beta) * c0) + 1e-30;
            worst = worse(worst, std::abs(got - want) / den);
        }
    return worst;
}

template <typename T>
double tol(int k) {
    return 16.0 * (k + 2) * double(std::numeric_limits<typename T::value_type>::epsilon());
}

template <typename T>
::testing::AssertionResult correct(const Prob<T>& p) {
    const Spec& s = p.s;
    std::vector<int> items;
    if (s.period > 0) for (int it = 0; it < std::min(s.period, s.batch); ++it) items.push_back(it);
    else if (s.batch <= 8) for (int it = 0; it < s.batch; ++it) items.push_back(it);
    else items = {0, 1, s.batch / 2, s.batch - 1};
    for (int it : items) {
        const double e = rel_error(p, it);
        if (!(std::isfinite(e) && e <= tol<T>(s.k)))
            return ::testing::AssertionFailure() << "item " << it << " relative error " << e;
    }
    std::vector<char> owned(p.mem.size(), 0);
    for (int it = 0; it < s.batch; ++it)
        for (int j = 0; j < s.n; ++j)
            for (int i = 0; i < s.n; ++i)
                if (p.ref(i, j)) owned[p.at(p.c, it, i, j)] = 1;
    for (std::size_t e = 0; e < p.mem.size(); ++e)
        if (!owned[e] && !same_bits(p.mem[e], p.mem0[e]))
            return ::testing::AssertionFailure() << "wrote outside C's triangle at element " << e;
    if (s.period > 0)
        for (int it = s.period; it < s.batch; ++it)
            for (int j = 0; j < s.n; ++j)
                for (int i = 0; i < s.n; ++i)
                    if (p.ref(i, j) && !same_bits(p.mem[p.at(p.c, it, i, j)], p.mem[p.at(p.c, it % s.period, i, j)]))
                        return ::testing::AssertionFailure() << "item " << it << " differs from its representative";
    return ::testing::AssertionSuccess();
}

template <typename T>
bool untouched(const Prob<T>& p) {
    for (std::size_t e = 0; e < p.mem.size(); ++e)
        if (!same_bits(p.mem[e], p.mem0[e])) return false;
    return true;
}

// The outermost her2k trace line for whatever `run` calls (nested gemm lines are indented).
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
        if (line.rfind("her2k ", 0) == 0 && line.find(" -> ") != std::string::npos) return line;
    return "<no her2k trace line in: " + err + ">";
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

void use_her2k_table(const std::string& dtype, const std::string& dev, const std::string& rows) {
    std::vector<std::pair<std::string, std::string>> files;
    for (const auto& t : select::embedded_tables())
        if (t.name.rfind("her2k.", 0) != 0) files.emplace_back(std::string(t.name), std::string(t.text));
    files.emplace_back("her2k." + dtype + "." + dev + ".txt",
                       "# op=her2k dtype=" + dtype + " device=" + dev + " kernels=unknown source=transcribed:abcd1234\n"
                       "# keys: n:log:2 k:log batch:log\n" + rows);
    select::testing::set_builtin_tables(std::move(files));
}

// The old rule (8cf7fd86 expansion_budget.hh her2k_gemm_preferred), restated, unpinned.
std::string old_auto(int n, int batch) { return batch >= 2 || n >= 128 ? "fold" : "vendor"; }

// A heterogeneous operand has no route on any backend (see herk_candidates_tests.cc).
template <Backend B, class T>
void expect_heterogeneous_has_no_route(Queue& ctx) {
    using R = typename T::value_type;
    const int n = 16, k = 8, batch = 4;
    Matrix<T, MatrixFormat::Dense> A(n, k, batch), Bm(n, k, batch), Cm(n, n, batch);
    A.fill(T(0.5, 0.25));
    Bm.fill(T(-0.5, 1));
    Cm.fill(T(1));
    UnifiedVector<int> rows(batch), cols(batch);
    for (int b = 0; b < batch; ++b) rows[b] = n - b, cols[b] = k - (b % 2);
    const auto hA = A.view().with_active_dims(rows.to_span(), cols.to_span());
    const auto hB = Bm.view().with_active_dims(rows.to_span(), cols.to_span());
    const auto hC = Cm.view().with_active_dims(rows.to_span(), rows.to_span());
    struct Case { const char* what; MVof<T> a, b, c; };
    const Case cases[] = {{"A heterogeneous", hA, Bm.view(), Cm.view()},
                          {"B heterogeneous", A.view(), hB, Cm.view()},
                          {"C heterogeneous", A.view(), Bm.view(), hC},
                          {"all three, consistent", hA, hB, hC}};
    for (const auto& kc : cases) {
        auto call = [&] {
            (void)her2k<B, T>(ctx, kc.a, kc.b, kc.c, T(1), R(0), Uplo::Lower, Transpose::NoTrans);
            ctx.wait();
        };
        auto expect_no_route = [&](const std::string& what) {
            if constexpr (select::level3_vendor_available<B>) EXPECT_THROW(call(), std::runtime_error) << what;
            else EXPECT_THROW(call(), batchlas::NoRouteError) << what;
        };
        for (const C& c : h2::candidates<T>()) {
            const Pin pin("her2k", c);
            if (std::holds_alternative<h2::Vendor>(c)) {
                expect_no_route(std::string(kc.what) + ": vendor pin");
                continue;
            }
            try {
                call();
                ADD_FAILURE() << kc.what << " " << select::to_string(c) << " was accepted";
            } catch (const std::invalid_argument& e) {
                EXPECT_NE(std::string(e.what()).find("cannot run this shape"), std::string::npos)
                    << kc.what << " " << select::to_string(c) << ": " << e.what();
            }
        }
        const ScopedEnvVar clear("BATCHLAS_HER2K_ROUTE", nullptr);
        expect_no_route(std::string(kc.what) + ": auto");
    }
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) ASSERT_EQ(Cm(i, j, b), T(1)) << "a refused call wrote C";
}

template <typename T, Backend B>
struct Cfg {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};
using Types = typename test_utils::backend_types_complex<Cfg>::type;

template <typename Config>
class Her2kCandidates : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    using R = typename T::value_type;
    using MV = MVof<T>;
    static constexpr Backend B = Config::BackendVal;
    static constexpr bool kVendor = batchlas::select::level3_vendor_available<B>;

    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        if (this->HasFatalFailure() || ::testing::Test::IsSkipped()) return;
        if (!this->ctx) GTEST_SKIP() << "no queue";
        if (this->ctx->device().type != DeviceType::GPU) GTEST_SKIP() << "the fold is a GPU kernel";
    }

    static bool expect_runs(const C& c, const Spec& s) {
        if (std::holds_alternative<h2::Vendor>(c)) return kVendor;
        return s.batch <= 65535;
    }
    static bool vendor_word_falls_back(const C& c) { return std::holds_alternative<h2::Vendor>(c) && !kVendor; }

    void run(Prob<T>& p) {
        (void)her2k<B, T>(*this->ctx, p.A(), p.B(), p.Cv(), p.alpha, p.beta, p.s.uplo, p.s.trans);
        this->ctx->wait();
    }
    void run_pinned(const C& c, Prob<T>& p) {
        const Pin pin("her2k", c);
        run(p);
    }
    std::string auto_choice(Prob<T>& p) {
        const ScopedEnvVar clear("BATCHLAS_HER2K_ROUTE", nullptr);
        return traced_choice([&] { run(p); });
    }
    static std::string auto_expect(const std::string& first) { return kVendor ? first : "fold"; }
    bool pin_accepted(const C& c, Prob<T>& p) {
        const Pin pin("her2k", c);
        try {
            run(p);
            return true;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find("cannot run this shape") == std::string::npos) throw;
            return false;
        }
    }
    bool fold_chosen(Prob<T>& p) {
        return h2::fold_chosen<B, T>(*this->ctx, p.A(), p.B(), p.Cv(), p.s.trans);
    }

    // The family's own code: Fold = one public gemm alpha op(A) op(B)^H into a scratch product at
    // the launcher's ld, then accumulate_hermitian<true>; Vendor = the library loop.
    bool direct(const C& c, Prob<T>& p, std::string* why = nullptr) {
        Queue& q = *this->ctx;
        const MV A = p.A(), Bm = p.B(), Cm = p.Cv();
        try {
            if (std::holds_alternative<h2::Fold>(c)) {
                const int n = Cm.rows(), batch = Cm.batch_size(), ld = backend::detail::expanded_ld<T>(n);
                UnifiedVector<T> scratch(std::size_t(ld) * n * std::max(batch, 1), T(0));
                const MV product(scratch.data(), n, n, ld, ld * n, batch);
                (void)gemm<B, T>(q, A, Bm, product, p.alpha, T(0), p.s.trans,
                                 p.s.trans == Transpose::NoTrans ? Transpose::ConjTrans : Transpose::NoTrans,
                                 ComputePrecision::Default);
                q.wait();
                (void)backend::detail::accumulate_hermitian<T, true>(q, Cm, product, p.beta, p.s.uplo);
            } else {
                if constexpr (kVendor) {
                    (void)backend::her2k_vendor<B, T>(q, A, Bm, Cm, p.alpha, p.beta, p.s.uplo, p.s.trans);
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
};

TYPED_TEST_SUITE(Her2kCandidates, Types);

// Shapes straddling the fold's 32-row group, the old n 127|128 threshold and ragged k.
const Spec kStraddle[] = {{1, 1, 2},   {31, 7, 1},  {32, 8, 2},   {33, 53, 3},  {64, 5, 1},
                          {127, 9, 2}, {128, 64, 1}, {129, 3, 2}, {200, 53, 1}};

TYPED_TEST(Her2kCandidates, PinnedCandidatesStraddleTheirLimits) {
    using T = typename TestFixture::T;
    for (const C& c : h2::candidates<T>())
        for (Spec s : kStraddle)
            for (Transpose t : {Transpose::NoTrans, Transpose::ConjTrans})
                for (Uplo u : {Uplo::Lower, Uplo::Upper}) {
                    s.trans = t;
                    s.uplo = u;
                    s.seed = 7u + s.n;
                    if (this->vendor_word_falls_back(c)) continue;
                    auto p = make_prob<T>(s);
                    const bool want = this->expect_runs(c, s);
                    EXPECT_EQ(this->pin_accepted(c, p), want) << name(c, s);
                    if (want) EXPECT_TRUE(correct(p)) << name(c, s);
                    else EXPECT_TRUE(untouched(p)) << name(c, s) << ": a refused pin wrote";
                }
}

TYPED_TEST(Her2kCandidates, FoldFitStraddlesTheScratchBudget) {
    using T = typename TestFixture::T;
    for (Spec s : {Spec{33, 9, 3}, Spec{64, 40, 2, Uplo::Upper, Transpose::ConjTrans}}) {
        const std::size_t bytes = backend::detail::expanded_workspace_bytes<T>(*this->ctx, s.n, s.batch);
        for (std::size_t cap : {bytes, bytes - 1}) {
            const ScopedEnvVar budget("BATCHLAS_EXPAND_MAX_BYTES", std::to_string(cap).c_str());
            auto p = make_prob<T>(s);
            const bool ok = this->pin_accepted(C{h2::Fold{}}, p);
            EXPECT_EQ(ok, cap == bytes) << label(s) << " budget " << cap << " of " << bytes;
            if (ok) EXPECT_TRUE(correct(p)) << label(s) << " at the exact budget";
            else EXPECT_TRUE(untouched(p)) << label(s);
        }
    }
}

TYPED_TEST(Her2kCandidates, EveryCombinationOnEveryCandidate) {
    using T = typename TestFixture::T;
    for (const Spec base : {Spec{48, 37, 1}, Spec{100, 53, 2}, Spec{200, 9, 1}})
        for (Uplo u : {Uplo::Lower, Uplo::Upper})
            for (Transpose t : {Transpose::NoTrans, Transpose::ConjTrans}) {
                Spec s = base;
                s.uplo = u;
                s.trans = t;
                for (const C& c : h2::candidates<T>()) {
                    if (this->vendor_word_falls_back(c)) continue;
                    auto p = make_prob<T>(s);
                    EXPECT_EQ(this->pin_accepted(c, p), this->expect_runs(c, s)) << name(c, s);
                    EXPECT_TRUE(correct(p)) << name(c, s);
                }
                auto p = make_prob<T>(s);
                EXPECT_EQ(this->auto_choice(p), TestFixture::auto_expect(old_auto(s.n, s.batch))) << label(s);
                EXPECT_TRUE(correct(p)) << "auto " << label(s);
            }
}

// BLAS: beta = 0 makes C output only, so NaN in C's triangle must not reach the result. The fold's
// gemm writes a never-zeroed arena lease at beta = 0 that the native direct and tiled gemms read
// (known-defects.md #11): the arena holds NaN bytes first, and the inner gemm also runs pinned to
// those two (vendor-free Auto reaches them at n = 8; the vendor tree never does).
TYPED_TEST(Her2kCandidates, BetaZeroDoesNotReadC) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;
    const ScopedEnvVar clear("BATCHLAS_HER2K_ROUTE", nullptr);
    std::vector<std::optional<C>> pins{std::nullopt};
    for (const C& c : h2::candidates<T>())
        if (!this->vendor_word_falls_back(c)) pins.push_back(c);
    const R nan = std::numeric_limits<R>::quiet_NaN();
    for (const char* gemm_route : {static_cast<const char*>(nullptr), "direct", "tiled"})
    for (const Spec& s : {Spec{40, 9, 5}, Spec{100, 33, 1, Uplo::Upper, Transpose::ConjTrans}, Spec{8, 8, 2}})
        for (const auto& c : pins) {
            const ScopedEnvVar inner("BATCHLAS_GEMM_ROUTE", gemm_route);
            poison_arena(*this->ctx, 4 * backend::detail::expanded_workspace_bytes<T>(*this->ctx, s.n, s.batch));
            auto p = make_prob<T>(s);
            p.beta = R(0);
            for (int it = 0; it < s.batch; ++it)
                for (int j = 0; j < s.n; ++j)
                    for (int i = 0; i < s.n; ++i)
                        if (p.ref(i, j)) p.mem[p.at(p.c, it, i, j)] = T(nan, nan);
            p.mem0.assign(p.mem.begin(), p.mem.end());
            std::optional<Pin> pin;
            if (c) pin.emplace("her2k", *c);
            this->run(p);
            EXPECT_TRUE(correct(p)) << (c ? select::to_string(*c) : std::string("auto")) << " " << label(s)
                                    << " gemm " << (gemm_route ? gemm_route : "auto");
        }
}

TYPED_TEST(Her2kCandidates, PinnedRunIsTheDirectKernelBitForBit) {
    using T = typename TestFixture::T;
    int compared = 0;
    for (const C& c : h2::candidates<T>()) {
        if (this->vendor_word_falls_back(c)) continue;
        for (Spec s : {Spec{48, 37, 3, Uplo::Upper, Transpose::ConjTrans}, Spec{100, 9, 2, Uplo::Lower}}) {
            s.seed = 11;
            auto pinned = make_prob<T>(s);
            this->run_pinned(c, pinned);
            auto direct = make_prob<T>(s);
            std::string why;
            ASSERT_TRUE(this->direct(c, direct, &why)) << name(c, s) << ": " << why;
            EXPECT_TRUE(correct(pinned)) << name(c, s);
            for (std::size_t e = 0; e < pinned.mem.size(); ++e)
                ASSERT_TRUE(same_bits(pinned.mem[e], direct.mem[e])) << name(c, s) << " element " << e;
            ++compared;
        }
    }
    EXPECT_GT(compared, 0);
}

// 1024 items repeating 7 problems: each item's C bit-identical to its representative's.
TYPED_TEST(Her2kCandidates, SaturatingBatchIsBitIdentical) {
    using T = typename TestFixture::T;
    for (Spec s : {Spec{48, 70, 1024, Uplo::Lower, Transpose::NoTrans}, Spec{130, 33, 1024, Uplo::Upper,
                                                                              Transpose::ConjTrans}}) {
        s.period = 7;
        auto p = make_prob<T>(s);
        this->run_pinned(C{h2::Fold{}}, p);
        EXPECT_TRUE(correct(p)) << label(s);
    }
}

// §8.2 (R3) over the straddle set: a pin is accepted exactly when the family's own code runs and
// answers correctly.
TYPED_TEST(Her2kCandidates, CanRunEqualsLaunch) {
    using T = typename TestFixture::T;
    for (const C& c : h2::candidates<T>()) {
        if (this->vendor_word_falls_back(c)) continue;
        for (Spec s : kStraddle) {
            s.uplo = Uplo::Upper;
            s.trans = Transpose::ConjTrans;
            auto p = make_prob<T>(s);
            const bool accepted = this->pin_accepted(c, p);
            auto d = make_prob<T>(s);
            std::string why;
            const bool launched = this->direct(c, d, &why);
            EXPECT_EQ(accepted, launched && correct(d)) << name(c, s) << (launched ? "" : " (direct: " + why + ")");
        }
    }
}

// The grid term: the fold puts the batch in grid z. At 65535 the pin launches; at 65536 the fold's
// own kernel throws, the pin is refused, and Auto takes the vendor (vendor-free: no route).
TYPED_TEST(Her2kCandidates, GridBatchCeiling) {
    using T = typename TestFixture::T;
    const int ceiling = int(h2::kMaxGridBatch);
    Spec at{1, 1, ceiling};
    at.period = 1;
    auto p = make_prob<T>(at);
    ASSERT_TRUE(this->pin_accepted(C{h2::Fold{}}, p));
    EXPECT_TRUE(correct(p));
    Spec past = at;
    past.batch = ceiling + 1;
    auto q = make_prob<T>(past);
    EXPECT_FALSE(this->pin_accepted(C{h2::Fold{}}, q));
    EXPECT_TRUE(untouched(q));
    UnifiedVector<T> product(std::size_t(past.batch) * 2, T(0));
    const MVof<T> prod(product.data(), 1, 1, 2, 2, past.batch);
    std::string why;
    try {
        (void)backend::detail::accumulate_hermitian<T, true>(*this->ctx, q.Cv(), prod, q.beta, Uplo::Lower);
        this->ctx->wait();
    } catch (const std::exception& x) {
        why = x.what();
    }
    EXPECT_NE(why.find("exceed limit"), std::string::npos) << why;
    auto r = make_prob<T>(past);
    if constexpr (TestFixture::kVendor) {
        EXPECT_EQ(this->auto_choice(r), "vendor");
        EXPECT_TRUE(correct(r)) << "auto past the grid ceiling";
    } else {
        const ScopedEnvVar clear("BATCHLAS_HER2K_ROUTE", nullptr);
        EXPECT_THROW(this->run(r), batchlas::NoRouteError);
    }
}

// The fold's capacity at batch 65535 (AGENTS §8.9, a launch at the ceiling), as herk's: the padded
// range of accumulate_hermitian<true> overflows an int from n = 169 (expand_grid_fits), inside
// expansion_fits' n^2 batch term. n = 168 launches and answers; 169 and 181 are refused before the
// gemm runs; Auto takes the vendor. Skipped where the memory budget refuses n = 168 first.
TYPED_TEST(Her2kCandidates, FoldGridCeiling) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;
    static constexpr Backend B = TestFixture::B;
    Queue& q = *this->ctx;
    const int batch = int(h2::kMaxGridBatch), k = 2;
    const std::size_t budget = q.device().get_property(DeviceProperty::GLOBAL_MEM_SIZE) / 4;
    if (backend::detail::expanded_workspace_bytes<T>(q, 168, batch) > budget)
        GTEST_SKIP() << "the memory budget refuses n = 168 at batch 65535 on this device";
    const T alpha(1.25, -0.5);
    for (int n : {168, 169, 181}) {
        UnifiedVector<T> a(std::size_t(n) * k * batch), b(std::size_t(n) * k * batch), c(std::size_t(n) * n * batch, T(-3, 7));
        for (std::size_t e = 0; e < a.size(); ++e) {
            a[e] = T(R(0.25) * R(e % 7) - R(0.5), R(0.125) * R(e % 5 + e % 3));
            b[e] = T(R(0.5) - R(0.125) * R(e % 9), R(0.25) * R(e % 4) - R(0.3));
        }
        const MVof<T> A(a.data(), n, k, n, n * k, batch), Bm(b.data(), n, k, n, n * k, batch);
        const MVof<T> Cm(c.data(), n, n, n, n * n, batch);
        auto call = [&] {
            (void)her2k<B, T>(q, A, Bm, Cm, alpha, R(0), Uplo::Upper, Transpose::NoTrans);
            q.wait();
        };
        auto worst = [&] {
            double w = 0;
            for (int it : {0, batch / 2, batch - 1})
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i <= j; ++i) {
                        const std::size_t ab = std::size_t(it) * n * k;
                        cd want = 0;
                        for (int l = 0; l < k; ++l)
                            want += wide(alpha) * wide(a[ab + i + l * n]) * std::conj(wide(b[ab + j + l * n])) +
                                    std::conj(wide(alpha)) * wide(b[ab + i + l * n]) * std::conj(wide(a[ab + j + l * n]));
                        if (i == j) want = cd(want.real(), 0.0);
                        w = worse(w, std::abs(wide(c[std::size_t(it) * n * n + i + std::size_t(j) * n]) - want));
                    }
            return w;
        };
        const std::string what = "n = " + std::to_string(n) + " batch = 65535";
        if (n == 168) {
            const Pin pin("her2k", C{h2::Fold{}});
            ASSERT_NO_THROW(call()) << what << ": inside both terms, the fold must launch";
            EXPECT_LE(worst(), tol<T>(k)) << what;
            continue;
        }
        {
            const Pin pin("her2k", C{h2::Fold{}});
            try {
                call();
                ADD_FAILURE() << what << ": the fold was accepted";
            } catch (const std::invalid_argument& e) {
                EXPECT_NE(std::string(e.what()).find("cannot run this shape"), std::string::npos) << what << ": " << e.what();
            }
        }
        EXPECT_TRUE(same_bits(c[0], T(-3, 7))) << what << ": a refused pin wrote C";
        if (n != 169) continue;
        const ScopedEnvVar clear("BATCHLAS_HER2K_ROUTE", nullptr);
        if constexpr (TestFixture::kVendor) {
            EXPECT_EQ(traced_choice(call), "vendor") << what;
            EXPECT_LE(worst(), tol<T>(k)) << what << " under Auto";
        } else {
            EXPECT_THROW(call(), batchlas::NoRouteError) << what;
        }
    }
}

TYPED_TEST(Her2kCandidates, InvalidShapesThrowBeforeChoose) {
    using T = typename TestFixture::T;
    using R = typename T::value_type;
    static constexpr Backend B = TestFixture::B;
    auto p = make_prob<T>(Spec{8, 5, 3});
    std::vector<std::optional<C>> pins{std::nullopt};
    for (const C& c : h2::candidates<T>()) pins.push_back(c);
    struct Bad { const char* what; MVof<T> a, b, c; Transpose t; };
    const Bad bads[] = {
        {"plain Trans (complex-symmetric)", p.A(), p.B(), p.Cv(), Transpose::Trans},
        {"C not square", p.A(), p.B(), MVof<T>(p.mem.data() + p.c.off, 8, 7, p.c.ld, p.c.stride, 3), Transpose::NoTrans},
        {"B's k differs", p.A(), MVof<T>(p.mem.data() + p.b.off, 8, 4, p.b.ld, p.b.stride, 3), p.Cv(), Transpose::NoTrans},
        {"batch mismatch", p.A(), MVof<T>(p.mem.data() + p.b.off, 8, 5, p.b.ld, p.b.stride, 2), p.Cv(), Transpose::NoTrans}};
    for (const auto& bad : bads)
        for (const auto& c : pins) {
            std::optional<Pin> pin;
            if (c) pin.emplace("her2k", *c);
            EXPECT_THROW(((void)her2k<B, T>(*this->ctx, bad.a, bad.b, bad.c, T(1), R(1), Uplo::Lower, bad.t)),
                         std::invalid_argument)
                << bad.what << " under " << (c ? select::to_string(*c) : std::string("auto"));
        }
    this->ctx->wait();
    EXPECT_TRUE(untouched(p)) << "a refused call wrote";
}

TYPED_TEST(Her2kCandidates, UnknownPinsThrow) {
    using T = typename TestFixture::T;
    for (const char* word : {"loop", "expand", "gemm", "gram", "triangular", "cublasdx", "native:fold",
                             "vendor:auto", "fold:1", "syr2k", "rank2k", "bogus"}) {
        auto p = make_prob<T>(Spec{});
        const Pin pin("her2k", std::string_view(word));
        EXPECT_THROW(this->run(p), std::invalid_argument) << word;
        EXPECT_TRUE(untouched(p)) << word;
    }
}

TYPED_TEST(Her2kCandidates, CanRunFalsePinsThrow) {
    using T = typename TestFixture::T;
    const ScopedEnvVar budget("BATCHLAS_EXPAND_MAX_BYTES", "0");
    auto p = make_prob<T>(Spec{16, 8, 2});
    const Pin pin("her2k", C{h2::Fold{}});
    try {
        this->run(p);
        ADD_FAILURE() << "fold over BATCHLAS_EXPAND_MAX_BYTES was accepted";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("cannot run this shape"), std::string::npos) << e.what();
    }
    EXPECT_TRUE(untouched(p));
}

TYPED_TEST(Her2kCandidates, HeterogeneousBatchHasNoRoute) {
    expect_heterogeneous_has_no_route<TestFixture::B, typename TestFixture::T>(*this->ctx);
}

TYPED_TEST(Her2kCandidates, EmptyProblemIsANoOp) {
    using T = typename TestFixture::T;
    using R = typename T::value_type;
    static constexpr Backend B = TestFixture::B;
    UnifiedVector<T> a(256, T(1)), b(256, T(3)), c(256, T(2, -1));
    struct Case { int n, k, batch; };
    const Case cases[] = {{8, 4, 0}, {0, 4, 2}};
    std::vector<std::optional<C>> pins{std::nullopt};
    for (const C& k : h2::candidates<T>()) pins.push_back(k);
    const ScopedEnvVar clear("BATCHLAS_HER2K_ROUTE", nullptr);
    for (const auto& k : cases) {
        const MVof<T> A(a.data(), k.n, k.k, std::max(k.n, 1), 64, k.batch);
        const MVof<T> Bm(b.data(), k.n, k.k, std::max(k.n, 1), 64, k.batch);
        const MVof<T> Cm(c.data(), k.n, k.n, std::max(k.n, 1), 64, k.batch);
        for (const auto& p : pins) {
            std::optional<Pin> pin;
            if (p) pin.emplace("her2k", *p);
            EXPECT_NO_THROW(((void)her2k<B, T>(*this->ctx, A, Bm, Cm, T(1.5), R(-0.75), Uplo::Lower, Transpose::NoTrans),
                             this->ctx->wait()))
                << (p ? select::to_string(*p) : std::string("auto")) << " n=" << k.n << " batch=" << k.batch;
        }
    }
    for (std::size_t e = 0; e < c.size(); ++e) ASSERT_EQ(c[e], T(2, -1)) << "an empty problem wrote element " << e;
}

TYPED_TEST(Her2kCandidates, ClassWordsAndSpellings) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_HER2K_ROUTE", nullptr);
    const Spec s{20, 24, 1, Uplo::Upper, Transpose::ConjTrans};  // batch 1, n < 128: vendor | fold
    std::string auto_pick;
    {
        auto p = make_prob<T>(s);
        auto_pick = this->auto_choice(p);
    }
    EXPECT_EQ(auto_pick, TestFixture::auto_expect("vendor"));
    const std::string vendor_pick = TestFixture::kVendor ? "vendor" : auto_pick;
    const std::pair<const char*, std::string> expect[] = {
        {"FOLD", "fold"}, {" fold ", "fold"}, {"native", "fold"}, {"vendor", vendor_pick}, {"Auto", auto_pick}};
    for (const auto& [word, spelling] : expect)
        for (bool via_env : {false, true}) {
            auto p = make_prob<T>(s);
            select::testing::reset_warnings();
            std::string err;
            const std::string got = traced_choice([&] {
                const ScopedEnvVar env("BATCHLAS_HER2K_ROUTE", via_env ? word : nullptr);
                std::optional<Pin> pin;
                if (!via_env) pin.emplace("her2k", std::string_view(word));
                this->run(p);
            }, &err);
            const std::string what = std::string("'") + word + (via_env ? "' via BATCHLAS_HER2K_ROUTE" : "' via ScopedPin");
            EXPECT_EQ(got, spelling) << what;
            const bool warns = std::string(word) == "vendor" && !TestFixture::kVendor;
            EXPECT_EQ(err.find("her2k pinned \"vendor\", but no vendor candidate") != std::string::npos, warns) << what << ": " << err;
            EXPECT_TRUE(correct(p)) << what;
        }
}

TYPED_TEST(Her2kCandidates, ScopedPinBeatsTheEnvironment) {
    using T = typename TestFixture::T;
    const Spec s{12, 7, 1};
    const ScopedEnvVar env("BATCHLAS_HER2K_ROUTE", "bogus");
    {
        auto p = make_prob<T>(s);
        EXPECT_THROW(this->run(p), std::invalid_argument) << "the environment pin was not read";
    }
    auto p = make_prob<T>(s);
    EXPECT_EQ(traced_choice([&] {
                  const Pin pin("her2k", C{h2::Fold{}});
                  this->run(p);
              }),
              "fold");
    EXPECT_TRUE(correct(p));
    const Pin outer("her2k", std::string_view("bogus"));
    {
        const Pin inner("her2k", C{h2::Fold{}});
        auto r = make_prob<T>(s);
        EXPECT_EQ(traced_choice([&] { this->run(r); }), "fold");
    }
    auto r = make_prob<T>(s);
    EXPECT_THROW(this->run(r), std::invalid_argument) << "the inner pin did not restore the outer";
}

// BATCHLAS_EXPAND_ROUTE is retired: neither word moves Auto off the table's choice.
TYPED_TEST(Her2kCandidates, ExpandRouteIsNotRead) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_HER2K_ROUTE", nullptr);
    for (const Spec& s : {Spec{40, 9, 1}, Spec{40, 9, 2}})
        for (const char* word : {"expand", "loop"}) {
            const ScopedEnvVar retired("BATCHLAS_EXPAND_ROUTE", word);
            auto p = make_prob<T>(s);
            EXPECT_EQ(this->auto_choice(p), TestFixture::auto_expect(old_auto(s.n, s.batch))) << word << " " << label(s);
            EXPECT_TRUE(correct(p)) << word;
        }
}

// Auto against the transcribed table, both sides of batch 1|2 and n 127|128, on and off the grid.
TYPED_TEST(Her2kCandidates, AutoReadsTheTranscribedTable) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    const ScopedEnvVar clear("BATCHLAS_HER2K_ROUTE", nullptr);
    const auto tables = select::tables_in_borrow_order("her2k", select::dtype_name<T>(), select::device_of<B>(*this->ctx));
    ASSERT_FALSE(tables.empty());
    if (tables.front()->source != "transcribed:8cf7fd86") GTEST_SKIP() << "this device reads a measured her2k table";
    struct Cell { int n, k, batch; };
    const Cell cells[] = {{64, 16, 1},  {64, 16, 2},  {127, 9, 1}, {128, 9, 1}, {129, 9, 1}, {100, 3, 1},
                          {150, 3, 1},  {120, 2, 3},  {5, 70, 1},  {1, 1, 1},   {1, 1, 2},   {300, 30, 1000},
                          {900, 2, 1},  {33, 50, 8}};
    for (const Cell& c : cells)
        for (Transpose t : {Transpose::NoTrans, Transpose::ConjTrans}) {
            Spec s{c.n, c.k, c.batch, Uplo::Lower, t};
            s.seed = 41u + c.n;
            auto p = make_prob<T>(s);
            EXPECT_EQ(traced_choice([&] { this->run(p); }), TestFixture::auto_expect(old_auto(c.n, c.batch))) << label(s);
            EXPECT_TRUE(correct(p)) << "auto " << label(s);
        }
}

// key_of's every field reaches select::pick: a synthetic table whose winner changes with n, k, batch.
TYPED_TEST(Her2kCandidates, AutoReadsEveryKeyField) {
    using T = typename TestFixture::T;
    if (!TestFixture::kVendor) GTEST_SKIP() << "only fold can run vendor-free, so no row can steer Auto";
    const ScopedEnvVar clear("BATCHLAS_HER2K_ROUTE", nullptr);
    const TableGuard restore;
    use_her2k_table(std::string(select::dtype_name<T>()), select::device_of<TestFixture::B>(*this->ctx).key,
                    "n=16 k=16 batch=128 | fold - | vendor -\n"
                    "n=16 k=16 batch=8192 | vendor - | fold -\n"
                    "n=4 k=16 batch=128 | vendor - | fold -\n"
                    "n=16 k=2 batch=128 | vendor - | fold -\n");
    struct Probe { Spec s; const char* expect; const char* field; };
    const Probe probes[] = {{Spec{16, 16, 128}, "fold", "base"},
                            {Spec{16, 16, 8192}, "vendor", "batch"},
                            {Spec{4, 16, 128}, "vendor", "n"},
                            {Spec{16, 2, 128}, "vendor", "k (NoTrans)"},
                            {Spec{16, 2, 128, Uplo::Upper, Transpose::ConjTrans}, "vendor", "k (ConjTrans)"}};
    for (const auto& k : probes) {
        Spec s = k.s;
        s.seed = 43u;
        if (s.batch > 4096) s.period = 1;
        auto p = make_prob<T>(s);
        EXPECT_EQ(traced_choice([&] { this->run(p); }), k.expect) << "the " << k.field << " row, " << label(s);
        EXPECT_TRUE(correct(p)) << k.field;
    }
}

TYPED_TEST(Her2kCandidates, TraceKeyIsNKBatch) {
    using T = typename TestFixture::T;
    for (Transpose t : {Transpose::NoTrans, Transpose::ConjTrans}) {
        Spec s{37, 12, 5, Uplo::Upper, t};
        auto p = make_prob<T>(s);
        const std::string line = traced_line([&] { this->run(p); });
        EXPECT_NE(line.find("n=37 k=12 batch=5 ->"), std::string::npos) << line;
        EXPECT_TRUE(correct(p)) << line;
    }
}

TYPED_TEST(Her2kCandidates, VendorFreeLastResortIsFold) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_HER2K_ROUTE", nullptr);
    const TableGuard restore;
    use_her2k_table(std::string(select::dtype_name<T>()), select::device_of<TestFixture::B>(*this->ctx).key,
                    "n=32 k=32 batch=128 | vendor -\n");
    for (const Spec& s : {Spec{32, 32, 64}, Spec{300, 3, 1, Uplo::Upper, Transpose::ConjTrans}}) {
        auto p = make_prob<T>(s);
        std::string all;
        EXPECT_EQ(traced_choice([&] { this->run(p); }, &all), TestFixture::kVendor ? "vendor" : "fold") << label(s);
        if (!TestFixture::kVendor) EXPECT_NE(all.find("last resort"), std::string::npos) << all;
        EXPECT_TRUE(correct(p)) << "vendor-only table, " << label(s);
    }
}

// sytrd_blocked's predictor is her2k's own select::pick (ops::her2k::fold_chosen): it agrees with
// the traced choice over cells straddling the old rule and the scratch budget, follows the pins,
// and throws on a bad one.
TYPED_TEST(Her2kCandidates, FoldChosenIsTheCallsChoice) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_HER2K_ROUTE", nullptr);
    int folds = 0, others = 0;
    for (const Spec& s : {Spec{64, 16, 1}, Spec{64, 16, 2}, Spec{127, 9, 1}, Spec{128, 9, 1}, Spec{224, 16, 4},
                          Spec{480, 32, 2, Uplo::Lower, Transpose::NoTrans}})
        for (const char* budget : {static_cast<const char*>(nullptr), "0"}) {
            if (budget && !TestFixture::kVendor) continue;  // nothing can run: NoRouteError, as the call
            const ScopedEnvVar cap("BATCHLAS_EXPAND_MAX_BYTES", budget);
            auto p = make_prob<T>(s);
            const bool predicted = this->fold_chosen(p);
            const std::string ran = this->auto_choice(p);
            EXPECT_EQ(predicted, ran == "fold") << label(s) << (budget ? " budget 0" : "") << " ran " << ran;
            (predicted ? folds : others) += 1;
        }
    EXPECT_GT(folds, 0);
    if (TestFixture::kVendor) EXPECT_GT(others, 0);
    auto p = make_prob<T>(Spec{224, 16, 4});
    {
        const Pin pin("her2k", std::string_view("vendor"));
        EXPECT_EQ(this->fold_chosen(p), !TestFixture::kVendor) << "the vendor pin";
    }
    {
        const Pin pin("her2k", C{h2::Fold{}});
        EXPECT_TRUE(this->fold_chosen(p));
    }
    const Pin bad("her2k", std::string_view("loop"));
    EXPECT_THROW((void)this->fold_chosen(p), std::invalid_argument);
}

TYPED_TEST(Her2kCandidates, CoverageRowCarriesBackendKeyAndNativeFlags) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    const std::string dir = ::testing::TempDir() + "her2k_cov." + std::string(select::dtype_name<T>());
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const std::string out = dir + "/cov";
    const Spec lo{16, 5, 2, Uplo::Lower, Transpose::NoTrans};
    const Spec hi{40, 7, 2, Uplo::Upper, Transpose::ConjTrans};
    auto child = [&] {
        const ScopedEnvVar cov("BATCHLAS_COVERAGE_OUT", out.c_str());
        const ScopedEnvVar clear("BATCHLAS_HER2K_ROUTE", nullptr);
        batchlas::coverage::g_dynamic_enabled = true;
        {
            const Pin pin("her2k", C{h2::Fold{}});
            auto p = make_prob<T>(lo);
            this->run(p);
        }
        const Pin pin("her2k", TestFixture::kVendor ? C{h2::Vendor{}} : C{h2::Fold{}});
        auto p = make_prob<T>(hi);
        this->run(p);
        std::exit(0);
    };
    EXPECT_EXIT(child(), ::testing::ExitedWithCode(0), "");
    std::map<std::string, std::vector<std::string>> rows;
    for (const auto& ent : std::filesystem::directory_iterator(dir)) {
        std::ifstream in(ent.path());
        for (std::string line; std::getline(in, line);) {
            if (line.rfind("reached,her2k,", 0) != 0) continue;
            std::vector<std::string> f;
            std::stringstream ss(line);
            for (std::string tok; std::getline(ss, tok, ',');) f.push_back(tok);
            ASSERT_GE(f.size(), 19u) << line;
            rows[f[5] + " " + f[6] + " " + f[7]] = f;
        }
    }
    std::filesystem::remove_all(dir);
    ASSERT_EQ(rows.size(), 2u);
    auto as_int = [](auto e) { return std::to_string(static_cast<int>(e)); };
    const std::string hi_algo = TestFixture::kVendor ? "vendor" : "fold";
    for (const auto& [key, s, algo] : {std::tuple{std::string("16 16 5"), lo, std::string("fold")},
                                       std::tuple{std::string("40 40 7"), hi, hi_algo}}) {
        ASSERT_TRUE(rows.count(key)) << "no " << key << " row";
        const auto& f = rows[key];
        EXPECT_EQ(f[2], std::string(batchlas::to_string(batchlas::scalar_kind_of<T>))) << key << ": the scalar column";
        EXPECT_EQ(f[3], B == Backend::CUDA ? "CUDA" : "?") << key << ": the backend column";
        EXPECT_EQ(f[9], algo == "vendor" ? "vendor" : "native") << key;
        EXPECT_EQ(f[10], algo) << key;
        EXPECT_EQ(f[12], "1") << key;
        EXPECT_EQ(f[13], "1") << key;
        EXPECT_EQ(f[15], as_int(s.uplo)) << key;
        EXPECT_EQ(f[18], as_int(s.trans)) << key;
    }
}

template <typename Config>
class Her2kCandidatesCpu : public test_utils::BatchLASTest<Config> {};
TYPED_TEST_SUITE(Her2kCandidatesCpu, Types);

TYPED_TEST(Her2kCandidatesCpu, CpuQueueRunsNoNativeFamily) {
    using T = typename TypeParam::ScalarType;
    static constexpr Backend B = TypeParam::BackendVal;
    if (!this->ctx) GTEST_SKIP() << "no queue";
    if (this->ctx->device().type == DeviceType::GPU) GTEST_SKIP() << "a GPU queue";
    const ScopedEnvVar clear("BATCHLAS_HER2K_ROUTE", nullptr);
    const Spec s{8, 3, 4};
    {
        auto p = make_prob<T>(s);
        const Pin pin("her2k", C{h2::Fold{}});
        EXPECT_THROW(((void)her2k<B, T>(*this->ctx, p.A(), p.B(), p.Cv(), p.alpha, p.beta, s.uplo, s.trans)),
                     std::invalid_argument);
    }
    auto p = make_prob<T>(s);
    auto call = [&] {
        (void)her2k<B, T>(*this->ctx, p.A(), p.B(), p.Cv(), p.alpha, p.beta, s.uplo, s.trans);
        this->ctx->wait();
    };
    if constexpr (batchlas::select::level3_vendor_available<B>) {
        EXPECT_EQ(traced_choice(call), "vendor");
        EXPECT_TRUE(correct(p));
    } else {
        EXPECT_THROW(call(), batchlas::NoRouteError);
    }
}

TYPED_TEST(Her2kCandidatesCpu, HeterogeneousBatchHasNoRoute) {
    if (!this->ctx) GTEST_SKIP() << "no queue";
    if (this->ctx->device().type == DeviceType::GPU) GTEST_SKIP() << "a GPU queue";
    expect_heterogeneous_has_no_route<TypeParam::BackendVal, typename TypeParam::ScalarType>(*this->ctx);
}

// The transcription (no GPU): each table holds exactly choice.hh's grid, sm_89 and sm_120 alike,
// names the transcribed commit and choice.hh's keys, and reads `fold | vendor` where the old rule
// folded and `vendor | fold` elsewhere.
TEST(Her2kTranscribedTable, HoldsTheChoiceGridAndTheOldRule) {
    std::set<std::string> want;
    for (int n : h2::grid_n)
        for (int k : h2::grid_k)
            for (int b : h2::grid_batch) want.insert(std::to_string(n) + " " + std::to_string(k) + " " + std::to_string(b));
    for (const char* dt : {"cfloat", "cdouble"}) {
        std::map<std::string, std::string> first;
        for (const char* dev : {"sm_89", "sm_120"}) {
            const auto tables = select::tables_in_borrow_order("her2k", dt, select::device_from_key(dev));
            ASSERT_FALSE(tables.empty()) << dt << " " << dev;
            const select::Table& t = *tables.front();
            ASSERT_EQ(t.device, dev) << t.file;
            EXPECT_EQ(t.source, "transcribed:8cf7fd86") << t.file;
            ASSERT_EQ(t.keys.size(), h2::key_names.size()) << t.file;
            for (std::size_t i = 0; i < t.keys.size(); ++i) {
                const std::string w = t.keys[i].weight == 1.0 ? "" : ":" + std::to_string(int(t.keys[i].weight));
                EXPECT_EQ(t.keys[i].name + (t.keys[i].log ? ":log" : ":exact") + w, std::string(h2::key_names[i]))
                    << t.file;
            }
            std::set<std::string> got;
            for (const auto& row : t.rows) {
                const std::string k = row.keys[0] + " " + row.keys[1] + " " + row.keys[2];
                got.insert(k);
                EXPECT_FALSE(row.timed) << t.file << ":" << row.line;
                const int n = std::stoi(row.keys[0]), b = std::stoi(row.keys[2]);
                std::string ranked;
                for (const auto& x : row.ranked) ranked += (ranked.empty() ? "" : "|") + x.spelling;
                EXPECT_EQ(ranked, old_auto(n, b) == "fold" ? "fold|vendor" : "vendor|fold") << t.file << ":" << row.line;
                if (std::string(dev) == "sm_89") first[k] = ranked;
                else EXPECT_EQ(ranked, first[k]) << t.file << " " << k << ": sm_120 differs from sm_89";
            }
            EXPECT_EQ(got, want) << t.file;
            EXPECT_EQ(t.rows.size(), want.size()) << t.file;
        }
    }
    struct Spot { int n, k, batch; const char* first; };
    const Spot spots[] = {{64, 9, 1, "vendor"}, {64, 9, 2, "fold"},     {120, 30, 1, "vendor"},
                          {136, 30, 1, "fold"}, {3000, 2, 1, "fold"},   {7, 7, 1, "vendor"},
                          {7, 7, 9000, "fold"}};
    for (const Spot& s : spots) {
        const select::Table& t = *select::tables_in_borrow_order("her2k", "cfloat", select::device_from_key("sm_89")).front();
        const select::Key key{{"n", s.n}, {"k", s.k}, {"batch", s.batch}};
        const select::TableRow* row = t.nearest(key);
        ASSERT_NE(row, nullptr);
        EXPECT_EQ(row->ranked.front().spelling, s.first) << t.file << ":" << row->line;
    }
}

}  // namespace
