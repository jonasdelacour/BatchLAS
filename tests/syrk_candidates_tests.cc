// Every syrk candidate, pinned: docs/design/flat-kernel-selection.md §8 and
// docs/design/flat-kernel-selection.md §12. The limit oracle reads the kernels' own constants; which
// kernel ran is read back from the select trace or a bit-for-bit comparison with the direct
// kernel call, never assumed from the pin being accepted.
#include <gtest/gtest.h>

#include <batchlas/blas/functions/syrk.hh>
#include <batchlas/no_route.hh>
#include <batchlas/sycl_interop.hh>
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>
#include "../src/select/coverage.hh"
#include "../src/select/vendor.hh"

#include "test_utils.hh"

#include "../src/backends/syrk_gram_tiles.hh"
#include "../src/backends/syrk_triangular_tiles.hh"
#include "../src/ops/syrk/choice.hh"

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

namespace sk = batchlas::ops::syrk;
using C = sk::SyrkChoice;
using Pin = select::ScopedPin<C>;
template <typename T>
using MVof = MatrixView<T, MatrixFormat::Dense>;

template <typename T>
bool same_bits(T a, T b) {
    return std::memcmp(&a, &b, sizeof(T)) == 0;
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

// A and C live in one buffer of a large finite poison, at padded lds and strides that are not
// ld * cols. C's other triangle, the ld pads and the gaps hold the poison; syrk may write only
// C's `uplo` triangle.
template <typename T>
struct Prob {
    Spec s;
    T alpha = T(1.25), beta = T(-0.75);
    Region a, c;
    UnifiedVector<T> mem;
    std::vector<T> mem0;
    UnifiedVector<T*> aptr, cptr;

    MVof<T> A() { return {mem.data() + a.off, a.rows, a.cols, a.ld, a.stride, s.batch, aptr.data()}; }
    MVof<T> Cv() { return {mem.data() + c.off, s.n, s.n, c.ld, c.stride, s.batch, cptr.data()}; }
    std::size_t ai(int it, int i, int j) const { return a.off + std::size_t(it) * a.stride + std::size_t(j) * a.ld + i; }
    std::size_t ci(int it, int i, int j) const { return c.off + std::size_t(it) * c.stride + std::size_t(j) * c.ld + i; }
    bool ref(int i, int j) const { return s.uplo == Uplo::Lower ? i >= j : i <= j; }
    // op(A)(i, l)
    T opa(const std::vector<T>& m, int it, int i, int l) const {
        return s.trans == Transpose::NoTrans ? m[ai(it, i, l)] : m[ai(it, l, i)];
    }
};

template <typename T>
T poison() {
    return T(4096.5);
}

template <typename T>
Prob<T> make_prob(const Spec& s) {
    Prob<T> p;
    p.s = s;
    const int ar = s.trans == Transpose::NoTrans ? s.n : s.k, ac = s.trans == Transpose::NoTrans ? s.k : s.n;
    p.a = {0, ar, ac, ar + 3, (ar + 3) * ac + 5};
    p.c = {std::size_t(p.a.stride) * s.batch + 11, s.n, s.n, s.n + 2, (s.n + 2) * s.n + 7};
    const std::size_t total = p.c.off + std::size_t(p.c.stride) * s.batch + 13;
    p.mem = UnifiedVector<T>(total, poison<T>());
    p.aptr = UnifiedVector<T*>(s.batch, nullptr);
    p.cptr = UnifiedVector<T*>(s.batch, nullptr);
    std::mt19937 gen(s.seed);
    std::uniform_real_distribution<T> u(T(-1), T(1));
    const int reps = s.period > 0 ? std::min(s.period, s.batch) : s.batch;
    for (int it = 0; it < s.batch; ++it) {
        const int r = it % reps;
        for (int j = 0; j < ac; ++j)
            for (int i = 0; i < ar; ++i) p.mem[p.ai(it, i, j)] = it < reps ? u(gen) : p.mem[p.ai(r, i, j)];
        for (int j = 0; j < s.n; ++j)
            for (int i = 0; i < s.n; ++i)
                if (p.ref(i, j)) p.mem[p.ci(it, i, j)] = it < reps ? u(gen) : p.mem[p.ci(r, i, j)];
    }
    p.mem0.assign(p.mem.begin(), p.mem.end());
    return p;
}

// Max over C's triangle of |C - ref| / (|alpha| sum |op(A)_il op(A)_jl| + |beta C0_ij|), in double.
template <typename T>
double rel_error(const Prob<T>& p, int it) {
    double worst = 0;
    for (int j = 0; j < p.s.n; ++j)
        for (int i = 0; i < p.s.n; ++i) {
            if (!p.ref(i, j)) continue;
            double acc = 0, mag = 0;
            for (int l = 0; l < p.s.k; ++l) {
                const double x = double(p.opa(p.mem0, it, i, l)) * double(p.opa(p.mem0, it, j, l));
                acc += x;
                mag += std::abs(x);
            }
            const double c0 = double(p.mem0[p.ci(it, i, j)]);
            const double want = double(p.alpha) * acc + double(p.beta) * c0;
            const double den = std::abs(double(p.alpha)) * mag + std::abs(double(p.beta) * c0) + 1e-30;
            worst = std::max(worst, std::abs(double(p.mem[p.ci(it, i, j)]) - want) / den);
        }
    return worst;
}

template <typename T>
double tol(int k) {
    return 16.0 * (k + 2) * double(std::numeric_limits<T>::epsilon());
}

// The checked items' triangles against the host reference, every element outside them bit for
// bit (A, the pads, C's other triangle), and for a repeating batch every item's C bit-identical
// to its representative's.
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
                if (p.ref(i, j)) owned[p.ci(it, i, j)] = 1;
    for (std::size_t e = 0; e < p.mem.size(); ++e)
        if (!owned[e] && !same_bits(p.mem[e], p.mem0[e]))
            return ::testing::AssertionFailure() << "wrote outside C's triangle at element " << e;
    if (s.period > 0)
        for (int it = s.period; it < s.batch; ++it)
            for (int j = 0; j < s.n; ++j)
                for (int i = 0; i < s.n; ++i)
                    if (p.ref(i, j) && !same_bits(p.mem[p.ci(it, i, j)], p.mem[p.ci(it % s.period, i, j)]))
                        return ::testing::AssertionFailure() << "item " << it << " differs from its representative";
    return ::testing::AssertionSuccess();
}

template <typename T>
bool untouched(const Prob<T>& p) {
    for (std::size_t e = 0; e < p.mem.size(); ++e)
        if (!same_bits(p.mem[e], p.mem0[e])) return false;
    return true;
}

// The outermost syrk trace line for whatever `run` calls, and the choice it names.
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
        if (line.rfind("syrk ", 0) == 0 && line.find(" -> ") != std::string::npos) return line;
    return "<no syrk trace line in: " + err + ">";
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

// Every embedded table but syrk's, plus `syrk_text` as this device's syrk table.
void use_syrk_table(const std::string& dtype, const std::string& dev, const std::string& rows) {
    std::vector<std::pair<std::string, std::string>> files;
    for (const auto& t : select::embedded_tables())
        if (t.name.rfind("syrk.", 0) != 0) files.emplace_back(std::string(t.name), std::string(t.text));
    files.emplace_back("syrk." + dtype + "." + dev + ".txt", "# op=syrk dtype=" + dtype + " device=" + dev +
                                                                 " kernels=unknown\n"
                                                                 "# keys: form:exact trans:exact n:log:2 k:log "
                                                                 "batch:log\n" +
                                                                 rows);
    select::testing::set_builtin_tables(std::move(files));
}

// A heterogeneous A or C has no route on any backend: one tile launch has one (n, k, ld, stride),
// and
// every vendor loop (cuBLAS, rocBLAS, netlib) runs each item at the top-level extents. A
// spelling pin throws invalid_argument (the vendor class word falls back to Auto), Auto throws
// runtime_error, or NoRouteError without a vendor library, and C is left untouched.
template <Backend B, class T>
void expect_heterogeneous_has_no_route(Queue& ctx) {
    const int n = 16, k = 8, batch = 4;
    Matrix<T, MatrixFormat::Dense> A(n, k, batch), Cm(n, n, batch);
    A.fill(T(0.5));
    Cm.fill(T(1));
    UnifiedVector<int> rows(batch), cols(batch);
    for (int b = 0; b < batch; ++b) rows[b] = n - b, cols[b] = k - (b % 2);
    const auto hetA = A.view().with_active_dims(rows.to_span(), cols.to_span());
    const auto hetC = Cm.view().with_active_dims(rows.to_span(), rows.to_span());
    ASSERT_TRUE(hetA.is_heterogeneous() && hetC.is_heterogeneous());
    struct Case { const char* what; MVof<T> a, c; };
    const Case cases[] = {
        {"A heterogeneous", hetA, Cm.view()}, {"C heterogeneous", A.view(), hetC}, {"both, consistent", hetA, hetC}};
    for (const auto& kc : cases) {
        auto call = [&] {
            (void)syrk<B, T>(ctx, kc.a, kc.c, T(1), T(0), Uplo::Lower, Transpose::NoTrans);
            ctx.wait();
        };
        auto expect_no_route = [&](const std::string& what) {
            if constexpr (select::level3_vendor_available<B>) EXPECT_THROW(call(), std::runtime_error) << what;
            else EXPECT_THROW(call(), batchlas::NoRouteError) << what;
        };
        for (const C& c : sk::candidates<T>()) {
            const Pin pin("syrk", c);
            if (std::holds_alternative<sk::Vendor>(c)) {
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
        const ScopedEnvVar clear("BATCHLAS_SYRK_ROUTE", nullptr);
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
using Types = typename test_utils::backend_types_filtered<Cfg, false>::type;

template <typename Config>
class SyrkCandidates : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    using MV = MVof<T>;
    static constexpr Backend B = Config::BackendVal;
    static constexpr bool kVendor = batchlas::select::level3_vendor_available<B>;
    static constexpr bool kFloat = std::is_same_v<T, float>;

    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        if (this->HasFatalFailure() || ::testing::Test::IsSkipped()) return;
        if (!this->ctx) GTEST_SKIP() << "no queue";
        if (this->ctx->device().type != DeviceType::GPU) GTEST_SKIP() << "the tile kernels are GPU kernels";
    }

    // ---- the limit oracle: the kernels' own constants, not syrk.cc's can_run ----
    // Real C is T to both kernels, so trans never refuses one.
    static bool expect_runs(const C& c, const Spec& s) {
        if (std::holds_alternative<sk::Vendor>(c)) return kVendor;
        if (s.batch > 65535) return false;
        if (std::holds_alternative<sk::Gram>(c)) return s.n <= backend::detail::kGramMaxTile;
        return kFloat && backend::detail::triangular_tile_count(s.n) <= 65535;
    }
    // The Vendor choice spells the class word `vendor`, which falls back to Auto where no vendor
    // can run instead of throwing (§5.3).
    static bool vendor_word_falls_back(const C& c) { return std::holds_alternative<sk::Vendor>(c) && !kVendor; }

    void run(Prob<T>& p) {
        (void)syrk<B, T>(*this->ctx, p.A(), p.Cv(), p.alpha, p.beta, p.s.uplo, p.s.trans);
        this->ctx->wait();
    }
    void run_pinned(const C& c, Prob<T>& p) {
        const Pin pin("syrk", c);
        run(p);
    }
    std::string auto_choice(Prob<T>& p) {
        const ScopedEnvVar clear("BATCHLAS_SYRK_ROUTE", nullptr);
        return traced_choice([&] { run(p); });
    }
    // syrk has no sizing call, so acceptance is a launch: `p` is consumed.
    bool pin_accepted(const C& c, Prob<T>& p) {
        const Pin pin("syrk", c);
        try {
            run(p);
            return true;
        } catch (const std::invalid_argument& e) {
            const std::string w = e.what();
            if (w.find("cannot run this shape") == std::string::npos && w.find("is not a compiled") == std::string::npos)
                throw;
            return false;
        }
    }

    // The family's own kernel, called directly.
    bool direct(const C& c, Prob<T>& p, std::string* why = nullptr) {
        Queue& q = *this->ctx;
        const MV A = p.A(), Cm = p.Cv();
        try {
            if (std::holds_alternative<sk::Gram>(c)) {
                (void)backend::detail::syrk_gram_tiles<T, false>(q, A, Cm, p.alpha, p.beta, p.s.uplo, p.s.trans);
            } else if (std::holds_alternative<sk::Triangular>(c)) {
                if constexpr (kFloat) {
                    (void)backend::detail::syrk_triangular_tiles<float>(q, A, Cm, p.alpha, p.beta, p.s.uplo,
                                                                        p.s.trans);
                } else {
                    if (why) *why = "triangular is a float kernel";
                    return false;
                }
            } else {
                if constexpr (kVendor) {
                    (void)backend::syrk_vendor<B, T>(q, A, Cm, p.alpha, p.beta, p.s.uplo, p.s.trans);
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

TYPED_TEST_SUITE(SyrkCandidates, Types);

// Shapes straddling gram's one-tile ceiling (128|129), its 32/64/128 tile widths, the
// triangular staging depth (k 7|8), partial and whole 128-tiles, both transposes and triangles.
const Spec kStraddle[] = {{1, 1, 2},     {31, 7, 3},     {32, 8, 2},      {33, 53, 2},    {64, 5, 2},
                          {65, 200, 2},  {127, 9, 2},    {128, 64, 2},    {129, 3, 2},    {200, 53, 2},
                          {256, 64, 2},  {257, 8, 2}};

// §8.1: each candidate on the straddle set; accepted shapes are correct and leave everything
// outside C's triangle bit for bit, refused ones throw from the pin and write nothing.
TYPED_TEST(SyrkCandidates, PinnedCandidatesStraddleTheirLimits) {
    using T = typename TestFixture::T;
    for (const C& c : sk::candidates<T>())
        for (Spec s : kStraddle)
            for (Transpose t : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans})
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

// Every (uplo, trans) on every candidate, ConjTrans included: real C is T, so every kernel runs
// it. Auto keeps C on the vendor (the trans=C rows); vendor-free it takes the native kernel.
TYPED_TEST(SyrkCandidates, EveryCombinationOnEveryCandidate) {
    using T = typename TestFixture::T;
    for (const Spec base : {Spec{48, 37, 3}, Spec{200, 53, 2}})
        for (Uplo u : {Uplo::Lower, Uplo::Upper})
            for (Transpose t : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans}) {
                Spec s = base;
                s.uplo = u;
                s.trans = t;
                for (const C& c : sk::candidates<T>()) {
                    if (this->vendor_word_falls_back(c)) continue;
                    auto p = make_prob<T>(s);
                    const bool want = this->expect_runs(c, s);
                    EXPECT_EQ(this->pin_accepted(c, p), want) << name(c, s);
                    if (want) EXPECT_TRUE(correct(p)) << name(c, s);
                }
                auto p = make_prob<T>(s);
                const bool no_native = !TestFixture::kFloat && s.n > 128;
                if (no_native && !TestFixture::kVendor) {
                    EXPECT_THROW(this->run(p), batchlas::NoRouteError) << label(s);
                    continue;
                }
                const std::string got = this->auto_choice(p);
                if (t == Transpose::ConjTrans)
                    EXPECT_EQ(got, TestFixture::kVendor ? "vendor" : (s.n <= 128 ? "gram" : "triangular")) << label(s);
                EXPECT_TRUE(correct(p)) << "auto " << label(s) << " -> " << got;
            }
}

// The pinned facade runs exactly that family's kernel: bit-identical to the direct call on the
// same input. A swapped launch arm shows here even where both kernels are correct.
TYPED_TEST(SyrkCandidates, PinnedRunIsTheDirectKernelBitForBit) {
    using T = typename TestFixture::T;
    for (const C& c : sk::candidates<T>()) {
        if (this->vendor_word_falls_back(c)) continue;
        for (Spec s : {Spec{48, 37, 3, Uplo::Upper, Transpose::Trans}, Spec{100, 9, 2, Uplo::Lower}}) {
            s.seed = 11;
            auto pinned = make_prob<T>(s);
            this->run_pinned(c, pinned);
            auto direct = make_prob<T>(s);
            ASSERT_TRUE(this->direct(c, direct)) << name(c, s);
            for (std::size_t e = 0; e < pinned.mem.size(); ++e)
                ASSERT_TRUE(same_bits(pinned.mem[e], direct.mem[e])) << name(c, s) << " element " << e;
        }
    }
}

// The SLM kernels at a saturating batch: 1024 copies of one problem, each bit-identical to item 0.
TYPED_TEST(SyrkCandidates, SaturatingBatchIsBitIdentical) {
    using T = typename TestFixture::T;
    for (const C& c : sk::candidates<T>()) {
        if (std::holds_alternative<sk::Vendor>(c)) continue;
        for (Spec s : {Spec{48, 70, 1024, Uplo::Lower, Transpose::NoTrans}, Spec{128, 33, 1024, Uplo::Upper,
                                                                                  Transpose::Trans},
                       Spec{200, 40, 1024, Uplo::Upper, Transpose::NoTrans}}) {
            s.period = 1;
            if (!this->expect_runs(c, s)) continue;
            auto p = make_prob<T>(s);
            this->run_pinned(c, p);
            EXPECT_TRUE(correct(p)) << name(c, s);
        }
    }
}

// §8.2 (R3) over the straddle set: a pin is accepted exactly when the family's own kernel,
// called directly, runs and answers correctly. Gram past its one tile does not throw: it answers
// wrongly, which is why its pin now throws there (the old BATCHLAS_SYRK_ROUTE=gram ran it).
// ConjTrans is in the set: real C is T, so it must be accepted wherever T is.
TYPED_TEST(SyrkCandidates, CanRunEqualsLaunch) {
    using T = typename TestFixture::T;
    for (const C& c : sk::candidates<T>()) {
        if (this->vendor_word_falls_back(c)) continue;
        for (Spec s : kStraddle)
            for (Transpose t : {Transpose::Trans, Transpose::ConjTrans}) {
                s.uplo = Uplo::Upper;
                s.trans = t;
                auto p = make_prob<T>(s);
                const bool accepted = this->pin_accepted(c, p);
                auto d = make_prob<T>(s);
                std::string why;
                const bool launched = this->direct(c, d, &why);
                const bool right = launched && correct(d);
                EXPECT_EQ(accepted, right) << name(c, s) << (launched ? "" : " (direct: " + why + ")");
            }
    }
}

// The grid term (AGENTS §8.9, a launch at the ceiling): the tile kernels put the batch in a grid
// dimension capped at 65535, and one past it the direct launch throws ("Number of work-groups
// exceed limit"). A pin at the ceiling launches; one past it is refused, and Auto takes the
// vendor (vendor-free: no route).
TYPED_TEST(SyrkCandidates, GridBatchCeiling) {
    using T = typename TestFixture::T;
    const int ceiling = int(sk::kMaxGridBatch);
    for (const C& c : sk::candidates<T>()) {
        if (std::holds_alternative<sk::Vendor>(c)) continue;
        Spec at{1, 1, ceiling};
        at.period = 1;
        auto p = make_prob<T>(at);
        ASSERT_TRUE(this->pin_accepted(c, p)) << name(c, at);
        EXPECT_TRUE(correct(p)) << name(c, at);
        auto d = make_prob<T>(at);
        EXPECT_TRUE(this->direct(c, d)) << name(c, at) << ": the direct launch at the ceiling";
        Spec past = at;
        past.batch = ceiling + 1;
        auto q = make_prob<T>(past);
        EXPECT_FALSE(this->pin_accepted(c, q)) << name(c, past);
        EXPECT_TRUE(untouched(q)) << name(c, past);
        std::string why;
        auto e = make_prob<T>(past);
        EXPECT_FALSE(this->direct(c, e, &why)) << name(c, past) << ": the direct launch past the ceiling ran";
        EXPECT_NE(why.find("exceed limit"), std::string::npos) << why;
    }
    Spec past{1, 1, ceiling + 1};
    past.period = 1;
    auto p = make_prob<T>(past);
    if constexpr (TestFixture::kVendor) {
        EXPECT_EQ(this->auto_choice(p), "vendor");
        EXPECT_TRUE(correct(p)) << "auto past the grid ceiling";
    } else {
        const ScopedEnvVar clear("BATCHLAS_SYRK_ROUTE", nullptr);
        EXPECT_THROW(this->run(p), batchlas::NoRouteError);
    }
}

// The tile-count term (AGENTS §8.9): triangular puts its T(T+1)/2 tiles (T = ceil(n/128)) in a
// grid dimension capped at 65535. At the last order whose tile count fits, the pin launches and
// writes C's last tile; one past it the direct launch throws, the pin is refused and writes
// nothing, and Auto takes the vendor (vendor-free: no route). k = 8, batch 1, C on the device.
TYPED_TEST(SyrkCandidates, TriangularTileGridCeiling) {
    static constexpr Backend B = TestFixture::B;
    if constexpr (!TestFixture::kFloat) {
        GTEST_SKIP() << "triangular is a float kernel";
    } else {
        int last = 1;
        while (backend::detail::triangular_tile_count(last + 1) <= 65535) ++last;
        auto& q = batchlas::sycl_queue(*this->ctx);
        const std::size_t need = std::size_t(last + 1) * std::size_t(last + 1) * sizeof(float);
        if (q.get_device().template get_info<sycl::info::device::global_mem_size>() < need + (std::size_t(4) << 30))
            GTEST_SKIP() << "needs " << (need >> 30) << " GiB of device memory";
        const int k = 8;
        for (const int n : {last, last + 1}) {
            UnifiedVector<float> a(std::size_t(n) * k);
            for (int l = 0; l < k; ++l)
                for (int i = 0; i < n; ++i) a[std::size_t(l) * n + i] = float((i * 7 + l * 3) % 11 - 5) / 4;
            float* c = sycl::malloc_device<float>(std::size_t(n) * n, q);
            ASSERT_NE(c, nullptr);
            q.fill(c, poison<float>(), std::size_t(n) * n).wait();
            UnifiedVector<float*> ap(1, nullptr), cp(1, nullptr);
            const MVof<float> A(a.data(), n, k, n, n * k, 1, ap.data());
            const MVof<float> Cm(c, n, n, n, n * n, 1, cp.data());
            auto at = [&](int i, int j) {
                float v = 0;
                q.memcpy(&v, c + std::size_t(j) * n + i, sizeof v).wait();
                return v;
            };
            auto want = [&](int i, int j) {  // multiples of 1/16: exact in float
                float s = 0;
                for (int l = 0; l < k; ++l) s += a[std::size_t(l) * n + i] * a[std::size_t(l) * n + j];
                return s;
            };
            auto call = [&] {
                (void)syrk<B, float>(*this->ctx, A, Cm, 1.0f, 0.0f, Uplo::Lower, Transpose::NoTrans);
                this->ctx->wait();
            };
            bool accepted = true;
            try {
                const Pin pin("syrk", C{sk::Triangular{}});
                call();
            } catch (const std::invalid_argument& e) {
                EXPECT_NE(std::string(e.what()).find("cannot run this shape"), std::string::npos) << e.what();
                accepted = false;
            }
            if (n == last) {
                EXPECT_TRUE(accepted) << "n=" << n;
                for (const auto& [i, j] : std::initializer_list<std::pair<int, int>>{
                         {n - 1, n - 1}, {n - 1, 0}, {0, 0}, {n - 1, n - 129}, {n - 200, n - 300}})
                    EXPECT_EQ(at(i, j), want(i, j)) << "n=" << n << " C(" << i << "," << j << ")";
                EXPECT_EQ(at(0, n - 1), poison<float>()) << "wrote the upper triangle";
            } else {
                EXPECT_FALSE(accepted) << "n=" << n;
                EXPECT_EQ(at(n - 1, n - 1), poison<float>()) << "a refused pin wrote";
                try {
                    (void)backend::detail::syrk_triangular_tiles<float>(*this->ctx, A, Cm, 1.0f, 0.0f, Uplo::Lower,
                                                                        Transpose::NoTrans);
                    this->ctx->wait();
                    ADD_FAILURE() << "the direct launch past the ceiling ran, n=" << n;
                } catch (const std::exception& e) {
                    EXPECT_NE(std::string(e.what()).find("exceed limit"), std::string::npos) << e.what();
                }
                const ScopedEnvVar clear("BATCHLAS_SYRK_ROUTE", nullptr);
                if constexpr (TestFixture::kVendor) {
                    EXPECT_EQ(traced_choice(call), "vendor");
                    EXPECT_EQ(at(n - 1, n - 1), want(n - 1, n - 1));
                    EXPECT_EQ(at(n - 1, 0), want(n - 1, 0));
                } else {
                    EXPECT_THROW(call(), batchlas::NoRouteError);
                }
            }
            sycl::free(c, q);
        }
    }
}

// R6: a spelling that names nothing compiled throws instead of meaning Auto, via ScopedPin and
// via the environment. Includes every removed level-3 word and the other ops' family words.
TYPED_TEST(SyrkCandidates, UnknownPinsThrow) {
    using T = typename TestFixture::T;
    for (const char* word : {"cublasdx", "tiles", "narrow", "gemm", "custom", "dx", "fused", "diag_full_gemm",
                             "triangular_tiles", "gram_tiles", "expand_gemm", "fused_device", "register_tiled",
                             "native:auto", "vendor:auto", "bogus", "expand", "cta", "blocked", "direct", "tiled",
                             "gram:1", "triangular:k=8", "native:gram"}) {
        for (bool via_env : {false, true}) {
            auto p = make_prob<T>(Spec{16, 8, 2});
            const ScopedEnvVar env("BATCHLAS_SYRK_ROUTE", via_env ? word : nullptr);
            std::optional<Pin> pin;
            if (!via_env) pin.emplace("syrk", std::string_view(word));
            EXPECT_THROW(this->run(p), std::invalid_argument) << word << (via_env ? " (env)" : " (ScopedPin)");
            EXPECT_TRUE(untouched(p)) << word;
        }
    }
}

// The named can_run-false cases, each with its reason.
TYPED_TEST(SyrkCandidates, CanRunFalsePinsThrow) {
    using T = typename TestFixture::T;
    struct Case { C c; Spec s; const char* msg; };
    // Triangular's float limit (the tile grid) is TriangularTileGridCeiling's: it needs 8.5 GB.
    std::vector<Case> cases{{sk::Gram{}, {129, 8, 2}, "cannot run this shape"},
                            {sk::Gram{}, {4096, 1, 1}, "cannot run this shape"},
                            {sk::Gram{}, {129, 8, 2, Uplo::Upper, Transpose::ConjTrans}, "cannot run this shape"}};
    if constexpr (!TestFixture::kFloat)
        cases.push_back({sk::Triangular{}, {300, 8, 2}, "is not a compiled syrk double candidate"});
    for (const auto& k : cases) {
        auto p = make_prob<T>(k.s);
        const Pin pin("syrk", k.c);
        try {
            this->run(p);
            ADD_FAILURE() << name(k.c, k.s) << " was accepted";
        } catch (const std::invalid_argument& e) {
            EXPECT_NE(std::string(e.what()).find(k.msg), std::string::npos) << name(k.c, k.s) << ": " << e.what();
        }
        EXPECT_TRUE(untouched(p)) << name(k.c, k.s);
    }
}

// A heterogeneous operand on CUDA (expect_heterogeneous_has_no_route).
TYPED_TEST(SyrkCandidates, HeterogeneousBatchHasNoRoute) {
    expect_heterogeneous_has_no_route<TestFixture::B, typename TestFixture::T>(*this->ctx);
}

// §5.3: spellings (case-folded) and the class words, via ScopedPin and via the environment.
// Bare `native` is the row's best runnable non-vendor; bare `vendor` falls back to Auto (with
// the warning) where there is no vendor.
TYPED_TEST(SyrkCandidates, ClassWordsAndSpellings) {
    using T = typename TestFixture::T;
    // float: gram cannot run and the old rule took the vendor (tall, n in 129..256); double: gram.
    const Spec s = TestFixture::kFloat ? Spec{200, 53, 2} : Spec{96, 20, 3};
    std::string auto_pick;
    {
        auto p = make_prob<T>(s);
        auto_pick = this->auto_choice(p);
    }
    std::vector<std::pair<const char*, std::string>> expect{{"vendor", TestFixture::kVendor ? "vendor" : auto_pick},
                                                            {"auto", auto_pick},
                                                            {" Auto ", auto_pick},
                                                            {"native", TestFixture::kFloat ? "triangular" : "gram"}};
    if constexpr (TestFixture::kFloat) expect.push_back({"TRIANGULAR", "triangular"});
    for (const auto& [word, spelling] : expect)
        for (bool via_env : {false, true}) {
            auto p = make_prob<T>(s);
            select::testing::reset_warnings();
            std::string err;
            const std::string got = traced_choice([&] {
                const ScopedEnvVar env("BATCHLAS_SYRK_ROUTE", via_env ? word : nullptr);
                std::optional<Pin> pin;
                if (!via_env) pin.emplace("syrk", std::string_view(word));
                this->run(p);
            }, &err);
            const std::string what = std::string(word) + (via_env ? " via BATCHLAS_SYRK_ROUTE" : " via ScopedPin");
            EXPECT_EQ(got, spelling) << what;
            const bool warns = std::string(word) == "vendor" && !TestFixture::kVendor;
            EXPECT_EQ(err.find("syrk pinned \"vendor\", but no vendor candidate") != std::string::npos, warns)
                << what << ": " << err;
            EXPECT_TRUE(correct(p)) << what;
        }
    // gram is a spelling at n <= 128, and `native` there is gram (the row's first).
    auto p = make_prob<T>(Spec{96, 20, 3});
    EXPECT_EQ(traced_choice([&] {
                  const Pin pin("syrk", std::string_view("Gram"));
                  this->run(p);
              }),
              "gram");
    EXPECT_TRUE(correct(p));
    auto q = make_prob<T>(Spec{96, 20, 3});
    EXPECT_EQ(traced_choice([&] {
                  const Pin pin("syrk", std::string_view("native"));
                  this->run(q);
              }),
              "gram");
}

// §5.3: a ScopedPin wins over BATCHLAS_SYRK_ROUTE, and nested pins restore the outer one.
TYPED_TEST(SyrkCandidates, ScopedPinBeatsTheEnvironment) {
    using T = typename TestFixture::T;
    const ScopedEnvVar env("BATCHLAS_SYRK_ROUTE", "gram");
    {
        auto p = make_prob<T>(Spec{200, 8, 2});
        EXPECT_THROW(this->run(p), std::invalid_argument) << "the environment pin was not read";
    }
    const C other = TestFixture::kFloat ? C{sk::Triangular{}} : C{sk::Vendor{}};
    if (this->vendor_word_falls_back(other)) GTEST_SKIP() << "double has one native family and no vendor here";
    auto p = make_prob<T>(Spec{48, 8, 2});
    EXPECT_EQ(traced_choice([&] {
                  const Pin pin("syrk", other);
                  this->run(p);
              }),
              select::to_string(other));
    EXPECT_TRUE(correct(p));
    const Pin outer("syrk", other);
    {
        const Pin inner("syrk", C{sk::Gram{}});
        auto r = make_prob<T>(Spec{48, 8, 2});
        EXPECT_EQ(traced_choice([&] { this->run(r); }), "gram");
    }
    auto r = make_prob<T>(Spec{48, 8, 2});
    EXPECT_EQ(traced_choice([&] { this->run(r); }), select::to_string(other)) << "the inner pin did not restore";
    EXPECT_TRUE(correct(r));
}

// Auto against the transcribed table, on each side of every threshold the old rule read:
// gram 128|129, triangular-if-squareish above, the batch floor of the 3-tile band (26|27) and of
// the 13-tile band (1|2), n 256|257, k 7|8, all three forms, N and T alike; C is the vendor
// everywhere. Vendor-free, `vendor` is taken by the row's next entry (float: triangular; C at
// n <= 128: gram) or nothing (double).
TYPED_TEST(SyrkCandidates, AutoReadsTheTranscribedTable) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    const auto tables = select::tables_in_borrow_order("syrk", select::dtype_name<T>(), select::device_of<B>(*this->ctx));
    if (tables.empty() || tables.front()->source != "transcribed:ff340fc6")
        GTEST_SKIP() << "this device reads another syrk table";
    struct Row { int n, k, batch; const char* f; const char* d; };  // float, double (vendor present)
    const Row rows[] = {{128, 64, 4, "gram", "gram"},           {1, 4096, 2, "gram", "gram"},
                        {129, 129, 4, "triangular", "vendor"},  {129, 8, 128, "vendor", "vendor"},
                        {200, 150, 2, "triangular", "vendor"},  {200, 500, 8, "vendor", "vendor"},
                        {256, 8, 128, "vendor", "vendor"},      {257, 8, 128, "triangular", "vendor"},
                        {257, 7, 128, "vendor", "vendor"},      {300, 64, 26, "vendor", "vendor"},
                        {300, 64, 27, "triangular", "vendor"},  {300, 700, 26, "vendor", "vendor"},
                        {300, 700, 27, "triangular", "vendor"}, {1600, 64, 1, "vendor", "vendor"},
                        {1600, 64, 2, "triangular", "vendor"}};
    for (const Row& r : rows)
        for (Transpose t : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans}) {
            Spec s{r.n, r.k, r.batch, Uplo::Lower, t};
            std::string want = t == Transpose::ConjTrans ? "vendor" : (TestFixture::kFloat ? r.f : r.d);
            if (want == "vendor" && !TestFixture::kVendor)
                want = t == Transpose::ConjTrans && r.n <= 128 ? "gram" : (TestFixture::kFloat ? "triangular" : "");
            auto p = make_prob<T>(s);
            if (want.empty()) {
                const ScopedEnvVar clear("BATCHLAS_SYRK_ROUTE", nullptr);
                EXPECT_THROW(this->run(p), batchlas::NoRouteError) << label(s);
                continue;
            }
            EXPECT_EQ(this->auto_choice(p), want) << label(s);
            EXPECT_TRUE(correct(p)) << "auto " << label(s);
        }
}

// key_of's every field reaches choose(): a synthetic table whose winner changes with form,
// trans, n, k (op(A)'s inner extent, so Trans reads A.rows) and batch alone. Breaking a field in
// key_of turns exactly its probe red. Double alternates gram with the vendor, so it needs one.
TYPED_TEST(SyrkCandidates, AutoReadsEveryKeyField) {
    using T = typename TestFixture::T;
    if (!TestFixture::kFloat && !TestFixture::kVendor) GTEST_SKIP() << "double's only alternative is the vendor";
    const ScopedEnvVar clear("BATCHLAS_SYRK_ROUTE", nullptr);
    const std::string dtype(select::dtype_name<T>());
    const std::string dev = select::device_of<TestFixture::B>(*this->ctx).key;
    const std::string alt = TestFixture::kFloat ? "triangular" : "vendor";
    const std::string ga = " | gram 1 | " + alt + " 2\n", ag = " | " + alt + " 1 | gram 2\n";
    const TableGuard restore;
    // N: gram only at the base cell. T: alt at the base cell, gram at k=100 and on the wide row.
    // C: one row, at batch 1024, where N and T both say alt.
    use_syrk_table(dtype, dev,
                   "form=sq trans=N n=64 k=64 batch=16" + ga + "form=sq trans=N n=64 k=64 batch=1024" + ag +
                       "form=sq trans=N n=96 k=64 batch=16" + ag + "form=sq trans=N n=64 k=100 batch=16" + ag +
                       "form=tall trans=N n=64 k=16 batch=16" + ag + "form=wide trans=N n=16 k=64 batch=16" + ag +
                       "form=sq trans=T n=64 k=64 batch=16" + ag + "form=sq trans=T n=64 k=100 batch=16" + ga +
                       "form=wide trans=T n=16 k=64 batch=16" + ga + "form=sq trans=C n=64 k=64 batch=1024" + ga);
    struct Probe { int n, k, batch; Transpose t; std::string expect; const char* field; };
    const Probe probes[] = {{64, 64, 16, Transpose::NoTrans, "gram", "base"},
                            {64, 64, 1024, Transpose::NoTrans, alt, "batch"},
                            {96, 64, 16, Transpose::NoTrans, alt, "n"},
                            {64, 100, 16, Transpose::NoTrans, alt, "k (NoTrans: A.cols)"},
                            {64, 100, 16, Transpose::Trans, "gram", "k (Trans: A.rows)"},
                            {64, 16, 16, Transpose::NoTrans, alt, "form tall"},
                            {16, 64, 16, Transpose::Trans, "gram", "form wide"},
                            {64, 64, 16, Transpose::Trans, alt, "trans T"},
                            {64, 64, 1024, Transpose::ConjTrans, "gram", "trans C"}};
    for (const auto& k : probes) {
        auto p = make_prob<T>(Spec{k.n, k.k, k.batch, Uplo::Upper, k.t});
        EXPECT_EQ(traced_choice([&] { this->run(p); }), k.expect) << "the " << k.field << " probe";
        EXPECT_TRUE(correct(p)) << k.field;
    }
}

// The trace line prints syrk's key: form, trans, n = C's order, k = op(A)'s inner extent, batch.
TYPED_TEST(SyrkCandidates, TraceLineCarriesTheKey) {
    using T = typename TestFixture::T;
    for (Transpose t : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans}) {
        auto p = make_prob<T>(Spec{12, 37, 5, Uplo::Upper, t});
        const std::string line = traced_line([&] { this->run(p); });
        const std::string key = std::string("form=wide trans=") + trans_s(t) + " n=12 k=37 batch=5 ->";
        EXPECT_NE(line.find(key), std::string::npos) << line;
        EXPECT_TRUE(correct(p));
    }
}

// With a table naming only the vendor, a vendor-free build takes the last resort: triangular
// for float, gram for double at n <= 128; double past 128 has no route (NoRouteError).
TYPED_TEST(SyrkCandidates, VendorFreeLastResort) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_SYRK_ROUTE", nullptr);
    const std::string dtype(select::dtype_name<T>());
    const std::string dev = select::device_of<TestFixture::B>(*this->ctx).key;
    const TableGuard restore;
    use_syrk_table(dtype, dev, "form=tall trans=N n=200 k=8 batch=128 | vendor 1\n");
    for (const Spec& s : {Spec{200, 9, 3}, Spec{64, 9, 3}}) {
        auto p = make_prob<T>(s);
        std::string want = "vendor";
        if (!TestFixture::kVendor) want = TestFixture::kFloat ? "triangular" : (s.n <= 128 ? "gram" : "");
        if (want.empty()) {
            EXPECT_THROW(this->run(p), batchlas::NoRouteError) << label(s);
            continue;
        }
        std::string all;
        EXPECT_EQ(traced_choice([&] { this->run(p); }, &all), want) << label(s);
        if (!TestFixture::kVendor) EXPECT_NE(all.find("last resort"), std::string::npos) << all;
        EXPECT_TRUE(correct(p)) << label(s);
    }
}

// The coverage rows (§5.6): the real backend and scalar (the old rows hard-coded CUDA/F32), the
// old key (m = n = C's order, k = op(A)'s inner extent, side Left, diag NonUnit, uplo, trans), the
// native flags; and a vendor-free call with no route records a `miss`.
// threadsafe: the child re-executes the binary, so CUDA is initialised fresh, never forked.
TYPED_TEST(SyrkCandidates, CoverageRowCarriesBackendKeyAndNativeFlags) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    const std::string dir = ::testing::TempDir() + "syrk_cov." + std::string(select::dtype_name<T>());
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const std::string out = dir + "/cov";
    const Spec lo{48, 20, 2, Uplo::Lower, Transpose::NoTrans};
    const Spec hi{300, 7, 2, Uplo::Upper, Transpose::Trans};
    const C hi_c = TestFixture::kFloat ? C{sk::Triangular{}} : C{sk::Vendor{}};
    const bool hi_runs = !this->vendor_word_falls_back(hi_c);
    auto child = [&] {
        const ScopedEnvVar cov("BATCHLAS_COVERAGE_OUT", out.c_str());
        const ScopedEnvVar clear("BATCHLAS_SYRK_ROUTE", nullptr);
        batchlas::coverage::g_dynamic_enabled = true;
        {
            const Pin pin("syrk", C{sk::Gram{}});
            auto p = make_prob<T>(lo);
            this->run(p);
        }
        auto p = make_prob<T>(hi);
        try {
            const Pin pin("syrk", hi_runs ? hi_c : C{sk::Gram{}});
            if (hi_runs) this->run(p);
        } catch (...) {
        }
        if (!hi_runs) {
            try {
                this->run(p);  // Auto: double n=300 vendor-free has no route
            } catch (const batchlas::NoRouteError&) {
            }
        }
        std::exit(0);
    };
    EXPECT_EXIT(child(), ::testing::ExitedWithCode(0), "");
    std::map<std::string, std::vector<std::string>> rows;  // "m n k" -> fields
    int misses = 0;
    for (const auto& ent : std::filesystem::directory_iterator(dir)) {
        std::ifstream in(ent.path());
        for (std::string line; std::getline(in, line);) {
            if (line.rfind("miss,syrk,", 0) == 0) ++misses;
            if (line.rfind("reached,syrk,", 0) != 0) continue;
            std::vector<std::string> f;
            std::stringstream ss(line);
            for (std::string tok; std::getline(ss, tok, ',');) f.push_back(tok);
            ASSERT_GE(f.size(), 19u) << line;
            rows[f[5] + " " + f[6] + " " + f[7]] = f;
        }
    }
    std::filesystem::remove_all(dir);
    const std::string backend = B == Backend::CUDA ? "CUDA" : (B == Backend::ROCM ? "ROCM" : "?");
    const std::string scalar = TestFixture::kFloat ? "float" : "double";
    auto as_int = [](auto e) { return std::to_string(static_cast<int>(e)); };
    std::vector<std::tuple<std::string, Spec, std::string>> want{{"48 48 20", lo, "gram"}};
    if (hi_runs) want.push_back({"300 300 7", hi, select::to_string(hi_c)});
    EXPECT_EQ(rows.size(), want.size());
    EXPECT_EQ(misses, hi_runs ? 0 : 1);
    for (const auto& [key, s, algo] : want) {
        ASSERT_TRUE(rows.count(key)) << "no " << key << " row";
        const auto& f = rows[key];
        EXPECT_EQ(f[2], scalar) << key;
        EXPECT_EQ(f[3], backend) << key << ": the backend column";
        EXPECT_EQ(f[8], "2") << key << ": batch";
        EXPECT_EQ(f[9], algo == "vendor" ? "vendor" : "native") << key;
        EXPECT_EQ(f[10], algo) << key;
        EXPECT_EQ(f[12], "1") << key << ": a native family exists";
        EXPECT_EQ(f[13], key == "48 48 20" || TestFixture::kFloat ? "1" : "0") << key << ": native supported";
        EXPECT_EQ(f[15], as_int(s.uplo)) << key;
        EXPECT_EQ(f[16], as_int(Side::Left)) << key;
        EXPECT_EQ(f[17], as_int(Diag::NonUnit)) << key;
        EXPECT_EQ(f[18], as_int(s.trans)) << key;
    }
}

// An empty batch launches nothing, under Auto, every pin and vendor-free.
TYPED_TEST(SyrkCandidates, EmptyBatchIsANoOp) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    UnifiedVector<T> a(64, T(1)), c(64, T(2));
    const MVof<T> A(a.data(), 8, 4, 8, 32, 0), Cm(c.data(), 8, 8, 8, 64, 0);
    std::vector<std::optional<C>> pins{std::nullopt};
    for (const C& k : sk::candidates<T>()) pins.push_back(k);
    for (const auto& k : pins) {
        std::optional<Pin> pin;
        if (k) pin.emplace("syrk", *k);
        EXPECT_NO_THROW(((void)syrk<B, T>(*this->ctx, A, Cm, T(1), T(0), Uplo::Lower, Transpose::NoTrans),
                         this->ctx->wait()))
            << (k ? select::to_string(*k) : std::string("auto"));
    }
    for (std::size_t e = 0; e < c.size(); ++e) ASSERT_EQ(c[e], T(2)) << e;
}

// On a CPU queue no tile family can run: native pins throw and Auto is the vendor (netlib), or
// no route without one.
template <typename Config>
class SyrkCandidatesCpu : public test_utils::BatchLASTest<Config> {};
TYPED_TEST_SUITE(SyrkCandidatesCpu, Types);

TYPED_TEST(SyrkCandidatesCpu, CpuQueueRunsNoNativeFamily) {
    using T = typename TypeParam::ScalarType;
    static constexpr Backend B = TypeParam::BackendVal;
    if (!this->ctx) GTEST_SKIP() << "no queue";
    if (this->ctx->device().type == DeviceType::GPU) GTEST_SKIP() << "a GPU queue";
    const ScopedEnvVar clear("BATCHLAS_SYRK_ROUTE", nullptr);
    const Spec s{24, 9, 2, Uplo::Upper, Transpose::Trans};
    for (const C& c : sk::candidates<T>()) {
        if (std::holds_alternative<sk::Vendor>(c)) continue;
        auto p = make_prob<T>(s);
        const Pin pin("syrk", c);
        EXPECT_THROW(((void)syrk<B, T>(*this->ctx, p.A(), p.Cv(), p.alpha, p.beta, s.uplo, s.trans)),
                     std::invalid_argument)
            << select::to_string(c);
    }
    auto p = make_prob<T>(s);
    auto call = [&] {
        (void)syrk<B, T>(*this->ctx, p.A(), p.Cv(), p.alpha, p.beta, s.uplo, s.trans);
        this->ctx->wait();
    };
    if constexpr (batchlas::select::level3_vendor_available<B>) {
        EXPECT_EQ(traced_choice(call), "vendor");
        EXPECT_TRUE(correct(p));
    } else {
        EXPECT_THROW(call(), batchlas::NoRouteError);
    }
}

// The netlib loop shares the vendor loops' single top-level (n, k), so a heterogeneous batch has
// no route on a CPU queue either.
TYPED_TEST(SyrkCandidatesCpu, HeterogeneousBatchHasNoRoute) {
    if (!this->ctx) GTEST_SKIP() << "no queue";
    if (this->ctx->device().type == DeviceType::GPU) GTEST_SKIP() << "a GPU queue";
    expect_heterogeneous_has_no_route<TypeParam::BackendVal, typename TypeParam::ScalarType>(*this->ctx);
}

// herk's opt-in reads syrk's pin, the word `gram` only (case-folded, trimmed, ScopedPin or env).
TEST(SyrkHerkHook, HerkGramPinnedReadsOnlyTheWordGram) {
    const ScopedEnvVar clear("BATCHLAS_SYRK_ROUTE", nullptr);
    EXPECT_FALSE(sk::herk_gram_pinned());
    for (const char* yes : {"gram", " GRAM ", "Gram"}) {
        const ScopedEnvVar env("BATCHLAS_SYRK_ROUTE", yes);
        EXPECT_TRUE(sk::herk_gram_pinned()) << yes;
    }
    for (const char* no : {"triangular", "vendor", "native", "auto", "bogus", "gram:1"}) {
        const ScopedEnvVar env("BATCHLAS_SYRK_ROUTE", no);
        EXPECT_FALSE(sk::herk_gram_pinned()) << no;
    }
    {
        const Pin pin("syrk", C{sk::Gram{}});
        EXPECT_TRUE(sk::herk_gram_pinned());
    }
    const ScopedEnvVar env("BATCHLAS_SYRK_ROUTE", "gram");
    const Pin pin("syrk", C{sk::Triangular{}});
    EXPECT_FALSE(sk::herk_gram_pinned()) << "the ScopedPin wins over the environment";
}

// The transcription (no GPU): each table holds exactly choice.hh's grid (the full product for N
// and T, the n axis at k = batch = 1 for C), sm_120's rows equal sm_89's, every row untimed under
// source=transcribed:ff340fc6, and the keys line is choice.hh's key_names.
TEST(SyrkTranscribedTable, RowsAreExactlyTheChoiceGridOnBothDevices) {
    std::set<std::string> want;
    for (const char* f : {"sq", "tall", "wide"})
        for (std::string_view t : sk::grid_trans)
            for (int n : sk::grid_n)
                for (int k : sk::grid_k)
                    for (int b : sk::grid_batch)
                        if (t != "C" || (k == 1 && b == 1))
                            want.insert(std::string(f) + " " + std::string(t) + " " + std::to_string(n) + " " +
                                        std::to_string(k) + " " + std::to_string(b));
    std::string keys = "# keys:";
    for (auto k : sk::key_names) keys += " " + std::string(k);
    for (const char* dt : {"float", "double"}) {
        std::map<std::string, std::string> rows_89;
        for (const char* dev : {"sm_89", "sm_120"}) {
            const auto tables = select::tables_in_borrow_order("syrk", dt, select::device_from_key(dev));
            ASSERT_FALSE(tables.empty()) << dt << " " << dev;
            const select::Table& t = *tables.front();
            ASSERT_EQ(t.device, dev) << dt;
            EXPECT_EQ(t.source, "transcribed:ff340fc6") << t.file;
            std::set<std::string> got;
            for (const auto& row : t.rows) {
                ASSERT_EQ(row.keys.size(), 5u) << t.file << ":" << row.line;
                const std::string key =
                    row.keys[0] + " " + row.keys[1] + " " + row.keys[2] + " " + row.keys[3] + " " + row.keys[4];
                got.insert(key);
                EXPECT_FALSE(row.timed) << t.file << ":" << row.line;
                std::string ranked;
                for (const auto& e : row.ranked) ranked += e.spelling + "|";
                if (std::string(dev) == "sm_89") rows_89[key] = ranked;
                else EXPECT_EQ(ranked, rows_89[key]) << dt << " " << key << ": sm_120 differs from sm_89";
            }
            EXPECT_EQ(got, want) << t.file;
            EXPECT_EQ(t.rows.size(), want.size()) << t.file;
        }
        for (const auto& e : select::embedded_tables())
            if (std::string_view(e.name).rfind(std::string("syrk.") + dt + ".", 0) == 0)
                EXPECT_NE(std::string(e.text).find("\n" + keys + "\n"), std::string::npos) << e.name;
    }
    for (const char* dt : {"cfloat", "cdouble"})
        for (const auto& e : select::embedded_tables())
            EXPECT_NE(std::string_view(e.name).rfind(std::string("syrk.") + dt + ".", 0), 0u)
                << e.name << ": syrk is real-only";
}

// Spot rows through Table::nearest, off the grid: the old rule's decision at the snapped cell.
TEST(SyrkTranscribedTable, OffGridRowsHoldTheOldRule) {
    struct Row { const char* dtype; const char* t; int n, k, batch; const char* ranked; };
    const Row rows[] = {{"float", "N", 100, 3000, 7, "gram|triangular|vendor|"},
                        {"float", "T", 140, 150, 9, "triangular|vendor|"},
                        {"float", "N", 140, 30, 9, "vendor|triangular|"},
                        {"float", "T", 400, 50, 20, "triangular|vendor|"},
                        {"float", "N", 400, 50, 12, "vendor|triangular|"},
                        {"float", "N", 3000, 4, 32768, "vendor|triangular|"},
                        {"float", "T", 3000, 9000, 1, "triangular|vendor|"},
                        {"float", "C", 100, 3000, 7, "vendor|gram|triangular|"},
                        {"float", "C", 3000, 9000, 1, "vendor|triangular|"},
                        {"double", "N", 100, 3, 50000, "gram|vendor|"},
                        {"double", "T", 140, 140, 50, "vendor|"},
                        {"double", "C", 100, 3, 50000, "vendor|gram|"},
                        {"double", "C", 140, 140, 50, "vendor|"}};
    for (const Row& r : rows) {
        const auto tables = select::tables_in_borrow_order("syrk", r.dtype, select::device_from_key("sm_89"));
        ASSERT_FALSE(tables.empty());
        const select::Key key{
            {"form", sk::form_of(r.n, r.k)}, {"trans", r.t}, {"n", r.n}, {"k", r.k}, {"batch", r.batch}};
        const select::TableRow* row = tables.front()->nearest(key);
        ASSERT_NE(row, nullptr);
        std::string ranked;
        for (const auto& e : row->ranked) ranked += e.spelling + "|";
        EXPECT_EQ(ranked, r.ranked) << r.dtype << " " << r.t << " n=" << r.n << " k=" << r.k << " batch=" << r.batch
                                    << " (line " << row->line << ")";
    }
}

}  // namespace
