// Every herk candidate, pinned: docs/design/flat-kernel-selection.md §8 and §12 "Hermitian three".
// The limit oracle reads the kernels' own constants; which kernel ran is read back from the select
// trace or a bit-for-bit comparison with the direct launch, never assumed from the pin.
#include <gtest/gtest.h>

#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/functions/herk.hh>
#include <batchlas/no_route.hh>
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>
#include "../src/select/coverage.hh"
#include "../src/select/vendor.hh"

#include "test_utils.hh"

#include "../src/backends/accumulate_hermitian.hh"
#include "../src/backends/syrk_gram_tiles.hh"
#include "../src/expansion_budget.hh"
#include "../src/ops/herk/choice.hh"

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

namespace hk = batchlas::ops::herk;
using C = hk::HerkChoice;
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

// A and C live in one buffer of a large finite poison, at padded lds and strides that are not
// ld * cols. C's other triangle, the pads and the gaps hold the poison, and C's diagonal carries
// an imaginary part (555) that C = C^H says is not an input. herk may write only C's triangle.
template <typename T>
struct Prob {
    using R = typename T::value_type;
    Spec s;
    R alpha = R(1.25), beta = R(-0.75);
    Region a, c;
    UnifiedVector<T> mem;
    std::vector<T> mem0;
    UnifiedVector<T*> aptr, cptr;

    MVof<T> A() { return {mem.data() + a.off, a.rows, a.cols, a.ld, a.stride, s.batch, aptr.data()}; }
    MVof<T> Cv() { return {mem.data() + c.off, s.n, s.n, c.ld, c.stride, s.batch, cptr.data()}; }
    std::size_t ai(int it, int i, int j) const { return a.off + std::size_t(it) * a.stride + std::size_t(j) * a.ld + i; }
    std::size_t ci(int it, int i, int j) const { return c.off + std::size_t(it) * c.stride + std::size_t(j) * c.ld + i; }
    bool ref(int i, int j) const { return s.uplo == Uplo::Lower ? i >= j : i <= j; }
    // op(A)(i, l): A, or A^H for ConjTrans.
    cd opa(int it, int i, int l) const {
        return s.trans == Transpose::NoTrans ? wide(mem0[ai(it, i, l)]) : std::conj(wide(mem0[ai(it, l, i)]));
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
    p.c = {std::size_t(p.a.stride) * s.batch + 11, s.n, s.n, s.n + 2, (s.n + 2) * s.n + 7};
    p.mem = UnifiedVector<T>(p.c.off + std::size_t(p.c.stride) * s.batch + 13, poison<T>());
    p.aptr = UnifiedVector<T*>(s.batch, nullptr);
    p.cptr = UnifiedVector<T*>(s.batch, nullptr);
    std::mt19937 gen(s.seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    auto z = [&] { return T(R(u(gen)), R(u(gen))); };
    const int reps = s.period > 0 ? std::min(s.period, s.batch) : s.batch;
    for (int it = 0; it < s.batch; ++it) {
        const int r = it % reps;
        for (int j = 0; j < ac; ++j)
            for (int i = 0; i < ar; ++i) p.mem[p.ai(it, i, j)] = it < reps ? z() : p.mem[p.ai(r, i, j)];
        for (int j = 0; j < s.n; ++j)
            for (int i = 0; i < s.n; ++i)
                if (p.ref(i, j))
                    p.mem[p.ci(it, i, j)] = it >= reps ? p.mem[p.ci(r, i, j)] : (i == j ? T(R(u(gen)), R(555)) : z());
    }
    p.mem0.assign(p.mem.begin(), p.mem.end());
    return p;
}

// Max over C's triangle of |C - ref| / (|alpha| sum |op(A)_il conj(op(A)_jl)| + |beta C0_ij|), in
// double, with the diagonal's input imaginary part ignored; inf if a diagonal leaves nonzero imag.
template <typename T>
double rel_error(const Prob<T>& p, int it) {
    double worst = 0;
    for (int j = 0; j < p.s.n; ++j)
        for (int i = 0; i < p.s.n; ++i) {
            if (!p.ref(i, j)) continue;
            cd acc = 0;
            double mag = 0;
            for (int l = 0; l < p.s.k; ++l) {
                const cd x = p.opa(it, i, l) * std::conj(p.opa(it, j, l));
                acc += x;
                mag += std::abs(x);
            }
            cd c0 = wide(p.mem0[p.ci(it, i, j)]);
            if (i == j) c0 = cd(c0.real(), 0.0);
            if (p.beta == 0) c0 = 0;  // BLAS: beta = 0 makes C output only
            cd want = double(p.alpha) * acc + double(p.beta) * c0;
            if (i == j) want = cd(want.real(), 0.0);
            const cd got = wide(p.mem[p.ci(it, i, j)]);
            if (i == j && got.imag() != 0.0) return std::numeric_limits<double>::infinity();
            const double den = std::abs(double(p.alpha)) * mag + std::abs(double(p.beta) * c0) + 1e-30;
            worst = worse(worst, std::abs(got - want) / den);
        }
    return worst;
}

template <typename T>
double tol(int k) {
    return 16.0 * (k + 2) * double(std::numeric_limits<typename T::value_type>::epsilon());
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

// The outermost herk trace line for whatever `run` calls (nested gemm lines are indented).
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
        if (line.rfind("herk ", 0) == 0 && line.find(" -> ") != std::string::npos) return line;
    return "<no herk trace line in: " + err + ">";
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

// Every embedded table but herk's, plus `rows` as this device's herk table.
void use_herk_table(const std::string& dtype, const std::string& dev, const std::string& rows) {
    std::vector<std::pair<std::string, std::string>> files;
    for (const auto& t : select::embedded_tables())
        if (t.name.rfind("herk.", 0) != 0) files.emplace_back(std::string(t.name), std::string(t.text));
    files.emplace_back("herk." + dtype + "." + dev + ".txt",
                       "# op=herk dtype=" + dtype + " device=" + dev + " kernels=unknown source=transcribed:abcd1234\n"
                       "# keys: n:log:2 k:log batch:log\n" + rows);
    select::testing::set_builtin_tables(std::move(files));
}

// The old rule (8cf7fd86 cublas.cc herk_vendor, herk_gemm_preferred), restated, unpinned.
std::string old_auto(int n, int batch) { return batch >= 4 && n <= 768 ? "fold" : "vendor"; }

// A heterogeneous A or C has no route on any backend: each native launch has one (n, k, ld,
// stride), and every vendor loop (cuBLAS, netlib) runs each item at the top-level extents. A
// spelling pin throws invalid_argument (the vendor class word falls back to Auto), Auto throws
// runtime_error, or NoRouteError without a vendor library, and C is left untouched.
template <Backend B, class T>
void expect_heterogeneous_has_no_route(Queue& ctx) {
    const int n = 16, k = 8, batch = 4;
    Matrix<T, MatrixFormat::Dense> A(n, k, batch), Cm(n, n, batch);
    A.fill(T(0.5, 0.25));
    Cm.fill(T(1));
    UnifiedVector<int> rows(batch), cols(batch);
    for (int b = 0; b < batch; ++b) rows[b] = n - b, cols[b] = k - (b % 2);
    const auto hetA = A.view().with_active_dims(rows.to_span(), cols.to_span());
    const auto hetC = Cm.view().with_active_dims(rows.to_span(), rows.to_span());
    ASSERT_TRUE(hetA.is_heterogeneous() && hetC.is_heterogeneous());
    struct Case { const char* what; MVof<T> a, c; };
    const Case cases[] = {
        {"A heterogeneous", hetA, Cm.view()}, {"C heterogeneous", A.view(), hetC}, {"both, consistent", hetA, hetC}};
    using R = typename T::value_type;
    for (const auto& kc : cases) {
        auto call = [&] {
            (void)herk<B, T>(ctx, kc.a, kc.c, R(1), R(0), Uplo::Lower, Transpose::NoTrans);
            ctx.wait();
        };
        auto expect_no_route = [&](const std::string& what) {
            if constexpr (select::level3_vendor_available<B>) EXPECT_THROW(call(), std::runtime_error) << what;
            else EXPECT_THROW(call(), batchlas::NoRouteError) << what;
        };
        for (const C& c : hk::candidates<T>()) {
            const Pin pin("herk", c);
            if (std::holds_alternative<hk::Vendor>(c)) {
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
        const ScopedEnvVar clear("BATCHLAS_HERK_ROUTE", nullptr);
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
class HerkCandidates : public test_utils::BatchLASTest<Config> {
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
        if (this->ctx->device().type != DeviceType::GPU) GTEST_SKIP() << "fold and gram are GPU kernels";
    }

    // ---- the limit oracle: the kernels' own constants, not herk.cc's can_run ----
    static bool expect_runs(const C& c, const Spec& s) {
        if (std::holds_alternative<hk::Vendor>(c)) return kVendor;
        if (s.batch > 65535) return false;
        if (std::holds_alternative<hk::Gram>(c)) return s.n <= backend::detail::kGramMaxTile;
        return true;
    }
    static bool vendor_word_falls_back(const C& c) { return std::holds_alternative<hk::Vendor>(c) && !kVendor; }

    void run(Prob<T>& p) {
        (void)herk<B, T>(*this->ctx, p.A(), p.Cv(), p.alpha, p.beta, p.s.uplo, p.s.trans);
        this->ctx->wait();
    }
    void run_pinned(const C& c, Prob<T>& p) {
        const Pin pin("herk", c);
        run(p);
    }
    std::string auto_choice(Prob<T>& p) {
        const ScopedEnvVar clear("BATCHLAS_HERK_ROUTE", nullptr);
        return traced_choice([&] { run(p); });
    }
    static std::string auto_expect(const std::string& first) { return kVendor ? first : "fold"; }
    // herk has no sizing call, so acceptance is a launch: `p` is consumed.
    bool pin_accepted(const C& c, Prob<T>& p) {
        const Pin pin("herk", c);
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

    // The family's own code: Fold = the public gemm into a scratch product at the launcher's ld,
    // then accumulate_hermitian<false>; Gram = syrk_gram_tiles<T, conj>; Vendor = the library loop.
    bool direct(const C& c, const MV& A, const MV& Cm, R alpha, R beta, Uplo uplo, Transpose trans,
                std::string* why = nullptr) {
        Queue& q = *this->ctx;
        try {
            if (std::holds_alternative<hk::Fold>(c)) {
                const int n = Cm.rows(), batch = Cm.batch_size(), ld = backend::detail::expanded_ld<T>(n);
                UnifiedVector<T> scratch(std::size_t(ld) * n * std::max(batch, 1), T(0));
                const MV product(scratch.data(), n, n, ld, ld * n, batch);
                (void)gemm<B, T>(q, A, A, product, T(alpha), T(0), trans,
                                 trans == Transpose::NoTrans ? Transpose::ConjTrans : Transpose::NoTrans,
                                 ComputePrecision::Default);
                q.wait();
                (void)backend::detail::accumulate_hermitian<T, false>(q, Cm, product, beta, uplo);
            } else if (std::holds_alternative<hk::Gram>(c)) {
                (void)backend::detail::syrk_gram_tiles<T, true>(q, A, Cm, T(alpha), T(beta), uplo, trans);
            } else {
                if constexpr (kVendor) {
                    (void)backend::herk_vendor<B, T>(q, A, Cm, alpha, beta, uplo, trans);
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
        return direct(c, p.A(), p.Cv(), p.alpha, p.beta, p.s.uplo, p.s.trans, why);
    }
};

TYPED_TEST_SUITE(HerkCandidates, Types);

// Shapes straddling gram's one-tile ceiling (128|129) and its 32/64/128 tile widths (with the
// complex 8-wide thread tile at 128), the fold's 32-row group, and k past gram's 32-deep chunk.
const Spec kStraddle[] = {{1, 1, 2},    {31, 7, 3},   {32, 8, 2},    {33, 53, 2},  {64, 5, 2},  {65, 200, 2},
                          {127, 9, 2},  {128, 64, 2}, {129, 3, 2},   {200, 53, 2}};

// §8.1: each candidate on the straddle set, both transposes and triangles; accepted shapes are
// correct and leave everything outside C's triangle bit for bit, refused ones write nothing.
TYPED_TEST(HerkCandidates, PinnedCandidatesStraddleTheirLimits) {
    using T = typename TestFixture::T;
    for (const C& c : hk::candidates<T>())
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

// The fold's scratch budget straddled: BATCHLAS_EXPAND_MAX_BYTES at exactly the lease, one short.
TYPED_TEST(HerkCandidates, FoldFitStraddlesTheScratchBudget) {
    using T = typename TestFixture::T;
    for (Spec s : {Spec{33, 9, 3}, Spec{64, 40, 2, Uplo::Upper, Transpose::ConjTrans}}) {
        const std::size_t bytes = backend::detail::expanded_workspace_bytes<T>(*this->ctx, s.n, s.batch);
        for (std::size_t cap : {bytes, bytes - 1}) {
            const ScopedEnvVar budget("BATCHLAS_EXPAND_MAX_BYTES", std::to_string(cap).c_str());
            auto p = make_prob<T>(s);
            const bool ok = this->pin_accepted(C{hk::Fold{}}, p);
            EXPECT_EQ(ok, cap == bytes) << label(s) << " budget " << cap << " of " << bytes;
            if (ok) EXPECT_TRUE(correct(p)) << label(s) << " at the exact budget";
            else EXPECT_TRUE(untouched(p)) << label(s);
        }
    }
}

// Auto on a fold row whose fold cannot fit its scratch takes what the old code took, the vendor
// loop, not the untimed gram (both sides of gram's n <= 128 and of the budget). Vendor-free the
// next native runs: gram at n <= 128, nothing past it.
TYPED_TEST(HerkCandidates, AutoTakesTheVendorWhenTheFoldCannotFit) {
    using T = typename TestFixture::T;
    for (Spec s : {Spec{128, 9, 4}, Spec{129, 9, 4, Uplo::Upper, Transpose::ConjTrans}, Spec{33, 40, 5}}) {
        s.seed = 19;
        const std::size_t bytes = backend::detail::expanded_workspace_bytes<T>(*this->ctx, s.n, s.batch);
        for (std::size_t cap : {bytes, bytes - 1}) {
            const ScopedEnvVar budget("BATCHLAS_EXPAND_MAX_BYTES", std::to_string(cap).c_str());
            auto p = make_prob<T>(s);
            const std::string what = label(s) + " budget " + std::to_string(cap) + " of " + std::to_string(bytes);
            std::string want = cap == bytes ? "fold" : (TestFixture::kVendor ? "vendor" : "gram");
            if (want == "gram" && s.n > backend::detail::kGramMaxTile) {
                const ScopedEnvVar clear("BATCHLAS_HERK_ROUTE", nullptr);
                EXPECT_THROW(this->run(p), batchlas::NoRouteError) << what;
                EXPECT_TRUE(untouched(p)) << what;
                continue;
            }
            EXPECT_EQ(this->auto_choice(p), want) << what;
            EXPECT_TRUE(correct(p)) << what;
        }
    }
}

// Every (uplo, trans) on every candidate, and Auto on the same shapes.
TYPED_TEST(HerkCandidates, EveryCombinationOnEveryCandidate) {
    using T = typename TestFixture::T;
    for (const Spec base : {Spec{48, 37, 3}, Spec{100, 53, 5}, Spec{200, 9, 4}})
        for (Uplo u : {Uplo::Lower, Uplo::Upper})
            for (Transpose t : {Transpose::NoTrans, Transpose::ConjTrans}) {
                Spec s = base;
                s.uplo = u;
                s.trans = t;
                for (const C& c : hk::candidates<T>()) {
                    if (this->vendor_word_falls_back(c)) continue;
                    auto p = make_prob<T>(s);
                    const bool want = this->expect_runs(c, s);
                    EXPECT_EQ(this->pin_accepted(c, p), want) << name(c, s);
                    if (want) EXPECT_TRUE(correct(p)) << name(c, s);
                }
                auto p = make_prob<T>(s);
                EXPECT_EQ(this->auto_choice(p), TestFixture::auto_expect(old_auto(s.n, s.batch))) << label(s);
                EXPECT_TRUE(correct(p)) << "auto " << label(s);
            }
}

// BLAS: beta = 0 makes C output only. NaN in C's triangle (the diagonal too) must not reach the
// result under any family or Auto: 0 * NaN is NaN, so a finite poison could not tell. The fold's
// gemm writes a never-zeroed arena lease at beta = 0, and the native direct and tiled gemms read
// it (known-defects.md #11), so the arena holds NaN bytes first and the inner gemm also runs
// pinned to those two (vendor-free Auto reaches them at n = 8; the vendor tree never does).
TYPED_TEST(HerkCandidates, BetaZeroDoesNotReadC) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;
    const ScopedEnvVar clear("BATCHLAS_HERK_ROUTE", nullptr);
    std::vector<std::optional<C>> pins{std::nullopt};
    for (const C& c : hk::candidates<T>())
        if (!this->vendor_word_falls_back(c)) pins.push_back(c);
    const R nan = std::numeric_limits<R>::quiet_NaN();
    for (const char* gemm_route : {static_cast<const char*>(nullptr), "direct", "tiled"})
    for (const Spec& s : {Spec{40, 9, 5}, Spec{100, 33, 3, Uplo::Upper, Transpose::ConjTrans}, Spec{8, 8, 4}})
        for (const auto& c : pins) {
            const ScopedEnvVar inner("BATCHLAS_GEMM_ROUTE", gemm_route);
            poison_arena(*this->ctx, 4 * backend::detail::expanded_workspace_bytes<T>(*this->ctx, s.n, s.batch));
            auto p = make_prob<T>(s);
            p.beta = R(0);
            for (int it = 0; it < s.batch; ++it)
                for (int j = 0; j < s.n; ++j)
                    for (int i = 0; i < s.n; ++i)
                        if (p.ref(i, j)) p.mem[p.ci(it, i, j)] = T(nan, nan);
            p.mem0.assign(p.mem.begin(), p.mem.end());
            std::optional<Pin> pin;
            if (c) pin.emplace("herk", *c);
            this->run(p);
            EXPECT_TRUE(correct(p)) << (c ? select::to_string(*c) : std::string("auto")) << " " << label(s)
                                    << " gemm " << (gemm_route ? gemm_route : "auto");
        }
}

// The pinned facade runs exactly that family's code: bit-identical to the direct call.
TYPED_TEST(HerkCandidates, PinnedRunIsTheDirectKernelBitForBit) {
    using T = typename TestFixture::T;
    int compared = 0;
    for (const C& c : hk::candidates<T>()) {
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

// The SLM and grid kernels at a saturating batch: 1024 items repeating 7 problems, each item
// bit-identical to its representative.
TYPED_TEST(HerkCandidates, SaturatingBatchIsBitIdentical) {
    using T = typename TestFixture::T;
    for (const C& c : hk::candidates<T>()) {
        if (std::holds_alternative<hk::Vendor>(c)) continue;
        for (Spec s : {Spec{48, 70, 1024, Uplo::Lower, Transpose::NoTrans},
                       Spec{128, 33, 1024, Uplo::Upper, Transpose::ConjTrans}}) {
            s.period = 7;
            if (!this->expect_runs(c, s)) continue;
            auto p = make_prob<T>(s);
            this->run_pinned(c, p);
            EXPECT_TRUE(correct(p)) << name(c, s);
        }
    }
}

// §8.2 (R3) over the straddle set: a pin is accepted exactly when the family's own code, called
// directly, runs and answers correctly. Gram past its one tile does not throw: it answers wrongly
// (n = 129), which is why its pin throws there.
TYPED_TEST(HerkCandidates, CanRunEqualsLaunch) {
    using T = typename TestFixture::T;
    for (const C& c : hk::candidates<T>()) {
        if (this->vendor_word_falls_back(c)) continue;
        for (Spec s : kStraddle)
            for (Transpose t : {Transpose::NoTrans, Transpose::ConjTrans}) {
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

// The grid term (AGENTS §8.9, a launch at the ceiling): the fold and gram put the batch in a
// grid dimension capped at 65535; one past it the direct launch throws. A pin at the ceiling
// launches; one past it is refused, and Auto takes the vendor (vendor-free: no route).
TYPED_TEST(HerkCandidates, GridBatchCeiling) {
    using T = typename TestFixture::T;
    const int ceiling = int(hk::kMaxGridBatch);
    for (const C& c : hk::candidates<T>()) {
        if (std::holds_alternative<hk::Vendor>(c)) continue;
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
        if (std::holds_alternative<hk::Gram>(c)) {
            EXPECT_FALSE(this->direct(c, e, &why)) << name(c, past) << ": the direct launch past the ceiling ran";
        } else {
            // The fold's own kernel, not the gemm in front of it (which picks its own route).
            UnifiedVector<T> product(std::size_t(past.batch) * 2, T(0));
            const MVof<T> prod(product.data(), 1, 1, 2, 2, past.batch);
            try {
                (void)backend::detail::accumulate_hermitian<T, false>(*this->ctx, e.Cv(), prod, e.beta, Uplo::Lower);
                this->ctx->wait();
            } catch (const std::exception& x) {
                why = x.what();
            }
        }
        EXPECT_NE(why.find("exceed limit"), std::string::npos) << name(c, past) << ": " << why;
    }
    Spec past{1, 1, ceiling + 1};
    past.period = 1;
    auto p = make_prob<T>(past);
    if constexpr (TestFixture::kVendor) {
        EXPECT_EQ(this->auto_choice(p), "vendor");
        EXPECT_TRUE(correct(p)) << "auto past the grid ceiling";
    } else {
        const ScopedEnvVar clear("BATCHLAS_HERK_ROUTE", nullptr);
        EXPECT_THROW(this->run(p), batchlas::NoRouteError);
    }
}

// The fold's capacity at batch 65535 (AGENTS §8.9, a launch at the ceiling): accumulate_hermitian's
// padded range overflows an int from n = 169 (expand_grid_fits), inside expansion_fits' n^2 batch
// term (n <= 181). n = 168 must launch and answer; 169 and 181 are refused before the gemm runs,
// and Auto takes the vendor there. Where the memory budget refuses n = 168 first (a 24 GB card),
// the grid term is not the binding one and there is nothing to launch.
TYPED_TEST(HerkCandidates, FoldGridCeiling) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;
    static constexpr Backend B = TestFixture::B;
    Queue& q = *this->ctx;
    const int batch = int(hk::kMaxGridBatch), k = 3;
    ASSERT_TRUE(backend::detail::expand_grid_fits(168, batch));
    ASSERT_FALSE(backend::detail::expand_grid_fits(169, batch));
    const std::size_t budget = q.device().get_property(DeviceProperty::GLOBAL_MEM_SIZE) / 4;
    if (backend::detail::expanded_workspace_bytes<T>(q, 168, batch) > budget)
        GTEST_SKIP() << "the memory budget refuses n = 168 at batch 65535 on this device";
    for (int n : {168, 169, 181}) {
        UnifiedVector<T> a(std::size_t(n) * k * batch), c(std::size_t(n) * n * batch, T(-3, 7));
        for (std::size_t e = 0; e < a.size(); ++e) a[e] = T(R(0.25) * R(e % 7) - R(0.5), R(0.125) * R(e % 5 + e % 3));
        const MVof<T> A(a.data(), n, k, n, n * k, batch), Cm(c.data(), n, n, n, n * n, batch);
        auto call = [&] {
            (void)herk<B, T>(q, A, Cm, R(1.25), R(0), Uplo::Lower, Transpose::NoTrans);
            q.wait();
        };
        auto worst = [&] {
            double w = 0;
            for (int it : {0, batch / 2, batch - 1})
                for (int j = 0; j < n; ++j)
                    for (int i = j; i < n; ++i) {
                        const std::size_t ab = std::size_t(it) * n * k;
                        cd want = 0;
                        for (int l = 0; l < k; ++l) want += wide(a[ab + i + l * n]) * std::conj(wide(a[ab + j + l * n]));
                        want *= 1.25;
                        if (i == j) want = cd(want.real(), 0.0);
                        w = worse(w, std::abs(wide(c[std::size_t(it) * n * n + i + std::size_t(j) * n]) - want));
                    }
            return w;
        };
        const std::string what = "n = " + std::to_string(n) + " batch = 65535";
        if (n == 168) {
            const Pin pin("herk", C{hk::Fold{}});
            ASSERT_NO_THROW(call()) << what << ": inside both terms, the fold must launch";
            EXPECT_LE(worst(), tol<T>(k)) << what;
            continue;
        }
        {
            const Pin pin("herk", C{hk::Fold{}});
            try {
                call();
                ADD_FAILURE() << what << ": the fold was accepted";
            } catch (const std::invalid_argument& e) {
                EXPECT_NE(std::string(e.what()).find("cannot run this shape"), std::string::npos) << what << ": " << e.what();
            }
        }
        EXPECT_TRUE(same_bits(c[0], T(-3, 7))) << what << ": a refused pin wrote C";
        if (n != 169) continue;
        const ScopedEnvVar clear("BATCHLAS_HERK_ROUTE", nullptr);
        if constexpr (TestFixture::kVendor) {
            EXPECT_EQ(traced_choice(call), "vendor") << what;
            EXPECT_LE(worst(), tol<T>(k)) << what << " under Auto";
        } else {
            EXPECT_THROW(call(), batchlas::NoRouteError) << what;
        }
    }
}

// Shape errors throw invalid_argument before choose(), under Auto and every pin, and write nothing.
TYPED_TEST(HerkCandidates, InvalidShapesThrowBeforeChoose) {
    using T = typename TestFixture::T;
    using R = typename T::value_type;
    static constexpr Backend B = TestFixture::B;
    auto p = make_prob<T>(Spec{8, 5, 3});
    std::vector<std::optional<C>> pins{std::nullopt};
    for (const C& c : hk::candidates<T>()) pins.push_back(c);
    struct Bad { const char* what; MVof<T> a, c; Transpose t; };
    const Bad bads[] = {
        {"plain Trans (complex-symmetric)", p.A(), p.Cv(), Transpose::Trans},
        {"C not square", p.A(), MVof<T>(p.mem.data() + p.c.off, 8, 7, p.c.ld, p.c.stride, 3), Transpose::NoTrans},
        {"op(A) rows != n", p.A(), MVof<T>(p.mem.data() + p.c.off, 6, 6, p.c.ld, p.c.stride, 3), Transpose::NoTrans},
        {"batch mismatch", MVof<T>(p.mem.data() + p.a.off, 8, 5, p.a.ld, p.a.stride, 2), p.Cv(), Transpose::NoTrans}};
    for (const auto& bad : bads)
        for (const auto& c : pins) {
            std::optional<Pin> pin;
            if (c) pin.emplace("herk", *c);
            EXPECT_THROW(((void)herk<B, T>(*this->ctx, bad.a, bad.c, R(1), R(1), Uplo::Lower, bad.t)),
                         std::invalid_argument)
                << bad.what << " under " << (c ? select::to_string(*c) : std::string("auto"));
        }
    this->ctx->wait();
    EXPECT_TRUE(untouched(p)) << "a refused call wrote";
}

// R6: a spelling that names nothing compiled throws instead of meaning Auto: the retired
// BATCHLAS_EXPAND_ROUTE words, other ops' families, fields on a fieldless family.
TYPED_TEST(HerkCandidates, UnknownPinsThrow) {
    using T = typename TestFixture::T;
    for (const char* word : {"loop", "expand", "gemm", "triangular", "cublasdx", "gram_tiles", "native:fold",
                             "native:gram", "vendor:auto", "fold:1", "gram:tile=32", "syrk", "bogus", "direct"}) {
        auto p = make_prob<T>(Spec{});
        const Pin pin("herk", std::string_view(word));
        EXPECT_THROW(this->run(p), std::invalid_argument) << word;
        EXPECT_TRUE(untouched(p)) << word;
    }
}

// The named can_run-false cases, each with its message: gram past one tile, the fold over its
// scratch budget.
TYPED_TEST(HerkCandidates, CanRunFalsePinsThrow) {
    using T = typename TestFixture::T;
    struct Case { C choice; Spec s; const char* budget; const char* what; };
    const Case cases[] = {{C{hk::Gram{}}, Spec{129, 8, 2}, nullptr, "gram at n = 129"},
                          {C{hk::Gram{}}, Spec{300, 8, 2, Uplo::Upper, Transpose::ConjTrans}, nullptr, "gram at n = 300"},
                          {C{hk::Fold{}}, Spec{16, 8, 2}, "0", "fold over BATCHLAS_EXPAND_MAX_BYTES"}};
    for (const auto& k : cases) {
        const ScopedEnvVar budget("BATCHLAS_EXPAND_MAX_BYTES", k.budget);
        auto p = make_prob<T>(k.s);
        const Pin pin("herk", k.choice);
        try {
            this->run(p);
            ADD_FAILURE() << k.what << " was accepted";
        } catch (const std::invalid_argument& e) {
            EXPECT_NE(std::string(e.what()).find("cannot run this shape"), std::string::npos) << k.what << ": " << e.what();
        }
        EXPECT_TRUE(untouched(p)) << k.what;
    }
}

TYPED_TEST(HerkCandidates, HeterogeneousBatchHasNoRoute) {
    expect_heterogeneous_has_no_route<TestFixture::B, typename TestFixture::T>(*this->ctx);
}

// An empty batch or an empty C launches nothing under Auto and every pin, vendor-free too.
TYPED_TEST(HerkCandidates, EmptyProblemIsANoOp) {
    using T = typename TestFixture::T;
    using R = typename T::value_type;
    static constexpr Backend B = TestFixture::B;
    UnifiedVector<T> a(256, T(1)), c(256, T(2, -1));
    struct Case { int n, k, batch; };
    const Case cases[] = {{8, 4, 0}, {0, 4, 2}};
    std::vector<std::optional<C>> pins{std::nullopt};
    for (const C& k : hk::candidates<T>()) pins.push_back(k);
    const ScopedEnvVar clear("BATCHLAS_HERK_ROUTE", nullptr);
    for (const auto& k : cases) {
        const MVof<T> A(a.data(), k.n, k.k, std::max(k.n, 1), 64, k.batch);
        const MVof<T> Cm(c.data(), k.n, k.n, std::max(k.n, 1), 64, k.batch);
        for (const auto& p : pins) {
            std::optional<Pin> pin;
            if (p) pin.emplace("herk", *p);
            EXPECT_NO_THROW(((void)herk<B, T>(*this->ctx, A, Cm, R(1.5), R(-0.75), Uplo::Lower, Transpose::NoTrans),
                             this->ctx->wait()))
                << (p ? select::to_string(*p) : std::string("auto")) << " n=" << k.n << " batch=" << k.batch;
        }
    }
    for (std::size_t e = 0; e < c.size(); ++e) ASSERT_EQ(c[e], T(2, -1)) << "an empty problem wrote element " << e;
}

// §5.3: spellings (case-folded) and the class words, via ScopedPin and via the environment.
TYPED_TEST(HerkCandidates, ClassWordsAndSpellings) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_HERK_ROUTE", nullptr);
    const Spec s{20, 24, 2, Uplo::Upper, Transpose::ConjTrans};  // batch < 4: vendor | fold | gram
    std::string auto_pick;
    {
        auto p = make_prob<T>(s);
        auto_pick = this->auto_choice(p);
    }
    EXPECT_EQ(auto_pick, TestFixture::auto_expect("vendor"));
    const std::string vendor_pick = TestFixture::kVendor ? "vendor" : auto_pick;
    const std::pair<const char*, std::string> expect[] = {{"FOLD", "fold"},     {" gram ", "gram"},
                                                          {"Gram", "gram"},     {"native", "fold"},
                                                          {"vendor", vendor_pick}, {"Auto", auto_pick}};
    for (const auto& [word, spelling] : expect)
        for (bool via_env : {false, true}) {
            auto p = make_prob<T>(s);
            select::testing::reset_warnings();
            std::string err;
            const std::string got = traced_choice([&] {
                const ScopedEnvVar env("BATCHLAS_HERK_ROUTE", via_env ? word : nullptr);
                std::optional<Pin> pin;
                if (!via_env) pin.emplace("herk", std::string_view(word));
                this->run(p);
            }, &err);
            const std::string what = std::string("'") + word + (via_env ? "' via BATCHLAS_HERK_ROUTE" : "' via ScopedPin");
            EXPECT_EQ(got, spelling) << what;
            const bool warns = std::string(word) == "vendor" && !TestFixture::kVendor;
            EXPECT_EQ(err.find("herk pinned \"vendor\", but no vendor candidate") != std::string::npos, warns) << what << ": " << err;
            EXPECT_TRUE(correct(p)) << what;
        }
}

// §5.3: a ScopedPin wins over BATCHLAS_HERK_ROUTE, and nested pins restore the outer one.
TYPED_TEST(HerkCandidates, ScopedPinBeatsTheEnvironment) {
    using T = typename TestFixture::T;
    const Spec s{12, 7, 2};
    const ScopedEnvVar env("BATCHLAS_HERK_ROUTE", "bogus");
    {
        auto p = make_prob<T>(s);
        EXPECT_THROW(this->run(p), std::invalid_argument) << "the environment pin was not read";
    }
    auto p = make_prob<T>(s);
    EXPECT_EQ(traced_choice([&] {
                  const Pin pin("herk", C{hk::Gram{}});
                  this->run(p);
              }),
              "gram");
    EXPECT_TRUE(correct(p));
    const Pin outer("herk", std::string_view("bogus"));
    {
        const Pin inner("herk", C{hk::Fold{}});
        auto r = make_prob<T>(s);
        EXPECT_EQ(traced_choice([&] { this->run(r); }), "fold");
    }
    auto r = make_prob<T>(s);
    EXPECT_THROW(this->run(r), std::invalid_argument) << "the inner pin did not restore the outer";
}

// herk's gram is its own family now: syrk's pin and the retired BATCHLAS_EXPAND_ROUTE move
// nothing, and BATCHLAS_HERK_ROUTE=gram is the only way to the gram kernel.
TYPED_TEST(HerkCandidates, OnlyItsOwnRouteVariableIsRead) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_HERK_ROUTE", nullptr);
    for (const Spec& s : {Spec{40, 9, 2}, Spec{40, 9, 8}})
        for (const auto& [var, word] : {std::pair{"BATCHLAS_SYRK_ROUTE", "gram"}, std::pair{"BATCHLAS_EXPAND_ROUTE", "expand"},
                                        std::pair{"BATCHLAS_EXPAND_ROUTE", "loop"}}) {
            const ScopedEnvVar other(var, word);
            auto p = make_prob<T>(s);
            EXPECT_EQ(this->auto_choice(p), TestFixture::auto_expect(old_auto(s.n, s.batch)))
                << var << "=" << word << " " << label(s);
            EXPECT_TRUE(correct(p)) << var << "=" << word;
        }
    const ScopedEnvVar gram("BATCHLAS_HERK_ROUTE", "gram");
    auto p = make_prob<T>(Spec{40, 9, 2});
    EXPECT_EQ(traced_choice([&] { this->run(p); }), "gram");
    EXPECT_TRUE(correct(p));
}

// Auto against the transcribed table, at cells on both sides of every threshold the old rule read
// (batch 3|4, n 768|769) and of gram's tile (n 128|129), on and off the grid.
TYPED_TEST(HerkCandidates, AutoReadsTheTranscribedTable) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    const ScopedEnvVar clear("BATCHLAS_HERK_ROUTE", nullptr);
    const auto tables = select::tables_in_borrow_order("herk", select::dtype_name<T>(), select::device_of<B>(*this->ctx));
    ASSERT_FALSE(tables.empty());
    if (tables.front()->source != "transcribed:8cf7fd86") GTEST_SKIP() << "this device reads a measured herk table";
    struct Cell { int n, k, batch; };
    const Cell cells[] = {{64, 16, 3},  {64, 16, 4},  {128, 9, 3},  {129, 9, 4}, {768, 4, 4}, {769, 4, 4},
                          {700, 3, 5},  {800, 3, 5},  {760, 2, 3},  {5, 70, 9},  {1, 1, 1},   {300, 30, 1000},
                          {900, 2, 64}, {100, 50, 2}, {640, 8, 4}};
    for (const Cell& c : cells)
        for (Transpose t : {Transpose::NoTrans, Transpose::ConjTrans}) {
            Spec s{c.n, c.k, c.batch, Uplo::Lower, t};
            s.seed = 41u + c.n;
            auto p = make_prob<T>(s);
            EXPECT_EQ(traced_choice([&] { this->run(p); }), TestFixture::auto_expect(old_auto(c.n, c.batch))) << label(s);
            EXPECT_TRUE(correct(p)) << "auto " << label(s);
        }
}

// key_of's every field reaches choose(): a synthetic table whose winner changes with n, with k
// (op(A)'s inner extent: A's columns for NoTrans, its rows for ConjTrans) and with batch.
TYPED_TEST(HerkCandidates, AutoReadsEveryKeyField) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_HERK_ROUTE", nullptr);
    const TableGuard restore;
    use_herk_table(std::string(select::dtype_name<T>()), select::device_of<TestFixture::B>(*this->ctx).key,
                   "n=16 k=16 batch=128 | fold - | gram - | vendor -\n"
                   "n=16 k=16 batch=8192 | gram - | fold - | vendor -\n"
                   "n=4 k=16 batch=128 | gram - | fold - | vendor -\n"
                   "n=16 k=2 batch=128 | gram - | fold - | vendor -\n");
    struct Probe { Spec s; const char* expect; const char* field; };
    const Probe probes[] = {{Spec{16, 16, 128}, "fold", "base"},
                            {Spec{16, 16, 8192}, "gram", "batch"},
                            {Spec{4, 16, 128}, "gram", "n"},
                            {Spec{16, 2, 128}, "gram", "k (NoTrans)"},
                            {Spec{16, 2, 128, Uplo::Upper, Transpose::ConjTrans}, "gram", "k (ConjTrans)"},
                            {Spec{16, 16, 128, Uplo::Upper, Transpose::ConjTrans}, "fold", "base (ConjTrans)"}};
    for (const auto& k : probes) {
        Spec s = k.s;
        s.seed = 43u;
        if (s.batch > 4096) s.period = 1;
        auto p = make_prob<T>(s);
        EXPECT_EQ(traced_choice([&] { this->run(p); }), k.expect) << "the " << k.field << " row, " << label(s);
        EXPECT_TRUE(correct(p)) << k.field;
    }
}

// The trace key: C's order, op(A)'s inner extent and the batch.
TYPED_TEST(HerkCandidates, TraceKeyIsNKBatch) {
    using T = typename TestFixture::T;
    for (Transpose t : {Transpose::NoTrans, Transpose::ConjTrans}) {
        Spec s{37, 12, 5, Uplo::Upper, t};
        auto p = make_prob<T>(s);
        const std::string line = traced_line([&] { this->run(p); });
        EXPECT_NE(line.find("n=37 k=12 batch=5 ->"), std::string::npos) << line;
        EXPECT_TRUE(correct(p)) << line;
    }
}

// With a table naming only the vendor, a vendor-free build takes the last resort, fold.
TYPED_TEST(HerkCandidates, VendorFreeLastResortIsFold) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_HERK_ROUTE", nullptr);
    const TableGuard restore;
    use_herk_table(std::string(select::dtype_name<T>()), select::device_of<TestFixture::B>(*this->ctx).key,
                   "n=32 k=32 batch=128 | vendor -\n");
    for (const Spec& s : {Spec{32, 32, 64}, Spec{300, 3, 1, Uplo::Upper, Transpose::ConjTrans}}) {
        auto p = make_prob<T>(s);
        std::string all;
        EXPECT_EQ(traced_choice([&] { this->run(p); }, &all), TestFixture::kVendor ? "vendor" : "fold") << label(s);
        if (!TestFixture::kVendor) EXPECT_NE(all.find("last resort"), std::string::npos) << all;
        EXPECT_TRUE(correct(p)) << "vendor-only table, " << label(s);
    }
}

// The coverage row (§5.6): the real backend and scalar, the key (m = n = C's order, k = op(A)'s
// inner extent, uplo, transA) and the native flags. The old cublas.cc branches recorded no row.
// threadsafe: the child re-executes the binary, so CUDA is initialised fresh, never forked.
TYPED_TEST(HerkCandidates, CoverageRowCarriesBackendKeyAndNativeFlags) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    const std::string dir = ::testing::TempDir() + "herk_cov." + std::string(select::dtype_name<T>());
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const std::string out = dir + "/cov";
    const Spec lo{16, 5, 2, Uplo::Lower, Transpose::NoTrans};
    const Spec hi{40, 7, 2, Uplo::Upper, Transpose::ConjTrans};
    const Spec mid{24, 3, 2, Uplo::Lower, Transpose::ConjTrans};
    auto child = [&] {
        const ScopedEnvVar cov("BATCHLAS_COVERAGE_OUT", out.c_str());
        const ScopedEnvVar clear("BATCHLAS_HERK_ROUTE", nullptr);
        batchlas::coverage::g_dynamic_enabled = true;
        {
            const Pin pin("herk", C{hk::Fold{}});
            auto p = make_prob<T>(lo);
            this->run(p);
        }
        {
            const Pin pin("herk", C{hk::Gram{}});
            auto p = make_prob<T>(mid);
            this->run(p);
        }
        const Pin pin("herk", TestFixture::kVendor ? C{hk::Vendor{}} : C{hk::Fold{}});
        auto p = make_prob<T>(hi);
        this->run(p);
        std::exit(0);
    };
    EXPECT_EXIT(child(), ::testing::ExitedWithCode(0), "");
    std::map<std::string, std::vector<std::string>> rows;
    for (const auto& ent : std::filesystem::directory_iterator(dir)) {
        std::ifstream in(ent.path());
        for (std::string line; std::getline(in, line);) {
            if (line.rfind("reached,herk,", 0) != 0) continue;
            std::vector<std::string> f;
            std::stringstream ss(line);
            for (std::string tok; std::getline(ss, tok, ',');) f.push_back(tok);
            ASSERT_GE(f.size(), 19u) << line;
            rows[f[5] + " " + f[6] + " " + f[7]] = f;
        }
    }
    std::filesystem::remove_all(dir);
    ASSERT_EQ(rows.size(), 3u);
    auto as_int = [](auto e) { return std::to_string(static_cast<int>(e)); };
    const std::string hi_algo = TestFixture::kVendor ? "vendor" : "fold";
    for (const auto& [key, s, algo] : {std::tuple{std::string("16 16 5"), lo, std::string("fold")},
                                       std::tuple{std::string("24 24 3"), mid, std::string("gram")},
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

// On a CPU queue no native family can run: native pins throw, Auto is the vendor (or no route).
template <typename Config>
class HerkCandidatesCpu : public test_utils::BatchLASTest<Config> {};
TYPED_TEST_SUITE(HerkCandidatesCpu, Types);

TYPED_TEST(HerkCandidatesCpu, CpuQueueRunsNoNativeFamily) {
    using T = typename TypeParam::ScalarType;
    static constexpr Backend B = TypeParam::BackendVal;
    if (!this->ctx) GTEST_SKIP() << "no queue";
    if (this->ctx->device().type == DeviceType::GPU) GTEST_SKIP() << "a GPU queue";
    const ScopedEnvVar clear("BATCHLAS_HERK_ROUTE", nullptr);
    const Spec s{8, 3, 4};
    for (const C& c : hk::candidates<T>()) {
        if (std::holds_alternative<hk::Vendor>(c)) continue;
        auto p = make_prob<T>(s);
        const Pin pin("herk", c);
        EXPECT_THROW(((void)herk<B, T>(*this->ctx, p.A(), p.Cv(), p.alpha, p.beta, s.uplo, s.trans)),
                     std::invalid_argument)
            << select::to_string(c);
    }
    auto p = make_prob<T>(s);
    auto call = [&] {
        (void)herk<B, T>(*this->ctx, p.A(), p.Cv(), p.alpha, p.beta, s.uplo, s.trans);
        this->ctx->wait();
    };
    if constexpr (batchlas::select::level3_vendor_available<B>) {
        EXPECT_EQ(traced_choice(call), "vendor");
        EXPECT_TRUE(correct(p));
    } else {
        EXPECT_THROW(call(), batchlas::NoRouteError);
    }
}

TYPED_TEST(HerkCandidatesCpu, HeterogeneousBatchHasNoRoute) {
    if (!this->ctx) GTEST_SKIP() << "no queue";
    if (this->ctx->device().type == DeviceType::GPU) GTEST_SKIP() << "a GPU queue";
    expect_heterogeneous_has_no_route<TypeParam::BackendVal, typename TypeParam::ScalarType>(*this->ctx);
}

// The transcription (no GPU): each table holds exactly choice.hh's grid, sm_89 and sm_120 alike,
// names the transcribed commit and choice.hh's keys; a row is the old Auto choice, then vendor
// (the old code's fallback when the fold could not fit; gram was pin-only), then the remaining
// natives in candidate order (gram only at n <= 128).
TEST(HerkTranscribedTable, HoldsTheChoiceGridAndTheOldRule) {
    std::set<std::string> want;
    for (int n : hk::grid_n)
        for (int k : hk::grid_k)
            for (int b : hk::grid_batch) want.insert(std::to_string(n) + " " + std::to_string(k) + " " + std::to_string(b));
    for (const char* dt : {"cfloat", "cdouble"}) {
        std::map<std::string, std::string> first;
        for (const char* dev : {"sm_89", "sm_120"}) {
            const auto tables = select::tables_in_borrow_order("herk", dt, select::device_from_key(dev));
            ASSERT_FALSE(tables.empty()) << dt << " " << dev;
            const select::Table& t = *tables.front();
            ASSERT_EQ(t.device, dev) << t.file;
            EXPECT_EQ(t.source, "transcribed:8cf7fd86") << t.file;
            ASSERT_EQ(t.keys.size(), hk::key_names.size()) << t.file;
            for (std::size_t i = 0; i < t.keys.size(); ++i) {
                const std::string w = t.keys[i].weight == 1.0 ? "" : ":" + std::to_string(int(t.keys[i].weight));
                EXPECT_EQ(t.keys[i].name + (t.keys[i].log ? ":log" : ":exact") + w, std::string(hk::key_names[i]))
                    << t.file;
            }
            std::set<std::string> got;
            for (const auto& row : t.rows) {
                const std::string k = row.keys[0] + " " + row.keys[1] + " " + row.keys[2];
                got.insert(k);
                EXPECT_FALSE(row.timed) << t.file << ":" << row.line;
                const int n = std::stoi(row.keys[0]), b = std::stoi(row.keys[2]);
                const std::string gram = n <= 128 ? "|gram" : "";
                const std::string expect = (old_auto(n, b) == "fold" ? "fold|vendor" : "vendor|fold") + gram;
                std::string ranked;
                for (const auto& x : row.ranked) ranked += (ranked.empty() ? "" : "|") + x.spelling;
                EXPECT_EQ(ranked, expect) << t.file << ":" << row.line;
                if (std::string(dev) == "sm_89") first[k] = ranked;
                else EXPECT_EQ(ranked, first[k]) << t.file << " " << k << ": sm_120 differs from sm_89";
            }
            EXPECT_EQ(got, want) << t.file;
            EXPECT_EQ(t.rows.size(), want.size()) << t.file;
        }
    }
    struct Spot { int n, k, batch; const char* first; };
    const Spot spots[] = {{64, 9, 3, "vendor"},   {64, 9, 4, "fold"},     {760, 30, 6, "fold"},
                          {780, 30, 6, "vendor"}, {3000, 2, 9000, "vendor"}, {7, 7, 1, "vendor"},
                          {500, 500, 20000, "fold"}};
    for (const Spot& s : spots) {
        const select::Table& t = *select::tables_in_borrow_order("herk", "cdouble", select::device_from_key("sm_120")).front();
        const select::Key key{{"n", s.n}, {"k", s.k}, {"batch", s.batch}};
        const select::TableRow* row = t.nearest(key);
        ASSERT_NE(row, nullptr);
        EXPECT_EQ(row->ranked.front().spelling, s.first) << t.file << ":" << row->line;
    }
}

}  // namespace
