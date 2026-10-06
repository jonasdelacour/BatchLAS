// Every hemm candidate, pinned: docs/design/flat-kernel-selection.md §8 and §12 "Hermitian three".
// Which kernel ran is read back from the select trace or a bit-for-bit comparison with the direct
// call, never assumed from the pin being accepted.
#include <gtest/gtest.h>

#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/functions/hemm.hh>
#include <batchlas/no_route.hh>
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>
#include "../src/select/coverage.hh"
#include "../src/select/vendor.hh"

#include "test_utils.hh"

#include "../src/backends/triangular_expand.hh"
#include "../src/expansion_budget.hh"
#include "../src/ops/hemm/choice.hh"

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

namespace hm = batchlas::ops::hemm;
using C = hm::HemmChoice;
using Pin = select::ScopedPin<C>;
template <typename T>
using MVof = MatrixView<T, MatrixFormat::Dense>;
using cd = std::complex<double>;

// A max that keeps a NaN: std::max(w, NaN) returns w, so a NaN result would pass.
double worse(double w, double x) { return std::isnan(x) || x > w ? x : w; }

template <typename T>
bool same_bits(T a, T b) {
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}
template <typename T>
cd wide(T x) {
    return cd(double(x.real()), double(x.imag()));
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
// ld * cols. A's unreferenced triangle, the pads and the gaps hold the poison, and A's diagonal
// carries an imaginary part (555) that A = A^H says is not an input; hemm may write only C.
template <typename T>
struct Prob {
    Spec s;
    T alpha = T(1.25, -0.75), beta = T(-0.5, 0.25);
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
    // The Hermitian operand A stands for: conjugate mirror, real diagonal.
    cd herm(int it, int i, int j) const {
        if (i == j) return cd(double(mem0[at(a, it, i, i)].real()), 0.0);
        return referenced(i, j) ? wide(mem0[at(a, it, i, j)]) : std::conj(wide(mem0[at(a, it, j, i)]));
    }
};

template <typename T>
T poison() {
    return T(-997.0, 431.0);
}

template <typename T>
Prob<T> make_prob(const Spec& s) {
    using R = typename T::value_type;
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
    auto z = [&] { return T(R(u(gen)), R(u(gen))); };
    const int reps = s.period > 0 ? std::min(s.period, s.batch) : s.batch;
    for (int it = 0; it < s.batch; ++it) {
        const int r = it % reps;
        for (int j = 0; j < k; ++j)
            for (int i = 0; i < k; ++i)
                if (p.referenced(i, j))
                    p.mem[p.at(p.a, it, i, j)] = it >= reps ? p.mem[p.at(p.a, r, i, j)]
                                                            : (i == j ? T(R(u(gen)), R(555)) : z());
        for (int j = 0; j < s.n; ++j)
            for (int i = 0; i < s.m; ++i) {
                p.mem[p.at(p.b, it, i, j)] = it < reps ? z() : p.mem[p.at(p.b, r, i, j)];
                p.mem[p.at(p.c, it, i, j)] = it < reps ? z() : p.mem[p.at(p.c, r, i, j)];
            }
    }
    p.mem0.assign(p.mem.begin(), p.mem.end());
    return p;
}

template <typename T>
double tol(int k) {
    return 64.0 * std::max(k, 1) * double(std::numeric_limits<typename T::value_type>::epsilon());
}

// Each checked item against a double host reference of the Hermitian product, every element
// outside C bit for bit, and for a repeating batch every item bit-identical to its representative.
template <typename T>
void expect_hemm(const Prob<T>& p, const std::string& what) {
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
    const cd alpha = wide(p.alpha), beta = wide(p.beta);
    for (int it : items)
        for (int j = 0; j < s.n; ++j)
            for (int i = 0; i < s.m; ++i) {
                cd acc = 0;
                double mag = 0;
                for (int t = 0; t < k; ++t) {
                    const cd x = s.side == Side::Left ? p.herm(it, i, t) * wide(p.mem0[p.at(p.b, it, t, j)])
                                                      : wide(p.mem0[p.at(p.b, it, i, t)]) * p.herm(it, t, j);
                    acc += x;
                    mag += std::abs(x);
                }
                const cd c0 = p.beta == T(0) ? cd(0) : wide(p.mem0[p.at(p.c, it, i, j)]);  // beta = 0: C is output only
                const cd want = alpha * acc + beta * c0;
                const double bound = tol<T>(k) * (std::abs(alpha) * mag + std::abs(beta * c0)) + 1e-30;
                const cd got = wide(p.mem[p.at(p.c, it, i, j)]);
                ASSERT_TRUE(std::isfinite(std::abs(got)) && std::abs(got - want) <= bound)
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

// The outermost hemm trace line for whatever `run` calls (nested gemm lines are indented).
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
        if (line.rfind("hemm ", 0) == 0 && line.find(" -> ") != std::string::npos) return line;
    return "<no hemm trace line in: " + err + ">";
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

// Every embedded table but hemm's, plus `rows` as this device's hemm table.
void use_hemm_table(const std::string& dtype, const std::string& dev, const std::string& rows) {
    std::vector<std::pair<std::string, std::string>> files;
    for (const auto& t : select::embedded_tables())
        if (t.name.rfind("hemm.", 0) != 0) files.emplace_back(std::string(t.name), std::string(t.text));
    files.emplace_back("hemm." + dtype + "." + dev + ".txt",
                       "# op=hemm dtype=" + dtype + " device=" + dev + " kernels=unknown source=transcribed:abcd1234\n"
                       "# keys: order:log:2 q:log batch:log\n" + rows);
    select::testing::set_builtin_tables(std::move(files));
}

// The old rule (8cf7fd86 cublas.cc hemm_vendor + triangular_expand.hh expansion_preferred),
// restated: batch >= 4 or max(m, n, k) >= 256 took the expansion, everything else the loop.
std::string old_auto(int m, int n, int batch) {
    return batch >= 4 || std::max(m, n) >= 256 ? "expand" : "vendor";
}

// A heterogeneous operand has no route on any backend: expand reads A at its storage order and
// its gemm rejects a heterogeneous B or C; every vendor loop (cuBLAS, netlib) runs each item at
// the top-level extents. A spelling pin throws invalid_argument (the vendor class word falls back
// to Auto), Auto throws runtime_error, or NoRouteError without a vendor library; C is untouched.
template <Backend B, class T>
void expect_heterogeneous_has_no_route(Queue& ctx) {
    const int n = 16, batch = 4;
    auto A = Matrix<T, MatrixFormat::Dense>::Random(n, n, false, batch, 3);
    auto Bm = Matrix<T, MatrixFormat::Dense>::Random(n, n, false, batch, 5);
    Matrix<T, MatrixFormat::Dense> Cm(n, n, batch);
    Cm.fill(T(0.25, -0.5));
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
            (void)hemm<B, T>(ctx, k.a, k.b, k.c, T(1), T(0.5), Side::Left, Uplo::Lower);
            ctx.wait();
        };
        auto expect_no_route = [&](const std::string& what) {
            if constexpr (select::level3_vendor_available<B>) EXPECT_THROW(call(), std::runtime_error) << what;
            else EXPECT_THROW(call(), batchlas::NoRouteError) << what;
        };
        for (const C& c : hm::candidates<T>()) {
            const Pin pin("hemm", c);
            if (std::holds_alternative<hm::Vendor>(c)) expect_no_route(std::string(k.what) + ": vendor pin");
            else EXPECT_THROW(call(), std::invalid_argument) << k.what << ": " << select::to_string(c);
        }
        const ScopedEnvVar clear("BATCHLAS_HEMM_ROUTE", nullptr);
        expect_no_route(std::string(k.what) + ": auto");
    }
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) ASSERT_EQ(Cm(i, j, b), T(0.25, -0.5)) << "a refused call wrote C";
}

template <typename T, Backend B>
struct Cfg {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};
using Types = typename test_utils::backend_types_complex<Cfg>::type;

template <typename Config>
class HemmCandidates : public test_utils::BatchLASTest<Config> {
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
    static bool vendor_word_falls_back(const C& c) { return std::holds_alternative<hm::Vendor>(c) && !kVendor; }
    static bool expect_runs(const C& c) { return std::holds_alternative<hm::Expand>(c) || kVendor; }

    void run(Prob<T>& p) {
        (void)hemm<B, T>(*this->ctx, p.A(), p.B(), p.Cv(), p.alpha, p.beta, p.s.side, p.s.uplo);
        this->ctx->wait();
    }
    void run_pinned(const C& c, Prob<T>& p) {
        const Pin pin("hemm", c);
        run(p);
    }
    bool pin_accepted(const C& c, Prob<T>& p) {
        const Pin pin("hemm", c);
        try {
            run(p);
            return true;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find("cannot run this shape") == std::string::npos) throw;
            return false;
        }
    }
    std::string auto_choice(Prob<T>& p) {
        const ScopedEnvVar clear("BATCHLAS_HEMM_ROUTE", nullptr);
        return traced_choice([&] { run(p); });
    }
    // What Auto takes where the table ranks `first`: a vendor-free build skips the vendor.
    static std::string auto_expect(const std::string& first) { return kVendor ? first : "expand"; }

    // The family's own code: Expand = expand_mirrored<conj> into scratch + the public gemm, with
    // the expanded ld the launcher derives; Vendor = the library loop.
    bool direct(const C& c, const MV& A, const MV& Bm, const MV& Cm, T alpha, T beta, Side side, Uplo uplo,
                std::string* why = nullptr) {
        Queue& q = *this->ctx;
        try {
            if (std::holds_alternative<hm::Expand>(c)) {
                const int k = A.rows(), batch = A.batch_size(), ld = backend::detail::expanded_ld<T>(k);
                UnifiedVector<T> scratch(std::size_t(ld) * k * std::max(batch, 1), T(0));
                const MV full(scratch.data(), k, k, ld, ld * k, batch);
                backend::detail::expand_mirrored<T, true>(q, full, A, uplo).wait();
                if (side == Side::Left) (void)gemm<B, T>(q, full, Bm, Cm, alpha, beta, Transpose::NoTrans, Transpose::NoTrans, ComputePrecision::Default);
                else (void)gemm<B, T>(q, Bm, full, Cm, alpha, beta, Transpose::NoTrans, Transpose::NoTrans, ComputePrecision::Default);
            } else {
                if constexpr (kVendor) {
                    (void)backend::hemm_vendor<B, T>(q, A, Bm, Cm, alpha, beta, side, uplo);
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

TYPED_TEST_SUITE(HemmCandidates, Types);

// §8.1: each candidate on extents straddling the expansion's 32-wide mirror tile (31/32/33, a
// ragged 77) and the old 255|256 threshold, both sides, both triangles, batch 1 and 5. Then the
// scratch budget straddled: BATCHLAS_EXPAND_MAX_BYTES at exactly the lease, and one byte short.
TYPED_TEST(HemmCandidates, PinnedCandidatesStraddleTheirLimits) {
    using T = typename TestFixture::T;
    int ran = 0;
    for (const C& c : hm::candidates<T>())
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
                        expect_hemm(p, what + " (vendor-free: Auto)");
                        continue;
                    }
                    ASSERT_TRUE(this->pin_accepted(c, p)) << what << " was refused";
                    expect_hemm(p, what);
                    ++ran;
                }
    EXPECT_GT(ran, 0);
    for (auto [k, batch] : {std::pair{33, 3}, std::pair{64, 2}}) {
        Spec s{Side::Left, Uplo::Lower, k, 5, batch};
        const std::size_t bytes = backend::detail::expanded_workspace_bytes<T>(*this->ctx, k, batch);
        for (std::size_t cap : {bytes, bytes - 1}) {
            const ScopedEnvVar budget("BATCHLAS_EXPAND_MAX_BYTES", std::to_string(cap).c_str());
            auto p = make_prob<T>(s);
            const bool ok = this->pin_accepted(C{hm::Expand{}}, p);
            EXPECT_EQ(ok, cap == bytes) << label(s) << " budget " << cap << " of " << bytes;
            if (ok) expect_hemm(p, label(s) + " at the exact budget");
        }
    }
}

// Every (side, uplo) at three shapes, rows != cols, complex alpha and beta != 0.
TYPED_TEST(HemmCandidates, EveryCombinationOnEveryCandidate) {
    using T = typename TestFixture::T;
    for (const C& c : hm::candidates<T>()) {
        if (!TestFixture::expect_runs(c)) continue;
        for (auto [m, n] : {std::pair{13, 7}, std::pair{40, 33}, std::pair{70, 129}})
            for (Side side : {Side::Left, Side::Right})
                for (Uplo uplo : {Uplo::Lower, Uplo::Upper}) {
                    Spec s{side, uplo, m, n, 3};
                    s.seed = 77u + m;
                    auto p = make_prob<T>(s);
                    this->run_pinned(c, p);
                    expect_hemm(p, name(c, s));
                    if (::testing::Test::HasFatalFailure()) return;
                }
    }
}

// BLAS: beta = 0 makes C output only. NaN in every element of C must not reach the result under
// any family or Auto (expand's gemm included): 0 * NaN is NaN, so a finite poison could not tell.
// The native direct and tiled gemms read C at beta = 0 (known-defects.md #11), so expand's gemm
// also runs pinned to those two (vendor-free Auto reaches them at 8 x 8; the vendor tree never does).
TYPED_TEST(HemmCandidates, BetaZeroDoesNotReadC) {
    using T = typename TestFixture::T;
    using R = typename T::value_type;
    const ScopedEnvVar clear("BATCHLAS_HEMM_ROUTE", nullptr);
    std::vector<std::optional<C>> pins{std::nullopt};
    for (const C& c : hm::candidates<T>())
        if (!TestFixture::vendor_word_falls_back(c)) pins.push_back(c);
    const R nan = std::numeric_limits<R>::quiet_NaN();
    for (const char* gemm_route : {static_cast<const char*>(nullptr), "direct", "tiled"})
    for (const Spec& s : {Spec{Side::Left, Uplo::Lower, 40, 9, 5}, Spec{Side::Right, Uplo::Upper, 7, 100, 2},
                          Spec{Side::Left, Uplo::Upper, 8, 8, 4}})
        for (const auto& c : pins) {
            const ScopedEnvVar inner("BATCHLAS_GEMM_ROUTE", gemm_route);
            auto p = make_prob<T>(s);
            p.beta = T(0);
            for (int it = 0; it < s.batch; ++it)
                for (int j = 0; j < s.n; ++j)
                    for (int i = 0; i < s.m; ++i) p.mem[p.at(p.c, it, i, j)] = T(nan, nan);
            p.mem0.assign(p.mem.begin(), p.mem.end());
            std::optional<Pin> pin;
            if (c) pin.emplace("hemm", *c);
            this->run(p);
            expect_hemm(p, (c ? select::to_string(*c) : std::string("auto")) + " beta = 0 " + label(s) + " gemm " +
                               (gemm_route ? gemm_route : "auto"));
        }
}

// The pinned facade runs exactly that family's code: bit-identical to the direct call.
TYPED_TEST(HemmCandidates, PinnedRunIsTheDirectKernelBitForBit) {
    using T = typename TestFixture::T;
    int compared = 0;
    for (const C& c : hm::candidates<T>()) {
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
                expect_hemm(pinned, what);
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
TYPED_TEST(HemmCandidates, SaturatingBatchIsBitIdenticalToItsRepresentative) {
    using T = typename TestFixture::T;
    for (const C& c : hm::candidates<T>()) {
        if (!TestFixture::expect_runs(c)) continue;
        for (auto [side, m, n] : {std::tuple{Side::Left, 40, 9}, std::tuple{Side::Right, 6, 70}}) {
            Spec s{side, Uplo::Lower, m, n, 1024};
            s.period = 7;
            s.seed = 777u + m;
            auto p = make_prob<T>(s);
            this->run_pinned(c, p);
            expect_hemm(p, name(c, s));
            if (::testing::Test::HasFatalFailure()) return;
        }
    }
}

// §8.2 (R3): a pin is accepted exactly when the family's own code launches, over shapes, and a
// heterogeneous B/C (the gemm under expand rejects it).
TYPED_TEST(HemmCandidates, CanRunEqualsLaunch) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    int disagreements = 0;
    for (const C& c : hm::candidates<T>()) {
        if (TestFixture::vendor_word_falls_back(c)) continue;
        for (Side side : {Side::Left, Side::Right})
            for (auto [m, n] : {std::pair{1, 5}, std::pair{33, 2}, std::pair{64, 64}}) {
                Spec s{side, Uplo::Lower, m, n, 2};
                auto a = make_prob<T>(s);
                auto b = make_prob<T>(s);
                const bool pin = this->pin_accepted(c, a);
                std::string why;
                const bool run = this->direct(c, b, &why);
                EXPECT_EQ(pin, run) << name(c, s) << ": can_run says " << pin << ", the code "
                                    << (run ? "launches" : "refuses: " + why);
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
    EXPECT_FALSE(this->direct(C{hm::Expand{}}, A.view(), hB, hC, T(1), T(0), Side::Left, Uplo::Lower, &why))
        << "expand ran a heterogeneous B/C";
    const Pin pin("hemm", C{hm::Expand{}});
    EXPECT_THROW(((void)hemm<B, T>(*this->ctx, A.view(), hB, hC, T(1), T(0), Side::Left, Uplo::Lower)),
                 std::invalid_argument);
    EXPECT_EQ(disagreements, 0);
}

// expand_mirrored puts the batch in grid z: 65535 items launch, 65536 do not, so the pin refuses
// 65536 and Auto takes the vendor (vendor-free: no route).
TYPED_TEST(HemmCandidates, GridBatchCeiling) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    const int ceiling = int(hm::kMaxGridBatch);
    for (int batch : {ceiling, ceiling + 1}) {
        Matrix<T, MatrixFormat::Dense> A(1, 1, batch), Bm(1, 2, batch), Cm(1, 2, batch);
        A.fill(T(2, 9)), Bm.fill(T(3, 1)), Cm.fill(T(1));
        auto call = [&] {
            (void)hemm<B, T>(*this->ctx, A.view(), Bm.view(), Cm.view(), T(1), T(1), Side::Left, Uplo::Lower);
            this->ctx->wait();
        };
        std::string why;
        Matrix<T, MatrixFormat::Dense> Cd(1, 2, batch);
        Cd.fill(T(1));
        const bool launches = this->direct(C{hm::Expand{}}, A.view(), Bm.view(), Cd.view(), T(1), T(1),
                                           Side::Left, Uplo::Lower, &why);
        EXPECT_EQ(launches, batch <= ceiling) << "direct expand at batch " << batch << ": " << why;
        // A = 2 (the 9i is the diagonal's ignored imaginary part): 2 * (3 + i) + 1 = 7 + 2i.
        const T want(7, 2);
        {
            const Pin pin("hemm", C{hm::Expand{}});
            if (batch <= ceiling) {
                call();
                for (int it : {0, batch / 2, batch - 1}) EXPECT_EQ(Cm(0, 1, it), want) << "item " << it;
                continue;
            }
            try {
                call();
                ADD_FAILURE() << "expand accepted batch " << batch;
            } catch (const std::invalid_argument& e) {
                EXPECT_NE(std::string(e.what()).find("cannot run this shape"), std::string::npos) << e.what();
            }
        }
        const ScopedEnvVar clear("BATCHLAS_HEMM_ROUTE", nullptr);
        if constexpr (TestFixture::kVendor) {
            EXPECT_EQ(traced_choice(call), "vendor");
            EXPECT_EQ(Cm(0, 1, batch - 1), want);
        } else {
            EXPECT_THROW(call(), batchlas::NoRouteError);
        }
    }
}

// expansion_fits' element term at batch 65535 (AGENTS §8.9, a launch at the ceiling): order 181
// is the last with order^2 batch <= INT_MAX. expand_mirrored's tiled range stays far below an int
// there (it is not the padded range herk's fold and trmm's expand hit from 169), so order 181
// launches and answers and 182 is refused. Skipped where the memory budget refuses 181 first.
TYPED_TEST(HemmCandidates, ExpandElementCeiling) {
    using T = typename TestFixture::T;
    using R = typename T::value_type;
    static constexpr Backend B = TestFixture::B;
    Queue& q = *this->ctx;
    const int batch = int(hm::kMaxGridBatch);
    const std::size_t budget = q.device().get_property(DeviceProperty::GLOBAL_MEM_SIZE) / 4;
    if (backend::detail::expanded_workspace_bytes<T>(q, 181, batch) > budget)
        GTEST_SKIP() << "the memory budget refuses order 181 at batch 65535 on this device";
    const T alpha(1.25, -0.5);
    for (int k : {181, 182}) {
        UnifiedVector<T> a(std::size_t(k) * k * batch), b(std::size_t(k) * batch), c(std::size_t(k) * batch, T(-3, 7));
        for (std::size_t e = 0; e < a.size(); ++e) a[e] = T(R(0.25) * R(e % 7) - R(0.5), R(0.125) * R(e % 5 + e % 3));
        for (std::size_t e = 0; e < b.size(); ++e) b[e] = T(R(0.5) - R(0.125) * R(e % 9), R(0.25) * R(e % 4) - R(0.3));
        const MVof<T> A(a.data(), k, k, k, k * k, batch), Bm(b.data(), k, 1, k, k, batch), Cm(c.data(), k, 1, k, k, batch);
        const Pin pin("hemm", C{hm::Expand{}});
        auto call = [&] {
            (void)hemm<B, T>(q, A, Bm, Cm, alpha, T(0), Side::Left, Uplo::Upper);
            q.wait();
        };
        if (k == 182) {
            try {
                call();
                ADD_FAILURE() << "order 182 batch 65535: expand was accepted";
            } catch (const std::invalid_argument& e) {
                EXPECT_NE(std::string(e.what()).find("cannot run this shape"), std::string::npos) << e.what();
            }
            EXPECT_TRUE(same_bits(c[0], T(-3, 7))) << "a refused pin wrote C";
            continue;
        }
        ASSERT_NO_THROW(call()) << "order 181 batch 65535: inside the element term, expand must launch";
        double worst = 0;
        for (int it : {0, batch / 2, batch - 1})
            for (int i = 0; i < k; ++i) {
                const std::size_t ab = std::size_t(it) * k * k;
                cd want = 0;
                for (int t = 0; t < k; ++t) {
                    const cd h = i == t ? cd(double(a[ab + i + std::size_t(i) * k].real()), 0.0)
                                 : i < t ? wide(a[ab + i + std::size_t(t) * k])
                                         : std::conj(wide(a[ab + t + std::size_t(i) * k]));
                    want += h * wide(b[std::size_t(it) * k + t]);
                }
                want *= wide(alpha);
                worst = worse(worst, std::abs(wide(c[std::size_t(it) * k + i]) - want) / (std::abs(want) + 1.0));
            }
        EXPECT_LE(worst, tol<T>(k)) << "order 181 at the element ceiling";
    }
}

// Shape and batch errors throw invalid_argument before choose(), under Auto and every pin, and
// write nothing.
TYPED_TEST(HemmCandidates, InvalidShapesThrowBeforeChoose) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    auto p = make_prob<T>(Spec{Side::Left, Uplo::Lower, 8, 5, 3});
    std::vector<std::optional<C>> pins{std::nullopt};
    for (const C& c : hm::candidates<T>()) pins.push_back(c);
    struct Bad { const char* what; MVof<T> a, b, c; Side side; };
    const Bad bads[] = {
        {"A not square", MVof<T>(p.mem.data() + p.a.off, 8, 7, p.a.ld, p.a.stride, 3), p.B(), p.Cv(), Side::Left},
        {"order is n on the right", p.A(), p.B(), p.Cv(), Side::Right},
        {"batch mismatch", MVof<T>(p.mem.data() + p.a.off, 8, 8, p.a.ld, p.a.stride, 2), p.B(), p.Cv(), Side::Left}};
    for (const auto& bad : bads)
        for (const auto& c : pins) {
            std::optional<Pin> pin;
            if (c) pin.emplace("hemm", *c);
            EXPECT_THROW(((void)hemm<B, T>(*this->ctx, bad.a, bad.b, bad.c, T(1), T(1), bad.side, Uplo::Lower)),
                         std::invalid_argument)
                << bad.what << " under " << (c ? select::to_string(*c) : std::string("auto"));
        }
    this->ctx->wait();
    for (std::size_t e = 0; e < p.mem.size(); ++e)
        ASSERT_TRUE(same_bits(p.mem[e], p.mem0[e])) << "a refused call wrote element " << e;
}

// R6: a spelling that names nothing compiled throws instead of meaning Auto: the retired
// BATCHLAS_EXPAND_ROUTE words, other ops' families, fields on a fieldless family.
TYPED_TEST(HemmCandidates, UnknownPinsThrow) {
    using T = typename TestFixture::T;
    for (const char* word : {"loop", "gemm", "fold", "gram", "triangular", "cublasdx", "native:expand",
                             "vendor:auto", "native:auto", "expand:1", "expand:tile=32", "symm", "bogus",
                             "direct", "tiled", "cta", "blocked"}) {
        auto p = make_prob<T>(Spec{});
        const Pin pin("hemm", std::string_view(word));
        EXPECT_THROW(this->run(p), std::invalid_argument) << word;
    }
}

// The named can_run-false cases for expand, each with its message.
TYPED_TEST(HemmCandidates, CanRunFalsePinsThrow) {
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
        const Pin pin("hemm", C{hm::Expand{}});
        try {
            (void)hemm<B, T>(*this->ctx, k.a, k.b, k.c, T(1), T(0), Side::Left, Uplo::Lower);
            this->ctx->wait();
            ADD_FAILURE() << k.what << " was accepted";
        } catch (const std::invalid_argument& e) {
            EXPECT_NE(std::string(e.what()).find("cannot run this shape"), std::string::npos) << k.what << ": " << e.what();
        }
    }
}

TYPED_TEST(HemmCandidates, HeterogeneousBatchHasNoRoute) {
    expect_heterogeneous_has_no_route<TestFixture::B, typename TestFixture::T>(*this->ctx);
}

// An empty batch or an empty C is a no-op under Auto and every pin, vendor-free too.
TYPED_TEST(HemmCandidates, EmptyProblemIsANoOp) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    UnifiedVector<T> a(256, T(1)), b(256, T(3)), c(256, T(2, -1));
    struct Case { int m, n, batch; Side side; };
    const Case cases[] = {{8, 6, 0, Side::Left}, {8, 6, 0, Side::Right}, {8, 0, 2, Side::Left}, {0, 6, 2, Side::Left},
                          {8, 0, 2, Side::Right}};
    std::vector<std::optional<C>> pins{std::nullopt};
    for (const C& k : hm::candidates<T>()) pins.push_back(k);
    const ScopedEnvVar clear("BATCHLAS_HEMM_ROUTE", nullptr);
    for (const auto& k : cases) {
        const int ka = k.side == Side::Left ? k.m : k.n;
        const MVof<T> A(a.data(), ka, ka, std::max(ka, 1), 64, k.batch);
        const MVof<T> Bm(b.data(), k.m, k.n, std::max(k.m, 1), 64, k.batch);
        const MVof<T> Cm(c.data(), k.m, k.n, std::max(k.m, 1), 64, k.batch);
        for (const auto& p : pins) {
            std::optional<Pin> pin;
            if (p) pin.emplace("hemm", *p);
            EXPECT_NO_THROW(((void)hemm<B, T>(*this->ctx, A, Bm, Cm, T(1.5), T(-0.75), k.side, Uplo::Lower),
                             this->ctx->wait()))
                << (p ? select::to_string(*p) : std::string("auto")) << " m=" << k.m << " n=" << k.n
                << " batch=" << k.batch << " " << side_s(k.side);
        }
    }
    for (std::size_t e = 0; e < c.size(); ++e) ASSERT_EQ(c[e], T(2, -1)) << "an empty problem wrote element " << e;
}

// §5.3: spellings (case-folded) and the class words, via ScopedPin and via the environment.
TYPED_TEST(HemmCandidates, ClassWordsAndSpellings) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_HEMM_ROUTE", nullptr);
    const Spec s{Side::Right, Uplo::Upper, 20, 24, 2};  // batch < 4, max < 256: vendor-ranked
    std::string auto_pick;
    {
        auto p = make_prob<T>(s);
        auto_pick = traced_choice([&] { this->run(p); });
    }
    EXPECT_EQ(auto_pick, TestFixture::auto_expect("vendor"));
    const std::string vendor_pick = TestFixture::kVendor ? "vendor" : auto_pick;
    const std::pair<const char*, std::string> expect[] = {
        {"EXPAND", "expand"}, {" expand ", "expand"}, {"native", "expand"}, {"vendor", vendor_pick}, {"Auto", auto_pick}};
    for (const auto& [word, spelling] : expect)
        for (bool via_env : {false, true}) {
            auto p = make_prob<T>(s);
            select::testing::reset_warnings();
            std::string err;
            const std::string got = traced_choice([&] {
                const ScopedEnvVar env("BATCHLAS_HEMM_ROUTE", via_env ? word : nullptr);
                std::optional<Pin> pin;
                if (!via_env) pin.emplace("hemm", std::string_view(word));
                this->run(p);
            }, &err);
            const std::string what = std::string("'") + word + (via_env ? "' via BATCHLAS_HEMM_ROUTE" : "' via ScopedPin");
            EXPECT_EQ(got, spelling) << what;
            const bool warns = std::string(word) == "vendor" && !TestFixture::kVendor;
            EXPECT_EQ(err.find("hemm pinned \"vendor\", but no vendor candidate") != std::string::npos, warns) << what << ": " << err;
            expect_hemm(p, what);
        }
}

// §5.3: a ScopedPin wins over BATCHLAS_HEMM_ROUTE, and nested pins restore the outer one.
TYPED_TEST(HemmCandidates, ScopedPinBeatsTheEnvironment) {
    using T = typename TestFixture::T;
    const Spec s{Side::Left, Uplo::Lower, 12, 7, 2};
    const ScopedEnvVar env("BATCHLAS_HEMM_ROUTE", "bogus");
    {
        auto p = make_prob<T>(s);
        EXPECT_THROW(this->run(p), std::invalid_argument) << "the environment pin was not read";
    }
    auto p = make_prob<T>(s);
    EXPECT_EQ(traced_choice([&] {
                  const Pin pin("hemm", C{hm::Expand{}});
                  this->run(p);
              }),
              "expand");
    expect_hemm(p, "expand over env bogus");
    const Pin outer("hemm", std::string_view("bogus"));
    {
        const Pin inner("hemm", C{hm::Expand{}});
        auto r = make_prob<T>(s);
        EXPECT_EQ(traced_choice([&] { this->run(r); }), "expand");
    }
    auto r = make_prob<T>(s);
    EXPECT_THROW(this->run(r), std::invalid_argument) << "the inner pin did not restore the outer";
}

// BATCHLAS_EXPAND_ROUTE is retired: neither word moves Auto off the table's choice.
TYPED_TEST(HemmCandidates, ExpandRouteIsNotRead) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_HEMM_ROUTE", nullptr);
    for (const Spec& s : {Spec{Side::Left, Uplo::Lower, 20, 24, 2}, Spec{Side::Right, Uplo::Upper, 20, 24, 8}})
        for (const char* word : {"expand", "loop"}) {
            const ScopedEnvVar retired("BATCHLAS_EXPAND_ROUTE", word);
            auto p = make_prob<T>(s);
            EXPECT_EQ(traced_choice([&] { this->run(p); }), TestFixture::auto_expect(old_auto(s.m, s.n, s.batch)))
                << word << " " << label(s);
            expect_hemm(p, std::string("BATCHLAS_EXPAND_ROUTE=") + word);
        }
}

// Auto against the transcribed table, at cells on both sides of every threshold the old rule
// read (batch 3|4, max(m, n) 255|256), on and off the grid.
TYPED_TEST(HemmCandidates, AutoReadsTheTranscribedTable) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    const ScopedEnvVar clear("BATCHLAS_HEMM_ROUTE", nullptr);
    const auto tables = select::tables_in_borrow_order("hemm", select::dtype_name<T>(), select::device_of<B>(*this->ctx));
    ASSERT_FALSE(tables.empty());
    if (tables.front()->source != "transcribed:8cf7fd86") GTEST_SKIP() << "this device reads a measured hemm table";
    struct Cell { int m, n, batch; };
    const Cell cells[] = {{64, 64, 3},   {64, 64, 4},   {255, 255, 1}, {256, 256, 1}, {128, 255, 3}, {129, 256, 1},
                          {256, 100, 2}, {100, 256, 2}, {300, 200, 2}, {200, 150, 5}, {200, 150, 3}, {40, 19, 1000},
                          {19, 40, 1000}, {5, 9, 4},    {9, 5, 3},     {1, 1, 4},     {600, 290, 1}, {3, 600, 2},
                          {254, 7, 3},   {7, 257, 1}};
    for (const Cell& c : cells)
        for (Side side : {Side::Left, Side::Right}) {
            Spec s{side, Uplo::Lower, c.m, c.n, c.batch};
            s.seed = 41u + c.m;
            auto p = make_prob<T>(s);
            EXPECT_EQ(traced_choice([&] { this->run(p); }), TestFixture::auto_expect(old_auto(c.m, c.n, c.batch)))
                << label(s);
            expect_hemm(p, "auto " + label(s));
        }
}

// key_of's every field reaches choose(): a synthetic table whose winner changes with batch, with
// order (A's order, not C's rows: a Right call swaps them) and with q.
TYPED_TEST(HemmCandidates, AutoReadsEveryKeyField) {
    using T = typename TestFixture::T;
    if (!TestFixture::kVendor) GTEST_SKIP() << "only expand can run vendor-free, so no row can steer Auto";
    const ScopedEnvVar clear("BATCHLAS_HEMM_ROUTE", nullptr);
    const TableGuard restore;
    use_hemm_table(std::string(select::dtype_name<T>()), select::device_of<TestFixture::B>(*this->ctx).key,
                   "order=64 q=64 batch=128 | vendor - | expand -\n"
                   "order=64 q=64 batch=32768 | expand - | vendor -\n"
                   "order=16 q=64 batch=128 | expand - | vendor -\n"
                   "order=64 q=16 batch=128 | vendor - | expand -\n");
    struct Probe { Side side; int m, n, batch; const char* expect; const char* field; };
    const Probe probes[] = {{Side::Left, 64, 64, 128, "vendor", "base"},
                            {Side::Left, 64, 64, 32768, "expand", "batch"},
                            {Side::Left, 16, 64, 128, "expand", "order (Left: m)"},
                            {Side::Right, 64, 16, 128, "expand", "order (Right: n)"},
                            {Side::Left, 64, 16, 128, "vendor", "q (Left: n)"},
                            {Side::Right, 16, 64, 128, "vendor", "q (Right: m)"}};
    for (const auto& k : probes) {
        Spec s{k.side, Uplo::Upper, k.m, k.n, k.batch};
        s.seed = 43u;
        auto p = make_prob<T>(s);
        EXPECT_EQ(traced_choice([&] { this->run(p); }), k.expect) << "the " << k.field << " row, " << label(s);
        expect_hemm(p, std::string("the ") + k.field + " row");
    }
}

// The trace key: A's order and C's other extent, whatever the side.
TYPED_TEST(HemmCandidates, TraceKeyIsOrderAndQ) {
    using T = typename TestFixture::T;
    for (auto [side, m, n] : {std::tuple{Side::Left, 37, 12}, std::tuple{Side::Right, 12, 37}}) {
        Spec s{side, Uplo::Upper, m, n, 5};
        auto p = make_prob<T>(s);
        const std::string line = traced_line([&] { this->run(p); });
        EXPECT_NE(line.find("order=37 q=12 batch=5 ->"), std::string::npos) << line;
        expect_hemm(p, line);
    }
}

// With a table naming only the vendor, a vendor-free build takes the last resort, expand.
TYPED_TEST(HemmCandidates, VendorFreeLastResortIsExpand) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_HEMM_ROUTE", nullptr);
    const TableGuard restore;
    use_hemm_table(std::string(select::dtype_name<T>()), select::device_of<TestFixture::B>(*this->ctx).key,
                   "order=32 q=32 batch=128 | vendor -\n");
    for (const Spec& s : {Spec{Side::Left, Uplo::Lower, 32, 32, 64}, Spec{Side::Right, Uplo::Upper, 3, 300, 1}}) {
        auto p = make_prob<T>(s);
        std::string all;
        EXPECT_EQ(traced_choice([&] { this->run(p); }, &all), TestFixture::kVendor ? "vendor" : "expand") << label(s);
        if (!TestFixture::kVendor) EXPECT_NE(all.find("last resort"), std::string::npos) << all;
        expect_hemm(p, "vendor-only table, " + label(s));
    }
}

// The coverage row (§5.6): the real backend and scalar, the key (m, n = C's extents, k = A's
// order, uplo, side) and the native flags. The old cublas.cc branches recorded no row at all.
// threadsafe: the child re-executes the binary, so CUDA is initialised fresh, never forked.
TYPED_TEST(HemmCandidates, CoverageRowCarriesBackendKeyAndNativeFlags) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    const std::string dir = ::testing::TempDir() + "hemm_cov." + std::string(select::dtype_name<T>());
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const std::string out = dir + "/cov";
    const Spec lo{Side::Left, Uplo::Lower, 16, 5, 2};
    const Spec hi{Side::Right, Uplo::Upper, 7, 40, 2};
    auto child = [&] {
        const ScopedEnvVar cov("BATCHLAS_COVERAGE_OUT", out.c_str());
        const ScopedEnvVar clear("BATCHLAS_HEMM_ROUTE", nullptr);
        batchlas::coverage::g_dynamic_enabled = true;
        {
            const Pin pin("hemm", C{hm::Expand{}});
            auto p = make_prob<T>(lo);
            this->run(p);
        }
        const Pin pin("hemm", TestFixture::kVendor ? C{hm::Vendor{}} : C{hm::Expand{}});
        auto p = make_prob<T>(hi);
        this->run(p);
        std::exit(0);
    };
    EXPECT_EXIT(child(), ::testing::ExitedWithCode(0), "");
    std::map<std::string, std::vector<std::string>> rows;
    for (const auto& ent : std::filesystem::directory_iterator(dir)) {
        std::ifstream in(ent.path());
        for (std::string line; std::getline(in, line);) {
            if (line.rfind("reached,hemm,", 0) != 0) continue;
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
    auto as_int = [](auto e) { return std::to_string(static_cast<int>(e)); };
    const std::string hi_algo = TestFixture::kVendor ? "vendor" : "expand";
    for (const auto& [key, s, algo] : {std::tuple{klo, lo, std::string("expand")}, std::tuple{khi, hi, hi_algo}}) {
        const auto& f = rows[key];
        EXPECT_EQ(f[2], std::string(batchlas::to_string(batchlas::scalar_kind_of<T>))) << key << ": the scalar column";
        EXPECT_EQ(f[3], B == Backend::CUDA ? "CUDA" : "?") << key << ": the backend column";
        EXPECT_EQ(f[9], algo == "vendor" ? "vendor" : "native") << key;
        EXPECT_EQ(f[10], algo) << key;
        EXPECT_EQ(f[12], "1") << key;
        EXPECT_EQ(f[13], "1") << key;
        EXPECT_EQ(f[15], as_int(s.uplo)) << key;
        EXPECT_EQ(f[16], as_int(s.side)) << key;
    }
}

// On a CPU queue no native family can run: expand pins throw, Auto is the vendor (or no route).
template <typename Config>
class HemmCandidatesCpu : public test_utils::BatchLASTest<Config> {};
TYPED_TEST_SUITE(HemmCandidatesCpu, Types);

TYPED_TEST(HemmCandidatesCpu, CpuQueueRunsNoNativeFamily) {
    using T = typename TypeParam::ScalarType;
    static constexpr Backend B = TypeParam::BackendVal;
    if (!this->ctx) GTEST_SKIP() << "no queue";
    if (this->ctx->device().type == DeviceType::GPU) GTEST_SKIP() << "a GPU queue";
    const ScopedEnvVar clear("BATCHLAS_HEMM_ROUTE", nullptr);
    const Spec s{Side::Left, Uplo::Lower, 8, 3, 4};
    {
        auto p = make_prob<T>(s);
        const Pin pin("hemm", C{hm::Expand{}});
        EXPECT_THROW(((void)hemm<B, T>(*this->ctx, p.A(), p.B(), p.Cv(), p.alpha, p.beta, s.side, s.uplo)),
                     std::invalid_argument);
    }
    auto p = make_prob<T>(s);
    auto call = [&] {
        (void)hemm<B, T>(*this->ctx, p.A(), p.B(), p.Cv(), p.alpha, p.beta, s.side, s.uplo);
        this->ctx->wait();
    };
    if constexpr (batchlas::select::level3_vendor_available<B>) {
        EXPECT_EQ(traced_choice(call), "vendor");
        expect_hemm(p, "cpu vendor");
    } else {
        EXPECT_THROW(call(), batchlas::NoRouteError);
    }
}

// The netlib loop shares the vendor loops' single top-level (m, n), so a heterogeneous batch has
// no route on a CPU queue either.
TYPED_TEST(HemmCandidatesCpu, HeterogeneousBatchHasNoRoute) {
    if (!this->ctx) GTEST_SKIP() << "no queue";
    if (this->ctx->device().type == DeviceType::GPU) GTEST_SKIP() << "a GPU queue";
    expect_heterogeneous_has_no_route<TypeParam::BackendVal, typename TypeParam::ScalarType>(*this->ctx);
}

// The transcription (no GPU): each table holds exactly choice.hh's grid, sm_89 and sm_120 alike,
// names the transcribed commit and choice.hh's keys, ranks both candidates, and its first entries
// restate the old rule at every cell.
TEST(HemmTranscribedTable, HoldsTheChoiceGridAndTheOldRule) {
    std::set<std::string> want;
    for (int o : hm::grid_extent)
        for (int q : hm::grid_extent)
            for (int b : hm::grid_batch) want.insert(std::to_string(o) + " " + std::to_string(q) + " " + std::to_string(b));
    for (const char* dt : {"cfloat", "cdouble"}) {
        std::map<std::string, std::string> first;
        for (const char* dev : {"sm_89", "sm_120"}) {
            const auto tables = select::tables_in_borrow_order("hemm", dt, select::device_from_key(dev));
            ASSERT_FALSE(tables.empty()) << dt << " " << dev;
            const select::Table& t = *tables.front();
            ASSERT_EQ(t.device, dev) << t.file;
            EXPECT_EQ(t.source, "transcribed:8cf7fd86") << t.file;
            ASSERT_EQ(t.keys.size(), hm::key_names.size()) << t.file;
            for (std::size_t i = 0; i < t.keys.size(); ++i) {
                const std::string w = t.keys[i].weight == 1.0 ? "" : ":" + std::to_string(int(t.keys[i].weight));
                EXPECT_EQ(t.keys[i].name + (t.keys[i].log ? ":log" : ":exact") + w, std::string(hm::key_names[i]))
                    << t.file;
            }
            std::set<std::string> got;
            for (const auto& row : t.rows) {
                const std::string k = row.keys[0] + " " + row.keys[1] + " " + row.keys[2];
                got.insert(k);
                ASSERT_EQ(row.ranked.size(), 2u) << t.file << ":" << row.line;
                EXPECT_FALSE(row.timed) << t.file << ":" << row.line;
                const int o = std::stoi(row.keys[0]), q = std::stoi(row.keys[1]), b = std::stoi(row.keys[2]);
                EXPECT_EQ(row.ranked.front().spelling, old_auto(o, q, b)) << t.file << ":" << row.line;
                const std::string ranked = row.ranked[0].spelling + "|" + row.ranked[1].spelling;
                if (std::string(dev) == "sm_89") first[k] = ranked;
                else EXPECT_EQ(ranked, first[k]) << t.file << " " << k << ": sm_120 differs from sm_89";
            }
            EXPECT_EQ(got, want) << t.file;
            EXPECT_EQ(t.rows.size(), want.size()) << t.file;
        }
    }
    struct Spot { int order, q, batch; const char* first; };
    const Spot spots[] = {{64, 64, 3, "vendor"},   {64, 64, 4, "expand"},   {250, 30, 2, "vendor"},
                          {260, 30, 2, "expand"},  {30, 300, 1, "expand"},  {900, 40, 9000, "expand"},
                          {3, 2000, 50, "expand"}, {200, 200, 3, "vendor"}, {7, 7, 1, "vendor"}};
    for (const Spot& s : spots) {
        const select::Table& t = *select::tables_in_borrow_order("hemm", "cfloat", select::device_from_key("sm_89")).front();
        const select::Key key{{"order", s.order}, {"q", s.q}, {"batch", s.batch}};
        const select::TableRow* row = t.nearest(key);
        ASSERT_NE(row, nullptr);
        EXPECT_EQ(row->ranked.front().spelling, s.first) << t.file << ":" << row->line;
    }
}

}  // namespace
