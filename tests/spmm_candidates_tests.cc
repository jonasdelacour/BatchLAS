// Every spmm candidate, pinned: docs/design/flat-kernel-selection.md §8 and
// docs/design/flat-select-p5/spmm.md. Which kernel ran is read back from the select trace or a
// bit-for-bit comparison with the direct driver, never assumed from the pin being accepted.
// Ports the RouteSpmm.* cases of route_vocabulary_tests that still describe behaviour.
#include <gtest/gtest.h>

#include <batchlas/blas/functions/spmm.hh>
#include <batchlas/blas/dispatch/coverage.hh>
#include <batchlas/blas/dispatch/no_route.hh>
#include <batchlas/blas/dispatch/vendor_available.hh>
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>

#include "test_utils.hh"

#include "../src/ops/spmm/choice.hh"
#include "../src/sycl/spmm_native.hh"

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
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <variant>
#include <vector>

using namespace batchlas;

namespace {

namespace sp = batchlas::ops::spmm;
using C = sp::SpmmChoice;
using Pin = select::ScopedPin<C>;
template <typename T>
using Dense = MatrixView<T, MatrixFormat::Dense>;
template <typename T>
using Csr = MatrixView<T, MatrixFormat::CSR>;

template <typename T>
using RealOf = typename batchlas::base_type<T>::type;
template <typename T>
constexpr bool kCx = test_utils::is_complex<T>::value;

template <typename T>
T mk(RealOf<T> r, RealOf<T> i) {
    if constexpr (kCx<T>) return T(r, i);
    else return r;
}
template <typename T>
T cj(T v) {
    if constexpr (kCx<T>) return std::conj(v);
    else return v;
}
template <typename T>
bool same_bits(T a, T b) {
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}

const char* tr_s(Transpose t) { return t == Transpose::NoTrans ? "N" : (t == Transpose::Trans ? "T" : "C"); }
constexpr Transpose kAllTrans[3] = {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans};

struct Spec {
    int m = 7, k = 5, nrhs = 3, batch = 3;
    Transpose ta = Transpose::NoTrans, tb = Transpose::NoTrans;
    int nnz_row = 3;  // rows hold 0 .. 2*nnz_row entries, differently per item
    int period = 0;   // > 0: item b repeats item b % period
    unsigned seed = 1;
};

std::string label(const Spec& s) {
    return std::string(tr_s(s.ta)) + tr_s(s.tb) + " m=" + std::to_string(s.m) + " k=" + std::to_string(s.k) +
           " nrhs=" + std::to_string(s.nrhs) + " batch=" + std::to_string(s.batch);
}
std::string name(const C& c, const Spec& s) { return select::to_string(c) + " " + label(s); }

// A batched CSR A with item-local offsets, an offset stride above m + 1 and a value stride above
// the capacity; B and C share one buffer of a large finite poison at padded lds and strides that
// are not ld * cols. Slots above an item's nnz hold the poison with an in-range column index.
template <typename T>
struct Problem {
    Spec s;
    T alpha{}, beta{};
    int out_rows = 0, red_rows = 0, b_rows = 0, b_cols = 0;
    int ldb = 1, ldc = 1, sb = 1, sc = 1, cap = 1, vstride = 1, ostride = 1;
    std::size_t boff = 0, coff = 0;
    UnifiedVector<T> val, mem;
    UnifiedVector<int> ro, ci;
    std::vector<T> mem0;

    Csr<T> A() { return {val.data(), ro.data(), ci.data(), s.m, s.k, NonZeros{cap}, vstride, ostride, s.batch}; }
    Dense<T> B() { return {mem.data() + boff, b_rows, b_cols, ldb, sb, s.batch}; }
    Dense<T> Cv() { return {mem.data() + coff, out_rows, s.nrhs, ldc, sc, s.batch}; }
    std::size_t bi(int b, int i, int j) const { return boff + std::size_t(b) * sb + std::size_t(j) * ldb + i; }
    std::size_t cidx(int b, int i, int j) const { return coff + std::size_t(b) * sc + std::size_t(j) * ldc + i; }
};

template <typename T>
T poison() {
    return mk<T>(RealOf<T>(-999), RealOf<T>(777));
}

template <typename T>
Problem<T> make(const Spec& s) {
    using R = RealOf<T>;
    Problem<T> p;
    p.s = s;
    p.alpha = mk<T>(R(1.25), R(-0.5));
    p.beta = mk<T>(R(-0.75), R(0.25));
    const bool an = s.ta == Transpose::NoTrans, bn = s.tb == Transpose::NoTrans;
    p.out_rows = an ? s.m : s.k;
    p.red_rows = an ? s.k : s.m;
    p.b_rows = bn ? p.red_rows : s.nrhs;
    p.b_cols = bn ? s.nrhs : p.red_rows;
    p.ldb = std::max(1, p.b_rows) + 3;
    p.ldc = std::max(1, p.out_rows) + 2;
    p.sb = p.ldb * std::max(1, p.b_cols) + 7;
    p.sc = p.ldc * std::max(1, s.nrhs) + 5;
    p.ostride = s.m + 3;
    std::mt19937 gen(s.seed);
    std::uniform_real_distribution<R> u(R(-1), R(1));
    const int reps = s.period > 0 ? std::min(s.period, s.batch) : s.batch;
    std::vector<std::vector<int>> rows(reps), cols(reps);
    for (int b = 0; b < reps; ++b) {
        rows[b].push_back(0);
        // Distinct, unsorted columns: cuSPARSE refuses nnz > rows * cols (duplicates: spmm_tests).
        std::vector<int> pool(s.k);
        for (int i = 0; i < s.m; ++i) {
            const int cnt = int(gen() % unsigned(std::min(2 * s.nnz_row, s.k) + 1));
            for (int j = 0; j < s.k; ++j) pool[j] = j;
            std::shuffle(pool.begin(), pool.end(), gen);
            cols[b].insert(cols[b].end(), pool.begin(), pool.begin() + cnt);
            rows[b].push_back(int(cols[b].size()));
        }
    }
    p.cap = 0;
    for (int b = 0; b < reps; ++b) p.cap = std::max(p.cap, int(cols[b].size()));
    p.vstride = p.cap + 2;
    p.val.resize(std::size_t(p.vstride) * std::max(1, s.batch));
    p.ci.resize(p.val.size());
    p.ro.resize(std::size_t(p.ostride) * std::max(1, s.batch));
    std::vector<T> vals_rep(std::size_t(p.vstride) * reps);
    for (auto& v : vals_rep) v = mk<T>(u(gen), u(gen));
    for (int b = 0; b < s.batch; ++b) {
        const int r = b % reps;
        const int nnz = int(cols[r].size());
        for (int i = 0; i < p.ostride; ++i) p.ro[std::size_t(b) * p.ostride + i] = i <= s.m ? rows[r][i] : p.cap;
        for (int q = 0; q < p.vstride; ++q) {
            const std::size_t at = std::size_t(b) * p.vstride + q;
            p.val[at] = q < nnz ? vals_rep[std::size_t(r) * p.vstride + q] : mk<T>(R(1e6), R(-1e6));
            p.ci[at] = q < nnz ? cols[r][q] : 0;
        }
    }
    p.boff = 5;
    p.coff = p.boff + std::size_t(p.sb) * std::max(1, s.batch) + 11;
    p.mem.resize(p.coff + std::size_t(p.sc) * std::max(1, s.batch) + 13);
    for (std::size_t e = 0; e < p.mem.size(); ++e) p.mem[e] = poison<T>();
    std::vector<T> brep(std::size_t(std::max(1, p.b_rows)) * std::max(1, p.b_cols) * reps),
        crep(std::size_t(std::max(1, p.out_rows)) * std::max(1, s.nrhs) * reps);
    for (auto& v : brep) v = mk<T>(u(gen), u(gen));
    for (auto& v : crep) v = mk<T>(u(gen), u(gen));
    for (int b = 0; b < s.batch; ++b) {
        const std::size_t r = std::size_t(b % reps);
        for (int j = 0; j < p.b_cols; ++j)
            for (int i = 0; i < p.b_rows; ++i)
                p.mem[p.bi(b, i, j)] = brep[(r * std::max(1, p.b_cols) + j) * std::max(1, p.b_rows) + i];
        for (int j = 0; j < s.nrhs; ++j)
            for (int i = 0; i < p.out_rows; ++i)
                p.mem[p.cidx(b, i, j)] = crep[(r * std::max(1, s.nrhs) + j) * std::max(1, p.out_rows) + i];
    }
    p.mem0.assign(p.mem.begin(), p.mem.end());
    return p;
}

// C = alpha op(A) op(B) + beta C from the definition, accumulated over the nonzeros so duplicate
// columns sum; every in-range C element within a scale-relative bound, every other element of
// the shared buffer bit for bit.
template <typename T>
void expect_correct(const Problem<T>& p, const std::string& what) {
    using R = RealOf<T>;
    const Spec& s = p.s;
    const R eps = std::numeric_limits<R>::epsilon();
    int bad = 0;
    for (int b = 0; b < s.batch && bad < 5; ++b) {
        const std::size_t n_out = std::size_t(std::max(1, p.out_rows)) * std::max(1, s.nrhs);
        std::vector<T> acc(n_out, T(0));
        std::vector<R> scale(n_out, R(0));
        auto opb = [&](int j, int c) {
            if (s.tb == Transpose::NoTrans) return p.mem0[p.bi(b, j, c)];
            const T v = p.mem0[p.bi(b, c, j)];
            return s.tb == Transpose::ConjTrans ? cj(v) : v;
        };
        for (int i = 0; i < s.m; ++i)
            for (int q = p.ro[std::size_t(b) * p.ostride + i]; q < p.ro[std::size_t(b) * p.ostride + i + 1]; ++q) {
                T a = p.val[std::size_t(b) * p.vstride + q];
                const int j = p.ci[std::size_t(b) * p.vstride + q];
                const bool an = s.ta == Transpose::NoTrans;
                if (s.ta == Transpose::ConjTrans) a = cj(a);
                const int orow = an ? i : j, rrow = an ? j : i;
                for (int c = 0; c < s.nrhs; ++c) {
                    acc[std::size_t(c) * p.out_rows + orow] += a * opb(rrow, c);
                    scale[std::size_t(c) * p.out_rows + orow] += std::abs(a) * std::abs(opb(rrow, c));
                }
            }
        for (int c = 0; c < s.nrhs; ++c)
            for (int i = 0; i < p.out_rows; ++i) {
                const std::size_t e = std::size_t(c) * p.out_rows + i;
                const T c0 = p.mem0[p.cidx(b, i, c)];
                const T want = p.alpha * acc[e] + p.beta * c0;
                const R bound = R(64) * eps * (std::abs(p.alpha) * scale[e] + std::abs(p.beta) * std::abs(c0) + R(1));
                const T got = p.mem[p.cidx(b, i, c)];
                if (!(std::abs(got - want) <= bound) && bad++ < 5)
                    ADD_FAILURE() << what << ": C(" << i << "," << c << ") item " << b << " = " << got << ", want "
                                  << want;
            }
    }
    std::vector<char> in_c(p.mem.size(), 0);
    for (int b = 0; b < s.batch; ++b)
        for (int c = 0; c < s.nrhs; ++c)
            for (int i = 0; i < p.out_rows; ++i) in_c[p.cidx(b, i, c)] = 1;
    for (std::size_t e = 0; e < p.mem.size(); ++e)
        if (!in_c[e] && !same_bits(p.mem[e], p.mem0[e])) {
            ADD_FAILURE() << what << ": element " << e << " outside C was written";
            return;
        }
}

// The outermost spmm trace line for whatever `run` calls (a throw after the line is kept).
template <class F>
std::string traced_line(F&& run, std::string* all = nullptr, bool* threw = nullptr) {
    const ScopedEnvVar trace("BATCHLAS_SELECT_TRACE", "1");
    ::testing::internal::CaptureStderr();
    bool t = false;
    try {
        run();
    } catch (...) {
        t = true;
        if (!threw) {
            (void)::testing::internal::GetCapturedStderr();
            throw;
        }
    }
    if (threw) *threw = t;
    const std::string err = ::testing::internal::GetCapturedStderr();
    if (all) *all = err;
    std::istringstream in(err);
    for (std::string line; std::getline(in, line);)
        if (line.rfind("spmm ", 0) == 0 && line.find(" -> ") != std::string::npos) return line;
    return "<no spmm trace line in: " + err + ">";
}
template <class F>
std::string traced_choice(F&& run, std::string* all = nullptr, bool* threw = nullptr) {
    const std::string line = traced_line(std::forward<F>(run), all, threw);
    const auto arrow = line.find(" -> ");
    if (arrow == std::string::npos) return line;
    const std::string tail = line.substr(arrow + 4);
    return tail.substr(0, tail.find(' '));
}

struct TableGuard {
    ~TableGuard() { select::testing::use_embedded_tables(); }
};

template <typename T, Backend B>
struct Cfg {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};
using Types = typename test_utils::backend_types<Cfg>::type;

constexpr std::size_t kGuard = 256;

template <typename Config>
class SpmmCandidates : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    static constexpr Backend B = Config::BackendVal;
    static constexpr bool kVendor = dispatch::sparse_vendor_available<B>;
    // netlib's spmm refuses every transpose at run time; can_run does not model it (it never did,
    // see docs/design/flat-select-p5/spmm.md "known R3 gaps").
    static constexpr bool kVendorNoTransOnly = B == Backend::NETLIB;

    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        if (this->HasFatalFailure() || ::testing::Test::IsSkipped()) return;
        if (!this->ctx) GTEST_SKIP() << "no queue";
    }

    // ---- the limit oracle: what the driver needs, written out here, not spmm.cc's can_run ----
    // cuSPARSE: a conjugated single-row B is an error status the vendor arm never checked.
    static bool vendor_conj_row(const Spec& s) {
        return B == Backend::CUDA && kCx<T> && s.tb == Transpose::ConjTrans && s.nrhs == 1;
    }
    // cuSPARSE: complex<double> N/N with one column segfaults on the host (known-defects #13).
    static bool vendor_zz_column(const Spec& s) {
        return B == Backend::CUDA && std::is_same_v<T, std::complex<double>> && s.ta == Transpose::NoTrans &&
               s.tb == Transpose::NoTrans && s.nrhs == 1;
    }
    static bool expect_runs(const C& c, const Spec& s) {
        return std::holds_alternative<sp::Direct>(c) || (kVendor && !vendor_conj_row(s) && !vendor_zz_column(s));
    }
    // `vendor` is also the class word: a vendor pin nothing can serve warns and runs Auto (§5.3).
    static bool vendor_word_falls_back(const C& c, const Spec& s) {
        return std::holds_alternative<sp::Vendor>(c) && !expect_runs(c, s);
    }
    static bool vendor_refuses(const C& c, const Spec& s) {
        return std::holds_alternative<sp::Vendor>(c) && kVendorNoTransOnly &&
               (s.ta != Transpose::NoTrans || s.tb != Transpose::NoTrans);
    }

    // Sizes under the current pins, then runs in an arena of exactly that size whose guard
    // bytes past the end must survive (§8.3).
    void run(Problem<T>& p) {
        Queue& q = *this->ctx;
        const std::size_t need = spmm_buffer_size<B, T>(q, p.A(), p.B(), p.Cv(), p.alpha, p.beta, p.s.ta, p.s.tb);
        UnifiedVector<std::byte> arena(need + kGuard, std::byte{0xA5});
        (void)spmm<B, T>(q, p.A(), p.B(), p.Cv(), p.alpha, p.beta, p.s.ta, p.s.tb,
                         Span<std::byte>(arena.data(), need));
        q.wait();
        for (std::size_t e = need; e < arena.size(); ++e)
            ASSERT_EQ(arena[e], std::byte{0xA5}) << "workspace overrun at byte " << e << " of " << need;
    }
    void run_pinned(const C& c, Problem<T>& p) {
        const Pin pin("spmm", c);
        run(p);
    }
    bool pin_accepted(const C& c, Problem<T>& p) {
        try {
            run_pinned(c, p);
            return true;
        } catch (const std::invalid_argument& e) {
            EXPECT_NE(std::string(e.what()).find("cannot run this shape"), std::string::npos) << e.what();
            return false;
        }
    }
    std::string auto_choice(Problem<T>& p, bool* threw = nullptr) {
        const ScopedEnvVar clear("BATCHLAS_SPMM_ROUTE", nullptr);
        return traced_choice([&] { run(p); }, nullptr, threw);
    }
    // The family's own driver, called directly.
    void direct(const C& c, Problem<T>& p) {
        Queue& q = *this->ctx;
        if (std::holds_alternative<sp::Direct>(c)) {
            (void)sycl_spmm::spmm_native_csr<T>(q, p.A(), p.B(), p.Cv(), p.alpha, p.beta, p.s.ta, p.s.tb);
        } else if constexpr (kVendor) {
            const std::size_t need =
                backend::spmm_vendor_buffer_size<B, T>(q, p.A(), p.B(), p.Cv(), p.alpha, p.beta, p.s.ta, p.s.tb);
            UnifiedVector<std::byte> ws(std::max<std::size_t>(need, 1));
            (void)backend::spmm_vendor<B, T>(q, p.A(), p.B(), p.Cv(), p.alpha, p.beta, p.s.ta, p.s.tb,
                                             Span<std::byte>(ws.data(), need));
            q.wait();
        }
        q.wait();
    }
};

TYPED_TEST_SUITE(SpmmCandidates, Types);

// §8.1 / RouteSpmm.AllNineTransposeCombinationsSupported: each candidate on all nine (transA,
// transB) spellings, rectangular both ways, nrhs straddling the gather's 2-column block, distinct
// per-item patterns with heterogeneous nnz, complex alpha/beta, beta != 0.
TYPED_TEST(SpmmCandidates, PinnedCandidatesRunEveryTransposeCombination) {
    using T = typename TestFixture::T;
    int ran = 0;
    for (const C& c : sp::candidates<T>())
        for (Transpose ta : kAllTrans)
            for (Transpose tb : kAllTrans)
                for (auto [m, k] : {std::pair{9, 4}, std::pair{4, 11}})
                    for (int nrhs : {1, 2, 3, 17}) {
                        Spec s{m, k, nrhs, 3, ta, tb};
                        s.seed = 100u + 7u * m + 3u * nrhs + unsigned(ta) * 11u + unsigned(tb);
                        auto p = make<T>(s);
                        const std::string what = name(c, s);
                        if (TestFixture::vendor_word_falls_back(c, s)) {
                            auto q = make<T>(s);
                            const std::string want = this->auto_choice(q);
                            EXPECT_EQ(want, "direct") << what << ": Auto took a vendor can_run refuses";
                            EXPECT_EQ(traced_choice([&] { this->run_pinned(c, p); }), want) << what;
                            expect_correct(p, what + " (refused vendor pin: Auto)");
                            continue;
                        }
                        if (TestFixture::vendor_refuses(c, s)) {
                            EXPECT_ANY_THROW(this->run_pinned(c, p)) << what << ": netlib took a transpose";
                            continue;
                        }
                        ASSERT_TRUE(this->pin_accepted(c, p)) << what << " was refused";
                        expect_correct(p, what);
                        ++ran;
                        if (::testing::Test::HasFatalFailure()) return;
                    }
    EXPECT_GT(ran, 0);
}

// The pinned facade runs exactly that family's driver: bit-identical to the direct call. Only the
// gather (transA == N) is deterministic; the scatter's atomics reorder sums run to run. Long rows
// make the two families' roundings differ, so a swapped launch arm shows. cuSPARSE's default
// algorithm is not bit-reproducible run to run (measured on CUDA), so a vendor whose own two runs
// differ is identified by the trace line alone.
TYPED_TEST(SpmmCandidates, PinnedRunIsTheDirectKernelBitForBit) {
    using T = typename TestFixture::T;
    int compared = 0;
    for (const C& c : sp::candidates<T>()) {
        for (Transpose tb : kAllTrans)
            for (int nrhs : {2, 9}) {
                Spec s{40, 33, nrhs, 3, Transpose::NoTrans, tb, 12};
                s.seed = 4242u + nrhs + unsigned(tb);
                if (!TestFixture::expect_runs(c, s) || TestFixture::vendor_refuses(c, s)) continue;
                auto pinned = make<T>(s);
                auto direct = make<T>(s);
                const std::string what = name(c, s);
                EXPECT_EQ(traced_choice([&] { this->run_pinned(c, pinned); }), select::to_string(c)) << what;
                this->direct(c, direct);
                expect_correct(pinned, what);
                if (std::holds_alternative<sp::Vendor>(c)) {
                    auto again = make<T>(s);
                    this->direct(c, again);
                    bool reproducible = true;
                    for (std::size_t e = 0; e < again.mem.size(); ++e)
                        reproducible = reproducible && same_bits(again.mem[e], direct.mem[e]);
                    if (!reproducible) continue;
                }
                for (std::size_t e = 0; e < pinned.mem.size(); ++e)
                    ASSERT_TRUE(same_bits(pinned.mem[e], direct.mem[e]))
                        << what << ": the pinned facade did not run this family's driver; element " << e;
                ++compared;
            }
    }
    EXPECT_GT(compared, 0);
}

// A saturating batch: 1024 items repeating 5 distinct problems; every item correct, and on the
// deterministic gather each item bit-identical to its representative (not cuSPARSE: see above).
TYPED_TEST(SpmmCandidates, SaturatingBatchIsBitIdenticalToItsRepresentative) {
    using T = typename TestFixture::T;
    constexpr int kBatch = 1024, kPeriod = 5;
    for (const C& c : sp::candidates<T>())
        for (Transpose ta : {Transpose::NoTrans, Transpose::ConjTrans})
            for (Transpose tb : {Transpose::NoTrans, Transpose::Trans}) {
                Spec s{24, 19, 5, kBatch, ta, tb, 4, kPeriod, 777u};
                if (!TestFixture::expect_runs(c, s) || TestFixture::vendor_refuses(c, s)) continue;
                auto p = make<T>(s);
                const std::string what = name(c, s);
                this->run_pinned(c, p);
                expect_correct(p, what);
                if (ta != Transpose::NoTrans || std::holds_alternative<sp::Vendor>(c)) continue;
                for (int b = kPeriod; b < kBatch; ++b)
                    for (int j = 0; j < s.nrhs; ++j)
                        for (int i = 0; i < p.out_rows; ++i)
                            ASSERT_TRUE(same_bits(p.mem[p.cidx(b, i, j)], p.mem[p.cidx(b % kPeriod, i, j)]))
                                << what << ": item " << b << " differs from item " << b % kPeriod;
            }
}

// RouteSpmm.ZeroExtentsAreSupportedNegativeAreNot, ported: m, k or nrhs 0 is a legal call that
// every candidate accepts (Direct quick-returns on the host); C stays C = beta C where it exists.
TYPED_TEST(SpmmCandidates, ZeroExtentsAreAccepted) {
    using T = typename TestFixture::T;
    for (const C& c : sp::candidates<T>())
        for (Transpose ta : {Transpose::NoTrans, Transpose::Trans})
            for (auto [m, k, nrhs] : {std::tuple{0, 6, 3}, std::tuple{6, 0, 3}, std::tuple{6, 5, 0}}) {
                Spec s{m, k, nrhs, 2, ta, Transpose::NoTrans};
                if (!TestFixture::expect_runs(c, s) || TestFixture::vendor_refuses(c, s)) continue;
                auto p = make<T>(s);
                const std::string what = name(c, s);
                ASSERT_TRUE(this->pin_accepted(c, p)) << what;
                expect_correct(p, what);
            }
}

// §8.2 (R3): Direct's can_run against the oracle on shapes straddling each of its terms. Its
// driver has no checks of its own (a refused shape would read out of bounds), so refusal is
// checked through the pin and acceptance through a correct answer.
TYPED_TEST(SpmmCandidates, CanRunEqualsTheOracle) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    const C direct{sp::Direct{}};
    Spec s{8, 6, 3, 3};
    {
        auto p = make<T>(s);
        ASSERT_TRUE(this->pin_accepted(direct, p)) << "the control shape";
        expect_correct(p, "control");
    }
    struct Bad { const char* what; int a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, ab, bb, cb, ostride; };
    const Bad cases[] = {{"opA cols != opB rows", 8, 6, 5, 3, 8, 3, 3, 3, 3, 11},
                         {"C rows != opA rows", 8, 6, 6, 3, 7, 3, 3, 3, 3, 11},
                         {"C cols != opB cols", 8, 6, 6, 3, 8, 2, 3, 3, 3, 11},
                         {"B batch != A batch", 8, 6, 6, 3, 8, 3, 3, 2, 3, 11},
                         {"C batch != A batch", 8, 6, 6, 3, 8, 3, 3, 3, 2, 11},
                         {"offset stride m (needs m + 1)", 8, 6, 6, 3, 8, 3, 3, 3, 3, 8},
                         {"empty batch", 8, 6, 6, 3, 8, 3, 0, 0, 0, 11}};
    for (const auto& k : cases) {
        auto p = make<T>(s);
        const Csr<T> A(p.val.data(), p.ro.data(), p.ci.data(), k.a_rows, k.a_cols, NonZeros{p.cap}, p.vstride,
                       k.ostride, k.ab);
        const Dense<T> Bm(p.mem.data() + p.boff, k.b_rows, k.b_cols, p.ldb, p.sb, k.bb);
        const Dense<T> Cm(p.mem.data() + p.coff, k.c_rows, k.c_cols, p.ldc, p.sc, k.cb);
        const Pin pin("spmm", direct);
        try {
            (void)spmm<B, T>(*this->ctx, A, Bm, Cm, p.alpha, p.beta, Transpose::NoTrans, Transpose::NoTrans,
                             Span<std::byte>());
            this->ctx->wait();
            ADD_FAILURE() << k.what << ": direct was accepted";
        } catch (const std::invalid_argument& e) {
            EXPECT_NE(std::string(e.what()).find("cannot run this shape"), std::string::npos) << k.what;
        }
        for (std::size_t e = 0; e < p.mem.size(); ++e)
            ASSERT_TRUE(same_bits(p.mem[e], p.mem0[e])) << k.what << ": a refused call wrote element " << e;
    }
}

// The two Vendor terms (R3): cuSPARSE refuses a ConjTrans single-row B ("opB ==
// OPERATION_CONJUGATE_TRANSPOSE is unsupported when B is a single row"), the vendor arm drops the
// status and C is never written; complex<double> N/N with one column segfaults. Straddled on
// nrhs 1/2, transA N/T and transB N/T/C; Auto then runs Direct.
TYPED_TEST(SpmmCandidates, VendorRefusesAConjugatedSingleRowB) {
    using T = typename TestFixture::T;
    if (!TestFixture::kVendor) GTEST_SKIP() << "no vendor in this build";
    for (Transpose ta : {Transpose::NoTrans, Transpose::Trans})
        for (Transpose tb : kAllTrans)
            for (int nrhs : {1, 2}) {
                Spec s{5, 7, nrhs, 3, ta, tb};
                const bool refused = TestFixture::vendor_conj_row(s) || TestFixture::vendor_zz_column(s);
                auto p = make<T>(s);
                const std::string what = name(C{sp::Vendor{}}, s);
                if (TestFixture::vendor_refuses(C{sp::Vendor{}}, s)) continue;  // netlib: no transposes
                // A refused `vendor` pin is the class word's fallback to Auto, not an error (§5.3).
                EXPECT_EQ(traced_choice([&] { this->run_pinned(C{sp::Vendor{}}, p); }), refused ? "direct" : "vendor")
                    << what;
                expect_correct(p, what);
                auto q = make<T>(s);
                const std::string got = this->auto_choice(q);
                if (refused) EXPECT_EQ(got, "direct") << label(s);
                expect_correct(q, "auto " + label(s));
            }
}

// RouteSpmm.HeterogeneousBatchRefused, ported: a heterogeneous B or C has no native route (one
// launch has one ld/stride per dense operand); Auto takes the vendor, or there is no route.
TYPED_TEST(SpmmCandidates, HeterogeneousDenseBatchHasNoNativeRoute) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    Spec s{8, 6, 3, 4};
    auto p = make<T>(s);
    UnifiedVector<int> rows(s.batch), cols(s.batch), crow(s.batch), ccol(s.batch);
    for (int b = 0; b < s.batch; ++b) rows[b] = 6 - (b % 2), cols[b] = 3, crow[b] = 8 - (b % 2), ccol[b] = 3;
    const auto hetB = p.B().with_active_dims(rows.to_span(), cols.to_span());
    ASSERT_TRUE(hetB.is_heterogeneous());
    for (bool on_c : {false, true}) {
        auto q = make<T>(s);
        const auto hb = on_c ? q.B() : q.B().with_active_dims(rows.to_span(), cols.to_span());
        const auto hc = on_c ? q.Cv().with_active_dims(crow.to_span(), ccol.to_span()) : q.Cv();
        ASSERT_TRUE(hb.is_heterogeneous() || hc.is_heterogeneous());
        const Pin pin("spmm", C{sp::Direct{}});
        EXPECT_THROW(((void)spmm<B, T>(*this->ctx, q.A(), hb, hc, q.alpha, q.beta, Transpose::NoTrans,
                                       Transpose::NoTrans, Span<std::byte>())),
                     std::invalid_argument)
            << (on_c ? "C" : "B") << " heterogeneous";
        for (std::size_t e = 0; e < q.mem.size(); ++e)
            ASSERT_TRUE(same_bits(q.mem[e], q.mem0[e])) << "a refused call wrote element " << e;
    }
    const ScopedEnvVar clear("BATCHLAS_SPMM_ROUTE", nullptr);
    auto call = [&] {
        UnifiedVector<std::byte> ws(1 << 20);
        (void)spmm<B, T>(*this->ctx, p.A(), hetB, p.Cv(), p.alpha, p.beta, Transpose::NoTrans, Transpose::NoTrans,
                         ws.to_span());
        this->ctx->wait();
    };
    if constexpr (TestFixture::kVendor) {
        bool threw = false;
        EXPECT_EQ(traced_choice(call, nullptr, &threw), "vendor");
    } else {
        EXPECT_THROW(call(), dispatch::NoRouteError);
    }
}

// R6: a spelling that names nothing compiled throws instead of meaning Auto (RouteSpmm's
// SilentPinFallThrough is now an error: `cta` and `blocked` have no spmm body).
TYPED_TEST(SpmmCandidates, UnknownPinsThrow) {
    using T = typename TestFixture::T;
    for (const char* word : {"bogus", "cta", "blocked", "native:cta", "direct:1", "native:vendor", "gather",
                             "native:direct:2"}) {
        auto p = make<T>(Spec{});
        const Pin pin("spmm", std::string_view(word));
        EXPECT_THROW(this->run(p), std::invalid_argument) << word;
    }
    const ScopedEnvVar env("BATCHLAS_SPMM_ROUTE", "not-a-route");
    auto p = make<T>(Spec{});
    EXPECT_THROW(this->run(p), std::invalid_argument) << "an environment typo must throw, not mean Auto";
}

// §5.3 / RouteSpmm.BatchlasSpmmRouteIsActuallyRead and ForcedNativeStillReachesTheRefusedScatter:
// the legacy spellings and class words via ScopedPin and the environment, on a transposed A
// (outside the old preferred window) so `native` must reach the scatter.
TYPED_TEST(SpmmCandidates, LegacyAliasesAndClassWords) {
    using T = typename TestFixture::T;
    for (Transpose ta : {Transpose::NoTrans, Transpose::ConjTrans}) {
        const Spec s{9, 7, 4, 3, ta, Transpose::Trans};
        std::string auto_pick;
        {
            auto p = make<T>(s);
            bool threw = false;
            auto_pick = this->auto_choice(p, &threw);
        }
        ASSERT_TRUE(auto_pick == "direct" || auto_pick == "vendor") << auto_pick;
        const std::string vendor_pick = TestFixture::kVendor ? "vendor" : auto_pick;
        const std::pair<const char*, std::string> expect[] = {
            {"native:direct", "direct"}, {"NATIVE:DIRECT", "direct"}, {"direct", "direct"}, {"Direct", "direct"},
            {"native", "direct"},        {"vendor", vendor_pick},     {"auto", auto_pick}};
        for (const auto& [word, spelling] : expect)
            for (bool via_env : {false, true}) {
                auto p = make<T>(s);
                select::testing::reset_warnings();
                std::string err;
                bool threw = false;
                const std::string got = traced_choice(
                    [&] {
                        const ScopedEnvVar env("BATCHLAS_SPMM_ROUTE", via_env ? word : nullptr);
                        std::optional<Pin> pin;
                        if (!via_env) pin.emplace("spmm", std::string_view(word));
                        this->run(p);
                    },
                    &err, &threw);
                const std::string what = std::string(word) + (via_env ? " via env" : " via ScopedPin") + " " + label(s);
                EXPECT_EQ(got, spelling) << what;
                const bool warns = std::string(word) == "vendor" && !TestFixture::kVendor;
                EXPECT_EQ(err.find("spmm pinned \"vendor\", but no vendor candidate") != std::string::npos, warns)
                    << what << ": " << err;
                if (got == "vendor" && TestFixture::vendor_refuses(C{sp::Vendor{}}, s)) {
                    EXPECT_TRUE(threw) << what;
                    continue;
                }
                EXPECT_FALSE(threw) << what;
                expect_correct(p, what);
            }
    }
}

// §5.3: a ScopedPin wins over BATCHLAS_SPMM_ROUTE, and nested pins restore the outer one.
TYPED_TEST(SpmmCandidates, ScopedPinBeatsTheEnvironment) {
    using T = typename TestFixture::T;
    if (!TestFixture::kVendor) GTEST_SKIP() << "needs two runnable candidates";
    const Spec s{6, 6, 2, 2};
    const ScopedEnvVar env("BATCHLAS_SPMM_ROUTE", "cta");
    {
        auto p = make<T>(s);
        EXPECT_THROW(this->run(p), std::invalid_argument) << "the environment pin was not read";
    }
    auto p = make<T>(s);
    EXPECT_EQ(traced_choice([&] {
                  const Pin pin("spmm", C{sp::Vendor{}});
                  this->run(p);
              }),
              "vendor");
    expect_correct(p, "vendor over env cta");
    const Pin outer("spmm", C{sp::Direct{}});
    {
        const Pin inner("spmm", C{sp::Vendor{}});
        auto r = make<T>(s);
        EXPECT_EQ(traced_choice([&] { this->run(r); }), "vendor");
    }
    auto r = make<T>(s);
    EXPECT_EQ(traced_choice([&] { this->run(r); }), "direct") << "the inner pin did not restore the outer";
    expect_correct(r, "outer direct");
}

// The old preferred() clause, read back through Auto on the shipped tables (RouteSpmm's
// Preferred* and AutoTakesNativeWhereTheClauseFiresAndVendorWhereItDoesNot): direct for
// transA == N except complex<float> with transB != N; the vendor otherwise; direct for
// everything once the vendor is gone. No extent, batch or GPU term: grid and off-grid sizes, and
// the CPU queue (its own cpu table) read the same.
TYPED_TEST(SpmmCandidates, AutoReadsTheTranscribedTables) {
    using T = typename TestFixture::T;
    const auto& d = select::device_of<TestFixture::B>(*this->ctx);
    const auto tables = select::tables_in_borrow_order("spmm", select::dtype_name<T>(), d);
    ASSERT_FALSE(tables.empty()) << "no spmm table for " << d.key;
    struct Size { int m, k, nrhs, batch; };
    const Size sizes[] = {{1, 3, 1, 1}, {37, 21, 3, 5}, {300, 260, 2, 200}, {16, 16, 17, 1024}};
    for (const Size& z : sizes)
        for (Transpose ta : kAllTrans)
            for (Transpose tb : kAllTrans) {
                const bool cf = std::is_same_v<T, std::complex<float>>;
                const bool window = ta == Transpose::NoTrans && !(cf && tb != Transpose::NoTrans);
                Spec s{z.m, z.k, z.nrhs, z.batch, ta, tb};
                const bool vendor_ok = TestFixture::expect_runs(C{sp::Vendor{}}, s);
                const std::string want = (window || !vendor_ok) ? "direct" : "vendor";
                s.seed = 31u + z.m;
                auto p = make<T>(s);
                bool threw = false;
                EXPECT_EQ(this->auto_choice(p, &threw), want) << label(s) << " on " << tables.front()->file;
                if (want == "vendor" && TestFixture::vendor_refuses(C{sp::Vendor{}}, s)) continue;
                EXPECT_FALSE(threw) << label(s);
                expect_correct(p, "auto " + label(s));
            }
}

// key_of's every field reaches choose(): a synthetic table for this device whose winner changes
// with transA, transB (ConjTrans folding to T), m (A.rows, not the output extent), nrhs and batch.
TYPED_TEST(SpmmCandidates, AutoReadsEveryKeyField) {
    using T = typename TestFixture::T;
    if (!TestFixture::kVendor) GTEST_SKIP() << "the vendor-free build has one runnable candidate";
    const ScopedEnvVar clear("BATCHLAS_SPMM_ROUTE", nullptr);
    const std::string dtype(select::dtype_name<T>());
    const std::string dev = select::device_of<TestFixture::B>(*this->ctx).key;
    std::vector<std::pair<std::string, std::string>> files;
    for (const auto& t : select::embedded_tables())
        if (t.name.rfind("spmm.", 0) != 0) files.emplace_back(std::string(t.name), std::string(t.text));
    files.emplace_back("spmm." + dtype + "." + dev + ".txt",
                       "# op=spmm dtype=" + dtype + " device=" + dev + " kernels=unknown\n"
                       "# keys: transA:exact transB:exact m:log nrhs:log batch:log\n"
                       "transA=N transB=N m=16 nrhs=4 batch=8 | vendor 1 | direct 2\n"
                       "transA=N transB=N m=16 nrhs=4 batch=4096 | direct 1 | vendor 2\n"
                       "transA=N transB=N m=16 nrhs=64 batch=8 | direct 1 | vendor 2\n"
                       "transA=N transB=N m=1024 nrhs=4 batch=8 | direct 1 | vendor 2\n"
                       "transA=N transB=T m=16 nrhs=4 batch=8 | direct 1 | vendor 2\n"
                       "transA=T transB=N m=16 nrhs=4 batch=8 | direct 1 | vendor 2\n"
                       "transA=T transB=N m=1024 nrhs=4 batch=8 | vendor 1 | direct 2\n");
    const TableGuard restore;
    select::testing::set_builtin_tables(std::move(files));
    struct Probe { Transpose ta, tb; int m, k, nrhs, batch; const char* expect; const char* field; };
    const Probe probes[] = {
        {Transpose::NoTrans, Transpose::NoTrans, 16, 16, 4, 8, "vendor", "base"},
        {Transpose::NoTrans, Transpose::NoTrans, 16, 16, 4, 4096, "direct", "batch"},
        {Transpose::NoTrans, Transpose::NoTrans, 16, 16, 64, 8, "direct", "nrhs"},
        {Transpose::NoTrans, Transpose::NoTrans, 1024, 16, 4, 8, "direct", "m"},
        {Transpose::NoTrans, Transpose::Trans, 16, 16, 4, 8, "direct", "transB"},
        {Transpose::NoTrans, Transpose::ConjTrans, 16, 16, 4, 8, "direct", "transB (C folds to T)"},
        {Transpose::Trans, Transpose::NoTrans, 16, 16, 4, 8, "direct", "transA"},
        {Transpose::ConjTrans, Transpose::NoTrans, 16, 16, 4, 8, "direct", "transA (C folds to T)"},
        {Transpose::Trans, Transpose::NoTrans, 1024, 16, 4, 8, "vendor", "m is A.rows under transA=T"}};
    for (const auto& k : probes) {
        Spec s{k.m, k.k, k.nrhs, k.batch, k.ta, k.tb};
        s.seed = 43u;
        auto p = make<T>(s);
        bool threw = false;
        EXPECT_EQ(traced_choice([&] { this->run(p); }, nullptr, &threw), k.expect) << "the " << k.field << " row";
    }
}

// The trace line prints key_of itself: m is A.rows as stored (not the output extent), ConjTrans
// prints as T (RouteSpmm.OutRowsAndRedRowsSwapWithTransA: the key never spells out_rows).
TYPED_TEST(SpmmCandidates, TraceKeyIsKeyOf) {
    using T = typename TestFixture::T;
    const Pin pin("spmm", C{sp::Direct{}});
    for (Transpose ta : {Transpose::NoTrans, Transpose::ConjTrans}) {
        Spec s{13, 37, 5, 2, ta, Transpose::ConjTrans};
        auto p = make<T>(s);
        const std::string line = traced_line([&] { this->run(p); });
        const std::string want = std::string("transA=") + (ta == Transpose::NoTrans ? "N" : "T") +
                                 " transB=T m=13 nrhs=5 batch=2 -> direct";
        EXPECT_NE(line.find(want), std::string::npos) << line;
        expect_correct(p, line);
    }
}

// RouteSpmm.NoGpuGateOnDirect, ported: on a CPU queue Direct runs all nine spellings, and Auto
// on a NoTrans CPU spmm is Direct even with netlib present.
TYPED_TEST(SpmmCandidates, DirectHasNoGpuGate) {
    using T = typename TestFixture::T;
    if (this->ctx->device().type == DeviceType::GPU) GTEST_SKIP() << "a GPU queue";
    for (Transpose ta : kAllTrans)
        for (Transpose tb : kAllTrans) {
            Spec s{12, 10, 2, 8, ta, tb};
            auto p = make<T>(s);
            ASSERT_TRUE(this->pin_accepted(C{sp::Direct{}}, p)) << label(s);
            expect_correct(p, "cpu direct " + label(s));
        }
    auto p = make<T>(Spec{12, 10, 2, 8});
    EXPECT_EQ(this->auto_choice(p), "direct");
}

// The coverage row (§5.6): the real backend, m = A.rows, n = nrhs, k = A.cols, transA and transB
// as passed (not folded), the chosen spelling and origin, and the native flags.
// threadsafe: the child re-executes the binary, so CUDA is initialised fresh, never forked.
TYPED_TEST(SpmmCandidates, CoverageRowCarriesBackendKeyAndNativeFlags) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    const std::string dir = ::testing::TempDir() + "spmm_cov." + std::string(select::dtype_name<T>()) +
                            (B == Backend::NETLIB ? ".netlib" : ".gpu");
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const std::string out = dir + "/cov";
    const Spec lo{17, 9, 3, 2, Transpose::NoTrans, Transpose::NoTrans};
    const Spec hi{40, 23, 5, 2, Transpose::ConjTrans, Transpose::Trans};
    auto child = [&] {
        const ScopedEnvVar cov("BATCHLAS_COVERAGE_OUT", out.c_str());
        const ScopedEnvVar clear("BATCHLAS_SPMM_ROUTE", nullptr);
        dispatch::coverage::g_dynamic_enabled = true;
        const Pin pin("spmm", C{sp::Direct{}});
        auto a = make<T>(lo);
        this->run(a);
        auto b = make<T>(hi);
        this->run(b);
        std::exit(0);
    };
    EXPECT_EXIT(child(), ::testing::ExitedWithCode(0), "");
    std::map<std::string, std::vector<std::string>> rows;  // "m n k" -> fields
    for (const auto& ent : std::filesystem::directory_iterator(dir)) {
        std::ifstream in(ent.path());
        for (std::string line; std::getline(in, line);) {
            if (line.rfind("reached,spmm,", 0) != 0) continue;
            std::vector<std::string> f;
            std::stringstream ss(line);
            for (std::string tok; std::getline(ss, tok, ',');) f.push_back(tok);
            ASSERT_GE(f.size(), 20u) << line;
            rows[f[5] + " " + f[6] + " " + f[7]] = f;
        }
    }
    std::filesystem::remove_all(dir);
    ASSERT_EQ(rows.size(), 2u);
    const std::string backend = B == Backend::CUDA ? "CUDA" : (B == Backend::ROCM ? "ROCM" : "NETLIB");
    auto as_int = [](auto e) { return std::to_string(static_cast<int>(e)); };
    for (const Spec& s : {lo, hi}) {
        const std::string key = std::to_string(s.m) + " " + std::to_string(s.nrhs) + " " + std::to_string(s.k);
        ASSERT_TRUE(rows.count(key)) << "no row " << key;
        const auto& f = rows[key];
        EXPECT_EQ(f[3], backend) << key;
        EXPECT_EQ(f[9], "native") << key;
        EXPECT_EQ(f[10], "direct") << key;
        EXPECT_EQ(f[12], "1") << key;
        EXPECT_EQ(f[13], "1") << key;
        EXPECT_EQ(f[18], as_int(s.ta)) << key;
        EXPECT_EQ(f[19], as_int(s.tb)) << key;
    }
}

// The transcribed rows, read with Table::nearest directly so every build checks them on every
// device key: the old clause at grid and off-grid sizes (no size term, so any size reads the same).
TEST(SpmmTranscribedTable, RowsHoldTheOldPreference) {
    for (const char* dev : {"sm_89", "sm_120", "cpu"})
        for (const char* dtype : {"float", "double", "cfloat", "cdouble"})
            for (const char* ta : {"N", "T"})
                for (const char* tb : {"N", "T"})
                    for (auto [m, nrhs, batch] : {std::tuple{1, 1, 1}, std::tuple{3000, 7, 300},
                                                  std::tuple{1 << 20, 1000, 100000}}) {
                        const auto tables = select::tables_in_borrow_order("spmm", dtype, select::device_from_key(dev));
                        ASSERT_FALSE(tables.empty()) << dtype << " " << dev;
                        const select::Table& t = *tables.front();
                        ASSERT_EQ(t.device, dev) << dtype;
                        EXPECT_EQ(t.source.rfind("transcribed:", 0), 0u) << t.file;
                        const select::Key key{{"transA", ta}, {"transB", tb}, {"m", m}, {"nrhs", nrhs}, {"batch", batch}};
                        const select::TableRow* row = t.nearest(key);
                        ASSERT_NE(row, nullptr) << t.file;
                        const std::string what = t.file + ":" + std::to_string(row->line);
                        const bool window = std::string(ta) == "N" && !(std::string(dtype) == "cfloat" && std::string(tb) == "T");
                        ASSERT_EQ(row->ranked.size(), 2u) << what;
                        EXPECT_EQ(row->ranked[0].spelling, window ? "direct" : "vendor") << what;
                        EXPECT_EQ(row->ranked[1].spelling, window ? "vendor" : "direct") << what;
                        EXPECT_FALSE(row->timed) << what;
                    }
}

}  // namespace
