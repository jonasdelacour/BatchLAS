// Every syr2k candidate, pinned: docs/design/flat-kernel-selection.md §8 and
// docs/design/flat-kernel-selection.md §12. Which kernel ran is read back from the select trace, the
// kernel trace or a bit-for-bit comparison with the direct vendor call, never assumed from the
// pin being accepted.
#include <gtest/gtest.h>

#include <batchlas/blas/functions/syr2k.hh>
#include "../src/select/coverage.hh"
#include <batchlas/no_route.hh>
#include "../src/select/vendor.hh"
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/sycl_interop.hh>
#include <batchlas/util/sycl-vector.hh>

#include "test_utils.hh"

#include "../src/ops/syr2k/choice.hh"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
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
#include <variant>
#include <vector>

using namespace batchlas;

namespace {

namespace s2 = batchlas::ops::syr2k;
using C = s2::Syr2kChoice;
using Pin = select::ScopedPin<C>;
template <typename T>
using MVof = MatrixView<T, MatrixFormat::Dense>;

template <typename T>
bool same_bits(T a, T b) {
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}

const char* uplo_s(Uplo u) { return u == Uplo::Lower ? "L" : "U"; }
const char* trans_s(Transpose t) { return t == Transpose::NoTrans ? "N" : (t == Transpose::Trans ? "T" : "C"); }

// One operand's footprint inside the shared buffer: item `it` starts at off + it * stride.
struct Region {
    std::size_t off = 0;
    int rows = 0, cols = 0, ld = 0, stride = 0;
};

struct Spec {
    Uplo uplo = Uplo::Lower;
    Transpose trans = Transpose::NoTrans;
    int n = 8, k = 5, batch = 3;
    int period = 0;        // > 0: item it repeats item it % period
    bool natural = false;  // packed, 16-byte aligned operands: the kernel's unpredicated leg
    unsigned seed = 1;
};

std::string label(const Spec& s) {
    return std::string(uplo_s(s.uplo)) + trans_s(s.trans) + " n=" + std::to_string(s.n) + " k=" +
           std::to_string(s.k) + " batch=" + std::to_string(s.batch) + (s.natural ? " packed" : "");
}
std::string name(const C& c, const Spec& s) { return select::to_string(c) + " " + label(s); }

// A, B and C in one buffer of poison. A and B are op-shaped (n x k, or k x n under T/C); C's
// unreferenced triangle, every ld pad and every gap hold distinct large finite values that a
// correct syr2k neither writes nor lets into the referenced triangle.
template <typename T>
struct Prob {
    Spec s;
    T alpha = T(1.25), beta = T(-0.5);
    Region a, b, c;
    UnifiedVector<T> mem;
    std::vector<T> mem0;
    MVof<T> A() { return {mem.data() + a.off, a.rows, a.cols, a.ld, a.stride, s.batch}; }
    MVof<T> B() { return {mem.data() + b.off, b.rows, b.cols, b.ld, b.stride, s.batch}; }
    MVof<T> Cv() { return {mem.data() + c.off, s.n, s.n, c.ld, c.stride, s.batch}; }
    std::size_t idx(const Region& r, int it, int i, int j) const {
        return r.off + std::size_t(it) * r.stride + std::size_t(j) * r.ld + i;
    }
    bool referenced(int i, int j) const { return s.uplo == Uplo::Lower ? i >= j : i <= j; }
};

std::size_t round4(std::size_t v) { return (v + 3) / 4 * 4; }

template <typename T>
Prob<T> make_prob(const Spec& s) {
    Prob<T> p;
    p.s = s;
    const bool nt = s.trans == Transpose::NoTrans;
    const int ar = nt ? s.n : s.k, ac = nt ? s.k : s.n;
    const int pad = s.natural ? 0 : 3, gap = s.natural ? 0 : 5;
    p.a = {0, ar, ac, ar + pad, (ar + pad) * ac + gap};
    p.b = {round4(std::size_t(p.a.stride) * s.batch + (s.natural ? 0 : 7)), ar, ac, ar + pad + 1,
           (ar + pad + 1) * ac + gap};
    p.c = {round4(p.b.off + std::size_t(p.b.stride) * s.batch + (s.natural ? 0 : 3)), s.n, s.n, s.n + 2 * pad,
           (s.n + 2 * pad) * s.n + gap};
    const std::size_t total = p.c.off + std::size_t(p.c.stride) * s.batch + 11;
    p.mem = UnifiedVector<T>(total, T(0));
    for (std::size_t e = 0; e < total; ++e) p.mem[e] = -T(1000) - T(e % 4093);
    std::mt19937 gen(s.seed);
    std::uniform_real_distribution<T> u(T(-1), T(1));
    const int reps = s.period > 0 ? std::min(s.period, s.batch) : s.batch;
    for (int it = 0; it < s.batch; ++it) {
        const int r = it % reps;
        for (int j = 0; j < ac; ++j)
            for (int i = 0; i < ar; ++i) {
                p.mem[p.idx(p.a, it, i, j)] = it < reps ? u(gen) : p.mem[p.idx(p.a, r, i, j)];
                p.mem[p.idx(p.b, it, i, j)] = it < reps ? u(gen) : p.mem[p.idx(p.b, r, i, j)];
            }
        for (int j = 0; j < s.n; ++j)
            for (int i = 0; i < s.n; ++i)
                if (p.referenced(i, j)) p.mem[p.idx(p.c, it, i, j)] = it < reps ? u(gen) : p.mem[p.idx(p.c, r, i, j)];
    }
    p.mem0.assign(p.mem.begin(), p.mem.end());
    return p;
}

// The referenced triangle against a double reference; every other element bit for bit; for a
// repeating batch, every item bit-identical to its representative.
template <typename T>
void expect_syr2k(const Prob<T>& p, const std::string& what) {
    const Spec& s = p.s;
    const bool nt = s.trans == Transpose::NoTrans;
    auto opa = [&](int it, int i, int l) { return double(p.mem0[nt ? p.idx(p.a, it, i, l) : p.idx(p.a, it, l, i)]); };
    auto opb = [&](int it, int i, int l) { return double(p.mem0[nt ? p.idx(p.b, it, i, l) : p.idx(p.b, it, l, i)]); };
    const double eps = std::numeric_limits<T>::epsilon();
    std::vector<int> items;
    if (s.period > 0) for (int it = 0; it < std::min(s.period, s.batch); ++it) items.push_back(it);
    else if (s.batch <= 6) for (int it = 0; it < s.batch; ++it) items.push_back(it);
    else items = {0, 1, s.batch / 2, s.batch - 1};
    for (int it : items)
        for (int j = 0; j < s.n; ++j)
            for (int i = 0; i < s.n; ++i) {
                if (!p.referenced(i, j)) continue;
                double acc = 0, mag = 0;
                for (int l = 0; l < s.k; ++l) {
                    const double t = opa(it, i, l) * opb(it, j, l) + opb(it, i, l) * opa(it, j, l);
                    acc += t;
                    mag += std::abs(opa(it, i, l) * opb(it, j, l)) + std::abs(opb(it, i, l) * opa(it, j, l));
                }
                const double c0 = double(p.mem0[p.idx(p.c, it, i, j)]);
                const double ref = double(p.alpha) * acc + double(p.beta) * c0;
                const double bound = 8.0 * (s.k + 2) * eps * (std::abs(double(p.alpha)) * mag + std::abs(double(p.beta) * c0)) + 1e-30;
                const double got = double(p.mem[p.idx(p.c, it, i, j)]);
                ASSERT_LE(std::abs(got - ref), bound) << what << " item " << it << " (" << i << "," << j << ")";
            }
    std::vector<char> out(p.mem.size(), 0);
    for (int it = 0; it < s.batch; ++it)
        for (int j = 0; j < s.n; ++j)
            for (int i = 0; i < s.n; ++i)
                if (p.referenced(i, j)) out[p.idx(p.c, it, i, j)] = 1;
    for (std::size_t e = 0; e < p.mem.size(); ++e)
        if (!out[e]) ASSERT_TRUE(same_bits(p.mem[e], p.mem0[e])) << what << ": wrote element " << e << " outside C's triangle";
    if (s.period > 0)
        for (int it = s.period; it < s.batch; ++it)
            for (int j = 0; j < s.n; ++j)
                for (int i = 0; i < s.n; ++i)
                    if (p.referenced(i, j))
                        ASSERT_TRUE(same_bits(p.mem[p.idx(p.c, it, i, j)], p.mem[p.idx(p.c, it % s.period, i, j)]))
                            << what << ": item " << it << " differs from its representative at (" << i << "," << j << ")";
}

// The outermost syr2k trace line for whatever `run` calls, and the choice it names.
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
        if (line.rfind("syr2k ", 0) == 0 && line.find(" -> ") != std::string::npos) return line;
    return "<no syr2k trace line in: " + err + ">";
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

// A heterogeneous operand has no route on any backend: the tile launch has one (n, k, ld, stride),
// and
// every vendor loop (cuBLAS, rocBLAS, netlib) runs each item at the top-level extents. A
// spelling pin throws invalid_argument (the vendor class word falls back to Auto), Auto throws
// runtime_error, or NoRouteError without a vendor library, and C is left untouched.
template <Backend B, class T>
void expect_heterogeneous_has_no_route(Queue& ctx) {
    const int n = 16, k = 3, batch = 4;
    Matrix<T, MatrixFormat::Dense> A(n, k, batch), Bm(n, k, batch), Cm(n, n, batch);
    A.fill(T(0.5));
    Bm.fill(T(0.25));
    Cm.fill(T(1));
    UnifiedVector<int> rows(batch), cols(batch);
    for (int b = 0; b < batch; ++b) rows[b] = n - b, cols[b] = k - (b % 2);
    const auto hA = A.view().with_active_dims(rows.to_span(), cols.to_span());
    const auto hB = Bm.view().with_active_dims(rows.to_span(), cols.to_span());
    const auto hC = Cm.view().with_active_dims(rows.to_span(), rows.to_span());
    ASSERT_TRUE(hA.is_heterogeneous() && hB.is_heterogeneous() && hC.is_heterogeneous());
    struct Case { const char* what; MVof<T> a, b, c; };
    const Case cases[] = {{"A heterogeneous", hA, Bm.view(), Cm.view()},
                          {"C heterogeneous", A.view(), Bm.view(), hC},
                          {"all three, consistent per item", hA, hB, hC}};
    for (const auto& kc : cases) {
        auto call = [&] {
            (void)syr2k<B, T>(ctx, kc.a, kc.b, kc.c, T(1), T(0), Uplo::Lower, Transpose::NoTrans);
            ctx.wait();
        };
        auto expect_no_route = [&](const std::string& what) {
            if constexpr (select::level3_vendor_available<B>) EXPECT_THROW(call(), std::runtime_error) << what;
            else EXPECT_THROW(call(), batchlas::NoRouteError) << what;
        };
        for (const C& c : s2::candidates<T>()) {
            const Pin pin("syr2k", c);
            if (std::holds_alternative<s2::Vendor>(c)) expect_no_route(std::string(kc.what) + ": vendor pin");
            else EXPECT_THROW(call(), std::invalid_argument) << kc.what << ": " << select::to_string(c);
        }
        const ScopedEnvVar clear("BATCHLAS_SYR2K_ROUTE", nullptr);
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
class Syr2kCandidates : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    static constexpr Backend B = Config::BackendVal;
    static constexpr bool kVendor = batchlas::select::level3_vendor_available<B>;
    static constexpr bool kFloat = std::is_same_v<T, float>;

    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        if (this->HasFatalFailure() || ::testing::Test::IsSkipped()) return;
        if (!this->ctx) GTEST_SKIP() << "no queue";
        if (this->ctx->device().type != DeviceType::GPU) GTEST_SKIP() << "the tile kernel is a GPU kernel";
    }

    // The limit oracle, from the kernel's launch geometry rather than syr2k.cc's can_run.
    static bool expect_runs(const C& c, const Spec& s) {
        if (std::holds_alternative<s2::Vendor>(c)) return kVendor;
        const long t = (s.n + 127) / 128;
        return kFloat && B == Backend::CUDA && s.trans != Transpose::ConjTrans && s.n >= 1 && s.k >= 1 &&
               s.batch >= 1 && s.batch <= 65535 && t * (t + 1) / 2 <= 65535;
    }
    // `vendor` is a class word: with no vendor it falls back to Auto (with a warning), §5.3.
    static bool vendor_word_falls_back(const C& c) { return std::holds_alternative<s2::Vendor>(c) && !kVendor; }

    void run(Prob<T>& p) {
        (void)syr2k<B, T>(*this->ctx, p.A(), p.B(), p.Cv(), p.alpha, p.beta, p.s.uplo, p.s.trans);
        this->ctx->wait();
    }
    void run_pinned(const C& c, Prob<T>& p) {
        const Pin pin("syr2k", c);
        run(p);
    }
    bool pin_accepted(const C& c, Prob<T>& p) {
        const Pin pin("syr2k", c);
        try {
            run(p);
            return true;
        } catch (const std::invalid_argument& e) {
            const std::string w = e.what();
            if (w.find("cannot run this shape") == std::string::npos &&
                w.find("is not a compiled") == std::string::npos)
                throw;
            return false;
        }
    }
    std::string auto_choice(const Spec& s) {
        const ScopedEnvVar clear("BATCHLAS_SYR2K_ROUTE", nullptr);
        auto p = make_prob<T>(s);
        const std::string got = traced_choice([&] { run(p); });
        expect_syr2k(p, "auto " + label(s));
        return got;
    }
};

TYPED_TEST_SUITE(Syr2kCandidates, Types);

// §8.1: each candidate across the tile edges (127|128|129, 256|257), k under, at and over one
// staging step (7|8|9), batch 1|2|3, both triangles and N/T. Accepted shapes run and are
// correct; refused ones throw from the pin; the vendor word without a vendor runs Auto.
TYPED_TEST(Syr2kCandidates, PinnedCandidatesStraddleTheirLimits) {
    using T = typename TestFixture::T;
    for (const C& c : {C{s2::Triangular{}}, C{s2::Vendor{}}})
        for (int n : {1, 127, 128, 129, 257})
            for (int k : {1, 7, 8, 9})
                for (int batch : {1, 2, 3}) {
                    const Spec s{n % 2 ? Uplo::Lower : Uplo::Upper, k % 2 ? Transpose::NoTrans : Transpose::Trans,
                                 n, k, batch, 0, false, unsigned(n * 31 + k * 7 + batch)};
                    auto p = make_prob<T>(s);
                    if (TestFixture::vendor_word_falls_back(c)) continue;
                    const bool want = TestFixture::expect_runs(c, s);
                    EXPECT_EQ(this->pin_accepted(c, p), want) << name(c, s);
                    if (want) expect_syr2k(p, name(c, s));
                }
}

// Every (uplo, trans) on every candidate, at rows != cols operands with padded ld and stride,
// alpha != 1, beta != 0. Real ConjTrans is refused by the tile kernel (the old reach) and runs
// on the vendor, or has no route vendor-free.
TYPED_TEST(Syr2kCandidates, EveryCombinationOnEveryCandidate) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
        for (Transpose t : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans}) {
            const Spec s{uplo, t, 141, 37, 3, 0, false, 7u};
            for (const C& c : s2::candidates<T>()) {
                if (TestFixture::vendor_word_falls_back(c)) continue;
                auto p = make_prob<T>(s);
                const bool want = TestFixture::expect_runs(c, s);
                EXPECT_EQ(this->pin_accepted(c, p), want) << name(c, s);
                if (want) expect_syr2k(p, name(c, s));
            }
            if (t != Transpose::ConjTrans) continue;
            const ScopedEnvVar clear("BATCHLAS_SYR2K_ROUTE", nullptr);
            auto p = make_prob<T>(s);
            if constexpr (TestFixture::kVendor) {
                EXPECT_EQ(traced_choice([&] { this->run(p); }), "vendor") << label(s);
                expect_syr2k(p, "auto " + label(s));
            } else {
                EXPECT_THROW(this->run(p), batchlas::NoRouteError) << label(s);
            }
            (void)B;
        }
}

// The pinned vendor is exactly backend::syr2k_vendor (bit for bit on the same input), and the
// pinned triangular launches the tile kernel and nothing else (kernel trace, in a child that has
// BATCHLAS_KERNEL_TRACE from the start). A swapped launch arm shows here even when both kernels
// would be correct.
TYPED_TEST(Syr2kCandidates, PinnedChoiceRunsItsOwnKernel) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    const Spec s{Uplo::Upper, Transpose::Trans, 130, 19, 3, 0, false, 11u};
    if constexpr (TestFixture::kVendor) {
        auto p = make_prob<T>(s), q = make_prob<T>(s);
        this->run_pinned(C{s2::Vendor{}}, p);
        (void)backend::syr2k_vendor<B, T>(*this->ctx, q.A(), q.B(), q.Cv(), q.alpha, q.beta, s.uplo, s.trans);
        this->ctx->wait();
        for (std::size_t e = 0; e < p.mem.size(); ++e)
            ASSERT_TRUE(same_bits(p.mem[e], q.mem[e])) << "pinned vendor != syr2k_vendor at element " << e;
    }
    if constexpr (!TestFixture::kFloat) return;
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    const std::string path = ::testing::TempDir() + "syr2k_ktrace.json";
    std::filesystem::remove(path);
    const ScopedEnvVar on("BATCHLAS_KERNEL_TRACE", "1");
    const ScopedEnvVar where("BATCHLAS_KERNEL_TRACE_PATH", path.c_str());
    auto child = [&] {
        auto p = make_prob<T>(s);
        this->run_pinned(C{s2::Triangular{}}, p);
        if constexpr (TestFixture::kVendor) {
            auto q = make_prob<T>(s);
            this->run_pinned(C{s2::Vendor{}}, q);
        }
        std::exit(0);
    };
    EXPECT_EXIT(child(), ::testing::ExitedWithCode(0), "");
    int tiles = 0, other = 0;
    std::ifstream in(path);
    for (std::string line; std::getline(in, line);) {
        if (line.find("\"name\":\"") == std::string::npos) continue;
        (line.find("\"name\":\"syr2k_cuda_custom.triangular_tiles\"") != std::string::npos ? tiles : other)++;
    }
    std::filesystem::remove(path);
    EXPECT_EQ(tiles, 1) << "the triangular pin launches the tile kernel once; the vendor pin never";
    EXPECT_EQ(other, 0) << "no other native kernel ran";
}

// The tile kernel stages through shared memory: 1024 items repeating 3 distinct problems, on
// both legs (packed 256x64 is the unpredicated one, padded 200x53 the predicated one with a
// partial diagonal tile). Each representative is correct and every other item matches it bit
// for bit.
TYPED_TEST(Syr2kCandidates, SaturatingBatchIsBitIdentical) {
    using T = typename TestFixture::T;
    if constexpr (!TestFixture::kFloat) GTEST_SKIP() << "no native family for this dtype";
    for (const Spec& s : {Spec{Uplo::Lower, Transpose::NoTrans, 256, 64, 1024, 3, true, 5u},
                          Spec{Uplo::Upper, Transpose::Trans, 200, 53, 1024, 3, false, 6u}}) {
        auto p = make_prob<T>(s);
        this->run_pinned(C{s2::Triangular{}}, p);
        expect_syr2k(p, name(C{s2::Triangular{}}, s));
    }
}

// The grid terms (R3, AGENTS §8.9): the batch is grid z, so a pin launches at 65535 and is
// refused at 65536 (without the term that launch throws from the driver; the break is in
// docs/design/flat-kernel-selection.md §12). Auto past it takes the vendor (no route vendor-free).
// The tile list is grid y: a launch at n = 46208 (361 tiles a side, 65341 tiles), a refusal at
// 46209 (65703 tiles).
TYPED_TEST(Syr2kCandidates, GridCeilingsAreCanRunTerms) {
    using T = typename TestFixture::T;
    if constexpr (!TestFixture::kFloat) GTEST_SKIP() << "no native family for this dtype";
    else {
        Queue& q = *this->ctx;
        const int ceiling = int(s2::kMaxGridBatch);
        for (int batch : {ceiling, ceiling + 1}) {
            UnifiedVector<T> a(std::size_t(batch) * 2, T(0.5)), c(std::size_t(batch) * 4, T(1));
            const MVof<T> A(a.data(), 2, 1, 2, 2, batch), Cm(c.data(), 2, 2, 2, 4, batch);
            const bool fits = batch <= ceiling;
            {
                const Pin pin("syr2k", C{s2::Triangular{}});
                if (fits) {
                    (void)syr2k<TestFixture::B, T>(q, A, A, Cm, T(1), T(0), Uplo::Lower, Transpose::NoTrans);
                    q.wait();
                    EXPECT_EQ(c[0], T(0.5)) << "batch " << batch;
                    EXPECT_EQ(c[std::size_t(batch) * 4 - 1], T(0.5)) << "batch " << batch;
                    EXPECT_EQ(c[std::size_t(batch) * 4 - 2], T(1)) << "the upper element was written";
                } else {
                    EXPECT_THROW(((void)syr2k<TestFixture::B, T>(q, A, A, Cm, T(1), T(0), Uplo::Lower,
                                                                 Transpose::NoTrans)),
                                 std::invalid_argument);
                }
            }
            if (fits) continue;
            EXPECT_EQ(c[0], T(1)) << "a refused pin wrote C";
            const ScopedEnvVar clear("BATCHLAS_SYR2K_ROUTE", nullptr);
            auto call = [&] {
                (void)syr2k<TestFixture::B, T>(q, A, A, Cm, T(1), T(0), Uplo::Lower, Transpose::NoTrans);
                q.wait();
            };
            if constexpr (TestFixture::kVendor) EXPECT_EQ(traced_choice(call), "vendor");
            else EXPECT_THROW(call(), batchlas::NoRouteError);
        }
        const auto mem = q.device().get_property(DeviceProperty::GLOBAL_MEM_SIZE);
        if (mem < (std::size_t(16) << 30)) GTEST_SKIP() << "the tile ceiling needs ~9 GB for C";
        const int n_fit = 46208, n_past = 46209;
        sycl::queue& sq = batchlas::sycl_queue(q);
        T* cbuf = sycl::malloc_device<T>(std::size_t(n_past) * n_past, sq);
        T* abuf = sycl::malloc_device<T>(n_past, sq);
        ASSERT_NE(cbuf, nullptr);
        sq.fill(abuf, T(1), n_past).wait();
        sq.fill(cbuf, T(7), std::size_t(n_past) * n_past).wait();
        {
            const MVof<T> A(abuf, n_fit, 1, n_fit, n_fit, 1), Cm(cbuf, n_fit, n_fit, n_past, n_past * n_fit, 1);
            const Pin pin("syr2k", C{s2::Triangular{}});
            (void)syr2k<TestFixture::B, T>(q, A, A, Cm, T(1), T(0), Uplo::Lower, Transpose::NoTrans);
            q.wait();
            T probe[3];
            const std::size_t at[3] = {0, std::size_t(n_fit - 1) * n_past + n_fit - 1, std::size_t(n_fit - 2) * n_past + n_fit - 1};
            for (int i = 0; i < 3; ++i) sq.memcpy(&probe[i], cbuf + at[i], sizeof(T)).wait();
            EXPECT_EQ(probe[0], T(2)) << "first diagonal element";
            EXPECT_EQ(probe[1], T(2)) << "last diagonal element (the last tile ran)";
            EXPECT_EQ(probe[2], T(2)) << "last row, one column in";
        }
        {
            const MVof<T> A(abuf, n_past, 1, n_past, n_past, 1), Cm(cbuf, n_past, n_past, n_past, n_past * n_past, 1);
            const Pin pin("syr2k", C{s2::Triangular{}});
            EXPECT_THROW(((void)syr2k<TestFixture::B, T>(q, A, A, Cm, T(1), T(0), Uplo::Lower, Transpose::NoTrans)),
                         std::invalid_argument);
        }
        sycl::free(cbuf, sq);
        sycl::free(abuf, sq);
    }
}

// R6: a spelling that names nothing compiled throws instead of meaning Auto: the removed level-3
// words, cuBLASDx, and the other ops' family words.
TYPED_TEST(Syr2kCandidates, UnknownPinsThrow) {
    using T = typename TestFixture::T;
    for (const char* word : {"cublasdx", "tiles", "narrow", "gemm", "custom", "dx", "fused", "diag_full_gemm",
                             "triangular_tiles", "gram_tiles", "expand_gemm", "fused_device", "register_tiled",
                             "native:auto", "vendor:auto", "bogus", "gram", "expand", "cta", "blocked",
                             "triangular:1", "native:triangular"}) {
        auto p = make_prob<T>(Spec{});
        const Pin pin("syr2k", std::string_view(word));
        EXPECT_THROW(this->run(p), std::invalid_argument) << word;
        for (std::size_t e = 0; e < p.mem.size(); ++e)
            ASSERT_TRUE(same_bits(p.mem[e], p.mem0[e])) << word << ": a refused pin wrote element " << e;
    }
}

// The named can_run-false pins, each with its message: real ConjTrans, a heterogeneous operand,
// and a double triangular (not a compiled candidate: the kernel is float only).
TYPED_TEST(Syr2kCandidates, CanRunFalsePinsThrow) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    const C tri{s2::Triangular{}};
    if constexpr (!TestFixture::kFloat) {
        auto p = make_prob<T>(Spec{});
        const Pin pin("syr2k", tri);
        try {
            this->run(p);
            ADD_FAILURE() << "double triangular was accepted";
        } catch (const std::invalid_argument& e) {
            EXPECT_NE(std::string(e.what()).find("is not a compiled syr2k double candidate"), std::string::npos) << e.what();
        }
        return;
    }
    {
        auto p = make_prob<T>(Spec{Uplo::Lower, Transpose::ConjTrans, 16, 4, 2});
        const Pin pin("syr2k", tri);
        try {
            this->run(p);
            ADD_FAILURE() << "ConjTrans triangular was accepted";
        } catch (const std::invalid_argument& e) {
            EXPECT_NE(std::string(e.what()).find("cannot run this shape"), std::string::npos) << e.what();
        }
    }
    const int n = 16, k = 3, batch = 4;
    Matrix<T, MatrixFormat::Dense> A(n, k, batch), Bm(n, k, batch), Cm(n, n, batch);
    A.fill(T(0.5));
    Bm.fill(T(0.25));
    Cm.fill(T(1));
    UnifiedVector<int> rows(batch), cols(batch), crows(batch);
    for (int b = 0; b < batch; ++b) rows[b] = n - b, cols[b] = k, crows[b] = n - b;
    const auto hetA = A.view().with_active_dims(rows.to_span(), cols.to_span());
    const auto hetB = Bm.view().with_active_dims(rows.to_span(), cols.to_span());
    const auto hetC = Cm.view().with_active_dims(crows.to_span(), crows.to_span());
    struct Case { const char* what; MVof<T> a, b, c; };
    const Case cases[] = {{"A heterogeneous", hetA, Bm.view(), Cm.view()},
                          {"B heterogeneous", A.view(), hetB, Cm.view()},
                          {"C heterogeneous", A.view(), Bm.view(), hetC}};
    for (const auto& kc : cases) {
        const Pin pin("syr2k", tri);
        try {
            (void)syr2k<B, T>(*this->ctx, kc.a, kc.b, kc.c, T(1), T(0), Uplo::Lower, Transpose::NoTrans);
            this->ctx->wait();
            ADD_FAILURE() << kc.what << " triangular was accepted";
        } catch (const std::invalid_argument& e) {
            EXPECT_NE(std::string(e.what()).find("cannot run this shape"), std::string::npos) << kc.what << ": " << e.what();
        }
    }
}

// A heterogeneous operand on CUDA (expect_heterogeneous_has_no_route).
TYPED_TEST(Syr2kCandidates, HeterogeneousBatchHasNoRoute) {
    expect_heterogeneous_has_no_route<TestFixture::B, typename TestFixture::T>(*this->ctx);
}

// An empty batch launches nothing, under Auto, every pin and vendor-free (the cuBLAS loop threw
// CUBLAS error 7 on a batch of 0).
TYPED_TEST(Syr2kCandidates, EmptyBatchIsANoOp) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    UnifiedVector<T> a(64, T(1)), b(64, T(3)), c(64, T(2));
    const MVof<T> A(a.data(), 8, 4, 8, 32, 0), Bm(b.data(), 8, 4, 8, 32, 0), Cm(c.data(), 8, 8, 8, 64, 0);
    std::vector<std::optional<C>> pins{std::nullopt};
    for (const C& k : s2::candidates<T>()) pins.push_back(k);
    const ScopedEnvVar clear("BATCHLAS_SYR2K_ROUTE", nullptr);
    for (const auto& k : pins) {
        std::optional<Pin> pin;
        if (k) pin.emplace("syr2k", *k);
        EXPECT_NO_THROW(((void)syr2k<B, T>(*this->ctx, A, Bm, Cm, T(1), T(0), Uplo::Lower, Transpose::NoTrans),
                         this->ctx->wait()))
            << (k ? select::to_string(*k) : std::string("auto"));
    }
    for (std::size_t e = 0; e < c.size(); ++e) ASSERT_EQ(c[e], T(2)) << e;
}

// Shape validation runs before choose(): a batch or extent mismatch throws invalid_argument
// under Auto and every pin, vendor-free too, and touches nothing.
TYPED_TEST(Syr2kCandidates, InvalidShapesThrowBeforeChoosing) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    auto p = make_prob<T>(Spec{Uplo::Lower, Transpose::NoTrans, 8, 5, 3});
    std::vector<std::optional<C>> pins{std::nullopt};
    for (const C& c : s2::candidates<T>()) pins.push_back(c);
    struct Bad { const char* what; MVof<T> a, b, c; };
    const Bad bad[] = {
        {"B batch 2", p.A(), MVof<T>(p.mem.data() + p.b.off, 8, 5, p.b.ld, p.b.stride, 2), p.Cv()},
        {"C not square", p.A(), p.B(), MVof<T>(p.mem.data() + p.c.off, 8, 7, p.c.ld, p.c.stride, 3)},
        {"B k differs", p.A(), MVof<T>(p.mem.data() + p.b.off, 8, 4, p.b.ld, p.b.stride, 3), p.Cv()},
        {"A rows differ from C", MVof<T>(p.mem.data() + p.a.off, 7, 5, p.a.ld, p.a.stride, 3), p.B(), p.Cv()}};
    for (const auto& k : bad)
        for (const auto& c : pins) {
            std::optional<Pin> pin;
            if (c) pin.emplace("syr2k", *c);
            EXPECT_THROW(((void)syr2k<B, T>(*this->ctx, k.a, k.b, k.c, T(1), T(1), Uplo::Lower, Transpose::NoTrans)),
                         std::invalid_argument)
                << k.what << " under " << (c ? select::to_string(*c) : std::string("auto"));
        }
    for (std::size_t e = 0; e < p.mem.size(); ++e)
        ASSERT_TRUE(same_bits(p.mem[e], p.mem0[e])) << "a refused call wrote element " << e;
}

// §5.3: spellings and the class words, via ScopedPin and the environment (which is case-folded
// and trimmed). `native` is the row's best non-vendor; `vendor` and `native` with nothing of
// their class fall back to Auto with a warning.
TYPED_TEST(Syr2kCandidates, ClassWordsAndSpellings) {
    using T = typename TestFixture::T;
    const Spec s{Uplo::Lower, Transpose::NoTrans, 40, 9, 1};  // batch 1: the vendor's row
    const bool routed = TestFixture::kFloat || TestFixture::kVendor;
    const std::string auto_pick = !routed ? "<no route>" : (TestFixture::kVendor ? "vendor" : "triangular");
    const std::string native_pick = TestFixture::kFloat ? "triangular" : auto_pick;
    struct Want { const char* word; bool env; std::string pick; bool warns; };
    std::vector<Want> wants{{"auto", false, auto_pick, false},         {"auto", true, auto_pick, false},
                            {" Auto ", true, auto_pick, false},        {"vendor", false, auto_pick, !TestFixture::kVendor},
                            {"VENDOR", true, auto_pick, !TestFixture::kVendor},
                            {"native", false, native_pick, !TestFixture::kFloat},
                            {"native", true, native_pick, !TestFixture::kFloat}};
    if constexpr (TestFixture::kFloat) {
        wants.push_back({"triangular", false, "triangular", false});
        wants.push_back({" Triangular", true, "triangular", false});
    }
    for (const Want& w : wants) {
        auto p = make_prob<T>(s);
        select::testing::reset_warnings();
        std::string err, got;
        try {
            got = traced_choice([&] {
                const ScopedEnvVar env("BATCHLAS_SYR2K_ROUTE", w.env ? w.word : nullptr);
                std::optional<Pin> pin;
                if (!w.env) pin.emplace("syr2k", std::string_view(w.word));
                this->run(p);
            }, &err);
        } catch (const batchlas::NoRouteError&) {
            got = "<no route>";
        }
        const std::string what = "'" + std::string(w.word) + (w.env ? "' via BATCHLAS_SYR2K_ROUTE" : "' via ScopedPin");
        EXPECT_EQ(got, w.pick) << what;
        if (got != "<no route>") EXPECT_EQ(err.find("but no ") != std::string::npos, w.warns) << what << ": " << err;
        if (got == w.pick && got != "<no route>") expect_syr2k(p, what);
    }
}

// §5.3: a ScopedPin wins over BATCHLAS_SYR2K_ROUTE, and nested pins restore the outer one.
TYPED_TEST(Syr2kCandidates, ScopedPinBeatsTheEnvironment) {
    using T = typename TestFixture::T;
    if constexpr (!TestFixture::kFloat || !TestFixture::kVendor) GTEST_SKIP() << "needs two runnable candidates";
    const Spec s{Uplo::Upper, Transpose::Trans, 33, 6, 2};
    const ScopedEnvVar env("BATCHLAS_SYR2K_ROUTE", "vendor");
    {
        auto p = make_prob<T>(s);
        EXPECT_EQ(traced_choice([&] { this->run(p); }), "vendor") << "the environment pin was not read";
    }
    auto p = make_prob<T>(s);
    EXPECT_EQ(traced_choice([&] {
                  const Pin pin("syr2k", C{s2::Triangular{}});
                  this->run(p);
              }),
              "triangular");
    expect_syr2k(p, "triangular over env vendor");
    const Pin outer("syr2k", C{s2::Triangular{}});
    {
        const Pin inner("syr2k", C{s2::Vendor{}});
        auto r = make_prob<T>(s);
        EXPECT_EQ(traced_choice([&] { this->run(r); }), "vendor");
    }
    auto r = make_prob<T>(s);
    EXPECT_EQ(traced_choice([&] { this->run(r); }), "triangular") << "the inner pin did not restore the outer";
}

// Auto against the shipped transcription on either side of its only threshold (batch 1|2, the
// old prefer_triangular_tiles), at grid and off-grid n, k and batch: float batch >= 2 runs the
// tile kernel, batch 1 the vendor (vendor-free: the row's second entry, triangular); double is
// the vendor at every shape (vendor-free: no route).
TYPED_TEST(Syr2kCandidates, AutoReadsTheTranscribedTable) {
    using T = typename TestFixture::T;
    struct Cell { int n, k, batch; Transpose t; };
    const Cell cells[] = {{200, 53, 1, Transpose::NoTrans}, {200, 53, 2, Transpose::Trans},
                          {1, 1, 1, Transpose::Trans},       {1, 1, 2, Transpose::NoTrans},
                          {300, 4096, 1, Transpose::NoTrans}, {129, 9, 3, Transpose::Trans},
                          {64, 64, 128, Transpose::NoTrans},  {17, 600, 5, Transpose::Trans},
                          {700, 2, 1, Transpose::Trans},      {2, 700, 2, Transpose::NoTrans}};
    for (const Cell& c : cells) {
        const Spec s{Uplo::Lower, c.t, c.n, c.k, c.batch, 0, false, unsigned(c.n + c.k)};
        std::string want;
        if constexpr (TestFixture::kFloat) want = (c.batch >= 2 || !TestFixture::kVendor) ? "triangular" : "vendor";
        else want = TestFixture::kVendor ? "vendor" : "<no route>";
        std::string got;
        try {
            got = this->auto_choice(s);
        } catch (const batchlas::NoRouteError&) {
            got = "<no route>";
        }
        EXPECT_EQ(got, want) << label(s);
    }
}

// key_of's every field reaches choose(): a synthetic table whose winner changes with n, k (op(A)'s
// inner extent: A.cols under N, A.rows under T) and batch alone. Fixing a field in key_of, or
// taking k from the wrong extent, turns exactly its probe red. Needs both float candidates.
TYPED_TEST(Syr2kCandidates, AutoReadsEveryKeyField) {
    using T = typename TestFixture::T;
    if constexpr (!TestFixture::kFloat || !TestFixture::kVendor) GTEST_SKIP() << "needs two runnable candidates";
    const ScopedEnvVar clear("BATCHLAS_SYR2K_ROUTE", nullptr);
    const std::string dev = select::device_of<TestFixture::B>(*this->ctx).key;
    std::vector<std::pair<std::string, std::string>> files;
    for (const auto& t : select::embedded_tables())
        if (t.name.rfind("syr2k.", 0) != 0) files.emplace_back(std::string(t.name), std::string(t.text));
    files.emplace_back("syr2k.float." + dev + ".txt",
                       "# op=syr2k dtype=float device=" + dev + " kernels=unknown\n"
                       "# keys: n:log:2 k:log batch:log\n"
                       "n=16 k=4 batch=4 | triangular 1 | vendor 2\n"
                       "n=256 k=4 batch=4 | vendor 1 | triangular 2\n"
                       "n=16 k=256 batch=4 | vendor 1 | triangular 2\n"
                       "n=16 k=4 batch=1024 | vendor 1 | triangular 2\n");
    const TableGuard restore;
    select::testing::set_builtin_tables(std::move(files));
    struct Probe { int n, k, batch; Transpose t; const char* expect; const char* field; };
    const Probe probes[] = {{16, 4, 4, Transpose::NoTrans, "triangular", "base"},
                            {16, 4, 4, Transpose::Trans, "triangular", "base (T)"},
                            {256, 4, 4, Transpose::NoTrans, "vendor", "n"},
                            {16, 256, 4, Transpose::NoTrans, "vendor", "k (N: A.cols)"},
                            {16, 256, 4, Transpose::Trans, "vendor", "k (T: A.rows)"},
                            {16, 4, 1024, Transpose::Trans, "vendor", "batch"}};
    for (const auto& k : probes) {
        const Spec s{Uplo::Lower, k.t, k.n, k.k, k.batch, 0, false, 43u};
        auto p = make_prob<T>(s);
        EXPECT_EQ(traced_choice([&] { this->run(p); }), k.expect) << "the " << k.field << " probe";
        expect_syr2k(p, std::string("the ") + k.field + " probe");
    }
}

// The trace key: n is C's order and k op(A)'s inner extent, on both transposes.
TYPED_TEST(Syr2kCandidates, TraceKeyKFollowsTrans) {
    using T = typename TestFixture::T;
    if constexpr (!TestFixture::kFloat && !TestFixture::kVendor) GTEST_SKIP() << "no route";
    for (Transpose t : {Transpose::NoTrans, Transpose::Trans}) {
        const Spec s{Uplo::Upper, t, 12, 37, 5};
        auto p = make_prob<T>(s);
        const std::string line = traced_line([&] { this->run(p); });
        EXPECT_NE(line.find("n=12 k=37 batch=5 ->"), std::string::npos) << line;
        expect_syr2k(p, line);
    }
}

// With a table naming only the vendor, a vendor-free float build takes the last resort,
// triangular; with the vendor present the row's vendor runs.
TYPED_TEST(Syr2kCandidates, VendorFreeLastResortIsTriangular) {
    using T = typename TestFixture::T;
    if constexpr (!TestFixture::kFloat) GTEST_SKIP() << "no native family for this dtype";
    const ScopedEnvVar clear("BATCHLAS_SYR2K_ROUTE", nullptr);
    const std::string dev = select::device_of<TestFixture::B>(*this->ctx).key;
    std::vector<std::pair<std::string, std::string>> files;
    for (const auto& t : select::embedded_tables())
        if (t.name.rfind("syr2k.", 0) != 0) files.emplace_back(std::string(t.name), std::string(t.text));
    files.emplace_back("syr2k.float." + dev + ".txt", "# op=syr2k dtype=float device=" + dev +
                                                          " kernels=unknown\n# keys: n:log:2 k:log batch:log\n"
                                                          "n=64 k=8 batch=128 | vendor 1\n");
    const TableGuard restore;
    select::testing::set_builtin_tables(std::move(files));
    const Spec s{Uplo::Lower, Transpose::Trans, 70, 9, 6};
    auto p = make_prob<T>(s);
    std::string all;
    const std::string got = traced_choice([&] { this->run(p); }, &all);
    EXPECT_EQ(got, TestFixture::kVendor ? "vendor" : "triangular");
    if (!TestFixture::kVendor) EXPECT_NE(all.find("last resort"), std::string::npos) << all;
    expect_syr2k(p, "vendor-only table");
}

// Nothing runnable vendor-free (double) throws NoRouteError and records the coverage miss row.
// threadsafe: the child re-executes the binary, so CUDA is initialised fresh, never forked.
TYPED_TEST(Syr2kCandidates, NoRouteRecordsAMissRow) {
    using T = typename TestFixture::T;
    if constexpr (TestFixture::kVendor || TestFixture::kFloat) GTEST_SKIP() << "a route exists";
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    const std::string dir = ::testing::TempDir() + "syr2k_miss";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const std::string out = dir + "/cov";
    auto child = [&] {
        const ScopedEnvVar cov("BATCHLAS_COVERAGE_OUT", out.c_str());
        batchlas::coverage::g_dynamic_enabled = true;
        auto p = make_prob<T>(Spec{});
        try {
            this->run(p);
        } catch (const batchlas::NoRouteError&) {
            std::exit(0);
        }
        std::exit(1);
    };
    EXPECT_EXIT(child(), ::testing::ExitedWithCode(0), "");
    int miss = 0;
    for (const auto& ent : std::filesystem::directory_iterator(dir)) {
        std::ifstream in(ent.path());
        for (std::string line; std::getline(in, line);)
            if (line.rfind("miss,syr2k,double,", 0) == 0) ++miss;
    }
    std::filesystem::remove_all(dir);
    EXPECT_EQ(miss, 1);
}

// The coverage row (§5.6): the real backend and scalar (the old recorder hard-coded F32/CUDA),
// the old key (m = n = C's order, k = op(A)'s inner extent, uplo, transA, side Left, diag
// NonUnit) and the native flags. threadsafe: the child re-executes the binary.
TYPED_TEST(Syr2kCandidates, CoverageRowCarriesBackendKeyAndNativeFlags) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    if constexpr (!TestFixture::kFloat && !TestFixture::kVendor) GTEST_SKIP() << "no route";
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    const std::string dt(select::dtype_name<T>());
    const std::string dir = ::testing::TempDir() + "syr2k_cov." + dt;
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const std::string out = dir + "/cov";
    const Spec lo{Uplo::Lower, Transpose::NoTrans, 20, 7, 2};
    const Spec hi{Uplo::Upper, Transpose::Trans, 48, 3, 1};
    auto child = [&] {
        const ScopedEnvVar cov("BATCHLAS_COVERAGE_OUT", out.c_str());
        const ScopedEnvVar clear("BATCHLAS_SYR2K_ROUTE", nullptr);
        batchlas::coverage::g_dynamic_enabled = true;
        for (const Spec& s : {lo, hi}) {
            auto p = make_prob<T>(s);
            this->run(p);
        }
        std::exit(0);
    };
    EXPECT_EXIT(child(), ::testing::ExitedWithCode(0), "");
    std::map<std::string, std::vector<std::string>> rows;  // "m n k" -> fields
    for (const auto& ent : std::filesystem::directory_iterator(dir)) {
        std::ifstream in(ent.path());
        for (std::string line; std::getline(in, line);) {
            if (line.rfind("reached,syr2k,", 0) != 0) continue;
            std::vector<std::string> f;
            std::stringstream ss(line);
            for (std::string tok; std::getline(ss, tok, ',');) f.push_back(tok);
            ASSERT_GE(f.size(), 19u) << line;
            rows[f[5] + " " + f[6] + " " + f[7]] = f;
        }
    }
    std::filesystem::remove_all(dir);
    ASSERT_EQ(rows.size(), 2u);
    const std::string backend = B == Backend::CUDA ? "CUDA" : (B == Backend::ROCM ? "ROCM" : "?");
    auto as_int = [](auto e) { return std::to_string(static_cast<int>(e)); };
    const bool f = TestFixture::kFloat;
    const std::string algo_lo = f ? "triangular" : "vendor";
    const std::string algo_hi = f && !TestFixture::kVendor ? "triangular" : "vendor";
    for (const auto& [key, s, algo] : {std::tuple{std::string("20 20 7"), lo, algo_lo},
                                       std::tuple{std::string("48 48 3"), hi, algo_hi}}) {
        ASSERT_TRUE(rows.count(key)) << "no " << key << " row";
        const auto& r = rows[key];
        EXPECT_EQ(r[2], dt) << key << ": the scalar column";
        EXPECT_EQ(r[3], backend) << key << ": the backend column";
        EXPECT_EQ(r[10], algo) << key;
        EXPECT_EQ(r[12], f ? "1" : "0") << key << ": native_route_existed";
        EXPECT_EQ(r[13], f ? "1" : "0") << key << ": native_route_supported";
        EXPECT_EQ(r[15], as_int(s.uplo)) << key;
        EXPECT_EQ(r[16], as_int(Side::Left)) << key;
        EXPECT_EQ(r[17], as_int(Diag::NonUnit)) << key;
        EXPECT_EQ(r[18], as_int(s.trans)) << key;
    }
}

// On a CPU queue no native family can run: the triangular pin throws and Auto is the vendor
// (or no route without one).
template <typename Config>
class Syr2kCandidatesCpu : public test_utils::BatchLASTest<Config> {};
TYPED_TEST_SUITE(Syr2kCandidatesCpu, Types);

TYPED_TEST(Syr2kCandidatesCpu, CpuQueueRunsNoNativeFamily) {
    using T = typename TypeParam::ScalarType;
    static constexpr Backend B = TypeParam::BackendVal;
    if (!this->ctx) GTEST_SKIP() << "no queue";
    if (this->ctx->device().type == DeviceType::GPU) GTEST_SKIP() << "a GPU queue";
    const ScopedEnvVar clear("BATCHLAS_SYR2K_ROUTE", nullptr);
    const Spec s{Uplo::Lower, Transpose::Trans, 9, 4, 2};
    if constexpr (std::is_same_v<T, float>) {
        auto p = make_prob<T>(s);
        const Pin pin("syr2k", C{s2::Triangular{}});
        EXPECT_THROW(((void)syr2k<B, T>(*this->ctx, p.A(), p.B(), p.Cv(), p.alpha, p.beta, s.uplo, s.trans)),
                     std::invalid_argument);
    }
    auto p = make_prob<T>(s);
    auto call = [&] {
        (void)syr2k<B, T>(*this->ctx, p.A(), p.B(), p.Cv(), p.alpha, p.beta, s.uplo, s.trans);
        this->ctx->wait();
    };
    if constexpr (batchlas::select::level3_vendor_available<B>) {
        EXPECT_EQ(traced_choice(call), "vendor");
        expect_syr2k(p, "cpu vendor");
    } else {
        EXPECT_THROW(call(), batchlas::NoRouteError);
    }
}

// The netlib loop shares the vendor loops' single top-level (n, k), so a heterogeneous batch has
// no route on a CPU queue either.
TYPED_TEST(Syr2kCandidatesCpu, HeterogeneousBatchHasNoRoute) {
    if (!this->ctx) GTEST_SKIP() << "no queue";
    if (this->ctx->device().type == DeviceType::GPU) GTEST_SKIP() << "a GPU queue";
    expect_heterogeneous_has_no_route<TypeParam::BackendVal, typename TypeParam::ScalarType>(*this->ctx);
}

// The shipped tables (no GPU): one row per grid cell of choice.hh on sm_89 and sm_120, the same
// rows on both, untimed, source transcribed:ff340fc6; float `triangular | vendor` from batch 2
// and `vendor | triangular` at batch 1, double `vendor` everywhere; spot rows through nearest().
TEST(Syr2kTranscribedTable, RowsAreTheGridAndHoldTheOldRule) {
    std::map<std::string, std::vector<std::string>> by_dev;
    for (const char* dt : {"float", "double"})
        for (const char* dev : {"sm_89", "sm_120"}) {
            const auto tables = select::tables_in_borrow_order("syr2k", dt, select::device_from_key(dev));
            ASSERT_FALSE(tables.empty()) << dt << " " << dev;
            const select::Table& t = *tables.front();
            ASSERT_EQ(t.device, dev) << t.file;
            EXPECT_EQ(t.source, "transcribed:ff340fc6") << t.file;
            std::string keys = "# keys:";
            for (auto k : s2::key_names) keys += " " + std::string(k);
            const std::string file = "syr2k." + std::string(dt) + "." + dev + ".txt";
            const auto& all = select::embedded_tables();
            const auto e = std::find_if(all.begin(), all.end(), [&](const auto& x) { return x.name == file; });
            ASSERT_NE(e, all.end()) << file;
            EXPECT_NE(std::string(e->text).find("\n" + keys + "\n"), std::string::npos) << file;
            ASSERT_EQ(t.rows.size(), s2::grid_n.size() * s2::grid_k.size() * s2::grid_batch.size()) << t.file;
            std::size_t i = 0;
            for (int n : s2::grid_n)
                for (int k : s2::grid_k)
                    for (int b : s2::grid_batch) {
                        const select::TableRow& row = t.rows[i++];
                        const std::string want_key = std::to_string(n) + " " + std::to_string(k) + " " + std::to_string(b);
                        ASSERT_EQ(row.keys.size(), 3u) << t.file << ":" << row.line;
                        EXPECT_EQ(row.keys[0] + " " + row.keys[1] + " " + row.keys[2], want_key)
                            << t.file << ":" << row.line;
                        std::vector<std::string> ranked;
                        for (const auto& e : row.ranked) ranked.push_back(e.spelling);
                        const std::vector<std::string> want =
                            std::string(dt) == "double" ? std::vector<std::string>{"vendor"}
                            : b >= 2 ? std::vector<std::string>{"triangular", "vendor"}
                                     : std::vector<std::string>{"vendor", "triangular"};
                        EXPECT_EQ(ranked, want) << t.file << ":" << row.line;
                        EXPECT_FALSE(row.timed) << t.file << ":" << row.line;
                        const select::TableRow* hit = t.nearest({{"n", n}, {"k", k}, {"batch", b}});
                        ASSERT_EQ(hit, &row) << t.file << ": grid cell n=" << n << " k=" << k << " batch=" << b;
                        by_dev[std::string(dt) + dev].push_back(std::to_string(row.line) + ":" + ranked.front());
                    }
            const select::TableRow* off = t.nearest({{"n", 300}, {"k", 33}, {"batch", 2}});
            ASSERT_NE(off, nullptr);
            EXPECT_EQ(off->ranked.front().spelling, std::string(dt) == "double" ? "vendor" : "triangular") << t.file;
            off = t.nearest({{"n", 3000}, {"k", 2}, {"batch", 1}});
            EXPECT_EQ(off->ranked.front().spelling, "vendor") << t.file;
        }
    EXPECT_EQ(by_dev["floatsm_89"], by_dev["floatsm_120"]);
    EXPECT_EQ(by_dev["doublesm_89"], by_dev["doublesm_120"]);
}

}  // namespace
