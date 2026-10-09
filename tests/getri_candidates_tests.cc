// Every getri candidate, pinned: docs/design/flat-kernel-selection.md §8. The limit oracle is the
// drivers' own checks; which kernel ran is read back from the select trace or a bit-for-bit
// comparison with the direct driver, never assumed from the pin being accepted. The old
// RouteGetri.* cases of route_vocabulary_tests and getrf_tests' getri route readbacks live here.
#include <gtest/gtest.h>

#include <batchlas/backend_config.h>
#include <batchlas/blas/functions/getrf.hh>
#include <batchlas/blas/functions/getri.hh>
#include <batchlas/blas/functions/trsm.hh>
#include "../src/select/coverage.hh"
#include <batchlas/no_route.hh>
#include "../src/select/vendor.hh"
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>

#include "test_utils.hh"
#include "shipped_table_pick.hh"

#include "../src/ops/getri/choice.hh"
#include "../src/extensions/getri_native.hh"

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
#include <variant>
#include <vector>

using namespace batchlas;

namespace {

namespace gi = batchlas::ops::getri;
using C = gi::GetriChoice;
using Pin = select::ScopedPin<C>;
template <typename T>
using MVof = MatrixView<T, MatrixFormat::Dense>;
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
std::complex<double> up(T v) {
    if constexpr (kCx<T>) return {double(v.real()), double(v.imag())};
    else return {double(v), 0.0};
}
template <typename T>
bool same_bits(T a, T b) {
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}

// The old router's preference (route_getri.hh at 424a45bc, the transcribed commit): Blocked for
// float from order 128 and cfloat from order 256, the vendor otherwise. Written from the deleted
// predicate, not read from the tables, so a table that drifts from it goes red.
template <typename T>
const char* old_auto(int n) {
    if constexpr (std::is_same_v<T, float>) return n >= 128 ? "blocked" : "vendor";
    else if constexpr (std::is_same_v<T, std::complex<float>>) return n >= 256 ? "blocked" : "vendor";
    else return "vendor";
}
const char* old_auto_dtype(const std::string& dt, int n) {
    if (dt == "float") return old_auto<float>(n);
    if (dt == "cfloat") return old_auto<std::complex<float>>(n);
    return "vendor";
}

struct Spec {
    int n = 8, batch = 3;
    int period = 0;  // > 0: item it repeats item it % period
    unsigned seed = 1;
};

// A (factored in place by the public getrf) and C share one buffer of a large finite poison,
// at padded lds and strides that are not ld * n. A must survive getri bit for bit; C's pads
// and the gaps must too.
template <typename T>
struct Inv {
    Spec s;
    int lda = 0, sa = 0, ldc = 0, sc = 0;
    std::size_t coff = 0;
    UnifiedVector<T> mem;
    std::vector<T> mem0;                   // after getrf, before getri
    std::vector<std::complex<double>> a0;  // the matrix getrf factored, column-major per item
    UnifiedVector<int64_t> piv;
    UnifiedVector<int32_t> info;
    UnifiedVector<T*> aptr, cptr;

    MVof<T> A() { return {mem.data(), s.n, s.n, lda, sa, s.batch, aptr.data()}; }
    MVof<T> Cv() { return {mem.data() + coff, s.n, s.n, ldc, sc, s.batch, cptr.data()}; }
    std::size_t ai(int it, int i, int j) const { return std::size_t(it) * sa + std::size_t(j) * lda + i; }
    std::size_t ci(int it, int i, int j) const { return coff + std::size_t(it) * sc + std::size_t(j) * ldc + i; }
};

template <typename T>
T poison() {
    return mk<T>(RealOf<T>(-9.75e3), RealOf<T>(4.5e3));
}

// Strictly diagonally dominant with its rows permuted, so getrf pivots; every entry has a
// nonzero imaginary part.
template <typename T>
void fill_a(Inv<T>& p) {
    using R = RealOf<T>;
    const Spec& s = p.s;
    std::mt19937 gen(s.seed);
    std::uniform_real_distribution<R> u(R(-1), R(1));
    const int reps = s.period > 0 ? std::min(s.period, s.batch) : s.batch;
    p.a0.assign(std::size_t(reps) * s.n * s.n, 0.0);
    for (int it = 0; it < reps; ++it) {
        std::vector<int> perm(s.n);
        for (int i = 0; i < s.n; ++i) perm[i] = i;
        std::shuffle(perm.begin(), perm.end(), gen);
        for (int j = 0; j < s.n; ++j)
            for (int i = 0; i < s.n; ++i) {
                const T v = i == j ? mk<T>(R(s.n) + R(1.5), R(0.25)) : mk<T>(u(gen), u(gen));
                p.mem[p.ai(it, perm[i], j)] = v;
                p.a0[std::size_t(it) * s.n * s.n + std::size_t(j) * s.n + perm[i]] = up(v);
            }
    }
    for (int it = reps; it < s.batch; ++it)
        for (int j = 0; j < s.n; ++j)
            for (int i = 0; i < s.n; ++i) p.mem[p.ai(it, i, j)] = p.mem[p.ai(it % reps, i, j)];
}

template <typename T>
double tol(int n) {
    return 64.0 * std::max(n, 1) * double(std::numeric_limits<RealOf<T>>::epsilon());
}

// ||A0 C - I||_F / (||A0||_F ||C||_F) for one item, in double.
template <typename T>
double residual(const Inv<T>& p, int it) {
    const int n = p.s.n;
    const int rep = p.s.period > 0 ? it % std::min(p.s.period, p.s.batch) : it;
    const std::complex<double>* a = p.a0.data() + std::size_t(rep) * n * n;
    double num = 0, an = 0, cn = 0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            std::complex<double> acc = i == j ? -1.0 : 0.0;
            for (int t = 0; t < n; ++t) acc += a[std::size_t(t) * n + i] * up(p.mem[p.ci(it, t, j)]);
            num += std::norm(acc);
            an += std::norm(a[std::size_t(j) * n + i]);
            cn += std::norm(up(p.mem[p.ci(it, i, j)]));
        }
    return std::sqrt(num) / std::sqrt(an * cn);
}

std::string label(const Spec& s) { return "n=" + std::to_string(s.n) + " batch=" + std::to_string(s.batch); }
std::string name(const C& c, const Spec& s) { return select::to_string(c) + " " + label(s); }

// The outermost getri trace line for whatever `run` calls, and the choice it names.
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
        if (line.rfind("getri ", 0) == 0 && line.find(" -> ") != std::string::npos) return line;
    return "<no getri trace line in: " + err + ">";
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

// The embedded tables with getri's replaced by one synthetic table for `dev`.
void install_getri_table(const std::string& dtype, const std::string& dev, const std::string& rows) {
    std::vector<std::pair<std::string, std::string>> files;
    for (const auto& t : select::embedded_tables())
        if (t.name.rfind("getri.", 0) != 0) files.emplace_back(std::string(t.name), std::string(t.text));
    files.emplace_back("getri." + dtype + "." + dev + ".txt", "# op=getri dtype=" + dtype + " device=" + dev +
                                                                  " kernels=unknown\n# keys: n:log:3 batch:log\n" + rows);
    select::testing::set_builtin_tables(std::move(files));
}

// A view whose data is never dereferenced: getri_buffer_size reads metadata only.
template <typename T>
MVof<T> meta_view(int rows, int cols, int batch) {
    return {static_cast<T*>(nullptr), rows, cols, std::max(rows, 1), std::max(rows, 1) * std::max(cols, 1), batch};
}

template <typename T, Backend B>
struct Cfg {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};
using Types = typename test_utils::backend_types<Cfg>::type;

template <typename Config>
class GetriCandidates : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    using MV = MatrixView<T, MatrixFormat::Dense>;
    static constexpr Backend B = Config::BackendVal;
    static constexpr bool kVendor = batchlas::select::factorization_vendor_available<B>;

    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        if (this->HasFatalFailure() || ::testing::Test::IsSkipped()) return;
        if (!this->ctx) GTEST_SKIP() << "no queue";
        if (this->ctx->device().type != DeviceType::GPU) GTEST_SKIP() << "the native tier is GPU-only";
        if (!this->ctx->device().supports_sub_group_size(32)) GTEST_SKIP() << "no sub-group size 32";
    }

    // ---- the limit oracle: getri_blocked_dispatch's own checks, not getri.cc's can_run ----
    static bool expect_runs(const C& c, int rows, int cols, int batch, bool het = false) {
        if (std::holds_alternative<gi::Vendor>(c)) return kVendor;
        return rows == cols && rows >= 1 && batch >= 1 && !het && sycl_getri::getri_blocked_available<T>();
    }
    static bool vendor_word_falls_back(const C& c) { return std::holds_alternative<gi::Vendor>(c) && !kVendor; }

    // Factored by the public getrf (Auto, or `getrf_route`): the pivots a caller hands getri.
    Inv<T> make(const Spec& s, const char* getrf_route = nullptr) {
        Inv<T> p;
        p.s = s;
        p.lda = s.n + 3;
        p.sa = p.lda * s.n + 5;
        p.ldc = s.n + 2;
        p.sc = p.ldc * s.n + 7;
        p.coff = std::size_t(p.sa) * s.batch + 11;
        p.mem = UnifiedVector<T>(p.coff + std::size_t(p.sc) * s.batch + 13, poison<T>());
        p.aptr = UnifiedVector<T*>(std::max(s.batch, 1), nullptr);
        p.cptr = UnifiedVector<T*>(std::max(s.batch, 1), nullptr);
        p.piv = UnifiedVector<int64_t>(std::max<std::size_t>(1, std::size_t(s.n) * s.batch), 0);
        p.info = UnifiedVector<int32_t>(std::max(s.batch, 1), -12345);
        fill_a(p);
        const ScopedEnvVar route("BATCHLAS_GETRF_ROUTE", getrf_route);
        UnifiedVector<std::byte> ws(std::max<std::size_t>(1, getrf_buffer_size<B, T>(*this->ctx, p.A())));
        (void)getrf<B, T>(*this->ctx, p.A(), p.piv.to_span(), ws.to_span());
        this->ctx->wait();
        p.mem0.assign(p.mem.begin(), p.mem.end());
        return p;
    }

    void run(Inv<T>& p) {
        UnifiedVector<std::byte> ws(std::max<std::size_t>(1, getri_buffer_size<B, T>(*this->ctx, p.A())));
        (void)getri<B, T>(*this->ctx, p.A(), p.Cv(), p.piv.to_span(), ws.to_span(), p.info.to_span());
        this->ctx->wait();
    }
    void run_pinned(const C& c, Inv<T>& p) {
        const Pin pin("getri", c);
        run(p);
    }
    // Acceptance is asked of the sizing call, which resolves the pin exactly as the run does.
    bool pin_accepted(const C& c, const MV& A) {
        const Pin pin("getri", c);
        try {
            (void)getri_buffer_size<B, T>(*this->ctx, A);
            return true;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find("cannot run this shape") == std::string::npos) throw;
            return false;
        }
    }

    // The family's own driver: Blocked with the public trsm, Vendor the library.
    bool direct(const C& c, Inv<T>& p, std::string* why = nullptr) {
        Queue& q = *this->ctx;
        try {
            if (std::holds_alternative<gi::Blocked>(c)) {
                UnifiedVector<std::byte> ws(1);
                (void)sycl_getri::getri_blocked_dispatch<T>(
                    q, p.A(), p.Cv(), p.piv.to_span(), ws.to_span(), p.info.to_span(),
                    [](Queue& c2, const MV& ta, const MV& tb, T al, Side sd, Uplo ul, Transpose tr, Diag dg) {
                        return trsm<B, T>(c2, ta, tb, al, sd, ul, tr, dg);
                    });
            } else {
                if constexpr (kVendor) {
                    UnifiedVector<std::byte> ws(
                        std::max<std::size_t>(1, backend::getri_vendor_buffer_size<B, T>(q, p.A())));
                    (void)backend::getri_vendor<B, T>(q, p.A(), p.Cv(), p.piv.to_span(), ws.to_span(),
                                                      p.info.to_span());
                } else {
                    if (why) *why = "no vendor solver in this build";
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

    // Residual of the checked items, info 0, A and everything outside C bit for bit, and for a
    // repeating batch every item's C bit-identical to its representative's.
    void expect_inverted(const Inv<T>& p, const std::string& what) {
        const Spec& s = p.s;
        std::vector<int> items;
        if (s.period > 0) {
            for (int it = 0; it < std::min(s.period, s.batch); ++it) items.push_back(it);
            items.push_back(s.batch - 1);
        } else if (s.batch <= 8) {
            for (int it = 0; it < s.batch; ++it) items.push_back(it);
        } else {
            items = {0, 1, s.batch / 2, s.batch - 1};
        }
        for (int it : items) {
            const double r = residual(p, it);
            ASSERT_TRUE(std::isfinite(r) && r <= tol<T>(s.n)) << what << " item " << it << " residual " << r;
        }
        for (int it = 0; it < s.batch; ++it) ASSERT_EQ(p.info[it], 0) << what << " info of item " << it;
        std::vector<char> inc(p.mem.size(), 0);
        for (int it = 0; it < s.batch; ++it)
            for (int j = 0; j < s.n; ++j)
                for (int i = 0; i < s.n; ++i) inc[p.ci(it, i, j)] = 1;
        for (std::size_t e = 0; e < p.mem.size(); ++e)
            if (!inc[e]) ASSERT_TRUE(same_bits(p.mem[e], p.mem0[e])) << what << ": wrote outside C at element " << e;
        if (s.period > 0)
            for (int it = s.period; it < s.batch; ++it)
                for (int j = 0; j < s.n; ++j)
                    for (int i = 0; i < s.n; ++i)
                        ASSERT_TRUE(same_bits(p.mem[p.ci(it, i, j)], p.mem[p.ci(it % s.period, i, j)]))
                            << what << ": C of item " << it << " differs from its representative at (" << i << ","
                            << j << ")";
    }
};

TYPED_TEST_SUITE(GetriCandidates, Types);

// §8.1: each candidate across the old window edges (127/128, 255/256), the permutation kernel's
// work-group ladder (32/33, 64/65, 256/257) and the trsm CTA ceiling, at batch 1-3. Accepted
// shapes run and are correct; the vendor word falls back to Auto where there is no vendor.
TYPED_TEST(GetriCandidates, PinnedCandidatesStraddleTheirLimits) {
    using T = typename TestFixture::T;
    for (const C& c : gi::candidates<T>())
        for (int n : {1, 2, 32, 33, 64, 65, 127, 128, 255, 256, 257})
            for (int batch : {1, 3}) {
                const Spec s{n, batch, 0, 7u + unsigned(n)};
                auto p = this->make(s);
                if (!this->expect_runs(c, n, n, batch) && !this->vendor_word_falls_back(c)) {
                    EXPECT_THROW(this->run_pinned(c, p), std::invalid_argument) << name(c, s);
                    continue;
                }
                this->run_pinned(c, p);
                this->expect_inverted(p, name(c, s));
            }
}

// The pinned facade runs exactly that family's driver: bit-identical to the direct call.
TYPED_TEST(GetriCandidates, PinnedRunIsTheDirectKernelBitForBit) {
    using T = typename TestFixture::T;
    for (const C& c : gi::candidates<T>()) {
        if (this->vendor_word_falls_back(c)) continue;
        for (int n : {5, 40, 129}) {
            const Spec s{n, 3, 0, 91u + unsigned(n)};
            auto a = this->make(s);
            auto b = this->make(s);
            this->run_pinned(c, a);
            std::string why;
            ASSERT_TRUE(this->direct(c, b, &why)) << name(c, s) << ": " << why;
            for (std::size_t e = 0; e < a.mem.size(); ++e)
                ASSERT_TRUE(same_bits(a.mem[e], b.mem[e])) << name(c, s) << ": pinned != direct at element " << e;
            this->expect_inverted(a, name(c, s));
        }
    }
}

// A saturating batch: 1024 items of one matrix, each inverse bit-identical to item 0's.
TYPED_TEST(GetriCandidates, SaturatingBatchIsBitIdenticalToItsRepresentative) {
    using T = typename TestFixture::T;
    for (const C& c : gi::candidates<T>()) {
        if (this->vendor_word_falls_back(c)) continue;
        for (int n : {16, 129}) {
            const Spec s{n, 1024, 1, 5u};
            auto p = this->make(s);
            this->run_pinned(c, p);
            this->expect_inverted(p, name(c, s));
        }
    }
}

// §8.2 (R3): a pin is accepted exactly when the family's own driver launches.
TYPED_TEST(GetriCandidates, CanRunEqualsLaunch) {
    using T = typename TestFixture::T;
    for (const C& c : gi::candidates<T>()) {
        if (this->vendor_word_falls_back(c)) continue;
        for (const Spec& s : {Spec{1, 1}, Spec{8, 2}, Spec{33, 3}, Spec{300, 2}}) {
            auto p = this->make(s);
            const bool accepted = this->pin_accepted(c, p.A());
            std::string why;
            const bool launched = this->direct(c, p, &why);
            EXPECT_EQ(accepted, launched) << name(c, s) << ": " << why;
            EXPECT_EQ(accepted, this->expect_runs(c, s.n, s.n, s.batch)) << name(c, s);
        }
    }
    // The degenerate and non-square shapes: Blocked's driver throws, so the pin must refuse.
    for (const auto& [r, cc, batch] : {std::tuple{0, 0, 2}, std::tuple{8, 8, 0}, std::tuple{8, 6, 2}}) {
        UnifiedVector<T> a(std::max(1, r * cc * std::max(batch, 1)) + 64, T(1)), cm(a.size(), T(0));
        const MVof<T> A(a.data(), r, cc, std::max(r, 1), std::max(r * cc, 1), batch);
        const MVof<T> Cm(cm.data(), r, cc, std::max(r, 1), std::max(r * cc, 1), batch);
        UnifiedVector<int64_t> piv(64, 1);
        UnifiedVector<std::byte> ws(1);
        bool launched = true;
        try {
            (void)sycl_getri::getri_blocked_dispatch<T>(
                *this->ctx, A, Cm, piv.to_span(), ws.to_span(), Span<int32_t>{},
                [](Queue& c2, const MVof<T>& ta, const MVof<T>& tb, T al, Side sd, Uplo ul, Transpose tr, Diag dg) {
                    return trsm<TestFixture::B, T>(c2, ta, tb, al, sd, ul, tr, dg);
                });
            this->ctx->wait();
        } catch (const std::exception&) {
            launched = false;
        }
        EXPECT_FALSE(launched) << r << "x" << cc << " batch " << batch;
        EXPECT_EQ(this->pin_accepted(C{gi::Blocked{}}, A), launched) << r << "x" << cc << " batch " << batch;
    }
}

// §8.3 (R5): run every candidate with exactly getri_buffer_size bytes inside a poisoned arena;
// nothing past the end changes, and the size is the chosen family's own (Blocked needs none).
TYPED_TEST(GetriCandidates, ExactWorkspaceInAPoisonedArena) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    constexpr std::size_t kGuard = 4096;
    for (const C& c : gi::candidates<T>()) {
        if (this->vendor_word_falls_back(c)) continue;
        for (int n : {12, 130}) {
            const Spec s{n, 4, 0, 13u};
            auto p = this->make(s);
            const Pin pin("getri", c);
            const std::size_t need = getri_buffer_size<B, T>(*this->ctx, p.A());
            if (std::holds_alternative<gi::Blocked>(c)) EXPECT_EQ(need, 0u) << name(c, s);
            if constexpr (TestFixture::kVendor)
                if (std::holds_alternative<gi::Vendor>(c))
                    EXPECT_EQ(need, (backend::getri_vendor_buffer_size<B, T>)(*this->ctx, p.A())) << name(c, s);
            UnifiedVector<std::byte> arena(need + kGuard, std::byte{0xA5});
            (void)getri<B, T>(*this->ctx, p.A(), p.Cv(), p.piv.to_span(), Span<std::byte>(arena.data(), need),
                              p.info.to_span());
            this->ctx->wait();
            for (std::size_t i = need; i < arena.size(); ++i)
                ASSERT_EQ(arena[i], std::byte{0xA5}) << name(c, s) << ": wrote " << i - need << " bytes past the end";
            this->expect_inverted(p, name(c, s));
        }
    }
}

// getri_buffer_size runs under BumpAllocator::measuring() (inv.cc): it must size from metadata
// alone. A null data pointer under Auto and every pin proves it never dereferences A.
TYPED_TEST(GetriCandidates, SizingReadsMetadataOnly) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    const ScopedEnvVar clear("BATCHLAS_GETRI_ROUTE", nullptr);
    for (int n : {8, 512}) {
        const auto A = meta_view<T>(n, n, 4096);
        EXPECT_NO_THROW((void)(getri_buffer_size<B, T>)(*this->ctx, A)) << n;
        for (const C& c : gi::candidates<T>()) {
            const Pin pin("getri", c);
            EXPECT_NO_THROW((void)(getri_buffer_size<B, T>)(*this->ctx, A)) << select::to_string(c) << " " << n;
        }
    }
}

// R6: a spelling that names nothing compiled throws instead of meaning Auto.
TYPED_TEST(GetriCandidates, UnknownPinsThrow) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    const auto A = meta_view<T>(64, 64, 8);
    for (const char* word : {"bogus", "blocked:1", "blocked:nb=32", "native:vendor", "cta", "native:cta", "tiny",
                             "native:tiny", "vendor:auto",
                             // removed aliases (phase 5): each must stay an error
                             "native:blocked"}) {
        const Pin pin("getri", std::string_view(word));
        EXPECT_THROW((void)(getri_buffer_size<B, T>)(*this->ctx, A), std::invalid_argument) << word;
    }
    const ScopedEnvVar env("BATCHLAS_GETRI_ROUTE", "not-a-route");
    EXPECT_THROW((void)(getri_buffer_size<B, T>)(*this->ctx, A), std::invalid_argument)
        << "a typo in the environment must be reported, not silently Auto";
}

// The named can_run-false cases (RouteGetri.CorrectnessGatesIncludeTheOnesInheritedFromTrsm,
// ported): order 0, batch 0, a non-square A, and a heterogeneous A, each with its message;
// batch 1 and 2, n = 32 at a saturating batch and a huge order are NOT gates.
TYPED_TEST(GetriCandidates, CanRunFalsePinsThrow) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    for (const auto& [r, c, batch] : {std::tuple{0, 0, 4}, std::tuple{64, 64, 0}, std::tuple{64, 1024, 4}}) {
        const Pin pin("getri", C{gi::Blocked{}});
        try {
            (void)getri_buffer_size<B, T>(*this->ctx, meta_view<T>(r, c, batch));
            ADD_FAILURE() << r << "x" << c << " batch " << batch << " was accepted";
        } catch (const std::invalid_argument& e) {
            EXPECT_NE(std::string(e.what()).find("cannot run this shape"), std::string::npos) << e.what();
        }
    }
    for (const auto& [n, batch] : {std::pair{64, 1}, std::pair{40, 2}, std::pair{32, 8192}, std::pair{1 << 20, 1}})
        EXPECT_TRUE(this->pin_accepted(C{gi::Blocked{}}, meta_view<T>(n, n, batch))) << n << " batch " << batch;
}

// A heterogeneous A: the pivot list is read at b * n + k with one n, so Blocked cannot run it and
// its pin throws. Auto takes the vendor; vendor-free there is no route.
TYPED_TEST(GetriCandidates, HeterogeneousBatchHasNoNativeRoute) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    const int n = 16, batch = 4;
    Matrix<T, MatrixFormat::Dense> A(n, n, batch);
    UnifiedVector<int> act(batch);
    for (int b = 0; b < batch; ++b) act[b] = n - b;
    const auto het = A.view().with_active_dims(act.to_span(), act.to_span());
    ASSERT_TRUE(het.is_heterogeneous());
    EXPECT_FALSE(this->pin_accepted(C{gi::Blocked{}}, het));
    const ScopedEnvVar clear("BATCHLAS_GETRI_ROUTE", nullptr);
    if constexpr (TestFixture::kVendor) {
        EXPECT_NO_THROW((void)(getri_buffer_size<B, T>)(*this->ctx, het)) << "Auto must take the vendor";
    } else {
        EXPECT_THROW((void)(getri_buffer_size<B, T>)(*this->ctx, het), batchlas::NoRouteError);
    }
}

#if BATCHLAS_HAS_HOST_BACKEND
// RouteLuPivotFormat's getri half, ported: GPU backends pack 1-based int32 pivots into the int64
// span and netlib writes genuine int64, so on a GPU queue a NETLIB getri must never take Blocked.
// evidence: docs/perf/lu.md#lu-correctness-findings
TYPED_TEST(GetriCandidates, NetlibBackendOnAGpuQueueRefusesBlocked) {
    using T = typename TestFixture::T;
    const auto A = meta_view<T>(512, 512, 256);
    ASSERT_TRUE(this->pin_accepted(C{gi::Blocked{}}, A)) << "guard: the shape is Blocked's on " << TestFixture::B;
    {
        const Pin pin("getri", C{gi::Blocked{}});
        try {
            (void)getri_buffer_size<Backend::NETLIB, T>(*this->ctx, A);
            ADD_FAILURE() << "a NETLIB getri accepted the native pin on a GPU queue";
        } catch (const std::invalid_argument& e) {
            EXPECT_NE(std::string(e.what()).find("cannot run this shape"), std::string::npos) << e.what();
        }
    }
    select::testing::reset_warnings();
    const Pin pin("getri", std::string_view("native"));
    ::testing::internal::CaptureStderr();
    bool no_route = false;
    try {
        (void)getri_buffer_size<Backend::NETLIB, T>(*this->ctx, A);
    } catch (const batchlas::NoRouteError&) {
        no_route = true;
    }
    const std::string err = ::testing::internal::GetCapturedStderr();
    EXPECT_NE(err.find("getri pinned \"native\", but no native candidate"), std::string::npos) << err;
    // Without netlib the warned fall-back to Auto has nothing left: a NoRouteError, never Blocked.
    EXPECT_EQ(no_route, !batchlas::select::factorization_vendor_available<Backend::NETLIB>);
}
#endif

// §5.3: spellings (case-folded) and the class words, via ScopedPin and via the environment
// (RouteGetri.BareOriginResolvesToASpecificAlgorithm and BatchlasGetriRouteIsActuallyRead).
TYPED_TEST(GetriCandidates, ClassWordsAndSpellings) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_GETRI_ROUTE", nullptr);
    const Spec s{40, 2, 0, 3u};
    std::string auto_pick;
    {
        auto p = this->make(s);
        auto_pick = traced_choice([&] { this->run(p); });
    }
    ASSERT_TRUE(auto_pick == "blocked" || auto_pick == "vendor") << auto_pick;
    const std::string vendor_pick = TestFixture::kVendor ? "vendor" : auto_pick;
    const std::pair<const char*, std::string> expect[] = {
        {"BLOCKED", "blocked"}, {"blocked", "blocked"}, {"Blocked", "blocked"},
        {"native", "blocked"},         {"vendor", vendor_pick},       {"auto", auto_pick}};
    for (const auto& [word, spelling] : expect)
        for (bool via_env : {false, true}) {
            auto p = this->make(s);
            select::testing::reset_warnings();
            std::string err;
            const std::string got = traced_choice([&] {
                const ScopedEnvVar env("BATCHLAS_GETRI_ROUTE", via_env ? word : nullptr);
                std::optional<Pin> pin;
                if (!via_env) pin.emplace("getri", std::string_view(word));
                this->run(p);
            }, &err);
            const std::string what = std::string(word) + (via_env ? " via BATCHLAS_GETRI_ROUTE" : " via ScopedPin");
            EXPECT_EQ(got, spelling) << what;
            const bool warns = std::string(word) == "vendor" && !TestFixture::kVendor;
            EXPECT_EQ(err.find("getri pinned \"vendor\", but no vendor candidate") != std::string::npos, warns)
                << what << ": " << err;
            this->expect_inverted(p, what);
        }
}

// §5.3: a ScopedPin wins over BATCHLAS_GETRI_ROUTE, and nested pins restore the outer one.
TYPED_TEST(GetriCandidates, ScopedPinBeatsTheEnvironment) {
    using T = typename TestFixture::T;
    const Spec s{24, 2, 0, 17u};
    const ScopedEnvVar env("BATCHLAS_GETRI_ROUTE", "bogus");
    {
        auto p = this->make(s);
        EXPECT_THROW(this->run(p), std::invalid_argument) << "the environment pin was not read";
    }
    auto p = this->make(s);
    EXPECT_EQ(traced_choice([&] { this->run_pinned(C{gi::Blocked{}}, p); }), "blocked");
    this->expect_inverted(p, "blocked over env bogus");
    const Pin outer("getri", C{gi::Blocked{}});
    {
        const Pin inner("getri", std::string_view("vendor"));
        auto q = this->make(s);
        EXPECT_EQ(traced_choice([&] { this->run(q); }), TestFixture::kVendor ? "vendor" : "blocked");
    }
    auto r = this->make(s);
    EXPECT_EQ(traced_choice([&] { this->run(r); }), "blocked") << "the inner pin did not restore the outer";
    this->expect_inverted(r, "outer blocked");
}

// Auto against the shipped table of this device (shipped_table_pick.hh): both sides of each old
// window edge, on and off the grid, at batch 2 and 3 (inverse_tests' extents, below the grid) and
// 300. Vendor-free that is Blocked everywhere (RouteGetri.VendorFreeFallbackHandsOverTheNativeRoute
// and getrf_tests' L12 getri readbacks, ported).
TYPED_TEST(GetriCandidates, AutoReadsTheShippedTable) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    const ScopedEnvVar clear("BATCHLAS_GETRI_ROUTE", nullptr);
    for (const auto& [n, batch] : {std::pair{40, 2}, std::pair{127, 3}, std::pair{128, 3}, std::pair{129, 300},
                                   std::pair{200, 2}, std::pair{255, 2}, std::pair{256, 2}, std::pair{512, 2}}) {
        const Spec s{n, batch, 0, 23u + unsigned(n)};
        auto p = this->make(s);
        const std::string want = test_utils::shipped_table_pick<C>(
            "getri", select::dtype_name<T>(), select::device_of<B>(*this->ctx), select::Key{{"n", n}, {"batch", batch}},
            [&] {
                auto q = this->make(s);
                this->run(q);
            });
        if (!TestFixture::kVendor) EXPECT_EQ(want, "blocked") << label(s);
        EXPECT_EQ(traced_choice([&] { this->run(p); }), want) << label(s);
        this->expect_inverted(p, "auto " + label(s));
    }
}

// key_of's every field reaches choose(): a synthetic table whose winner changes with n alone and
// with batch alone. Fixing either field in key_of turns exactly its row red.
TYPED_TEST(GetriCandidates, AutoReadsEveryKeyField) {
    using T = typename TestFixture::T;
    if (!TestFixture::kVendor) GTEST_SKIP() << "one runnable family: no key can change the choice";
    const ScopedEnvVar clear("BATCHLAS_GETRI_ROUTE", nullptr);
    const TableGuard restore;
    install_getri_table(std::string(select::dtype_name<T>()), select::device_of<TestFixture::B>(*this->ctx).key,
                        "n=8 batch=128 | vendor 1 | blocked 2\n"
                        "n=8 batch=32768 | blocked 1 | vendor 2\n"
                        "n=64 batch=128 | blocked 1 | vendor 2\n");
    struct Probe { int n, batch; const char* expect; const char* field; };
    for (const Probe& k : {Probe{8, 128, "vendor", "base"}, Probe{8, 32768, "blocked", "batch"},
                           Probe{64, 128, "blocked", "n"}}) {
        const Spec s{k.n, k.batch, 4, 43u};
        auto p = this->make(s);
        EXPECT_EQ(traced_choice([&] { this->run(p); }), k.expect) << "the " << k.field << " row";
        this->expect_inverted(p, std::string("the ") + k.field + " row");
    }
}

// The trace line names the key the table lookup used.
TYPED_TEST(GetriCandidates, TraceLineShowsTheKey) {
    using T = typename TestFixture::T;
    const Spec s{12, 5, 0, 29u};
    auto p = this->make(s);
    const std::string line = traced_line([&] { this->run(p); });
    EXPECT_NE(line.find(std::string("getri ") + std::string(select::dtype_name<T>()) + " n=12 batch=5 ->"),
              std::string::npos)
        << line;
    this->expect_inverted(p, line);
}

// RouteGetri.AbsentDriverIsUnsupported, ported as far as a build can show it: every type's
// driver is compiled here; with a table naming only the vendor, a vendor-free build takes the
// last resort, Blocked, and a vendor build the vendor.
TYPED_TEST(GetriCandidates, VendorFreeLastResortIsBlocked) {
    using T = typename TestFixture::T;
    ASSERT_TRUE(sycl_getri::getri_blocked_available<T>());
    const ScopedEnvVar clear("BATCHLAS_GETRI_ROUTE", nullptr);
    const TableGuard restore;
    install_getri_table(std::string(select::dtype_name<T>()), select::device_of<TestFixture::B>(*this->ctx).key,
                        "n=64 batch=128 | vendor 1\n");
    for (int n : {40, 300}) {
        const Spec s{n, 2, 0, 31u};
        auto p = this->make(s);
        std::string all;
        const std::string got = traced_choice([&] { this->run(p); }, &all);
        EXPECT_EQ(got, TestFixture::kVendor ? "vendor" : "blocked") << n;
        if (!TestFixture::kVendor) EXPECT_NE(all.find("last resort"), std::string::npos) << all;
        this->expect_inverted(p, "vendor-only table, n " + std::to_string(n));
    }
}

// A pinned vendor getri reading a natively-written getrf factor, and the reverse: the pivot
// contract across two independently pinned ops (RouteLuFamily's getri half, now with kernels).
TYPED_TEST(GetriCandidates, PinsAreIndependentOfGetrf) {
    if (!TestFixture::kVendor) GTEST_SKIP() << "no vendor getri";
    const std::pair<const char*, C> pairs[] = {{"native", C{gi::Vendor{}}}, {"vendor", C{gi::Blocked{}}}};
    for (const auto& [getrf_pin, getri_pin] : pairs) {
        auto p = this->make(Spec{48, 3, 0, 37u}, getrf_pin);
        this->run_pinned(getri_pin, p);
        this->expect_inverted(p, std::string("getrf ") + getrf_pin + ", getri " + select::to_string(getri_pin));
    }
}

// Every entry of every row of this device's own table is accepted as a pin at that row's key.
TYPED_TEST(GetriCandidates, ShippedRowsRunOnTheirOwnDevice) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    const auto& d = select::device_of<B>(*this->ctx);
    const auto tables = select::tables_in_borrow_order("getri", select::dtype_name<T>(), d);
    if (tables.empty() || tables.front()->device != d.key) GTEST_SKIP() << "no own getri table for " << d.key;
    for (const auto& row : tables.front()->rows) {
        const int n = std::stoi(row.keys[0]), batch = std::stoi(row.keys[1]);
        for (const auto& e : row.ranked) {
            const auto c = select::parse<C>(e.spelling);
            ASSERT_TRUE(c.has_value()) << e.spelling;
            if (this->vendor_word_falls_back(*c)) continue;
            EXPECT_TRUE(this->pin_accepted(*c, meta_view<T>(n, n, batch)))
                << tables.front()->file << ":" << row.line << " " << e.spelling;
        }
    }
}

// The coverage row (§5.6): the real backend, m = k = order and n = A.cols, the choice, and the
// native flags. threadsafe: the child re-executes the binary, so CUDA is initialised fresh.
TYPED_TEST(GetriCandidates, CoverageRowCarriesBackendKeyAndNativeFlags) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    const std::string dir = ::testing::TempDir() + "getri_cov." + std::string(select::dtype_name<T>());
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const std::string out = dir + "/cov";
    auto child = [&] {
        const ScopedEnvVar cov("BATCHLAS_COVERAGE_OUT", out.c_str());
        const ScopedEnvVar clear("BATCHLAS_GETRI_ROUTE", nullptr);
        batchlas::coverage::g_dynamic_enabled = true;
        {
            auto p = this->make(Spec{16, 2, 0, 3u});
            this->run_pinned(C{gi::Blocked{}}, p);
        }
        if constexpr (TestFixture::kVendor) {
            auto p = this->make(Spec{48, 2, 0, 4u});
            this->run_pinned(C{gi::Vendor{}}, p);
        }
        std::exit(0);
    };
    EXPECT_EXIT(child(), ::testing::ExitedWithCode(0), "");
    std::map<std::string, std::vector<std::string>> rows;  // "m n k" -> fields
    for (const auto& ent : std::filesystem::directory_iterator(dir)) {
        std::ifstream in(ent.path());
        for (std::string line; std::getline(in, line);) {
            if (line.rfind("reached,getri,", 0) != 0) continue;
            std::vector<std::string> f;
            std::stringstream ss(line);
            for (std::string tok; std::getline(ss, tok, ',');) f.push_back(tok);
            ASSERT_GE(f.size(), 14u) << line;
            rows[f[5] + " " + f[6] + " " + f[7]] = f;
        }
    }
    std::filesystem::remove_all(dir);
    ASSERT_EQ(rows.size(), TestFixture::kVendor ? 2u : 1u);
    const std::string backend = B == Backend::CUDA ? "CUDA" : (B == Backend::ROCM ? "ROCM" : "?");
    std::vector<std::pair<std::string, std::string>> want{{"16 16 16", "blocked"}};
    if (TestFixture::kVendor) want.emplace_back("48 48 48", "vendor");
    for (const auto& [key, algo] : want) {
        ASSERT_TRUE(rows.count(key)) << "no " << key << " row";
        const auto& f = rows[key];
        EXPECT_EQ(f[3], backend) << key;
        EXPECT_EQ(f[9], algo == "vendor" ? "vendor" : "native") << key;
        EXPECT_EQ(f[10], algo) << key;
        EXPECT_EQ(f[12], "1") << key << ": native_route_existed";
        EXPECT_EQ(f[13], "1") << key << ": native_route_supported";
    }
}

// RouteGetri's CPU case, ported: on a CPU queue Blocked cannot run, so its pin throws and Auto is
// the vendor (netlib, with its genuine int64 pivots) or, without one, no route.
template <typename Config>
class GetriCandidatesCpu : public test_utils::BatchLASTest<Config> {};
TYPED_TEST_SUITE(GetriCandidatesCpu, Types);

TYPED_TEST(GetriCandidatesCpu, CpuQueueRunsNoNativeFamily) {
    using T = typename TypeParam::ScalarType;
    static constexpr Backend B = TypeParam::BackendVal;
    if (!this->ctx) GTEST_SKIP() << "no queue";
    if (this->ctx->device().type == DeviceType::GPU) GTEST_SKIP() << "a GPU queue";
    const ScopedEnvVar clear("BATCHLAS_GETRI_ROUTE", nullptr);
    {
        const Pin pin("getri", C{gi::Blocked{}});
        EXPECT_THROW((void)(getri_buffer_size<B, T>)(*this->ctx, meta_view<T>(8, 8, 2)), std::invalid_argument);
    }
    const int n = 8, batch = 2;
    Matrix<T, MatrixFormat::Dense> A(n, n, batch), Ai(n, n, batch);
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) A(i, j, b) = mk<T>(RealOf<T>(i == j ? n + 1 : 0.25), RealOf<T>(0.1));
    UnifiedVector<int64_t> piv(std::size_t(n) * batch);
    auto call = [&] {
        UnifiedVector<std::byte> fws(std::max<std::size_t>(1, getrf_buffer_size<B, T>(*this->ctx, A.view())));
        (void)getrf<B, T>(*this->ctx, A.view(), piv.to_span(), fws.to_span());
        UnifiedVector<std::byte> ws(std::max<std::size_t>(1, getri_buffer_size<B, T>(*this->ctx, A.view())));
        (void)getri<B, T>(*this->ctx, A.view(), Ai.view(), piv.to_span(), ws.to_span());
        this->ctx->wait();
    };
    if constexpr (batchlas::select::factorization_vendor_available<B>) EXPECT_EQ(traced_choice(call), "vendor");
    else EXPECT_THROW(call(), batchlas::NoRouteError);
}

// The sm_89 transcribed rows, read with Table::nearest directly so every device checks them: the
// old preference, every dtype, at on-grid, edge and off-grid keys. sm_120 is deep-measured since
// 2026-10-09 (tuned_tables_tests checks its provenance).
TEST(GetriTranscribedTable, RowsHoldTheOldPreference) {
    for (const char* dev : {"sm_89"})
        for (const char* dt : {"float", "double", "cfloat", "cdouble"}) {
            const auto tables = select::tables_in_borrow_order("getri", dt, select::device_from_key(dev));
            ASSERT_FALSE(tables.empty()) << dt;
            const select::Table& t = *tables.front();
            ASSERT_EQ(t.device, dev) << dt;
            EXPECT_EQ(t.source.rfind("transcribed:", 0), 0u) << t.file;
            for (int n : {1, 5, 100, 112, 126, 127, 128, 129, 160, 254, 255, 256, 257, 300, 3000, 9000})
                for (int batch : {1, 2, 3, 300, 100000}) {
                    const select::TableRow* row = t.nearest({{"n", n}, {"batch", batch}});
                    ASSERT_NE(row, nullptr) << t.file;
                    const std::string what = t.file + ":" + std::to_string(row->line) + " for n=" +
                                             std::to_string(n) + " batch=" + std::to_string(batch);
                    ASSERT_EQ(row->ranked.size(), 2u) << what;
                    EXPECT_EQ(row->ranked.front().spelling, old_auto_dtype(dt, n)) << what;
                    EXPECT_FALSE(row->timed) << what;
                }
        }
}

}  // namespace
