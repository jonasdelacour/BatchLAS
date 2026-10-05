// Every getrs candidate, pinned: docs/design/flat-kernel-selection.md §8 and
// docs/design/flat-select-p5/getrs.md. The limit oracle reads the drivers' own capacity
// constants; which kernel ran is read back from the select trace or a bit-for-bit comparison
// with the direct driver, never assumed from the pin being accepted. The factors are built on
// the host (no getrf in the loop), so the reference never shares code with what it checks.
#include <gtest/gtest.h>

#include <batchlas/blas/functions/getrs.hh>
#include <batchlas/blas/functions/trsm.hh>
#include "../src/select/coverage.hh"
#include <batchlas/no_route.hh>
#include "../src/select/vendor.hh"
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>

#include "test_utils.hh"

#include "../src/extensions/getrs_native.hh"
#include "../src/ops/getrs/choice.hh"

#include <algorithm>
#include <array>
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
#include <variant>
#include <vector>

using namespace batchlas;

namespace {

namespace gs = batchlas::ops::getrs;
using C = gs::GetrsChoice;
using Pin = select::ScopedPin<C>;
template <typename T>
using MVof = MatrixView<T, MatrixFormat::Dense>;
template <typename T>
using RealOf = typename batchlas::base_type<T>::type;
template <typename T>
constexpr bool kCx = test_utils::is_complex<T>::value;
using cd = std::complex<double>;

template <typename T>
T mk(RealOf<T> r, RealOf<T> i) {
    if constexpr (kCx<T>) return T(r, i);
    else return r;
}
template <typename T>
cd up(T v) {
    if constexpr (kCx<T>) return {double(v.real()), double(v.imag())};
    else return {double(v), 0.0};
}
template <typename T>
bool same_bits(T a, T b) {
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}
const char* trans_s(Transpose t) { return t == Transpose::NoTrans ? "N" : (t == Transpose::Trans ? "T" : "C"); }

struct Spec {
    int n = 8, nrhs = 2, batch = 3;
    Transpose trans = Transpose::NoTrans;
    int period = 0;  // > 0: item it repeats item it % period
    unsigned seed = 1;
};
std::string label(const Spec& s) {
    return std::string(trans_s(s.trans)) + " n=" + std::to_string(s.n) + " nrhs=" + std::to_string(s.nrhs) +
           " batch=" + std::to_string(s.batch);
}
std::string name(const C& c, const Spec& s) { return select::to_string(c) + " " + label(s); }

// One factored system per item: packed L\U at ld n+3 and a non-ld*n stride, B at ld n+2, the
// interchange list as packed 1-based int32 in the first half of the int64 span. Everything a
// solve may not touch -- ld pads, stride gaps, the second half of the pivot span -- holds a
// large finite poison the kernels would accept (in-range pivot 1, finite matrix entries).
template <typename T>
struct Sys {
    Spec s;
    int lda = 0, sa = 0, ldb = 0, sb = 0;
    UnifiedVector<T> a, b;
    std::vector<T> b0;
    UnifiedVector<std::int64_t> piv;
    UnifiedVector<T*> aptr, bptr;        // the vendor's batched pointer arrays
    std::vector<std::vector<int>> ipiv;  // per representative item, 0-based

    MVof<T> A() { return {a.data(), s.n, s.n, lda, sa, s.batch, aptr.data()}; }
    MVof<T> B() { return {b.data(), s.n, s.nrhs, ldb, sb, s.batch, bptr.data()}; }
    T& f(int it, int i, int j) { return a[std::size_t(it) * sa + std::size_t(j) * lda + i]; }
    T& x(int it, int i, int j) { return b[std::size_t(it) * sb + std::size_t(j) * ldb + i]; }
    int* p32() { return reinterpret_cast<int*>(piv.data()); }
};

template <typename T>
T poison() {
    return mk<T>(RealOf<T>(-999), RealOf<T>(777));
}

template <typename T>
Sys<T> make_sys(const Spec& s) {
    using R = RealOf<T>;
    Sys<T> p;
    p.s = s;
    p.lda = std::max(1, s.n + 3);
    p.sa = p.lda * std::max(1, s.n) + 5;
    p.ldb = std::max(1, s.n + 2);
    p.sb = p.ldb * std::max(1, s.nrhs) + 7;
    p.a = UnifiedVector<T>(std::size_t(p.sa) * std::max(1, s.batch), poison<T>());
    p.b = UnifiedVector<T>(std::size_t(p.sb) * std::max(1, s.batch), poison<T>());
    p.aptr = UnifiedVector<T*>(std::max(1, s.batch), nullptr);
    p.bptr = UnifiedVector<T*>(std::max(1, s.batch), nullptr);
    p.piv = UnifiedVector<std::int64_t>(std::size_t(std::max(1, s.n)) * std::max(1, s.batch), 0);
    int* pv = p.p32();
    for (std::size_t k = 0; k < 2 * p.piv.size(); ++k) pv[k] = 1;  // second half: accepted poison
    std::mt19937 gen(s.seed);
    std::uniform_real_distribution<R> d(R(0.1), R(1)), u(R(-1), R(1));
    const int reps = s.period > 0 ? std::min(s.period, s.batch) : s.batch;
    for (int it = 0; it < reps; ++it) {
        std::vector<int> ip(s.n);
        for (int k = 0; k < s.n; ++k) ip[k] = k + int(gen() % unsigned(s.n - k));
        p.ipiv.push_back(ip);
        for (int j = 0; j < s.n; ++j)
            for (int i = 0; i < s.n; ++i) {
                const R sg = gen() & 1 ? R(1) : R(-1);
                if (i == j) p.f(it, i, j) = mk<T>(R(2.5) + R(0.5) * d(gen), R(0.3) * d(gen));
                else p.f(it, i, j) = mk<T>(sg * d(gen) / R(s.n), d(gen) / R(s.n));
            }
        for (int j = 0; j < s.nrhs; ++j)
            for (int i = 0; i < s.n; ++i) p.x(it, i, j) = mk<T>(u(gen), u(gen));
    }
    for (int it = 0; it < s.batch; ++it) {
        const int r = it % reps;
        for (int k = 0; k < s.n; ++k) pv[std::size_t(it) * s.n + k] = p.ipiv[r][k] + 1;
        if (it < reps) continue;
        for (int j = 0; j < s.n; ++j)
            for (int i = 0; i < s.n; ++i) p.f(it, i, j) = p.f(r, i, j);
        for (int j = 0; j < s.nrhs; ++j)
            for (int i = 0; i < s.n; ++i) p.x(it, i, j) = p.x(r, i, j);
    }
    p.b0.assign(p.b.begin(), p.b.end());
    return p;
}

// ||op(M) X - B0|| / (||L|| ||U|| ||X|| + ||B0||) with M = P^-1 L U, applied factor by factor
// (O(n^2 nrhs), so the capacity cases stay cheap). getrf's list swaps row k with ipiv[k] in
// order, so P^-1 is the reversed walk; op(M) = op(U) op(L) P for T and C.
template <typename T>
double residual(Sys<T>& p, int it) {
    const int n = p.s.n, nr = p.s.nrhs;
    const int rep = p.s.period > 0 ? it % p.s.period : it;
    const auto& ip = p.ipiv[rep < int(p.ipiv.size()) ? rep : 0];
    const Transpose t = p.s.trans;
    auto L = [&](int i, int j) -> cd { return i == j ? cd(1) : (i > j ? up(p.f(it, i, j)) : cd(0)); };
    auto U = [&](int i, int j) -> cd { return i <= j ? up(p.f(it, i, j)) : cd(0); };
    auto op = [&](cd v) { return t == Transpose::ConjTrans ? std::conj(v) : v; };
    double ln = 0, un = 0, xn = 0, bn = 0, num = 0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) ln += std::norm(L(i, j)), un += std::norm(U(i, j));
    std::vector<cd> v(n), w(n);
    for (int c = 0; c < nr; ++c) {
        for (int i = 0; i < n; ++i) v[i] = up(p.x(it, i, c)), xn += std::norm(v[i]);
        if (t == Transpose::NoTrans) {
            for (int i = 0; i < n; ++i) { w[i] = 0; for (int k = i; k < n; ++k) w[i] += U(i, k) * v[k]; }
            for (int i = 0; i < n; ++i) { v[i] = 0; for (int k = 0; k <= i; ++k) v[i] += L(i, k) * w[k]; }
            for (int k = n - 1; k >= 0; --k) std::swap(v[k], v[ip[k]]);
        } else {
            for (int k = 0; k < n; ++k) std::swap(v[k], v[ip[k]]);
            for (int i = 0; i < n; ++i) { w[i] = 0; for (int k = i; k < n; ++k) w[i] += op(L(k, i)) * v[k]; }
            for (int i = 0; i < n; ++i) { v[i] = 0; for (int k = 0; k <= i; ++k) v[i] += op(U(k, i)) * w[k]; }
        }
        for (int i = 0; i < n; ++i) {
            const cd b0 = up(p.b0[std::size_t(it) * p.sb + std::size_t(c) * p.ldb + i]);
            num += std::norm(v[i] - b0);
            bn += std::norm(b0);
        }
    }
    const double den = std::sqrt(ln * un * xn) + std::sqrt(bn);
    return den == 0 ? std::sqrt(num) : std::sqrt(num) / den;
}

template <typename T>
double tol(int n) {
    return 64.0 * std::max(n, 1) * double(std::numeric_limits<RealOf<T>>::epsilon());
}

// Residuals of the checked items, everything outside B's footprint and A bit for bit, and for a
// repeating batch every item bit-identical to its representative.
template <typename T>
void expect_solved(Sys<T>& p, const std::string& what, const std::vector<T>* a0 = nullptr) {
    const Spec& s = p.s;
    std::vector<int> items;
    if (s.period > 0) for (int it = 0; it < std::min(s.period, s.batch); ++it) items.push_back(it);
    else if (s.batch <= 8) for (int it = 0; it < s.batch; ++it) items.push_back(it);
    else items = {0, 1, s.batch / 2, s.batch - 1};
    for (int it : items) {
        const double r = residual(p, it);
        ASSERT_TRUE(std::isfinite(r) && r <= tol<T>(s.n)) << what << " item " << it << " residual " << r;
    }
    for (std::size_t e = 0; e < p.b.size(); ++e) {
        const std::size_t off = e % std::size_t(p.sb), col = off / std::size_t(p.ldb), row = off % std::size_t(p.ldb);
        if (col >= std::size_t(s.nrhs) || row >= std::size_t(s.n))
            ASSERT_TRUE(same_bits(p.b[e], p.b0[e])) << what << ": wrote outside B at element " << e;
    }
    if (a0)
        for (std::size_t e = 0; e < p.a.size(); ++e)
            ASSERT_TRUE(same_bits(p.a[e], (*a0)[e])) << what << ": wrote A at element " << e;
    if (s.period > 0)
        for (int it = s.period; it < s.batch; ++it)
            for (int j = 0; j < s.nrhs; ++j)
                for (int i = 0; i < s.n; ++i)
                    ASSERT_TRUE(same_bits(p.x(it, i, j), p.x(it % s.period, i, j)))
                        << what << ": X of item " << it << " differs from its representative at (" << i << "," << j << ")";
}

// The outermost getrs trace line for whatever `run` calls, and the choice it names.
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
        if (line.rfind("getrs ", 0) == 0 && line.find(" -> ") != std::string::npos) return line;
    return "<no getrs trace line in: " + err + ">";
}
template <class F>
std::string traced_choice(F&& run, std::string* all = nullptr) {
    const std::string line = traced_line(std::forward<F>(run), all);
    const auto arrow = line.find(" -> ");
    if (arrow == std::string::npos) return line;
    const std::string tail = line.substr(arrow + 4);
    return tail.substr(0, tail.find(' '));
}

// The old predicate (route_getrs.hh at 424a45bc), restated by hand from its source so the
// table checks below are not read from the table: clause A/B for the fused tier at order >= 32,
// the composition at batch >= 128 for float nrhs >= 64 and double nrhs >= 128, else the vendor;
// without a vendor the fused tier where it fits, else the composition.
std::string old_auto(const std::string& dtype, long n, long nrhs, long batch, bool vendor, bool cta_fits) {
    const bool cta_pref = n >= 32 && (nrhs <= 2 || (dtype == "float" && nrhs <= 4));
    const bool blk_pref = batch >= 128 && ((dtype == "float" && nrhs >= 64) || (dtype == "double" && nrhs >= 128));
    if (cta_pref && cta_fits) return "cta";
    if (blk_pref) return "blocked";
    if (vendor) return "vendor";
    return cta_fits ? "cta" : "blocked";
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

template <typename Config>
class GetrsCandidates : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    using MV = MatrixView<T, MatrixFormat::Dense>;
    static constexpr Backend B = Config::BackendVal;
    static constexpr bool kVendor = batchlas::select::factorization_vendor_available<B>;

    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        if (this->HasFatalFailure() || ::testing::Test::IsSkipped()) return;
        if (!this->ctx) GTEST_SKIP() << "no queue";
        if (this->ctx->device().type != DeviceType::GPU) GTEST_SKIP() << "the native tiers are GPU kernels";
        if (!this->ctx->device().supports_sub_group_size(32)) GTEST_SKIP() << "no sub-group size 32";
    }

    std::size_t budget() const {
        const std::size_t lm = this->ctx->device().get_property(DeviceProperty::LOCAL_MEM_SIZE);
        return lm > 4096 ? lm - 4096 : 0;
    }
    std::size_t cap() const { return sycl_getrs::getrs_fused_max_rhs_elems<T>(budget()); }

    // ---- the limit oracle: the drivers' own constants, not getrs.cc's can_run ----
    bool expect_runs(const C& c, int n, int nrhs) const {
        if (n < 1 || nrhs < 1) return false;
        if (std::holds_alternative<gs::Cta>(c))
            return nrhs <= sycl_getrs::kGetrsFusedMaxRhs && std::size_t(n) * std::size_t(nrhs) <= cap();
        if (std::holds_alternative<gs::Blocked>(c)) return sycl_getrs::getrs_blocked_available<T>();
        return kVendor;
    }
    bool cta_fits(int n, int nrhs) const { return expect_runs(C{gs::Cta{}}, n, nrhs); }

    static bool vendor_word_falls_back(const C& c) { return std::holds_alternative<gs::Vendor>(c) && !kVendor; }

    void run(Sys<T>& p) {
        auto A = p.A();
        auto Bm = p.B();
        UnifiedVector<std::byte> ws(std::max<std::size_t>(1, getrs_buffer_size<B, T>(*this->ctx, A, Bm, p.s.trans)));
        (void)getrs<B, T>(*this->ctx, A, Bm, p.s.trans, p.piv.to_span(), ws.to_span());
        this->ctx->wait();
    }
    void run_pinned(const C& c, Sys<T>& p) {
        const Pin pin("getrs", c);
        run(p);
    }
    std::string auto_choice(Sys<T>& p) {
        const ScopedEnvVar clear("BATCHLAS_GETRS_ROUTE", nullptr);
        return traced_choice([&] { run(p); });
    }
    // Acceptance is the sizing call under the pin: a can_run-false pin throws there.
    bool pin_accepted(const C& c, Sys<T>& p) {
        const Pin pin("getrs", c);
        try {
            auto A = p.A();
            auto Bm = p.B();
            (void)getrs_buffer_size<B, T>(*this->ctx, A, Bm, p.s.trans);
            return true;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find("cannot run this shape") == std::string::npos) throw;
            return false;
        }
    }

    // The family's own driver: Cta = the fused kernel, Blocked = laswp + the public trsm.
    bool direct(const C& c, Sys<T>& p, std::string* why = nullptr) {
        Queue& q = *this->ctx;
        const MV A = p.A(), Bm = p.B();
        UnifiedVector<std::byte> ws(4096);
        try {
            if (std::holds_alternative<gs::Cta>(c)) {
                (void)sycl_getrs::getrs_fused_dispatch<T>(q, A, Bm, p.s.trans, p.piv.to_span(), ws.to_span());
            } else if (std::holds_alternative<gs::Blocked>(c)) {
                (void)sycl_getrs::getrs_blocked_dispatch<T>(
                    q, A, Bm, p.s.trans, p.piv.to_span(), ws.to_span(),
                    [](Queue& c2, const MV& ta, const MV& tb, T al, Side sd, Uplo ul, Transpose tr, Diag dg) {
                        return trsm<B, T>(c2, ta, tb, al, sd, ul, tr, dg);
                    });
            } else {
                if constexpr (kVendor) {
                    UnifiedVector<std::byte> vw(
                        std::max<std::size_t>(1, backend::getrs_vendor_buffer_size<B, T>(q, A, Bm, p.s.trans)));
                    (void)backend::getrs_vendor<B, T>(q, A, Bm, p.s.trans, p.piv.to_span(), vw.to_span());
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

    // Rows of one shipped-table file replaced by `text`, everything else embedded as built.
    void use_table(const std::string& text) {
        const std::string dtype(select::dtype_name<T>());
        const std::string dev = select::device_of<B>(*this->ctx).key;
        std::vector<std::pair<std::string, std::string>> files;
        for (const auto& t : select::embedded_tables())
            if (t.name.rfind("getrs.", 0) != 0) files.emplace_back(std::string(t.name), std::string(t.text));
        files.emplace_back("getrs." + dtype + "." + dev + ".txt",
                           "# op=getrs dtype=" + dtype + " device=" + dev + " kernels=unknown\n"
                           "# keys: n:log:2 nrhs:log batch:log\n" + text);
        select::testing::set_builtin_tables(std::move(files));
    }
};

TYPED_TEST_SUITE(GetrsCandidates, Types);

// §8.1: each candidate across the fused tier's accumulator ladder (1/2/4/8, with 3 and 5 inside
// a wider accumulator), its width ceiling 8/9, the nb=16 block edge and a final block of one
// (97 = 6*16 + 1), the old order floor 31/32, and all three transposes. Refused shapes throw.
TYPED_TEST(GetrsCandidates, PinnedCandidatesStraddleTheirLimits) {
    using T = typename TestFixture::T;
    int ran = 0, refused = 0, i = 0;
    for (const C& c : gs::candidates<T>())
        for (int n : {1, 2, 16, 17, 31, 32, 97})
            for (int nrhs : {1, 2, 3, 4, 5, 8, 9, 17}) {
                const Transpose t = std::array{Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans}[i++ % 3];
                Spec s{n, nrhs, 3, t};
                s.seed = 1000u + 31u * n + nrhs;
                auto p = make_sys<T>(s);
                const std::string what = name(c, s);
                if (TestFixture::vendor_word_falls_back(c)) {
                    auto q = make_sys<T>(s);
                    const std::string want = this->auto_choice(q);
                    EXPECT_EQ(traced_choice([&] { this->run_pinned(c, p); }), want) << what;
                    expect_solved(p, what + " (vendor-free: Auto)");
                    continue;
                }
                if (!this->expect_runs(c, n, nrhs)) {
                    EXPECT_FALSE(this->pin_accepted(c, p)) << what << " was accepted";
                    ++refused;
                    continue;
                }
                ASSERT_TRUE(this->pin_accepted(c, p)) << what << " was refused";
                const std::vector<T> a0(p.a.begin(), p.a.end());
                EXPECT_EQ(traced_choice([&] { this->run_pinned(c, p); }), select::to_string(c)) << what;
                expect_solved(p, what, &a0);
                if (::testing::Test::HasFatalFailure()) return;
                ++ran;
            }
    EXPECT_GT(ran, 0);
    EXPECT_GT(refused, 0);
}

// R7 / AGENTS.md §8.9: the fused tier LAUNCHED at its device capacity n * nrhs = cap at nrhs 8,
// and refused one order past it; Blocked serves the refused shape.
TYPED_TEST(GetrsCandidates, CtaLaunchesAtItsResidentCapacity) {
    using T = typename TestFixture::T;
    const int nrhs = int(sycl_getrs::kGetrsFusedMaxRhs);
    const int fit = int(this->cap() / std::size_t(nrhs));
    ASSERT_GT(fit, 32) << "the fused tier has no useful capacity on this device";
    Spec s{fit, nrhs, 2, Transpose::ConjTrans};
    s.seed = 99u;
    auto p = make_sys<T>(s);
    ASSERT_TRUE(this->pin_accepted(C{gs::Cta{}}, p));
    this->run_pinned(C{gs::Cta{}}, p);
    expect_solved(p, "cta at capacity " + label(s));
    Spec over = s;
    over.n = fit + 1;
    auto q = make_sys<T>(over);
    EXPECT_FALSE(this->pin_accepted(C{gs::Cta{}}, q)) << label(over);
    std::string why;
    EXPECT_FALSE(this->direct(C{gs::Cta{}}, q, &why)) << "the fused driver launched past its capacity";
    auto r = make_sys<T>(over);
    this->run_pinned(C{gs::Blocked{}}, r);
    expect_solved(r, "blocked past the cta capacity " + label(over));
}

// The pinned facade runs exactly that family's driver: bit-identical to the direct call.
TYPED_TEST(GetrsCandidates, PinnedRunIsTheDirectKernelBitForBit) {
    using T = typename TestFixture::T;
    int compared = 0;
    for (const C& c : gs::candidates<T>())
        for (Transpose t : {Transpose::NoTrans, Transpose::ConjTrans})
            for (auto [n, nrhs] : {std::pair{13, 3}, std::pair{40, 8}, std::pair{70, 20}}) {
                if (!this->expect_runs(c, n, nrhs)) continue;
                Spec s{n, nrhs, 3, t};
                s.seed = 4242u + n;
                auto pinned = make_sys<T>(s);
                auto direct = make_sys<T>(s);
                const std::string what = name(c, s);
                this->run_pinned(c, pinned);
                std::string why;
                ASSERT_TRUE(this->direct(c, direct, &why)) << what << ": " << why;
                expect_solved(pinned, what);
                for (std::size_t e = 0; e < pinned.b.size(); ++e)
                    ASSERT_TRUE(same_bits(pinned.b[e], direct.b[e]))
                        << what << ": the pinned facade did not run this family's driver; element " << e;
                ++compared;
            }
    EXPECT_GT(compared, 0);
}

// SLM tiers at a saturating batch: 1024 items repeating 7 systems; every item bit-identical to
// its representative (the fused kernel's resident RHS, Blocked's SLM gather at nrhs >= 16).
TYPED_TEST(GetrsCandidates, SaturatingBatchIsBitIdenticalToItsRepresentative) {
    using T = typename TestFixture::T;
    struct Case { C c; int n, nrhs; };
    const Case cases[] = {{gs::Cta{}, 40, 3}, {gs::Cta{}, 97, 8}, {gs::Blocked{}, 70, 20},
                          {gs::Blocked{}, 33, 5}, {gs::Vendor{}, 24, 4}};
    for (const auto& k : cases) {
        if (!this->expect_runs(k.c, k.n, k.nrhs)) continue;
        for (Transpose t : {Transpose::NoTrans, Transpose::Trans}) {
            Spec s{k.n, k.nrhs, 1024, t};
            s.period = 7;
            s.seed = 777u + k.n;
            auto p = make_sys<T>(s);
            this->run_pinned(k.c, p);
            expect_solved(p, name(k.c, s));
            if (::testing::Test::HasFatalFailure()) return;
        }
    }
}

// §8.2 (R3): on every straddling shape a pin is accepted exactly when the family's driver launches.
TYPED_TEST(GetrsCandidates, CanRunEqualsLaunch) {
    using T = typename TestFixture::T;
    const int nfit = int(this->cap() / 8);
    int disagreements = 0;
    for (const C& c : gs::candidates<T>()) {
        if (TestFixture::vendor_word_falls_back(c)) continue;
        for (auto [n, nrhs] : {std::pair{1, 1}, std::pair{5, 8}, std::pair{5, 9}, std::pair{40, 4},
                               std::pair{40, 64}, std::pair{nfit, 8}, std::pair{nfit + 1, 8}}) {
            Spec s{n, nrhs, 2, Transpose::Trans};
            s.seed = 61u + n;
            auto a = make_sys<T>(s);
            auto b = make_sys<T>(s);
            const bool pin = this->pin_accepted(c, a);
            std::string why;
            const bool run = this->direct(c, b, &why);
            EXPECT_EQ(pin, run) << name(c, s) << ": can_run says " << pin << ", the driver "
                                << (run ? "launches" : "refuses: " + why);
            EXPECT_EQ(pin, this->expect_runs(c, n, nrhs)) << name(c, s) << ": can_run disagrees with the oracle";
            disagreements += pin != run;
        }
    }
    EXPECT_EQ(disagreements, 0);
}

// §8.3 (R5): under every pin, a workspace of exactly getrs_buffer_size bytes inside a poisoned
// arena; nothing past it is written.
TYPED_TEST(GetrsCandidates, WorkspaceIsExactlyWhatTheChoiceNeeds) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    for (const C& c : gs::candidates<T>()) {
        if (TestFixture::vendor_word_falls_back(c)) continue;
        for (auto [n, nrhs] : {std::pair{24, 2}, std::pair{70, 20}}) {
            if (!this->expect_runs(c, n, nrhs)) continue;
            Spec s{n, nrhs, 5, Transpose::NoTrans};
            s.seed = 5u + n;
            auto p = make_sys<T>(s);
            const Pin pin("getrs", c);
            auto A = p.A();
            auto Bm = p.B();
            const std::size_t need = getrs_buffer_size<B, T>(*this->ctx, A, Bm, s.trans);
            if (std::holds_alternative<gs::Cta>(c))
                EXPECT_EQ(need, sycl_getrs::getrs_fused_buffer_size<T>(*this->ctx, A, Bm, s.trans));
            if (std::holds_alternative<gs::Blocked>(c))
                EXPECT_EQ(need, sycl_getrs::getrs_blocked_buffer_size<T>(*this->ctx, A, Bm, s.trans));
            constexpr std::size_t kTail = 4096;
            UnifiedVector<std::byte> arena(need + kTail, std::byte{0xA5});
            (void)getrs<B, T>(*this->ctx, A, Bm, s.trans, p.piv.to_span(), Span<std::byte>(arena.data(), need));
            this->ctx->wait();
            for (std::size_t e = need; e < arena.size(); ++e)
                ASSERT_EQ(arena[e], std::byte{0xA5}) << name(c, s) << ": wrote past the workspace at byte " << e;
            expect_solved(p, name(c, s) + " (exact workspace)");
        }
    }
}

// R6: a spelling that names nothing compiled throws instead of meaning Auto.
TYPED_TEST(GetrsCandidates, UnknownPinsThrow) {
    using T = typename TestFixture::T;
    for (const char* word : {"bogus", "cta:1", "blocked:nb=16", "native:vendor", "native:tiny", "fused", "v1",
                             // removed aliases (phase 5): each must stay an error
                             "native:cta", "native:blocked"}) {
        auto p = make_sys<T>(Spec{});
        const Pin pin("getrs", std::string_view(word));
        EXPECT_THROW(this->run(p), std::invalid_argument) << word;
    }
}

// The named can_run-false shapes (absorbs RouteGetrs.CorrectnessGatesAreNotSpeedGates and
// FusedGetrsHandsBackAtBothCeilings' table half): empty order or rhs, one past the width,
// a non-square A, B's rows or batch disagreeing with A, an empty batch on both sides.
TYPED_TEST(GetrsCandidates, CanRunFalsePinsThrow) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    struct Case { C c; int an, ac, bn, bc, ab, bb; const char* what; };
    const int w = int(sycl_getrs::kGetrsFusedMaxRhs);
    const Case cases[] = {{gs::Cta{}, 0, 0, 0, 2, 1, 1, "order 0"},        {gs::Blocked{}, 0, 0, 0, 2, 1, 1, "order 0"},
                          {gs::Cta{}, 8, 8, 8, 0, 1, 1, "nrhs 0"},         {gs::Blocked{}, 8, 8, 8, 0, 1, 1, "nrhs 0"},
                          {gs::Cta{}, 8, 8, 8, w + 1, 1, 1, "nrhs past the width"},
                          {gs::Cta{}, 8, 10, 8, 2, 1, 1, "A not square"},  {gs::Blocked{}, 8, 10, 8, 2, 1, 1, "A not square"},
                          {gs::Cta{}, 8, 8, 9, 2, 1, 1, "B rows != n"},    {gs::Blocked{}, 8, 8, 9, 2, 1, 1, "B rows != n"},
                          {gs::Cta{}, 8, 8, 8, 2, 2, 1, "batch mismatch"}, {gs::Blocked{}, 8, 8, 8, 2, 2, 1, "batch mismatch"},
                          {gs::Cta{}, 8, 8, 8, 2, 0, 0, "batch 0"},        {gs::Blocked{}, 8, 8, 8, 2, 0, 0, "batch 0"}};
    for (const auto& k : cases) {
        UnifiedVector<T> a(std::size_t(std::max(1, k.an * k.ac)) * 2, T(1)), b(std::size_t(std::max(1, k.bn * k.bc)) * 2, T(1));
        const MVof<T> A(a.data(), k.an, k.ac, std::max(1, k.an), std::max(1, k.an * k.ac), k.ab);
        const MVof<T> Bm(b.data(), k.bn, k.bc, std::max(1, k.bn), std::max(1, k.bn * k.bc), k.bb);
        const Pin pin("getrs", k.c);
        try {
            (void)getrs_buffer_size<B, T>(*this->ctx, A, Bm, Transpose::NoTrans);
            ADD_FAILURE() << select::to_string(k.c) << " " << k.what << " was accepted";
        } catch (const std::invalid_argument& e) {
            EXPECT_NE(std::string(e.what()).find("cannot run this shape"), std::string::npos)
                << select::to_string(k.c) << " " << k.what << ": " << e.what();
        }
    }
}

// RouteLuPivotFormat's getrs half, ported: under Backend::NETLIB (true int64 pivots) no native
// family runs even on a GPU queue, because they read packed int32.
TYPED_TEST(GetrsCandidates, NetlibBackendRunsNoNativeFamily) {
#if BATCHLAS_HAS_HOST_BACKEND
    using T = typename TestFixture::T;
    auto p = make_sys<T>(Spec{40, 3, 2});
    for (const C& c : {C{gs::Cta{}}, C{gs::Blocked{}}}) {
        const Pin pin("getrs", c);
        auto A = p.A();
        auto Bm = p.B();
        EXPECT_THROW(((void)getrs_buffer_size<Backend::NETLIB, T>(*this->ctx, A, Bm, Transpose::NoTrans)),
                     std::invalid_argument)
            << select::to_string(c);
        EXPECT_TRUE(this->pin_accepted(c, p)) << "guard: the same pin runs under the GPU backend";
    }
#else
    GTEST_SKIP() << "no host backend in this build";
#endif
}

// A heterogeneous A or B: one launch has one order, so every native pin throws. Auto takes the
// vendor as before; vendor-free there is no route.
TYPED_TEST(GetrsCandidates, HeterogeneousBatchHasNoNativeRoute) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    const int n = 16, q = 3, batch = 4;
    Matrix<T, MatrixFormat::Dense> A(n, n, batch), Bm(n, q, batch);
    UnifiedVector<int> act(batch), cols(batch);
    for (int b = 0; b < batch; ++b) act[b] = n - b, cols[b] = q - (b % 2);
    const auto hetA = A.view().with_active_dims(act.to_span(), act.to_span());
    const auto hetB = Bm.view().with_active_dims(act.to_span(), cols.to_span());
    ASSERT_TRUE(hetA.is_heterogeneous());
    struct Case { const char* what; MVof<T> a, b; };
    const Case cases[] = {{"A heterogeneous", hetA, Bm.view()}, {"B heterogeneous", A.view(), hetB}};
    for (const auto& k : cases) {
        for (const C& c : {C{gs::Cta{}}, C{gs::Blocked{}}}) {
            const Pin pin("getrs", c);
            EXPECT_THROW(((void)getrs_buffer_size<B, T>(*this->ctx, k.a, k.b, Transpose::NoTrans)), std::invalid_argument)
                << k.what << " " << select::to_string(c);
        }
        const ScopedEnvVar clear("BATCHLAS_GETRS_ROUTE", nullptr);
        if constexpr (!TestFixture::kVendor)
            EXPECT_THROW(((void)getrs_buffer_size<B, T>(*this->ctx, k.a, k.b, Transpose::NoTrans)), batchlas::NoRouteError)
                << k.what;
    }
}

// §5.3: spellings (case-folded) and the class words, via ScopedPin and via the environment.
TYPED_TEST(GetrsCandidates, ClassWordsAndSpellings) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_GETRS_ROUTE", nullptr);
    const Spec s{40, 6, 4};  // the old router: vendor with a vendor, cta without
    std::string auto_pick;
    {
        auto p = make_sys<T>(s);
        auto_pick = this->auto_choice(p);
    }
    ASSERT_EQ(auto_pick, TestFixture::kVendor ? "vendor" : "cta");
    const std::string native_pick = "cta";
    const std::pair<const char*, std::string> expect[] = {
        {"CTA", "cta"}, {"cta", "cta"},
        {"Blocked", "blocked"}, {"vendor", TestFixture::kVendor ? "vendor" : auto_pick}, {"native", native_pick},
        {"auto", auto_pick}};
    for (const auto& [word, spelling] : expect)
        for (bool via_env : {false, true}) {
            auto p = make_sys<T>(s);
            select::testing::reset_warnings();
            std::string err;
            const std::string got = traced_choice([&] {
                const ScopedEnvVar env("BATCHLAS_GETRS_ROUTE", via_env ? word : nullptr);
                std::optional<Pin> pin;
                if (!via_env) pin.emplace("getrs", std::string_view(word));
                this->run(p);
            }, &err);
            const std::string what = std::string(word) + (via_env ? " via BATCHLAS_GETRS_ROUTE" : " via ScopedPin");
            EXPECT_EQ(got, spelling) << what;
            expect_solved(p, what);
        }
}

// §5.3: a ScopedPin wins over BATCHLAS_GETRS_ROUTE, and nested pins restore the outer one.
TYPED_TEST(GetrsCandidates, ScopedPinBeatsTheEnvironment) {
    using T = typename TestFixture::T;
    const ScopedEnvVar env("BATCHLAS_GETRS_ROUTE", "cta");
    const Spec wide{40, int(sycl_getrs::kGetrsFusedMaxRhs) + 1, 2};
    {
        auto p = make_sys<T>(wide);
        EXPECT_THROW(this->run(p), std::invalid_argument) << "the environment pin was not read";
    }
    auto p = make_sys<T>(wide);
    EXPECT_EQ(traced_choice([&] {
                  const Pin pin("getrs", C{gs::Blocked{}});
                  this->run(p);
              }),
              "blocked");
    expect_solved(p, "blocked over env cta");
    const Spec s{20, 2, 2};
    const Pin outer("getrs", C{gs::Cta{}});
    {
        const Pin inner("getrs", C{gs::Blocked{}});
        auto r = make_sys<T>(s);
        EXPECT_EQ(traced_choice([&] { this->run(r); }), "blocked");
    }
    auto r = make_sys<T>(s);
    EXPECT_EQ(traced_choice([&] { this->run(r); }), "cta") << "the inner pin did not restore the outer";
    expect_solved(r, "outer cta");
}

// Auto on the shipped (transcribed) table equals the old router on both sides of every
// threshold it read, with and without a vendor: n 31/32, nrhs 2/3, 4/5 (float), 8/9, 63/64,
// 127/128, batch 127/128; on- and off-grid. Skips on a device without its own getrs table.
TYPED_TEST(GetrsCandidates, AutoReadsTheTranscribedTable) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    const ScopedEnvVar clear("BATCHLAS_GETRS_ROUTE", nullptr);
    const std::string dt(select::dtype_name<T>());
    const auto& d = select::device_of<B>(*this->ctx);
    const auto tables = select::tables_in_borrow_order("getrs", dt, d);
    if (tables.empty() || tables.front()->device != d.key) GTEST_SKIP() << "no own getrs table for " << d.key;
    struct Cell { int n, nrhs, batch; };
    const Cell cells[] = {{31, 1, 128}, {32, 1, 128}, {40, 2, 200},  {40, 3, 200},   {45, 4, 128},  {45, 5, 128},
                          {36, 8, 128}, {36, 9, 128}, {20, 63, 128}, {20, 64, 128},  {20, 64, 127}, {24, 100, 300},
                          {20, 127, 128}, {20, 128, 128}, {20, 128, 100}, {300, 1, 3}, {600, 2, 1000}};
    for (const Cell& k : cells) {
        Spec s{k.n, k.nrhs, k.batch, Transpose::Trans};
        s.seed = 41u + k.n;
        auto p = make_sys<T>(s);
        const std::string want = old_auto(dt, k.n, k.nrhs, k.batch, TestFixture::kVendor, this->cta_fits(k.n, k.nrhs));
        EXPECT_EQ(traced_choice([&] { this->run(p); }), want) << label(s);
        expect_solved(p, "auto " + label(s));
    }
}

// key_of's every field reaches choose(): a synthetic table whose winner changes with n, nrhs
// and batch alone. Fixing a field in key_of turns exactly its row red.
TYPED_TEST(GetrsCandidates, AutoReadsEveryKeyField) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_GETRS_ROUTE", nullptr);
    const TableGuard restore;
    this->use_table("n=8 nrhs=2 batch=128 | cta 1 | blocked 2\n"
                    "n=8 nrhs=2 batch=32768 | blocked 1 | cta 2\n"
                    "n=8 nrhs=6 batch=128 | blocked 1 | cta 2\n"
                    "n=64 nrhs=2 batch=128 | blocked 1 | cta 2\n");
    struct Probe { int n, nrhs, batch; const char* expect; const char* field; };
    const Probe probes[] = {{8, 2, 128, "cta", "base"}, {8, 2, 32768, "blocked", "batch"},
                            {8, 6, 128, "blocked", "nrhs (B.cols)"}, {64, 2, 128, "blocked", "n (A.rows)"}};
    for (const auto& k : probes) {
        Spec s{k.n, k.nrhs, k.batch, Transpose::NoTrans};
        s.seed = 43u;
        auto p = make_sys<T>(s);
        EXPECT_EQ(traced_choice([&] { this->run(p); }), k.expect) << "the " << k.field << " row";
        expect_solved(p, std::string("the ") + k.field + " row");
    }
}

// The trace prints the lookup key itself.
TYPED_TEST(GetrsCandidates, TraceLineShowsTheKey) {
    using T = typename TestFixture::T;
    Spec s{12, 7, 5, Transpose::ConjTrans};
    auto p = make_sys<T>(s);
    const std::string line = traced_line([&] { this->run(p); });
    EXPECT_NE(line.find("n=12 nrhs=7 batch=5 ->"), std::string::npos) << line;
    expect_solved(p, line);
}

// RouteGetrs.VendorFreeFallbackHandsOverTheNativeRoute, ported: with a table naming only the
// vendor, a vendor-free build takes the last resort, blocked, at any width.
TYPED_TEST(GetrsCandidates, VendorFreeLastResortIsBlocked) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_GETRS_ROUTE", nullptr);
    const TableGuard restore;
    this->use_table("n=32 nrhs=1 batch=128 | vendor 1\n");
    for (const Spec& s : {Spec{32, 1, 4}, Spec{200, 17, 2}}) {
        auto p = make_sys<T>(s);
        std::string all;
        const std::string got = traced_choice([&] { this->run(p); }, &all);
        EXPECT_EQ(got, TestFixture::kVendor ? "vendor" : "blocked") << label(s);
        if (!TestFixture::kVendor) EXPECT_NE(all.find("last resort"), std::string::npos) << all;
        expect_solved(p, "vendor-only table, " + label(s));
    }
}

// The coverage row (§5.6): the real backend, getrs's key (m = k = order, n = nrhs, transA) and
// the native flags. threadsafe: the child re-executes the binary, so CUDA starts fresh.
TYPED_TEST(GetrsCandidates, CoverageRowCarriesBackendKeyAndNativeFlags) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    const std::string dir = ::testing::TempDir() + "getrs_cov." + std::string(select::dtype_name<T>());
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const std::string out = dir + "/cov";
    const Spec lo{16, 4, 2, Transpose::NoTrans};
    const Spec hi{48, 3, 2, Transpose::ConjTrans};
    auto child = [&] {
        const ScopedEnvVar cov("BATCHLAS_COVERAGE_OUT", out.c_str());
        const ScopedEnvVar clear("BATCHLAS_GETRS_ROUTE", nullptr);
        batchlas::coverage::g_dynamic_enabled = true;
        {
            auto p = make_sys<T>(lo);
            this->run_pinned(C{gs::Cta{}}, p);
        }
        auto p = make_sys<T>(hi);
        this->run_pinned(C{gs::Blocked{}}, p);
        std::exit(0);
    };
    EXPECT_EXIT(child(), ::testing::ExitedWithCode(0), "");
    std::map<std::string, std::vector<std::string>> rows;  // "m n k" -> fields
    for (const auto& ent : std::filesystem::directory_iterator(dir)) {
        std::ifstream in(ent.path());
        for (std::string line; std::getline(in, line);) {
            if (line.rfind("reached,getrs,", 0) != 0) continue;
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
    for (const auto& [key, s, algo] : {std::tuple{std::string("16 4 16"), lo, "cta"},
                                       std::tuple{std::string("48 3 48"), hi, "blocked"}}) {
        ASSERT_TRUE(rows.count(key)) << "no row " << key;
        const auto& f = rows[key];
        EXPECT_EQ(f[3], backend) << key;
        EXPECT_EQ(f[10], algo) << key;
        EXPECT_EQ(f[12], "1") << key;
        EXPECT_EQ(f[13], "1") << key;
        EXPECT_EQ(f[18], as_int(s.trans)) << key;
    }
}

// RouteGetrs.CorrectnessGatesAreNotSpeedGates' CPU case, ported: on a CPU queue no native
// family runs, so native pins throw and Auto is the vendor (or no route without one).
template <typename Config>
class GetrsCandidatesCpu : public test_utils::BatchLASTest<Config> {};
TYPED_TEST_SUITE(GetrsCandidatesCpu, Types);

TYPED_TEST(GetrsCandidatesCpu, CpuQueueRunsNoNativeFamily) {
    using T = typename TypeParam::ScalarType;
    static constexpr Backend B = TypeParam::BackendVal;
    if (!this->ctx) GTEST_SKIP() << "no queue";
    if (this->ctx->device().type == DeviceType::GPU) GTEST_SKIP() << "a GPU queue";
    const ScopedEnvVar clear("BATCHLAS_GETRS_ROUTE", nullptr);
    auto p = make_sys<T>(Spec{8, 3, 2});
    auto A = p.A();
    auto Bm = p.B();
    for (const C& c : {C{gs::Cta{}}, C{gs::Blocked{}}}) {
        const Pin pin("getrs", c);
        EXPECT_THROW(((void)getrs_buffer_size<B, T>(*this->ctx, A, Bm, Transpose::NoTrans)), std::invalid_argument)
            << select::to_string(c);
    }
    if constexpr (batchlas::select::factorization_vendor_available<B>)
        EXPECT_NO_THROW(((void)getrs_buffer_size<B, T>(*this->ctx, A, Bm, Transpose::NoTrans)));
    else
        EXPECT_THROW(((void)getrs_buffer_size<B, T>(*this->ctx, A, Bm, Transpose::NoTrans)), batchlas::NoRouteError);
}

// The transcribed rows, read with Table::nearest directly so every machine checks them: for
// both devices and every dtype, the first entry (vendor present) and the first native entry
// that fits (vendor-free) equal the old router at straddles of each threshold.
TEST(GetrsTranscribedTable, RowsHoldTheOldPreferenceOnBothSidesOfEveryThreshold) {
    const int w = int(sycl_getrs::kGetrsFusedMaxRhs);
    for (const char* dev : {"sm_89", "sm_120"})
        for (const char* dt : {"float", "double", "cfloat", "cdouble"}) {
            const auto tables = select::tables_in_borrow_order("getrs", dt, select::device_from_key(dev));
            ASSERT_FALSE(tables.empty()) << dt;
            const select::Table& t = *tables.front();
            ASSERT_EQ(t.device, dev) << dt;
            for (int n : {1, 20, 31, 32, 33, 700, 5000})
                for (int nrhs : {1, 2, 3, 4, 5, 8, 9, 40, 63, 64, 65, 127, 128, 129, 2000})
                    for (int batch : {1, 100, 127, 128, 129, 40000}) {
                        const select::TableRow* row = t.nearest({{"n", n}, {"nrhs", nrhs}, {"batch", batch}});
                        ASSERT_NE(row, nullptr);
                        const std::string what = t.file + ":" + std::to_string(row->line) + " n=" + std::to_string(n) +
                                                 " nrhs=" + std::to_string(nrhs) + " batch=" + std::to_string(batch);
                        EXPECT_FALSE(row->timed) << what;
                        const bool fits = nrhs <= w;  // unlimited SLM: the width is the only cta term here
                        std::string first, native;
                        for (const auto& e : row->ranked) {
                            if (e.spelling == "cta" && !fits) continue;
                            if (first.empty()) first = e.spelling;
                            if (native.empty() && e.spelling != "vendor") native = e.spelling;
                        }
                        EXPECT_EQ(first, old_auto(dt, n, nrhs, batch, true, fits)) << what << " (vendor present)";
                        EXPECT_EQ(native, old_auto(dt, n, nrhs, batch, false, fits)) << what << " (vendor-free)";
                    }
        }
}

}  // namespace
