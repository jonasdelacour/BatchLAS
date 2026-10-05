// Every posv candidate, pinned: docs/design/flat-kernel-selection.md §8 and
// flat-kernel-selection-phase3-plan.md §1.1. The limit oracle reads the drivers' own capacity
// queries; which kernel ran is read back from the select trace or a bit-for-bit comparison
// with the direct driver, never assumed from the pin being accepted.
#include <gtest/gtest.h>

#include <batchlas/blas/functions/posv.hh>
#include <batchlas/blas/functions/potrf.hh>
#include <batchlas/blas/functions/trsm.hh>
#include <batchlas/blas/dispatch/no_route.hh>
#include <batchlas/blas/dispatch/vendor_available.hh>
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>

#include "test_utils.hh"

#include "../src/extensions/getrs_native.hh"
#include "../src/extensions/solve_native.hh"
#include "../src/ops/posv/choice.hh"
#include "../src/ops/potrf/choice.hh"
#include "../src/util/resident_capacity.hh"

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
#include <variant>
#include <vector>

using namespace batchlas;

namespace {

namespace ps = batchlas::ops::posv;
using C = ps::PosvChoice;
using Pin = select::ScopedPin<C>;
using PotrfPin = select::ScopedPin<ops::potrf::PotrfChoice>;

template <typename T>
using RealOf = typename batchlas::base_type<T>::type;
template <typename T>
constexpr bool kCx = test_utils::is_complex<T>::value;

template <typename T>
T cj(T v) {
    if constexpr (kCx<T>) return std::conj(v);
    else return v;
}
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

// Hermitian, strictly diagonally dominant (condition number below 8); every off-diagonal
// entry has a nonzero imaginary part.
template <typename T>
std::vector<T> make_hpd(int n, std::mt19937& gen) {
    using R = RealOf<T>;
    std::uniform_real_distribution<R> d(R(0.1), R(1));
    std::vector<T> A(static_cast<size_t>(n) * n);
    for (int j = 0; j < n; ++j) {
        A[j + static_cast<size_t>(j) * n] = mk<T>(R(2.5) + R(0.5) * d(gen), R(0));
        for (int i = j + 1; i < n; ++i) {
            const T v = mk<T>((gen() & 1 ? d(gen) : -d(gen)) / R(n), d(gen) / R(n));
            A[i + static_cast<size_t>(j) * n] = v;
            A[j + static_cast<size_t>(i) * n] = cj(v);
        }
    }
    return A;
}

// A and B at padded lds and strides that are not ld*cols. Everything outside the owned
// triangle of A and the n x nrhs block of B -- the other triangle, the ld pads and the
// inter-item gaps -- holds a large finite value a kernel would happily consume.
template <typename T>
struct Sys {
    int n = 0, nrhs = 0, batch = 0, lda = 0, stra = 0, ldb = 0, strb = 0;
    Uplo uplo = Uplo::Lower;
    UnifiedVector<T> a, b;
    UnifiedVector<T*> aptr, bptr;
    std::vector<T> a0, b0;
    std::vector<std::vector<T>> full;  // the whole Hermitian A per item

    MatrixView<T, MatrixFormat::Dense> A() { return {a.data(), n, n, lda, stra, batch, aptr.data()}; }
    MatrixView<T, MatrixFormat::Dense> B() { return {b.data(), n, nrhs, ldb, strb, batch, bptr.data()}; }
    bool in_tri(int i, int j) const { return uplo == Uplo::Lower ? i >= j : i <= j; }
};

template <typename T>
Sys<T> make_sys(int n, int nrhs, int batch, Uplo uplo, unsigned seed, bool identical = false) {
    using R = RealOf<T>;
    Sys<T> p;
    p.n = n, p.nrhs = nrhs, p.batch = batch, p.uplo = uplo;
    p.lda = n + 3, p.stra = p.lda * n + 5, p.ldb = n + 2, p.strb = p.ldb * nrhs + 7;
    const T poison = mk<T>(R(-999), R(777));
    p.a = UnifiedVector<T>(static_cast<size_t>(p.stra) * batch, poison);
    p.b = UnifiedVector<T>(static_cast<size_t>(p.strb) * batch, poison);
    p.aptr = UnifiedVector<T*>(batch, nullptr);
    p.bptr = UnifiedVector<T*>(batch, nullptr);
    p.full.resize(batch);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<R> d(R(-1), R(1));
    std::vector<T> rhs(static_cast<size_t>(n) * nrhs);
    for (int it = 0; it < batch; ++it) {
        p.full[it] = (identical && it > 0) ? p.full[0] : make_hpd<T>(n, gen);
        if (!identical || it == 0)
            for (auto& v : rhs) v = mk<T>(d(gen), d(gen));
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                if (p.in_tri(i, j))
                    p.a[size_t(it) * p.stra + size_t(j) * p.lda + i] = p.full[it][i + size_t(j) * n];
        for (int k = 0; k < nrhs; ++k)
            for (int i = 0; i < n; ++i) p.b[size_t(it) * p.strb + size_t(k) * p.ldb + i] = rhs[i + size_t(k) * n];
    }
    p.a0.assign(p.a.begin(), p.a.end());
    p.b0.assign(p.b.begin(), p.b.end());
    return p;
}

// ||A X - B||_F / (||A||_F ||X||_F) in double.
template <typename T>
double residual(const Sys<T>& p, int it) {
    double num = 0, an = 0, xn = 0;
    for (const T& e : p.full[it]) an += std::norm(up(e));
    for (int k = 0; k < p.nrhs; ++k)
        for (int i = 0; i < p.n; ++i) {
            const size_t col = size_t(it) * p.strb + size_t(k) * p.ldb;
            xn += std::norm(up(p.b[col + i]));
            std::complex<double> acc = -up(p.b0[col + i]);
            for (int t = 0; t < p.n; ++t) acc += up(p.full[it][i + size_t(t) * p.n]) * up(p.b[col + t]);
            num += std::norm(acc);
        }
    return (an == 0 || xn == 0) ? std::sqrt(num) : std::sqrt(num / (an * xn));
}

template <typename T>
double tol(int n) {
    return 64.0 * std::max(n, 1) * double(std::numeric_limits<RealOf<T>>::epsilon());
}

// info, the residual of the first and last item, and every element the solve must not touch,
// bit for bit. `other_is_scratch`: a vendor potrf ran (cuSOLVER Upper writes the lower triangle).
template <typename T>
void expect_solved(const Sys<T>& p, const std::vector<int32_t>& info, const std::string& what,
                   bool other_is_scratch = false) {
    for (int it = 0; it < p.batch; ++it) ASSERT_EQ(info[it], 0) << what << " item " << it;
    for (int it : {0, p.batch - 1}) {
        const double r = residual(p, it);
        EXPECT_TRUE(std::isfinite(r) && r <= tol<T>(p.n)) << what << " item " << it << " residual " << r;
    }
    for (size_t e = 0; e < p.a0.size(); ++e) {
        const int r = int(e % p.stra), i = r % p.lda, j = r / p.lda;
        const bool owned = j < p.n && i < p.n && (other_is_scratch || p.in_tri(i, j));
        if (!owned)
            ASSERT_TRUE(same_bits(p.a[e], p.a0[e])) << what << ": A written outside its triangle at element " << e
                                                    << " (item " << e / p.stra << " i=" << i << " j=" << j << ")";
    }
    for (size_t e = 0; e < p.b0.size(); ++e) {
        const int r = int(e % p.strb), i = r % p.ldb, k = r / p.ldb;
        if (!(k < p.nrhs && i < p.n))
            ASSERT_TRUE(same_bits(p.b[e], p.b0[e])) << what << ": B written in its padding at element " << e;
    }
}

// The choice the outermost posv trace line names for whatever `run` calls.
template <class F>
std::string traced_choice(F&& run, std::string* all = nullptr) {
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
    for (std::string line; std::getline(in, line);) {
        const auto arrow = line.find(" -> ");
        if (line.rfind("posv ", 0) != 0 || arrow == std::string::npos) continue;
        const std::string tail = line.substr(arrow + 4);
        return tail.substr(0, tail.find(' '));
    }
    return "<no posv trace line in: " + err + ">";
}

// Restores the embedded tables on scope exit, whatever a test installed.
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
class PosvCandidates : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    using MV = MatrixView<T, MatrixFormat::Dense>;
    static constexpr Backend B = Config::BackendVal;

    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        if (this->HasFatalFailure() || ::testing::Test::IsSkipped()) return;
        if (!this->ctx) GTEST_SKIP() << "no queue";
        if (this->ctx->device().type != DeviceType::GPU) GTEST_SKIP() << "the native tiers are GPU kernels";
        if (!this->ctx->device().supports_sub_group_size(32)) GTEST_SKIP() << "no sub-group size 32";
    }

    // ---- the limit oracle: each driver's own capacity query, not posv.cc's can_run ----
    int max_wg() const {
        return static_cast<int>(this->ctx->device().get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    }
    int tiny_n() const {
        return max_wg() >= sycl_posv::kPosvTinyWgSize ? sycl_posv::posv_tiny_max_n<T>() : 0;
    }
    std::size_t cta_elems() const {
        return sycl_getrs::getrs_fused_max_rhs_elems<T>(
            resident::device_slm_budget(this->ctx->device().get_property(DeviceProperty::LOCAL_MEM_SIZE)));
    }
    bool expect_runs(const C& c, int n, int nrhs) const {
        if (std::holds_alternative<ps::Tiny>(c)) return n <= tiny_n() && nrhs <= sycl_posv::kPosvTinyMaxRhs;
        if (std::holds_alternative<ps::Cta>(c))
            return nrhs <= sycl_getrs::kGetrsFusedMaxRhs && std::size_t(n) * std::size_t(nrhs) <= cta_elems();
        return true;
    }

    // Whether some non-vendor potrf candidate runs this A: then a `native` potrf pin keeps
    // the other triangle; otherwise it falls to the vendor (or, vendor-free, has no route).
    bool potrf_native(Sys<T>& p) {
        for (const auto& c : ops::potrf::candidates<T>()) {
            if (select::family_of(c) == "vendor") continue;
            const PotrfPin pin("potrf", c);
            try {
                (void)potrf_buffer_size<B, T>(*this->ctx, p.A(), p.uplo);
                return true;
            } catch (const std::invalid_argument&) {
            }
        }
        return false;
    }
    // Tiny never calls potrf, so only the composed families depend on it.
    bool keeps_other(const C& c, Sys<T>& p) { return std::holds_alternative<ps::Tiny>(c) || potrf_native(p); }
    bool child_has_no_route(const C& c, Sys<T>& p) {
        return !keeps_other(c, p) && !dispatch::solver_vendor_available<B>;
    }

    // Whether a pin of `c` is accepted: posv_buffer_size runs choose() and launches nothing.
    // A child (potrf) without a route still means the posv pin itself was taken.
    bool pin_accepted(const C& c, const MV& A, const MV& Bm, Uplo uplo) {
        const Pin pin("posv", c);
        try {
            (void)posv_buffer_size<B, T>(*this->ctx, A, Bm, uplo);
            return true;
        } catch (const dispatch::NoRouteError&) {
            return true;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find("cannot run this shape") == std::string::npos) throw;
            return false;
        }
    }

    // Whatever posv pin is in force, sized by the same choose(); potrf is pinned native so
    // the poisoned other triangle survives the factor wherever a native potrf exists.
    std::vector<int32_t> run_auto(Sys<T>& p, bool pass_info = true) {
        const PotrfPin native("potrf", "native");
        const std::size_t bytes = posv_buffer_size<B, T>(*this->ctx, p.A(), p.B(), p.uplo);
        UnifiedVector<std::byte> ws(std::max<std::size_t>(bytes, 1));
        UnifiedVector<int32_t> info(p.batch, int32_t(-7));
        (void)posv<B, T>(*this->ctx, p.A(), p.B(), p.uplo, Span<std::byte>(ws.data(), bytes),
                         pass_info ? info.to_span() : Span<int32_t>{});
        this->ctx->wait();
        return pass_info ? std::vector<int32_t>(info.begin(), info.end()) : std::vector<int32_t>(p.batch, 0);
    }
    std::vector<int32_t> run_pinned(const C& c, Sys<T>& p) {
        const Pin pin("posv", c);
        return run_auto(p);
    }

    // The family's own kernels, called directly: Tiny's driver; Cta = the public potrf plus
    // potrs_fused_dispatch; Blocked = the public potrf plus the two public trsm calls.
    void run_direct(const C& c, const MV& A, const MV& Bm, Uplo uplo, Span<int32_t> info) {
        Queue& q = *this->ctx;
        const PotrfPin native("potrf", "native");
        auto factor = [&] {
            const std::size_t bytes = potrf_buffer_size<B, T>(q, A, uplo);
            UnifiedVector<std::byte> ws(std::max<std::size_t>(bytes, 1));
            (void)potrf<B, T>(q, A, uplo, Span<std::byte>(ws.data(), bytes), info);
            q.wait();
        };
        if (std::holds_alternative<ps::Tiny>(c)) {
            const std::size_t bytes = sycl_posv::posv_tiny_buffer_size<T>(q, A, Bm);
            UnifiedVector<std::byte> ws(std::max<std::size_t>(bytes, 1));
            (void)sycl_posv::posv_tiny_dispatch<T>(q, A, Bm, uplo, Span<std::byte>(ws.data(), bytes), info);
        } else if (std::holds_alternative<ps::Cta>(c)) {
            factor();
            (void)sycl_getrs::potrs_fused_dispatch<T>(q, A, Bm, uplo);
        } else {
            factor();
            const Transpose adj = kCx<T> ? Transpose::ConjTrans : Transpose::Trans;
            const T one(1);
            const Transpose first = uplo == Uplo::Lower ? Transpose::NoTrans : adj;
            const Transpose second = uplo == Uplo::Lower ? adj : Transpose::NoTrans;
            (void)trsm<B, T>(q, A, Bm, one, Side::Left, uplo, first, Diag::NonUnit);
            (void)trsm<B, T>(q, A, Bm, one, Side::Left, uplo, second, Diag::NonUnit);
        }
        q.wait();
    }
    // Only the driver's own argument checks: no factor, so a refusal is the driver's.
    bool direct_launches(const C& c, const MV& A, const MV& Bm, Uplo uplo, std::string* why = nullptr) {
        Queue& q = *this->ctx;
        UnifiedVector<int32_t> info(std::max<int>(A.batch_size(), 1), int32_t(0));
        try {
            if (std::holds_alternative<ps::Tiny>(c)) {
                const std::size_t bytes = sycl_posv::posv_tiny_buffer_size<T>(q, A, Bm);
                UnifiedVector<std::byte> ws(std::max<std::size_t>(bytes, 1));
                (void)sycl_posv::posv_tiny_dispatch<T>(q, A, Bm, uplo, Span<std::byte>(ws.data(), bytes),
                                                       info.to_span());
            } else if (std::holds_alternative<ps::Cta>(c)) {
                (void)sycl_getrs::potrs_fused_dispatch<T>(q, A, Bm, uplo);
            } else {
                return true;  // Blocked has no driver of its own; its children decide
            }
            q.wait();
            return true;
        } catch (const std::exception& e) {
            if (why) *why = e.what();
            return false;
        }
    }

    // (n, nrhs) at and one past Cta's resident-RHS capacity, n * nrhs exactly cap and cap + 1.
    // The widest nrhs dividing the product keeps n, and so the host check, small.
    std::optional<std::pair<int, int>> cta_shape(std::size_t elems, std::size_t max_n = 8192) const {
        for (int r = int(sycl_getrs::kGetrsFusedMaxRhs); r >= 1; --r)
            if (elems % r == 0 && elems / r <= max_n) return std::make_pair(int(elems / r), r);
        return std::nullopt;
    }

    static std::string name(const C& c, Uplo uplo, int n, int nrhs) {
        return select::to_string(c) + (uplo == Uplo::Lower ? " L" : " U") + " n=" + std::to_string(n) +
               " nrhs=" + std::to_string(nrhs);
    }
};

TYPED_TEST_SUITE(PosvCandidates, Types);

// §8.1: each candidate on shapes straddling both its order and its nrhs ceiling. Accepted
// shapes run and are correct; refused ones throw from the pin, for buffer_size and posv alike.
TYPED_TEST(PosvCandidates, PinnedCandidatesStraddleTheirLimits) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    const int tiny = this->tiny_n();
    ASSERT_GT(tiny, 0) << "the oracle has no Tiny ceiling on this device";
    std::vector<int> ns{1, 2, 16, 17, 32, 33, tiny, tiny + 1};
    std::sort(ns.begin(), ns.end());
    ns.erase(std::unique(ns.begin(), ns.end()), ns.end());
    int ran = 0, refused = 0;
    for (const C& c : ps::candidates<T>()) {
        for (Uplo uplo : {Uplo::Lower, Uplo::Upper}) {
            for (int n : ns) {
                for (int nrhs : {1, 4, 5, 8, 9}) {
                    auto p = make_sys<T>(n, nrhs, 3, uplo, 1000u + 31u * n + nrhs);
                    const std::string what = this->name(c, uplo, n, nrhs);
                    if (!this->expect_runs(c, n, nrhs)) {
                        EXPECT_FALSE(this->pin_accepted(c, p.A(), p.B(), uplo)) << what << " was accepted";
                        const Pin pin("posv", c);
                        UnifiedVector<std::byte> ws(1 << 16);
                        UnifiedVector<int32_t> info(p.batch, 0);
                        EXPECT_THROW(((void)posv<B, T>(*this->ctx, p.A(), p.B(), uplo, ws.to_span(), info.to_span())),
                                     std::invalid_argument) << what;
                        ++refused;
                        continue;
                    }
                    ASSERT_TRUE(this->pin_accepted(c, p.A(), p.B(), uplo)) << what << " was refused";
                    if (this->child_has_no_route(c, p)) {
                        EXPECT_THROW((void)this->run_pinned(c, p), dispatch::NoRouteError) << what;
                        continue;
                    }
                    expect_solved(p, this->run_pinned(c, p), what, !this->keeps_other(c, p));
                    ++ran;
                }
            }
        }
    }
    EXPECT_GT(ran, 0);
    EXPECT_GT(refused, 0);
}

// The plan's capacity launch: Cta at exactly n * nrhs == getrs_fused_max_rhs_elems runs and is
// correct; at cap + 1 the pin throws. A launch, not a re-derivation of the arithmetic.
TYPED_TEST(PosvCandidates, CtaLaunchesAtItsCapacityAndRefusesOnePast) {
    using T = typename TestFixture::T;
    const std::size_t cap = this->cta_elems();
    ASSERT_GT(cap, 0u);
    const auto at = this->cta_shape(cap, 4096);
    const auto past = this->cta_shape(cap + 1);
    ASSERT_TRUE(at.has_value()) << "no (n, nrhs <= 8) with n * nrhs == " << cap;
    const C cta{ps::Cta{}};
    {
        auto p = make_sys<T>(at->first, at->second, 2, Uplo::Lower, 71u);
        const std::string what = this->name(cta, Uplo::Lower, at->first, at->second) + " (cap " +
                                 std::to_string(cap) + ")";
        ASSERT_TRUE(this->pin_accepted(cta, p.A(), p.B(), Uplo::Lower)) << what;
        ASSERT_TRUE(this->direct_launches(cta, p.A(), p.B(), Uplo::Lower)) << what;
        auto q = make_sys<T>(at->first, at->second, 2, Uplo::Lower, 71u);
        expect_solved(q, this->run_pinned(cta, q), what);
    }
    std::vector<std::pair<int, int>> over{{at->first + 1, at->second}};
    if (past) over.push_back(*past);
    for (auto [n, nrhs] : over) {
        auto p = make_sys<T>(n, nrhs, 1, Uplo::Lower, 72u);
        const std::string what = this->name(cta, Uplo::Lower, n, nrhs) + " (cap " + std::to_string(cap) + ")";
        EXPECT_FALSE(this->pin_accepted(cta, p.A(), p.B(), Uplo::Lower)) << what << " is past the capacity";
        EXPECT_FALSE(this->direct_launches(cta, p.A(), p.B(), Uplo::Lower)) << what << " launched";
    }
}

// The pinned facade runs exactly that family's kernels: bit-identical to the direct calls on
// the same input. Complex data, both triangles: a wrong adjoint in Blocked's solves shows here.
TYPED_TEST(PosvCandidates, PinnedRunIsTheDirectKernelBitForBit) {
    using T = typename TestFixture::T;
    const int tiny = this->tiny_n();
    for (const C& c : ps::candidates<T>()) {
        for (Uplo uplo : {Uplo::Lower, Uplo::Upper}) {
            for (auto [n, nrhs] : {std::pair{std::min(tiny, 13), 3}, std::pair{tiny, 4}, std::pair{40, 7}}) {
                if (!this->expect_runs(c, n, nrhs)) continue;
                auto pinned = make_sys<T>(n, nrhs, 3, uplo, 4242u + n);
                auto direct = make_sys<T>(n, nrhs, 3, uplo, 4242u + n);
                const std::string what = this->name(c, uplo, n, nrhs);
                if (this->child_has_no_route(c, pinned)) continue;
                const auto info = this->run_pinned(c, pinned);
                UnifiedVector<int32_t> dinfo(3, int32_t(-7));
                ASSERT_NO_THROW(this->run_direct(c, direct.A(), direct.B(), uplo, dinfo.to_span())) << what;
                for (int it = 0; it < 3; ++it) ASSERT_EQ(info[it], dinfo[it]) << what;
                expect_solved(pinned, info, what, !this->keeps_other(c, pinned));
                for (size_t e = 0; e < pinned.b.size(); ++e)
                    ASSERT_TRUE(same_bits(pinned.b[e], direct.b[e]))
                        << what << ": the pinned facade did not run this family's kernels; X element " << e;
                for (size_t e = 0; e < pinned.a.size(); ++e)
                    ASSERT_TRUE(same_bits(pinned.a[e], direct.a[e])) << what << ": factor element " << e;
            }
        }
    }
}

// SLM tiers at a saturating batch: 1024 identical items come back bit-identical to item 0.
// Tiny stages through SLM for float/cfloat at N >= 16; Cta stages the RHS and blocks of L.
TYPED_TEST(PosvCandidates, SaturatingBatchIsBitIdentical) {
    using T = typename TestFixture::T;
    constexpr int kBatch = 1024;
    const int tiny = this->tiny_n();
    struct Case { C c; int n, nrhs; };
    const Case cases[] = {{ps::Tiny{}, std::min(tiny, 17), 4}, {ps::Tiny{}, tiny, 3},
                          {ps::Cta{}, 17, 8}, {ps::Cta{}, 96, 5}};
    for (const auto& k : cases) {
        for (Uplo uplo : {Uplo::Lower, Uplo::Upper}) {
            auto p = make_sys<T>(k.n, k.nrhs, kBatch, uplo, 777u, /*identical=*/true);
            const std::string what = this->name(k.c, uplo, k.n, k.nrhs) + " batch=1024";
            if (this->child_has_no_route(k.c, p)) continue;
            expect_solved(p, this->run_pinned(k.c, p), what, !this->keeps_other(k.c, p));
            for (int it = 1; it < kBatch; ++it) {
                for (int col = 0; col < k.nrhs; ++col)
                    for (int i = 0; i < k.n; ++i) {
                        const size_t o = size_t(col) * p.ldb + i;
                        ASSERT_TRUE(same_bits(p.b[size_t(it) * p.strb + o], p.b[o]))
                            << what << ": X of item " << it << " differs from item 0 at (" << i << "," << col << ")";
                    }
                for (int j = 0; j < k.n; ++j)
                    for (int i = 0; i < k.n; ++i)
                        if (p.in_tri(i, j)) {
                            const size_t o = size_t(j) * p.lda + i;
                            ASSERT_TRUE(same_bits(p.a[size_t(it) * p.stra + o], p.a[o]))
                                << what << ": the factor of item " << it << " differs from item 0";
                        }
            }
        }
    }
}

// §8.2 (R3): on every straddling shape a pin is accepted exactly when the family's own driver
// launches (posv_tiny_dispatch, potrs_fused_dispatch). Blocked has no driver: can_run is
// true on every homogeneous batch, and the public call runs. Absorbs posv_tests' old P6.
TYPED_TEST(PosvCandidates, CanRunEqualsLaunch) {
    using T = typename TestFixture::T;
    const int tiny = this->tiny_n();
    std::vector<std::pair<int, int>> shapes;
    for (int n : {1, 2, 16, 17, 32, 33, tiny, tiny + 1, 100})
        for (int nrhs : {1, 4, 5, 8, 9}) shapes.push_back({n, nrhs});
    for (std::size_t e : {this->cta_elems(), this->cta_elems() + 1})
        if (auto s = this->cta_shape(e)) shapes.push_back(*s);
    int disagreements = 0;
    for (const C& c : ps::candidates<T>()) {
        for (Uplo uplo : {Uplo::Lower, Uplo::Upper}) {
            for (auto [n, nrhs] : shapes) {
                auto p = make_sys<T>(n, nrhs, n > 256 ? 1 : 2, uplo, 61u);
                const std::string what = this->name(c, uplo, n, nrhs);
                const bool pin = this->pin_accepted(c, p.A(), p.B(), uplo);
                std::string why;
                const bool run = this->direct_launches(c, p.A(), p.B(), uplo, &why);
                EXPECT_EQ(pin, run) << what << ": can_run says " << pin << ", the driver "
                                    << (run ? "launches" : "refuses: " + why);
                EXPECT_EQ(pin, this->expect_runs(c, n, nrhs)) << what << ": can_run disagrees with the oracle";
                disagreements += pin != run;
            }
        }
    }
    EXPECT_EQ(disagreements, 0);
}

// A heterogeneous A or B (the other homogeneous) is refused before choose(), under Auto and
// under every pin, with the old router's internal_error. Blocked's vendor children would
// otherwise solve at the full storage order; the padding is nonzero and indefinite so such a
// solve would be wrong, not merely harmless. The natives' own drivers refuse it as well.
TYPED_TEST(PosvCandidates, HeterogeneousBatchIsRefusedUnderEveryPin) {
    using T = typename TestFixture::T;
    using R = RealOf<T>;
    static constexpr Backend B = TestFixture::B;
    const int n = 16, nrhs = 3, batch = 4;
    Matrix<T, MatrixFormat::Dense> A(n, n, batch), Bm(n, nrhs, batch);
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) A(i, j, b) = mk<T>(R(i == j ? 4 : -3), R(0));
    Bm.fill(mk<T>(R(1), R(0.5)));
    UnifiedVector<int> act(batch), cols(batch);
    for (int b = 0; b < batch; ++b) act[b] = n - b, cols[b] = nrhs - (b % 2);
    const auto hetA = A.view().with_active_dims(act.to_span(), act.to_span());
    const auto hetB = Bm.view().with_active_dims(act.to_span(), cols.to_span());
    ASSERT_TRUE(hetA.is_heterogeneous());
    ASSERT_TRUE(hetB.is_heterogeneous());
    struct Case { const char* what; MatrixView<T, MatrixFormat::Dense> a, b; };
    const Case cases[] = {{"A heterogeneous", hetA, Bm.view()}, {"B heterogeneous", A.view(), hetB}};
    std::vector<std::optional<C>> pins{std::nullopt};
    for (const C& c : ps::candidates<T>()) pins.push_back(c);
    for (const auto& k : cases)
        for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
            for (const auto& c : pins) {
                std::optional<Pin> pin;
                if (c) pin.emplace("posv", *c);
                const std::string what = std::string(k.what) + " " + (c ? select::to_string(*c) : "auto") +
                                         (uplo == Uplo::Lower ? " L" : " U");
                EXPECT_THROW(((void)posv_buffer_size<B, T>(*this->ctx, k.a, k.b, uplo)), batchlas::internal_error)
                    << what << ": posv_buffer_size";
                UnifiedVector<std::byte> ws(1 << 20);
                UnifiedVector<int32_t> info(batch);
                EXPECT_THROW(((void)posv<B, T>(*this->ctx, k.a, k.b, uplo, ws.to_span(), info.to_span())),
                             batchlas::internal_error)
                    << what << ": posv";
                if (c && !std::holds_alternative<ps::Blocked>(*c))
                    EXPECT_FALSE(this->direct_launches(*c, k.a, k.b, uplo)) << what << ": the driver launched";
            }
    this->ctx->wait();
}

// §8.3 (R5): exactly posv_buffer_size bytes inside a larger arena whose tail is a guard. The
// workspace starts as all-ones (a NaN pattern), so reading scratch before writing it shows in
// the residual. With an empty info span every family draws its info scratch from the span.
TYPED_TEST(PosvCandidates, ExactWorkspaceInAPoisonedArena) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    constexpr std::size_t kGuard = 4096;
    const int tiny = this->tiny_n();
    for (const C& c : ps::candidates<T>()) {
        for (Uplo uplo : {Uplo::Lower, Uplo::Upper}) {
            for (auto [n, nrhs] : {std::pair{1, 1}, std::pair{tiny, 4}, std::pair{33, 8}, std::pair{130, 9}}) {
                if (!this->expect_runs(c, n, nrhs)) continue;
                for (bool pass_info : {true, false}) {
                    auto p = make_sys<T>(n, nrhs, 5, uplo, 99u + n);
                    if (this->child_has_no_route(c, p)) continue;
                    const Pin pin("posv", c);
                    const PotrfPin native("potrf", "native");
                    const std::size_t bytes = posv_buffer_size<B, T>(*this->ctx, p.A(), p.B(), uplo);
                    UnifiedVector<std::byte> arena(bytes + kGuard);
                    std::memset(arena.data(), 0xFF, bytes);
                    std::memset(arena.data() + bytes, 0xA5, kGuard);
                    UnifiedVector<int32_t> info(p.batch, int32_t(-7));
                    const std::string what = this->name(c, uplo, n, nrhs) + " bytes=" + std::to_string(bytes) +
                                             (pass_info ? " info" : " no-info");
                    ASSERT_NO_THROW(((void)posv<B, T>(*this->ctx, p.A(), p.B(), uplo,
                                                      Span<std::byte>(arena.data(), bytes),
                                                      pass_info ? info.to_span() : Span<int32_t>{}),
                                     this->ctx->wait()))
                        << what;
                    for (std::size_t i = 0; i < kGuard; ++i)
                        ASSERT_EQ(static_cast<unsigned>(arena[bytes + i]), 0xA5u)
                            << what << ": wrote " << i << " bytes past the sized workspace";
                    expect_solved(p, pass_info ? std::vector<int32_t>(info.begin(), info.end())
                                               : std::vector<int32_t>(p.batch, 0),
                                  what, !this->keeps_other(c, p));
                }
            }
        }
    }
}

// R6: a spelling that names nothing compiled throws instead of meaning Auto.
TYPED_TEST(PosvCandidates, UnknownPinsThrow) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    Matrix<T, MatrixFormat::Dense> A(8, 8, 2), Bm(8, 2, 2);
    for (const char* word : {"bogus", "tiny:1", "cta:nb=8", "native:lpanel", "lpanel:panel=8", "native:tiny:4"}) {
        const Pin pin("posv", std::string_view(word));
        EXPECT_THROW(((void)posv_buffer_size<B, T>(*this->ctx, A.view(), Bm.view(), Uplo::Lower)),
                     std::invalid_argument) << word;
    }
    for (const C& c : ps::candidates<T>()) {
        const Pin pin("posv", c);
        EXPECT_NO_THROW(((void)posv_buffer_size<B, T>(*this->ctx, A.view(), Bm.view(), Uplo::Lower)))
            << select::to_string(c);
    }
}

// The named can_run-false cases, each with its message.
TYPED_TEST(PosvCandidates, CanRunFalsePinsThrow) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    const int tiny = this->tiny_n();
    const auto past = this->cta_shape(this->cta_elems() + 1);
    struct Case { C c; Uplo uplo; int n, nrhs; };
    std::vector<Case> cases{{ps::Tiny{}, Uplo::Lower, tiny + 1, 1}, {ps::Tiny{}, Uplo::Upper, tiny + 1, 1},
                            {ps::Tiny{}, Uplo::Lower, 8, 5},         {ps::Cta{}, Uplo::Upper, 8, 9}};
    if (past) cases.push_back({ps::Cta{}, Uplo::Lower, past->first, past->second});
    for (const auto& k : cases) {
        Matrix<T, MatrixFormat::Dense> A(k.n, k.n, 1), Bm(k.n, k.nrhs, 1);
        const Pin pin("posv", k.c);
        try {
            (void)posv_buffer_size<B, T>(*this->ctx, A.view(), Bm.view(), k.uplo);
            ADD_FAILURE() << this->name(k.c, k.uplo, k.n, k.nrhs) << " was accepted";
        } catch (const std::invalid_argument& e) {
            EXPECT_NE(std::string(e.what()).find("cannot run this shape"), std::string::npos)
                << this->name(k.c, k.uplo, k.n, k.nrhs) << ": " << e.what();
        }
    }
}

// §5.3: the legacy spellings select their choice, via ScopedPin and via the environment. posv
// has no vendor family, so bare `vendor` warns and runs Auto; bare `native` is Auto's pick
// (whatever this device's table ranks first at n=16 nrhs=2: tiny on sm_89, measured on sm_120).
TYPED_TEST(PosvCandidates, LegacyAliasesSelectTheirChoice) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_POSV_ROUTE", nullptr);
    std::string auto_pick;
    {
        auto p = make_sys<T>(16, 2, 4, Uplo::Lower, 5u);
        auto_pick = traced_choice([&] { (void)this->run_auto(p); });
    }
    const char* a = auto_pick.c_str();
    const std::pair<const char*, const char*> expect[] = {
        {"native:tiny", "tiny"}, {"native:cta", "cta"}, {"native:blocked", "blocked"}, {"NATIVE:CTA", "cta"},
        {"tiny", "tiny"},        {"Blocked", "blocked"}, {"vendor", a},                {"native", a},
        {"auto", a}};
    for (const auto& [word, spelling] : expect) {
        for (bool via_env : {false, true}) {
            auto p = make_sys<T>(16, 2, 4, Uplo::Lower, 5u);
            std::vector<int32_t> info;
            select::testing::reset_warnings();
            std::string err;
            const std::string got = traced_choice([&] {
                const ScopedEnvVar env("BATCHLAS_POSV_ROUTE", via_env ? word : nullptr);
                std::optional<Pin> pin;
                if (!via_env) pin.emplace("posv", std::string_view(word));
                info = this->run_auto(p);
            }, &err);
            const std::string what = std::string(word) + (via_env ? " via BATCHLAS_POSV_ROUTE" : " via ScopedPin");
            EXPECT_EQ(got, spelling) << what;
            const bool warns = std::string(word) == "vendor";
            EXPECT_EQ(err.find("posv pinned \"vendor\", but no vendor candidate") != std::string::npos, warns)
                << what << ": " << err;
            expect_solved(p, info, what);
        }
    }
}

// §5.3: a ScopedPin wins over BATCHLAS_POSV_ROUTE, and nested pins restore the outer one.
TYPED_TEST(PosvCandidates, ScopedPinBeatsTheEnvironment) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    const int n = this->tiny_n() + 1;
    const ScopedEnvVar env("BATCHLAS_POSV_ROUTE", "tiny");
    {
        Matrix<T, MatrixFormat::Dense> A(n, n, 1), Bm(n, 1, 1);
        EXPECT_THROW(((void)posv_buffer_size<B, T>(*this->ctx, A.view(), Bm.view(), Uplo::Lower)),
                     std::invalid_argument)
            << "the environment pin was not read, so nothing below proves it lost";
    }
    std::vector<int32_t> info;
    auto p = make_sys<T>(n, 2, 2, Uplo::Lower, 31u);
    EXPECT_EQ(traced_choice([&] {
                  const Pin pin("posv", C{ps::Cta{}});
                  info = this->run_auto(p);
              }),
              "cta");
    expect_solved(p, info, "cta over env tiny");
    std::string auto_pick;
    {
        const ScopedEnvVar clear("BATCHLAS_POSV_ROUTE", nullptr);
        auto probe = make_sys<T>(n, 2, 2, Uplo::Lower, 32u);
        auto_pick = traced_choice([&] { (void)this->run_auto(probe); });
    }
    ASSERT_NE(auto_pick, "tiny") << "n = tiny_n() + 1 must be past tiny's ceiling";
    auto q = make_sys<T>(n, 2, 2, Uplo::Lower, 32u);
    EXPECT_EQ(traced_choice([&] {
                  const Pin pin("posv", "auto");
                  info = this->run_auto(q);
              }),
              auto_pick);
    expect_solved(q, info, "auto over env tiny");

    const Pin outer("posv", C{ps::Cta{}});
    {
        const Pin inner("posv", C{ps::Blocked{}});
        auto r = make_sys<T>(n, 2, 2, Uplo::Lower, 33u);
        EXPECT_EQ(traced_choice([&] { (void)this->run_auto(r); }), "blocked");
    }
    auto s = make_sys<T>(n, 2, 2, Uplo::Lower, 34u);
    EXPECT_EQ(traced_choice([&] { (void)this->run_auto(s); }), "cta") << "the inner pin did not restore the outer";
}

// Auto against the sm_89 transcribed table on a device that reads it (its own, or borrowed by
// another sm device): hand-read rows straddling n (cfloat 24 vs 28 at nrhs 2) and nrhs (cfloat
// n=28 at 2 vs 4), where both answers can run, so only the key decides.
TYPED_TEST(PosvCandidates, AutoReadsTheSm89TranscribedTable) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    const ScopedEnvVar clear("BATCHLAS_POSV_ROUTE", nullptr);
    const std::string dtype(select::dtype_name<T>());
    const auto tables = select::tables_in_borrow_order("posv", dtype, select::device_of<B>(*this->ctx));
    if (tables.empty() || tables.front()->device != "sm_89") GTEST_SKIP() << "this device does not read sm_89";
    struct Row { const char* dtype; int n, nrhs; const char* expect; };
    const Row rows[] = {{"cfloat", 24, 2, "tiny"}, {"cfloat", 28, 2, "cta"}, {"cfloat", 28, 4, "tiny"},
                        {"float", 28, 2, "tiny"},  {"float", 32, 4, "tiny"}, {"float", 32, 5, "cta"},
                        {"cdouble", 16, 4, "tiny"}, {"double", 32, 9, "blocked"}};
    int checked = 0;
    for (const Row& r : rows) {
        if (dtype != r.dtype) continue;
        for (Uplo uplo : {Uplo::Lower, Uplo::Upper}) {
            auto p = make_sys<T>(r.n, r.nrhs, 512, uplo, 41u);
            std::vector<int32_t> info;
            const std::string what = std::string(uplo == Uplo::Lower ? "L" : "U") + " n=" + std::to_string(r.n) +
                                     " nrhs=" + std::to_string(r.nrhs);
            EXPECT_EQ(traced_choice([&] { info = this->run_auto(p); }), r.expect) << what;
            if (!this->child_has_no_route(C{ps::Blocked{}}, p)) expect_solved(p, info, what, !this->potrf_native(p));
            ++checked;
        }
    }
    EXPECT_GT(checked, 0) << dtype;
}

// key_of's every field reaches choose(): a synthetic table for this device whose winner changes
// with each of uplo, n, nrhs and batch alone. Fixing any field in key_of turns exactly its row
// red; the transcribed tables cannot show uplo or batch, which they do not depend on.
TYPED_TEST(PosvCandidates, AutoReadsEveryKeyField) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    const ScopedEnvVar clear("BATCHLAS_POSV_ROUTE", nullptr);
    const std::string dtype(select::dtype_name<T>());
    const std::string dev = select::device_of<B>(*this->ctx).key;
    std::vector<std::pair<std::string, std::string>> files;
    for (const auto& t : select::embedded_tables())
        if (t.name.rfind("posv.", 0) != 0) files.emplace_back(std::string(t.name), std::string(t.text));
    files.emplace_back("posv." + dtype + "." + dev + ".txt",
                       "# op=posv dtype=" + dtype + " device=" + dev + " kernels=unknown\n"
                       "# keys: uplo:exact n:log:3 nrhs:log batch:log\n"
                       "uplo=L n=8 nrhs=1 batch=128 | tiny 1 | cta 2 | blocked 3\n"
                       "uplo=L n=8 nrhs=1 batch=32768 | cta 1 | tiny 2 | blocked 3\n"
                       "uplo=L n=8 nrhs=4 batch=128 | blocked 1 | tiny 2 | cta 3\n"
                       "uplo=L n=12 nrhs=1 batch=128 | cta 1 | blocked 2 | tiny 3\n"
                       "uplo=U n=8 nrhs=1 batch=128 | blocked 1 | cta 2 | tiny 3\n");
    const TableGuard restore;
    select::testing::set_builtin_tables(std::move(files));
    struct Probe { Uplo uplo; int n, nrhs, batch; const char* expect; const char* field; };
    const Probe probes[] = {{Uplo::Lower, 8, 1, 128, "tiny", "base"},
                            {Uplo::Lower, 8, 1, 32768, "cta", "batch"},
                            {Uplo::Lower, 8, 4, 128, "blocked", "nrhs"},
                            {Uplo::Lower, 12, 1, 128, "cta", "n"},
                            {Uplo::Upper, 8, 1, 128, "blocked", "uplo"}};
    for (const auto& k : probes) {
        auto p = make_sys<T>(k.n, k.nrhs, k.batch, k.uplo, 43u);
        std::vector<int32_t> info;
        EXPECT_EQ(traced_choice([&] { info = this->run_auto(p); }), k.expect) << "the " << k.field << " row";
        expect_solved(p, info, std::string("the ") + k.field + " row");
    }
}

// The coverage row's native flags come from the candidate list and can_run (§5.6): every
// posv shape has a runnable native candidate (Blocked), so existed = supported = 1. The row
// also carries the posv key: m = order, n = nrhs, and uplo.
// threadsafe: the child re-executes the binary, so CUDA is initialised fresh, never forked.
TYPED_TEST(PosvCandidates, CoverageRowCarriesNativeFlags) {
    using T = typename TestFixture::T;
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    const std::string dir = ::testing::TempDir() + "posv_cov." + std::string(select::dtype_name<T>());
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const std::string out = dir + "/cov";
    std::string want_lo;  // this device's table decides the Lower n=8 nrhs=2 tier
    {
        const ScopedEnvVar clear("BATCHLAS_POSV_ROUTE", nullptr);
        auto lo = make_sys<T>(8, 2, 2, Uplo::Lower, 51u);
        want_lo = traced_choice([&] { (void)this->run_auto(lo); });
    }
    auto child = [&] {
        const ScopedEnvVar cov("BATCHLAS_COVERAGE_OUT", out.c_str());
        const ScopedEnvVar clear("BATCHLAS_POSV_ROUTE", nullptr);
        dispatch::coverage::g_dynamic_enabled = true;
        auto lo = make_sys<T>(8, 2, 2, Uplo::Lower, 51u);
        (void)this->run_auto(lo);
        auto hi = make_sys<T>(8, 9, 2, Uplo::Upper, 52u);
        (void)this->run_auto(hi);
        std::exit(0);
    };
    EXPECT_EXIT(child(), ::testing::ExitedWithCode(0), "");
    std::map<std::string, std::vector<std::string>> rows;  // "m n uplo" -> fields
    for (const auto& ent : std::filesystem::directory_iterator(dir)) {
        std::ifstream in(ent.path());
        for (std::string line; std::getline(in, line);) {
            if (line.rfind("reached,posv,", 0) != 0) continue;
            std::vector<std::string> f;
            std::stringstream ss(line);
            for (std::string tok; std::getline(ss, tok, ',');) f.push_back(tok);
            ASSERT_GE(f.size(), 16u) << line;
            rows[f[5] + " " + f[6] + " " + f[15]] = f;
        }
    }
    std::filesystem::remove_all(dir);
    const std::string lo = "8 2 " + std::to_string(static_cast<int>(Uplo::Lower));
    const std::string hi = "8 9 " + std::to_string(static_cast<int>(Uplo::Upper));
    ASSERT_EQ(rows.size(), 2u);
    ASSERT_TRUE(rows.count(lo)) << "no Lower n=8 nrhs=2 row";
    ASSERT_TRUE(rows.count(hi)) << "no Upper n=8 nrhs=9 row";
    for (const auto& key : {lo, hi}) {
        EXPECT_EQ(rows[key][12], "1") << key;
        EXPECT_EQ(rows[key][13], "1") << key;
    }
    EXPECT_EQ(rows[lo][10], want_lo);
    EXPECT_EQ(rows[hi][10], "blocked");  // nrhs 9 is past tiny's and cta's ceilings
}

// An empty problem has no kernel: an explicit throw before choose(), never a silent no-op.
TYPED_TEST(PosvCandidates, EmptyShapesThrowBeforeChoose) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    for (auto [n, nrhs, batch] : {std::tuple{0, 1, 1}, std::tuple{4, 0, 1}, std::tuple{4, 1, 0}}) {
        Matrix<T, MatrixFormat::Dense> A(n, n, batch), Bm(n, nrhs, batch);
        const Pin pin("posv", C{ps::Blocked{}});
        EXPECT_THROW(((void)posv_buffer_size<B, T>(*this->ctx, A.view(), Bm.view(), Uplo::Lower)),
                     batchlas::internal_error)
            << n << " " << nrhs << " " << batch;
    }
}

// The transcribed sm_89 rows, read with Table::nearest directly so every device checks them:
// the old router's window (float/double/cdouble to their tier ceiling, cfloat above 24 only at
// nrhs > 2) as the first entry, then cta, then blocked.
TEST(PosvTranscribedTable, Sm89RowsHoldTheOldWindow) {
    struct Row { const char* dtype; const char* uplo; int n, nrhs, batch; const char* first; std::size_t len; };
    const Row rows[] = {
        {"cfloat", "L", 24, 2, 8192, "tiny", 3},  {"cfloat", "L", 28, 2, 8192, "cta", 2},
        {"cfloat", "L", 28, 4, 8192, "tiny", 3},  {"cfloat", "U", 28, 2, 128, "cta", 2},
        {"cdouble", "L", 16, 1, 8192, "tiny", 3}, {"cdouble", "U", 20, 1, 8192, "cta", 2},
        {"float", "L", 32, 1, 32768, "tiny", 3},  {"float", "L", 36, 1, 32768, "cta", 2},
        {"double", "U", 32, 64, 512, "tiny", 3},  {"double", "L", 1024, 64, 512, "cta", 2},
    };
    for (const Row& r : rows) {
        const auto tables = select::tables_in_borrow_order("posv", r.dtype, select::device_from_key("sm_89"));
        ASSERT_FALSE(tables.empty()) << r.dtype;
        const select::Table& t = *tables.front();
        ASSERT_EQ(t.device, "sm_89") << r.dtype;
        EXPECT_EQ(t.source.rfind("transcribed:", 0), 0u) << t.file;
        const select::Key key{{"uplo", r.uplo}, {"n", r.n}, {"nrhs", r.nrhs}, {"batch", r.batch}};
        const select::TableRow* row = t.nearest(key);
        ASSERT_NE(row, nullptr) << t.file;
        const std::string what = t.file + ":" + std::to_string(row->line);
        ASSERT_EQ(row->ranked.size(), r.len) << what;
        EXPECT_EQ(row->ranked.front().spelling, r.first) << what;
        EXPECT_EQ(row->ranked.back().spelling, "blocked") << what;
        EXPECT_FALSE(row->timed) << what;
    }
}

}  // namespace
