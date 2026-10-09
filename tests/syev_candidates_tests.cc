// Every syev candidate, pinned: docs/design/flat-kernel-selection.md §8 and
// docs/design/flat-kernel-selection.md#phase-5-syev. Inputs have a known spectrum (A = Q diag(lambda) Q^H with a
// host Householder Q), so no reference goes through the code under test. Which kernel ran is
// read back from the select trace or a bit-for-bit comparison with the direct driver.
#include <gtest/gtest.h>

#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/functions/syev.hh>
#include "../src/select/coverage.hh"
#include <batchlas/no_route.hh>
#include "../src/select/vendor.hh"
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>

#include "test_utils.hh"
#include "shipped_table_pick.hh"

#include "../src/ops/syev/choice.hh"

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
#include "../src/ops/syev/vendor.hh"

using namespace batchlas;

namespace {

namespace sy = batchlas::ops::syev;
using C = sy::SyevChoice;
using Pin = select::ScopedPin<C>;
template <typename T>
using MVof = MatrixView<T, MatrixFormat::Dense>;
template <typename T>
using RealOf = typename batchlas::base_type<T>::type;
template <typename T>
constexpr bool kCx = test_utils::is_complex<T>::value;
using cd = std::complex<double>;

template <typename T>
T from(cd v) {
    if constexpr (kCx<T>) return T(RealOf<T>(v.real()), RealOf<T>(v.imag()));
    else return T(v.real());
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

bool is_small(const C& c) {
    return std::holds_alternative<sy::Cta>(c) || std::holds_alternative<sy::CtaFused>(c) ||
           std::holds_alternative<sy::Jacobi>(c);
}

struct Spec {
    int n = 8, batch = 3;
    JobType jobz = JobType::EigenVectors;
    Uplo uplo = Uplo::Lower;
    int period = 0;  // > 0: item it repeats item it % period
    unsigned seed = 1;
    bool poison_other = false;  // the other triangle holds poison instead of the mirror
};

// A in a buffer of large finite poison at ld = n + 3 and a stride that is not ld * n. The other
// triangle holds the mirror, or poison under Spec::poison_other (OtherTriangleIsNeverRead).
template <typename T>
struct Eig {
    Spec s;
    int ld = 0, stride = 0;
    UnifiedVector<T> mem;
    std::vector<T> mem0;
    UnifiedVector<RealOf<T>> w;
    UnifiedVector<T*> ptrs;
    std::vector<std::vector<double>> lambda;  // sorted, per representative item

    MVof<T> A() { return {mem.data(), s.n, s.n, ld, stride, s.batch, ptrs.data()}; }
    std::size_t at(int it, int i, int j) const { return std::size_t(it) * stride + std::size_t(j) * ld + i; }
    bool stored(int i, int j) const { return s.uplo == Uplo::Lower ? i >= j : i <= j; }
};

template <typename T>
T poison() {
    return from<T>(cd(-9.0e3, 7.0e3));
}

// Graded spectrum +-10^[-2, 2] and a dense complex Householder Q, built in double on the host.
template <typename T>
Eig<T> make_eig(const Spec& s) {
    Eig<T> p;
    p.s = s;
    p.ld = s.n + 3;
    p.stride = p.ld * s.n + 5;
    p.mem = UnifiedVector<T>(std::size_t(p.stride) * s.batch + 7, poison<T>());
    p.w = UnifiedVector<RealOf<T>>(std::size_t(s.n) * s.batch, RealOf<T>(0));
    p.ptrs = UnifiedVector<T*>(s.batch, nullptr);
    const int reps = s.period > 0 ? std::min(s.period, s.batch) : s.batch;
    std::mt19937 gen(s.seed);
    std::uniform_real_distribution<double> u(-1, 1);
    const int n = s.n;
    for (int it = 0; it < reps; ++it) {
        std::vector<double> lam(n);
        for (int i = 0; i < n; ++i)
            lam[i] = ((i + it) % 2 ? -1.0 : 1.0) * std::pow(10.0, n == 1 ? 0.0 : -2.0 + 4.0 * i / (n - 1)) * (1 + 0.01 * it);
        std::vector<cd> v(n);
        double vv = 0;
        for (int i = 0; i < n; ++i) {
            v[i] = kCx<T> ? cd(u(gen), u(gen)) : cd(u(gen), 0.0);
            vv += std::norm(v[i]);
        }
        auto q = [&](int i, int j) { return (i == j ? 1.0 : 0.0) - 2.0 * v[i] * std::conj(v[j]) / vv; };
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                if (!p.stored(i, j) && s.poison_other) continue;
                cd a = 0;
                for (int k = 0; k < n; ++k) a += q(i, k) * lam[k] * std::conj(q(j, k));
                if (i == j) a = cd(a.real(), 0.0);
                p.mem[p.at(it, i, j)] = from<T>(a);
            }
        std::sort(lam.begin(), lam.end());
        p.lambda.push_back(lam);
    }
    for (int it = reps; it < s.batch; ++it)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                if (p.stored(i, j) || !s.poison_other) p.mem[p.at(it, i, j)] = p.mem[p.at(it % reps, i, j)];
    p.mem0.assign(p.mem.begin(), p.mem.end());
    return p;
}

template <typename T>
double tol(int n) {
    return 200.0 * std::max(n, 4) * double(std::numeric_limits<RealOf<T>>::epsilon());
}

// Eigenvalues against the construction (relative to max |lambda| = 100), eigenvector residual
// and orthonormality from the original A, nothing written outside A's n x n footprint (LAPACK
// lets jobz = N destroy A), and a repeating batch bit-identical to its representatives.
template <typename T>
void expect_solved(const Eig<T>& p, const std::string& what) {
    const Spec& s = p.s;
    const int n = s.n;
    const int reps = s.period > 0 ? std::min(s.period, s.batch) : s.batch;
    std::vector<int> items;
    for (int it = 0; it < std::min(reps, 4); ++it) items.push_back(it);
    if (reps > 4) items.push_back(reps - 1);
    const double t = tol<T>(n);
    for (int it : items) {
        const auto& lam = p.lambda[it];
        for (int i = 0; i < n; ++i)
            ASSERT_NEAR(double(p.w[std::size_t(it) * n + i]), lam[i], t * 100.0)
                << what << " item " << it << " eigenvalue " << i;
        if (s.jobz != JobType::EigenVectors) continue;
        auto a0 = [&](int i, int j) {
            const bool st = p.stored(i, j);
            const cd v = up(p.mem0[st ? p.at(it, i, j) : p.at(it, j, i)]);
            return st ? v : std::conj(v);
        };
        double worst_r = 0, worst_o = 0;
        for (int k = 0; k < n; ++k) {
            const double lk = double(p.w[std::size_t(it) * n + k]);
            for (int i = 0; i < n; ++i) {
                cd r = -lk * up(p.mem[p.at(it, i, k)]);
                for (int j = 0; j < n; ++j) r += a0(i, j) * up(p.mem[p.at(it, j, k)]);
                worst_r = std::max(worst_r, std::abs(r) / 100.0);
            }
            for (int l = 0; l < n; ++l) {
                cd d = 0;
                for (int i = 0; i < n; ++i) d += std::conj(up(p.mem[p.at(it, i, k)])) * up(p.mem[p.at(it, i, l)]);
                worst_o = std::max(worst_o, std::abs(d - (k == l ? 1.0 : 0.0)));
            }
        }
        ASSERT_LE(worst_r, t) << what << " item " << it << " residual";
        ASSERT_LE(worst_o, t) << what << " item " << it << " orthonormality";
    }
    for (int it = 0; it < s.batch; ++it)
        for (std::size_t e = std::size_t(it) * p.stride; e < std::size_t(it + 1) * p.stride && e < p.mem.size(); ++e) {
            const std::size_t off = e - std::size_t(it) * p.stride;
            const bool inside = off / p.ld < std::size_t(n) && off % p.ld < std::size_t(n);
            if (!inside) ASSERT_TRUE(same_bits(p.mem[e], p.mem0[e])) << what << ": wrote element " << e;
        }
    if (s.period > 0)
        for (int it = s.period; it < s.batch; ++it)
            for (int i = 0; i < n; ++i) {
                ASSERT_TRUE(same_bits(p.w[std::size_t(it) * n + i], p.w[std::size_t(it % s.period) * n + i]))
                    << what << ": W of item " << it << " differs from its representative";
                if (s.jobz == JobType::EigenVectors)
                    for (int k = 0; k < n; ++k)
                        ASSERT_TRUE(same_bits(p.mem[p.at(it, i, k)], p.mem[p.at(it % s.period, i, k)]))
                            << what << ": V of item " << it << " differs from its representative";
            }
}

std::string label(const Spec& s) {
    return std::string(s.jobz == JobType::EigenVectors ? "V" : "N") + (s.uplo == Uplo::Lower ? "L" : "U") +
           " n=" + std::to_string(s.n) + " batch=" + std::to_string(s.batch);
}
std::string name(const C& c, const Spec& s) { return select::to_string(c) + " " + label(s); }

// The outermost syev trace line for whatever `run` calls, and the choice it names.
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
        if (line.rfind("syev ", 0) == 0 && line.find(" -> ") != std::string::npos) return line;
    return "<no syev trace line in: " + err + ">";
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

std::vector<std::pair<std::string, std::string>> tables_without_syev() {
    std::vector<std::pair<std::string, std::string>> files;
    for (const auto& t : select::embedded_tables())
        if (t.name.rfind("syev.", 0) != 0) files.emplace_back(std::string(t.name), std::string(t.text));
    return files;
}

template <typename T, Backend B>
struct Cfg {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};
using Types = typename test_utils::backend_types<Cfg>::type;

template <typename Config>
class SyevCandidates : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    using R = RealOf<T>;
    static constexpr Backend B = Config::BackendVal;
    static constexpr bool kVendor = batchlas::select::solver_vendor_available<B>;

    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        if (this->HasFatalFailure() || ::testing::Test::IsSkipped()) return;
        if (!this->ctx) GTEST_SKIP() << "no queue";
        if (this->ctx->device().type != DeviceType::GPU) GTEST_SKIP() << "the native tiers are GPU kernels";
        if (!this->ctx->device().supports_sub_group_size(32)) GTEST_SKIP() << "no sub-group size 32";
    }

    // The limit oracle: the drivers' own checks (1 <= n <= 32 in syev_cta/_fused/jacobi_cta.cc).
    static bool expect_runs(const C& c, int n) {
        if (n < 1) return false;
        if (is_small(c)) return n <= 32;
        if (std::holds_alternative<sy::Vendor>(c)) return kVendor;
        return true;
    }
    static bool vendor_word_falls_back(const C& c) { return std::holds_alternative<sy::Vendor>(c) && !kVendor; }

    std::size_t size(Eig<T>& p) {
        return syev_buffer_size<B, T>(*this->ctx, p.A(), p.w.to_span(), p.s.jobz, p.s.uplo);
    }
    void run(Eig<T>& p) {
        UnifiedVector<std::byte> ws(std::max<std::size_t>(size(p), 1));
        (void)syev<B, T>(*this->ctx, p.A(), p.w.to_span(), p.s.jobz, p.s.uplo, ws.to_span(), Span<int32_t>{});
        this->ctx->wait();
    }
    void run_pinned(const C& c, Eig<T>& p) {
        const Pin pin("syev", c);
        run(p);
    }
    // Acceptance is the sizing call under the pin: it runs the same choose() as the solve.
    bool pin_accepted(const C& c, Eig<T>& p) {
        const Pin pin("syev", c);
        try {
            (void)size(p);
            return true;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find("cannot run this shape") == std::string::npos) throw;
            return false;
        }
    }

    // The family's own driver, with syev's parameters, on an in-order queue (blocked needs one).
    bool direct(const C& c, Eig<T>& p, std::string* why = nullptr) {
        std::optional<Queue> io;
        Queue* q = this->ctx.get();
        if (!q->in_order()) q = &io.emplace(*this->ctx, true);
        const auto A = p.A();
        const auto w = p.w.to_span();
        const JobType j = p.s.jobz;
        const Uplo u = p.s.uplo;
        SteqrParams<T> sp{};
        sp.max_sweeps = 400;
        sp.cta_shift_strategy = SteqrShiftStrategy::Wilkinson;
        try {
            std::size_t need = 0;
            if (std::holds_alternative<sy::Cta>(c)) need = syev_cta_buffer_size<B, T>(*q, A, j, sp);
            else if (std::holds_alternative<sy::CtaFused>(c)) need = syev_cta_fused_buffer_size<B, T>(*q, A, j, sp);
            else if (std::holds_alternative<sy::Jacobi>(c)) need = syev_jacobi_cta_buffer_size<B, T>(*q, A, j);
            else if (std::holds_alternative<sy::Blocked>(c)) need = syev_blocked_buffer_size<B, T>(*q, A, j, u);
            else if (std::holds_alternative<sy::TwoStage>(c)) need = syev_two_stage_buffer_size<B, T>(*q, A, j, u);
            else need = blas::dispatch::detail::syev_vendor_buffer_size_or_throw<B, T>(*q, A, w, j, u);
            UnifiedVector<std::byte> ws(std::max<std::size_t>(need, 1));
            const auto wsp = ws.to_span();
            if (std::holds_alternative<sy::Cta>(c)) (void)syev_cta<B, T>(*q, A, w, j, u, wsp, sp, 0);
            else if (std::holds_alternative<sy::CtaFused>(c)) (void)syev_cta_fused<B, T>(*q, A, w, j, u, wsp, sp, 0);
            else if (std::holds_alternative<sy::Jacobi>(c)) (void)syev_jacobi_cta<B, T>(*q, A, w, j, u, wsp);
            else if (std::holds_alternative<sy::Blocked>(c)) (void)syev_blocked<B, T>(*q, A, w, j, u, wsp);
            else if (std::holds_alternative<sy::TwoStage>(c)) (void)syev_two_stage<B, T>(*q, A, w, j, u, wsp);
            else (void)blas::dispatch::detail::syev_vendor_or_throw<B, T>(*q, A, w, j, u, wsp);
            q->wait();
            return true;
        } catch (const std::exception& e) {
            if (why) *why = e.what();
            return false;
        }
    }
};

TYPED_TEST_SUITE(SyevCandidates, Types);

// §8.1: every candidate across the small-kernel ceiling (32/33), the P buckets (4/5, 8/9,
// 16/17) and n = 1, both jobz and both triangles. Accepted shapes are correct; refused throw.
TYPED_TEST(SyevCandidates, PinnedCandidatesStraddleTheirLimits) {
    using T = typename TestFixture::T;
    for (const C& c : sy::candidates<T>())
        for (int n : {1, 4, 5, 9, 17, 32, 33, 40})
            for (JobType j : {JobType::NoEigenVectors, JobType::EigenVectors})
                for (Uplo u : {Uplo::Lower, Uplo::Upper}) {
                    if (TestFixture::vendor_word_falls_back(c)) continue;
                    Spec s{n, 3, j, u};
                    s.seed = 7u + n;
                    auto p = make_eig<T>(s);
                    const bool want = this->expect_runs(c, n);
                    if (!want) {
                        EXPECT_FALSE(this->pin_accepted(c, p)) << name(c, s);
                        continue;
                    }
                    this->run_pinned(c, p);
                    expect_solved(p, name(c, s));
                }
}

// uplo's other triangle holds a large finite poison: every candidate must ignore it. Three
// drivers read it (docs/design/known-defects.md#defect-14-the-hermitian-drivers-read-the-unreferenced-triangle); they are listed and
// skipped here, the rest are held to it. The pins reach the drivers whatever Auto picks.
bool reads_other_triangle(const C& c, int n, Uplo u) {
    if (std::holds_alternative<sy::Cta>(c)) return u == Uplo::Upper;
    if (std::holds_alternative<sy::Blocked>(c)) return u == Uplo::Lower && n <= 32;
    if (std::holds_alternative<sy::TwoStage>(c)) return u == Uplo::Lower && n > 32;
    return false;
}

TYPED_TEST(SyevCandidates, OtherTriangleIsNeverRead) {
    using T = typename TestFixture::T;
    for (const C& c : sy::candidates<T>())
        for (int n : {5, 17, 32, 40})
            for (Uplo u : {Uplo::Lower, Uplo::Upper}) {
                if (!this->expect_runs(c, n) || TestFixture::vendor_word_falls_back(c)) continue;
                if (reads_other_triangle(c, n, u)) continue;
                Spec s{n, 3, JobType::EigenVectors, u};
                s.poison_other = true;
                s.seed = 17u + n;
                auto p = make_eig<T>(s);
                this->run_pinned(c, p);
                expect_solved(p, "poisoned other triangle, " + name(c, s));
            }
}

// The pinned public call is exactly the family's driver: same W (and V) bit for bit.
TYPED_TEST(SyevCandidates, PinnedRunIsTheDirectKernelBitForBit) {
    using T = typename TestFixture::T;
    for (const C& c : sy::candidates<T>())
        for (int n : {6, 24, 48}) {
            if (!this->expect_runs(c, n) || TestFixture::vendor_word_falls_back(c)) continue;
            const Spec s{n, 4, JobType::EigenVectors, Uplo::Upper, 0, 11u};
            auto a = make_eig<T>(s), b = make_eig<T>(s);
            this->run_pinned(c, a);
            std::string why;
            ASSERT_TRUE(this->direct(c, b, &why)) << name(c, s) << ": " << why;
            for (std::size_t i = 0; i < a.w.size(); ++i)
                ASSERT_TRUE(same_bits(a.w[i], b.w[i])) << name(c, s) << " W[" << i << "]";
            for (std::size_t e = 0; e < a.mem.size(); ++e)
                ASSERT_TRUE(same_bits(a.mem[e], b.mem[e])) << name(c, s) << " A[" << e << "]";
        }
}

// §8.1 saturating batch: 1024 copies of 4 matrices; every item bit-identical to its
// representative (the small solvers keep a matrix per sub-group partition in SLM).
TYPED_TEST(SyevCandidates, SaturatingBatchIsBitIdenticalToItsRepresentative) {
    using T = typename TestFixture::T;
    for (const C& c : sy::candidates<T>())
        for (int n : {8, 32}) {
            if (!this->expect_runs(c, n) || TestFixture::vendor_word_falls_back(c)) continue;
            if (std::holds_alternative<sy::TwoStage>(c) && n == 8) continue;
            Spec s{n, 1024, JobType::EigenVectors, Uplo::Lower, 4, 5u};
            auto p = make_eig<T>(s);
            this->run_pinned(c, p);
            expect_solved(p, name(c, s));
        }
}

// §8.2 R3: can_run (the pin accepted by the sizing call) equals the driver accepting the shape,
// including the degenerate n = 0 and batch = 0 (small and large n) that the drivers check.
TYPED_TEST(SyevCandidates, CanRunEqualsLaunch) {
    using T = typename TestFixture::T;
    const std::pair<int, int> shapes[] = {{1, 2}, {31, 2}, {32, 2}, {33, 2}, {64, 2}, {0, 2}, {8, 0}, {40, 0}};
    for (const C& c : sy::candidates<T>())
        for (const auto& [n, batch] : shapes)
            for (JobType j : {JobType::NoEigenVectors, JobType::EigenVectors})
                for (Uplo u : {Uplo::Lower, Uplo::Upper}) {
                    if (TestFixture::vendor_word_falls_back(c)) continue;
                    const Spec s{n, batch, j, u};
                    auto p = make_eig<T>(s), q = make_eig<T>(s);
                    std::string why;
                    EXPECT_EQ(this->pin_accepted(c, p), this->direct(c, q, &why)) << name(c, s) << ": " << why;
                }
}

// §8.3 R5: under each pin, a workspace of exactly syev_buffer_size bytes inside a poisoned
// arena suffices and nothing past its end is written.
TYPED_TEST(SyevCandidates, ExactWorkspaceInAPoisonedArena) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    constexpr std::size_t kGuard = 4096;
    for (const C& c : sy::candidates<T>())
        for (int n : {8, 32, 40}) {
            if (!this->expect_runs(c, n) || TestFixture::vendor_word_falls_back(c)) continue;
            const Spec s{n, 5, JobType::EigenVectors, Uplo::Lower, 0, 13u};
            auto p = make_eig<T>(s);
            const Pin pin("syev", c);
            const std::size_t need = this->size(p);
            UnifiedVector<std::byte> arena(need + kGuard, std::byte{0xA5});
            (void)syev<B, T>(*this->ctx, p.A(), p.w.to_span(), s.jobz, s.uplo, Span<std::byte>(arena.data(), need),
                             Span<int32_t>{});
            this->ctx->wait();
            for (std::size_t i = need; i < arena.size(); ++i)
                ASSERT_EQ(arena[i], std::byte{0xA5}) << name(c, s) << " wrote past the workspace at +" << i - need;
            expect_solved(p, name(c, s));
            if (need > 0) {
                auto r = make_eig<T>(s);
                UnifiedVector<std::byte> small(need - 1);
                EXPECT_THROW(((void)syev<B, T>(*this->ctx, r.A(), r.w.to_span(), s.jobz, s.uplo, small.to_span(),
                                               Span<int32_t>{})),
                             std::exception)
                    << name(c, s) << " accepted need - 1 bytes";
            }
        }
}

// R6: a spelling that names nothing compiled throws instead of meaning Auto.
TYPED_TEST(SyevCandidates, UnknownPinsThrow) {
    using T = typename TestFixture::T;
    for (const char* word : {"bogus", "cta:1", "jacobi_cta", "native:vendor", "native:cta_fused:8", "twostage",
                             "native:tiny",
                             // removed aliases (phase 5): each must stay an error
                             "native:cta", "native:blocked", "native:two_stage", "native:two-stage",
                             "two-stage", "native:jacobi", "batchlas_cta", "batchlas-cta", "batchlas_blocked",
                             "batchlas-blocked", "batchlas_two_stage", "batchlas-two-stage", "fused",
                             "native:cta_fused", "netlib", "vendor:auto"}) {
        auto p = make_eig<T>(Spec{});
        const Pin pin("syev", std::string_view(word));
        EXPECT_THROW(this->run(p), std::invalid_argument) << word;
    }
}

// The named can_run-false pins and their message; a non-square A throws under every pin.
TYPED_TEST(SyevCandidates, CanRunFalsePinsThrow) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    for (const C& c : {C{sy::Cta{}}, C{sy::CtaFused{}}, C{sy::Jacobi{}}}) {
        auto p = make_eig<T>(Spec{33, 2});
        const Pin pin("syev", c);
        try {
            this->run(p);
            ADD_FAILURE() << select::to_string(c) << " n=33 was accepted";
        } catch (const std::invalid_argument& e) {
            EXPECT_NE(std::string(e.what()).find("cannot run this shape"), std::string::npos) << e.what();
        }
    }
    UnifiedVector<T> a(64, T(1));
    UnifiedVector<RealOf<T>> w(8);
    const MVof<T> rect(a.data(), 8, 4, 8, 32, 2);
    for (const char* word : {"auto", "blocked", "vendor"}) {
        const Pin pin("syev", std::string_view(word));
        EXPECT_THROW(((void)syev_buffer_size<B, T>(*this->ctx, rect, w.to_span(), JobType::EigenVectors, Uplo::Lower)),
                     std::invalid_argument)
            << word;
    }
    // The Python binding's introspection asks the same can_run: no family takes a rectangle.
    namespace det = blas::dispatch::detail;
    EXPECT_FALSE(det::syev_supports_cta<T>(*this->ctx, rect));
    EXPECT_FALSE(det::syev_supports_blocked<T>(*this->ctx, rect, Uplo::Lower));
    EXPECT_FALSE(det::syev_supports_two_stage<T>(*this->ctx, rect, Uplo::Upper));
    const MVof<T> sq32(a.data(), 32, 32, 32, 0, 1), sq33(a.data(), 33, 33, 33, 0, 1);
    EXPECT_TRUE(det::syev_supports_cta<T>(*this->ctx, sq32));
    EXPECT_FALSE(det::syev_supports_cta<T>(*this->ctx, sq33));
    EXPECT_TRUE(det::syev_supports_blocked<T>(*this->ctx, sq33, Uplo::Upper));
    EXPECT_TRUE(det::syev_supports_two_stage<T>(*this->ctx, sq33, Uplo::Lower));
}

// §5.3: spellings (case-folded) and the class words, via ScopedPin and BATCHLAS_SYEV_ROUTE. The
// retired BATCHLAS_SYEV_PROVIDER is not read: setting it changes nothing.
TYPED_TEST(SyevCandidates, ClassWordsAndSpellings) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_SYEV_ROUTE", nullptr);

    const Spec s{12, 3};
    std::string auto_pick;
    {
        auto p = make_eig<T>(s);
        auto_pick = traced_choice([&] { this->run(p); });
    }
    const auto table_pick = [&](bool native_only) {
        const select::Key key{{"jobz", s.jobz == JobType::EigenVectors ? "V" : "N"}, {"n", s.n}, {"batch", s.batch}};
        return test_utils::shipped_table_pick<C>("syev", select::dtype_name<T>(),
                                                 select::device_of<TestFixture::B>(*this->ctx), key, [&] {
                                                     auto q = make_eig<T>(s);
                                                     this->run(q);
                                                 }, native_only);
    };
    EXPECT_EQ(auto_pick, table_pick(false)) << "Auto against the shipped table";
    const std::string vendor_pick = TestFixture::kVendor ? "vendor" : auto_pick;
    const std::string native_pick = table_pick(true);  // the row's best runnable non-vendor
    ASSERT_NE(native_pick, test_utils::kNoTableEntryRuns);
    {
        const ScopedEnvVar retired("BATCHLAS_SYEV_PROVIDER", auto_pick == "blocked" ? "cta" : "blocked");
        auto p = make_eig<T>(s);
        EXPECT_EQ(traced_choice([&] { this->run(p); }), auto_pick) << "BATCHLAS_SYEV_PROVIDER was read";
    }
    const std::pair<const char*, std::string> expect[] = {
        {"cta", "cta"},             {"BLOCKED", "blocked"}, {"Two_Stage", "two_stage"},
        {"cta_fused", "cta_fused"}, {"jacobi", "jacobi"},   {"vendor", vendor_pick},
        {"native", native_pick},    {"auto", auto_pick}};
    for (const auto& [word, spelling] : expect)
        for (const char* var : {"", "BATCHLAS_SYEV_ROUTE"}) {
            auto p = make_eig<T>(s);
            const std::string got = traced_choice([&] {
                const ScopedEnvVar env(*var ? var : "BATCHLAS_SYEV_UNUSED", word);
                std::optional<Pin> pin;
                if (!*var) pin.emplace("syev", std::string_view(word));
                this->run(p);
            });
            const std::string what = std::string(word) + " via " + (*var ? var : "ScopedPin");
            EXPECT_EQ(got, spelling) << what;
            expect_solved(p, what);
        }
}

// ScopedPin beats BATCHLAS_SYEV_ROUTE.
TYPED_TEST(SyevCandidates, PinPrecedence) {
    using T = typename TestFixture::T;
    const Spec s{10, 2};
    const ScopedEnvVar route("BATCHLAS_SYEV_ROUTE", "blocked");
    auto b = make_eig<T>(s);
    EXPECT_EQ(traced_choice([&] { this->run(b); }), "blocked");
    const Pin pin("syev", C{sy::CtaFused{}});
    auto c = make_eig<T>(s);
    EXPECT_EQ(traced_choice([&] { this->run(c); }), "cta_fused");
    expect_solved(c, "cta_fused over the variable");
}

// key_of's fields reach choose(): a synthetic table whose first entry changes with jobz, n and
// batch alone. Fixing a field in key_of turns exactly its row red.
TYPED_TEST(SyevCandidates, AutoReadsEveryKeyField) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_SYEV_ROUTE", nullptr);
    const std::string dtype(select::dtype_name<T>());
    const std::string dev = select::device_of<TestFixture::B>(*this->ctx).key;
    auto files = tables_without_syev();
    files.emplace_back("syev." + dtype + "." + dev + ".txt",
                       "# op=syev dtype=" + dtype + " device=" + dev + " kernels=unknown\n"
                       "# keys: jobz:exact n:log:3 batch:log\n"
                       "jobz=V n=8 batch=128 | jacobi 1 | cta 2\n"
                       "jobz=V n=8 batch=32768 | cta_fused 1 | cta 2\n"
                       "jobz=V n=16 batch=128 | blocked 1 | cta 2\n"
                       "jobz=N n=8 batch=128 | cta 1 | jacobi 2\n");
    const TableGuard restore;
    select::testing::set_builtin_tables(std::move(files));
    struct Probe { JobType j; int n, batch; const char* expect; const char* field; };
    const Probe probes[] = {{JobType::EigenVectors, 8, 100, "jacobi", "base"},
                            {JobType::EigenVectors, 8, 30000, "cta_fused", "batch"},
                            {JobType::EigenVectors, 15, 100, "blocked", "n"},
                            {JobType::NoEigenVectors, 8, 100, "cta", "jobz"}};
    for (const auto& k : probes) {
        Spec s{k.n, k.batch, k.j};
        auto p = make_eig<T>(s);
        EXPECT_EQ(traced_choice([&] { this->run(p); }), k.expect) << "the " << k.field << " row";
        expect_solved(p, std::string("the ") + k.field + " row");
    }
}

// The trace key is key_of itself: jobz, n and batch.
TYPED_TEST(SyevCandidates, TraceLineShowsTheKey) {
    using T = typename TestFixture::T;
    auto p = make_eig<T>(Spec{7, 5, JobType::NoEigenVectors, Uplo::Upper});
    const std::string line = traced_line([&] { this->run(p); });
    EXPECT_NE(line.find("jobz=N n=7 batch=5 ->"), std::string::npos) << line;
}

// Auto against the shipped transcribed tables, on the device's own (or borrowed) table: the
// old small-kernel pick per type at n 8|9, the cdouble CTA-vs-vendor edge 24|25, CTA's 32|33.
TYPED_TEST(SyevCandidates, AutoReadsTheTranscribedTable) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_SYEV_ROUTE", nullptr);
    const auto tables = select::tables_in_borrow_order("syev", select::dtype_name<T>(),
                                                       select::device_of<TestFixture::B>(*this->ctx));
    if (tables.empty() || tables.front()->source.rfind("transcribed:", 0) != 0)
        GTEST_SKIP() << "this device reads no transcribed syev table";
    const std::string dt(select::dtype_name<T>());
    const std::string v = TestFixture::kVendor ? "vendor" : "cta";
    const std::map<std::string, std::vector<std::pair<int, std::string>>> want{
        {"float", {{8, "jacobi"}, {9, "cta_fused"}, {32, "cta_fused"}, {33, "blocked"}}},
        {"double", {{8, "jacobi"}, {9, "jacobi"}, {32, "jacobi"}, {33, "blocked"}}},
        {"cfloat", {{8, "cta_fused"}, {9, "cta"}, {32, "cta"}, {33, "blocked"}}},
        {"cdouble", {{8, "cta"}, {24, "cta"}, {25, v}, {33, "blocked"}}}};
    for (const auto& [n, expect] : want.at(dt)) {
        Spec s{n, 3, JobType::EigenVectors};
        auto p = make_eig<T>(s);
        EXPECT_EQ(traced_choice([&] { this->run(p); }), expect) << label(s);
        expect_solved(p, "auto " + label(s));
    }
}

// With a table naming only the vendor, a vendor-free build takes the last resort, blocked.
TYPED_TEST(SyevCandidates, VendorFreeLastResortIsBlocked) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_SYEV_ROUTE", nullptr);
    const std::string dtype(select::dtype_name<T>());
    const std::string dev = select::device_of<TestFixture::B>(*this->ctx).key;
    auto files = tables_without_syev();
    files.emplace_back("syev." + dtype + "." + dev + ".txt",
                       "# op=syev dtype=" + dtype + " device=" + dev + " kernels=unknown\n"
                       "# keys: jobz:exact n:log:3 batch:log\njobz=V n=64 batch=128 | vendor 1\n");
    const TableGuard restore;
    select::testing::set_builtin_tables(std::move(files));
    for (int n : {16, 48}) {
        auto p = make_eig<T>(Spec{n, 2});
        std::string all;
        EXPECT_EQ(traced_choice([&] { this->run(p); }, &all), TestFixture::kVendor ? "vendor" : "blocked") << n;
        if (!TestFixture::kVendor) EXPECT_NE(all.find("last resort"), std::string::npos) << all;
        expect_solved(p, "vendor-only table, n " + std::to_string(n));
    }
}

// §5.6: the coverage row carries the backend, the choice spelling and the native flags.
// threadsafe: the child re-executes the binary, so CUDA is initialised fresh, never forked.
TYPED_TEST(SyevCandidates, CoverageRowCarriesChoiceAndNativeFlags) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    const std::string dir = ::testing::TempDir() + "syev_cov." + std::string(select::dtype_name<T>());
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const std::string out = dir + "/cov";
    auto child = [&] {
        const ScopedEnvVar cov("BATCHLAS_COVERAGE_OUT", out.c_str());
        batchlas::coverage::g_dynamic_enabled = true;
        {
            const Pin pin("syev", C{sy::Jacobi{}});
            auto p = make_eig<T>(Spec{12, 2});
            this->run(p);
        }
        const Pin pin("syev", C{sy::Blocked{}});
        auto p = make_eig<T>(Spec{40, 2, JobType::NoEigenVectors, Uplo::Upper});
        this->run(p);
        std::exit(0);
    };
    EXPECT_EXIT(child(), ::testing::ExitedWithCode(0), "");
    std::map<std::string, std::vector<std::string>> rows;
    for (const auto& ent : std::filesystem::directory_iterator(dir)) {
        std::ifstream in(ent.path());
        for (std::string line; std::getline(in, line);) {
            if (line.rfind("reached,syev,", 0) != 0) continue;
            std::vector<std::string> f;
            std::stringstream ss(line);
            for (std::string tok; std::getline(ss, tok, ',');) f.push_back(tok);
            ASSERT_GE(f.size(), 16u) << line;
            rows[f[5]] = f;
        }
    }
    std::filesystem::remove_all(dir);
    ASSERT_EQ(rows.size(), 2u);
    const std::string backend = B == Backend::CUDA ? "CUDA" : (B == Backend::ROCM ? "ROCM" : "?");
    for (const auto& [m, algo, uplo] : {std::tuple{"12", "jacobi", Uplo::Lower}, std::tuple{"40", "blocked", Uplo::Upper}}) {
        ASSERT_TRUE(rows.count(m)) << m;
        const auto& f = rows[m];
        EXPECT_EQ(f[3], backend) << m;
        EXPECT_EQ(f[10], algo) << m;
        EXPECT_EQ(f[12], "1") << m;
        EXPECT_EQ(f[13], "1") << m;
        EXPECT_EQ(f[15], std::to_string(static_cast<int>(uplo))) << m;
    }
}

// On a CPU queue no native family can run: native pins throw and Auto is the vendor.
template <typename Config>
class SyevCandidatesCpu : public test_utils::BatchLASTest<Config> {};
TYPED_TEST_SUITE(SyevCandidatesCpu, Types);

TYPED_TEST(SyevCandidatesCpu, CpuQueueRunsNoNativeFamily) {
    using T = typename TypeParam::ScalarType;
    static constexpr Backend B = TypeParam::BackendVal;
    if (!this->ctx) GTEST_SKIP() << "no queue";
    if (this->ctx->device().type == DeviceType::GPU) GTEST_SKIP() << "a GPU queue";
    const ScopedEnvVar clear("BATCHLAS_SYEV_ROUTE", nullptr);
    const Spec s{8, 2};
    for (const C& c : sy::candidates<T>()) {
        if (std::holds_alternative<sy::Vendor>(c)) continue;
        auto p = make_eig<T>(s);
        const Pin pin("syev", c);
        EXPECT_THROW(((void)syev_buffer_size<B, T>(*this->ctx, p.A(), p.w.to_span(), s.jobz, s.uplo)),
                     std::invalid_argument)
            << select::to_string(c);
    }
    // The syev_supports_* bindings ask can_run with Backend::AUTO, so only the is_gpu term refuses.
    {
        namespace det = blas::dispatch::detail;
        auto sq = make_eig<T>(s);
        EXPECT_FALSE(det::syev_supports_cta<T>(*this->ctx, sq.A()));
        EXPECT_FALSE(det::syev_supports_blocked<T>(*this->ctx, sq.A(), Uplo::Lower));
        EXPECT_FALSE(det::syev_supports_two_stage<T>(*this->ctx, sq.A(), Uplo::Upper));
    }
    auto p = make_eig<T>(s);
    auto call = [&] {
        UnifiedVector<std::byte> ws(
            std::max<std::size_t>(syev_buffer_size<B, T>(*this->ctx, p.A(), p.w.to_span(), s.jobz, s.uplo), 1));
        (void)syev<B, T>(*this->ctx, p.A(), p.w.to_span(), s.jobz, s.uplo, ws.to_span(), Span<int32_t>{});
        this->ctx->wait();
    };
    if constexpr (batchlas::select::solver_vendor_available<B>) {
        EXPECT_EQ(traced_choice(call), "vendor");
        expect_solved(p, "cpu vendor");
    } else {
        EXPECT_THROW(call(), batchlas::NoRouteError);
    }
}

// The sm_89 transcribed rows, read with Table::nearest directly so every device checks them: each
// old threshold on its two sides, per type and jobz (the old predicates, syev.hh@424a45bc). sm_120
// is deep-measured since 2026-10-09 (tuned_tables_tests checks its provenance).
TEST(SyevTranscribedTable, RowsHoldTheOldPreference) {
    struct Row { const char* dtype; const char* jobz; int n; const char* ranked; };
    const Row rows[] = {
        {"float", "V", 8, "jacobi|blocked|two_stage|vendor"},     {"float", "V", 9, "cta_fused|blocked|two_stage|vendor"},
        {"float", "V", 33, "blocked|vendor|two_stage"},            {"float", "V", 448, "blocked|vendor|two_stage"},
        {"float", "V", 449, "two_stage|vendor|blocked"},           {"float", "V", 1024, "two_stage|vendor|blocked"},
        {"float", "V", 1025, "vendor|blocked|two_stage"},          {"float", "N", 320, "blocked|vendor|two_stage"},
        {"float", "N", 321, "two_stage|vendor|blocked"},           {"double", "V", 449, "vendor|blocked|two_stage"},
        {"double", "N", 9, "jacobi|blocked|two_stage|vendor"},     {"cfloat", "V", 8, "cta_fused|blocked|two_stage|vendor"},
        {"cfloat", "V", 512, "blocked|vendor|two_stage"},          {"cfloat", "V", 513, "vendor|blocked|two_stage"},
        {"cdouble", "V", 24, "cta|blocked|two_stage|vendor"},      {"cdouble", "V", 25, "vendor|cta|blocked|two_stage"},
        {"cdouble", "N", 25, "cta|blocked|two_stage|vendor"},      {"cdouble", "V", 256, "blocked|vendor|two_stage"},
        {"cdouble", "V", 257, "vendor|blocked|two_stage"},         {"cdouble", "N", 4000, "two_stage|vendor|blocked"}};
    for (const char* dev : {"sm_89"})
        for (const Row& r : rows)
            for (int batch : {1, 300, 100000}) {
                const auto tables = select::tables_in_borrow_order("syev", r.dtype, select::device_from_key(dev));
                ASSERT_FALSE(tables.empty()) << r.dtype;
                const select::Table& t = *tables.front();
                ASSERT_EQ(t.device, dev) << r.dtype;
                EXPECT_EQ(t.source.rfind("transcribed:", 0), 0u) << t.file;
                const select::TableRow* row = t.nearest({{"jobz", r.jobz}, {"n", r.n}, {"batch", batch}});
                ASSERT_NE(row, nullptr) << t.file;
                std::string got;
                for (const auto& e : row->ranked) got += (got.empty() ? "" : "|") + e.spelling;
                EXPECT_EQ(got, r.ranked) << t.file << ":" << row->line << " n=" << r.n << " batch=" << batch;
                EXPECT_FALSE(row->timed) << t.file;
            }
}

}  // namespace
