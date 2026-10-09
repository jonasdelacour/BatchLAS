// Every potrf candidate, pinned: docs/design/flat-kernel-selection.md §8.1-§8.3 and §8.5.
// The limit oracle below reads the drivers' own capacity queries; whether a pin was taken is
// read back from the select trace, never assumed from the pin being accepted.
#include <gtest/gtest.h>

#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/functions/potrf.hh>
#include <batchlas/blas/functions/trsm.hh>
#include <batchlas/no_route.hh>
#include "../src/select/vendor.hh"
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>

#include "test_utils.hh"

#include "../src/extensions/potrf_native.hh"
#include "../src/ops/potrf/choice.hh"

#include <algorithm>
#include <climits>
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

namespace pc = batchlas::ops::potrf;
using C = pc::PotrfChoice;
using Pin = select::ScopedPin<C>;

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
RealOf<T> re(T v) {
    if constexpr (kCx<T>) return v.real();
    else return v;
}
template <typename T>
RealOf<T> im(T v) {
    if constexpr (kCx<T>) return v.imag();
    else return RealOf<T>(0);
}
template <typename T>
T mk(RealOf<T> r, RealOf<T> i) {
    if constexpr (kCx<T>) return T(r, i);
    else return r;
}
template <typename T>
bool same_bits(T a, T b) {
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}

// Hermitian and strictly diagonally dominant, so PD with condition number below 8. Every
// off-diagonal entry has a nonzero imaginary part. O(n^2), so the wide tiers' ceilings stay
// affordable on the host.
template <typename T>
std::vector<T> make_hpd(int n, unsigned seed) {
    using R = RealOf<T>;
    std::mt19937 gen(seed);
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

template <typename T>
RealOf<T> tol(int n) {
    return RealOf<T>(16) * RealOf<T>(std::max(n, 1)) * std::numeric_limits<RealOf<T>>::epsilon();
}

// One batch at a padded ld and a stride that is not ld*n. Everything outside the factored
// triangle -- the other triangle, the ld padding and the inter-item gap -- holds a large
// finite value a kernel would happily consume if it read it.
template <typename T>
struct Prob {
    int n = 0, batch = 0, ld = 0, stride = 0;
    Uplo uplo = Uplo::Lower;
    UnifiedVector<T> buf;
    UnifiedVector<T*> ptrs;  // the vendor's batched call needs a pointer array
    std::vector<T> before;
    std::vector<std::vector<T>> ref;

    MatrixView<T, MatrixFormat::Dense> view() {
        return MatrixView<T, MatrixFormat::Dense>(buf.data(), n, n, ld, stride, batch, ptrs.data());
    }
    size_t at(int i, int j, int b) const {
        return static_cast<size_t>(b) * stride + i + static_cast<size_t>(j) * ld;
    }
    bool in_tri(int i, int j) const { return uplo == Uplo::Lower ? i >= j : i <= j; }
};

template <typename T>
Prob<T> make_prob(int n, int batch, Uplo uplo, unsigned seed, bool identical = false) {
    using R = RealOf<T>;
    Prob<T> p;
    p.n = n;
    p.batch = batch;
    p.uplo = uplo;
    p.ld = n + 3;
    p.stride = p.ld * n + 5;
    p.buf = UnifiedVector<T>(static_cast<size_t>(p.stride) * batch, mk<T>(R(-999), R(777)));
    p.ptrs = UnifiedVector<T*>(batch, nullptr);
    p.ref.resize(batch);
    for (int b = 0; b < batch; ++b) {
        p.ref[b] = (identical && b > 0) ? p.ref[0] : make_hpd<T>(n, seed + 7919u * b);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                if (p.in_tri(i, j)) p.buf[p.at(i, j, b)] = p.ref[b][i + static_cast<size_t>(j) * n];
    }
    p.before.assign(p.buf.begin(), p.buf.end());
    return p;
}

// ||L L^H - A||_F / ||A||_F over the lower triangle (A is Hermitian).
template <typename T>
RealOf<T> residual(const Prob<T>& p, int b) {
    using R = RealOf<T>;
    const int n = p.n;
    auto L = [&](int i, int j) -> T {
        return p.uplo == Uplo::Lower ? p.buf[p.at(i, j, b)] : cj(p.buf[p.at(j, i, b)]);
    };
    R num = 0, den = 0;
    for (int j = 0; j < n; ++j)
        for (int i = j; i < n; ++i) {
            T acc{};
            for (int k = 0; k <= j; ++k) acc += L(i, k) * cj(L(j, k));
            const T a = p.ref[b][i + static_cast<size_t>(j) * n];
            const T d = acc - a;
            num += re(d) * re(d) + im(d) * im(d);
            den += re(a) * re(a) + im(a) * im(a);
        }
    return den == R(0) ? R(0) : std::sqrt(num / den);
}

// info, the residual of the first and last item, and every element the factor must not
// touch -- the other triangle, the ld padding and the inter-item gap -- bit for bit.
template <typename T>
void expect_factored(const Prob<T>& p, const std::vector<int32_t>& info, const std::string& what,
                     bool other_triangle_is_scratch = false) {
    for (int b = 0; b < p.batch; ++b) ASSERT_EQ(info[b], 0) << what << " b=" << b;
    for (int b : {0, p.batch - 1}) EXPECT_LE(residual(p, b), tol<T>(p.n)) << what << " b=" << b;
    for (size_t e = 0; e < p.before.size(); ++e) {
        const int b = static_cast<int>(e / p.stride);
        const int r = static_cast<int>(e % p.stride);
        const int i = r % p.ld, j = r / p.ld;
        const bool owned = b < p.batch && j < p.n && i < p.n && (other_triangle_is_scratch || p.in_tri(i, j));
        if (!owned)
            ASSERT_TRUE(same_bits(p.buf[e], p.before[e]))
                << what << ": wrote outside the factored triangle at element " << e << " (b=" << b
                << " i=" << i << " j=" << j << ")";
    }
}

// cuSOLVER's Upper potrf writes the strictly lower triangle (docs/developer/agent-guide.md §8 rule 4); the
// ld padding and the inter-item gap must still survive it.
bool clobbers_other(const C& c, Uplo uplo) { return std::holds_alternative<pc::Vendor>(c) && uplo == Uplo::Upper; }

// The spelling the outermost potrf trace line names for whatever `run` calls.
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
    std::string line;
    while (std::getline(in, line)) {
        const auto arrow = line.find(" -> ");
        if (line.rfind("potrf ", 0) != 0 || arrow == std::string::npos) continue;
        const std::string tail = line.substr(arrow + 4);
        return tail.substr(0, tail.find(' '));
    }
    return "<no potrf trace line in: " + err + ">";
}

template <typename T, Backend B>
struct Cfg {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};
using Types = typename test_utils::backend_types<Cfg>::type;

template <typename Config>
class PotrfCandidates : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    using R = RealOf<T>;
    static constexpr Backend B = Config::BackendVal;
    static constexpr int kUnbounded = INT_MAX;

    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        if (this->HasFatalFailure() || ::testing::Test::IsSkipped()) return;
        if (!this->ctx) GTEST_SKIP() << "no queue";
        if (this->ctx->device().type != DeviceType::GPU) GTEST_SKIP() << "the native tiers are GPU kernels";
        if (!this->ctx->device().supports_sub_group_size(32)) GTEST_SKIP() << "no sub-group size 32";
    }

    std::size_t budget() const {
        return resident::device_slm_budget(this->ctx->device().get_property(DeviceProperty::LOCAL_MEM_SIZE));
    }
    int max_wg() const {
        return static_cast<int>(this->ctx->device().get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    }
    int blocked_nb() const {
        return static_cast<int>(sycl_potrf::potrf_blocked_debug_params<T>(*this->ctx, 1 << 20) & 0xffffu);
    }

    // The largest order a candidate claims for `uplo`: 0 for none, kUnbounded for no ceiling.
    // Read from each driver's own capacity query.
    int limit(const C& c, Uplo uplo) const {
        const bool lower = uplo == Uplo::Lower;
        if (std::holds_alternative<pc::Tiny>(c))
            return max_wg() >= sycl_potrf::kPotrfTinyWgSize ? sycl_potrf::potrf_tiny_max_n<T>() : 0;
        if (std::holds_alternative<pc::Cta>(c)) return sycl_potrf::potrf_cta_max_n_for_slm<T>(budget());
        if (const auto* l = std::get_if<pc::Lpanel>(&c))
            return lower ? sycl_potrf::potrf_lpanel_max_n_for_slm<T>(budget(), max_wg(),
                                                                     resident::kMinBlocksPerSm, l->panel)
                         : 0;
        if (std::holds_alternative<pc::Blocked>(c))
            return lower && sycl_potrf::potrf_blocked_available<T>() ? kUnbounded : 0;
        return batchlas::select::solver_vendor_available<B> ? kUnbounded : 0;
    }

    // Orders that straddle every candidate's ceiling, for every candidate to face.
    std::vector<int> straddle_orders() const {
        std::vector<int> ns{1, 2};
        for (const C& c : pc::candidates<T>()) {
            const int lim = limit(c, Uplo::Lower);
            if (lim > 0 && lim != kUnbounded) {
                ns.push_back(lim);
                ns.push_back(lim + 1);
            }
        }
        std::sort(ns.begin(), ns.end());
        ns.erase(std::unique(ns.begin(), ns.end()), ns.end());
        return ns;
    }

    // Whether a pin of `c` is accepted: potrf_buffer_size runs choose() and launches nothing.
    bool pin_accepted(const C& c, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo) {
        const Pin pin("potrf", c);
        try {
            (void)potrf_buffer_size<B, T>(*this->ctx, A, uplo);
            return true;
        } catch (const std::invalid_argument&) {
            return false;
        }
    }

    std::vector<int32_t> run_pinned(const C& c, Prob<T>& p) {
        const Pin pin("potrf", c);
        return run_auto(p);
    }

    // Whatever pin is in force, sized by the same choose().
    std::vector<int32_t> run_auto(Prob<T>& p) {
        const std::size_t bytes = potrf_buffer_size<B, T>(*this->ctx, p.view(), p.uplo);
        UnifiedVector<std::byte> ws(std::max<std::size_t>(bytes, 1));
        UnifiedVector<int32_t> info(p.batch, int32_t(-7));
        (void)potrf<B, T>(*this->ctx, p.view(), p.uplo, Span<std::byte>(ws.data(), bytes), info.to_span());
        this->ctx->wait();
        return std::vector<int32_t>(info.begin(), info.end());
    }

    // The family's own driver, sized by its own query; Blocked gets the public gemm and trsm,
    // which is what the facade injects.
    void run_direct(const C& c, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo, Span<int32_t> info) {
        using MV = MatrixView<T, MatrixFormat::Dense>;
        Queue& q = *this->ctx;
        auto go = [&](std::size_t bytes, auto&& launch) {
            UnifiedVector<std::byte> ws(std::max<std::size_t>(bytes, 1));
            (void)launch(Span<std::byte>(ws.data(), bytes));
            q.wait();
        };
        std::visit([&](const auto& f) {
            using F = std::decay_t<decltype(f)>;
            if constexpr (std::is_same_v<F, pc::Tiny>) {
                go(sycl_potrf::potrf_tiny_buffer_size<T>(q, A),
                   [&](Span<std::byte> w) { return sycl_potrf::potrf_tiny_dispatch<T>(q, A, uplo, w, info); });
            } else if constexpr (std::is_same_v<F, pc::Cta>) {
                go(sycl_potrf::potrf_cta_buffer_size<T>(q, A),
                   [&](Span<std::byte> w) { return sycl_potrf::potrf_cta_dispatch<T>(q, A, uplo, w, info); });
            } else if constexpr (std::is_same_v<F, pc::Lpanel>) {
                go(sycl_potrf::potrf_lpanel_buffer_size<T>(q, A), [&](Span<std::byte> w) {
                    return sycl_potrf::potrf_lpanel_dispatch<T>(q, A, uplo, w, info, resident::kMinBlocksPerSm,
                                                                f.panel);
                });
            } else if constexpr (std::is_same_v<F, pc::Blocked>) {
                go(sycl_potrf::potrf_blocked_buffer_size<T>(q, A, uplo), [&](Span<std::byte> w) {
                    return sycl_potrf::potrf_blocked_dispatch<T>(
                        q, A, uplo, w, info,
                        [](Queue& c2, const MV& a, const MV& b, const MV& r, T al, T be, Transpose ta, Transpose tb,
                           ComputePrecision pr) { return gemm<B, T>(c2, a, b, r, al, be, ta, tb, pr); },
                        [](Queue& c2, const MV& a, const MV& b, T al, Side s, Uplo u, Transpose t, Diag dg) {
                            return trsm<B, T>(c2, a, b, al, s, u, t, dg);
                        });
                });
            } else {
                if constexpr (batchlas::select::solver_vendor_available<B>) {
                    go(backend::potrf_vendor_buffer_size<B, T>(q, A, uplo), [&](Span<std::byte> w) {
                        return backend::potrf_vendor<B, T>(q, A, uplo, w, info);
                    });
                } else {
                    throw std::runtime_error("no vendor solver in this build");
                }
            }
        }, c);
    }

    bool direct_launches(const C& c, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo,
                         std::string* why = nullptr) {
        UnifiedVector<int32_t> info(std::max<int>(A.batch_size(), 1), int32_t(0));
        try {
            run_direct(c, A, uplo, info.to_span());
            return true;
        } catch (const std::exception& e) {
            if (why) *why = e.what();
            return false;
        }
    }

    static std::string name(const C& c, Uplo uplo, int n) {
        return select::to_string(c) + (uplo == Uplo::Lower ? " L" : " U") + " n=" + std::to_string(n);
    }
};

TYPED_TEST_SUITE(PotrfCandidates, Types);

// §8.1: at a candidate's ceiling the pinned call runs and is correct; one past it, or on a
// triangle the candidate does not serve, the pin throws.
TYPED_TEST(PotrfCandidates, PinnedCandidatesStraddleTheirLimits) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    const int nb = this->blocked_nb();
    const int cta = this->limit(C{pc::Cta{}}, Uplo::Lower);
    ASSERT_GT(cta, 0);
    for (const C& c : pc::candidates<T>()) {
        for (Uplo uplo : {Uplo::Lower, Uplo::Upper}) {
            const int lim = this->limit(c, uplo);
            if (lim == 0 && std::holds_alternative<pc::Vendor>(c)) {
                // Vendor compiled out: bare `vendor` is a class word, so it runs Auto (§12).
                auto p = make_prob<T>(8, 2, uplo, 11u);
                expect_factored(p, this->run_pinned(c, p), this->name(c, uplo, 8) + " (vendor-free fallback)");
                continue;
            }
            if (lim == 0) {
                auto p = make_prob<T>(8, 2, uplo, 11u);
                EXPECT_FALSE(this->pin_accepted(c, p.view(), uplo)) << this->name(c, uplo, 8);
                const Pin pin("potrf", c);
                UnifiedVector<std::byte> ws(1 << 16);
                UnifiedVector<int32_t> info(2, 0);
                EXPECT_THROW(((void)potrf<B, T>(*this->ctx, p.view(), uplo, ws.to_span(), info.to_span())),
                             std::invalid_argument) << this->name(c, uplo, 8);
                continue;
            }
            std::vector<int> runs{1, 2};
            if (lim == TestFixture::kUnbounded) {
                runs.push_back(cta + 1);
                runs.push_back(2 * nb + nb / 2);  // several blocks and a short final one
            } else {
                runs.push_back(lim);
                auto past = make_prob<T>(lim + 1, 1, uplo, 13u);
                EXPECT_FALSE(this->pin_accepted(c, past.view(), uplo))
                    << this->name(c, uplo, lim + 1) << " is one past the ceiling and was accepted";
            }
            for (int n : runs) {
                auto p = make_prob<T>(n, n > 256 ? 2 : 3, uplo, 1000u + 31u * n);
                ASSERT_TRUE(this->pin_accepted(c, p.view(), uplo)) << this->name(c, uplo, n);
                expect_factored(p, this->run_pinned(c, p), this->name(c, uplo, n), clobbers_other(c, uplo));
            }
        }
    }
}

// The pinned facade runs exactly that family's driver: bit-identical to the direct call on
// the same input. A pin being accepted says nothing about what ran.
TYPED_TEST(PotrfCandidates, PinnedRunIsTheDirectKernelBitForBit) {
    using T = typename TestFixture::T;
    for (const C& c : pc::candidates<T>()) {
        for (Uplo uplo : {Uplo::Lower, Uplo::Upper}) {
            const int lim = this->limit(c, uplo);
            if (lim == 0) continue;
            for (int n : {std::min(lim, 24), std::min(lim, 100)}) {
                auto pinned = make_prob<T>(n, 3, uplo, 4242u + n);
                auto direct = make_prob<T>(n, 3, uplo, 4242u + n);
                const auto info = this->run_pinned(c, pinned);
                UnifiedVector<int32_t> dinfo(3, int32_t(-7));
                ASSERT_NO_THROW(this->run_direct(c, direct.view(), uplo, dinfo.to_span()))
                    << this->name(c, uplo, n);
                for (int b = 0; b < 3; ++b) ASSERT_EQ(info[b], dinfo[b]) << this->name(c, uplo, n);
                for (size_t e = 0; e < pinned.before.size(); ++e)
                    ASSERT_TRUE(same_bits(pinned.buf[e], direct.buf[e]))
                        << this->name(c, uplo, n) << ": the pinned facade did not run this family's "
                        << "driver; element " << e << " differs";
            }
        }
    }
}

// SLM tiers at a saturating batch: 1024 identical items must come back bit-identical to item 0.
// Small batches cannot race, and the work-group-size ladder hides cross-sub-group races at 32.
TYPED_TEST(PotrfCandidates, SaturatingBatchIsBitIdentical) {
    using T = typename TestFixture::T;
    constexpr int kBatch = 1024;
    const int cta = this->limit(C{pc::Cta{}}, Uplo::Lower);
    for (const C& c : pc::candidates<T>()) {
        if (std::holds_alternative<pc::Vendor>(c)) continue;
        const bool blocked = std::holds_alternative<pc::Blocked>(c);
        for (Uplo uplo : {Uplo::Lower, Uplo::Upper}) {
            const int lim = this->limit(c, uplo);
            if (lim == 0) continue;
            for (int n : {std::min(lim, 17), blocked ? cta + 5 : std::min(lim, 96)}) {
                auto p = make_prob<T>(n, kBatch, uplo, 777u, /*identical=*/true);
                const auto info = this->run_pinned(c, p);
                expect_factored(p, info, this->name(c, uplo, n));
                for (int b = 1; b < kBatch; ++b)
                    for (int j = 0; j < n; ++j)
                        for (int i = 0; i < n; ++i)
                            if (p.in_tri(i, j))
                                ASSERT_TRUE(same_bits(p.buf[p.at(i, j, b)], p.buf[p.at(i, j, 0)]))
                                    << this->name(c, uplo, n) << ": item " << b << " differs from item 0 at ("
                                    << i << "," << j << ")";
            }
        }
    }
}

// §8.2 (R3): on every straddling shape, a pin is accepted exactly when the family's own
// driver launches.
TYPED_TEST(PotrfCandidates, CanRunEqualsLaunch) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;
    std::vector<std::pair<int, int>> shapes;  // (n, batch)
    for (int n : this->straddle_orders()) shapes.push_back({n, 1});
    shapes.push_back({0, 2});
    shapes.push_back({16, 5});
    int disagreements = 0;
    for (const C& c : pc::candidates<T>()) {
        // Compiled out, the vendor pin falls back to Auto (§12): PinnedCandidates covers it.
        if (std::holds_alternative<pc::Vendor>(c) && !batchlas::select::solver_vendor_available<TestFixture::B>) continue;
        for (Uplo uplo : {Uplo::Lower, Uplo::Upper}) {
            for (auto [n, batch] : shapes) {
                auto p = make_prob<T>(n, batch, uplo, 61u);
                const bool pin = this->pin_accepted(c, p.view(), uplo);
                std::string why;
                const bool run = this->direct_launches(c, p.view(), uplo, &why);
                EXPECT_EQ(pin, run) << this->name(c, uplo, n) << " batch=" << batch << ": can_run says "
                                    << pin << ", the driver " << (run ? "launches" : "refuses: " + why);
                disagreements += pin != run;
            }
            // A heterogeneous batch: the natives would silently factor the full order.
            const int n = 16;
            Matrix<T, MatrixFormat::Dense> A(n, n, 4);
            A.fill(T{});
            for (int b = 0; b < 4; ++b)
                for (int i = 0; i < n; ++i) A(i, i, b) = mk<T>(R(2), R(0));
            UnifiedVector<int> act(4);
            for (int b = 0; b < 4; ++b) act[b] = n - b;
            auto V = A.view().with_active_dims(act.to_span(), act.to_span());
            ASSERT_TRUE(V.is_heterogeneous());
            if (!std::holds_alternative<pc::Vendor>(c)) {
                EXPECT_FALSE(this->pin_accepted(c, V, uplo)) << this->name(c, uplo, n) << " heterogeneous";
                EXPECT_FALSE(this->direct_launches(c, V, uplo)) << this->name(c, uplo, n) << " heterogeneous";
            }
        }
    }
    EXPECT_EQ(disagreements, 0);
}

// §8.3 (R5): exactly potrf_buffer_size bytes, inside a larger arena whose tail is a guard. The
// workspace itself starts as all-ones (a NaN pattern), so reading scratch before writing it
// shows in the residual. Empty and full info spans both: an empty span draws info scratch.
TYPED_TEST(PotrfCandidates, ExactWorkspaceInAPoisonedArena) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    constexpr std::size_t kGuard = 4096;
    const int nb = this->blocked_nb();
    for (const C& c : pc::candidates<T>()) {
        for (Uplo uplo : {Uplo::Lower, Uplo::Upper}) {
            const int lim = this->limit(c, uplo);
            if (lim == 0) continue;
            const int big = lim == TestFixture::kUnbounded ? 2 * nb + 3 : lim;
            for (int n : {1, std::min(lim, 33), big}) {
                for (bool pass_info : {true, false}) {
                    auto p = make_prob<T>(n, 5, uplo, 99u + n);
                    const Pin pin("potrf", c);
                    const std::size_t bytes = potrf_buffer_size<B, T>(*this->ctx, p.view(), uplo);
                    UnifiedVector<std::byte> arena(bytes + kGuard);
                    std::memset(arena.data(), 0xFF, bytes);
                    std::memset(arena.data() + bytes, 0xA5, kGuard);
                    UnifiedVector<int32_t> info(p.batch, int32_t(-7));
                    const std::string what = this->name(c, uplo, n) + " bytes=" + std::to_string(bytes) +
                                             (pass_info ? " info" : " no-info");
                    ASSERT_NO_THROW(((void)potrf<B, T>(*this->ctx, p.view(), uplo, Span<std::byte>(arena.data(), bytes),
                                                       pass_info ? info.to_span() : Span<int32_t>{}),
                                     this->ctx->wait()))
                        << what;
                    for (std::size_t i = 0; i < kGuard; ++i)
                        ASSERT_EQ(static_cast<unsigned>(arena[bytes + i]), 0xA5u)
                            << what << ": wrote " << i << " bytes past the sized workspace";
                    if (pass_info) {
                        expect_factored(p, std::vector<int32_t>(info.begin(), info.end()), what,
                                        clobbers_other(c, uplo));
                    } else {
                        expect_factored(p, std::vector<int32_t>(p.batch, 0), what, clobbers_other(c, uplo));
                    }
                }
            }
        }
    }
}

// §8.5 / R6: a spelling that names nothing compiled throws instead of meaning Auto.
TYPED_TEST(PotrfCandidates, UnknownAndUncompiledPinsThrow) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    Matrix<T, MatrixFormat::Dense> A(8, 8, 2);
    std::vector<std::string> bad{"regsiter_tiled", "lpanel:panel=12", "lpanel:panel", "native:lpanel:8", "cta:1",
                             // removed aliases (phase 5): each must stay an error
                             "native:tiny", "native:cta", "native:lpanel", "native:blocked", "lpanel"};
    if (!std::is_same_v<T, float>) bad.push_back("lpanel:panel=16");
    for (const auto& word : bad) {
        const Pin pin("potrf", std::string_view(word));
        EXPECT_THROW(((void)potrf_buffer_size<B, T>(*this->ctx, A.view(), Uplo::Lower)), std::invalid_argument)
            << word;
    }
    for (const C& c : pc::candidates<T>()) {
        const Pin pin("potrf", c);
        EXPECT_NO_THROW(((void)potrf_buffer_size<B, T>(*this->ctx, A.view(), Uplo::Lower))) << select::to_string(c);
    }
}

// The coverage row's native flags come from the candidate list and can_run (§5.6), not the
// old constants 1/-1: Lower n=16 has a native tier, Upper above every Upper-capable tier none.
// threadsafe: the child re-executes the binary, so CUDA is initialised fresh, never forked.
TYPED_TEST(PotrfCandidates, CoverageRowCarriesNativeFlags) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    constexpr bool kVendor = batchlas::select::solver_vendor_available<B>;  // vendor-free, Upper has no route
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    const int up = std::max(this->limit(C{pc::Tiny{}}, Uplo::Upper), this->limit(C{pc::Cta{}}, Uplo::Upper)) + 1;
    const std::string dir = ::testing::TempDir() + "potrf_cov." + std::string(select::dtype_name<T>());
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const std::string out = dir + "/cov";
    auto child = [&] {
        const ScopedEnvVar cov("BATCHLAS_COVERAGE_OUT", out.c_str());
        const ScopedEnvVar clear("BATCHLAS_POTRF_ROUTE", nullptr);
        batchlas::coverage::g_dynamic_enabled = true;
        auto lo = make_prob<T>(16, 2, Uplo::Lower, 51u);
        (void)this->run_auto(lo);
        if (kVendor) {
            auto hi = make_prob<T>(up, 2, Uplo::Upper, 52u);
            (void)this->run_auto(hi);
        }
        std::exit(0);
    };
    EXPECT_EXIT(child(), ::testing::ExitedWithCode(0), "");
    std::map<std::string, std::pair<std::string, std::string>> flags;  // "n uplo" -> (existed, supported)
    for (const auto& ent : std::filesystem::directory_iterator(dir)) {
        std::ifstream in(ent.path());
        for (std::string line; std::getline(in, line);) {
            if (line.rfind("reached,potrf,", 0) != 0) continue;
            std::vector<std::string> f;
            std::stringstream ss(line);
            for (std::string tok; std::getline(ss, tok, ',');) f.push_back(tok);
            ASSERT_GE(f.size(), 16u) << line;
            flags[f[6] + " " + f[15]] = {f[12], f[13]};
        }
    }
    std::filesystem::remove_all(dir);
    const std::string lower = "16 " + std::to_string(static_cast<int>(Uplo::Lower));
    const std::string upper = std::to_string(up) + " " + std::to_string(static_cast<int>(Uplo::Upper));
    ASSERT_EQ(flags.size(), kVendor ? 2u : 1u);
    ASSERT_TRUE(flags.count(lower)) << "no Lower n=16 row";
    EXPECT_EQ(flags[lower], std::make_pair(std::string("1"), std::string("1")));
    if (kVendor) {
        ASSERT_TRUE(flags.count(upper)) << "no Upper n=" << up << " row";
        EXPECT_EQ(flags[upper], std::make_pair(std::string("1"), std::string("0")));
    }
}

// §8.5: the named can_run-false cases, each with its message.
TYPED_TEST(PotrfCandidates, CanRunFalsePinsThrow) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    const int tiny = this->limit(C{pc::Tiny{}}, Uplo::Lower);
    struct Case { C c; Uplo uplo; int n; };
    std::vector<Case> cases{{pc::Lpanel{8}, Uplo::Upper, 40}, {pc::Blocked{}, Uplo::Upper, 40},
                            {pc::Tiny{}, Uplo::Lower, tiny + 1}, {pc::Tiny{}, Uplo::Upper, tiny + 1}};
    for (const auto& k : cases) {
        Matrix<T, MatrixFormat::Dense> A(k.n, k.n, 1);
        const Pin pin("potrf", k.c);
        try {
            (void)potrf_buffer_size<B, T>(*this->ctx, A.view(), k.uplo);
            ADD_FAILURE() << this->name(k.c, k.uplo, k.n) << " was accepted";
        } catch (const std::invalid_argument& e) {
            EXPECT_NE(std::string(e.what()).find("cannot run this shape"), std::string::npos)
                << this->name(k.c, k.uplo, k.n) << ": " << e.what();
        }
    }
}

// §5.3: spellings (case-folded, trimmed, positional) and the class words select their choice.
// Read back from the trace line, not assumed. The removed aliases are in UnknownAndUncompiledPinsThrow.
TYPED_TEST(PotrfCandidates, ClassWordsAndSpellingsSelectTheirChoice) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    const ScopedEnvVar clear("BATCHLAS_POTRF_ROUTE", nullptr);
    // Without the vendor library, bare `vendor` falls back to the automatic choice (§12).
    std::string auto_pick;
    {
        auto p = make_prob<T>(16, 4, Uplo::Lower, 5u);
        auto_pick = traced_choice([&] { (void)this->run_auto(p); });
    }
    const std::string vendor_pick = batchlas::select::solver_vendor_available<B> ? "vendor" : auto_pick;
    ASSERT_NE(vendor_pick, "") << "no auto trace line";
    const std::pair<const char*, const char*> expect[] = {
        {"tiny", "tiny"},       {" CTA ", "cta"},  {"lpanel:8", "lpanel:panel=8"}, {"Blocked", "blocked"},
        {"vendor", vendor_pick.c_str()}, {"LPanel:Panel=8", "lpanel:panel=8"}};
    for (const auto& [word, spelling] : expect) {
        for (bool via_env : {false, true}) {
            auto p = make_prob<T>(16, 4, Uplo::Lower, 5u);
            std::vector<int32_t> info;
            select::testing::reset_warnings();
            std::string err;
            const std::string got = traced_choice([&] {
                const ScopedEnvVar env("BATCHLAS_POTRF_ROUTE", via_env ? word : nullptr);
                std::optional<Pin> pin;
                if (!via_env) pin.emplace("potrf", std::string_view(word));
                info = this->run_auto(p);
            }, &err);
            const std::string what = std::string(word) + (via_env ? " via BATCHLAS_POTRF_ROUTE" : " via ScopedPin");
            EXPECT_EQ(got, spelling) << what;
            const bool fell_back = std::string(word) == "vendor" && !batchlas::select::solver_vendor_available<B>;
            EXPECT_EQ(err.find("pinned \"vendor\", but no vendor candidate") != std::string::npos, fell_back)
                << what << ": " << err;
            expect_factored(p, info, what);
        }
    }
}

// §5.3: bare `native` is the best runnable non-vendor entry of the row, not the first native
// candidate in list order; with none runnable it warns and runs the automatic choice.
TYPED_TEST(PotrfCandidates, BareNativePicksTheBestRunnableNonVendor) {
    using T = typename TestFixture::T;
    if (test_utils::kSlmCappedAt48KiB) GTEST_SKIP() << test_utils::kSlmCappedReason << "; the shipped sm_120 table rows were measured at the 99 KiB budget; per-implementation tables are deferred (sycl-implementations.md)";
    static constexpr Backend B = TestFixture::B;
    const ScopedEnvVar clear("BATCHLAS_POTRF_ROUTE", nullptr);
    const std::string dtype(select::dtype_name<T>());
    const bool sm120 = select::device_of<B>(*this->ctx).key == "sm_120";

    // Rows of tuned/potrf.<dtype>.sm_120.txt that rank vendor first, so `native` must move;
    // the expected pick is the row's first entry that can run, read off the file by hand.
    struct Row { const char* dtype; int n; const char* native; };
    const Row rows[] = {{"float", 128, "lpanel:panel=8"}, {"float", 512, "blocked"},
                        {"double", 64, "lpanel:panel=8"}, {"double", 256, "blocked"},
                        {"cfloat", 64, "lpanel:panel=8"}, {"cfloat", 256, "lpanel:panel=8"},
                        {"cdouble", 16, "tiny"},          {"cdouble", 256, "blocked"}};
    int checked = 0;
    for (const Row& r : rows) {
        if (!sm120 || dtype != r.dtype) continue;
        auto a = make_prob<T>(r.n, 128, Uplo::Lower, 21u);
        std::vector<int32_t> info;
        // Vendor-free, Auto skips the vendor entry and lands where `native` does.
        const char* auto_pick = batchlas::select::solver_vendor_available<B> ? "vendor" : r.native;
        EXPECT_EQ(traced_choice([&] { info = this->run_auto(a); }), auto_pick) << "auto, n=" << r.n;
        expect_factored(a, info, "auto");
        auto p = make_prob<T>(r.n, 128, Uplo::Lower, 22u);
        EXPECT_EQ(traced_choice([&] {
                      const Pin pin("potrf", "native");
                      info = this->run_auto(p);
                  }),
                  r.native)
            << "native, n=" << r.n;
        expect_factored(p, info, "native");
        ++checked;
    }
    if (sm120) EXPECT_EQ(checked, 2) << "no sm_120 rows for " << dtype;

    for (int n : {16, 64, 200}) {
        auto p = make_prob<T>(n, 4, Uplo::Lower, 23u);
        const std::string got = traced_choice([&] {
            const Pin pin("potrf", "native");
            (void)this->run_auto(p);
        });
        EXPECT_NE(got.rfind("vendor", 0), 0u) << "native ran " << got << " at n=" << n;
    }

    // Upper above both Upper-capable tiers: nothing native runs, so a warning and Auto --
    // which, vendor-free, has nothing either and throws the no-route error.
    const int n = std::max(this->limit(C{pc::Tiny{}}, Uplo::Upper), this->limit(C{pc::Cta{}}, Uplo::Upper)) + 1;
    auto p = make_prob<T>(n, 2, Uplo::Upper, 24u);
    select::testing::reset_warnings();
    std::string err;
    std::vector<int32_t> info;
    if constexpr (!batchlas::select::solver_vendor_available<B>) {
        const Pin pin("potrf", "native");
        ::testing::internal::CaptureStderr();
        EXPECT_THROW((void)this->run_auto(p), batchlas::NoRouteError);
        err = ::testing::internal::GetCapturedStderr();
        EXPECT_NE(err.find("pinned \"native\", but no native candidate"), std::string::npos) << err;
        return;
    }
    EXPECT_EQ(traced_choice([&] {
                  const Pin pin("potrf", "native");
                  info = this->run_auto(p);
              }, &err),
              "vendor");
    EXPECT_NE(err.find("pinned \"native\", but no native candidate"), std::string::npos) << err;
    expect_factored(p, info, "native fallback", /*other_triangle_is_scratch=*/true);
}

// §5.3: a ScopedPin wins over BATCHLAS_POTRF_ROUTE, and nested pins restore the outer one.
TYPED_TEST(PotrfCandidates, ScopedPinBeatsTheEnvironment) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    const int n = this->limit(C{pc::Tiny{}}, Uplo::Lower) + 1;
    const ScopedEnvVar env("BATCHLAS_POTRF_ROUTE", "tiny");
    {
        Matrix<T, MatrixFormat::Dense> A(n, n, 1);
        EXPECT_THROW(((void)potrf_buffer_size<B, T>(*this->ctx, A.view(), Uplo::Lower)), std::invalid_argument)
            << "the environment pin was not read, so nothing below proves it lost";
    }
    std::vector<int32_t> info;
    auto p = make_prob<T>(n, 2, Uplo::Lower, 31u);
    EXPECT_EQ(traced_choice([&] {
                  const Pin pin("potrf", C{pc::Cta{}});
                  info = this->run_auto(p);
              }),
              "cta");
    expect_factored(p, info, "cta over env tiny");

    auto q = make_prob<T>(n, 2, Uplo::Lower, 32u);
    std::string got = traced_choice([&] {
        const Pin pin("potrf", "auto");
        info = this->run_auto(q);
    });
    EXPECT_NE(got, "tiny");
    expect_factored(q, info, "auto over env tiny");

    const Pin outer("potrf", C{pc::Cta{}});
    {
        const Pin inner("potrf", C{pc::Blocked{}});
        auto r = make_prob<T>(n, 2, Uplo::Lower, 33u);
        EXPECT_EQ(traced_choice([&] { (void)this->run_auto(r); }), "blocked");
    }
    auto s = make_prob<T>(n, 2, Uplo::Lower, 34u);
    EXPECT_EQ(traced_choice([&] { (void)this->run_auto(s); }), "cta") << "the inner pin did not restore the outer";
}

// §8.4, device half: every entry of every row of this device's own table was timed at that
// row's key, so a pin of it must be accepted there. A can_run stricter than the drivers would
// otherwise skip a measured winner in Auto and run the runner-up, with every result correct.
// can_run reads no batch beyond >= 1, so batch 1 stands in for the row's batch.
TYPED_TEST(PotrfCandidates, ShippedRowsRunOnTheirOwnDevice) {
    using T = typename TestFixture::T;
    if (test_utils::kSlmCappedAt48KiB) GTEST_SKIP() << test_utils::kSlmCappedReason << "; the shipped sm_120 table rows were measured at the 99 KiB budget; per-implementation tables are deferred (sycl-implementations.md)";
    static constexpr Backend B = TestFixture::B;
    const ScopedEnvVar clear("BATCHLAS_POTRF_ROUTE", nullptr);
    const select::Device& d = select::device_of<B>(*this->ctx);
    const auto tables = select::tables_in_borrow_order("potrf", select::dtype_name<T>(), d);
    if (tables.empty() || tables.front()->device != d.key) GTEST_SKIP() << "no shipped table for " << d.key;
    const select::Table& t = *tables.front();
    std::size_t iu = t.keys.size(), in = t.keys.size();
    for (std::size_t k = 0; k < t.keys.size(); ++k) {
        if (t.keys[k].name == "uplo") iu = k;
        if (t.keys[k].name == "n") in = k;
    }
    ASSERT_LT(iu, t.keys.size());
    ASSERT_LT(in, t.keys.size());
    int checked = 0;
    for (const auto& row : t.rows) {
        const Uplo uplo = row.keys[iu] == "U" ? Uplo::Upper : Uplo::Lower;
        const int n = std::stoi(row.keys[in]);
        Matrix<T, MatrixFormat::Dense> A(n, n, 1);
        for (const auto& e : row.ranked) {
            if (e.spelling == "vendor" && !batchlas::select::solver_vendor_available<B>) continue;  // compiled out
            const Pin pin("potrf", std::string_view(e.spelling));
            try {
                (void)potrf_buffer_size<B, T>(*this->ctx, A.view(), uplo);
                ++checked;
            } catch (const std::invalid_argument& ex) {
                ADD_FAILURE() << t.file << ":" << row.line << ": " << e.spelling << " was measured at this key "
                              << "but its pin is refused: " << ex.what();
            }
        }
    }
    EXPECT_GT(checked, 0);
}

// key_of's every field reaches the table: hand-read Auto rows on each side of a batch edge
// (per dtype) and of the uplo key (float, the only dtype with Upper rows). Fixing batch or
// uplo in key_of turns exactly these red; ScopedPin-based tests bypass the key entirely.
TYPED_TEST(PotrfCandidates, AutoReadsEveryKeyField) {
    using T = typename TestFixture::T;
    if (test_utils::kSlmCappedAt48KiB) GTEST_SKIP() << test_utils::kSlmCappedReason << "; the shipped sm_120 table rows were measured at the 99 KiB budget; per-implementation tables are deferred (sycl-implementations.md)";
    static constexpr Backend B = TestFixture::B;
    const ScopedEnvVar clear("BATCHLAS_POTRF_ROUTE", nullptr);
    const std::string dev = select::device_of<B>(*this->ctx).key;
    const std::string dtype(select::dtype_name<T>());
    // `vf`: the vendor-free pick, the row's first non-vendor entry (each runs at its n).
    struct Row { const char* dev; const char* dtype; Uplo uplo; int n, batch; const char* expect; const char* vf; };
    const Row rows[] = {
        {"sm_120", "float", Uplo::Lower, 128, 128, "vendor", "lpanel:panel=8"},
        {"sm_120", "float", Uplo::Lower, 128, 512, "lpanel:panel=8", "lpanel:panel=8"},
        {"sm_120", "float", Uplo::Upper, 64, 8192, "cta", "cta"},  // the L row gives lpanel, then vendor
        {"sm_120", "float", Uplo::Upper, 16, 8192, "tiny", "tiny"},
        {"sm_120", "double", Uplo::Lower, 32, 512, "vendor", "lpanel:panel=8"},
        {"sm_120", "double", Uplo::Lower, 32, 2048, "lpanel:panel=8", "lpanel:panel=8"},
        {"sm_120", "cfloat", Uplo::Lower, 24, 2048, "tiny", "tiny"},
        {"sm_120", "cfloat", Uplo::Lower, 24, 8192, "lpanel:panel=8", "lpanel:panel=8"},
        {"sm_120", "cdouble", Uplo::Lower, 16, 512, "vendor", "tiny"},
        {"sm_120", "cdouble", Uplo::Lower, 16, 8192, "tiny", "tiny"},
        {"sm_89", "float", Uplo::Lower, 44, 8192, "vendor", "cta"},
        {"sm_89", "float", Uplo::Lower, 44, 16384, "cta", "cta"},
    };
    int checked = 0;
    for (const Row& r : rows) {
        if (dev != r.dev || dtype != r.dtype) continue;
        auto p = make_prob<T>(r.n, r.batch, r.uplo, 41u);
        std::vector<int32_t> info;
        const std::string what = std::string(r.uplo == Uplo::Lower ? "L" : "U") + " n=" + std::to_string(r.n) +
                                 " batch=" + std::to_string(r.batch);
        const char* want = batchlas::select::solver_vendor_available<B> ? r.expect : r.vf;
        EXPECT_EQ(traced_choice([&] { info = this->run_auto(p); }), want) << what;
        expect_factored(p, info, what, r.uplo == Uplo::Upper && std::string(want) == "vendor");
        ++checked;
    }
    if (dev == "sm_120" || (dev == "sm_89" && dtype == "float")) EXPECT_GT(checked, 0) << dev << " " << dtype;
    if (checked == 0) GTEST_SKIP() << "no hand-read rows for " << dev << " " << dtype;
}

// Not square never reaches choose() (potrf_validate_params), so each driver checks it itself.
TYPED_TEST(PotrfCandidates, DirectDriversRefuseNonSquare) {
    using T = typename TestFixture::T;
    Matrix<T, MatrixFormat::Dense> A(40, 33, 1);
    A.fill(T{});
    for (const C& c : pc::candidates<T>()) {
        if (std::holds_alternative<pc::Vendor>(c)) continue;
        EXPECT_FALSE(this->direct_launches(c, A.view(), Uplo::Lower)) << select::to_string(c);
    }
}

}  // namespace
