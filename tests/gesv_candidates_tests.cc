// Every gesv candidate, pinned: docs/design/flat-kernel-selection.md §8 and
// docs/design/flat-kernel-selection.md#phase-5-gesv. The limit oracle reads the driver's own capacity
// queries; which kernel ran is read back from the select trace or a bit-for-bit comparison
// with the direct driver, never assumed from the pin being accepted.
#include <gtest/gtest.h>

#include <batchlas/blas/functions/gesv.hh>
#include <batchlas/blas/functions/getrf.hh>
#include <batchlas/blas/functions/getrs.hh>
#include <batchlas/no_route.hh>
#include "../src/select/vendor.hh"
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>

#include "test_utils.hh"

#include "../src/extensions/solve_native.hh"
#include "../src/ops/gesv/choice.hh"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
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

namespace gs = batchlas::ops::gesv;
using C = gs::GesvChoice;
using Pin = select::ScopedPin<C>;

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

// A and B at padded lds and strides that are not ld*cols; the pads hold a large finite value a
// kernel would happily consume. A is diagonally dominant with its rows cyclically shifted by an
// item-dependent offset, so partial pivoting must interchange rows (an identity pivot list would
// hide a dropped interchange); every off-diagonal entry has a nonzero imaginary part.
template <typename T>
struct Sys {
    int n = 0, nrhs = 0, batch = 0, lda = 0, stra = 0, ldb = 0, strb = 0;
    UnifiedVector<T> a, b;
    UnifiedVector<T*> aptr, bptr;
    UnifiedVector<int64_t> piv;
    std::vector<T> a0, b0;

    MatrixView<T, MatrixFormat::Dense> A() { return {a.data(), n, n, lda, stra, batch, aptr.data()}; }
    MatrixView<T, MatrixFormat::Dense> B() { return {b.data(), n, nrhs, ldb, strb, batch, bptr.data()}; }
};

template <typename T>
Sys<T> make_sys(int n, int nrhs, int batch, unsigned seed, bool identical = false) {
    using R = RealOf<T>;
    Sys<T> p;
    p.n = n, p.nrhs = nrhs, p.batch = batch;
    p.lda = n + 3, p.stra = p.lda * n + 5, p.ldb = n + 2, p.strb = p.ldb * nrhs + 7;
    const T poison = mk<T>(R(-999), R(777));
    p.a = UnifiedVector<T>(static_cast<size_t>(p.stra) * batch, poison);
    p.b = UnifiedVector<T>(static_cast<size_t>(p.strb) * batch, poison);
    p.aptr = UnifiedVector<T*>(batch, nullptr);
    p.bptr = UnifiedVector<T*>(batch, nullptr);
    p.piv = UnifiedVector<int64_t>(static_cast<size_t>(n) * batch, int64_t(0x0BADBEEF0BADBEEFLL));
    std::mt19937 gen(seed);
    std::uniform_real_distribution<R> d(R(-1), R(1));
    std::vector<T> a1(static_cast<size_t>(n) * n), b1(static_cast<size_t>(n) * nrhs);
    for (int it = 0; it < batch; ++it) {
        if (!identical || it == 0) {
            const int shift = n > 1 ? 1 + int(gen() % unsigned(n - 1)) : 0;
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i) {
                    const T v = (i == j) ? mk<T>(R(3) * R(n) * (d(gen) >= 0 ? R(1) : R(-1)), d(gen))
                                         : mk<T>(d(gen), d(gen));
                    a1[(i + shift) % n + size_t(j) * n] = v;
                }
            for (auto& v : b1) v = mk<T>(d(gen), d(gen));
        }
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) p.a[size_t(it) * p.stra + size_t(j) * p.lda + i] = a1[i + size_t(j) * n];
        for (int k = 0; k < nrhs; ++k)
            for (int i = 0; i < n; ++i) p.b[size_t(it) * p.strb + size_t(k) * p.ldb + i] = b1[i + size_t(k) * n];
    }
    p.a0.assign(p.a.begin(), p.a.end());
    p.b0.assign(p.b.begin(), p.b.end());
    return p;
}

// ||A X - B||_F / (||A||_F ||X||_F) in double, against the pristine A.
template <typename T>
double residual(const Sys<T>& p, int it) {
    double num = 0, an = 0, xn = 0;
    const size_t ao = size_t(it) * p.stra, bo = size_t(it) * p.strb;
    for (int j = 0; j < p.n; ++j)
        for (int i = 0; i < p.n; ++i) an += std::norm(up(p.a0[ao + size_t(j) * p.lda + i]));
    for (int k = 0; k < p.nrhs; ++k)
        for (int i = 0; i < p.n; ++i) {
            const size_t col = bo + size_t(k) * p.ldb;
            xn += std::norm(up(p.b[col + i]));
            std::complex<double> acc = -up(p.b0[col + i]);
            for (int t = 0; t < p.n; ++t) acc += up(p.a0[ao + size_t(t) * p.lda + i]) * up(p.b[col + t]);
            num += std::norm(acc);
        }
    return (an == 0 || xn == 0) ? std::sqrt(num) : std::sqrt(num / (an * xn));
}

template <typename T>
double tol(int n) {
    return 64.0 * std::max(n, 1) * double(std::numeric_limits<RealOf<T>>::epsilon());
}

// The GPU arms pack 1-based int32 pivots into the int64 span (AGENTS.md §9).
template <typename T>
int32_t piv32(const Sys<T>& p, int it, int i) {
    return reinterpret_cast<const int32_t*>(p.piv.data())[size_t(it) * p.n + i];
}

// info, the residual of the first and last item, a valid nontrivial pivot list, and every
// padding element of A and B untouched, bit for bit.
template <typename T>
void expect_solved(const Sys<T>& p, const std::vector<int32_t>& info, const std::string& what) {
    for (int it = 0; it < p.batch; ++it) ASSERT_EQ(info[it], 0) << what << " item " << it;
    for (int it : {0, p.batch - 1}) {
        const double r = residual(p, it);
        EXPECT_TRUE(std::isfinite(r) && r <= tol<T>(p.n)) << what << " item " << it << " residual " << r;
        bool moved = false;
        for (int i = 0; i < p.n; ++i) {
            const int32_t pv = piv32(p, it, i);
            ASSERT_TRUE(pv >= i + 1 && pv <= p.n) << what << " item " << it << " pivot " << i << " = " << pv;
            moved |= pv != i + 1;
        }
        if (p.n > 1) EXPECT_TRUE(moved) << what << ": identity pivots on a row-shifted matrix";
    }
    for (size_t e = 0; e < p.a0.size(); ++e) {
        const int r = int(e % p.stra), i = r % p.lda, j = r / p.lda;
        if (!(j < p.n && i < p.n))
            ASSERT_TRUE(same_bits(p.a[e], p.a0[e])) << what << ": A written in its padding at element " << e;
    }
    for (size_t e = 0; e < p.b0.size(); ++e) {
        const int r = int(e % p.strb), i = r % p.ldb, k = r / p.ldb;
        if (!(k < p.nrhs && i < p.n))
            ASSERT_TRUE(same_bits(p.b[e], p.b0[e])) << what << ": B written in its padding at element " << e;
    }
}

// The choice the outermost gesv trace line names for whatever `run` calls.
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
        if (line.rfind("gesv ", 0) != 0 || arrow == std::string::npos) continue;
        const std::string tail = line.substr(arrow + 4);
        return tail.substr(0, tail.find(' '));
    }
    return "<no gesv trace line in: " + err + ">";
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
class GesvCandidates : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    using MV = MatrixView<T, MatrixFormat::Dense>;
    static constexpr Backend B = Config::BackendVal;

    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        if (this->HasFatalFailure() || ::testing::Test::IsSkipped()) return;
        if (!this->ctx) GTEST_SKIP() << "no queue";
        if (this->ctx->device().type != DeviceType::GPU) GTEST_SKIP() << "the native tier is a GPU kernel";
        if (!this->ctx->device().supports_sub_group_size(32)) GTEST_SKIP() << "no sub-group size 32";
    }

    // ---- the limit oracle: the driver's own capacity query, not gesv.cc's can_run ----
    int tiny_n() const {
        const int wg = static_cast<int>(this->ctx->device().get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
        return wg >= sycl_gesv::kGesvTinyWgSize ? sycl_gesv::gesv_tiny_max_n<T>() : 0;
    }
    bool expect_runs(const C& c, int n, int nrhs) const {
        if (std::holds_alternative<gs::Tiny>(c)) return n <= tiny_n() && nrhs <= sycl_gesv::kGesvTinyMaxRhs;
        return true;
    }

    // Whether a pin of `c` is accepted: gesv_buffer_size runs choose() and launches nothing.
    bool pin_accepted(const C& c, const MV& A, const MV& Bm) {
        const Pin pin("gesv", c);
        try {
            (void)gesv_buffer_size<B, T>(*this->ctx, A, Bm);
            return true;
        } catch (const batchlas::NoRouteError&) {
            return true;  // a child (getrf/getrs) without a route: the gesv pin itself was taken
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find("cannot run this shape") == std::string::npos) throw;
            return false;
        }
    }

    // Whatever gesv pin is in force, sized by the same choose().
    std::vector<int32_t> run_auto(Sys<T>& p, bool pass_info = true) {
        const std::size_t bytes = gesv_buffer_size<B, T>(*this->ctx, p.A(), p.B());
        UnifiedVector<std::byte> ws(std::max<std::size_t>(bytes, 1));
        UnifiedVector<int32_t> info(p.batch, int32_t(-7));
        (void)gesv<B, T>(*this->ctx, p.A(), p.B(), p.piv.to_span(), Span<std::byte>(ws.data(), bytes),
                         pass_info ? info.to_span() : Span<int32_t>{});
        this->ctx->wait();
        return pass_info ? std::vector<int32_t>(info.begin(), info.end()) : std::vector<int32_t>(p.batch, 0);
    }
    std::vector<int32_t> run_pinned(const C& c, Sys<T>& p) {
        const Pin pin("gesv", c);
        return run_auto(p);
    }

    // The family's own kernels, called directly: Tiny's driver; Blocked = public getrf + getrs.
    void run_direct(const C& c, Sys<T>& p, Span<int32_t> info) {
        Queue& q = *this->ctx;
        if (std::holds_alternative<gs::Tiny>(c)) {
            const std::size_t bytes = sycl_gesv::gesv_tiny_buffer_size<T>(q, p.A(), p.B());
            UnifiedVector<std::byte> ws(std::max<std::size_t>(bytes, 1));
            (void)sycl_gesv::gesv_tiny_dispatch<T>(q, p.A(), p.B(), p.piv.to_span(), Span<std::byte>(ws.data(), bytes),
                                                   info);
        } else {
            const std::size_t fb = getrf_buffer_size<B, T>(q, p.A());
            UnifiedVector<std::byte> fws(std::max<std::size_t>(fb, 1));
            (void)getrf<B, T>(q, p.A(), p.piv.to_span(), Span<std::byte>(fws.data(), fb), info);
            q.wait();
            const std::size_t sb = getrs_buffer_size<B, T>(q, p.A(), p.B(), Transpose::NoTrans);
            UnifiedVector<std::byte> sws(std::max<std::size_t>(sb, 1));
            (void)getrs<B, T>(q, p.A(), p.B(), Transpose::NoTrans, p.piv.to_span(), Span<std::byte>(sws.data(), sb));
        }
        q.wait();
    }
    // Only the driver's own argument checks. Blocked has no driver of its own; its children decide.
    bool direct_launches(const C& c, const MV& A, const MV& Bm, std::string* why = nullptr) {
        if (!std::holds_alternative<gs::Tiny>(c)) return true;
        Queue& q = *this->ctx;
        UnifiedVector<int32_t> info(std::max<int>(A.batch_size(), 1), int32_t(0));
        UnifiedVector<int64_t> piv(std::max<std::size_t>(std::size_t(A.rows()) * A.batch_size(), 1));
        try {
            const std::size_t bytes = sycl_gesv::gesv_tiny_buffer_size<T>(q, A, Bm);
            UnifiedVector<std::byte> ws(std::max<std::size_t>(bytes, 1));
            (void)sycl_gesv::gesv_tiny_dispatch<T>(q, A, Bm, piv.to_span(), Span<std::byte>(ws.data(), bytes),
                                                   info.to_span());
            q.wait();
            return true;
        } catch (const std::exception& e) {
            if (why) *why = e.what();
            return false;
        }
    }
    // The vendor-free build may have no getrf/getrs route for a child shape.
    static bool child_may_lack_route() { return !batchlas::select::factorization_vendor_available<B>; }

    static std::string name(const C& c, int n, int nrhs) {
        return select::to_string(c) + " n=" + std::to_string(n) + " nrhs=" + std::to_string(nrhs);
    }
};

TYPED_TEST_SUITE(GesvCandidates, Types);

// §8.1: each candidate on shapes straddling both its order and its nrhs ceiling. Accepted
// shapes run and are correct; refused ones throw from the pin, for buffer_size and gesv alike.
TYPED_TEST(GesvCandidates, PinnedCandidatesStraddleTheirLimits) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    const int tiny = this->tiny_n();
    ASSERT_GT(tiny, 0) << "the oracle has no Tiny ceiling on this device";
    std::vector<int> ns{1, 2, 16, 17, 32, 33, tiny, tiny + 1, 70};
    std::sort(ns.begin(), ns.end());
    ns.erase(std::unique(ns.begin(), ns.end()), ns.end());
    int ran = 0, refused = 0;
    for (const C& c : gs::candidates<T>()) {
        for (int n : ns) {
            for (int nrhs : {1, 3, 4, 5, 9}) {
                auto p = make_sys<T>(n, nrhs, 3, 1000u + 31u * n + nrhs);
                const std::string what = this->name(c, n, nrhs);
                if (!this->expect_runs(c, n, nrhs)) {
                    EXPECT_FALSE(this->pin_accepted(c, p.A(), p.B())) << what << " was accepted";
                    const Pin pin("gesv", c);
                    UnifiedVector<std::byte> ws(1 << 16);
                    UnifiedVector<int32_t> info(p.batch, 0);
                    EXPECT_THROW(((void)gesv<B, T>(*this->ctx, p.A(), p.B(), p.piv.to_span(), ws.to_span(),
                                                   info.to_span())),
                                 std::invalid_argument) << what;
                    ++refused;
                    continue;
                }
                ASSERT_TRUE(this->pin_accepted(c, p.A(), p.B())) << what << " was refused";
                try {
                    expect_solved(p, this->run_pinned(c, p), what);
                    ++ran;
                } catch (const batchlas::NoRouteError& e) {
                    if (!this->child_may_lack_route()) ADD_FAILURE() << what << ": " << e.what();
                }
            }
        }
    }
    EXPECT_GT(ran, 0);
    EXPECT_GT(refused, 0);
}

// The pinned facade runs exactly that family's kernels: bit-identical to the direct calls on
// the same input, factor, pivots and X.
TYPED_TEST(GesvCandidates, PinnedRunIsTheDirectKernelBitForBit) {
    using T = typename TestFixture::T;
    const int tiny = this->tiny_n();
    for (const C& c : gs::candidates<T>()) {
        for (auto [n, nrhs] : {std::pair{std::min(tiny, 13), 3}, std::pair{tiny, 4}, std::pair{40, 7}}) {
            if (!this->expect_runs(c, n, nrhs)) continue;
            auto pinned = make_sys<T>(n, nrhs, 3, 4242u + n);
            auto direct = make_sys<T>(n, nrhs, 3, 4242u + n);
            const std::string what = this->name(c, n, nrhs);
            std::vector<int32_t> info;
            try {
                info = this->run_pinned(c, pinned);
            } catch (const batchlas::NoRouteError&) {
                if (this->child_may_lack_route()) continue;
                throw;
            }
            UnifiedVector<int32_t> dinfo(3, int32_t(-7));
            ASSERT_NO_THROW(this->run_direct(c, direct, dinfo.to_span())) << what;
            for (int it = 0; it < 3; ++it) ASSERT_EQ(info[it], dinfo[it]) << what;
            expect_solved(pinned, info, what);
            for (size_t e = 0; e < pinned.b.size(); ++e)
                ASSERT_TRUE(same_bits(pinned.b[e], direct.b[e]))
                    << what << ": the pinned facade did not run this family's kernels; X element " << e;
            for (size_t e = 0; e < pinned.a.size(); ++e)
                ASSERT_TRUE(same_bits(pinned.a[e], direct.a[e])) << what << ": factor element " << e;
            for (size_t e = 0; e < pinned.piv.size(); ++e) ASSERT_EQ(pinned.piv[e], direct.piv[e]) << what << " pivot";
        }
    }
}

// The tiny tier at a saturating batch: 1024 identical items come back bit-identical to item 0
// (the packed launch puts several matrices in one work-group; SLM staging for float/cfloat).
TYPED_TEST(GesvCandidates, SaturatingBatchIsBitIdentical) {
    using T = typename TestFixture::T;
    constexpr int kBatch = 1024;
    const int tiny = this->tiny_n();
    struct Case { C c; int n, nrhs; };
    const Case cases[] = {{gs::Tiny{}, std::min(tiny, 17), 4}, {gs::Tiny{}, tiny, 3}, {gs::Tiny{}, 5, 1},
                          {gs::Blocked{}, 24, 2}};
    for (const auto& k : cases) {
        auto p = make_sys<T>(k.n, k.nrhs, kBatch, 777u, /*identical=*/true);
        const std::string what = this->name(k.c, k.n, k.nrhs) + " batch=1024";
        std::vector<int32_t> info;
        try {
            info = this->run_pinned(k.c, p);
        } catch (const batchlas::NoRouteError&) {
            if (this->child_may_lack_route()) continue;
            throw;
        }
        expect_solved(p, info, what);
        for (int it = 1; it < kBatch; ++it) {
            for (int col = 0; col < k.nrhs; ++col)
                for (int i = 0; i < k.n; ++i) {
                    const size_t o = size_t(col) * p.ldb + i;
                    ASSERT_TRUE(same_bits(p.b[size_t(it) * p.strb + o], p.b[o]))
                        << what << ": X of item " << it << " differs from item 0 at (" << i << "," << col << ")";
                }
            for (int j = 0; j < k.n; ++j)
                for (int i = 0; i < k.n; ++i) {
                    const size_t o = size_t(j) * p.lda + i;
                    ASSERT_TRUE(same_bits(p.a[size_t(it) * p.stra + o], p.a[o]))
                        << what << ": the factor of item " << it << " differs from item 0";
                }
            for (int i = 0; i < k.n; ++i)
                ASSERT_EQ(piv32(p, it, i), piv32(p, 0, i)) << what << " pivot of item " << it;
        }
    }
}

// §8.2 (R3): on every straddling shape a pin is accepted exactly when the family's own driver
// launches (gesv_tiny_dispatch). Blocked has no driver: can_run is true on every homogeneous
// batch. Absorbs gesv_tests' old G8 (SupportsAgreesWithTheLauncherCeilings).
TYPED_TEST(GesvCandidates, CanRunEqualsLaunch) {
    using T = typename TestFixture::T;
    const int tiny = this->tiny_n();
    int disagreements = 0;
    for (const C& c : gs::candidates<T>()) {
        for (int n : {1, 2, 8, 16, 17, 32, 33, tiny, tiny + 1, 64, 100}) {
            for (int nrhs : {1, 2, 3, 4, 5, 8}) {
                auto p = make_sys<T>(n, nrhs, 2, 61u);
                const std::string what = this->name(c, n, nrhs);
                const bool pin = this->pin_accepted(c, p.A(), p.B());
                std::string why;
                const bool run = this->direct_launches(c, p.A(), p.B(), &why);
                EXPECT_EQ(pin, run) << what << ": can_run says " << pin << ", the driver "
                                    << (run ? "launches" : "refuses: " + why);
                EXPECT_EQ(pin, this->expect_runs(c, n, nrhs)) << what << ": can_run disagrees with the oracle";
                disagreements += pin != run;
            }
        }
    }
    EXPECT_EQ(disagreements, 0);
}

// A heterogeneous A or B is refused before choose(), under Auto and under every pin, with the
// old router's internal_error. The tiny driver refuses it as well.
TYPED_TEST(GesvCandidates, HeterogeneousBatchIsRefusedUnderEveryPin) {
    using T = typename TestFixture::T;
    using R = RealOf<T>;
    static constexpr Backend B = TestFixture::B;
    const int n = 16, nrhs = 3, batch = 4;
    Matrix<T, MatrixFormat::Dense> A(n, n, batch), Bm(n, nrhs, batch);
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) A(i, j, b) = mk<T>(R(i == j ? 40 : -1), R(0.25));
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
    for (const C& c : gs::candidates<T>()) pins.push_back(c);
    UnifiedVector<int64_t> piv(size_t(n) * batch);
    for (const auto& k : cases)
        for (const auto& c : pins) {
            std::optional<Pin> pin;
            if (c) pin.emplace("gesv", *c);
            const std::string what = std::string(k.what) + " " + (c ? select::to_string(*c) : "auto");
            EXPECT_THROW(((void)gesv_buffer_size<B, T>(*this->ctx, k.a, k.b)), batchlas::internal_error)
                << what << ": gesv_buffer_size";
            UnifiedVector<std::byte> ws(1 << 20);
            UnifiedVector<int32_t> info(batch);
            EXPECT_THROW(((void)gesv<B, T>(*this->ctx, k.a, k.b, piv.to_span(), ws.to_span(), info.to_span())),
                         batchlas::internal_error)
                << what << ": gesv";
            if (c && std::holds_alternative<gs::Tiny>(*c))
                EXPECT_FALSE(this->direct_launches(*c, k.a, k.b)) << what << ": the driver launched";
        }
    this->ctx->wait();
}

// §8.3 (R5): exactly gesv_buffer_size bytes inside a larger arena whose tail is a guard. The
// workspace starts as all-ones (a NaN pattern). With an empty info span every family draws its
// info scratch from the span.
TYPED_TEST(GesvCandidates, ExactWorkspaceInAPoisonedArena) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    constexpr std::size_t kGuard = 4096;
    const int tiny = this->tiny_n();
    for (const C& c : gs::candidates<T>()) {
        for (auto [n, nrhs] : {std::pair{1, 1}, std::pair{tiny, 4}, std::pair{33, 8}, std::pair{130, 9}}) {
            if (!this->expect_runs(c, n, nrhs)) continue;
            for (bool pass_info : {true, false}) {
                auto p = make_sys<T>(n, nrhs, 5, 99u + n);
                const Pin pin("gesv", c);
                std::size_t bytes = 0;
                try {
                    bytes = gesv_buffer_size<B, T>(*this->ctx, p.A(), p.B());
                } catch (const batchlas::NoRouteError&) {
                    if (this->child_may_lack_route()) continue;
                    throw;
                }
                UnifiedVector<std::byte> arena(bytes + kGuard);
                std::memset(arena.data(), 0xFF, bytes);
                std::memset(arena.data() + bytes, 0xA5, kGuard);
                UnifiedVector<int32_t> info(p.batch, int32_t(-7));
                const std::string what = this->name(c, n, nrhs) + " bytes=" + std::to_string(bytes) +
                                         (pass_info ? " info" : " no-info");
                ASSERT_NO_THROW(((void)gesv<B, T>(*this->ctx, p.A(), p.B(), p.piv.to_span(),
                                                  Span<std::byte>(arena.data(), bytes),
                                                  pass_info ? info.to_span() : Span<int32_t>{}),
                                 this->ctx->wait()))
                    << what;
                for (std::size_t i = 0; i < kGuard; ++i)
                    ASSERT_EQ(static_cast<unsigned>(arena[bytes + i]), 0xA5u)
                        << what << ": wrote " << i << " bytes past the sized workspace";
                expect_solved(p, pass_info ? std::vector<int32_t>(info.begin(), info.end())
                                           : std::vector<int32_t>(p.batch, 0),
                              what);
            }
        }
    }
}

// R6: a spelling that names nothing compiled throws instead of meaning Auto.
TYPED_TEST(GesvCandidates, UnknownPinsThrow) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    Matrix<T, MatrixFormat::Dense> A(8, 8, 2), Bm(8, 2, 2);
    for (const char* word : {"bogus", "tiny:1", "cta", "native:cta", "composed", "native:tiny:4",
                             // removed aliases (phase 5): each must stay an error
                             "native:tiny", "native:blocked"}) {
        const Pin pin("gesv", std::string_view(word));
        EXPECT_THROW(((void)gesv_buffer_size<B, T>(*this->ctx, A.view(), Bm.view())), std::invalid_argument) << word;
    }
    for (const C& c : gs::candidates<T>()) {
        const Pin pin("gesv", c);
        EXPECT_NO_THROW(((void)gesv_buffer_size<B, T>(*this->ctx, A.view(), Bm.view()))) << select::to_string(c);
    }
}

// The named can_run-false cases, each with its message.
TYPED_TEST(GesvCandidates, CanRunFalsePinsThrow) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    const int tiny = this->tiny_n();
    for (auto [n, nrhs] : {std::pair{tiny + 1, 1}, std::pair{8, 5}, std::pair{tiny + 1, 5}}) {
        Matrix<T, MatrixFormat::Dense> A(n, n, 1), Bm(n, nrhs, 1);
        const Pin pin("gesv", C{gs::Tiny{}});
        try {
            (void)gesv_buffer_size<B, T>(*this->ctx, A.view(), Bm.view());
            ADD_FAILURE() << this->name(gs::Tiny{}, n, nrhs) << " was accepted";
        } catch (const std::invalid_argument& e) {
            EXPECT_NE(std::string(e.what()).find("cannot run this shape"), std::string::npos)
                << this->name(gs::Tiny{}, n, nrhs) << ": " << e.what();
        }
    }
}

// §5.3: spellings (case-folded) and the class words select their choice, via ScopedPin and via the environment. gesv
// has no vendor family, so bare `vendor` warns and runs Auto; bare `native` is Auto's pick.
// float n=16 is inside every transcribed window except double/cdouble's, so Auto differs by type.
TYPED_TEST(GesvCandidates, ClassWordsAndSpellings) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_GESV_ROUTE", nullptr);
    std::string auto_pick;
    {
        auto p = make_sys<T>(16, 2, 4, 5u);
        auto_pick = traced_choice([&] { (void)this->run_auto(p); });
    }
    const bool window = std::is_same_v<T, float> || std::is_same_v<T, std::complex<float>>;
    ASSERT_EQ(auto_pick, window ? "tiny" : "blocked") << "the transcribed tables at n=16 nrhs=2";
    const std::pair<const char*, const char*> expect[] = {
        {"BLOCKED", "blocked"}, {"tiny", "tiny"},
        {"Blocked", "blocked"},  {"vendor", auto_pick.c_str()}, {"native", auto_pick.c_str()},
        {"auto", auto_pick.c_str()}};
    for (const auto& [word, spelling] : expect) {
        for (bool via_env : {false, true}) {
            auto p = make_sys<T>(16, 2, 4, 5u);
            std::vector<int32_t> info;
            select::testing::reset_warnings();
            std::string err;
            const std::string got = traced_choice([&] {
                const ScopedEnvVar env("BATCHLAS_GESV_ROUTE", via_env ? word : nullptr);
                std::optional<Pin> pin;
                if (!via_env) pin.emplace("gesv", std::string_view(word));
                info = this->run_auto(p);
            }, &err);
            const std::string what = std::string(word) + (via_env ? " via BATCHLAS_GESV_ROUTE" : " via ScopedPin");
            EXPECT_EQ(got, spelling) << what;
            const bool warns = std::string(word) == "vendor";
            EXPECT_EQ(err.find("gesv pinned \"vendor\", but no vendor candidate") != std::string::npos, warns)
                << what << ": " << err;
            expect_solved(p, info, what);
        }
    }
}

// §5.3: a ScopedPin wins over BATCHLAS_GESV_ROUTE, and nested pins restore the outer one.
TYPED_TEST(GesvCandidates, ScopedPinBeatsTheEnvironment) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    const int n = this->tiny_n() + 1;
    const ScopedEnvVar env("BATCHLAS_GESV_ROUTE", "tiny");
    {
        Matrix<T, MatrixFormat::Dense> A(n, n, 1), Bm(n, 1, 1);
        EXPECT_THROW(((void)gesv_buffer_size<B, T>(*this->ctx, A.view(), Bm.view())), std::invalid_argument)
            << "the environment pin was not read, so nothing below proves it lost";
    }
    std::vector<int32_t> info;
    auto p = make_sys<T>(n, 2, 2, 31u);
    EXPECT_EQ(traced_choice([&] {
                  const Pin pin("gesv", C{gs::Blocked{}});
                  info = this->run_auto(p);
              }),
              "blocked");
    expect_solved(p, info, "blocked over env tiny");
    auto q = make_sys<T>(n, 2, 2, 32u);
    EXPECT_EQ(traced_choice([&] {
                  const Pin pin("gesv", "auto");
                  info = this->run_auto(q);
              }),
              "blocked");
    expect_solved(q, info, "auto over env tiny");

    const int m = 8;
    const Pin outer("gesv", C{gs::Blocked{}});
    {
        const Pin inner("gesv", C{gs::Tiny{}});
        auto r = make_sys<T>(m, 2, 2, 33u);
        EXPECT_EQ(traced_choice([&] { (void)this->run_auto(r); }), "tiny");
    }
    auto s = make_sys<T>(m, 2, 2, 34u);
    EXPECT_EQ(traced_choice([&] { (void)this->run_auto(s); }), "blocked") << "the inner pin did not restore the outer";
}

// Auto against the transcribed table this device reads (sm_89 or sm_120, or borrowed): rows
// straddling the old window (float 32|33, cfloat 16|17) and the tiny nrhs ceiling (4|5), where
// both answers can run unless can_run says otherwise. Absorbs gesv_tests' old G7.
TYPED_TEST(GesvCandidates, AutoReadsTheTranscribedTables) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    const ScopedEnvVar clear("BATCHLAS_GESV_ROUTE", nullptr);
    const std::string dtype(select::dtype_name<T>());
    const auto tables = select::tables_in_borrow_order("gesv", dtype, select::device_of<B>(*this->ctx));
    if (tables.empty() || (tables.front()->device != "sm_89" && tables.front()->device != "sm_120"))
        GTEST_SKIP() << "this device reads no transcribed gesv table";
    struct Row { const char* dtype; int n, nrhs; const char* expect; };
    const Row rows[] = {{"float", 4, 1, "tiny"},      {"float", 32, 4, "tiny"},     {"float", 33, 1, "blocked"},
                        {"float", 25, 3, "tiny"},     {"float", 8, 5, "blocked"},   {"float", 64, 1, "blocked"},
                        {"cfloat", 16, 4, "tiny"},    {"cfloat", 17, 1, "blocked"}, {"cfloat", 9, 2, "tiny"},
                        {"cfloat", 32, 1, "blocked"}, {"double", 4, 1, "blocked"},  {"double", 32, 2, "blocked"},
                        {"cdouble", 8, 1, "blocked"}, {"cdouble", 16, 4, "blocked"}};
    int checked = 0;
    for (const Row& r : rows) {
        if (dtype != r.dtype) continue;
        auto p = make_sys<T>(r.n, r.nrhs, 512, 41u);
        std::vector<int32_t> info;
        const std::string what = "n=" + std::to_string(r.n) + " nrhs=" + std::to_string(r.nrhs);
        try {
            EXPECT_EQ(traced_choice([&] { info = this->run_auto(p); }), r.expect) << what;
            expect_solved(p, info, what);
        } catch (const batchlas::NoRouteError& e) {
            if (!this->child_may_lack_route()) ADD_FAILURE() << what << ": " << e.what();
        }
        ++checked;
    }
    EXPECT_GT(checked, 0) << dtype;
}

// key_of's every field reaches choose(): a synthetic table for this device whose winner changes
// with each of n and nrhs alone. Fixing either field in key_of turns exactly its row red.
TYPED_TEST(GesvCandidates, AutoReadsEveryKeyField) {
    using T = typename TestFixture::T;
    const ScopedEnvVar clear("BATCHLAS_GESV_ROUTE", nullptr);
    const std::string dtype(select::dtype_name<T>());
    const std::string dev = select::device_of<TestFixture::B>(*this->ctx).key;
    std::vector<std::pair<std::string, std::string>> files;
    for (const auto& t : select::embedded_tables())
        if (t.name.rfind("gesv.", 0) != 0) files.emplace_back(std::string(t.name), std::string(t.text));
    files.emplace_back("gesv." + dtype + "." + dev + ".txt",
                       "# op=gesv dtype=" + dtype + " device=" + dev + " kernels=unknown\n"
                       "# keys: n:log:3 nrhs:log\n"
                       "n=8 nrhs=1 | tiny 1 | blocked 2\n"
                       "n=8 nrhs=4 | blocked 1 | tiny 2\n"
                       "n=12 nrhs=1 | blocked 1 | tiny 2\n");
    const TableGuard restore;
    select::testing::set_builtin_tables(std::move(files));
    struct Probe { int n, nrhs; const char* expect; const char* field; };
    const Probe probes[] = {{8, 1, "tiny", "base"}, {8, 4, "blocked", "nrhs"}, {12, 1, "blocked", "n"}};
    for (const auto& k : probes) {
        auto p = make_sys<T>(k.n, k.nrhs, 128, 43u);
        std::vector<int32_t> info;
        EXPECT_EQ(traced_choice([&] { info = this->run_auto(p); }), k.expect) << "the " << k.field << " row";
        expect_solved(p, info, std::string("the ") + k.field + " row");
    }
}

// The coverage row's native flags come from the candidate list and can_run (§5.6): every gesv
// shape has a runnable native candidate (Blocked), so existed = supported = 1. The row carries
// the gesv key: m = order, n = nrhs.
// threadsafe: the child re-executes the binary, so CUDA is initialised fresh, never forked.
TYPED_TEST(GesvCandidates, CoverageRowCarriesNativeFlags) {
    using T = typename TestFixture::T;
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    const std::string dir = ::testing::TempDir() + "gesv_cov." + std::string(select::dtype_name<T>());
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const std::string out = dir + "/cov";
    auto child = [&] {
        const ScopedEnvVar cov("BATCHLAS_COVERAGE_OUT", out.c_str());
        const ScopedEnvVar clear("BATCHLAS_GESV_ROUTE", nullptr);
        batchlas::coverage::g_dynamic_enabled = true;
        auto lo = make_sys<T>(8, 2, 2, 51u);
        {
            const Pin pin("gesv", C{gs::Tiny{}});
            (void)this->run_auto(lo);
        }
        auto hi = make_sys<T>(8, 9, 2, 52u);
        (void)this->run_auto(hi);
        std::exit(0);
    };
    EXPECT_EXIT(child(), ::testing::ExitedWithCode(0), "");
    std::map<std::string, std::vector<std::string>> rows;  // "m n" -> fields
    for (const auto& ent : std::filesystem::directory_iterator(dir)) {
        std::ifstream in(ent.path());
        for (std::string line; std::getline(in, line);) {
            if (line.rfind("reached,gesv,", 0) != 0) continue;
            std::vector<std::string> f;
            std::stringstream ss(line);
            for (std::string tok; std::getline(ss, tok, ',');) f.push_back(tok);
            ASSERT_GE(f.size(), 14u) << line;
            rows[f[5] + " " + f[6]] = f;
        }
    }
    std::filesystem::remove_all(dir);
    ASSERT_EQ(rows.size(), 2u);
    ASSERT_TRUE(rows.count("8 2")) << "no n=8 nrhs=2 row";
    ASSERT_TRUE(rows.count("8 9")) << "no n=8 nrhs=9 row";
    for (const auto& key : {"8 2", "8 9"}) {
        EXPECT_EQ(rows[key][12], "1") << key;
        EXPECT_EQ(rows[key][13], "1") << key;
    }
    EXPECT_EQ(rows["8 2"][10], "tiny");
    EXPECT_EQ(rows["8 9"][10], "blocked");
}

// An empty problem has no kernel: an explicit throw before choose(), never a silent no-op.
TYPED_TEST(GesvCandidates, EmptyShapesThrowBeforeChoose) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::B;
    for (auto [n, nrhs, batch] : {std::tuple{0, 1, 1}, std::tuple{4, 0, 1}, std::tuple{4, 1, 0}}) {
        Matrix<T, MatrixFormat::Dense> A(n, n, batch), Bm(n, nrhs, batch);
        const Pin pin("gesv", C{gs::Blocked{}});
        EXPECT_THROW(((void)gesv_buffer_size<B, T>(*this->ctx, A.view(), Bm.view())), batchlas::internal_error)
            << n << " " << nrhs << " " << batch;
    }
}

// The transcribed rows of both devices, read with Table::nearest directly so every device checks
// them: the old window (float n <= 32, cfloat n <= 16, none for double/cdouble) as the first
// entry, then blocked. Off-grid probes (25, 30, 36, 3000) included.
TEST(GesvTranscribedTable, RowsHoldTheOldWindowOnBothDevices) {
    struct Row { const char* dtype; int n, nrhs; const char* first; std::size_t len; };
    const Row rows[] = {
        {"float", 1, 1, "tiny", 2},        {"float", 32, 64, "tiny", 2},   {"float", 33, 1, "blocked", 1},
        {"float", 30, 3, "tiny", 2},       {"float", 36, 3, "blocked", 1}, {"cfloat", 16, 4, "tiny", 2},
        {"cfloat", 17, 1, "blocked", 1},   {"cfloat", 25, 2, "blocked", 1}, {"double", 1, 1, "blocked", 1},
        {"cdouble", 16, 4, "blocked", 1},  {"double", 3000, 300, "blocked", 1},
    };
    for (const char* dev : {"sm_89", "sm_120"}) {
        for (const Row& r : rows) {
            const auto tables = select::tables_in_borrow_order("gesv", r.dtype, select::device_from_key(dev));
            ASSERT_FALSE(tables.empty()) << r.dtype;
            const select::Table& t = *tables.front();
            ASSERT_EQ(t.device, dev) << r.dtype;
            EXPECT_EQ(t.source.rfind("transcribed:", 0), 0u) << t.file;
            const select::TableRow* row = t.nearest(select::Key{{"n", r.n}, {"nrhs", r.nrhs}});
            ASSERT_NE(row, nullptr) << t.file;
            const std::string what = t.file + ":" + std::to_string(row->line) + " n=" + std::to_string(r.n);
            ASSERT_EQ(row->ranked.size(), r.len) << what;
            EXPECT_EQ(row->ranked.front().spelling, r.first) << what;
            EXPECT_EQ(row->ranked.back().spelling, "blocked") << what;
            EXPECT_FALSE(row->timed) << what;
        }
    }
}

#if BATCHLAS_HAS_HOST_BACKEND
// The typed suite skips NETLIB (CPU queue). Tiny refuses the NETLIB backend even on a GPU device:
// netlib getrf writes true int64 pivots where the kernel packs int32. Blocked solves on the CPU.
TEST(GesvNetlib, TinyRefusedBlockedSolves) {
    using T = float;
    constexpr Backend B = Backend::NETLIB;
    for (const char* dev : {"gpu", "cpu"}) {
        std::unique_ptr<Queue> q;
        try {
            q = std::make_unique<Queue>(Device(dev), B, true);
        } catch (const std::exception&) {
            continue;
        }
        if (std::string(dev) == "gpu" &&
            (q->device().type != DeviceType::GPU || !q->device().supports_sub_group_size(32)))
            continue;
        Matrix<T, MatrixFormat::Dense> A(8, 8, 2), Bm(8, 2, 2);
        const Pin tiny("gesv", C{gs::Tiny{}});
        try {
            (void)gesv_buffer_size<B, T>(*q, A.view(), Bm.view());
            ADD_FAILURE() << dev << ": a Tiny pin was accepted on NETLIB";
        } catch (const std::invalid_argument& e) {
            EXPECT_NE(std::string(e.what()).find("cannot run this shape"), std::string::npos) << dev << e.what();
        }
        // Blocked's children need netlib LAPACKE, absent from a vendor-free build.
        if (std::string(dev) != "cpu" || !batchlas::select::factorization_vendor_available<B>) continue;
#if BATCHLAS_HAS_CPU_TARGET
        auto p = make_sys<T>(8, 2, 3, 71u);
        std::vector<int32_t> info(p.batch, -7);
        const std::string got = traced_choice([&] {
            const Pin automatic("gesv", "auto");
            const std::size_t bytes = gesv_buffer_size<B, T>(*q, p.A(), p.B());
            UnifiedVector<std::byte> ws(std::max<std::size_t>(bytes, 1));
            UnifiedVector<int32_t> inf(p.batch, int32_t(-7));
            (void)gesv<B, T>(*q, p.A(), p.B(), p.piv.to_span(), Span<std::byte>(ws.data(), bytes), inf.to_span());
            q->wait();
            info.assign(inf.begin(), inf.end());
        });
        EXPECT_EQ(got, "blocked");
        for (int it = 0; it < p.batch; ++it) ASSERT_EQ(info[it], 0) << "item " << it;
        for (int it : {0, p.batch - 1}) EXPECT_LE(residual(p, it), tol<T>(p.n)) << "item " << it;
#endif
    }
}
#endif

}  // namespace
