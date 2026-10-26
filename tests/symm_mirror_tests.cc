// SPDX-License-Identifier: MIT
// The symm / hemm `mirror` family (src/ops/symm/kernels/symm_mirror.cc), pinned, on every scalar:
// the symmetric operand is read from its named triangle only, inside wide v2's tile loader (the
// tiled kernel for double). A, B and C each sit in a buffer of large finite poison at padded ld and
// batch stride; A's unnamed triangle holds poison the kernel would accept, and for hemm the diagonal
// carries an imaginary part. The reference is the host mirror of the named triangle; which kernel
// ran is read from the kernel trace and the coverage row. Deliberate breaks: docs/perf/level3.md.
#include <gtest/gtest.h>

#include <batchlas/blas/functions/hemm.hh>
#include <batchlas/blas/functions/symm.hh>
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/verify/tolerance.hh>

#include <src/ops/hemm/choice.hh>
#include <src/ops/symm/choice.hh>
#include <src/ops/symm/mirror_tile.hh>
#include <src/select/coverage.hh>

#include <tests/test_utils.hh>

#include <complex>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

using namespace batchlas;

namespace {

template <typename T>
using MV = MatrixView<T, MatrixFormat::Dense>;
template <typename T>
inline constexpr bool kComplex = !std::is_floating_point_v<T>;

template <typename T>
T val(double re, double im) {
    if constexpr (kComplex<T>) return T(re, im);
    else return T(re);
}
template <typename T>
T conj_of(T x) {
    if constexpr (kComplex<T>) return std::conj(x);
    else return x;
}
template <typename T>
bool same_bits(const T& a, const T& b) {
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}

struct Spec {
    Side side = Side::Left;
    Uplo uplo = Uplo::Lower;
    int m = 8, n = 5, batch = 3;
    bool herm = false;
    int pad = 3, gap = 5;  // ld = rows + pad, stride = ld cols + gap; 0, 0 is the packed (vector) leg
    int period = 0;        // > 0: item it repeats item it % period
    unsigned seed = 1;
    int k() const { return side == Side::Left ? m : n; }
};

std::string label(const Spec& s) {
    return std::string(s.herm ? "hemm " : "symm ") + (s.side == Side::Left ? "L" : "R") +
           (s.uplo == Uplo::Lower ? "L" : "U") + " m=" + std::to_string(s.m) + " n=" + std::to_string(s.n) +
           " batch=" + std::to_string(s.batch) + " pad=" + std::to_string(s.pad);
}

template <typename T>
struct Operand {
    int rows = 0, cols = 0, batch = 0, ld = 0, stride = 0;
    UnifiedVector<T> mem;
    std::vector<T> mem0;
    Operand(int r, int c, int b, int pad, int gap)
        : rows(r), cols(c), batch(b), ld(r + pad), stride((r + pad) * c + gap),
          mem(std::size_t(stride) * std::size_t(b) + 16) {
        for (std::size_t e = 0; e < mem.size(); ++e) mem[e] = val<T>(3000.0 + double(e % 89), -1500.0 - double(e % 7));
    }
    T& at(int it, int i, int j) { return mem[std::size_t(it) * stride + std::size_t(j) * ld + i]; }
    MV<T> view() { return MV<T>(mem.data(), rows, cols, ld, stride, batch); }
    void snapshot() { mem0.assign(mem.begin(), mem.end()); }
};

template <typename T>
struct Prob {
    Spec s;
    T alpha, beta;
    Operand<T> A, B, C;
    Matrix<T, MatrixFormat::Dense> Afull, B0, C0;
    explicit Prob(const Spec& sp, bool beta_zero = false)
        : s(sp), alpha(val<T>(1.25, -0.5)), beta(beta_zero ? T(0) : val<T>(-0.75, 0.25)),
          A(sp.k(), sp.k(), sp.batch, sp.pad, sp.gap), B(sp.m, sp.n, sp.batch, sp.pad, sp.gap),
          C(sp.m, sp.n, sp.batch, sp.pad + 2, sp.gap), Afull(sp.k(), sp.k(), sp.batch), B0(sp.m, sp.n, sp.batch),
          C0(sp.m, sp.n, sp.batch) {
        std::mt19937 gen(sp.seed);
        std::uniform_real_distribution<double> u(-1, 1);
        const int k = sp.k(), reps = sp.period > 0 ? std::min(sp.period, sp.batch) : sp.batch;
        auto named = [&](int i, int j) { return sp.uplo == Uplo::Lower ? i >= j : i <= j; };
        for (int it = 0; it < sp.batch; ++it) {
            const int r = it % reps;
            for (int j = 0; j < k; ++j)
                for (int i = 0; i < k; ++i) {
                    if (!named(i, j)) continue;  // the poison stays: in range, large, finite
                    A.at(it, i, j) = it < reps ? val<T>(u(gen), u(gen)) : A.at(r, i, j);
                }
            for (int j = 0; j < sp.n; ++j)
                for (int i = 0; i < sp.m; ++i) {
                    B.at(it, i, j) = it < reps ? val<T>(u(gen), u(gen)) : B.at(r, i, j);
                    C.at(it, i, j) = beta_zero ? val<T>(std::numeric_limits<double>::quiet_NaN(), 1.0)
                                               : (it < reps ? val<T>(u(gen), u(gen)) : C.at(r, i, j));
                }
        }
        // The reference: the named triangle mirrored on the host (conjugated, real diagonal for hemm).
        for (int it = 0; it < sp.batch; ++it) {
            for (int j = 0; j < k; ++j)
                for (int i = 0; i < k; ++i) {
                    T a = named(i, j) ? A.at(it, i, j) : A.at(it, j, i);
                    if (sp.herm && !named(i, j)) a = conj_of(a);
                    if constexpr (kComplex<T>)
                        if (sp.herm && i == j) a = T(a.real(), 0);
                    Afull(i, j, it) = a;
                }
            for (int j = 0; j < sp.n; ++j)
                for (int i = 0; i < sp.m; ++i) B0(i, j, it) = B.at(it, i, j), C0(i, j, it) = C.at(it, i, j);
        }
        A.snapshot(), B.snapshot(), C.snapshot();
    }
};

template <typename T>
void expect_product(Queue& q, Prob<T>& p, const std::string& what) {
    const Spec& s = p.s;
    std::vector<int> items;
    if (s.period > 0 || s.batch <= 6) {
        for (int it = 0; it < (s.period > 0 ? std::min(s.period, s.batch) : s.batch); ++it) items.push_back(it);
    } else {
        items = {0, 1, s.batch / 2, s.batch - 1};
    }
    const MV<T> C1 = p.C.view();
    const double err = s.side == Side::Left
        ? MV<T>::gemm_backward_error(q, p.Afull, Transpose::NoTrans, p.B0, Transpose::NoTrans, p.C0, C1,
                                     MV<T>::up(p.alpha), MV<T>::up(p.beta), items)
        : MV<T>::gemm_backward_error(q, p.B0, Transpose::NoTrans, p.Afull, Transpose::NoTrans, p.C0, C1,
                                     MV<T>::up(p.alpha), MV<T>::up(p.beta), items);
    EXPECT_VERIFY(T, verify::Check::blas, s.k(), err) << what;
    for (std::size_t e = 0; e < p.A.mem.size(); ++e)
        ASSERT_TRUE(same_bits(p.A.mem[e], p.A.mem0[e])) << what << ": wrote A at " << e;
    for (std::size_t e = 0; e < p.B.mem.size(); ++e)
        ASSERT_TRUE(same_bits(p.B.mem[e], p.B.mem0[e])) << what << ": wrote B at " << e;
    std::vector<char> inC(p.C.mem.size(), 0);
    for (int it = 0; it < s.batch; ++it)
        for (int j = 0; j < s.n; ++j)
            for (int i = 0; i < s.m; ++i) inC[std::size_t(it) * p.C.stride + std::size_t(j) * p.C.ld + i] = 1;
    for (std::size_t e = 0; e < p.C.mem.size(); ++e)
        if (!inC[e]) ASSERT_TRUE(same_bits(p.C.mem[e], p.C.mem0[e])) << what << ": wrote outside C at " << e;
    if (s.period > 0)
        for (int it = s.period; it < s.batch; ++it)
            for (int j = 0; j < s.n; ++j)
                for (int i = 0; i < s.m; ++i)
                    ASSERT_TRUE(same_bits(p.C.at(it, i, j), p.C.at(it % s.period, i, j)))
                        << what << ": item " << it << " differs from item " << it % s.period << " at (" << i << ","
                        << j << ")";
}

template <typename T, Backend B>
struct Cfg {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};
using Types = typename test_utils::backend_types_filtered<Cfg, true>::type;

template <typename Config>
class SymmMirror : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    static constexpr Backend B = Config::BackendVal;

    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        if (this->HasFatalFailure() || ::testing::Test::IsSkipped()) return;
        if (!this->ctx) GTEST_SKIP() << "no queue";
        if (this->ctx->device().type != DeviceType::GPU) GTEST_SKIP() << "mirror is a GPU kernel";
    }
    // symm, and hemm for a complex scalar, pinned to mirror.
    std::vector<bool> ops() const { return kComplex<T> ? std::vector<bool>{false, true} : std::vector<bool>{false}; }
    void run(Prob<T>& p) {
        Queue& q = *this->ctx;
        if (p.s.herm) {
            if constexpr (kComplex<T>) {
                const select::ScopedPin<ops::hemm::HemmChoice> pin("hemm", ops::hemm::Mirror{});
                (void)hemm<B, T>(q, p.A.view(), p.B.view(), p.C.view(), p.alpha, p.beta, p.s.side, p.s.uplo);
            }
        } else {
            const select::ScopedPin<ops::symm::SymmChoice> pin("symm", ops::symm::Mirror{});
            (void)symm<B, T>(q, p.A.view(), p.B.view(), p.C.view(), p.alpha, p.beta, p.s.side, p.s.uplo);
        }
        q.wait();
    }
    void check(Spec s, const std::string& tag = "", bool beta_zero = false) {
        Prob<T> p(s, beta_zero);
        run(p);
        expect_product(*this->ctx, p, tag + label(s));
    }
};

TYPED_TEST_SUITE(SymmMirror, Types);

// Every side and triangle on extents straddling the 64-wide tile (63/64/65, 128/130) and the
// 16-deep slab (15/16/17), so each staging mode runs: a slab inside the named triangle, inside
// the other one, and across the diagonal, at the predicated edge and in the interior.
TYPED_TEST(SymmMirror, NamedTriangleOnlyOnEveryShape) {
    for (bool herm : this->ops())
        for (Side side : {Side::Left, Side::Right})
            for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
                for (auto [m, n] : {std::pair{1, 1}, std::pair{15, 7}, std::pair{16, 33}, std::pair{17, 64},
                                    std::pair{63, 65}, std::pair{64, 5}, std::pair{65, 32}, std::pair{130, 70},
                                    std::pair{200, 9}}) {
                    Spec s{side, uplo, side == Side::Left ? m : n, side == Side::Left ? n : m, 3, herm};
                    s.seed = 11u + unsigned(m) * 3u + unsigned(n);
                    this->check(s);
                    if (::testing::Test::HasFatalFailure()) return;
                }
}

// Both global-load legs of wide v2: packed 16-byte operands take the vector leg, an odd ld the
// element leg; the kernel trace names the leg (double: the tiled kernel), once per launch.
TYPED_TEST(SymmMirror, BothLoadLegsAreReachedAndRight) {
    using T = typename TestFixture::T;
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    const std::string path = ::testing::TempDir() + "symm_mirror_ktrace." + std::string(select::dtype_name<T>());
    std::filesystem::remove(path);
    const ScopedEnvVar on("BATCHLAS_KERNEL_TRACE", "1");
    const ScopedEnvVar where("BATCHLAS_KERNEL_TRACE_PATH", path.c_str());
    const std::vector<Spec> specs{Spec{Side::Left, Uplo::Lower, 128, 32, 4, false, 0, 0},
                                  Spec{Side::Right, Uplo::Upper, 64, 192, 4, false, 0, 0},
                                  Spec{Side::Left, Uplo::Upper, 129, 31, 4, false, 1, 1},
                                  Spec{Side::Right, Uplo::Lower, 33, 130, 4, false, 1, 1}};
    auto child = [&] {
        for (bool herm : this->ops())
            for (Spec s : specs) {
                s.herm = herm;
                this->check(s);
                if (::testing::Test::HasFailure()) std::exit(1);
            }
        std::exit(0);
    };
    EXPECT_EXIT(child(), ::testing::ExitedWithCode(0), "");
    std::map<std::string, int> legs;
    std::ifstream in(path);
    for (std::string line; std::getline(in, line);)
        for (const char* name : {"symm.mirror.vec", "symm.mirror.elem", "symm.mirror.tiled", "hemm.mirror.vec",
                                 "hemm.mirror.elem"})
            if (line.find(std::string("\"name\":\"") + name + "\"") != std::string::npos) ++legs[name];
    std::filesystem::remove(path);
    if constexpr (std::is_same_v<T, double>) {
        EXPECT_EQ(legs["symm.mirror.tiled"], 4);
        EXPECT_EQ(legs.size(), 1u);
    } else {
        // A complex<double> element is one 16-byte granule: every ld is whole granules, so its views
        // always take the vector leg.
        const int vec = sizeof(T) == 16 ? 4 : 2, elem = 4 - vec;
        EXPECT_EQ(legs["symm.mirror.vec"], vec);
        EXPECT_EQ(legs["symm.mirror.elem"], elem);
        EXPECT_EQ(legs["hemm.mirror.vec"], kComplex<T> ? vec : 0);
        EXPECT_EQ(legs["hemm.mirror.elem"], kComplex<T> ? elem : 0);
        EXPECT_EQ(legs.count("symm.mirror.tiled"), 0u);
    }
}

// beta = 0 never reads C: C holds NaN, on both legs, both sides.
TYPED_TEST(SymmMirror, BetaZeroNeverReadsC) {
    for (bool herm : this->ops())
        for (int pad : {0, 1})
            for (Side side : {Side::Left, Side::Right}) {
                Spec s{side, Uplo::Lower, 96, 64, 3, herm, pad, pad};
                this->check(s, "beta 0 ", true);
                if (::testing::Test::HasFatalFailure()) return;
            }
}

// Shared-memory staging at saturation: 1024 items of one problem, every item bit-identical to the
// first, which is right; packed and padded, a tile edge and a diagonal slab in each.
TYPED_TEST(SymmMirror, SaturatingBatchIsBitIdentical) {
    for (bool herm : this->ops())
        for (Spec s : {Spec{Side::Left, Uplo::Lower, 130, 33, 1024, false, 0, 0},
                       Spec{Side::Right, Uplo::Upper, 40, 100, 1024, false, 3, 5}}) {
            s.herm = herm;
            s.period = 1;
            this->check(s, "saturating ");
            if (::testing::Test::HasFatalFailure()) return;
        }
}

// Grid z holds 65535 items: the pin runs at the ceiling and one past it (two launches), every
// item right. Grid y holds the row tiles: C with 65535 tiles of rows runs, one row more is refused.
TYPED_TEST(SymmMirror, LaunchAtTheGridCeilings) {
    using T = typename TestFixture::T;
    const int z = int(test_utils::kCudaGridYZ);
    for (bool herm : this->ops())
        for (int batch : {z, z + 1}) {
            Spec s{Side::Left, Uplo::Upper, 2, 3, batch, herm, 1, 1};
            s.period = 5;
            this->check(s, "grid z ");
            if (::testing::Test::HasFatalFailure()) return;
        }
    const std::int64_t ceiling = test_utils::expected_grid_yz(select::device_of<TestFixture::B>(*this->ctx).family);
    if (ceiling > test_utils::kCudaGridYZ) GTEST_SKIP() << "no grid-y cap on this device";
    const int rows = int(ceiling) * ops::mirror::tile<T>();
    for (int m : {rows, rows + 1}) {
        Spec s{Side::Right, Uplo::Lower, m, 1, 1, false, 0, 0};
        Prob<T> p(s);
        bool ran = true;
        try {
            this->run(p);
        } catch (const std::invalid_argument& e) {
            ran = false;
            EXPECT_NE(std::string(e.what()).find("cannot run this shape"), std::string::npos) << e.what();
        }
        EXPECT_EQ(ran, m == rows) << "m = " << m << " (" << ceiling << " row tiles of " << ops::mirror::tile<T>() << ")";
        if (ran) expect_product(*this->ctx, p, "grid y " + label(s));
    }
}

// The pin is reached (coverage `reached` rows): symm, and hemm for a complex scalar, chose mirror.
TYPED_TEST(SymmMirror, PinIsReachedPerCoverage) {
    using T = typename TestFixture::T;
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    const std::string dir = ::testing::TempDir() + "symm_mirror_cov." + std::string(select::dtype_name<T>());
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const std::string out = dir + "/cov";
    auto child = [&] {
        const ScopedEnvVar cov("BATCHLAS_COVERAGE_OUT", out.c_str());
        batchlas::coverage::g_dynamic_enabled = true;
        for (bool herm : this->ops()) {
            Prob<T> p(Spec{Side::Left, Uplo::Lower, 70, 9, 2, herm});
            this->run(p);
        }
        std::exit(0);
    };
    EXPECT_EXIT(child(), ::testing::ExitedWithCode(0), "");
    std::map<std::string, std::string> chosen;
    for (const auto& ent : std::filesystem::directory_iterator(dir)) {
        std::ifstream in(ent.path());
        for (std::string line; std::getline(in, line);) {
            if (line.rfind("reached,", 0) != 0) continue;
            std::vector<std::string> f;
            std::stringstream ss(line);
            for (std::string tok; std::getline(ss, tok, ',');) f.push_back(tok);
            ASSERT_GE(f.size(), 11u) << line;
            if (f[1] == "symm" || f[1] == "hemm") chosen[f[1]] = f[9] + " " + f[10];
        }
    }
    std::filesystem::remove_all(dir);
    EXPECT_EQ(chosen["symm"], "native mirror");
    if constexpr (kComplex<T>) EXPECT_EQ(chosen["hemm"], "native mirror");
}

}  // namespace
