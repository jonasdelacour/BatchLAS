// SPDX-License-Identifier: MIT
// Every hemm, herk and her2k candidate, and the complex-symmetric symm/syrk/syr2k ones, pinned (docs/design/flat-kernel-selection.md §8, §12).
// Operands sit in buffers of large finite poison with padded ld and batch stride, complex data has
// nonzero imaginary parts, alpha is complex where the op allows it and beta != 0. Which family ran
// is read back from the select trace, never assumed from the pin being accepted.
#include <gtest/gtest.h>

#include <batchlas/blas/functions/hemm.hh>
#include <batchlas/blas/functions/her2k.hh>
#include <batchlas/blas/functions/herk.hh>
#include <batchlas/blas/functions/symm.hh>
#include <batchlas/blas/functions/syr2k.hh>
#include <batchlas/blas/functions/syrk.hh>
#include <batchlas/no_route.hh>
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/verify/tolerance.hh>

#include <src/ops/hemm/choice.hh>
#include <src/ops/her2k/choice.hh>
#include <src/ops/herk/choice.hh>
#include <src/select/vendor.hh>

#include <tests/test_utils.hh>
#include <tests/usm_bulk_copy.hh>

#include <cctype>
#include <complex>
#include <cstring>
#include <functional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace batchlas;

namespace {

template <typename T>
using MV = MatrixView<T, MatrixFormat::Dense>;
namespace ub = test_utils::usm_bulk;

const char* uplo_s(Uplo u) { return u == Uplo::Lower ? "L" : "U"; }

// One operand in its own buffer of poison: ld = rows + pad, stride = ld * cols + gap. Items
// it >= period repeat item it % period (period 0: all distinct).
template <typename T>
struct Operand {
    int rows = 0, cols = 0, batch = 0, ld = 0, stride = 0;
    UnifiedVector<T> mem, mem0;
    Operand(int r, int c, int b, unsigned seed, bool natural = false, int period = 0)
        : rows(r), cols(c), batch(b), ld(r + (natural ? 0 : 3)), stride(ld * c + (natural ? 0 : 5)),
          mem(std::size_t(stride) * b + 7), mem0(1) {
        for (std::size_t e = 0; e < mem.size(); ++e) mem[e] = T(1000 + double(e % 97), -500 - double(e % 13));
        std::mt19937 gen(seed);
        std::uniform_real_distribution<double> u(-1, 1);
        const int reps = period > 0 ? std::min(period, b) : b;
        for (int it = 0; it < b; ++it)
            for (int j = 0; j < c; ++j)
                for (int i = 0; i < r; ++i)
                    at(it, i, j) = it < reps ? T(u(gen), u(gen)) : at(it % reps, i, j);
    }
    T& at(int it, int i, int j) { return mem[std::size_t(it) * stride + std::size_t(j) * ld + i]; }
    MV<T> view() { return MV<T>(mem.data(), rows, cols, ld, stride, batch); }
    MV<T> view0() { return MV<T>(mem0.data(), rows, cols, ld, stride, batch); }
    // Poison the triangle opposite `uplo` (the op must not read it); optionally a real diagonal.
    void poison_opposite(Uplo uplo, bool real_diagonal) {
        for (int it = 0; it < batch; ++it)
            for (int j = 0; j < cols; ++j)
                for (int i = 0; i < rows; ++i) {
                    if (i == j && real_diagonal) at(it, i, j) = T(at(it, i, j).real(), 0);
                    if (uplo == Uplo::Lower ? i < j : i > j) at(it, i, j) = T(4096 + i, -2048 - j);
                }
    }
    void snapshot() {
        mem0 = UnifiedVector<T>(mem.size());
        ub::copy(mem0.data(), mem.data(), mem.size() * sizeof(T), mem.data());
    }
    // Every element outside the `uplo` triangle of each item (padding and gaps included) is unchanged.
    ::testing::AssertionResult untouched_outside(Uplo uplo) const {
        std::vector<char> owned(mem.size(), 0);
        for (int it = 0; it < batch; ++it)
            for (int j = 0; j < cols; ++j)
                for (int i = 0; i < rows; ++i)
                    if (uplo == Uplo::Lower ? i >= j : i <= j) owned[std::size_t(it) * stride + std::size_t(j) * ld + i] = 1;
        const T* got = ub::download(mem.data(), mem.size(), 0);
        const T* want = ub::download(mem0.data(), mem.size(), 1);
        for (std::size_t e = 0; e < mem.size(); ++e)
            if (!owned[e] && std::memcmp(&got[e], &want[e], sizeof(T)) != 0)
                return ::testing::AssertionFailure() << "wrote element " << e << " outside its triangle";
        return ::testing::AssertionSuccess();
    }
    ::testing::AssertionResult diagonal_real() const {
        const T* got = ub::download(mem.data(), mem.size(), 0);
        for (int it = 0; it < batch; ++it)
            for (int d = 0; d < rows; ++d)
                if (got[std::size_t(it) * stride + std::size_t(d) * ld + d].imag() != 0)
                    return ::testing::AssertionFailure() << "item " << it << " diagonal " << d << " not real";
        return ::testing::AssertionSuccess();
    }
    ::testing::AssertionResult repeats(int period) const {  // item it bit-identical to item it % period
        const T* got = ub::download(mem.data(), mem.size(), 0);
        for (int it = period; it < batch; ++it)
            for (int j = 0; j < cols; ++j)
                for (int i = 0; i < rows; ++i) {
                    const std::size_t a = std::size_t(it) * stride + std::size_t(j) * ld + i;
                    const std::size_t b = std::size_t(it % period) * stride + std::size_t(j) * ld + i;
                    if (std::memcmp(&got[a], &got[b], sizeof(T)) != 0)
                        return ::testing::AssertionFailure() << "item " << it << " (" << i << "," << j << ")";
                }
        return ::testing::AssertionSuccess();
    }
};

// The outermost `op` trace line's choice for whatever `run` calls.
template <typename F>
std::string traced_choice(const std::string& op, F&& run) {
    const ScopedEnvVar trace("BATCHLAS_SELECT_TRACE", "1");
    ::testing::internal::CaptureStderr();
    try {
        run();
    } catch (...) {
        (void)::testing::internal::GetCapturedStderr();
        throw;
    }
    std::istringstream in(::testing::internal::GetCapturedStderr());
    for (std::string line; std::getline(in, line);)
        if (line.rfind(op + " ", 0) == 0 && line.find(" -> ") != std::string::npos) {
            const std::string tail = line.substr(line.find(" -> ") + 4);
            return tail.substr(0, tail.find(' '));
        }
    return "<no " + op + " trace line>";
}

template <typename T, Backend B>
struct Cfg {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};
using Types = typename test_utils::backend_types_complex<Cfg>::type;

template <typename Config>
class HermitianCandidates : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    using R = batchlas::float_t<T>;
    static constexpr Backend B = Config::BackendVal;
    static constexpr bool kVendor = select::level3_vendor_available<B> && B != Backend::ROCM;
    static constexpr bool kCfloat = std::is_same_v<T, std::complex<float>>;
    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        if (this->HasFatalFailure() || ::testing::Test::IsSkipped()) return;
        if (!this->ctx) GTEST_SKIP() << "no queue";
        if (this->ctx->device().type != DeviceType::GPU) GTEST_SKIP() << "the native families are GPU families";
        ub::attach(this->ctx.get());
    }
    void TearDown() override {
        ub::detach();
        test_utils::BatchLASTest<Config>::TearDown();
    }
    Queue& q() { return *this->ctx; }

    // herk on fresh operands; returns whether the pin was accepted, and checks the answer if so.
    bool herk_case(const std::string& pin, int n, int k, int batch, Transpose t, Uplo uplo, int period = 0,
                   bool natural = false) {
        const bool nt = t == Transpose::NoTrans;
        Operand<T> A(nt ? n : k, nt ? k : n, batch, 11u + n + k, natural, period);
        Operand<T> C(n, n, batch, 13u + n, natural, period);
        C.poison_opposite(uplo, true);
        C.snapshot();
        const R alpha = R(0.75), beta = R(-0.5);
        const std::string what = pin + " herk " + uplo_s(uplo) + (nt ? "N" : "C") + " n=" + std::to_string(n) +
                                 " k=" + std::to_string(k) + " batch=" + std::to_string(batch);
        std::string ran;
        try {
            const ScopedEnvVar env("BATCHLAS_HERK_ROUTE", pin.c_str());
            ran = traced_choice("herk", [&] { herk<B, T>(q(), A.view(), C.view(), alpha, beta, uplo, t).wait(); });
        } catch (const std::invalid_argument&) {
            return false;
        }
        if (pin != "vendor" || kVendor) EXPECT_EQ(ran, pin) << what;
        std::vector<int> items{0, batch - 1};
        EXPECT_VERIFY(T, verify::Check::blas, k,
                      MV<T>::gemm_backward_error(q(), A.view(), t, A.view(), nt ? Transpose::ConjTrans : Transpose::NoTrans,
                                                 C.view0(), C.view(), MV<T>::up(T(alpha)), MV<T>::up(T(beta)), items,
                                                 uplo))
            << what;
        EXPECT_TRUE(C.untouched_outside(uplo)) << what;
        EXPECT_TRUE(C.diagonal_real()) << what;
        // fold chunks past grid z, and each chunk's gemm picks its kernel by its own batch: only one
        // chunk is bit-for-bit (every item is still verified above).
        if (period > 0 && (pin != "fold" || batch <= 65535)) EXPECT_TRUE(C.repeats(period)) << what;
        return true;
    }

    bool her2k_case(const std::string& pin, int n, int k, int batch, Transpose t, Uplo uplo, int period = 0) {
        const bool nt = t == Transpose::NoTrans;
        Operand<T> A(nt ? n : k, nt ? k : n, batch, 17u + n, false, period);
        Operand<T> Bm(nt ? n : k, nt ? k : n, batch, 19u + k, false, period);
        Operand<T> C(n, n, batch, 23u + n, false, period);
        C.poison_opposite(uplo, false);
        C.snapshot();
        const T alpha = T(0.5, -1.25);
        const R beta = R(0.75);
        const std::string what = pin + " her2k " + uplo_s(uplo) + (nt ? "N" : "C") + " n=" + std::to_string(n) +
                                 " k=" + std::to_string(k) + " batch=" + std::to_string(batch);
        std::string ran;
        try {
            const ScopedEnvVar env("BATCHLAS_HER2K_ROUTE", pin.c_str());
            ran = traced_choice("her2k", [&] {
                her2k<B, T>(q(), A.view(), Bm.view(), C.view(), alpha, beta, uplo, t).wait();
            });
        } catch (const std::invalid_argument&) {
            return false;
        }
        if (pin != "vendor" || kVendor) EXPECT_EQ(ran, pin) << what;
        std::vector<int> items{0, batch - 1};
        EXPECT_VERIFY(T, verify::Check::blas, 2 * k,
                      MV<T>::rank2k_backward_error(q(), A.view(), Bm.view(), t, C.view0(), C.view(), uplo,
                                                   MV<T>::up(alpha), MV<T>::up(T(beta)), true, items))
            << what;
        EXPECT_TRUE(C.untouched_outside(uplo)) << what;
        EXPECT_TRUE(C.diagonal_real()) << what;
        // fold chunks past grid z, and each chunk's gemm picks its kernel by its own batch: only one
        // chunk is bit-for-bit (every item is still verified above).
        if (period > 0 && (pin != "fold" || batch <= 65535)) EXPECT_TRUE(C.repeats(period)) << what;
        return true;
    }

    bool hemm_case(const std::string& pin, int m, int n, int batch, Side side, Uplo uplo) {
        const int ka = side == Side::Left ? m : n;
        Operand<T> A(ka, ka, batch, 29u + m, false);
        Operand<T> Bm(m, n, batch, 31u + n, false);
        Operand<T> C(m, n, batch, 37u + m + n, false);
        A.poison_opposite(uplo, false);  // the diagonal keeps an imaginary part hemm must not read
        C.snapshot();
        const T alpha = T(1.25, 0.5), beta = T(-0.5, 0.25);
        const std::string what = pin + " hemm " + (side == Side::Left ? "L" : "R") + uplo_s(uplo) +
                                 " m=" + std::to_string(m) + " n=" + std::to_string(n) + " batch=" + std::to_string(batch);
        std::string ran;
        try {
            const ScopedEnvVar env("BATCHLAS_HEMM_ROUTE", pin.c_str());
            ran = traced_choice("hemm", [&] {
                hemm<B, T>(q(), A.view(), Bm.view(), C.view(), alpha, beta, side, uplo).wait();
            });
        } catch (const std::invalid_argument&) {
            return false;
        }
        if (pin != "vendor" || kVendor) EXPECT_EQ(ran, pin) << what;
        Matrix<T, MatrixFormat::Dense> Aref(ka, ka, batch);
        MV<T>::copy(q(), Aref.view(), A.view()).wait();
        Aref.hermitize(q(), uplo).wait();
        const Transpose N = Transpose::NoTrans;
        std::vector<int> items{0, batch - 1};
        const double err = side == Side::Left
                               ? MV<T>::gemm_backward_error(q(), Aref.view(), N, Bm.view(), N, C.view0(), C.view(),
                                                            MV<T>::up(alpha), MV<T>::up(beta), items)
                               : MV<T>::gemm_backward_error(q(), Bm.view(), N, Aref.view(), N, C.view0(), C.view(),
                                                            MV<T>::up(alpha), MV<T>::up(beta), items);
        EXPECT_VERIFY(T, verify::Check::blas, ka, err) << what;
        return true;
    }
};

TYPED_TEST_SUITE(HermitianCandidates, Types);

// herk: every family across gram's n = 128|129 edge, the 128-tile edges and k under, at and over a
// staging step, both triangles, N and C, padded operands, batch 1 and 3. A pin is accepted exactly
// where the family can run; an accepted one is the family that ran and is right.
TYPED_TEST(HermitianCandidates, HerkPinnedFamiliesStraddleTheirLimits) {
    for (const char* fam : {"gram", "triangular", "fold", "vendor"})
        for (int n : {1, 33, 128, 129, 200})
            for (int k : {1, 7, 9})
                for (int batch : {1, 3}) {
                    const Transpose t = (n + k) % 2 ? Transpose::NoTrans : Transpose::ConjTrans;
                    const Uplo uplo = batch == 1 ? Uplo::Lower : Uplo::Upper;
                    const std::string f = fam;
                    bool want = true;
                    if (f == "gram") want = n <= 128;
                    if (f == "triangular") want = TestFixture::kCfloat;
                    if (f == "vendor" && !TestFixture::kVendor) continue;  // the class word falls back to Auto
                    EXPECT_EQ(this->herk_case(f, n, k, batch, t, uplo), want) << f << " n=" << n << " k=" << k;
                }
}

TYPED_TEST(HermitianCandidates, Her2kPinnedFamiliesStraddleTheirLimits) {
    for (const char* fam : {"triangular", "fold", "vendor"})
        for (int n : {1, 127, 128, 129, 257})
            for (int k : {1, 7, 9})
                for (Transpose t : {Transpose::NoTrans, Transpose::ConjTrans}) {
                    const std::string f = fam;
                    if (f == "vendor" && !TestFixture::kVendor) continue;
                    const bool want = f != "triangular" || TestFixture::kCfloat;
                    const Uplo uplo = (n + k) % 2 ? Uplo::Lower : Uplo::Upper;
                    EXPECT_EQ(this->her2k_case(f, n, k, 2, t, uplo), want) << f << " n=" << n << " k=" << k;
                }
}

TYPED_TEST(HermitianCandidates, HemmPinnedFamiliesOnBothSides) {
    for (const char* fam : {"mirror", "expand", "vendor"})
        for (auto [m, n] : {std::pair{1, 1}, std::pair{7, 33}, std::pair{33, 7}, std::pair{130, 40}})
            for (Side side : {Side::Left, Side::Right})
                for (Uplo uplo : {Uplo::Lower, Uplo::Upper}) {
                    const std::string f = fam;
                    if (f == "vendor" && !TestFixture::kVendor) continue;
                    EXPECT_TRUE(this->hemm_case(f, m, n, 3, side, uplo)) << f;
                }
}

// The SLM tile kernels at a saturating batch: 1024 items repeating 3 problems, every item
// bit-identical to its representative (a cross-sub-group race shows only at wg > 32 and a full grid).
TYPED_TEST(HermitianCandidates, SaturatingBatchIsBitIdentical) {
    EXPECT_TRUE(this->herk_case("gram", 100, 40, 1024, Transpose::NoTrans, Uplo::Lower, 3));
    EXPECT_TRUE(this->herk_case("fold", 150, 24, 1024, Transpose::ConjTrans, Uplo::Upper, 3));
    EXPECT_TRUE(this->her2k_case("fold", 150, 24, 1024, Transpose::NoTrans, Uplo::Lower, 3));
    if constexpr (TestFixture::kCfloat) {
        EXPECT_TRUE(this->herk_case("triangular", 256, 64, 1024, Transpose::NoTrans, Uplo::Lower, 3, true));
        EXPECT_TRUE(this->herk_case("triangular", 200, 53, 1024, Transpose::ConjTrans, Uplo::Upper, 3));
        EXPECT_TRUE(this->her2k_case("triangular", 200, 53, 1024, Transpose::ConjTrans, Uplo::Upper, 3));
    }
}

// The scratch families chunk the batch to fit BATCHLAS_EXPAND_MAX_BYTES: a budget of a few items
// splits batch 11 into several launches, and every item (first, last) is right.
TYPED_TEST(HermitianCandidates, ScratchFamiliesChunkToTheBudget) {
    using T = typename TestFixture::T;
    const int n = 40;
    const std::string bytes = std::to_string(3 * n * n * sizeof(T) + 512);  // three items of n x n scratch
    const ScopedEnvVar budget("BATCHLAS_EXPAND_MAX_BYTES", bytes.c_str());
    EXPECT_TRUE(this->herk_case("fold", n, 9, 11, Transpose::NoTrans, Uplo::Lower));
    EXPECT_TRUE(this->her2k_case("fold", n, 9, 11, Transpose::ConjTrans, Uplo::Upper));
    EXPECT_TRUE(this->hemm_case("expand", n, 17, 11, Side::Left, Uplo::Lower));
    EXPECT_TRUE(this->hemm_case("expand", 17, n, 11, Side::Right, Uplo::Upper));
    // One item that does not fit is the only refusal.
    const ScopedEnvVar tiny("BATCHLAS_EXPAND_MAX_BYTES", "64");
    EXPECT_FALSE(this->herk_case("fold", n, 9, 2, Transpose::NoTrans, Uplo::Lower));
}

// Grid z holds 65535 items on CUDA: every native family chunks a batch one past it.
TYPED_TEST(HermitianCandidates, NativesChunkABatchPastTheGrid) {
    const int batch = 65536;
    EXPECT_TRUE(this->herk_case("gram", 8, 5, batch, Transpose::NoTrans, Uplo::Lower, 7));
    EXPECT_TRUE(this->herk_case("fold", 8, 5, batch, Transpose::ConjTrans, Uplo::Upper, 7));
    EXPECT_TRUE(this->her2k_case("fold", 8, 5, batch, Transpose::NoTrans, Uplo::Lower, 7));
    if constexpr (TestFixture::kCfloat) {
        EXPECT_TRUE(this->herk_case("triangular", 8, 5, batch, Transpose::NoTrans, Uplo::Upper, 7));
        EXPECT_TRUE(this->her2k_case("triangular", 8, 5, batch, Transpose::ConjTrans, Uplo::Lower, 7));
    }
}

// Auto on both sides of the transcribed rules (tuned/<op>.sm_89.txt, sm_120 likewise): with a vendor
// library herk takes fold iff batch >= 4 and n <= 768, her2k iff batch >= 2 or n >= 128, hemm expand
// iff batch >= 4 or max(m, n) >= 256; vendor-free every shape is native.
TYPED_TEST(HermitianCandidates, AutoFollowsTheTranscribedRules) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;
    const std::string key = select::device_of<TestFixture::B>(this->q()).key;
    if (key != "sm_89" && key != "sm_120") GTEST_SKIP() << "the transcribed tables are sm_89 and sm_120";
    const bool v = TestFixture::kVendor;
    auto herk_auto = [&](int n, int batch) {
        Operand<T> A(n, 16, batch, 3), C(n, n, batch, 5);
        return traced_choice("herk", [&] {
            herk<TestFixture::B, T>(this->q(), A.view(), C.view(), R(1), R(0), Uplo::Lower, Transpose::NoTrans).wait();
        });
    };
    const char* nat = TestFixture::kCfloat ? "triangular" : "fold";  // the first native entry of a vendor row
    EXPECT_EQ(herk_auto(64, 3), v ? "vendor" : nat);
    EXPECT_EQ(herk_auto(64, 4), "fold");
    EXPECT_EQ(herk_auto(768, 4), nat) << "cfloat ranks triangular first at 256 <= n <= 768";
    EXPECT_EQ(herk_auto(769, 4), v ? "vendor" : nat);
    auto her2k_auto = [&](int n, int batch) {
        Operand<T> A(n, 8, batch, 3), Bm(n, 8, batch, 4), C(n, n, batch, 5);
        return traced_choice("her2k", [&] {
            her2k<TestFixture::B, T>(this->q(), A.view(), Bm.view(), C.view(), T(1), R(0), Uplo::Lower,
                                     Transpose::NoTrans)
                .wait();
        });
    };
    EXPECT_EQ(her2k_auto(127, 1), v ? "vendor" : "fold");
    EXPECT_EQ(her2k_auto(128, 1), "fold");
    EXPECT_EQ(her2k_auto(64, 2), "fold");
    auto hemm_auto = [&](int m, int n, int batch) {
        Operand<T> A(m, m, batch, 3), Bm(m, n, batch, 4), C(m, n, batch, 5);
        return traced_choice("hemm", [&] {
            hemm<TestFixture::B, T>(this->q(), A.view(), Bm.view(), C.view(), T(1), T(0), Side::Left, Uplo::Lower)
                .wait();
        });
    };
    EXPECT_EQ(hemm_auto(255, 8, 3), v ? "vendor" : "expand");
    EXPECT_EQ(hemm_auto(256, 8, 3), "expand");
    EXPECT_EQ(hemm_auto(32, 32, 3), v ? "vendor" : "expand");
    EXPECT_EQ(hemm_auto(32, 32, 4), "expand");
}

// R6: a spelling that names nothing compiled, or a family that cannot run the shape, throws.
TYPED_TEST(HermitianCandidates, BadPinsThrow) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;
    Operand<T> A(200, 8, 2, 3), C(200, 200, 2, 5);
    auto run = [&](const char* pin) {
        const ScopedEnvVar env("BATCHLAS_HERK_ROUTE", pin);
        herk<TestFixture::B, T>(this->q(), A.view(), C.view(), R(1), R(0), Uplo::Lower, Transpose::NoTrans).wait();
    };
    EXPECT_THROW(run("cta"), std::invalid_argument);
    EXPECT_THROW(run("expand"), std::invalid_argument) << "hemm's family";
    EXPECT_THROW(run("gram"), std::invalid_argument) << "n = 200 > 128";
    EXPECT_THROW(run("fold:1"), std::invalid_argument);
    if constexpr (!TestFixture::kCfloat) EXPECT_THROW(run("triangular"), std::invalid_argument) << "cdouble";
    const ScopedEnvVar env("BATCHLAS_HEMM_ROUTE", "fold");
    Operand<T> H(8, 8, 1, 3), Bm(8, 8, 1, 4), Cm(8, 8, 1, 5);
    EXPECT_THROW((hemm<TestFixture::B, T>(this->q(), H.view(), Bm.view(), Cm.view(), T(1), T(0), Side::Left,
                                         Uplo::Lower)),
                 std::invalid_argument);
}

// ---- complex-symmetric (non-Hermitian) symm, syrk, syr2k: op(A) op(A)^T with no conjugate ----

// Runs `call` under BATCHLAS_<OP>_ROUTE=pin; false when the pin is refused, else the family the
// trace names in `ran`.
bool sym_pin(const std::string& op, const std::string& pin, const std::function<void()>& call, std::string* ran) {
    std::string var = "BATCHLAS_";
    for (char ch : op) var += char(std::toupper(static_cast<unsigned char>(ch)));
    var += "_ROUTE";
    try {
        const ScopedEnvVar env(var.c_str(), pin.c_str());
        *ran = traced_choice(op, call);
        return true;
    } catch (const std::invalid_argument&) {
        return false;
    }
}

// syrk: every family on padded operands, N and T, both triangles; ConjTrans is herk's and throws.
TYPED_TEST(HermitianCandidates, ComplexSymmetricSyrkFamilies) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::B;
    for (const char* fam : {"gram", "triangular", "vendor"})
        for (int n : {7, 128, 200})
            for (Transpose t : {Transpose::NoTrans, Transpose::Trans})
                for (int period : {0, 3}) {
                    const std::string f = fam;
                    if (f == "vendor" && !TestFixture::kVendor) continue;
                    const int k = 9, batch = period ? 1024 : 3;
                    const Uplo uplo = t == Transpose::NoTrans ? Uplo::Lower : Uplo::Upper;
                    const bool nt = t == Transpose::NoTrans;
                    Operand<T> A(nt ? n : k, nt ? k : n, batch, 41u + n, false, period);
                    Operand<T> C(n, n, batch, 43u + n, false, period);
                    C.poison_opposite(uplo, false);
                    C.snapshot();
                    const T alpha = T(0.5, 1.5), beta = T(-0.75, 0.25);
                    std::string ran;
                    const bool ok = sym_pin("syrk", f, [&] {
                        syrk<B, T>(this->q(), A.view(), C.view(), alpha, beta, uplo, t).wait();
                    }, &ran);
                    const bool want = (f != "gram" || n <= 128) && (f != "triangular" || TestFixture::kCfloat);
                    const std::string what = f + " syrk n=" + std::to_string(n) + (nt ? " N" : " T");
                    ASSERT_EQ(ok, want) << what;
                    if (!ok) continue;
                    EXPECT_EQ(ran, f) << what;
                    std::vector<int> items{0, batch - 1};
                    EXPECT_VERIFY(T, verify::Check::blas, k,
                                  MV<T>::gemm_backward_error(this->q(), A.view(), t, A.view(),
                                                             nt ? Transpose::Trans : Transpose::NoTrans, C.view0(),
                                                             C.view(), MV<T>::up(alpha), MV<T>::up(beta), items, uplo))
                        << what;
                    EXPECT_TRUE(C.untouched_outside(uplo)) << what;
                    if (period) EXPECT_TRUE(C.repeats(period)) << what;
                }
    Operand<T> A(8, 4, 1, 3), C(8, 8, 1, 5);
    EXPECT_THROW((syrk<B, T>(this->q(), A.view(), C.view(), T(1), T(0), Uplo::Lower, Transpose::ConjTrans)),
                 std::invalid_argument);
}

TYPED_TEST(HermitianCandidates, ComplexSymmetricSyr2kFamilies) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::B;
    for (const char* fam : {"triangular", "vendor"})
        for (int n : {7, 129, 200})
            for (Transpose t : {Transpose::NoTrans, Transpose::Trans})
                for (int period : {0, 3}) {
                    const std::string f = fam;
                    if (f == "vendor" && !TestFixture::kVendor) continue;
                    const int k = 9, batch = period ? 1024 : 3;
                    const Uplo uplo = t == Transpose::NoTrans ? Uplo::Upper : Uplo::Lower;
                    const bool nt = t == Transpose::NoTrans;
                    Operand<T> A(nt ? n : k, nt ? k : n, batch, 47u + n, false, period);
                    Operand<T> Bm(nt ? n : k, nt ? k : n, batch, 53u + n, false, period);
                    Operand<T> C(n, n, batch, 59u + n, false, period);
                    C.poison_opposite(uplo, false);
                    C.snapshot();
                    const T alpha = T(0.5, -1.25), beta = T(0.75, 0.5);
                    std::string ran;
                    const bool ok = sym_pin("syr2k", f, [&] {
                        syr2k<B, T>(this->q(), A.view(), Bm.view(), C.view(), alpha, beta, uplo, t).wait();
                    }, &ran);
                    const bool want = f != "triangular" || TestFixture::kCfloat;
                    const std::string what = f + " syr2k n=" + std::to_string(n) + (nt ? " N" : " T");
                    ASSERT_EQ(ok, want) << what;
                    if (!ok) continue;
                    EXPECT_EQ(ran, f) << what;
                    std::vector<int> items{0, batch - 1};
                    EXPECT_VERIFY(T, verify::Check::blas, 2 * k,
                                  MV<T>::rank2k_backward_error(this->q(), A.view(), Bm.view(), t, C.view0(), C.view(),
                                                               uplo, MV<T>::up(alpha), MV<T>::up(beta), false, items))
                        << what;
                    EXPECT_TRUE(C.untouched_outside(uplo)) << what;
                    if (period) EXPECT_TRUE(C.repeats(period)) << what;
                }
}

// symm: the reference mirrors A's triangle WITHOUT conjugating and keeps the complex diagonal.
TYPED_TEST(HermitianCandidates, ComplexSymmetricSymmFamilies) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::B;
    for (const char* fam : {"mirror", "expand", "vendor"})
        for (auto [m, n] : {std::pair{7, 33}, std::pair{130, 40}})
            for (Side side : {Side::Left, Side::Right})
                for (Uplo uplo : {Uplo::Lower, Uplo::Upper}) {
                    const std::string f = fam;
                    if (f == "vendor" && !TestFixture::kVendor) continue;
                    const int ka = side == Side::Left ? m : n, batch = 3;
                    Operand<T> A(ka, ka, batch, 61u + m, false), Bm(m, n, batch, 67u + n, false),
                        C(m, n, batch, 71u + m, false);
                    C.snapshot();
                    Operand<T> Aref = A;  // the full symmetric matrix the triangle stands for
                    for (int it = 0; it < batch; ++it)
                        for (int j = 0; j < ka; ++j)
                            for (int i = 0; i < ka; ++i)
                                if (uplo == Uplo::Lower ? i < j : i > j) Aref.at(it, i, j) = A.at(it, j, i);
                    A.poison_opposite(uplo, false);
                    const T alpha = T(1.25, 0.5), beta = T(-0.5, 0.25);
                    std::string ran;
                    ASSERT_TRUE(sym_pin("symm", f, [&] {
                        symm<B, T>(this->q(), A.view(), Bm.view(), C.view(), alpha, beta, side, uplo).wait();
                    }, &ran)) << f;
                    EXPECT_EQ(ran, f);
                    const Transpose N = Transpose::NoTrans;
                    std::vector<int> items{0, batch - 1};
                    const double err =
                        side == Side::Left
                            ? MV<T>::gemm_backward_error(this->q(), Aref.view(), N, Bm.view(), N, C.view0(), C.view(),
                                                         MV<T>::up(alpha), MV<T>::up(beta), items)
                            : MV<T>::gemm_backward_error(this->q(), Bm.view(), N, Aref.view(), N, C.view0(), C.view(),
                                                         MV<T>::up(alpha), MV<T>::up(beta), items);
                    EXPECT_VERIFY(T, verify::Check::blas, ka, err) << f << " m=" << m << " n=" << n;
                }
}

}  // namespace
