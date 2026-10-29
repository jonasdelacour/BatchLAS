// SPDX-License-Identifier: MIT
// trsm honours a batch mask in every native family (W1b), each pinned and reached, on
// tests/masked_contract.hh: a device-only mask, complex data, padded ld and stride, both sides,
// every transpose form. cta returns per work-group (I1); sg_left packs 32 / N items in a sub-group
// (I2, the holes patterns); blocked re-attaches the mask to every block view and pins the public
// gemm it calls. The heavy tier adds the saturating identical-item batch.
// evidence: docs/design/masked-batches.md#masked-batches-tests
#include <gtest/gtest.h>

#include <batchlas/blas/functions/trsm.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/error.hh>

#include <src/ops/gemm/choice.hh>
#include <src/ops/trsm/choice.hh>
#include <src/select/vendor.hh>
#include <tests/masked_contract.hh>
#include <tests/test_support/storage.hh>
#include <tests/test_utils.hh>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

using namespace batchlas;

namespace {

namespace mc = masked_contract;
namespace ot = ops::trsm;

template <typename T>
using MV = MatrixView<T, MatrixFormat::Dense>;

struct Shape {
    std::string spelling;
    int order, q, batch;
    Side side = Side::Left;
    Uplo uplo = Uplo::Lower;
    Transpose ta = Transpose::NoTrans;
    Diag diag = Diag::NonUnit;
    int wg_items = 0;        ///< items per packed sub-group (sg_left: 32 / bucket)
    bool identical = false;  ///< every item the same input (the saturating case)
};

std::string label(const Shape& s) {
    const char* w[3] = {"N", "T", "C"};
    return s.spelling + (s.side == Side::Left ? " L" : " R") + (s.uplo == Uplo::Lower ? "L" : "U") + w[int(s.ta)] +
           (s.diag == Diag::Unit ? "U" : "N") + " order " + std::to_string(s.order) + " q " + std::to_string(s.q) +
           " batch " + std::to_string(s.batch);
}

template <typename T, Backend B>
struct Problem {
    Queue& q;
    Shape s;
    T alpha;
    Matrix<T> A, Bm;

    Problem(Queue& queue, const Shape& shape)
        : q(queue), s(shape), alpha(MV<T>::make(0.75, -0.5)), A(test_support::padded<T>(s.order, s.order, s.batch)),
          Bm(s.side == Side::Left ? test_support::padded<T>(s.order, s.q, s.batch)
                                  : test_support::padded<T>(s.q, s.order, s.batch)) {}

    // Small values over 8 and a heavy diagonal: distinct, well-conditioned items.
    void fill(Matrix<T>& M, int salt) {
        T* p = M.data_ptr();
        const std::size_t per = std::size_t(M.stride()), ld = std::size_t(M.ld());
        for (std::size_t e = 0; e < per * std::size_t(s.batch); ++e) {
            const std::size_t item = s.identical ? 0 : e / per, off = e % per;
            double re = double(int((off * 7 + item * 3 + salt) % 23) - 11) / 8;
            const double im = double(int((off * 5 + item + salt) % 13) - 6) / 8;
            if (off % ld == off / ld) re += 6;
            p[e] = MV<T>::make(re, im);
        }
    }
    mc::Case make_case() {
        mc::Case c;
        c.batch = s.batch;
        c.wg_items = s.wg_items;
        c.identical_items = s.identical;
        c.prepare = [this] {
            fill(A, 1);
            fill(Bm, 5);
        };
        c.call = [this](Span<const std::uint8_t> mask, Span<std::byte>) {
            trsm<B, T>(q, A.view().with_batch_mask(mask), Bm.view().with_batch_mask(mask), alpha, s.side, s.uplo, s.ta,
                       s.diag);
        };
        c.outputs = [this] { return std::vector<mc::Output>{mc::Output::matrix("B", Bm.view())}; };
        return c;
    }
};

template <typename T, Backend B>
struct Cfg {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};
using Types = typename test_utils::backend_types<Cfg>::type;

template <typename Config>
class TrsmMask : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    static constexpr Backend B = Config::BackendVal;
    static constexpr Transpose kT = test_utils::is_complex<T>::value ? Transpose::ConjTrans : Transpose::Trans;

    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        if (this->HasFatalFailure() || ::testing::Test::IsSkipped()) return;
        if (!this->ctx || this->ctx->device().type != DeviceType::GPU) GTEST_SKIP() << "the native families are GPU kernels here";
    }

    // Pinned and reached (unless `reach` is off); blocked pins its trailing gemm too, so the
    // unmasked reference cannot take the vendor the masked call refuses.
    void check(const Shape& s, bool reach = true) {
        SCOPED_TRACE(label(s));
        Queue& q = *this->ctx;
        std::optional<select::ScopedPin<ops::gemm::GemmChoice>> gemm_pin;
        if (s.spelling == "blocked") gemm_pin.emplace("gemm", "tiled", select::StrictPin{});
        Problem<T, B> p(q, s);
        if (reach) {
            mc::check<ot::TrsmChoice>(q, "trsm", s.spelling, p.make_case());
        } else {
            const select::ScopedPin<ot::TrsmChoice> pin("trsm", s.spelling, select::StrictPin{});
            mc::check(q, p.make_case());
        }
    }

    // Every bucket of each register solver, both sides of cta, blocked across its outer block.
    static std::vector<Shape> shapes(int batch, bool identical) {
        const auto L = Uplo::Lower, U = Uplo::Upper;
        const auto N = Transpose::NoTrans;
        std::vector<Shape> v{
            {"cta", 13, 9, batch, Side::Left, L, N},
            {"cta", 7, 11, batch, Side::Right, U, kT, Diag::Unit},
            {"cta", 29, 5, batch, Side::Left, U, kT},
            {"sg_left", 3, 5, batch, Side::Left, L, N, Diag::NonUnit, 8},
            {"sg_left", 7, 3, batch, Side::Left, U, kT, Diag::Unit, 4},
            {"sg_left", 13, 9, batch, Side::Left, L, kT, Diag::NonUnit, 2},
            {"sg_left", 27, 17, batch, Side::Left, U, N, Diag::NonUnit, 1},
            {"blocked", 45, 7, batch, Side::Left, L, N},
            {"blocked", 41, 6, batch, Side::Right, U, kT},
        };
        for (Shape& s : v) s.identical = identical;
        return v;
    }
};

TYPED_TEST_SUITE(TrsmMask, Types);

TYPED_TEST(TrsmMask, EveryNativeFamilyLeavesInactiveItemsUntouched) {
    for (const Shape& s : TestFixture::shapes(24, false)) this->check(s);
}

// The saturating batch (>= 1024, identical items): cta's one-group-per-item launch, sg_left's
// packed sub-groups, and blocked's chain of both.
TYPED_TEST(TrsmMask, SaturatingEveryFamily) {
    for (const Shape& s : TestFixture::shapes(1024, true)) this->check(s, false);
}

// A masked call never reaches the vendor: a strict vendor pin finds no candidate and throws
// before writing B.
TYPED_TEST(TrsmMask, VendorRefusesAMaskedCall) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::B;
    if constexpr (!select::level3_vendor_available<B>) {
        GTEST_SKIP() << "no vendor trsm in this build";
    } else {
        Queue& q = *this->ctx;
        const Shape s{"vendor", 9, 5, 6};
        Problem<T, B> p(q, s);
        const mc::Case c = p.make_case();
        c.prepare();
        q.wait();
        const auto before = mc::detail::snapshot(c.outputs(), s.batch);
        const mc::DeviceMask mask(q, mc::patterns(s.batch)[2].bytes);
        const select::ScopedPin<ot::TrsmChoice> pin("trsm", "vendor", select::StrictPin{});
        try {
            c.call(mask.span(), {});
            q.wait();
            ADD_FAILURE() << "a vendor pin ran a masked call";
        } catch (const batchlas::invalid_argument& e) {
            EXPECT_NE(std::string(e.what()).find("run this shape"), std::string::npos) << e.what();
        }
        EXPECT_TRUE(mc::detail::snapshot(c.outputs(), s.batch) == before) << "the refused vendor pin wrote B";
    }
}

}  // namespace
