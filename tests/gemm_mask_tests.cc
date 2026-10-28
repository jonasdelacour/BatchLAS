// SPDX-License-Identifier: MIT
// gemm honours a batch mask in every native family (W1a), each pinned and reached, on
// tests/masked_contract.hh: every mask pattern (packed work-group holes for `small`), a device-only
// mask, complex data, padded ld and stride, beta != 0 and beta = 0 (an inactive C is untouched even
// where it is never read). The heavy tier adds every reg / wide row and split-k slice count, the
// saturating identical-item batch, and the 65535-item grid chunk and split-k's part loop.
// evidence: docs/design/masked-batches.md#masked-batches-tests
#include <gtest/gtest.h>

#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/error.hh>
#include <batchlas/util/sycl-vector.hh>

#include <src/ops/gemm/choice.hh>
#include <src/select/vendor.hh>
#include <tests/masked_contract.hh>
#include <tests/test_support/storage.hh>
#include <tests/test_utils.hh>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

using namespace batchlas;

namespace {

namespace og = ops::gemm;
namespace mc = masked_contract;

constexpr Transpose kOps[3] = {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans};

// One masked gemm problem: the shape, the forms, the layout, the scalars and the mask pattern knobs.
struct Shape {
    std::string spelling;  ///< the pinned choice
    int m, n, k, batch;
    Transpose ta = Transpose::NoTrans, tb = Transpose::NoTrans;
    int wg_items = 0;       ///< items per packed work-group (small's kG)
    bool aligned = false;   ///< ld and stride whole 16-byte granules: the vector-load legs
    bool identical = false; ///< every item the same input (the saturating case)
    int chunk_items = 0;    ///< the launch's part size: chunk-edge patterns
};

std::string label(const Shape& s) {
    const char* w[3] = {"N", "T", "C"};
    return s.spelling + " " + w[int(s.ta)] + w[int(s.tb)] + " " + std::to_string(s.m) + "x" + std::to_string(s.n) +
           "x" + std::to_string(s.k) + " batch " + std::to_string(s.batch) + (s.aligned ? " aligned" : "");
}

template <typename T, Backend B>
struct Problem {
    Queue& q;
    Shape s;
    T alpha, beta;
    Matrix<T> A, Bm, C;

    static Matrix<T> make(const Shape& s, int rows, int cols) {
        return s.aligned ? test_support::padded<T>(rows, cols, s.batch, 4, 8) : test_support::padded<T>(rows, cols, s.batch);
    }
    Problem(Queue& queue, const Shape& shape, T beta_)
        : q(queue), s(shape), alpha(MatrixView<T>::make(0.75, -0.5)), beta(beta_),
          A(make(s, s.ta == Transpose::NoTrans ? s.m : s.k, s.ta == Transpose::NoTrans ? s.k : s.m)),
          Bm(make(s, s.tb == Transpose::NoTrans ? s.k : s.n, s.tb == Transpose::NoTrans ? s.n : s.k)),
          C(make(s, s.m, s.n)) {}

    // Small integers over 8: exact products, so the bit comparison is about the mask alone.
    void fill(Matrix<T>& M, int salt) {
        T* p = M.data_ptr();
        const std::size_t per = std::size_t(M.stride());
        for (std::size_t e = 0; e < per * std::size_t(s.batch); ++e) {
            const std::size_t item = s.identical ? 0 : e / per, off = e % per;
            const double re = double(int((off * 7 + item * 3 + salt) % 23) - 11) / 8;
            const double im = double(int((off * 5 + item + salt) % 13) - 6) / 8;
            p[e] = MatrixView<T>::make(re, im);
        }
    }
    mc::Case make_case() {
        mc::Case c;
        c.batch = s.batch;
        c.wg_items = s.wg_items;
        c.chunk_items = s.chunk_items;
        c.identical_items = s.identical;
        c.prepare = [this] {
            fill(A, 1);
            fill(Bm, 5);
            std::fill_n(C.data_ptr(), std::size_t(C.stride()) * s.batch, test_support::poison<T>());
            if (beta != T(0)) fill(C, 9);
        };
        c.call = [this](Span<const std::uint8_t> mask, Span<std::byte>) {
            gemm<B, T>(q, A.view().with_batch_mask(mask), Bm.view().with_batch_mask(mask), C.view().with_batch_mask(mask),
                       alpha, beta, s.ta, s.tb);
        };
        c.outputs = [this] { return std::vector<mc::Output>{mc::Output::matrix("C", C.view())}; };
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
class GemmMask : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    static constexpr Backend B = Config::BackendVal;

    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        if (this->HasFatalFailure() || ::testing::Test::IsSkipped()) return;
        if (!this->ctx || this->ctx->device().type != DeviceType::GPU) GTEST_SKIP() << "the native families are GPU kernels here";
    }

    // The contract at beta != 0 (pinned, reached unless `reach` is off), then at beta = 0 under the same pin.
    void check(const Shape& s, bool reach = true) {
        SCOPED_TRACE(label(s));
        Queue& q = *this->ctx;
        {
            Problem<T, B> p(q, s, MatrixView<T>::make(-0.25, 0.5));
            if (reach) {
                mc::check<og::GemmChoice>(q, "gemm", s.spelling, p.make_case());
            } else {
                const select::ScopedPin<og::GemmChoice> pin("gemm", s.spelling, select::StrictPin{});
                mc::check(q, p.make_case());
            }
        }
        const select::ScopedPin<og::GemmChoice> pin("gemm", s.spelling, select::StrictPin{});
        Problem<T, B> p(q, s, T(0));
        mc::check(q, p.make_case());
    }

    static constexpr bool kReal = !test_utils::is_complex<T>::value;
    static constexpr bool kFloat = std::is_same_v<T, float>;

    // The small family's legs: every batched bucket in a transposed and a plain form (holes in a
    // packed work-group of kG = 128 / (2 NB) items), and float NN's one-item-per-group tiled leg.
    static std::vector<Shape> small_shapes(int batch) {
        std::vector<Shape> v{{"small", 5, 7, 6, batch, Transpose::NoTrans, Transpose::NoTrans, 8},
                             {"small", 6, 5, 7, batch, Transpose::ConjTrans, Transpose::NoTrans, 8},
                             {"small", 12, 9, 14, batch, Transpose::NoTrans, Transpose::Trans, 4},
                             {"small", 30, 17, 25, batch, Transpose::Trans, Transpose::ConjTrans, 2}};
        if (og::small_max_dim<T>() >= 64) v.push_back({"small", 50, 40, 61, batch, Transpose::NoTrans, Transpose::Trans, 1});
        if (kFloat) {
            v.push_back({"small", 40, 36, 34, batch});  // tiled leg, NB 48
            v.push_back({"small", 52, 50, 55, batch});  // tiled leg, NB 56
        }
        return v;
    }
    // A form row r instantiates for T (NN when it has one).
    static std::pair<Transpose, Transpose> reg_form(const og::RegCfg& r) {
        if (r.forms.nn) return {Transpose::NoTrans, Transpose::NoTrans};
        if (r.forms.nt) return {Transpose::NoTrans, Transpose::Trans};
        if (r.forms.tn) return {Transpose::Trans, Transpose::NoTrans};
        return {Transpose::Trans, Transpose::Trans};
    }
    static std::pair<Transpose, Transpose> wide_form(const og::WideCfg& c) {
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                if (og::wide_serves<T>(c, a, b)) return {kOps[a], kOps[b]};
        return {Transpose::NoTrans, Transpose::NoTrans};
    }
    static std::string reg_word(const og::RegCfg& r) {
        return "reg:m=" + std::to_string(r.m) + ":n=" + std::to_string(r.n) + ":k=" + std::to_string(r.k) +
               ":u=" + std::to_string(r.u);
    }
    static std::string wide_word(const og::WideCfg& c, const char* fam = "wide", int s = 0) {
        return std::string(fam) + ":m=" + std::to_string(c.m) + ":n=" + std::to_string(c.n) + ":k=" +
               std::to_string(c.k) + (s ? ":s=" + std::to_string(s) : "");
    }
    // Every reg and wide row at one form, ragged in m, n and k (the predicated legs).
    static std::vector<Shape> row_shapes(int batch) {
        std::vector<Shape> v;
        if constexpr (kFloat)
            for (const og::RegCfg& r : og::reg_configs) {
                const auto [ta, tb] = reg_form(r);
                v.push_back({reg_word(r), 37, 29, 23, batch, ta, tb});
            }
        for (const og::WideCfg& c : og::wide_configs) {
            if (!c.compiles<T>()) continue;
            const auto [ta, tb] = wide_form(c);
            v.push_back({wide_word(c), 37, 29, 23, batch, ta, tb});
        }
        return v;
    }
    static std::vector<Shape> splitk_shapes(int batch, bool every_slice) {
        std::vector<Shape> v;
        for (const og::WideCfg& c : og::wide_configs) {
            if (!og::splitk_host(c)) continue;
            const auto [ta, tb] = wide_form(c);
            for (int s : og::kSplitKSlices)
                if (every_slice || s == 2) v.push_back({wide_word(c, "splitk", s), 21, 19, 150, batch, ta, tb});
        }
        return v;
    }
    // The aligned legs: 128x128's unpredicated body, 256x128's 16-byte copies, wide v2's vector
    // loads, double 64x64x16's own fast path.
    static std::vector<Shape> aligned_shapes(int batch) {
        std::vector<Shape> v;
        if constexpr (kFloat) {
            v.push_back({"reg:m=128:n=128:k=8:u=1", 128, 128, 24, batch, Transpose::NoTrans, Transpose::NoTrans, 0, true});
            v.push_back({"reg:m=128:n=128:k=8:u=4", 128, 256, 16, batch, Transpose::NoTrans, Transpose::Trans, 0, true});
            v.push_back({"reg:m=256:n=128:k=8:u=4", 132, 68, 20, batch, Transpose::NoTrans, Transpose::NoTrans, 0, true});
        }
        v.push_back({"wide:m=64:n=64:k=16", 68, 60, 32, batch, Transpose::NoTrans, Transpose::NoTrans, 0, true});
        if constexpr (std::is_same_v<T, double>)
            v.push_back({"wide:m=64:n=64:k=16", 64, 128, 32, batch, Transpose::NoTrans, Transpose::NoTrans, 0, true});
        if constexpr (std::is_same_v<T, std::complex<float>>)
            v.push_back({"wide:m=128:n=128:k=8", 132, 72, 24, batch, Transpose::NoTrans, Transpose::ConjTrans, 0, true});
        return v;
    }
};

TYPED_TEST_SUITE(GemmMask, Types);

// One representative per kernel: direct, tiled, every small leg, the 128x128 / 256x128 / launch_reg
// bodies, wide v2, the double 64x64x16 kernel, a transposed wide tile, split-k's slices and reduce.
TYPED_TEST(GemmMask, EveryNativeFamilyLeavesInactiveItemsUntouched) {
    using T = typename TestFixture::T;
    const int batch = 24;
    std::vector<Shape> v{{"direct", 9, 7, 5, batch, Transpose::Trans, Transpose::NoTrans},
                         {"tiled", 33, 20, 17, batch, Transpose::ConjTrans, Transpose::NoTrans}};
    for (const Shape& s : TestFixture::small_shapes(batch)) v.push_back(s);
    if constexpr (TestFixture::kFloat) {
        v.push_back({"reg:m=128:n=128:k=8:u=1", 37, 29, 23, batch});
        v.push_back({"reg:m=128:n=128:k=8:u=4", 37, 29, 23, batch, Transpose::Trans, Transpose::NoTrans});
        v.push_back({"reg:m=256:n=128:k=8:u=4", 37, 29, 23, batch, Transpose::NoTrans, Transpose::Trans});
        v.push_back({"reg:m=128:n=32:k=32:u=1", 37, 29, 23, batch});
    }
    v.push_back({"wide:m=64:n=64:k=16", 37, 29, 23, batch});
    v.push_back({"wide:m=32:n=32:k=16", 37, 29, 23, batch, Transpose::ConjTrans, Transpose::NoTrans});
    if constexpr (std::is_same_v<T, std::complex<float>>) v.push_back({"wide:m=128:n=128:k=8", 37, 29, 23, batch});
    v.push_back({"splitk:m=32:n=32:k=16:s=2", 21, 19, 150, batch, Transpose::NoTrans, Transpose::ConjTrans});
    for (const Shape& s : v) this->check(s);
}

// Every reg and wide row at one form, every split-k slice count, the aligned legs, and the
// saturating batch (>= 1024, half masked, identical items) for the local-memory kernels.
TYPED_TEST(GemmMask, SaturatingEveryRowSliceAndLeg) {
    for (const Shape& s : TestFixture::row_shapes(16)) this->check(s);
    for (const Shape& s : TestFixture::splitk_shapes(16, true)) this->check(s, s.spelling.ends_with(":s=2"));
    for (const Shape& s : TestFixture::aligned_shapes(8)) this->check(s);
    std::vector<Shape> sat;
    for (Shape s : TestFixture::small_shapes(1024)) {
        s.identical = true;
        sat.push_back(s);
    }
    sat.push_back({"tiled", 33, 20, 17, 1024, Transpose::NoTrans, Transpose::NoTrans, 0, false, true});
    if constexpr (TestFixture::kFloat)
        sat.push_back({"reg:m=128:n=128:k=8:u=1", 37, 29, 23, 1024, Transpose::NoTrans, Transpose::NoTrans, 0, false, true});
    sat.push_back({"wide:m=64:n=64:k=16", 37, 29, 23, 1024, Transpose::NoTrans, Transpose::NoTrans, 0, false, true});
    sat.push_back({"splitk:m=32:n=32:k=16:s=2", 21, 19, 150, 1024, Transpose::NoTrans, Transpose::NoTrans, 0, false, true});
    for (const Shape& s : sat) this->check(s, false);
}

// The 65535-item grid chunk (gemm.cc's for_batch_chunks): each part takes its slice of the mask.
// chunk-edges-off turns off the first and last item of every part, part-inactive all of part 1.
// Split-k's part is 65535 / s items, its partials laid out per part item.
TYPED_TEST(GemmMask, PastTheGridEachPartTakesItsMaskSlice) {
    const int grid = int(og::kMaxGridBatch);
    this->check({"direct", 2, 3, 2, 2 * grid + 7, Transpose::NoTrans, Transpose::Trans, 0, false, false, grid}, false);
    this->check({"splitk:m=32:n=32:k=16:s=8", 3, 2, 40, 2 * (grid / 8) + 5, Transpose::NoTrans, Transpose::NoTrans, 0,
                 false, false, grid / 8},
                false);
}

// A masked call never reaches the vendor: Auto walks on to a native family (the contract holds
// with no pin), and a strict vendor pin finds no candidate and throws before writing.
TYPED_TEST(GemmMask, VendorRefusesAndAutoRunsNatively) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::B;
    Queue& q = *this->ctx;
    const Shape s{"auto", 64, 64, 64, 32};
    {
        // Auto may route the unmasked reference to the vendor, so active items are verified instead.
        using MV = MatrixView<T, MatrixFormat::Dense>;
        Problem<T, B> p(q, s, MatrixView<T>::make(-0.25, 0.5));
        mc::Case c = p.make_case();
        c.prepare();
        q.wait();
        Matrix<T> C0 = Problem<T, B>::make(s, s.m, s.n);
        std::copy_n(p.C.data_ptr(), std::size_t(p.C.stride()) * s.batch, C0.data_ptr());
        c.verify_active = [&](const std::string&, std::span<const int> active) {
            if (active.empty()) return;
            const double err = MV::gemm_backward_error(q, p.A.view(), s.ta, p.Bm.view(), s.tb, C0.view(), p.C.view(),
                                                       MV::up(p.alpha), MV::up(p.beta), active);
            EXPECT_VERIFY(T, verify::Check::blas, s.k, err);
        };
        mc::check(q, c);
    }
    if constexpr (!select::level3_vendor_available<B>) GTEST_SKIP() << "no vendor gemm in this build";
    Problem<T, B> p(q, s, T(0));
    const mc::Case c = p.make_case();
    c.prepare();
    q.wait();
    const auto before = mc::detail::snapshot(c.outputs(), s.batch);
    const mc::DeviceMask mask(q, mc::patterns(s.batch)[2].bytes);
    const select::ScopedPin<og::GemmChoice> pin("gemm", "vendor", select::StrictPin{});
    try {
        c.call(mask.span(), {});
        q.wait();
        ADD_FAILURE() << "a vendor pin ran a masked call";
    } catch (const batchlas::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("run this shape"), std::string::npos) << e.what();
    }
    EXPECT_TRUE(mc::detail::snapshot(c.outputs(), s.batch) == before) << "the refused vendor pin wrote C";
}

}  // namespace
