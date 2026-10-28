// SPDX-License-Identifier: MIT
// trmm, symm, hemm, syrk, herk, syr2k and her2k honour a batch mask in every native family (W1b),
// each pinned and reached, on tests/masked_contract.hh: a device-only mask, complex data, padded ld
// and stride, beta != 0 and beta = 0 (an inactive C is untouched even where it is never read). The
// families over the public gemm (expand, fold) pin gemm too, so the unmasked reference cannot take
// the vendor the masked call refuses. The heavy tier adds the saturating identical-item batch and
// the 65535-item grid chunk; the expansion chunk runs under a lowered BATCHLAS_EXPAND_MAX_BYTES.
// evidence: docs/design/masked-batches.md#masked-batches-tests
#include <gtest/gtest.h>

#include <batchlas/blas/functions/hemm.hh>
#include <batchlas/blas/functions/her2k.hh>
#include <batchlas/blas/functions/herk.hh>
#include <batchlas/blas/functions/symm.hh>
#include <batchlas/blas/functions/syr2k.hh>
#include <batchlas/blas/functions/syrk.hh>
#include <batchlas/blas/functions/trmm.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/error.hh>
#include <batchlas/util/env.hh>

#include <src/expansion_budget.hh>
#include <src/ops/gemm/choice.hh>
#include <src/ops/hemm/choice.hh>
#include <src/ops/her2k/choice.hh>
#include <src/ops/herk/choice.hh>
#include <src/ops/symm/choice.hh>
#include <src/ops/syr2k/choice.hh>
#include <src/ops/syrk/choice.hh>
#include <src/ops/trmm/choice.hh>
#include <src/select/vendor.hh>
#include <tests/masked_contract.hh>
#include <tests/test_support/storage.hh>
#include <tests/test_utils.hh>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

using namespace batchlas;

namespace {

namespace mc = masked_contract;

template <typename T>
using MV = MatrixView<T, MatrixFormat::Dense>;

struct Dims {
    int rows, cols;
};

// One masked level-3 problem: A, B (unused by the rank-k ops) and the output C. `op` runs the call
// on the three views, each carrying the mask; beta0 asks it for beta = 0 (C poisoned, never read).
template <typename T>
struct Problem {
    using Op = std::function<void(Queue&, const MV<T>&, const MV<T>&, const MV<T>&, bool beta0)>;
    Queue& q;
    int batch;
    bool identical, beta0;
    Matrix<T> A, Bm, C;
    Op op;

    Problem(Queue& queue, Dims a, Dims b, Dims c, int batch_, bool identical_, bool beta0_, Op op_)
        : q(queue), batch(batch_), identical(identical_), beta0(beta0_),
          A(test_support::padded<T>(a.rows, a.cols, batch_)), Bm(test_support::padded<T>(b.rows, b.cols, batch_)),
          C(test_support::padded<T>(c.rows, c.cols, batch_)), op(std::move(op_)) {}

    // Small values over 8 and a heavy diagonal, so every item is distinct and well conditioned.
    void fill(Matrix<T>& M, int salt) {
        T* p = M.data_ptr();
        const std::size_t per = std::size_t(M.stride()), ld = std::size_t(M.ld());
        for (std::size_t e = 0; e < per * std::size_t(batch); ++e) {
            const std::size_t item = identical ? 0 : e / per, off = e % per;
            double re = double(int((off * 7 + item * 3 + salt) % 23) - 11) / 8;
            const double im = double(int((off * 5 + item + salt) % 13) - 6) / 8;
            if (off % ld == off / ld) re += 4;
            p[e] = MV<T>::make(re, im);
        }
    }
    mc::Case make_case(int chunk_items = 0) {
        mc::Case c;
        c.batch = batch;
        c.chunk_items = chunk_items;
        c.identical_items = identical;
        c.prepare = [this] {
            fill(A, 1);
            fill(Bm, 5);
            std::fill_n(C.data_ptr(), std::size_t(C.stride()) * batch, test_support::poison<T>());
            if (!beta0) fill(C, 9);
        };
        c.call = [this](Span<const std::uint8_t> mask, Span<std::byte>) {
            op(q, A.view().with_batch_mask(mask), Bm.view().with_batch_mask(mask), C.view().with_batch_mask(mask), beta0);
        };
        c.outputs = [this] { return std::vector<mc::Output>{mc::Output::matrix("C", C.view())}; };
        return c;
    }
};

struct Spec {
    std::string spelling;
    Dims a, b, c;
    int batch = 12;
    bool identical = false;
    bool over_gemm = false;  ///< expand / fold: pin the public gemm as well
    int chunk_items = 0;
};

template <typename T, Backend B>
struct Cfg {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};
using Types = typename test_utils::backend_types<Cfg>::type;

template <typename Config>
class Level3Mask : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    using R = typename base_type<T>::type;
    static constexpr Backend B = Config::BackendVal;
    static constexpr bool kComplex = test_utils::is_complex<T>::value;
    static constexpr bool kCDouble = std::is_same_v<T, std::complex<double>>;
    static constexpr bool kCFloat = std::is_same_v<T, std::complex<float>>;

    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        if (this->HasFatalFailure() || ::testing::Test::IsSkipped()) return;
        if (!this->ctx || this->ctx->device().type != DeviceType::GPU) GTEST_SKIP() << "the native families are GPU kernels here";
    }

    static T alpha() { return MV<T>::make(0.75, -0.5); }
    static T beta() { return MV<T>::make(-0.25, 0.5); }

    // The contract at beta != 0 (pinned, reached unless `reach` is off), then at beta = 0 on the same pin.
    template <typename Choice>
    void check(const char* opname, const Spec& s, const typename Problem<T>::Op& op, bool has_beta = true,
               bool reach = true) {
        SCOPED_TRACE(std::string(opname) + " " + s.spelling + " batch " + std::to_string(s.batch));
        Queue& q = *this->ctx;
        std::optional<select::ScopedPin<ops::gemm::GemmChoice>> gemm_pin;
        if (s.over_gemm) gemm_pin.emplace("gemm", "tiled", select::StrictPin{});
        {
            Problem<T> p(q, s.a, s.b, s.c, s.batch, s.identical, !has_beta, op);
            if (reach) {
                mc::check<Choice>(q, opname, s.spelling, p.make_case(s.chunk_items));
            } else {
                const select::ScopedPin<Choice> pin(opname, s.spelling, select::StrictPin{});
                mc::check(q, p.make_case(s.chunk_items));
            }
        }
        if (!has_beta) return;
        const select::ScopedPin<Choice> pin(opname, s.spelling, select::StrictPin{});
        Problem<T> p(q, s.a, s.b, s.c, s.batch, s.identical, true, op);
        mc::check(q, p.make_case(s.chunk_items));
    }

    // Each op's call, at Left/Right, uplo and op(A) spelled per case.
    static typename Problem<T>::Op trmm_op(Side side, Uplo uplo, Transpose ta, Diag diag) {
        return [=](Queue& q, const MV<T>& A, const MV<T>& Bm, const MV<T>& C, bool) {
            trmm<B, T>(q, A, Bm, C, alpha(), side, uplo, ta, diag);
        };
    }
    static typename Problem<T>::Op symm_op(Side side, Uplo uplo) {
        return [=](Queue& q, const MV<T>& A, const MV<T>& Bm, const MV<T>& C, bool beta0) {
            symm<B, T>(q, A, Bm, C, alpha(), beta0 ? T(0) : beta(), side, uplo);
        };
    }
    static typename Problem<T>::Op hemm_op(Side side, Uplo uplo) {
        return [=](Queue& q, const MV<T>& A, const MV<T>& Bm, const MV<T>& C, bool beta0) {
            if constexpr (kComplex) hemm<B, T>(q, A, Bm, C, alpha(), beta0 ? T(0) : beta(), side, uplo);
        };
    }
    static typename Problem<T>::Op syrk_op(Uplo uplo, Transpose ta) {
        return [=](Queue& q, const MV<T>& A, const MV<T>&, const MV<T>& C, bool beta0) {
            syrk<B, T>(q, A, C, alpha(), beta0 ? T(0) : beta(), uplo, ta);
        };
    }
    static typename Problem<T>::Op herk_op(Uplo uplo, Transpose ta) {
        return [=](Queue& q, const MV<T>& A, const MV<T>&, const MV<T>& C, bool beta0) {
            if constexpr (kComplex) herk<B, T>(q, A, C, R(0.75), beta0 ? R(0) : R(-0.5), uplo, ta);
        };
    }
    static typename Problem<T>::Op syr2k_op(Uplo uplo, Transpose ta) {
        return [=](Queue& q, const MV<T>& A, const MV<T>& Bm, const MV<T>& C, bool beta0) {
            syr2k<B, T>(q, A, Bm, C, alpha(), beta0 ? T(0) : beta(), uplo, ta);
        };
    }
    static typename Problem<T>::Op her2k_op(Uplo uplo, Transpose ta) {
        return [=](Queue& q, const MV<T>& A, const MV<T>& Bm, const MV<T>& C, bool beta0) {
            if constexpr (kComplex) her2k<B, T>(q, A, Bm, C, alpha(), beta0 ? R(0) : R(-0.5), uplo, ta);
        };
    }

    // Every family once per op at batch `batch` (identical items when `identical`), ragged sizes.
    void every_family(int batch, bool identical) {
        const int m = 37, n = 29, k = 23;
        const auto L = Uplo::Lower, U = Uplo::Upper;
        const auto N = Transpose::NoTrans, Tr = Transpose::Trans, C = Transpose::ConjTrans;
        const Transpose tc = kComplex ? C : Tr;
        auto sp = [&](const char* w, Dims a, Dims b, Dims c, bool over = false) {
            return Spec{w, a, b, c, batch, identical, over};
        };
        check<ops::trmm::TrmmChoice>("trmm", sp("triangular", {m, m}, {m, n}, {m, n}),
                                     trmm_op(Side::Left, L, N, Diag::NonUnit), false);
        check<ops::trmm::TrmmChoice>("trmm", sp("triangular", {m, m}, {m, n}, {m, n}),
                                     trmm_op(Side::Left, U, tc, Diag::Unit), false);
        check<ops::trmm::TrmmChoice>("trmm", sp("expand", {n, n}, {m, n}, {m, n}, true),
                                     trmm_op(Side::Right, U, tc, Diag::NonUnit), false);
        check<ops::symm::SymmChoice>("symm", sp("mirror", {m, m}, {m, n}, {m, n}), symm_op(Side::Left, L));
        check<ops::symm::SymmChoice>("symm", sp("mirror", {n, n}, {m, n}, {m, n}), symm_op(Side::Right, U));
        check<ops::symm::SymmChoice>("symm", sp("expand", {m, m}, {m, n}, {m, n}, true), symm_op(Side::Left, U));
        check<ops::syrk::SyrkChoice>("syrk", sp("gram", {m, k}, {1, 1}, {m, m}), syrk_op(L, N));
        check<ops::syrk::SyrkChoice>("syrk", sp("gram", {k, m}, {1, 1}, {m, m}), syrk_op(U, Tr));
        if constexpr (!kCDouble) {
            check<ops::syrk::SyrkChoice>("syrk", sp("triangular", {m, k}, {1, 1}, {m, m}), syrk_op(U, N));
            check<ops::syr2k::Syr2kChoice>("syr2k", sp("triangular", {m, k}, {m, k}, {m, m}), syr2k_op(L, N));
            check<ops::syr2k::Syr2kChoice>("syr2k", sp("triangular", {k, m}, {k, m}, {m, m}), syr2k_op(U, Tr));
        }
        if constexpr (kComplex) {
            check<ops::hemm::HemmChoice>("hemm", sp("mirror", {m, m}, {m, n}, {m, n}), hemm_op(Side::Left, U));
            check<ops::hemm::HemmChoice>("hemm", sp("mirror", {n, n}, {m, n}, {m, n}), hemm_op(Side::Right, L));
            check<ops::hemm::HemmChoice>("hemm", sp("expand", {n, n}, {m, n}, {m, n}, true), hemm_op(Side::Right, U));
            check<ops::herk::HerkChoice>("herk", sp("gram", {m, k}, {1, 1}, {m, m}), herk_op(L, N));
            check<ops::herk::HerkChoice>("herk", sp("fold", {k, m}, {1, 1}, {m, m}, true), herk_op(U, C));
            check<ops::her2k::Her2kChoice>("her2k", sp("fold", {m, k}, {m, k}, {m, m}, true), her2k_op(L, N));
            if constexpr (kCFloat) {
                check<ops::herk::HerkChoice>("herk", sp("triangular", {k, m}, {1, 1}, {m, m}), herk_op(L, C));
                check<ops::her2k::Her2kChoice>("her2k", sp("triangular", {k, m}, {k, m}, {m, m}), her2k_op(U, C));
            }
        }
    }
};

TYPED_TEST_SUITE(Level3Mask, Types);

// One case per native family of every op: the tile kernels (trmm, syrk, syr2k, herk and her2k
// triangular, syrk/herk gram), symm/hemm mirror (wide v2 body, double's tiled body), and the
// families over the public gemm (expand, fold).
TYPED_TEST(Level3Mask, EveryNativeFamilyLeavesInactiveItemsUntouched) { this->every_family(12, false); }

// The saturating batch (>= 1024, identical items) on every family; then the 65535-item grid chunk
// of the tile kernels and mirror takes its slice of the mask.
TYPED_TEST(Level3Mask, SaturatingEveryFamilyAndTheGridChunk) {
    this->every_family(1024, true);
    const int grid = int(ops::trmm::kMaxGridBatch), big = 2 * grid + 7;
    const auto L = Uplo::Lower;
    const auto N = Transpose::NoTrans;
    this->template check<ops::trmm::TrmmChoice>("trmm", Spec{"triangular", {2, 2}, {2, 3}, {2, 3}, big, false, false, grid},
                                                TestFixture::trmm_op(Side::Left, L, N, Diag::NonUnit), false, false);
    this->template check<ops::symm::SymmChoice>("symm", Spec{"mirror", {2, 2}, {2, 3}, {2, 3}, big, false, false, grid},
                                                TestFixture::symm_op(Side::Left, L), false, false);
    this->template check<ops::syrk::SyrkChoice>("syrk", Spec{"gram", {3, 2}, {1, 1}, {3, 3}, big, false, false, grid},
                                                TestFixture::syrk_op(L, N), false, false);
}

// Each expansion chunk of expand and fold takes its slice of the mask: seven items per chunk under
// a lowered BATCHLAS_EXPAND_MAX_BYTES, the chunk-edge and wholly inactive part patterns.
TYPED_TEST(Level3Mask, ExpansionChunksTakeTheirMaskSlice) {
    using T = typename TestFixture::T;
    const int grid = int(ops::trmm::kMaxGridBatch);
    const auto L = Uplo::Lower;
    const auto N = Transpose::NoTrans;
    const int n = 9, batch = 31;
    Queue& q = *this->ctx;
    const std::string cap = std::to_string(backend::detail::expanded_workspace_bytes<T>(q, n, 7));
    const ScopedEnvVar env("BATCHLAS_EXPAND_MAX_BYTES", cap.c_str());
    const int chunk = backend::detail::expansion_chunk<T>(q, n, batch, grid);
    ASSERT_GT(chunk, 1);
    ASSERT_LT(chunk, batch / 2) << "the lowered budget did not chunk the expansion";
    this->template check<ops::trmm::TrmmChoice>("trmm", Spec{"expand", {n, n}, {n, 5}, {n, 5}, batch, false, true, chunk},
                                                TestFixture::trmm_op(Side::Left, L, N, Diag::NonUnit), false);
    this->template check<ops::symm::SymmChoice>("symm", Spec{"expand", {n, n}, {n, 5}, {n, 5}, batch, false, true, chunk},
                                                TestFixture::symm_op(Side::Left, L));
    if constexpr (TestFixture::kComplex) {
        this->template check<ops::herk::HerkChoice>("herk", Spec{"fold", {n, 4}, {1, 1}, {n, n}, batch, false, true, chunk},
                                                    TestFixture::herk_op(L, N));
        this->template check<ops::her2k::Her2kChoice>(
            "her2k", Spec{"fold", {n, 4}, {n, 4}, {n, n}, batch, false, true, chunk}, TestFixture::her2k_op(L, N));
    }
}

// A masked call never reaches the vendor: a strict vendor pin finds no candidate and throws
// before writing C, on every op with a vendor in this build.
TYPED_TEST(Level3Mask, VendorPinsRefuseAMaskedCall) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::B;
    if constexpr (!select::level3_vendor_available<B>) {
        GTEST_SKIP() << "no level-3 vendor in this build";
    } else {
        Queue& q = *this->ctx;
        const int m = 9, n = 7, batch = 6;
        const auto L = Uplo::Lower;
        const auto N = Transpose::NoTrans;
        auto refuses = [&](const char* opname, auto choice_tag, Dims a, Dims b, Dims c,
                           const typename Problem<T>::Op& op) {
            using Choice = decltype(choice_tag);
            SCOPED_TRACE(opname);
            Problem<T> p(q, a, b, c, batch, false, false, op);
            const mc::Case cs = p.make_case();
            cs.prepare();
            q.wait();
            const auto before = mc::detail::snapshot(cs.outputs(), batch);
            const mc::DeviceMask mask(q, mc::patterns(batch)[2].bytes);
            const select::ScopedPin<Choice> pin(opname, "vendor", select::StrictPin{});
            try {
                cs.call(mask.span(), {});
                q.wait();
                ADD_FAILURE() << "a vendor pin ran a masked call";
            } catch (const batchlas::invalid_argument& e) {
                EXPECT_NE(std::string(e.what()).find("run this shape"), std::string::npos) << e.what();
            }
            EXPECT_TRUE(mc::detail::snapshot(cs.outputs(), batch) == before) << "the refused vendor pin wrote C";
        };
        refuses("trmm", ops::trmm::TrmmChoice{}, {m, m}, {m, n}, {m, n}, TestFixture::trmm_op(Side::Left, L, N, Diag::NonUnit));
        refuses("symm", ops::symm::SymmChoice{}, {m, m}, {m, n}, {m, n}, TestFixture::symm_op(Side::Left, L));
        refuses("syrk", ops::syrk::SyrkChoice{}, {m, n}, {1, 1}, {m, m}, TestFixture::syrk_op(L, N));
        refuses("syr2k", ops::syr2k::Syr2kChoice{}, {m, n}, {m, n}, {m, m}, TestFixture::syr2k_op(L, N));
        if constexpr (TestFixture::kComplex) {
            refuses("hemm", ops::hemm::HemmChoice{}, {m, m}, {m, n}, {m, n}, TestFixture::hemm_op(Side::Left, L));
            refuses("herk", ops::herk::HerkChoice{}, {m, n}, {1, 1}, {m, m}, TestFixture::herk_op(L, N));
            refuses("her2k", ops::her2k::Her2kChoice{}, {m, n}, {m, n}, {m, m}, TestFixture::her2k_op(L, N));
        }
    }
}

}  // namespace
