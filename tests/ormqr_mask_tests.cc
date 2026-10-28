// SPDX-License-Identifier: MIT
// ormqr, orgqr and ormqx_cta honour a batch mask (W2c) on tests/masked_contract.hh: an inactive
// item's C (ormqr, ormqx_cta) or A (orgqr) is bit-for-bit untouched, an active one equals the
// unmasked call on the same pin. Device-only masks, complex data, padded ld and stride, a tau batch
// stride with a gap. ormqr's and orgqr's blocked family runs a masked view on both sides, its Left
// apply's trmm leg included (W1b); NETLIB's host loops skip an inactive item;
// ormqx_cta folds the mask into its packed partitions (holes inside a sub-group).
// evidence: docs/design/masked-batches.md#masked-batches-tests
#include <gtest/gtest.h>

#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/functions/orgqr.hh>
#include <batchlas/blas/functions/ormqr.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/error.hh>
#include <batchlas/util/sycl-vector.hh>

#include <src/ops/orgqr/choice.hh>
#include <src/ops/ormqr/choice.hh>
#include <tests/masked_contract.hh>
#include <tests/test_support/storage.hh>
#include <tests/test_utils.hh>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <string>
#include <vector>

using namespace batchlas;

namespace {

namespace mc = masked_contract;

// Small values over 8 and |tau| <= 1.5: the reflectors need not be a factorization's for the
// mask's bit comparison, only finite.
template <typename T>
void fill(T* p, std::size_t elems, std::size_t per, int salt, bool identical, double scale = 1.0) {
    for (std::size_t e = 0; e < elems; ++e) {
        const std::size_t item = identical ? 0 : e / per, off = e % per;
        const double re = double(int((off * 7 + item * 3 + salt) % 23) - 11) / 8 * scale;
        const double im = double(int((off * 5 + item + salt) % 13) - 6) / 8 * scale;
        p[e] = MatrixView<T>::make(re, im);
    }
}

struct Shape {
    std::string spelling;  ///< the pinned choice ("" for ormqx_cta, which has no selection)
    int m, n, k, batch;    ///< ormqr: C is m x n, Q's order is m (Left) or n (Right); orgqr: A is m x n
    Side side = Side::Left;
    Transpose trans = Transpose::NoTrans;
    int hint = 0;            ///< ormqr's block size hint: several WY blocks
    int wg_items = 0;        ///< the hole patterns
    int chunk_items = 0;     ///< the chunk-edge patterns
    bool identical = false;  ///< the saturating case
};

std::string label(const Shape& s) {
    return s.spelling + (s.side == Side::Left ? " L" : " R") + (s.trans == Transpose::NoTrans ? "N " : "C ") +
           std::to_string(s.m) + "x" + std::to_string(s.n) + " k " + std::to_string(s.k) + " batch " +
           std::to_string(s.batch);
}

template <typename T, Backend B>
struct OrmqrProblem {
    Queue& q;
    Shape s;
    Matrix<T> A, C;
    UnifiedVector<T> tau;

    int nq() const { return s.side == Side::Left ? s.m : s.n; }
    OrmqrProblem(Queue& queue, const Shape& shape)
        : q(queue), s(shape), A(test_support::padded<T>(shape.side == Side::Left ? shape.m : shape.n, shape.k, shape.batch)),
          C(test_support::padded<T>(shape.m, shape.n, shape.batch)), tau(std::size_t(shape.k) * shape.batch) {}

    mc::Case make_case() {
        mc::Case c;
        c.batch = s.batch;
        c.wg_items = s.wg_items;
        c.chunk_items = s.chunk_items;
        c.identical_items = s.identical;
        // blocked's gemm children take the vendor unmasked and a native family masked.
        c.masked_reference = s.spelling == "blocked" && (B == Backend::CUDA || B == Backend::ROCM);
        c.prepare = [this] {
            fill(A.data_ptr(), std::size_t(A.stride()) * s.batch, std::size_t(A.stride()), 1, s.identical);
            fill(C.data_ptr(), std::size_t(C.stride()) * s.batch, std::size_t(C.stride()), 9, s.identical);
            fill(tau.data(), tau.size(), std::size_t(s.k), 4, s.identical, 1.0 / 8);
        };
        c.call = [this](Span<const std::uint8_t> mask, Span<std::byte> ws) {
            ormqr<B, T>(q, A.view().with_batch_mask(mask), C.view().with_batch_mask(mask), s.side, s.trans,
                        tau.to_span(), ws, s.hint);
        };
        c.buffer_size = [this](Span<const std::uint8_t> mask) {
            return ormqr_buffer_size<B, T>(q, A.view().with_batch_mask(mask), C.view().with_batch_mask(mask), s.side,
                                           s.trans, tau.to_span(), s.hint);
        };
        c.outputs = [this] { return std::vector<mc::Output>{mc::Output::matrix("C", C.view())}; };
        return c;
    }
};

template <typename T, Backend B>
struct OrgqrProblem {
    Queue& q;
    Shape s;
    Matrix<T> A;
    UnifiedVector<T> tau;

    OrgqrProblem(Queue& queue, const Shape& shape)
        : q(queue), s(shape), A(test_support::padded<T>(shape.m, shape.n, shape.batch)),
          tau(std::size_t(shape.n) * shape.batch) {}

    mc::Case make_case() {
        mc::Case c;
        c.batch = s.batch;
        c.chunk_items = s.chunk_items;
        c.identical_items = s.identical;
        c.masked_reference = s.spelling == "blocked" && (B == Backend::CUDA || B == Backend::ROCM);
        c.prepare = [this] {
            fill(A.data_ptr(), std::size_t(A.stride()) * s.batch, std::size_t(A.stride()), 2, s.identical);
            fill(tau.data(), tau.size(), std::size_t(s.n), 6, s.identical, 1.0 / 8);
        };
        c.call = [this](Span<const std::uint8_t> mask, Span<std::byte> ws) {
            orgqr<B, T>(q, A.view().with_batch_mask(mask), tau.to_span(), ws);
        };
        c.buffer_size = [this](Span<const std::uint8_t> mask) {
            return orgqr_buffer_size<B, T>(q, A.view().with_batch_mask(mask), tau.to_span());
        };
        c.outputs = [this] { return std::vector<mc::Output>{mc::Output::matrix("A", A.view())}; };
        return c;
    }
};

// ormqx_cta: square n x n A and C, k reflectors, tau at a batch stride of n + 2.
template <typename T, Backend B>
struct CtaProblem {
    Queue& q;
    Shape s;
    Uplo fact;
    Matrix<T> A, C;
    UnifiedVector<T> tau;

    CtaProblem(Queue& queue, const Shape& shape, Uplo factorization)
        : q(queue), s(shape), fact(factorization), A(test_support::padded<T>(shape.n, shape.n, shape.batch)),
          C(test_support::padded<T>(shape.n, shape.n, shape.batch)), tau(std::size_t(shape.n + 2) * shape.batch) {}

    mc::Case make_case() {
        mc::Case c;
        c.batch = s.batch;
        c.wg_items = s.wg_items;
        c.identical_items = s.identical;
        c.prepare = [this] {
            fill(A.data_ptr(), std::size_t(A.stride()) * s.batch, std::size_t(A.stride()), 3, s.identical);
            fill(C.data_ptr(), std::size_t(C.stride()) * s.batch, std::size_t(C.stride()), 8, s.identical);
            fill(tau.data(), tau.size(), std::size_t(s.n + 2), 5, s.identical, 1.0 / 8);
        };
        c.call = [this](Span<const std::uint8_t> mask, Span<std::byte> ws) {
            const VectorView<T> tv(tau.data(), s.n, s.batch, 1, s.n + 2);
            ormqx_cta<B, T>(q, A.view().with_batch_mask(mask), tv.with_batch_mask(mask), C.view().with_batch_mask(mask),
                            fact, s.side, s.trans, s.k, ws);
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
class OrmqrMask : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    static constexpr Backend B = Config::BackendVal;
    static constexpr Transpose kC = Transpose::ConjTrans;

    bool gpu() const { return this->ctx->device().type == DeviceType::GPU; }

    void check_ormqr(const Shape& s) {
        SCOPED_TRACE(label(s));
        OrmqrProblem<T, B> p(*this->ctx, s);
        mc::check<ops::ormqr::OrmqrChoice>(*this->ctx, "ormqr", s.spelling, p.make_case());
    }
    void check_orgqr(const Shape& s) {
        SCOPED_TRACE(label(s));
        OrgqrProblem<T, B> p(*this->ctx, s);
        mc::check<ops::orgqr::OrgqrChoice>(*this->ctx, "orgqr", s.spelling, p.make_case());
    }
    // Every (factorization, side, trans) at order n with k reflectors.
    void check_cta(int n, int k, int batch, bool identical) {
        if (!this->ctx->device().supports_sub_group_size(32)) GTEST_SKIP() << "ormqx_cta needs sub-group size 32";
        const int p = n <= 4 ? 4 : n <= 8 ? 8 : n <= 16 ? 16 : 32;
        for (Uplo f : {Uplo::Upper, Uplo::Lower})
            for (Side side : {Side::Left, Side::Right})
                for (Transpose t : {Transpose::NoTrans, kC}) {
                    // Holes inside a sub-group: 32 / P partitions share its barriers.
                    const Shape s{"", n, n, k, batch, side, t, 0, 32 / p, 0, identical};
                    SCOPED_TRACE(label(s) + (f == Uplo::Lower ? " QL" : " QR"));
                    CtaProblem<T, B> pr(*this->ctx, s, f);
                    mc::check(*this->ctx, pr.make_case());
                }
    }
};

TYPED_TEST_SUITE(OrmqrMask, Types);

// Several WY blocks (hint 16), both sides and both forms where blocked runs masked, and parts of 8.
TYPED_TEST(OrmqrMask, BlockedLeavesInactiveItemsUntouched) {
    if (!this->gpu()) GTEST_SKIP() << "blocked is a GPU family";
    constexpr Transpose kC = TestFixture::kC;
    for (const Shape& s : {Shape{"blocked", 30, 21, 40, 24, Side::Right, Transpose::NoTrans, 16, 0, 8},
                           Shape{"blocked", 19, 37, 33, 12, Side::Right, kC, 16},
                           Shape{"blocked", 45, 13, 40, 24, Side::Left, Transpose::NoTrans, 16, 0, 8},
                           Shape{"blocked", 37, 9, 20, 12, Side::Left, kC, 16}})
        this->check_ormqr(s);
    this->check_orgqr({"blocked", 45, 40, 40, 24, Side::Left, Transpose::NoTrans, 0, 0, 8});
    this->check_orgqr({"blocked", 33, 33, 33, 12});
}

TYPED_TEST(OrmqrMask, NetlibHostLoopsSkipInactiveItems) {
    if constexpr (TestFixture::B != Backend::NETLIB) {
        GTEST_SKIP() << "the host loops are NETLIB's vendor family";
    } else {
        constexpr Transpose kC = TestFixture::kC;
        for (const Shape& s : {Shape{"vendor", 11, 7, 6, 12, Side::Left, Transpose::NoTrans},
                               Shape{"vendor", 5, 9, 4, 12, Side::Right, kC},
                               Shape{"vendor", 8, 8, 8, 7, Side::Left, kC}})
            this->check_ormqr(s);
        this->check_orgqr({"vendor", 11, 7, 7, 12});
        this->check_orgqr({"vendor", 6, 6, 6, 9});
    }
}

TYPED_TEST(OrmqrMask, OrmqxCtaLeavesInactiveItemsUntouched) {
    if (!this->gpu()) GTEST_SKIP() << "ormqx_cta is a GPU kernel";
    this->check_cta(7, 5, 40, false);   // P = 8: four partitions per sub-group
    this->check_cta(13, 13, 20, false); // P = 16
    this->check_cta(3, 2, 40, false);   // P = 4
}

// The saturating batch (>= 1024, identical items) for the packed kernel.
TYPED_TEST(OrmqrMask, SaturatingOrmqxCta) {
    if (!this->gpu()) GTEST_SKIP() << "ormqx_cta is a GPU kernel";
    this->check_cta(7, 6, 1024, true);
    this->check_cta(29, 20, 1024, true);
}

}  // namespace
