// SPDX-License-Identifier: MIT
// geqrf honours a batch mask in every family (W2c), each pinned and reached, on
// tests/masked_contract.hh: an inactive item's A and tau are bit-for-bit untouched, an active one
// equals the unmasked call on the same pin. Device-only masks, complex data, padded ld and stride.
// tiny folds the mask into a packed partition (holes inside a work-group); cta returns per sub-group
// (packed) or per work-group; blocked slices the mask per part (chunk patterns) through its panel
// leaves (resident, global, register), pack_v, larft and the gemm children; NETLIB's host loop skips.
// The Saturating case adds the >= 1024 identical-item batch for the packed kernels.
// evidence: docs/design/masked-batches.md#masked-batches-tests
#include <gtest/gtest.h>

#include <batchlas/blas/functions/geqrf.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-vector.hh>

#include <src/ops/geqrf/choice.hh>
#include <src/ops/geqrf/geqrf_native.hh>
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

namespace og = ops::geqrf;
namespace mc = masked_contract;

struct Shape {
    std::string spelling;  ///< the pinned choice
    int m, n, batch;
    int wg_items = 0;        ///< items per packed work-group: the hole patterns
    int chunk_items = 0;     ///< blocked's part size: the chunk-edge patterns
    bool identical = false;  ///< every item the same input (the saturating case)
    const char* leaf = nullptr;  ///< BATCHLAS_GEQRF_LEAF for blocked
};

std::string label(const Shape& s) {
    return s.spelling + " " + std::to_string(s.m) + "x" + std::to_string(s.n) + " batch " + std::to_string(s.batch) +
           (s.leaf ? std::string(" leaf ") + s.leaf : "");
}

template <typename T, Backend B>
struct Problem {
    Queue& q;
    Shape s;
    Matrix<T> A;
    UnifiedVector<T> tau;

    int k() const { return std::min(s.m, s.n); }
    Problem(Queue& queue, const Shape& shape)
        : q(queue), s(shape), A(test_support::padded<T>(s.m, s.n, s.batch)),
          tau(std::size_t(std::min(s.m, s.n)) * s.batch) {}

    void fill() {
        T* p = A.data_ptr();
        const std::size_t per = std::size_t(A.stride());
        for (std::size_t e = 0; e < per * std::size_t(s.batch); ++e) {
            const std::size_t item = s.identical ? 0 : e / per, off = e % per;
            const double re = double(int((off * 7 + item * 3 + 1) % 23) - 11) / 8;
            const double im = double(int((off * 5 + item + 4) % 13) - 6) / 8;
            p[e] = MatrixView<T>::make(re, im);
        }
        std::fill(tau.begin(), tau.end(), test_support::poison<T>());
    }
    mc::Case make_case() {
        mc::Case c;
        c.batch = s.batch;
        c.wg_items = s.wg_items;
        c.chunk_items = s.chunk_items;
        c.identical_items = s.identical;
        c.prepare = [this] { fill(); };
        c.call = [this](Span<const std::uint8_t> mask, Span<std::byte> ws) {
            geqrf<B, T>(q, A.view().with_batch_mask(mask), tau.to_span(), ws);
        };
        c.buffer_size = [this](Span<const std::uint8_t> mask) {
            return geqrf_buffer_size<B, T>(q, A.view().with_batch_mask(mask), tau.to_span());
        };
        c.outputs = [this] {
            return std::vector<mc::Output>{mc::Output::matrix("A", A.view()),
                                           mc::Output::span<T>("tau", {tau.data(), tau.size()}, std::size_t(k()))};
        };
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
class GeqrfMask : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    static constexpr Backend B = Config::BackendVal;

    bool gpu() const { return this->ctx->device().type == DeviceType::GPU; }

    void check(const Shape& s) {
        SCOPED_TRACE(label(s));
        std::optional<ScopedEnvVar> leaf;
        if (s.leaf) leaf.emplace("BATCHLAS_GEQRF_LEAF", s.leaf);
        Problem<T, B> p(*this->ctx, s);
        mc::check<og::GeqrfChoice>(*this->ctx, "geqrf", s.spelling, p.make_case());
    }
    // Items per tiny work-group at order n (0: the tier does not take n here).
    int tiny_items(int n) { return int(sycl_geqrf::geqrf_tiny_debug_launch<T>(*this->ctx, n) & 0xffffu); }
    // The blocked panel leaf at m x n: 1 resident, 2 global, 3 register.
    int blocked_leaf(int m, int n) { return int(sycl_geqrf::geqrf_blocked_debug_params<T>(*this->ctx, m, n) >> 16); }

    std::vector<Shape> tiny_shapes(int batch, bool identical) {
        std::vector<Shape> v;
        for (int n : {7, 13, 29})
            if (const int g = tiny_items(n); g > 0) v.push_back({"tiny", n, n, batch, g, 0, identical});
        return v;
    }
    // Packed (m, n <= 32: one item per sub-group, G per work-group) and one work-group per item.
    static std::vector<Shape> cta_shapes(int batch, bool identical) {
        return {{"cta", 20, 12, batch, 4, 0, identical}, {"cta", 9, 17, batch, 4, 0, identical},
                {"cta", 40, 24, batch, 0, 0, identical}};
    }
};

TYPED_TEST_SUITE(GeqrfMask, Types);

TYPED_TEST(GeqrfMask, TinyAndCtaLeaveInactiveItemsUntouched) {
    if (!this->gpu()) GTEST_SKIP() << "tiny and cta are GPU kernels";
    std::vector<Shape> v = this->tiny_shapes(40, false);
    ASSERT_FALSE(v.empty()) << "no tiny bucket runs here: the case would pass vacuously";
    for (const Shape& s : TestFixture::cta_shapes(40, false)) v.push_back(s);
    for (const Shape& s : v) this->check(s);
}

// Every panel leaf the driver reaches (resident, global, register by Auto and by the env pin), a
// wide and a tall form, and parts of 8 items: each part takes its mask slice.
TYPED_TEST(GeqrfMask, BlockedSlicesTheMaskPerPartThroughEveryLeaf) {
    if (!this->gpu()) GTEST_SKIP() << "blocked is a GPU driver";
    std::vector<Shape> v{{"blocked", 70, 50, 24, 0, 8}, {"blocked", 40, 70, 20}, {"blocked", 300, 40, 12}};
    v.push_back({"blocked", 70, 50, 12, 0, 0, false, "reg"});
    std::vector<int> leaves;
    for (const Shape& s : v) leaves.push_back(s.leaf ? 3 : this->blocked_leaf(s.m, s.n));
    EXPECT_NE(std::find(leaves.begin(), leaves.end(), 1), leaves.end()) << "no shape takes the resident leaf";
    EXPECT_NE(std::find(leaves.begin(), leaves.end(), 2), leaves.end()) << "no shape takes the global leaf";
    for (const Shape& s : v) this->check(s);
}

TYPED_TEST(GeqrfMask, NetlibHostLoopSkipsInactiveItems) {
    if constexpr (TestFixture::B != Backend::NETLIB) {
        GTEST_SKIP() << "the host loop is NETLIB's vendor family";
    } else {
        for (const Shape& s : {Shape{"vendor", 9, 6, 12}, Shape{"vendor", 5, 8, 12}, Shape{"vendor", 16, 16, 7}})
            this->check(s);
    }
}

// The saturating batch (>= 1024, identical items) for the packed and local-memory kernels.
TYPED_TEST(GeqrfMask, SaturatingPackedKernels) {
    if (!this->gpu()) GTEST_SKIP() << "tiny and cta are GPU kernels";
    std::vector<Shape> v = this->tiny_shapes(1024, true);
    for (const Shape& s : TestFixture::cta_shapes(1024, true)) v.push_back(s);
    v.push_back({"blocked", 70, 50, 1024, 0, 0, true});
    for (const Shape& s : v) this->check(s);
}

}  // namespace
