// SPDX-License-Identifier: MIT
// getrf honours a batch mask in every family that runs it (W2b), each pinned and reached, on
// tests/masked_contract.hh: every mask pattern (the packed work-group holes for tiny's partitions
// and cta's sub-group tiles), a device-only mask, complex data, padded ld and stride, a singular
// item (info > 0) and pivots that move. An inactive item's A, pivots and info are bit-identical to
// the poisoned input. Device: tiny, cta (square and m x n); NETLIB: the LAPACKE host loop (I4).
// blocked keeps refusing until trsm honours the mask, and a strided vendor refuses permanently.
// evidence: docs/design/masked-batches.md#masked-batches-tests
#include <gtest/gtest.h>

#include <batchlas/blas/functions/getrf.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/error.hh>
#include <batchlas/util/sycl-vector.hh>

#include <src/ops/getrf/choice.hh>
#include <src/ops/getrf/getrf_native.hh>
#include <src/select/vendor.hh>
#include <src/util/resident_capacity.hh>
#include <tests/masked_contract.hh>
#include <tests/test_support/storage.hh>
#include <tests/test_utils.hh>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

using namespace batchlas;

namespace {

namespace mc = masked_contract;
using GetrfPin = select::ScopedPin<ops::getrf::GetrfChoice>;

struct Shape {
    std::string spelling;
    int m, n, batch;
    int wg_items = 0;        ///< items per packed work-group: the hole patterns
    bool identical = false;  ///< every item the same input (the saturating case)
};

std::string label(const Shape& s) {
    return s.spelling + " " + std::to_string(s.m) + "x" + std::to_string(s.n) + " batch " + std::to_string(s.batch);
}

// A well-spread value in [-1, 1] per (item, element, salt): pivots move in every item.
inline double spread(std::size_t item, std::size_t e, int salt) {
    std::uint64_t h = (item * 0x9E3779B97F4A7C15ull) ^ (e * 0xC2B2AE3D27D4EB4Full) ^ std::uint64_t(salt * 7919);
    h ^= h >> 29;
    h *= 0xBF58476D1CE4E5B9ull;
    h ^= h >> 32;
    return double(int(h % 2001) - 1000) / 1000.0;
}

template <typename T, Backend B>
struct Problem {
    Queue& q;
    Shape s;
    Matrix<T> A;
    UnifiedVector<std::int64_t> piv;
    UnifiedVector<std::int32_t> info;

    Problem(Queue& queue, const Shape& shape)
        : q(queue), s(shape), A(test_support::padded<T>(s.m, s.n, s.batch)),
          piv(std::size_t(std::min(s.m, s.n)) * s.batch), info(std::size_t(s.batch)) {}

    // NETLIB writes int64 pivots; every native family packs 1-based int32 in the span's first half.
    static constexpr bool kInt64Pivots = B == Backend::NETLIB;

    void fill() {
        T* p = A.data_ptr();
        const std::size_t per = std::size_t(A.stride()), ld = std::size_t(A.ld());
        for (std::size_t b = 0; b < std::size_t(s.batch); ++b)
            for (std::size_t j = 0; j < std::size_t(s.n); ++j)
                for (std::size_t i = 0; i < std::size_t(s.m); ++i) {
                    const std::size_t item = s.identical ? 0 : b, e = i + j * ld;
                    // Item 5 has a zero third column: info = 3 there (an active item in the masks).
                    const bool zero = !s.identical && b == 5 && j == 2;
                    p[b * per + e] = zero ? T(0) : MatrixView<T>::make(spread(item, e, 1), spread(item, e, 2));
                }
    }
    mc::Case make_case() {
        mc::Case c;
        c.batch = s.batch;
        c.wg_items = s.wg_items;
        c.identical_items = s.identical;
        c.prepare = [this] {
            std::fill_n(A.data_ptr(), std::size_t(A.stride()) * s.batch, test_support::poison<T>());
            fill();
            std::fill(piv.begin(), piv.end(), std::int64_t(0x5A5A5A5A5A5A5A5A));
            std::fill(info.begin(), info.end(), std::int32_t(-77));
        };
        c.call = [this](Span<const std::uint8_t> mask, Span<std::byte> ws) {
            getrf<B, T>(q, A.view().with_batch_mask(mask), piv.to_span(), ws, info.to_span());
        };
        c.buffer_size = [this](Span<const std::uint8_t> mask) {
            return getrf_buffer_size<B, T>(q, A.view().with_batch_mask(mask));
        };
        c.outputs = [this] {
            const std::size_t mn = std::size_t(std::min(s.m, s.n));
            std::vector<mc::Output> out{mc::Output::matrix("A", A.view()),
                                        mc::Output::span<std::int32_t>("info", {info.data(), info.size()}, 1)};
            if constexpr (kInt64Pivots)
                out.push_back(mc::Output::span<std::int64_t>("pivots", {piv.data(), piv.size()}, mn));
            else
                out.push_back(mc::Output::span<std::int32_t>(
                    "pivots", {reinterpret_cast<const std::int32_t*>(piv.data()), mn * s.batch}, mn));
            return out;
        };
        return c;
    }
};

template <typename Config>
class GetrfMask : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    static constexpr Backend B = Config::BackendVal;

    bool device() const { return this->ctx->device().type == DeviceType::GPU && B != Backend::NETLIB; }

    void check(const Shape& s) {
        SCOPED_TRACE(label(s));
        Problem<T, B> p(*this->ctx, s);
        mc::check<ops::getrf::GetrfChoice>(*this->ctx, "getrf", s.spelling, p.make_case());
    }
    // Packed work-group width of the tiny partitions (64 lanes / the lane bucket N) and of cta.
    static int tiny_items(int n) { return 64 / (n <= 4 ? 4 : n <= 8 ? 8 : n <= 16 ? 16 : 32); }
    int cta_items(int m, int n) {
        return int(sycl_getrf::getrf_cta_debug_launch<T>(*this->ctx, m, n) & 0xFFFFu);
    }
    std::size_t slm_budget() const {
        return resident::device_slm_budget(this->ctx->device().get_property(DeviceProperty::LOCAL_MEM_SIZE));
    }
    int cta_cap() const { return sycl_getrf::getrf_cta_max_n_for_slm<T>(slm_budget()); }
    bool cta_fits(int m, int n) const { return sycl_getrf::getrf_cta_fits_rect<T>(m, n, slm_budget()); }
};

using Types = typename test_utils::backend_types<test_support::Config>::type;
TYPED_TEST_SUITE(GetrfMask, Types);

// Every family that runs on this backend, at a pivoting shape with a singular item.
TYPED_TEST(GetrfMask, EveryFamilyLeavesInactiveItemsUntouched) {
    constexpr Backend B = TestFixture::B;
    if (!this->device()) {
        if constexpr (B != Backend::NETLIB) GTEST_SKIP() << "no native family and no host loop here";
        if (!select::kHasNetlib) GTEST_SKIP() << "no LAPACKE in this build";
        this->check({"vendor", 7, 7, 12});
        this->check({"vendor", 9, 6, 9});   // m > n: min(m, n) pivots per item
        this->check({"vendor", 5, 8, 9});   // m < n
        return;
    }
    using T = typename TestFixture::T;
    for (int n : {3, 7, 13, 29}) {
        if (n > sycl_getrf::getrf_tiny_max_n<T>()) continue;
        this->check({"tiny", n, n, 24, TestFixture::tiny_items(n)});
    }
    // The advertised ceiling, a packed small order, and both m x n orientations where they fit.
    const int cap = this->cta_cap();
    ASSERT_GE(cap, 9) << "cta advertises no useful order here";
    this->check({"cta", cap, cap, 12, this->cta_items(cap, cap)});
    this->check({"cta", 9, 9, 24, this->cta_items(9, 9)});
    for (auto [m, n] : {std::pair{cap + 7, cap - 6}, std::pair{cap - 8, cap + 3}})
        if (this->cta_fits(m, n)) this->check({"cta", m, n, 10, this->cta_items(m, n)});
}

// The saturating batch (>= 1024, identical items) for the packed register and SLM tiles.
TYPED_TEST(GetrfMask, SaturatingBatchIsBitIdentical) {
    if (!this->device()) GTEST_SKIP() << "tiny and cta are device families";
    this->check({"tiny", 8, 8, 1024, 8, true});
    this->check({"tiny", 17, 17, 1024, 2, true});
    const int n = std::min(24, this->cta_cap());
    this->check({"cta", n, n, 1024, this->cta_items(n, n), true});
}

// blocked's trailing update calls trsm, which refuses a masked view until W1b: its pin is refused
// before anything is written. A strided vendor (cuBLAS, rocSOLVER) refuses permanently.
TYPED_TEST(GetrfMask, BlockedAndStridedVendorRefuseBeforeWriting) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::B;
    std::vector<std::string> refused;
    if (this->device()) refused.push_back("blocked");
    if constexpr (B != Backend::NETLIB)
        if (select::factorization_vendor_available<B>) refused.push_back("vendor");
    if (refused.empty()) GTEST_SKIP() << "nothing refuses on this backend";
    Queue& q = *this->ctx;
    for (const std::string& spelling : refused) {
        SCOPED_TRACE(spelling);
        const Shape s{spelling, 20, 20, 4};
        Problem<T, B> p(q, s);
        const mc::Case c = p.make_case();
        {
            // The pin runs unmasked first, so the masked throw is the mask's, not the shape's.
            const GetrfPin pin("getrf", spelling, select::StrictPin{});
            c.prepare();
            UnifiedVector<std::byte> ws(std::max<std::size_t>(c.buffer_size({}), 1));
            ASSERT_NO_THROW({
                c.call({}, ws.to_span());
                q.wait();
            });
        }
        c.prepare();
        q.wait();
        const auto before = mc::detail::snapshot(c.outputs(), s.batch);
        const mc::DeviceMask mask(q, mc::patterns(s.batch)[2].bytes);
        const GetrfPin pin("getrf", spelling, select::StrictPin{});
        EXPECT_ANY_THROW({
            UnifiedVector<std::byte> ws(1 << 20);
            c.call(mask.span(), ws.to_span());
            q.wait();
        });
        EXPECT_TRUE(mc::detail::snapshot(c.outputs(), s.batch) == before) << "a refused pin wrote an output";
    }
}

// Auto on a masked view runs a family that honours it (the contract with no pin).
TYPED_TEST(GetrfMask, AutoRunsAMaskedBatch) {
    constexpr Backend B = TestFixture::B;
    if (!this->device() && !(B == Backend::NETLIB && select::kHasNetlib)) GTEST_SKIP() << "no masked route here";
    using T = typename TestFixture::T;
    Problem<T, B> p(*this->ctx, {"auto", 12, 12, 16});
    mc::check(*this->ctx, p.make_case());
}

}  // namespace
