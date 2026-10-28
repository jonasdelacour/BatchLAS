// SPDX-License-Identifier: MIT
// The batch mask as a view property (W0): it travels through every view transform, call_mask()
// agrees across operands, and the utilities leave an inactive item bit-for-bit untouched with a
// device-only mask (sycl::malloc_device, written by a kernel), so any host read would fault.
#include <gtest/gtest.h>

#include <batchlas/blas/extra.hh>
#include <batchlas/blas/linalg-ops.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-vector.hh>

#include <src/kernels/common/info_span.hh>
#include <src/queue.hh>
#include <src/util/batch_mask.hh>

#include <complex>
#include <cstdint>
#include <cstring>
#include <functional>
#include <vector>

using namespace batchlas;

namespace {

constexpr int kBatch = 37;

// Items 0, 1, 4, 7, ... are inactive: the first item, interior holes, and an active last item.
bool pattern_active(int b) { return b != 0 && b % 3 != 1; }

UnifiedVector<std::uint8_t> host_mask(int batch, bool all_active = false) {
    UnifiedVector<std::uint8_t> m(static_cast<std::size_t>(batch));
    for (int b = 0; b < batch; ++b) m[b] = (all_active || pattern_active(b)) ? 1 : 0;
    return m;
}

// Device-only bytes written by a kernel: any host dereference of them faults on a GPU.
struct DeviceMask {
    Queue& ctx;
    std::uint8_t* ptr = nullptr;
    int batch = 0;
    DeviceMask(Queue& q, int n, bool all_active = false) : ctx(q), batch(n) {
        ptr = sycl::malloc_device<std::uint8_t>(static_cast<std::size_t>(n), *ctx.operator->());
        std::uint8_t* p = ptr;
        ctx->parallel_for(sycl::range<1>(static_cast<std::size_t>(n)), [=](sycl::id<1> id) {
            const int b = static_cast<int>(id[0]);
            p[b] = (all_active || (b != 0 && b % 3 != 1)) ? std::uint8_t(1) : std::uint8_t(0);
        });
        ctx.wait();
    }
    ~DeviceMask() { sycl::free(ptr, *ctx.operator->()); }
    Span<const std::uint8_t> span() const { return {ptr, static_cast<std::size_t>(batch)}; }
};

template <typename T>
T poison(std::size_t k) {
    const double re = 1000.0 + 0.25 * static_cast<double>(k % 4093);
    if constexpr (is_std_complex_v<T>) return T(batchlas::float_t<T>(re), batchlas::float_t<T>(-0.5 * re));
    else return T(re);
}

template <typename T>
void fill_poison(const MatrixView<T>& A, std::size_t salt = 0) {
    auto d = A.data();
    for (std::size_t k = 0; k < d.size(); ++k) d[k] = poison<T>(k + salt);
}

template <typename T>
std::vector<T> snapshot(const MatrixView<T>& A) { return std::vector<T>(A.data().begin(), A.data().end()); }

template <typename T>
bool item_bytes_equal(const T* a, const T* b, std::size_t n) { return std::memcmp(a, b, n * sizeof(T)) == 0; }

// Inactive item b: its whole stride range equals the poisoned input. Active: equals the unmasked run.
template <typename T>
void expect_masked_items(const std::vector<T>& before, const std::vector<T>& unmasked, const MatrixView<T>& got,
                         bool all_active, const char* what) {
    const std::size_t stride = static_cast<std::size_t>(got.stride());
    const auto d = got.data();
    for (int b = 0; b < got.batch_size(); ++b) {
        const std::size_t off = static_cast<std::size_t>(b) * stride;
        const std::size_t len = std::min(stride, d.size() - off);
        const bool active = all_active || pattern_active(b);
        const auto& ref = active ? unmasked : before;
        EXPECT_TRUE(item_bytes_equal(d.data() + off, ref.data() + off, len))
            << what << ": item " << b << (active ? " (active) differs from the unmasked call"
                                                 : " (inactive) was written");
    }
}

template <typename T>
using MatOp = std::function<void(Queue&, const MatrixView<T>&)>;

// Runs op on a poisoned strided batch unmasked, then masked (device-only mask), and compares.
template <typename T>
void check_utility(Queue& ctx, const MatOp<T>& op, const char* what, int rows, int cols, bool packed = false) {
    const int ld = packed ? rows : rows + 3;
    const int stride = packed ? rows * cols : ld * cols + 5;
    for (bool all_active : {false, true}) {
        Matrix<T> ref(rows, cols, kBatch, ld, stride);
        Matrix<T> got(rows, cols, kBatch, ld, stride);
        fill_poison<T>(ref);
        fill_poison<T>(got);
        const auto before = snapshot<T>(got);
        op(ctx, ref.view());
        ctx.wait();
        DeviceMask mask(ctx, kBatch, all_active);
        op(ctx, got.view().with_batch_mask(mask.span()));
        ctx.wait();
        expect_masked_items<T>(before, snapshot<T>(ref), got, all_active, what);
    }
}

template <typename T>
class BatchMaskUtilities : public ::testing::Test {
protected:
    Queue ctx{Device::default_device()};
};

using MaskTypes = ::testing::Types<float, double, std::complex<float>>;
TYPED_TEST_SUITE(BatchMaskUtilities, MaskTypes);

}  // namespace

// ---- representation and propagation ------------------------------------------------------------

TEST(BatchMaskView, WithBatchMaskChecksTheLengthOnlyAndEmptyClears) {
    Matrix<float> A(4, 3, 5);
    EXPECT_FALSE(A.has_batch_mask());
    auto m = host_mask(5);
    auto v = A.view().with_batch_mask(m.to_span());
    EXPECT_TRUE(v.has_batch_mask());
    EXPECT_EQ(v.batch_mask().data(), m.data());
    EXPECT_EQ(v.batch_mask().size(), 5u);
    EXPECT_FALSE(A.has_batch_mask());  // a copy is masked, never the source
    EXPECT_FALSE(v.with_batch_mask(Span<const std::uint8_t>()).has_batch_mask());
    auto short_mask = host_mask(4);
    EXPECT_THROW((void)A.view().with_batch_mask(short_mask.to_span()), batchlas::invalid_argument);
    auto long_mask = host_mask(6);
    EXPECT_THROW((void)A.view().with_batch_mask(long_mask.to_span()), batchlas::invalid_argument);
    // A raw construction starts unmasked.
    MatrixView<float> raw(A.data_ptr(), 4, 3, 4, 12, 5);
    EXPECT_FALSE(raw.has_batch_mask());
}

TEST(BatchMaskView, EveryViewTransformKeepsTheMask) {
    constexpr int rows = 6, cols = 5, ld = 8, stride = 45, batch = 7;
    Matrix<float> A(rows, cols, batch, ld, stride);
    auto m = host_mask(batch);
    const auto v = A.view().with_batch_mask(m.to_span());
    const auto* base = m.data();

    const auto range = v.batch_range(2, 4);
    EXPECT_EQ(range.batch_mask().data(), base + 2);
    EXPECT_EQ(range.batch_mask().size(), 4u);

    for (int b : {0, 3, 6}) {
        const auto item = v.batch_item(b);
        EXPECT_EQ(item.batch_mask().data(), base + b);
        EXPECT_EQ(item.batch_mask().size(), 1u);
        EXPECT_EQ(v[b].batch_mask().data(), base + b);
    }

    const auto sub = v(Slice(1, 4), Slice(2, 5));
    EXPECT_EQ(sub.batch_mask().data(), base);
    EXPECT_EQ(v(Slice(2, 3)).batch_mask().data(), base);
    EXPECT_EQ(v(1, Slice(0, 3)).batch_mask().data(), base);   // row vector
    EXPECT_EQ(v(Slice(0, 4), 2).batch_mask().data(), base);   // column vector

    UnifiedVector<int> ar(batch, 3), ac(batch, 2);
    EXPECT_EQ(v.with_active_dims(ar.to_span(), ac.to_span()).batch_mask().data(), base);

    const MatrixView<float> over(v, 4, 3);
    EXPECT_EQ(over.batch_mask().data(), base);
    EXPECT_THROW((void)MatrixView<float>(v, -1, -1, -1, -1, batch - 1), batchlas::invalid_argument);

    const auto col = v(Slice(0, rows), 1);
    const MatrixView<float> from_vector(col);
    EXPECT_EQ(from_vector.batch_mask().data(), base);

    const auto kv = v.kernel_view();
    EXPECT_EQ(kv.mask_, base);
    EXPECT_EQ(A.kernel_view().mask_, nullptr);
    EXPECT_EQ(kv.batch_item(3).mask_, base + 3);
    EXPECT_EQ(kv(Slice(0, 2), Slice(0, 2)).mask_, base);
    EXPECT_EQ(kv(1, Slice(0, 2)).batch_mask().data(), base);
    EXPECT_EQ(kv(Slice(0, 2), 1).batch_mask().data(), base);

    const auto converted = v.astype<double>();
    ASSERT_TRUE(converted.has_batch_mask());
    for (int b = 0; b < batch; ++b) EXPECT_EQ(converted.batch_mask()[b], m[b]);
}

TEST(BatchMaskView, VectorViewKeepsTheMask) {
    UnifiedVector<float> mem(5 * 9, 0.0f);
    auto m = host_mask(9);
    const VectorView<float> x(mem.data(), 5, 9, 1, 5);
    EXPECT_FALSE(x.has_batch_mask());
    const auto xm = x.with_batch_mask(m.to_span());
    EXPECT_EQ(xm.batch_mask().data(), m.data());
    EXPECT_EQ(xm(Slice(1, 4)).batch_mask().data(), m.data());
    EXPECT_EQ(xm.batch_item(4).batch_mask().data(), m.data() + 4);
    EXPECT_EQ(xm.batch_item(4).active(0), pattern_active(4));
    EXPECT_THROW((void)x.with_batch_mask(host_mask(8).to_span()), batchlas::invalid_argument);
}

TEST(BatchMaskView, MatrixOwnsItsMaskAndCopiesCarryIt) {
    constexpr int batch = 6;
    Matrix<float> A(3, 3, batch);
    auto store = A.enable_batch_mask();
    ASSERT_EQ(store.size(), std::size_t(batch));
    for (int b = 0; b < batch; ++b) EXPECT_EQ(store[b], 1);
    for (int b = 0; b < batch; ++b) store[b] = pattern_active(b) ? 1 : 0;
    EXPECT_TRUE(A.has_batch_mask());
    EXPECT_EQ(A.view().batch_mask().data(), store.data());
    EXPECT_EQ(static_cast<const MatrixView<float>&>(A).batch_mask().data(), store.data());

    Matrix<float> copy(A);
    ASSERT_TRUE(copy.has_batch_mask());
    EXPECT_NE(copy.batch_mask().data(), store.data());  // deep
    for (int b = 0; b < batch; ++b) EXPECT_EQ(copy.batch_mask()[b], store[b]);

    const auto cloned = A.clone();
    ASSERT_TRUE(cloned.has_batch_mask());
    for (int b = 0; b < batch; ++b) EXPECT_EQ(cloned.batch_mask()[b], store[b]);

    const auto* p = store.data();
    Matrix<float> moved(std::move(A));
    EXPECT_EQ(moved.batch_mask().data(), p);
    EXPECT_EQ(moved.view().batch_mask().data(), p);
    EXPECT_FALSE(A.has_batch_mask());  // NOLINT(bugprone-use-after-move): moved-from is empty

    const auto converted = moved.astype<double>();
    ASSERT_TRUE(converted.has_batch_mask());
    for (int b = 0; b < batch; ++b) EXPECT_EQ(converted.batch_mask()[b], store[b]);
}

// The kernel side: active(b), batch_item(b).active(0) and the row view's active(b), from a device-only mask.
TEST(BatchMaskView, KernelViewReadsADeviceOnlyMask) {
    Queue ctx(Device::default_device());
    Matrix<float> A(4, 4, kBatch);
    DeviceMask mask(ctx, kBatch);
    const auto kv = A.view().with_batch_mask(mask.span()).kernel_view();
    const auto plain = A.kernel_view();
    UnifiedVector<int> out(std::size_t(4) * kBatch, -1);
    int* o = out.data();
    ctx->parallel_for(sycl::range<1>(kBatch), [=](sycl::id<1> id) {
        const int b = static_cast<int>(id[0]);
        o[b] = kv.active(b);
        o[kBatch + b] = kv.batch_item(b).active(0);
        o[2 * kBatch + b] = kv(1, Slice(0, 3)).active(b);
        o[3 * kBatch + b] = plain.active(b);
    });
    ctx.wait();
    for (int b = 0; b < kBatch; ++b) {
        EXPECT_EQ(out[b], pattern_active(b) ? 1 : 0) << b;
        EXPECT_EQ(out[kBatch + b], pattern_active(b) ? 1 : 0) << b;
        EXPECT_EQ(out[2 * kBatch + b], pattern_active(b) ? 1 : 0) << b;
        EXPECT_EQ(out[3 * kBatch + b], 1) << b;
    }
}

// ---- call_mask ---------------------------------------------------------------------------------

TEST(BatchMaskCallMask, OneSharedSpanAndBothThrows) {
    constexpr int batch = 5;
    Matrix<float> A(3, 3, batch), B(3, 3, batch), C(3, 3, 1);
    auto m = host_mask(batch);
    auto other = host_mask(batch);  // same contents, different memory: still a different mask
    const auto Am = A.view().with_batch_mask(m.to_span());

    EXPECT_TRUE(detail::call_mask("op", A.view(), B.view()).empty());
    EXPECT_EQ(detail::call_mask("op", Am, B.view()).data(), m.data());  // unmasked operands adopt it
    EXPECT_EQ(detail::call_mask("op", B.view(), Am, Am).data(), m.data());
    EXPECT_EQ(detail::call_mask("op", Am, C.view()).data(), m.data());  // an unmasked broadcast is fine

    try {
        (void)detail::call_mask("gemm", Am, B.view().with_batch_mask(other.to_span()));
        ADD_FAILURE() << "different spans did not throw";
    } catch (const batchlas::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("gemm: operands carry different batch masks"), std::string::npos)
            << e.what();
    }
    // A masked broadcast operand (batch 1 of a batch-5 call) throws.
    UnifiedVector<std::uint8_t> one(1, 1);
    EXPECT_THROW((void)detail::call_mask("op", Am, C.view().with_batch_mask(one.to_span())), batchlas::invalid_argument);

    // A Matrix's own (mutable) mask takes part like a view's.
    Matrix<float> D(3, 3, batch);
    D.enable_batch_mask();
    EXPECT_EQ(detail::call_mask("op", D, B.view()).data(), D.batch_mask().data());
    EXPECT_THROW((void)detail::call_mask("op", D, Am), batchlas::invalid_argument);

    UnifiedVector<float> vmem(3 * batch, 0.0f);
    const VectorView<float> x(vmem.data(), 3, batch);
    EXPECT_EQ(detail::call_mask("op", Am, x).data(), m.data());
    EXPECT_THROW((void)detail::call_mask("op", Am, x.with_batch_mask(other.to_span())), batchlas::invalid_argument);
}

// ---- utilities: inactive items untouched, active items equal to the unmasked call ---------------

TYPED_TEST(BatchMaskUtilities, FillsAndTransforms) {
    using T = TypeParam;
    auto& ctx = this->ctx;
    const T value = poison<T>(17) * T(0.5);
    check_utility<T>(ctx, [](Queue& q, const MatrixView<T>& A) { (void)A.fill_zeros(q); }, "fill_zeros strided", 5, 4);
    check_utility<T>(ctx, [](Queue& q, const MatrixView<T>& A) { (void)A.fill_zeros(q); }, "fill_zeros packed (memset path)", 5, 4, true);
    check_utility<T>(ctx, [=](Queue& q, const MatrixView<T>& A) { (void)A.fill(q, value); }, "fill packed", 5, 4, true);
    check_utility<T>(ctx, [=](Queue& q, const MatrixView<T>& A) { (void)A.fill(q, value); }, "fill strided", 5, 4);
    check_utility<T>(ctx, [](Queue& q, const MatrixView<T>& A) { (void)A.fill_identity(q, T(2)); }, "fill_identity", 5, 5);
    check_utility<T>(ctx, [](Queue& q, const MatrixView<T>& A) { (void)A.fill_diagonal(q, T(3)); }, "fill_diagonal value", 5, 4);
    UnifiedVector<T> diag(std::size_t(4) * kBatch);
    for (std::size_t k = 0; k < diag.size(); ++k) diag[k] = poison<T>(k + 99);
    check_utility<T>(ctx, [&](Queue& q, const MatrixView<T>& A) { (void)A.fill_diagonal(q, diag.to_span(), -1); }, "fill_diagonal per-item span", 5, 4);
    check_utility<T>(ctx, [](Queue& q, const MatrixView<T>& A) { (void)A.fill_triangular(q, Uplo::Lower, T(2), T(-1)); }, "fill_triangular", 5, 5);
    check_utility<T>(ctx, [](Queue& q, const MatrixView<T>& A) { (void)A.fill_triangular_random(q, Uplo::Upper, Diag::NonUnit, 7); }, "fill_triangular_random", 5, 5, true);
    check_utility<T>(ctx, [](Queue& q, const MatrixView<T>& A) { (void)A.fill_tridiag_toeplitz(q); }, "fill_tridiag_toeplitz", 5, 5, true);
    check_utility<T>(ctx, [](Queue& q, const MatrixView<T>& A) { (void)A.fill_random(q, false, 3); }, "fill_random", 5, 4);
    check_utility<T>(ctx, [](Queue& q, const MatrixView<T>& A) { (void)A.fill_random(q, true, 3); }, "fill_random hermitian", 5, 5);
    check_utility<T>(ctx, [](Queue& q, const MatrixView<T>& A) { (void)A.fill_random_spd(q, 5); }, "fill_random_spd", 5, 5);
    check_utility<T>(ctx, [](Queue& q, const MatrixView<T>& A) { (void)A.fill_near_degenerate_hermitian(q, 1e-3, 5); }, "fill_near_degenerate_hermitian", 5, 5);
    check_utility<T>(ctx, [](Queue& q, const MatrixView<T>& A) { (void)A.fill_random_pivoted(q, 5); }, "fill_random_pivoted", 5, 5);
    check_utility<T>(ctx, [](Queue& q, const MatrixView<T>& A) { (void)A.fill_graded_hermitian(q, 3.0, 5); }, "fill_graded_hermitian", 5, 5);
    check_utility<T>(ctx, [](Queue& q, const MatrixView<T>& A) { (void)A.triangularize(q, Uplo::Upper, Diag::Unit); }, "triangularize", 5, 4);
    check_utility<T>(ctx, [](Queue& q, const MatrixView<T>& A) { (void)A.symmetrize(q, Uplo::Lower); }, "symmetrize", 5, 5);
    check_utility<T>(ctx, [](Queue& q, const MatrixView<T>& A) { (void)A.hermitize(q, Uplo::Upper); }, "hermitize", 5, 5);
    check_utility<T>(ctx, [](Queue& q, const MatrixView<T>& A) { (void)scale(q, T(-3), A); }, "scale", 5, 4);
    check_utility<T>(ctx, [](Queue& q, const MatrixView<T>& A) { (void)linalg::scale(q, A, T(2)); }, "linalg::scale", 5, 4);
}

// Two-operand utilities: the mask sits on the source only, so the destination adopts it.
TYPED_TEST(BatchMaskUtilities, CopiesAndElementwise) {
    using T = TypeParam;
    auto& ctx = this->ctx;
    for (bool packed : {true, false}) {
        Matrix<T> src(5, 4, kBatch, packed ? 5 : 7, packed ? 20 : 31);
        fill_poison<T>(src, 555);
        auto copy_from = [&](Queue& q, const MatrixView<T>& D) {
            const bool masked = D.has_batch_mask();
            const auto S = masked ? src.view().with_batch_mask(D.batch_mask()) : src.view();
            (void)MatrixView<T>::copy(q, D.with_batch_mask(Span<const std::uint8_t>()), S);
        };
        check_utility<T>(ctx, copy_from, packed ? "copy packed (memcpy path)" : "copy strided", 5, 4, packed);
    }
    Matrix<T> B(5, 4, kBatch);
    fill_poison<T>(B, 31);
    check_utility<T>(ctx, [&](Queue& q, const MatrixView<T>& C) { (void)linalg::elementwise_into<T, linalg::BinaryOp::Multiply>(q, B, B, C); }, "elementwise_into", 5, 4);
    check_utility<T>(ctx, [&](Queue& q, const MatrixView<T>& C) { (void)linalg::axpby_into<T>(q, T(2), B, T(-1), C, C); }, "axpby_into", 5, 4);
    check_utility<T>(ctx, [&](Queue& q, const MatrixView<T>& C) { (void)linalg::triangular_mask_into<T>(q, B, C, Uplo::Lower, 0); }, "triangular_mask_into", 5, 4);
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
        Matrix<T> At(4, 5, kBatch);
        fill_poison<T>(At, 77);
        check_utility<T>(ctx, [&](Queue& q, const MatrixView<T>& D) { (void)transpose(q, At.view(), D); }, "transpose", 5, 4, true);
    }
}

TYPED_TEST(BatchMaskUtilities, VectorCopy) {
    using T = TypeParam;
    auto& ctx = this->ctx;
    constexpr int n = 6, stride = 9;
    UnifiedVector<T> src(std::size_t(stride) * kBatch), dst(std::size_t(stride) * kBatch);
    for (std::size_t k = 0; k < src.size(); ++k) { src[k] = poison<T>(k + 3); dst[k] = poison<T>(k + 1000); }
    const std::vector<T> before(dst.begin(), dst.end());
    DeviceMask mask(ctx, kBatch);
    const VectorView<T> s(src.data(), n, kBatch, 1, stride), d(dst.data(), n, kBatch, 1, stride);
    (void)VectorView<T>::copy(ctx, d, s.with_batch_mask(mask.span()));
    ctx.wait();
    for (int b = 0; b < kBatch; ++b) {
        const std::size_t off = std::size_t(b) * stride;
        const auto& ref = pattern_active(b) ? std::vector<T>(src.begin(), src.end()) : before;
        EXPECT_TRUE(item_bytes_equal(dst.data() + off, ref.data() + off, n)) << "item " << b;
    }
}

// Inactive items compare true even where they differ; an active difference still fails.
TYPED_TEST(BatchMaskUtilities, AllCloseSkipsInactiveItems) {
    using T = TypeParam;
    auto& ctx = this->ctx;
    Matrix<T> A(4, 3, kBatch), B(4, 3, kBatch);
    fill_poison<T>(A);
    fill_poison<T>(B);
    B(2, 1, 1) = B(2, 1, 1) * T(3);  // item 1 is inactive
    DeviceMask mask(ctx, kBatch);
    EXPECT_FALSE(MatrixView<T>::all_close(ctx, A, B));
    EXPECT_TRUE(MatrixView<T>::all_close(ctx, A.view().with_batch_mask(mask.span()), B));
    B(0, 0, 2) = B(0, 0, 2) * T(3);  // item 2 is active
    EXPECT_FALSE(MatrixView<T>::all_close(ctx, A.view().with_batch_mask(mask.span()), B));

    UnifiedVector<T> x(std::size_t(4) * kBatch), y(std::size_t(4) * kBatch);
    for (std::size_t k = 0; k < x.size(); ++k) x[k] = y[k] = poison<T>(k);
    y[4 * 4 + 1] = y[4 * 4 + 1] * T(3);  // item 4 is inactive
    const VectorView<T> xv(x.data(), 4, kBatch), yv(y.data(), 4, kBatch);
    EXPECT_FALSE(VectorView<T>::all_close(ctx, xv, yv));
    EXPECT_TRUE(VectorView<T>::all_close(ctx, xv.with_batch_mask(mask.span()), yv));
}

TEST(BatchMaskInfo, MaskedClearAndFromFlagsLeaveInactiveSlotsUntouched) {
    Queue ctx(Device::default_device());
    DeviceMask mask(ctx, kBatch);
    UnifiedVector<int32_t> info(kBatch, 77), flags(kBatch, 0);
    for (int b = 0; b < kBatch; ++b) flags[b] = b % 2 ? 5 : 0;

    detail::info_clear(ctx, info.to_span(), kBatch, mask.span());
    ctx.wait();
    for (int b = 0; b < kBatch; ++b) EXPECT_EQ(info[b], pattern_active(b) ? 0 : 77) << b;

    for (int b = 0; b < kBatch; ++b) info[b] = 77;
    detail::info_from_flags(ctx, info.to_span(), flags.data(), kBatch, false, mask.span());
    ctx.wait();
    for (int b = 0; b < kBatch; ++b) EXPECT_EQ(info[b], pattern_active(b) ? flags[b] : 77) << b;

    for (int b = 0; b < kBatch; ++b) info[b] = 77;
    detail::info_from_flags(ctx, info.to_span(), flags.data(), kBatch, true, mask.span());
    ctx.wait();
    for (int b = 0; b < kBatch; ++b) EXPECT_EQ(info[b], pattern_active(b) ? (flags[b] ? 0 : 1) : 77) << b;

    // An empty mask is the unmasked clear: every slot.
    for (int b = 0; b < kBatch; ++b) info[b] = 77;
    detail::info_clear(ctx, info.to_span(), kBatch, Span<const std::uint8_t>());
    ctx.wait();
    for (int b = 0; b < kBatch; ++b) EXPECT_EQ(info[b], 0) << b;
}

TEST(BatchMaskView, SparseRandomFillRefusesAMaskedView) {
    Queue ctx(Device::default_device());
    auto A = Matrix<float, MatrixFormat::CSR>::RandomSparseHermitian(8, 0.3f, 4);
    auto m = host_mask(4);
    EXPECT_THROW((void)A.view().with_batch_mask(m.to_span()).fill_random_sparse_hermitian(ctx, 0.3f), batchlas::unsupported);
}
