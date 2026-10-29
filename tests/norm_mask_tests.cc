// SPDX-License-Identifier: MIT
// norm and cond honour a batch mask (W4c), on tests/masked_contract.hh: every norm type (the
// spectral one through the public syev, pinned to a family that honours the mask), and cond through
// inv and norm or through syev. A device-only mask on every pattern, complex Hermitian data, padded
// ld and stride; A and the per-item result are the outputs, and A must keep every byte. The heavy
// tier adds the saturating identical-item batch for the one-work-group-per-item norm kernel.
// evidence: docs/design/masked-batches.md#masked-batches-tests
#include <gtest/gtest.h>

#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/extra.hh>
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/error.hh>
#include <batchlas/util/sycl-vector.hh>

#include <src/ops/syev/choice.hh>
#include <tests/masked_contract.hh>
#include <tests/test_support/storage.hh>
#include <tests/test_utils.hh>

#include <algorithm>
#include <cstring>
#include <complex>
#include <cstdint>
#include <array>
#include <optional>
#include <variant>
#include <span>
#include <string>
#include <vector>

using namespace batchlas;

namespace {

namespace mc = masked_contract;

const char* norm_name(NormType t) {
    switch (t) {
        case NormType::Frobenius: return "Frobenius";
        case NormType::One: return "One";
        case NormType::Inf: return "Inf";
        case NormType::Max: return "Max";
        default: return "Spectral";
    }
}

// Hermitian and diagonally dominant (so inv and the spectral solve are well posed), with a distinct
// value per item; the pads keep the poison.
template <typename T>
void fill_hermitian(Matrix<T>& A, int n, int batch, bool identical) {
    using R = typename base_type<T>::type;
    std::fill_n(A.data_ptr(), std::size_t(A.stride()) * batch, test_support::poison<T>());
    for (int b = 0; b < batch; ++b) {
        const int item = identical ? 0 : b;
        T* a = A.data_ptr() + std::size_t(A.stride()) * b;
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                const int lo = std::min(i, j), hi = std::max(i, j);
                const double re = double(int((lo * 7 + hi * 3 + item * 5) % 17) - 8) / 8;
                double im = double(int((lo * 5 + hi + item * 3) % 11) - 5) / 8;
                if (i < j) im = -im;
                T v = i == j ? T(R(2 * n + item % 7)) : MatrixView<T>::make(re, im);
                a[i + std::size_t(A.ld()) * j] = v;
            }
    }
}

template <typename T, Backend B>
struct Cfg {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};
using Types = typename test_utils::backend_types<Cfg>::type;

template <typename Config>
class NormMask : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    using R = typename base_type<T>::type;
    static constexpr Backend B = Config::BackendVal;
    static constexpr bool kHost = B == Backend::NETLIB;

    // The spectral paths run the public syev: pin a family that honours the mask, so the masked and
    // the unmasked call take one route (Auto may refuse cta for the masked one and walk on).
    using SyevPin = std::optional<select::ScopedPin<ops::syev::SyevChoice>>;
    static void pin_syev(SyevPin& pin, int n) {
        if (!kHost && n <= 32) pin.emplace("syev", "cta_fused", select::StrictPin{});
    }

    void check_norm(NormType type, int n, int batch, bool identical = false) {
        SCOPED_TRACE(std::string("norm ") + norm_name(type) + " n " + std::to_string(n) + " batch " +
                     std::to_string(batch) + (identical ? " identical" : ""));
        Queue& q = *this->ctx;
        Matrix<T> A = test_support::padded<T>(n, n, batch);
        auto out = UnifiedVector<R>(static_cast<std::size_t>(batch));
        SyevPin pin;
        if (type == NormType::Spectral) pin_syev(pin, n);
        mc::Case c;
        c.batch = batch;
        c.identical_items = identical;
        c.prepare = [&] {
            fill_hermitian(A, n, batch, identical);
            std::fill_n(out.data(), out.size(), R(-77));
        };
        c.call = [&](Span<const std::uint8_t> m, Span<std::byte>) {
            (void)norm<T, MatrixFormat::Dense>(q, A.view().with_batch_mask(m), type, out.to_span());
        };
        c.outputs = [&] {
            return std::vector<mc::Output>{mc::Output::matrix("A", A.view()),
                                           mc::Output::span<R>("norm", std::span<const R>(out.data(), out.size()), 1)};
        };
        mc::check(q, c);
    }
};

TYPED_TEST_SUITE(NormMask, Types);

TYPED_TEST(NormMask, EveryNormType) {
    for (NormType t : {NormType::Frobenius, NormType::One, NormType::Inf, NormType::Max})
        for (int n : {5, 33}) this->check_norm(t, n, 6);
}

TYPED_TEST(NormMask, SpectralThroughSyev) { this->check_norm(NormType::Spectral, 8, 6); }

// 1024 identical items: every active item bit-identical to the first.
TYPED_TEST(NormMask, SaturatingIdenticalBatch) {
    for (NormType t : {NormType::Frobenius, NormType::One, NormType::Inf}) this->check_norm(t, 12, 1024, true);
}

template <typename Config>
class CondMask : public NormMask<Config> {
protected:
    using T = typename Config::ScalarType;
    using R = typename base_type<T>::type;
    static constexpr Backend B = Config::BackendVal;

    void check_cond(NormType type, int n, int batch) {
        if constexpr (RealScalar<T>) {
            SCOPED_TRACE(std::string("cond ") + norm_name(type) + " n " + std::to_string(n));
            Queue& q = *this->ctx;
            Matrix<T> A = test_support::padded<T>(n, n, batch);
            auto out = UnifiedVector<T>(static_cast<std::size_t>(batch));
            typename NormMask<Config>::SyevPin pin;
            if (type == NormType::Spectral) this->pin_syev(pin, n);
            mc::Case c;
            c.batch = batch;
            c.prepare = [&] {
                fill_hermitian(A, n, batch, false);
                std::fill_n(out.data(), out.size(), T(-77));
            };
            c.call = [&](Span<const std::uint8_t> m, Span<std::byte> ws) {
                (void)cond<B, T, MatrixFormat::Dense>(q, A.view().with_batch_mask(m), type, out.to_span(), ws);
            };
            c.buffer_size = [&](Span<const std::uint8_t> m) {
                return cond_buffer_size<B, T, MatrixFormat::Dense>(q, A.view().with_batch_mask(m), type);
            };
            c.outputs = [&] {
                return std::vector<mc::Output>{mc::Output::matrix("A", A.view()),
                                               mc::Output::span<T>("cond", std::span<const T>(out.data(), out.size()), 1)};
            };
            // On a GPU inv's children (getrf, getri, trsm, gemm) are pinned `native` for both runs:
            // the unmasked reference would otherwise route one to the vendor, which refuses a mask.
            std::array<std::optional<select::ScopedPin<std::monostate>>, 6> children;
            if (this->ctx->device().type == DeviceType::GPU) {
                const char* ops[] = {"getrf", "getri", "trsm", "trmm", "gemm", "gemv"};
                for (std::size_t i = 0; i < children.size(); ++i) children[i].emplace(ops[i], "native");
            }
            mc::check(q, c);
        }
    }
};

TYPED_TEST_SUITE(CondMask, Types);

TYPED_TEST(CondMask, ThroughInvAndNorm) {
    if constexpr (!RealScalar<typename TestFixture::T>) GTEST_SKIP() << "cond is real-only";
    for (NormType t : {NormType::Frobenius, NormType::One, NormType::Inf}) this->check_cond(t, 8, 6);
}

TYPED_TEST(CondMask, SpectralThroughSyev) {
    if constexpr (!RealScalar<typename TestFixture::T>) GTEST_SKIP() << "cond is real-only";
    this->check_cond(NormType::Spectral, 8, 6);
}

}  // namespace
