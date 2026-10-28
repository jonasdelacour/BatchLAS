// SPDX-License-Identifier: MIT
// sygvd and sygst honour a batch mask (W3c) in the host backend's LAPACKE family, pinned `vendor`
// and reached, on tests/masked_contract.hh: A, B, w and info of an inactive item untouched, active
// items the unmasked call's. Compose, Blocked and TwoSided keep refusing until their children
// (potrf, trsm, trmm, syr2k/her2k, syev's composites) honour the mask; select_tests' BatchPolicy
// holds that refusal. Device-only masks, complex data, padded ld and stride.
// evidence: docs/design/masked-batches.md#masked-batches-tests
#include <gtest/gtest.h>

#include <batchlas/blas/functions.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-vector.hh>

#include <src/ops/sygst/choice.hh>
#include <src/ops/sygvd/choice.hh>
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
namespace ts = test_support;

template <typename T, Backend B>
struct Cfg {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};
using Types = typename test_utils::backend_types<Cfg>::type;

template <typename Config>
class SygvdMask : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    using R = typename base_type<T>::type;
    static constexpr Backend B = Config::BackendVal;

    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        if (this->HasFatalFailure() || ::testing::Test::IsSkipped()) return;
        if constexpr (B != Backend::NETLIB) GTEST_SKIP() << "every device family still refuses a mask (W1b, W2a, W3a)";
    }

    // A: item-dependent small values over 8 (complex where T is). B: Hermitian positive definite,
    // diagonal n + 2 over off-diagonals of at most 1 / 2, so sygvd's potrf succeeds on every item.
    static void fill(Matrix<T>& A, Matrix<T>& Bm, int n, int batch) {
        for (int b = 0; b < batch; ++b)
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i) {
                    const int k = i + 3 * j + 5 * b;
                    A.data_ptr()[std::size_t(b) * A.stride() + std::size_t(j) * A.ld() + i] =
                        MatrixView<T>::make(double(k % 17 - 8) / 8, i == j ? 0.0 : double(k % 7 - 3) / 8);
                    const T off = MatrixView<T>::make(double((i + j + b) % 5 - 2) / 8, double((i * j + b) % 3 - 1) / 8);
                    T mirrored = off;
                    if constexpr (test_utils::is_complex<T>::value) mirrored = std::conj(off);
                    const T bij = i == j ? T(R(n + 2 + b % 3)) : (i > j ? off : mirrored);
                    Bm.data_ptr()[std::size_t(b) * Bm.stride() + std::size_t(j) * Bm.ld() + i] = bij;
                }
    }
};

TYPED_TEST_SUITE(SygvdMask, Types);

TYPED_TEST(SygvdMask, VendorLeavesInactiveItemsUntouched) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;
    constexpr Backend B = TestFixture::B;
    Queue& q = *this->ctx;
    for (const int n : {5, 19})
        for (const int itype : {1, 2})
            for (const JobType jobz : {JobType::EigenVectors, JobType::NoEigenVectors})
                for (const Uplo uplo : {Uplo::Lower, Uplo::Upper}) {
                    const int batch = 9;
                    SCOPED_TRACE("n " + std::to_string(n) + " itype " + std::to_string(itype) +
                                 (jobz == JobType::EigenVectors ? " V " : " N ") + ts::uplo_name(uplo));
                    Matrix<T> A = ts::padded<T>(n, n, batch), Bm = ts::padded<T>(n, n, batch);
                    UnifiedVector<R> w(std::size_t(n) * batch);
                    UnifiedVector<int32_t> info{std::size_t(batch)};
                    mc::Case c;
                    c.batch = batch;
                    c.prepare = [&] {
                        TestFixture::fill(A, Bm, n, batch);
                        std::fill_n(w.data(), w.size(), ts::poison<R>());
                        std::fill_n(info.data(), info.size(), int32_t(-77));
                    };
                    c.call = [&](Span<const std::uint8_t> m, Span<std::byte> ws) {
                        sygvd<B, T>(q, A.view().with_batch_mask(m), Bm.view().with_batch_mask(m), w.to_span(), itype, jobz,
                                    uplo, ws, info.to_span());
                    };
                    c.buffer_size = [&](Span<const std::uint8_t> m) {
                        return sygvd_buffer_size<B, T>(q, A.view().with_batch_mask(m), Bm.view().with_batch_mask(m),
                                                       w.to_span(), itype, jobz, uplo);
                    };
                    c.outputs = [&] {
                        return std::vector<mc::Output>{mc::Output::matrix("A", A.view()), mc::Output::matrix("B", Bm.view()),
                                                       mc::Output::span<R>("w", {w.data(), w.size()}, std::size_t(n)),
                                                       mc::Output::span<int32_t>("info", {info.data(), info.size()}, 1)};
                    };
                    mc::check<ops::sygvd::SygvdChoice>(q, "sygvd", "vendor", c);
                }
}

// sygst on a Cholesky factor: the factor is any triangle with a positive diagonal to LAPACKE.
TYPED_TEST(SygvdMask, SygstVendorLeavesInactiveItemsUntouched) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::B;
    Queue& q = *this->ctx;
    for (const int n : {6, 21})
        for (const int itype : {1, 2, 3})
            for (const Uplo uplo : {Uplo::Lower, Uplo::Upper}) {
                const int batch = 7;
                SCOPED_TRACE("n " + std::to_string(n) + " itype " + std::to_string(itype) + " " + ts::uplo_name(uplo));
                Matrix<T> A = ts::padded<T>(n, n, batch), Bm = ts::padded<T>(n, n, batch);
                mc::Case c;
                c.batch = batch;
                c.prepare = [&] { TestFixture::fill(A, Bm, n, batch); };
                c.call = [&](Span<const std::uint8_t> m, Span<std::byte> ws) {
                    sygst<B, T>(q, A.view().with_batch_mask(m), Bm.view().with_batch_mask(m), itype, uplo, ws);
                };
                c.buffer_size = [&](Span<const std::uint8_t> m) {
                    return sygst_buffer_size<B, T>(q, A.view().with_batch_mask(m), Bm.view().with_batch_mask(m), itype, uplo);
                };
                c.outputs = [&] {
                    return std::vector<mc::Output>{mc::Output::matrix("A", A.view()), mc::Output::matrix("B", Bm.view())};
                };
                mc::check<ops::sygst::SygstChoice>(q, "sygst", "vendor", c);
            }
}

}  // namespace
