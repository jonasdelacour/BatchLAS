// SPDX-License-Identifier: MIT
// sygvd and sygst honour a batch mask (W3c) in the host backend's LAPACKE family, pinned `vendor`
// and reached, on tests/masked_contract.hh: A, B, w and info of an inactive item untouched, active
// items the unmasked call's. sygst's Blocked and TwoSided run their native children on the masked
// views (lift 1). sygvd's Compose and sygvd_factored run potrf, sygst, syev and trsm / trmm on the
// masked views (lift 2) on a device, pinned and reached; on the host backend compose refuses (its
// trsm / trmm are the host BLAS). Auto on both. Device-only masks, complex data, padded ld and stride.
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

    bool host() const { return B == Backend::NETLIB; }
    bool gpu() const { return this->ctx->device().type == DeviceType::GPU; }

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

    static constexpr bool vendor() { return B == Backend::CUDA || B == Backend::ROCM; }

    // Items per 32-lane sub-group of syev's cta partitions: one item per power of two >= n.
    static int items_per_subgroup(int n) {
        int p = 4;
        while (p < n) p *= 2;
        return 32 / p;
    }

    // One sygvd, or sygvd_factored (B's uplo triangle, positive diagonal, is the factor), on
    // `spelling` (empty: Auto). On a vendor backend the children and Auto take the vendor unmasked,
    // which a masked view refuses, so masked patterns compare with the all-active masked call.
    void check_compose(const std::string& spelling, int n, int itype, JobType jobz, Uplo uplo, bool factored) {
        const int batch = 9;
        SCOPED_TRACE((spelling.empty() ? std::string("auto") : spelling) + (factored ? " factored" : "") + " n " +
                     std::to_string(n) + " itype " + std::to_string(itype) +
                     (jobz == JobType::EigenVectors ? " V " : " N ") + ts::uplo_name(uplo));
        Queue& q = *this->ctx;
        Matrix<T> A = ts::padded<T>(n, n, batch), Bm = ts::padded<T>(n, n, batch);
        UnifiedVector<R> w(std::size_t(n) * batch);
        UnifiedVector<int32_t> info{std::size_t(batch)};
        mc::Case c;
        c.batch = batch;
        c.wg_items = n <= 32 ? items_per_subgroup(n) : 0;
        c.chunk_items = n > 32 ? 3 : 0;
        c.masked_reference = vendor();
        c.prepare = [&] {
            fill(A, Bm, n, batch);
            std::fill_n(w.data(), w.size(), ts::poison<R>());
            std::fill_n(info.data(), info.size(), int32_t(-77));
        };
        c.call = [&](Span<const std::uint8_t> m, Span<std::byte> ws) {
            const auto a = A.view().with_batch_mask(m);
            const auto b = Bm.view().with_batch_mask(m);
            if (factored) sygvd_factored<B, T>(q, a, b, w.to_span(), itype, jobz, uplo, ws, info.to_span());
            else sygvd<B, T>(q, a, b, w.to_span(), itype, jobz, uplo, ws, info.to_span());
        };
        c.buffer_size = [&](Span<const std::uint8_t> m) {
            const auto a = A.view().with_batch_mask(m);
            const auto b = Bm.view().with_batch_mask(m);
            return factored ? sygvd_factored_buffer_size<B, T>(q, a, b, w.to_span(), itype, jobz, uplo)
                            : sygvd_buffer_size<B, T>(q, a, b, w.to_span(), itype, jobz, uplo);
        };
        c.outputs = [&] {
            return std::vector<mc::Output>{mc::Output::matrix("A", A.view()), mc::Output::matrix("B", Bm.view()),
                                           mc::Output::span<R>("w", {w.data(), w.size()}, std::size_t(n)),
                                           mc::Output::span<int32_t>("info", {info.data(), info.size()}, 1)};
        };
        if (spelling.empty()) mc::check(q, c);
        else mc::check<ops::sygvd::SygvdChoice>(q, "sygvd", spelling, c);
    }
};

TYPED_TEST_SUITE(SygvdMask, Types);

TYPED_TEST(SygvdMask, VendorLeavesInactiveItemsUntouched) {
    if (!this->host()) GTEST_SKIP() << "cuSOLVER cannot skip an item: the device vendor refuses a mask (compose runs it)";
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
    if (!this->host()) GTEST_SKIP() << "the LAPACKE family is the host backend's";
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

// sygst's device families, pinned and reached: blocked (the SLM diagonal kernel, trsm, trmm,
// gemm and syr2k / her2k per block; its part loop at 3 items under the chunk patterns) and twosided
// (hermitize, two trsm or trmm), one block (n = 6) and three (n = 70), every itype, both triangles.
// Their children route on their own Auto, which takes the vendor unmasked on a vendor backend.
TYPED_TEST(SygvdMask, SygstDeviceFamiliesLeaveInactiveItemsUntouched) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::B;
    if (this->host() || !this->gpu()) GTEST_SKIP() << "blocked and twosided are device families";
    Queue& q = *this->ctx;
    for (const char* fam : {"blocked:nb=32", "twosided"})
        for (const int n : {6, 70})
            for (const int itype : {1, 2, 3})
                for (const Uplo uplo : {Uplo::Lower, Uplo::Upper}) {
                    const int batch = 9;
                    SCOPED_TRACE(std::string(fam) + " n " + std::to_string(n) + " itype " + std::to_string(itype) + " " +
                                 ts::uplo_name(uplo));
                    Matrix<T> A = ts::padded<T>(n, n, batch), Bm = ts::padded<T>(n, n, batch);
                    mc::Case c;
                    c.batch = batch;
                    c.chunk_items = fam[0] == 'b' && n > 32 ? 3 : 0;
                    c.masked_reference = B == Backend::CUDA || B == Backend::ROCM;
                    c.prepare = [&] { TestFixture::fill(A, Bm, n, batch); };
                    c.call = [&](Span<const std::uint8_t> m, Span<std::byte> ws) {
                        sygst<B, T>(q, A.view().with_batch_mask(m), Bm.view().with_batch_mask(m), itype, uplo, ws);
                    };
                    c.buffer_size = [&](Span<const std::uint8_t> m) {
                        return sygst_buffer_size<B, T>(q, A.view().with_batch_mask(m), Bm.view().with_batch_mask(m), itype,
                                                       uplo);
                    };
                    c.outputs = [&] {
                        return std::vector<mc::Output>{mc::Output::matrix("A", A.view()), mc::Output::matrix("B", Bm.view())};
                    };
                    mc::check<ops::sygst::SygstChoice>(q, "sygst", fam, c);
                }
}

// compose, pinned and reached: potrf, sygst, syev and the trsm (itype 1, 2) or trmm (itype 3)
// back-transform on the masked views. syev's cta sizes with the sub-group hole patterns, then its
// blocked sizes with the part patterns; every itype, both jobz and triangles.
TYPED_TEST(SygvdMask, ComposeAtCtaSizesLeavesInactiveItemsUntouched) {
    if (this->host()) GTEST_SKIP() << "compose refuses a mask on the host backend: its trsm and trmm are the host BLAS";
    for (const int n : {7, 32})
        for (const int itype : {1, 2, 3})
            for (const JobType jobz : {JobType::EigenVectors, JobType::NoEigenVectors})
                for (const Uplo uplo : {Uplo::Lower, Uplo::Upper}) this->check_compose("compose", n, itype, jobz, uplo, false);
}
TYPED_TEST(SygvdMask, ComposeAtBlockedSizesLeavesInactiveItemsUntouched) {
    if (this->host()) GTEST_SKIP() << "compose refuses a mask on the host backend: its trsm and trmm are the host BLAS";
    for (const int n : {45, 70})
        for (const int itype : {1, 2, 3})
            for (const JobType jobz : {JobType::EigenVectors, JobType::NoEigenVectors})
                for (const Uplo uplo : {Uplo::Lower, Uplo::Upper}) this->check_compose("compose", n, itype, jobz, uplo, false);
}

// sygvd_factored always composes (no potrf): sygst, syev and the back-transform; B is read only.
TYPED_TEST(SygvdMask, FactoredComposeLeavesInactiveItemsUntouched) {
    if (this->host()) GTEST_SKIP() << "compose refuses a mask on the host backend: its trsm and trmm are the host BLAS";
    for (const int n : {7, 45})
        for (const int itype : {1, 2, 3})
            for (const JobType jobz : {JobType::EigenVectors, JobType::NoEigenVectors})
                for (const Uplo uplo : {Uplo::Lower, Uplo::Upper}) this->check_compose("compose", n, itype, jobz, uplo, true);
}

// The SCC call on Auto: the masked view routes to a family that honours the mask at every size.
TYPED_TEST(SygvdMask, AutoLeavesInactiveItemsUntouched) {
    for (const int n : {7, 32, 45, 70})
        for (const JobType jobz : {JobType::EigenVectors, JobType::NoEigenVectors}) {
            this->check_compose("", n, 1, jobz, Uplo::Lower, false);
            this->check_compose("", n, 3, jobz, Uplo::Upper, false);
        }
}

}  // namespace
