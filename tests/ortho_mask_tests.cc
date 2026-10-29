// SPDX-License-Identifier: MIT
// ortho and ritz_values honour a batch mask (W4c), on tests/masked_contract.hh. ortho runs every
// algorithm the device serves (a host device forces Householder) on both arms (columns, and rows
// through the transposed copy), and the external-basis overload, the Cholesky family included. The
// public syev under SVQB is pinned to a family that honours the mask, and on a GPU every other child
// to `native`, so the unmasked reference cannot route one to the vendor. A device-only mask on every pattern, complex data,
// padded ld and stride. evidence: docs/design/masked-batches.md#masked-batches-tests
#include <gtest/gtest.h>

#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/error.hh>
#include <batchlas/util/sycl-vector.hh>

#include <src/ops/syev/choice.hh>
#include <tests/masked_contract.hh>
#include <tests/test_support/storage.hh>
#include <tests/test_utils.hh>

#include <algorithm>
#include <array>
#include <complex>
#include <cstdint>
#include <cstring>
#include <optional>
#include <variant>
#include <span>
#include <string>
#include <vector>

using namespace batchlas;

namespace {

namespace mc = masked_contract;

// A composite over the public gemm on the host backend, whose BLAS cannot skip an item and refuses:
// the call ran unmasked, so the masked throw is the mask's, and it wrote no output.
inline void expect_refusal(batchlas::Queue& q, const mc::Case& c) {
    {
        UnifiedVector<std::byte> ws(std::max<std::size_t>(c.buffer_size ? c.buffer_size({}) : 0, 1));
        c.prepare();
        ASSERT_NO_THROW(c.call({}, ws.to_span()));
        q.wait();
    }
    const mc::DeviceMask mk(q, mc::by_rule("alternating", c.batch, [](int b) { return b % 2 == 1; }).bytes);
    c.prepare();
    q.wait();
    const auto outs = c.outputs();
    std::vector<std::vector<std::byte>> before;
    for (const auto& o : outs) before.emplace_back(o.base, o.base + o.stride_bytes * std::size_t(c.batch));
    UnifiedVector<std::byte> ws(std::size_t(1) << 24);
    EXPECT_ANY_THROW({
        c.call(mk.span(), ws.to_span());
        q.wait();
    });
    q.wait();
    for (std::size_t i = 0; i < outs.size(); ++i)
        EXPECT_EQ(0, std::memcmp(before[i].data(), outs[i].base, before[i].size())) << "the refused call wrote " << outs[i].name;
}

const char* algo_name(OrthoAlgorithm a) {
    switch (a) {
        case OrthoAlgorithm::Cholesky: return "Cholesky";
        case OrthoAlgorithm::Chol2: return "Chol2";
        case OrthoAlgorithm::ShiftChol3: return "ShiftChol3";
        case OrthoAlgorithm::Householder: return "Householder";
        case OrthoAlgorithm::CGS2: return "CGS2";
        case OrthoAlgorithm::SVQB: return "SVQB";
        default: return "SVQB2";
    }
}

// rows x cols items of full rank, distinct per item (or all item 0's when identical).
template <typename T>
void fill_full_rank(Matrix<T>& A, int rows, int cols, int batch, bool identical) {
    std::fill_n(A.data_ptr(), std::size_t(A.stride()) * batch, test_support::poison<T>());
    for (int b = 0; b < batch; ++b) {
        const int item = identical ? 0 : b;
        T* a = A.data_ptr() + std::size_t(A.stride()) * b;
        for (int j = 0; j < cols; ++j)
            for (int i = 0; i < rows; ++i) {
                const double re = double(int((i * 7 + j * 3 + item * 5) % 17) - 8) / 8 + (i == j ? 3.0 : 0.0);
                const double im = double(int((i * 5 + j + item * 3) % 11) - 5) / 8;
                a[i + std::size_t(A.ld()) * j] = MatrixView<T>::make(re, im);
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
class OrthoMask : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    using R = typename base_type<T>::type;
    static constexpr Backend B = Config::BackendVal;
    using SyevPin = std::optional<select::ScopedPin<ops::syev::SyevChoice>>;

    bool gpu() const { return this->ctx->device().type == DeviceType::GPU; }
    Transpose rows_trans() const { return is_std_complex_v<T> ? Transpose::ConjTrans : Transpose::Trans; }

    // The algorithms that honour the mask here; a host device runs Householder whatever is asked.
    std::vector<OrthoAlgorithm> algorithms() const {
        if (!gpu()) return {OrthoAlgorithm::Householder};
        // Householder on a GPU waits for orgqr: its blocked family refuses a mask while its ormqr
        // reaches trmm (orgqr.cc, "mask: lift after trmm").
        return {OrthoAlgorithm::CGS2, OrthoAlgorithm::SVQB, OrthoAlgorithm::SVQB2,
                OrthoAlgorithm::Cholesky, OrthoAlgorithm::Chol2, OrthoAlgorithm::ShiftChol3};
    }
    using ChildPins = std::array<std::optional<select::ScopedPin<std::monostate>>, 6>;
    void pin_syev(SyevPin& pin, ChildPins& children, int k) const {
        if (!gpu()) return;
        if (k <= 32) pin.emplace("syev", "cta_fused", select::StrictPin{});
        const char* ops[] = {"potrf", "trsm", "gemm", "gemv", "syrk", "herk"};
        for (std::size_t i = 0; i < children.size(); ++i) children[i].emplace(ops[i], "native");
    }

    // m vectors of length... : `m` x `k` columns (NoTrans) or k x m rows (the row arm).
    void check_ortho(OrthoAlgorithm algo, bool rows, int m, int k, int batch, bool identical = false) {
        SCOPED_TRACE(std::string("ortho ") + algo_name(algo) + (rows ? " rows" : " cols") + " m " + std::to_string(m) +
                     " k " + std::to_string(k) + " batch " + std::to_string(batch));
        Queue& q = *this->ctx;
        const int r = rows ? k : m, c = rows ? m : k;
        const Transpose tr = rows ? rows_trans() : Transpose::NoTrans;
        Matrix<T> A = test_support::padded<T>(r, c, batch);
        SyevPin pin;
        ChildPins children;
        pin_syev(pin, children, k);
        mc::Case cs;
        cs.batch = batch;
        cs.identical_items = identical;
        cs.prepare = [&] { fill_full_rank(A, r, c, batch, identical); };
        cs.call = [&](Span<const std::uint8_t> mk, Span<std::byte> ws) {
            (void)ortho<B, T>(q, A.view().with_batch_mask(mk), tr, ws, algo);
        };
        cs.buffer_size = [&](Span<const std::uint8_t> mk) {
            return ortho_buffer_size<B, T>(q, A.view().with_batch_mask(mk), tr, algo);
        };
        cs.outputs = [&] { return std::vector<mc::Output>{mc::Output::matrix("A", A.view())}; };
        mc::check(q, cs);
    }

    // A (m x nA) against M's nM orthonormal columns e_0 .. e_{nM-1}; only A carries the mask (M adopts it).
    void check_ortho_basis(OrthoAlgorithm algo, int m, int nA, int nM, int batch, bool mask_on_m) {
        SCOPED_TRACE(std::string("ortho(A, M) ") + algo_name(algo) + (mask_on_m ? " mask on M" : " mask on A"));
        Queue& q = *this->ctx;
        Matrix<T> A = test_support::padded<T>(m, nA, batch);
        Matrix<T> M = test_support::padded<T>(m, nM, batch, 2, 3);
        SyevPin pin;
        ChildPins children;
        pin_syev(pin, children, nA);
        const auto views = [&](Span<const std::uint8_t> mk) {
            return mask_on_m ? std::pair{A.view(), M.view().with_batch_mask(mk)}
                             : std::pair{A.view().with_batch_mask(mk), M.view()};
        };
        mc::Case cs;
        cs.batch = batch;
        cs.prepare = [&] {
            fill_full_rank(A, m, nA, batch, false);
            std::fill_n(M.data_ptr(), std::size_t(M.stride()) * batch, test_support::poison<T>());
            for (int b = 0; b < batch; ++b)
                for (int j = 0; j < nM; ++j)
                    for (int i = 0; i < m; ++i) M.data_ptr()[std::size_t(M.stride()) * b + i + std::size_t(M.ld()) * j] = T(i == j ? 1 : 0);
        };
        cs.call = [&](Span<const std::uint8_t> mk, Span<std::byte> ws) {
            const auto [a, mm] = views(mk);
            (void)ortho<B, T>(q, a, mm, Transpose::NoTrans, Transpose::NoTrans, ws, algo, 2);
        };
        cs.buffer_size = [&](Span<const std::uint8_t> mk) {
            const auto [a, mm] = views(mk);
            return ortho_buffer_size<B, T>(q, a, mm, Transpose::NoTrans, Transpose::NoTrans, algo, 2);
        };
        cs.outputs = [&] { return std::vector<mc::Output>{mc::Output::matrix("A", A.view()), mc::Output::matrix("M", M.view())}; };
        if (B == Backend::NETLIB) return expect_refusal(q, cs);  // the projection's gemm
        // AdaptiveCpp JIT-specialises a kernel on its argument values, so the null and the device
        // mask pointer compile apart: CGS2's gemv steps after the projection round differently on
        // Metal complex<float> (~1e-7 relative; the count moves with ACPP_ADAPTIVITY_LEVEL, and level
        // 0 cannot run on Metal). Active items are then checked by value: orthonormal, orthogonal to M.
        if (gpu() && is_std_complex_v<T>)
            cs.verify_active = [&](const std::string&, std::span<const int> active) {
                const double tol = 1e-4;
                for (int b : active) {
                    const T* a = A.data_ptr() + std::size_t(A.stride()) * b;
                    for (int i = 0; i < nA; ++i)
                        for (int j = 0; j < nA + nM; ++j) {
                            const T* x = j < nA ? a + std::size_t(A.ld()) * j : M.data_ptr() + std::size_t(M.stride()) * b + std::size_t(M.ld()) * (j - nA);
                            std::complex<double> dot = 0;
                            for (int r = 0; r < m; ++r)
                                dot += std::conj(std::complex<double>(a[r + std::size_t(A.ld()) * i])) * std::complex<double>(x[r]);
                            EXPECT_NEAR(std::abs(dot - (i == j ? 1.0 : 0.0)), 0.0, tol) << "item " << b << " (" << i << ", " << j << ")";
                        }
                }
            };
        mc::check(q, cs);
    }
};

TYPED_TEST_SUITE(OrthoMask, Types);

TYPED_TEST(OrthoMask, EveryAlgorithmBothArms) {
    for (OrthoAlgorithm a : this->algorithms())
        for (bool rows : {false, true}) this->check_ortho(a, rows, 12, 5, 6);
}

TYPED_TEST(OrthoMask, AgainstAnExternalBasis) {
    for (OrthoAlgorithm a : this->algorithms())
        for (bool on_m : {false, true}) this->check_ortho_basis(a, 12, 4, 3, 6, on_m);
}

// 1024 identical items through the one-work-group-per-item kernels: active items bit-identical.
TYPED_TEST(OrthoMask, SaturatingIdenticalBatch) {
    const auto algos = this->algorithms();
    this->check_ortho(algos.front(), false, 16, 4, 1024, true);
    if (this->gpu()) this->check_ortho(OrthoAlgorithm::ShiftChol3, false, 16, 4, 1024, true);  // potrf, trsm packed
}

// ritz_values(A, V): AV through the public gemm, then one work-group per (item, column). The ritz
// values sit at a padded per-item stride, so the pad keeps its poison too.
TYPED_TEST(OrthoMask, RitzValues) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;
    constexpr Backend B = TestFixture::B;
    Queue& q = *this->ctx;
    for (int batch : {6, 1024}) {
        const bool identical = batch >= 1024;
        SCOPED_TRACE("ritz_values batch " + std::to_string(batch));
        const int n = 10, k = 3, vstride = k + 2;
        Matrix<T> A = test_support::padded<T>(n, n, batch);
        Matrix<T> V = test_support::padded<T>(n, k, batch, 1, 4);
        auto rv = UnifiedVector<R>(static_cast<std::size_t>(vstride) * batch);
        mc::Case cs;
        cs.batch = batch;
        cs.identical_items = identical;
        cs.prepare = [&] {
            fill_full_rank(A, n, n, batch, identical);
            fill_full_rank(V, n, k, batch, identical);
            std::fill_n(rv.data(), rv.size(), R(-77));
        };
        const auto out = [&](Span<const std::uint8_t> mk) {
            return VectorView<R>(rv.data(), k, batch, 1, vstride).with_batch_mask(mk);
        };
        cs.call = [&](Span<const std::uint8_t> mk, Span<std::byte> ws) {
            (void)ritz_values<B, T, MatrixFormat::Dense>(q, A.view().with_batch_mask(mk), V.view().with_batch_mask(mk), out(mk), ws);
        };
        cs.buffer_size = [&](Span<const std::uint8_t> mk) {
            return ritz_values_buffer_size<B, T, MatrixFormat::Dense>(q, A.view().with_batch_mask(mk),
                                                                      V.view().with_batch_mask(mk), out(mk));
        };
        cs.outputs = [&] {
            const std::size_t s = sizeof(R) * vstride;
            return std::vector<mc::Output>{mc::Output::matrix("A", A.view()), mc::Output::matrix("V", V.view()),
                                           {"ritz", reinterpret_cast<const std::byte*>(rv.data()), s, s}};
        };
        if (B == Backend::NETLIB) {  // AV through the host BLAS gemm, which refuses
            if (!identical) expect_refusal(q, cs);
            continue;
        }
        mc::check(q, cs);
    }
}

}  // namespace
