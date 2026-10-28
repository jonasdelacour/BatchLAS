// SPDX-License-Identifier: MIT
// sytrf, sytrs and sysv honour a batch mask (W4b), each family pinned strictly and reached, on
// tests/masked_contract.hh: every mask pattern on a device-only mask, complex symmetric (not
// Hermitian) data, padded ld and stride, both triangles, and a singular item (info > 0, which sysv's
// solve skips). A, the packed int32 ipiv, info and B are all outputs: an inactive item keeps every
// byte. sytrs `blocked` runs its trsm on the masked part views (lift 1). The heavy tier adds the
// saturating identical-item batch. evidence: docs/design/masked-batches.md#masked-batches-tests
#include <gtest/gtest.h>

#include <batchlas/blas/functions/sysv.hh>
#include <batchlas/blas/functions/sytrf.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/error.hh>
#include <batchlas/util/sycl-vector.hh>

#include <src/ops/sysv/choice.hh>
#include <src/ops/sytrf/choice.hh>
#include <src/ops/sytrs/choice.hh>
#include <src/select/vendor.hh>
#include <tests/masked_contract.hh>
#include <tests/test_support/storage.hh>
#include <tests/test_utils.hh>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <vector>

using namespace batchlas;

namespace {

namespace mc = masked_contract;

enum class Op { sytrf, sytrs, sysv };

struct Shape {
    std::string spelling;
    int n, nrhs, batch;
    Uplo uplo = Uplo::Lower;
    bool identical = false;  ///< every item the same input (the saturating case)
    int chunk_items = 0;     ///< the blocked part size: chunk-edge patterns
};

std::string label(const Shape& s) {
    return s.spelling + " n " + std::to_string(s.n) + " nrhs " + std::to_string(s.nrhs) + " batch " +
           std::to_string(s.batch) + (s.uplo == Uplo::Lower ? " L" : " U") + (s.identical ? " identical" : "");
}

template <typename T, Backend B>
struct Problem {
    Queue& q;
    Op op;
    Shape s;
    Matrix<T> A, Bm;
    UnifiedVector<int64_t> piv;
    UnifiedVector<int32_t> info;

    Problem(Queue& queue, Op o, const Shape& shape)
        : q(queue), op(o), s(shape), A(test_support::padded<T>(s.n, s.n, s.batch)),
          Bm(test_support::padded<T>(s.n, std::max(s.nrhs, 1), s.batch, 2, 7)),
          piv(static_cast<std::size_t>(s.n) * s.batch), info(static_cast<std::size_t>(s.batch)) {}

    bool singular(int item) const { return !s.identical && s.batch > 2 && item == 1; }

    // Complex symmetric, indefinite: every third diagonal zero, so Bunch-Kaufman takes 2 x 2 blocks
    // and interchanges. Item 1 is all zero (singular, info = 1). The pads keep the poison.
    void fill_a() {
        std::fill_n(A.data_ptr(), std::size_t(A.stride()) * s.batch, test_support::poison<T>());
        for (int b = 0; b < s.batch; ++b) {
            const int item = s.identical ? 0 : b;
            T* a = A.data_ptr() + std::size_t(A.stride()) * b;
            for (int j = 0; j < s.n; ++j)
                for (int i = 0; i < s.n; ++i) {
                    const int lo = std::min(i, j), hi = std::max(i, j);
                    const double re = double(int((lo * 7 + hi * 3 + item * 5) % 17) - 8) / 8;
                    const double im = double(int((lo * 5 + hi + item * 3) % 11) - 5) / 8;
                    T v = MatrixView<T>::make(re, im);
                    if (i == j) v = (i % 3 == 0) ? T(0) : MatrixView<T>::make(re / 4, im / 4);
                    a[i + std::size_t(A.ld()) * j] = singular(b) ? T(0) : v;
                }
        }
    }
    void fill_b() {
        std::fill_n(Bm.data_ptr(), std::size_t(Bm.stride()) * s.batch, test_support::poison<T>());
        for (int b = 0; b < s.batch; ++b) {
            const int item = s.identical ? 0 : b;
            T* p = Bm.data_ptr() + std::size_t(Bm.stride()) * b;
            for (int c = 0; c < s.nrhs; ++c)
                for (int i = 0; i < s.n; ++i)
                    p[i + std::size_t(Bm.ld()) * c] =
                        MatrixView<T>::make(double(int((i * 3 + c * 7 + item) % 13) - 6) / 4, double((i + c) % 5) / 8);
        }
    }
    void poison_outputs() {
        std::memset(piv.data(), 0x5a, piv.size() * sizeof(int64_t));
        // sysv's solve skips an item whose info is nonzero: a 0 sentinel keeps that skip from hiding
        // an inactive item the solve wrote.
        std::fill_n(info.data(), info.size(), op == Op::sysv ? int32_t(0) : int32_t(-77));
    }
    MatrixView<T> a(Span<const std::uint8_t> m) { return A.view().with_batch_mask(m); }
    MatrixView<T> b(Span<const std::uint8_t> m) { return Bm.view().with_batch_mask(m); }
    Span<std::byte> sized(Span<std::byte> ws, std::size_t need) { return {ws.data(), need}; }

    mc::Case make_case() {
        mc::Case c;
        c.batch = s.batch;
        c.chunk_items = s.chunk_items;
        c.identical_items = s.identical;
        // sytrs blocked's two trsm take the vendor unmasked and a native family masked.
        c.masked_reference = op == Op::sytrs && s.spelling == "blocked" && (B == Backend::CUDA || B == Backend::ROCM);
        c.prepare = [this] {
            fill_a();
            fill_b();
            poison_outputs();
            if (op == Op::sytrs) {  // the factor every pattern solves with: unmasked, on Auto
                const std::size_t need = sytrf_buffer_size<B, T>(q, A.view(), s.uplo);
                UnifiedVector<std::byte> ws(std::max<std::size_t>(need, 1));
                (void)sytrf<B, T>(q, A.view(), s.uplo, piv.to_span(), sized(ws.to_span(), need), info.to_span());
                q.wait();
            }
        };
        c.call = [this](Span<const std::uint8_t> m, Span<std::byte> ws) {
            switch (op) {
                case Op::sytrf: (void)sytrf<B, T>(q, a(m), s.uplo, piv.to_span(), ws, info.to_span()); break;
                case Op::sytrs: (void)sytrs<B, T>(q, a(m), b(m), s.uplo, piv.to_span(), ws); break;
                case Op::sysv: (void)sysv<B, T>(q, a(m), b(m), s.uplo, piv.to_span(), ws, info.to_span()); break;
            }
        };
        c.buffer_size = [this](Span<const std::uint8_t> m) -> std::size_t {
            switch (op) {
                case Op::sytrf: return sytrf_buffer_size<B, T>(q, a(m), s.uplo);
                case Op::sytrs: return sytrs_buffer_size<B, T>(q, a(m), b(m), s.uplo);
                case Op::sysv: return sysv_buffer_size<B, T>(q, a(m), b(m), s.uplo);
            }
            return 0;
        };
        c.outputs = [this] {
            const std::span<const int> p32(reinterpret_cast<const int*>(piv.data()), std::size_t(s.n) * s.batch);
            std::vector<mc::Output> o{mc::Output::matrix("A", A.view()), mc::Output::span<int>("ipiv", p32, s.n)};
            if (op != Op::sytrs) o.push_back(mc::Output::span<int32_t>("info", std::span<const int32_t>(info.data(), info.size()), 1));
            if (op != Op::sytrf) o.push_back(mc::Output::matrix("B", Bm.view()));
            return o;
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
class SytrfMask : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    static constexpr Backend B = Config::BackendVal;
    static constexpr bool kHost = B == Backend::NETLIB;

    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        if (this->HasFatalFailure() || ::testing::Test::IsSkipped()) return;
        if (kHost && !select::kHasNetlib) GTEST_SKIP() << "no LAPACKE in this build";
        if (!kHost && this->ctx->device().type != DeviceType::GPU) GTEST_SKIP() << "the native families are GPU kernels";
    }

    template <typename Choice>
    void check(Op op, const char* name, const Shape& s) {
        SCOPED_TRACE(std::string(name) + " " + label(s));
        Problem<T, B> p(*this->ctx, op, s);
        mc::check<Choice>(*this->ctx, name, s.spelling, p.make_case());
    }

    // The families that honour the mask here: LAPACKE's host loop on NETLIB, the natives on a GPU.
    static std::vector<std::string> sytrf_families(int n) {
        if (kHost) return {"vendor"};
        std::vector<std::string> v{"wg", "blocked"};
        if (n <= 32) v.insert(v.begin(), "cta");
        return v;
    }
    static std::vector<std::string> sytrs_families() { return kHost ? std::vector<std::string>{"vendor"} : std::vector<std::string>{"cta", "wg"}; }
    static std::vector<std::string> sysv_families(int n) {
        if (kHost || n > 32) return {"compose"};
        return {"cta", "compose"};
    }
};

TYPED_TEST_SUITE(SytrfMask, Types);

TYPED_TEST(SytrfMask, SytrfEveryFamilyBothTriangles) {
    for (Uplo u : {Uplo::Lower, Uplo::Upper})
        for (int n : {8, 33})
            for (const auto& f : this->sytrf_families(n))
                this->template check<ops::sytrf::SytrfChoice>(Op::sytrf, "sytrf", {f, n, 0, 6, u});
}

// n = 170 runs blocked's panel loop through the public gemm; the parts {0..2}, {3..5}, {6..8}
// each carry their own slice of the mask (the drop hazard), on the gemm views as on the kernels.
TYPED_TEST(SytrfMask, SytrfBlockedPartsCarryTheirSlice) {
    if (this->kHost) GTEST_SKIP() << "blocked is a GPU family";
    if constexpr (std::is_same_v<typename TestFixture::T, float>)
        for (Uplo u : {Uplo::Lower, Uplo::Upper})
            this->template check<ops::sytrf::SytrfChoice>(Op::sytrf, "sytrf", {"blocked", 170, 0, 9, u, false, 3});
    this->template check<ops::sytrf::SytrfChoice>(Op::sytrf, "sytrf", {"blocked", 45, 0, 9, Uplo::Lower, false, 3});
}

TYPED_TEST(SytrfMask, SytrsEveryFamilyBothTriangles) {
    for (Uplo u : {Uplo::Lower, Uplo::Upper})
        for (int n : {8, 40})
            for (const auto& f : this->sytrs_families())
                this->template check<ops::sytrs::SytrsChoice>(Op::sytrs, "sytrs", {f, n, 3, 6, u});
}

// sytrs blocked: the converted factor and gathered B per part, its two trsm on the masked part
// views; parts of 3 and 4 items carry their own slice (the chunk patterns).
TYPED_TEST(SytrfMask, SytrsBlockedPartsCarryTheirSlice) {
    if (this->kHost) GTEST_SKIP() << "blocked is a GPU family";
    for (Uplo u : {Uplo::Lower, Uplo::Upper}) {
        this->template check<ops::sytrs::SytrsChoice>(Op::sytrs, "sytrs", {"blocked", 8, 3, 6, u});
        this->template check<ops::sytrs::SytrsChoice>(Op::sytrs, "sytrs", {"blocked", 45, 2, 9, u, false, 3});
    }
    this->template check<ops::sytrs::SytrsChoice>(Op::sytrs, "sytrs", {"blocked", 70, 4, 12, Uplo::Lower, false, 4});
}

TYPED_TEST(SytrfMask, SysvEveryFamilyBothTriangles) {
    for (Uplo u : {Uplo::Lower, Uplo::Upper})
        for (int n : {8, 32, 40})
            for (const auto& f : this->sysv_families(n))
                this->template check<ops::sysv::SysvChoice>(Op::sysv, "sysv", {f, n, 3, 6, u});
}

// 1024 identical items: the local-memory (cta) families and the host loops at a saturating batch,
// every active item bit-identical to the first.
TYPED_TEST(SytrfMask, SaturatingIdenticalBatch) {
    constexpr int kBatch = 1024;
    for (const auto& f : this->sytrf_families(16))
        if (f != "blocked") this->template check<ops::sytrf::SytrfChoice>(Op::sytrf, "sytrf", {f, 16, 0, kBatch, Uplo::Lower, true});
    for (const auto& f : this->sytrs_families())
        this->template check<ops::sytrs::SytrsChoice>(Op::sytrs, "sytrs", {f, 16, 2, kBatch, Uplo::Upper, true});
    for (const auto& f : this->sysv_families(16))
        this->template check<ops::sysv::SysvChoice>(Op::sysv, "sysv", {f, 16, 2, kBatch, Uplo::Lower, true});
}

}  // namespace
