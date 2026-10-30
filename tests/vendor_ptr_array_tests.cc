// SPDX-License-Identifier: MIT
// Pointer-array vendor routes (cublas<t>{trsm,geqrf,getrf,getrs,getri}Batched,
// cusolverDn<t>potrfBatched) on views that bring no pointer array of their own: a view over
// the caller's USM, and two slices of one Matrix, which share theirs. Each case pins the op's
// vendor route, uses padded ld and stride, and checks every item against a host computation
// on the original data plus the poison in the padding.
#include <gtest/gtest.h>

#include <batchlas/blas/functions/geqrf.hh>
#include <batchlas/blas/functions/getrf.hh>
#include <batchlas/blas/functions/getri.hh>
#include <batchlas/blas/functions/getrs.hh>
#include <batchlas/blas/functions/potrf.hh>
#include <batchlas/blas/functions/trsm.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/verify/tolerance.hh>
#include <src/select/vendor.hh>

#include <tests/test_utils.hh>

#include <cmath>
#include <complex>
#include <cstdint>
#include <string>
#include <vector>

using namespace batchlas;

namespace {

template <typename T>
using MV = MatrixView<T, MatrixFormat::Dense>;
template <typename T>
using Real = typename base_type<T>::type;
using verify::Check;

constexpr int kBatch = 5;

template <typename T> T cj(T v) {
    if constexpr (is_std_complex_v<T>) return std::conj(v);
    else return v;
}

inline std::uint32_t mix(std::uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return x;
}

// Deterministic entry in [-1, 1] (both parts for complex T), distinct per item and salt.
template <typename T> T entry(int i, int j, int b, int salt) {
    const std::uint32_t s = mix(std::uint32_t(i) * 7919U + std::uint32_t(j) * 104729U +
                                std::uint32_t(b) * 1299709U + std::uint32_t(salt) * 15485863U);
    const double re = mix(s) / 4294967295.0 * 2.0 - 1.0;
    if constexpr (is_std_complex_v<T>) {
        const double im = mix(s + 1U) / 4294967295.0 * 2.0 - 1.0;
        return T(Real<T>(re), Real<T>(im));
    } else {
        return T(re);
    }
}

template <typename T> T poison() {
    if constexpr (is_std_complex_v<T>) return T(Real<T>(777), Real<T>(-777));
    else return T(777);
}

// A batch in the caller's own USM at ld = rows + 3 and stride = ld * cols + 5.
template <typename T>
struct RawBatch {
    int rows, cols, ld, stride;
    UnifiedVector<T> buf;
    RawBatch(int r, int c)
        : rows(r), cols(c), ld(r + 3), stride((r + 3) * c + 5), buf(std::size_t((r + 3) * c + 5) * kBatch, poison<T>()) {}
    T& at(int i, int j, int b) { return buf[std::size_t(b) * stride + i + std::size_t(j) * ld]; }
    MV<T> view() { return MV<T>(buf.data(), rows, cols, ld, stride, kBatch); }
    std::vector<T> item(int b) {
        std::vector<T> out(std::size_t(rows) * cols);
        for (int j = 0; j < cols; ++j)
            for (int i = 0; i < rows; ++i) out[i + std::size_t(j) * rows] = at(i, j, b);
        return out;
    }
    // Every element outside the rows x cols block of every item still holds the poison.
    ::testing::AssertionResult padding_intact() {
        for (int b = 0; b < kBatch; ++b)
            for (int o = 0; o < stride; ++o) {
                const int j = o / ld, i = o % ld;
                if (j < cols && i < rows) continue;
                if (buf[std::size_t(b) * stride + o] != poison<T>())
                    return ::testing::AssertionFailure() << "item " << b << " offset " << o << " overwritten";
            }
        return ::testing::AssertionSuccess();
    }
};

template <typename T>
double fro(const std::vector<T>& v) {
    double s = 0;
    for (const T& x : v) s += double(std::norm(std::complex<double>(x)));
    return std::sqrt(s);
}

// C (m x n) = A (m x k) * B (k x n), column-major, natural ld.
template <typename T>
std::vector<T> matmul(const std::vector<T>& A, const std::vector<T>& B, int m, int k, int n) {
    std::vector<T> C(std::size_t(m) * n, T(0));
    for (int j = 0; j < n; ++j)
        for (int p = 0; p < k; ++p)
            for (int i = 0; i < m; ++i) C[i + std::size_t(j) * m] += A[i + std::size_t(p) * m] * B[p + std::size_t(j) * k];
    return C;
}

template <typename T>
std::vector<T> op_of(const std::vector<T>& A, int n, Transpose t) {
    if (t == Transpose::NoTrans) return A;
    std::vector<T> out(A.size());
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            out[i + std::size_t(j) * n] = t == Transpose::ConjTrans ? cj(A[j + std::size_t(i) * n]) : A[j + std::size_t(i) * n];
    return out;
}

template <typename T>
std::vector<T> diff(const std::vector<T>& a, const std::vector<T>& b) {
    std::vector<T> d(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) d[i] = a[i] - b[i];
    return d;
}

// Packed 1-based int32 pivots of item b (the layout every getrf route writes).
inline std::vector<int> pivots_of(const UnifiedVector<int64_t>& piv, int n, int b) {
    const auto* p = reinterpret_cast<const std::int32_t*>(piv.data());
    return std::vector<int>(p + std::size_t(b) * n, p + std::size_t(b + 1) * n);
}

// Row-permute A (n x cols) by the LAPACK swap sequence: P * A.
template <typename T>
std::vector<T> permute_rows(std::vector<T> A, int n, int cols, const std::vector<int>& ipiv) {
    for (int i = 0; i < n; ++i) {
        const int r = ipiv[i] - 1;
        if (r != i)
            for (int j = 0; j < cols; ++j) std::swap(A[i + std::size_t(j) * n], A[r + std::size_t(j) * n]);
    }
    return A;
}

// Fill a general n x n item with a shifted diagonal so the solves stay well conditioned.
template <typename T>
void fill_general(RawBatch<T>& M, int salt, double shift) {
    for (int b = 0; b < kBatch; ++b)
        for (int j = 0; j < M.cols; ++j)
            for (int i = 0; i < M.rows; ++i)
                M.at(i, j, b) = entry<T>(i, j, b, salt) + (i == j ? T(Real<T>(shift)) : T(0));
}

// Success, or the exception's message: a refusal must say what threw.
template <typename F>
::testing::AssertionResult runs(F&& f) {
    try {
        f();
        return ::testing::AssertionSuccess();
    } catch (const std::exception& e) {
        return ::testing::AssertionFailure() << "threw: " << e.what();
    }
}

}  // namespace

template <typename T, Backend B>
struct TestConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

using VendorPtrTypes = typename test_utils::backend_types<TestConfig>::type;

template <typename Config>
class VendorPtrArrayTest : public test_utils::BatchLASTest<Config> {
protected:
    static constexpr Backend Bk = Config::BackendVal;
    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        // The host vendor (LAPACKE loop) takes no pointer arrays.
        if (Bk == Backend::NETLIB) GTEST_SKIP() << "pointer-array vendors are the GPU libraries";
    }
};

TYPED_TEST_SUITE(VendorPtrArrayTest, VendorPtrTypes);

TYPED_TEST(VendorPtrArrayTest, TrsmRawViews) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::Bk;
    if constexpr (!select::level3_vendor_available<B>) {
        GTEST_SKIP() << "no level-3 vendor";
    } else {
        const ScopedEnvVar pin("BATCHLAS_TRSM_ROUTE", "vendor");
        const int n = 12, nrhs = 7;
        const T alpha = is_std_complex_v<T> ? T(1) * entry<T>(1, 2, 3, 4) + T(1) : T(1.5);
        struct Cfg { Side side; Uplo uplo; Transpose t; };
        for (const Cfg c : {Cfg{Side::Left, Uplo::Lower, Transpose::NoTrans}, Cfg{Side::Right, Uplo::Upper, Transpose::ConjTrans}}) {
            const std::string where = c.side == Side::Left ? "Left/Lower/N" : "Right/Upper/C";
            const int bm = c.side == Side::Left ? n : nrhs, bn = c.side == Side::Left ? nrhs : n;
            RawBatch<T> A(n, n), X(bm, bn);
            // Only the uplo triangle is read: the other one keeps the poison.
            for (int b = 0; b < kBatch; ++b)
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < n; ++i)
                        if (c.uplo == Uplo::Lower ? i >= j : i <= j)
                            A.at(i, j, b) = entry<T>(i, j, b, 1) + (i == j ? T(Real<T>(n)) : T(0));
            for (int b = 0; b < kBatch; ++b)
                for (int j = 0; j < bn; ++j)
                    for (int i = 0; i < bm; ++i) X.at(i, j, b) = entry<T>(i, j, b, 2);
            std::vector<std::vector<T>> B0(kBatch);
            for (int b = 0; b < kBatch; ++b) B0[b] = X.item(b);
            ASSERT_TRUE(runs([&] { trsm<B, T>(*this->ctx, A.view(), X.view(), alpha, c.side, c.uplo, c.t, Diag::NonUnit); }))
                << where;
            this->ctx->wait();
            EXPECT_TRUE(X.padding_intact()) << where;
            for (int b = 0; b < kBatch; ++b) {
                std::vector<T> a = A.item(b);
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < n; ++i)
                        if (c.uplo == Uplo::Lower ? i < j : i > j) a[i + std::size_t(j) * n] = T(0);
                const auto opA = op_of(a, n, c.t);
                const auto x = X.item(b);
                const auto lhs = c.side == Side::Left ? matmul(opA, x, n, n, nrhs) : matmul(x, opA, nrhs, n, n);
                std::vector<T> rhs = B0[b];
                for (T& v : rhs) v *= alpha;
                const double err = fro(diff(lhs, rhs)) / (fro(opA) * fro(x) + fro(rhs));
                EXPECT_VERIFY(T, Check::solve, n, err) << where << " item " << b;
            }
        }
    }
}

TYPED_TEST(VendorPtrArrayTest, GeqrfRawView) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::Bk;
    if constexpr (!select::factorization_vendor_available<B>) {
        GTEST_SKIP() << "no factorization vendor";
    } else {
        const ScopedEnvVar pin("BATCHLAS_GEQRF_ROUTE", "vendor");
        const int m = 14, n = 10, k = 10;
        RawBatch<T> A(m, n);
        for (int b = 0; b < kBatch; ++b)
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < m; ++i) A.at(i, j, b) = entry<T>(i, j, b, 3);
        std::vector<std::vector<T>> A0(kBatch);
        for (int b = 0; b < kBatch; ++b) A0[b] = A.item(b);
        UnifiedVector<T> tau(std::size_t(k) * kBatch, poison<T>());
        UnifiedVector<std::byte> ws(geqrf_buffer_size<B, T>(*this->ctx, A.view(), tau.to_span()));
        ASSERT_TRUE(runs([&] { geqrf<B, T>(*this->ctx, A.view(), tau.to_span(), ws.to_span()); }));
        this->ctx->wait();
        EXPECT_TRUE(A.padding_intact());
        for (int b = 0; b < kBatch; ++b) {
            const auto f = A.item(b);
            // Q R = H_0 ... H_{k-1} R: apply the reflectors to R from the last one back.
            std::vector<T> X(std::size_t(m) * n, T(0));
            for (int j = 0; j < n; ++j)
                for (int i = 0; i <= std::min(j, m - 1); ++i) X[i + std::size_t(j) * m] = f[i + std::size_t(j) * m];
            for (int r = k - 1; r >= 0; --r) {
                const T t = tau[std::size_t(b) * k + r];
                std::vector<T> v(m, T(0));
                v[r] = T(1);
                for (int i = r + 1; i < m; ++i) v[i] = f[i + std::size_t(r) * m];
                for (int j = 0; j < n; ++j) {
                    T s = T(0);
                    for (int i = r; i < m; ++i) s += cj(v[i]) * X[i + std::size_t(j) * m];
                    for (int i = r; i < m; ++i) X[i + std::size_t(j) * m] -= t * v[i] * s;
                }
            }
            EXPECT_VERIFY(T, Check::factorization, m, fro(diff(X, A0[b])) / fro(A0[b])) << "item " << b;
        }
    }
}

TYPED_TEST(VendorPtrArrayTest, GetrfRawView) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::Bk;
    if constexpr (!select::factorization_vendor_available<B>) {
        GTEST_SKIP() << "no factorization vendor";
    } else {
        const ScopedEnvVar pin("BATCHLAS_GETRF_ROUTE", "vendor");
        const int n = 12;
        RawBatch<T> A(n, n);
        fill_general(A, 4, 0.0);  // no shift: the pivots move
        std::vector<std::vector<T>> A0(kBatch);
        for (int b = 0; b < kBatch; ++b) A0[b] = A.item(b);
        UnifiedVector<int64_t> piv(std::size_t(n) * kBatch, int64_t(0));
        UnifiedVector<int32_t> info(kBatch, int32_t(-1));
        UnifiedVector<std::byte> ws(getrf_buffer_size<B, T>(*this->ctx, A.view()));
        ASSERT_TRUE(runs([&] { getrf<B, T>(*this->ctx, A.view(), piv.to_span(), ws.to_span(), info.to_span()); }));
        this->ctx->wait();
        EXPECT_TRUE(A.padding_intact());
        for (int b = 0; b < kBatch; ++b) {
            EXPECT_EQ(info[b], 0) << "item " << b;
            const auto f = A.item(b);
            std::vector<T> L(std::size_t(n) * n, T(0)), U(std::size_t(n) * n, T(0));
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i) {
                    if (i > j) L[i + std::size_t(j) * n] = f[i + std::size_t(j) * n];
                    else U[i + std::size_t(j) * n] = f[i + std::size_t(j) * n];
                    if (i == j) L[i + std::size_t(j) * n] = T(1);
                }
            const auto PA = permute_rows(A0[b], n, n, pivots_of(piv, n, b));
            EXPECT_VERIFY(T, Check::factorization, n, fro(diff(matmul(L, U, n, n, n), PA)) / fro(A0[b])) << "item " << b;
        }
    }
}

TYPED_TEST(VendorPtrArrayTest, GetrsRawViews) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::Bk;
    if constexpr (!select::factorization_vendor_available<B>) {
        GTEST_SKIP() << "no factorization vendor";
    } else {
        // The factorization takes a register-resident route that needs no pointer array.
        const ScopedEnvVar pin_f("BATCHLAS_GETRF_ROUTE", "tiny");
        const ScopedEnvVar pin("BATCHLAS_GETRS_ROUTE", "vendor");
        const int n = 12, nrhs = 4;
        for (const Transpose t : {Transpose::NoTrans, Transpose::ConjTrans}) {
            const std::string where = t == Transpose::NoTrans ? "N" : "C";
            RawBatch<T> A(n, n), X(n, nrhs);
            fill_general(A, 5, 2.0);
            for (int b = 0; b < kBatch; ++b)
                for (int j = 0; j < nrhs; ++j)
                    for (int i = 0; i < n; ++i) X.at(i, j, b) = entry<T>(i, j, b, 6);
            std::vector<std::vector<T>> A0(kBatch), B0(kBatch);
            for (int b = 0; b < kBatch; ++b) { A0[b] = A.item(b); B0[b] = X.item(b); }
            UnifiedVector<int64_t> piv(std::size_t(n) * kBatch, int64_t(0));
            UnifiedVector<std::byte> wf(getrf_buffer_size<B, T>(*this->ctx, A.view()));
            getrf<B, T>(*this->ctx, A.view(), piv.to_span(), wf.to_span());
            UnifiedVector<std::byte> ws(getrs_buffer_size<B, T>(*this->ctx, A.view(), X.view(), t));
            ASSERT_TRUE(runs([&] { getrs<B, T>(*this->ctx, A.view(), X.view(), t, piv.to_span(), ws.to_span()); })) << where;
            this->ctx->wait();
            EXPECT_TRUE(X.padding_intact()) << where;
            for (int b = 0; b < kBatch; ++b) {
                const auto opA = op_of(A0[b], n, t);
                const auto x = X.item(b);
                const double err = fro(diff(matmul(opA, x, n, n, nrhs), B0[b])) / (fro(opA) * fro(x) + fro(B0[b]));
                EXPECT_VERIFY(T, Check::solve, n, err) << where << " item " << b;
            }
        }
    }
}

TYPED_TEST(VendorPtrArrayTest, GetriRawViews) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::Bk;
    if constexpr (!select::factorization_vendor_available<B>) {
        GTEST_SKIP() << "no factorization vendor";
    } else {
        const ScopedEnvVar pin_f("BATCHLAS_GETRF_ROUTE", "tiny");
        const ScopedEnvVar pin("BATCHLAS_GETRI_ROUTE", "vendor");
        const int n = 12;
        RawBatch<T> A(n, n), C(n, n);
        fill_general(A, 7, 2.0);
        std::vector<std::vector<T>> A0(kBatch);
        for (int b = 0; b < kBatch; ++b) A0[b] = A.item(b);
        UnifiedVector<int64_t> piv(std::size_t(n) * kBatch, int64_t(0));
        UnifiedVector<std::byte> wf(getrf_buffer_size<B, T>(*this->ctx, A.view()));
        getrf<B, T>(*this->ctx, A.view(), piv.to_span(), wf.to_span());
        UnifiedVector<int32_t> info(kBatch, int32_t(-1));
        UnifiedVector<std::byte> ws(getri_buffer_size<B, T>(*this->ctx, A.view()));
        ASSERT_TRUE(runs([&] { getri<B, T>(*this->ctx, A.view(), C.view(), piv.to_span(), ws.to_span(), info.to_span()); }));
        this->ctx->wait();
        EXPECT_TRUE(C.padding_intact());
        for (int b = 0; b < kBatch; ++b) {
            EXPECT_EQ(info[b], 0) << "item " << b;
            const auto c = C.item(b);
            auto r = matmul(A0[b], c, n, n, n);
            for (int i = 0; i < n; ++i) r[i + std::size_t(i) * n] -= T(1);
            EXPECT_VERIFY(T, Check::solve, n, fro(r) / (fro(A0[b]) * fro(c))) << "item " << b;
        }
    }
}

TYPED_TEST(VendorPtrArrayTest, PotrfRawView) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::Bk;
    if constexpr (!select::solver_vendor_available<B>) {
        GTEST_SKIP() << "no solver vendor";
    } else {
        const ScopedEnvVar pin("BATCHLAS_POTRF_ROUTE", "vendor");
        const int n = 12;
        RawBatch<T> A(n, n);
        std::vector<std::vector<T>> A0(kBatch);
        for (int b = 0; b < kBatch; ++b) {
            std::vector<T> M(std::size_t(n) * n);
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i) M[i + std::size_t(j) * n] = entry<T>(i, j, b, 8);
            A0[b] = matmul(M, op_of(M, n, Transpose::ConjTrans), n, n, n);
            for (int i = 0; i < n; ++i) A0[b][i + std::size_t(i) * n] += T(Real<T>(n));
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i) A.at(i, j, b) = A0[b][i + std::size_t(j) * n];
        }
        UnifiedVector<int32_t> info(kBatch, int32_t(-1));
        UnifiedVector<std::byte> ws(potrf_buffer_size<B, T>(*this->ctx, A.view(), Uplo::Lower));
        ASSERT_TRUE(runs([&] { potrf<B, T>(*this->ctx, A.view(), Uplo::Lower, ws.to_span(), info.to_span()); }));
        this->ctx->wait();
        EXPECT_TRUE(A.padding_intact());
        for (int b = 0; b < kBatch; ++b) {
            EXPECT_EQ(info[b], 0) << "item " << b;
            const auto f = A.item(b);
            std::vector<T> L(std::size_t(n) * n, T(0));
            for (int j = 0; j < n; ++j)
                for (int i = j; i < n; ++i) L[i + std::size_t(j) * n] = f[i + std::size_t(j) * n];
            const auto LLh = matmul(L, op_of(L, n, Transpose::ConjTrans), n, n, n);
            EXPECT_VERIFY(T, Check::factorization, n, fro(diff(LLh, A0[b])) / fro(A0[b])) << "item " << b;
        }
    }
}

// A and B as column slices of one Matrix share its pointer array: filling B's after A's left
// both arguments pointing at B.
TYPED_TEST(VendorPtrArrayTest, GetrsSlicesOfOneMatrix) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::Bk;
    if constexpr (!select::factorization_vendor_available<B>) {
        GTEST_SKIP() << "no factorization vendor";
    } else {
        const ScopedEnvVar pin_f("BATCHLAS_GETRF_ROUTE", "tiny");
        const ScopedEnvVar pin("BATCHLAS_GETRS_ROUTE", "vendor");
        const int n = 12, nrhs = 4;
        Matrix<T, MatrixFormat::Dense> M(n, n + nrhs, kBatch);
        auto Mv = M.view();
        for (int b = 0; b < kBatch; ++b)
            for (int j = 0; j < n + nrhs; ++j)
                for (int i = 0; i < n; ++i)
                    Mv(i, j, b) = j < n ? entry<T>(i, j, b, 9) + (i == j ? T(2) : T(0)) : entry<T>(i, j, b, 10);
        std::vector<std::vector<T>> A0(kBatch), B0(kBatch);
        for (int b = 0; b < kBatch; ++b) {
            A0[b].resize(std::size_t(n) * n);
            B0[b].resize(std::size_t(n) * nrhs);
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i) A0[b][i + std::size_t(j) * n] = Mv(i, j, b);
            for (int j = 0; j < nrhs; ++j)
                for (int i = 0; i < n; ++i) B0[b][i + std::size_t(j) * n] = Mv(i, n + j, b);
        }
        const auto A = Mv(Slice(), Slice(0, n));
        const auto X = Mv(Slice(), Slice(n, n + nrhs));
        UnifiedVector<int64_t> piv(std::size_t(n) * kBatch, int64_t(0));
        UnifiedVector<std::byte> wf(getrf_buffer_size<B, T>(*this->ctx, A));
        getrf<B, T>(*this->ctx, A, piv.to_span(), wf.to_span());
        UnifiedVector<std::byte> ws(getrs_buffer_size<B, T>(*this->ctx, A, X, Transpose::NoTrans));
        ASSERT_TRUE(runs([&] { getrs<B, T>(*this->ctx, A, X, Transpose::NoTrans, piv.to_span(), ws.to_span()); }));
        this->ctx->wait();
        for (int b = 0; b < kBatch; ++b) {
            std::vector<T> x(std::size_t(n) * nrhs);
            for (int j = 0; j < nrhs; ++j)
                for (int i = 0; i < n; ++i) x[i + std::size_t(j) * n] = Mv(i, n + j, b);
            const double err = fro(diff(matmul(A0[b], x, n, n, nrhs), B0[b])) / (fro(A0[b]) * fro(x) + fro(B0[b]));
            EXPECT_VERIFY(T, Check::solve, n, err) << "item " << b;
        }
    }
}
