#include <gtest/gtest.h>
#include <batchlas/blas/linalg.hh>
#include <batchlas/util/sycl-vector.hh>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <vector>
#include <batchlas/util/sycl-device-queue.hh>

using namespace batchlas;

template <typename T, Backend B>
struct OrgqrConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

#include "test_utils.hh"
using OrgqrTestTypes = typename test_utils::backend_types<OrgqrConfig>::type;

template <typename Config>
class OrgqrTest : public test_utils::BatchLASTest<Config> {
protected:
    Transpose trans_op = test_utils::is_complex<typename Config::ScalarType>() ? Transpose::ConjTrans : Transpose::Trans;
};

TYPED_TEST_SUITE(OrgqrTest, OrgqrTestTypes);

TYPED_TEST(OrgqrTest, SingleMatrix) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 4;

    Matrix<T, MatrixFormat::Dense> A = Matrix<T, MatrixFormat::Dense>::Random(n, n);
    UnifiedVector<T> tau(n);
    UnifiedVector<std::byte> ws_geqrf(geqrf_buffer_size(*this->ctx, A.view(), tau.to_span()));
    geqrf(*this->ctx, A.view(), tau.to_span(), ws_geqrf.to_span());
    this->ctx->wait();

    UnifiedVector<std::byte> ws_orgqr(orgqr_buffer_size(*this->ctx, A.view(), tau.to_span()));
    orgqr(*this->ctx, A.view(), tau.to_span(), ws_orgqr.to_span());
    this->ctx->wait();

    Matrix<T, MatrixFormat::Dense> Result(n, n);
    gemm(*this->ctx, A.view(), A.view(), Result.view(), {.transA = this->trans_op});
    this->ctx->wait();

    auto r = Result.data();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            T expected = (i == j) ? T(1) : T(0);
            test_utils::assert_near(r[i * Result.ld() + j], expected);
        }
    }
}

TYPED_TEST(OrgqrTest, BatchedMatrices) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 4;
    const int batch = 3;

    Matrix<T, MatrixFormat::Dense> A = Matrix<T, MatrixFormat::Dense>::Random(n, n, false, batch);
    UnifiedVector<T> tau(n * batch);
    UnifiedVector<std::byte> ws_geqrf(geqrf_buffer_size(*this->ctx, A.view(), tau.to_span()));
    geqrf(*this->ctx, A.view(), tau.to_span(), ws_geqrf.to_span());
    this->ctx->wait();

    UnifiedVector<std::byte> ws_orgqr(orgqr_buffer_size(*this->ctx, A.view(), tau.to_span()));
    orgqr(*this->ctx, A.view(), tau.to_span(), ws_orgqr.to_span());
    this->ctx->wait();

    Matrix<T, MatrixFormat::Dense> Result(n, n, batch);
    gemm(*this->ctx, A.view(), A.view(), Result.view(), {.transA = this->trans_op});
    this->ctx->wait();

    auto r = Result.data();
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                T expected = (i == j) ? T(1) : T(0);
                test_utils::assert_near(r[b * Result.stride() + i * Result.ld() + j], expected);
            }
        }
    }
}


// ===========================================================================
// WP5 -- ORGQR AGAINST A HOST REFERENCE.
//
// The two tests above are the pre-WP5 coverage: n = 4, Q^H Q == I through a
// device gemm, item 0 and its two neighbours. Three things they cannot see, and
// each has a recorded precedent in this repository:
//
//   * Q^H Q == I ALONE DOES NOT SAY Q IS *THIS* A's Q. Any orthonormal matrix
//     passes it -- a driver that dropped a reflector, or applied the reflectors
//     in the wrong order, still returns something orthonormal. So the test below
//     also checks || Q R - A ||_F / ||A||_F with R read out of the geqrf factor.
//
//   * ld == rows AND stride == ld*cols. Matrix<T>(n, n, batch) has both, so the
//     two lines of every launcher that read A.ld() and A.stride() were
//     structurally unfalsifiable here. trsm_native.cc:590-599 records that exact
//     failure: the 6-arg MatrixView constructor defaults stride to ld*cols, after
//     which every batch item but the first reads the wrong matrix.
//
//   * m == n ONLY. orgqr's whole point is the first n columns of an m x n Q.
//
// The oracle is a host multiply-back in double, computed here from the input this
// file generated -- never the vendor, because a vendor reference is inert in the
// vendor-free build this campaign exists for.
// ===========================================================================
namespace orgqr_wp5 {

template <class T> struct Prom { using type = double; };
template <class R> struct Prom<std::complex<R>> { using type = std::complex<double>; };

inline double up2(float x) { return double(x); }
inline double up2(double x) { return x; }
inline std::complex<double> up2(std::complex<float> x) { return {double(x.real()), double(x.imag())}; }
inline std::complex<double> up2(std::complex<double> x) { return x; }
inline double cj2(double x) { return x; }
inline std::complex<double> cj2(std::complex<double> x) { return std::conj(x); }
inline double re2(double x) { return x; }
inline double re2(std::complex<double> x) { return x.real(); }
inline double im2(double) { return 0.0; }
inline double im2(std::complex<double> x) { return x.imag(); }
inline double abs2v(double x) { return std::fabs(x); }
inline double abs2v(std::complex<double> x) { return std::abs(x); }

template <class T> inline T mk2(double re, double im);
template <> inline float mk2<float>(double re, double) { return float(re); }
template <> inline double mk2<double>(double re, double) { return re; }
template <> inline std::complex<float> mk2<std::complex<float>>(double re, double im) {
    return {float(re), float(im)};
}
template <> inline std::complex<double> mk2<std::complex<double>>(double re, double im) {
    return {re, im};
}

struct Rng2 {
    uint64_t s;
    explicit Rng2(uint64_t seed) : s(seed * 6364136223846793005ULL + 1442695040888963407ULL) {}
    double next() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return double(int32_t(uint32_t(s >> 32))) / 2147483648.0;
    }
};

// || Q^H Q - I ||_F / sqrt(n), in double.
template <typename T>
double orth(const T* Q, int m, int n, int ld) {
    using D = typename Prom<T>::type;
    double num = 0.0;
    for (int a = 0; a < n; ++a)
        for (int b = 0; b < n; ++b) {
            D acc(0);
            for (int r = 0; r < m; ++r)
                acc += cj2(up2(Q[static_cast<size_t>(a) * ld + r])) *
                       up2(Q[static_cast<size_t>(b) * ld + r]);
            const D d = acc - D(a == b ? 1 : 0);
            num += re2(d) * re2(d) + im2(d) * im2(d);
        }
    return std::sqrt(num) / std::sqrt(double(n));
}

// || Q R - A ||_F / ||A||_F, R read out of the geqrf factor's upper triangle.
template <typename T>
double recon(const T* Q, const T* F, const T* A0, int m, int n, int ld) {
    using D = typename Prom<T>::type;
    double num = 0.0, den = 0.0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            D acc(0);
            for (int p = 0; p <= j; ++p)
                acc += up2(Q[static_cast<size_t>(p) * ld + i]) *
                       up2(F[static_cast<size_t>(j) * ld + p]);
            const D a = up2(A0[static_cast<size_t>(j) * ld + i]);
            const D d = acc - a;
            num += re2(d) * re2(d) + im2(d) * im2(d);
            den += re2(a) * re2(a) + im2(a) * im2(a);
        }
    return den > 0 ? std::sqrt(num) / std::sqrt(den) : std::sqrt(num);
}

}  // namespace orgqr_wp5

TYPED_TEST(OrgqrTest, QIsOrthonormalAndReconstructsAAtEveryBatchItem) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    using namespace orgqr_wp5;

    struct S { int m, n, batch; };
    // m == n and m > n; one n that is a multiple of every block width this tree
    // uses and one that is not; and batches > 1 so a stride defect has somewhere
    // to hide.
    const S shapes[] = {{32, 32, 3}, {40, 24, 3}, {64, 33, 2}, {96, 96, 2}};
    for (const S& s : shapes) {
        const int ld = s.m + 5;
        const int stride = ld * s.n + 11;
        UnifiedVector<T> buf(static_cast<size_t>(stride) * s.batch, mk2<T>(-9.75e3, 4.5e3));
        UnifiedVector<T*> ptrs(static_cast<size_t>(s.batch), nullptr);
        Rng2 rg(4242u + 13u * unsigned(s.m) + unsigned(s.n));
        for (int b = 0; b < s.batch; ++b)
            for (int j = 0; j < s.n; ++j)
                for (int i = 0; i < s.m; ++i)
                    buf[static_cast<size_t>(b) * stride + static_cast<size_t>(j) * ld + i] =
                        mk2<T>(rg.next(), rg.next());
        const std::vector<T> A0(buf.begin(), buf.end());

        MatrixView<T, MatrixFormat::Dense> V(buf.data(), s.m, s.n, ld, stride, s.batch,
                                             ptrs.data());
        UnifiedVector<T> tau(static_cast<size_t>(std::min(s.m, s.n)) * s.batch,
                             mk2<T>(-12345.0, -12345.0));

        UnifiedVector<std::byte> wg(std::max<size_t>(
            1, geqrf_buffer_size<B, T>(*this->ctx, V, tau.to_span())));
        geqrf<B, T>(*this->ctx, V, tau.to_span(), wg.to_span());
        this->ctx->wait();
        const std::vector<T> F(buf.begin(), buf.end());   // orgqr overwrites it

        UnifiedVector<std::byte> wo(std::max<size_t>(
            1, orgqr_buffer_size<B, T>(*this->ctx, V, tau.to_span())));
        orgqr<B, T>(*this->ctx, V, tau.to_span(), wo.to_span());
        this->ctx->wait();

        const double tol = 0.5 * double(s.m + s.n) *
                           double(std::numeric_limits<typename base_type<T>::type>::epsilon());
        for (int b = 0; b < s.batch; ++b) {
            const size_t off = static_cast<size_t>(b) * stride;
            EXPECT_LE(orth<T>(buf.data() + off, s.m, s.n, ld), tol)
                << "Q is not orthonormal at b=" << b << " (m=" << s.m << " n=" << s.n << ")";
            EXPECT_LE(recon<T>(buf.data() + off, F.data() + off, A0.data() + off, s.m, s.n, ld),
                      tol)
                << "Q R != A at b=" << b << " (m=" << s.m << " n=" << s.n
                << ") -- Q is orthonormal but it is not THIS A's Q";
        }
        // A broadcast of item 0 over the batch would pass every check above.
        if (s.batch > 1) {
            bool differ = false;
            for (int j = 0; j < s.n && !differ; ++j)
                for (int i = 0; i < s.m && !differ; ++i)
                    if (abs2v(up2(buf[static_cast<size_t>(j) * ld + i]) -
                              up2(buf[static_cast<size_t>(s.batch - 1) * stride +
                                      static_cast<size_t>(j) * ld + i])) > 0.0)
                        differ = true;
            EXPECT_TRUE(differ) << "the first and last batch items' Q are identical, so this "
                                   "shape cannot see a batch-stride defect";
        }
        if (this->HasFailure()) return;
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

