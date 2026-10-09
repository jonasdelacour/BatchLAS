// Host-only tests of batchlas::verify (docs/design/verification.md): scalars, norms, items, tolerances, inputs.

#include <batchlas/verify/inputs.hh>
#include <batchlas/verify/items.hh>
#include <batchlas/verify/norms.hh>
#include <batchlas/verify/scalar.hh>
#include <batchlas/verify/tolerance.hh>

#include <gtest/gtest.h>

#if BATCHLAS_VERIFY_HAVE_LAPACKE
#include <lapacke.h>
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <complex>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using batchlas::MatrixFormat;
using batchlas::MatrixView;
using batchlas::verify::Check;

namespace {
const double kNaN = std::numeric_limits<double>::quiet_NaN();
}

TEST(Scalar, NanmaxPropagates) {
    EXPECT_TRUE(std::isnan(batchlas::verify::nanmax(kNaN, 1.0)));
    EXPECT_TRUE(std::isnan(batchlas::verify::nanmax(1.0, kNaN)));
    EXPECT_EQ(batchlas::verify::nanmax(1.0, 2.0), 2.0);
}

TEST(Scalar, EpsIsUnitRoundoff) {
    EXPECT_EQ(batchlas::verify::eps<float>(), std::ldexp(1.0, -24));
    EXPECT_EQ(batchlas::verify::eps<std::complex<float>>(), std::ldexp(1.0, -24));
    EXPECT_EQ(batchlas::verify::eps<double>(), std::ldexp(1.0, -53));
    EXPECT_EQ(batchlas::verify::eps<std::complex<double>>(), std::ldexp(1.0, -53));
}

TEST(Scalar, RngMatchesFactorBench) {
    // First four next() values of Rng(12345), recorded from a copy of the Rng struct in
    // benchmarks/factor_bench.cc as of commit b4e9c1b2.
    batchlas::verify::Rng rng(12345);
    EXPECT_EQ(rng.next(), 0x1.0fc129b8p-1);
    EXPECT_EQ(rng.next(), -0x1.d47befbp-3);
    EXPECT_EQ(rng.next(), -0x1.5068e7bp-2);
    EXPECT_EQ(rng.next(), 0x1.4d723aap-1);
}

TEST(Scalar, MakeAndPromote) {
    EXPECT_EQ(batchlas::verify::make<float>(1.5, 7.0), 1.5f);
    const auto z = batchlas::verify::make<std::complex<float>>(1.5, -2.0);
    EXPECT_EQ(batchlas::verify::up(z), std::complex<double>(1.5, -2.0));
    EXPECT_EQ(batchlas::verify::abs(std::complex<double>(3.0, 4.0)), 5.0);
    EXPECT_EQ(batchlas::verify::conj(std::complex<double>(3.0, 4.0)), std::complex<double>(3.0, -4.0));
}

TEST(Items, DefaultIncludesLastAndMiddle) {
    using batchlas::verify::default_items;
    EXPECT_EQ(default_items(1), (std::vector<int>{0}));
    EXPECT_EQ(default_items(2), (std::vector<int>{0, 1}));
    EXPECT_EQ(default_items(9), (std::vector<int>{0, 4, 8}));
    EXPECT_EQ(batchlas::verify::all_items(3), (std::vector<int>{0, 1, 2}));
}

TEST(Norms, ReadThroughLdAndStride) {
    constexpr int rows = 3, cols = 2, ld = 5, stride = 17, batch = 3;
    std::vector<double> buf(stride * batch, 1e30);
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < cols; ++j)
            for (int i = 0; i < rows; ++i) buf[b * stride + j * ld + i] = (b + 1) * (i + 1) * (j == 0 ? 1.0 : -1.0);
    MatrixView<double, MatrixFormat::Dense> A(buf.data(), rows, cols, ld, stride, batch);

    // Item 2 holds 3*(1,2,3) and -3*(1,2,3): sum of squares 2 * 9 * 14.
    EXPECT_DOUBLE_EQ(batchlas::verify::frobenius(A, 2), std::sqrt(2.0 * 9.0 * 14.0));
    EXPECT_DOUBLE_EQ(batchlas::verify::max_abs(A, 2), 9.0);
    EXPECT_DOUBLE_EQ(batchlas::verify::one_norm(A, 2), 18.0);

    buf[2 * stride + 1 * ld + 2] = kNaN;
    EXPECT_TRUE(std::isnan(batchlas::verify::frobenius(A, 2)));
    EXPECT_TRUE(std::isnan(batchlas::verify::max_abs(A, 2)));
    EXPECT_TRUE(std::isnan(batchlas::verify::one_norm(A, 2)));
    EXPECT_DOUBLE_EQ(batchlas::verify::frobenius(A, 0), std::sqrt(2.0 * 14.0));
}

TEST(Norms, ComplexUsesModulus) {
    std::vector<std::complex<float>> buf{{3.0f, 4.0f}, {0.0f, 0.0f}, {0.0f, -12.0f}, {5.0f, 0.0f}};
    MatrixView<std::complex<float>, MatrixFormat::Dense> A(buf.data(), 2, 2, 2, 4, 1);
    EXPECT_DOUBLE_EQ(batchlas::verify::frobenius(A, 0), std::sqrt(25.0 + 144.0 + 25.0));
    EXPECT_DOUBLE_EQ(batchlas::verify::max_abs(A, 0), 12.0);
    EXPECT_DOUBLE_EQ(batchlas::verify::one_norm(A, 0), 12.0 + 5.0);
}

TEST(Norms, KernelViewAccepted) {
    std::vector<double> buf{1.0, 2.0, 99.0, 3.0, 4.0, 99.0};
    batchlas::KernelMatrixView<double, MatrixFormat::Dense> A(buf.data(), 2, 2, 3, 6, 1);
    EXPECT_DOUBLE_EQ(batchlas::verify::frobenius(A, 0), std::sqrt(30.0));
}

TEST(Tolerance, BoundAndPass) {
    using batchlas::verify::bound;
    using batchlas::verify::pass;
    EXPECT_EQ(bound<float>(Check::blas, 8), 4.0 * 8.0 * std::ldexp(1.0, -24));
    EXPECT_EQ(bound<double>(Check::eigen_residual, 3), 32.0 * 3.0 * std::ldexp(1.0, -53));
    EXPECT_EQ(bound<double>(Check::solve, 0), bound<double>(Check::solve, 1));
    EXPECT_FALSE(pass<double>(Check::solve, 10, kNaN));
    const double b = bound<double>(Check::solve, 10);
    EXPECT_TRUE(pass<double>(Check::solve, 10, b));
    EXPECT_FALSE(pass<double>(Check::solve, 10, std::nextafter(b, 1.0)));
}

TEST(Tolerance, RecordsWhenAsked) {
    const std::string path = ::testing::TempDir() + "verify_record.txt";
    std::remove(path.c_str());
    ASSERT_EQ(setenv("BATCHLAS_VERIFY_RECORD", path.c_str(), 1), 0);
    (void)batchlas::verify::pass<std::complex<float>>(Check::orthogonality, 7, 1e-6);
    unsetenv("BATCHLAS_VERIFY_RECORD");
    (void)batchlas::verify::pass<float>(Check::solve, 7, 1e-6);  // unset: nothing appended

    std::ifstream in(path);
    std::string line;
    ASSERT_TRUE(std::getline(in, line));
    std::istringstream fields(line);
    std::string kind, dtype;
    int n = 0;
    double value = 0, bnd = 0;
    ASSERT_TRUE(bool(fields >> kind >> dtype >> n >> value >> bnd));
    EXPECT_EQ(kind, "orthogonality");
    EXPECT_EQ(dtype, "cfloat");
    EXPECT_EQ(n, 7);
    EXPECT_EQ(value, 1e-6);
    EXPECT_EQ(bnd, batchlas::verify::bound<std::complex<float>>(Check::orthogonality, 7));
    EXPECT_FALSE(std::getline(in, line));
    std::remove(path.c_str());
}

// ----------------------------------------------------------------------------------- Inputs

namespace {

using cfloat = std::complex<float>;
using cdouble = std::complex<double>;

constexpr int kN = 7, kLd = 9, kBatch = 3;
constexpr long long kStride = 70;
constexpr std::uint64_t kSeed = 12345;

template <class T> T sentinel() { return batchlas::verify::make<T>(-777.0, -555.0); }

template <class T> std::vector<T> padded_buffer() { return std::vector<T>(kStride * kBatch, sentinel<T>()); }

template <class T> MatrixView<T, MatrixFormat::Dense> square_view(std::vector<T>& buf) {
    return MatrixView<T, MatrixFormat::Dense>(buf.data(), kN, kN, kLd, kStride, kBatch);
}

// FNV-1a 64 over the raw bytes of the whole ld x cols x stride x batch buffer, padding included.
template <class T> std::uint64_t fnv1a(const std::vector<T>& v) {
    std::uint64_t h = 1469598103934665603ULL;
    const unsigned char* p = reinterpret_cast<const unsigned char*>(v.data());
    for (std::size_t i = 0; i < v.size() * sizeof(T); ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

// Recorded from copies of fill_spd / fill_lu / fill_gauss in benchmarks/factor_bench.cc (last changed
// in commit b4e9c1b2) run on a std::vector, built with /opt/dpcpp-cuda clang++ (left-to-right argument
// evaluation, which the bench relies on): n=7, ld=9, stride=70, batch=3, seed 12345, padding
// pre-filled with make<T>(-777, -555).
template <class T> struct Golden;
template <> struct Golden<float> {
    static constexpr std::uint64_t spd = 0x750b3e8ce6cbdfbeULL, lu = 0xfc830640c86ea95bULL, gauss = 0x00731694f7177b43ULL;
};
template <> struct Golden<double> {
    static constexpr std::uint64_t spd = 0xec9db1114108317fULL, lu = 0xbc2af2584858c945ULL, gauss = 0x9115ccbde0ccdd8dULL;
};
template <> struct Golden<cfloat> {
    static constexpr std::uint64_t spd = 0xb5ea33f1e79a9fccULL, lu = 0x38a4780ba4d641c9ULL, gauss = 0x3b2bd1b5208c3c27ULL;
};
template <> struct Golden<cdouble> {
    static constexpr std::uint64_t spd = 0x369193e572a23086ULL, lu = 0xefd18494761318c5ULL, gauss = 0xfcebb90a62babe52ULL;
};

template <class T> void check_spd() {
    auto buf = padded_buffer<T>();
    batchlas::verify::fill_spd(square_view(buf));
    EXPECT_EQ(fnv1a(buf), Golden<T>::spd);
}
template <class T> void check_lu() {
    auto buf = padded_buffer<T>();
    batchlas::verify::fill_lu(square_view(buf), kSeed);
    EXPECT_EQ(fnv1a(buf), Golden<T>::lu);
}
template <class T> void check_gauss() {
    auto buf = padded_buffer<T>();
    batchlas::verify::fill_gauss(square_view(buf), kSeed);
    EXPECT_EQ(fnv1a(buf), Golden<T>::gauss);
}

// Every padding element (rows kN..kLd-1 of each column, the gap between items) still holds the sentinel.
template <class T> void expect_padding(const std::vector<T>& buf, int rows, int cols, int ld) {
    for (int b = 0; b < kBatch; ++b)
        for (long long o = 0; o < kStride; ++o) {
            const long long idx = b * kStride + o;
            const bool inside = (o % ld) < rows && (o / ld) < cols;
            const T s = sentinel<T>();
            if (!inside) ASSERT_EQ(std::memcmp(&buf[idx], &s, sizeof(T)), 0) << "padding written at " << idx;
        }
}

template <class T> void check_padding() {
    {
        auto buf = padded_buffer<T>();
        batchlas::verify::fill_spd(square_view(buf));
        expect_padding(buf, kN, kN, kLd);
    }
    {
        auto buf = padded_buffer<T>();
        batchlas::verify::fill_lu(square_view(buf), kSeed);
        expect_padding(buf, kN, kN, kLd);
    }
    {
        auto buf = padded_buffer<T>();
        batchlas::verify::fill_graded_hermitian(square_view(buf), 3.0, kSeed);
        expect_padding(buf, kN, kN, kLd);
    }
    {  // rectangular: rows 6, cols 4 inside ld 9
        auto buf = padded_buffer<T>();
        MatrixView<T, MatrixFormat::Dense> A(buf.data(), 6, 4, kLd, kStride, kBatch);
        batchlas::verify::fill_gauss(A, kSeed);
        expect_padding(buf, 6, 4, kLd);
        buf = padded_buffer<T>();
        MatrixView<T, MatrixFormat::Dense> R(buf.data(), 6, 4, kLd, kStride, kBatch);
        batchlas::verify::fill_random(R, kSeed);
        expect_padding(buf, 6, 4, kLd);
    }
    {  // reflectors: m=8, k=5, tau inc 2 stride 13
        std::vector<T> buf(kStride * kBatch, sentinel<T>());
        std::vector<T> tau(13 * kBatch, sentinel<T>());
        MatrixView<T, MatrixFormat::Dense> A(buf.data(), 8, 5, kLd, kStride, kBatch);
        batchlas::VectorView<T> t(tau.data(), 5, kBatch, 2, 13);
        batchlas::verify::fill_reflectors(A, t, kSeed);
        expect_padding(buf, 8, 5, kLd);
        for (int b = 0; b < kBatch; ++b)
            for (int o = 0; o < 13; ++o)
                if (const T s = sentinel<T>(); o % 2 != 0 || o / 2 >= 5) ASSERT_EQ(std::memcmp(&tau[b * 13 + o], &s, sizeof(T)), 0);
    }
}

}  // namespace

TEST(Inputs, SpdMatchesFactorBench) {
    check_spd<float>();
    check_spd<double>();
    check_spd<cfloat>();
    check_spd<cdouble>();
}

TEST(Inputs, LuMatchesFactorBench) {
    check_lu<float>();
    check_lu<double>();
    check_lu<cfloat>();
    check_lu<cdouble>();
}

TEST(Inputs, GaussMatchesFactorBench) {
    check_gauss<float>();
    check_gauss<double>();
    check_gauss<cfloat>();
    check_gauss<cdouble>();
}

TEST(Inputs, PaddingUntouched) {
    check_padding<float>();
    check_padding<double>();
    check_padding<cfloat>();
    check_padding<cdouble>();
}

TEST(Inputs, RandomIsItemMajorColumnMajor) {
    std::vector<cdouble> buf(kStride * kBatch, sentinel<cdouble>());
    MatrixView<cdouble, MatrixFormat::Dense> A(buf.data(), 3, 2, kLd, kStride, kBatch);
    batchlas::verify::fill_random(A, 7);
    batchlas::verify::Rng rng(7);
    for (int b = 0; b < kBatch; ++b)
        for (int c = 0; c < 2; ++c)
            for (int r = 0; r < 3; ++r) {
                const double re = rng.next();
                const double im = rng.next();
                EXPECT_EQ(buf[b * kStride + c * kLd + r], cdouble(re, im));
            }
}

TEST(Inputs, SquareOnlyGeneratorsThrow) {
    std::vector<double> buf(kStride * kBatch, 0.0);
    MatrixView<double, MatrixFormat::Dense> A(buf.data(), 5, 4, kLd, kStride, kBatch);
    EXPECT_THROW(batchlas::verify::fill_spd(A), std::invalid_argument);
    EXPECT_THROW(batchlas::verify::fill_lu(A, 1), std::invalid_argument);
    EXPECT_THROW(batchlas::verify::fill_graded_hermitian(A, 2.0, 1), std::invalid_argument);
}

namespace {

template <class T> void check_graded_hermitian(double log10_cond, double tol) {
    constexpr int n = 9, ld = 12, batch = 3;
    constexpr long long stride = 130;
    std::vector<T> buf(stride * batch, sentinel<T>());
    MatrixView<T, MatrixFormat::Dense> A(buf.data(), n, n, ld, stride, batch);
    batchlas::verify::fill_graded_hermitian(A, log10_cond, 99);

    // lambda_i = 10^(-c i/(n-1)); trace = sum lambda and ||A||_F^2 = sum lambda^2 hold for any Q.
    double tr = 0, fro2 = 0;
    for (int i = 0; i < n; ++i) {
        const double l = std::pow(10.0, -log10_cond * i / (n - 1));
        tr += l;
        fro2 += l * l;
    }
    for (int b = 0; b < batch; ++b) {
        const T* p = buf.data() + b * stride;
        double rel = 0, trace = 0, imag_diag = 0;
        for (int c = 0; c < n; ++c) {
            const cdouble d = cdouble(batchlas::verify::up(p[c * ld + c]));
            trace += d.real();
            imag_diag = batchlas::verify::nanmax(imag_diag, std::abs(d.imag()));
            for (int r = 0; r < n; ++r)
                rel = batchlas::verify::nanmax(
                    rel, batchlas::verify::abs(batchlas::verify::up(p[c * ld + r]) - batchlas::verify::conj(batchlas::verify::up(p[r * ld + c]))));
        }
        const double fro = batchlas::verify::frobenius(A, b);
        EXPECT_LE(rel, 1e-12 * fro);
        EXPECT_EQ(imag_diag, 0.0);
        EXPECT_NEAR(trace, tr, tol * tr);
        EXPECT_NEAR(fro * fro, fro2, tol * fro2);
    }
    // Items differ (one shared Rng stream, item-major).
    EXPECT_NE(buf[0 * stride + 1], buf[1 * stride + 1]);
}

}  // namespace

TEST(Inputs, GradedHermitianIsHermitianWithSpectrum) {
    check_graded_hermitian<double>(6.0, 1e-12);
    check_graded_hermitian<cdouble>(6.0, 1e-12);
    check_graded_hermitian<float>(2.0, 1e-5);
    check_graded_hermitian<cfloat>(2.0, 1e-5);
}

TEST(Inputs, GradedHermitianHasRequestedCondition) {
#if BATCHLAS_VERIFY_HAVE_LAPACKE
    const int n = 9, ld = 12, batch = 3;
    const long long stride = 130;
    const double c = 6.0;
    {
        std::vector<double> buf(stride * batch, sentinel<double>());
        MatrixView<double, MatrixFormat::Dense> A(buf.data(), n, n, ld, stride, batch);
        batchlas::verify::fill_graded_hermitian(A, c, 99);
        for (int b = 0; b < batch; ++b) {
            std::vector<double> w(n);
            ASSERT_EQ(LAPACKE_dsyev(LAPACK_COL_MAJOR, 'N', 'L', n, buf.data() + b * stride, ld, w.data()), 0);
            EXPECT_NEAR(w[n - 1] / w[0] / std::pow(10.0, c), 1.0, 0.01);
        }
    }
    {
        std::vector<cdouble> buf(stride * batch, sentinel<cdouble>());
        MatrixView<cdouble, MatrixFormat::Dense> A(buf.data(), n, n, ld, stride, batch);
        batchlas::verify::fill_graded_hermitian(A, c, 99);
        for (int b = 0; b < batch; ++b) {
            std::vector<double> w(n);
            ASSERT_EQ(LAPACKE_zheev(LAPACK_COL_MAJOR, 'N', 'L', n, reinterpret_cast<lapack_complex_double*>(buf.data() + b * stride), ld, w.data()), 0);
            EXPECT_NEAR(w[n - 1] / w[0] / std::pow(10.0, c), 1.0, 0.01);
        }
    }
#else
    GTEST_SKIP() << "built without LAPACKE: eigenvalue-ratio half skipped";
#endif
}

namespace {

// H_i = I - tau_i v_i v_i^H, v_i = (0..0, 1, A(i+1:m, i)); unitary to roundoff for every larfg output.
template <class T> void check_reflectors(double tol) {
    constexpr int m = 8, k = 5, ld = 11, batch = 3, tinc = 2, tstride = 13;
    constexpr long long stride = 100;
    std::vector<T> buf(stride * batch, sentinel<T>());
    std::vector<T> tau(tstride * batch, sentinel<T>());
    MatrixView<T, MatrixFormat::Dense> A(buf.data(), m, k, ld, stride, batch);
    batchlas::VectorView<T> t(tau.data(), k, batch, tinc, tstride);
    batchlas::verify::fill_reflectors(A, t, 4242);
    using D = batchlas::verify::promoted_t<T>;
    for (int b = 0; b < batch; ++b)
        for (int i = 0; i < k; ++i) {
            std::vector<D> v(m, D(0));
            v[i] = D(1);
            for (int r = i + 1; r < m; ++r) v[r] = batchlas::verify::up(buf[b * stride + i * ld + r]);
            // Diagonal and above are large finite values nothing may read.
            for (int r = 0; r <= i; ++r) EXPECT_GE(batchlas::verify::abs(batchlas::verify::up(buf[b * stride + i * ld + r])), 32.0);
            const D ti = batchlas::verify::up(tau[b * tstride + i * tinc]);
            double worst = 0;
            for (int c = 0; c < m; ++c)
                for (int r = 0; r < m; ++r) {
                    // (H^H H)(r,c) = sum_q conj(H(q,r)) H(q,c)
                    D acc = 0;
                    for (int q = 0; q < m; ++q) {
                        const D hr = (q == r ? D(1) : D(0)) - ti * v[q] * batchlas::verify::conj(v[r]);
                        const D hc = (q == c ? D(1) : D(0)) - ti * v[q] * batchlas::verify::conj(v[c]);
                        acc += batchlas::verify::conj(hr) * hc;
                    }
                    worst = batchlas::verify::nanmax(worst, batchlas::verify::abs(acc - (r == c ? D(1) : D(0))));
                }
            EXPECT_LE(worst, tol) << "b=" << b << " i=" << i;
        }
}

}  // namespace

TEST(Inputs, ReflectorsHaveUnitTau) {
    check_reflectors<double>(1e-12);
    check_reflectors<cdouble>(1e-12);
    check_reflectors<float>(1e-5);
    check_reflectors<cfloat>(1e-5);
    // tau == 0 occurs for real data when the column below the diagonal is empty: last reflector of a square panel.
    std::vector<double> buf(kStride, 0.0), tau(4, -1.0);
    MatrixView<double, MatrixFormat::Dense> A(buf.data(), 4, 4, kLd, kStride, 1);
    batchlas::VectorView<double> t(tau.data(), 4, 1);
    batchlas::verify::fill_reflectors(A, t, 5);
    EXPECT_EQ(tau[3], 0.0);
}
