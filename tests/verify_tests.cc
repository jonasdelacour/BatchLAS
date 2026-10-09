// Host-only tests of batchlas::verify (docs/design/verification.md): scalars, norms, items, tolerances.

#include <batchlas/verify/items.hh>
#include <batchlas/verify/norms.hh>
#include <batchlas/verify/scalar.hh>
#include <batchlas/verify/tolerance.hh>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
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
