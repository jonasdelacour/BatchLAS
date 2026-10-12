#include <gtest/gtest.h>
#include <batchlas/blas/linalg.hh>
#include <sycl/sycl.hpp>

#include "../src/util/philox.hh"

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace batchlas;

namespace {

struct Kat {
    philox::Words ctr;
    std::uint32_t k0, k1;
    philox::Words want;
};

// Random123 kat_vectors, philox4x32 10 rounds.
const Kat kKats[] = {
    {{{0u, 0u, 0u, 0u}}, 0u, 0u, {{0x6627e8d5u, 0xe169c58du, 0xbc57ac4cu, 0x9b00dbd8u}}},
    {{{0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu}}, 0xffffffffu, 0xffffffffu,
     {{0x408f276du, 0x41c83b0eu, 0xa20bc7c6u, 0x6d5451fdu}}},
    {{{0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u}}, 0xa4093822u, 0x299f31d0u,
     {{0xd16cfe09u, 0x94fdccebu, 0x5001e420u, 0x24126ea1u}}},
};
constexpr int kNumKats = sizeof(kKats) / sizeof(kKats[0]);

template <typename T>
bool bit_equal(const T& a, const T& b) {
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}

struct Moments {
    double mean = 0, var = 0, m4 = 0;
};

template <typename F>
Moments moments(std::size_t n, F&& sample) {
    Moments m;
    std::vector<double> xs(n);
    for (std::size_t i = 0; i < n; ++i) {
        xs[i] = sample(i);
        m.mean += xs[i];
    }
    m.mean /= double(n);
    for (double x : xs) {
        const double d = (x - m.mean) * (x - m.mean);
        m.var += d;
        m.m4 += d * d;
    }
    m.var /= double(n);
    m.m4 /= double(n);
    return m;
}

template <typename F, typename G>
double correlation(std::size_t n, F&& fa, G&& fb) {
    double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const double a = fa(i), b = fb(i);
        sa += a; sb += b; saa += a * a; sbb += b * b; sab += a * b;
    }
    const double dn = double(n);
    const double cov = sab / dn - (sa / dn) * (sb / dn);
    return cov / std::sqrt((saa / dn - (sa / dn) * (sa / dn)) * (sbb / dn - (sb / dn) * (sb / dn)));
}

constexpr std::size_t kSamples = std::size_t(1) << 20;
const double kSigmaMean = 1.0 / std::sqrt(double(kSamples));

}  // namespace

TEST(PhiloxTest, HostMatchesRandom123KnownAnswers) {
    for (int t = 0; t < kNumKats; ++t) {
        const auto got = philox::philox4x32_10(kKats[t].ctr, kKats[t].k0, kKats[t].k1);
        for (int w = 0; w < 4; ++w) {
            EXPECT_EQ(got.w[w], kKats[t].want.w[w]) << "KAT " << t << " word " << w;
        }
    }
}

TEST(PhiloxTest, DeviceMatchesRandom123KnownAnswers) {
    sycl::queue q;
    auto* out = sycl::malloc_shared<philox::Words>(kNumKats, q);
    auto* in = sycl::malloc_shared<Kat>(kNumKats, q);
    for (int t = 0; t < kNumKats; ++t) in[t] = kKats[t];
    q.parallel_for(sycl::range<1>(kNumKats), [=](sycl::id<1> i) {
         out[i] = philox::philox4x32_10(in[i].ctr, in[i].k0, in[i].k1);
     }).wait();
    for (int t = 0; t < kNumKats; ++t) {
        for (int w = 0; w < 4; ++w) {
            EXPECT_EQ(out[t].w[w], kKats[t].want.w[w]) << "KAT " << t << " word " << w;
        }
    }
    sycl::free(out, q);
    sycl::free(in, q);
}

TEST(PhiloxTest, DrawPutsSeedInTheKeyAndIndexStreamInTheCounter) {
    const std::uint64_t seed = 0x299f31d0a4093822ull;
    const std::uint64_t index = 0x85a308d3243f6a88ull;
    const std::uint64_t stream = 0x0370734413198a2eull;
    const auto got = philox::draw(seed, stream, index);
    for (int w = 0; w < 4; ++w) EXPECT_EQ(got.w[w], kKats[2].want.w[w]) << "word " << w;
}

template <typename R>
class PhiloxRealTest : public ::testing::Test {};
using RealTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(PhiloxRealTest, RealTypes);

TYPED_TEST(PhiloxRealTest, UniformRangesAndMoments) {
    using R = TypeParam;
    for (std::size_t i = 0; i < kSamples; ++i) {
        const R u = philox::uniform_unit<R>(3, 0, i);
        const R s = philox::uniform_symmetric<R>(3, 0, i);
        ASSERT_GE(u, R(0)); ASSERT_LT(u, R(1));
        ASSERT_GE(s, R(-1)); ASSERT_LT(s, R(1));
    }
    const auto mu = moments(kSamples, [](std::size_t i) { return double(philox::uniform_unit<R>(3, 0, i)); });
    EXPECT_NEAR(mu.mean, 0.5, 6 * kSigmaMean * std::sqrt(1.0 / 12));
    EXPECT_NEAR(mu.var, 1.0 / 12, 6 * kSigmaMean * std::sqrt(1.0 / 80 - 1.0 / 144));
    const auto ms = moments(kSamples, [](std::size_t i) { return double(philox::uniform_symmetric<R>(3, 1, i)); });
    EXPECT_NEAR(ms.mean, 0.0, 6 * kSigmaMean * std::sqrt(1.0 / 3));
    EXPECT_NEAR(ms.var, 1.0 / 3, 6 * kSigmaMean * std::sqrt(16.0 / 80 - 16.0 / 144));
}

TYPED_TEST(PhiloxRealTest, NormalMoments) {
    using R = TypeParam;
    const auto m = moments(kSamples, [](std::size_t i) { return double(philox::normal<R>(11, 2, i)); });
    EXPECT_NEAR(m.mean, 0.0, 6 * kSigmaMean);
    EXPECT_NEAR(m.var, 1.0, 6 * kSigmaMean * std::sqrt(2.0));
    EXPECT_NEAR(m.m4, 3.0, 6 * kSigmaMean * std::sqrt(96.0));
    const auto mc = moments(kSamples, [](std::size_t i) { return double(philox::normal<std::complex<R>>(11, 3, i).imag()); });
    EXPECT_NEAR(mc.mean, 0.0, 6 * kSigmaMean);
    EXPECT_NEAR(mc.var, 1.0, 6 * kSigmaMean * std::sqrt(2.0));
}

TYPED_TEST(PhiloxRealTest, IndexStreamSeedAndComponentsAreUncorrelated) {
    using R = TypeParam;
    using C = std::complex<R>;
    const double tol = 6 * kSigmaMean;
    auto u = [](std::uint64_t seed, std::uint64_t stream, std::size_t i) {
        return double(philox::uniform_symmetric<R>(seed, stream, i));
    };
    EXPECT_NEAR(correlation(kSamples, [&](std::size_t i) { return u(5, 0, i); },
                            [&](std::size_t i) { return u(5, 0, i + 1); }), 0.0, tol);
    EXPECT_NEAR(correlation(kSamples, [&](std::size_t i) { return u(5, 0, i); },
                            [&](std::size_t i) { return u(5, 1, i); }), 0.0, tol);
    EXPECT_NEAR(correlation(kSamples, [&](std::size_t i) { return u(5, 0, i); },
                            [&](std::size_t i) { return u(6, 0, i); }), 0.0, tol);
    EXPECT_NEAR(correlation(kSamples, [](std::size_t i) { return double(philox::uniform_symmetric<C>(5, 0, i).real()); },
                            [](std::size_t i) { return double(philox::uniform_symmetric<C>(5, 0, i).imag()); }),
                0.0, tol);
    EXPECT_NEAR(correlation(kSamples, [](std::size_t i) { return double(philox::normal<C>(5, 0, i).real()); },
                            [](std::size_t i) { return double(philox::normal<C>(5, 0, i).imag()); }),
                0.0, tol);
}

template <typename T>
class PhiloxScalarTest : public ::testing::Test {};
using ScalarTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(PhiloxScalarTest, ScalarTypes);

TYPED_TEST(PhiloxScalarTest, DeviceUniformIsBitIdenticalToHost) {
    using T = TypeParam;
    constexpr int n = 4096;
    const std::uint64_t seed = 0xfedcba9876543210ull;
    const std::uint64_t stream = 0x100000003ull;
    const std::uint64_t base = 0xfffff000ull;  // straddles the 2^32 index word boundary
    sycl::queue q;
    auto* sym = sycl::malloc_shared<T>(n, q);
    auto* uni = sycl::malloc_shared<T>(n, q);
    auto* nrm = sycl::malloc_shared<T>(n, q);
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
         sym[i] = philox::uniform_symmetric<T>(seed, stream, base + i);
         uni[i] = philox::uniform_unit<T>(seed, stream, base + i);
         nrm[i] = philox::normal<T>(seed, stream, base + i);
     }).wait();
    using R = typename philox::real_of<T>::type;
    const double ntol = std::is_same_v<R, float> ? 1e-5 : 1e-12;
    for (int i = 0; i < n; ++i) {
        ASSERT_TRUE(bit_equal(sym[i], philox::uniform_symmetric<T>(seed, stream, base + i))) << "index " << i;
        ASSERT_TRUE(bit_equal(uni[i], philox::uniform_unit<T>(seed, stream, base + i))) << "index " << i;
        const T h = philox::normal<T>(seed, stream, base + i);
        ASSERT_LE(std::abs(nrm[i] - h), ntol * (1 + std::abs(h))) << "index " << i;
    }
    sycl::free(sym, q);
    sycl::free(uni, q);
    sycl::free(nrm, q);
}

// The library fill paths are pinned to their key, so both SYCL implementations fill identically.
TYPED_TEST(PhiloxScalarTest, FillRandomIsKeyedOnSeedAndFlatIndex) {
    using T = TypeParam;
    const unsigned int seed = 77;
    auto m = Matrix<T, MatrixFormat::Dense>::Random(5, 3, false, 4, seed);
    ASSERT_EQ(m.data().size(), std::size_t(5 * 3 * 4));
    for (std::size_t idx = 0; idx < m.data().size(); ++idx) {
        ASSERT_TRUE(bit_equal(m.data()[idx], philox::uniform_symmetric<T>(seed, 0, idx))) << "flat index " << idx;
    }
}

TYPED_TEST(PhiloxScalarTest, FillTriangularRandomGivesEveryItemTheSameKeyedMatrix) {
    using T = TypeParam;
    const int n = 6, batch = 3;
    const unsigned int seed = 9;
    auto m = Matrix<T, MatrixFormat::Dense>::RandomTriangular(n, Uplo::Lower, Diag::NonUnit, batch, seed);
    for (int b = 0; b < batch; ++b) {
        for (int pos = 0; pos < n * n; ++pos) {
            const int i = pos % n, j = pos / n;
            const T want = (i == j || i < j) ? philox::uniform_symmetric<T>(seed, 0, pos) : T(0);
            ASSERT_TRUE(bit_equal(m.data()[b * n * n + i * n + j], want)) << "item " << b << " position " << pos;
        }
    }
}
