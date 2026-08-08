#include <gtest/gtest.h>
#include <blas/extra.hh>
#include <blas/matrix.hh>
#include <util/sycl-device-queue.hh>
#include <util/sycl-vector.hh>
#include <batchlas/backend_config.h>
#include "test_utils.hh"
#include <cmath>
#include <tuple>
#include <vector>
#include <type_traits>

using namespace batchlas;

template <typename T, Backend B>
struct CondConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

template <template <typename, Backend> class Config>
struct backend_real_types {
    using tuple_type = decltype(std::tuple_cat(
#if BATCHLAS_HAS_HOST_BACKEND && BATCHLAS_HAS_CPU_TARGET
        std::tuple<Config<float, Backend::NETLIB>,
                   Config<double, Backend::NETLIB>>{},
#endif
#if BATCHLAS_HAS_CUDA_BACKEND
        std::tuple<Config<float, Backend::CUDA>,
                   Config<double, Backend::CUDA>>{},
#endif
#if BATCHLAS_HAS_ROCM_BACKEND
        std::tuple<Config<float, Backend::ROCM>,
                   Config<double, Backend::ROCM>>{},
#endif
#if BATCHLAS_HAS_MKL_BACKEND
        std::tuple<Config<float, Backend::MKL>,
                   Config<double, Backend::MKL>>{},
#endif
        std::tuple<>{}));

    using type = typename test_utils::tuple_to_types<tuple_type>::type;
};

using CondTestTypes = typename backend_real_types<CondConfig>::type;

// Typed test fixture for condition number computations
template <typename Config>
class CondTest : public test_utils::BatchLASTest<Config> {
protected:
    using ScalarType = typename Config::ScalarType;
    using real_t = typename base_type<ScalarType>::type;

    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        if (!this->ctx) {
            return;
        }
    }

    static constexpr real_t tolerance() {
        return test_utils::tolerance<ScalarType>();
    }

    // Compute expected condition number for a diagonal matrix
    static real_t expected_cond_diagonal(const std::vector<real_t>& diag, NormType nt) {
        real_t max_v = 0;
        real_t min_v = diag[0];
        real_t sum_sq = 0;
        real_t sum_inv_sq = 0;
        for (real_t v : diag) {
            real_t abs_v = std::abs(v);
            max_v = std::max(max_v, abs_v);
            min_v = std::min(min_v, abs_v);
            sum_sq += abs_v * abs_v;
            sum_inv_sq += real_t(1) / (abs_v * abs_v);
        }
        switch (nt) {
            case NormType::Frobenius:
                return std::sqrt(sum_sq * sum_inv_sq);
            case NormType::One:
            case NormType::Inf:
            case NormType::Max:
            case NormType::Spectral:
                return max_v / min_v;
        }
        return real_t(0);
    }

    static real_t expected_cond_frobenius_target(real_t log10_kappa) {
        return std::pow(real_t(10), log10_kappa);
    }

};

TYPED_TEST_SUITE(CondTest, CondTestTypes);

TYPED_TEST(CondTest, IdentityMatrix) {
    using T = typename TestFixture::ScalarType;
    using real_t = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 4;
    const int batch_size = 2;

    auto mat = Matrix<T, MatrixFormat::Dense>::Identity(n, batch_size);

    for (auto nt : {NormType::One, NormType::Inf, NormType::Max}) {
        auto conds = cond<B>(*this->ctx, mat.view(), nt);
        this->ctx->wait();
        for (int b = 0; b < batch_size; ++b) {
            test_utils::expect_near(conds[b], static_cast<T>(real_t(1)), this->tolerance());
        }
    }

    auto conds = cond<B>(*this->ctx, mat.view(), NormType::Frobenius);
    this->ctx->wait();
    for (int b = 0; b < batch_size; ++b) {
        test_utils::expect_near(conds[b], static_cast<T>(real_t(n)), this->tolerance());
    }
}

TYPED_TEST(CondTest, DiagonalMatrix) {
    using T = typename TestFixture::ScalarType;
    using real_t = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 3;
    const int batch_size = 2;

    std::vector<real_t> diag = {real_t(1), real_t(2), real_t(3)};
    UnifiedVector<T> diag_vals(n);
    for (int i = 0; i < n; ++i) {
        diag_vals[i] = static_cast<T>(diag[i]);
    }

    auto mat = Matrix<T, MatrixFormat::Dense>::Diagonal(diag_vals.to_span(), batch_size);

    for (auto nt : {NormType::Frobenius, NormType::One, NormType::Inf, NormType::Max}) {
        real_t expected = this->expected_cond_diagonal(diag, nt);
        auto conds = cond<B>(*this->ctx, mat.view(), nt);
        this->ctx->wait();
        for (int b = 0; b < batch_size; ++b) {
            test_utils::expect_near(conds[b], static_cast<T>(expected), this->tolerance());
        }
    }
}

TYPED_TEST(CondTest, RandomMatrixLogCondFrobenius) {
    using T = typename TestFixture::ScalarType;
    using real_t = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 32;
    const int batch_size = 4;
    const real_t log10_cond = real_t(5);

    auto mat = random_with_log10_cond_metric<B, T>(*this->ctx, n, log10_cond, NormType::Frobenius, batch_size, /*seed=*/123);

    // Expected Frobenius condition number from the requested conditioning metric.
    const real_t expected = this->expected_cond_frobenius_target(log10_cond);

    auto conds = cond<B>(*this->ctx, mat.view(), NormType::Frobenius);
    this->ctx->wait();

    const real_t rel_tol = std::is_same_v<real_t, float> ? real_t(2e-4f) : real_t(1e-8);
    const real_t tol = expected * rel_tol;
    for (int b = 0; b < batch_size; ++b) {
        test_utils::expect_near(conds[b], static_cast<T>(expected), tol);
    }
}

TYPED_TEST(CondTest, RandomHermitianLogCondFrobenius) {
    using T = typename TestFixture::ScalarType;
    using real_t = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 32;
    const int batch_size = 4;
    const real_t log10_cond = real_t(4);

    auto mat = random_hermitian_with_log10_cond_metric<B, T>(*this->ctx, n, log10_cond, NormType::Frobenius, batch_size, /*seed=*/321);
    const real_t expected = this->expected_cond_frobenius_target(log10_cond);

    auto conds = cond<B>(*this->ctx, mat.view(), NormType::Frobenius);
    this->ctx->wait();

    const real_t rel_tol = std::is_same_v<real_t, float> ? real_t(2e-4f) : real_t(1e-8);
    const real_t tol = expected * rel_tol;
    for (int b = 0; b < batch_size; ++b) {
        test_utils::expect_near(conds[b], static_cast<T>(expected), tol);
    }
}

TYPED_TEST(CondTest, RandomHermitianLogCondSpectral) {
    using T = typename TestFixture::ScalarType;
    using real_t = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 32;
    const int batch_size = 4;
    const real_t log10_cond = real_t(4);

    auto mat = random_hermitian_with_log10_cond_metric<B, T>(*this->ctx, n, log10_cond, NormType::Spectral, batch_size, /*seed=*/456);

    auto conds = cond<B>(*this->ctx, mat.view(), NormType::Spectral);
    this->ctx->wait();

    const real_t tol_log10 = std::is_same_v<real_t, float> ? real_t(1.0f) : real_t(2.0);
    for (int b = 0; b < batch_size; ++b) {
        const real_t val = static_cast<real_t>(conds[b]);
        ASSERT_GT(val, real_t(0));
        EXPECT_NEAR(std::log10(val), log10_cond, tol_log10);
    }
}

TYPED_TEST(CondTest, RandomBandedLogCondFrobenius) {
    using T = typename TestFixture::ScalarType;
    using real_t = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 32;
    const int kd = 6;
    const int batch_size = 3;
    const real_t log10_cond = real_t(3);

    auto mat = random_banded_with_log10_cond_metric<B, T>(*this->ctx, n, kd, log10_cond, NormType::Frobenius, batch_size, /*seed=*/7);
    const real_t expected = this->expected_cond_frobenius_target(log10_cond);

    auto conds = cond<B>(*this->ctx, mat.view(), NormType::Frobenius);
    this->ctx->wait();

    const real_t rel_tol = std::is_same_v<real_t, float> ? real_t(2e-4f) : real_t(1e-8);
    const real_t tol = expected * rel_tol;
    for (int b = 0; b < batch_size; ++b) {
        test_utils::expect_near(conds[b], static_cast<T>(expected), tol);
    }

    const real_t ztol = std::is_same_v<real_t, float> ? real_t(5e-5f) : real_t(1e-12);
    for (int b = 0; b < batch_size; ++b) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                if (std::abs(i - j) > kd) {
                    test_utils::expect_near(mat(i, j, b), T(0), ztol);
                }
            }
        }
    }
}

TYPED_TEST(CondTest, RandomHermitianBandedLogCondFrobenius) {
    using T = typename TestFixture::ScalarType;
    using real_t = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 32;
    const int kd = 6;
    const int batch_size = 3;
    const real_t log10_cond = real_t(3);

    auto mat = random_hermitian_banded_with_log10_cond_metric<B, T>(*this->ctx, n, kd, log10_cond, NormType::Frobenius, batch_size, /*seed=*/11);
    const real_t expected = this->expected_cond_frobenius_target(log10_cond);

    auto conds = cond<B>(*this->ctx, mat.view(), NormType::Frobenius);
    this->ctx->wait();

    const real_t rel_tol = std::is_same_v<real_t, float> ? real_t(2e-4f) : real_t(1e-8);
    const real_t tol = expected * rel_tol;
    for (int b = 0; b < batch_size; ++b) {
        test_utils::expect_near(conds[b], static_cast<T>(expected), tol);
    }

    const real_t ztol = std::is_same_v<real_t, float> ? real_t(5e-5f) : real_t(1e-12);
    for (int b = 0; b < batch_size; ++b) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                if (std::abs(i - j) > kd) {
                    test_utils::expect_near(mat(i, j, b), T(0), ztol);
                } else {
                    test_utils::expect_near(mat(i, j, b), mat(j, i, b), ztol);
                }
            }
        }
    }
}

TYPED_TEST(CondTest, RandomTridiagonalLogCondFrobenius) {
    using T = typename TestFixture::ScalarType;
    using real_t = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 32;
    const int batch_size = 4;
    const real_t log10_cond = real_t(4);

    auto mat = random_tridiagonal_with_log10_cond_metric<B, T>(*this->ctx, n, log10_cond, NormType::Frobenius, batch_size, /*seed=*/9);
    const real_t expected = this->expected_cond_frobenius_target(log10_cond);

    auto conds = cond<B>(*this->ctx, mat.view(), NormType::Frobenius);
    this->ctx->wait();

    const real_t rel_tol = std::is_same_v<real_t, float> ? real_t(2e-4f) : real_t(1e-8);
    const real_t tol = expected * rel_tol;
    for (int b = 0; b < batch_size; ++b) {
        test_utils::expect_near(conds[b], static_cast<T>(expected), tol);
    }

    const real_t ztol = std::is_same_v<real_t, float> ? real_t(1e-5f) : real_t(1e-12);
    for (int b = 0; b < batch_size; ++b) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                if (std::abs(i - j) > 1) {
                    test_utils::expect_near(mat(i, j, b), T(0), ztol);
                }
            }
        }
    }
}

TYPED_TEST(CondTest, RandomHermitianTridiagonalLogCondFrobenius) {
    using T = typename TestFixture::ScalarType;
    using real_t = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 32;
    const int batch_size = 4;
    const real_t log10_cond = real_t(4);

    auto mat = random_hermitian_tridiagonal_with_log10_cond_metric<B, T>(*this->ctx, n, log10_cond, NormType::Frobenius, batch_size, /*seed=*/13);
    const real_t expected = this->expected_cond_frobenius_target(log10_cond);

    auto conds = cond<B>(*this->ctx, mat.view(), NormType::Frobenius);
    this->ctx->wait();

    const real_t rel_tol = std::is_same_v<real_t, float> ? real_t(2e-4f) : real_t(1e-8);
    const real_t tol = expected * rel_tol;
    for (int b = 0; b < batch_size; ++b) {
        test_utils::expect_near(conds[b], static_cast<T>(expected), tol);
    }

    const real_t ztol = std::is_same_v<real_t, float> ? real_t(1e-5f) : real_t(1e-12);
    for (int b = 0; b < batch_size; ++b) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                if (std::abs(i - j) > 1) {
                    test_utils::expect_near(mat(i, j, b), T(0), ztol);
                } else {
                    test_utils::expect_near(mat(i, j, b), mat(j, i, b), ztol);
                }
            }
        }
    }
}

TYPED_TEST(CondTest, RandomHermitianTridiagonalLogCondSpectral) {
    using T = typename TestFixture::ScalarType;
    using real_t = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 32;
    const int batch_size = 4;
    const real_t log10_cond = real_t(8);

    auto mat = random_hermitian_tridiagonal_with_log10_cond_metric<B, T>(*this->ctx, n, log10_cond, NormType::Spectral, batch_size, /*seed=*/17);

    auto conds = cond<B>(*this->ctx, mat.view(), NormType::Spectral);
    this->ctx->wait();

    const real_t tol_log10 = real_t(0.5);
    for (int b = 0; b < batch_size; ++b) {
        const real_t val = static_cast<real_t>(conds[b]);
        ASSERT_GT(val, real_t(0));
        EXPECT_NEAR(std::log10(val), log10_cond, tol_log10);
    }
}

// The generator must never emit a non-finite matrix.
//
// It used to, intermittently: it orthonormalised two raw Matrix::Random factors
// with OrthoAlgorithm::Chol2, and Chol-QR forms the Gram matrix, squaring the
// condition number of its input. That input is UNconditioned -- the requested
// kappa is imposed afterwards by the diagonal -- so in float the squared Gram
// goes numerically indefinite whenever an ordinary random draw lands above
// kappa ~ 1e4, which happens a few times in a batch of 32 at n=64. potrf then
// fails, its info code is discarded, the following trsm back-substitutes
// through a garbage diagonal, and two gemms smear the NaN over every entry of
// that batch item.
//
// benchmarks/gesvd_relacc.cc works around it by re-drawing up to 8 times, so
// this has to test the generator directly; going through the benchmark would
// hide exactly the failure being guarded. The default is now Householder,
// which is unconditionally backward stable.
//
// Seeds matter: at n=64/batch=32 the reported reproduction failed on seed 1 and
// not on 123 or 2024, so a single seed proves nothing. Float only in practice
// (double needs kappa > 6.7e7), but both are swept here since the fixture is
// typed anyway.
TYPED_TEST(CondTest, RandomMatrixGeneratorIsAlwaysFinite) {
    using T = typename TestFixture::ScalarType;
    using real_t = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;

    const int n = 64;
    const int batch_size = 32;
    const real_t log10_cond = real_t(4);

    for (unsigned seed : {1u, 7u, 42u, 123u, 2024u}) {
        auto mat = random_with_log10_cond_metric<B, T>(
            *this->ctx, n, log10_cond, NormType::Spectral, batch_size, seed);
        this->ctx->wait();

        const auto view = mat.view();
        const T* base = view.data_ptr();
        const int64_t stride = view.stride();
        const int64_t ld = view.ld();

        int non_finite_items = 0;
        for (int b = 0; b < batch_size; ++b) {
            bool bad = false;
            for (int c = 0; c < n && !bad; ++c) {
                for (int r = 0; r < n; ++r) {
                    const T v = base[b * stride + static_cast<int64_t>(c) * ld + r];
                    if (!std::isfinite(static_cast<double>(std::abs(v)))) {
                        bad = true;
                        break;
                    }
                }
            }
            if (bad) ++non_finite_items;
        }

        EXPECT_EQ(non_finite_items, 0)
            << "seed " << seed << ": " << non_finite_items << " of " << batch_size
            << " batch items are non-finite";
    }
}
