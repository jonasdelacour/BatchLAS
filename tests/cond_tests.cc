#include <gtest/gtest.h>
#include <blas/extra.hh>
#include <blas/matrix.hh>
#include <util/sycl-device-queue.hh>
#include <util/sycl-vector.hh>
#include <batchlas/backend_config.h>
#include "test_utils.hh"
#include <cmath>
#include <vector>

using namespace batchlas;

#if BATCHLAS_HAS_GPU_BACKEND

// Typed test fixture for condition number computations
template <typename T>
class CondTest : public ::testing::Test {
protected:
    void SetUp() override {
        ctx = std::make_shared<Queue>();
    }

    void TearDown() override {
        if (ctx) ctx->wait();
    }

    using real_t = typename base_type<T>::type;

    static constexpr real_t tolerance() {
        return test_utils::tolerance<T>();
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
                return max_v / min_v;
        }
        return real_t(0);
    }

    std::shared_ptr<Queue> ctx;
};

using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(CondTest, TestTypes);

TYPED_TEST(CondTest, IdentityMatrix) {
    using T = TypeParam;
    const int n = 4;
    const int batch_size = 2;

    auto mat = Matrix<T, MatrixFormat::Dense>::Identity(n, batch_size);

    for (auto nt : {NormType::One, NormType::Inf, NormType::Max}) {
        auto conds = cond<test_utils::gpu_backend>(*this->ctx, mat.view(), nt);
        this->ctx->wait();
        for (int b = 0; b < batch_size; ++b) {
            EXPECT_NEAR(conds[b], static_cast<typename CondTest<T>::real_t>(1), this->tolerance())
                << "Batch " << b;
        }
    }

    auto conds = cond<test_utils::gpu_backend>(*this->ctx, mat.view(), NormType::Frobenius);
    this->ctx->wait();
    for (int b = 0; b < batch_size; ++b) {
        EXPECT_NEAR(conds[b], n, this->tolerance()) // Frobenius norm of identity is sqrt(n) so ||I|| * ||I^-1|| = sqrt(n) * sqrt(n) = n
            << "Batch " << b;
    }
}

TYPED_TEST(CondTest, DiagonalMatrix) {
    using T = TypeParam;
    using real_t = typename CondTest<T>::real_t;
    const int n = 3;
    const int batch_size = 2;

    std::vector<real_t> diag = {real_t(1), real_t(2), real_t(3)};
    UnifiedVector<T> diag_vals(n);
    for (int i = 0; i < n; ++i) {
        diag_vals[i] = static_cast<T>(diag[i]);
    }

    auto mat = Matrix<T, MatrixFormat::Dense>::Diagonal(diag_vals.to_span(), batch_size);

    for (auto nt : {NormType::Frobenius, NormType::One, NormType::Inf, NormType::Max}) {
        real_t expected = CondTest<T>::expected_cond_diagonal(diag, nt);
        auto conds = cond<test_utils::gpu_backend>(*this->ctx, mat.view(), nt);
        this->ctx->wait();
        for (int b = 0; b < batch_size; ++b) {
            EXPECT_NEAR(conds[b], expected, this->tolerance())
                << "Batch " << b;
        }
    }
}

#endif // BATCHLAS_HAS_GPU_BACKEND

