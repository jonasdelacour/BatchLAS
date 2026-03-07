#include <gtest/gtest.h>

#include <batchlas/backend_config.h>
#include <blas/linalg.hh>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <initializer_list>
#include <numeric>
#include <string>
#include <vector>

#include "test_utils.hh"

using namespace batchlas;

namespace {

template <typename T>
UnifiedVector<T> make_unified_vector(std::initializer_list<T> values) {
    UnifiedVector<T> out;
    out.reserve(values.size());
    for (const auto &value : values) {
        out.push_back(value);
    }
    return out;
}

struct ConvergenceRun {
        std::string label;
        int iluk_level = -1;
        UnifiedVector<float> W;
        UnifiedVector<float> best_hist;
        UnifiedVector<float> current_hist;
        UnifiedVector<float> rate_hist;
        UnifiedVector<float> ritz_hist;
        UnifiedVector<int32_t> iters_done;

        ConvergenceRun(std::string lbl, int k, size_t history_size, size_t w_size, size_t batch)
                : label(std::move(lbl)),
                    iluk_level(k),
                    W(w_size, 0.0f),
                    best_hist(history_size, std::nanf("")),
                    current_hist(history_size, std::nanf("")),
                    rate_hist(history_size, std::nanf("")),
                    ritz_hist(history_size, std::nanf("")),
                    iters_done(batch, 0) {}
};

struct SweepCase {
        std::string label;
        float density = 0.0f;
        float diagonal_boost = 0.0f;
        unsigned seed = 0;
};

template <typename T>
Matrix<T, MatrixFormat::Dense> make_identity_matrix(int n, int batch = 1) {
    Matrix<T, MatrixFormat::Dense> eye(n, n, batch);
    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                eye(i, j, b) = (i == j) ? T(1) : T(0);
            }
        }
    }
    return eye;
}

template <typename T>
double inverse_residual_frobenius(const MatrixView<T, MatrixFormat::CSR>& A,
                                  const MatrixView<T, MatrixFormat::Dense>& approx_inverse) {
    const int n = A.rows();
    const int batch = A.batch_size();
    const auto row_offsets = A.row_offsets();
    const auto col_indices = A.col_indices();
    const auto values = A.data();

    double sum_sq = 0.0;
    for (int b = 0; b < batch; ++b) {
        const int ro_base = b * A.offset_stride();
        const int val_base = b * A.matrix_stride();
        for (int col = 0; col < n; ++col) {
            for (int row = 0; row < n; ++row) {
                T product = T(0);
                for (int p = row_offsets[ro_base + row]; p < row_offsets[ro_base + row + 1]; ++p) {
                    const int k = col_indices[val_base + p];
                    product += values[val_base + p] * approx_inverse(k, col, b);
                }
                const T target = (row == col) ? T(1) : T(0);
                const double error = static_cast<double>(std::abs(product - target));
                sum_sq += error * error;
            }
        }
    }
    return std::sqrt(sum_sq);
}

template <typename T>
T csr_entry(const MatrixView<T, MatrixFormat::CSR>& A, int row, int col, int batch = 0) {
    const auto row_offsets = A.row_offsets();
    const auto col_indices = A.col_indices();
    const auto values = A.data();
    const int ro_base = batch * A.offset_stride();
    const int val_base = batch * A.matrix_stride();
    for (int p = row_offsets[ro_base + row]; p < row_offsets[ro_base + row + 1]; ++p) {
        if (col_indices[val_base + p] == col) {
            return values[val_base + p];
        }
    }
    return T(0);
}

}  // namespace

#if BATCHLAS_HAS_GPU_BACKEND

class ILUKTests : public ::testing::Test {
protected:
    void SetUp() override {
        ctx = std::make_shared<Queue>(Device::default_device());
    }

    std::shared_ptr<Queue> ctx;
};

TEST_F(ILUKTests, FactorApplyDiagonalBatchMultiRHS) {
    constexpr int n = 4;
    constexpr int batch = 2;
    constexpr int nrhs = 2;
    constexpr int nnz = n;

    UnifiedVector<int> row_offsets((n + 1) * batch);
    UnifiedVector<int> col_indices(nnz * batch);
    UnifiedVector<float> values(nnz * batch);

    for (int b = 0; b < batch; ++b) {
        const int ro_base = b * (n + 1);
        const int val_base = b * nnz;
        row_offsets[ro_base + 0] = 0;
        row_offsets[ro_base + 1] = 1;
        row_offsets[ro_base + 2] = 2;
        row_offsets[ro_base + 3] = 3;
        row_offsets[ro_base + 4] = 4;
        for (int i = 0; i < n; ++i) {
            col_indices[val_base + i] = i;
            values[val_base + i] = (b == 0 ? 2.0f : 4.0f) + i;
        }
    }

    MatrixView<float, MatrixFormat::CSR> A(values.data(), row_offsets.data(), col_indices.data(),
                                            nnz, n, n, nnz, n + 1, batch);

    Matrix<float, MatrixFormat::Dense> rhs(n, nrhs, batch);
    Matrix<float, MatrixFormat::Dense> out(n, nrhs, batch);

    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < nrhs; ++c) {
            for (int i = 0; i < n; ++i) {
                rhs(i, c, b) = 1.0f + static_cast<float>(b + c + i);
            }
        }
    }

    for (int k : {0, 1, 2}) {
        ILUKParams<float> params;
        params.levels_of_fill = k;
        auto M = iluk_factorize<test_utils::gpu_backend>(*ctx, A, params);

        iluk_apply<test_utils::gpu_backend>(*ctx, M, rhs.view(), out.view());
        ctx->wait_and_throw();

        for (int b = 0; b < batch; ++b) {
            for (int c = 0; c < nrhs; ++c) {
                for (int i = 0; i < n; ++i) {
                    const float denom = (b == 0 ? 2.0f : 4.0f) + i;
                    const float expected = rhs(i, c, b) / denom;
                    EXPECT_NEAR(out(i, c, b), expected, 1e-5f)
                        << "Mismatch at k=" << k << " b=" << b << " c=" << c << " i=" << i;
                }
            }
        }
    }
}

TEST_F(ILUKTests, SymbolicPatternUsesMaxLevelRecurrence) {
    constexpr int n = 5;
    constexpr int batch = 1;
    constexpr int nnz = 9;

    UnifiedVector<int> row_offsets = make_unified_vector<int>({0, 2, 4, 6, 8, 9});
    UnifiedVector<int> col_indices = make_unified_vector<int>({0, 2,
                                                               1, 4,
                                                               1, 2,
                                                               0, 3,
                                                               4});
    UnifiedVector<float> values(nnz, 1.0f);

    MatrixView<float, MatrixFormat::CSR> A(values.data(), row_offsets.data(), col_indices.data(),
                                            nnz, n, n, nnz, n + 1, batch);

    ILUKParams<float> params;
    params.levels_of_fill = 2;
    auto M = iluk_factorize<test_utils::gpu_backend>(*ctx, A, params);

    const auto lu = M.lu.view();
    const auto ro = lu.row_offsets();
    const auto ci = lu.col_indices();

    bool found_fill = false;
    const int row = 3;
    for (int p = ro[row]; p < ro[row + 1]; ++p) {
        if (ci[p] == 4) {
            found_fill = true;
            break;
        }
    }

    EXPECT_TRUE(found_fill)
        << "Expected row 3 to retain column 4 at ILU(2) under the textbook max-level recurrence";
}

TEST_F(ILUKTests, HeterogeneousBatchSparsityThrows) {
    constexpr int n = 3;
    constexpr int batch = 2;
    constexpr int nnz = 4;

    UnifiedVector<int> row_offsets = make_unified_vector<int>({0, 2, 3, 4,
                                                               0, 1, 3, 4});
    UnifiedVector<int> col_indices = make_unified_vector<int>({0, 1, 1, 2,
                                                               0, 1, 2, 2});
    UnifiedVector<float> values(nnz * batch, 1.0f);

    MatrixView<float, MatrixFormat::CSR> A(values.data(), row_offsets.data(), col_indices.data(),
                                            nnz, n, n, nnz, n + 1, batch);

    EXPECT_THROW((iluk_factorize<test_utils::gpu_backend>(*ctx, A, ILUKParams<float>{})), std::invalid_argument);
}

TEST_F(ILUKTests, ZeroShiftZeroPivotFactorizationThrows) {
    constexpr int n = 2;
    constexpr int batch = 1;
    constexpr int nnz = 4;

    UnifiedVector<int> row_offsets = make_unified_vector<int>({0, 2, 4});
    UnifiedVector<int> col_indices = make_unified_vector<int>({0, 1,
                                                               0, 1});
    UnifiedVector<float> values = make_unified_vector<float>({0.0f, 1.0f,
                                                              1.0f, 1.0f});

    MatrixView<float, MatrixFormat::CSR> A(values.data(), row_offsets.data(), col_indices.data(),
                                            nnz, n, n, nnz, n + 1, batch);

    ILUKParams<float> params;
    params.levels_of_fill = 0;
    params.diagonal_shift = 0.0f;
    params.diag_pivot_threshold = 0.0f;
    EXPECT_THROW((iluk_factorize<test_utils::gpu_backend>(*ctx, A, params)), std::runtime_error);
}

TEST_F(ILUKTests, ZeroShiftZeroDiagonalApplyThrows) {
    constexpr int n = 2;
    constexpr int batch = 1;
    constexpr int nnz = 2;

    ILUKPreconditioner<float> M;
    M.lu = Matrix<float, MatrixFormat::CSR>(n, n, nnz, batch);
    M.diag_positions = make_unified_vector<int>({0, 1});
    M.n = n;
    M.batch_size = batch;
    M.levels_of_fill = 0;
    M.diagonal_shift = 0.0f;

    auto lu = M.lu.view();
    auto ro = lu.row_offsets();
    auto ci = lu.col_indices();
    auto vals = lu.data();
    ro[0] = 0;
    ro[1] = 1;
    ro[2] = 2;
    ci[0] = 0;
    ci[1] = 1;
    vals[0] = 0.0f;
    vals[1] = 2.0f;

    Matrix<float, MatrixFormat::Dense> rhs(n, 1, batch);
    Matrix<float, MatrixFormat::Dense> out(n, 1, batch);
    rhs(0, 0, 0) = 1.0f;
    rhs(1, 0, 0) = 2.0f;

    EXPECT_THROW((iluk_apply<test_utils::gpu_backend>(*ctx, M, rhs.view(), out.view())), std::runtime_error);
}

TEST_F(ILUKTests, DropToleranceRemovesTinyOffDiagonalEntries) {
    constexpr int n = 3;
    constexpr int batch = 1;
    constexpr int nnz = 4;

    UnifiedVector<int> row_offsets = make_unified_vector<int>({0, 2, 3, 4});
    UnifiedVector<int> col_indices = make_unified_vector<int>({0, 1,
                                                               1,
                                                               2});
    UnifiedVector<float> values = make_unified_vector<float>({4.0f, 1e-6f,
                                                              5.0f,
                                                              6.0f});

    MatrixView<float, MatrixFormat::CSR> A(values.data(), row_offsets.data(), col_indices.data(),
                                           nnz, n, n, nnz, n + 1, batch);

    ILUKParams<float> params;
    params.levels_of_fill = 0;
    params.drop_tolerance = 1e-3f;
    auto M = iluk_factorize<test_utils::gpu_backend>(*ctx, A, params);

    const auto lu = M.lu.view();
    const auto ro = lu.row_offsets();
    const auto ci = lu.col_indices();
    EXPECT_EQ(ro[1] - ro[0], 1);
    EXPECT_EQ(ci[ro[0]], 0);
}

TEST_F(ILUKTests, FillFactorCapsAdditionalFill) {
    constexpr int n = 5;
    constexpr int batch = 1;
    constexpr int nnz = 9;

    UnifiedVector<int> row_offsets = make_unified_vector<int>({0, 2, 4, 6, 8, 9});
    UnifiedVector<int> col_indices = make_unified_vector<int>({0, 2,
                                                               1, 4,
                                                               1, 2,
                                                               0, 3,
                                                               4});
    UnifiedVector<float> values(nnz, 1.0f);

    MatrixView<float, MatrixFormat::CSR> A(values.data(), row_offsets.data(), col_indices.data(),
                                           nnz, n, n, nnz, n + 1, batch);

    ILUKParams<float> params;
    params.levels_of_fill = 2;
    params.drop_tolerance = 0.0f;
    params.fill_factor = 1.0f;
    auto M = iluk_factorize<test_utils::gpu_backend>(*ctx, A, params);

    EXPECT_LE(M.lu.nnz(), nnz);

    const auto lu = M.lu.view();
    const auto ro = lu.row_offsets();
    const auto ci = lu.col_indices();
    bool found_fill = false;
    for (int p = ro[3]; p < ro[4]; ++p) {
        if (ci[p] == 4) {
            found_fill = true;
            break;
        }
    }
    EXPECT_FALSE(found_fill);
}

TEST_F(ILUKTests, PivotThresholdStabilizesTinyDiagonal) {
    constexpr int n = 2;
    constexpr int batch = 1;
    constexpr int nnz = 2;

    UnifiedVector<int> row_offsets = make_unified_vector<int>({0, 1, 2});
    UnifiedVector<int> col_indices = make_unified_vector<int>({0, 1});
    UnifiedVector<float> values = make_unified_vector<float>({1e-8f, 2.0f});

    MatrixView<float, MatrixFormat::CSR> A(values.data(), row_offsets.data(), col_indices.data(),
                                           nnz, n, n, nnz, n + 1, batch);

    ILUKParams<float> params;
    params.levels_of_fill = 0;
    params.diagonal_shift = 0.0f;
    params.diag_pivot_threshold = 0.25f;
    auto M = iluk_factorize<test_utils::gpu_backend>(*ctx, A, params);

    const auto lu = M.lu.view();
    EXPECT_GE(lu.data()[M.diag_positions[0]], 0.25f);
}

TEST_F(ILUKTests, ModifiedIluAccumulatesDroppedMassOnDiagonal) {
    constexpr int n = 2;
    constexpr int batch = 1;
    constexpr int nnz = 3;

    UnifiedVector<int> row_offsets = make_unified_vector<int>({0, 2, 3});
    UnifiedVector<int> col_indices = make_unified_vector<int>({0, 1,
                                                               1});
    UnifiedVector<float> values = make_unified_vector<float>({4.0f, 1e-3f,
                                                              5.0f});

    MatrixView<float, MatrixFormat::CSR> A(values.data(), row_offsets.data(), col_indices.data(),
                                           nnz, n, n, nnz, n + 1, batch);

    ILUKParams<float> params;
    params.levels_of_fill = 0;
    params.drop_tolerance = 1e-2f;
    params.modified_ilu = true;
    auto M = iluk_factorize<test_utils::gpu_backend>(*ctx, A, params);

    const auto lu = M.lu.view();
    EXPECT_NEAR(lu.data()[M.diag_positions[0]], 4.001f, 1e-6f);
}

TEST_F(ILUKTests, InverseResidualIsNearZeroForExactTridiagonalFactorization) {
    constexpr int n = 4;
    constexpr int batch = 1;
    constexpr int nnz = 10;

    UnifiedVector<int> row_offsets = make_unified_vector<int>({0, 2, 5, 8, 10});
    UnifiedVector<int> col_indices = make_unified_vector<int>({0, 1,
                                                               0, 1, 2,
                                                               1, 2, 3,
                                                               2, 3});
    UnifiedVector<float> values = make_unified_vector<float>({4.0f, -1.0f,
                                                              -1.0f, 4.0f, -1.0f,
                                                              -1.0f, 4.0f, -1.0f,
                                                              -1.0f, 4.0f});

    MatrixView<float, MatrixFormat::CSR> A(values.data(), row_offsets.data(), col_indices.data(),
                                           nnz, n, n, nnz, n + 1, batch);

    ILUKParams<float> params;
    params.levels_of_fill = 0;
    params.drop_tolerance = 0.0f;
    params.fill_factor = 4.0f;
    params.diag_pivot_threshold = 0.0f;
    params.modified_ilu = false;

    auto M = iluk_factorize<test_utils::gpu_backend>(*ctx, A, params);
    auto identity = make_identity_matrix<float>(n, batch);
    Matrix<float, MatrixFormat::Dense> approx_inverse(n, n, batch);

    iluk_apply<test_utils::gpu_backend>(*ctx, M, identity.view(), approx_inverse.view());
    ctx->wait_and_throw();

    const double residual = inverse_residual_frobenius(A, approx_inverse.view());
    EXPECT_LT(residual, 1e-4) << "Expected ILU(0) to reproduce the exact inverse on a tridiagonal system";
}

TEST_F(ILUKTests, HigherFillImprovesApproximateInverseResidualOnLargeSparseMatrix) {
    constexpr int n = 256;
    constexpr int batch = 1;
    constexpr float density = 0.05f;
    constexpr float diagonal_boost = 4.0f;
    constexpr unsigned seed = 20250306u;

    auto A_storage = csr_generators::random_sparse_hermitian_csr<float>(
        n,
        density,
        batch,
        seed,
        diagonal_boost,
        true);
    auto A = A_storage.view();

    auto identity = make_identity_matrix<float>(n, batch);
    Matrix<float, MatrixFormat::Dense> approx_inverse_k0(n, n, batch);
    Matrix<float, MatrixFormat::Dense> approx_inverse_k4(n, n, batch);

    ILUKParams<float> params_k0;
    params_k0.levels_of_fill = 0;
    params_k0.drop_tolerance = 0.0f;
    params_k0.fill_factor = 16.0f;
    params_k0.diag_pivot_threshold = 0.0f;
    params_k0.modified_ilu = false;

    ILUKParams<float> params_k4 = params_k0;
    params_k4.levels_of_fill = 4;

    auto M0 = iluk_factorize<test_utils::gpu_backend>(*ctx, A, params_k0);
    auto M4 = iluk_factorize<test_utils::gpu_backend>(*ctx, A, params_k4);

    iluk_apply<test_utils::gpu_backend>(*ctx, M0, identity.view(), approx_inverse_k0.view());
    iluk_apply<test_utils::gpu_backend>(*ctx, M4, identity.view(), approx_inverse_k4.view());
    ctx->wait_and_throw();

    const double residual_k0 = inverse_residual_frobenius(A, approx_inverse_k0.view());
    const double residual_k4 = inverse_residual_frobenius(A, approx_inverse_k4.view());
    const double normalized_residual_k4 = residual_k4 / std::sqrt(static_cast<double>(n));

    EXPECT_LT(residual_k4, residual_k0)
        << "Higher fill should reduce the inverse residual on a realistic sparse matrix";
    EXPECT_LT(normalized_residual_k4, 1.0)
        << "The approximate inverse residual should be materially below O(sqrt(n)) on the large sparse test case";
}

TEST_F(ILUKTests, RandomSparseHermitianCSRProducesSortedHermitianRows) {
    constexpr int n = 48;
    constexpr int batch = 2;
    constexpr float density = 0.12f;
    constexpr float diagonal_boost = 3.0f;
    constexpr unsigned seed = 20260307u;

    auto csr = Matrix<std::complex<float>, MatrixFormat::CSR>::RandomSparseHermitian(
        n,
        density,
        batch,
        seed,
        diagonal_boost,
        false);
    auto view = csr.view();

    const auto row_offsets = view.row_offsets();
    const auto col_indices = view.col_indices();
    const auto values = view.data();

    for (int b = 0; b < batch; ++b) {
        const int ro_base = b * view.offset_stride();
        const int val_base = b * view.matrix_stride();
        for (int row = 0; row < n; ++row) {
            const int rs = row_offsets[ro_base + row];
            const int re = row_offsets[ro_base + row + 1];
            ASSERT_GT(re, rs);

            bool found_diag = false;
            float offdiag_abs_sum = 0.0f;
            for (int p = rs; p < re; ++p) {
                if (p > rs) {
                    EXPECT_LT(col_indices[val_base + p - 1], col_indices[val_base + p]);
                }
                const int col = col_indices[val_base + p];
                const auto value = values[val_base + p];
                if (col == row) {
                    found_diag = true;
                    EXPECT_FLOAT_EQ(value.imag(), 0.0f);
                } else {
                    offdiag_abs_sum += std::abs(value);
                    EXPECT_NEAR(std::abs(value - std::conj(csr_entry(view, col, row, b))), 0.0f, 1e-5f);
                }
            }

            ASSERT_TRUE(found_diag);
            const auto diag = csr_entry(view, row, row, b);
            EXPECT_GT(diag.real(), offdiag_abs_sum);
        }
    }
}

TEST_F(ILUKTests, SyevxInstrumentationAndPreconditioner) {
    constexpr int n = 256;
    constexpr int batch = 1;
    constexpr int neigs = 5;
    constexpr int extra_dirs = 5;
    constexpr int max_iters = 30;

    const std::vector<SweepCase> sweep_cases = {
        {"d0.02_b0.5_s1234", 0.02f, 0.5f, 1234u},
        {"d0.02_b0.5_s4321", 0.02f, 0.5f, 4321u},
        {"d0.02_b4.0_s1234", 0.02f, 4.0f, 1234u},
        {"d0.02_b4.0_s4321", 0.02f, 4.0f, 4321u},
        {"d0.06_b0.5_s1234", 0.06f, 0.5f, 1234u},
        {"d0.06_b0.5_s4321", 0.06f, 0.5f, 4321u},
        {"d0.06_b4.0_s1234", 0.06f, 4.0f, 1234u},
        {"d0.06_b4.0_s4321", 0.06f, 4.0f, 4321u},
    };

    const size_t history_size = static_cast<size_t>(max_iters) * static_cast<size_t>(batch) * static_cast<size_t>(neigs);
    const size_t w_size = static_cast<size_t>(neigs) * static_cast<size_t>(batch);

    std::vector<ConvergenceRun> runs;
    runs.emplace_back("baseline", -1, history_size, w_size, batch);
    runs.emplace_back("iluk_k2", 2, history_size, w_size, batch);
    runs.emplace_back("iluk_k3", 3, history_size, w_size, batch);
    runs.emplace_back("iluk_k4", 4, history_size, w_size, batch);
    runs.emplace_back("iluk_k5", 5, history_size, w_size, batch);

    auto run_case = [&](ConvergenceRun& run, const MatrixView<float, MatrixFormat::CSR>& csr_view) {
        std::fill(run.best_hist.begin(), run.best_hist.end(), std::nanf(""));
        std::fill(run.current_hist.begin(), run.current_hist.end(), std::nanf(""));
        std::fill(run.rate_hist.begin(), run.rate_hist.end(), std::nanf(""));
        std::fill(run.ritz_hist.begin(), run.ritz_hist.end(), std::nanf(""));
        std::fill(run.iters_done.begin(), run.iters_done.end(), 0);
        std::fill(run.W.begin(), run.W.end(), 0.0f);

        std::optional<ILUKPreconditioner<float>> precond;
        const ILUKPreconditioner<float>* precond_ptr = nullptr;
        if (run.iluk_level >= 0) {
            ILUKParams<float> iluk_params;
            iluk_params.levels_of_fill = run.iluk_level;
            precond = iluk_factorize<test_utils::gpu_backend>(*ctx, csr_view, iluk_params);
            precond_ptr = &(*precond);
        }

        SyevxInstrumentation<float> instr;
        instr.best_residual_history = run.best_hist;
        instr.current_residual_history = run.current_hist;
        instr.convergence_rate_history = run.rate_hist;
        instr.ritz_value_history = run.ritz_hist;
        instr.max_iterations = max_iters;
        instr.store_every = 1;
        instr.store_current_residual = true;
        instr.store_convergence_rate = true;
        instr.store_ritz_values = true;
        instr.iterations_done = run.iters_done.data();

        SyevxParams<float> params;
        params.iterations = max_iters;
        params.extra_directions = extra_dirs;
        params.find_largest = true;
        params.absolute_tolerance = 1e-6f;
        params.relative_tolerance = 1e-6f;
        params.preconditioner = precond_ptr;
        params.instrumentation = &instr;

        UnifiedVector<std::byte> workspace(
            syevx_buffer_size<test_utils::gpu_backend>(*ctx, csr_view, run.W, neigs, JobType::NoEigenVectors,
                                                       MatrixView<float, MatrixFormat::Dense>(), params));

        syevx<test_utils::gpu_backend>(*ctx, csr_view, run.W, neigs, workspace, JobType::NoEigenVectors,
                                       MatrixView<float, MatrixFormat::Dense>(), params);
        ctx->wait_and_throw();
    };

    auto compute_avg_final_best = [&](const UnifiedVector<float>& hist) {
        float sum = 0.0f;
        int count = 0;
        for (size_t b = 0; b < batch; ++b) {
            for (size_t i = 0; i < neigs; ++i) {
                float last = std::nanf("");
                for (int it = 0; it < max_iters; ++it) {
                    const size_t idx = static_cast<size_t>(it) * batch * neigs + b * neigs + i;
                    const float v = hist[idx];
                    if (std::isfinite(v)) last = v;
                }
                if (std::isfinite(last)) {
                    sum += last;
                    ++count;
                }
            }
        }
        return (count > 0) ? (sum / static_cast<float>(count)) : std::nanf("");
    };

    auto compute_avg_rate = [&](const UnifiedVector<float>& hist) {
        float sum = 0.0f;
        int count = 0;
        for (size_t b = 0; b < batch; ++b) {
            for (size_t i = 0; i < neigs; ++i) {
                for (int it = 1; it < std::min(max_iters, 10); ++it) {
                    const size_t cur_idx = static_cast<size_t>(it) * batch * neigs + b * neigs + i;
                    const size_t prev_idx = static_cast<size_t>(it - 1) * batch * neigs + b * neigs + i;
                    const float cur = hist[cur_idx];
                    const float prev = hist[prev_idx];
                    if (std::isfinite(cur) && std::isfinite(prev) && prev > 0.0f) {
                        sum += (cur / prev);
                        ++count;
                    }
                }
            }
        }
        return (count > 0) ? (sum / static_cast<float>(count)) : std::nanf("");
    };


    struct LevelAggregate {
        float ratio_sum = 0.0f;
        int ratio_count = 0;
        int win_count = 0;
        int lose_count = 0;
    };
    std::vector<LevelAggregate> aggregates(runs.size());

    const float tol = 1e-6f;

    const std::filesystem::path csv_path = "/home/jonaslacour/BatchLAS/output/iluk_convergence_trace.csv";
    std::filesystem::create_directories(csv_path.parent_path());
    std::ofstream csv(csv_path);
    ASSERT_TRUE(csv.good());
    csv << "case_label,density,diag_boost,seed,level,label,iter,batch,eig,best_res,current_res,rate,ritz,converged\n";

    std::cout << "\n[ILUK sweep summary] n=" << n << ", neigs=" << neigs << ", extra=" << extra_dirs
              << ", cases=" << sweep_cases.size() << "\n";

    for (const auto& sc : sweep_cases) {
        auto csr = csr_generators::random_sparse_hermitian_csr<float>(
            n,
            sc.density,
            batch,
            sc.seed,
            sc.diagonal_boost,
            true);
        ASSERT_GT(csr.nnz(), n * 2);

        for (auto& run : runs) {
            run_case(run, csr.view());
        }

        const float baseline_final = compute_avg_final_best(runs[0].best_hist);
        std::cout << "  case=" << sc.label << " density=" << sc.density
                  << " diag_boost=" << sc.diagonal_boost << " seed=" << sc.seed
                  << " nnz=" << csr.nnz() << "\n";

        for (size_t r = 0; r < runs.size(); ++r) {
            const auto& run = runs[r];
            const float final_best = compute_avg_final_best(run.best_hist);
            const float avg_rate = compute_avg_rate(run.best_hist);
            std::cout << "    " << run.label << ": avg_final_best=" << final_best
                      << ", avg_residual_ratio=" << avg_rate;
            if (run.iluk_level >= 0 && std::isfinite(baseline_final) && baseline_final > 0.0f && std::isfinite(final_best)) {
                const float ratio = final_best / baseline_final;
                aggregates[r].ratio_sum += ratio;
                aggregates[r].ratio_count += 1;
                if (ratio < 1.0f) {
                    aggregates[r].win_count += 1;
                } else {
                    aggregates[r].lose_count += 1;
                }
                std::cout << ", ratio_vs_baseline=" << ratio;
            }
            std::cout << "\n";

            for (int it = 0; it < max_iters; ++it) {
                for (int b = 0; b < batch; ++b) {
                    for (int i = 0; i < neigs; ++i) {
                        const size_t idx = static_cast<size_t>(it) * batch * neigs + static_cast<size_t>(b) * neigs + i;
                        const float best = run.best_hist[idx];
                        const float cur = run.current_hist[idx];
                        const float rate = run.rate_hist[idx];
                        const float ritz = run.ritz_hist[idx];
                        const int conv = (std::isfinite(best) && best <= tol) ? 1 : 0;
                        csv << sc.label << ',' << sc.density << ',' << sc.diagonal_boost << ',' << sc.seed << ','
                            << run.iluk_level << ',' << run.label << ',' << it << ',' << b << ',' << i << ','
                            << best << ',' << cur << ',' << rate << ',' << ritz << ',' << conv << '\n';
                    }
                }
            }
        }

        for (const auto& run : runs) {
            EXPECT_TRUE(std::isfinite(compute_avg_final_best(run.best_hist)));
            EXPECT_TRUE(std::isfinite(compute_avg_rate(run.best_hist)));
        }
    }

    std::cout << "  aggregate ratios vs baseline:\n";
    for (size_t r = 1; r < runs.size(); ++r) {
        const auto& agg = aggregates[r];
        const float avg_ratio = (agg.ratio_count > 0)
                                    ? (agg.ratio_sum / static_cast<float>(agg.ratio_count))
                                    : std::nanf("");
        std::cout << "    " << runs[r].label << ": avg_ratio=" << avg_ratio
                  << ", wins=" << agg.win_count << ", losses=" << agg.lose_count << "\n";
    }

    csv.close();
    std::cout << "[ILUK trace] wrote " << csv_path << "\n";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#endif  // BATCHLAS_HAS_GPU_BACKEND
