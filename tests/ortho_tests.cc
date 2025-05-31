\
#include <gtest/gtest.h>
#include "blas/extensions_new.hh"
#include "blas/matrix_handle_new.hh"
#include "blas/functions_matrixview.hh"
#include "util/sycl-device-queue.hh"
#include <complex>
#include <vector>
#include <iostream>

using namespace batchlas;

template <typename T>
void print_matrix(const MatrixView<T, MatrixFormat::Dense>& mat, const std::string& name) {
    std::cout << "Matrix: " << name << " (" << mat.rows() << "x" << mat.cols() << ") Batch: " << mat.batch_size() << std::endl;
    auto mat_host = mat.data().to_vector(); // Assuming to_vector copies to host

    for (int b = 0; b < mat.batch_size(); ++b) {
        std::cout << "Batch " << b << ":" << std::endl;
        for (int i = 0; i < mat.rows(); ++i) {
            for (int j = 0; j < mat.cols(); ++j) {
                // Correct indexing for row-major layout
                std::cout << mat_host[b * mat.stride() + i * mat.ld() + j] << "\\t";
            }
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
}


template <typename T>
class OrthoTest : public ::testing::Test {
protected:
    std::shared_ptr<Queue> ctx;

    void SetUp() override {
        ctx = std::make_shared<Queue>(Device::default_device());
    }

    // Helper to check for orthonormality: Q^T * Q = I or Q * Q^T = I
    // For column orthonormality (transA = NoTrans), check Q^T * Q = I
    // For row orthonormality (transA = Trans), check Q * Q^T = I
    void check_orthonormality(const MatrixView<T, MatrixFormat::Dense>& Q_view, Transpose transQ, T tolerance) {
        int m = transQ == Transpose::NoTrans ? Q_view.rows() : Q_view.cols();
        int k = transQ == Transpose::NoTrans ? Q_view.cols() : Q_view.rows();
        int batch_size = Q_view.batch_size();

        Matrix<T, MatrixFormat::Dense> I_expected = Matrix<T, MatrixFormat::Dense>::Identity(k, batch_size);
        Matrix<T, MatrixFormat::Dense> Result_actual(k, k, batch_size);
        auto I_expected_view = I_expected.view();
        auto Result_actual_view = Result_actual.view();

        // print_matrix(Q_view, "Q_view for check");

        if (transQ == Transpose::NoTrans) { // Columns are orthogonal, Q is m x k
            // Result_actual = Q^T * Q
            gemm<Backend::CUDA>(*ctx, Q_view, Q_view, Result_actual_view, T(1.0), T(0.0), Transpose::Trans, Transpose::NoTrans);
        } else { // Rows are orthogonal, Q is k x m
            // Result_actual = Q * Q^T
            gemm<Backend::CUDA>(*ctx, Q_view, Q_view, Result_actual_view, T(1.0), T(0.0), Transpose::NoTrans, Transpose::Trans);
        }
        ctx->wait();
        // print_matrix(Result_actual_view, "Result_actual (Q^T*Q or Q*Q^T)");
        // print_matrix(I_expected_view, "I_expected");


        auto res_data = Result_actual_view.data();
        auto eye_data = I_expected_view.data();

        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < k; ++i) {
                for (int j = 0; j < k; ++j) {
                    T expected_val = (i == j) ? T(1.0) : T(0.0);
                    // Indexing for Result_actual_view (k x k) and I_expected_view (k x k)
                    // These are always k x k, ld = k, stride = k*k
                    size_t idx = b * k * k + i * k + j;
                    if constexpr (sycl::detail::is_complex<T>::value) {
                        ASSERT_NEAR(res_data[idx].real(), expected_val.real(), tolerance.real());
                        ASSERT_NEAR(res_data[idx].imag(), expected_val.imag(), tolerance.imag());
                    } else {
                        ASSERT_NEAR(res_data[idx], expected_val, tolerance);
                    }
                }
            }
        }
    }

    // Helper to check M-orthogonality: Q^T * M * Q = I
    void check_M_orthonormality(const MatrixView<T, MatrixFormat::Dense>& Q_view,
                                const MatrixView<T, MatrixFormat::Dense>& M_view,
                                Transpose transQ, // Determines orientation of Q for M-orthogonality check
                                T tolerance) {
        int m_q = transQ == Transpose::NoTrans ? Q_view.rows() : Q_view.cols(); // num_vectors in Q
        int k_q = transQ == Transpose::NoTrans ? Q_view.cols() : Q_view.rows(); // dimension of vectors in Q

        int m_m = M_view.rows();
        int k_m = M_view.cols();
        int batch_size = Q_view.batch_size();

        ASSERT_EQ(batch_size, M_view.batch_size());

        // We expect Q^T * M * Q = I (if transQ = NoTrans, Q is dim x num_vecs)
        // or Q * M * Q^T = I (if transQ = Trans, Q is num_vecs x dim)
        // The identity matrix will be k_q x k_q

        Matrix<T, MatrixFormat::Dense> I_expected = Matrix<T, MatrixFormat::Dense>::Identity(k_q, batch_size);
        Matrix<T, MatrixFormat::Dense> Temp_MQ(m_m, k_q, batch_size); // M*Q or M^T*Q
        Matrix<T, MatrixFormat::Dense> Result_actual(k_q, k_q, batch_size);

        auto I_expected_view = I_expected.view();
        auto Temp_MQ_view = Temp_MQ.view();
        auto Result_actual_view = Result_actual.view();

        if (transQ == Transpose::NoTrans) { // Q is m x k_q (vectors are columns)
            // M is m x m (assuming M is symmetric and defines inner product for columns of Q)
            // Temp_MQ (m x k_q) = M (m x m) * Q (m x k_q)
            gemm<Backend::CUDA>(*ctx, M_view, Q_view, Temp_MQ_view, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans);
            // Result_actual (k_q x k_q) = Q^T (k_q x m) * Temp_MQ (m x k_q)
            gemm<Backend::CUDA>(*ctx, Q_view, Temp_MQ_view, Result_actual_view, T(1.0), T(0.0), Transpose::Trans, Transpose::NoTrans);
        } else { // Q is k_q x m (vectors are rows)
            // M is m x m
            // Temp_MQ (k_q x m) = Q (k_q x m) * M (m x m)
            gemm<Backend::CUDA>(*ctx, Q_view, M_view, Temp_MQ_view, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans);
            // Result_actual (k_q x k_q) = Temp_MQ (k_q x m) * Q^T (m x k_q)
            gemm<Backend::CUDA>(*ctx, Temp_MQ_view, Q_view, Result_actual_view, T(1.0), T(0.0), Transpose::NoTrans, Transpose::Trans);
        }
        ctx->wait();

        // print_matrix(Result_actual_view, "Result_actual (M-ortho)");
        // print_matrix(I_expected_view, "I_expected (M-ortho)");

        auto res_data = Result_actual_view.data().to_vector();
        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < k_q; ++i) {
                for (int j = 0; j < k_q; ++j) {
                    T expected_val = (i == j) ? T(1.0) : T(0.0);
                    size_t idx = b * k_q * k_q + i * k_q + j;
                     if constexpr (sycl::detail::is_complex<T>::value) {
                        ASSERT_NEAR(res_data[idx].real(), expected_val.real(), tolerance.real());
                        ASSERT_NEAR(res_data[idx].imag(), expected_val.imag(), tolerance.imag());
                    } else {
                        ASSERT_NEAR(res_data[idx], expected_val, tolerance);
                    }
                }
            }
        }
    }
    // Helper to check A is orthogonal to M_basis: A^T * M_basis = 0 or A * M_basis^T = 0
    void check_orthogonality_to_M(const MatrixView<T, MatrixFormat::Dense>& A_view,
                                   const MatrixView<T, MatrixFormat::Dense>& M_basis_view,
                                   Transpose transA, // Orientation of A
                                   Transpose transM, // Orientation of M_basis
                                   T tolerance) {
        int a_rows = A_view.rows();
        int a_cols = A_view.cols();
        int m_rows = M_basis_view.rows();
        int m_cols = M_basis_view.cols();
        int batch_size = A_view.batch_size();

        // Determine dimensions of the result matrix (should be all zeros)
        // If A has columns as vectors (transA=NoTrans, A is dim x nA)
        // and M_basis has columns as vectors (transM=NoTrans, M_basis is dim x nM)
        // Then A^T * M_basis should be zero (nA x nM)
        int res_rows, res_cols;
        Transpose opA, opM;

        if (transA == Transpose::NoTrans) { // A vectors are columns (dim x nA)
            opA = Transpose::Trans; // Use A^T (nA x dim)
            res_rows = a_cols; // nA
        } else { // A vectors are rows (nA x dim)
            opA = Transpose::NoTrans; // Use A (nA x dim)
            res_rows = a_rows; // nA
        }

        if (transM == Transpose::NoTrans) { // M_basis vectors are columns (dim x nM)
            opM = Transpose::NoTrans; // Use M_basis (dim x nM)
            res_cols = m_cols; // nM
            ASSERT_EQ( (transA == Transpose::NoTrans ? a_rows : a_cols) , m_rows ); // Dimensions must match
        } else { // M_basis vectors are rows (nM x dim)
            opM = Transpose::Trans; // Use M_basis^T (dim x nM)
            res_cols = m_rows; // nM
            ASSERT_EQ( (transA == Transpose::NoTrans ? a_rows : a_cols) , m_cols ); // Dimensions must match
        }


        Matrix<T, MatrixFormat::Dense> Result_AM(res_rows, res_cols, batch_size);
        auto Result_AM_view = Result_AM.view();

        gemm<Backend::CUDA>(*ctx, A_view, M_basis_view, Result_AM_view, T(1.0), T(0.0), opA, opM);
        ctx->wait();

        // print_matrix(Result_AM_view, "Result_AM (A vs M_basis)");

        auto res_data = Result_AM_view.data();
        for (int b = 0; b < batch_size; ++b) {
            for (int r = 0; r < res_rows; ++r) {
                for (int c = 0; c < res_cols; ++c) {
                    size_t idx = b * res_rows * res_cols + r * res_cols + c; // Assuming result is row-major
                     if constexpr (sycl::detail::is_complex<T>::value) {
                        ASSERT_NEAR(res_data[idx].real(), T(0.0).real(), tolerance.real());
                        ASSERT_NEAR(res_data[idx].imag(), T(0.0).imag(), tolerance.imag());
                    } else {
                        ASSERT_NEAR(res_data[idx], T(0.0), tolerance);
                    }
                }
            }
        }
    }
};

// Separate implementations for different types
// Float version of ortho matrix test
class OrthoMatrixFloatTest : public OrthoTest<float>,
                           public ::testing::WithParamInterface<std::tuple<Transpose, OrthoAlgorithm>> {
};

// Double version of ortho matrix test
class OrthoMatrixDoubleTest : public OrthoTest<double>,
                            public ::testing::WithParamInterface<std::tuple<Transpose, OrthoAlgorithm>> {
};

// Float version of ortho against M test
class OrthoAgainstMFloatTest : public OrthoTest<float>,
                             public ::testing::WithParamInterface<std::tuple<Transpose, Transpose, OrthoAlgorithm>> {
};

// Double version of ortho against M test
class OrthoAgainstMDoubleTest : public OrthoTest<double>,
                              public ::testing::WithParamInterface<std::tuple<Transpose, Transpose, OrthoAlgorithm>> {
};

// Test implementations
TEST_P(OrthoMatrixFloatTest, OrthogonalizeMatrix) {
    float tol = 1e-5f;
    Transpose transA = std::get<0>(GetParam());
    OrthoAlgorithm algo = std::get<1>(GetParam());
    
    const int m = 10, k = 5, batch_size = 2;
    int rows = (transA == Transpose::NoTrans) ? m : k;
    int cols = (transA == Transpose::NoTrans) ? k : m;

    Matrix<float, MatrixFormat::Dense> A = Matrix<float, MatrixFormat::Dense>::Random(rows, cols, batch_size);

    size_t buffer_size = ortho_buffer_size<Backend::CUDA, float>(*(this->ctx), A, transA, algo);
    UnifiedVector<std::byte> workspace(buffer_size);

    ortho<Backend::CUDA, float>(*(this->ctx), A, transA, workspace.to_span(), algo);
    this->ctx->wait();

    this->check_orthonormality(A, transA, tol);
}

/* TEST_F(OrthoMatrixFloatTest, DiagnosticTest) {
    float tol = 1e-5f;
    OrthoAlgorithm algo = OrthoAlgorithm::SVQB;

    const int m = 10, k = 5, batch_size = 2;
    int rows = m;
    int cols = k;

    Matrix<float, MatrixFormat::Dense> A = Matrix<float, MatrixFormat::Dense>::Random(rows, cols, batch_size);
    Matrix<float, MatrixFormat::Dense> Atransposed = A.to_row_major();

    size_t buffer_size = ortho_buffer_size<Backend::CUDA, float>(*(this->ctx), A, Transpose::NoTrans, algo);
    UnifiedVector<std::byte> workspace(buffer_size);

    ortho<Backend::CUDA, float>(*(this->ctx), A, Transpose::NoTrans, workspace.to_span(), algo);
    this->ctx->wait();
    std::cout << A << std::endl;
    ortho<Backend::CUDA, float>(*(this->ctx), MatrixView<float, MatrixFormat::Dense>(Atransposed, 0, 0, cols, rows, cols), Transpose::Trans, workspace.to_span(), algo);
    this->ctx->wait();
    std::cout << MatrixView<float, MatrixFormat::Dense>(Atransposed, 0, 0, cols, rows, cols) << std::endl;

} */

TEST_P(OrthoMatrixDoubleTest, OrthogonalizeMatrix) {
    double tol = 1e-9;
    Transpose transA = std::get<0>(GetParam());
    OrthoAlgorithm algo = std::get<1>(GetParam());
    
    const int m = 10, k = 5, batch_size = 2;
    int rows = (transA == Transpose::NoTrans) ? m : k;
    int cols = (transA == Transpose::NoTrans) ? k : m;

    Matrix<double, MatrixFormat::Dense> A = Matrix<double, MatrixFormat::Dense>::Random(rows, cols, batch_size);

    size_t buffer_size = ortho_buffer_size<Backend::CUDA, double>(*(this->ctx), A, transA, algo);
    UnifiedVector<std::byte> workspace(buffer_size);

    ortho<Backend::CUDA, double>(*(this->ctx), A, transA, workspace.to_span(), algo);
    this->ctx->wait();

    this->check_orthonormality(A, transA, tol);
}

TEST_P(OrthoAgainstMFloatTest, OrthogonalizeMatrixAgainstM) {
    float tol = 1e-5f;
    Transpose transA = std::get<0>(GetParam());
    Transpose transM = std::get<1>(GetParam());
    OrthoAlgorithm algo = std::get<2>(GetParam());
    
    const int dim = 12, nA = 3, nM = 2, batch_size = 2;
    
    int A_rows = (transA == Transpose::NoTrans) ? dim : nA;
    int A_cols = (transA == Transpose::NoTrans) ? nA : dim;
    int M_rows = (transM == Transpose::NoTrans) ? dim : nM;
    int M_cols = (transM == Transpose::NoTrans) ? nM : dim;

    Matrix<float, MatrixFormat::Dense> A = Matrix<float, MatrixFormat::Dense>::Random(A_rows, A_cols, batch_size);
    Matrix<float, MatrixFormat::Dense> M = Matrix<float, MatrixFormat::Dense>::Random(M_rows, M_cols, batch_size);

    size_t ortho_M_buffer_size = ortho_buffer_size<Backend::CUDA, float>(*(this->ctx), M, transM, algo);
    UnifiedVector<std::byte> workspace_M_ortho(ortho_M_buffer_size);
    ortho<Backend::CUDA, float>(*(this->ctx), M, transM, workspace_M_ortho.to_span(), algo);
    this->ctx->wait();

    this->check_orthonormality(M, transM, tol);

    size_t buffer_size = ortho_buffer_size<Backend::CUDA, float>(*(this->ctx), A, M, transA, transM, algo);
    UnifiedVector<std::byte> workspace(buffer_size);
    const size_t iterations = 2;

    ortho<Backend::CUDA, float>(*(this->ctx), A, M, transA, transM, workspace.to_span(), algo, iterations);
    this->ctx->wait();

    this->check_orthonormality(A, transA, tol);
    this->check_orthogonality_to_M(A, M, transA, transM, tol);
}



TEST_P(OrthoAgainstMDoubleTest, OrthogonalizeMatrixAgainstM) {
    double tol = 1e-9;
    Transpose transA = std::get<0>(GetParam());
    Transpose transM_basis = std::get<1>(GetParam());
    OrthoAlgorithm algo = std::get<2>(GetParam());
    
    const int dim = 12, nA = 3, nM = 2, batch_size = 2;
    
    int A_rows = (transA == Transpose::NoTrans) ? dim : nA;
    int A_cols = (transA == Transpose::NoTrans) ? nA : dim;
    int M_basis_rows = (transM_basis == Transpose::NoTrans) ? dim : nM;
    int M_basis_cols = (transM_basis == Transpose::NoTrans) ? nM : dim;

    Matrix<double, MatrixFormat::Dense> A = Matrix<double, MatrixFormat::Dense>::Random(A_rows, A_cols, batch_size);
    auto A_view = A.view();

    Matrix<double, MatrixFormat::Dense> M_basis_orig = Matrix<double, MatrixFormat::Dense>::Random(M_basis_rows, M_basis_cols, batch_size);
    auto M_basis_view_orig = M_basis_orig.view();
    size_t ortho_M_buffer_size = ortho_buffer_size<Backend::CUDA, double>(*(this->ctx), M_basis_view_orig, transM_basis, algo);
    UnifiedVector<std::byte> workspace_M_ortho(ortho_M_buffer_size);
    ortho<Backend::CUDA, double>(*(this->ctx), M_basis_view_orig, transM_basis, workspace_M_ortho.to_span(), algo);
    this->ctx->wait();
    
    this->check_orthonormality(M_basis_view_orig, transM_basis, tol);

    auto M_ortho_basis_view = M_basis_view_orig;

    size_t buffer_size = ortho_buffer_size<Backend::CUDA, double>(*(this->ctx), A_view, M_ortho_basis_view, transA, transM_basis, algo);
    UnifiedVector<std::byte> workspace(buffer_size);
    const size_t iterations = 2;

    ortho<Backend::CUDA, double>(*(this->ctx), A_view, M_ortho_basis_view, transA, transM_basis, workspace.to_span(), algo, iterations);
    this->ctx->wait();

    this->check_orthonormality(A_view, transA, tol);
    this->check_orthogonality_to_M(A_view, M_ortho_basis_view, transA, transM_basis, tol);
}

// Helper function for test name generation
std::string GetTestName(Transpose trans, OrthoAlgorithm algo) {
    std::string trans_str = (trans == Transpose::NoTrans) ? "NoTrans" : "Trans";
    std::string algo_str;
    switch (algo) {
        case OrthoAlgorithm::Chol2: algo_str = "Chol2"; break;
        case OrthoAlgorithm::ShiftChol3: algo_str = "ShiftChol3"; break;
        case OrthoAlgorithm::SVQB: algo_str = "SVQB"; break;
        case OrthoAlgorithm::CGS2: algo_str = "CGS2"; break;
        default: algo_str = "Unknown"; break;
    }
    return trans_str + "_" + algo_str;
}

std::string GetAgainstMTestName(Transpose transA, Transpose transM, OrthoAlgorithm algo) {
    std::string transA_str = (transA == Transpose::NoTrans) ? "NoTrans" : "Trans";
    std::string transM_str = (transM == Transpose::NoTrans) ? "NoTrans" : "Trans";
    std::string algo_str;
    switch (algo) {
        case OrthoAlgorithm::Chol2: algo_str = "Chol2"; break;
        case OrthoAlgorithm::ShiftChol3: algo_str = "ShiftChol3"; break;
        case OrthoAlgorithm::SVQB: algo_str = "SVQB"; break;
        case OrthoAlgorithm::CGS2: algo_str = "CGS2"; break;
        default: algo_str = "Unknown"; break;
    }
    return "A" + transA_str + "_M" + transM_str + "_" + algo_str;
}

// Instantiate the value-parameterized tests
INSTANTIATE_TEST_SUITE_P(
    Combinations, OrthoMatrixFloatTest,
    ::testing::Combine(
        ::testing::Values(Transpose::NoTrans, Transpose::Trans),
        ::testing::Values(OrthoAlgorithm::Chol2, OrthoAlgorithm::ShiftChol3, OrthoAlgorithm::CGS2)
    ),
    [](const ::testing::TestParamInfo<OrthoMatrixFloatTest::ParamType>& info) {
        Transpose trans = std::get<0>(info.param);
        OrthoAlgorithm algo = std::get<1>(info.param);
        return GetTestName(trans, algo);
    }
);

INSTANTIATE_TEST_SUITE_P(
    Combinations, OrthoMatrixDoubleTest,
    ::testing::Combine(
        ::testing::Values(Transpose::NoTrans, Transpose::Trans),
        ::testing::Values(OrthoAlgorithm::Chol2, OrthoAlgorithm::ShiftChol3, OrthoAlgorithm::CGS2)
    ),
    [](const ::testing::TestParamInfo<OrthoMatrixDoubleTest::ParamType>& info) {
        Transpose trans = std::get<0>(info.param);
        OrthoAlgorithm algo = std::get<1>(info.param);
        return GetTestName(trans, algo);
    }
);

INSTANTIATE_TEST_SUITE_P(
    Combinations, OrthoAgainstMFloatTest,
    ::testing::Combine(
        ::testing::Values(Transpose::NoTrans, Transpose::Trans),
        ::testing::Values(Transpose::NoTrans, Transpose::Trans),
        ::testing::Values(OrthoAlgorithm::Chol2, OrthoAlgorithm::ShiftChol3, OrthoAlgorithm::CGS2)
    ),
    [](const ::testing::TestParamInfo<OrthoAgainstMFloatTest::ParamType>& info) {
        Transpose transA = std::get<0>(info.param);
        Transpose transM = std::get<1>(info.param);
        OrthoAlgorithm algo = std::get<2>(info.param);
        return GetAgainstMTestName(transA, transM, algo);
    }
);

INSTANTIATE_TEST_SUITE_P(
    Combinations, OrthoAgainstMDoubleTest,
    ::testing::Combine(
        ::testing::Values(Transpose::NoTrans, Transpose::Trans),
        ::testing::Values(Transpose::NoTrans, Transpose::Trans),
        ::testing::Values(OrthoAlgorithm::Chol2, OrthoAlgorithm::ShiftChol3, OrthoAlgorithm::CGS2)
    ),
    [](const ::testing::TestParamInfo<OrthoAgainstMDoubleTest::ParamType>& info) {
        Transpose transA = std::get<0>(info.param);
        Transpose transM = std::get<1>(info.param);
        OrthoAlgorithm algo = std::get<2>(info.param);
        return GetAgainstMTestName(transA, transM, algo);
    }
);

