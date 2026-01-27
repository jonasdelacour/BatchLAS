#include <gtest/gtest.h>
#include <blas/linalg.hh>
#include <util/sycl-device-queue.hh>
#include <batchlas/backend_config.h>
#include "test_utils.hh"
#include <complex>
#include <vector>
#include <iostream>
#include <algorithm>

using namespace batchlas;

std::string GetTestName(Transpose trans, OrthoAlgorithm algo);
std::string GetAgainstMTestName(Transpose transA, Transpose transM, OrthoAlgorithm algo);

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


template <typename T, Backend B>
struct OrthoConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

using OrthoTestTypes = typename test_utils::backend_types<OrthoConfig>::type;

template <typename Config>
class OrthoTest : public test_utils::BatchLASTest<Config> {
protected:
    using ScalarType = typename Config::ScalarType;
    static constexpr Backend BackendType = Config::BackendVal;

    // Helper to check for orthonormality: Q^T * Q = I or Q * Q^T = I
    // For column orthonormality (transA = NoTrans), check Q^T * Q = I
    // For row orthonormality (transA = Trans), check Q * Q^T = I
    void check_orthonormality(const MatrixView<ScalarType, MatrixFormat::Dense>& Q_view, Transpose transQ, typename base_type<ScalarType>::type tolerance) {
        int m = transQ == Transpose::NoTrans ? Q_view.rows() : Q_view.cols();
        int k = transQ == Transpose::NoTrans ? Q_view.cols() : Q_view.rows();
        int batch_size = Q_view.batch_size();

        Matrix<ScalarType, MatrixFormat::Dense> I_expected = Matrix<ScalarType, MatrixFormat::Dense>::Identity(k, batch_size);
        Matrix<ScalarType, MatrixFormat::Dense> Result_actual(k, k, batch_size);
        auto I_expected_view = I_expected.view();
        auto Result_actual_view = Result_actual.view();

        // print_matrix(Q_view, "Q_view for check");

        auto transp = std::is_same_v<ScalarType, std::complex<typename base_type<ScalarType>::type>> ? Transpose::ConjTrans : Transpose::Trans;
        if (transQ == Transpose::NoTrans) { // Columns are orthogonal, Q is m x k
            // Result_actual = Q^T * Q
            gemm<BackendType>(*(this->ctx), Q_view, Q_view, Result_actual_view, ScalarType(1.0), ScalarType(0.0), transp, Transpose::NoTrans);
        } else { // Rows are orthogonal, Q is k x m
            // Result_actual = Q * Q^T
            gemm<BackendType>(*(this->ctx), Q_view, Q_view, Result_actual_view, ScalarType(1.0), ScalarType(0.0), Transpose::NoTrans, transp);
        }
        this->ctx->wait();
        // print_matrix(Result_actual_view, "Result_actual (Q^T*Q or Q*Q^T)");
        // print_matrix(I_expected_view, "I_expected");


        auto res_data = Result_actual_view.data();
        auto eye_data = I_expected_view.data();

        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < k; ++i) {
                for (int j = 0; j < k; ++j) {
                    ScalarType expected_val = (i == j) ? ScalarType(1.0) : ScalarType(0.0);
                    // Indexing for Result_actual_view (k x k) and I_expected_view (k x k)
                    // These are always k x k, ld = k, stride = k*k
                    size_t idx = b * k * k + i * k + j;
                    test_utils::assert_near(res_data[idx], expected_val, tolerance);
                }
            }
        }
    }

    // Helper to check M-orthogonality: Q^T * M * Q = I
    void check_M_orthonormality(const MatrixView<ScalarType, MatrixFormat::Dense>& Q_view,
                                const MatrixView<ScalarType, MatrixFormat::Dense>& M_view,
                                Transpose transQ, // Determines orientation of Q for M-orthogonality check
                                typename base_type<ScalarType>::type tolerance) {
        int m_q = transQ == Transpose::NoTrans ? Q_view.rows() : Q_view.cols(); // num_vectors in Q
        int k_q = transQ == Transpose::NoTrans ? Q_view.cols() : Q_view.rows(); // dimension of vectors in Q

        int m_m = M_view.rows();
        int k_m = M_view.cols();
        int batch_size = Q_view.batch_size();

        ASSERT_EQ(batch_size, M_view.batch_size());

        // We expect Q^T * M * Q = I (if transQ = NoTrans, Q is dim x num_vecs)
        // or Q * M * Q^T = I (if transQ = Trans, Q is num_vecs x dim)
        // The identity matrix will be k_q x k_q

        Matrix<ScalarType, MatrixFormat::Dense> I_expected = Matrix<ScalarType, MatrixFormat::Dense>::Identity(k_q, batch_size);
        Matrix<ScalarType, MatrixFormat::Dense> Temp_MQ(m_m, k_q, batch_size); // M*Q or M^T*Q
        Matrix<ScalarType, MatrixFormat::Dense> Result_actual(k_q, k_q, batch_size);

        auto I_expected_view = I_expected.view();
        auto Temp_MQ_view = Temp_MQ.view();
        auto Result_actual_view = Result_actual.view();

        if (transQ == Transpose::NoTrans) { // Q is m x k_q (vectors are columns)
            // M is m x m (assuming M is symmetric and defines inner product for columns of Q)
            // Temp_MQ (m x k_q) = M (m x m) * Q (m x k_q)
            gemm<BackendType>(*(this->ctx), M_view, Q_view, Temp_MQ_view, ScalarType(1.0), ScalarType(0.0), Transpose::NoTrans, Transpose::NoTrans);
            // Result_actual (k_q x k_q) = Q^T (k_q x m) * Temp_MQ (m x k_q)
            gemm<BackendType>(*(this->ctx), Q_view, Temp_MQ_view, Result_actual_view, ScalarType(1.0), ScalarType(0.0), Transpose::Trans, Transpose::NoTrans);
        } else { // Q is k_q x m (vectors are rows)
            // M is m x m
            // Temp_MQ (k_q x m) = Q (k_q x m) * M (m x m)
            gemm<BackendType>(*(this->ctx), Q_view, M_view, Temp_MQ_view, ScalarType(1.0), ScalarType(0.0), Transpose::NoTrans, Transpose::NoTrans);
            // Result_actual (k_q x k_q) = Temp_MQ (k_q x m) * Q^T (m x k_q)
            gemm<BackendType>(*(this->ctx), Temp_MQ_view, Q_view, Result_actual_view, ScalarType(1.0), ScalarType(0.0), Transpose::NoTrans, Transpose::Trans);
        }
        this->ctx->wait();

        // print_matrix(Result_actual_view, "Result_actual (M-ortho)");
        // print_matrix(I_expected_view, "I_expected (M-ortho)");

        auto res_data = Result_actual_view.data().to_vector();
        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < k_q; ++i) {
                for (int j = 0; j < k_q; ++j) {
                    ScalarType expected_val = (i == j) ? ScalarType(1.0) : ScalarType(0.0);
                    size_t idx = b * k_q * k_q + i * k_q + j;
                    test_utils::assert_near(res_data[idx], expected_val, tolerance);
                }
            }
        }
    }
    // Helper to check A is orthogonal to M_basis: A^T * M_basis = 0 or A * M_basis^T = 0
    void check_orthogonality_to_M(const MatrixView<ScalarType, MatrixFormat::Dense>& A_view,
                                   const MatrixView<ScalarType, MatrixFormat::Dense>& M_basis_view,
                                   Transpose transA, // Orientation of A
                                   Transpose transM, // Orientation of M_basis
                                   typename base_type<ScalarType>::type tolerance) {
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
        auto trans = std::is_same_v<ScalarType, std::complex<typename base_type<ScalarType>::type>> ? Transpose::ConjTrans : Transpose::Trans;
        if (transA == Transpose::NoTrans) { // A vectors are columns (dim x nA)
            opA = trans;
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


        Matrix<ScalarType, MatrixFormat::Dense> Result_AM(res_rows, res_cols, batch_size);
        auto Result_AM_view = Result_AM.view();

        gemm<BackendType>(*(this->ctx), A_view, M_basis_view, Result_AM_view, ScalarType(1.0), ScalarType(0.0), opA, opM);
        this->ctx->wait();

        // print_matrix(Result_AM_view, "Result_AM (A vs M_basis)");

        auto res_data = Result_AM_view.data();
        for (int b = 0; b < batch_size; ++b) {
            for (int r = 0; r < res_rows; ++r) {
                for (int c = 0; c < res_cols; ++c) {
                    size_t idx = b * res_rows * res_cols + r * res_cols + c; // Assuming result is row-major
                    test_utils::assert_near(res_data[idx], ScalarType(0.0), tolerance);
                }
            }
        }
    }
};

template <typename Config>
class OrthoMatrixTest : public OrthoTest<Config> {};

template <typename Config>
class OrthoAgainstMTest : public OrthoTest<Config> {};

TYPED_TEST_SUITE(OrthoMatrixTest, OrthoTestTypes);
TYPED_TEST_SUITE(OrthoAgainstMTest, OrthoTestTypes);

// Test implementations
TYPED_TEST(OrthoMatrixTest, OrthogonalizeMatrix) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;
    auto tol = test_utils::tolerance<T>();

    const std::vector<Transpose> transposes = {Transpose::NoTrans};
    std::vector<OrthoAlgorithm> algos = {
        OrthoAlgorithm::Chol2,
        OrthoAlgorithm::ShiftChol3,
        OrthoAlgorithm::CGS2,
        OrthoAlgorithm::SVQB,
        OrthoAlgorithm::SVQB2,
        OrthoAlgorithm::Householder
    };
    if constexpr (std::is_same_v<T, std::complex<double>>) {
        algos.erase(std::remove(algos.begin(), algos.end(), OrthoAlgorithm::Householder), algos.end());
    }

    for (auto transA : transposes) {
        for (auto algo : algos) {
            SCOPED_TRACE(test_utils::backend_to_string(BackendType));
            SCOPED_TRACE(GetTestName(transA, algo));

            const int m = 10, k = 5, batch_size = 2;
            int rows = (transA == Transpose::NoTrans) ? m : k;
            int cols = (transA == Transpose::NoTrans) ? k : m;

            Matrix<T, MatrixFormat::Dense> A = Matrix<T, MatrixFormat::Dense>::Random(rows, cols, false, batch_size);

            size_t buffer_size = ortho_buffer_size<BackendType, T>(*(this->ctx), A, transA, algo);
            UnifiedVector<std::byte> workspace(buffer_size);

            ortho<BackendType, T>(*(this->ctx), A, transA, workspace.to_span(), algo);
            this->ctx->wait();

            this->check_orthonormality(A, transA, tol);
        }
    }
}

/* TEST_F(OrthoMatrixFloatTest, DiagnosticTest) {
    float tol = 1e-5f;
    OrthoAlgorithm algo = OrthoAlgorithm::SVQB;

    const int m = 10, k = 5, batch_size = 2;
    int rows = m;
    int cols = k;

    Matrix<float, MatrixFormat::Dense> A = Matrix<float, MatrixFormat::Dense>::Random(rows, cols, batch_size);
    Matrix<float, MatrixFormat::Dense> Atransposed = A.to_row_major();

    size_t buffer_size = ortho_buffer_size<test_utils::gpu_backend, float>(*(this->ctx), A, Transpose::NoTrans, algo);
    UnifiedVector<std::byte> workspace(buffer_size);

    ortho<test_utils::gpu_backend, float>(*(this->ctx), A, Transpose::NoTrans, workspace.to_span(), algo);
    this->ctx->wait();
    std::cout << A << std::endl;
    ortho<test_utils::gpu_backend, float>(*(this->ctx), MatrixView<float, MatrixFormat::Dense>(Atransposed, 0, 0, cols, rows, cols), Transpose::Trans, workspace.to_span(), algo);
    this->ctx->wait();
    std::cout << MatrixView<float, MatrixFormat::Dense>(Atransposed, 0, 0, cols, rows, cols) << std::endl;

} */

TYPED_TEST(OrthoAgainstMTest, OrthogonalizeMatrixAgainstM) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;
    auto tol = test_utils::tolerance<T>();

    const std::vector<Transpose> transposes = {Transpose::NoTrans};
    const std::vector<OrthoAlgorithm> algos = {
        OrthoAlgorithm::Chol2,
        OrthoAlgorithm::ShiftChol3,
        OrthoAlgorithm::CGS2,
        OrthoAlgorithm::SVQB,
        OrthoAlgorithm::SVQB2
    };

    for (auto transA : transposes) {
        for (auto transM : transposes) {
            for (auto algo : algos) {
                SCOPED_TRACE(test_utils::backend_to_string(BackendType));
                SCOPED_TRACE(GetAgainstMTestName(transA, transM, algo));

                const int dim = 12, nA = 3, nM = 2, batch_size = 2;

                int A_rows = (transA == Transpose::NoTrans) ? dim : nA;
                int A_cols = (transA == Transpose::NoTrans) ? nA : dim;
                int M_rows = (transM == Transpose::NoTrans) ? dim : nM;
                int M_cols = (transM == Transpose::NoTrans) ? nM : dim;

                Matrix<T, MatrixFormat::Dense> A = Matrix<T, MatrixFormat::Dense>::Random(A_rows, A_cols, false, batch_size);
                Matrix<T, MatrixFormat::Dense> M = Matrix<T, MatrixFormat::Dense>::Random(M_rows, M_cols, false, batch_size);

                size_t ortho_M_buffer_size = ortho_buffer_size<BackendType, T>(*(this->ctx), M, transM, algo);
                UnifiedVector<std::byte> workspace_M_ortho(ortho_M_buffer_size);
                ortho<BackendType, T>(*(this->ctx), M, transM, workspace_M_ortho.to_span(), algo);
                this->ctx->wait();

                this->check_orthonormality(M, transM, tol);

                size_t buffer_size = ortho_buffer_size<BackendType, T>(*(this->ctx), A, M, transA, transM, algo);
                UnifiedVector<std::byte> workspace(buffer_size);
                const size_t iterations = 2;

                ortho<BackendType, T>(*(this->ctx), A, M, transA, transM, workspace.to_span(), algo, iterations);
                this->ctx->wait();

                this->check_orthonormality(A, transA, tol);
                this->check_orthogonality_to_M(A, M, transA, transM, tol);
            }
        }
    }
}

// Helper function for test name generation
std::string GetTestName(Transpose trans, OrthoAlgorithm algo) {
    std::string trans_str = (trans == Transpose::NoTrans) ? "NoTrans" : "Trans";
    std::string algo_str;
    switch (algo) {
        case OrthoAlgorithm::Chol2: algo_str = "Chol2"; break;
        case OrthoAlgorithm::ShiftChol3: algo_str = "ShiftChol3"; break;
        case OrthoAlgorithm::SVQB: algo_str = "SVQB"; break;
        case OrthoAlgorithm::SVQB2: algo_str = "SVQB2"; break;
        case OrthoAlgorithm::CGS2: algo_str = "CGS2"; break;
        case OrthoAlgorithm::Householder: algo_str = "Householder"; break;
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
        case OrthoAlgorithm::SVQB2: algo_str = "SVQB2"; break;
        case OrthoAlgorithm::CGS2: algo_str = "CGS2"; break;
        case OrthoAlgorithm::Householder: algo_str = "Householder"; break;
        default: algo_str = "Unknown"; break;
    }
    return "A" + transA_str + "_M" + transM_str + "_" + algo_str;
}

