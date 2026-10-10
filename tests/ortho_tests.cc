#include <gtest/gtest.h>
#include <batchlas/blas/linalg.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/backend_config.h>
#include "test_utils.hh"
#include <batchlas/verify/norms.hh>
#include <batchlas/verify/residuals.hh>
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

    // The vectors of Q (its columns for NoTrans, its rows for Trans) as the columns of a host
    // dim x count batch, conjugated for rows, so one orthogonality call checks either orientation.
    static std::vector<ScalarType> vectors_of(const MatrixView<ScalarType, MatrixFormat::Dense>& Q, Transpose transQ, int& dim, int& count) {
        dim = transQ == Transpose::NoTrans ? Q.rows() : Q.cols();
        count = transQ == Transpose::NoTrans ? Q.cols() : Q.rows();
        std::vector<ScalarType> out(static_cast<size_t>(dim) * count * Q.batch_size());
        for (int b = 0; b < Q.batch_size(); ++b)
            for (int v = 0; v < count; ++v)
                for (int i = 0; i < dim; ++i)
                    out[(static_cast<size_t>(b) * count + v) * dim + i] =
                        transQ == Transpose::NoTrans ? Q(i, v, b) : verify::conj(Q(v, i, b));
        return out;
    }

    // ||Q^H Q - I||_F over every item on the host, judged at @p kind with n = the vector length.
    void check_orthonormality(const MatrixView<ScalarType, MatrixFormat::Dense>& Q, Transpose transQ, verify::Check kind) {
        int dim = 0, count = 0;
        const auto q = vectors_of(Q, transQ, dim, count);
        const auto V = verify::view(q.data(), dim, count, dim, dim * count, Q.batch_size());
        EXPECT_VERIFY(ScalarType, kind, dim, verify::orthogonality(V, verify::all_items(Q.batch_size())));
    }

    // A orthonormal and orthogonal to the orthonormal basis M: the columns of [M A] are orthonormal,
    // so its Gram matrix holds M^H A in the off-diagonal block.
    void check_orthogonality_to_M(const MatrixView<ScalarType, MatrixFormat::Dense>& A, const MatrixView<ScalarType, MatrixFormat::Dense>& M,
                                  Transpose transA, Transpose transM, verify::Check kind) {
        int dim = 0, na = 0, dm = 0, nm = 0;
        const auto a = vectors_of(A, transA, dim, na);
        const auto m = vectors_of(M, transM, dm, nm);
        ASSERT_EQ(dim, dm);
        const int batch = A.batch_size(), cols = nm + na;
        std::vector<ScalarType> ma(static_cast<size_t>(dim) * cols * batch);
        for (int b = 0; b < batch; ++b) {
            std::copy_n(m.begin() + static_cast<std::ptrdiff_t>(b) * dim * nm, dim * nm, ma.begin() + static_cast<std::ptrdiff_t>(b) * dim * cols);
            std::copy_n(a.begin() + static_cast<std::ptrdiff_t>(b) * dim * na, dim * na, ma.begin() + static_cast<std::ptrdiff_t>(b) * dim * cols + dim * nm);
        }
        const auto V = verify::view(ma.data(), dim, cols, dim, dim * cols, batch);
        EXPECT_VERIFY(ScalarType, kind, dim, verify::orthogonality(V, verify::all_items(batch)));
    }
};

// Every algorithm orthogonalizes the input columns directly (reflectors, Cholesky QR, Gram-Schmidt
// or the Gram matrix's eigenbasis, twice where it has a second pass): Householder-grade orthogonality,
// measured c <= 6 at n = 10, 12 (verification.md).
constexpr verify::Check kOrtho = verify::Check::orthogonality;

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

    const std::vector<Transpose> transposes = {Transpose::NoTrans};
    std::vector<OrthoAlgorithm> algos = {
        OrthoAlgorithm::Chol2,
        OrthoAlgorithm::ShiftChol3,
        OrthoAlgorithm::CGS2,
        OrthoAlgorithm::SVQB,
        OrthoAlgorithm::SVQB2,
        OrthoAlgorithm::Householder
    };
    if constexpr (BackendType == Backend::NETLIB) {
        algos.erase(std::remove(algos.begin(), algos.end(), OrthoAlgorithm::Chol2), algos.end());
        algos.erase(std::remove(algos.begin(), algos.end(), OrthoAlgorithm::ShiftChol3), algos.end());
    }
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

            size_t buffer_size = ortho_buffer_size(*(this->ctx), A.view(), transA, algo);
            UnifiedVector<std::byte> workspace(buffer_size);

            (void)ortho(*(this->ctx), A.view(), transA, workspace.to_span(), algo);
            this->ctx->wait();

            this->check_orthonormality(A, transA, kOrtho);
        }
    }
}

TYPED_TEST(OrthoAgainstMTest, OrthogonalizeMatrixAgainstM) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    const std::vector<Transpose> transposes = {Transpose::NoTrans};
    std::vector<OrthoAlgorithm> algos = {
        OrthoAlgorithm::Chol2,
        OrthoAlgorithm::ShiftChol3,
        OrthoAlgorithm::CGS2,
        OrthoAlgorithm::SVQB,
        OrthoAlgorithm::SVQB2
    };
    if constexpr (BackendType == Backend::NETLIB) {
        algos.erase(std::remove(algos.begin(), algos.end(), OrthoAlgorithm::Chol2), algos.end());
        algos.erase(std::remove(algos.begin(), algos.end(), OrthoAlgorithm::ShiftChol3), algos.end());
    }

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

                size_t ortho_M_buffer_size = ortho_buffer_size(*(this->ctx), M.view(), transM, algo);
                UnifiedVector<std::byte> workspace_M_ortho(ortho_M_buffer_size);
                (void)ortho(*(this->ctx), M.view(), transM, workspace_M_ortho.to_span(), algo);
                this->ctx->wait();

                this->check_orthonormality(M, transM, kOrtho);

                size_t buffer_size = ortho_buffer_size(*(this->ctx), A.view(), M.view(), transA, transM, algo);
                UnifiedVector<std::byte> workspace(buffer_size);
                const size_t iterations = 2;

                (void)ortho(*(this->ctx), A.view(), M.view(), transA, transM, workspace.to_span(), algo, iterations);
                this->ctx->wait();

                this->check_orthonormality(A, transA, kOrtho);
                this->check_orthogonality_to_M(A, M, transA, transM, kOrtho);
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

