#pragma once
#include <complex>
#include <util/sycl-device-queue.hh>
#include <util/sycl-span.hh>
#include <blas/enums.hh>
#include <blas/matrix.hh>
#include <numeric>
#include <limits>
#include <cstddef>


namespace batchlas {
    // Forward declarations for interface compatibility

    /**
     * @brief Parameters for the Syevx algorithm (eigenvalues calculation)
     * 
     * @tparam T Data type
     */
    #ifndef SYEVSTRUCTS
    #define SYEVSTRUCTS
    template <typename T>
    struct SyevxParams {
        using float_type = typename base_type<T>::type;
        OrthoAlgorithm algorithm = OrthoAlgorithm::Chol2;  // Default orthogonalization algorithm
        size_t ortho_iterations = 2;                       // Number of orthogonalization iterations
        size_t iterations = 100;                           // Default number of iterations
        size_t extra_directions = 0;                       // Number of extra search directions
        bool find_largest = true;                          // Whether to find largest eigenvalues
        T absolute_tolerance = T(std::numeric_limits<float_type>::epsilon());  // Absolute tolerance
        T relative_tolerance = T(std::numeric_limits<float_type>::epsilon());  // Relative tolerance
    };

    /**
     * @brief Parameters for the Lanczos algorithm
     * 
     * @tparam T Data type
     */
    template <typename T>
    struct LanczosParams {
        using float_type = typename base_type<T>::type;
        OrthoAlgorithm ortho_algorithm = OrthoAlgorithm::CGS2;      // Default orthogonalization algorithm
        size_t ortho_iterations = 2;                                // Number of orthogonalization iterations
        size_t reorthogonalization_iterations = 2;                  // Number of iterations before reorthogonalization
        bool sort_enabled = true;                                   // Whether to sort eigenvalues and eigenvectors
        SortOrder sort_order = SortOrder::Ascending;                // Order of sorted eigenvalues and eigenvectors
    };
    #endif

    /**
     * @brief Orthogonalizes a matrix in-place
     * 
     * @param ctx Execution context/device queue
     * @param A Matrix to orthogonalize, overwritten with result
     * @param transA Whether to orthogonalize columns (NoTrans) or rows (Trans)
     * @param workspace Pre-allocated workspace buffer
     * @param algo Algorithm to use for orthogonalization
     * @return Event Event to track operation completion
     */
    template <Backend B, typename T>
    Event ortho(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& A, //A is overwritten with orthogonal vectors
            Transpose transA,
            Span<std::byte> workspace,
            OrthoAlgorithm algo = OrthoAlgorithm::Chol2);

    /**
     * @brief Orthogonalizes a matrix with respect to another matrix in-place
     * 
     * @param ctx Execution context/device queue
     * @param A Matrix to orthogonalize, overwritten with result
     * @param M External metric matrix
     * @param transA Whether to orthogonalize columns (NoTrans) or rows (Trans) of A
     * @param transM Whether to use columns (NoTrans) or rows (Trans) of M
     * @param workspace Pre-allocated workspace buffer
     * @param algo Algorithm to use for orthogonalization
     * @param iterations Number of iterations for improved stability
     * @return Event Event to track operation completion
     */
    template <Backend B, typename T>
    Event ortho(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& A, //A is overwritten with orthogonal vectors
            const MatrixView<T, MatrixFormat::Dense>& M, //External metric
            Transpose transA,
            Transpose transM,
            Span<std::byte> workspace,
            OrthoAlgorithm algo = OrthoAlgorithm::Chol2,
            size_t iterations = 2);
    
    /**
     * @brief Get required buffer size for orthogonalization
     * 
     * @param ctx Execution context/device queue
     * @param A Matrix to orthogonalize
     * @param transA Whether to orthogonalize columns (NoTrans) or rows (Trans)
     * @param algo Algorithm to use for orthogonalization
     * @return size_t Required workspace size in bytes
     */
    template <Backend B, typename T>
    size_t ortho_buffer_size(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& A,
            Transpose transA,
            OrthoAlgorithm algo = OrthoAlgorithm::Chol2);

    /**
     * @brief Get required buffer size for orthogonalization with external metric
     * 
     * @param ctx Execution context/device queue
     * @param A Matrix to orthogonalize
     * @param M External metric matrix
     * @param transA Whether to orthogonalize columns (NoTrans) or rows (Trans) of A
     * @param transM Whether to use columns (NoTrans) or rows (Trans) of M
     * @param algo Algorithm to use for orthogonalization
     * @param iterations Number of iterations for improved stability
     * @return size_t Required workspace size in bytes
     */
    template <Backend B, typename T>
    size_t ortho_buffer_size(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& A,
            const MatrixView<T, MatrixFormat::Dense>& M,
            Transpose transA,
            Transpose transM,
            OrthoAlgorithm algo = OrthoAlgorithm::Chol2,
            size_t iterations = 2);

    /**
     * @brief Computes selected eigenvalues and optionally eigenvectors of a sparse matrix
     * 
     * @param ctx Execution context/device queue
     * @param A Sparse matrix A handle
     * @param W Output array for eigenvalues
     * @param neigs Number of eigenvalues to compute
     * @param workspace Pre-allocated workspace buffer
     * @param jobz Whether to compute eigenvectors
     * @param V Dense matrix to store eigenvectors (if jobz = EigenVectors)
     * @param params Additional parameters for the algorithm
     * @return Event Event to track operation completion
     */
    template <Backend B, typename T, MatrixFormat MFormat>
    Event syevx(Queue& ctx,
                const MatrixView<T, MFormat>& A,
                Span<typename base_type<T>::type> W,
                size_t neigs,
                Span<std::byte> workspace,
                JobType jobz = JobType::NoEigenVectors,
                const MatrixView<T, MatrixFormat::Dense>& V = MatrixView<T, MatrixFormat::Dense>(),
                const SyevxParams<T>& params = SyevxParams<T>());

    /**
     * @brief Get required buffer size for the syevx operation
     * 
     * @param ctx Execution context/device queue
     * @param A Sparse matrix A handle
     * @param W Output array for eigenvalues
     * @param neigs Number of eigenvalues to compute
     * @param jobz Whether to compute eigenvectors
     * @param V Dense matrix to store eigenvectors (if jobz = EigenVectors)
     * @param params Additional parameters for the algorithm
     * @return size_t Required workspace size in bytes
     */
    template <Backend B, typename T, MatrixFormat MFormat>
    size_t syevx_buffer_size(Queue& ctx,
                const MatrixView<T, MFormat>& A,
                Span<typename base_type<T>::type> W,
                size_t neigs,
                JobType jobz = JobType::NoEigenVectors,
                const MatrixView<T, MatrixFormat::Dense>& V = MatrixView<T, MatrixFormat::Dense>(),
                const SyevxParams<T>& params = SyevxParams<T>());

    /**
     * @brief Computes eigenvalues and optionally eigenvectors of a sparse matrix using the Lanczos algorithm
     * 
     * @param ctx Execution context/device queue
     * @param A Sparse matrix A handle
     * @param W Output array for eigenvalues
     * @param workspace Pre-allocated workspace buffer
     * @param jobz Whether to compute eigenvectors
     * @param V Dense matrix to store eigenvectors (if jobz = EigenVectors)
     * @param params Additional parameters for the algorithm
     * @return Event Event to track operation completion
     */
    template <Backend B, typename T, MatrixFormat MFormat>
    Event lanczos(Queue& ctx,
        const MatrixView<T, MFormat>& A,
        Span<typename base_type<T>::type> W,
        Span<std::byte> workspace,
        JobType jobz = JobType::NoEigenVectors,
        const MatrixView<T, MatrixFormat::Dense>& V = MatrixView<T, MatrixFormat::Dense>(),
        const LanczosParams<T>& params = LanczosParams<T>());

    /**
     * @brief Get required buffer size for the Lanczos algorithm
     * 
     * @param ctx Execution context/device queue
     * @param A Sparse matrix A handle
     * @param W Output array for eigenvalues
     * @param jobz Whether to compute eigenvectors
     * @param V Dense matrix to store eigenvectors (if jobz = EigenVectors)
     * @param params Additional parameters for the algorithm
     * @return size_t Required workspace size in bytes
     */
    template <Backend B, typename T, MatrixFormat MFormat>
    size_t lanczos_buffer_size(Queue& ctx,
        const MatrixView<T, MFormat>& A,
        Span<typename base_type<T>::type> W,
        JobType jobz = JobType::NoEigenVectors,
        const MatrixView<T, MatrixFormat::Dense>& V = MatrixView<T, MatrixFormat::Dense>(),
        const LanczosParams<T>& params = LanczosParams<T>());

    template <Backend B, typename T>
    Event tridiagonal_solver(Queue& ctx,
        Span<T> alphas,
        Span<T> betas,
        Span<typename base_type<T>::type> W,
        Span<std::byte> workspace,
        JobType jobz,
        const MatrixView<T, MatrixFormat::Dense>& Q,
        size_t n,
        size_t batch_size);

    template <Backend B, typename T>
    size_t tridiagonal_solver_buffer_size(Queue& ctx, size_t n, size_t batch_size, JobType jobz);
    /**
     * @brief Computes the explicit inverse of a dense matrix
     *
     * @param ctx Execution context/device queue
     * @param A Input matrix to invert
     * @param Ainv Output matrix storing the inverse
     * @param workspace Pre-allocated workspace buffer
     * @return Event Event to track operation completion
     */
    template <Backend B, typename T>
    Event inv(Queue& ctx,
        const MatrixView<T, MatrixFormat::Dense>& A,
        const MatrixView<T, MatrixFormat::Dense>& Ainv,
        Span<std::byte> workspace);

    /**
     * @brief Get required workspace size for matrix inversion
     *
     * @param ctx Execution context/device queue
     * @param A Matrix to invert
     * @return size_t Required workspace size in bytes
     */
    template <Backend B, typename T>
    size_t inv_buffer_size(Queue& ctx,
        const MatrixView<T, MatrixFormat::Dense>& A);

    /**
     * @brief Convenience overload allocating output matrix internally
     *
     * @param ctx Execution context/device queue
     * @param A Matrix to invert
     * @return Matrix<T, MatrixFormat::Dense> Inverted matrix
     */
    template <Backend B, typename T>
    Matrix<T, MatrixFormat::Dense> inv(Queue& ctx,
        const MatrixView<T, MatrixFormat::Dense>& A);
}
