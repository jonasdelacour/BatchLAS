#pragma once
#include <complex>
#include <util/sycl-device-queue.hh>
#include <util/sycl-span.hh>
#include "enums.hh"
#include "matrix_handle.hh"
#include <numeric>
#include <limits>

namespace batchlas {
    template<typename T>
    struct base_type {
        using type = T;
    };

    template<typename T>
    struct base_type<std::complex<T>> {
        using type = T;
    };
    // BLAS Level 3 operations

    /**
     * @brief General matrix multiplication: C = alpha*op(A)*op(B) + beta*C
     * 
     * @param ctx Execution context/device queue
     * @param descrA Matrix A view
     * @param descrB Matrix B view
     * @param descrC Matrix C view (result)
     * @param alpha Scalar multiplier for op(A)*op(B)
     * @param beta Scalar multiplier for C
     * @param transA Operation to apply to A (NoTrans or Trans)
     * @param transB Operation to apply to B (NoTrans or Trans)
     * @param precision Computation precision
     * @return Event Event to track operation completion
     */
    template <Backend B, typename T, BatchType BT>
    Event gemm(Queue& ctx,
        const DenseMatView<T,BT>& descrA,
        const DenseMatView<T,BT>& descrB,
        const DenseMatView<T,BT>& descrC,
        T alpha,
        T beta,
        Transpose transA,        
        Transpose transB,
        ComputePrecision precision = ComputePrecision::Default);

    /**
     * @brief Sparse matrix-dense matrix multiplication: C = alpha*op(A)*op(B) + beta*C
     * 
     * @param ctx Execution context/device queue
     * @param descrA Sparse matrix A handle
     * @param descrB Dense matrix B view
     * @param descrC Dense matrix C view (result)
     * @param alpha Scalar multiplier for op(A)*op(B)
     * @param beta Scalar multiplier for C
     * @param transA Operation to apply to A (NoTrans or Trans)
     * @param transB Operation to apply to B (NoTrans or Trans)
     * @param workspace Pre-allocated workspace buffer
     * @return Event Event to track operation completion
     */
    template <Backend B, typename T, Format F, BatchType BT>
    Event spmm(Queue& ctx,
        SparseMatHandle<T, F, BT>& descrA,
        DenseMatView<T,BT>& descrB,
        DenseMatView<T,BT>& descrC,
        T alpha,
        T beta,
        Transpose transA,
        Transpose transB,
        Span<std::byte> workspace);


    /**
     * @brief Batch generalized matrix-vector multiplication: y = alpha * op(A) * x + beta * y.
     *
     * @tparam B The backend to use for the computation (e.g., CPU, GPU)
     * @tparam T The data type of the matrix and vectors (e.g., float, double)
     * @tparam BT The batch type (e.g., Strided, Array)
     * 
     * @param ctx The queue to execute the operation on
     * @param descrA View of the dense matrix A
     * @param descrX Handle to the dense vector x
     * @param descrY Handle to the dense vector y (input/output)
     * @param alpha Scalar multiplier for op(A) * x
     * @param beta Scalar multiplier for y
     * @param transA Transpose operation to apply to matrix A
     * 
     * @return Event object representing the completion of this operation
     */
    template <Backend B, typename T, BatchType BT>
    Event gemv(Queue& ctx,
        const DenseMatView<T,BT>& descrA,
        const DenseVecHandle<T,BT>& descrX,
        const DenseVecHandle<T,BT>& descrY,
        T alpha,
        T beta,
        Transpose transA);

    /**
     * @brief Get required buffer size for spmm operation
     * 
     * @param ctx Execution context/device queue
     * @param descrA Sparse matrix A handle
     * @param descrB Dense matrix B view
     * @param descrC Dense matrix C view (result)
     * @param alpha Scalar multiplier for op(A)*op(B)
     * @param beta Scalar multiplier for C
     * @param transA Operation to apply to A (NoTrans or Trans)
     * @param transB Operation to apply to B (NoTrans or Trans)
     * @return size_t Required workspace size in bytes
     */
    template <Backend B, typename T, Format F, BatchType BT>
    size_t spmm_buffer_size(Queue& ctx,
        SparseMatHandle<T, F, BT>& descrA,
        DenseMatView<T,BT>& descrB,
        DenseMatView<T,BT>& descrC,
        T alpha,
        T beta,
        Transpose transA,        
        Transpose transB);

    /**
     * @brief Triangular matrix solve: Solves op(A)*X = alpha*B or X*op(A) = alpha*B
     * 
     * @param ctx Execution context/device queue
     * @param descrA Triangular matrix A view
     * @param descrB Right-hand side matrix B view, overwritten with solution X
     * @param side Whether A is on the left or right
     * @param uplo Whether A is upper or lower triangular
     * @param transA Operation to apply to A (NoTrans or Trans)
     * @param diag Whether A is unit diagonal
     * @param alpha Scalar multiplier for B
     * @return Event Event to track operation completion
     */
    template <Backend B, typename T, BatchType BT>
    Event trsm(Queue& ctx,
        const DenseMatView<T,BT>& descrA,
        const DenseMatView<T,BT>& descrB,
        Side side,
        Uplo uplo,
        Transpose transA,
        Diag diag,
        T alpha);

    // LAPACK operations

    /**
     * @brief Get required buffer size for Cholesky factorization
     * 
     * @param ctx Execution context/device queue
     * @param A Matrix to factorize
     * @param uplo Whether to use upper or lower triangle of A
     * @return size_t Required workspace size in bytes
     */
    template <Backend B, typename T, BatchType BT>
    size_t potrf_buffer_size(Queue& ctx,
                        const DenseMatView<T,BT>& A,
                        Uplo uplo);

    /**
     * @brief Cholesky factorization: A = L*L^T or A = U^T*U
     * 
     * @param ctx Execution context/device queue
     * @param descrA Matrix to factorize, overwritten with the result
     * @param uplo Whether to use upper or lower triangle of A
     * @param workspace Pre-allocated workspace buffer
     * @return Event Event to track operation completion
     */ 
    template <Backend B, typename T, BatchType BT>
    Event potrf(Queue& ctx,
            const DenseMatView<T,BT>& descrA,
            Uplo uplo,
            Span<std::byte> workspace);

    /**
     * @brief Symmetric eigenvalue problem: A*x = lambda*x
     * 
     * @param ctx Execution context/device queue
     * @param descrA Symmetric matrix, overwritten with eigenvectors
     * @param eigenvalues Output array for eigenvalues
     * @param jobtype Job type for the operation
     * @param uplo Whether to use upper or lower triangle of A
     * @param workspace Pre-allocated workspace buffer
     * @return Event Event to track operation completion
     */
    template <Backend B, typename T, BatchType BT>
    Event syev(Queue& ctx,
            const DenseMatView<T,BT>& descrA, //A is overwritten with eigenvectors
            Span<typename base_type<T>::type> eigenvalues,
            JobType jobtype,
            Uplo uplo,
            Span<std::byte> workspace);

    /**
     * @brief Get required buffer size for symmetric eigenvalue problem
     * 
     * @param ctx Execution context/device queue
     * @param A Symmetric matrix
     * @param eigenvalues Output array for eigenvalues
     * @param jobtype Job type for the operation
     * @param uplo Whether to use upper or lower triangle of A
     * @return size_t Required workspace size in bytes
     */
    template <Backend B, typename T, BatchType BT>
    size_t syev_buffer_size(Queue& ctx,
            const DenseMatView<T,BT>& A,
            Span<typename base_type<T>::type> eigenvalues,
            JobType jobtype,
            Uplo uplo);

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
    template <Backend B, typename T, BatchType BT>
    Event ortho(Queue& ctx,
            const DenseMatView<T,BT>& A, //A is overwritten with orthogonal vectors
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
    template <Backend B, typename T, BatchType BT>
    Event ortho(Queue& ctx,
            const DenseMatView<T,BT>& A, //A is overwritten with orthogonal vectors
            const DenseMatView<T,BT>& M, //External metric
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
    template <Backend B, typename T, BatchType BT>
    size_t ortho_buffer_size(Queue& ctx,
            const DenseMatView<T,BT>& A,
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
    template <Backend B, typename T, BatchType BT>
    size_t ortho_buffer_size(Queue& ctx,
            const DenseMatView<T,BT>& A,
            const DenseMatView<T,BT>& M,
            Transpose transA,
            Transpose transM,
            OrthoAlgorithm algo = OrthoAlgorithm::Chol2,
            size_t iterations = 2);

    template <typename T>
    struct SyevxParams {
        using float_type = typename base_type<T>::type;
        OrthoAlgorithm algorithm = OrthoAlgorithm::Chol2;  // Default orthogonalization algorithm
        size_t ortho_iterations = 2;                       // Number of orthogonalization iterations
        size_t iterations = 100;                         // Default number of iterations
        size_t extra_directions = 0;                    // Number of extra search directions
        bool find_largest = true;                      // Whether to find largest eigenvalues
        T absolute_tolerance = T(std::numeric_limits<float_type>::epsilon());                 // Absolute tolerance
        T relative_tolerance = T(std::numeric_limits<float_type>::epsilon());                  // Relative tolerance
    };

    template <Backend B, typename T, Format F, BatchType BT>
    Event syevx(Queue& ctx,
                SparseMatHandle<T, F, BT>& A,
                Span<typename base_type<T>::type> W,
                size_t neigs,
                Span<std::byte> workspace,
                JobType jobz = JobType::NoEigenVectors,
                const DenseMatView<T, BT>& V = DenseMatView<T, BT>(),
                const SyevxParams<T>& params = SyevxParams<T>());

    template <Backend B, typename T, Format F, BatchType BT>
    size_t syevx_buffer_size(Queue& ctx,
                SparseMatHandle<T, F, BT>& A,
                Span<typename base_type<T>::type> W,
                size_t neigs,
                JobType jobz = JobType::NoEigenVectors,
                const DenseMatView<T, BT>& V = DenseMatView<T, BT>(),
                const SyevxParams<T>& params = SyevxParams<T>());

    template <typename T>
    struct LanczosParams {
        using float_type = typename base_type<T>::type;
        OrthoAlgorithm ortho_algorithm = OrthoAlgorithm::CGS2;      // Default orthogonalization algorithm
        size_t ortho_iterations = 2;                                // Number of orthogonalization iterations
        size_t reorthogonalization_iterations = 2;                  // Number of iterations before reorthogonalization, increasing this value will reduce accuracy.
        bool sort_enabled = true;                                   // Whether to sort eigenvalues and eigenvectors
        SortOrder sort_order = SortOrder::Ascending;                // Order of sorted eigenvalues and eigenvectors
    };


    template <Backend B, typename T, Format F, BatchType BT>
    Event lanczos(Queue& ctx,
        SparseMatHandle<T, F, BT>& A,
        Span<typename base_type<T>::type> W,
        Span<std::byte> workspace,
        JobType jobz = JobType::NoEigenVectors,
        const DenseMatView<T, BT>& V = DenseMatView<T, BT>(),
        const LanczosParams<T>& params = LanczosParams<T>());

    template <Backend B, typename T, Format F, BatchType BT>
    size_t lanczos_buffer_size(Queue& ctx,
        SparseMatHandle<T, F, BT>& A,
        Span<typename base_type<T>::type> W,
        JobType jobz = JobType::NoEigenVectors,
        const DenseMatView<T, BT>& V = DenseMatView<T, BT>(),
        const LanczosParams<T>& params = LanczosParams<T>());
}