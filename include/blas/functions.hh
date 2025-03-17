#pragma once
#include <complex>
#include <util/sycl-device-queue.hh>
#include <util/sycl-span.hh>
#include "enums.hh"
#include "matrix_handle.hh"

namespace batchlas {
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
        DenseMatView<T,BT> descrA,
        DenseMatView<T,BT> descrB,
        DenseMatView<T,BT> descrC,
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
        DenseMatView<T,BT> descrB,
        DenseMatView<T,BT> descrC,
        T alpha,
        T beta,
        Transpose transA,
        Transpose transB,
        Span<std::byte> workspace);

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
        DenseMatView<T,BT> descrB,
        DenseMatView<T,BT> descrC,
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
        DenseMatView<T,BT> descrA,
        DenseMatView<T,BT> descrB,
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
                        DenseMatView<T,BT> A,
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
            DenseMatView<T,BT> descrA,
            Uplo uplo,
            Span<std::byte> workspace);

    /**
     * @brief Symmetric eigenvalue problem: A*x = lambda*x
     * 
     * @param ctx Execution context/device queue
     * @param descrA Symmetric matrix, overwritten with eigenvectors
     * @param eigenvalues Output array for eigenvalues
     * @param uplo Whether to use upper or lower triangle of A
     * @param workspace Pre-allocated workspace buffer
     * @return Event Event to track operation completion
     */
    template <Backend B, typename T, BatchType BT>
    Event syev(Queue& ctx,
            DenseMatView<T,BT> descrA, //A is overwritten with eigenvectors
            Span<T> eigenvalues,
            Uplo uplo,
            Span<std::byte> workspace);

    /**
     * @brief Get required buffer size for symmetric eigenvalue problem
     * 
     * @param ctx Execution context/device queue
     * @param A Symmetric matrix
     * @param eigenvalues Output array for eigenvalues
     * @param uplo Whether to use upper or lower triangle of A
     * @return size_t Required workspace size in bytes
     */
    template <Backend B, typename T, BatchType BT>
    size_t syev_buffer_size(Queue& ctx,
            DenseMatView<T,BT> A,
            Span<T> eigenvalues,
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
            DenseMatView<T,BT> A, //A is overwritten with orthogonal vectors
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
            DenseMatView<T,BT> A, //A is overwritten with orthogonal vectors
            DenseMatView<T,BT> M, //External metric
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
            DenseMatView<T,BT> A,
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
            DenseMatView<T,BT> A,
            DenseMatView<T,BT> M,
            Transpose transA,
            Transpose transM,
            OrthoAlgorithm algo = OrthoAlgorithm::Chol2,
            size_t iterations = 2);
}