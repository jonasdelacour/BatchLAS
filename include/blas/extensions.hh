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

    // Forwarding overload accepting owning Matrix A
    template <Backend B, typename T>
    inline Event ortho(Queue& ctx,
            const Matrix<T, MatrixFormat::Dense>& A,
            Transpose transA,
            Span<std::byte> workspace,
            OrthoAlgorithm algo = OrthoAlgorithm::Chol2) {
        return ortho<B,T>(ctx, MatrixView<T, MatrixFormat::Dense>(A), transA, workspace, algo);
    }

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

    // Forwarding overload accepting owning Matrices A and M
    template <Backend B, typename T>
    inline Event ortho(Queue& ctx,
            const Matrix<T, MatrixFormat::Dense>& A,
            const Matrix<T, MatrixFormat::Dense>& M,
            Transpose transA,
            Transpose transM,
            Span<std::byte> workspace,
            OrthoAlgorithm algo = OrthoAlgorithm::Chol2,
            size_t iterations = 2) {
        return ortho<B,T>(ctx,
                          MatrixView<T, MatrixFormat::Dense>(A),
                          MatrixView<T, MatrixFormat::Dense>(M),
                          transA, transM, workspace, algo, iterations);
    }
    
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

    // Forwarding overload (owning A)
    template <Backend B, typename T>
    inline size_t ortho_buffer_size(Queue& ctx,
            const Matrix<T, MatrixFormat::Dense>& A,
            Transpose transA,
            OrthoAlgorithm algo = OrthoAlgorithm::Chol2) {
        return ortho_buffer_size<B,T>(ctx, MatrixView<T, MatrixFormat::Dense>(A), transA, algo);
    }

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

    // Forwarding overload (owning A and M)
    template <Backend B, typename T>
    inline size_t ortho_buffer_size(Queue& ctx,
            const Matrix<T, MatrixFormat::Dense>& A,
            const Matrix<T, MatrixFormat::Dense>& M,
            Transpose transA,
            Transpose transM,
            OrthoAlgorithm algo = OrthoAlgorithm::Chol2,
            size_t iterations = 2) {
        return ortho_buffer_size<B,T>(ctx,
                                      MatrixView<T, MatrixFormat::Dense>(A),
                                      MatrixView<T, MatrixFormat::Dense>(M),
                                      transA, transM, algo, iterations);
    }

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

    // Forwarding overload (owning A only, eigenvalues only)
    template <Backend B, typename T, MatrixFormat MFormat>
    inline Event syevx(Queue& ctx,
                const Matrix<T, MFormat>& A,
                Span<typename base_type<T>::type> W,
                size_t neigs,
                Span<std::byte> workspace,
                JobType jobz = JobType::NoEigenVectors,
                const SyevxParams<T>& params = SyevxParams<T>()) {
        return syevx<B,T,MFormat>(ctx, MatrixView<T, MFormat>(A), W, neigs, workspace, jobz, MatrixView<T, MatrixFormat::Dense>(), params);
    }

    // Forwarding overload (owning A and V)
    template <Backend B, typename T, MatrixFormat MFormat>
    inline Event syevx(Queue& ctx,
                const Matrix<T, MFormat>& A,
                Span<typename base_type<T>::type> W,
                size_t neigs,
                Span<std::byte> workspace,
                JobType jobz,
                const Matrix<T, MatrixFormat::Dense>& V,
                const SyevxParams<T>& params = SyevxParams<T>()) {
        return syevx<B,T,MFormat>(ctx, MatrixView<T, MFormat>(A), W, neigs, workspace, jobz, MatrixView<T, MatrixFormat::Dense>(V), params);
    }

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

    // Forwarding overload (owning A only)
    template <Backend B, typename T, MatrixFormat MFormat>
    inline size_t syevx_buffer_size(Queue& ctx,
                const Matrix<T, MFormat>& A,
                Span<typename base_type<T>::type> W,
                size_t neigs,
                JobType jobz = JobType::NoEigenVectors,
                const SyevxParams<T>& params = SyevxParams<T>()) {
        return syevx_buffer_size<B,T,MFormat>(ctx, MatrixView<T, MFormat>(A), W, neigs, jobz, MatrixView<T, MatrixFormat::Dense>(), params);
    }

    // Forwarding overload (owning A and V)
    template <Backend B, typename T, MatrixFormat MFormat>
    inline size_t syevx_buffer_size(Queue& ctx,
                const Matrix<T, MFormat>& A,
                Span<typename base_type<T>::type> W,
                size_t neigs,
                JobType jobz,
                const Matrix<T, MatrixFormat::Dense>& V,
                const SyevxParams<T>& params = SyevxParams<T>()) {
        return syevx_buffer_size<B,T,MFormat>(ctx, MatrixView<T, MFormat>(A), W, neigs, jobz, MatrixView<T, MatrixFormat::Dense>(V), params);
    }

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

    // Forwarding overload (owning A only)
    template <Backend B, typename T, MatrixFormat MFormat>
    inline Event lanczos(Queue& ctx,
        const Matrix<T, MFormat>& A,
        Span<typename base_type<T>::type> W,
        Span<std::byte> workspace,
        JobType jobz = JobType::NoEigenVectors,
        const LanczosParams<T>& params = LanczosParams<T>()) {
        return lanczos<B,T,MFormat>(ctx, MatrixView<T,MFormat>(A), W, workspace, jobz, MatrixView<T, MatrixFormat::Dense>(), params);
    }

    // Forwarding overload (owning A and V)
    template <Backend B, typename T, MatrixFormat MFormat>
    inline Event lanczos(Queue& ctx,
        const Matrix<T, MFormat>& A,
        Span<typename base_type<T>::type> W,
        Span<std::byte> workspace,
        JobType jobz,
        const Matrix<T, MatrixFormat::Dense>& V,
        const LanczosParams<T>& params = LanczosParams<T>()) {
        return lanczos<B,T,MFormat>(ctx, MatrixView<T,MFormat>(A), W, workspace, jobz, MatrixView<T, MatrixFormat::Dense>(V), params);
    }

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

    // Forwarding overload (owning Q)
    template <Backend B, typename T>
    inline Event tridiagonal_solver(Queue& ctx,
         Span<T> alphas,
         Span<T> betas,
         Span<typename base_type<T>::type> W,
         Span<std::byte> workspace,
         JobType jobz,
         const Matrix<T, MatrixFormat::Dense>& Q,
         size_t n,
         size_t batch_size) {
        return tridiagonal_solver<B,T>(ctx, alphas, betas, W, workspace, jobz, MatrixView<T, MatrixFormat::Dense>(Q), n, batch_size);
    }

    template <Backend B, typename T>
    size_t tridiagonal_solver_buffer_size(Queue& ctx, size_t n, size_t batch_size, JobType jobz);

    template <typename T>
    Event francis_sweep(Queue& ctx, const VectorView<T>& d, const VectorView<T>& e, const MatrixView<std::array<T,2>, MatrixFormat::Dense>& givens_rotations = {}, size_t n_sweeps = 1, T zero_threshold = std::numeric_limits<T>::epsilon());

    // Forwarding overloads to allow passing owning Vector<T> directly
    template <typename T>
    inline Event francis_sweep(Queue& ctx, const Vector<T>& d, const Vector<T>& e,
                               const MatrixView<std::array<T,2>, MatrixFormat::Dense>& givens_rotations = {},
                               size_t n_sweeps = 1,
                               T zero_threshold = std::numeric_limits<T>::epsilon()) {
        return francis_sweep<T>(ctx, static_cast<VectorView<T>>(d), static_cast<VectorView<T>>(e), givens_rotations, n_sweeps, zero_threshold);
    }

    enum class SteqrShiftStrategy {
        // LAPACK-style implicit shift (stable formulation used by dsteqr-style iterations).
        Lapack = 0,
        // Wilkinson shift computed from the relevant 2x2 block.
        Wilkinson = 1,
    };

    template <typename T>
    struct SteqrParams {
        //Givens rotations are applied in blocks of this size, increasing this number will lead to excess FLOPs but memory reuse and hence arithmetic intensity improves.
        //Setting this number to 1 is equivalent to full serialization of givens rotation applications, i.e. rotations are applied 1 at a time in the order they were applied to the tridiagonal matrix.
        size_t block_size = 32;
        //Maximum number of sweeps in each Francis QR iteration on average 2-3 iteartions are sufficient to converge to an eigenvalue. 
        size_t max_sweeps = 5; 
        //Threshold for regarding off-diagonal elements as zero  
        T zero_threshold = std::numeric_limits<T>::epsilon(); 
        //Use this toggle to control whether rotations are applied to the eigenvectors matrix passed to STEQR. If false, the matrix will be set to Identity and have rotations applied to this.
        bool back_transform = false; 
        bool block_rotations = false;
        bool sort = true;
        bool transpose_working_vectors = true;
        SortOrder sort_order = SortOrder::Ascending;

        // CTA STEQR only: multiplies the baseline work-group size.
        // Baseline is LCM(N, sub_group_size). The effective work-group size becomes:
        //   wg_size = LCM(N, sub_group_size) * cta_wg_size_multiplier
        // This lets you tune the number of sub-groups per work-group at runtime.
        size_t cta_wg_size_multiplier = 1;

        // CTA STEQR only: select the shift strategy used in the implicit QR/QL steps.
        // - Lapack: stable LAPACK-style implicit shift formulation.
        // - Wilkinson: explicit Wilkinson shift via the eigenvalues of the 2x2 block.
        SteqrShiftStrategy cta_shift_strategy = SteqrShiftStrategy::Lapack;
    };

    template <Backend B, typename T>
    Event steqr(Queue& ctx, const VectorView<T>& d, const VectorView<T>& e,
                const VectorView<T>& eigenvalues, const Span<std::byte>& ws, JobType jobz = JobType::NoEigenVectors, SteqrParams<T> params = SteqrParams<T>(),
                const MatrixView<T, MatrixFormat::Dense>& eigvects = MatrixView<T, MatrixFormat::Dense>());

    // CTA-optimized STEQR for small power-of-two N (runtime-dispatched, compile-time specialized kernels).
    template <Backend B, typename T>
    Event steqr_cta(Queue& ctx, const VectorView<T>& d, const VectorView<T>& e,
                    const VectorView<T>& eigenvalues, const Span<std::byte>& ws,
                    JobType jobz = JobType::NoEigenVectors,
                    SteqrParams<T> params = SteqrParams<T>(),
                    const MatrixView<T, MatrixFormat::Dense>& eigvects = MatrixView<T, MatrixFormat::Dense>());
  
    // Forwarding overload for steqr taking owning Vectors
    template <Backend B, typename T>
    inline Event steqr(Queue& ctx, const Vector<T>& d, const Vector<T>& e,
                       const Vector<T>& eigenvalues, const Span<std::byte>& ws,
                       JobType jobz = JobType::NoEigenVectors,
                       SteqrParams<T> params = SteqrParams<T>(),
                       const Matrix<T, MatrixFormat::Dense>& eigvects = Matrix<T, MatrixFormat::Dense>()) {
        return steqr<B, T>(ctx,
                        static_cast<VectorView<T>>(d),
                        static_cast<VectorView<T>>(e),
                        static_cast<VectorView<T>>(eigenvalues),
                        ws,
                        jobz,
                        params,
                        MatrixView<T, MatrixFormat::Dense>(eigvects));
    }

    // Forwarding overload for steqr_cta taking owning Vectors
    template <Backend B, typename T>
    inline Event steqr_cta(Queue& ctx, const Vector<T>& d, const Vector<T>& e,
                           const Vector<T>& eigenvalues, const Span<std::byte>& ws,
                           JobType jobz = JobType::NoEigenVectors,
                           SteqrParams<T> params = SteqrParams<T>(),
                           const Matrix<T, MatrixFormat::Dense>& eigvects = Matrix<T, MatrixFormat::Dense>()) {
        return steqr_cta<B, T>(ctx,
                               static_cast<VectorView<T>>(d),
                               static_cast<VectorView<T>>(e),
                               static_cast<VectorView<T>>(eigenvalues),
                               ws,
                               jobz,
                               params,
                               MatrixView<T, MatrixFormat::Dense>(eigvects));
    }

    template <typename T>
    size_t steqr_buffer_size(Queue& ctx, const VectorView<T>& d, const VectorView<T>& e,
                            const VectorView<T>& eigenvalues, JobType jobz = JobType::NoEigenVectors, SteqrParams<T> params = SteqrParams<T>());

    template <typename T>
    size_t steqr_cta_buffer_size(Queue& ctx, const VectorView<T>& d, const VectorView<T>& e,
                                 const VectorView<T>& eigenvalues, JobType jobz = JobType::NoEigenVectors, SteqrParams<T> params = SteqrParams<T>());

    // Forwarding overload for steqr_buffer_size taking owning Vectors
    template <typename T>
    inline size_t steqr_buffer_size(Queue& ctx, const Vector<T>& d, const Vector<T>& e,
                                    const Vector<T>& eigenvalues,
                                    JobType jobz = JobType::NoEigenVectors,
                                    SteqrParams<T> params = SteqrParams<T>()) {
        return steqr_buffer_size<T>(ctx,
                                    static_cast<VectorView<T>>(d),
                                    static_cast<VectorView<T>>(e),
                                    static_cast<VectorView<T>>(eigenvalues),
                                    jobz,
                                    params);
    }

    // Forwarding overload for steqr_cta_buffer_size taking owning Vectors
    template <typename T>
    inline size_t steqr_cta_buffer_size(Queue& ctx, const Vector<T>& d, const Vector<T>& e,
                                        const Vector<T>& eigenvalues,
                                        JobType jobz = JobType::NoEigenVectors,
                                        SteqrParams<T> params = SteqrParams<T>()) {
        return steqr_cta_buffer_size<T>(ctx,
                                        static_cast<VectorView<T>>(d),
                                        static_cast<VectorView<T>>(e),
                                        static_cast<VectorView<T>>(eigenvalues),
                                        jobz,
                                        params);
    }


    template <typename T>
    struct StedcParams {
        int64_t recursion_threshold = 32; //Threshold below which the algorithm switches to a non-recursive method such as STEQR
    };

    template <Backend B, typename T>
    Event stedc(Queue& ctx, const VectorView<T>& d, const VectorView<T>& e, const VectorView<T>& eigenvalues, const Span<std::byte>& ws,
            JobType jobz, StedcParams<T> params, const MatrixView<T, MatrixFormat::Dense>& eigvects);

        // Flattened STEDC (non-recursive) for testing/comparison; kept separate from the default path.
        template <Backend B, typename T>
        Event stedc_flat(Queue& ctx, const VectorView<T>& d, const VectorView<T>& e, const VectorView<T>& eigenvalues, const Span<std::byte>& ws,
            JobType jobz, StedcParams<T> params, const MatrixView<T, MatrixFormat::Dense>& eigvects);

    template <Backend B, typename T>
    inline Event stedc(Queue& ctx, const Vector<T>& d, const Vector<T>& e, const Vector<T>& eigenvalues, const Span<std::byte>& ws,
            JobType jobz, StedcParams<T> params, const Matrix<T, MatrixFormat::Dense>& eigvects) {
        return stedc<B,T>(ctx, static_cast<VectorView<T>>(d), static_cast<VectorView<T>>(e), static_cast<VectorView<T>>(eigenvalues), ws, jobz, params, MatrixView<T, MatrixFormat::Dense>(eigvects));
    }

    template <Backend B, typename T>
    inline Event stedc_flat(Queue& ctx, const Vector<T>& d, const Vector<T>& e, const Vector<T>& eigenvalues, const Span<std::byte>& ws,
            JobType jobz, StedcParams<T> params, const Matrix<T, MatrixFormat::Dense>& eigvects) {
        return stedc_flat<B,T>(ctx, static_cast<VectorView<T>>(d), static_cast<VectorView<T>>(e), static_cast<VectorView<T>>(eigenvalues), ws, jobz, params, MatrixView<T, MatrixFormat::Dense>(eigvects));
    }
    
    template <Backend B, typename T>
    size_t stedc_workspace_size(Queue& ctx, size_t n, size_t batch_size, JobType jobz, StedcParams<T> params);


    /**
     * @brief Computes the Ritz values given a matrix and trial vectors
     * 
     * Ritz values are approximations to eigenvalues computed from the Rayleigh quotient:
     * For each column v_j of V: ritz_value[j] = (v_j^T * A * v_j) / (v_j^T * v_j)
     * 
     * @param ctx Execution context/device queue
     * @param A Matrix (can be sparse or dense)
     * @param V Trial vectors (dense matrix, columns are trial eigenvectors)
     * @param ritz_vals Output vector for Ritz values
     * @param workspace Pre-allocated workspace buffer
     * @return Event Event to track operation completion
     */
    template <Backend B, typename T, MatrixFormat MFormat>
    Event ritz_values(Queue& ctx,
                      const MatrixView<T, MFormat>& A,
                      const MatrixView<T, MatrixFormat::Dense>& V,
                      const VectorView<typename base_type<T>::type>& ritz_vals,
                      Span<std::byte> workspace);

    // Forwarding overload (owning A, V, and ritz_vals)
    template <Backend B, typename T, MatrixFormat MFormat>
    inline Event ritz_values(Queue& ctx,
                            const Matrix<T, MFormat>& A,
                            const Matrix<T, MatrixFormat::Dense>& V,
                            const Vector<typename base_type<T>::type>& ritz_vals,
                            Span<std::byte> workspace) {
        return ritz_values<B,T,MFormat>(ctx, MatrixView<T, MFormat>(A), MatrixView<T, MatrixFormat::Dense>(V), static_cast<VectorView<typename base_type<T>::type>>(ritz_vals), workspace);
    }

    //Convenience overload allocating workspace internally
    template <Backend B, typename T, MatrixFormat MFormat>
    inline auto ritz_values(Queue& ctx,
                            const MatrixView<T, MFormat>& A,
                            const MatrixView<T, MatrixFormat::Dense>& V) {
        using float_type = typename base_type<T>::type;
        size_t nRitz = V.cols();
        Vector<float_type> ritz_vals(nRitz, V.batch_size());
        size_t workspace_size = ritz_values_workspace<B,T,MFormat>(ctx, A, V, static_cast<VectorView<float_type>>(ritz_vals));
        UnifiedVector<std::byte> workspace(workspace_size);
        ctx.wait();
        ritz_values<B,T,MFormat>(ctx, A, V, static_cast<VectorView<float_type>>(ritz_vals), workspace);
        ctx.wait();
        return ritz_vals;
    }

    template <Backend B, typename T, MatrixFormat MFormat>
    inline auto ritz_values(Queue& ctx,
                            const Matrix<T, MFormat>& A,
                            const Matrix<T, MatrixFormat::Dense>& V) {
        return ritz_values<B,T,MFormat>(ctx, MatrixView<T, MFormat>(A), MatrixView<T, MatrixFormat::Dense>(V));
    }


    /**
     * @brief Computes the required workspace size for ritz_values
     * 
     * @param ctx Execution context/device queue
     * @param A Matrix (can be sparse or dense)
     * @param V Trial vectors (dense matrix)
     * @param ritz_vals Output vector for Ritz values
     * @return size_t Required workspace size in bytes
     */
    template <Backend B, typename T, MatrixFormat MFormat>
    size_t ritz_values_workspace(Queue& ctx,
                                 const MatrixView<T, MFormat>& A,
                                 const MatrixView<T, MatrixFormat::Dense>& V,
                                 const VectorView<typename base_type<T>::type>& ritz_vals);

    // Forwarding overload (owning A, V, and ritz_vals)
    template <Backend B, typename T, MatrixFormat MFormat>
    inline size_t ritz_values_workspace(Queue& ctx,
                                       const Matrix<T, MFormat>& A,
                                       const Matrix<T, MatrixFormat::Dense>& V,
                                       const Vector<typename base_type<T>::type>& ritz_vals) {
        return ritz_values_workspace<B,T,MFormat>(ctx, MatrixView<T, MFormat>(A), MatrixView<T, MatrixFormat::Dense>(V), static_cast<VectorView<typename base_type<T>::type>>(ritz_vals));
    }

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

    // Forwarding overload (owning A and Ainv)
    template <Backend B, typename T>
    inline Event inv(Queue& ctx,
        const Matrix<T, MatrixFormat::Dense>& A,
        const Matrix<T, MatrixFormat::Dense>& Ainv,
        Span<std::byte> workspace) {
        return inv<B,T>(ctx, MatrixView<T, MatrixFormat::Dense>(A), MatrixView<T, MatrixFormat::Dense>(Ainv), workspace);
    }

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

    // Forwarding overload (owning A)
    template <Backend B, typename T>
    inline size_t inv_buffer_size(Queue& ctx,
        const Matrix<T, MatrixFormat::Dense>& A) {
        return inv_buffer_size<B,T>(ctx, MatrixView<T, MatrixFormat::Dense>(A));
    }

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

    // Forwarding convenience overload (owning A)
    template <Backend B, typename T>
    inline Matrix<T, MatrixFormat::Dense> inv_matrix(Queue& ctx,
        const Matrix<T, MatrixFormat::Dense>& A) {
        return inv<B,T>(ctx, MatrixView<T, MatrixFormat::Dense>(A));
    }

    template <MatrixFormat MType, typename T>
    Event lascl(Queue& ctx, const MatrixView<T, MType>& mat, T cfrom, T cto);
}
