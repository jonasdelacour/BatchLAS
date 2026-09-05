#pragma once
#include <complex>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/tuning_params.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/queue-dispatch.hh>
#include <batchlas/blas/functions/iluk.hh>
#include <numeric>
#include <limits>
#include <cstddef>
#include <cstdint>
#include <vector>


namespace batchlas {

    template <typename T>
    struct SyevxInstrumentation;

    template <typename T>
    struct StedcParams;

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
        SyevxAlgorithm method = SyevxAlgorithm::Auto;      // Algorithm family (see SyevxAlgorithm)
        OrthoAlgorithm algorithm = OrthoAlgorithm::Chol2;  // Default orthogonalization algorithm
        size_t ortho_iterations = 2;                       // Number of orthogonalization iterations
        size_t iterations = 100;                           // Default number of iterations
        size_t extra_directions = 0;                       // Number of extra search directions
        bool find_largest = true;                          // Whether to find largest eigenvalues
        T absolute_tolerance = T(std::numeric_limits<float_type>::epsilon());  // Absolute tolerance
        T relative_tolerance = T(std::numeric_limits<float_type>::epsilon());  // Relative tolerance
        // ILU(k) approximates A^{-1}, so it is a valid LOBPCG preconditioner only for the
        // SMALLEST eigenpairs; find_largest = true is rejected, not silently accepted.
        const ILUKPreconditioner<T>* preconditioner = nullptr;
        // `Auto`: ILU(k) when a factor is supplied or requested below, otherwise none
        // (unless BATCHLAS_SYEVX_PRECONDITIONER names a default).
        SyevxPreconditioner preconditioner_type = SyevxPreconditioner::Auto;
        // Builds the ILU(k) factor inside syevx from the caller's workspace. Requires a
        // CSR A and find_largest = false; mutually exclusive with the pointer above.
        bool build_preconditioner = false;
        ILUKParams<T> iluk_params{};
        // Chebyshev filter degree for SyevxAlgorithm::Filtered; 0 selects a default.
        size_t filter_degree = 0;
        // LOBPCG only: power-iteration steps on the random start block; -1 default,
        // 0 disables. Ignored unless find_largest is true.
        int init_power_iterations = -1;
        const SyevxInstrumentation<T>* instrumentation = nullptr;               // Optional convergence instrumentation sink

        // ---- Range selection (LAPACK ?syevx's RANGE argument) --------------

        SyevxSelect select = SyevxSelect::Extremal;

        // select == Index: inclusive 0-based bounds into the ASCENDING spectrum.
        // iu < 0 means n-1. il > iu is an empty request and is rejected.
        int64_t il = 0;
        int64_t iu = -1;
        // select == Value: the half-open interval (vl, vu], as in LAPACK. The count is
        // data-dependent per item, so it is reported through syevx's `m` output.
        float_type vl = float_type(0);
        float_type vu = float_type(0);

        // Absolute tolerance per eigenvalue; non-positive means eps * ||T||. Forwarded
        // to StebzParams::abstol; ignored by paths that use a full decomposition.
        float_type abstol = float_type(0);

        // Honoured for Index and Value only; for Extremal the order follows find_largest.
        SortOrder order = SortOrder::Ascending;
    };

    template <typename T>
    struct SyevxInstrumentation {
        using float_type = typename base_type<T>::type;

        // Histories are laid out as: [iter][batch][eig], with optional custom strides.
        // If iteration_stride/batch_stride are zero they default to batch_size*neigs and neigs.
        Span<float_type> best_residual_history{};
        Span<float_type> current_residual_history{};
        Span<float_type> convergence_rate_history{};
        Span<float_type> ritz_value_history{};

        int32_t* iterations_done = nullptr;  // Optional per-batch output

        size_t max_iterations = 0;
        size_t store_every = 1;
        size_t iteration_stride = 0;
        size_t batch_stride = 0;

        bool store_current_residual = false;
        bool store_convergence_rate = true;
        bool store_ritz_values = false;
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
     * @brief Computes selected eigenvalues and optionally eigenvectors of a
     *        Hermitian/symmetric matrix, dense or sparse.
     *
     * `SyevxParams::select` picks the part of the spectrum: the `neigs` extremal
     * eigenpairs, an index block `il..iu`, or the interval `(vl, vu]`.
     *
     * @param ctx Execution context/device queue
     * @param A Matrix A (dense or CSR)
     * @param W Output array for eigenvalues, `neigs` entries per batch item
     * @param neigs CAPACITY of `W` and of `V`'s columns per batch item -- not
     *        necessarily the number produced.
     * @param workspace Pre-allocated workspace buffer
     * @param jobz Whether to compute eigenvectors
     * @param V Dense matrix to store eigenvectors (if jobz = EigenVectors)
     * @param params Additional parameters for the algorithm
     * @return Event Event to track operation completion
     *
     * @throws std::invalid_argument if `select` is `Value` (use the `m`-taking overload
     *         below) or if a non-extremal range is asked of a path that cannot answer
     *         one. See `syevx_select_algorithm`.
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
     * @brief `syevx` with a per-batch-item count of how many eigenpairs were
     *        actually found. Required for `SyevxSelect::Value`.
     *
     * LAPACK's contract: `neigs` is a capacity, `min(m[b], neigs)` eigenpairs are written
     * into the leading slots of item `b`, the rest of that item's `W` is left untouched,
     * the remaining columns of its `V` are written as EXACTLY ZERO, and `m[b]` reports the
     * TRUE count -- so `m[b] > neigs` is the caller's overflow signal. When truncating,
     * the LOWEST `neigs` eigenvalues of the interval are kept. Zeroing `V` but not `W` is
     * a CONTRACT: the subset path's back-transforms run over a uniform column count and
     * need the unused columns inert.
     *
     * @param m Per-item count, at least `A.batch_size()` entries. Device-writable.
     *
     * OVERLOAD-RESOLUTION INVARIANT: these stay unambiguous against the `m`-less forms
     * only because argument positions 4 and 5 diverge into mutually non-convertible types,
     * which relies on `Span`'s scalar constructor staying `explicit`. Never spell a bare
     * `{}` in positions 4-6: it matches all of them at once and the call is ambiguous.
     */
    template <Backend B, typename T, MatrixFormat MFormat>
    Event syevx(Queue& ctx,
                const MatrixView<T, MFormat>& A,
                Span<typename base_type<T>::type> W,
                Span<int32_t> m,
                size_t neigs,
                Span<std::byte> workspace,
                JobType jobz = JobType::NoEigenVectors,
                const MatrixView<T, MatrixFormat::Dense>& V = MatrixView<T, MatrixFormat::Dense>(),
                const SyevxParams<T>& params = SyevxParams<T>());

    template <Backend B, typename T, MatrixFormat MFormat>
    inline Event syevx(Queue& ctx,
                const Matrix<T, MFormat>& A,
                Span<typename base_type<T>::type> W,
                Span<int32_t> m,
                size_t neigs,
                Span<std::byte> workspace,
                JobType jobz = JobType::NoEigenVectors,
                const SyevxParams<T>& params = SyevxParams<T>()) {
        return syevx<B,T,MFormat>(ctx, MatrixView<T, MFormat>(A), W, m, neigs, workspace, jobz, MatrixView<T, MatrixFormat::Dense>(), params);
    }

    template <Backend B, typename T, MatrixFormat MFormat>
    inline Event syevx(Queue& ctx,
                const Matrix<T, MFormat>& A,
                Span<typename base_type<T>::type> W,
                Span<int32_t> m,
                size_t neigs,
                Span<std::byte> workspace,
                JobType jobz,
                const Matrix<T, MatrixFormat::Dense>& V,
                const SyevxParams<T>& params = SyevxParams<T>()) {
        return syevx<B,T,MFormat>(ctx, MatrixView<T, MFormat>(A), W, m, neigs, workspace, jobz, MatrixView<T, MatrixFormat::Dense>(V), params);
    }

    /**
     * @brief Get required buffer size for the syevx operation
     *
     * Unlike the solve, this accepts `SyevxSelect::Value`: sizing writes no counts. It
     * still resolves the range through `syevx_resolve_range`, because
     * `syevx_direct_subset`'s workspace depends on it.
     *
     * @param ctx Execution context/device queue
     * @param A Matrix A (dense or CSR)
     * @param W Output array for eigenvalues
     * @param neigs Capacity of W and V per batch item (see `syevx`)
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

    template <Backend B, typename T, MatrixFormat MFormat>
    inline size_t syevx_buffer_size(Queue& ctx,
                const Matrix<T, MFormat>& A,
                Span<typename base_type<T>::type> W,
                size_t neigs,
                JobType jobz = JobType::NoEigenVectors,
                const SyevxParams<T>& params = SyevxParams<T>()) {
        return syevx_buffer_size<B,T,MFormat>(ctx, MatrixView<T, MFormat>(A), W, neigs, jobz, MatrixView<T, MatrixFormat::Dense>(), params);
    }

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

    // `m`-taking sizing forms. `m` is ACCEPTED AND IGNORED; they exist so a value-range
    // caller can write the sizing call and the solve call with the same argument list.
    template <Backend B, typename T, MatrixFormat MFormat>
    inline size_t syevx_buffer_size(Queue& ctx,
                const MatrixView<T, MFormat>& A,
                Span<typename base_type<T>::type> W,
                Span<int32_t> m,
                size_t neigs,
                JobType jobz = JobType::NoEigenVectors,
                const MatrixView<T, MatrixFormat::Dense>& V = MatrixView<T, MatrixFormat::Dense>(),
                const SyevxParams<T>& params = SyevxParams<T>()) {
        (void)m;
        return syevx_buffer_size<B,T,MFormat>(ctx, A, W, neigs, jobz, V, params);
    }

    template <Backend B, typename T, MatrixFormat MFormat>
    inline size_t syevx_buffer_size(Queue& ctx,
                const Matrix<T, MFormat>& A,
                Span<typename base_type<T>::type> W,
                Span<int32_t> m,
                size_t neigs,
                JobType jobz = JobType::NoEigenVectors,
                const SyevxParams<T>& params = SyevxParams<T>()) {
        (void)m;
        return syevx_buffer_size<B,T,MFormat>(ctx, MatrixView<T, MFormat>(A), W, neigs, jobz, MatrixView<T, MatrixFormat::Dense>(), params);
    }

    template <Backend B, typename T, MatrixFormat MFormat>
    inline size_t syevx_buffer_size(Queue& ctx,
                const Matrix<T, MFormat>& A,
                Span<typename base_type<T>::type> W,
                Span<int32_t> m,
                size_t neigs,
                JobType jobz,
                const Matrix<T, MatrixFormat::Dense>& V,
                const SyevxParams<T>& params = SyevxParams<T>()) {
        (void)m;
        return syevx_buffer_size<B,T,MFormat>(ctx, MatrixView<T, MFormat>(A), W, neigs, jobz, MatrixView<T, MatrixFormat::Dense>(V), params);
    }

    /**
     * @brief Resolved, algorithm-independent description of what the caller asked
     *        for, produced by `syevx_resolve_range`.
     *
     * Every solver and every `*_buffer_size` derives its behaviour from this struct
     * and never from SyevxParams::select / il / iu / find_largest directly, so that
     * the solve and the sizing call can never disagree. `vl`/`vu` are deliberately
     * absent -- they are forwarded verbatim, so read `params.vl` / `params.vu`.
     */
    struct SyevxResolvedRange {
        bool    value_range;  // true: (vl, vu]; false: the index block [il, iu]
        int64_t il;           // valid iff !value_range; 0-based, inclusive
        int64_t iu;           // valid iff !value_range; 0-based, inclusive
        // Upper bound on eigenpairs per item, clamped to [0, n] so a consumer may index
        // [il, il + max_count) into an n-entry array without a bound of its own. Exact for
        // an index block; for a value range the true m[b] may be larger.
        int64_t max_count;
        bool    reverse;      // write the selected block in descending order
    };

    /**
     * @brief Normalizes a range request into a SyevxResolvedRange.
     *
     * Legality is NOT checked here: the request must already have passed the host-side
     * validator in syevx.cc. It clamps rather than throws so it stays usable when sizing.
     *
     * @param n     Matrix dimension
     * @param neigs Capacity of W and V per batch item (see `syevx`)
     * @param select      SyevxParams::select
     * @param find_largest SyevxParams::find_largest (Extremal only)
     * @param il,iu       SyevxParams::il / iu (Index only; iu < 0 means n-1)
     * @param order       SyevxParams::order (Index and Value only)
     */
    SyevxResolvedRange syevx_resolve_range(int64_t n,
                                           size_t neigs,
                                           SyevxSelect select,
                                           bool find_largest,
                                           int64_t il,
                                           int64_t iu,
                                           SortOrder order);

    // Convenience adaptor, distinguished from the 7-argument form by arity.
    template <typename T>
    inline SyevxResolvedRange syevx_resolve_range(int64_t n,
                                                  size_t neigs,
                                                  const SyevxParams<T>& params) {
        return syevx_resolve_range(n, neigs, params.select, params.find_largest,
                                   params.il, params.iu, params.order);
    }

    /**
     * @brief Resolves SyevxParams::method (and the BATCHLAS_SYEVX_ALGORITHM override)
     *        to a concrete, implemented algorithm.
     *
     * Never returns `Auto`: an unavailable tier falls back to its nearest implemented
     * neighbour. Deterministic in its inputs so that `syevx` and `syevx_buffer_size`
     * always agree on the choice.
     *
     * @param format Matrix format of A (sparse formats always use LOBPCG)
     * @param n Matrix dimension
     * @param neigs Number of requested eigenpairs
     * @param requested Algorithm requested via SyevxParams::method
     * @param subset_supported Whether DirectSubset is available for this T/format
     * @param jobz Whether eigenvectors are wanted -- load-bearing: the subset solver's
     *        only advantage is the narrowed back-transform.
     * @param batch_size Also load-bearing: the subset solver's reduction is parallel over
     *        the batch, so it starves at small batch and wins at large.
     * @param select Which part of the spectrum was asked for. This does not choose between
     *        algorithms; it EXCLUDES the ones that cannot answer.
     * @return SyevxAlgorithm A concrete, implemented algorithm
     * @throws std::invalid_argument for sparse input, or an explicit LOBPCG/Filtered
     *         `method`, with a non-extremal range. BATCHLAS_SYEVX_ALGORITHM naming one
     *         degrades to Direct and warns instead.
     */
    SyevxAlgorithm syevx_select_algorithm(MatrixFormat format,
                                          int64_t n,
                                          size_t neigs,
                                          SyevxAlgorithm requested,
                                          bool subset_supported,
                                          JobType jobz = JobType::EigenVectors,
                                          int64_t batch_size = 1,
                                          SyevxSelect select = SyevxSelect::Extremal);

    /**
     * @brief Resolves SyevxParams::preconditioner_type to a concrete family.
     *
     * Never returns `Auto`. Deterministic so that `syevx`, `syevx_buffer_size` and
     * `syevx_lobpcg` always agree -- the Jacobi path adds a pool allocation the sizing
     * call has to predict. Legality is checked by `syevx`, not here.
     *
     * @param requested SyevxParams::preconditioner_type
     * @param iluk_configured Whether an ILU(k) factor was supplied or requested
     * @param find_largest An environment-supplied default that is illegal for the
     *        requested end degrades to `None`; an explicit request throws.
     */
    SyevxPreconditioner syevx_select_preconditioner(SyevxPreconditioner requested,
                                                    bool iluk_configured,
                                                    bool find_largest);

    /**
     * @brief Partial eigensolve by full decomposition followed by selection.
     *
     * Runs `syev` on a private copy of A (A is not modified) and extracts the requested
     * part of the spectrum. Dense input only, but every scalar type and every SyevxSelect
     * range, which is what makes it the universal fallback. Ordering: descending when
     * `params.find_largest` for the default `Extremal` selection, otherwise `params.order`.
     *
     * @param W Eigenvalue output. `neigs` is a CAPACITY: entries past `min(m[b], neigs)`
     *        are left untouched. The stride is always `neigs`.
     * @param V Eigenvector output. Columns past `min(m[b], neigs)` are written as EXACTLY
     *        ZERO -- not left untouched, unlike `W` -- so this path and
     *        `syevx_direct_subset` answer the same question the same way.
     * @param m Per-item count in the requested range, or an empty span to not report it.
     *        `m[b] > neigs` is the truncation signal; the LOWEST `neigs` are kept.
     */
    template <Backend B, typename T, MatrixFormat MFormat>
    Event syevx_direct(Queue& ctx,
                const MatrixView<T, MFormat>& A,
                Span<typename base_type<T>::type> W,
                Span<int32_t> m,
                size_t neigs,
                Span<std::byte> workspace,
                JobType jobz,
                const MatrixView<T, MatrixFormat::Dense>& V,
                const SyevxParams<T>& params);

    /**
     * @brief `syevx_direct` without the `m` output. Legal for `Extremal` and `Index`; a
     *        `Value` range needs the form above. Distinguished by ARITY, so neither the
     *        parameter-pack trap nor the trailing-`{}` trap can fire between them.
     */
    template <Backend B, typename T, MatrixFormat MFormat>
    inline Event syevx_direct(Queue& ctx,
                const MatrixView<T, MFormat>& A,
                Span<typename base_type<T>::type> W,
                size_t neigs,
                Span<std::byte> workspace,
                JobType jobz,
                const MatrixView<T, MatrixFormat::Dense>& V,
                const SyevxParams<T>& params) {
        return syevx_direct<B, T, MFormat>(ctx, A, W, Span<int32_t>(), neigs, workspace,
                                           jobz, V, params);
    }

    // No `m` parameter: sizing writes no counts, and this is range-independent anyway.
    template <Backend B, typename T, MatrixFormat MFormat>
    size_t syevx_direct_buffer_size(Queue& ctx,
                const MatrixView<T, MFormat>& A,
                Span<typename base_type<T>::type> W,
                size_t neigs,
                JobType jobz,
                const MatrixView<T, MatrixFormat::Dense>& V,
                const SyevxParams<T>& params);

    /**
     * @brief Partial eigensolve by two-stage reduction plus a subset tridiagonal
     *        solve (`stebz` + `stein`) and a back-transform narrowed to the
     *
     * Real scalar types and dense input only; `syevx` routes complex or sparse input
     * elsewhere. Supports every `SyevxSelect` range. Zero capacity is rejected (stein
     * requires k >= 1).
     *
     * A descending reversal is applied last, in the finalize kernel: `stein`'s cluster
     * detection walks consecutive eigenvalues and requires ascending input, so `stebz` is
     * never asked for descending mid-chain.
     *
     * @param W Eigenvalue output. `neigs` is a CAPACITY: entries past `min(m[b], neigs)`
     *        are left untouched. The stride is always `neigs`.
     * @param V Eigenvector output. Columns past `min(m[b], neigs)` are written as EXACTLY
     *        ZERO by `stein`, and the back-transforms preserve that.
     * @param m Per-item count in the requested range, or an empty span to not report it.
     *        `m[b] > neigs` is the truncation signal; the LOWEST `neigs` are kept.
     */
    template <Backend B, typename T, MatrixFormat MFormat>
    Event syevx_direct_subset(Queue& ctx,
                const MatrixView<T, MFormat>& A,
                Span<typename base_type<T>::type> W,
                Span<int32_t> m,
                size_t neigs,
                Span<std::byte> workspace,
                JobType jobz,
                const MatrixView<T, MatrixFormat::Dense>& V,
                const SyevxParams<T>& params);

    /**
     * @brief `syevx_direct_subset` without the `m` output; same reason and same
     *        arity-based disambiguation as the `syevx_direct` form above.
     */
    template <Backend B, typename T, MatrixFormat MFormat>
    inline Event syevx_direct_subset(Queue& ctx,
                const MatrixView<T, MFormat>& A,
                Span<typename base_type<T>::type> W,
                size_t neigs,
                Span<std::byte> workspace,
                JobType jobz,
                const MatrixView<T, MatrixFormat::Dense>& V,
                const SyevxParams<T>& params) {
        return syevx_direct_subset<B, T, MFormat>(ctx, A, W, Span<int32_t>(), neigs, workspace,
                                                  jobz, V, params);
    }

    // Sizing writes no counts, but it is NOT range-independent: a Value range needs room
    // for up to n eigenvalues per item, so the sizes come from `syevx_resolve_range`.
    template <Backend B, typename T, MatrixFormat MFormat>
    size_t syevx_direct_subset_buffer_size(Queue& ctx,
                const MatrixView<T, MFormat>& A,
                Span<typename base_type<T>::type> W,
                size_t neigs,
                JobType jobz,
                const MatrixView<T, MatrixFormat::Dense>& V,
                const SyevxParams<T>& params);

    /**
     * @brief Whether `syevx_direct_subset` supports this scalar type and format.
     */
    template <typename T, MatrixFormat MFormat>
    inline constexpr bool syevx_direct_subset_supported() {
        return MFormat == MatrixFormat::Dense && std::is_same_v<T, typename base_type<T>::type>;
    }

    /**
     * @brief Partial eigensolve by LOBPCG. Supports dense and sparse input.
     */
    template <Backend B, typename T, MatrixFormat MFormat>
    Event syevx_lobpcg(Queue& ctx,
                const MatrixView<T, MFormat>& A,
                Span<typename base_type<T>::type> W,
                size_t neigs,
                Span<std::byte> workspace,
                JobType jobz,
                const MatrixView<T, MatrixFormat::Dense>& V,
                const SyevxParams<T>& params);

    template <Backend B, typename T, MatrixFormat MFormat>
    size_t syevx_lobpcg_buffer_size(Queue& ctx,
                const MatrixView<T, MFormat>& A,
                Span<typename base_type<T>::type> W,
                size_t neigs,
                JobType jobz,
                const MatrixView<T, MatrixFormat::Dense>& V,
                const SyevxParams<T>& params);

    /**
     * @brief Chebyshev-filtered subspace iteration.
     *
     * Applies a Chebyshev polynomial in A to a block of vectors so that the wanted
     * end of the spectrum is amplified, then does a Rayleigh-Ritz extraction. Needs
     * only matvecs -- no preconditioner and no factorization -- and works for dense
     * and CSR input.
     */
    template <Backend B, typename T, MatrixFormat MFormat>
    Event syevx_filtered(Queue& ctx,
                const MatrixView<T, MFormat>& A,
                Span<typename base_type<T>::type> W,
                size_t neigs,
                Span<std::byte> workspace,
                JobType jobz,
                const MatrixView<T, MatrixFormat::Dense>& V,
                const SyevxParams<T>& params);

    template <Backend B, typename T, MatrixFormat MFormat>
    size_t syevx_filtered_buffer_size(Queue& ctx,
                const MatrixView<T, MFormat>& A,
                Span<typename base_type<T>::type> W,
                size_t neigs,
                JobType jobz,
                const MatrixView<T, MatrixFormat::Dense>& V,
                const SyevxParams<T>& params);

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

    template <Backend B, typename T, MatrixFormat MFormat>
    inline Event lanczos(Queue& ctx,
        const Matrix<T, MFormat>& A,
        Span<typename base_type<T>::type> W,
        Span<std::byte> workspace,
        JobType jobz = JobType::NoEigenVectors,
        const LanczosParams<T>& params = LanczosParams<T>()) {
        return lanczos<B,T,MFormat>(ctx, MatrixView<T,MFormat>(A), W, workspace, jobz, MatrixView<T, MatrixFormat::Dense>(), params);
    }

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

    template <typename T>
    inline Event francis_sweep(Queue& ctx, const Vector<T>& d, const Vector<T>& e,
                               const MatrixView<std::array<T,2>, MatrixFormat::Dense>& givens_rotations = {},
                               size_t n_sweeps = 1,
                               T zero_threshold = std::numeric_limits<T>::epsilon()) {
        return francis_sweep<T>(ctx, static_cast<VectorView<T>>(d), static_cast<VectorView<T>>(e), givens_rotations, n_sweeps, zero_threshold);
    }

    // Vector-typed parameters in the tridiagonal group below (`stebz`, `stein`, `steqr`,
    // `steqr_cta`, `stedc`): `VectorView<T>` for anything that is one vector per batch
    // item, because it carries inc/stride/batch_size; `Span<...>` for flat per-item arrays
    // (`m`, `counts`) and byte workspaces. A `VectorView` demoted to a `Span` silently
    // drops the stride, so the two do not interconvert. See docs/cpp-api.md.

    /**
     * @brief How a subset of the spectrum is selected.
     */
    enum class EigenRangeType {
        All,    // Every eigenvalue
        Index,  // Eigenvalues il..iu inclusive, 0-based, in ascending order
        Value   // Eigenvalues in the half-open interval (vl, vu]
    };

    /**
     * @brief Parameters for `stebz` (bisection on a symmetric tridiagonal matrix).
     */
    template <typename T>
    struct StebzParams {
        EigenRangeType range = EigenRangeType::All;
        int64_t il = 0;    // First wanted index (0-based, inclusive), range == Index
        int64_t iu = -1;   // Last wanted index (0-based, inclusive), range == Index
        T vl = T(0);       // Lower bound (exclusive), range == Value
        T vu = T(0);       // Upper bound (inclusive), range == Value
        // Absolute tolerance on each eigenvalue. Non-positive means eps * ||T||,
        // i.e. full working precision.
        T abstol = T(0);
        SortOrder order = SortOrder::Ascending;
        // Safety cap on bisection steps per eigenvalue; the loop also exits on
        // interval convergence.
        int32_t max_iterations = 128;
    };

    /**
     * @brief Computes selected eigenvalues of a batch of symmetric tridiagonal
     *        matrices by bisection on Sturm sequence sign counts.
     *
     * One work-item bisects one eigenvalue, so a subset costs proportionally less than
     * QR iteration or divide-and-conquer. Eigenvalues only; use `stein` for the vectors.
     *
     * @param ctx Execution context/device queue
     * @param d Diagonal, n entries per batch item
     * @param e Off-diagonal, n-1 entries per batch item
     * @param w Output eigenvalues; must hold at least the number selected
     * @param m Output count of eigenvalues found, per batch item
     * @param ws Pre-allocated workspace buffer
     * @param params Selection range, tolerance and ordering
     * @return Event Event to track operation completion
     */
    template <Backend B, typename T>
    Event stebz(Queue& ctx,
                const VectorView<T>& d,
                const VectorView<T>& e,
                const VectorView<T>& w,
                Span<int32_t> m,
                const Span<std::byte>& ws,
                StebzParams<T> params = StebzParams<T>());

    /**
     * @brief Required workspace size, in bytes, for `stebz`.
     *
     * `params` is REQUIRED, not defaulted: it is the only argument carrying `T`, and
     * defaulting it makes the queue-deducing dispatch wrapper's requires-clause fail and
     * drop the overload, with no hint that `T` was undeducible. Pass `StebzParams<T>{}`.
     */
    template <Backend B, typename T>
    size_t stebz_buffer_size(Queue& ctx,
                             size_t n,
                             size_t batch_size,
                             StebzParams<T> params);

    template <Backend B, typename T>
    inline Event stebz(Queue& ctx,
                       const Vector<T>& d,
                       const Vector<T>& e,
                       const Vector<T>& w,
                       Span<int32_t> m,
                       const Span<std::byte>& ws,
                       StebzParams<T> params = StebzParams<T>()) {
        return stebz<B, T>(ctx,
                           static_cast<VectorView<T>>(d),
                           static_cast<VectorView<T>>(e),
                           static_cast<VectorView<T>>(w),
                           m, ws, params);
    }

    /**
     * @brief Parameters for `stein` (inverse iteration on a symmetric tridiagonal).
     */
    template <typename T>
    struct SteinParams {
        // Two or three steps suffice for eigenvalues accurate to working precision.
        int32_t max_iterations = 3;
        // Eigenvalues closer than ortho_threshold * ||T|| form one cluster and have their
        // vectors explicitly reorthogonalized (LAPACK dstein uses 1e-3).
        T ortho_threshold = T(1e-3);
        uint32_t seed = 0x5eed1234u;
    };

    /**
     * @brief Computes eigenvectors of a batch of symmetric tridiagonal matrices by
     *        inverse iteration, given previously computed eigenvalues.
     *
     * Pairs with `stebz`. Each vector solves (T - lambda*I) x = b with a tridiagonal
     * LU factorization (partial pivoting) from a pseudo-random start; clustered
     * eigenvalues have their vectors reorthogonalized afterwards.
     *
     * @param ctx Execution context/device queue
     * @param d Diagonal, n entries per batch item
     * @param e Off-diagonal, n-1 entries per batch item
     * @param w Eigenvalues, k per batch item, in ascending order
     * @param k Number of eigenvectors to compute
     * @param Z Output eigenvectors, n x k per batch item, columns matching w
     * @param ws Pre-allocated workspace buffer
     * @param params Iteration count and clustering threshold
     * @return Event Event to track operation completion
     */
    template <Backend B, typename T>
    Event stein(Queue& ctx,
                const VectorView<T>& d,
                const VectorView<T>& e,
                const VectorView<T>& w,
                size_t k,
                const MatrixView<T, MatrixFormat::Dense>& Z,
                const Span<std::byte>& ws,
                SteinParams<T> params = SteinParams<T>());

    /**
     * @brief Sentinel meaning "every batch item has all `k` eigenvalues valid".
     *
     * Spell this rather than a bare `{}` at `counts`: a bare `{}` in an argument position
     * two overloads both accept has previously selected the wrong overload in this
     * codebase and silently changed results.
     */
    inline constexpr Span<const int32_t> stein_all_counts{};

    /**
     * @brief `stein` with a per-batch-item count of valid eigenvalues.
     *
     * `k` is a *capacity* -- the columns of `Z` and entries of `w` per item -- while
     * `counts[b]` is the number of leading entries of item `b`'s `w` that are real
     * eigenvalues; the rest hold whatever the workspace last contained. Inverse iteration
     * is not run on those invalid shifts and the cluster walk stops at `counts[b]`, so a
     * real eigenvalue is never grouped with a garbage neighbour.
     *
     * Columns `[counts[b], k)` of `Z` are **written as exactly zero**, not left untouched,
     * so callers may run a uniform-width back-transform over all `k` columns.
     *
     * `counts` is read on the device, so it may be the `m` span `stebz` just wrote.
     *
     * @param counts Per-item valid prefix length, at least `d.batch_size()` entries;
     *               empty (or `stein_all_counts`) means "all `k`". Clamped to `[0, k]`.
     */
    template <Backend B, typename T>
    Event stein(Queue& ctx,
                const VectorView<T>& d,
                const VectorView<T>& e,
                const VectorView<T>& w,
                size_t k,
                Span<const int32_t> counts,
                const MatrixView<T, MatrixFormat::Dense>& Z,
                const Span<std::byte>& ws,
                SteinParams<T> params = SteinParams<T>());

    // Forwarding overloads taking owning Vectors. These cannot collide with the
    // primaries: deduction does not consider the implicit `VectorView(const Vector<T>&)`
    // conversion, so exactly one set is ever viable.
    template <Backend B, typename T>
    inline Event stein(Queue& ctx,
                       const Vector<T>& d,
                       const Vector<T>& e,
                       const Vector<T>& w,
                       size_t k,
                       const MatrixView<T, MatrixFormat::Dense>& Z,
                       const Span<std::byte>& ws,
                       SteinParams<T> params = SteinParams<T>()) {
        return stein<B, T>(ctx,
                           static_cast<VectorView<T>>(d),
                           static_cast<VectorView<T>>(e),
                           static_cast<VectorView<T>>(w),
                           k, Z, ws, params);
    }

    template <Backend B, typename T>
    inline Event stein(Queue& ctx,
                       const Vector<T>& d,
                       const Vector<T>& e,
                       const Vector<T>& w,
                       size_t k,
                       Span<const int32_t> counts,
                       const MatrixView<T, MatrixFormat::Dense>& Z,
                       const Span<std::byte>& ws,
                       SteinParams<T> params = SteinParams<T>()) {
        return stein<B, T>(ctx,
                           static_cast<VectorView<T>>(d),
                           static_cast<VectorView<T>>(e),
                           static_cast<VectorView<T>>(w),
                           k, counts, Z, ws, params);
    }

    /**
     * @brief Required workspace size, in bytes, for `stein`.
     *
     * Sizes on the capacity `k` regardless of any per-item `counts`, so the `counts`
     * overload needs no separate sizing entry point. `params` is REQUIRED for the same
     * reason as `stebz_buffer_size`: it is the only argument carrying `T`.
     */
    template <Backend B, typename T>
    size_t stein_buffer_size(Queue& ctx,
                             size_t n,
                             size_t k,
                             size_t batch_size,
                             SteinParams<T> params);

    enum class SteqrShiftStrategy {
        // LAPACK-style implicit shift (stable formulation used by dsteqr-style iterations).
        Lapack = 0,
        // Wilkinson shift computed from the relevant 2x2 block.
        Wilkinson = 1,
    };

    enum class SteqrUpdateScheme {
        // Parlett-Gray style recurrence (current default).
        PG = 0,
        // Explicit similarity update mirroring steqr.cc bulge-chasing math.
        EXP = 1,
    };

    template <typename T>
    struct SteqrParams {
        // Rotations are applied in blocks of this size; larger means excess FLOPs but
        // better memory reuse. 1 fully serializes them.
        size_t block_size = 32;
        // Cap on Francis QR sweeps; 2-3 typically suffice per eigenvalue.
        size_t max_sweeps = 50; 
        T zero_threshold = std::numeric_limits<T>::epsilon(); 
        // If false, the eigenvector matrix is set to Identity and rotations applied to it.
        bool back_transform = false; 
        bool block_rotations = false;
        bool sort = true;
        bool transpose_working_vectors = true;
        SortOrder sort_order = SortOrder::Ascending;

        // CTA STEQR only: multiplies the baseline work-group size, LCM(N, sub_group_size).
        size_t cta_wg_size_multiplier = 1;

        // CTA STEQR only: shift strategy for the implicit QR/QL steps.
        SteqrShiftStrategy cta_shift_strategy = SteqrShiftStrategy::Lapack;
        // CTA STEQR only: update scheme for the implicit QR/QL steps.
        // CTA STEQR only: select the update scheme used in implicit QR/QL steps.
        SteqrUpdateScheme cta_update_scheme = SteqrUpdateScheme::EXP;
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
  
    template <Backend B, typename T>
    inline Event steqr(Queue& ctx, const Vector<T>& d, const Vector<T>& e,
                       const Vector<T>& eigenvalues, const Span<std::byte>& ws,
                       JobType jobz = JobType::NoEigenVectors,
                       SteqrParams<T> params = SteqrParams<T>()) {
        return steqr<B, T>(ctx,
                        static_cast<VectorView<T>>(d),
                        static_cast<VectorView<T>>(e),
                        static_cast<VectorView<T>>(eigenvalues),
                        ws,
                        jobz,
                        params,
                        MatrixView<T, MatrixFormat::Dense>());
    }

    template <Backend B, typename T>
    inline Event steqr(Queue& ctx, const Vector<T>& d, const Vector<T>& e,
                       const Vector<T>& eigenvalues, const Span<std::byte>& ws,
                       JobType jobz,
                       SteqrParams<T> params,
                       const Matrix<T, MatrixFormat::Dense>& eigvects) {
        return steqr<B, T>(ctx,
                        static_cast<VectorView<T>>(d),
                        static_cast<VectorView<T>>(e),
                        static_cast<VectorView<T>>(eigenvalues),
                        ws,
                        jobz,
                        params,
                        MatrixView<T, MatrixFormat::Dense>(eigvects));
    }

    template <Backend B, typename T>
    inline Event steqr_cta(Queue& ctx, const Vector<T>& d, const Vector<T>& e,
                           const Vector<T>& eigenvalues, const Span<std::byte>& ws,
                           JobType jobz = JobType::NoEigenVectors,
                           SteqrParams<T> params = SteqrParams<T>()) {
        return steqr_cta<B, T>(ctx,
                               static_cast<VectorView<T>>(d),
                               static_cast<VectorView<T>>(e),
                               static_cast<VectorView<T>>(eigenvalues),
                               ws,
                               jobz,
                               params,
                               MatrixView<T, MatrixFormat::Dense>());
    }

    template <Backend B, typename T>
    inline Event steqr_cta(Queue& ctx, const Vector<T>& d, const Vector<T>& e,
                           const Vector<T>& eigenvalues, const Span<std::byte>& ws,
                           JobType jobz,
                           SteqrParams<T> params,
                           const Matrix<T, MatrixFormat::Dense>& eigvects) {
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


    // === CTA small-matrix extensions (n <= 32) ===

    enum class OrmqCtaFactorization {
        QR,
        QL,
    };

    /**
     * @brief CTA-optimized symmetric tridiagonal reduction for very small matrices.
     *
     * Overwrites A with the tridiagonal and reflector storage (SYTD2-style), and returns
     * the diagonal/off-diagonal in (d,e) plus reflector scalars in tau. Intended for
     * n <= 32; `ws` is unused but kept for API compatibility.
     */
    template <Backend B, typename T>
    Event sytrd_cta(Queue& ctx,
                    const MatrixView<T, MatrixFormat::Dense>& a_in,
                    const VectorView<T>& d_out,
                    const VectorView<T>& e_out,
                    const VectorView<T>& tau_out,
                    Uplo uplo,
                    const Span<std::byte>& ws,
                    size_t cta_wg_size_multiplier = 1);

    /**
     * @brief LATRD-like panel factorization used by blocked SYTRD (Lower only).
     *
     * Computes Householder vectors (stored in A) and W for a block of columns starting at
     * j0, in the same SYTD2-style reflector layout used by sytrd. `Uplo::Lower` only;
     * writes only the first `ib` columns of W (W is treated as n x nb).
     */
    template <Backend B, typename T>
    Event latrd_lower_panel(Queue& ctx,
                            const MatrixView<T, MatrixFormat::Dense>& a_in,
                            const VectorView<T>& e_out,
                            const VectorView<T>& tau_out,
                            const MatrixView<T, MatrixFormat::Dense>& w_in,
                            int32_t j0,
                            int32_t ib,
                            int32_t wg_hint = 0,
                            bool fuse_trailing_update = false);

    /**
     * @brief LATRD-like panel factorization (Lower only), view-based overload.
     *
     * Pass pre-sliced views instead of (j0, ib):
     *  - a_panel = A({j0, SliceEnd()}, {j0, SliceEnd()})  (must be square)
     *  - e_panel = E(Slice(j0, j0 + ib))
     *  - tau_panel = TAU(Slice(j0, j0 + ib))
     *  - w_panel = Wmat({j0, SliceEnd()}, {0, ib})
     *
     * e_panel/tau_panel size must match w_panel.cols().
     */
    template <Backend B, typename T>
    Event latrd_lower_panel(Queue& ctx,
                            const MatrixView<T, MatrixFormat::Dense>& a_panel_in,
                            const VectorView<T>& e_panel_out,
                            const VectorView<T>& tau_panel_out,
                            const MatrixView<T, MatrixFormat::Dense>& w_panel_in,
                            int32_t wg_hint = 0,
                            bool fuse_trailing_update = false);

    /**
     * @brief Blocked symmetric/Hermitian tridiagonal reduction for medium/large matrices.
     *
     * Overwrites A with the tridiagonal and reflector storage (SYTD2-style), and returns
     * the diagonal/off-diagonal in (d,e) plus reflector scalars in tau. Intended for
     * n > 32: blocked LATRD-style panel plus BLAS-3 trailing update, in-order queue only.
     */
    template <Backend B, typename T>
    Event sytrd_blocked(Queue& ctx,
                        const MatrixView<T, MatrixFormat::Dense>& a_in,
                        const VectorView<T>& d_out,
                        const VectorView<T>& e_out,
                        const VectorView<T>& tau_out,
                        Uplo uplo,
                        const Span<std::byte>& ws,
                        int32_t block_size = tuning::SYTRD_BLOCK_SIZE_MEDIUM);

    template <Backend B, typename T>
    size_t sytrd_blocked_buffer_size(Queue& ctx,
                                     const MatrixView<T, MatrixFormat::Dense>& a,
                                     const VectorView<T>& d,
                                     const VectorView<T>& e,
                                     const VectorView<T>& tau,
                                     Uplo uplo,
                                     int32_t block_size = tuning::SYTRD_BLOCK_SIZE_MEDIUM);

    /**
     * @brief First stage of two-stage reduction: symmetric/Hermitian dense -> band (SY2SB).
     *
     * Analogous to LAPACK xSYTRD_SY2SB: overwrites `A` with Householder reflector storage
     * and writes the band matrix into `AB`. `tau_out` has size (n-kd). Requires an
     * in-order queue; only `Uplo::Lower` is implemented.
     *
     * Band storage (AB), shape (kd+1) x n:
     *  - `Uplo::Lower`: AB(1+i-j,j) = A(i,j) for j<=i<=min(n,j+kd).
     *  - `Uplo::Upper`: AB(kd+1+i-j,j) = A(i,j) for max(1,j-kd)<=i<=j.
     */
    template <Backend B, typename T>
    Event sytrd_sy2sb(Queue& ctx,
                      const MatrixView<T, MatrixFormat::Dense>& a_in,
                      const MatrixView<T, MatrixFormat::Dense>& ab_out,
                      const VectorView<T>& tau_out,
                      Uplo uplo,
                      int32_t kd,
                      const Span<std::byte>& ws);

    template <Backend B, typename T>
    size_t sytrd_sy2sb_buffer_size(Queue& ctx,
                                   const MatrixView<T, MatrixFormat::Dense>& a_in,
                                   const MatrixView<T, MatrixFormat::Dense>& ab_out,
                                   const VectorView<T>& tau_out,
                                   Uplo uplo,
                                   int32_t kd);

    /**
     * @brief Second stage of two-stage reduction: symmetric/Hermitian band -> tridiagonal (SB2ST/HB2ST).
     *
     * Analogous to LAPACK xSB2ST/xHB2ST (bulge chasing) with VECT='N': reduces the band
     * matrix to tridiagonal without forming eigenvectors. Requires an in-order queue;
     * only `Uplo::Lower` is implemented.
     *
     * Band storage (AB), shape (kd+1) x n:
     *  - `Uplo::Lower`: AB(1+i-j,j) = A(i,j) for j<=i<=min(n,j+kd).
     *  - `Uplo::Upper`: AB(kd+1+i-j,j) = A(i,j) for max(1,j-kd)<=i<=j.
     *
     * `d_out` (size n) and `e_out` (size n-1) are always real-valued; `tau_out` (size n-1)
     * is unused downstream and is set to 0.
     */
    template <Backend B, typename T>
    Event sytrd_sb2st(Queue& ctx,
                      const MatrixView<T, MatrixFormat::Dense>& ab_in,
                      const VectorView<typename base_type<T>::type>& d_out,
                      const VectorView<typename base_type<T>::type>& e_out,
                      const VectorView<T>& tau_out,
                      Uplo uplo,
                      int32_t kd,
                      const Span<std::byte>& ws,
                      int32_t block_size);

    template <Backend B, typename T>
    size_t sytrd_sb2st_buffer_size(Queue& ctx,
                                   const MatrixView<T, MatrixFormat::Dense>& ab_in,
                                   const VectorView<typename base_type<T>::type>& d_out,
                                   const VectorView<typename base_type<T>::type>& e_out,
                                   const VectorView<T>& tau_out,
                                   Uplo uplo,
                                   int32_t kd,
                                   int32_t block_size);

    /**
     * @brief Symmetric/Hermitian band -> tridiagonal reduction (BANDR1-style).
     *
     * Requires an in-order queue; only `Uplo::Lower` is implemented. Produces real-valued
     * (d,e); for complex input `e_out` is the magnitude of the (possibly phased)
     * subdiagonal. `tau_out` is set to 0 (VECT='N' style).
     */

    struct SytrdBandReductionParams {
        // Diagonals eliminated per sweep (Algorithm 2: d^(i)); 0 means the default
        // schedule. If sweeps exceed the sequence length, the last value is reused.
        std::vector<int32_t> d_seq{0};

        // Block size per sweep (Algorithm 2: nb^(i)).
        // If sweeps exceed sequence length, the last value is reused.
        std::vector<int32_t> block_size_seq{32};

        // Maximum number of sweeps. max_sweeps < 0 means use the implementation default.
        int32_t max_sweeps = -1;

        // Debug/testing: chase steps for `sytrd_band_reduction_single_step`;
        // <= 0 means exactly one step.
        int32_t max_steps = 1;

        // Working band semibandwidth. kd_work <= 0 means use the implementation default.
        int32_t kd_work = 0;
    };

    template <Backend B, typename T>
    Event sytrd_band_reduction(Queue& ctx,
                               const MatrixView<T, MatrixFormat::Dense>& ab_in,
                               const VectorView<typename base_type<T>::type>& d_out,
                               const VectorView<typename base_type<T>::type>& e_out,
                               const VectorView<T>& tau_out,
                               Uplo uplo,
                               int32_t kd,
                               const Span<std::byte>& ws,
                               int32_t block_size);

    // Overload exposing schedule parameters (Python-style flexibility).
    template <Backend B, typename T>
    Event sytrd_band_reduction(Queue& ctx,
                               const MatrixView<T, MatrixFormat::Dense>& ab_in,
                               const VectorView<typename base_type<T>::type>& d_out,
                               const VectorView<typename base_type<T>::type>& e_out,
                               const VectorView<T>& tau_out,
                               Uplo uplo,
                               int32_t kd,
                               const Span<std::byte>& ws,
                               SytrdBandReductionParams params);

    template <Backend B, typename T>
    Event sytrd_band_reduction_single_step(Queue& ctx,
                                           const MatrixView<T, MatrixFormat::Dense>& ab_in,
                                           const MatrixView<T, MatrixFormat::Dense>& abw_out,
                                           Uplo uplo,
                                           int32_t kd,
                                           const Span<std::byte>& ws,
                                           SytrdBandReductionParams params);

    template <Backend B, typename T>
    size_t sytrd_band_reduction_buffer_size(Queue& ctx,
                                            const MatrixView<T, MatrixFormat::Dense>& ab_in,
                                            const VectorView<typename base_type<T>::type>& d_out,
                                            const VectorView<typename base_type<T>::type>& e_out,
                                            const VectorView<T>& tau_out,
                                            Uplo uplo,
                                            int32_t kd,
                                            int32_t block_size);

    // Buffer size query matching the schedule-parameter overload.
    template <Backend B, typename T>
    size_t sytrd_band_reduction_buffer_size(Queue& ctx,
                                            const MatrixView<T, MatrixFormat::Dense>& ab_in,
                                            const VectorView<typename base_type<T>::type>& d_out,
                                            const VectorView<typename base_type<T>::type>& e_out,
                                            const VectorView<T>& tau_out,
                                            Uplo uplo,
                                            int32_t kd,
                                            SytrdBandReductionParams params);

    template <Backend B, typename T>
    size_t sytrd_band_reduction_single_step_buffer_size(Queue& ctx,
                                                        const MatrixView<T, MatrixFormat::Dense>& ab_in,
                                                        const MatrixView<T, MatrixFormat::Dense>& abw_out,
                                                        Uplo uplo,
                                                        int32_t kd,
                                                        SytrdBandReductionParams params);

    /**
     * @brief Debug/testing hook: execute exactly one BANDR1 “chase step”.
     *
     * Runs a single QR-panel + similarity update (Pre/Sym/Post/Right) on a working band
     * matrix ABw: `ab_in` is lower-band with rows == kd+1, `abw_out` lower-band with rows
     * == kd_work+1 (kd_work from params). Requires an in-order queue; `Uplo::Lower` only.
     * Not a stable public API.
     */
    template <Backend B, typename T>
    Event sytrd_band_reduction_single_step(Queue& ctx,
                                           const MatrixView<T, MatrixFormat::Dense>& ab_in,
                                           const MatrixView<T, MatrixFormat::Dense>& abw_out,
                                           Uplo uplo,
                                           int32_t kd,
                                           const Span<std::byte>& ws,
                                           SytrdBandReductionParams params);

    template <Backend B, typename T>
    size_t sytrd_band_reduction_single_step_buffer_size(Queue& ctx,
                                                        const MatrixView<T, MatrixFormat::Dense>& ab_in,
                                                        const MatrixView<T, MatrixFormat::Dense>& abw_out,
                                                        Uplo uplo,
                                                        int32_t kd,
                                                        SytrdBandReductionParams params);

    // Naming aliases for Hermitian band -> tridiagonal (HB2ST) to match LAPACK terminology.
    template <Backend B, typename T>
    inline Event hetrd_hb2st(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& ab_in,
                             const VectorView<typename base_type<T>::type>& d_out,
                             const VectorView<typename base_type<T>::type>& e_out,
                             const VectorView<T>& tau_out,
                             Uplo uplo,
                             int32_t kd,
                             const Span<std::byte>& ws,
                             int32_t block_size) {
        return sytrd_sb2st<B, T>(ctx, ab_in, d_out, e_out, tau_out, uplo, kd, ws, block_size);
    }

    template <Backend B, typename T>
    inline size_t hetrd_hb2st_buffer_size(Queue& ctx,
                                          const MatrixView<T, MatrixFormat::Dense>& ab_in,
                                          const VectorView<typename base_type<T>::type>& d_out,
                                          const VectorView<typename base_type<T>::type>& e_out,
                                          const VectorView<T>& tau_out,
                                          Uplo uplo,
                                          int32_t kd,
                                          int32_t block_size) {
        return sytrd_sb2st_buffer_size<B, T>(ctx, ab_in, d_out, e_out, tau_out, uplo, kd, block_size);
    }

    /**
     * @brief CTA-optimized symmetric eigen-solver (SYEV-like) for very small matrices.
     *
     * Pipeline: sytrd_cta -> steqr_cta -> ormqx_cta (back-transform). Intended for
     * n <= 32; real symmetric and complex Hermitian. Overwrites A with eigenvectors when
     * jobz == EigenVectors. Eigenvalues ascend when SteqrParams::sort is set (default).
     */
    template <Backend B, typename T>
    Event syev_cta(Queue& ctx,
                   const MatrixView<T, MatrixFormat::Dense>& a_in,
                   Span<typename base_type<T>::type> eigenvalues,
                   JobType jobz,
                   Uplo uplo,
                   const Span<std::byte>& ws,
                   SteqrParams<T> steqr_params = SteqrParams<T>(),
                   size_t cta_wg_size_multiplier = 1);

    template <Backend B, typename T>
    size_t syev_cta_buffer_size(Queue& ctx,
                                const MatrixView<T, MatrixFormat::Dense>& a,
                                JobType jobz,
                                SteqrParams<T> steqr_params = SteqrParams<T>());

    /**
     * @brief Fused (single-kernel) variant of syev_cta for very small matrices.
     *
     * Same algorithm as syev_cta, but run end to end inside a single sub-group partition,
     * so d, e, tau, the packed reflectors and the intermediate eigenvectors never reach
     * global memory; results track syev_cta to within the reassociation implied by fusing.
     * Unlike syev_cta, A is left untouched when jobz == NoEigenVectors. Requires no global
     * workspace; `ws` is accepted for API symmetry and ignored.
     */
    template <Backend B, typename T>
    Event syev_cta_fused(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& a_in,
                         Span<typename base_type<T>::type> eigenvalues,
                         JobType jobz,
                         Uplo uplo,
                         const Span<std::byte>& ws = Span<std::byte>(),
                         SteqrParams<T> steqr_params = SteqrParams<T>(),
                         size_t cta_wg_size_multiplier = 1);

    template <Backend B, typename T>
    size_t syev_cta_fused_buffer_size(Queue& ctx,
                                      const MatrixView<T, MatrixFormat::Dense>& a,
                                      JobType jobz,
                                      SteqrParams<T> steqr_params = SteqrParams<T>());

    template <typename T>
    struct JacobiParams {
        using Real = typename base_type<T>::type;

        // A rotation is applied to pivot pair (p,q) only when
        //     |a_pq| > tol_multiplier * n * eps * sqrt(|a_pp| * |a_qq|)
        // The relative form, rather than the classical absolute test against max|a_kl|, is
        // what yields the high relative accuracy; raising it trades accuracy for sweeps.
        Real tol_multiplier = Real(1);

        // Cap on cyclic sweeps; a safety net for pathological inputs.
        size_t max_sweeps = 30;

        bool sort = true;
        SortOrder sort_order = SortOrder::Ascending;

        // Multiplies the baseline work-group size, LCM(P, sub_group_size) for the
        // compile-time partition width P chosen from n. Clamped by device limits.
        size_t cta_wg_size_multiplier = 1;
    };

    /**
     * @brief CTA-optimized Jacobi symmetric/Hermitian eigen-solver for very small matrices.
     *
     * Cyclic two-sided Jacobi run entirely inside a single sub-group partition: A and the
     * eigenvector accumulator Z stay in local memory for the whole solve, so one kernel
     * launch performs the complete eigendecomposition.
     *
     * Relative to syev_cta it trades throughput for high *relative* accuracy on graded or
     * badly scaled input. The underlying theorem (Demmel & Veselic, SIMAX 13(4), 1992) is
     * proved for symmetric positive definite input; indefinite matrices are solved
     * correctly but do not inherit the bound.
     *
     * Intended for n <= 32; real symmetric and complex Hermitian. Overwrites A with
     * eigenvectors when jobz == EigenVectors and leaves it untouched otherwise.
     * Eigenvalues ascend when JacobiParams::sort is set (default). Requires no global
     * workspace; `ws` is accepted for API symmetry and ignored.
     */
    template <Backend B, typename T>
    Event syev_jacobi_cta(Queue& ctx,
                          const MatrixView<T, MatrixFormat::Dense>& a_in,
                          Span<typename base_type<T>::type> eigenvalues,
                          JobType jobz,
                          Uplo uplo,
                          const Span<std::byte>& ws = Span<std::byte>(),
                          JacobiParams<T> params = JacobiParams<T>());

    template <Backend B, typename T>
    size_t syev_jacobi_cta_buffer_size(Queue& ctx,
                                       const MatrixView<T, MatrixFormat::Dense>& a,
                                       JobType jobz,
                                       JacobiParams<T> params = JacobiParams<T>());

    /**
     * @brief Blocked symmetric/Hermitian eigen-solver (SYEV-like) for medium/large matrices.
     *
     * Pipeline: sytrd_blocked -> stedc -> ormqr_blocked (back-transform). Overwrites A
     * with eigenvectors when jobz == EigenVectors. Only Uplo::Lower is supported.
     */
    template <Backend B, typename T>
    Event syev_blocked(Queue& ctx,
                       const MatrixView<T, MatrixFormat::Dense>& a_in,
                       Span<typename base_type<T>::type> eigenvalues,
                       JobType jobz,
                       Uplo uplo,
                       const Span<std::byte>& ws,
                       StedcParams<typename base_type<T>::type> stedc_params = StedcParams<typename base_type<T>::type>());

    template <Backend B, typename T>
    size_t syev_blocked_buffer_size(Queue& ctx,
                                    const MatrixView<T, MatrixFormat::Dense>& a,
                                    JobType jobz,
                                    Uplo uplo,
                                    StedcParams<typename base_type<T>::type> stedc_params = StedcParams<typename base_type<T>::type>());

    /**
     * @brief Two-stage symmetric/Hermitian eigen-solver (SYEV-like), opt-in path.
     *
     * Pipeline: sytrd_sy2sb (dense -> band) -> sytrd_sb2st (band -> tridiagonal) -> stedc.
     * Eigenvectors are recovered with explicit phase/sign recovery and a reflector
     * back-transform. Both modes use a real band width (choose_two_stage_kd, overridable
     * via BATCHLAS_SYEV_TWO_STAGE_KD). Only Uplo::Lower is supported.
     */
    template <Backend B, typename T>
    Event syev_two_stage(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& a_in,
                         Span<typename base_type<T>::type> eigenvalues,
                         JobType jobz,
                         Uplo uplo,
                         const Span<std::byte>& ws,
                         StedcParams<typename base_type<T>::type> stedc_params = StedcParams<typename base_type<T>::type>());

    template <Backend B, typename T>
    size_t syev_two_stage_buffer_size(Queue& ctx,
                                      const MatrixView<T, MatrixFormat::Dense>& a,
                                      JobType jobz,
                                      Uplo uplo,
                                      StedcParams<typename base_type<T>::type> stedc_params = StedcParams<typename base_type<T>::type>());

    /**
     * @brief Unblocked GEBRD-like reduction to real bidiagonal form.
     *
     * Reduces each square matrix A to bidiagonal form in-place while returning
     * bidiagonal coefficients (d,e) and Householder scalars (tauq,taup).
     */
    template <Backend B, typename T>
    Event gebrd_unblocked(Queue& ctx,
                          const MatrixView<T, MatrixFormat::Dense>& a,
                          const VectorView<typename base_type<T>::type>& d,
                          const VectorView<typename base_type<T>::type>& e,
                          const VectorView<T>& tauq,
                          const VectorView<T>& taup);

    /**
     * @brief CTA-parallel small-matrix GEBRD for real square matrices.
     *
     * Intended for very small dense problems (`1 <= n <= 32`) where one CTA can
     * cooperatively reduce one matrix to upper bidiagonal form.
     */
    template <Backend B, typename T>
    Event gebrd_cta(Queue& ctx,
                    const MatrixView<T, MatrixFormat::Dense>& a,
                    const VectorView<typename base_type<T>::type>& d,
                    const VectorView<typename base_type<T>::type>& e,
                    const VectorView<T>& tauq,
                    const VectorView<T>& taup,
                    size_t cta_wg_size_multiplier = 1);

    /**
     * @brief Blocked GEBRD for real square dense matrices.
     *
     * Uses a DLABRD-style blocked panel factorization plus GEMM trailing updates.
     */
    template <Backend B, typename T>
    Event gebrd_blocked(Queue& ctx,
                        const MatrixView<T, MatrixFormat::Dense>& a,
                        const VectorView<typename base_type<T>::type>& d,
                        const VectorView<typename base_type<T>::type>& e,
                        const VectorView<T>& tauq,
                        const VectorView<T>& taup,
                        const Span<std::byte>& ws,
                        int32_t block_size = 16);

    template <Backend B, typename T>
    size_t gebrd_blocked_buffer_size(Queue& ctx,
                                     const MatrixView<T, MatrixFormat::Dense>& a,
                                     const VectorView<typename base_type<T>::type>& d,
                                     const VectorView<typename base_type<T>::type>& e,
                                     const VectorView<T>& tauq,
                                     const VectorView<T>& taup,
                                     int32_t block_size = 16);

    /**
     * @brief Bidiagonal QR iteration for a real upper bidiagonal matrix.
     *
     * Computes singular values from bidiagonal coefficients `(d,e)`. The matrix overload
     * additionally accumulates the alternating right and left Givens rotations into `vh`
     * and `u`, matching LAPACK's `BDSQR` contract of returning `P^T * vh` and `u * Q`.
     */
    template <Backend B, typename T>
    Event bdsqr(Queue& ctx,
                const VectorView<T>& d,
                const VectorView<T>& e,
                Span<T> singular_values_out,
                const Span<std::byte>& ws,
                bool sort_desc = true);

    template <Backend B, typename T>
    Event bdsqr(Queue& ctx,
                const VectorView<T>& d,
                const VectorView<T>& e,
                Span<T> singular_values_out,
                const Span<std::byte>& ws,
                const MatrixView<T, MatrixFormat::Dense>& u,
                const MatrixView<T, MatrixFormat::Dense>& vh,
                bool sort_desc = true);

    template <typename T>
    size_t bdsqr_buffer_size(Queue& ctx,
                             const VectorView<T>& d,
                             const VectorView<T>& e,
                             Span<T> singular_values_out);

    template <typename T>
    inline size_t bdsqr_buffer_size(Queue& ctx,
                                    const VectorView<T>& d,
                                    const VectorView<T>& e,
                                    Span<T> singular_values_out,
                                    const MatrixView<T, MatrixFormat::Dense>& u,
                                    const MatrixView<T, MatrixFormat::Dense>& vh) {
        static_cast<void>(u);
        static_cast<void>(vh);
        return bdsqr_buffer_size(ctx, d, e, singular_values_out);
    }

    /**
     * @brief Bidiagonal divide-and-conquer SVD for a real upper bidiagonal matrix.
     *
     * Same problem as `bdsqr`, but parallel instead of a serial Golub-Kahan sweep: it
     * reduces the bidiagonal SVD to the symmetric tridiagonal eigenproblem of the
     * interleaved Golub-Kahan form (order `2n`) and hands that to `stedc`. Nothing is
     * squared, so the error stays proportional to `kappa`, not `kappa^2`.
     *
     * Unlike `bdsqr`, which *accumulates* into whatever it is handed (`u <- u*Q`), `bdsdc`
     * *writes* the leading `n x n` block of `u` and of `vh` and leaves the rest of those
     * views untouched -- seed them with the identity if the trailing columns matter.
     *
     * Workspace is dominated by a `2n x 2n` eigenvector matrix per batch item.
     */
    template <Backend B, typename T>
    Event bdsdc(Queue& ctx,
                const VectorView<T>& d,
                const VectorView<T>& e,
                Span<T> singular_values_out,
                const Span<std::byte>& ws,
                bool sort_desc = true);

    template <Backend B, typename T>
    Event bdsdc(Queue& ctx,
                const VectorView<T>& d,
                const VectorView<T>& e,
                Span<T> singular_values_out,
                const Span<std::byte>& ws,
                const MatrixView<T, MatrixFormat::Dense>& u,
                const MatrixView<T, MatrixFormat::Dense>& vh,
                bool sort_desc = true);

    template <Backend B, typename T>
    size_t bdsdc_buffer_size(Queue& ctx,
                             const VectorView<T>& d,
                             const VectorView<T>& e,
                             Span<T> singular_values_out,
                             bool want_vectors);

    
    /**
     * @brief ORMBR/UNMBR-style application of bidiagonal reduction reflectors.
     *
        * Current implementation supports `vect='Q'` (tauq, CTA/blocked ORMQR path)
        * and `vect='P'` (taup, blocked compact-WY path over the right reflectors).
     */
    template <Backend B, typename T>
    Event ormbr(Queue& ctx,
                const MatrixView<T, MatrixFormat::Dense>& a,
                const VectorView<T>& tau,
                const MatrixView<T, MatrixFormat::Dense>& c,
                char vect,
                Side side,
                Transpose trans,
                const Span<std::byte>& ws,
                int32_t block_size = 32);

    template <Backend B, typename T>
    size_t ormbr_buffer_size(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& a,
                             const VectorView<T>& tau,
                             const MatrixView<T, MatrixFormat::Dense>& c,
                             char vect,
                             Side side,
                             Transpose trans,
                             int32_t block_size = 32);

    /**
     * @brief Blocked native SVD for real dense matrices.
     *
     * Pipeline: GEBRD-style dense -> bidiagonal, then BDSQR (or BDSDC when explicitly
     * selected in blocked mode), then ORMBR-style backtransforms for full U and V^H.
     */
    template <Backend B, typename T>
    Event gesvd_blocked(Queue& ctx,
                        const MatrixView<T, MatrixFormat::Dense>& a_in,
                        Span<typename base_type<T>::type> singular_values,
                        const MatrixView<T, MatrixFormat::Dense>& u_out,
                        const MatrixView<T, MatrixFormat::Dense>& vh_out,
                        SvdVectors jobu,
                        SvdVectors jobvh,
                        const Span<std::byte>& ws);

    template <Backend B, typename T>
    Event gesvd_blocked(Queue& ctx,
                        const MatrixView<T, MatrixFormat::Dense>& a_in,
                        Span<typename base_type<T>::type> singular_values,
                        const MatrixView<T, MatrixFormat::Dense>& u_out,
                        const MatrixView<T, MatrixFormat::Dense>& vh_out,
                        SvdVectors jobu,
                        SvdVectors jobvh,
                        Uplo hermitian_uplo,
                        const Span<std::byte>& ws);

    template <Backend B, typename T>
    size_t gesvd_blocked_buffer_size(Queue& ctx,
                                     const MatrixView<T, MatrixFormat::Dense>& a,
                                     Span<typename base_type<T>::type> singular_values,
                                     const MatrixView<T, MatrixFormat::Dense>& u_out,
                                     const MatrixView<T, MatrixFormat::Dense>& vh_out,
                                     SvdVectors jobu,
                                     SvdVectors jobvh);

    template <Backend B, typename T>
    size_t gesvd_blocked_buffer_size(Queue& ctx,
                                     const MatrixView<T, MatrixFormat::Dense>& a,
                                     Span<typename base_type<T>::type> singular_values,
                                     const MatrixView<T, MatrixFormat::Dense>& u_out,
                                     const MatrixView<T, MatrixFormat::Dense>& vh_out,
                                     SvdVectors jobu,
                                     SvdVectors jobvh,
                                     Uplo hermitian_uplo);

    /**
     * @brief CTA-oriented native SVD entry point for very small real square matrices.
     *
     * Current scope matches the native blocked path except it is intended for
     * `1 <= n <= 32` and small-batch CUDA execution.
     */
    template <Backend B, typename T>
    Event gesvd_cta(Queue& ctx,
                    const MatrixView<T, MatrixFormat::Dense>& a_in,
                    Span<typename base_type<T>::type> singular_values,
                    const MatrixView<T, MatrixFormat::Dense>& u_out,
                    const MatrixView<T, MatrixFormat::Dense>& vh_out,
                    SvdVectors jobu,
                    SvdVectors jobvh,
                    const Span<std::byte>& ws);

    template <Backend B, typename T>
    Event gesvd_cta(Queue& ctx,
                    const MatrixView<T, MatrixFormat::Dense>& a_in,
                    Span<typename base_type<T>::type> singular_values,
                    const MatrixView<T, MatrixFormat::Dense>& u_out,
                    const MatrixView<T, MatrixFormat::Dense>& vh_out,
                    SvdVectors jobu,
                    SvdVectors jobvh,
                    Uplo hermitian_uplo,
                    const Span<std::byte>& ws);

    template <Backend B, typename T>
    size_t gesvd_cta_buffer_size(Queue& ctx,
                                 const MatrixView<T, MatrixFormat::Dense>& a,
                                 Span<typename base_type<T>::type> singular_values,
                                 const MatrixView<T, MatrixFormat::Dense>& u_out,
                                 const MatrixView<T, MatrixFormat::Dense>& vh_out,
                                 SvdVectors jobu,
                                 SvdVectors jobvh);

    template <Backend B, typename T>
    size_t gesvd_cta_buffer_size(Queue& ctx,
                                 const MatrixView<T, MatrixFormat::Dense>& a,
                                 Span<typename base_type<T>::type> singular_values,
                                 const MatrixView<T, MatrixFormat::Dense>& u_out,
                                 const MatrixView<T, MatrixFormat::Dense>& vh_out,
                                 SvdVectors jobu,
                                 SvdVectors jobvh,
                                 Uplo hermitian_uplo);

    /**
     * @brief Parameters for the one-sided Jacobi SVD (`gesvdj_cta`).
     *
     * Deliberately NOT a reuse of JacobiParams: that struct's `sort_order` defaults
     * to Ascending while gesvd's contract admits exactly one order (descending), so
     * reusing it invites a silent reversal.
     */
    template <typename T>
    struct GesvdjParams {
        using Real = typename base_type<T>::type;

        // A rotation is applied to pivot pair (p,q) only when
        //     |a_pq| > tol_multiplier * n * eps * sqrt(|a_pp| * |a_qq|)
        // over the 2x2 Gram entries of the current columns. This relative form is what
        Real tol_multiplier = Real(1);

        // Cap on cyclic sweeps. Convergence normally occurs in well under 10.
        size_t max_sweeps = 30;

        // Multiplies the baseline problems-per-work-group. Baseline is
        // 32 / P problems, clamped by local memory and max work-group size.
        size_t cta_wg_size_multiplier = 1;

        // sigma_j <= zero_sigma_multiplier * eps * sigma_max means U_j is not
        // determined by A and is filled from the orthogonal complement.
        Real zero_sigma_multiplier = Real(1);

        // Optional per-problem diagnostic, the analogue of cusolverDnXgesvdjGetSweeps:
        // when non-empty (size >= batch_size) the kernel writes each problem's sweeps.
        Span<int32_t> sweep_counts = Span<int32_t>();
    };

    /**
     * @brief One-sided (Hestenes) Jacobi SVD for batches of small matrices.
     *
     * Computes A = U * diag(s) * Vh with high RELATIVE accuracy: the error in the singular
     * values is governed by the condition number of the column-equilibrated matrix rather
     * than of A itself, so graded and badly scaled inputs keep their small singular values.
     * The gebrd -> tridiagonal -> steqr path in `gesvd_cta` and `gesvd_blocked` does not.
     *
     * Supports max(m, n) <= 32, real and complex, rectangular in both orientations. A is
     * DESTROYED. Singular values are returned descending. Requires no workspace.
     */
    template <Backend B, typename T>
    Event gesvdj_cta(Queue& ctx,
                     const MatrixView<T, MatrixFormat::Dense>& a_in,
                     Span<typename base_type<T>::type> singular_values,
                     const MatrixView<T, MatrixFormat::Dense>& u_out,
                     const MatrixView<T, MatrixFormat::Dense>& vh_out,
                     SvdVectors jobu,
                     SvdVectors jobvh,
                     const Span<std::byte>& ws = Span<std::byte>(),
                     GesvdjParams<T> params = GesvdjParams<T>());

    template <Backend B, typename T>
    size_t gesvdj_cta_buffer_size(Queue& ctx,
                                  const MatrixView<T, MatrixFormat::Dense>& a,
                                  Span<typename base_type<T>::type> singular_values,
                                  const MatrixView<T, MatrixFormat::Dense>& u_out,
                                  const MatrixView<T, MatrixFormat::Dense>& vh_out,
                                  SvdVectors jobu,
                                  SvdVectors jobvh,
                                  GesvdjParams<T> params = GesvdjParams<T>());

    /**
     * @brief CTA-optimized application of Q from a QR/QL factorization (ORMQx/UNMQx semantics) for very small matrices.
     *
     * This applies the implicit orthogonal/unitary matrix Q represented by Householder
     * reflectors (A, TAU) from GEQRF (Upper) or GEQLF (Lower) to C.
     */
    template <Backend B, typename T>
    Event ormqx_cta(Queue& ctx,
                   const MatrixView<T, MatrixFormat::Dense>& a_in,
                   const VectorView<T>& tau_in,
                   const MatrixView<T, MatrixFormat::Dense>& c_in,
                   Uplo factorization,
                   Side side,
                   Transpose trans,
                   int32_t k,
                   const Span<std::byte>& ws,
                   size_t cta_wg_size_multiplier = 1);

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


    enum class StedcSecularSolver {
        Rocm,
        Legacy,
    };

    // Controls how the merge step (secular solve + eigenvector formation) is dispatched.
    enum class StedcMergeVariant {
        Auto = -1,          // Use BatchLAS tuning tables for the current problem size
        Baseline,           // Current serial-per-root path (3 separate kernels)
        Fused,              // One kernel: warp-parallel root solve + build/normalize Qprime columns
        FusedCta,           // CTA-partitioned root solve using tunable threads per root
    };

    // Which divide-and-conquer driver runs the merge tree.
    enum class StedcAlgorithm {
        Auto = -1,      // Currently: Levels
        Levels,         // Level-synchronous: every node at a tree level merges in one launch
        Recursive,      // Depth-first: one node per launch (kept for A/B comparison)
    };

    template <typename T>
    struct StedcParams {
        int64_t recursion_threshold = 0; // <=0 uses BatchLAS tuning; otherwise this exact threshold is used
        StedcAlgorithm algorithm = StedcAlgorithm::Auto;
        StedcSecularSolver secular_solver = StedcSecularSolver::Rocm;
        SteqrParams<T> leaf_steqr_params = SteqrParams<T>();

        // --- Merge-step tuning knobs ---
        StedcMergeVariant merge_variant = StedcMergeVariant::Auto;
        int merge_threads = 128;       // work-group size for fused kernel
        int max_sec_iter = 50;         // iteration cap for secular root solver
        bool enable_rescale = true;    // keep ROCm-style v rescale (disable for perf experiments)
        int secular_threads_per_root = 0;       // <=0 uses BatchLAS tuning; otherwise this exact partition width is used
        int secular_cta_wg_size_multiplier = 0;  // <=0 uses BatchLAS tuning; otherwise this exact multiplier is used
    };

    template <Backend B, typename T>
    Event stedc(Queue& ctx, const VectorView<T>& d, const VectorView<T>& e, const VectorView<T>& eigenvalues, const Span<std::byte>& ws,
            JobType jobz, StedcParams<T> params, const MatrixView<T, MatrixFormat::Dense>& eigvects);

    template <Backend B, typename T>
    inline Event stedc(Queue& ctx, const Vector<T>& d, const Vector<T>& e, const Vector<T>& eigenvalues, const Span<std::byte>& ws,
            JobType jobz, StedcParams<T> params, const Matrix<T, MatrixFormat::Dense>& eigvects) {
        return stedc<B,T>(ctx, static_cast<VectorView<T>>(d), static_cast<VectorView<T>>(e), static_cast<VectorView<T>>(eigenvalues), ws, jobz, params, MatrixView<T, MatrixFormat::Dense>(eigvects));
    }

    template <Backend B, typename T>
    size_t stedc_buffer_size(Queue& ctx, size_t n, size_t batch_size, JobType jobz, StedcParams<T> params);

    // Deprecated spelling, kept so an out-of-tree caller gets a warning and not a link
    // error. docs/cpp-api.md states the `*_buffer_size` rule.
    template <Backend B, typename T>
    [[deprecated("renamed to stedc_buffer_size")]]
    inline size_t stedc_workspace_size(Queue& ctx, size_t n, size_t batch_size, JobType jobz, StedcParams<T> params) {
        return stedc_buffer_size<B, T>(ctx, n, batch_size, jobz, params);
    }


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
        size_t workspace_size = ritz_values_buffer_size<B,T,MFormat>(ctx, A, V, static_cast<VectorView<float_type>>(ritz_vals));
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
    size_t ritz_values_buffer_size(Queue& ctx,
                                 const MatrixView<T, MFormat>& A,
                                 const MatrixView<T, MatrixFormat::Dense>& V,
                                 const VectorView<typename base_type<T>::type>& ritz_vals);

    template <Backend B, typename T, MatrixFormat MFormat>
    inline size_t ritz_values_buffer_size(Queue& ctx,
                                       const Matrix<T, MFormat>& A,
                                       const Matrix<T, MatrixFormat::Dense>& V,
                                       const Vector<typename base_type<T>::type>& ritz_vals) {
        return ritz_values_buffer_size<B,T,MFormat>(ctx, MatrixView<T, MFormat>(A), MatrixView<T, MatrixFormat::Dense>(V), static_cast<VectorView<typename base_type<T>::type>>(ritz_vals));
    }

    // Deprecated spellings, one per overload above; see `stedc_workspace_size`.
    template <Backend B, typename T, MatrixFormat MFormat>
    [[deprecated("renamed to ritz_values_buffer_size")]]
    inline size_t ritz_values_workspace(Queue& ctx,
                                        const MatrixView<T, MFormat>& A,
                                        const MatrixView<T, MatrixFormat::Dense>& V,
                                        const VectorView<typename base_type<T>::type>& ritz_vals) {
        return ritz_values_buffer_size<B,T,MFormat>(ctx, A, V, ritz_vals);
    }

    template <Backend B, typename T, MatrixFormat MFormat>
    [[deprecated("renamed to ritz_values_buffer_size")]]
    inline size_t ritz_values_workspace(Queue& ctx,
                                        const Matrix<T, MFormat>& A,
                                        const Matrix<T, MatrixFormat::Dense>& V,
                                        const Vector<typename base_type<T>::type>& ritz_vals) {
        return ritz_values_buffer_size<B,T,MFormat>(ctx, A, V, ritz_vals);
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

namespace batchlas {

// Backend-deducing overloads for the extension surface: `f(ctx, ...)` uses ctx.backend().
// The macro is constrained, so a name whose remaining template parameters are not
// deducible from the arguments simply gets no overload here rather than an ill-formed one.

BATCHLAS_DISPATCH_ON_QUEUE(ortho)
BATCHLAS_DISPATCH_ON_QUEUE(ortho_buffer_size)
BATCHLAS_DISPATCH_ON_QUEUE(syevx)
BATCHLAS_DISPATCH_ON_QUEUE(syevx_buffer_size)
BATCHLAS_DISPATCH_ON_QUEUE(syevx_direct)
BATCHLAS_DISPATCH_ON_QUEUE(syevx_direct_buffer_size)
BATCHLAS_DISPATCH_ON_QUEUE(syevx_direct_subset)
BATCHLAS_DISPATCH_ON_QUEUE(syevx_direct_subset_buffer_size)
BATCHLAS_DISPATCH_ON_QUEUE(syevx_lobpcg)
BATCHLAS_DISPATCH_ON_QUEUE(syevx_lobpcg_buffer_size)
BATCHLAS_DISPATCH_ON_QUEUE(syevx_filtered)
BATCHLAS_DISPATCH_ON_QUEUE(syevx_filtered_buffer_size)
BATCHLAS_DISPATCH_ON_QUEUE(lanczos)
BATCHLAS_DISPATCH_ON_QUEUE(lanczos_buffer_size)
BATCHLAS_DISPATCH_ON_QUEUE(tridiagonal_solver)
BATCHLAS_DISPATCH_ON_QUEUE(tridiagonal_solver_buffer_size)
BATCHLAS_DISPATCH_ON_QUEUE(stebz)
BATCHLAS_DISPATCH_ON_QUEUE(stebz_buffer_size)
BATCHLAS_DISPATCH_ON_QUEUE(stein)
BATCHLAS_DISPATCH_ON_QUEUE(stein_buffer_size)
BATCHLAS_DISPATCH_ON_QUEUE(steqr)
BATCHLAS_DISPATCH_ON_QUEUE(steqr_cta)
BATCHLAS_DISPATCH_ON_QUEUE(sytrd_cta)
BATCHLAS_DISPATCH_ON_QUEUE(latrd_lower_panel)
BATCHLAS_DISPATCH_ON_QUEUE(sytrd_blocked)
BATCHLAS_DISPATCH_ON_QUEUE(sytrd_blocked_buffer_size)
BATCHLAS_DISPATCH_ON_QUEUE(sytrd_sy2sb)
BATCHLAS_DISPATCH_ON_QUEUE(sytrd_sy2sb_buffer_size)
BATCHLAS_DISPATCH_ON_QUEUE(sytrd_sb2st)
BATCHLAS_DISPATCH_ON_QUEUE(sytrd_sb2st_buffer_size)
BATCHLAS_DISPATCH_ON_QUEUE(sytrd_band_reduction)
BATCHLAS_DISPATCH_ON_QUEUE(sytrd_band_reduction_single_step)
BATCHLAS_DISPATCH_ON_QUEUE(sytrd_band_reduction_buffer_size)
BATCHLAS_DISPATCH_ON_QUEUE(sytrd_band_reduction_single_step_buffer_size)
BATCHLAS_DISPATCH_ON_QUEUE(hetrd_hb2st)
BATCHLAS_DISPATCH_ON_QUEUE(hetrd_hb2st_buffer_size)
BATCHLAS_DISPATCH_ON_QUEUE(syev_cta)
BATCHLAS_DISPATCH_ON_QUEUE(syev_cta_buffer_size)
BATCHLAS_DISPATCH_ON_QUEUE(syev_cta_fused)
BATCHLAS_DISPATCH_ON_QUEUE(syev_cta_fused_buffer_size)
BATCHLAS_DISPATCH_ON_QUEUE(syev_jacobi_cta)
BATCHLAS_DISPATCH_ON_QUEUE(syev_jacobi_cta_buffer_size)
BATCHLAS_DISPATCH_ON_QUEUE(syev_blocked)
BATCHLAS_DISPATCH_ON_QUEUE(syev_blocked_buffer_size)
BATCHLAS_DISPATCH_ON_QUEUE(syev_two_stage)
BATCHLAS_DISPATCH_ON_QUEUE(syev_two_stage_buffer_size)
BATCHLAS_DISPATCH_ON_QUEUE(gebrd_unblocked)
BATCHLAS_DISPATCH_ON_QUEUE(gebrd_cta)
BATCHLAS_DISPATCH_ON_QUEUE(gebrd_blocked)
BATCHLAS_DISPATCH_ON_QUEUE(gebrd_blocked_buffer_size)
BATCHLAS_DISPATCH_ON_QUEUE(bdsqr)
BATCHLAS_DISPATCH_ON_QUEUE(bdsdc)
BATCHLAS_DISPATCH_ON_QUEUE(bdsdc_buffer_size)
BATCHLAS_DISPATCH_ON_QUEUE(ormbr)
BATCHLAS_DISPATCH_ON_QUEUE(ormbr_buffer_size)
BATCHLAS_DISPATCH_ON_QUEUE(gesvd_blocked)
BATCHLAS_DISPATCH_ON_QUEUE(gesvd_blocked_buffer_size)
BATCHLAS_DISPATCH_ON_QUEUE(gesvd_cta)
BATCHLAS_DISPATCH_ON_QUEUE(gesvd_cta_buffer_size)
BATCHLAS_DISPATCH_ON_QUEUE(gesvdj_cta)
BATCHLAS_DISPATCH_ON_QUEUE(gesvdj_cta_buffer_size)
BATCHLAS_DISPATCH_ON_QUEUE(ormqx_cta)
BATCHLAS_DISPATCH_ON_QUEUE(stedc)
BATCHLAS_DISPATCH_ON_QUEUE(stedc_buffer_size)
BATCHLAS_DISPATCH_ON_QUEUE(ritz_values)
BATCHLAS_DISPATCH_ON_QUEUE(ritz_values_buffer_size)

// Queue-deducing forms of the two deprecated spellings, hand-written because
// BATCHLAS_DISPATCH_ON_QUEUE has nowhere to put the [[deprecated]] attribute.
template <typename... Args>
    requires requires(Queue& probe_ctx, Args&&... probe_args) {
        stedc_buffer_size<::batchlas::detail::kProbeBackend>(probe_ctx, std::forward<Args>(probe_args)...);
    }
[[deprecated("renamed to stedc_buffer_size")]]
inline auto stedc_workspace_size(Queue& ctx, Args&&... args) {
    return stedc_buffer_size(ctx, std::forward<Args>(args)...);
}

template <typename... Args>
    requires requires(Queue& probe_ctx, Args&&... probe_args) {
        ritz_values_buffer_size<::batchlas::detail::kProbeBackend>(probe_ctx, std::forward<Args>(probe_args)...);
    }
[[deprecated("renamed to ritz_values_buffer_size")]]
inline auto ritz_values_workspace(Queue& ctx, Args&&... args) {
    return ritz_values_buffer_size(ctx, std::forward<Args>(args)...);
}

BATCHLAS_DISPATCH_ON_QUEUE(inv)
BATCHLAS_DISPATCH_ON_QUEUE(inv_buffer_size)
BATCHLAS_DISPATCH_ON_QUEUE(inv_matrix)

// ---- ortho: option-struct and arena-backed spellings -----------------------
//
// Same layer as blas/options.hh, but it has to live HERE: blas/linalg.hh includes
// blas/functions.hh (which ends by including options.hh) BEFORE this header, so `ortho`
// is not yet declared when options.hh is parsed. None of options.hh's helpers are
// reachable from here, so the MatrixView and Matrix spellings are written out separately.
//
// A lease is released when the call returns, which on an out-of-order Queue drains the
// device first: the arena spelling is synchronous where the span-taking one is not.

struct OrthoOptions {
    Transpose transA = Transpose::NoTrans;
    OrthoAlgorithm algorithm = OrthoAlgorithm::Chol2;
};

struct OrthoAgainstOptions {
    Transpose transA = Transpose::NoTrans;
    Transpose transM = Transpose::NoTrans;
    OrthoAlgorithm algorithm = OrthoAlgorithm::Chol2;
    size_t iterations = 2;
};

// ---- in-place orthogonalisation --------------------------------------------

template <Backend B, typename T>
inline Event ortho(Queue& ctx,
        const MatrixView<T, MatrixFormat::Dense>& A,
        const OrthoOptions& opts,
        Span<std::byte> workspace) {
    return ortho<B, T>(ctx, A, opts.transA, workspace, opts.algorithm);
}

template <Backend B, typename T>
inline Event ortho(Queue& ctx,
        const MatrixView<T, MatrixFormat::Dense>& A,
        const OrthoOptions& opts) {
    // Sized with the algorithm the call actually runs: Chol2, SVQB and the Householder
    // path need different scratch, so a default-algorithm query under-sizes the rest.
    auto lease = ctx.workspace(ortho_buffer_size<B, T>(ctx, A, opts.transA, opts.algorithm));
    return ortho<B, T>(ctx, A, opts.transA, lease.span(), opts.algorithm);
}

template <Backend B, typename T>
inline Event ortho(Queue& ctx,
        const Matrix<T, MatrixFormat::Dense>& A,
        const OrthoOptions& opts,
        Span<std::byte> workspace) {
    return ortho<B, T>(ctx, MatrixView<T, MatrixFormat::Dense>(A), opts, workspace);
}

template <Backend B, typename T>
inline Event ortho(Queue& ctx,
        const Matrix<T, MatrixFormat::Dense>& A,
        const OrthoOptions& opts) {
    return ortho<B, T>(ctx, MatrixView<T, MatrixFormat::Dense>(A), opts);
}

template <typename T>
inline Event ortho(Queue& ctx,
        const MatrixView<T, MatrixFormat::Dense>& A,
        const OrthoOptions& opts,
        Span<std::byte> workspace) {
    if (detail::pointer_checks_enabled()) {
        detail::require_arg_accessible(ctx, A, "ortho: A");
        detail::require_arg_accessible(ctx, workspace, "ortho: workspace");
    }
    return with_backend(ctx, [&](auto Back) { return ortho<Back.value, T>(ctx, A, opts, workspace); });
}

template <typename T>
inline Event ortho(Queue& ctx,
        const MatrixView<T, MatrixFormat::Dense>& A,
        const OrthoOptions& opts = {}) {
    if (detail::pointer_checks_enabled()) detail::require_arg_accessible(ctx, A, "ortho: A");
    return with_backend(ctx, [&](auto Back) { return ortho<Back.value, T>(ctx, A, opts); });
}

template <typename T>
inline Event ortho(Queue& ctx,
        const Matrix<T, MatrixFormat::Dense>& A,
        const OrthoOptions& opts,
        Span<std::byte> workspace) {
    return ortho<T>(ctx, MatrixView<T, MatrixFormat::Dense>(A), opts, workspace);
}

template <typename T>
inline Event ortho(Queue& ctx,
        const Matrix<T, MatrixFormat::Dense>& A,
        const OrthoOptions& opts = {}) {
    return ortho<T>(ctx, MatrixView<T, MatrixFormat::Dense>(A), opts);
}

namespace detail {
// The bare-`{}` trap, in the one place ortho has it. `ortho(ctx, A, {}, ws)` matches BOTH
// the positional overload `(ctx, A, Transpose, ws, algo)` and the option overload
// `(ctx, A, OrthoOptions, ws)`; `{}` is an exact match for a scoped enum but only a
// user-defined conversion to a class type, so the positional overload would win silently
// and hand the caller `Transpose{}` with no diagnostic. A third candidate at the same
// exact-match rank makes the bare-`{}` call ambiguous instead, while every spelled-out
// call still resolves exactly as before. Same job as options.hh's
// EmptyBracesAreAmbiguous, which is not visible here.
enum class OrthoEmptyBracesAreAmbiguous {};
}  // namespace detail

template <Backend B, typename T>
Event ortho(Queue&, const MatrixView<T, MatrixFormat::Dense>&,
            detail::OrthoEmptyBracesAreAmbiguous, Span<std::byte>) = delete;

template <Backend B, typename T>
Event ortho(Queue&, const Matrix<T, MatrixFormat::Dense>&,
            detail::OrthoEmptyBracesAreAmbiguous, Span<std::byte>) = delete;

template <typename T>
Event ortho(Queue&, const MatrixView<T, MatrixFormat::Dense>&,
            detail::OrthoEmptyBracesAreAmbiguous, Span<std::byte>) = delete;

template <typename T>
Event ortho(Queue&, const Matrix<T, MatrixFormat::Dense>&,
            detail::OrthoEmptyBracesAreAmbiguous, Span<std::byte>) = delete;

// ---- orthogonalisation against an external metric ---------------------------
//
// No `{}` guard is needed on this family: the positional metric form needs at least six
// arguments while the option forms take four and five, so no argument list can reach both.

template <Backend B, typename T>
inline Event ortho(Queue& ctx,
        const MatrixView<T, MatrixFormat::Dense>& A,
        const MatrixView<T, MatrixFormat::Dense>& M,
        const OrthoAgainstOptions& opts,
        Span<std::byte> workspace) {
    return ortho<B, T>(ctx, A, M, opts.transA, opts.transM, workspace, opts.algorithm,
                       opts.iterations);
}

template <Backend B, typename T>
inline Event ortho(Queue& ctx,
        const MatrixView<T, MatrixFormat::Dense>& A,
        const MatrixView<T, MatrixFormat::Dense>& M,
        const OrthoAgainstOptions& opts) {
    auto lease = ctx.workspace(ortho_buffer_size<B, T>(ctx, A, M, opts.transA, opts.transM,
                                                       opts.algorithm, opts.iterations));
    return ortho<B, T>(ctx, A, M, opts.transA, opts.transM, lease.span(), opts.algorithm,
                       opts.iterations);
}

template <Backend B, typename T>
inline Event ortho(Queue& ctx,
        const Matrix<T, MatrixFormat::Dense>& A,
        const Matrix<T, MatrixFormat::Dense>& M,
        const OrthoAgainstOptions& opts,
        Span<std::byte> workspace) {
    return ortho<B, T>(ctx, MatrixView<T, MatrixFormat::Dense>(A),
                       MatrixView<T, MatrixFormat::Dense>(M), opts, workspace);
}

template <Backend B, typename T>
inline Event ortho(Queue& ctx,
        const Matrix<T, MatrixFormat::Dense>& A,
        const Matrix<T, MatrixFormat::Dense>& M,
        const OrthoAgainstOptions& opts) {
    return ortho<B, T>(ctx, MatrixView<T, MatrixFormat::Dense>(A),
                       MatrixView<T, MatrixFormat::Dense>(M), opts);
}

template <typename T>
inline Event ortho(Queue& ctx,
        const MatrixView<T, MatrixFormat::Dense>& A,
        const MatrixView<T, MatrixFormat::Dense>& M,
        const OrthoAgainstOptions& opts,
        Span<std::byte> workspace) {
    if (detail::pointer_checks_enabled()) {
        detail::require_arg_accessible(ctx, A, "ortho: A");
        detail::require_arg_accessible(ctx, M, "ortho: M");
        detail::require_arg_accessible(ctx, workspace, "ortho: workspace");
    }
    return with_backend(ctx,
                        [&](auto Back) { return ortho<Back.value, T>(ctx, A, M, opts, workspace); });
}

template <typename T>
inline Event ortho(Queue& ctx,
        const MatrixView<T, MatrixFormat::Dense>& A,
        const MatrixView<T, MatrixFormat::Dense>& M,
        const OrthoAgainstOptions& opts = {}) {
    if (detail::pointer_checks_enabled()) {
        detail::require_arg_accessible(ctx, A, "ortho: A");
        detail::require_arg_accessible(ctx, M, "ortho: M");
    }
    return with_backend(ctx, [&](auto Back) { return ortho<Back.value, T>(ctx, A, M, opts); });
}

template <typename T>
inline Event ortho(Queue& ctx,
        const Matrix<T, MatrixFormat::Dense>& A,
        const Matrix<T, MatrixFormat::Dense>& M,
        const OrthoAgainstOptions& opts,
        Span<std::byte> workspace) {
    return ortho<T>(ctx, MatrixView<T, MatrixFormat::Dense>(A),
                    MatrixView<T, MatrixFormat::Dense>(M), opts, workspace);
}

template <typename T>
inline Event ortho(Queue& ctx,
        const Matrix<T, MatrixFormat::Dense>& A,
        const Matrix<T, MatrixFormat::Dense>& M,
        const OrthoAgainstOptions& opts = {}) {
    return ortho<T>(ctx, MatrixView<T, MatrixFormat::Dense>(A),
                    MatrixView<T, MatrixFormat::Dense>(M), opts);
}

}  // namespace batchlas
