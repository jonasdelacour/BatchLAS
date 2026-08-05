#pragma once
#include <complex>
#include <util/sycl-device-queue.hh>
#include <util/sycl-span.hh>
#include <batchlas/tuning_params.hh>
#include <blas/enums.hh>
#include <blas/matrix.hh>
#include <blas/functions/iluk.hh>
#include <numeric>
#include <limits>
#include <cstddef>
#include <cstdint>
#include <vector>


namespace batchlas {
    // Forward declarations for interface compatibility

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
        // Optional ILU(k) preconditioner. An ILU(k) factorization of A approximates
        // A^{-1}, which is the correct LOBPCG preconditioner only when searching for
        // the *smallest* eigenpairs. With find_largest = true it damps exactly the
        // components being sought and amplifies the rest, so syevx rejects that
        // combination rather than silently converging more slowly.
        const ILUKPreconditioner<T>* preconditioner = nullptr;
        // Which preconditioner family the LOBPCG path should use. `Auto` keeps the
        // pre-existing behaviour exactly: ILU(k) when a factor is supplied or
        // requested below, otherwise none (unless BATCHLAS_SYEVX_PRECONDITIONER
        // names a default). See SyevxPreconditioner and
        // `syevx_select_preconditioner`.
        SyevxPreconditioner preconditioner_type = SyevxPreconditioner::Auto;
        // Build the ILU(k) factor inside syevx instead of supplying one. The factor
        // is carved out of the same workspace the caller passes to syevx, so an
        // end-to-end timing covers formation as well as application. Requires a CSR
        // A and find_largest = false, and is mutually exclusive with the pointer above.
        bool build_preconditioner = false;
        ILUKParams<T> iluk_params{};
        // Chebyshev filter degree for SyevxAlgorithm::Filtered. 0 selects a
        // default. Higher degrees separate the wanted end of the spectrum more
        // aggressively per outer iteration, at one matvec each; the useful range
        // is roughly 8-25 and the optimum depends on the spectral gap.
        size_t filter_degree = 0;
        // LOBPCG only: number of block power-iteration steps applied to the random
        // starting block before the first Rayleigh-Ritz. Each step is one matvec plus
        // one orthogonalization and biases the start toward the largest eigenpairs.
        // -1 selects the built-in default, 0 disables.
        //
        // Ignored unless find_largest is true: powers of A amplify the largest
        // eigendirections, so with find_largest = false they would drive the start
        // away from what is wanted. The shifted operator that would fix that was
        // measured and gave no useful speedup, so it is not implemented -- see the
        // measurements in src/extensions/syevx_lobpcg.cc.
        int init_power_iterations = -1;
        const SyevxInstrumentation<T>* instrumentation = nullptr;               // Optional convergence instrumentation sink

        // ---- Range selection (LAPACK ?syevx's RANGE argument) --------------
        // All defaulted so that an existing caller's behaviour is byte-for-byte
        // unchanged: Extremal + find_largest is exactly the historical top-k.

        // Which part of the spectrum to return. See SyevxSelect.
        SyevxSelect select = SyevxSelect::Extremal;

        // select == Index: inclusive 0-based bounds into the ASCENDING spectrum.
        // iu < 0 means n-1. il > iu is an empty request and is rejected.
        int64_t il = 0;
        int64_t iu = -1;

        // select == Value: the half-open interval (vl, vu], matching LAPACK. The
        // count is data-dependent and differs per batch item, so it is reported
        // through the `m` output of syevx rather than being known in advance.
        //
        // These are float_type, not T: the eigenvalues of a Hermitian matrix are
        // real and W is already Span<base_type<T>::type>, so typing them T would
        // force complex callers to write std::complex<float>(vl) for a real
        // quantity. This deliberately differs from absolute_tolerance /
        // relative_tolerance above, which are a pre-existing wart.
        float_type vl = float_type(0);
        float_type vu = float_type(0);

        // Absolute tolerance on each eigenvalue for the bisection-based paths.
        // Non-positive means eps * ||T||, i.e. full working precision. Forwarded
        // to StebzParams::abstol; ignored by paths that get their eigenvalues from
        // a full decomposition (syevx_direct).
        float_type abstol = float_type(0);

        // Output order within the selected block. Honoured for Index and Value
        // only: for Extremal the order comes from find_largest (descending for the
        // largest, ascending for the smallest), which is what preserves the
        // historical contract, and this member is ignored.
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
     * @brief Computes selected eigenvalues and optionally eigenvectors of a
     *        Hermitian/symmetric matrix, dense or sparse.
     *
     * Which part of the spectrum is returned is set by `SyevxParams::select`:
     * the `neigs` extremal eigenpairs (the default and the historical
     * behaviour), an index block `il..iu` of the ascending spectrum, or every
     * eigenvalue in a half-open interval `(vl, vu]`. See SyevxSelect.
     *
     * @param ctx Execution context/device queue
     * @param A Matrix A (dense or CSR)
     * @param W Output array for eigenvalues, `neigs` entries per batch item
     * @param neigs CAPACITY of `W` and of `V`'s columns, per batch item -- not
     *        necessarily the number produced. For `Extremal` it is the number
     *        wanted and for `Index` it must equal `iu - il + 1`, so in both
     *        cases capacity and count coincide; for `Value` the count is
     *        data-dependent, differs per batch item, and is reported through
     *        the `m` output of the overload below.
     * @param workspace Pre-allocated workspace buffer
     * @param jobz Whether to compute eigenvectors
     * @param V Dense matrix to store eigenvectors (if jobz = EigenVectors)
     * @param params Additional parameters for the algorithm
     * @return Event Event to track operation completion
     *
     * @throws std::invalid_argument if `SyevxParams::select` is `Value`: the
     *         count is only known on the device, so a Value range requires the
     *         `m`-taking overload below. Also if a non-extremal range is asked
     *         of a path that cannot answer one -- sparse input, or an explicit
     *         `method` of LOBPCG/Filtered. See `syevx_select_algorithm`.
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
     * @brief `syevx` with a per-batch-item count of how many eigenpairs were
     *        actually found. Required for `SyevxSelect::Value`.
     *
     * The contract for a value range is LAPACK's: the caller declares a
     * capacity (`neigs`), the library writes `min(m[b], neigs)` eigenpairs into
     * slots `[0, min(m[b], neigs))` of item `b` and leaves the rest of that
     * item's `W` and `V` untouched, and `m[b]` reports the TRUE count. So
     * `m[b] > neigs` is the caller's overflow signal -- one comparison per item,
     * no extra synchronization beyond reading `m`, which a value-range caller
     * has to do anyway. When truncating, the LOWEST `neigs` eigenvalues of the
     * interval are the ones kept, regardless of the requested output order.
     *
     * For `Extremal` and `Index` the count is static (`neigs` and `iu - il + 1`
     * respectively) and `m` is filled with it for uniformity.
     *
     * @param m Per-item count, at least `A.batch_size()` entries. Device-writable.
     *
     * OVERLOAD-RESOLUTION INVARIANT, do not break it: these forms are
     * unambiguous against the `m`-less ones above because parameter 5 of this
     * form (`size_t neigs`) and parameter 5 of the `m`-less form
     * (`Span<std::byte>` / `JobType`) are mutually non-convertible. It is NOT
     * because `Span<int32_t>` and `size_t` differ in position 4 -- `Span` has a
     * non-explicit `Span(T&)` constructor, so `Span<int32_t>` IS implicitly
     * constructible from an `int32_t` lvalue and position 4 alone does not
     * discriminate. Never let this form's parameter k+1 be a type the `m`-less
     * form's parameter k+1 converts to, and never spell a bare `{}` in argument
     * positions 4-6 (it is an identity conversion to both, hence ambiguous).
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

    // Forwarding overload (owning A only, eigenvalues only)
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

    // Forwarding overload (owning A and V)
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
     * Unlike the solve, this accepts `SyevxSelect::Value` on the `m`-less form:
     * sizing writes no counts, and if it threw here then sizing the workspace
     * for a value-range solve would be impossible. It does have to resolve the
     * range, though -- `syevx_direct_subset`'s workspace depends on it -- which
     * it does through the same `syevx_resolve_range` call the solve makes.
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

    // `m`-taking sizing forms. `m` is ACCEPTED AND IGNORED -- sizing writes no
    // counts and needs none; these exist only so that a value-range caller can
    // write the sizing call and the solve call with the same argument list
    // instead of dropping one argument in the middle of it. They are inline
    // forwarders, so they add no instantiated symbols.
    //
    // The same overload-resolution invariant as `syevx` applies: parameter 5
    // here is `size_t neigs`, parameter 5 of the `m`-less form is `JobType`,
    // and the two are mutually non-convertible.
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
     * the solve and the sizing call can never disagree about what was requested.
     *
     * `vl`/`vu` are deliberately absent: they are typed `float_type` and are
     * forwarded verbatim from SyevxParams to StebzParams (or compared against the
     * computed spectrum) without any normalization, so keeping them here would
     * force this struct -- and the resolver -- to be a template for no benefit.
     * Read `params.vl` / `params.vu` directly when `value_range` is true.
     */
    struct SyevxResolvedRange {
        bool    value_range;  // true: (vl, vu]; false: the index block [il, iu]
        int64_t il;           // valid iff !value_range; 0-based, inclusive
        int64_t iu;           // valid iff !value_range; 0-based, inclusive
        // Upper bound on the number of eigenpairs that can be produced per item,
        // already clamped to n. For an index block this is exactly iu-il+1 and
        // hence exactly m[b]; for a value range it is the caller's capacity and
        // the true m[b] may be larger (the answer is then truncated, see `syevx`).
        int64_t max_count;
        bool    reverse;      // write the selected block in descending order
    };

    /**
     * @brief Normalizes a range request into a SyevxResolvedRange.
     *
     * A plain (non-template) function, exactly like `syevx_select_algorithm`, so
     * that it links from every translation unit that needs it. Nothing about range
     * resolution depends on the scalar type.
     *
     * Legality is NOT checked here -- this function assumes the request already
     * passed the host-side validator in syevx.cc. It clamps rather than throws so
     * that it stays usable from a sizing path.
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

    // Convenience adaptor. Distinguished from the 7-argument form by arity, so
    // there is no overload ambiguity between them.
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
     * Never returns `Auto`: an unavailable tier falls back to its nearest
     * implemented neighbour (`DirectSubset` degrades to `Direct` when the scalar
     * type or format does not support it). Deterministic in its inputs so that
     * `syevx` and `syevx_buffer_size` always agree on the choice.
     *
     * @param format Matrix format of A (sparse formats always use LOBPCG)
     * @param n Matrix dimension
     * @param neigs Number of requested eigenpairs
     * @param requested Algorithm requested via SyevxParams::method
     * @param subset_supported Whether DirectSubset is available for this T/format
     * @param jobz Whether eigenvectors are wanted -- load-bearing, because the
     *        subset solver's only measured advantage is the narrowed
     *        back-transform, which does not exist in eigenvalues-only mode.
     * @param batch_size Number of matrices in the batch -- also load-bearing.
     *        The subset solver's reduction is parallel over the batch, so at
     *        small batch it starves; measured, it loses to Direct by up to 16x
     *        at batch 1 and wins by up to 2.1x at batch 256, for the same n.
     * @param select Which part of the spectrum was asked for
     *        (SyevxParams::select). Only `Direct` and `DirectSubset` implement
     *        anything other than `Extremal`, so this parameter does not choose
     *        between algorithms -- it EXCLUDES the two that cannot answer:
     *          - sparse + non-extremal throws (LOBPCG is the only sparse path);
     *          - an explicit `method` of LOBPCG/Filtered + non-extremal throws;
     *          - BATCHLAS_SYEVX_ALGORITHM=lobpcg|filtered + non-extremal
     *            degrades to Direct and warns once per process.
     *        Substituting an *algorithm* is a performance decision and this
     *        function does it freely; substituting the requested *part of the
     *        spectrum* would change the answer, so it throws instead -- except
     *        for the environment override, whose whole purpose is to force a
     *        whole suite onto one algorithm for diagnosis, which aborting on
     *        the first interior call would make impossible. Same asymmetry as
     *        `syevx_select_preconditioner`.
     * @return SyevxAlgorithm A concrete, implemented algorithm
     * @throws std::invalid_argument for the two rejected combinations above.
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
     * Never returns `Auto`. Deterministic in its inputs so that `syevx`,
     * `syevx_buffer_size` and `syevx_lobpcg` always agree -- the Jacobi path adds a
     * pool allocation, and the sizing call has to make the same decision the solve
     * will. Legality (e.g. ILU(k) requested with no factor supplied) is checked by
     * `syevx`, not here.
     *
     * @param requested SyevxParams::preconditioner_type
     * @param iluk_configured Whether an ILU(k) factor was supplied or requested
     * @param find_largest SyevxParams::find_largest; an environment-supplied default
     *        that is illegal for the requested end degrades to `None` instead of
     *        throwing (an explicit request still throws -- see `syevx`).
     */
    SyevxPreconditioner syevx_select_preconditioner(SyevxPreconditioner requested,
                                                    bool iluk_configured,
                                                    bool find_largest);

    /**
     * @brief Partial eigensolve by full decomposition followed by selection.
     *
     * Runs `syev` on a private copy of A (A is not modified) and extracts the
     * requested part of the spectrum. Dense input only, but every scalar type
     * (including complex) and every SyevxSelect range, which is what makes it the
     * universal fallback.
     *
     * Ordering: descending when `params.find_largest` for the default `Extremal`
     * selection (matching the LOBPCG path), otherwise `params.order`.
     *
     * @param W Eigenvalue output, `neigs` entries per batch item. `neigs` is a
     *        CAPACITY: for a Value range only `min(m[b], neigs)` entries are
     *        written and the rest are left untouched. The stride is always `neigs`.
     * @param m Per-item count of eigenvalues in the requested range, or an empty
     *        span to not report it. For Index/Extremal this is always the block
     *        size; for Value it is data-dependent, and `m[b] > neigs` is the
     *        caller's truncation signal. When truncating, the LOWEST `neigs`
     *        eigenvalues of the interval are the ones kept.
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

    // No `m` parameter: sizing writes no counts, and this function is
    // range-independent anyway (it sizes a full syev on n and batch alone).
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
     *        requested eigenvectors.
     *
     * Real scalar types and dense input only; callers should route complex or
     * sparse input elsewhere (`syevx` does this automatically).
     *
     * Supports every `SyevxSelect` range. Position within the spectrum costs it
     * nothing: the reduction, the band width `kd` and the reflector schedule
     * depend on `n` alone, `stebz` runs the same number of bisection steps for
     * every index, and both back-transforms act on the same fixed `n x neigs`
     * slice wherever the block sits.
     *
     * Ordering: descending when `params.find_largest` for the default `Extremal`
     * selection, otherwise `params.order`. The reversal is applied at the very
     * end, in the finalize kernel: `stein`'s cluster detection walks consecutive
     * eigenvalues and requires ascending input, so `stebz` is never asked for
     * descending mid-chain.
     *
     * @param W Eigenvalue output, `neigs` entries per batch item. `neigs` is a
     *        CAPACITY: for a Value range only `min(m[b], neigs)` entries are
     *        written and the rest are left untouched. The stride is always `neigs`.
     * @param m Per-item count of eigenvalues in the requested range, or an empty
     *        span to not report it. For Index/Extremal this is the block size; for
     *        Value it is data-dependent, and `m[b] > neigs` is the caller's
     *        truncation signal. When truncating, the LOWEST `neigs` eigenvalues of
     *        the interval are kept.
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

    // No `m` parameter: sizing writes no counts. It is NOT range-independent,
    // though -- a Value range needs room for up to n eigenvalues per item in the
    // internal stebz output regardless of the caller's capacity -- so it derives
    // its sizes from the same `syevx_resolve_range` call the solver makes.
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
     * @brief Chebyshev-filtered subspace iteration (SYEVX_PLAN.md Tier 3).
     *
     * Applies a Chebyshev polynomial in A to a block of vectors so that the
     * wanted end of the spectrum is amplified relative to the rest, then does a
     * Rayleigh-Ritz extraction. Needs no preconditioner and no factorization --
     * only matvecs -- so unlike LOBPCG it does not depend on having a good
     * preconditioner to make progress. Works for dense and CSR input.
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
        // Absolute tolerance on each eigenvalue. Non-positive means "use
        // eps * ||T||", which yields eigenvalues to full working precision.
        T abstol = T(0);
        SortOrder order = SortOrder::Ascending;
        // Safety cap on bisection steps per eigenvalue. The loop also exits on
        // interval convergence, so this only bounds pathological cases.
        int32_t max_iterations = 128;
    };

    /**
     * @brief Computes selected eigenvalues of a batch of symmetric tridiagonal
     *        matrices by bisection on Sturm sequence sign counts.
     *
     * Every eigenvalue is independent of every other, so this is embarrassingly
     * parallel: one work-item bisects one eigenvalue. Unlike QR iteration or
     * divide-and-conquer it can compute a subset at proportionally reduced cost,
     * which is what makes it the tridiagonal kernel for `syevx`. Eigenvalues only;
     * use `stein` for the corresponding eigenvectors.
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
     */
    template <Backend B, typename T>
    size_t stebz_buffer_size(Queue& ctx,
                             size_t n,
                             size_t batch_size,
                             StebzParams<T> params = StebzParams<T>());

    // Forwarding overload taking owning Vectors.
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
        // Inverse iteration steps per vector. With eigenvalues accurate to working
        // precision (as `stebz` produces) two to three steps are sufficient.
        int32_t max_iterations = 3;
        // Eigenvalues closer than ortho_threshold * ||T|| are treated as one
        // cluster and the corresponding vectors are explicitly reorthogonalized.
        // This is the mechanism that keeps inverse iteration usable on clustered
        // spectra; it matches LAPACK dstein's default of 1e-3.
        T ortho_threshold = T(1e-3);
        uint32_t seed = 0x5eed1234u;
    };

    /**
     * @brief Computes eigenvectors of a batch of symmetric tridiagonal matrices by
     *        inverse iteration, given previously computed eigenvalues.
     *
     * Pairs with `stebz`. Each vector is obtained by solving (T - lambda*I) x = b
     * with a tridiagonal LU factorization (partial pivoting), repeated a few times
     * from a pseudo-random start. Vectors whose eigenvalues form a cluster are
     * reorthogonalized against each other afterwards.
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
     * @brief Sentinel meaning "every batch item has all `k` eigenvalues valid",
     *        i.e. exactly the behaviour of the `counts`-less `stein` overload.
     *
     * Spell this rather than a bare `{}` at the `counts` argument: a bare `{}` in
     * an argument position that two overloads both accept has previously selected
     * the wrong overload in this codebase and silently changed results.
     */
    inline constexpr Span<const int32_t> stein_all_counts{};

    /**
     * @brief `stein` with a per-batch-item count of valid eigenvalues.
     *
     * Identical to the overload above except that `k` is a *capacity* -- the number
     * of columns of `Z` and entries of `w` per item -- while `counts[b]` is the
     * number of leading entries of item `b`'s `w` that are real eigenvalues. This is
     * what a `stebz` value range (`EigenRangeType::Value`) produces: the count is
     * data-dependent and differs from one batch item to the next, so the slots
     * `[counts[b], k)` of `w` hold whatever the workspace last contained.
     *
     * Two consequences, both guaranteed here:
     *
     *  - inverse iteration is not run on those invalid shifts, and
     *  - the cluster walk of phase 2 stops at `counts[b]`, so a real eigenvalue is
     *    never grouped with a garbage neighbour.
     *
     * Columns `[counts[b], k)` of `Z` are **written as exactly zero**, not left
     * untouched. Callers may therefore run a uniform-width back-transform over all
     * `k` columns: an orthogonal transform maps zero to zero, so nothing propagates.
     *
     * `counts` is read on the device, so it may be the `m` span `stebz` just wrote;
     * no host synchronization is introduced between the two calls. Pass
     * `stein_all_counts` (or an empty span) for the uniform-`k` behaviour.
     *
     * @param counts Per-item valid prefix length, at least `d.batch_size()` entries;
     *               empty means "all `k` for every item". Values are clamped to
     *               `[0, k]`.
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

    /**
     * @brief Required workspace size, in bytes, for `stein`.
     *
     * Sizes on the capacity `k`, which is what is allocated regardless of any
     * per-item `counts`: every scratch array is indexed by `(batch, column)` over
     * the full `n * k * batch_size` grid whether or not a column is used. So the
     * `counts` overload needs no separate sizing entry point.
     */
    template <Backend B, typename T>
    size_t stein_buffer_size(Queue& ctx,
                             size_t n,
                             size_t k,
                             size_t batch_size,
                             SteinParams<T> params = SteinParams<T>());

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
        //Givens rotations are applied in blocks of this size, increasing this number will lead to excess FLOPs but memory reuse and hence arithmetic intensity improves.
        //Setting this number to 1 is equivalent to full serialization of givens rotation applications, i.e. rotations are applied 1 at a time in the order they were applied to the tridiagonal matrix.
        size_t block_size = 32;
        //Maximum number of sweeps in each Francis QR iteration on average 2-3 iteartions are sufficient to converge to an eigenvalue. 
        size_t max_sweeps = 50; 
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
  
    // Forwarding overload for steqr taking owning Vectors
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

    // Forwarding overload for steqr_cta taking owning Vectors
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
     * This overwrites A with the tridiagonal and reflector storage (SYTD2-style), and
     * returns the diagonal/off-diagonal in (d,e) plus reflector scalars in tau.
     *
     * Notes:
     * - Intended for n <= 32.
     * - `ws` is currently unused but kept for API compatibility.
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
     * Computes Householder vectors (stored in A) and W (workspace) for a block
     * of columns starting at j0. This is the panel factorization stage of the
     * blocked reduction and is useful to benchmark/optimize independently.
     *
     * Notes:
     * - Currently implements only `Uplo::Lower` semantics.
     * - Overwrites A in the same SYTD2-style reflector layout used by sytrd.
     * - Writes only the first `ib` columns of W (W is treated as n x nb).
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
     * Pass pre-sliced views instead of (j0, ib). Typical usage in blocked SYTRD:
     *  - a_panel = A({j0, SliceEnd()}, {j0, SliceEnd()})
     *  - e_panel = E(Slice(j0, j0 + ib))
     *  - tau_panel = TAU(Slice(j0, j0 + ib))
     *  - w_panel = Wmat({j0, SliceEnd()}, {0, ib})
     *
     * Notes:
     * - a_panel must be square.
     * - e_panel/tau_panel size must match w_panel.cols().
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
     * This overwrites A with the tridiagonal and reflector storage (SYTD2-style), and
     * returns the diagonal/off-diagonal in (d,e) plus reflector scalars in tau.
     *
     * Notes:
     * - Intended for n > 32.
     * - Requires an in-order queue.
     * - Uses a blocked panel (LATRD-style) + BLAS-3 trailing update.
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
     * This is analogous to LAPACK xSYTRD_SY2SB: it overwrites `A` with Householder
     * reflector storage and writes the band matrix into `AB` (LAPACK band storage).
     *
     * Band storage (AB):
     *  - `AB` has shape (kd+1) x n.
     *  - For `Uplo::Lower`: AB(1+i-j,j) = A(i,j) for j<=i<=min(n,j+kd).
     *  - For `Uplo::Upper`: AB(kd+1+i-j,j) = A(i,j) for max(1,j-kd)<=i<=j.
     *
     * Reflectors:
     *  - `tau_out` has size (n-kd).
     *
     * Notes:
     *  - Requires an in-order queue.
     *  - Currently only `Uplo::Lower` is implemented.
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
     * This is analogous to LAPACK xSB2ST/xHB2ST (bulge chasing) in the VECT='N' sense:
     * it reduces the band matrix to tridiagonal without forming eigenvectors.
     *
     * Band storage (AB):
     *  - `AB` has shape (kd+1) x n.
     *  - For `Uplo::Lower`: AB(1+i-j,j) = A(i,j) for j<=i<=min(n,j+kd).
     *  - For `Uplo::Upper`: AB(kd+1+i-j,j) = A(i,j) for max(1,j-kd)<=i<=j.
     *
     * Outputs:
    *  - `d_out` has size n (diagonal of the tridiagonal), always real-valued.
    *  - `e_out` has size (n-1) (off-diagonal of the tridiagonal), always real-valued.
    *  - `tau_out` has size (n-1). For the current VECT='N' style implementation this output
    *    is not used by downstream routines and is set to 0.
     *
    * Notes:
    *  - Requires an in-order queue.
    *  - Currently only `Uplo::Lower` is implemented.
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
     * This is the dedicated entrypoint for the new blocked band→tridiagonal reduction
     * algorithm (BANDR1-like schedule).
     *
     * Notes:
     *  - Requires an in-order queue.
     *  - Currently only `Uplo::Lower` is implemented.
     *  - Produces real-valued tridiagonal outputs (d,e). For complex inputs, `e_out` is the
     *    magnitude of the (possibly phased) subdiagonal.
     *  - `tau_out` is currently set to 0 (VECT='N' style).
     */

    struct SytrdBandReductionParams {
        // Number of diagonals to eliminate per sweep (Algorithm 2: d^(i)).
        // d == 0 means use the implementation default schedule for that sweep.
        // If sweeps exceed sequence length, the last value is reused.
        std::vector<int32_t> d_seq{0};

        // Block size per sweep (Algorithm 2: nb^(i)).
        // If sweeps exceed sequence length, the last value is reused.
        std::vector<int32_t> block_size_seq{32};

        // Maximum number of sweeps. max_sweeps < 0 means use the implementation default.
        int32_t max_sweeps = -1;

        // Debug/testing: maximum number of chase steps to execute in the
        // `sytrd_band_reduction_single_step` entrypoint.
        // max_steps <= 0 means run exactly one step.
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
     * This runs a single QR-panel + similarity update (Pre/Sym/Post/Right) on a
     * working band matrix ABw.
     *
     * Inputs:
     *  - ab_in: lower-band storage with rows == kd+1
     * Outputs:
     *  - abw_out: lower-band storage with rows == kd_work+1 (kd_work from params)
     *
     * Notes:
     *  - Requires an in-order queue.
     *  - Currently only `Uplo::Lower` is implemented.
     *  - Intended for unit tests / debugging; not a stable public API.
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
     * Pipeline:
     *  1) sytrd_cta: reduce symmetric A -> tridiagonal (d,e) and Householder reflectors (A,tau)
     *  2) steqr_cta: solve tridiagonal eigenproblem (eigenvalues and optionally eigenvectors)
     *  3) ormqx_cta: back-transform eigenvectors with the Householder reflectors
     *
     * Notes:
     * - Intended for n <= 32.
    * - Supports both real symmetric and complex Hermitian inputs.
     * - Overwrites A with eigenvectors when jobz == EigenVectors.
     * - Eigenvalues are returned in ascending order when SteqrParams::sort is enabled (default).
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
     * Same algorithm as syev_cta -- SYTD2 tridiagonalization, implicit QL/QR on
     * the tridiagonal, Householder back-transform -- but run end to end inside a
     * single sub-group partition instead of as three kernel launches with
     * global-memory intermediates. The reduced tile stays in local memory and
     * doubles as the reflector store for the back-transform, so d, e, tau, the
     * packed reflector matrix and the intermediate eigenvector matrix never
     * reach global memory: traffic is one read of A plus one write of the
     * results.
     *
     * The three stages share their device code with sytrd_cta and steqr_cta
     * (sytrd_cta_device.hh / steqr_cta_device.hh), so results track syev_cta to
     * within the reassociation implied by fusing.
     *
     * Notes:
     * - Intended for n <= 32; supports real symmetric and complex Hermitian input.
     * - Overwrites A with eigenvectors when jobz == EigenVectors; unlike
     *   syev_cta, A is left untouched when jobz == NoEigenVectors.
     * - Eigenvalues are returned in ascending order when SteqrParams::sort is
     *   enabled (default).
     * - Requires no global workspace; ws is accepted for API symmetry and ignored.
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

        // Multiplier on the relative off-diagonal threshold. A rotation is
        // applied to pivot pair (p,q) only when
        //     |a_pq| > tol_multiplier * n * eps * sqrt(|a_pp| * |a_qq|)
        // so the default of 1 gives a threshold of order n*eps, matching the
        // criterion of Demmel & Veselic / LAWN 169. This relative form (rather
        // than the classical absolute test against max|a_kl|) is what yields the
        // high relative accuracy; raising this trades accuracy for sweeps.
        Real tol_multiplier = Real(1);

        // Cap on cyclic sweeps. Convergence normally occurs in well under 10;
        // this is a safety net for pathological inputs.
        size_t max_sweeps = 30;

        // Sort eigenvalues (and permute eigenvectors to match) on output.
        bool sort = true;
        SortOrder sort_order = SortOrder::Ascending;

        // Multiplies the baseline work-group size. Baseline is
        // LCM(P, sub_group_size) where P is the compile-time partition width
        // chosen from n; the result is clamped by the device's maximum
        // work-group size and by available local memory.
        size_t cta_wg_size_multiplier = 1;
    };

    /**
     * @brief CTA-optimized Jacobi symmetric/Hermitian eigen-solver for very small matrices.
     *
     * Cyclic two-sided Jacobi run entirely inside a single sub-group partition:
     * A (and the eigenvector accumulator Z) stay in local memory for the whole
     * solve, so one kernel launch performs the complete eigendecomposition with
     * no intermediate global-memory traffic.
     *
     * Relative to syev_cta (sytrd_cta -> steqr_cta -> ormqx_cta), this trades
     * throughput on generic matrices for high *relative* accuracy on graded or
     * badly scaled input: with the relative stopping criterion the eigenvalue
     * error is governed by the condition number of the column-equilibrated
     * matrix rather than that of the tridiagonalized matrix. Note the underlying
     * theorem (Demmel & Veselic, SIMAX 13(4), 1992) is proved for symmetric
     * positive definite input; indefinite matrices are solved correctly but do
     * not inherit the relative-accuracy bound.
     *
     * Notes:
     * - Intended for n <= 32.
     * - Supports both real symmetric and complex Hermitian inputs.
     * - Overwrites A with eigenvectors when jobz == EigenVectors; A is left
     *   untouched when jobz == NoEigenVectors.
     * - Eigenvalues are returned in ascending order when JacobiParams::sort is
     *   enabled (default).
     * - Requires no global workspace; the ws argument is accepted for API
     *   symmetry and ignored.
     *
     * See JACOBI_EIGENSOLVER_PLAN.md for the design rationale and references.
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
     * Pipeline:
     *  1) sytrd_blocked: reduce dense A -> tridiagonal (d,e) and Householder reflectors (A,tau)
     *  2) stedc: solve tridiagonal eigenproblem (eigenvalues and optionally eigenvectors)
     *  3) ormqr_blocked: back-transform eigenvectors with the Householder reflectors
     *
     * Notes:
     * - Overwrites A with eigenvectors when jobz == EigenVectors.
     * - Currently supports only Uplo::Lower (matches current sytrd_blocked support).
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
     * Pipeline (eigenvalues mode):
     *  1) sytrd_sy2sb: dense -> band
     *  2) sytrd_sb2st: band -> tridiagonal (d,e)
     *  3) stedc: tridiagonal eigensolve
     *
     * Notes:
    * - JobType::EigenVectors is supported via two-stage reduction with
    *   explicit phase/sign recovery and reflector backtransform.
    * - Both modes use a real band width (choose_two_stage_kd, env-overridable
    *   via BATCHLAS_SYEV_TWO_STAGE_KD); eigenvector mode no longer forces kd=1.
     * - Currently supports only Uplo::Lower.
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
     * Computes singular values from bidiagonal coefficients `(d,e)`. The
     * values-only overload writes singular values to `singular_values_out`.
     * The matrix overload additionally accumulates the alternating right and
     * left Givens rotations into `vh` and `u`, matching LAPACK's `BDSQR`
     * contract of returning `P^T * vh` and `u * Q`.
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
    * Current pipeline:
     *  1) GEBRD-style dense -> bidiagonal reduction
    *  2) direct bidiagonal SVD solve via BDSQR by default, with optional
    *     explicit BDSDC selection for blocked mode
     *  3) ORMBR-style backtransforms for full U and V^H
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

    template <typename T>
    struct StedcParams {
        int64_t recursion_threshold = 0; // <=0 uses BatchLAS tuning; otherwise this exact threshold is used
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

    template <Backend B, typename T>
    size_t stedc_flat_workspace_size(Queue& ctx, size_t n, size_t batch_size, JobType jobz, StedcParams<T> params);


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
