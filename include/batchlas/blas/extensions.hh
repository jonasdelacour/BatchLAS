#pragma once
#include <batchlas/export.hh>
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

    #ifndef SYEVSTRUCTS
    #define SYEVSTRUCTS
    /// @addtogroup api_eigen
    /// @{

    /**
     * @brief Options for syevx(): algorithm family, convergence controls, preconditioning
     *        and which part of the spectrum to compute.
     *
     * Every field has a default, so designated initialisers name only what changes. The
     * range fields (`select`, `il`, `iu`, `vl`, `vu`, `abstol`, `order`) mirror LAPACK
     * `?syevx`'s RANGE, IL/IU, VL/VU and ABSTOL arguments.
     *
     * @tparam T scalar type of the matrix (real or complex); tolerances on the real type
     * @see @ref design_syevx_range for the range semantics, @ref perf_syevx for routing
     */
    template <typename T>
    struct SyevxParams {
        using float_type = typename base_type<T>::type;
        SyevxAlgorithm method = SyevxAlgorithm::Auto;      ///< Algorithm family; `Auto` defers to syevx_select_algorithm().
        OrthoAlgorithm algorithm = OrthoAlgorithm::Chol2;  ///< Orthogonalisation used inside the iterative solvers.
        size_t ortho_iterations = 2;                       ///< Orthogonalisation passes per use.
        size_t iterations = 100;                           ///< Cap on outer iterations of LOBPCG / Filtered.
        size_t extra_directions = 0;                       ///< Extra search directions carried beyond `neigs`.
        bool find_largest = true;                          ///< Extremal selection: largest (`true`) or smallest (`false`) eigenpairs.
        T absolute_tolerance = T(std::numeric_limits<float_type>::epsilon());  ///< Absolute residual tolerance of the iterative paths.
        T relative_tolerance = T(std::numeric_limits<float_type>::epsilon());  ///< Relative residual tolerance of the iterative paths.
        /// Caller-owned ILU(k) factor used as preconditioner, or nullptr.
        /// ILU(k) approximates \f$ A^{-1} \f$, so it is valid only for the SMALLEST
        /// eigenpairs: `find_largest = true` with a preconditioner is rejected, not ignored.
        const ILUKPreconditioner<T>* preconditioner = nullptr;
        /// Preconditioner family. `Auto`: ILU(k) when a factor is supplied or requested,
        /// otherwise none (unless `BATCHLAS_SYEVX_PRECONDITIONER` names a default).
        SyevxPreconditioner preconditioner_type = SyevxPreconditioner::Auto;
        /// Build the ILU(k) factor inside syevx from the caller's workspace. Requires a CSR
        /// `A` and `find_largest = false`; mutually exclusive with #preconditioner.
        bool build_preconditioner = false;
        ILUKParams<T> iluk_params{};                       ///< Parameters of the factor built under #build_preconditioner.
        size_t filter_degree = 0;                          ///< Chebyshev degree for SyevxAlgorithm::Filtered; 0 selects a default.
        /// LOBPCG only: power-iteration steps on the random start block; -1 selects the
        /// default, 0 disables. Ignored unless #find_largest is true.
        int init_power_iterations = -1;
        const SyevxInstrumentation<T>* instrumentation = nullptr;  ///< Optional convergence-history sink; nullptr records nothing.

        SyevxSelect select = SyevxSelect::Extremal;        ///< Range kind: `Extremal` (with #find_largest), `Index` or `Value`.

        /// `select == Index`: first wanted index, 0-based and inclusive, into the
        /// ASCENDING spectrum. `il > iu` is an empty request and is rejected.
        int64_t il = 0;
        int64_t iu = -1;                                   ///< `select == Index`: last wanted index, inclusive; `iu < 0` means n-1.
        /// `select == Value`: lower end of the half-open interval (vl, vu], as in LAPACK.
        /// The count is data-dependent per item and is reported through syevx's `m` output.
        float_type vl = float_type(0);
        float_type vu = float_type(0);                     ///< `select == Value`: upper end (inclusive).

        /// Absolute tolerance per eigenvalue; non-positive means \f$ \varepsilon \|T\| \f$.
        /// Forwarded to StebzParams::abstol; ignored by paths that use a full decomposition.
        float_type abstol = float_type(0);

        SortOrder order = SortOrder::Ascending;            ///< Output order for `Index` and `Value`; for `Extremal` it follows #find_largest.
    };

    /**
     * @brief Optional per-iteration convergence histories recorded by the iterative syevx paths.
     *
     * Every history span is laid out `[iter][batch][eig]`. With `iteration_stride` and
     * `batch_stride` zero the strides default to `batch_size * neigs` and `neigs`. Empty
     * spans and null pointers record nothing. Passed through SyevxParams::instrumentation.
     */
    template <typename T>
    struct SyevxInstrumentation {
        using float_type = typename base_type<T>::type;

        Span<float_type> best_residual_history{};         ///< Best residual norm seen so far, per eigenpair.
        Span<float_type> current_residual_history{};      ///< Residual norm of the current iterate (needs #store_current_residual).
        Span<float_type> convergence_rate_history{};      ///< Per-iteration residual reduction (needs #store_convergence_rate).
        Span<float_type> ritz_value_history{};            ///< Ritz values per iteration (needs #store_ritz_values).

        int32_t* iterations_done = nullptr;  ///< Optional per-batch-item iteration count.

        size_t max_iterations = 0;           ///< Rows the history spans can hold.
        size_t store_every = 1;              ///< Record every k-th iteration.
        size_t iteration_stride = 0;         ///< Stride between iterations; 0 means `batch_size * neigs`.
        size_t batch_stride = 0;             ///< Stride between batch items; 0 means `neigs`.

        bool store_current_residual = false; ///< Record #current_residual_history.
        bool store_convergence_rate = true;  ///< Record #convergence_rate_history.
        bool store_ritz_values = false;      ///< Record #ritz_value_history.
    };

    /** @brief Parameters for lanczos(). */
    template <typename T>
    struct LanczosParams {
        using float_type = typename base_type<T>::type;
        OrthoAlgorithm ortho_algorithm = OrthoAlgorithm::CGS2;      ///< Orthogonalisation of each new Lanczos vector.
        size_t ortho_iterations = 2;                                ///< Orthogonalisation passes per vector.
        size_t reorthogonalization_iterations = 2;                  ///< Iterations between full reorthogonalisations.
        bool sort_enabled = true;                                   ///< Sort eigenvalues (and eigenvectors) on output.
        SortOrder sort_order = SortOrder::Ascending;                ///< Order used when #sort_enabled.
    };
    /// @}
    #endif

    /// @addtogroup api_qr
    /// @{

    /**
     * @brief Orthonormalises the columns (or rows) of each batch item in place.
     *
     * With `transA == NoTrans` each n-column block of `A` is replaced by an orthonormal
     * basis of its column space, \f$ A \leftarrow Q \f$ with \f$ Q^H Q = I \f$; with
     * `Trans` the rows are orthonormalised instead. Asynchronous.
     *
     * @param ctx       queue the work is enqueued on
     * @param A         batch of m x n matrices; overwritten with the orthonormal vectors
     * @param transA    `NoTrans` for columns, `Trans` for rows
     * @param workspace at least ortho_buffer_size() bytes for the same `algo`
     * @param algo      orthogonalisation algorithm (see OrthoAlgorithm)
     * @return event of the last enqueued kernel
     * @note The workspace depends on `algo`: size it with the algorithm the call runs.
     */
    template <Backend B, typename T>
    BATCHLAS_API Event ortho(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A, //A is overwritten with orthogonal vectors
                         Transpose transA,
                         Span<std::byte> workspace,
                         OrthoAlgorithm algo = OrthoAlgorithm::Chol2);

    /**
     * @brief Orthonormalises `A` against the columns of an external basis `M`, in place.
     *
     * Each of `iterations` passes projects out the span of `M`,
     * \f$ A \leftarrow A - M (M^H A) \f$, and then orthonormalises `A` with `algo`.
     *
     * @param ctx        queue the work is enqueued on
     * @param A          batch of vectors to orthonormalise; overwritten
     * @param M          external basis, read only
     * @param transA     `NoTrans`: the vectors are the columns of `A`; transposed: its rows
     * @param transM     `NoTrans`: the basis vectors are the columns of `M`; transposed: its rows
     * @param workspace  at least ortho_buffer_size() bytes for the same arguments
     * @param algo       orthogonalisation algorithm run after each projection
     * @param iterations number of project-then-orthonormalise passes
     * @return event of the last enqueued kernel
     * @pre `M`'s vectors are orthonormal, and the vector counts of `A` and `M` sum to at
     *      most the vector length.
     * @throws batchlas::invalid_argument if the vector counts exceed the vector length
     */
    template <Backend B, typename T>
    BATCHLAS_API Event ortho(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A, //A is overwritten with orthogonal vectors
                         const MatrixView<T, MatrixFormat::Dense>& M, //External metric
                         Transpose transA,
                         Transpose transM,
                         Span<std::byte> workspace,
                         OrthoAlgorithm algo = OrthoAlgorithm::Chol2,
                         size_t iterations = 2);

    /** @brief Required workspace, in bytes, for ortho(); arguments as for the call itself. */
    template <Backend B, typename T>
    BATCHLAS_API size_t ortho_buffer_size(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         Transpose transA,
                         OrthoAlgorithm algo = OrthoAlgorithm::Chol2);

    /** @brief Required workspace, in bytes, for the external-basis ortho(); same arguments. */
    template <Backend B, typename T>
    BATCHLAS_API size_t ortho_buffer_size(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         const MatrixView<T, MatrixFormat::Dense>& M,
                         Transpose transA,
                         Transpose transM,
                         OrthoAlgorithm algo = OrthoAlgorithm::Chol2,
                         size_t iterations = 2);

    /// @}

    /// @addtogroup api_eigen
    /// @{

    /**
     * @brief Selected eigenpairs of a batch of symmetric/Hermitian matrices, dense or CSR.
     *
     * Computes \f$ A v = \lambda v \f$ for the part of the spectrum that
     * `params.select` names: the `neigs` largest or smallest eigenpairs (`Extremal`, with
     * SyevxParams::find_largest) or an index block (`Index`). A value interval
     * (`Value`) has a data-dependent count and needs the overload that reports `m`.
     *
     * The algorithm comes from syevx_select_algorithm(): `Direct` (full syev() on a private
     * copy, then selection), `DirectSubset` (two-stage reduction plus stebz()/stein()),
     * `LOBPCG` or `Filtered` (Chebyshev subspace iteration). `A` is never modified; the
     * direct paths read its lower triangle. Asynchronous: returns once the work is enqueued.
     *
     * @tparam MFormat  `MatrixFormat::Dense` or `MatrixFormat::CSR` (CSR always runs LOBPCG
     *                  or Filtered)
     * @param ctx       queue the work is enqueued on
     * @param A         batch of n x n symmetric/Hermitian matrices, dense or CSR
     * @param W         eigenvalue output (real type), `neigs` entries per batch item
     * @param neigs     CAPACITY of `W` and of `V`'s columns per item, not the number produced
     * @param workspace at least syevx_buffer_size() bytes
     * @param jobz      whether eigenvectors are written to `V`
     * @param V         n x `neigs` eigenvector output per item; used only for `EigenVectors`
     * @param params    selection range and algorithm parameters
     * @param info      per-item convergence status (0 = converged, > 0 LAPACK-like), or an
     *                  empty span to not request it; see @ref md_docs_2cpp-api
     * @return event of the last enqueued kernel
     * @throws batchlas::invalid_argument for `SyevxSelect::Value` (use the `m`-taking
     *         overload); an `Index` block outside [0, n) or with `neigs != iu - il + 1`;
     *         a non-extremal range with CSR input or an explicit LOBPCG/Filtered `method`;
     *         an ILU(k) or `Jacobi` preconditioner with `find_largest`, or conflicting
     *         preconditioner fields.
     * @see @ref perf_syevx, @ref design_syevx_range
     */
    template <Backend B, typename T, MatrixFormat MFormat>
    BATCHLAS_API Event syevx(Queue& ctx,
                             const MatrixView<T, MFormat>& A,
                             Span<typename base_type<T>::type> W,
                             size_t neigs,
                             Span<std::byte> workspace,
                             JobType jobz = JobType::NoEigenVectors,
                             const MatrixView<T, MatrixFormat::Dense>& V = MatrixView<T, MatrixFormat::Dense>(),
                             const SyevxParams<T>& params = SyevxParams<T>(),
                             Span<int32_t> info = Span<int32_t>());

    /** @brief syevx() on an owning `A` without eigenvectors: no `V` argument. */
    template <Backend B, typename T, MatrixFormat MFormat>
    inline Event syevx(Queue& ctx,
                const Matrix<T, MFormat>& A,
                Span<typename base_type<T>::type> W,
                size_t neigs,
                Span<std::byte> workspace,
                JobType jobz = JobType::NoEigenVectors,
                const SyevxParams<T>& params = SyevxParams<T>(),
                Span<int32_t> info = Span<int32_t>()) {
        return syevx<B,T,MFormat>(ctx, MatrixView<T, MFormat>(A), W, neigs, workspace, jobz, MatrixView<T, MatrixFormat::Dense>(), params, info);
    }

    /**
     * @brief syevx() with a per-batch-item count of eigenpairs actually found.
     *        Required for `SyevxSelect::Value`.
     *
     * Item `b` gets `min(m[b], neigs)` eigenpairs (the lowest ones, when truncating), the
     * rest of its `W` is left untouched, the remaining columns of its `V` are written as
     * EXACTLY ZERO so the back-transform can run over a uniform column count, and `m[b]`
     * reports the TRUE count, so `m[b] > neigs` is the caller's overflow signal.
     * Other parameters as for the `m`-less form.
     *
     * @param m Per-item count, at least `A.batch_size()` entries. Device-writable.
     *
     * @trap Unambiguous against the `m`-less forms only because `Span`'s scalar
     *       constructor is `explicit`; a bare `{}` in positions 4-6 matches both and is
     *       ambiguous.
     */
    template <Backend B, typename T, MatrixFormat MFormat>
    BATCHLAS_API Event syevx(Queue& ctx,
                             const MatrixView<T, MFormat>& A,
                             Span<typename base_type<T>::type> W,
                             Span<int32_t> m,
                             size_t neigs,
                             Span<std::byte> workspace,
                             JobType jobz = JobType::NoEigenVectors,
                             const MatrixView<T, MatrixFormat::Dense>& V = MatrixView<T, MatrixFormat::Dense>(),
                             const SyevxParams<T>& params = SyevxParams<T>(),
                             Span<int32_t> info = Span<int32_t>());

    /** @brief The `m`-reporting syevx() on an owning `A` without eigenvectors. */
    template <Backend B, typename T, MatrixFormat MFormat>
    inline Event syevx(Queue& ctx,
                const Matrix<T, MFormat>& A,
                Span<typename base_type<T>::type> W,
                Span<int32_t> m,
                size_t neigs,
                Span<std::byte> workspace,
                JobType jobz = JobType::NoEigenVectors,
                const SyevxParams<T>& params = SyevxParams<T>(),
                Span<int32_t> info = Span<int32_t>()) {
        return syevx<B,T,MFormat>(ctx, MatrixView<T, MFormat>(A), W, m, neigs, workspace, jobz, MatrixView<T, MatrixFormat::Dense>(), params, info);
    }

    /**
     * @brief Required workspace, in bytes, for syevx(); arguments as for the call minus
     *        the workspace. Unlike the solve, this accepts `SyevxSelect::Value`: sizing
     *        writes no counts.
     */
    template <Backend B, typename T, MatrixFormat MFormat>
    BATCHLAS_API size_t syevx_buffer_size(Queue& ctx,
                             const MatrixView<T, MFormat>& A,
                             Span<typename base_type<T>::type> W,
                             size_t neigs,
                             JobType jobz = JobType::NoEigenVectors,
                             const MatrixView<T, MatrixFormat::Dense>& V = MatrixView<T, MatrixFormat::Dense>(),
                             const SyevxParams<T>& params = SyevxParams<T>());

    /** @brief syevx_buffer_size() on an owning `A` without eigenvectors. */
    template <Backend B, typename T, MatrixFormat MFormat>
    inline size_t syevx_buffer_size(Queue& ctx,
                const Matrix<T, MFormat>& A,
                Span<typename base_type<T>::type> W,
                size_t neigs,
                JobType jobz = JobType::NoEigenVectors,
                const SyevxParams<T>& params = SyevxParams<T>()) {
        return syevx_buffer_size<B,T,MFormat>(ctx, MatrixView<T, MFormat>(A), W, neigs, jobz, MatrixView<T, MatrixFormat::Dense>(), params);
    }

    /**
     * @brief syevx_buffer_size() with the solve's `m` argument, which is ACCEPTED AND
     *        IGNORED, so a value-range caller writes sizing and solve with one argument list.
     */
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

    /** @brief The `m`-taking syevx_buffer_size() on an owning `A`; `m` is ignored. */
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

    /**
     * @brief Resolved, algorithm-independent description of the caller's request.
     *
     * Solvers and `*_buffer_size` must derive their behaviour from this struct, never from
     * SyevxParams::select / il / iu / find_largest, so solve and sizing cannot disagree.
     * `vl`/`vu` are deliberately absent: they are forwarded verbatim from `params`.
     */
    struct SyevxResolvedRange {
        bool    value_range;  ///< true: (vl, vu]; false: the index block [il, iu]
        int64_t il;           ///< valid iff !value_range; 0-based, inclusive
        int64_t iu;           ///< valid iff !value_range; 0-based, inclusive
        /// Upper bound on eigenpairs per item, clamped to [0, n] so a consumer may index
        /// [il, il + max_count) unguarded. Exact for an index block; a value range may exceed it.
        int64_t max_count;
        bool    reverse;      ///< write the selected block in descending order
    };

    /**
     * @brief Normalizes a range request into a SyevxResolvedRange. Legality is NOT checked
     *        here; it clamps rather than throws so it stays usable when sizing.
     *
     * @param n     Matrix dimension
     * @param neigs Capacity of W and V per batch item (see `syevx`)
     * @param select      SyevxParams::select
     * @param find_largest SyevxParams::find_largest (Extremal only)
     * @param il,iu       SyevxParams::il / iu (Index only; iu < 0 means n-1)
     * @param order       SyevxParams::order (Index and Value only)
     * @return the normalised request
     */
    BATCHLAS_API SyevxResolvedRange syevx_resolve_range(int64_t n,
                                                        size_t neigs,
                                                        SyevxSelect select,
                                                        bool find_largest,
                                                        int64_t il,
                                                        int64_t iu,
                                                        SortOrder order);

    /// @brief syevx_resolve_range() reading the range fields from `params`; distinguished
    ///        from the 7-argument form by arity.
    template <typename T>
    inline SyevxResolvedRange syevx_resolve_range(int64_t n,
                                                  size_t neigs,
                                                  const SyevxParams<T>& params) {
        return syevx_resolve_range(n, neigs, params.select, params.find_largest,
                                   params.il, params.iu, params.order);
    }

    /**
     * @brief Resolves SyevxParams::method (and the BATCHLAS_SYEVX_ALGORITHM override) to a
     *        concrete, implemented algorithm. Never returns `Auto`, and is deterministic
     *        so that `syevx` and `syevx_buffer_size` always agree on the choice.
     *
     * @param format Matrix format of A (sparse formats run LOBPCG unless `Filtered` is
     *        requested explicitly)
     * @param n Matrix dimension
     * @param neigs Number of requested eigenpairs
     * @param requested Algorithm requested via SyevxParams::method
     * @param subset_supported Whether DirectSubset is available for this T/format
     * @param jobz Whether eigenvectors are wanted; load-bearing for the choice
     * @param batch_size Load-bearing: the subset solver starves at small batch
     * @param select Excludes the algorithms that cannot answer the requested range
     * @return the algorithm syevx() will run
     * @throws batchlas::invalid_argument for sparse input, or an explicit LOBPCG/Filtered
     *         `method`, with a non-extremal range. The same request made through
     *         `BATCHLAS_SYEVX_ALGORITHM` degrades to `Direct` with a one-time warning.
     * @see @ref perf_syevx for the measured thresholds behind the choice
     */
    BATCHLAS_API SyevxAlgorithm syevx_select_algorithm(MatrixFormat format,
                                                       int64_t n,
                                                       size_t neigs,
                                                       SyevxAlgorithm requested,
                                                       bool subset_supported,
                                                       JobType jobz = JobType::EigenVectors,
                                                       int64_t batch_size = 1,
                                                       SyevxSelect select = SyevxSelect::Extremal);

    /**
     * @brief Resolves SyevxParams::preconditioner_type to a concrete family. Never returns
     *        `Auto`; deterministic so solve and sizing agree on the pool allocation.
     *
     * @param requested SyevxParams::preconditioner_type
     * @param iluk_configured Whether an ILU(k) factor was supplied or requested
     * @param find_largest An environment default of `Jacobi` degrades to `None` when this
     *        is true; an explicit request is returned as is (syevx() rejects it earlier).
     * @return the preconditioner family syevx() will build or use
     */
    SyevxPreconditioner syevx_select_preconditioner(SyevxPreconditioner requested,
                                                    bool iluk_configured,
                                                    bool find_largest);

    /**
     * @brief Partial eigensolve by full decomposition followed by selection. Runs `syev`
     *        on a private copy of A. Dense input only, but every scalar type and every
     *        `SyevxSelect` range: the universal fallback.
     *
     * @param W Eigenvalue output, stride `neigs`; entries past `min(m[b], neigs)` untouched
     * @param V Eigenvector output. Columns past `min(m[b], neigs)` are written as EXACTLY
     *        ZERO -- unlike `W` -- so this path and `syevx_direct_subset` agree.
     * @param m Per-item count in the requested range, or an empty span to not report it.
     *        `m[b] > neigs` is the truncation signal; the LOWEST `neigs` are kept.
     *
     * Other parameters as for syevx().
     */
    template <Backend B, typename T, MatrixFormat MFormat>
    BATCHLAS_API Event syevx_direct(Queue& ctx,
                             const MatrixView<T, MFormat>& A,
                             Span<typename base_type<T>::type> W,
                             Span<int32_t> m,
                             size_t neigs,
                             Span<std::byte> workspace,
                             JobType jobz,
                             const MatrixView<T, MatrixFormat::Dense>& V,
                             const SyevxParams<T>& params,
                             Span<int32_t> info = Span<int32_t>());

    /**
     * @brief `syevx_direct` without the `m` output; `Extremal` and `Index` only.
     *        Distinguished from the form above by ARITY, so no `{}` trap can fire.
     */
    template <Backend B, typename T, MatrixFormat MFormat>
    inline Event syevx_direct(Queue& ctx,
                const MatrixView<T, MFormat>& A,
                Span<typename base_type<T>::type> W,
                size_t neigs,
                Span<std::byte> workspace,
                JobType jobz,
                const MatrixView<T, MatrixFormat::Dense>& V,
                const SyevxParams<T>& params,
                Span<int32_t> info = Span<int32_t>()) {
        return syevx_direct<B, T, MFormat>(ctx, A, W, Span<int32_t>(), neigs, workspace,
                                           jobz, V, params, info);
    }

    /**
     * @brief Required workspace, in bytes, for syevx_direct(). Takes no `m`: sizing writes
     *        no counts, and this path's workspace is range-independent.
     */
    template <Backend B, typename T, MatrixFormat MFormat>
    BATCHLAS_API size_t syevx_direct_buffer_size(Queue& ctx,
                             const MatrixView<T, MFormat>& A,
                             Span<typename base_type<T>::type> W,
                             size_t neigs,
                             JobType jobz,
                             const MatrixView<T, MatrixFormat::Dense>& V,
                             const SyevxParams<T>& params);

    /**
     * @brief Partial eigensolve by two-stage reduction plus a subset tridiagonal solve
     *        (`stebz` + `stein`) and a narrowed back-transform. Real scalars and dense
     *        input only; zero capacity is rejected (stein requires k >= 1).
     *
     * `W`, `V` and `m` follow the same contract as `syevx_direct`, with `stein` doing the
     * zeroing. The descending reversal is applied last, in the finalize kernel: `stein`'s
     * cluster detection walks consecutive eigenvalues and requires ascending input.
     * Other parameters as for syevx().
     *
     * @pre in-order Queue; `A` square
     * @throws batchlas::unsupported for complex or sparse input
     * @throws batchlas::invalid_argument for `neigs == 0`, an out-of-range index block,
     *         `vl >= vu`, a too-short `m`, or an out-of-order Queue
     * @see syevx_direct_subset_supported()
     */
    template <Backend B, typename T, MatrixFormat MFormat>
    BATCHLAS_API Event syevx_direct_subset(Queue& ctx,
                             const MatrixView<T, MFormat>& A,
                             Span<typename base_type<T>::type> W,
                             Span<int32_t> m,
                             size_t neigs,
                             Span<std::byte> workspace,
                             JobType jobz,
                             const MatrixView<T, MatrixFormat::Dense>& V,
                             const SyevxParams<T>& params,
                             Span<int32_t> info = Span<int32_t>());

    /** @brief `syevx_direct_subset` without the `m` output; arity-disambiguated as above. */
    template <Backend B, typename T, MatrixFormat MFormat>
    inline Event syevx_direct_subset(Queue& ctx,
                const MatrixView<T, MFormat>& A,
                Span<typename base_type<T>::type> W,
                size_t neigs,
                Span<std::byte> workspace,
                JobType jobz,
                const MatrixView<T, MatrixFormat::Dense>& V,
                const SyevxParams<T>& params,
                Span<int32_t> info = Span<int32_t>()) {
        return syevx_direct_subset<B, T, MFormat>(ctx, A, W, Span<int32_t>(), neigs, workspace,
                                                  jobz, V, params, info);
    }

    /**
     * @brief Required workspace, in bytes, for syevx_direct_subset().
     *
     * Sizing writes no counts, but it is NOT range-independent: a `Value` range needs
     * room for up to n eigenvalues per item, so the sizes come from syevx_resolve_range().
     */
    template <Backend B, typename T, MatrixFormat MFormat>
    BATCHLAS_API size_t syevx_direct_subset_buffer_size(Queue& ctx,
                             const MatrixView<T, MFormat>& A,
                             Span<typename base_type<T>::type> W,
                             size_t neigs,
                             JobType jobz,
                             const MatrixView<T, MatrixFormat::Dense>& V,
                             const SyevxParams<T>& params);

    /** @brief Whether `syevx_direct_subset` supports this scalar type and format. */
    template <typename T, MatrixFormat MFormat>
    inline constexpr bool syevx_direct_subset_supported() {
        return MFormat == MatrixFormat::Dense && std::is_same_v<T, typename base_type<T>::type>;
    }

    /**
     * @brief Partial eigensolve by LOBPCG (locally optimal block preconditioned conjugate
     *        gradient). Dense and CSR input; extremal ranges only.
     *
     * Iterates a block of `neigs` (+ SyevxParams::extra_directions) vectors until the
     * residuals meet the tolerances or SyevxParams::iterations is reached; optionally
     * preconditioned by ILU(k) for the smallest eigenpairs. Parameters as for syevx().
     */
    template <Backend B, typename T, MatrixFormat MFormat>
    BATCHLAS_API Event syevx_lobpcg(Queue& ctx,
                             const MatrixView<T, MFormat>& A,
                             Span<typename base_type<T>::type> W,
                             size_t neigs,
                             Span<std::byte> workspace,
                             JobType jobz,
                             const MatrixView<T, MatrixFormat::Dense>& V,
                             const SyevxParams<T>& params,
                             Span<int32_t> info = Span<int32_t>());

    /** @brief Required workspace, in bytes, for syevx_lobpcg(). */
    template <Backend B, typename T, MatrixFormat MFormat>
    BATCHLAS_API size_t syevx_lobpcg_buffer_size(Queue& ctx,
                             const MatrixView<T, MFormat>& A,
                             Span<typename base_type<T>::type> W,
                             size_t neigs,
                             JobType jobz,
                             const MatrixView<T, MatrixFormat::Dense>& V,
                             const SyevxParams<T>& params);

    /**
     * @brief Chebyshev-filtered subspace iteration: amplifies the wanted end of the
     *        spectrum, then Rayleigh-Ritz. Needs only matvecs; dense and CSR input,
     *        extremal ranges only.
     *
     * The polynomial degree is SyevxParams::filter_degree (0 picks a default).
     * Parameters as for syevx().
     */
    template <Backend B, typename T, MatrixFormat MFormat>
    BATCHLAS_API Event syevx_filtered(Queue& ctx,
                             const MatrixView<T, MFormat>& A,
                             Span<typename base_type<T>::type> W,
                             size_t neigs,
                             Span<std::byte> workspace,
                             JobType jobz,
                             const MatrixView<T, MatrixFormat::Dense>& V,
                             const SyevxParams<T>& params,
                             Span<int32_t> info = Span<int32_t>());

    /** @brief Required workspace, in bytes, for syevx_filtered(). */
    template <Backend B, typename T, MatrixFormat MFormat>
    BATCHLAS_API size_t syevx_filtered_buffer_size(Queue& ctx,
                             const MatrixView<T, MFormat>& A,
                             Span<typename base_type<T>::type> W,
                             size_t neigs,
                             JobType jobz,
                             const MatrixView<T, MatrixFormat::Dense>& V,
                             const SyevxParams<T>& params);

    /**
     * @brief All eigenvalues, and optionally eigenvectors, of a batch of symmetric/Hermitian
     *        matrices by a full-length Lanczos tridiagonalisation.
     *
     * Builds an n-step Krylov basis with reorthogonalisation, then solves the resulting
     * tridiagonal problem. Intended for sparse (CSR) input; dense works too.
     *
     * @param ctx       queue the work is enqueued on
     * @param A         batch of n x n symmetric/Hermitian matrices (CSR or dense); read only
     * @param W         eigenvalue output (real type), n entries per batch item
     * @param workspace at least lanczos_buffer_size() bytes
     * @param jobz      whether eigenvectors are written to `V`
     * @param V         n x n eigenvector output per item; used only for `EigenVectors`
     * @param params    orthogonalisation and sorting options
     * @return event of the last enqueued kernel
     */
    template <Backend B, typename T, MatrixFormat MFormat>
    BATCHLAS_API Event lanczos(Queue& ctx,
                     const MatrixView<T, MFormat>& A,
                     Span<typename base_type<T>::type> W,
                     Span<std::byte> workspace,
                     JobType jobz = JobType::NoEigenVectors,
                     const MatrixView<T, MatrixFormat::Dense>& V = MatrixView<T, MatrixFormat::Dense>(),
                     const LanczosParams<T>& params = LanczosParams<T>());

    /** @brief lanczos() on an owning `A` without eigenvectors: no `V` argument. */
    template <Backend B, typename T, MatrixFormat MFormat>
    inline Event lanczos(Queue& ctx,
        const Matrix<T, MFormat>& A,
        Span<typename base_type<T>::type> W,
        Span<std::byte> workspace,
        JobType jobz = JobType::NoEigenVectors,
        const LanczosParams<T>& params = LanczosParams<T>()) {
        return lanczos<B,T,MFormat>(ctx, MatrixView<T,MFormat>(A), W, workspace, jobz, MatrixView<T, MatrixFormat::Dense>(), params);
    }

    /** @brief Required workspace, in bytes, for lanczos(); arguments as for the call itself. */
    template <Backend B, typename T, MatrixFormat MFormat>
    BATCHLAS_API size_t lanczos_buffer_size(Queue& ctx,
                     const MatrixView<T, MFormat>& A,
                     Span<typename base_type<T>::type> W,
                     JobType jobz = JobType::NoEigenVectors,
                     const MatrixView<T, MatrixFormat::Dense>& V = MatrixView<T, MatrixFormat::Dense>(),
                     const LanczosParams<T>& params = LanczosParams<T>());

    /// @}

    /// @addtogroup api_tridiag
    /// @{

    /**
     * @brief Eigenvalues, and optionally eigenvectors, of a batch of small symmetric
     *        tridiagonal matrices by shifted QR with 2x2 reflections.
     *
     * One work-group of 32 per batch item; the working arrays live in local memory when
     * they fit and in the workspace otherwise. With `EigenVectors`, `Q` is first set to
     * the identity.
     *
     * @param ctx        queue the work is enqueued on
     * @param alphas     diagonals, packed: item `b` at `alphas[b*n .. b*n+n)`
     * @param betas      off-diagonals, packed with the same stride `n`
     * @param W          eigenvalue output (real type), packed with stride `n`
     * @param workspace  at least tridiagonal_solver_buffer_size() bytes
     * @param jobz       whether `Q` receives the eigenvectors
     * @param Q          n x n eigenvector output per item
     * @param n          order of each tridiagonal matrix
     * @param batch_size number of batch items
     * @return event of the kernel
     * @pre `Q` is packed (`ld() == n`): the rotation update addresses it with stride `n`.
     * @note Each eigenvalue gets at most six QR steps and no convergence status is
     *       reported; steqr() and stedc() are the maintained tridiagonal solvers.
     */
    template <Backend B, typename T>
    BATCHLAS_API Event tridiagonal_solver(Queue& ctx,
                     Span<T> alphas,
                     Span<T> betas,
                     Span<typename base_type<T>::type> W,
                     Span<std::byte> workspace,
                     JobType jobz,
                     const MatrixView<T, MatrixFormat::Dense>& Q,
                     size_t n,
                     size_t batch_size);

    /** @brief Required workspace, in bytes, for tridiagonal_solver(). */
    template <Backend B, typename T>
    BATCHLAS_API size_t tridiagonal_solver_buffer_size(Queue& ctx, size_t n, size_t batch_size, JobType jobz);

    /**
     * @brief One or more implicit Francis QR sweeps over a batch of tridiagonals (d, e).
     * @warning Declared only: the library defines and instantiates no `francis_sweep`, so a
     *          call fails at link time.
     */
    template <typename T>
    BATCHLAS_API Event francis_sweep(Queue& ctx, const VectorView<T>& d, const VectorView<T>& e, const MatrixView<std::array<T,2>, MatrixFormat::Dense>& givens_rotations = {}, size_t n_sweeps = 1, T zero_threshold = std::numeric_limits<T>::epsilon());

    // Convention in the tridiagonal group below: `VectorView<T>` for one vector per batch
    // item (it carries inc/stride/batch_size), `Span<...>` for flat per-item arrays and byte
    // workspaces. Demoting a `VectorView` to a `Span` silently drops the stride.

    /** @brief How stebz() selects a subset of the spectrum. */
    enum class EigenRangeType {
        All,    ///< Every eigenvalue
        Index,  ///< Eigenvalues il..iu inclusive, 0-based, in ascending order
        Value   ///< Eigenvalues in the half-open interval (vl, vu]
    };

    /** @brief Parameters for stebz() (bisection on a symmetric tridiagonal matrix). */
    template <typename T>
    struct StebzParams {
        EigenRangeType range = EigenRangeType::All;  ///< Which eigenvalues to compute.
        int64_t il = 0;    ///< First wanted index (0-based, inclusive), range == Index
        int64_t iu = -1;   ///< Last wanted index (0-based, inclusive), range == Index; < 0 means n-1
        T vl = T(0);       ///< Lower bound (exclusive), range == Value
        T vu = T(0);       ///< Upper bound (inclusive), range == Value
        /// Absolute tolerance on each eigenvalue. Non-positive means
        /// \f$ \varepsilon \|T\| \f$, i.e. full working precision.
        T abstol = T(0);
        SortOrder order = SortOrder::Ascending;  ///< Output order of `w`.
        /// Safety cap on bisection steps per eigenvalue; the loop also exits on
        /// interval convergence.
        int32_t max_iterations = 128;
    };

    /**
     * @brief Computes selected eigenvalues of a batch of symmetric tridiagonal matrices by
     *        bisection on Sturm sequence sign counts. Eigenvalues only; one work-item per
     *        eigenvalue, so a subset costs proportionally less. Use stein() for vectors.
     *
     * @param ctx Execution context/device queue
     * @param d Diagonal, n entries per batch item
     * @param e Off-diagonal, n-1 entries per batch item
     * @param w Output eigenvalues; must hold at least the number selected (n for `Value`)
     * @param m Output count of eigenvalues found, per batch item (device-written)
     * @param ws Pre-allocated workspace buffer, at least stebz_buffer_size() bytes
     * @param params Selection range, tolerance and ordering
     * @return Event Event to track operation completion
     * @throws batchlas::invalid_argument for n < 1, a short `e`, `m` or `w`, or an invalid
     *         index range
     */
    template <Backend B, typename T>
    BATCHLAS_API Event stebz(Queue& ctx,
                             const VectorView<T>& d,
                             const VectorView<T>& e,
                             const VectorView<T>& w,
                             Span<int32_t> m,
                             const Span<std::byte>& ws,
                             StebzParams<T> params = StebzParams<T>());

    /**
     * @brief Required workspace size, in bytes, for stebz().
     *
     * @trap `params` is REQUIRED, not defaulted: it is the only argument carrying `T`, and
     *       defaulting it silently drops the queue-deducing overload. Pass `StebzParams<T>{}`.
     */
    template <Backend B, typename T>
    BATCHLAS_API size_t stebz_buffer_size(Queue& ctx,
                                          size_t n,
                                          size_t batch_size,
                                          StebzParams<T> params);

    /** @brief Parameters for stein() (inverse iteration on a symmetric tridiagonal). */
    template <typename T>
    struct SteinParams {
        /// Fixed number of inverse-iteration steps; two or three suffice for eigenvalues
        /// accurate to working precision. No convergence test is made.
        int32_t max_iterations = 3;
        /// Eigenvalues closer than `ortho_threshold * ||T||` form one cluster and have their
        /// vectors explicitly reorthogonalised (LAPACK dstein uses 1e-3).
        T ortho_threshold = T(1e-3);
        uint32_t seed = 0x5eed1234u;  ///< Seed of the random starting vectors.
    };

    /**
     * @brief Computes eigenvectors of a batch of symmetric tridiagonal matrices by inverse
     *        iteration, given previously computed eigenvalues. Pairs with stebz().
     *
     * @param ctx Execution context/device queue
     * @param d Diagonal, n entries per batch item
     * @param e Off-diagonal, n-1 entries per batch item
     * @param w Eigenvalues, k per batch item, in ascending order (cluster detection walks
     *          consecutive values)
     * @param k Number of eigenvectors to compute, at least 1
     * @param Z Output eigenvectors, n x k per batch item, columns matching w
     * @param ws Pre-allocated workspace buffer, at least stein_buffer_size() bytes
     * @param params Iteration count and clustering threshold
     * @return Event Event to track operation completion
     * @throws batchlas::invalid_argument for n < 1, k < 1, or a short `e`, `w` or `Z`
     */
    template <Backend B, typename T>
    BATCHLAS_API Event stein(Queue& ctx,
                             const VectorView<T>& d,
                             const VectorView<T>& e,
                             const VectorView<T>& w,
                             size_t k,
                             const MatrixView<T, MatrixFormat::Dense>& Z,
                             const Span<std::byte>& ws,
                             SteinParams<T> params = SteinParams<T>());

    /**
     * @brief Sentinel meaning: every batch item has all `k` eigenvalues valid.
     *
     * @trap Spell this rather than a bare `{}` at `counts`: `{}` in a position two
     *       overloads both accept has silently selected the wrong one in this codebase before.
     */
    inline constexpr Span<const int32_t> stein_all_counts{};

    /**
     * @brief stein() with a per-batch-item count of valid eigenvalues.
     *
     * `k` is a capacity (columns of `Z`, entries of `w` per item); `counts[b]` is how many
     * leading entries of item `b`'s `w` are real eigenvalues -- the rest hold stale
     * workspace, and are neither iterated on nor joined to a cluster. Columns
     * `[counts[b], k)` of `Z` are written as EXACTLY ZERO, not left untouched, so callers
     * may back-transform a uniform `k` columns. `counts` is read on the device, so it may
     * be the `m` span `stebz` just wrote; empty (or `stein_all_counts`) means all `k` are valid,
     * and it is clamped to `[0, k]`. Other parameters as for the `counts`-less form.
     *
     * @throws batchlas::invalid_argument additionally if a non-empty `counts` is shorter
     *         than the batch
     */
    template <Backend B, typename T>
    BATCHLAS_API Event stein(Queue& ctx,
                             const VectorView<T>& d,
                             const VectorView<T>& e,
                             const VectorView<T>& w,
                             size_t k,
                             Span<const int32_t> counts,
                             const MatrixView<T, MatrixFormat::Dense>& Z,
                             const Span<std::byte>& ws,
                             SteinParams<T> params = SteinParams<T>());

    /**
     * @brief Required workspace size, in bytes, for stein().
     *
     * Sizes on the capacity `k`, so the `counts` overload needs no sizing entry of its own.
     * @trap `params` is REQUIRED, as for stebz_buffer_size(): it is the only argument
     *       carrying `T`.
     */
    template <Backend B, typename T>
    BATCHLAS_API size_t stein_buffer_size(Queue& ctx,
                                          size_t n,
                                          size_t k,
                                          size_t batch_size,
                                          SteinParams<T> params);

    /** @brief Shift used by the CTA STEQR's implicit QR/QL steps. @see @ref algo_steqr */
    enum class SteqrShiftStrategy {
        Lapack = 0,     ///< LAPACK-style implicit shift (the stable dsteqr formulation).
        Wilkinson = 1,  ///< Wilkinson shift computed from the trailing 2x2 block.
    };

    /**
     * @brief Update scheme of the CTA STEQR's implicit QR/QL steps.
     *
     * The default (SteqrParams::cta_update_scheme) is `EXP`.
     * @see @ref algo_steqr, section "steqr: the CTA update schemes EXP and PG"
     */
    enum class SteqrUpdateScheme {
        PG = 0,   ///< Parlett-Gray style scalar recurrence.
        EXP = 1,  ///< Explicit similarity update mirroring steqr.cc's bulge-chasing math.
    };

    /** @brief Parameters for steqr() and steqr_cta(), and for stedc()'s leaf solves. */
    template <typename T>
    struct SteqrParams {
        /// Rotations are applied in blocks of this size; larger means excess FLOPs but
        /// better memory reuse. 1 fully serialises them.
        size_t block_size = 32;
        size_t max_sweeps = 50;  ///< Cap on Francis QR sweeps; 2-3 typically suffice per eigenvalue.
        T zero_threshold = std::numeric_limits<T>::epsilon();  ///< Deflation threshold on the off-diagonal.
        /// If false, the eigenvector matrix is set to the identity and the rotations are
        /// applied to it; if true, they are applied to the caller's matrix (a back-transform).
        bool back_transform = false;
        bool block_rotations = false;          ///< Apply the rotations in #block_size blocks.
        bool sort = true;                      ///< Sort eigenpairs on output.
        bool transpose_working_vectors = true; ///< Interleave the working copies of d and e across the batch.
        SortOrder sort_order = SortOrder::Ascending;  ///< Order used when #sort is set.

        /// CTA STEQR only: multiplies the baseline work-group size, LCM(N, sub_group_size).
        /// 0 picks a tuned value per scalar type and partition width.
        size_t cta_wg_size_multiplier = 0;

        SteqrShiftStrategy cta_shift_strategy = SteqrShiftStrategy::Lapack;  ///< CTA STEQR only: shift strategy.
        SteqrUpdateScheme cta_update_scheme = SteqrUpdateScheme::EXP;        ///< CTA STEQR only: update scheme.
    };


    // `info` on steqr/stedc/syev/syevx/gesvd and their tiers: per-item convergence status,
    // an ACCUMULATOR zeroed once by the entry point the caller invoked and only raised
    // below (contract and helpers: src/extensions/info_span.hh).
    // evidence: docs/cpp-api.md#convergence-status-syev-syevx-gesvd-steqr-stedc
    // A DEFAULTED trailing parameter, not an old-arity forwarder as in potrf: these are
    // instantiated from macros that spell the list out, and with `jobz`, `params` and
    // `eigvects` already defaulted a forwarder would be AMBIGUOUS with the primary.
    // evidence: docs/design/vendor-independence.md#info-spans-on-syev-gesvd-and-steqr-forwarder-or-default

    /**
     * @brief Eigenvalues, and optionally eigenvectors, of a batch of symmetric tridiagonal
     *        matrices by implicit QR/QL iteration (LAPACK `?steqr`).
     *
     * Real `T` only. Runs the CTA kernel (steqr_cta()) when n is at most the device's
     * largest sub-group size (never on ROCm) and the work-group kernel otherwise;
     * steqr_buffer_size() returns the larger of the two needs.
     *
     * @param ctx         queue the work is enqueued on
     * @param d           diagonal, n entries per batch item
     * @param e           off-diagonal, n-1 entries per batch item
     * @param eigenvalues output, n per batch item; ascending when SteqrParams::sort is set
     * @param ws          at least steqr_buffer_size() bytes
     * @param jobz        whether `eigvects` receives the eigenvectors
     * @param params      sweep cap, deflation threshold, sorting and CTA tuning
     * @param eigvects    n x n per item; with SteqrParams::back_transform the rotations
     *                    are applied to its contents, otherwise it is set to the identity
     *                    first
     * @param info        per-item convergence status (0 = converged, > 0 LAPACK-like), or
     *                    empty to not request it; see @ref md_docs_2cpp-api
     * @throws batchlas::invalid_argument if `eigvects` is not n x n with the same batch
     * @throws batchlas::convergence_error as steqr_cta() does, when the CTA kernel runs
     * @return event of the last enqueued kernel
     * @see @ref algo_steqr
     */
    template <Backend B, typename T>
    BATCHLAS_API Event steqr(Queue& ctx, const VectorView<T>& d, const VectorView<T>& e,
                             const VectorView<T>& eigenvalues, const Span<std::byte>& ws, JobType jobz = JobType::NoEigenVectors, SteqrParams<T> params = SteqrParams<T>(),
                             const MatrixView<T, MatrixFormat::Dense>& eigvects = MatrixView<T, MatrixFormat::Dense>(),
                             Span<int32_t> info = Span<int32_t>());

    /**
     * @brief steqr() pinned to the CTA kernel: one sub-group partition per matrix, for small
     *        n (runtime-dispatched to compile-time specialised kernels). Parameters as for
     *        steqr().
     * @throws batchlas::invalid_argument unless 1 <= n <= 32
     * @throws batchlas::convergence_error if `BATCHLAS_STEQR_CTA_CHECK` is set and an item
     *         ran out of sweeps (the check waits on the queue; unset, nothing is thrown)
     */
    template <Backend B, typename T>
    BATCHLAS_API Event steqr_cta(Queue& ctx, const VectorView<T>& d, const VectorView<T>& e,
                                 const VectorView<T>& eigenvalues, const Span<std::byte>& ws,
                                 JobType jobz = JobType::NoEigenVectors,
                                 SteqrParams<T> params = SteqrParams<T>(),
                                 const MatrixView<T, MatrixFormat::Dense>& eigvects = MatrixView<T, MatrixFormat::Dense>(),
                                 Span<int32_t> info = Span<int32_t>());

    /** @brief Required workspace, in bytes, for steqr(); arguments as for the call. */
    template <typename T>
    BATCHLAS_API size_t steqr_buffer_size(Queue& ctx, const VectorView<T>& d, const VectorView<T>& e,
                                         const VectorView<T>& eigenvalues, JobType jobz = JobType::NoEigenVectors, SteqrParams<T> params = SteqrParams<T>());

    /** @brief Required workspace, in bytes, for steqr_cta(); arguments as for the call. */
    template <typename T>
    BATCHLAS_API size_t steqr_cta_buffer_size(Queue& ctx, const VectorView<T>& d, const VectorView<T>& e,
                                              const VectorView<T>& eigenvalues, JobType jobz = JobType::NoEigenVectors, SteqrParams<T> params = SteqrParams<T>());


    /// @}

    // === CTA small-matrix extensions (n <= 32) ===

    /**
     * @brief QR (GEQRF) or QL (GEQLF) reflector layout. Not referenced by any entry point.
     * @ingroup api_qr
     */
    enum class OrmqCtaFactorization {
        QR,
        QL,
    };

    /// @addtogroup api_tridiag
    /// @{

    /**
     * @brief CTA-optimized symmetric tridiagonal reduction (SYTD2-style), n <= 32.
     *        Overwrites A with the tridiagonal and reflector storage and returns (d,e)
     *        plus reflector scalars in tau. `ws` is unused, kept for API compatibility.
     */
    template <Backend B, typename T>
    BATCHLAS_API Event sytrd_cta(Queue& ctx,
                                 const MatrixView<T, MatrixFormat::Dense>& a_in,
                                 const VectorView<T>& d_out,
                                 const VectorView<T>& e_out,
                                 const VectorView<T>& tau_out,
                                 Uplo uplo,
                                 const Span<std::byte>& ws,
                                 size_t cta_wg_size_multiplier = 1);

    /**
     * @brief LATRD-like panel factorization used by blocked SYTRD, `Uplo::Lower` only.
     *        Householder vectors go into A, in sytrd's SYTD2-style reflector layout;
     *        only the first `ib` columns of W are written (W is treated as n x nb).
     *
     * @param j0   first column of the panel
     * @param ib   panel width
     * @param wg_hint work-group size hint; 0 lets the implementation choose
     * @param fuse_trailing_update fold the panel's share of the trailing update into the
     *        kernel
     */
    template <Backend B, typename T>
    BATCHLAS_API Event latrd_lower_panel(Queue& ctx,
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
     * e_panel/tau_panel size must match w_panel.cols().
     */
    template <Backend B, typename T>
    BATCHLAS_API Event latrd_lower_panel(Queue& ctx,
                                         const MatrixView<T, MatrixFormat::Dense>& a_panel_in,
                                         const VectorView<T>& e_panel_out,
                                         const VectorView<T>& tau_panel_out,
                                         const MatrixView<T, MatrixFormat::Dense>& w_panel_in,
                                         int32_t wg_hint = 0,
                                         bool fuse_trailing_update = false);

    /**
     * @brief Blocked symmetric/Hermitian tridiagonal reduction, n > 32: LATRD-style panel
     *        plus a BLAS-3 trailing update. Same SYTD2-style output as `sytrd_cta`.
     *        In-order queue only.
     */
    template <Backend B, typename T>
    BATCHLAS_API Event sytrd_blocked(Queue& ctx,
                                     const MatrixView<T, MatrixFormat::Dense>& a_in,
                                     const VectorView<T>& d_out,
                                     const VectorView<T>& e_out,
                                     const VectorView<T>& tau_out,
                                     Uplo uplo,
                                     const Span<std::byte>& ws,
                                     int32_t block_size = tuning::SYTRD_BLOCK_SIZE_MEDIUM);

    /** @brief Required workspace, in bytes, for sytrd_blocked(); arguments as for the call. */
    template <Backend B, typename T>
    BATCHLAS_API size_t sytrd_blocked_buffer_size(Queue& ctx,
                                                  const MatrixView<T, MatrixFormat::Dense>& a,
                                                  const VectorView<T>& d,
                                                  const VectorView<T>& e,
                                                  const VectorView<T>& tau,
                                                  Uplo uplo,
                                                  int32_t block_size = tuning::SYTRD_BLOCK_SIZE_MEDIUM);

    /**
     * @brief First stage of two-stage reduction: dense -> band (LAPACK xSYTRD_SY2SB).
     *        Overwrites `A` with reflector storage and writes the band into `AB`;
     *        `tau_out` has size (n-kd). In-order queue, `Uplo::Lower` only.
     *
     * Band storage (AB), shape (kd+1) x n:
     *  - `Uplo::Lower`: AB(1+i-j,j) = A(i,j) for j<=i<=min(n,j+kd).
     *  - `Uplo::Upper`: AB(kd+1+i-j,j) = A(i,j) for max(1,j-kd)<=i<=j.
     */
    template <Backend B, typename T>
    BATCHLAS_API Event sytrd_sy2sb(Queue& ctx,
                                   const MatrixView<T, MatrixFormat::Dense>& a_in,
                                   const MatrixView<T, MatrixFormat::Dense>& ab_out,
                                   const VectorView<T>& tau_out,
                                   Uplo uplo,
                                   int32_t kd,
                                   const Span<std::byte>& ws);

    /** @brief Required workspace, in bytes, for sytrd_sy2sb(); arguments as for the call. */
    template <Backend B, typename T>
    BATCHLAS_API size_t sytrd_sy2sb_buffer_size(Queue& ctx,
                                                const MatrixView<T, MatrixFormat::Dense>& a_in,
                                                const MatrixView<T, MatrixFormat::Dense>& ab_out,
                                                const VectorView<T>& tau_out,
                                                Uplo uplo,
                                                int32_t kd);

    /**
     * @brief Second stage of two-stage reduction: band -> tridiagonal (LAPACK
     *        xSB2ST/xHB2ST bulge chasing, VECT='N'). In-order queue, `Uplo::Lower` only.
     *
     * Band storage (AB) is as for `sytrd_sy2sb`. `d_out` (n) and `e_out` (n-1) are always
     * real-valued; `tau_out` (n-1) is unused downstream and is set to 0.
     */
    template <Backend B, typename T>
    BATCHLAS_API Event sytrd_sb2st(Queue& ctx,
                                   const MatrixView<T, MatrixFormat::Dense>& ab_in,
                                   const VectorView<typename base_type<T>::type>& d_out,
                                   const VectorView<typename base_type<T>::type>& e_out,
                                   const VectorView<T>& tau_out,
                                   Uplo uplo,
                                   int32_t kd,
                                   const Span<std::byte>& ws,
                                   int32_t block_size);

    /** @brief Required workspace, in bytes, for sytrd_sb2st(); arguments as for the call. */
    template <Backend B, typename T>
    BATCHLAS_API size_t sytrd_sb2st_buffer_size(Queue& ctx,
                                                const MatrixView<T, MatrixFormat::Dense>& ab_in,
                                                const VectorView<typename base_type<T>::type>& d_out,
                                                const VectorView<typename base_type<T>::type>& e_out,
                                                const VectorView<T>& tau_out,
                                                Uplo uplo,
                                                int32_t kd,
                                                int32_t block_size);

    /** @brief Sweep schedule for sytrd_band_reduction() (successive band reduction). */
    struct SytrdBandReductionParams {
        /// Diagonals eliminated per sweep (Algorithm 2: d^(i)); 0 means the default
        /// schedule. If sweeps exceed the sequence length, the last value is reused.
        std::vector<int32_t> d_seq{0};

        /// Block size per sweep (Algorithm 2: nb^(i)).
        /// If sweeps exceed sequence length, the last value is reused.
        std::vector<int32_t> block_size_seq{32};

        int32_t max_sweeps = -1;  ///< Maximum number of sweeps; < 0 means the implementation default.

        /// Debug/testing: chase steps for sytrd_band_reduction_single_step();
        /// <= 0 means exactly one step.
        int32_t max_steps = 1;

        int32_t kd_work = 0;      ///< Working band semibandwidth; <= 0 means the implementation default.
    };

    /**
     * @brief Symmetric/Hermitian band -> tridiagonal reduction (BANDR1-style). In-order
     *        queue, `Uplo::Lower` only. (d,e) are real; for complex input `e_out` is the
     *        magnitude of the (possibly phased) subdiagonal, and `tau_out` is set to 0.
     *
     * Band storage is as for sytrd_sy2sb(). This overload uses one block size for every
     * sweep.
     */
    template <Backend B, typename T>
    BATCHLAS_API Event sytrd_band_reduction(Queue& ctx,
                                            const MatrixView<T, MatrixFormat::Dense>& ab_in,
                                            const VectorView<typename base_type<T>::type>& d_out,
                                            const VectorView<typename base_type<T>::type>& e_out,
                                            const VectorView<T>& tau_out,
                                            Uplo uplo,
                                            int32_t kd,
                                            const Span<std::byte>& ws,
                                            int32_t block_size);

    /** @brief sytrd_band_reduction() with an explicit per-sweep schedule. */
    template <Backend B, typename T>
    BATCHLAS_API Event sytrd_band_reduction(Queue& ctx,
                                            const MatrixView<T, MatrixFormat::Dense>& ab_in,
                                            const VectorView<typename base_type<T>::type>& d_out,
                                            const VectorView<typename base_type<T>::type>& e_out,
                                            const VectorView<T>& tau_out,
                                            Uplo uplo,
                                            int32_t kd,
                                            const Span<std::byte>& ws,
                                            SytrdBandReductionParams params);

    template <Backend B, typename T>
    BATCHLAS_API Event sytrd_band_reduction_single_step(Queue& ctx,
                                                        const MatrixView<T, MatrixFormat::Dense>& ab_in,
                                                        const MatrixView<T, MatrixFormat::Dense>& abw_out,
                                                        Uplo uplo,
                                                        int32_t kd,
                                                        const Span<std::byte>& ws,
                                                        SytrdBandReductionParams params);

    /** @brief Required workspace, in bytes, for sytrd_band_reduction() with one block size. */
    template <Backend B, typename T>
    BATCHLAS_API size_t sytrd_band_reduction_buffer_size(Queue& ctx,
                                                         const MatrixView<T, MatrixFormat::Dense>& ab_in,
                                                         const VectorView<typename base_type<T>::type>& d_out,
                                                         const VectorView<typename base_type<T>::type>& e_out,
                                                         const VectorView<T>& tau_out,
                                                         Uplo uplo,
                                                         int32_t kd,
                                                         int32_t block_size);

    /** @brief Required workspace, in bytes, for the schedule-parameter sytrd_band_reduction(). */
    template <Backend B, typename T>
    BATCHLAS_API size_t sytrd_band_reduction_buffer_size(Queue& ctx,
                                                         const MatrixView<T, MatrixFormat::Dense>& ab_in,
                                                         const VectorView<typename base_type<T>::type>& d_out,
                                                         const VectorView<typename base_type<T>::type>& e_out,
                                                         const VectorView<T>& tau_out,
                                                         Uplo uplo,
                                                         int32_t kd,
                                                         SytrdBandReductionParams params);

    template <Backend B, typename T>
    BATCHLAS_API size_t sytrd_band_reduction_single_step_buffer_size(Queue& ctx,
                                                                     const MatrixView<T, MatrixFormat::Dense>& ab_in,
                                                                     const MatrixView<T, MatrixFormat::Dense>& abw_out,
                                                                     Uplo uplo,
                                                                     int32_t kd,
                                                                     SytrdBandReductionParams params);

    /**
     * @brief Debug/testing hook: execute exactly one BANDR1 “chase step”. Not a stable
     *        public API. `ab_in` is lower-band with rows == kd+1, `abw_out` lower-band
     *        with rows == kd_work+1 (kd_work from params). In-order queue, Lower only.
     */
    template <Backend B, typename T>
    BATCHLAS_API Event sytrd_band_reduction_single_step(Queue& ctx,
                                                        const MatrixView<T, MatrixFormat::Dense>& ab_in,
                                                        const MatrixView<T, MatrixFormat::Dense>& abw_out,
                                                        Uplo uplo,
                                                        int32_t kd,
                                                        const Span<std::byte>& ws,
                                                        SytrdBandReductionParams params);

    /** @brief Required workspace, in bytes, for sytrd_band_reduction_single_step(). */
    template <Backend B, typename T>
    BATCHLAS_API size_t sytrd_band_reduction_single_step_buffer_size(Queue& ctx,
                                                                     const MatrixView<T, MatrixFormat::Dense>& ab_in,
                                                                     const MatrixView<T, MatrixFormat::Dense>& abw_out,
                                                                     Uplo uplo,
                                                                     int32_t kd,
                                                                     SytrdBandReductionParams params);

    /** @brief LAPACK-named alias (HB2ST) of sytrd_sb2st(); identical arguments. */
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

    /** @brief LAPACK-named alias of sytrd_sb2st_buffer_size(). */
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

    /// @}

    /// @addtogroup api_eigen
    /// @{

    /**
     * @brief CTA-optimized symmetric eigen-solver (SYEV-like), n <= 32; real symmetric and
     *        complex Hermitian. Overwrites A with eigenvectors when jobz == EigenVectors.
     *        Eigenvalues ascend when SteqrParams::sort is set (default).
     *        cta_wg_size_multiplier == 0 lets the tridiagonal solve pick its tuned value
     *        and runs the reduction and back-transform at 1.
     *
     * This and the other `syev_*` tiers are what syev() routes between; call one directly
     * only to pin a tier. As in syev(), `a_in`'s `uplo` triangle is read, `eigenvalues`
     * gets n real values per item, and `info` is the per-item convergence status or empty.
     * @see @ref perf_syev (small-n kernel choice)
     */
    template <Backend B, typename T>
    BATCHLAS_API Event syev_cta(Queue& ctx,
                                const MatrixView<T, MatrixFormat::Dense>& a_in,
                                Span<typename base_type<T>::type> eigenvalues,
                                JobType jobz,
                                Uplo uplo,
                                const Span<std::byte>& ws,
                                SteqrParams<T> steqr_params = SteqrParams<T>(),
                                size_t cta_wg_size_multiplier = 0,
                                Span<int32_t> info = Span<int32_t>());

    /** @brief Required workspace, in bytes, for syev_cta(). */
    template <Backend B, typename T>
    BATCHLAS_API size_t syev_cta_buffer_size(Queue& ctx,
                                             const MatrixView<T, MatrixFormat::Dense>& a,
                                             JobType jobz,
                                             SteqrParams<T> steqr_params = SteqrParams<T>());

    /**
     * @brief Fused (single-kernel) variant of syev_cta.
     *
     * Runs end to end inside one sub-group partition, so results track syev_cta only to
     * within the reassociation that fusing implies. Unlike syev_cta, A is left untouched
     * when jobz == NoEigenVectors. `ws` is accepted for API symmetry and ignored.
     * cta_wg_size_multiplier == 0 picks a tuned value per scalar type and width.
     */
    template <Backend B, typename T>
    BATCHLAS_API Event syev_cta_fused(Queue& ctx,
                                      const MatrixView<T, MatrixFormat::Dense>& a_in,
                                      Span<typename base_type<T>::type> eigenvalues,
                                      JobType jobz,
                                      Uplo uplo,
                                      const Span<std::byte>& ws = Span<std::byte>(),
                                      SteqrParams<T> steqr_params = SteqrParams<T>(),
                                      size_t cta_wg_size_multiplier = 0,
                                      Span<int32_t> info = Span<int32_t>());

    /** @brief Required workspace, in bytes, for syev_cta_fused(). */
    template <Backend B, typename T>
    BATCHLAS_API size_t syev_cta_fused_buffer_size(Queue& ctx,
                                                   const MatrixView<T, MatrixFormat::Dense>& a,
                                                   JobType jobz,
                                                   SteqrParams<T> steqr_params = SteqrParams<T>());

    /** @brief Parameters for syev_jacobi_cta() (cyclic two-sided Jacobi). */
    template <typename T>
    struct JacobiParams {
        using Real = typename base_type<T>::type;

        /// A rotation is applied to pivot pair (p,q) only when
        /// \f$ |a_{pq}| > \mathrm{tol\_multiplier} \cdot n \varepsilon \sqrt{|a_{pp}| |a_{qq}|} \f$.
        /// This relative test, not the classical absolute one, is what yields the high
        /// relative accuracy; raising it trades accuracy for sweeps.
        Real tol_multiplier = Real(1);

        size_t max_sweeps = 30;  ///< Cap on cyclic sweeps; a safety net for pathological inputs.

        bool sort = true;                             ///< Sort eigenpairs on output.
        SortOrder sort_order = SortOrder::Ascending;  ///< Order used when #sort is set.

        /// Multiplies the baseline work-group size, LCM(P, sub_group_size) for the
        /// compile-time partition width P chosen from n. Clamped by device limits.
        size_t cta_wg_size_multiplier = 1;
    };

    /**
     * @brief CTA-optimized Jacobi symmetric/Hermitian eigen-solver, n <= 32; real
     *        symmetric and complex Hermitian.
     *
     * Trades throughput for high *relative* accuracy on graded or badly scaled input. The
     * underlying theorem (Demmel & Veselic, SIMAX 13(4), 1992) is proved for symmetric
     * positive definite input; indefinite matrices are solved correctly but do not inherit
     * the bound. Overwrites A with eigenvectors when jobz == EigenVectors and leaves it
     * untouched otherwise; `ws` is accepted for API symmetry and ignored.
     */
    template <Backend B, typename T>
    BATCHLAS_API Event syev_jacobi_cta(Queue& ctx,
                                       const MatrixView<T, MatrixFormat::Dense>& a_in,
                                       Span<typename base_type<T>::type> eigenvalues,
                                       JobType jobz,
                                       Uplo uplo,
                                       const Span<std::byte>& ws = Span<std::byte>(),
                                       JacobiParams<T> params = JacobiParams<T>(),
                                       Span<int32_t> info = Span<int32_t>());

    /** @brief Required workspace, in bytes, for syev_jacobi_cta(). */
    template <Backend B, typename T>
    BATCHLAS_API size_t syev_jacobi_cta_buffer_size(Queue& ctx,
                                                    const MatrixView<T, MatrixFormat::Dense>& a,
                                                    JobType jobz,
                                                    JacobiParams<T> params = JacobiParams<T>());

    /**
     * @brief Blocked symmetric/Hermitian eigen-solver (SYEV-like) for medium/large n:
     *        sytrd_blocked -> stedc -> ormqr_blocked. Overwrites A with eigenvectors when
     *        jobz == EigenVectors. `Uplo::Upper` input is first mirrored into the lower
     *        triangle, in place, and then takes the Lower path.
     */
    template <Backend B, typename T>
    BATCHLAS_API Event syev_blocked(Queue& ctx,
                                    const MatrixView<T, MatrixFormat::Dense>& a_in,
                                    Span<typename base_type<T>::type> eigenvalues,
                                    JobType jobz,
                                    Uplo uplo,
                                    const Span<std::byte>& ws,
                                    StedcParams<typename base_type<T>::type> stedc_params = StedcParams<typename base_type<T>::type>(),
                                    Span<int32_t> info = Span<int32_t>());

    /** @brief Required workspace, in bytes, for syev_blocked(); the same for either `uplo`. */
    template <Backend B, typename T>
    BATCHLAS_API size_t syev_blocked_buffer_size(Queue& ctx,
                                                 const MatrixView<T, MatrixFormat::Dense>& a,
                                                 JobType jobz,
                                                 Uplo uplo,
                                                 StedcParams<typename base_type<T>::type> stedc_params = StedcParams<typename base_type<T>::type>());

    /**
     * @brief Two-stage symmetric/Hermitian eigen-solver (SYEV-like) for large n:
     *        sytrd_sy2sb -> sytrd_sb2st -> stedc, then phase/sign recovery and a reflector
     *        back-transform. Band width from choose_two_stage_kd (override with
     *        BATCHLAS_SYEV_TWO_STAGE_KD). `Uplo::Upper` input is mirrored into the lower
     *        triangle first, as in syev_blocked().
     * @see @ref perf_syev for where syev() routes to it
     */
    template <Backend B, typename T>
    BATCHLAS_API Event syev_two_stage(Queue& ctx,
                                      const MatrixView<T, MatrixFormat::Dense>& a_in,
                                      Span<typename base_type<T>::type> eigenvalues,
                                      JobType jobz,
                                      Uplo uplo,
                                      const Span<std::byte>& ws,
                                      StedcParams<typename base_type<T>::type> stedc_params = StedcParams<typename base_type<T>::type>(),
                                      Span<int32_t> info = Span<int32_t>());

    /** @brief Required workspace, in bytes, for syev_two_stage(). */
    template <Backend B, typename T>
    BATCHLAS_API size_t syev_two_stage_buffer_size(Queue& ctx,
                                                   const MatrixView<T, MatrixFormat::Dense>& a,
                                                   JobType jobz,
                                                   Uplo uplo,
                                                   StedcParams<typename base_type<T>::type> stedc_params = StedcParams<typename base_type<T>::type>());

    /// @}

    /// @addtogroup api_svd
    /// @{

    /**
     * @brief Unblocked GEBRD-like reduction to real bidiagonal form: reduces each square A
     *        in place, returning (d,e) and the Householder scalars (tauq,taup).
     *
     * Computes \f$ A = Q B P^H \f$ with `B` upper bidiagonal: `d` gets its n diagonal and
     * `e` its n-1 superdiagonal entries (real), and the reflectors defining `Q` and `P`
     * are left in `A` with their scalars in `tauq` and `taup`, as in LAPACK `?gebrd`.
     * @see @ref perf_gesvd (where the unblocked form stops paying)
     */
    template <Backend B, typename T>
    BATCHLAS_API Event gebrd_unblocked(Queue& ctx,
                                       const MatrixView<T, MatrixFormat::Dense>& a,
                                       const VectorView<typename base_type<T>::type>& d,
                                       const VectorView<typename base_type<T>::type>& e,
                                       const VectorView<T>& tauq,
                                       const VectorView<T>& taup);

    /**
     * @brief CTA-parallel small-matrix GEBRD for real square matrices, `1 <= n <= 32`,
     *        where one CTA cooperatively reduces one matrix to upper bidiagonal form.
     */
    template <Backend B, typename T>
    BATCHLAS_API Event gebrd_cta(Queue& ctx,
                                 const MatrixView<T, MatrixFormat::Dense>& a,
                                 const VectorView<typename base_type<T>::type>& d,
                                 const VectorView<typename base_type<T>::type>& e,
                                 const VectorView<T>& tauq,
                                 const VectorView<T>& taup,
                                 size_t cta_wg_size_multiplier = 1);

    /**
     * @brief Blocked GEBRD for real square dense matrices: DLABRD-style panel + GEMM.
     *        Output layout as for gebrd_unblocked().
     */
    template <Backend B, typename T>
    BATCHLAS_API Event gebrd_blocked(Queue& ctx,
                                     const MatrixView<T, MatrixFormat::Dense>& a,
                                     const VectorView<typename base_type<T>::type>& d,
                                     const VectorView<typename base_type<T>::type>& e,
                                     const VectorView<T>& tauq,
                                     const VectorView<T>& taup,
                                     const Span<std::byte>& ws,
                                     int32_t block_size = 16);

    /** @brief Required workspace, in bytes, for gebrd_blocked(); arguments as for the call. */
    template <Backend B, typename T>
    BATCHLAS_API size_t gebrd_blocked_buffer_size(Queue& ctx,
                                                  const MatrixView<T, MatrixFormat::Dense>& a,
                                                  const VectorView<typename base_type<T>::type>& d,
                                                  const VectorView<typename base_type<T>::type>& e,
                                                  const VectorView<T>& tauq,
                                                  const VectorView<T>& taup,
                                                  int32_t block_size = 16);

    /**
     * @brief Bidiagonal QR iteration for a real upper bidiagonal matrix. The matrix
     *        overload also accumulates the alternating Givens rotations into `u` and `vh`,
     *        matching LAPACK `BDSQR`: it returns `u * Q` and `P^T * vh`.
     *
     * @param d,e                  diagonal (n) and superdiagonal (n-1) per batch item
     * @param singular_values_out  n singular values per item, packed
     * @param ws                   at least bdsqr_buffer_size() bytes
     * @param sort_desc            sort the singular values (and vectors) in descending order
     * @param info                 per-item convergence status, or empty; see @ref md_docs_2cpp-api
     */
    template <Backend B, typename T>
    BATCHLAS_API Event bdsqr(Queue& ctx,
                             const VectorView<T>& d,
                             const VectorView<T>& e,
                             Span<T> singular_values_out,
                             const Span<std::byte>& ws,
                             bool sort_desc = true,
                             Span<int32_t> info = Span<int32_t>());

    /**
     * @brief bdsqr() that also accumulates the rotations: `u <- u * Q`, `vh <- P^T * vh`.
     *        Other parameters as for the values-only form.
     */
    template <Backend B, typename T>
    BATCHLAS_API Event bdsqr(Queue& ctx,
                             const VectorView<T>& d,
                             const VectorView<T>& e,
                             Span<T> singular_values_out,
                             const Span<std::byte>& ws,
                             const MatrixView<T, MatrixFormat::Dense>& u,
                             const MatrixView<T, MatrixFormat::Dense>& vh,
                             bool sort_desc = true,
                             Span<int32_t> info = Span<int32_t>());

    /** @brief Required workspace, in bytes, for bdsqr(). */
    template <typename T>
    BATCHLAS_API size_t bdsqr_buffer_size(Queue& ctx,
                                          const VectorView<T>& d,
                                          const VectorView<T>& e,
                                          Span<T> singular_values_out);

    /** @brief bdsqr_buffer_size() for the vector-accumulating form; `u` and `vh` do not change it. */
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
     * @brief Bidiagonal divide-and-conquer SVD for a real upper bidiagonal matrix. Same
     *        problem as `bdsqr`, via the symmetric tridiagonal eigenproblem of the
     *        interleaved Golub-Kahan form (order `2n`) handed to `stedc`.
     *
     * Unlike `bdsqr`, which accumulates into whatever it is handed (`u <- u*Q`), `bdsdc`
     * WRITES the leading `n x n` block of `u` and of `vh` and leaves the rest untouched --
     * seed them with the identity if the trailing columns matter. Workspace is dominated
     * by a `2n x 2n` eigenvector matrix per batch item. Parameters as for bdsqr().
     * @see @ref design_gesvd, section "gesvd design: why bdsdc goes through Golub-Kahan"
     */
    template <Backend B, typename T>
    BATCHLAS_API Event bdsdc(Queue& ctx,
                             const VectorView<T>& d,
                             const VectorView<T>& e,
                             Span<T> singular_values_out,
                             const Span<std::byte>& ws,
                             bool sort_desc = true,
                             Span<int32_t> info = Span<int32_t>());

    /** @brief bdsdc() that also writes the leading n x n blocks of `u` and `vh`. */
    template <Backend B, typename T>
    BATCHLAS_API Event bdsdc(Queue& ctx,
                             const VectorView<T>& d,
                             const VectorView<T>& e,
                             Span<T> singular_values_out,
                             const Span<std::byte>& ws,
                             const MatrixView<T, MatrixFormat::Dense>& u,
                             const MatrixView<T, MatrixFormat::Dense>& vh,
                             bool sort_desc = true,
                             Span<int32_t> info = Span<int32_t>());

    /** @brief Required workspace, in bytes, for bdsdc(); `want_vectors` selects the overload. */
    template <Backend B, typename T>
    BATCHLAS_API size_t bdsdc_buffer_size(Queue& ctx,
                                          const VectorView<T>& d,
                                          const VectorView<T>& e,
                                          Span<T> singular_values_out,
                                          bool want_vectors);

    
    /**
     * @brief ORMBR/UNMBR-style application of bidiagonal reduction reflectors. Supports
     *        `vect='Q'` (tauq, CTA/blocked ORMQR path) and `vect='P'` (taup, blocked
     *        compact-WY path over the right reflectors).
     *
     * Overwrites `c` with `op(Q) * c`, `c * op(Q)`, `op(P) * c` or `c * op(P)` according to
     * `vect`, `side` and `trans`, where `Q`/`P` are the reflectors that gebrd_blocked() (or
     * gebrd_unblocked()) left in `a` and `tau`, as LAPACK `?ormbr` / `?unmbr`: `'Q'` is of
     * order `a.rows()`, `'P'` of order `a.cols()`.
     *
     * @pre `tau` is unit-stride and packed by batch
     * @throws batchlas::invalid_argument on mismatched batch sizes or orders, a `vect`
     *         other than `'Q'`/`'P'`, or a short or strided `tau`
     * @throws batchlas::unsupported for `Transpose::Trans` with complex `T` and `'P'`
     *         (use `ConjTrans`)
     */
    template <Backend B, typename T>
    BATCHLAS_API Event ormbr(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& a,
                             const VectorView<T>& tau,
                             const MatrixView<T, MatrixFormat::Dense>& c,
                             char vect,
                             Side side,
                             Transpose trans,
                             const Span<std::byte>& ws,
                             int32_t block_size = 32);

    /** @brief Required workspace, in bytes, for ormbr(); arguments as for the call. */
    template <Backend B, typename T>
    BATCHLAS_API size_t ormbr_buffer_size(Queue& ctx,
                                          const MatrixView<T, MatrixFormat::Dense>& a,
                                          const VectorView<T>& tau,
                                          const MatrixView<T, MatrixFormat::Dense>& c,
                                          char vect,
                                          Side side,
                                          Transpose trans,
                                          int32_t block_size = 32);

    /**
     * @brief Blocked native SVD for real dense matrices: GEBRD-style dense -> bidiagonal,
     *        then BDSQR (or BDSDC when explicitly selected), then ORMBR-style
     *        back-transforms for full U and V^H.
     * @pre in-order Queue
     * @throws batchlas::invalid_argument for an out-of-order Queue or mis-shaped outputs
     * @throws batchlas::unsupported for complex input (use the Hermitian overload)
     */
    template <Backend B, typename T>
    BATCHLAS_API Event gesvd_blocked(Queue& ctx,
                                     const MatrixView<T, MatrixFormat::Dense>& a_in,
                                     Span<typename base_type<T>::type> singular_values,
                                     const MatrixView<T, MatrixFormat::Dense>& u_out,
                                     const MatrixView<T, MatrixFormat::Dense>& vh_out,
                                     SvdVectors jobu,
                                     SvdVectors jobvh,
                                     const Span<std::byte>& ws,
                                     Span<int32_t> info = Span<int32_t>());

    /**
     * @brief gesvd_blocked() for Hermitian input: only the `hermitian_uplo` triangle of
     *        `a_in` is read.
     */
    template <Backend B, typename T>
    BATCHLAS_API Event gesvd_blocked(Queue& ctx,
                                     const MatrixView<T, MatrixFormat::Dense>& a_in,
                                     Span<typename base_type<T>::type> singular_values,
                                     const MatrixView<T, MatrixFormat::Dense>& u_out,
                                     const MatrixView<T, MatrixFormat::Dense>& vh_out,
                                     SvdVectors jobu,
                                     SvdVectors jobvh,
                                     Uplo hermitian_uplo,
                                     const Span<std::byte>& ws,
                                     Span<int32_t> info = Span<int32_t>());

    /** @brief Required workspace, in bytes, for gesvd_blocked(). */
    template <Backend B, typename T>
    BATCHLAS_API size_t gesvd_blocked_buffer_size(Queue& ctx,
                                                  const MatrixView<T, MatrixFormat::Dense>& a,
                                                  Span<typename base_type<T>::type> singular_values,
                                                  const MatrixView<T, MatrixFormat::Dense>& u_out,
                                                  const MatrixView<T, MatrixFormat::Dense>& vh_out,
                                                  SvdVectors jobu,
                                                  SvdVectors jobvh);

    /** @brief Required workspace, in bytes, for the Hermitian gesvd_blocked(). */
    template <Backend B, typename T>
    BATCHLAS_API size_t gesvd_blocked_buffer_size(Queue& ctx,
                                                  const MatrixView<T, MatrixFormat::Dense>& a,
                                                  Span<typename base_type<T>::type> singular_values,
                                                  const MatrixView<T, MatrixFormat::Dense>& u_out,
                                                  const MatrixView<T, MatrixFormat::Dense>& vh_out,
                                                  SvdVectors jobu,
                                                  SvdVectors jobvh,
                                                  Uplo hermitian_uplo);

    /**
     * @brief CTA-oriented native SVD for small matrices, `max(m, n) <= 32`: bidiagonal
     *        reduction, then the eigenproblem of \f$ B^T B \f$ (the normal equations).
     *        Real input, rectangular either way.
     *
     * The normal equations square the condition number; use gesvdj_cta() where small
     * singular values matter.
     * @pre in-order Queue
     * @throws batchlas::invalid_argument for `max(m, n) > 32`, a genuinely `Thin` U or
     *         V^H, or an out-of-order Queue
     * @throws batchlas::unsupported for complex input (use the Hermitian overload)
     * @see @ref perf_gesvd
     */
    template <Backend B, typename T>
    BATCHLAS_API Event gesvd_cta(Queue& ctx,
                                 const MatrixView<T, MatrixFormat::Dense>& a_in,
                                 Span<typename base_type<T>::type> singular_values,
                                 const MatrixView<T, MatrixFormat::Dense>& u_out,
                                 const MatrixView<T, MatrixFormat::Dense>& vh_out,
                                 SvdVectors jobu,
                                 SvdVectors jobvh,
                                 const Span<std::byte>& ws,
                                 Span<int32_t> info = Span<int32_t>());

    /** @brief gesvd_cta() for Hermitian input: only the `hermitian_uplo` triangle is read. */
    template <Backend B, typename T>
    BATCHLAS_API Event gesvd_cta(Queue& ctx,
                                 const MatrixView<T, MatrixFormat::Dense>& a_in,
                                 Span<typename base_type<T>::type> singular_values,
                                 const MatrixView<T, MatrixFormat::Dense>& u_out,
                                 const MatrixView<T, MatrixFormat::Dense>& vh_out,
                                 SvdVectors jobu,
                                 SvdVectors jobvh,
                                 Uplo hermitian_uplo,
                                 const Span<std::byte>& ws,
                                 Span<int32_t> info = Span<int32_t>());

    /** @brief Required workspace, in bytes, for gesvd_cta(). */
    template <Backend B, typename T>
    BATCHLAS_API size_t gesvd_cta_buffer_size(Queue& ctx,
                                              const MatrixView<T, MatrixFormat::Dense>& a,
                                              Span<typename base_type<T>::type> singular_values,
                                              const MatrixView<T, MatrixFormat::Dense>& u_out,
                                              const MatrixView<T, MatrixFormat::Dense>& vh_out,
                                              SvdVectors jobu,
                                              SvdVectors jobvh);

    /** @brief Required workspace, in bytes, for the Hermitian gesvd_cta(). */
    template <Backend B, typename T>
    BATCHLAS_API size_t gesvd_cta_buffer_size(Queue& ctx,
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
     * @trap Deliberately NOT a reuse of JacobiParams: its `sort_order` defaults to
     *       Ascending while gesvd admits exactly one order (descending), so reuse invites
     *       a silent flip.
     */
    template <typename T>
    struct GesvdjParams {
        using Real = typename base_type<T>::type;

        /// A rotation is applied to pivot pair (p,q) only when
        /// \f$ |a_{pq}| > \mathrm{tol\_multiplier} \cdot n \varepsilon \sqrt{|a_{pp}| |a_{qq}|} \f$
        /// over the 2x2 Gram entries of the current columns; again the relative test.
        Real tol_multiplier = Real(1);

        size_t max_sweeps = 30;  ///< Cap on cyclic sweeps; convergence normally takes well under 10.

        /// Multiplies the baseline problems-per-work-group. Baseline is
        /// 32 / P problems, clamped by local memory and max work-group size.
        size_t cta_wg_size_multiplier = 1;

        /// \f$ \sigma_j \le \mathrm{zero\_sigma\_multiplier} \cdot \varepsilon \sigma_{\max} \f$
        /// means \f$ U_j \f$ is not determined by A and is filled from the orthogonal complement.
        Real zero_sigma_multiplier = Real(1);

        /// Optional per-problem diagnostic, the analogue of cusolverDnXgesvdjGetSweeps:
        /// when non-empty (size >= batch_size) the kernel writes each problem's sweeps.
        Span<int32_t> sweep_counts = Span<int32_t>();
    };

    /**
     * @brief One-sided (Hestenes) Jacobi SVD for batches of small matrices.
     *
     * Computes \f$ A = U \operatorname{diag}(s) V^H \f$ with high RELATIVE accuracy: the
     * singular-value error is governed by the condition number of the column-equilibrated
     * matrix rather than of A, so graded input keeps its small values;
     * gesvd_cta() / gesvd_blocked() do not.
     *
     * max(m, n) <= 64 (32 for `complex<double>` with vectors), real and complex,
     * rectangular either way; needs sub-group size 32. A is DESTROYED, singular values are
     * returned descending, and no workspace is required.
     *
     * @throws batchlas::invalid_argument for a size above the cap or undersized outputs
     * @throws batchlas::unsupported if the device has no sub-group size 32
     * @see @ref design_gesvd, @ref perf_gesvd
     */
    template <Backend B, typename T>
    BATCHLAS_API Event gesvdj_cta(Queue& ctx,
                                  const MatrixView<T, MatrixFormat::Dense>& a_in,
                                  Span<typename base_type<T>::type> singular_values,
                                  const MatrixView<T, MatrixFormat::Dense>& u_out,
                                  const MatrixView<T, MatrixFormat::Dense>& vh_out,
                                  SvdVectors jobu,
                                  SvdVectors jobvh,
                                  const Span<std::byte>& ws = Span<std::byte>(),
                                  GesvdjParams<T> params = GesvdjParams<T>(),
                                  Span<int32_t> info = Span<int32_t>());

    /** @brief Required workspace, in bytes, for gesvdj_cta() (currently 0: all state is in local memory). */
    template <Backend B, typename T>
    BATCHLAS_API size_t gesvdj_cta_buffer_size(Queue& ctx,
                                               const MatrixView<T, MatrixFormat::Dense>& a,
                                               Span<typename base_type<T>::type> singular_values,
                                               const MatrixView<T, MatrixFormat::Dense>& u_out,
                                               const MatrixView<T, MatrixFormat::Dense>& vh_out,
                                               SvdVectors jobu,
                                               SvdVectors jobvh,
                                               GesvdjParams<T> params = GesvdjParams<T>());

    /// @}

    /**
     * @brief CTA-optimized application of Q from a QR/QL factorization (ORMQx/UNMQx
     *        semantics) for very small matrices: applies the implicit Q of the Householder
     *        reflectors (A, TAU) from GEQRF (Upper) or GEQLF (Lower) to C.
     *
     * Overwrites `c_in` with `op(Q) * C` (`Side::Left`) or `C * op(Q)` (`Side::Right`),
     * using the first `k` reflectors.
     * @ingroup api_qr
     */
    template <Backend B, typename T>
    BATCHLAS_API Event ormqx_cta(Queue& ctx,
                                const MatrixView<T, MatrixFormat::Dense>& a_in,
                                const VectorView<T>& tau_in,
                                const MatrixView<T, MatrixFormat::Dense>& c_in,
                                Uplo factorization,
                                Side side,
                                Transpose trans,
                                int32_t k,
                                const Span<std::byte>& ws,
                                size_t cta_wg_size_multiplier = 1);

    /// @addtogroup api_tridiag
    /// @{

    /** @brief Secular-equation root finder used by stedc()'s merge step. */
    enum class StedcSecularSolver {
        Rocm,    ///< rocSOLVER-style solver (default).
        Legacy,  ///< Original BatchLAS solver.
    };

    /** @brief How the merge step (secular solve + eigenvector formation) is dispatched. */
    enum class StedcMergeVariant {
        Auto = -1,          ///< Use BatchLAS tuning tables for the current problem size
        Baseline,           ///< Serial-per-root path (3 separate kernels)
        Fused,              ///< One kernel: warp-parallel root solve + build/normalize Qprime columns
        FusedCta,           ///< CTA-partitioned root solve using tunable threads per root
    };

    /** @brief Which divide-and-conquer driver runs the merge tree. */
    enum class StedcAlgorithm {
        Auto = -1,      ///< Currently: Levels
        Levels,         ///< Level-synchronous: every node at a tree level merges in one launch
        Recursive,      ///< Depth-first: one node per launch (kept for A/B comparison)
    };

    /**
     * @brief Parameters for stedc(). Every tuning knob at 0 (or `Auto`) takes the value
     *        from the checked-in tuning tables.
     * @see @ref perf_stedc, section "stedc: current tuning values"
     */
    template <typename T>
    struct StedcParams {
        int64_t recursion_threshold = 0; ///< Leaf size; <= 0 uses BatchLAS tuning, otherwise this exact threshold
        StedcAlgorithm algorithm = StedcAlgorithm::Auto;               ///< Merge-tree driver.
        StedcSecularSolver secular_solver = StedcSecularSolver::Rocm;  ///< Secular root finder.
        SteqrParams<T> leaf_steqr_params = SteqrParams<T>();           ///< Parameters of the leaf steqr() solves.

        StedcMergeVariant merge_variant = StedcMergeVariant::Auto;     ///< Merge-step dispatch.
        int merge_threads = 128;       ///< work-group size for fused kernel
        int max_sec_iter = 50;         ///< iteration cap for secular root solver
        bool enable_rescale = true;    ///< keep ROCm-style v rescale (disable for perf experiments)
        int secular_threads_per_root = 0;       ///< <= 0 uses BatchLAS tuning; otherwise this exact partition width
        int secular_cta_wg_size_multiplier = 0;  ///< <= 0 uses BatchLAS tuning; otherwise this exact multiplier
    };

    /**
     * @brief Eigenvalues and eigenvectors of a batch of symmetric tridiagonal matrices by
     *        divide and conquer (LAPACK `?stedc`).
     *
     * Splits each matrix recursively down to leaves of `recursion_threshold`, solves the
     * leaves with steqr(), and merges pairs by solving the secular equation and forming
     * the merged eigenvectors. Eigenvalues are returned ascending.
     *
     * @param ctx         queue the work is enqueued on
     * @param d           diagonal, n entries per batch item
     * @param e           off-diagonal, n-1 entries per batch item
     * @param eigenvalues output, n per batch item
     * @param ws          at least stedc_buffer_size() bytes
     * @param jobz        whether `eigvects` receives the eigenvectors
     * @param params      leaf size, merge driver and tuning overrides
     * @param eigvects    n x n eigenvector output per item
     * @param info        per-item convergence status, or empty; see @ref md_docs_2cpp-api.
     *                    Raised by a merge whose secular solve hits its iteration cap and
     *                    by a leaf steqr() that runs out of sweeps.
     * @return event of the last enqueued kernel
     * @see @ref perf_stedc
     */
    template <Backend B, typename T>
    BATCHLAS_API Event stedc(Queue& ctx, const VectorView<T>& d, const VectorView<T>& e, const VectorView<T>& eigenvalues, const Span<std::byte>& ws,
                         JobType jobz, StedcParams<T> params, const MatrixView<T, MatrixFormat::Dense>& eigvects,
                         Span<int32_t> info = Span<int32_t>());

    /** @brief Required workspace, in bytes, for stedc(). */
    template <Backend B, typename T>
    BATCHLAS_API size_t stedc_buffer_size(Queue& ctx, size_t n, size_t batch_size, JobType jobz, StedcParams<T> params);

    /**
     * @brief Old name of stedc_buffer_size(), kept so an out-of-tree caller gets a warning
     *        and not a link error.
     * @deprecated Use stedc_buffer_size(); the `*_buffer_size` naming rule is in
     *             @ref md_docs_2cpp-api.
     */
    template <Backend B, typename T>
    [[deprecated("renamed to stedc_buffer_size")]]
    inline size_t stedc_workspace_size(Queue& ctx, size_t n, size_t batch_size, JobType jobz, StedcParams<T> params) {
        return stedc_buffer_size<B, T>(ctx, n, batch_size, jobz, params);
    }

    /// @}

    /// @addtogroup api_eigen
    /// @{

    /**
     * @brief Computes the Ritz values (Rayleigh quotients) of a set of trial vectors.
     *
     * \f$ \theta_j = \dfrac{v_j^H A v_j}{v_j^H v_j} \f$ for each column \f$ v_j \f$ of `V`,
     * per batch item.
     *
     * @param ctx Execution context/device queue
     * @param A Matrix (can be sparse or dense)
     * @param V Trial vectors (dense matrix, columns are trial eigenvectors)
     * @param ritz_vals Output vector for Ritz values, `V.cols()` real entries per item
     * @param workspace Pre-allocated workspace buffer, at least ritz_values_buffer_size() bytes
     * @return Event Event to track operation completion
     * @see @ref guide_ritz_values
     */
    template <Backend B, typename T, MatrixFormat MFormat>
    BATCHLAS_API Event ritz_values(Queue& ctx,
                                   const MatrixView<T, MFormat>& A,
                                   const MatrixView<T, MatrixFormat::Dense>& V,
                                   const VectorView<typename base_type<T>::type>& ritz_vals,
                                   Span<std::byte> workspace);

    /**
     * @brief ritz_values() returning a new `Vector` of `V.cols()` Ritz values per item.
     *
     * Allocates its own workspace and waits on `ctx` before returning, so the result is
     * ready to read.
     * @see @ref guide_ritz_values
     */
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
        // (void) on an Event: deliberate. This Queue is in-order, so the next submission
        // is already ordered after this one and the Event carries nothing the caller needs.
        (void)ritz_values<B,T,MFormat>(ctx, A, V, static_cast<VectorView<float_type>>(ritz_vals), workspace);
        ctx.wait();
        return ritz_vals;
    }

    /**
     * @brief The allocating ritz_values() on owning `A` and `V`.
     *
     * @trap The one owning-argument twin BATCHLAS_ACCEPT_OWNING cannot replace: a call
     *       with some template arguments explicit and the rest deduced, such as
     *       `ritz_values<B, float_type>(ctx, A, V)`, puts `float_type` into the generated
     *       forwarder's pack, and the primary cannot deduce `MFormat` through
     *       `Matrix -> MatrixView`. `ritz_values<B>(ctx, A, V)` needs no overload.
     */
    template <Backend B, typename T, MatrixFormat MFormat>
    inline auto ritz_values(Queue& ctx,
                            const Matrix<T, MFormat>& A,
                            const Matrix<T, MatrixFormat::Dense>& V) {
        return ritz_values<B,T,MFormat>(ctx, MatrixView<T, MFormat>(A), MatrixView<T, MatrixFormat::Dense>(V));
    }


    /** @brief Required workspace size, in bytes, for ritz_values(). */
    template <Backend B, typename T, MatrixFormat MFormat>
    BATCHLAS_API size_t ritz_values_buffer_size(Queue& ctx,
                                              const MatrixView<T, MFormat>& A,
                                              const MatrixView<T, MatrixFormat::Dense>& V,
                                              const VectorView<typename base_type<T>::type>& ritz_vals);

    /**
     * @brief Old name of ritz_values_buffer_size() (view arguments).
     * @deprecated Use ritz_values_buffer_size(); kept so out-of-tree callers get a warning.
     */
    template <Backend B, typename T, MatrixFormat MFormat>
    [[deprecated("renamed to ritz_values_buffer_size")]]
    inline size_t ritz_values_workspace(Queue& ctx,
                                        const MatrixView<T, MFormat>& A,
                                        const MatrixView<T, MatrixFormat::Dense>& V,
                                        const VectorView<typename base_type<T>::type>& ritz_vals) {
        return ritz_values_buffer_size<B,T,MFormat>(ctx, A, V, ritz_vals);
    }

    /**
     * @brief Old name of ritz_values_buffer_size() (owning arguments).
     * @deprecated Use ritz_values_buffer_size(); kept so out-of-tree callers get a warning.
     */
    template <Backend B, typename T, MatrixFormat MFormat>
    [[deprecated("renamed to ritz_values_buffer_size")]]
    inline size_t ritz_values_workspace(Queue& ctx,
                                        const Matrix<T, MFormat>& A,
                                        const Matrix<T, MatrixFormat::Dense>& V,
                                        const Vector<typename base_type<T>::type>& ritz_vals) {
        return ritz_values_buffer_size<B,T,MFormat>(ctx, A, V, ritz_vals);
    }

    /// @}

    /// @addtogroup api_extra
    /// @{

    /**
     * @brief Computes the explicit inverse of each dense square matrix in the batch,
     *        \f$ A_{inv} := A^{-1} \f$, by LU factorisation (getrf()) and getri().
     *
     * `A` is copied into the workspace first and is not modified. Singular items are not
     * reported: check the result where the input may be ill-conditioned.
     *
     * @param ctx Execution context/device queue
     * @param A Input matrices to invert, n x n per item; read only
     * @param Ainv Output matrices storing the inverses, n x n per item
     * @param workspace Pre-allocated workspace buffer, at least inv_buffer_size() bytes
     * @return Event Event to track operation completion
     */
    template <Backend B, typename T>
    BATCHLAS_API Event inv(Queue& ctx,
                     const MatrixView<T, MatrixFormat::Dense>& A,
                     const MatrixView<T, MatrixFormat::Dense>& Ainv,
                     Span<std::byte> workspace);

    /** @brief Required workspace size, in bytes, for inv(). */
    template <Backend B, typename T>
    BATCHLAS_API size_t inv_buffer_size(Queue& ctx,
                     const MatrixView<T, MatrixFormat::Dense>& A);

    /**
     * @brief inv() returning a newly allocated result; workspace is leased from the
     *        Queue's arena. Returns without waiting.
     */
    template <Backend B, typename T>
    BATCHLAS_API Matrix<T, MatrixFormat::Dense> inv(Queue& ctx,
                     const MatrixView<T, MatrixFormat::Dense>& A);

    /** @brief The allocating inv() on an owning `A`. */
    template <Backend B, typename T>
    inline Matrix<T, MatrixFormat::Dense> inv_matrix(Queue& ctx,
        const Matrix<T, MatrixFormat::Dense>& A) {
        return inv<B,T>(ctx, MatrixView<T, MatrixFormat::Dense>(A));
    }

    /**
     * @brief Scales `mat` by `cto / cfrom` without overflow or underflow (LAPACK `?lascl`).
     *
     * The factor is applied in safe steps when forming `cto / cfrom` directly would
     * over- or underflow. Instantiated for real `float` and `double`, dense format.
     * @return event of the last scaling step
     */
    template <MatrixFormat MType, typename T>
    BATCHLAS_API Event lascl(Queue& ctx, const MatrixView<T, MType>& mat, T cfrom, T cto);

    /// @}
}

namespace batchlas {

// Backend-deducing (BATCHLAS_DISPATCH_ON_QUEUE) and owning-argument (BATCHLAS_ACCEPT_OWNING)
// overloads; see blas/queue-dispatch.hh. A name absent from the owning list has its own
// arity-changing forwarder above (the A-only `syevx` and `lanczos`), which no pack expresses.
// evidence: docs/cpp-api.md#which-type-each-parameter-takes

BATCHLAS_ACCEPT_OWNING(ortho)
BATCHLAS_ACCEPT_OWNING(ortho_buffer_size)
BATCHLAS_ACCEPT_OWNING(syevx)
BATCHLAS_ACCEPT_OWNING(syevx_buffer_size)
BATCHLAS_ACCEPT_OWNING(lanczos)
BATCHLAS_ACCEPT_OWNING(tridiagonal_solver)
BATCHLAS_ACCEPT_OWNING(stebz)
BATCHLAS_ACCEPT_OWNING(stein)
BATCHLAS_ACCEPT_OWNING(steqr)
BATCHLAS_ACCEPT_OWNING(steqr_cta)
BATCHLAS_ACCEPT_OWNING(stedc)
BATCHLAS_ACCEPT_OWNING(ritz_values)
BATCHLAS_ACCEPT_OWNING(ritz_values_buffer_size)
BATCHLAS_ACCEPT_OWNING(inv)
BATCHLAS_ACCEPT_OWNING(inv_buffer_size)
BATCHLAS_ACCEPT_OWNING_NB(francis_sweep)
BATCHLAS_ACCEPT_OWNING_NB(steqr_buffer_size)
BATCHLAS_ACCEPT_OWNING_NB(steqr_cta_buffer_size)

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

// Hand-written because BATCHLAS_DISPATCH_ON_QUEUE has nowhere to put [[deprecated]].

/**
 * @brief Queue-deducing form of the old stedc_workspace_size() name.
 * @deprecated Use stedc_buffer_size().
 * @ingroup api_tridiag
 */
template <typename... Args>
    requires requires(Queue& probe_ctx, Args&&... probe_args) {
        stedc_buffer_size<::batchlas::detail::kProbeBackend>(probe_ctx, std::forward<Args>(probe_args)...);
    }
[[deprecated("renamed to stedc_buffer_size")]]
inline auto stedc_workspace_size(Queue& ctx, Args&&... args) {
    return stedc_buffer_size(ctx, std::forward<Args>(args)...);
}

/**
 * @brief Queue-deducing form of the old ritz_values_workspace() name.
 * @deprecated Use ritz_values_buffer_size().
 * @ingroup api_eigen
 */
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
// blas/functions.hh (which ends by including options.hh) BEFORE this header, so `ortho` is
// not yet declared when options.hh is parsed, and none of its helpers are reachable here.

/// @addtogroup api_qr
/// @{

/** @brief Options for the option-struct ortho(); fields as the positional arguments. */
struct OrthoOptions {
    Transpose transA = Transpose::NoTrans;             ///< Columns (`NoTrans`) or rows.
    OrthoAlgorithm algorithm = OrthoAlgorithm::Chol2;  ///< Orthogonalisation algorithm.
};

/** @brief Options for the option-struct ortho() against an external basis `M`. */
struct OrthoAgainstOptions {
    Transpose transA = Transpose::NoTrans;             ///< Vectors of `A` are its columns (`NoTrans`) or rows.
    Transpose transM = Transpose::NoTrans;             ///< Vectors of `M` are its columns (`NoTrans`) or rows.
    OrthoAlgorithm algorithm = OrthoAlgorithm::Chol2;  ///< Orthogonalisation algorithm.
    size_t iterations = 2;                             ///< Project-then-orthonormalise passes.
};

// ---- in-place orthogonalisation --------------------------------------------

/** @brief ortho() with an options struct and a caller-provided workspace. */
template <Backend B, typename T>
inline Event ortho(Queue& ctx,
        const MatrixView<T, MatrixFormat::Dense>& A,
        const OrthoOptions& opts,
        Span<std::byte> workspace) {
    return ortho<B, T>(ctx, A, opts.transA, workspace, opts.algorithm);
}

/**
 * @brief ortho() with an options struct; workspace is leased from the Queue's arena.
 *
 * The lease is released when the call returns, which on an out-of-order Queue drains
 * the device first: this spelling is synchronous there, the span-taking one is not.
 */
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

/// @}

namespace detail {
// The bare-`{}` trap, in the one place ortho has it: `ortho(ctx, A, {}, ws)` matches both
// the positional `(ctx, A, Transpose, ws, algo)` and the option `(ctx, A, OrthoOptions, ws)`
// overload, and the positional one would win silently, handing the caller `Transpose{}`. A
// third candidate at the same exact-match rank makes the bare-`{}` call ambiguous instead.
enum class OrthoEmptyBracesAreAmbiguous {};
}  // namespace detail

/// @addtogroup api_qr
/// @{

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
// No `{}` guard is needed here: the positional metric form needs at least six arguments
// while the option forms take four and five, so no argument list can reach both.

/** @brief ortho() against `M` with an options struct and a caller-provided workspace. */
template <Backend B, typename T>
inline Event ortho(Queue& ctx,
        const MatrixView<T, MatrixFormat::Dense>& A,
        const MatrixView<T, MatrixFormat::Dense>& M,
        const OrthoAgainstOptions& opts,
        Span<std::byte> workspace) {
    return ortho<B, T>(ctx, A, M, opts.transA, opts.transM, workspace, opts.algorithm,
                       opts.iterations);
}

/** @brief ortho() against `M` with an options struct; workspace leased from the arena. */
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

/// @}

}  // namespace batchlas
