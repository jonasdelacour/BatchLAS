// SPDX-License-Identifier: MIT
#pragma once
#include <batchlas/export.hh>
#include <complex>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
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
        /// ASCENDING spectrum. \f$\mathit{il} > \mathit{iu}\f$ is an empty request and is rejected.
        int64_t il = 0;
        int64_t iu = -1;                                   ///< `select == Index`: last wanted index, inclusive; \f$\mathit{iu} < 0\f$ means \f$n-1\f$.
        /// `select == Value`: lower end of the half-open interval \f$(\mathit{vl}, \mathit{vu}]\f$, as in LAPACK.
        /// The count is data-dependent per item and is reported through syevx's `m` output.
        float_type vl = float_type(0);
        float_type vu = float_type(0);                     ///< `select == Value`: upper end (inclusive).

        /// Absolute tolerance per eigenvalue; non-positive means \f$ \varepsilon \lVert T \rVert \f$.
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

    /// @addtogroup api_qr
    /// @{

    /**
     * @brief Orthonormalises the columns (or rows) of each batch item in place.
     *
     * With `transA == NoTrans` each \f$n\f$-column block of `A` is replaced by an orthonormal
     * basis of its column space, \f$ A \leftarrow Q \f$ with \f$ Q^H Q = I \f$; with
     * `Trans` the rows are orthonormalised instead. Asynchronous.
     *
     * @param[in]     ctx       queue the work is enqueued on
     * @param[in,out] A         batch of \f$m \times n\f$ matrices; overwritten with the orthonormal vectors
     * @param[in]     transA    `NoTrans` for columns, `Trans` for rows
     * @param[out]    workspace scratch of at least ortho_buffer_size() bytes for the same `algo`;
     *                          contents undefined on return
     * @param[in]     algo      orthogonalisation algorithm (see OrthoAlgorithm)
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
     * @param[in]     ctx        queue the work is enqueued on
     * @param[in,out] A          batch of vectors to orthonormalise; overwritten with the result
     * @param[in]     M          external basis, read only
     * @param[in]     transA     `NoTrans`: the vectors are the columns of `A`; transposed: its rows
     * @param[in]     transM     `NoTrans`: the basis vectors are the columns of `M`; transposed: its rows
     * @param[out]    workspace  scratch of at least ortho_buffer_size() bytes for the same
     *                           arguments; contents undefined on return
     * @param[in]     algo       orthogonalisation algorithm run after each projection
     * @param[in]     iterations number of project-then-orthonormalise passes
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

    /**
     * @brief Required workspace, in bytes, for ortho(); arguments as for the call itself.
     * @param[in] ctx    queue the call will run on
     * @param[in] A      batch of \f$m \times n\f$ matrices to be orthonormalised (shape only)
     * @param[in] transA `NoTrans` for columns, `Trans` for rows
     * @param[in] algo   orthogonalisation algorithm the call will run
     * @return the workspace size in bytes
     * @ingroup api_qr_lowlevel
     */
    template <Backend B, typename T>
    BATCHLAS_API size_t ortho_buffer_size(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         Transpose transA,
                         OrthoAlgorithm algo = OrthoAlgorithm::Chol2);

    /**
     * @brief Required workspace, in bytes, for the external-basis ortho(); same arguments.
     * @param[in] ctx        queue the call will run on
     * @param[in] A          batch of vectors to be orthonormalised (shape only)
     * @param[in] M          external basis (shape only)
     * @param[in] transA     `NoTrans`: the vectors are the columns of `A`; transposed: its rows
     * @param[in] transM     `NoTrans`: the basis vectors are the columns of `M`; transposed: its rows
     * @param[in] algo       orthogonalisation algorithm the call will run
     * @param[in] iterations number of project-then-orthonormalise passes
     * @return the workspace size in bytes
     * @ingroup api_qr_lowlevel
     */
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
     * @param[in]  ctx       queue the work is enqueued on
     * @param[in]  A         batch of \f$n \times n\f$ symmetric/Hermitian matrices, dense or CSR; not modified
     * @param[out] W         eigenvalue output (real type), `neigs` entries per batch item
     * @param[in]  neigs     CAPACITY of `W` and of `V`'s columns per item, not the number produced
     * @param[out] workspace scratch of at least syevx_buffer_size() bytes; contents undefined on return
     * @param[in]  jobz      whether eigenvectors are written to `V`
     * @param[out] V         \f$n \times \mathit{neigs}\f$ eigenvector output per item; used only for `EigenVectors`
     * @param[in]  params    selection range and algorithm parameters
     * @param[out] info      per-item convergence status (0 means converged, a positive value is
     *                       LAPACK-like), or an empty span to not request it; see @ref md_docs_2cpp-api
     * @return event of the last enqueued kernel
     * @throws batchlas::invalid_argument for `SyevxSelect::Value` (use the `m`-taking
     *         overload); an `Index` block outside \f$[0, n)\f$ or with \f$\mathit{neigs} \ne \mathit{iu} - \mathit{il} + 1\f$;
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

    /**
     * @brief syevx() on an owning `A` without eigenvectors: no `V` argument.
     * @param[in]  ctx       queue the work is enqueued on
     * @param[in]  A         batch of \f$n \times n\f$ symmetric/Hermitian matrices; not modified
     * @param[out] W         eigenvalue output (real type), `neigs` entries per batch item
     * @param[in]  neigs     capacity of `W` per item, not the number produced
     * @param[out] workspace scratch of at least syevx_buffer_size() bytes; contents undefined on return
     * @param[in]  jobz      whether eigenvectors are wanted; this form forwards an empty `V`
     * @param[in]  params    selection range and algorithm parameters
     * @param[out] info      per-item convergence status, or empty; see @ref md_docs_2cpp-api
     * @return event of the last enqueued kernel
     */
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
     * Item \f$b\f$ gets \f$\min(m_b, \mathit{neigs})\f$ eigenpairs, where \f$m_b\f$ is `m[b]`
     * (the lowest ones, when truncating), the rest of its `W` is left untouched, the
     * remaining columns of its `V` are written as EXACTLY ZERO so the back-transform can run
     * over a uniform column count, and `m[b]` reports the TRUE count, so
     * \f$m_b > \mathit{neigs}\f$ is the caller's overflow signal.
     * Other parameters as for the `m`-less form.
     *
     * @param[in]  ctx       queue the work is enqueued on
     * @param[in]  A         batch of \f$n \times n\f$ symmetric/Hermitian matrices, dense or CSR; not modified
     * @param[out] W         eigenvalue output (real type), `neigs` entries per batch item
     * @param[out] m         per-item TRUE count of eigenpairs in the range, at least
     *                       `A.batch_size()` entries; device-writable
     * @param[in]  neigs     CAPACITY of `W` and of `V`'s columns per item
     * @param[out] workspace scratch of at least syevx_buffer_size() bytes; contents undefined on return
     * @param[in]  jobz      whether eigenvectors are written to `V`
     * @param[out] V         \f$n \times \mathit{neigs}\f$ eigenvector output per item; used only for `EigenVectors`
     * @param[in]  params    selection range and algorithm parameters
     * @param[out] info      per-item convergence status, or empty; see @ref md_docs_2cpp-api
     * @return event of the last enqueued kernel
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

    /**
     * @brief The `m`-reporting syevx() on an owning `A` without eigenvectors.
     * @param[in]  ctx       queue the work is enqueued on
     * @param[in]  A         batch of \f$n \times n\f$ symmetric/Hermitian matrices; not modified
     * @param[out] W         eigenvalue output (real type), `neigs` entries per batch item
     * @param[out] m         per-item TRUE count of eigenvalues in the range, at least
     *                       `A.batch_size()` entries
     * @param[in]  neigs     capacity of `W` per item
     * @param[out] workspace scratch of at least syevx_buffer_size() bytes; contents undefined on return
     * @param[in]  jobz      whether eigenvectors are wanted; this form forwards an empty `V`
     * @param[in]  params    selection range and algorithm parameters
     * @param[out] info      per-item convergence status, or empty; see @ref md_docs_2cpp-api
     * @return event of the last enqueued kernel
     */
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
     * @param[in] ctx    queue the call will run on
     * @param[in] A      batch of \f$n \times n\f$ symmetric/Hermitian matrices (shape and format only)
     * @param[in] W      eigenvalue output the call will use; not written
     * @param[in] neigs  capacity of `W` and of `V`'s columns per item
     * @param[in] jobz   whether the call will write eigenvectors
     * @param[in] V      eigenvector output the call will use; not written
     * @param[in] params selection range and algorithm parameters of the call
     * @return the workspace size in bytes
     * @ingroup api_eigen_lowlevel
     */
    template <Backend B, typename T, MatrixFormat MFormat>
    BATCHLAS_API size_t syevx_buffer_size(Queue& ctx,
                             const MatrixView<T, MFormat>& A,
                             Span<typename base_type<T>::type> W,
                             size_t neigs,
                             JobType jobz = JobType::NoEigenVectors,
                             const MatrixView<T, MatrixFormat::Dense>& V = MatrixView<T, MatrixFormat::Dense>(),
                             const SyevxParams<T>& params = SyevxParams<T>());

    /**
     * @brief syevx_buffer_size() on an owning `A` without eigenvectors.
     * @param[in] ctx    queue the call will run on
     * @param[in] A      batch of \f$n \times n\f$ symmetric/Hermitian matrices (shape and format only)
     * @param[in] W      eigenvalue output the call will use; not written
     * @param[in] neigs  capacity of `W` per item
     * @param[in] jobz   whether eigenvectors are wanted; sized with an empty `V`
     * @param[in] params selection range and algorithm parameters of the call
     * @return the workspace size in bytes
     * @ingroup api_eigen_lowlevel
     */
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
     * @param[in] ctx    queue the call will run on
     * @param[in] A      batch of \f$n \times n\f$ symmetric/Hermitian matrices (shape and format only)
     * @param[in] W      eigenvalue output the call will use; not written
     * @param[in] m      the solve's count output; ignored and not written
     * @param[in] neigs  capacity of `W` and of `V`'s columns per item
     * @param[in] jobz   whether the call will write eigenvectors
     * @param[in] V      eigenvector output the call will use; not written
     * @param[in] params selection range and algorithm parameters of the call
     * @return the workspace size in bytes
     * @ingroup api_eigen_lowlevel
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

    /**
     * @brief The `m`-taking syevx_buffer_size() on an owning `A`; `m` is ignored.
     * @param[in] ctx    queue the call will run on
     * @param[in] A      batch of \f$n \times n\f$ symmetric/Hermitian matrices (shape and format only)
     * @param[in] W      eigenvalue output the call will use; not written
     * @param[in] m      the solve's count output; ignored and not written
     * @param[in] neigs  capacity of `W` per item
     * @param[in] jobz   whether eigenvectors are wanted; sized with an empty `V`
     * @param[in] params selection range and algorithm parameters of the call
     * @return the workspace size in bytes
     * @ingroup api_eigen_lowlevel
     */
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
        bool    value_range;  ///< true: \f$(\mathit{vl}, \mathit{vu}]\f$; false: the index block \f$[\mathit{il}, \mathit{iu}]\f$
        int64_t il;           ///< valid iff !value_range; 0-based, inclusive
        int64_t iu;           ///< valid iff !value_range; 0-based, inclusive
        /// Upper bound on eigenpairs per item, clamped to \f$[0, n]\f$ so a consumer may index
        /// \f$[\mathit{il}, \mathit{il} + \mathit{max\_count})\f$ unguarded. Exact for an index block; a value range may exceed it.
        int64_t max_count;
        bool    reverse;      ///< write the selected block in descending order
    };

    /**
     * @brief Normalizes a range request into a SyevxResolvedRange. Legality is NOT checked
     *        here; it clamps rather than throws so it stays usable when sizing.
     *
     * @param[in] n            matrix dimension
     * @param[in] neigs        capacity of `W` and `V` per batch item (see `syevx`)
     * @param[in] select       SyevxParams::select
     * @param[in] find_largest SyevxParams::find_largest (Extremal only)
     * @param[in] il,iu        SyevxParams::il / iu (Index only; \f$\mathit{iu} < 0\f$ means \f$n-1\f$)
     * @param[in] order        SyevxParams::order (Index and Value only)
     * @return the normalised request
     * @ingroup api_eigen_lowlevel
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
    /// @param[in] n      matrix dimension
    /// @param[in] neigs  capacity of `W` and `V` per batch item (see `syevx`)
    /// @param[in] params source of `select`, `find_largest`, `il`, `iu` and `order`
    /// @return the normalised request
    /// @ingroup api_eigen_lowlevel
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
     * @param[in] format           matrix format of `A` (sparse formats run LOBPCG unless
     *                             `Filtered` is requested explicitly)
     * @param[in] n                matrix dimension
     * @param[in] neigs            number of requested eigenpairs
     * @param[in] requested        algorithm requested via SyevxParams::method
     * @param[in] subset_supported whether DirectSubset is available for this `T`/format
     * @param[in] jobz             whether eigenvectors are wanted; load-bearing for the choice
     * @param[in] batch_size       number of batch items; load-bearing: the subset solver
     *                             starves at small batch
     * @param[in] select           range kind; excludes the algorithms that cannot answer it
     * @return the algorithm syevx() will run
     * @throws batchlas::invalid_argument for sparse input, or an explicit LOBPCG/Filtered
     *         `method`, with a non-extremal range. The same request made through
     *         `BATCHLAS_SYEVX_ALGORITHM` degrades to `Direct` with a one-time warning.
     * @see @ref perf_syevx for the measured thresholds behind the choice
     * @ingroup api_eigen_lowlevel
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
     * @brief Partial eigensolve by full decomposition followed by selection. Runs `syev`
     *        on a private copy of A. Dense input only, but every scalar type and every
     *        `SyevxSelect` range: the universal fallback.
     *
     * @param[in]  ctx       queue the work is enqueued on
     * @param[in]  A         batch of \f$n \times n\f$ symmetric/Hermitian dense matrices; not modified
     * @param[out] W         eigenvalue output, stride `neigs`; entries past \f$\min(m_b, \mathit{neigs})\f$
     *                       untouched, where \f$m_b\f$ is `m[b]`
     * @param[out] m         per-item count in the requested range, or an empty span to not report it.
     *                       \f$m_b > \mathit{neigs}\f$ is the truncation signal; the LOWEST `neigs` are kept.
     * @param[in]  neigs     capacity of `W` and of `V`'s columns per item
     * @param[out] workspace scratch of at least syevx_direct_buffer_size() bytes; contents undefined on return
     * @param[in]  jobz      whether eigenvectors are written to `V`
     * @param[out] V         \f$n \times \mathit{neigs}\f$ eigenvector output. Columns past \f$\min(m_b, \mathit{neigs})\f$
     *                       are written as EXACTLY ZERO -- unlike `W` -- so this path and
     *                       `syevx_direct_subset` agree.
     * @param[in]  params    selection range and algorithm parameters
     * @param[out] info      per-item convergence status, or empty; see @ref md_docs_2cpp-api
     * @return event of the last enqueued kernel
     * @ingroup api_eigen_lowlevel
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
     * @param[in]  ctx       queue the work is enqueued on
     * @param[in]  A         batch of \f$n \times n\f$ symmetric/Hermitian dense matrices; not modified
     * @param[out] W         eigenvalue output, `neigs` entries per batch item
     * @param[in]  neigs     capacity of `W` and of `V`'s columns per item
     * @param[out] workspace scratch of at least syevx_direct_buffer_size() bytes; contents undefined on return
     * @param[in]  jobz      whether eigenvectors are written to `V`
     * @param[out] V         \f$n \times \mathit{neigs}\f$ eigenvector output per item; used only for `EigenVectors`
     * @param[in]  params    selection range and algorithm parameters
     * @param[out] info      per-item convergence status, or empty; see @ref md_docs_2cpp-api
     * @return event of the last enqueued kernel
     * @ingroup api_eigen_lowlevel
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
     * @param[in] ctx    queue the call will run on
     * @param[in] A      batch of \f$n \times n\f$ dense matrices (shape only)
     * @param[in] W      eigenvalue output the call will use; not written
     * @param[in] neigs  capacity of `W` and of `V`'s columns per item
     * @param[in] jobz   whether the call will write eigenvectors
     * @param[in] V      eigenvector output the call will use; not written
     * @param[in] params selection range and algorithm parameters of the call
     * @return the workspace size in bytes
     * @ingroup api_eigen_lowlevel
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
     *        input only; zero capacity is rejected (stein requires \f$k \ge 1\f$).
     *
     * `W`, `V` and `m` follow the same contract as `syevx_direct`, with `stein` doing the
     * zeroing. The descending reversal is applied last, in the finalize kernel: `stein`'s
     * cluster detection walks consecutive eigenvalues and requires ascending input.
     * Other parameters as for syevx().
     *
     * @param[in]  ctx       in-order queue the work is enqueued on
     * @param[in]  A         batch of \f$n \times n\f$ real symmetric dense matrices; not modified
     * @param[out] W         eigenvalue output, stride `neigs`; entries past \f$\min(m_b, \mathit{neigs})\f$
     *                       untouched, where \f$m_b\f$ is `m[b]`
     * @param[out] m         per-item count in the requested range, or an empty span to not report it
     * @param[in]  neigs     capacity of `W` and of `V`'s columns per item, at least 1
     * @param[out] workspace scratch of at least syevx_direct_subset_buffer_size() bytes;
     *                       contents undefined on return
     * @param[in]  jobz      whether eigenvectors are written to `V`
     * @param[out] V         \f$n \times \mathit{neigs}\f$ eigenvector output; columns past
     *                       \f$\min(m_b, \mathit{neigs})\f$ are written as EXACTLY ZERO
     * @param[in]  params    selection range and algorithm parameters
     * @param[out] info      per-item convergence status, or empty; see @ref md_docs_2cpp-api
     * @return event of the last enqueued kernel
     * @pre in-order Queue; `A` square
     * @throws batchlas::unsupported for complex or sparse input
     * @throws batchlas::invalid_argument for `neigs == 0`, an out-of-range index block,
     *         \f$\mathit{vl} \ge \mathit{vu}\f$, a too-short `m`, or an out-of-order Queue
     * @see syevx_direct_subset_supported()
     * @ingroup api_eigen_lowlevel
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

    /**
     * @brief `syevx_direct_subset` without the `m` output; arity-disambiguated as above.
     * @param[in]  ctx       in-order queue the work is enqueued on
     * @param[in]  A         batch of \f$n \times n\f$ real symmetric dense matrices; not modified
     * @param[out] W         eigenvalue output, `neigs` entries per batch item
     * @param[in]  neigs     capacity of `W` and of `V`'s columns per item, at least 1
     * @param[out] workspace scratch of at least syevx_direct_subset_buffer_size() bytes;
     *                       contents undefined on return
     * @param[in]  jobz      whether eigenvectors are written to `V`
     * @param[out] V         \f$n \times \mathit{neigs}\f$ eigenvector output per item; used only for `EigenVectors`
     * @param[in]  params    selection range and algorithm parameters
     * @param[out] info      per-item convergence status, or empty; see @ref md_docs_2cpp-api
     * @return event of the last enqueued kernel
     * @ingroup api_eigen_lowlevel
     */
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
     * room for up to \f$n\f$ eigenvalues per item, so the sizes come from syevx_resolve_range().
     * @param[in] ctx    queue the call will run on
     * @param[in] A      batch of \f$n \times n\f$ dense matrices (shape only)
     * @param[in] W      eigenvalue output the call will use; not written
     * @param[in] neigs  capacity of `W` and of `V`'s columns per item
     * @param[in] jobz   whether the call will write eigenvectors
     * @param[in] V      eigenvector output the call will use; not written
     * @param[in] params selection range and algorithm parameters of the call
     * @return the workspace size in bytes
     * @ingroup api_eigen_lowlevel
     */
    template <Backend B, typename T, MatrixFormat MFormat>
    BATCHLAS_API size_t syevx_direct_subset_buffer_size(Queue& ctx,
                             const MatrixView<T, MFormat>& A,
                             Span<typename base_type<T>::type> W,
                             size_t neigs,
                             JobType jobz,
                             const MatrixView<T, MatrixFormat::Dense>& V,
                             const SyevxParams<T>& params);

    /**
     * @brief Whether `syevx_direct_subset` supports this scalar type and format.
     * @tparam T       scalar type
     * @tparam MFormat matrix format
     * @return true for real `T` with `MatrixFormat::Dense`
     * @ingroup api_eigen_lowlevel
     */
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
     *
     * @param[in]  ctx       queue the work is enqueued on
     * @param[in]  A         batch of \f$n \times n\f$ symmetric/Hermitian matrices, dense or CSR; not modified
     * @param[out] W         eigenvalue output (real type), `neigs` entries per batch item
     * @param[in]  neigs     number of eigenpairs per item
     * @param[out] workspace scratch of at least syevx_lobpcg_buffer_size() bytes; contents undefined on return
     * @param[in]  jobz      whether eigenvectors are written to `V`
     * @param[out] V         \f$n \times \mathit{neigs}\f$ eigenvector output per item; used only for `EigenVectors`
     * @param[in]  params    tolerances, iteration cap, preconditioner and extremal selection
     * @param[out] info      per-item convergence status, or empty; see @ref md_docs_2cpp-api
     * @return event of the last enqueued kernel
     * @ingroup api_eigen_lowlevel
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

    /**
     * @brief Required workspace, in bytes, for syevx_lobpcg().
     * @param[in] ctx    queue the call will run on
     * @param[in] A      batch of \f$n \times n\f$ matrices, dense or CSR (shape and format only)
     * @param[in] W      eigenvalue output the call will use; not written
     * @param[in] neigs  number of eigenpairs per item
     * @param[in] jobz   whether the call will write eigenvectors
     * @param[in] V      eigenvector output the call will use; not written
     * @param[in] params parameters of the call
     * @return the workspace size in bytes
     * @ingroup api_eigen_lowlevel
     */
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
     *
     * @param[in]  ctx       queue the work is enqueued on
     * @param[in]  A         batch of \f$n \times n\f$ symmetric/Hermitian matrices, dense or CSR; not modified
     * @param[out] W         eigenvalue output (real type), `neigs` entries per batch item
     * @param[in]  neigs     number of eigenpairs per item
     * @param[out] workspace scratch of at least syevx_filtered_buffer_size() bytes; contents undefined on return
     * @param[in]  jobz      whether eigenvectors are written to `V`
     * @param[out] V         \f$n \times \mathit{neigs}\f$ eigenvector output per item; used only for `EigenVectors`
     * @param[in]  params    filter degree, tolerances, iteration cap and extremal selection
     * @param[out] info      per-item convergence status, or empty; see @ref md_docs_2cpp-api
     * @return event of the last enqueued kernel
     * @ingroup api_eigen_lowlevel
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

    /**
     * @brief Required workspace, in bytes, for syevx_filtered().
     * @param[in] ctx    queue the call will run on
     * @param[in] A      batch of \f$n \times n\f$ matrices, dense or CSR (shape and format only)
     * @param[in] W      eigenvalue output the call will use; not written
     * @param[in] neigs  number of eigenpairs per item
     * @param[in] jobz   whether the call will write eigenvectors
     * @param[in] V      eigenvector output the call will use; not written
     * @param[in] params parameters of the call
     * @return the workspace size in bytes
     * @ingroup api_eigen_lowlevel
     */
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
     * Builds an \f$n\f$-step Krylov basis with reorthogonalisation, then solves the resulting
     * tridiagonal problem. Intended for sparse (CSR) input; dense works too.
     *
     * @param[in]  ctx       queue the work is enqueued on
     * @param[in]  A         batch of \f$n \times n\f$ symmetric/Hermitian matrices (CSR or dense); read only
     * @param[out] W         eigenvalue output (real type), \f$n\f$ entries per batch item
     * @param[out] workspace scratch of at least lanczos_buffer_size() bytes; contents undefined on return
     * @param[in]  jobz      whether eigenvectors are written to `V`
     * @param[out] V         \f$n \times n\f$ eigenvector output per item; used only for `EigenVectors`
     * @param[in]  params    orthogonalisation and sorting options
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

    /**
     * @brief lanczos() on an owning `A` without eigenvectors: no `V` argument.
     * @param[in]  ctx       queue the work is enqueued on
     * @param[in]  A         batch of \f$n \times n\f$ symmetric/Hermitian matrices (CSR or dense); read only
     * @param[out] W         eigenvalue output (real type), \f$n\f$ entries per batch item
     * @param[out] workspace scratch of at least lanczos_buffer_size() bytes; contents undefined on return
     * @param[in]  jobz      whether eigenvectors are wanted; this form forwards an empty `V`
     * @param[in]  params    orthogonalisation and sorting options
     * @return event of the last enqueued kernel
     */
    template <Backend B, typename T, MatrixFormat MFormat>
    inline Event lanczos(Queue& ctx,
        const Matrix<T, MFormat>& A,
        Span<typename base_type<T>::type> W,
        Span<std::byte> workspace,
        JobType jobz = JobType::NoEigenVectors,
        const LanczosParams<T>& params = LanczosParams<T>()) {
        return lanczos<B,T,MFormat>(ctx, MatrixView<T,MFormat>(A), W, workspace, jobz, MatrixView<T, MatrixFormat::Dense>(), params);
    }

    /**
     * @brief Required workspace, in bytes, for lanczos(); arguments as for the call itself.
     * @param[in] ctx    queue the call will run on
     * @param[in] A      batch of \f$n \times n\f$ matrices, CSR or dense (shape and format only)
     * @param[in] W      eigenvalue output the call will use; not written
     * @param[in] jobz   whether the call will write eigenvectors
     * @param[in] V      eigenvector output the call will use; not written
     * @param[in] params orthogonalisation and sorting options of the call
     * @return the workspace size in bytes
     * @ingroup api_eigen_lowlevel
     */
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
     *        tridiagonal matrices by shifted QR with \f$2 \times 2\f$ reflections.
     *
     * One work-group of 32 per batch item; the working arrays live in local memory when
     * they fit and in the workspace otherwise. With `EigenVectors`, `Q` is first set to
     * the identity.
     *
     * @param[in]  ctx        queue the work is enqueued on
     * @param[in]  alphas     diagonals, packed: item \f$b\f$ occupies entries \f$[bn,\, bn+n)\f$ of `alphas`; read only
     * @param[in]  betas      off-diagonals, packed with the same stride `n`; read only
     * @param[out] W          eigenvalue output (real type), packed with stride `n`
     * @param[out] workspace  scratch of at least tridiagonal_solver_buffer_size() bytes;
     *                        contents undefined on return
     * @param[in]  jobz       whether `Q` receives the eigenvectors
     * @param[out] Q          \f$n \times n\f$ eigenvector output per item
     * @param[in]  n          order of each tridiagonal matrix
     * @param[in]  batch_size number of batch items
     * @return event of the kernel; the call waits for it before returning
     * @throws batchlas::convergence_error if an eigenvalue of any item does not deflate
     *         within 30 QR steps
     * @note steqr() and stedc() are the maintained tridiagonal solvers.
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

    /**
     * @brief Required workspace, in bytes, for tridiagonal_solver().
     * @param[in] ctx        queue the call will run on
     * @param[in] n          order of each tridiagonal matrix
     * @param[in] batch_size number of batch items
     * @param[in] jobz       whether the call will compute eigenvectors
     * @return the workspace size in bytes
     * @ingroup api_tridiag_lowlevel
     */
    template <Backend B, typename T>
    BATCHLAS_API size_t tridiagonal_solver_buffer_size(Queue& ctx, size_t n, size_t batch_size, JobType jobz);

    // Convention in the tridiagonal group below: `VectorView<T>` for one vector per batch
    // item (it carries inc/stride/batch_size), `Span<...>` for flat per-item arrays and byte
    // workspaces. Demoting a `VectorView` to a `Span` silently drops the stride.

    /** @brief How stebz() selects a subset of the spectrum. */
    enum class EigenRangeType {
        All,    ///< Every eigenvalue
        Index,  ///< Eigenvalues \f$\mathit{il}, \dots, \mathit{iu}\f$ inclusive, 0-based, in ascending order
        Value   ///< Eigenvalues in the half-open interval \f$(\mathit{vl}, \mathit{vu}]\f$
    };

    /** @brief Parameters for stebz() (bisection on a symmetric tridiagonal matrix). */
    template <typename T>
    struct StebzParams {
        EigenRangeType range = EigenRangeType::All;  ///< Which eigenvalues to compute.
        int64_t il = 0;    ///< First wanted index (0-based, inclusive), range == Index
        int64_t iu = -1;   ///< Last wanted index (0-based, inclusive), range == Index; \f$< 0\f$ means \f$n-1\f$
        T vl = T(0);       ///< Lower bound (exclusive), range == Value
        T vu = T(0);       ///< Upper bound (inclusive), range == Value
        /// Absolute tolerance on each eigenvalue. Non-positive means
        /// \f$ \varepsilon \lVert T \rVert \f$, i.e. full working precision.
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
     * @param[in]  ctx    execution context/device queue
     * @param[in]  d      diagonal, \f$n\f$ entries per batch item
     * @param[in]  e      off-diagonal, \f$n-1\f$ entries per batch item
     * @param[out] w      eigenvalue output; must hold at least the number selected (\f$n\f$ for `Value`)
     * @param[out] m      count of eigenvalues found, per batch item (device-written)
     * @param[out] ws     scratch of at least stebz_buffer_size() bytes; contents undefined on return
     * @param[in]  params selection range, tolerance and ordering
     * @return event to track operation completion
     * @throws batchlas::invalid_argument for \f$n < 1\f$, a short `e`, `m` or `w`, or an invalid
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
     * @param[in] ctx        queue the call will run on
     * @param[in] n          order of each tridiagonal matrix
     * @param[in] batch_size number of batch items
     * @param[in] params     selection range, tolerance and ordering of the call
     * @return the workspace size in bytes
     * @ingroup api_tridiag_lowlevel
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
        /// Eigenvalues closer than \f$\mathit{ortho\_threshold} \cdot \lVert T \rVert\f$ form one cluster and have their
        /// vectors explicitly reorthogonalised (LAPACK dstein uses \f$10^{-3}\f$).
        T ortho_threshold = T(1e-3);
        uint32_t seed = 0x5eed1234u;  ///< Seed of the random starting vectors.
    };

    /**
     * @brief Computes eigenvectors of a batch of symmetric tridiagonal matrices by inverse
     *        iteration, given previously computed eigenvalues. Pairs with stebz().
     *
     * @param[in]  ctx    execution context/device queue
     * @param[in]  d      diagonal, \f$n\f$ entries per batch item
     * @param[in]  e      off-diagonal, \f$n-1\f$ entries per batch item
     * @param[in]  w      eigenvalues, \f$k\f$ per batch item, in ascending order (cluster detection
     *                    walks consecutive values)
     * @param[in]  k      number of eigenvectors to compute, \f$k \ge 1\f$
     * @param[out] Z      eigenvector output, \f$n \times k\f$ per batch item, columns matching `w`
     * @param[out] ws     scratch of at least stein_buffer_size() bytes; contents undefined on return
     * @param[in]  params iteration count and clustering threshold
     * @return event to track operation completion
     * @throws batchlas::invalid_argument for \f$n < 1\f$, \f$k < 1\f$, or a short `e`, `w` or `Z`
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
     * @param[in]  ctx    execution context/device queue
     * @param[in]  d      diagonal, \f$n\f$ entries per batch item
     * @param[in]  e      off-diagonal, \f$n-1\f$ entries per batch item
     * @param[in]  w      eigenvalues, \f$k\f$ slots per batch item, the valid ones leading and ascending
     * @param[in]  k      capacity: columns of `Z` and entries of `w` per item, \f$k \ge 1\f$
     * @param[in]  counts per-item number of valid eigenvalues (device-read), or empty /
     *                    `stein_all_counts` for all \f$k\f$
     * @param[out] Z      eigenvector output, \f$n \times k\f$ per batch item; columns past
     *                    `counts[b]` are written as zero
     * @param[out] ws     scratch of at least stein_buffer_size() bytes; contents undefined on return
     * @param[in]  params iteration count and clustering threshold
     * @return event to track operation completion
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
     * @param[in] ctx        queue the call will run on
     * @param[in] n          order of each tridiagonal matrix
     * @param[in] k          capacity: eigenvectors per item
     * @param[in] batch_size number of batch items
     * @param[in] params     iteration count and clustering threshold of the call
     * @return the workspace size in bytes
     * @ingroup api_tridiag_lowlevel
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
        Wilkinson = 1,  ///< Wilkinson shift computed from the trailing \f$2 \times 2\f$ block.
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
        /// If false, the eigenvector matrix is set to the identity and the rotations are
        /// applied to it; if true, they are applied to the caller's matrix (a back-transform).
        bool back_transform = false;
        bool block_rotations = false;          ///< Apply the rotations in #block_size blocks.
        bool sort = true;                      ///< Sort eigenpairs on output.
        bool transpose_working_vectors = true; ///< Interleave the working copies of d and e across the batch.
        SortOrder sort_order = SortOrder::Ascending;  ///< Order used when #sort is set.

        /// CTA STEQR only: multiplies the baseline work-group size, \f$\operatorname{lcm}(P, s)\f$
        /// for sub-group size \f$s\f$ and the compile-time partition width \f$P\f$ chosen from \f$n\f$.
        /// 0 picks a tuned value per scalar type and partition width.
        size_t cta_wg_size_multiplier = 0;

        SteqrShiftStrategy cta_shift_strategy = SteqrShiftStrategy::Lapack;  ///< CTA STEQR only: shift strategy.
        SteqrUpdateScheme cta_update_scheme = SteqrUpdateScheme::EXP;        ///< CTA STEQR only: update scheme.
    };


    // `info` on steqr/stedc/syev/syevx/gesvd and their tiers: per-item convergence status,
    // an ACCUMULATOR zeroed once by the entry point the caller invoked and only raised
    // below (contract and helpers: src/kernels/common/info_span.hh).
    // evidence: docs/cpp-api.md#convergence-status-syev-syevx-gesvd-steqr-stedc
    // A DEFAULTED trailing parameter, not an old-arity forwarder as in potrf: these are
    // instantiated from macros that spell the list out, and with `jobz`, `params` and
    // `eigvects` already defaulted a forwarder would be AMBIGUOUS with the primary.
    // evidence: docs/design/vendor-independence.md#info-spans-on-syev-gesvd-and-steqr-forwarder-or-default

    /**
     * @brief Eigenvalues, and optionally eigenvectors, of a batch of symmetric tridiagonal
     *        matrices by implicit QR/QL iteration (LAPACK `?steqr`).
     *
     * Real `T` only. Runs the CTA kernel (steqr_cta()) when \f$n\f$ is at most the device's
     * largest sub-group size (never on ROCm) and the work-group kernel otherwise;
     * steqr_buffer_size() returns the larger of the two needs.
     *
     * @param[in]     ctx         queue the work is enqueued on
     * @param[in]     d           diagonal, \f$n\f$ entries per batch item; not modified
     * @param[in]     e           off-diagonal, \f$n-1\f$ entries per batch item; not modified
     * @param[out]    eigenvalues eigenvalue output, \f$n\f$ per batch item; ascending when
     *                            SteqrParams::sort is set
     * @param[out]    ws          scratch of at least steqr_buffer_size() bytes; contents undefined on return
     * @param[in]     jobz        whether `eigvects` receives the eigenvectors
     * @param[in]     params      sweep cap, rotation blocking, sorting and CTA tuning
     * @param[in,out] eigvects    \f$n \times n\f$ per item; with SteqrParams::back_transform the
     *                            rotations are applied to its contents, otherwise it is set to
     *                            the identity first; holds the eigenvectors on return
     * @param[out]    info        per-item convergence status (0 means converged, a positive value
     *                            is LAPACK-like), or empty to not request it; see @ref md_docs_2cpp-api
     * @throws batchlas::invalid_argument if `eigvects` is not \f$n \times n\f$ with the same batch
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
     *        \f$n\f$ (runtime-dispatched to compile-time specialised kernels). Parameters as for
     *        steqr().
     * @param[in]     ctx         queue the work is enqueued on
     * @param[in]     d           diagonal, \f$n\f$ entries per batch item; not modified
     * @param[in]     e           off-diagonal, \f$n-1\f$ entries per batch item; not modified
     * @param[out]    eigenvalues eigenvalue output, \f$n\f$ per batch item
     * @param[out]    ws          scratch of at least steqr_cta_buffer_size() bytes; contents undefined on return
     * @param[in]     jobz        whether `eigvects` receives the eigenvectors
     * @param[in]     params      sweep cap, sorting and CTA tuning
     * @param[in,out] eigvects    \f$n \times n\f$ per item; back-transformed with
     *                            SteqrParams::back_transform, otherwise set to the identity first
     * @param[out]    info        per-item convergence status, or empty; see @ref md_docs_2cpp-api
     * @return event of the last enqueued kernel
     * @throws batchlas::invalid_argument unless \f$1 \le n \le 32\f$
     * @throws batchlas::convergence_error if `BATCHLAS_STEQR_CTA_CHECK` is set and an item
     *         ran out of sweeps (the check waits on the queue; unset, nothing is thrown)
     * @ingroup api_tridiag_lowlevel
     */
    template <Backend B, typename T>
    BATCHLAS_API Event steqr_cta(Queue& ctx, const VectorView<T>& d, const VectorView<T>& e,
                                 const VectorView<T>& eigenvalues, const Span<std::byte>& ws,
                                 JobType jobz = JobType::NoEigenVectors,
                                 SteqrParams<T> params = SteqrParams<T>(),
                                 const MatrixView<T, MatrixFormat::Dense>& eigvects = MatrixView<T, MatrixFormat::Dense>(),
                                 Span<int32_t> info = Span<int32_t>());

    /**
     * @brief Required workspace, in bytes, for steqr(); arguments as for the call.
     * @param[in] ctx         queue the call will run on
     * @param[in] d           diagonal the call will use (shape only)
     * @param[in] e           off-diagonal the call will use (shape only)
     * @param[in] eigenvalues eigenvalue output the call will use; not written
     * @param[in] jobz        whether the call will compute eigenvectors
     * @param[in] params      parameters of the call
     * @return the workspace size in bytes
     * @ingroup api_tridiag_lowlevel
     */
    template <typename T>
    BATCHLAS_API size_t steqr_buffer_size(Queue& ctx, const VectorView<T>& d, const VectorView<T>& e,
                                         const VectorView<T>& eigenvalues, JobType jobz = JobType::NoEigenVectors, SteqrParams<T> params = SteqrParams<T>());

    /**
     * @brief Required workspace, in bytes, for steqr_cta(); arguments as for the call.
     * @param[in] ctx         queue the call will run on
     * @param[in] d           diagonal the call will use (shape only)
     * @param[in] e           off-diagonal the call will use (shape only)
     * @param[in] eigenvalues eigenvalue output the call will use; not written
     * @param[in] jobz        whether the call will compute eigenvectors
     * @param[in] params      parameters of the call
     * @return the workspace size in bytes
     * @ingroup api_tridiag_lowlevel
     */
    template <typename T>
    BATCHLAS_API size_t steqr_cta_buffer_size(Queue& ctx, const VectorView<T>& d, const VectorView<T>& e,
                                              const VectorView<T>& eigenvalues, JobType jobz = JobType::NoEigenVectors, SteqrParams<T> params = SteqrParams<T>());


    /// @}

    // === CTA small-matrix extensions (n <= 32) ===

    /// @addtogroup api_tridiag
    /// @{

    /**
     * @brief CTA-optimized symmetric tridiagonal reduction (SYTD2-style), \f$n \le 32\f$.
     *        Overwrites A with the tridiagonal and reflector storage and returns (d,e)
     *        plus reflector scalars in tau. `ws` is unused, kept for API compatibility.
     * @param[in]     ctx                    queue the work is enqueued on
     * @param[in,out] a_in                   batch of \f$n \times n\f$ symmetric/Hermitian matrices;
     *                                       overwritten with the tridiagonal and reflector storage
     * @param[out]    d_out                  diagonal of the tridiagonal, \f$n\f$ per item
     * @param[out]    e_out                  off-diagonal of the tridiagonal, \f$n-1\f$ per item
     * @param[out]    tau_out                Householder scalars, \f$n-1\f$ per item
     * @param[in]     uplo                   triangle of `a_in` that is read
     * @param[in]     ws                     unused; kept for API compatibility
     * @param[in]     cta_wg_size_multiplier multiplier on the baseline work-group size
     * @return event of the last enqueued kernel
     * @ingroup api_tridiag_lowlevel
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
     *        only the first `ib` columns of W are written (W is treated as \f$n \times \mathit{nb}\f$).
     *
     * @param[in]     ctx                  queue the work is enqueued on
     * @param[in,out] a_in                 batch of \f$n \times n\f$ matrices being reduced; the
     *                                     panel's reflectors are written into it
     * @param[out]    e_out                off-diagonal entries, \f$n-1\f$ per item; the panel's
     *                                     `ib` entries are written
     * @param[out]    tau_out              Householder scalars, \f$n-1\f$ per item; the panel's
     *                                     `ib` entries are written
     * @param[out]    w_in                 \f$n \times \mathit{nb}\f$ panel matrix \f$W\f$; its first
     *                                     `ib` columns are written
     * @param[in]     j0                   first column of the panel
     * @param[in]     ib                   panel width
     * @param[in]     wg_hint              work-group size hint; 0 lets the implementation choose
     * @param[in]     fuse_trailing_update fold the panel's share of the trailing update into the
     *                                     kernel
     * @return event of the last enqueued kernel
     * @ingroup api_tridiag_lowlevel
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
     *
     * @param[in]     ctx                  queue the work is enqueued on
     * @param[in,out] a_panel_in           square trailing block `A({j0, SliceEnd()}, {j0, SliceEnd()})`;
     *                                     the panel's reflectors are written into it
     * @param[out]    e_panel_out          the panel's off-diagonal entries, `w_panel_in.cols()` per item
     * @param[out]    tau_panel_out        the panel's Householder scalars, `w_panel_in.cols()` per item
     * @param[out]    w_panel_in           panel matrix \f$W\f$; its column count is the panel width; written
     * @param[in]     wg_hint              work-group size hint; 0 lets the implementation choose
     * @param[in]     fuse_trailing_update fold the panel's share of the trailing update into the
     *                                     kernel
     * @return event of the last enqueued kernel
     * @ingroup api_tridiag_lowlevel
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
     * @brief Blocked symmetric/Hermitian tridiagonal reduction, \f$n > 32\f$: LATRD-style panel
     *        plus a BLAS-3 trailing update. Same SYTD2-style output as `sytrd_cta`.
     *        In-order queue only.
     *
     * A is overwritten with the reflector storage (LAPACK `sytd2` layout, with the tridiagonal
     * entries restored on the first off-diagonal), the diagonal goes to `d_out`, the
     * off-diagonal to `e_out` and the reflector scalars to `tau_out`.
     *
     * @param[in]     ctx        in-order queue the work is enqueued on
     * @param[in,out] a_in       batch of \f$n \times n\f$ symmetric/Hermitian matrices (lower
     *                           triangle read); overwritten with the reflector storage
     * @param[out]    d_out      diagonal of the tridiagonal, \f$n\f$ per item
     * @param[out]    e_out      off-diagonal of the tridiagonal, \f$n-1\f$ per item
     * @param[out]    tau_out    Householder scalars, \f$n-1\f$ per item
     * @param[in]     uplo       must be `Uplo::Lower`
     * @param[out]    ws         scratch of at least sytrd_blocked_buffer_size() bytes;
     *                           contents undefined on return
     * @param[in]     block_size panel width; 0 (the default) selects `tuning::SYTRD_BLOCK_SIZE_MEDIUM`
     *                           (`batchlas/tuning_params.hh`), and a negative value is treated as 1.
     *                           Pass the same value to sytrd_blocked_buffer_size().
     * @return event of the last enqueued kernel
     * @pre `ctx` is in-order (SYCL kernels and backend BLAS calls are not separately ordered)
     * @pre `ws` holds at least sytrd_blocked_buffer_size() bytes (the W panel)
     * @throws batchlas::invalid_argument on a non-square A, short d/e/tau, a batch mismatch or
     *         an out-of-order `ctx`
     * @throws batchlas::unsupported for `Uplo::Upper` (only Lower is implemented)
     */
    template <Backend B, typename T>
    BATCHLAS_API Event sytrd_blocked(Queue& ctx,
                                     const MatrixView<T, MatrixFormat::Dense>& a_in,
                                     const VectorView<T>& d_out,
                                     const VectorView<T>& e_out,
                                     const VectorView<T>& tau_out,
                                     Uplo uplo,
                                     const Span<std::byte>& ws,
                                     int32_t block_size = 0);

    /**
     * @brief Required workspace, in bytes, for sytrd_blocked(); arguments as for the call.
     * @param[in] ctx        queue the call will run on
     * @param[in] a          matrices the call will reduce (shape only)
     * @param[in] d          diagonal output the call will use; not written
     * @param[in] e          off-diagonal output the call will use; not written
     * @param[in] tau        Householder-scalar output the call will use; not written
     * @param[in] uplo       triangle the call will read
     * @param[in] block_size panel width the call will use (0 selects the default)
     * @return the workspace size in bytes
     * @ingroup api_tridiag_lowlevel
     */
    template <Backend B, typename T>
    BATCHLAS_API size_t sytrd_blocked_buffer_size(Queue& ctx,
                                                  const MatrixView<T, MatrixFormat::Dense>& a,
                                                  const VectorView<T>& d,
                                                  const VectorView<T>& e,
                                                  const VectorView<T>& tau,
                                                  Uplo uplo,
                                                  int32_t block_size = 0);

    /**
     * @brief First stage of two-stage reduction: dense \f$\to\f$ band (LAPACK xSYTRD_SY2SB).
     *        Overwrites `A` with reflector storage and writes the band into `AB`;
     *        `tau_out` has size \f$n - \mathit{kd}\f$. In-order queue, `Uplo::Lower` only.
     *
     * Band storage (AB), shape \f$(\mathit{kd}+1) \times n\f$, 1-based as in LAPACK:
     *  - `Uplo::Lower`: \f$AB(1+i-j,\, j) = A(i, j)\f$ for \f$j \le i \le \min(n, j+\mathit{kd})\f$.
     *  - `Uplo::Upper`: \f$AB(\mathit{kd}+1+i-j,\, j) = A(i, j)\f$ for \f$\max(1, j-\mathit{kd}) \le i \le j\f$.
     *
     * @param[in]     ctx     in-order queue the work is enqueued on
     * @param[in,out] a_in    batch of \f$n \times n\f$ symmetric/Hermitian matrices; overwritten
     *                        with the reflector storage
     * @param[out]    ab_out  band output, \f$(\mathit{kd}+1) \times n\f$ per item
     * @param[out]    tau_out Householder scalars, \f$n - \mathit{kd}\f$ per item
     * @param[in]     uplo    must be `Uplo::Lower`
     * @param[in]     kd      semibandwidth of the band
     * @param[out]    ws      scratch of at least sytrd_sy2sb_buffer_size() bytes; contents undefined on return
     * @return event of the last enqueued kernel
     * @ingroup api_tridiag_lowlevel
     */
    template <Backend B, typename T>
    BATCHLAS_API Event sytrd_sy2sb(Queue& ctx,
                                   const MatrixView<T, MatrixFormat::Dense>& a_in,
                                   const MatrixView<T, MatrixFormat::Dense>& ab_out,
                                   const VectorView<T>& tau_out,
                                   Uplo uplo,
                                   int32_t kd,
                                   const Span<std::byte>& ws);

    /**
     * @brief Required workspace, in bytes, for sytrd_sy2sb(); arguments as for the call.
     * @param[in] ctx     queue the call will run on
     * @param[in] a_in    matrices the call will reduce (shape only)
     * @param[in] ab_out  band output the call will use; not written
     * @param[in] tau_out Householder-scalar output the call will use; not written
     * @param[in] uplo    triangle the call will read
     * @param[in] kd      semibandwidth of the band
     * @return the workspace size in bytes
     * @ingroup api_tridiag_lowlevel
     */
    template <Backend B, typename T>
    BATCHLAS_API size_t sytrd_sy2sb_buffer_size(Queue& ctx,
                                                const MatrixView<T, MatrixFormat::Dense>& a_in,
                                                const MatrixView<T, MatrixFormat::Dense>& ab_out,
                                                const VectorView<T>& tau_out,
                                                Uplo uplo,
                                                int32_t kd);

    /**
     * @brief Second stage of two-stage reduction: band \f$\to\f$ tridiagonal (LAPACK
     *        xSB2ST/xHB2ST bulge chasing, VECT='N'). In-order queue, `Uplo::Lower` only.
     *
     * Band storage (AB) is as for `sytrd_sy2sb`. `d_out` (\f$n\f$) and `e_out` (\f$n-1\f$) are always
     * real-valued; `tau_out` (\f$n-1\f$) is unused downstream and is set to 0.
     *
     * @param[in]  ctx        in-order queue the work is enqueued on
     * @param[in]  ab_in      band input, \f$(\mathit{kd}+1) \times n\f$ per item; copied, not modified
     * @param[out] d_out      diagonal of the tridiagonal (real), \f$n\f$ per item
     * @param[out] e_out      off-diagonal of the tridiagonal (real), \f$n-1\f$ per item
     * @param[out] tau_out    \f$n-1\f$ per item; set to 0
     * @param[in]  uplo       must be `Uplo::Lower`
     * @param[in]  kd         semibandwidth of the band
     * @param[out] ws         scratch of at least sytrd_sb2st_buffer_size() bytes; contents undefined on return
     * @param[in]  block_size accepted for API symmetry; currently unused
     * @return event of the last enqueued kernel
     * @ingroup api_tridiag_lowlevel
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

    /**
     * @brief Required workspace, in bytes, for sytrd_sb2st(); arguments as for the call.
     * @param[in] ctx        queue the call will run on
     * @param[in] ab_in      band input the call will use (shape only)
     * @param[in] d_out      diagonal output the call will use; not written
     * @param[in] e_out      off-diagonal output the call will use; not written
     * @param[in] tau_out    `tau_out` the call will use; not written
     * @param[in] uplo       triangle of the band
     * @param[in] kd         semibandwidth of the band
     * @param[in] block_size accepted for API symmetry; currently unused
     * @return the workspace size in bytes
     * @ingroup api_tridiag_lowlevel
     */
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
        /// Diagonals eliminated per sweep (Algorithm 2: \f$d^{(i)}\f$); 0 means the default
        /// schedule. If sweeps exceed the sequence length, the last value is reused.
        std::vector<int32_t> d_seq{0};

        /// Block size per sweep (Algorithm 2: \f$\mathit{nb}^{(i)}\f$).
        /// If sweeps exceed sequence length, the last value is reused.
        std::vector<int32_t> block_size_seq{32};

        int32_t max_sweeps = -1;  ///< Maximum number of sweeps; \f$< 0\f$ means the implementation default.

        /// Debug/testing: chase steps for sytrd_band_reduction_single_step();
        /// \f$\le 0\f$ means exactly one step.
        int32_t max_steps = 1;

        int32_t kd_work = 0;      ///< Working band semibandwidth; \f$\le 0\f$ means the implementation default.
    };

    /**
     * @brief Symmetric/Hermitian band \f$\to\f$ tridiagonal reduction (BANDR1-style). In-order
     *        queue, `Uplo::Lower` only. (d,e) are real; for complex input `e_out` is the
     *        magnitude of the (possibly phased) subdiagonal, and `tau_out` is set to 0.
     *
     * Band storage is as for sytrd_sy2sb(). This overload uses one block size for every
     * sweep.
     *
     * @param[in]  ctx        in-order queue the work is enqueued on
     * @param[in]  ab_in      band input, \f$(\mathit{kd}+1) \times n\f$ per item; copied, not modified
     * @param[out] d_out      diagonal of the tridiagonal (real), \f$n\f$ per item
     * @param[out] e_out      off-diagonal of the tridiagonal (real), \f$n-1\f$ per item
     * @param[out] tau_out    set to 0
     * @param[in]  uplo       must be `Uplo::Lower`
     * @param[in]  kd         semibandwidth of the band
     * @param[out] ws         scratch of at least sytrd_band_reduction_buffer_size() bytes;
     *                        contents undefined on return
     * @param[in]  block_size block size used for every sweep
     * @return event of the last enqueued kernel
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

    /**
     * @brief sytrd_band_reduction() with an explicit per-sweep schedule.
     * @param[in]  ctx     in-order queue the work is enqueued on
     * @param[in]  ab_in   band input, \f$(\mathit{kd}+1) \times n\f$ per item; copied, not modified
     * @param[out] d_out   diagonal of the tridiagonal (real), \f$n\f$ per item
     * @param[out] e_out   off-diagonal of the tridiagonal (real), \f$n-1\f$ per item
     * @param[out] tau_out set to 0
     * @param[in]  uplo    must be `Uplo::Lower`
     * @param[in]  kd      semibandwidth of the band
     * @param[out] ws      scratch of at least sytrd_band_reduction_buffer_size() bytes;
     *                     contents undefined on return
     * @param[in]  params  per-sweep schedule
     * @return event of the last enqueued kernel
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
                                            SytrdBandReductionParams params);

    template <Backend B, typename T>
    BATCHLAS_API Event sytrd_band_reduction_single_step(Queue& ctx,
                                                        const MatrixView<T, MatrixFormat::Dense>& ab_in,
                                                        const MatrixView<T, MatrixFormat::Dense>& abw_out,
                                                        Uplo uplo,
                                                        int32_t kd,
                                                        const Span<std::byte>& ws,
                                                        SytrdBandReductionParams params);

    /**
     * @brief Required workspace, in bytes, for sytrd_band_reduction() with one block size.
     * @param[in] ctx        queue the call will run on
     * @param[in] ab_in      band input the call will use (shape only)
     * @param[in] d_out      diagonal output the call will use; not written
     * @param[in] e_out      off-diagonal output the call will use; not written
     * @param[in] tau_out    `tau_out` the call will use; not written
     * @param[in] uplo       triangle of the band
     * @param[in] kd         semibandwidth of the band
     * @param[in] block_size block size of the call
     * @return the workspace size in bytes
     * @ingroup api_tridiag_lowlevel
     */
    template <Backend B, typename T>
    BATCHLAS_API size_t sytrd_band_reduction_buffer_size(Queue& ctx,
                                                         const MatrixView<T, MatrixFormat::Dense>& ab_in,
                                                         const VectorView<typename base_type<T>::type>& d_out,
                                                         const VectorView<typename base_type<T>::type>& e_out,
                                                         const VectorView<T>& tau_out,
                                                         Uplo uplo,
                                                         int32_t kd,
                                                         int32_t block_size);

    /**
     * @brief Required workspace, in bytes, for the schedule-parameter sytrd_band_reduction().
     * @param[in] ctx     queue the call will run on
     * @param[in] ab_in   band input the call will use (shape only)
     * @param[in] d_out   diagonal output the call will use; not written
     * @param[in] e_out   off-diagonal output the call will use; not written
     * @param[in] tau_out `tau_out` the call will use; not written
     * @param[in] uplo    triangle of the band
     * @param[in] kd      semibandwidth of the band
     * @param[in] params  per-sweep schedule of the call
     * @return the workspace size in bytes
     * @ingroup api_tridiag_lowlevel
     */
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
     *        public API. `ab_in` is lower-band with \f$\mathit{kd}+1\f$ rows, `abw_out` lower-band
     *        with \f$\mathit{kd\_work}+1\f$ rows (`kd_work` from params). In-order queue, Lower only.
     * @param[in]  ctx     in-order queue the work is enqueued on
     * @param[in]  ab_in   lower band input, \f$(\mathit{kd}+1) \times n\f$ per item; not modified
     * @param[out] abw_out working band, \f$(\mathit{kd\_work}+1) \times n\f$ per item; receives the
     *                     band after the chase step
     * @param[in]  uplo    must be `Uplo::Lower`
     * @param[in]  kd      semibandwidth of `ab_in`
     * @param[out] ws      scratch of at least sytrd_band_reduction_single_step_buffer_size() bytes;
     *                     contents undefined on return
     * @param[in]  params  schedule; `kd_work` and `max_steps` are read
     * @return event of the last enqueued kernel
     * @ingroup api_tridiag_lowlevel
     */
    template <Backend B, typename T>
    BATCHLAS_API Event sytrd_band_reduction_single_step(Queue& ctx,
                                                        const MatrixView<T, MatrixFormat::Dense>& ab_in,
                                                        const MatrixView<T, MatrixFormat::Dense>& abw_out,
                                                        Uplo uplo,
                                                        int32_t kd,
                                                        const Span<std::byte>& ws,
                                                        SytrdBandReductionParams params);

    /**
     * @brief Required workspace, in bytes, for sytrd_band_reduction_single_step().
     * @param[in] ctx     queue the call will run on
     * @param[in] ab_in   band input the call will use (shape only)
     * @param[in] abw_out working band the call will use; not written
     * @param[in] uplo    triangle of the band
     * @param[in] kd      semibandwidth of `ab_in`
     * @param[in] params  schedule of the call
     * @return the workspace size in bytes
     * @ingroup api_tridiag_lowlevel
     */
    template <Backend B, typename T>
    BATCHLAS_API size_t sytrd_band_reduction_single_step_buffer_size(Queue& ctx,
                                                                     const MatrixView<T, MatrixFormat::Dense>& ab_in,
                                                                     const MatrixView<T, MatrixFormat::Dense>& abw_out,
                                                                     Uplo uplo,
                                                                     int32_t kd,
                                                                     SytrdBandReductionParams params);

    /// @}

    /// @addtogroup api_eigen
    /// @{

    /**
     * @brief CTA-optimized symmetric eigen-solver (SYEV-like), \f$n \le 32\f$; real symmetric and
     *        complex Hermitian. Overwrites A with eigenvectors when jobz == EigenVectors.
     *        Eigenvalues ascend when SteqrParams::sort is set (default).
     *        cta_wg_size_multiplier == 0 lets the tridiagonal solve pick its tuned value
     *        and runs the reduction and back-transform at 1.
     *
     * This and the other `syev_*` tiers are what syev() routes between; call one directly
     * only to pin a tier. As in syev(), `a_in`'s `uplo` triangle is read, `eigenvalues`
     * gets \f$n\f$ real values per item, and `info` is the per-item convergence status or empty.
     *
     * @param[in]     ctx                    queue the work is enqueued on
     * @param[in,out] a_in                   batch of \f$n \times n\f$ symmetric/Hermitian matrices
     *                                       (`uplo` triangle read); overwritten, with the
     *                                       eigenvectors when `jobz == EigenVectors`
     * @param[out]    eigenvalues            eigenvalue output, \f$n\f$ real values per item
     * @param[in]     jobz                   whether eigenvectors are written to `a_in`
     * @param[in]     uplo                   triangle of `a_in` that is read
     * @param[out]    ws                     scratch of at least syev_cta_buffer_size() bytes;
     *                                       contents undefined on return
     * @param[in]     steqr_params           parameters of the tridiagonal solve
     * @param[in]     cta_wg_size_multiplier work-group size multiplier; 0 picks the tuned value
     * @param[out]    info                   per-item convergence status, or empty; see @ref md_docs_2cpp-api
     * @return event of the last enqueued kernel
     * @see @ref perf_syev (small-n kernel choice)
     * @ingroup api_eigen_lowlevel
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

    /**
     * @brief Required workspace, in bytes, for syev_cta().
     * @param[in] ctx          queue the call will run on
     * @param[in] a            matrices the call will solve (shape only)
     * @param[in] jobz         whether the call will compute eigenvectors
     * @param[in] steqr_params parameters of the call's tridiagonal solve
     * @return the workspace size in bytes
     * @ingroup api_eigen_lowlevel
     */
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
     *
     * @param[in]     ctx                    queue the work is enqueued on
     * @param[in,out] a_in                   batch of \f$n \times n\f$ symmetric/Hermitian matrices
     *                                       (`uplo` triangle read); overwritten with the eigenvectors
     *                                       when `jobz == EigenVectors`, untouched otherwise
     * @param[out]    eigenvalues           eigenvalue output, \f$n\f$ real values per item
     * @param[in]     jobz                   whether eigenvectors are written to `a_in`
     * @param[in]     uplo                   triangle of `a_in` that is read
     * @param[in]     ws                     ignored; accepted for API symmetry
     * @param[in]     steqr_params           sorting and solver parameters
     * @param[in]     cta_wg_size_multiplier work-group size multiplier; 0 picks the tuned value
     * @param[out]    info                   per-item convergence status, or empty; see @ref md_docs_2cpp-api
     * @return event of the kernel
     * @ingroup api_eigen_lowlevel
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

    /**
     * @brief Required workspace, in bytes, for syev_cta_fused().
     * @param[in] ctx          queue the call will run on
     * @param[in] a            matrices the call will solve (shape only)
     * @param[in] jobz         whether the call will compute eigenvectors
     * @param[in] steqr_params parameters of the call
     * @return the workspace size in bytes
     * @ingroup api_eigen_lowlevel
     */
    template <Backend B, typename T>
    BATCHLAS_API size_t syev_cta_fused_buffer_size(Queue& ctx,
                                                   const MatrixView<T, MatrixFormat::Dense>& a,
                                                   JobType jobz,
                                                   SteqrParams<T> steqr_params = SteqrParams<T>());

    /** @brief Parameters for syev_jacobi_cta() (cyclic two-sided Jacobi). */
    template <typename T>
    struct JacobiParams {
        using Real = typename base_type<T>::type;

        /// A rotation is applied to pivot pair \f$(p, q)\f$ only when
        /// \f$ \lvert a_{pq} \rvert > \mathrm{tol\_multiplier} \cdot n \varepsilon \sqrt{\lvert a_{pp} \rvert \, \lvert a_{qq} \rvert} \f$.
        /// This relative test, not the classical absolute one, is what yields the high
        /// relative accuracy; raising it trades accuracy for sweeps.
        Real tol_multiplier = Real(1);

        size_t max_sweeps = 30;  ///< Cap on cyclic sweeps; a safety net for pathological inputs.

        bool sort = true;                             ///< Sort eigenpairs on output.
        SortOrder sort_order = SortOrder::Ascending;  ///< Order used when #sort is set.

        /// Multiplies the baseline work-group size, \f$\operatorname{lcm}(P, s)\f$ for sub-group
        /// size \f$s\f$ and the compile-time partition width \f$P\f$ chosen from \f$n\f$. Clamped by
        /// device limits.
        size_t cta_wg_size_multiplier = 1;
    };

    /**
     * @brief CTA-optimized Jacobi symmetric/Hermitian eigen-solver, \f$n \le 32\f$; real
     *        symmetric and complex Hermitian.
     *
     * Trades throughput for high *relative* accuracy on graded or badly scaled input. The
     * underlying theorem (Demmel & Veselic, SIMAX 13(4), 1992) is proved for symmetric
     * positive definite input; indefinite matrices are solved correctly but do not inherit
     * the bound. Overwrites A with eigenvectors when jobz == EigenVectors and leaves it
     * untouched otherwise; `ws` is accepted for API symmetry and ignored.
     *
     * @param[in]     ctx         queue the work is enqueued on
     * @param[in,out] a_in        batch of \f$n \times n\f$ symmetric/Hermitian matrices (`uplo`
     *                            triangle read); overwritten with the eigenvectors when
     *                            `jobz == EigenVectors`, untouched otherwise
     * @param[out]    eigenvalues eigenvalue output, \f$n\f$ real values per item
     * @param[in]     jobz        whether eigenvectors are written to `a_in`
     * @param[in]     uplo        triangle of `a_in` that is read
     * @param[in]     ws          ignored; accepted for API symmetry
     * @param[in]     params      rotation threshold, sweep cap, sorting and work-group size
     * @param[out]    info        per-item convergence status, or empty; see @ref md_docs_2cpp-api
     * @return event of the kernel
     * @ingroup api_eigen_lowlevel
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

    /**
     * @brief Required workspace, in bytes, for syev_jacobi_cta().
     * @param[in] ctx    queue the call will run on
     * @param[in] a      matrices the call will solve (shape only)
     * @param[in] jobz   whether the call will compute eigenvectors
     * @param[in] params parameters of the call
     * @return the workspace size in bytes
     * @ingroup api_eigen_lowlevel
     */
    template <Backend B, typename T>
    BATCHLAS_API size_t syev_jacobi_cta_buffer_size(Queue& ctx,
                                                    const MatrixView<T, MatrixFormat::Dense>& a,
                                                    JobType jobz,
                                                    JacobiParams<T> params = JacobiParams<T>());

    /**
     * @brief Blocked symmetric/Hermitian eigen-solver (SYEV-like) for medium/large \f$n\f$:
     *        sytrd_blocked \f$\to\f$ stedc \f$\to\f$ ormqr_blocked. Overwrites A with eigenvectors when
     *        jobz == EigenVectors. `Uplo::Upper` input is first mirrored into the lower
     *        triangle, in place, and then takes the Lower path.
     *
     * @param[in]     ctx          in-order queue the work is enqueued on
     * @param[in,out] a_in         batch of \f$n \times n\f$ symmetric/Hermitian matrices (`uplo`
     *                             triangle read); overwritten, with the eigenvectors when
     *                             `jobz == EigenVectors`
     * @param[out]    eigenvalues  eigenvalue output, \f$n\f$ real values per item, ascending
     * @param[in]     jobz         whether eigenvectors are written to `a_in`
     * @param[in]     uplo         triangle of `a_in` that is read
     * @param[out]    ws           scratch of at least syev_blocked_buffer_size() bytes;
     *                             contents undefined on return
     * @param[in]     stedc_params parameters of the tridiagonal divide-and-conquer solve
     * @param[out]    info         per-item convergence status, or empty; see @ref md_docs_2cpp-api
     * @return event of the last enqueued kernel
     * @ingroup api_eigen_lowlevel
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

    /**
     * @brief Required workspace, in bytes, for syev_blocked(); the same for either `uplo`.
     * @param[in] ctx          queue the call will run on
     * @param[in] a            matrices the call will solve (shape only)
     * @param[in] jobz         whether the call will compute eigenvectors
     * @param[in] uplo         triangle the call will read
     * @param[in] stedc_params parameters of the call's tridiagonal solve
     * @return the workspace size in bytes
     * @ingroup api_eigen_lowlevel
     */
    template <Backend B, typename T>
    BATCHLAS_API size_t syev_blocked_buffer_size(Queue& ctx,
                                                 const MatrixView<T, MatrixFormat::Dense>& a,
                                                 JobType jobz,
                                                 Uplo uplo,
                                                 StedcParams<typename base_type<T>::type> stedc_params = StedcParams<typename base_type<T>::type>());

    /**
     * @brief Two-stage symmetric/Hermitian eigen-solver (SYEV-like) for large \f$n\f$:
     *        sytrd_sy2sb \f$\to\f$ sytrd_sb2st \f$\to\f$ stedc, then phase/sign recovery and a reflector
     *        back-transform. Band width from choose_two_stage_kd (override with
     *        BATCHLAS_SYEV_TWO_STAGE_KD). `Uplo::Upper` input is mirrored into the lower
     *        triangle first, as in syev_blocked().
     *
     * @param[in]     ctx          in-order queue the work is enqueued on
     * @param[in,out] a_in         batch of \f$n \times n\f$ symmetric/Hermitian matrices (`uplo`
     *                             triangle read); overwritten, with the eigenvectors when
     *                             `jobz == EigenVectors`
     * @param[out]    eigenvalues  eigenvalue output, \f$n\f$ real values per item, ascending
     * @param[in]     jobz         whether eigenvectors are written to `a_in`
     * @param[in]     uplo         triangle of `a_in` that is read
     * @param[out]    ws           scratch of at least syev_two_stage_buffer_size() bytes;
     *                             contents undefined on return
     * @param[in]     stedc_params parameters of the tridiagonal divide-and-conquer solve
     * @param[out]    info         per-item convergence status, or empty; see @ref md_docs_2cpp-api
     * @return event of the last enqueued kernel
     * @see @ref perf_syev for where syev() routes to it
     * @ingroup api_eigen_lowlevel
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

    /**
     * @brief Required workspace, in bytes, for syev_two_stage().
     * @param[in] ctx          queue the call will run on
     * @param[in] a            matrices the call will solve (shape only)
     * @param[in] jobz         whether the call will compute eigenvectors
     * @param[in] uplo         triangle the call will read
     * @param[in] stedc_params parameters of the call's tridiagonal solve
     * @return the workspace size in bytes
     * @ingroup api_eigen_lowlevel
     */
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
     * Computes \f$ A = Q B P^H \f$ with `B` upper bidiagonal: `d` gets its \f$n\f$ diagonal and
     * `e` its \f$n-1\f$ superdiagonal entries (real), and the reflectors defining `Q` and `P`
     * are left in `A` with their scalars in `tauq` and `taup`, as in LAPACK `?gebrd`.
     *
     * @param[in]     ctx  queue the work is enqueued on
     * @param[in,out] a    batch of \f$n \times n\f$ matrices; overwritten with the reflectors
     * @param[out]    d    diagonal of \f$B\f$ (real), \f$n\f$ per item
     * @param[out]    e    superdiagonal of \f$B\f$ (real), \f$n-1\f$ per item
     * @param[out]    tauq scalars of the reflectors defining \f$Q\f$
     * @param[out]    taup scalars of the reflectors defining \f$P\f$
     * @return event of the last enqueued kernel
     * @see @ref perf_gesvd (where the unblocked form stops paying)
     * @ingroup api_svd_lowlevel
     */
    template <Backend B, typename T>
    BATCHLAS_API Event gebrd_unblocked(Queue& ctx,
                                       const MatrixView<T, MatrixFormat::Dense>& a,
                                       const VectorView<typename base_type<T>::type>& d,
                                       const VectorView<typename base_type<T>::type>& e,
                                       const VectorView<T>& tauq,
                                       const VectorView<T>& taup);

    /**
     * @brief CTA-parallel small-matrix GEBRD for real square matrices, \f$1 \le n \le 32\f$,
     *        where one CTA cooperatively reduces one matrix to upper bidiagonal form.
     *        Output layout as for gebrd_unblocked().
     * @param[in]     ctx                    queue the work is enqueued on
     * @param[in,out] a                      batch of \f$n \times n\f$ real matrices; overwritten with
     *                                       the reflectors
     * @param[out]    d                      diagonal of the bidiagonal, \f$n\f$ per item
     * @param[out]    e                      superdiagonal of the bidiagonal, \f$n-1\f$ per item
     * @param[out]    tauq                   scalars of the left (\f$Q\f$) reflectors
     * @param[out]    taup                   scalars of the right (\f$P\f$) reflectors
     * @param[in]     cta_wg_size_multiplier multiplier on the baseline work-group size
     * @return event of the kernel
     * @ingroup api_svd_lowlevel
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
     * @param[in]     ctx        queue the work is enqueued on
     * @param[in,out] a          batch of \f$n \times n\f$ real matrices; overwritten with the reflectors
     * @param[out]    d          diagonal of the bidiagonal, \f$n\f$ per item
     * @param[out]    e          superdiagonal of the bidiagonal, \f$n-1\f$ per item
     * @param[out]    tauq       scalars of the left (\f$Q\f$) reflectors
     * @param[out]    taup       scalars of the right (\f$P\f$) reflectors
     * @param[out]    ws         scratch of at least gebrd_blocked_buffer_size() bytes;
     *                           contents undefined on return
     * @param[in]     block_size panel width
     * @return event of the last enqueued kernel
     * @ingroup api_svd_lowlevel
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

    /**
     * @brief Required workspace, in bytes, for gebrd_blocked(); arguments as for the call.
     * @param[in] ctx        queue the call will run on
     * @param[in] a          matrices the call will reduce (shape only)
     * @param[in] d          diagonal output the call will use; not written
     * @param[in] e          superdiagonal output the call will use; not written
     * @param[in] tauq       left-reflector scalars the call will use; not written
     * @param[in] taup       right-reflector scalars the call will use; not written
     * @param[in] block_size panel width of the call
     * @return the workspace size in bytes
     * @ingroup api_svd_lowlevel
     */
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
     *        matching LAPACK `BDSQR`: it returns \f$u\,Q\f$ and \f$P^T \mathit{vh}\f$.
     *
     * @param[in]     ctx                  queue the work is enqueued on
     * @param[in,out] d,e                  diagonal (\f$n\f$) and superdiagonal (\f$n-1\f$) per batch
     *                                     item; used as working storage, contents undefined on return
     * @param[out]    singular_values_out  \f$n\f$ singular values per item, packed
     * @param[out]    ws                   scratch of at least bdsqr_buffer_size() bytes;
     *                                     contents undefined on return
     * @param[in]     sort_desc            sort the singular values (and vectors) in descending order
     * @param[out]    info                 per-item convergence status, or empty; see @ref md_docs_2cpp-api
     * @return event of the last enqueued kernel
     * @throws batchlas::convergence_error if an item does not converge and `info` is empty or
     *         shorter than the batch (with `info`, the status is reported per item instead)
     * @ingroup api_svd_lowlevel
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
     * @brief bdsqr() that also accumulates the rotations: \f$u \leftarrow u\,Q\f$, \f$\mathit{vh} \leftarrow P^T \mathit{vh}\f$.
     *        Other parameters as for the values-only form.
     * @param[in]     ctx                  queue the work is enqueued on
     * @param[in,out] d,e                  diagonal (\f$n\f$) and superdiagonal (\f$n-1\f$) per batch
     *                                     item; used as working storage, contents undefined on return
     * @param[out]    singular_values_out  \f$n\f$ singular values per item, packed
     * @param[out]    ws                   scratch of at least bdsqr_buffer_size() bytes;
     *                                     contents undefined on return
     * @param[in,out] u                    matrix with \f$n\f$ columns per item; overwritten with
     *                                     \f$u \, Q\f$
     * @param[in,out] vh                   matrix with \f$n\f$ rows per item; overwritten with
     *                                     \f$P^T \, \mathit{vh}\f$
     * @param[in]     sort_desc            sort the singular values (and vectors) in descending order
     * @param[out]    info                 per-item convergence status, or empty; see @ref md_docs_2cpp-api
     * @return event of the last enqueued kernel
     * @throws batchlas::convergence_error if an item does not converge and `info` is empty or
     *         shorter than the batch (with `info`, the status is reported per item instead)
     * @ingroup api_svd_lowlevel
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

    /**
     * @brief Required workspace, in bytes, for bdsqr().
     * @param[in] ctx                 queue the call will run on
     * @param[in] d                   diagonal the call will use (shape only)
     * @param[in] e                   superdiagonal the call will use (shape only)
     * @param[in] singular_values_out singular-value output the call will use; not written
     * @return the workspace size in bytes
     * @ingroup api_svd_lowlevel
     */
    template <typename T>
    BATCHLAS_API size_t bdsqr_buffer_size(Queue& ctx,
                                          const VectorView<T>& d,
                                          const VectorView<T>& e,
                                          Span<T> singular_values_out);

    /**
     * @brief bdsqr_buffer_size() for the vector-accumulating form; `u` and `vh` do not change it.
     * @param[in] ctx                 queue the call will run on
     * @param[in] d                   diagonal the call will use (shape only)
     * @param[in] e                   superdiagonal the call will use (shape only)
     * @param[in] singular_values_out singular-value output the call will use; not written
     * @param[in] u                   ignored
     * @param[in] vh                  ignored
     * @return the workspace size in bytes
     * @ingroup api_svd_lowlevel
     */
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
     *        interleaved Golub-Kahan form (order \f$2n\f$) handed to `stedc`.
     *
     * Unlike `bdsqr`, which accumulates into whatever it is handed (\f$u \leftarrow u\,Q\f$), `bdsdc`
     * WRITES the leading \f$n \times n\f$ block of `u` and of `vh` and leaves the rest untouched --
     * seed them with the identity if the trailing columns matter. Workspace is dominated
     * by a \f$2n \times 2n\f$ eigenvector matrix per batch item. Parameters as for bdsqr().
     *
     * @param[in]  ctx                 in-order queue the work is enqueued on
     * @param[in]  d                   diagonal, \f$n\f$ entries per batch item; not modified
     * @param[in]  e                   superdiagonal, \f$n-1\f$ entries per batch item; not modified
     * @param[out] singular_values_out \f$n\f$ singular values per item, packed
     * @param[out] ws                  scratch of at least bdsdc_buffer_size() bytes;
     *                                 contents undefined on return
     * @param[in]  sort_desc           sort the singular values in descending order
     * @param[out] info                per-item convergence status, or empty; see @ref md_docs_2cpp-api
     * @return event of the last enqueued kernel
     * @see @ref design_gesvd, section "gesvd design: why bdsdc goes through Golub-Kahan"
     * @ingroup api_svd_lowlevel
     */
    template <Backend B, typename T>
    BATCHLAS_API Event bdsdc(Queue& ctx,
                             const VectorView<T>& d,
                             const VectorView<T>& e,
                             Span<T> singular_values_out,
                             const Span<std::byte>& ws,
                             bool sort_desc = true,
                             Span<int32_t> info = Span<int32_t>());

    /**
     * @brief bdsdc() that also writes the leading \f$n \times n\f$ blocks of `u` and `vh`.
     * @param[in]  ctx                 in-order queue the work is enqueued on
     * @param[in]  d                   diagonal, \f$n\f$ entries per batch item; not modified
     * @param[in]  e                   superdiagonal, \f$n-1\f$ entries per batch item; not modified
     * @param[out] singular_values_out \f$n\f$ singular values per item, packed
     * @param[out] ws                  scratch of at least bdsdc_buffer_size() bytes with
     *                                 `want_vectors`; contents undefined on return
     * @param[out] u                   left singular vectors: the leading \f$n \times n\f$ block is
     *                                 written, the rest left untouched
     * @param[out] vh                  right singular vectors (as rows): the leading \f$n \times n\f$
     *                                 block is written, the rest left untouched
     * @param[in]  sort_desc           sort the singular values (and vectors) in descending order
     * @param[out] info                per-item convergence status, or empty; see @ref md_docs_2cpp-api
     * @return event of the last enqueued kernel
     * @ingroup api_svd_lowlevel
     */
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

    /**
     * @brief Required workspace, in bytes, for bdsdc(); `want_vectors` selects the overload.
     * @param[in] ctx                 queue the call will run on
     * @param[in] d                   diagonal the call will use (shape only)
     * @param[in] e                   superdiagonal the call will use (shape only)
     * @param[in] singular_values_out singular-value output the call will use; not written
     * @param[in] want_vectors        true to size the overload that writes `u` and `vh`
     * @return the workspace size in bytes
     * @ingroup api_svd_lowlevel
     */
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
     * Overwrites `c` (\f$C\f$) with \f$\operatorname{op}(Q)\,C\f$, \f$C\,\operatorname{op}(Q)\f$,
     * \f$\operatorname{op}(P)\,C\f$ or \f$C\,\operatorname{op}(P)\f$ according to
     * `vect`, `side` and `trans`, where `Q`/`P` are the reflectors that gebrd_blocked() (or
     * gebrd_unblocked()) left in `a` and `tau`, as LAPACK `?ormbr` / `?unmbr`: `'Q'` is of
     * order `a.rows()`, `'P'` of order `a.cols()`.
     *
     * @param[in]     ctx        queue the work is enqueued on
     * @param[in]     a          reflectors left by gebrd_blocked() / gebrd_unblocked(); read only
     * @param[in]     tau        the matching scalars: `tauq` for `'Q'`, `taup` for `'P'`
     * @param[in,out] c          matrices to transform; overwritten with the product
     * @param[in]     vect       `'Q'` or `'P'`: which reflector set to apply
     * @param[in]     side       `Side::Left` multiplies from the left, `Side::Right` from the right
     * @param[in]     trans      apply \f$Q\f$ / \f$P\f$ or its (conjugate) transpose
     * @param[out]    ws         scratch of at least ormbr_buffer_size() bytes; contents undefined on return
     * @param[in]     block_size block size of the blocked path
     * @return event of the last enqueued kernel
     * @pre `tau` is unit-stride and packed by batch
     * @throws batchlas::invalid_argument on mismatched batch sizes or orders, a `vect`
     *         other than `'Q'`/`'P'`, or a short or strided `tau`
     * @throws batchlas::unsupported for `Transpose::Trans` with complex `T` and `'P'`
     *         (use `ConjTrans`)
     * @ingroup api_svd_lowlevel
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

    /**
     * @brief Required workspace, in bytes, for ormbr(); arguments as for the call.
     * @param[in] ctx        queue the call will run on
     * @param[in] a          reflectors the call will apply (shape only)
     * @param[in] tau        reflector scalars the call will use (shape only)
     * @param[in] c          matrices the call will transform; not written
     * @param[in] vect       `'Q'` or `'P'`
     * @param[in] side       side of the multiplication
     * @param[in] trans      whether the transpose is applied
     * @param[in] block_size block size of the call
     * @return the workspace size in bytes
     * @ingroup api_svd_lowlevel
     */
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
     * @brief Blocked native SVD for real dense matrices: GEBRD-style dense \f$\to\f$ bidiagonal,
     *        then BDSQR (or BDSDC when explicitly selected), then ORMBR-style
     *        back-transforms for full \f$U\f$ and \f$V^H\f$.
     * @param[in]     ctx             in-order queue the work is enqueued on
     * @param[in,out] a_in            batch of \f$m \times n\f$ matrices; destroyed (used as
     *                                working storage by the bidiagonal reduction)
     * @param[out]    singular_values \f$\min(m, n)\f$ singular values per item, descending
     * @param[out]    u_out           left singular vectors, as selected by `jobu`
     * @param[out]    vh_out          right singular vectors \f$V^H\f$, as selected by `jobvh`
     * @param[in]     jobu            which columns of \f$U\f$ to compute
     * @param[in]     jobvh           which rows of \f$V^H\f$ to compute
     * @param[out]    ws              scratch of at least gesvd_blocked_buffer_size() bytes;
     *                                contents undefined on return
     * @param[out]    info            per-item convergence status, or empty; see @ref md_docs_2cpp-api
     * @return event of the last enqueued kernel
     * @pre in-order Queue
     * @throws batchlas::invalid_argument for an out-of-order Queue or mis-shaped outputs
     * @throws batchlas::unsupported for complex input (use the Hermitian overload)
     * @ingroup api_svd_lowlevel
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
     * @param[in]     ctx             in-order queue the work is enqueued on
     * @param[in,out] a_in            batch of \f$n \times n\f$ Hermitian matrices; destroyed (used as
     *                                working storage)
     * @param[out]    singular_values \f$n\f$ singular values per item, descending
     * @param[out]    u_out           left singular vectors, as selected by `jobu`
     * @param[out]    vh_out          right singular vectors \f$V^H\f$, as selected by `jobvh`
     * @param[in]     jobu            which columns of \f$U\f$ to compute
     * @param[in]     jobvh           which rows of \f$V^H\f$ to compute
     * @param[in]     hermitian_uplo  triangle of `a_in` that is read
     * @param[out]    ws              scratch of at least gesvd_blocked_buffer_size() bytes;
     *                                contents undefined on return
     * @param[out]    info            per-item convergence status, or empty; see @ref md_docs_2cpp-api
     * @return event of the last enqueued kernel
     * @ingroup api_svd_lowlevel
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

    /**
     * @brief Required workspace, in bytes, for gesvd_blocked().
     * @param[in] ctx             queue the call will run on
     * @param[in] a               matrices the call will decompose (shape only)
     * @param[in] singular_values singular-value output the call will use; not written
     * @param[in] u_out           left-vector output the call will use; not written
     * @param[in] vh_out          right-vector output the call will use; not written
     * @param[in] jobu            which columns of \f$U\f$ the call computes
     * @param[in] jobvh           which rows of \f$V^H\f$ the call computes
     * @return the workspace size in bytes
     * @ingroup api_svd_lowlevel
     */
    template <Backend B, typename T>
    BATCHLAS_API size_t gesvd_blocked_buffer_size(Queue& ctx,
                                                  const MatrixView<T, MatrixFormat::Dense>& a,
                                                  Span<typename base_type<T>::type> singular_values,
                                                  const MatrixView<T, MatrixFormat::Dense>& u_out,
                                                  const MatrixView<T, MatrixFormat::Dense>& vh_out,
                                                  SvdVectors jobu,
                                                  SvdVectors jobvh);

    /**
     * @brief Required workspace, in bytes, for the Hermitian gesvd_blocked().
     * @param[in] ctx             queue the call will run on
     * @param[in] a               matrices the call will decompose (shape only)
     * @param[in] singular_values singular-value output the call will use; not written
     * @param[in] u_out           left-vector output the call will use; not written
     * @param[in] vh_out          right-vector output the call will use; not written
     * @param[in] jobu            which columns of \f$U\f$ the call computes
     * @param[in] jobvh           which rows of \f$V^H\f$ the call computes
     * @param[in] hermitian_uplo  triangle the call will read
     * @return the workspace size in bytes
     * @ingroup api_svd_lowlevel
     */
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
     * @brief CTA-oriented native SVD for small matrices, \f$\max(m, n) \le 32\f$: bidiagonal
     *        reduction, then the eigenproblem of \f$ B^T B \f$ (the normal equations).
     *        Real input, rectangular either way.
     *
     * The normal equations square the condition number; use gesvdj_cta() where small
     * singular values matter.
     * @param[in]     ctx             in-order queue the work is enqueued on
     * @param[in,out] a_in            batch of \f$m \times n\f$ real matrices; destroyed (used as
     *                                working storage)
     * @param[out]    singular_values \f$\min(m, n)\f$ singular values per item, descending
     * @param[out]    u_out           left singular vectors, as selected by `jobu`
     * @param[out]    vh_out          right singular vectors \f$V^H\f$, as selected by `jobvh`
     * @param[in]     jobu            which columns of \f$U\f$ to compute (not genuinely `Thin`)
     * @param[in]     jobvh           which rows of \f$V^H\f$ to compute (not genuinely `Thin`)
     * @param[out]    ws              scratch of at least gesvd_cta_buffer_size() bytes;
     *                                contents undefined on return
     * @param[out]    info            per-item convergence status, or empty; see @ref md_docs_2cpp-api
     * @return event of the last enqueued kernel
     * @pre in-order Queue
     * @throws batchlas::invalid_argument for \f$\max(m, n) > 32\f$, a genuinely `Thin` \f$U\f$ or
     *         \f$V^H\f$, or an out-of-order Queue
     * @throws batchlas::unsupported for complex input (use the Hermitian overload)
     * @see @ref perf_gesvd
     * @ingroup api_svd_lowlevel
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

    /**
     * @brief gesvd_cta() for Hermitian input: only the `hermitian_uplo` triangle is read.
     * @param[in]     ctx             in-order queue the work is enqueued on
     * @param[in,out] a_in            batch of \f$n \times n\f$ Hermitian matrices, \f$n \le 32\f$;
     *                                destroyed (used as working storage)
     * @param[out]    singular_values \f$n\f$ singular values per item, descending
     * @param[out]    u_out           left singular vectors, as selected by `jobu`
     * @param[out]    vh_out          right singular vectors \f$V^H\f$, as selected by `jobvh`
     * @param[in]     jobu            which columns of \f$U\f$ to compute
     * @param[in]     jobvh           which rows of \f$V^H\f$ to compute
     * @param[in]     hermitian_uplo  triangle of `a_in` that is read
     * @param[out]    ws              scratch of at least gesvd_cta_buffer_size() bytes;
     *                                contents undefined on return
     * @param[out]    info            per-item convergence status, or empty; see @ref md_docs_2cpp-api
     * @return event of the last enqueued kernel
     * @ingroup api_svd_lowlevel
     */
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

    /**
     * @brief Required workspace, in bytes, for gesvd_cta().
     * @param[in] ctx             queue the call will run on
     * @param[in] a               matrices the call will decompose (shape only)
     * @param[in] singular_values singular-value output the call will use; not written
     * @param[in] u_out           left-vector output the call will use; not written
     * @param[in] vh_out          right-vector output the call will use; not written
     * @param[in] jobu            which columns of \f$U\f$ the call computes
     * @param[in] jobvh           which rows of \f$V^H\f$ the call computes
     * @return the workspace size in bytes
     * @ingroup api_svd_lowlevel
     */
    template <Backend B, typename T>
    BATCHLAS_API size_t gesvd_cta_buffer_size(Queue& ctx,
                                              const MatrixView<T, MatrixFormat::Dense>& a,
                                              Span<typename base_type<T>::type> singular_values,
                                              const MatrixView<T, MatrixFormat::Dense>& u_out,
                                              const MatrixView<T, MatrixFormat::Dense>& vh_out,
                                              SvdVectors jobu,
                                              SvdVectors jobvh);

    /**
     * @brief Required workspace, in bytes, for the Hermitian gesvd_cta().
     * @param[in] ctx             queue the call will run on
     * @param[in] a               matrices the call will decompose (shape only)
     * @param[in] singular_values singular-value output the call will use; not written
     * @param[in] u_out           left-vector output the call will use; not written
     * @param[in] vh_out          right-vector output the call will use; not written
     * @param[in] jobu            which columns of \f$U\f$ the call computes
     * @param[in] jobvh           which rows of \f$V^H\f$ the call computes
     * @param[in] hermitian_uplo  triangle the call will read
     * @return the workspace size in bytes
     * @ingroup api_svd_lowlevel
     */
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

        /// A rotation is applied to pivot pair \f$(p, q)\f$ only when
        /// \f$ \lvert a_{pq} \rvert > \mathrm{tol\_multiplier} \cdot n \varepsilon \sqrt{\lvert a_{pp} \rvert \, \lvert a_{qq} \rvert} \f$
        /// over the \f$2 \times 2\f$ Gram entries of the current columns; again the relative test.
        Real tol_multiplier = Real(1);

        size_t max_sweeps = 30;  ///< Cap on cyclic sweeps; convergence normally takes well under 10.

        /// Multiplies the baseline problems-per-work-group. Baseline is
        /// \f$32 / P\f$ problems, clamped by local memory and max work-group size.
        size_t cta_wg_size_multiplier = 1;

        /// \f$ \sigma_j \le \mathrm{zero\_sigma\_multiplier} \cdot \varepsilon \sigma_{\max} \f$
        /// means \f$ U_j \f$ is not determined by A and is filled from the orthogonal complement.
        Real zero_sigma_multiplier = Real(1);

        /// Optional per-problem diagnostic, the analogue of cusolverDnXgesvdjGetSweeps:
        /// when non-empty (size \f$\ge\f$ `batch_size`) the kernel writes each problem's sweeps.
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
     * \f$\max(m, n) \le 64\f$ (32 for `complex<double>` with vectors), real and complex,
     * rectangular either way; needs sub-group size 32. A is DESTROYED, singular values are
     * returned descending, and no workspace is required.
     *
     * @param[in]     ctx             queue the work is enqueued on
     * @param[in,out] a_in            batch of \f$m \times n\f$ matrices; DESTROYED
     * @param[out]    singular_values \f$\min(m, n)\f$ singular values per item, descending
     * @param[out]    u_out           left singular vectors, as selected by `jobu`
     * @param[out]    vh_out          right singular vectors \f$V^H\f$, as selected by `jobvh`
     * @param[in]     jobu            which columns of \f$U\f$ to compute
     * @param[in]     jobvh           which rows of \f$V^H\f$ to compute
     * @param[in]     ws              ignored; no workspace is required
     * @param[in]     params          rotation threshold, sweep cap, work-group size and the
     *                                optional `sweep_counts` output span
     * @param[out]    info            per-item convergence status, or empty; see @ref md_docs_2cpp-api
     * @return event of the kernel
     *
     * @throws batchlas::invalid_argument for a size above the cap or undersized outputs
     * @throws batchlas::unsupported if the device has no sub-group size 32
     * @see @ref design_gesvd, @ref perf_gesvd
     * @ingroup api_svd_lowlevel
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

    /**
     * @brief Required workspace, in bytes, for gesvdj_cta() (currently 0: all state is in local memory).
     * @param[in] ctx             queue the call will run on
     * @param[in] a               matrices the call will decompose (shape only)
     * @param[in] singular_values singular-value output the call will use; not written
     * @param[in] u_out           left-vector output the call will use; not written
     * @param[in] vh_out          right-vector output the call will use; not written
     * @param[in] jobu            which columns of \f$U\f$ the call computes
     * @param[in] jobvh           which rows of \f$V^H\f$ the call computes
     * @param[in] params          parameters of the call
     * @return the workspace size in bytes
     * @ingroup api_svd_lowlevel
     */
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
     *        reflectors (`a_in`, `tau_in`) from GEQRF (Upper) or GEQLF (Lower) to `c_in` (\f$C\f$).
     *
     * Overwrites `c_in` with \f$\operatorname{op}(Q)\,C\f$ (`Side::Left`) or \f$C\,\operatorname{op}(Q)\f$ (`Side::Right`),
     * using the first `k` reflectors.
     *
     * @param[in]     ctx                    queue the work is enqueued on
     * @param[in]     a_in                   Householder reflectors from GEQRF / GEQLF; read only
     * @param[in]     tau_in                 the reflectors' scalars, at least `k` per item
     * @param[in,out] c_in                   matrices \f$C\f$; overwritten with the product
     * @param[in]     factorization          `Uplo::Upper` for a QR (GEQRF) factor, `Uplo::Lower`
     *                                       for a QL (GEQLF) factor
     * @param[in]     side                   `Side::Left` or `Side::Right`
     * @param[in]     trans                  apply \f$Q\f$ or its (conjugate) transpose
     * @param[in]     k                      number of reflectors applied
     * @param[in]     ws                     unused; accepted for API symmetry
     * @param[in]     cta_wg_size_multiplier multiplier on the baseline work-group size
     * @return event of the kernel
     * @ingroup api_qr_lowlevel
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
        int64_t recursion_threshold = 0; ///< Leaf size; \f$\le 0\f$ uses BatchLAS tuning, otherwise this exact threshold
        StedcAlgorithm algorithm = StedcAlgorithm::Auto;               ///< Merge-tree driver.
        SteqrParams<T> leaf_steqr_params = SteqrParams<T>();           ///< Parameters of the leaf steqr() solves.

        StedcMergeVariant merge_variant = StedcMergeVariant::Auto;     ///< Merge-step dispatch.
        int merge_threads = 128;       ///< work-group size for fused kernel
        int max_sec_iter = 50;         ///< iteration cap for secular root solver
        bool enable_rescale = true;    ///< keep ROCm-style v rescale (disable for perf experiments)
        int secular_threads_per_root = 0;       ///< \f$\le 0\f$ uses BatchLAS tuning; otherwise this exact partition width
        int secular_cta_wg_size_multiplier = 0;  ///< \f$\le 0\f$ uses BatchLAS tuning; otherwise this exact multiplier
    };

    /**
     * @brief Eigenvalues and eigenvectors of a batch of symmetric tridiagonal matrices by
     *        divide and conquer (LAPACK `?stedc`).
     *
     * Splits each matrix recursively down to leaves of `recursion_threshold`, solves the
     * leaves with steqr(), and merges pairs by solving the secular equation and forming
     * the merged eigenvectors. Eigenvalues are returned ascending.
     *
     * @param[in]     ctx         queue the work is enqueued on
     * @param[in,out] d           diagonal, \f$n\f$ entries per batch item; the recursive driver
     *                            edits it in place, so treat its contents as undefined on return
     * @param[in]     e           off-diagonal, \f$n-1\f$ entries per batch item
     * @param[out]    eigenvalues eigenvalue output, \f$n\f$ per batch item
     * @param[out]    ws          scratch of at least stedc_buffer_size() bytes; contents undefined on return
     * @param[in]     jobz        whether `eigvects` receives the eigenvectors
     * @param[in]     params      leaf size, merge driver and tuning overrides
     * @param[out]    eigvects    \f$n \times n\f$ eigenvector output per item
     * @param[out]    info        per-item convergence status, or empty; see @ref md_docs_2cpp-api.
     *                            Raised by a merge whose secular solve hits its iteration cap and
     *                            by a leaf steqr() that runs out of sweeps.
     * @return event of the last enqueued kernel
     * @see @ref perf_stedc
     */
    template <Backend B, typename T>
    BATCHLAS_API Event stedc(Queue& ctx, const VectorView<T>& d, const VectorView<T>& e, const VectorView<T>& eigenvalues, const Span<std::byte>& ws,
                         JobType jobz, StedcParams<T> params, const MatrixView<T, MatrixFormat::Dense>& eigvects,
                         Span<int32_t> info = Span<int32_t>());

    /**
     * @brief Required workspace, in bytes, for stedc().
     * @param[in] ctx        queue the call will run on
     * @param[in] n          order of each tridiagonal matrix
     * @param[in] batch_size number of batch items
     * @param[in] jobz       whether the call will compute eigenvectors
     * @param[in] params     leaf size, merge driver and tuning overrides of the call
     * @return the workspace size in bytes
     * @ingroup api_tridiag_lowlevel
     */
    template <Backend B, typename T>
    BATCHLAS_API size_t stedc_buffer_size(Queue& ctx, size_t n, size_t batch_size, JobType jobz, StedcParams<T> params);

    /// @}

    /// @addtogroup api_eigen
    /// @{

    /**
     * @brief Computes the Ritz values (Rayleigh quotients) of a set of trial vectors.
     *
     * \f$ \theta_j = \dfrac{v_j^H A v_j}{v_j^H v_j} \f$ for each column \f$ v_j \f$ of `V`,
     * per batch item.
     *
     * @param[in]  ctx       execution context/device queue
     * @param[in]  A         batch of \f$n \times n\f$ matrices, sparse or dense; read only
     * @param[in]  V         trial vectors, \f$n \times k\f$ dense per item (columns are trial eigenvectors)
     * @param[out] ritz_vals Ritz values, \f$k\f$ (= `V.cols()`) real entries per item
     * @param[out] workspace scratch of at least ritz_values_buffer_size() bytes; contents undefined on return
     * @return event to track operation completion
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
     * @param[in] ctx queue the work runs on; waited on before returning
     * @param[in] A   batch of \f$n \times n\f$ matrices, sparse or dense; read only
     * @param[in] V   trial vectors, \f$n \times k\f$ dense per item
     * @return a `Vector` of \f$k\f$ (= `V.cols()`) real Ritz values per item
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
     * @brief Required workspace size, in bytes, for ritz_values().
     * @param[in] ctx       queue the call will run on
     * @param[in] A         matrices the call will use (shape and format only)
     * @param[in] V         trial vectors the call will use (shape only)
     * @param[in] ritz_vals Ritz-value output the call will use; not written
     * @return the workspace size in bytes
     * @ingroup api_eigen_lowlevel
     */
    template <Backend B, typename T, MatrixFormat MFormat>
    BATCHLAS_API size_t ritz_values_buffer_size(Queue& ctx,
                                              const MatrixView<T, MFormat>& A,
                                              const MatrixView<T, MatrixFormat::Dense>& V,
                                              const VectorView<typename base_type<T>::type>& ritz_vals);

    /// @}

    /// @addtogroup api_extra
    /// @{

    /**
     * @brief Computes the explicit inverse of each dense square matrix in the batch,
     *        \f$ A_{\mathrm{inv}} := A^{-1} \f$, by LU factorisation (getrf()) and getri().
     *
     * `A` is copied into the workspace first and is not modified. Singular items are not
     * reported: check the result where the input may be ill-conditioned.
     *
     * @param[in]  ctx       execution context/device queue
     * @param[in]  A         matrices to invert, \f$n \times n\f$ per item; read only
     * @param[out] Ainv      the inverses \f$A^{-1}\f$, \f$n \times n\f$ per item
     * @param[out] workspace scratch of at least inv_buffer_size() bytes; contents undefined on return
     * @return event to track operation completion
     */
    template <Backend B, typename T>
    BATCHLAS_API Event inv(Queue& ctx,
                     const MatrixView<T, MatrixFormat::Dense>& A,
                     const MatrixView<T, MatrixFormat::Dense>& Ainv,
                     Span<std::byte> workspace);

    /**
     * @brief Required workspace size, in bytes, for inv().
     * @param[in] ctx queue the call will run on
     * @param[in] A   matrices the call will invert (shape only)
     * @return the workspace size in bytes
     * @ingroup api_extra_lowlevel
     */
    template <Backend B, typename T>
    BATCHLAS_API size_t inv_buffer_size(Queue& ctx,
                     const MatrixView<T, MatrixFormat::Dense>& A);

    /**
     * @brief inv() returning a newly allocated result; workspace is leased from the
     *        Queue's arena. Returns without waiting.
     * @param[in] ctx queue the work is enqueued on
     * @param[in] A   matrices to invert, \f$n \times n\f$ per item; read only
     * @return a new batch holding \f$A^{-1}\f$, \f$n \times n\f$ per item
     */
    template <Backend B, typename T>
    BATCHLAS_API Matrix<T, MatrixFormat::Dense> inv(Queue& ctx,
                     const MatrixView<T, MatrixFormat::Dense>& A);

    /**
     * @brief Scales `mat` by \f$\mathit{cto} / \mathit{cfrom}\f$ without overflow or underflow (LAPACK `?lascl`).
     *
     * The factor is applied in safe steps when forming \f$\mathit{cto} / \mathit{cfrom}\f$ directly would
     * over- or underflow. Instantiated for real `float` and `double`, dense format.
     * @param[in]     ctx   queue the work is enqueued on
     * @param[in,out] mat   batch of matrices; scaled in place
     * @param[in]     cfrom denominator of the factor; must be nonzero
     * @param[in]     cto   numerator of the factor
     * @return event of the last scaling step
     */
    template <MatrixFormat MType, typename T>
    BATCHLAS_API Event lascl(Queue& ctx, const MatrixView<T, MType>& mat, T cfrom, T cto);

    /// @}
}

namespace batchlas {

// Backend-deducing overloads (BATCHLAS_DISPATCH_ON_QUEUE); see blas/queue-dispatch.hh.
// evidence: docs/cpp-api.md#which-type-each-parameter-takes


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

BATCHLAS_DISPATCH_ON_QUEUE(inv)
BATCHLAS_DISPATCH_ON_QUEUE(inv_buffer_size)

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

/**
 * @brief ortho() with an options struct and a caller-provided workspace.
 * @param[in]     ctx       queue the work is enqueued on
 * @param[in,out] A         batch of \f$m \times n\f$ matrices; overwritten with the orthonormal vectors
 * @param[in]     opts      columns or rows, and the orthogonalisation algorithm
 * @param[out]    workspace scratch of at least ortho_buffer_size() bytes for `opts.algorithm`;
 *                          contents undefined on return
 * @return event of the last enqueued kernel
 */
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
 * @param[in]     ctx  queue the work is enqueued on; supplies the workspace arena
 * @param[in,out] A    batch of \f$m \times n\f$ matrices; overwritten with the orthonormal vectors
 * @param[in]     opts columns or rows, and the orthogonalisation algorithm
 * @return event of the last enqueued kernel
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

/**
 * @brief Backend-deducing ortho() with an options struct and a caller-provided workspace.
 * @param[in]     ctx       queue the work is enqueued on; its device picks the backend
 * @param[in,out] A         batch of \f$m \times n\f$ matrices; overwritten with the orthonormal vectors
 * @param[in]     opts      columns or rows, and the orthogonalisation algorithm
 * @param[out]    workspace scratch of at least ortho_buffer_size() bytes for `opts.algorithm`;
 *                          contents undefined on return
 * @return event of the last enqueued kernel
 */
template <typename T>
inline Event ortho(Queue& ctx,
        const MatrixView<T, MatrixFormat::Dense>& A,
        const OrthoOptions& opts,
        Span<std::byte> workspace) {
    if (detail::pointer_checks_enabled()) {
        detail::require_arg_accessible(ctx, A, "ortho: A");
        detail::require_arg_accessible(ctx, workspace, "ortho: workspace");
    }
    return with_backend_for<T>(ctx, "ortho",
                               [&](auto Back) { return ortho<Back.value, T>(ctx, A, opts, workspace); });
}

/**
 * @brief Backend-deducing ortho() with an options struct; workspace leased from the arena.
 * @param[in]     ctx  queue the work is enqueued on; its device picks the backend and it
 *                     supplies the workspace arena
 * @param[in,out] A    batch of \f$m \times n\f$ matrices; overwritten with the orthonormal vectors
 * @param[in]     opts columns or rows, and the orthogonalisation algorithm
 * @return event of the last enqueued kernel
 */
template <typename T>
inline Event ortho(Queue& ctx,
        const MatrixView<T, MatrixFormat::Dense>& A,
        const OrthoOptions& opts = {}) {
    if (detail::pointer_checks_enabled()) detail::require_arg_accessible(ctx, A, "ortho: A");
    return with_backend_for<T>(ctx, "ortho",
                               [&](auto Back) { return ortho<Back.value, T>(ctx, A, opts); });
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

template <typename T>
Event ortho(Queue&, const MatrixView<T, MatrixFormat::Dense>&,
            detail::OrthoEmptyBracesAreAmbiguous, Span<std::byte>) = delete;

// ---- orthogonalisation against an external metric ---------------------------
// No `{}` guard is needed here: the positional metric form needs at least six arguments
// while the option forms take four and five, so no argument list can reach both.

/**
 * @brief ortho() against `M` with an options struct and a caller-provided workspace.
 * @param[in]     ctx       queue the work is enqueued on
 * @param[in,out] A         batch of vectors to orthonormalise; overwritten with the result
 * @param[in]     M         external basis, read only
 * @param[in]     opts      vector orientation of `A` and `M`, algorithm and pass count
 * @param[out]    workspace scratch of at least ortho_buffer_size() bytes for the same options;
 *                          contents undefined on return
 * @return event of the last enqueued kernel
 */
template <Backend B, typename T>
inline Event ortho(Queue& ctx,
        const MatrixView<T, MatrixFormat::Dense>& A,
        const MatrixView<T, MatrixFormat::Dense>& M,
        const OrthoAgainstOptions& opts,
        Span<std::byte> workspace) {
    return ortho<B, T>(ctx, A, M, opts.transA, opts.transM, workspace, opts.algorithm,
                       opts.iterations);
}

/**
 * @brief ortho() against `M` with an options struct; workspace leased from the arena.
 * @param[in]     ctx  queue the work is enqueued on; supplies the workspace arena
 * @param[in,out] A    batch of vectors to orthonormalise; overwritten with the result
 * @param[in]     M    external basis, read only
 * @param[in]     opts vector orientation of `A` and `M`, algorithm and pass count
 * @return event of the last enqueued kernel
 */
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

/**
 * @brief Backend-deducing ortho() against `M` with an options struct and a caller-provided workspace.
 * @param[in]     ctx       queue the work is enqueued on; its device picks the backend
 * @param[in,out] A         batch of vectors to orthonormalise; overwritten with the result
 * @param[in]     M         external basis, read only
 * @param[in]     opts      vector orientation of `A` and `M`, algorithm and pass count
 * @param[out]    workspace scratch of at least ortho_buffer_size() bytes for the same options;
 *                          contents undefined on return
 * @return event of the last enqueued kernel
 */
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
    return with_backend_for<T>(ctx, "ortho",
                               [&](auto Back) { return ortho<Back.value, T>(ctx, A, M, opts, workspace); });
}

/**
 * @brief Backend-deducing ortho() against `M` with an options struct; workspace leased from the arena.
 * @param[in]     ctx  queue the work is enqueued on; its device picks the backend and it
 *                     supplies the workspace arena
 * @param[in,out] A    batch of vectors to orthonormalise; overwritten with the result
 * @param[in]     M    external basis, read only
 * @param[in]     opts vector orientation of `A` and `M`, algorithm and pass count
 * @return event of the last enqueued kernel
 */
template <typename T>
inline Event ortho(Queue& ctx,
        const MatrixView<T, MatrixFormat::Dense>& A,
        const MatrixView<T, MatrixFormat::Dense>& M,
        const OrthoAgainstOptions& opts = {}) {
    if (detail::pointer_checks_enabled()) {
        detail::require_arg_accessible(ctx, A, "ortho: A");
        detail::require_arg_accessible(ctx, M, "ortho: M");
    }
    return with_backend_for<T>(ctx, "ortho",
                               [&](auto Back) { return ortho<Back.value, T>(ctx, A, M, opts); });
}

/// @}

}  // namespace batchlas
