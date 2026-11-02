// SPDX-License-Identifier: MIT
#pragma once

#include <batchlas/export.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>

#include <cstddef>
#include <cstdint>
#include <batchlas/blas/queue-dispatch.hh>

namespace batchlas {

/// @brief Parameters of an incomplete LU factorization with level-of-fill k, ILU(k).
///
/// The symbolic phase keeps every entry whose level of fill is at most
/// `levels_of_fill`. The numeric phase then prunes each row: entries at or below
/// `drop_tolerance` times the row's scale (floored at 1) are dropped, and at most
/// \f$\lceil f \cdot r \rceil - 1\f$ off-diagonal entries survive, largest magnitude
/// first (ties by column), where \f$f\f$ is `fill_factor` and \f$r\f$ the row's original
/// non-zero count. A pivot smaller than
/// \f$\max(\lvert \delta \rvert, \tau \cdot \max(s, 1))\f$, with \f$\delta\f$ = `diagonal_shift`,
/// \f$\tau\f$ = `diag_pivot_threshold` and \f$s\f$ the row scale, is shifted by
/// `diagonal_shift`, replaced by it, or raised to that threshold.
/// @tparam T  scalar type; real-valued fields use its real counterpart
/// @ingroup api_sparse
template <typename T>
struct ILUKParams {
    int levels_of_fill = 0;  ///< \f$k\f$; 0 keeps exactly A's pattern. Must be \f$\ge 0\f$.
    T diagonal_shift = T(1e-8);  ///< added to (or substituted for) a pivot that is too small
    typename base_type<T>::type drop_tolerance = typename base_type<T>::type(1e-4);  ///< relative drop threshold; \f$\ge 0\f$
    typename base_type<T>::type fill_factor = typename base_type<T>::type(10);  ///< per-row cap on kept entries relative to A's row; \f$\ge 1\f$
    typename base_type<T>::type diag_pivot_threshold = typename base_type<T>::type(0.1);  ///< relative pivot threshold; \f$\ge 0\f$
    bool modified_ilu = true;  ///< add dropped entries to the diagonal (MILU) instead of discarding them
    /// Check that every batch item shares one CSR pattern. Must stay true for
    /// \f$\mathrm{batch\_size} > 1\f$: disabling the check is rejected.
    bool validate_batch_sparsity = true;
};

/// @brief Non-owning description of a factored ILU(k) preconditioner.
///
/// This is what iluk_apply consumes, so the factor can live in an
/// ILUKPreconditioner or in a caller-supplied workspace (the `Span` overload of
/// iluk_factorize). Valid only while that storage is.
/// @tparam T  scalar type
/// @ingroup api_sparse
template <typename T>
struct ILUKView {
    MatrixView<T, MatrixFormat::CSR> lu;  ///< L (strict lower, unit diagonal implied) and U (upper, with diagonal)
    Span<int> diag_positions;  ///< size \f$n \cdot \mathrm{batch\_size}\f$, absolute indices into lu's values

    Span<int> l_rows;       ///< row indices ordered by forward-solve level
    Span<int> l_level_ptr;  ///< size \f$\mathrm{l\_levels} + 1\f$
    Span<int> u_rows;       ///< row indices ordered by backward-solve level
    Span<int> u_level_ptr;  ///< size \f$\mathrm{u\_levels} + 1\f$
    int l_levels = 0;       ///< number of forward-solve levels
    int u_levels = 0;       ///< number of backward-solve levels

    bool u_diagonals_usable = false;  ///< every U diagonal is non-zero after the shift

    int n = 0;           ///< order of each matrix
    int batch_size = 0;  ///< number of matrices
    T diagonal_shift = T(1e-8);  ///< shift the apply uses on U's diagonal
};

/// @brief Owning ILU(k) factor of a batch of CSR matrices sharing one pattern.
///
/// Holds L (unit diagonal, strict lower triangle) and U (upper triangle with its
/// diagonal) in one CSR matrix, plus a level schedule for the two triangular
/// solves: rows inside one level depend only on earlier levels, so iluk_apply
/// walks `l_levels` / `u_levels` steps instead of \f$n\f$. One schedule serves the
/// whole batch. Build it with iluk_factorize; `view()` gives the ILUKView that
/// iluk_apply takes.
/// @tparam T  scalar type
/// @ingroup api_sparse
template <typename T>
struct ILUKPreconditioner {
    /// \f$1 \times 1\f$ placeholder with a single stored non-zero; iluk_factorize replaces it.
    ILUKPreconditioner() : lu(1, 1, NonZeros{1}, 1) {}

    Matrix<T, MatrixFormat::CSR> lu;  ///< L (strict lower, unit diagonal implied) and U (upper, with diagonal)
    UnifiedVector<int> diag_positions;  ///< size \f$n \cdot \mathrm{batch\_size}\f$

    UnifiedVector<int> l_rows;       ///< row indices ordered by forward-solve level
    UnifiedVector<int> l_level_ptr;  ///< size \f$\mathrm{l\_levels} + 1\f$
    UnifiedVector<int> u_rows;       ///< row indices ordered by backward-solve level
    UnifiedVector<int> u_level_ptr;  ///< size \f$\mathrm{u\_levels} + 1\f$
    int l_levels = 0;  ///< number of forward-solve levels
    int u_levels = 0;  ///< number of backward-solve levels

    /// Whether every U diagonal can be made non-singular with `diagonal_shift`.
    /// Decided once when the factor is built, so iluk_apply never syncs with the
    /// host to check it.
    bool u_diagonals_usable = false;

    int n = 0;               ///< order of each matrix
    int batch_size = 0;      ///< number of matrices
    int levels_of_fill = 0;  ///< the ILUKParams value it was built with
    T diagonal_shift = T(1e-8);  ///< the ILUKParams value it was built with
    typename base_type<T>::type drop_tolerance = typename base_type<T>::type(1e-4);  ///< the ILUKParams value it was built with
    typename base_type<T>::type fill_factor = typename base_type<T>::type(10);  ///< the ILUKParams value it was built with
    typename base_type<T>::type diag_pivot_threshold = typename base_type<T>::type(0.1);  ///< the ILUKParams value it was built with
    bool modified_ilu = true;  ///< the ILUKParams value it was built with

    /// @brief Non-owning view of this factor; valid while `*this` is alive and unmodified.
    /// @return an ILUKView aliasing this factor's storage
    ILUKView<T> view() const {
        ILUKView<T> v;
        v.lu = lu.view();
        v.diag_positions = diag_positions;
        v.l_rows = l_rows;
        v.l_level_ptr = l_level_ptr;
        v.u_rows = u_rows;
        v.u_level_ptr = u_level_ptr;
        v.l_levels = l_levels;
        v.u_levels = u_levels;
        v.u_diagonals_usable = u_diagonals_usable;
        v.n = n;
        v.batch_size = batch_size;
        v.diagonal_shift = diagonal_shift;
        return v;
    }
};

/// @brief Builds the triangular-solve level schedule from `M.lu`'s sparsity pattern.
///
/// iluk_factorize already does this; call it only for an ILUKPreconditioner
/// assembled by hand, because iluk_apply requires a valid schedule. Runs on the
/// host and reads `M.lu`'s row offsets, column indices and diagonal values (at `diag_positions`).
/// @param[in,out] M  factor whose `lu`, `n`, `batch_size`, `diag_positions` and
///                   `diagonal_shift` are set; its schedule fields (`l_rows`, `l_level_ptr`,
///                   `l_levels`, `u_rows`, `u_level_ptr`, `u_levels`) and
///                   `u_diagonals_usable` are overwritten
/// @throws batchlas::invalid_argument if `M.n <= 0`
/// @ingroup api_sparse_lowlevel
template <typename T>
BATCHLAS_API void iluk_build_level_schedule(ILUKPreconditioner<T>& M);

/// @brief Computes the ILU(k) factor of a batch of CSR matrices into owned storage.
///
/// Every item of `A` must share one sparsity pattern; the factor's pattern is the
/// union of what each item kept, so the batch keeps sharing one pattern. Blocking:
/// returns once the factor is complete. The numeric phase runs on the device for
/// large batches and on the host otherwise (`BATCHLAS_ILUK_DEVICE` forces either).
/// @tparam B  backend the call is compiled for
/// @tparam T  scalar type
/// @param[in] ctx     queue used for the device phase and allocations
/// @param[in] A       batch of \f$n \times n\f$ CSR matrices sharing one sparsity pattern; not modified
/// @param[in] params  fill, drop and pivot controls
/// @return the owning factor, ready for iluk_apply
/// @throws batchlas::invalid_argument if `A` is not square, a parameter is out of
///         range, or the batch items do not share one pattern
/// @throws batchlas::convergence_error if a pivot is zero and `diagonal_shift`
///         cannot stabilise it
/// @ingroup api_sparse
template <Backend B, typename T>
BATCHLAS_API ILUKPreconditioner<T> iluk_factorize(Queue& ctx,
                                                  const MatrixView<T, MatrixFormat::CSR>& A,
                                                  const ILUKParams<T>& params = ILUKParams<T>());

/// @brief Bytes of workspace the `Span` overload of iluk_factorize needs for `A`.
///
/// An upper bound: it sizes against the symbolic ILU(k) pattern, and the numeric
/// phase only prunes. Costs one host-side symbolic factorization of item 0.
/// @tparam B  backend the call is compiled for
/// @tparam T  scalar type
/// @param[in] ctx     queue the factorization will run on
/// @param[in] A       the batch of \f$n \times n\f$ CSR matrices that will be factored
/// @param[in] params  the parameters that will be used
/// @return required workspace size in bytes
/// @throws batchlas::invalid_argument for a non-square `A`, out-of-range params,
///         or `validate_batch_sparsity == false` with a batch of more than one
/// @ingroup api_sparse_lowlevel
template <Backend B, typename T>
BATCHLAS_API size_t iluk_buffer_size(Queue& ctx,
                                     const MatrixView<T, MatrixFormat::CSR>& A,
                                     const ILUKParams<T>& params = ILUKParams<T>());

/// @brief Computes the ILU(k) factor into caller-supplied memory instead of allocating.
///
/// Same factorization as the owning overload. Lets an iterative solver carve the
/// preconditioner out of a workspace it already holds. The exact footprint is
/// only known after the numeric phase, so it is reported through `bytes_used`
/// rather than predicted.
/// @tparam B  backend the call is compiled for
/// @tparam T  scalar type
/// @param[in] ctx          queue used for the device phase
/// @param[in] A            batch of \f$n \times n\f$ CSR matrices sharing one pattern; not modified
/// @param[out] workspace   at least iluk_buffer_size bytes of USM the host can write
///                         (the level schedule is copied in from the host); receives the
///                         factor and must outlive the returned view
/// @param[in] params       fill, drop and pivot controls
/// @param[out] bytes_used  if non-null, receives the bytes of `workspace` the factor occupies
/// @return a view that aliases `workspace` and is valid only while it is
/// @throws batchlas::invalid_argument, batchlas::convergence_error as the owning overload
/// @ingroup api_sparse
template <Backend B, typename T>
BATCHLAS_API ILUKView<T> iluk_factorize(Queue& ctx,
                                        const MatrixView<T, MatrixFormat::CSR>& A,
                                        Span<std::byte> workspace,
                                        const ILUKParams<T>& params,
                                        size_t* bytes_used = nullptr);

/// @brief Applies the ILU(k) preconditioner: `out` := \f$(LU)^{-1}\f$ `rhs`.
///
/// For every batch item and every column of `rhs`, solves \f$L y = r\f$ (unit
/// diagonal) and then \f$U x = y\f$ by level-scheduled sparse triangular solves,
/// and writes \f$x\f$ to `out`. Asynchronous: nothing is waited on, so repeated
/// applications inside an iterative solver stay pipelined.
/// @tparam B  backend the call is compiled for
/// @tparam T  scalar type
/// @param[in] ctx        queue the kernel is enqueued on
/// @param[in] M          factor from iluk_factorize, with a level schedule
/// @param[in] rhs        batch of \f$n \times \mathrm{nrhs}\f$ right-hand sides; not modified
/// @param[out] out       batch of \f$n \times \mathrm{nrhs}\f$ results \f$x = (LU)^{-1} r\f$
/// @param[in] workspace  unused (iluk_apply_buffer_size is 0); may be empty
/// @return event of the solve kernel
/// @throws batchlas::invalid_argument if `rhs` / `out` rows differ from `M.n`,
///         their column counts differ, their batch size differs from
///         `M.batch_size`, or `M` has no level schedule
/// @throws batchlas::convergence_error if `M.u_diagonals_usable` is false
/// @ingroup api_sparse
template <Backend B, typename T>
BATCHLAS_API Event iluk_apply(Queue& ctx,
                              const ILUKView<T>& M,
                              const MatrixView<T, MatrixFormat::Dense>& rhs,
                              const MatrixView<T, MatrixFormat::Dense>& out,
                              Span<std::byte> workspace = Span<std::byte>());

/// @brief iluk_apply on an owning factor; forwards `M.view()`.
/// @tparam B  backend the call is compiled for
/// @tparam T  scalar type
/// @param[in] ctx       queue the kernel is enqueued on
/// @param[in] M          owning factor from iluk_factorize
/// @param[in] rhs        batch of \f$n \times \mathrm{nrhs}\f$ right-hand sides; not modified
/// @param[out] out       batch of \f$n \times \mathrm{nrhs}\f$ results \f$x = (LU)^{-1} r\f$
/// @param[in] workspace  unused (iluk_apply_buffer_size is 0); may be empty
/// @return event of the solve kernel
/// @ingroup api_sparse
template <Backend B, typename T>
Event iluk_apply(Queue& ctx,
                 const ILUKPreconditioner<T>& M,
                 const MatrixView<T, MatrixFormat::Dense>& rhs,
                 const MatrixView<T, MatrixFormat::Dense>& out,
                 Span<std::byte> workspace = Span<std::byte>()) {
    return iluk_apply<B, T>(ctx, M.view(), rhs, out, workspace);
}

/// @brief Bytes of workspace iluk_apply needs; currently always 0.
/// @tparam B  backend the call is compiled for
/// @tparam T  scalar type
/// @param[in] ctx queue the apply will run on
/// @param[in] M    factor that will be applied
/// @param[in] rhs  batch of \f$n \times \mathrm{nrhs}\f$ right-hand sides that will be passed
/// @param[in] out  batch of \f$n \times \mathrm{nrhs}\f$ results that will be passed
/// @return required workspace size in bytes (0)
/// @ingroup api_sparse_lowlevel
template <Backend B, typename T>
BATCHLAS_API size_t iluk_apply_buffer_size(Queue& ctx,
                                           const ILUKView<T>& M,
                                           const MatrixView<T, MatrixFormat::Dense>& rhs,
                                           const MatrixView<T, MatrixFormat::Dense>& out);

/// @brief iluk_apply_buffer_size on an owning factor; forwards `M.view()`.
/// @tparam B  backend the call is compiled for
/// @tparam T  scalar type
/// @param[in] ctx queue the apply will run on
/// @param[in] M    owning factor that will be applied
/// @param[in] rhs  batch of \f$n \times \mathrm{nrhs}\f$ right-hand sides that will be passed
/// @param[in] out  batch of \f$n \times \mathrm{nrhs}\f$ results that will be passed
/// @return required workspace size in bytes (0)
/// @ingroup api_sparse_lowlevel
template <Backend B, typename T>
size_t iluk_apply_buffer_size(Queue& ctx,
                              const ILUKPreconditioner<T>& M,
                              const MatrixView<T, MatrixFormat::Dense>& rhs,
                              const MatrixView<T, MatrixFormat::Dense>& out) {
    return iluk_apply_buffer_size<B, T>(ctx, M.view(), rhs, out);
}

}  // namespace batchlas

namespace batchlas {

// Backend-deducing overloads: `f(ctx, ...)` uses ctx.backend().
// See BATCHLAS_DISPATCH_ON_QUEUE in blas/queue-dispatch.hh.

BATCHLAS_DISPATCH_ON_QUEUE(iluk_factorize)
BATCHLAS_DISPATCH_ON_QUEUE(iluk_buffer_size)
BATCHLAS_DISPATCH_ON_QUEUE(iluk_apply)
BATCHLAS_DISPATCH_ON_QUEUE(iluk_apply_buffer_size)

}  // namespace batchlas
