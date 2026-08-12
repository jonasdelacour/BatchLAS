#pragma once

#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>

#include <cstddef>
#include <cstdint>
#include <batchlas/blas/queue-dispatch.hh>

namespace batchlas {

template <typename T>
struct ILUKParams {
    int levels_of_fill = 0;
    T diagonal_shift = T(1e-8);
    typename base_type<T>::type drop_tolerance = typename base_type<T>::type(1e-4);
    typename base_type<T>::type fill_factor = typename base_type<T>::type(10);
    typename base_type<T>::type diag_pivot_threshold = typename base_type<T>::type(0.1);
    bool modified_ilu = true;
    // Current implementation supports batches only when all matrices share the same CSR pattern.
    bool validate_batch_sparsity = true;
};

// Non-owning description of a factored ILU(k) preconditioner. This is what the
// apply kernel actually consumes, so a factor can live either in storage owned by
// an ILUKPreconditioner or in a caller-supplied workspace (see the Span overload
// of iluk_factorize). The view is valid only while that storage is.
template <typename T>
struct ILUKView {
    MatrixView<T, MatrixFormat::CSR> lu;
    Span<int> diag_positions;  // size = n * batch_size, absolute indices into lu's values

    Span<int> l_rows;
    Span<int> l_level_ptr;
    Span<int> u_rows;
    Span<int> u_level_ptr;
    int l_levels = 0;
    int u_levels = 0;

    bool u_diagonals_usable = false;

    int n = 0;
    int batch_size = 0;
    T diagonal_shift = T(1e-8);
};

template <typename T>
struct ILUKPreconditioner {
    // 1x1 placeholder with a single stored non-zero; iluk_factorize replaces it.
    ILUKPreconditioner() : lu(1, 1, NonZeros{1}, 1) {}

    // Factor storage uses unit-diagonal L in the strict lower triangle and explicit-diagonal U on/above the diagonal.
    Matrix<T, MatrixFormat::CSR> lu;
    UnifiedVector<int> diag_positions;  // size = n * batch_size

    // Level schedule for the two sparse triangular solves. Rows inside one level
    // depend only on rows in earlier levels, so they can be solved concurrently;
    // this turns the n-step serial walk in iluk_apply into a walk over
    // `l_levels`/`u_levels` steps. The sparsity pattern is shared across the
    // batch, so a single schedule serves every batch element.
    UnifiedVector<int> l_rows;       // row indices ordered by forward-solve level
    UnifiedVector<int> l_level_ptr;  // size l_levels + 1
    UnifiedVector<int> u_rows;       // row indices ordered by backward-solve level
    UnifiedVector<int> u_level_ptr;  // size u_levels + 1
    int l_levels = 0;
    int u_levels = 0;

    // Whether every U diagonal can be made non-singular with `diagonal_shift`.
    // The factor values do not change between applications, so this is decided once
    // (by iluk_factorize / iluk_build_level_schedule) rather than re-checked on the
    // host after every solve, which would force a device sync per LOBPCG iteration.
    bool u_diagonals_usable = false;

    int n = 0;
    int batch_size = 0;
    int levels_of_fill = 0;
    T diagonal_shift = T(1e-8);
    typename base_type<T>::type drop_tolerance = typename base_type<T>::type(1e-4);
    typename base_type<T>::type fill_factor = typename base_type<T>::type(10);
    typename base_type<T>::type diag_pivot_threshold = typename base_type<T>::type(0.1);
    bool modified_ilu = true;

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

// Populate the triangular-solve level schedule from M.lu's sparsity pattern.
// iluk_factorize does this already; call it only when building an
// ILUKPreconditioner by hand, since iluk_apply requires a valid schedule.
template <typename T>
void iluk_build_level_schedule(ILUKPreconditioner<T>& M);

template <Backend B, typename T>
ILUKPreconditioner<T> iluk_factorize(Queue& ctx,
                                     const MatrixView<T, MatrixFormat::CSR>& A,
                                     const ILUKParams<T>& params = ILUKParams<T>());

// Bytes of workspace the Span overload of iluk_factorize needs for A. The exact
// fill count is only known after the numeric phase (drop tolerance and fill
// control both prune entries), so this sizes against the symbolic pattern, which
// is an upper bound. Running it costs one symbolic factorization.
template <Backend B, typename T>
size_t iluk_buffer_size(Queue& ctx,
                        const MatrixView<T, MatrixFormat::CSR>& A,
                        const ILUKParams<T>& params = ILUKParams<T>());

// Factorize A into caller-supplied memory instead of allocating. Lets an
// iterative solver carve the preconditioner out of the workspace it was already
// given. The returned view aliases `workspace` and dies with it.
// `bytes_used`, when given, receives how much of `workspace` the factor occupies.
// The exact figure is only knowable after the numeric phase, so a caller that
// wants to keep sub-allocating from the same pool needs it reported back rather
// than predicted -- predicting it means running the symbolic phase a second time.
template <Backend B, typename T>
ILUKView<T> iluk_factorize(Queue& ctx,
                           const MatrixView<T, MatrixFormat::CSR>& A,
                           Span<std::byte> workspace,
                           const ILUKParams<T>& params,
                           size_t* bytes_used = nullptr);

template <Backend B, typename T>
Event iluk_apply(Queue& ctx,
                 const ILUKView<T>& M,
                 const MatrixView<T, MatrixFormat::Dense>& rhs,
                 const MatrixView<T, MatrixFormat::Dense>& out,
                 Span<std::byte> workspace = Span<std::byte>());

// Convenience overload so callers holding an owning factor need not spell .view().
template <Backend B, typename T>
Event iluk_apply(Queue& ctx,
                 const ILUKPreconditioner<T>& M,
                 const MatrixView<T, MatrixFormat::Dense>& rhs,
                 const MatrixView<T, MatrixFormat::Dense>& out,
                 Span<std::byte> workspace = Span<std::byte>()) {
    return iluk_apply<B, T>(ctx, M.view(), rhs, out, workspace);
}

template <Backend B, typename T>
size_t iluk_apply_buffer_size(Queue& ctx,
                              const ILUKView<T>& M,
                              const MatrixView<T, MatrixFormat::Dense>& rhs,
                              const MatrixView<T, MatrixFormat::Dense>& out);

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
