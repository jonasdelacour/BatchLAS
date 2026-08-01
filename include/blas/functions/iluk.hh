#pragma once

#include <blas/matrix.hh>
#include <blas/enums.hh>
#include <util/sycl-device-queue.hh>
#include <util/sycl-span.hh>
#include <util/sycl-vector.hh>

#include <cstddef>
#include <cstdint>

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

template <typename T>
struct ILUKPreconditioner {
    ILUKPreconditioner() : lu(1, 1, 1, 1) {}

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

template <Backend B, typename T>
Event iluk_apply(Queue& ctx,
                 const ILUKPreconditioner<T>& M,
                 const MatrixView<T, MatrixFormat::Dense>& rhs,
                 const MatrixView<T, MatrixFormat::Dense>& out,
                 Span<std::byte> workspace = Span<std::byte>());

template <Backend B, typename T>
size_t iluk_apply_buffer_size(Queue& ctx,
                              const ILUKPreconditioner<T>& M,
                              const MatrixView<T, MatrixFormat::Dense>& rhs,
                              const MatrixView<T, MatrixFormat::Dense>& out);

}  // namespace batchlas
