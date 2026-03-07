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
    int n = 0;
    int batch_size = 0;
    int levels_of_fill = 0;
    T diagonal_shift = T(1e-8);
    typename base_type<T>::type drop_tolerance = typename base_type<T>::type(1e-4);
    typename base_type<T>::type fill_factor = typename base_type<T>::type(10);
    typename base_type<T>::type diag_pivot_threshold = typename base_type<T>::type(0.1);
    bool modified_ilu = true;
};

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
