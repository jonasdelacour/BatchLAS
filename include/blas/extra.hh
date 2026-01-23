#pragma once
#include <blas/matrix.hh>
#include <blas/enums.hh>
#include <util/sycl-vector.hh>
#include <util/sycl-device-queue.hh>

namespace batchlas
{

    // Memory passed from outside
    template <typename T, MatrixFormat MF>
    Event norm(Queue &ctx,
              const MatrixView<T, MF> &A,
              const NormType norm_type,
              const Span<float_t<T>> norms);

    // Forwarding overload (owning A)
    template <typename T, MatrixFormat MF>
    inline Event norm(Queue &ctx,
              const Matrix<T, MF> &A,
              const NormType norm_type,
              const Span<float_t<T>> norms) {
        return norm<T,MF>(ctx, MatrixView<T,MF>(A), norm_type, norms);
    }

    // Convenience function which allocates and returns the results stored in an array.
    template <typename T, MatrixFormat MF>
    UnifiedVector<float_t<T>> norm(Queue &ctx,
                          const MatrixView<T, MF> &A,
                          const NormType norm_type = NormType::Frobenius);

    // Forwarding overload (owning A)
    template <typename T, MatrixFormat MF>
    inline UnifiedVector<float_t<T>> norm(Queue &ctx,
                          const Matrix<T, MF> &A,
                          const NormType norm_type = NormType::Frobenius) {
        return norm<T,MF>(ctx, MatrixView<T,MF>(A), norm_type);
    }

    //Memory passed from outside
    template <Backend B, typename T, MatrixFormat MF>
    Event cond(Queue &ctx,
              const MatrixView<T, MF> &A,
              const NormType norm_type,
              const Span<T> conds,
              const Span<std::byte> workspace);

    // Forwarding overload (owning A)
    template <Backend B, typename T, MatrixFormat MF>
    inline Event cond(Queue &ctx,
              const Matrix<T, MF> &A,
              const NormType norm_type,
              const Span<T> conds,
              const Span<std::byte> workspace) {
        return cond<B,T,MF>(ctx, MatrixView<T,MF>(A), norm_type, conds, workspace);
    }

    size_t cond_buffer_size(Queue &ctx,
                            const MatrixView<float, MatrixFormat::Dense> &A,
                            const NormType norm_type);

    //Convenience function which allocates memory internally
    template <Backend B, typename T, MatrixFormat MF>
    UnifiedVector<T> cond(Queue &ctx,
                          const MatrixView<T, MF> &A,
                          const NormType norm_type);

    // Forwarding overload (owning A)
    template <Backend B, typename T, MatrixFormat MF>
    inline UnifiedVector<T> cond(Queue &ctx,
                          const Matrix<T, MF> &A,
                          const NormType norm_type) {
        return cond<B,T,MF>(ctx, MatrixView<T,MF>(A), norm_type);
    }

    // Create a batch of random dense matrices with a specified log10 condition number.
    // The condition number is enforced in the 2-norm via singular values spanning [1, 10^log10_cond].
    template <Backend B, typename T>
    Matrix<T, MatrixFormat::Dense> random_with_log10_cond(Queue &ctx,
                                                          int n,
                                                          float_t<T> log10_cond,
                                                          int batch_size = 1,
                                                          unsigned int seed = 42,
                                                          OrthoAlgorithm algo = OrthoAlgorithm::Chol2);

    // Create a batch of random symmetric/Hermitian dense matrices with a specified log10 condition number.
    // The condition number is enforced in the 2-norm via eigenvalues spanning [1, 10^log10_cond].
    template <Backend B, typename T>
    Matrix<T, MatrixFormat::Dense> random_hermitian_with_log10_cond(Queue &ctx,
                                                                    int n,
                                                                    float_t<T> log10_cond,
                                                                    int batch_size = 1,
                                                                    unsigned int seed = 42,
                                                                    OrthoAlgorithm algo = OrthoAlgorithm::Chol2);

    // Create a batch of random dense banded matrices (general) with a specified log10 condition number.
    // The resulting bandwidth is <= kd. For small kd, this may produce diagonal matrices.
    template <Backend B, typename T>
    Matrix<T, MatrixFormat::Dense> random_banded_with_log10_cond(Queue &ctx,
                                                                 int n,
                                                                 int kd,
                                                                 float_t<T> log10_cond,
                                                                 int batch_size = 1,
                                                                 unsigned int seed = 42);

    // Create a batch of random symmetric/Hermitian banded matrices with a specified log10 condition number.
    // The resulting bandwidth is <= kd. For small kd, this may produce diagonal matrices.
    template <Backend B, typename T>
    Matrix<T, MatrixFormat::Dense> random_hermitian_banded_with_log10_cond(Queue &ctx,
                                                                           int n,
                                                                           int kd,
                                                                           float_t<T> log10_cond,
                                                                           int batch_size = 1,
                                                                           unsigned int seed = 42);

    // Create a batch of random tridiagonal dense matrices (general) with a specified log10 condition number.
    // The condition number is enforced via the diagonal spectrum; off-diagonals are zero.
    template <Backend B, typename T>
    Matrix<T, MatrixFormat::Dense> random_tridiagonal_with_log10_cond(Queue &ctx,
                                                                      int n,
                                                                      float_t<T> log10_cond,
                                                                      int batch_size = 1,
                                                                      unsigned int seed = 42);

    // Create a batch of random symmetric/Hermitian tridiagonal dense matrices with a specified log10 condition number.
    // The condition number is enforced via the diagonal spectrum; off-diagonals are zero.
    template <Backend B, typename T>
    Matrix<T, MatrixFormat::Dense> random_hermitian_tridiagonal_with_log10_cond(Queue &ctx,
                                                                                int n,
                                                                                float_t<T> log10_cond,
                                                                                int batch_size = 1,
                                                                                unsigned int seed = 42);

    // Batched matrix transpose into preallocated output
    template <typename T, MatrixFormat MF>
    Event transpose(Queue &ctx,
                    const MatrixView<T, MF> &A,
                    const MatrixView<T, MF> &B);

    // Forwarding overload (owning A and B)
    template <typename T, MatrixFormat MF>
    inline Event transpose(Queue &ctx,
                    const Matrix<T, MF> &A,
                    const Matrix<T, MF> &B) {
        return transpose<T,MF>(ctx, MatrixView<T,MF>(A), MatrixView<T,MF>(B));
    }

    // Convenience overload allocating the output matrix
    template <typename T, MatrixFormat MF>
    Matrix<T, MF> transpose(Queue &ctx,
                            const MatrixView<T, MF> &A);

    // Forwarding overload (owning A)
    template <typename T, MatrixFormat MF>
    inline Matrix<T, MF> transpose(Queue &ctx,
                            const Matrix<T, MF> &A) {
        return transpose<T,MF>(ctx, MatrixView<T,MF>(A));
    }

}
