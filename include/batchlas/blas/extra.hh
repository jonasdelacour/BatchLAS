#pragma once
#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/blas/queue-dispatch.hh>

namespace batchlas
{

    template <typename T, MatrixFormat MF>
    Event norm(Queue &ctx,
              const MatrixView<T, MF> &A,
              const NormType norm_type,
              const Span<float_t<T>> norms);

    template <typename T, MatrixFormat MF>
    inline Event norm(Queue &ctx,
              const Matrix<T, MF> &A,
              const NormType norm_type,
              const Span<float_t<T>> norms) {
        return norm<T,MF>(ctx, MatrixView<T,MF>(A), norm_type, norms);
    }

    template <typename T, MatrixFormat MF>
    UnifiedVector<float_t<T>> norm(Queue &ctx,
                          const MatrixView<T, MF> &A,
                          const NormType norm_type = NormType::Frobenius);

    template <typename T, MatrixFormat MF>
    inline UnifiedVector<float_t<T>> norm(Queue &ctx,
                          const Matrix<T, MF> &A,
                          const NormType norm_type = NormType::Frobenius) {
        return norm<T,MF>(ctx, MatrixView<T,MF>(A), norm_type);
    }

    template <Backend B, typename T, MatrixFormat MF>
    Event cond(Queue &ctx,
              const MatrixView<T, MF> &A,
              const NormType norm_type,
              const Span<T> conds,
              const Span<std::byte> workspace);

    template <Backend B, typename T, MatrixFormat MF>
    inline Event cond(Queue &ctx,
              const Matrix<T, MF> &A,
              const NormType norm_type,
              const Span<T> conds,
              const Span<std::byte> workspace) {
        return cond<B,T,MF>(ctx, MatrixView<T,MF>(A), norm_type, conds, workspace);
    }

    // Workspace size for the `cond` overload above. Instantiated only for T in
    // {float, double} with MF == MatrixFormat::Dense (COND_INSTANTIATE in
    // src/extra/cond.cc); any other T or MF is a link error, not a compile error.
    template <Backend B, typename T, MatrixFormat MF>
    size_t cond_buffer_size(Queue &ctx,
                            const MatrixView<T, MF> &A,
                            const NormType norm_type);

    template <Backend B, typename T, MatrixFormat MF>
    inline size_t cond_buffer_size(Queue &ctx,
                            const Matrix<T, MF> &A,
                            const NormType norm_type) {
        return cond_buffer_size<B,T,MF>(ctx, MatrixView<T,MF>(A), norm_type);
    }

    template <Backend B, typename T, MatrixFormat MF>
    UnifiedVector<T> cond(Queue &ctx,
                          const MatrixView<T, MF> &A,
                          const NormType norm_type);

    template <Backend B, typename T, MatrixFormat MF>
    inline UnifiedVector<T> cond(Queue &ctx,
                          const Matrix<T, MF> &A,
                          const NormType norm_type) {
        return cond<B,T,MF>(ctx, MatrixView<T,MF>(A), norm_type);
    }

    // log10_kappa is log10(κ2) or log10(κF) depending on metric (Spectral or Frobenius only).
    //
    // `algo` defaults to CGS2, not to a Cholesky variant, deliberately: Chol-QR
    // squares the condition number of its input, which here is an uncontrolled
    // Matrix::Random, so in float potrf fails, its info code is discarded (see
    // src/extra/random_cond.cc) and whole batch items come back non-finite.
    // Householder is the tempting alternative and is worse: it removes the NaNs
    // but leaves some items singular, so the requested kappa is then silently
    // not honoured.
    template <Backend B, typename T>
    Matrix<T, MatrixFormat::Dense> random_with_log10_cond_metric(Queue &ctx,
                                                                  int n,
                                                                  float_t<T> log10_kappa,
                                                                  NormType metric,
                                                                  int batch_size = 1,
                                                                  unsigned int seed = 42,
                                                                  OrthoAlgorithm algo = OrthoAlgorithm::CGS2);

    template <Backend B, typename T>
    Matrix<T, MatrixFormat::Dense> random_hermitian_with_log10_cond_metric(Queue &ctx,
                                                                           int n,
                                                                           float_t<T> log10_kappa,
                                                                           NormType metric,
                                                                           int batch_size = 1,
                                                                           unsigned int seed = 42,
                                                                           OrthoAlgorithm algo = OrthoAlgorithm::CGS2);

    // The resulting bandwidth is <= kd. For small kd, this may produce diagonal matrices.
    template <Backend B, typename T>
    Matrix<T, MatrixFormat::Dense> random_banded_with_log10_cond_metric(Queue &ctx,
                                                                         int n,
                                                                         int kd,
                                                                         float_t<T> log10_kappa,
                                                                         NormType metric,
                                                                         int batch_size = 1,
                                                                         unsigned int seed = 42);

    template <Backend B, typename T>
    Matrix<T, MatrixFormat::Dense> random_hermitian_banded_with_log10_cond_metric(Queue &ctx,
                                                                                  int n,
                                                                                  int kd,
                                                                                  float_t<T> log10_kappa,
                                                                                  NormType metric,
                                                                                  int batch_size = 1,
                                                                                  unsigned int seed = 42);

    // The condition number is enforced via the diagonal spectrum; off-diagonals are zero.
    template <Backend B, typename T>
    Matrix<T, MatrixFormat::Dense> random_tridiagonal_with_log10_cond_metric(Queue &ctx,
                                                                             int n,
                                                                             float_t<T> log10_kappa,
                                                                             NormType metric,
                                                                             int batch_size = 1,
                                                                             unsigned int seed = 42);

    template <Backend B, typename T>
    Matrix<T, MatrixFormat::Dense> random_hermitian_tridiagonal_with_log10_cond_metric(Queue &ctx,
                                                                                       int n,
                                                                                       float_t<T> log10_kappa,
                                                                                       NormType metric,
                                                                                       int batch_size = 1,
                                                                                       unsigned int seed = 42);

    template <typename T, MatrixFormat MF>
    Event transpose(Queue &ctx,
                    const MatrixView<T, MF> &A,
                    const MatrixView<T, MF> &B);

    template <typename T, MatrixFormat MF>
    inline Event transpose(Queue &ctx,
                    const Matrix<T, MF> &A,
                    const Matrix<T, MF> &B) {
        return transpose<T,MF>(ctx, MatrixView<T,MF>(A), MatrixView<T,MF>(B));
    }

    template <typename T, MatrixFormat MF>
    Matrix<T, MF> transpose(Queue &ctx,
                            const MatrixView<T, MF> &A);

    template <typename T, MatrixFormat MF>
    inline Matrix<T, MF> transpose(Queue &ctx,
                            const Matrix<T, MF> &A) {
        return transpose<T,MF>(ctx, MatrixView<T,MF>(A));
    }

    // Backend-deducing overloads; see BATCHLAS_DISPATCH_ON_QUEUE in
    // blas/queue-dispatch.hh. The random_*_with_log10_cond_metric generators are
    // deliberately absent: their `T` appears only as `float_t<T>` (a non-deduced
    // context) and in the return type, so the macro would expand to nothing and
    // read as if they were dispatchable. They keep the explicit f<Backend, T>
    // spelling; see docs/cpp-api.md#which-spelling-each-entry-point-takes.
    BATCHLAS_DISPATCH_ON_QUEUE(cond)
    BATCHLAS_DISPATCH_ON_QUEUE(cond_buffer_size)

}
