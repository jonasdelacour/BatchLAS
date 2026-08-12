#pragma once
#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/util/sycl-device-queue.hh>

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

    // Create a batch of random dense matrices with a specified log10 conditioning metric.
    // log10_kappa is log10(κ2) or log10(κF) depending on metric (Spectral or Frobenius only).
    //
    // `algo` orthonormalises the two random factors. It defaults to CGS2, NOT to
    // one of the Cholesky variants, and that is deliberate: Chol-QR forms the
    // Gram matrix and so squares the condition number of its input. The input
    // here is a raw Matrix::Random, whose conditioning is NOT controlled (the
    // requested kappa is imposed afterwards, by the diagonal), and in float the
    // squared Gram goes numerically indefinite once kappa exceeds about 1e4 --
    // which ordinary random draws reach a few times in a batch of 32 at n=64.
    // potrf then fails, its info code is discarded (see the note in
    // src/extra/random_cond.cc), the following trsm back-substitutes through a
    // garbage diagonal, and the two gemms smear the resulting NaN across every
    // entry of that batch item. That is the "generator intermittently emits an
    // entirely non-finite matrix" defect reported against PR #66.
    //
    // Measured at n=64, float, batch=32, seed 1, requesting log10(kappa_F) = 5;
    // "ortho err" is ||Q^H Q - I||_max and the bracket is the range of log10 of
    // the ACHIEVED condition number over the batch:
    //
    //   Chol2 (old default)  3.9e-7   [5.000,  5.000]   2/32 non-finite
    //   Cholesky             3.2e-2   [4.992,  5.003]   2/32 non-finite
    //   ShiftChol3           3.4e-7   [5.000,  5.000]   0/32
    //   Householder          4.3e-7   [5.000, 17.390]   0/32
    //   CGS2                 2.3e-7   [5.000,  5.000]   0/32
    //   SVQB2                4.0e-6   [5.000,  5.000]   0/32
    //
    // Householder is the obvious candidate and is the WRONG one: it removes the
    // NaNs but returns a factor that leaves some batch items numerically
    // singular, so the generator silently stops honouring the requested kappa --
    // a worse failure than the one being fixed, because it is invisible. That is
    // a latent defect in ortho's geqrf+orgqr path, not in this generator.
    //
    // CGS2 has the best orthogonality of the six, honours the requested kappa
    // exactly, and uses no potrf at all, so it cannot be caught by the unchecked
    // info code. Callers who want the faster Cholesky path on input they know to
    // be well conditioned can still ask for it explicitly.
    template <Backend B, typename T>
    Matrix<T, MatrixFormat::Dense> random_with_log10_cond_metric(Queue &ctx,
                                                                  int n,
                                                                  float_t<T> log10_kappa,
                                                                  NormType metric,
                                                                  int batch_size = 1,
                                                                  unsigned int seed = 42,
                                                                  OrthoAlgorithm algo = OrthoAlgorithm::CGS2);

    // Create a batch of random symmetric/Hermitian dense matrices with a specified log10 conditioning metric.
    // log10_kappa is log10(κ2) or log10(κF) depending on metric (Spectral or Frobenius only).
    // See random_with_log10_cond_metric above for why `algo` defaults to CGS2.
    template <Backend B, typename T>
    Matrix<T, MatrixFormat::Dense> random_hermitian_with_log10_cond_metric(Queue &ctx,
                                                                           int n,
                                                                           float_t<T> log10_kappa,
                                                                           NormType metric,
                                                                           int batch_size = 1,
                                                                           unsigned int seed = 42,
                                                                           OrthoAlgorithm algo = OrthoAlgorithm::CGS2);

    // Create a batch of random dense banded matrices (general) with a specified log10 conditioning metric.
    // The resulting bandwidth is <= kd. For small kd, this may produce diagonal matrices.
    template <Backend B, typename T>
    Matrix<T, MatrixFormat::Dense> random_banded_with_log10_cond_metric(Queue &ctx,
                                                                         int n,
                                                                         int kd,
                                                                         float_t<T> log10_kappa,
                                                                         NormType metric,
                                                                         int batch_size = 1,
                                                                         unsigned int seed = 42);

    // Create a batch of random symmetric/Hermitian banded matrices with a specified log10 conditioning metric.
    // log10_kappa is log10(κ2) or log10(κF) depending on metric (Spectral or Frobenius only).
    template <Backend B, typename T>
    Matrix<T, MatrixFormat::Dense> random_hermitian_banded_with_log10_cond_metric(Queue &ctx,
                                                                                  int n,
                                                                                  int kd,
                                                                                  float_t<T> log10_kappa,
                                                                                  NormType metric,
                                                                                  int batch_size = 1,
                                                                                  unsigned int seed = 42);

    // Create a batch of random tridiagonal dense matrices (general) with a specified log10 conditioning metric.
    // The condition number is enforced via the diagonal spectrum; off-diagonals are zero.
    template <Backend B, typename T>
    Matrix<T, MatrixFormat::Dense> random_tridiagonal_with_log10_cond_metric(Queue &ctx,
                                                                             int n,
                                                                             float_t<T> log10_kappa,
                                                                             NormType metric,
                                                                             int batch_size = 1,
                                                                             unsigned int seed = 42);

    // Create a batch of random symmetric/Hermitian tridiagonal dense matrices with a specified log10 conditioning metric.
    // log10_kappa is log10(κ2) or log10(κF) depending on metric (Spectral or Frobenius only).
    template <Backend B, typename T>
    Matrix<T, MatrixFormat::Dense> random_hermitian_tridiagonal_with_log10_cond_metric(Queue &ctx,
                                                                                       int n,
                                                                                       float_t<T> log10_kappa,
                                                                                       NormType metric,
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
