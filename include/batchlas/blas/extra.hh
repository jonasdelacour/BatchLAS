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
    UnifiedVector<float_t<T>> norm(Queue &ctx,
                          const MatrixView<T, MF> &A,
                          const NormType norm_type = NormType::Frobenius);

    template <Backend B, typename T, MatrixFormat MF>
    Event cond(Queue &ctx,
              const MatrixView<T, MF> &A,
              const NormType norm_type,
              const Span<T> conds,
              const Span<std::byte> workspace);

    // Workspace size for the `cond` overload above. Instantiated only for T in
    // {float, double} with MF == MatrixFormat::Dense (COND_INSTANTIATE in
    // src/extra/cond.cc); any other T or MF is a link error, not a compile error.
    template <Backend B, typename T, MatrixFormat MF>
    size_t cond_buffer_size(Queue &ctx,
                            const MatrixView<T, MF> &A,
                            const NormType norm_type);

    template <Backend B, typename T, MatrixFormat MF>
    UnifiedVector<T> cond(Queue &ctx,
                          const MatrixView<T, MF> &A,
                          const NormType norm_type);

    // log10_kappa is log10(κ2) or log10(κF) depending on metric (Spectral or Frobenius only).
    // `algo` defaults to CGS2 deliberately: Chol-QR squares the condition number of an
    // uncontrolled random input and returns non-finite items in float, and Householder
    // leaves some items singular, silently not honouring the requested kappa.
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
    Matrix<T, MF> transpose(Queue &ctx,
                            const MatrixView<T, MF> &A);

    // Owning-argument and backend-deducing overloads: `f(ctx, Matrix, ...)` accepts
    // owning containers where the primary takes views, and `f(ctx, ...)` uses
    // ctx.backend(). See BATCHLAS_ACCEPT_OWNING and BATCHLAS_DISPATCH_ON_QUEUE in
    // blas/queue-dispatch.hh.
    //
    // `norm` and `transpose` are not Backend-templated, so they need no dispatch
    // overload, but they do need the owning-argument one -- hence its _NB spelling,
    // which solves deduction of `T` and `MF` either way.
    //
    // The random_*_with_log10_cond_metric generators are deliberately absent from
    // both: their `T` is non-deduced (it appears only as `float_t<T>` and in the
    // return type), so either macro would expand to nothing at all, reading as if
    // they were dispatchable when they are not. They keep the explicit
    // f<Backend, T>(...) spelling. See docs/cpp-api.md#which-spelling-each-entry-point-takes.
    BATCHLAS_ACCEPT_OWNING(cond)
    BATCHLAS_ACCEPT_OWNING(cond_buffer_size)
    BATCHLAS_ACCEPT_OWNING_NB(norm)
    BATCHLAS_ACCEPT_OWNING_NB(transpose)

    BATCHLAS_DISPATCH_ON_QUEUE(cond)
    BATCHLAS_DISPATCH_ON_QUEUE(cond_buffer_size)

}
