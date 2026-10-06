#pragma once
#include <batchlas/export.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/blas/queue-dispatch.hh>

/// @file
/// @brief Matrix norms, condition numbers, conditioned test-matrix generators and transpose.
/// @ingroup extra

namespace batchlas
{

    /// @addtogroup extra
    /// @{

    /// @brief Per-item matrix norm, written into a caller-owned span.
    ///
    /// For each batch item \f$A_i\f$, `norms[i]` receives \f$\|A_i\|\f$ in the
    /// requested norm: Frobenius, One (max column sum), Inf (max row sum), Max
    /// (max absolute entry) or Spectral (largest \f$|\lambda|\f$).
    /// @tparam T      float, double, std::complex<float> or std::complex<double>
    /// @tparam MF     matrix format; only MatrixFormat::Dense is instantiated
    /// @param ctx       queue the kernels are enqueued on
    /// @param A         batch of matrices; not modified
    /// @param norm_type which norm
    /// @param norms     device-accessible output, at least `A.batch_size()` elements
    /// @return          event of the last enqueued kernel; `norms` is valid after it
    /// @pre  NormType::Spectral requires square symmetric/Hermitian `A` (the lower
    ///       triangle is read).
    /// @throws batchlas::invalid_argument for Spectral on a non-square `A`
    /// @throws batchlas::unsupported for Spectral on a non-dense format or when no
    ///         vendor `syev` is available
    template <typename T, MatrixFormat MF>
    BATCHLAS_API Event norm(Queue &ctx,
                           const MatrixView<T, MF> &A,
                           const NormType norm_type,
                           const Span<float_t<T>> norms);

    /// @brief Per-item matrix norm, returned in a new vector. Blocks until done.
    ///
    /// Same contract as the span overload, but allocates the result and waits on it
    /// before returning.
    /// @return one real norm per batch item
    template <typename T, MatrixFormat MF>
    BATCHLAS_API UnifiedVector<float_t<T>> norm(Queue &ctx,
                                       const MatrixView<T, MF> &A,
                                       const NormType norm_type = NormType::Frobenius);

    /// @brief Per-item condition number, written into a caller-owned span.
    ///
    /// For norm types other than Spectral, `conds[i]` = \f$\|A_i\|\,\|A_i^{-1}\|\f$,
    /// computed by explicit inversion. For NormType::Spectral, the ratio of the
    /// largest to the smallest eigenvalue magnitude of a symmetric/Hermitian \f$A_i\f$
    /// (lower triangle read; infinity when an eigenvalue is exactly zero).
    /// @tparam B        backend; deduced from the Queue on the `cond(ctx, ...)` spelling
    /// @tparam T        float or double (real only)
    /// @tparam MF       MatrixFormat::Dense
    /// @param ctx       queue the kernels are enqueued on
    /// @param A         batch of square matrices; not modified
    /// @param norm_type which norm
    /// @param conds     device-accessible output, at least `A.batch_size()` elements
    /// @param workspace scratch of at least cond_buffer_size() bytes; must outlive the kernels
    /// @return          event of the last enqueued kernel
    /// @pre  `A` is square and nonsingular
    /// @throws batchlas::invalid_argument for Spectral on a non-square `A`
    /// @throws batchlas::unsupported for Spectral on a non-dense format
    /// @note The Spectral path calls the vendor `syev` directly and throws in a
    ///       vendor-free build; see @ref md_docs_2design_2known-defects (defect 2).
    template <Backend B, typename T, MatrixFormat MF>
    BATCHLAS_API Event cond(Queue &ctx,
                           const MatrixView<T, MF> &A,
                           const NormType norm_type,
                           const Span<T> conds,
                           const Span<std::byte> workspace);

    /// @brief Workspace bytes needed by the span overload of cond().
    ///
    /// Instantiated only for T in {float, double} with MatrixFormat::Dense; any other
    /// combination is a link error, not a compile error.
    /// @return size in bytes for the `workspace` argument of cond()
    template <Backend B, typename T, MatrixFormat MF>
    BATCHLAS_API size_t cond_buffer_size(Queue &ctx,
                                         const MatrixView<T, MF> &A,
                                         const NormType norm_type);

    /// @brief Per-item condition number, returned in a new vector. Blocks until done.
    ///
    /// Leases its workspace from the queue's arena and waits before returning.
    /// @return one condition number per batch item
    template <Backend B, typename T, MatrixFormat MF>
    BATCHLAS_API UnifiedVector<T> cond(Queue &ctx,
                                       const MatrixView<T, MF> &A,
                                       const NormType norm_type);

    /// @brief Random dense matrices with a prescribed log10 condition number.
    ///
    /// Builds \f$A_i = U \Sigma V^H\f$ with random orthonormal factors and a singular
    /// spectrum chosen so that \f$\log_{10}\kappa(A_i)\f$ equals @p log10_kappa in the
    /// requested metric. Blocks until the result is written.
    /// @note On Backend::NETLIB the result is the diagonal \f$\Sigma\f$ itself, with
    ///       no random rotation.
    /// @tparam B  backend (not deducible; spell `f<Backend, T>(...)`)
    /// @tparam T  scalar type (not deducible)
    /// @param ctx          queue the kernels are enqueued on
    /// @param n            order of each matrix, > 0
    /// @param log10_kappa  target \f$\log_{10}\kappa\f$, >= 0 (>= \f$\log_{10} n\f$ for Frobenius)
    /// @param metric       NormType::Spectral (\f$\kappa_2\f$) or NormType::Frobenius (\f$\kappa_F\f$)
    /// @param batch_size   number of matrices, > 0
    /// @param seed         RNG seed; equal seeds give equal batches
    /// @param algo         orthogonalisation used to build the random factors. Keep
    ///                     the CGS2 default: Chol-QR and Householder can silently
    ///                     miss the requested kappa.
    /// @return a new n x n x batch_size matrix
    /// @throws batchlas::invalid_argument on a non-positive size, a negative
    ///         `log10_kappa`, or a metric other than Spectral/Frobenius
    // evidence: docs/design/api-conventions.md#api-conventions-cond-generators-default-to-cgs2
    template <Backend B, typename T>
    BATCHLAS_API Matrix<T, MatrixFormat::Dense> random_with_log10_cond_metric(Queue &ctx,
                                                                               int n,
                                                                               float_t<T> log10_kappa,
                                                                               NormType metric,
                                                                               int batch_size = 1,
                                                                               unsigned int seed = 42,
                                                                               OrthoAlgorithm algo = OrthoAlgorithm::CGS2);

    /// @brief Random symmetric/Hermitian matrices with a prescribed log10 condition number.
    ///
    /// As random_with_log10_cond_metric(), with \f$A_i = Q \Lambda Q^H\f$.
    template <Backend B, typename T>
    BATCHLAS_API Matrix<T, MatrixFormat::Dense> random_hermitian_with_log10_cond_metric(Queue &ctx,
                                                                                        int n,
                                                                                        float_t<T> log10_kappa,
                                                                                        NormType metric,
                                                                                        int batch_size = 1,
                                                                                        unsigned int seed = 42,
                                                                                        OrthoAlgorithm algo = OrthoAlgorithm::CGS2);

    /// @brief Random banded matrices with a prescribed log10 condition number.
    ///
    /// The resulting bandwidth is at most @p kd; for small `kd` the result may be
    /// diagonal.
    /// @param kd  maximum bandwidth, >= 0
    /// @throws batchlas::invalid_argument on a negative `kd`, and as random_with_log10_cond_metric()
    template <Backend B, typename T>
    BATCHLAS_API Matrix<T, MatrixFormat::Dense> random_banded_with_log10_cond_metric(Queue &ctx,
                                                                                      int n,
                                                                                      int kd,
                                                                                      float_t<T> log10_kappa,
                                                                                      NormType metric,
                                                                                      int batch_size = 1,
                                                                                      unsigned int seed = 42);

    /// @brief Random symmetric/Hermitian banded matrices with a prescribed log10 condition number.
    /// @param kd  maximum bandwidth, >= 0
    template <Backend B, typename T>
    BATCHLAS_API Matrix<T, MatrixFormat::Dense> random_hermitian_banded_with_log10_cond_metric(Queue &ctx,
                                                                                               int n,
                                                                                               int kd,
                                                                                               float_t<T> log10_kappa,
                                                                                               NormType metric,
                                                                                               int batch_size = 1,
                                                                                               unsigned int seed = 42);

    /// @brief Random tridiagonal matrices with a prescribed log10 condition number.
    ///
    /// The condition number is set through the diagonal spectrum; the off-diagonals
    /// are zero.
    template <Backend B, typename T>
    BATCHLAS_API Matrix<T, MatrixFormat::Dense> random_tridiagonal_with_log10_cond_metric(Queue &ctx,
                                                                                          int n,
                                                                                          float_t<T> log10_kappa,
                                                                                          NormType metric,
                                                                                          int batch_size = 1,
                                                                                          unsigned int seed = 42);

    /// @brief Random Hermitian tridiagonal matrices with a prescribed log10 condition number.
    template <Backend B, typename T>
    BATCHLAS_API Matrix<T, MatrixFormat::Dense> random_hermitian_tridiagonal_with_log10_cond_metric(Queue &ctx,
                                                                                                    int n,
                                                                                                    float_t<T> log10_kappa,
                                                                                                    NormType metric,
                                                                                                    int batch_size = 1,
                                                                                                    unsigned int seed = 42);

    /// @brief Batched plain transpose \f$B_i := A_i^T\f$ (no conjugation).
    /// @tparam T   float or double
    /// @param ctx  queue the kernel is enqueued on
    /// @param A    batch of m x n matrices; not modified
    /// @param B    batch of n x m matrices, same batch size; must not alias `A`
    /// @return     event of the enqueued kernel
    template <typename T, MatrixFormat MF>
    BATCHLAS_API Event transpose(Queue &ctx,
                                 const MatrixView<T, MF> &A,
                                 const MatrixView<T, MF> &B);

    /// @brief Batched plain transpose into a newly allocated matrix.
    /// @return a new n x m x batch matrix holding \f$A_i^T\f$; valid after the queue is waited on
    template <typename T, MatrixFormat MF>
    BATCHLAS_API Matrix<T, MF> transpose(Queue &ctx,
                                         const MatrixView<T, MF> &A);

    /// @}

    // norm/transpose are not Backend-templated, hence the _NB spelling. The
    // random_*_with_log10_cond_metric generators are deliberately absent: their T
    // is non-deduced, so either macro would be an overload that never applies.
    // evidence: docs/design/api-conventions.md#api-conventions-generators-without-a-dispatch-or-owning-overload
    BATCHLAS_ACCEPT_OWNING(cond)
    BATCHLAS_ACCEPT_OWNING(cond_buffer_size)
    BATCHLAS_ACCEPT_OWNING_NB(norm)
    BATCHLAS_ACCEPT_OWNING_NB(transpose)

    BATCHLAS_DISPATCH_ON_QUEUE(cond)
    BATCHLAS_DISPATCH_ON_QUEUE(cond_buffer_size)

}
