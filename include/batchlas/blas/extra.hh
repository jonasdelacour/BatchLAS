// SPDX-License-Identifier: MIT
#pragma once
#include <batchlas/export.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/blas/queue-dispatch.hh>

/// @file
/// @brief Matrix norms, condition numbers, conditioned test-matrix generators and transpose.
/// @ingroup api_extra

namespace batchlas
{

    /// @addtogroup api_extra
    /// @{

    /// @brief Per-item matrix norm, written into a caller-owned span.
    ///
    /// For each batch item \f$A_i\f$, `norms[i]` receives \f$\lVert A_i \rVert\f$ in the
    /// requested norm: Frobenius, One (max column sum), Inf (max row sum), Max
    /// (max absolute entry) or Spectral (largest \f$\lvert\lambda\rvert\f$).
    /// @tparam T      float, double, std::complex<float> or std::complex<double>
    /// @tparam MF     matrix format; only MatrixFormat::Dense is instantiated
    /// @param[in]  ctx       queue the kernels are enqueued on
    /// @param[in]  A         batch of \f$m \times n\f$ matrices; not modified
    /// @param[in]  norm_type which norm
    /// @param[out] norms     device-accessible output of at least `A.batch_size()` elements;
    ///                       `norms[i]` receives \f$\lVert A_i \rVert\f$
    /// @return          event of the last enqueued kernel; `norms` is valid after it (Spectral
    ///                  waits before returning, so `norms` is valid on return)
    /// @pre  NormType::Spectral requires square symmetric/Hermitian `A` (the lower
    ///       triangle is read).
    /// @throws batchlas::invalid_argument for Spectral on a non-square `A`
    /// @throws batchlas::unsupported for Spectral on a non-dense format or when no
    ///         backend with a vendor `syev` is compiled in
    /// @throws batchlas::NoRouteError for Spectral when the chosen backend's solver
    ///         library is not compiled in (the vendor `syev` is called directly;
    ///         Backend::SYCL calls the native `syev` instead)
    /// @throws batchlas::unsupported for double or complex<double> on a GPU without fp64
    template <typename T, MatrixFormat MF>
    BATCHLAS_API Event norm(Queue &ctx,
                           const MatrixView<T, MF> &A,
                           const NormType norm_type,
                           const Span<float_t<T>> norms);

    /// @brief Per-item matrix norm, returned in a new vector. Blocks until done.
    ///
    /// Same contract as the span overload, but allocates the result and waits on it
    /// before returning.
    /// @param[in] ctx       queue the kernels are enqueued on
    /// @param[in] A         batch of \f$m \times n\f$ matrices; not modified
    /// @param[in] norm_type which norm (default Frobenius)
    /// @return one real norm \f$\lVert A_i \rVert\f$ per batch item
    template <typename T, MatrixFormat MF>
    BATCHLAS_API UnifiedVector<float_t<T>> norm(Queue &ctx,
                                       const MatrixView<T, MF> &A,
                                       const NormType norm_type = NormType::Frobenius);

    /// @brief Per-item condition number, written into a caller-owned span.
    ///
    /// For norm types other than Spectral, `conds[i]` receives
    /// \f$\kappa(A_i) = \lVert A_i \rVert\,\lVert A_i^{-1} \rVert\f$, computed by explicit
    /// inversion. For NormType::Spectral, the ratio
    /// \f$\max_j \lvert\lambda_j\rvert / \min_j \lvert\lambda_j\rvert\f$ of the
    /// largest to the smallest eigenvalue magnitude of a symmetric/Hermitian \f$A_i\f$
    /// (lower triangle read; \f$\infty\f$ when an eigenvalue is exactly zero).
    /// @tparam B        backend; deduced from the Queue on the `cond(ctx, ...)` spelling
    /// @tparam T        float or double (real only)
    /// @tparam MF       MatrixFormat::Dense
    /// @param[in]  ctx       queue the kernels are enqueued on
    /// @param[in]  A         batch of \f$n \times n\f$ matrices; not modified
    /// @param[in]  norm_type which norm
    /// @param[out] conds     device-accessible output of at least `A.batch_size()` elements;
    ///                       `conds[i]` receives the condition number of \f$A_i\f$
    /// @param[out] workspace scratch of at least cond_buffer_size() bytes; must outlive the
    ///                       kernels; contents undefined on return
    /// @return          event of the last enqueued kernel
    /// @pre  `A` is square and nonsingular
    /// @throws batchlas::invalid_argument for Spectral on a non-square `A`
    /// @throws batchlas::unsupported for Spectral on a non-dense format
    /// @note The Spectral path calls the vendor `syev` directly, bypassing syev's
    ///       kernel selection, and throws batchlas::NoRouteError in a vendor-free
    ///       build; see @ref md_docs_2design_2known-defects (defect 2).
    ///       Backend::SYCL calls the native `syev` instead.
    template <Backend B, typename T, MatrixFormat MF>
        requires RealScalar<T>
    BATCHLAS_API Event cond(Queue &ctx,
                           const MatrixView<T, MF> &A,
                           const NormType norm_type,
                           const Span<T> conds,
                           const Span<std::byte> workspace);

    /// @brief Workspace bytes needed by the span overload of cond().
    ///
    /// Instantiated only for T in {float, double} with MatrixFormat::Dense: a complex T is a
    /// compile error (`requires RealScalar<T>`), another format a link error.
    /// @param[in] ctx       queue cond() will run on
    /// @param[in] A         batch of \f$n \times n\f$ matrices; only its shape is read
    /// @param[in] norm_type which norm cond() will use
    /// @return size in bytes for the `workspace` argument of cond()
    /// @ingroup api_extra_lowlevel
    template <Backend B, typename T, MatrixFormat MF>
        requires RealScalar<T>
    BATCHLAS_API size_t cond_buffer_size(Queue &ctx,
                                         const MatrixView<T, MF> &A,
                                         const NormType norm_type);

    /// @brief Per-item condition number, returned in a new vector. Blocks until done.
    ///
    /// Leases its workspace from the queue's arena and waits before returning.
    /// @param[in] ctx       queue the kernels are enqueued on
    /// @param[in] A         batch of \f$n \times n\f$ matrices; not modified
    /// @param[in] norm_type which norm
    /// @return one condition number per batch item
    template <Backend B, typename T, MatrixFormat MF>
        requires RealScalar<T>
    BATCHLAS_API UnifiedVector<T> cond(Queue &ctx,
                                       const MatrixView<T, MF> &A,
                                       const NormType norm_type);

    /// @brief Random dense matrices with a prescribed \f$\log_{10}\f$ condition number.
    ///
    /// Builds \f$A_i = U \Sigma V^H\f$ with random orthonormal factors and a singular
    /// spectrum chosen so that \f$\log_{10}\kappa(A_i)\f$ equals @p log10_kappa in the
    /// requested metric. Blocks until the result is written.
    /// @note On Backend::NETLIB the result is the diagonal \f$\Sigma\f$ itself, with
    ///       no random rotation.
    /// @tparam B  backend (not deducible; spell `f<Backend, T>(...)`)
    /// @tparam T  scalar type (not deducible)
    /// @param[in] ctx          queue the kernels are enqueued on
    /// @param[in] n            order \f$n > 0\f$ of each matrix
    /// @param[in] log10_kappa  target \f$\log_{10}\kappa \ge 0\f$
    ///                         (\f$\ge \log_{10} n\f$ for Frobenius)
    /// @param[in] metric       NormType::Spectral (\f$\kappa_2\f$) or NormType::Frobenius (\f$\kappa_F\f$)
    /// @param[in] batch_size   number of matrices, \f$> 0\f$
    /// @param[in] seed         RNG seed; equal seeds give equal batches
    /// @param[in] algo         orthogonalisation used to build the random factors. Keep
    ///                         the CGS2 default: Chol-QR and Householder can silently
    ///                         miss the requested \f$\kappa\f$.
    /// @return a new batch of `batch_size` \f$n \times n\f$ matrices
    /// @throws batchlas::invalid_argument on a non-positive size, a negative
    ///         `log10_kappa`, a metric other than Spectral/Frobenius, or, for
    ///         NormType::Frobenius, \f$\log_{10}\kappa < \log_{10} n\f$
    // evidence: docs/design/api-conventions.md#api-conventions-cond-generators-default-to-cgs2
    template <Backend B, typename T>
    BATCHLAS_API Matrix<T, MatrixFormat::Dense> random_with_log10_cond_metric(Queue &ctx,
                                                                               int n,
                                                                               float_t<T> log10_kappa,
                                                                               NormType metric,
                                                                               int batch_size = 1,
                                                                               unsigned int seed = 42,
                                                                               OrthoAlgorithm algo = OrthoAlgorithm::CGS2);

    /// @brief Random symmetric/Hermitian matrices with a prescribed \f$\log_{10}\f$ condition number.
    ///
    /// As random_with_log10_cond_metric(), with \f$A_i = Q \Lambda Q^H\f$.
    /// @param[in] ctx          queue the kernels are enqueued on
    /// @param[in] n            order \f$n > 0\f$ of each matrix
    /// @param[in] log10_kappa  target \f$\log_{10}\kappa \ge 0\f$
    ///                         (\f$\ge \log_{10} n\f$ for Frobenius)
    /// @param[in] metric       NormType::Spectral (\f$\kappa_2\f$) or NormType::Frobenius (\f$\kappa_F\f$)
    /// @param[in] batch_size   number of matrices, \f$> 0\f$
    /// @param[in] seed         RNG seed; equal seeds give equal batches
    /// @param[in] algo         orthogonalisation used to build \f$Q\f$; keep the CGS2 default
    /// @return a new batch of `batch_size` symmetric/Hermitian \f$n \times n\f$ matrices
    template <Backend B, typename T>
    BATCHLAS_API Matrix<T, MatrixFormat::Dense> random_hermitian_with_log10_cond_metric(Queue &ctx,
                                                                                        int n,
                                                                                        float_t<T> log10_kappa,
                                                                                        NormType metric,
                                                                                        int batch_size = 1,
                                                                                        unsigned int seed = 42,
                                                                                        OrthoAlgorithm algo = OrthoAlgorithm::CGS2);

    /// @brief Random banded matrices with a prescribed \f$\log_{10}\f$ condition number.
    ///
    /// The resulting bandwidth is at most @p kd; for small `kd` the result may be
    /// diagonal.
    /// @param[in] ctx          queue the kernels are enqueued on
    /// @param[in] n            order \f$n > 0\f$ of each matrix
    /// @param[in] kd           maximum bandwidth, \f$\mathit{kd} \ge 0\f$; \f$\mathit{kd} \ge n - 1\f$
    ///                         delegates to random_with_log10_cond_metric() (dense). On
    ///                         Backend::NETLIB every result is diagonal.
    /// @param[in] log10_kappa  target \f$\log_{10}\kappa \ge 0\f$
    ///                         (\f$\ge \log_{10} n\f$ for Frobenius)
    /// @param[in] metric       NormType::Spectral (\f$\kappa_2\f$) or NormType::Frobenius (\f$\kappa_F\f$)
    /// @param[in] batch_size   number of matrices, \f$> 0\f$
    /// @param[in] seed         RNG seed; equal seeds give equal batches
    /// @return a new batch of `batch_size` \f$n \times n\f$ matrices, stored dense
    /// @throws batchlas::invalid_argument on a negative `kd`, and as random_with_log10_cond_metric()
    template <Backend B, typename T>
    BATCHLAS_API Matrix<T, MatrixFormat::Dense> random_banded_with_log10_cond_metric(Queue &ctx,
                                                                                      int n,
                                                                                      int kd,
                                                                                      float_t<T> log10_kappa,
                                                                                      NormType metric,
                                                                                      int batch_size = 1,
                                                                                      unsigned int seed = 42);

    /// @brief Random symmetric/Hermitian banded matrices with a prescribed \f$\log_{10}\f$ condition number.
    /// @param[in] ctx          queue the kernels are enqueued on
    /// @param[in] n            order \f$n > 0\f$ of each matrix
    /// @param[in] kd           maximum bandwidth, \f$\mathit{kd} \ge 0\f$; \f$\mathit{kd} \ge n - 1\f$
    ///                         delegates to random_hermitian_with_log10_cond_metric() (dense).
    ///                         On Backend::NETLIB every result is diagonal.
    /// @param[in] log10_kappa  target \f$\log_{10}\kappa \ge 0\f$
    ///                         (\f$\ge \log_{10} n\f$ for Frobenius)
    /// @param[in] metric       NormType::Spectral (\f$\kappa_2\f$) or NormType::Frobenius (\f$\kappa_F\f$)
    /// @param[in] batch_size   number of matrices, \f$> 0\f$
    /// @param[in] seed         RNG seed; equal seeds give equal batches
    /// @return a new batch of `batch_size` symmetric/Hermitian \f$n \times n\f$ matrices, stored dense
    /// @throws batchlas::invalid_argument on a negative `kd`, and as random_with_log10_cond_metric()
    template <Backend B, typename T>
    BATCHLAS_API Matrix<T, MatrixFormat::Dense> random_hermitian_banded_with_log10_cond_metric(Queue &ctx,
                                                                                               int n,
                                                                                               int kd,
                                                                                               float_t<T> log10_kappa,
                                                                                               NormType metric,
                                                                                               int batch_size = 1,
                                                                                               unsigned int seed = 42);

    /// @brief Random tridiagonal matrices with a prescribed \f$\log_{10}\f$ condition number.
    ///
    /// The condition number is set through the diagonal spectrum; the off-diagonals
    /// are zero.
    /// @param[in] ctx          queue the kernels are enqueued on
    /// @param[in] n            order \f$n > 0\f$ of each matrix
    /// @param[in] log10_kappa  target \f$\log_{10}\kappa \ge 0\f$
    ///                         (\f$\ge \log_{10} n\f$ for Frobenius)
    /// @param[in] metric       NormType::Spectral (\f$\kappa_2\f$) or NormType::Frobenius (\f$\kappa_F\f$)
    /// @param[in] batch_size   number of matrices, \f$> 0\f$
    /// @param[in] seed         unused: the result is deterministic
    /// @return a new batch of `batch_size` \f$n \times n\f$ matrices, stored dense
    /// @throws batchlas::invalid_argument as random_with_log10_cond_metric()
    template <Backend B, typename T>
    BATCHLAS_API Matrix<T, MatrixFormat::Dense> random_tridiagonal_with_log10_cond_metric(Queue &ctx,
                                                                                          int n,
                                                                                          float_t<T> log10_kappa,
                                                                                          NormType metric,
                                                                                          int batch_size = 1,
                                                                                          unsigned int seed = 42);

    /// @brief Random Hermitian tridiagonal matrices with a prescribed \f$\log_{10}\f$ condition number.
    ///
    /// Builds \f$Q \Lambda Q^H\f$ and reduces it to tridiagonal form, so the spectrum
    /// (not just the diagonal) sets the condition number. On Backend::NETLIB the
    /// result is the diagonal \f$\Lambda\f$ itself.
    /// @param[in] ctx          queue the kernels are enqueued on
    /// @param[in] n            order \f$n > 0\f$ of each matrix
    /// @param[in] log10_kappa  target \f$\log_{10}\kappa \ge 0\f$
    ///                         (\f$\ge \log_{10} n\f$ for Frobenius)
    /// @param[in] metric       NormType::Spectral (\f$\kappa_2\f$) or NormType::Frobenius (\f$\kappa_F\f$)
    /// @param[in] batch_size   number of matrices, \f$> 0\f$
    /// @param[in] seed         RNG seed; equal seeds give equal batches
    /// @return a new batch of `batch_size` Hermitian tridiagonal \f$n \times n\f$ matrices, stored dense
    /// @throws batchlas::invalid_argument as random_with_log10_cond_metric()
    template <Backend B, typename T>
    BATCHLAS_API Matrix<T, MatrixFormat::Dense> random_hermitian_tridiagonal_with_log10_cond_metric(Queue &ctx,
                                                                                                    int n,
                                                                                                    float_t<T> log10_kappa,
                                                                                                    NormType metric,
                                                                                                    int batch_size = 1,
                                                                                                    unsigned int seed = 42);

    /// @brief Batched plain transpose \f$B_i := A_i^T\f$ (no conjugation).
    /// @tparam T   float or double
    /// @param[in]  ctx  queue the kernel is enqueued on
    /// @param[in]  A    batch of \f$m \times n\f$ matrices; not modified
    /// @param[out] B    batch of \f$n \times m\f$ matrices, same batch size; must not alias `A`;
    ///                  receives \f$A_i^T\f$
    /// @return     event of the enqueued kernel
    template <typename T, MatrixFormat MF>
    BATCHLAS_API Event transpose(Queue &ctx,
                                 const MatrixView<T, MF> &A,
                                 const MatrixView<T, MF> &B);

    /// @brief Batched plain transpose into a newly allocated matrix.
    /// @param[in] ctx  queue the kernel is enqueued on
    /// @param[in] A    batch of \f$m \times n\f$ matrices; not modified
    /// @return a new batch of \f$n \times m\f$ matrices holding \f$A_i^T\f$; valid after the
    ///         queue is waited on
    template <typename T, MatrixFormat MF>
    BATCHLAS_API Matrix<T, MF> transpose(Queue &ctx,
                                         const MatrixView<T, MF> &A);

    /// @}

    // The random_*_with_log10_cond_metric generators are deliberately absent: their T
    // is non-deduced, so the macro would be an overload that never applies.
    // evidence: docs/design/api-conventions.md#api-conventions-generators-without-a-dispatch-overload

    BATCHLAS_DISPATCH_ON_QUEUE(cond)
    BATCHLAS_DISPATCH_ON_QUEUE(cond_buffer_size)

}
