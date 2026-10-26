#pragma once

/// @file
/// @brief Eigenvector continuity (syev_align): rotate a new eigenbasis onto a previous one.
/// @ingroup api_eigen

#include <batchlas/export.hh>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/queue-dispatch.hh>

namespace batchlas {

// Signature aliases for explicit instantiation (BATCHLAS_INSTANTIATE in
// src/util/template-instantiations.hh); keep in sync with the declarations below.
namespace sig {
template <typename T>
using syev_align = Event(Queue&,
                         const MatrixView<T, MatrixFormat::Dense>&,
                         Span<typename base_type<T>::type>,
                         const MatrixView<T, MatrixFormat::Dense>&,
                         typename base_type<T>::type,
                         Span<std::byte>);

template <typename T>
using syev_align_buffer_size = size_t(Queue&,
                                      const MatrixView<T, MatrixFormat::Dense>&,
                                      Span<typename base_type<T>::type>,
                                      const MatrixView<T, MatrixFormat::Dense>&);
}  // namespace sig

/// @brief Validates the arguments of syev_align() and syev_align_buffer_size().
///
/// Checks that @p V_old and @p V share their shape and batch size, that `V.cols() <= V.rows()`,
/// that neither is heterogeneous, that `V.rows() * V.cols()` fits in an int, and that
/// @p eigenvalues holds `V.cols() * batch` values.
/// @throws batchlas::invalid_argument if any check fails
/// @ingroup api_eigen_lowlevel
template <typename T>
inline void syev_align_validate_params(const MatrixView<T, MatrixFormat::Dense>& V_old,
                                       Span<typename base_type<T>::type> eigenvalues,
                                       const MatrixView<T, MatrixFormat::Dense>& V) {
    if (V_old.rows() != V.rows() || V_old.cols() != V.cols() || V_old.batch_size() != V.batch_size()) {
        throw batchlas::invalid_argument(
            "syev_align: V_old and V must have the same shape and batch size, got " +
            std::to_string(V_old.rows()) + "x" + std::to_string(V_old.cols()) + " (batch " +
            std::to_string(V_old.batch_size()) + ") and " + std::to_string(V.rows()) + "x" +
            std::to_string(V.cols()) + " (batch " + std::to_string(V.batch_size()) + ")");
    }
    if (V.cols() > V.rows()) {
        throw batchlas::invalid_argument("syev_align: V has more columns (" + std::to_string(V.cols()) +
                                         ") than rows (" + std::to_string(V.rows()) + ")");
    }
    if (V_old.is_heterogeneous() || V.is_heterogeneous()) {
        throw batchlas::invalid_argument("syev_align: heterogeneous batches are not supported");
    }
    if (int64_t(V.rows()) * int64_t(V.cols()) > int64_t(std::numeric_limits<int>::max())) {
        throw batchlas::invalid_argument("syev_align: V.rows() * V.cols() = " +
                                         std::to_string(int64_t(V.rows()) * int64_t(V.cols())) +
                                         " exceeds the int range of the workspace views");
    }
    const size_t need = size_t(V.cols()) * size_t(V.batch_size());
    if (eigenvalues.size() < need) {
        throw batchlas::invalid_argument("syev_align: eigenvalues holds " + std::to_string(eigenvalues.size()) +
                                         " values, need V.cols() * batch = " + std::to_string(need));
    }
}

/// @brief Aligns a batch of new eigenvectors with a previous basis, cluster by cluster.
///
/// For every batch item, with \f$ m = \f$ `V.cols()` and \f$ M = V_{old}^H V \f$ (one gemm()):
/// columns j-1 and j share a cluster when
/// \f$ \lambda_j - \lambda_{j-1} \le \mathrm{tol} \cdot \max(|\lambda_0|, |\lambda_{m-1}|) \f$
/// (chained, so a cluster is a maximal run of such neighbours). Each cluster's columns
/// \f$ V_c \f$ are replaced by \f$ V_c R \f$, where \f$ R = W U^H \f$ is the orthogonal
/// (complex: unitary) Procrustes solution from the SVD \f$ M_{cc} = U \Sigma W^H \f$ of the
/// cluster's diagonal block, computed by a one-sided Jacobi SVD in the kernel. Then
/// \f$ V_{old,c}^H V_c \f$ is Hermitian positive semidefinite, so every diagonal entry of
/// \f$ V_{old}^H V \f$ is real and \f$ \ge 0 \f$: for a single column this is the sign
/// (complex: phase) fix. When \f$ V_c \f$ spans the same subspace as \f$ V_{old,c} \f$ the
/// result is \f$ V_{old,c} \f$ up to rounding, whatever basis of that subspace \f$ V_c \f$ was.
/// Singular directions of a rank-deficient \f$ M_{cc} \f$ are completed to an orthonormal
/// basis, so \f$ V_c R \f$ stays orthonormal.
///
/// Asynchronous: V is readable after the returned event is waited on.
/// @pre @p eigenvalues ascending per item (syev()'s order); column j of V pairs with value j
/// @tparam B  backend
/// @tparam T  float, double, std::complex<float> or std::complex<double>
/// @param ctx         queue the work is enqueued on
/// @param V_old       batch of n x m previous eigenvectors; read only
/// @param eigenvalues m real eigenvalues per item, packed (`V.cols() * batch` values)
/// @param V           batch of n x m new eigenvectors; overwritten with the aligned ones
/// @param tol         cluster tolerance relative to \f$ \max|\lambda| \f$ of the item; 0 clusters
///                    only exactly equal values. No absolute floor, so the rule is scale invariant.
///                    When the values passed do not reach the scale of \f$ \|A\| \f$ (a subset
///                    of eigenpairs near zero), pass \f$ \mathrm{tol} \cdot \|A\| / \max|\lambda| \f$;
///                    see @ref design_syev_align
/// @param workspace   at least syev_align_buffer_size() bytes
/// @return event of the last enqueued kernel
/// @throws batchlas::invalid_argument if syev_align_validate_params() rejects the call or @p tol < 0
/// @throws batchlas::workspace_error if @p workspace is too short
/// @ingroup api_eigen
template <Backend B, typename T>
BATCHLAS_API Event syev_align(Queue& ctx,
                              const MatrixView<T, MatrixFormat::Dense>& V_old,
                              Span<typename base_type<T>::type> eigenvalues,
                              const MatrixView<T, MatrixFormat::Dense>& V,
                              typename base_type<T>::type tol,
                              Span<std::byte> workspace);

/// @brief Workspace, in bytes, that syev_align() needs: about \f$ (m^2 + n m + m) \f$ scalars per item.
/// @throws batchlas::invalid_argument if syev_align_validate_params() rejects the operands
/// @ingroup api_eigen_lowlevel
template <Backend B, typename T>
BATCHLAS_API size_t syev_align_buffer_size(Queue& ctx,
                                           const MatrixView<T, MatrixFormat::Dense>& V_old,
                                           Span<typename base_type<T>::type> eigenvalues,
                                           const MatrixView<T, MatrixFormat::Dense>& V);

BATCHLAS_DISPATCH_ON_QUEUE(syev_align)
BATCHLAS_DISPATCH_ON_QUEUE(syev_align_buffer_size)

}  // namespace batchlas
