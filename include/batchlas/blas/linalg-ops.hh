#pragma once

#include <batchlas/export.hh>
#include <algorithm>
#include <stdexcept>
#include <utility>

#include <batchlas/blas/enums.hh>
// Needed here: the qualified ids in the forwards below bind at definition context.
#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/extra.hh>
// Called directly by svd(), solve() and solve_spd(); never rely on them transitively.
#include <batchlas/blas/functions/gesvd.hh>
#include <batchlas/blas/functions/gesv.hh>
#include <batchlas/blas/functions/posv.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/options.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>

/// @file
/// @brief The batchlas::linalg convenience layer: elementwise operations and value-returning wrappers.
/// @ingroup linalg

/**
 * @addtogroup linalg
 * @details
 * Free functions only, no operator overloads. Membership rule: value-returning,
 * backend from the Queue, workspace from the Queue's arena => `linalg::`;
 * out-parameter, workspace yours => `batchlas::`.
 *
 * Every function here **enqueues and returns without waiting**, including the
 * value-returning ones: a returned matrix is readable only after `ctx.wait()`.
 * The exceptions are norm(), cond() and svd(), which wait. Inputs are never
 * modified, except by the `_into` forms (which write their last matrix argument)
 * and scale(). A value-returning call allocates its result, so prefer the `_into`
 * forms in an inner loop. User guide: @ref md_docs_2cpp-api (the `linalg` section);
 * design notes: @ref design_api_conventions.
 */

namespace batchlas::linalg {

/// @addtogroup linalg
/// @{

// ---- elementwise -----------------------------------------------------------

/// @brief Elementwise operation selected by elementwise_into().
// BATCHLAS_API is load-bearing: an enum used as a template argument must be
// exported or every instantiation over it is hidden (see blas/enums.hh).
// evidence: docs/design/symbol-visibility.md#symbol-visibility-binaryop-is-a-template-argument-enum-too
enum class BATCHLAS_API BinaryOp { Add, Subtract, Multiply, Divide };

// Signature aliases for explicit instantiation; see BATCHLAS_INSTANTIATE in
// src/util/template-instantiations.hh. Keep in sync with the declarations below.
namespace sig {
template <typename T, BinaryOp Op>
using elementwise_into = Event(Queue&,
                               const MatrixView<T, MatrixFormat::Dense>&,
                               const MatrixView<T, MatrixFormat::Dense>&,
                               const MatrixView<T, MatrixFormat::Dense>&);

template <typename T>
using axpby_into = Event(Queue&,
                         T,
                         const MatrixView<T, MatrixFormat::Dense>&,
                         T,
                         const MatrixView<T, MatrixFormat::Dense>&,
                         const MatrixView<T, MatrixFormat::Dense>&);

template <typename T>
using scale = Event(Queue&, const MatrixView<T, MatrixFormat::Dense>&, T);

template <typename T>
using triangular_mask_into = Event(Queue&,
                                   const MatrixView<T, MatrixFormat::Dense>&,
                                   const MatrixView<T, MatrixFormat::Dense>&,
                                   Uplo,
                                   int64_t);
}  // namespace sig

/// @brief Elementwise \f$C := A \circ B\f$ for \f$\circ\f$ = Op, per batch item.
/// @tparam T   scalar type
/// @tparam Op  Add, Subtract, Multiply (Hadamard) or Divide
/// @param ctx  queue the kernel is enqueued on
/// @param A    first operand
/// @param B    second operand, same shape and batch size as `A`
/// @param C    output of the same shape; may alias `A` or `B`
/// @return     event of the enqueued kernel
template <typename T, BinaryOp Op>
BATCHLAS_API Event elementwise_into(Queue& ctx,
                                    const MatrixView<T, MatrixFormat::Dense>& A,
                                    const MatrixView<T, MatrixFormat::Dense>& B,
                                    const MatrixView<T, MatrixFormat::Dense>& C);

/// @brief \f$C := \alpha A + \beta B\f$, elementwise, per batch item.
/// @param ctx    queue the kernel is enqueued on
/// @param alpha  scale of `A`
/// @param A      first operand
/// @param beta   scale of `B`
/// @param B      second operand, same shape and batch size as `A`
/// @param C      output of the same shape; may alias `A` or `B`
/// @return       event of the enqueued kernel
template <typename T>
BATCHLAS_API Event axpby_into(Queue& ctx,
                              T alpha,
                              const MatrixView<T, MatrixFormat::Dense>& A,
                              T beta,
                              const MatrixView<T, MatrixFormat::Dense>& B,
                              const MatrixView<T, MatrixFormat::Dense>& C);

/// @brief In-place scaling \f$A := \alpha A\f$.
/// @return event of the enqueued kernel
template <typename T>
BATCHLAS_API Event scale(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, T alpha);

/// @brief \f$C :=\f$ A with everything outside the requested triangle zeroed.
/// @param ctx   queue the kernel is enqueued on
/// @param A     input, m x n per item
/// @param C     output of the same shape; may alias `A`
/// @param uplo  Upper keeps the part on and above diagonal `k`, Lower on and below
/// @param k     diagonal offset, as NumPy `triu`/`tril`: 0 is the main diagonal,
///              > 0 toward the upper right, < 0 toward the lower left
/// @return      event of the enqueued kernel
template <typename T>
BATCHLAS_API Event triangular_mask_into(Queue& ctx,
                                        const MatrixView<T, MatrixFormat::Dense>& A,
                                        const MatrixView<T, MatrixFormat::Dense>& C,
                                        Uplo uplo,
                                        int64_t k = 0);

/// @brief \f$C := A + B\f$ elementwise; see elementwise_into().
template <typename T>
inline Event add_into(Queue& ctx,
                      const MatrixView<T, MatrixFormat::Dense>& A,
                      const MatrixView<T, MatrixFormat::Dense>& B,
                      const MatrixView<T, MatrixFormat::Dense>& C) {
    return elementwise_into<T, BinaryOp::Add>(ctx, A, B, C);
}

/// @brief \f$C := A - B\f$ elementwise; see elementwise_into().
template <typename T>
inline Event subtract_into(Queue& ctx,
                           const MatrixView<T, MatrixFormat::Dense>& A,
                           const MatrixView<T, MatrixFormat::Dense>& B,
                           const MatrixView<T, MatrixFormat::Dense>& C) {
    return elementwise_into<T, BinaryOp::Subtract>(ctx, A, B, C);
}

/// @brief \f$C := A \odot B\f$ elementwise (Hadamard), not the matrix product; see matmul().
template <typename T>
inline Event multiply_into(Queue& ctx,
                           const MatrixView<T, MatrixFormat::Dense>& A,
                           const MatrixView<T, MatrixFormat::Dense>& B,
                           const MatrixView<T, MatrixFormat::Dense>& C) {
    return elementwise_into<T, BinaryOp::Multiply>(ctx, A, B, C);
}

/// @brief \f$C := A \oslash B\f$ elementwise; see elementwise_into().
template <typename T>
inline Event divide_into(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         const MatrixView<T, MatrixFormat::Dense>& B,
                         const MatrixView<T, MatrixFormat::Dense>& C) {
    return elementwise_into<T, BinaryOp::Divide>(ctx, A, B, C);
}

/// @}

namespace detail {
template <typename T>
inline Matrix<T, MatrixFormat::Dense> like(const MatrixView<T, MatrixFormat::Dense>& A) {
    return Matrix<T, MatrixFormat::Dense>(A.rows(), A.cols(), A.batch_size());
}
}  // namespace detail

/// @addtogroup linalg
/// @{

/// @brief Returns a new matrix \f$A + B\f$ (elementwise).
template <typename T>
inline Matrix<T, MatrixFormat::Dense> add(Queue& ctx,
                                          const MatrixView<T, MatrixFormat::Dense>& A,
                                          const MatrixView<T, MatrixFormat::Dense>& B) {
    auto C = detail::like(A);
    // (void) on an Event is deliberate: the in-order Queue already orders the next call.
    (void)add_into<T>(ctx, A, B, C.view());
    return C;
}

/// @brief Returns a new matrix \f$A - B\f$ (elementwise).
template <typename T>
inline Matrix<T, MatrixFormat::Dense> subtract(Queue& ctx,
                                               const MatrixView<T, MatrixFormat::Dense>& A,
                                               const MatrixView<T, MatrixFormat::Dense>& B) {
    auto C = detail::like(A);
    (void)subtract_into<T>(ctx, A, B, C.view());
    return C;
}

/// @brief Returns a new matrix \f$A \odot B\f$ (Hadamard, not matmul()).
template <typename T>
inline Matrix<T, MatrixFormat::Dense> multiply(Queue& ctx,
                                               const MatrixView<T, MatrixFormat::Dense>& A,
                                               const MatrixView<T, MatrixFormat::Dense>& B) {
    auto C = detail::like(A);
    (void)multiply_into<T>(ctx, A, B, C.view());
    return C;
}

/// @brief Returns a new matrix \f$A \oslash B\f$ (elementwise).
template <typename T>
inline Matrix<T, MatrixFormat::Dense> divide(Queue& ctx,
                                             const MatrixView<T, MatrixFormat::Dense>& A,
                                             const MatrixView<T, MatrixFormat::Dense>& B) {
    auto C = detail::like(A);
    (void)divide_into<T>(ctx, A, B, C.view());
    return C;
}

/// @brief Returns a new matrix \f$\alpha A\f$; `A` is not modified (scale() works in place).
template <typename T>
inline Matrix<T, MatrixFormat::Dense> scaled(Queue& ctx,
                                             const MatrixView<T, MatrixFormat::Dense>& A,
                                             T alpha) {
    auto C = detail::like(A);
    (void)axpby_into<T>(ctx, alpha, A, T(0), A, C.view());
    return C;
}

// ---- value-returning wrappers ---------------------------------------------

/// @brief Options for matmul(). Deliberately has no `beta`: C is freshly allocated.
template <typename T>
struct MatmulOptions {
    T alpha = T(1);                                          ///< scale of the product
    Transpose transA = Transpose::NoTrans;                   ///< op(A)
    Transpose transB = Transpose::NoTrans;                   ///< op(B)
    ComputePrecision precision = ComputePrecision::Default;  ///< internal compute type
};

/// @brief Returns a new matrix \f$C = \alpha\,\mathrm{op}(A)\,\mathrm{op}(B)\f$.
/// @param ctx   queue; supplies the backend
/// @param A     `op(A)` is m x k per item
/// @param B     `op(B)` is k x n per item, same batch size
/// @param opts  alpha, transA, transB, precision
/// @return      m x n x batch matrix, fully written once the queue is waited on
template <typename T>
inline Matrix<T, MatrixFormat::Dense> matmul(Queue& ctx,
                                             const MatrixView<T, MatrixFormat::Dense>& A,
                                             const MatrixView<T, MatrixFormat::Dense>& B,
                                             const MatmulOptions<T>& opts = {}) {
    const bool ta = opts.transA != Transpose::NoTrans;
    const bool tb = opts.transB != Transpose::NoTrans;
    const auto m = ta ? A.cols() : A.rows();
    const auto n = tb ? B.rows() : B.cols();
    Matrix<T, MatrixFormat::Dense> C(m, n, A.batch_size());
    (void)gemm(ctx, A, B, C.view(),
         GemmOptions<T>{.alpha = opts.alpha,
                        .beta = T(0),  // C is fresh
                        .transA = opts.transA,
                        .transB = opts.transB,
                        .precision = opts.precision});
    return C;
}

/// @brief Default triangle of the linalg wrappers that read one.
inline constexpr Uplo kDefaultUplo = Uplo::Lower;

/// @brief Returns the Cholesky factor of a Hermitian positive-definite A; A is not modified.
/// @param ctx   queue; supplies the backend and the arena
/// @param A     n x n per item; the `uplo` triangle is read
/// @param uplo  Lower gives \f$L\f$ with \f$A = LL^H\f$, Upper gives \f$U\f$ with \f$A = U^HU\f$
/// @return      new matrix holding the factor in the `uplo` triangle; the other
///              triangle holds a copy of A's
/// @note  No status is reported; a non-positive-definite item yields an undefined factor.
template <typename T>
inline Matrix<T, MatrixFormat::Dense> cholesky(Queue& ctx,
                                               const MatrixView<T, MatrixFormat::Dense>& A,
                                               Uplo uplo = kDefaultUplo) {
    auto L = detail::like(A);
    (void)MatrixView<T, MatrixFormat::Dense>::copy(ctx, L.view(), A);
    (void)potrf(ctx, L.view(), {.uplo = uplo});
    return L;
}

/// @brief Returns the eigenvalues of a symmetric/Hermitian A, ascending per item; A is not modified.
/// @param ctx   queue; supplies the backend and the arena
/// @param A     n x n per item; the `uplo` triangle is read
/// @param uplo  triangle of A that is read
/// @return      real vector of `n * batch_size` values, item b at offset `b * n`
template <typename T>
inline UnifiedVector<typename base_type<T>::type> eigvalsh(Queue& ctx,
                                                           const MatrixView<T, MatrixFormat::Dense>& A,
                                                           Uplo uplo = kDefaultUplo) {
    UnifiedVector<typename base_type<T>::type> W(static_cast<size_t>(A.rows()) *
                                                 static_cast<size_t>(A.batch_size()));
    auto work = detail::like(A);
    (void)MatrixView<T, MatrixFormat::Dense>::copy(ctx, work.view(), A);
    (void)syev(ctx, work.view(), W.to_span(), {.jobz = JobType::NoEigenVectors, .uplo = uplo});
    return W;
}

/// @brief Result of eigh().
template <typename T>
struct Eigh {
    UnifiedVector<typename base_type<T>::type> values;  ///< ascending, `n` per batch item
    Matrix<T, MatrixFormat::Dense> vectors;             ///< eigenvectors as columns, n x n per item
    /// Per-item convergence status: 0 converged, > 0 LAPACK-like. Always filled;
    /// check it, since a non-converged item otherwise looks like a converged one.
    UnifiedVector<int32_t> info;
};

/// @brief Eigenvalues and eigenvectors of a symmetric/Hermitian A; A is not modified.
/// @param ctx   queue; supplies the backend and the arena
/// @param A     n x n per item; the `uplo` triangle is read
/// @param uplo  triangle of A that is read
/// @return      values, vectors and per-item info, valid after the queue is waited on
/// @throws batchlas::invalid_argument if `A` is not square
template <typename T>
inline Eigh<T> eigh(Queue& ctx,
                    const MatrixView<T, MatrixFormat::Dense>& A,
                    Uplo uplo = kDefaultUplo) {
    UnifiedVector<typename base_type<T>::type> W(static_cast<size_t>(A.rows()) *
                                                 static_cast<size_t>(A.batch_size()));
    UnifiedVector<int32_t> info(static_cast<size_t>(A.batch_size()));
    // The positional syev below does not check squareness. Qualified: `detail::`
    // here would find batchlas::linalg::detail.
    ::batchlas::detail::require_square("eigh", "A", A);
    auto V = detail::like(A);
    (void)MatrixView<T, MatrixFormat::Dense>::copy(ctx, V.view(), A);
    // Positional because SyevOptions has no `info` field.
    auto lease = ctx.workspace(
        syev_buffer_size(ctx, V.view(), W.to_span(), JobType::EigenVectors, uplo));
    (void)syev(ctx, V.view(), W.to_span(), JobType::EigenVectors, uplo, lease.span(),
               info.to_span());
    // No wait needed: `info` is moved into the result (unlike svd()'s local scratch).
    return Eigh<T>{std::move(W), std::move(V), std::move(info)};
}

/// @brief Solves \f$\mathrm{op}(A_i)X_i = B_i\f$ by LU with partial pivoting; returns X.
///
/// Neither A nor B is modified; pivots and workspace come from the arena. No
/// status is reported: a singular or near-singular item yields a plausible-looking
/// but wrong X, so check the residual where conditioning is not known.
/// @param ctx    queue; supplies the backend and the arena
/// @param A      n x n per item
/// @param B      n x nrhs per item, same batch size
/// @param trans  op(A)
/// @return       new n x nrhs x batch matrix
/// @throws batchlas::exception (NoTrans) when an extent (n, nrhs or batch) is below
///         1: no route serves a degenerate shape
template <typename T>
inline Matrix<T, MatrixFormat::Dense> solve(Queue& ctx,
                                            const MatrixView<T, MatrixFormat::Dense>& A,
                                            const MatrixView<T, MatrixFormat::Dense>& B,
                                            Transpose trans = Transpose::NoTrans) {
    auto LU = detail::like(A);
    (void)MatrixView<T, MatrixFormat::Dense>::copy(ctx, LU.view(), A);

    Matrix<T, MatrixFormat::Dense> X(B.rows(), B.cols(), B.batch_size());
    (void)MatrixView<T, MatrixFormat::Dense>::copy(ctx, X.view(), B);

    const size_t n_pivots = static_cast<size_t>(A.rows()) * static_cast<size_t>(A.batch_size());
    auto pivot_bytes = ctx.workspace(n_pivots * sizeof(int64_t));
    Span<int64_t> pivots(reinterpret_cast<int64_t*>(pivot_bytes.data()), n_pivots);

    // Not stylistic, do not flatten: gesv has no Transpose parameter. No shape gate
    // here either; gesv routes itself. Degenerate extents throw (deliberate).
    // evidence: docs/design/api-conventions.md#api-conventions-linalgsolve-keeps-its-transpose-branch
    if (trans == Transpose::NoTrans) {
        // Released before the pivot lease (reverse order): the arena reuses it at once.
        auto lease = ctx.workspace(gesv_buffer_size(ctx, LU.view(), X.view()));
        (void)gesv(ctx, LU.view(), X.view(), pivots, lease.span(), Span<int32_t>{});
        return X;
    }

    (void)getrf(ctx, LU.view(), pivots);
    (void)getrs(ctx, LU.view(), X.view(), pivots, {.trans = trans});
    return X;
}

/// @brief Solves \f$A_iX_i = B_i\f$ for Hermitian positive-definite A by Cholesky; returns X.
///
/// About half the arithmetic of solve(), which stays the safe default: the SPD
/// claim is the caller's. A matrix that is not positive definite gives an
/// undefined X for that item, as LAPACK `?POSV`, and nothing is reported.
/// Neither A nor B is modified; the workspace comes from the arena.
/// @param ctx   queue; supplies the backend and the arena
/// @param A     n x n per item; the `uplo` triangle is read
/// @param B     n x nrhs per item
/// @param uplo  triangle of A that is read
/// @return      new n x nrhs x batch matrix
/// @throws batchlas::exception when an extent (n, nrhs or batch) is below 1
template <typename T>
inline Matrix<T, MatrixFormat::Dense> solve_spd(Queue& ctx,
                                                const MatrixView<T, MatrixFormat::Dense>& A,
                                                const MatrixView<T, MatrixFormat::Dense>& B,
                                                Uplo uplo = Uplo::Lower) {
    auto F = detail::like(A);
    (void)MatrixView<T, MatrixFormat::Dense>::copy(ctx, F.view(), A);

    Matrix<T, MatrixFormat::Dense> X(B.rows(), B.cols(), B.batch_size());
    (void)MatrixView<T, MatrixFormat::Dense>::copy(ctx, X.view(), B);

    auto lease = ctx.workspace(posv_buffer_size(ctx, F.view(), X.view(), uplo));
    (void)posv(ctx, F.view(), X.view(), uplo, lease.span(), Span<int32_t>{});
    return X;
}

/// @brief Returns a copy of A with everything below diagonal `k` zeroed (NumPy `triu`).
template <typename T>
inline Matrix<T, MatrixFormat::Dense> triu(Queue& ctx,
                                           const MatrixView<T, MatrixFormat::Dense>& A,
                                           int64_t k = 0) {
    auto C = detail::like(A);
    (void)triangular_mask_into<T>(ctx, A, C.view(), Uplo::Upper, k);
    return C;
}

/// @brief Returns a copy of A with everything above diagonal `k` zeroed (NumPy `tril`).
template <typename T>
inline Matrix<T, MatrixFormat::Dense> tril(Queue& ctx,
                                           const MatrixView<T, MatrixFormat::Dense>& A,
                                           int64_t k = 0) {
    auto C = detail::like(A);
    (void)triangular_mask_into<T>(ctx, A, C.view(), Uplo::Lower, k);
    return C;
}

// ---- forwarding aliases ----------------------------------------------------
// Every call below must stay qualified with ::batchlas::: linalg is nested inside
// batchlas, so `return inv(ctx, A);` here is infinite recursion.

/// @brief Returns \f$A^{-1}\f$ as a new matrix; A is not modified.
/// @pre `A` is square
template <typename T>
inline Matrix<T, MatrixFormat::Dense> inv(Queue& ctx,
                                          const MatrixView<T, MatrixFormat::Dense>& A) {
    return ::batchlas::inv(ctx, A);
}

/// @brief Returns \f$A^T\f$ as a new matrix. Real `T` only.
///
/// Plain transpose; a complex call is a compile error because nothing conjugates.
template <typename T>
    requires RealScalar<T>
inline Matrix<T, MatrixFormat::Dense> transpose(Queue& ctx,
                                                const MatrixView<T, MatrixFormat::Dense>& A) {
    return ::batchlas::transpose<T, MatrixFormat::Dense>(ctx, A);
}

/// @brief Returns one norm per batch item. **Blocks** until the result is ready.
/// @see batchlas::norm for the asynchronous out-parameter form
template <typename T>
inline UnifiedVector<typename base_type<T>::type> norm(
        Queue& ctx,
        const MatrixView<T, MatrixFormat::Dense>& A,
        NormType norm_type = NormType::Frobenius) {
    return ::batchlas::norm<T, MatrixFormat::Dense>(ctx, A, norm_type);
}

/// @brief Returns \f$\|A_i\|\,\|A_i^{-1}\|\f$ per item (eigenvalue-magnitude ratio for Spectral).
///
/// **Blocks** until the result is ready. Real `T` only.
template <typename T>
    requires RealScalar<T>
inline UnifiedVector<T> cond(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             NormType norm_type = NormType::Frobenius) {
    return ::batchlas::cond(ctx, A, norm_type);
}

// ---- composites ------------------------------------------------------------

/// @brief Result of svd().
template <typename T>
struct Svd {
    Matrix<T, MatrixFormat::Dense> U;                   ///< m x m (All) or m x k (Thin)
    UnifiedVector<typename base_type<T>::type> values;  ///< k = min(m, n) per batch item
    Matrix<T, MatrixFormat::Dense> Vh;                  ///< n x n (All) or k x n (Thin)
    /// Per-item convergence status, as Eigh::info. Always filled.
    UnifiedVector<int32_t> info;
};

/// @brief Singular value decomposition \f$A_i = U_i\Sigma_iV_i^H\f$; A is not modified.
///
/// **Blocks** before returning (its internal copy of A must outlive the kernels).
/// @param ctx      queue; supplies the backend and the arena
/// @param A        m x n per item
/// @param vectors  All or Thin
/// @return         U, singular values, Vh and per-item info
/// @throws batchlas::invalid_argument for SvdVectors::None; call batchlas::gesvd
///         directly for values only
/// @note Complex input can throw at run time: no blocked route has a complex path,
///       and the cta route rejects max(m, n) > 32.
template <typename T>
inline Svd<T> svd(Queue& ctx,
                  const MatrixView<T, MatrixFormat::Dense>& A,
                  SvdVectors vectors = SvdVectors::All) {
    if (vectors == SvdVectors::None) {
        throw batchlas::invalid_argument(
            "linalg::svd: SvdVectors::None would leave U and Vh empty; use "
            "::batchlas::gesvd directly for a values-only decomposition");
    }
    const int64_t m = A.rows();
    const int64_t n = A.cols();
    const int64_t k = std::min(m, n);
    const int batch = A.batch_size();

    auto work = detail::like(A);
    (void)MatrixView<T, MatrixFormat::Dense>::copy(ctx, work.view(), A);

    Matrix<T, MatrixFormat::Dense> U(static_cast<int>(m),
                                     static_cast<int>(svd_u_cols(vectors, m, k)), batch);
    Matrix<T, MatrixFormat::Dense> Vh(static_cast<int>(svd_vh_rows(vectors, n, k)),
                                      static_cast<int>(n), batch);
    UnifiedVector<typename base_type<T>::type> S(static_cast<size_t>(k) *
                                                 static_cast<size_t>(batch));

    UnifiedVector<int32_t> info(static_cast<size_t>(batch));
    // Positional because GesvdOptions has no `info` field.
    auto lease = ctx.workspace(::batchlas::gesvd_buffer_size(
        ctx, work.view(), S.to_span(), U.view(), Vh.view(), vectors, vectors));
    (void)::batchlas::gesvd(ctx, work.view(), S.to_span(), U.view(), Vh.view(), vectors, vectors,
                      lease.span(), info.to_span());

    // ~Matrix frees `work` without waiting; returning early frees it under the kernels.
    ctx.wait();
    return Svd<T>{std::move(U), std::move(S), std::move(Vh), std::move(info)};
}

/// @brief Result of lu().
template <typename T>
struct Lu {
    Matrix<T, MatrixFormat::Dense> factors;  ///< L and U packed; L's unit diagonal is implicit
    UnifiedVector<int64_t> pivots;           ///< rows * batch_size, in getrf()'s pivot format
};

/// @brief LU factorisation with partial pivoting, \f$A_i = P_iL_iU_i\f$; A is not modified.
/// @pre `A` is square
/// @return packed factors and pivots, valid after the queue is waited on
// Pivots are a UnifiedVector, not an arena lease as in solve(): they outlive the call.
template <typename T>
inline Lu<T> lu(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A) {
    auto LU = detail::like(A);
    (void)MatrixView<T, MatrixFormat::Dense>::copy(ctx, LU.view(), A);
    UnifiedVector<int64_t> pivots(static_cast<size_t>(A.rows()) *
                                  static_cast<size_t>(A.batch_size()));
    (void)::batchlas::getrf(ctx, LU.view(), pivots.to_span());
    return Lu<T>{std::move(LU), std::move(pivots)};
}

// linalg::qr is deliberately absent: an unexplained cross-Queue wrong answer.
// evidence: docs/design/known-defects.md#linalgqr-returns-a-wrong-qr-after-an-earlier-call-in-the-process


/// @}

}  // namespace batchlas::linalg
