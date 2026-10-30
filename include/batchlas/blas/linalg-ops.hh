// SPDX-License-Identifier: MIT
#pragma once

#include <batchlas/export.hh>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

#include <batchlas/blas/enums.hh>
#include <batchlas/error.hh>
// Needed here: the qualified ids in the forwards below bind at definition context.
#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/extra.hh>
// Every entry point a wrapper below calls; never rely on them transitively.
#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/functions/gemv.hh>
#include <batchlas/blas/functions/gesv.hh>
#include <batchlas/blas/functions/gesvd.hh>
#include <batchlas/blas/functions/getrf.hh>
#include <batchlas/blas/functions/getri.hh>
#include <batchlas/blas/functions/getrs.hh>
#include <batchlas/blas/functions/herk.hh>
#include <batchlas/blas/functions/posv.hh>
#include <batchlas/blas/functions/potrf.hh>
#include <batchlas/blas/functions/syev.hh>
#include <batchlas/blas/functions/syrk.hh>
#include <batchlas/blas/functions/trsm.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/options.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>

/// @file
/// @brief The batchlas::linalg convenience layer: elementwise operations and value-returning wrappers.
/// @ingroup api_linalg

/**
 * @addtogroup api_linalg
 * @details
 * Free functions only, no operator overloads. Membership rule: value-returning,
 * backend from the Queue, workspace from the Queue's arena => `linalg::`;
 * out-parameter, workspace yours => `batchlas::`.
 *
 * Every function here **enqueues and returns without waiting**, including the
 * value-returning ones: a returned matrix is readable only after `ctx.wait()`.
 * The exceptions are norm(), cond(), svd(), solve(), solve_spd(), inv() and the
 * full eigvalsh(), which wait because they free local scratch on return. A
 * wrapper with more than one launch orders them itself, so an out-of-order Queue
 * is safe. Factorisations and solves return a per-item `info`. Inputs are never
 * modified, except by the `_into` forms (which write their last matrix argument)
 * and scale(). A value-returning call allocates its result, so prefer the `_into`
 * forms in an inner loop. User guide: @ref md_docs_2cpp-api (the `linalg` section);
 * design notes: @ref design_api_conventions.
 */

namespace batchlas::linalg {

/// @addtogroup api_linalg
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
// First in every value-returning wrapper: its work copies and results carry no batch mask, so a
// masked operand would run every item.
template <typename... V>
inline void refuse_batch_mask(const char* who, const V&... v) {
    if ((v.has_batch_mask() || ...))
        throw ::batchlas::unsupported(std::string("linalg::") + who + ": batch mask not supported yet");
}

template <typename T>
inline Matrix<T, MatrixFormat::Dense> like(const MatrixView<T, MatrixFormat::Dense>& A) {
    return Matrix<T, MatrixFormat::Dense>(A.rows(), A.cols(), A.batch_size());
}
}  // namespace detail

/// @addtogroup api_linalg
/// @{

/// @brief Returns a new matrix \f$A + B\f$ (elementwise).
template <typename T>
inline Matrix<T, MatrixFormat::Dense> add(Queue& ctx,
                                          const MatrixView<T, MatrixFormat::Dense>& A,
                                          const MatrixView<T, MatrixFormat::Dense>& B) {
    detail::refuse_batch_mask("add", A, B);
    auto C = detail::like(A);
    // (void) on an Event is deliberate: the in-order Queue already orders the next call.
    (void)add_into<T>(ctx, A, B, C);
    return C;
}

/// @brief Returns a new matrix \f$A - B\f$ (elementwise).
template <typename T>
inline Matrix<T, MatrixFormat::Dense> subtract(Queue& ctx,
                                               const MatrixView<T, MatrixFormat::Dense>& A,
                                               const MatrixView<T, MatrixFormat::Dense>& B) {
    detail::refuse_batch_mask("subtract", A, B);
    auto C = detail::like(A);
    (void)subtract_into<T>(ctx, A, B, C);
    return C;
}

/// @brief Returns a new matrix \f$A \odot B\f$ (Hadamard, not matmul()).
template <typename T>
inline Matrix<T, MatrixFormat::Dense> multiply(Queue& ctx,
                                               const MatrixView<T, MatrixFormat::Dense>& A,
                                               const MatrixView<T, MatrixFormat::Dense>& B) {
    detail::refuse_batch_mask("multiply", A, B);
    auto C = detail::like(A);
    (void)multiply_into<T>(ctx, A, B, C);
    return C;
}

/// @brief Returns a new matrix \f$A \oslash B\f$ (elementwise).
template <typename T>
inline Matrix<T, MatrixFormat::Dense> divide(Queue& ctx,
                                             const MatrixView<T, MatrixFormat::Dense>& A,
                                             const MatrixView<T, MatrixFormat::Dense>& B) {
    detail::refuse_batch_mask("divide", A, B);
    auto C = detail::like(A);
    (void)divide_into<T>(ctx, A, B, C);
    return C;
}

/// @brief Returns a new matrix \f$\alpha A\f$; `A` is not modified (scale() works in place).
template <typename T>
inline Matrix<T, MatrixFormat::Dense> scaled(Queue& ctx,
                                             const MatrixView<T, MatrixFormat::Dense>& A,
                                             T alpha) {
    detail::refuse_batch_mask("scaled", A);
    auto C = detail::like(A);
    (void)axpby_into<T>(ctx, alpha, A, T(0), A, C);
    return C;
}

// ---- value-returning wrappers ---------------------------------------------
// A wrapper that submits more than once runs its body through on_in_order_queue: on an
// out-of-order Queue the copy of A would otherwise race the factorisation that reads it.
// The Hermitian wrappers mirror the named triangle on their copy: some drivers read the
// other one (defect 14).
// evidence: docs/design/api-conventions.md#api-conventions-linalg-wrappers-order-their-own-launches

/// @brief Default triangle of the linalg wrappers that read one.
inline constexpr Uplo kDefaultUplo = Uplo::Lower;

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
    detail::refuse_batch_mask("matmul", A, B);
    const bool ta = opts.transA != Transpose::NoTrans;
    const bool tb = opts.transB != Transpose::NoTrans;
    const auto m = ta ? A.cols() : A.rows();
    const auto n = tb ? B.rows() : B.cols();
    Matrix<T, MatrixFormat::Dense> C(m, n, A.batch_size());
    (void)gemm(ctx, A, B, C,
         GemmOptions<T>{.alpha = opts.alpha,
                        .beta = T(0),  // C is fresh
                        .transA = opts.transA,
                        .transB = opts.transB,
                        .precision = opts.precision});
    return C;
}

/// @brief Returns a new vector \f$y = \mathrm{op}(A)\,x\f$ per item (NumPy `matvec`).
/// @param ctx    queue; supplies the backend
/// @param A      `op(A)` is m x n per item
/// @param x      length n per item, same batch size
/// @param trans  op(A)
/// @return       length-m vector per item (stride m), valid after the queue is waited on
template <typename T>
inline Vector<T> matvec(Queue& ctx,
                        const MatrixView<T, MatrixFormat::Dense>& A,
                        const VectorView<T>& x,
                        Transpose trans = Transpose::NoTrans) {
    detail::refuse_batch_mask("matvec", A, x);
    Vector<T> y(trans == Transpose::NoTrans ? A.rows() : A.cols(), A.batch_size());
    (void)gemv(ctx, A, x, y, GemvOptions<T>{.transA = trans});
    return y;
}

/// @brief Returns the Hermitian Gram matrix \f$A^HA\f$ (default) or \f$AA^H\f$, both triangles filled.
///
/// Real `T` runs syrk(); complex `T` runs herk(), which has no native kernel, so a
/// complex call in a build without the vendor BLAS throws batchlas::NoRouteError.
/// @param ctx    queue; supplies the backend
/// @param A      m x n per item
/// @param trans  NoTrans gives \f$AA^H\f$ (m x m); Trans or ConjTrans gives \f$A^HA\f$ (n x n)
/// @return       new matrix, valid after the queue is waited on
template <typename T>
inline Matrix<T, MatrixFormat::Dense> gram(Queue& ctx,
                                           const MatrixView<T, MatrixFormat::Dense>& A,
                                           Transpose trans = Transpose::ConjTrans) {
    detail::refuse_batch_mask("gram", A);
    const bool inner = trans != Transpose::NoTrans;
    const int n = inner ? A.cols() : A.rows();
    Matrix<T, MatrixFormat::Dense> C(n, n, A.batch_size());
    (void)::batchlas::detail::on_in_order_queue(ctx, [&](Queue& q) {
        if constexpr (RealScalar<T>) {
            (void)syrk(q, A, C,
                       SyrkOptions<T>{.uplo = kDefaultUplo,
                                      .trans = inner ? Transpose::Trans : Transpose::NoTrans});
            return C.symmetrize(q, kDefaultUplo);
        } else {
            (void)herk(q, A, C,
                       HerkOptions<T>{.uplo = kDefaultUplo,
                                      .trans = inner ? Transpose::ConjTrans : Transpose::NoTrans});
            return C.hermitize(q, kDefaultUplo);
        }
    });
    return C;
}

/// @brief Returns X with \f$\mathrm{op}(A)X = \alpha B\f$ (Left) or \f$X\,\mathrm{op}(A) = \alpha B\f$
///        (Right), A triangular (SciPy `solve_triangular`); neither A nor B is modified.
///
/// No status is reported: a zero diagonal entry gives infinities in that item.
/// @param ctx   queue; supplies the backend
/// @param A     triangular, m x m (Left) or n x n (Right) per item; only the `uplo` triangle is read
/// @param B     m x n per item
/// @param opts  alpha, side, uplo (default Lower), trans, diag (Unit: SciPy `unit_diagonal`)
/// @return      new m x n x batch matrix, valid after the queue is waited on
template <typename T>
inline Matrix<T, MatrixFormat::Dense> solve_triangular(Queue& ctx,
                                                       const MatrixView<T, MatrixFormat::Dense>& A,
                                                       const MatrixView<T, MatrixFormat::Dense>& B,
                                                       const TrsmOptions<T>& opts = {}) {
    detail::refuse_batch_mask("solve_triangular", A, B);
    Matrix<T, MatrixFormat::Dense> X(B.rows(), B.cols(), B.batch_size());
    (void)::batchlas::detail::on_in_order_queue(ctx, [&](Queue& q) {
        (void)MatrixView<T, MatrixFormat::Dense>::copy(q, X, B);
        return trsm(q, A, X, opts);
    });
    return X;
}

/// @brief Result of cholesky().
template <typename T>
struct Cholesky {
    Matrix<T, MatrixFormat::Dense> factor;  ///< the factor in the `uplo` triangle, zeros in the other
    /// Per-item status, as potrf(): 0 factorised, i > 0 the leading minor of order i is not
    /// positive definite (that item's factor is incomplete). Always filled; check it.
    UnifiedVector<int32_t> info;
};

/// @brief Cholesky factor of a Hermitian positive-definite A; A is not modified.
/// @param ctx   queue; supplies the backend and the arena
/// @param A     n x n per item; the `uplo` triangle is read
/// @param uplo  Lower gives \f$L\f$ with \f$A = LL^H\f$, Upper gives \f$U\f$ with \f$A = U^HU\f$
/// @return      factor and per-item info, valid after the queue is waited on
/// @throws batchlas::invalid_argument if `A` is not square
template <typename T>
inline Cholesky<T> cholesky(Queue& ctx,
                            const MatrixView<T, MatrixFormat::Dense>& A,
                            Uplo uplo = kDefaultUplo) {
    detail::refuse_batch_mask("cholesky", A);
    ::batchlas::detail::require_square("cholesky", "A", A);
    auto F = detail::like(A);
    UnifiedVector<int32_t> info(static_cast<size_t>(A.batch_size()));
    (void)::batchlas::detail::on_in_order_queue(ctx, [&](Queue& q) {
        (void)MatrixView<T, MatrixFormat::Dense>::copy(q, F, A);
        (void)potrf(q, F, PotrfOptions{.uplo = uplo, .info = info.to_span()});
        // Vendor Upper potrf overwrites the other triangle; zero it for a clean factor.
        return triangular_mask_into<T>(q, F, F, uplo, 0);
    });
    return Cholesky<T>{std::move(F), std::move(info)};
}

/// @brief Returns the eigenvalues of a symmetric/Hermitian A, ascending per item; A is not modified.
///
/// **Blocks** before returning (its internal copy of A must outlive the kernels).
/// @param ctx   queue; supplies the backend and the arena
/// @param A     n x n per item; the `uplo` triangle is read
/// @param uplo  triangle of A that is read
/// @return      real vector of `n * batch_size` values, item b at offset `b * n`
template <typename T>
inline UnifiedVector<typename base_type<T>::type> eigvalsh(Queue& ctx,
                                                           const MatrixView<T, MatrixFormat::Dense>& A,
                                                           Uplo uplo = kDefaultUplo) {
    detail::refuse_batch_mask("eigvalsh", A);
    UnifiedVector<typename base_type<T>::type> W(static_cast<size_t>(A.rows()) *
                                                 static_cast<size_t>(A.batch_size()));
    auto work = detail::like(A);
    (void)::batchlas::detail::on_in_order_queue(ctx, [&](Queue& q) {
        (void)MatrixView<T, MatrixFormat::Dense>::copy(q, work, A);
        (void)work.hermitize(q, uplo);
        (void)syev(q, work, W.to_span(), {.jobz = JobType::NoEigenVectors, .uplo = uplo});
        q.wait();  // ~Matrix frees `work` without waiting
        return q.get_event();
    });
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
    detail::refuse_batch_mask("eigh", A);
    UnifiedVector<typename base_type<T>::type> W(static_cast<size_t>(A.rows()) *
                                                 static_cast<size_t>(A.batch_size()));
    UnifiedVector<int32_t> info(static_cast<size_t>(A.batch_size()));
    // The positional syev below does not check squareness. Qualified: `detail::`
    // here would find batchlas::linalg::detail.
    ::batchlas::detail::require_square("eigh", "A", A);
    auto V = detail::like(A);
    (void)::batchlas::detail::on_in_order_queue(ctx, [&](Queue& q) {
        (void)MatrixView<T, MatrixFormat::Dense>::copy(q, V, A);
        (void)V.hermitize(q, uplo);
        // Positional because SyevOptions has no `info` field.
        auto lease = q.workspace(
            syev_buffer_size(q, V, W.to_span(), JobType::EigenVectors, uplo));
        return syev(q, V, W.to_span(), JobType::EigenVectors, uplo, lease.span(),
                    info.to_span());
    });
    return Eigh<T>{std::move(W), std::move(V), std::move(info)};
}

/// @brief Result of the subset forms of eigh() and eigvalsh().
template <typename T>
struct EighSubset {
    /// `k` slots per batch item (item b at offset `b * k`; `k` = `iu - il + 1` for an index
    /// range, `n` for a value range); the first `count[b]` are written, ascending unless
    /// SyevxParams::order says otherwise. Slots past `count[b]` are not written.
    UnifiedVector<typename base_type<T>::type> values;
    /// n x k per item, the eigenvectors as columns; columns past `count[b]` are zero.
    /// Empty (0 x 0) from eigvalsh().
    Matrix<T, MatrixFormat::Dense> vectors;
    UnifiedVector<int32_t> count;  ///< eigenpairs found per item: `k` for an index range
    UnifiedVector<int32_t> info;   ///< per-item convergence status, as Eigh::info
};

namespace detail {
template <typename T>
inline EighSubset<T> eigh_subset(Queue& ctx,
                                 const MatrixView<T, MatrixFormat::Dense>& A,
                                 const SyevxParams<T>& subset,
                                 JobType jobz) {
    refuse_batch_mask("eigh", A);
    ::batchlas::detail::require_square("eigh", "A", A);
    const int64_t n = A.rows();
    int64_t k = n;
    if (subset.select == SyevxSelect::Index) {
        k = std::max<int64_t>(0, (subset.iu < 0 ? n - 1 : subset.iu) - subset.il + 1);
    } else if (subset.select != SyevxSelect::Value) {
        throw batchlas::invalid_argument(
            "linalg::eigh: a subset needs select = Index or Value; Extremal has no count here");
    }
    const int batch = A.batch_size();
    const bool vectors = jobz == JobType::EigenVectors;
    EighSubset<T> r{
        UnifiedVector<typename base_type<T>::type>(static_cast<size_t>(k) * static_cast<size_t>(batch)),
        Matrix<T, MatrixFormat::Dense>(vectors ? static_cast<int>(n) : 0,
                                       vectors ? static_cast<int>(k) : 0, vectors ? batch : 1),
        UnifiedVector<int32_t>(static_cast<size_t>(batch)),
        UnifiedVector<int32_t>(static_cast<size_t>(batch))};
    const MatrixView<T, MatrixFormat::Dense> V =
        vectors ? MatrixView<T, MatrixFormat::Dense>(r.vectors) : MatrixView<T, MatrixFormat::Dense>();
    (void)::batchlas::detail::on_in_order_queue(ctx, [&](Queue& q) {
        auto lease = q.workspace(syevx_buffer_size(q, A, r.values.to_span(), r.count.to_span(),
                                                   static_cast<size_t>(k), jobz, V, subset));
        return syevx(q, A, r.values.to_span(), r.count.to_span(), static_cast<size_t>(k),
                     lease.span(), jobz, V, subset, r.info.to_span());
    });
    return r;
}
}  // namespace detail

/// @brief Eigenpairs of a symmetric/Hermitian A by index or value range (SciPy `eigh` with
///        `subset_by_index` / `subset_by_value`); runs syevx(). A is not modified.
/// @param ctx     queue; supplies the backend and the arena
/// @param A       n x n per item; the lower triangle is read
/// @param subset  `select = SyevxSelect::Index` with `il`, `iu` (0-based, inclusive, ascending
///                spectrum; `iu < 0` means n - 1), or `SyevxSelect::Value` with the interval
///                (`vl`, `vu`]; `order`, `abstol` and `method` are honoured
/// @return        values, vectors, per-item counts and info, valid after the queue is waited on
/// @throws batchlas::invalid_argument for `SyevxSelect::Extremal`, a non-square `A`, or a range
///         syevx() rejects (an index block outside [0, n), `vl >= vu`)
template <typename T>
inline EighSubset<T> eigh(Queue& ctx,
                          const MatrixView<T, MatrixFormat::Dense>& A,
                          const SyevxParams<T>& subset) {
    return detail::eigh_subset(ctx, A, subset, JobType::EigenVectors);
}

/// @brief eigh() with a subset, eigenvalues only; `vectors` is empty. Does not wait.
template <typename T>
inline EighSubset<T> eigvalsh(Queue& ctx,
                              const MatrixView<T, MatrixFormat::Dense>& A,
                              const SyevxParams<T>& subset) {
    return detail::eigh_subset(ctx, A, subset, JobType::NoEigenVectors);
}

/// @brief Result of solve() and solve_spd().
template <typename T>
struct Solve {
    Matrix<T, MatrixFormat::Dense> x;  ///< the solution, n x nrhs per item
    /// Per-item status of the factorisation: getrf's for solve() (i > 0: U(i,i) is exactly
    /// zero), potrf's for solve_spd() (i > 0: not positive definite). That item's `x` is not
    /// meaningful. Always filled; a near-singular item still reports 0.
    UnifiedVector<int32_t> info;
};

/// @brief Solves \f$\mathrm{op}(A_i)X_i = B_i\f$ by LU with partial pivoting.
///
/// Neither A nor B is modified; pivots and workspace come from the arena.
/// **Blocks** before returning (its internal copy of A must outlive the kernels).
/// @param ctx    queue; supplies the backend and the arena
/// @param A      n x n per item
/// @param B      n x nrhs per item, same batch size
/// @param trans  op(A)
/// @return       the solution and per-item info
/// @throws batchlas::internal_error for NoTrans when an extent (n, nrhs or batch)
///         is below 1: `gesv` has no kernel for an empty problem
template <typename T>
inline Solve<T> solve(Queue& ctx,
                      const MatrixView<T, MatrixFormat::Dense>& A,
                      const MatrixView<T, MatrixFormat::Dense>& B,
                      Transpose trans = Transpose::NoTrans) {
    detail::refuse_batch_mask("solve", A, B);
    auto LU = detail::like(A);
    Matrix<T, MatrixFormat::Dense> X(B.rows(), B.cols(), B.batch_size());
    UnifiedVector<int32_t> info(static_cast<size_t>(A.batch_size()));
    (void)::batchlas::detail::on_in_order_queue(ctx, [&](Queue& q) {
        (void)MatrixView<T, MatrixFormat::Dense>::copy(q, LU, A);
        (void)MatrixView<T, MatrixFormat::Dense>::copy(q, X, B);
        const size_t n_pivots = static_cast<size_t>(A.rows()) * static_cast<size_t>(A.batch_size());
        auto pivot_bytes = q.workspace(n_pivots * sizeof(int64_t));
        Span<int64_t> pivots(reinterpret_cast<int64_t*>(pivot_bytes.data()), n_pivots);
        // Not stylistic, do not flatten: gesv has no Transpose parameter. No shape gate
        // here either; gesv selects its own kernel. Degenerate extents throw (deliberate).
        // evidence: docs/design/api-conventions.md#api-conventions-linalgsolve-keeps-its-transpose-branch
        if (trans == Transpose::NoTrans) {
            // Released before the pivot lease (reverse order): the arena reuses it at once.
            auto lease = q.workspace(gesv_buffer_size(q, LU, X));
            (void)gesv(q, LU, X, pivots, lease.span(), info.to_span());
        } else {
            (void)::batchlas::getrf(q, LU, pivots, info.to_span());
            (void)getrs(q, LU, X, pivots, {.trans = trans});
        }
        q.wait();  // ~Matrix frees `LU` without waiting
        return q.get_event();
    });
    return Solve<T>{std::move(X), std::move(info)};
}

/// @brief Solves \f$A_iX_i = B_i\f$ for Hermitian positive-definite A by Cholesky.
///
/// About half the arithmetic of solve(), which stays the safe default: the SPD
/// claim is the caller's, and `info` reports the items where it is false.
/// Neither A nor B is modified; the workspace comes from the arena.
/// **Blocks** before returning (its internal copy of A must outlive the kernels).
/// @param ctx   queue; supplies the backend and the arena
/// @param A     n x n per item; the `uplo` triangle is read
/// @param B     n x nrhs per item
/// @param uplo  triangle of A that is read
/// @return      the solution and per-item info
/// @throws batchlas::internal_error when an extent (n, nrhs or batch) is below 1:
///         `posv` has no kernel for an empty problem
template <typename T>
inline Solve<T> solve_spd(Queue& ctx,
                          const MatrixView<T, MatrixFormat::Dense>& A,
                          const MatrixView<T, MatrixFormat::Dense>& B,
                          Uplo uplo = kDefaultUplo) {
    detail::refuse_batch_mask("solve_spd", A, B);
    auto F = detail::like(A);
    Matrix<T, MatrixFormat::Dense> X(B.rows(), B.cols(), B.batch_size());
    UnifiedVector<int32_t> info(static_cast<size_t>(A.batch_size()));
    (void)::batchlas::detail::on_in_order_queue(ctx, [&](Queue& q) {
        (void)MatrixView<T, MatrixFormat::Dense>::copy(q, F, A);
        (void)MatrixView<T, MatrixFormat::Dense>::copy(q, X, B);
        auto lease = q.workspace(posv_buffer_size(q, F, X, uplo));
        (void)posv(q, F, X, uplo, lease.span(), info.to_span());
        q.wait();  // ~Matrix frees `F` without waiting
        return q.get_event();
    });
    return Solve<T>{std::move(X), std::move(info)};
}

/// @brief Returns a copy of A with everything below diagonal `k` zeroed (NumPy `triu`).
template <typename T>
inline Matrix<T, MatrixFormat::Dense> triu(Queue& ctx,
                                           const MatrixView<T, MatrixFormat::Dense>& A,
                                           int64_t k = 0) {
    detail::refuse_batch_mask("triu", A);
    auto C = detail::like(A);
    (void)triangular_mask_into<T>(ctx, A, C, Uplo::Upper, k);
    return C;
}

/// @brief Returns a copy of A with everything above diagonal `k` zeroed (NumPy `tril`).
template <typename T>
inline Matrix<T, MatrixFormat::Dense> tril(Queue& ctx,
                                           const MatrixView<T, MatrixFormat::Dense>& A,
                                           int64_t k = 0) {
    detail::refuse_batch_mask("tril", A);
    auto C = detail::like(A);
    (void)triangular_mask_into<T>(ctx, A, C, Uplo::Lower, k);
    return C;
}

/// @brief Result of inv().
template <typename T>
struct Inv {
    Matrix<T, MatrixFormat::Dense> inverse;  ///< \f$A^{-1}\f$, n x n per item
    UnifiedVector<int32_t> info;             ///< per-item status as Solve::info (getrf's)
};

/// @brief Inverse of A by LU with partial pivoting (getrf + getri); A is not modified.
///
/// **Blocks** before returning (its internal copy of A must outlive the kernels).
/// @return the inverse and per-item info
/// @throws batchlas::invalid_argument if `A` is not square
template <typename T>
inline Inv<T> inv(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A) {
    detail::refuse_batch_mask("inv", A);
    ::batchlas::detail::require_square("inv", "A", A);
    auto LU = detail::like(A);
    auto Ainv = detail::like(A);
    UnifiedVector<int32_t> info(static_cast<size_t>(A.batch_size()));
    (void)::batchlas::detail::on_in_order_queue(ctx, [&](Queue& q) {
        (void)MatrixView<T, MatrixFormat::Dense>::copy(q, LU, A);
        const size_t n_pivots = static_cast<size_t>(A.rows()) * static_cast<size_t>(A.batch_size());
        auto pivot_bytes = q.workspace(n_pivots * sizeof(int64_t));
        Span<int64_t> pivots(reinterpret_cast<int64_t*>(pivot_bytes.data()), n_pivots);
        (void)::batchlas::getrf(q, LU, pivots, info.to_span());
        (void)getri(q, LU, Ainv, pivots);
        q.wait();  // ~Matrix frees `LU` without waiting
        return q.get_event();
    });
    return Inv<T>{std::move(Ainv), std::move(info)};
}

// ---- forwarding aliases ----------------------------------------------------
// Every call below must stay qualified with ::batchlas::: linalg is nested inside
// batchlas, so `return transpose(ctx, A);` here is infinite recursion.

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
    Matrix<T, MatrixFormat::Dense> U;                   ///< m x m (All), m x k (Thin) or m x 0 (None)
    UnifiedVector<typename base_type<T>::type> values;  ///< k = min(m, n) per batch item, descending
    Matrix<T, MatrixFormat::Dense> Vh;                  ///< n x n (All), k x n (Thin) or 0 x n (None)
    /// Per-item convergence status, as Eigh::info. Always filled.
    UnifiedVector<int32_t> info;
};

/// @brief Singular value decomposition \f$A_i = U_i\Sigma_iV_i^H\f$; A is not modified.
///
/// **Blocks** before returning (its internal copy of A must outlive the kernels).
/// @param ctx   queue; supplies the backend and the arena
/// @param A     m x n per item (square when `hermitian_uplo` is engaged)
/// @param opts  jobu and jobvh (All, Thin or None; they may differ: Thin is NumPy
///              `full_matrices=False`), and `hermitian_uplo` for Hermitian input
/// @return      U, singular values, Vh and per-item info
/// @throws batchlas::invalid_argument if both jobs are None (call batchlas::gesvd for
///         values only), or `A` is not square with `hermitian_uplo` engaged
/// @note Complex input the one-sided Jacobi kernel cannot take (beyond its size
///       limit, or on a device without sub-group 32) needs the vendor library: no
///       other native gesvd kernel takes complex general input,
///       so a build without it throws batchlas::NoRouteError.
template <typename T>
inline Svd<T> svd(Queue& ctx,
                  const MatrixView<T, MatrixFormat::Dense>& A,
                  const GesvdOptions& opts) {
    detail::refuse_batch_mask("svd", A);
    if (opts.jobu == SvdVectors::None && opts.jobvh == SvdVectors::None) {
        throw batchlas::invalid_argument(
            "linalg::svd: SvdVectors::None for both U and Vh leaves them empty; use "
            "::batchlas::gesvd directly for a values-only decomposition");
    }
    if (opts.hermitian_uplo) ::batchlas::detail::require_square("svd", "A", A);
    const int64_t m = A.rows();
    const int64_t n = A.cols();
    const int64_t k = std::min(m, n);
    const int batch = A.batch_size();

    auto work = detail::like(A);
    Matrix<T, MatrixFormat::Dense> U(static_cast<int>(m),
                                     static_cast<int>(svd_u_cols(opts.jobu, m, k)), batch);
    Matrix<T, MatrixFormat::Dense> Vh(static_cast<int>(svd_vh_rows(opts.jobvh, n, k)),
                                      static_cast<int>(n), batch);
    UnifiedVector<typename base_type<T>::type> S(static_cast<size_t>(k) *
                                                 static_cast<size_t>(batch));
    UnifiedVector<int32_t> info(static_cast<size_t>(batch));
    // A None side is passed as an empty view: gesvd's spelling of "not computed".
    using MV = MatrixView<T, MatrixFormat::Dense>;
    const MV Uv = opts.jobu == SvdVectors::None ? MV() : MV(U);
    const MV Vhv = opts.jobvh == SvdVectors::None ? MV() : MV(Vh);

    (void)::batchlas::detail::on_in_order_queue(ctx, [&](Queue& q) {
        (void)MV::copy(q, work, A);
        if (opts.hermitian_uplo) (void)work.hermitize(q, *opts.hermitian_uplo);
        // Positional because GesvdOptions has no `info` field; size and call take one branch.
        if (opts.hermitian_uplo) {
            auto lease = q.workspace(::batchlas::gesvd_buffer_size(
                q, work, S.to_span(), Uv, Vhv, opts.jobu, opts.jobvh, *opts.hermitian_uplo));
            (void)::batchlas::gesvd(q, work, S.to_span(), Uv, Vhv, opts.jobu, opts.jobvh,
                                    *opts.hermitian_uplo, lease.span(), info.to_span());
        } else {
            auto lease = q.workspace(::batchlas::gesvd_buffer_size(
                q, work, S.to_span(), Uv, Vhv, opts.jobu, opts.jobvh));
            (void)::batchlas::gesvd(q, work, S.to_span(), Uv, Vhv, opts.jobu, opts.jobvh,
                                    lease.span(), info.to_span());
        }
        q.wait();  // ~Matrix frees `work` without waiting
        return q.get_event();
    });
    return Svd<T>{std::move(U), std::move(S), std::move(Vh), std::move(info)};
}

/// @brief svd() with one job for both U and Vh (All or Thin).
template <typename T>
inline Svd<T> svd(Queue& ctx,
                  const MatrixView<T, MatrixFormat::Dense>& A,
                  SvdVectors vectors = SvdVectors::All) {
    return svd(ctx, A, GesvdOptions{.jobu = vectors, .jobvh = vectors});
}

/// @brief Result of lu().
template <typename T>
struct Lu {
    Matrix<T, MatrixFormat::Dense> factors;  ///< L and U packed; L's unit diagonal is implicit
    UnifiedVector<int64_t> pivots;           ///< rows * batch_size, in getrf()'s pivot format
    UnifiedVector<int32_t> info;             ///< per-item status as getrf(): i > 0 if U(i,i) is exactly zero
};

/// @brief LU factorisation with partial pivoting, \f$A_i = P_iL_iU_i\f$; A is not modified.
/// @return packed factors, pivots and per-item info, valid after the queue is waited on
/// @throws batchlas::invalid_argument if `A` is not square
// Pivots are a UnifiedVector, not an arena lease as in solve(): they outlive the call.
template <typename T>
inline Lu<T> lu(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A) {
    detail::refuse_batch_mask("lu", A);
    ::batchlas::detail::require_square("lu", "A", A);
    auto LU = detail::like(A);
    UnifiedVector<int64_t> pivots(static_cast<size_t>(A.rows()) *
                                  static_cast<size_t>(A.batch_size()));
    UnifiedVector<int32_t> info(static_cast<size_t>(A.batch_size()));
    (void)::batchlas::detail::on_in_order_queue(ctx, [&](Queue& q) {
        (void)MatrixView<T, MatrixFormat::Dense>::copy(q, LU, A);
        return ::batchlas::getrf(q, LU, pivots.to_span(), info.to_span());
    });
    return Lu<T>{std::move(LU), std::move(pivots), std::move(info)};
}

/// @brief Result of qr().
template <typename T>
struct Qr {
    Matrix<T, MatrixFormat::Dense> q;  ///< m x n per item, orthonormal columns
    Matrix<T, MatrixFormat::Dense> r;  ///< n x n per item, upper triangular, zeros below
};

/// @brief Reduced QR factorisation \f$A = QR\f$ of a tall or square A; A is not modified.
///
/// **Blocks** before returning (its Householder scalars must outlive the kernels).
/// @param ctx   queue; supplies the backend and the arena
/// @param A     m x n per item, m >= n
/// @return      Q and R, valid after the queue is waited on
/// @throws batchlas::invalid_argument if `A` has more columns than rows
template <typename T>
inline Qr<T> qr(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A) {
    detail::refuse_batch_mask("qr", A);
    const int m = A.rows(), n = A.cols(), batch = A.batch_size();
    if (n > m)
        throw batchlas::invalid_argument("linalg::qr: A is " + std::to_string(m) + " x " + std::to_string(n) +
                                         "; the reduced QR needs rows >= cols");
    auto Q = detail::like(A);
    Matrix<T, MatrixFormat::Dense> R(n, n, batch);
    UnifiedVector<T> tau(static_cast<size_t>(n) * static_cast<size_t>(batch));
    (void)::batchlas::detail::on_in_order_queue(ctx, [&](Queue& q) {
        (void)MatrixView<T, MatrixFormat::Dense>::copy(q, Q, A);
        (void)geqrf(q, Q, tau.to_span());
        if (n > 0) (void)triangular_mask_into<T>(q, Q(Slice(0, n), Slice()), R, Uplo::Upper);
        (void)orgqr(q, Q, tau.to_span());
        q.wait();  // ~UnifiedVector frees `tau` without waiting
        return q.get_event();
    });
    return Qr<T>{std::move(Q), std::move(R)};
}

/// @}

}  // namespace batchlas::linalg
