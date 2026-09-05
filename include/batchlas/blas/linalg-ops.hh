#pragma once

#include <algorithm>
#include <stdexcept>
#include <utility>

#include <batchlas/blas/enums.hh>
// Needed here, not merely by our includers: the qualified ids in the forwards
// below bind at template-definition context.
#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/extra.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/options.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>

// batchlas::linalg -- the convenience layer: elementwise operations, and
// value-returning wrappers that allocate their own result. Free functions only.
// Membership rule: value-returning, backend from the Queue, workspace from the
// arena => linalg::; out-parameter, workspace yours => batchlas::.
// See docs/cpp-api.md#the-linalg-convenience-layer.
namespace batchlas::linalg {

// ---- elementwise -----------------------------------------------------------

enum class BinaryOp { Add, Subtract, Multiply, Divide };

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

// C = op(A, B), elementwise, for matching shapes. Aliasing C with A or B is
// allowed: each work-item reads and writes one element.
template <typename T, BinaryOp Op>
Event elementwise_into(Queue& ctx,
                       const MatrixView<T, MatrixFormat::Dense>& A,
                       const MatrixView<T, MatrixFormat::Dense>& B,
                       const MatrixView<T, MatrixFormat::Dense>& C);

// C = alpha*A + beta*B.
template <typename T>
Event axpby_into(Queue& ctx,
                 T alpha,
                 const MatrixView<T, MatrixFormat::Dense>& A,
                 T beta,
                 const MatrixView<T, MatrixFormat::Dense>& B,
                 const MatrixView<T, MatrixFormat::Dense>& C);

template <typename T>
Event scale(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, T alpha);

// C = A with everything outside the requested triangle zeroed. `k` follows
// NumPy: 0 keeps the main diagonal, > 0 moves the boundary toward the upper
// right, < 0 toward the lower left. Aliasing C with A is allowed.
template <typename T>
Event triangular_mask_into(Queue& ctx,
                           const MatrixView<T, MatrixFormat::Dense>& A,
                           const MatrixView<T, MatrixFormat::Dense>& C,
                           Uplo uplo,
                           int64_t k = 0);

template <typename T>
inline Event add_into(Queue& ctx,
                      const MatrixView<T, MatrixFormat::Dense>& A,
                      const MatrixView<T, MatrixFormat::Dense>& B,
                      const MatrixView<T, MatrixFormat::Dense>& C) {
    return elementwise_into<T, BinaryOp::Add>(ctx, A, B, C);
}

template <typename T>
inline Event subtract_into(Queue& ctx,
                           const MatrixView<T, MatrixFormat::Dense>& A,
                           const MatrixView<T, MatrixFormat::Dense>& B,
                           const MatrixView<T, MatrixFormat::Dense>& C) {
    return elementwise_into<T, BinaryOp::Subtract>(ctx, A, B, C);
}

// Elementwise (Hadamard), not matrix multiplication -- see matmul for that.
template <typename T>
inline Event multiply_into(Queue& ctx,
                           const MatrixView<T, MatrixFormat::Dense>& A,
                           const MatrixView<T, MatrixFormat::Dense>& B,
                           const MatrixView<T, MatrixFormat::Dense>& C) {
    return elementwise_into<T, BinaryOp::Multiply>(ctx, A, B, C);
}

template <typename T>
inline Event divide_into(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         const MatrixView<T, MatrixFormat::Dense>& B,
                         const MatrixView<T, MatrixFormat::Dense>& C) {
    return elementwise_into<T, BinaryOp::Divide>(ctx, A, B, C);
}

namespace detail {
template <typename T>
inline Matrix<T, MatrixFormat::Dense> like(const MatrixView<T, MatrixFormat::Dense>& A) {
    return Matrix<T, MatrixFormat::Dense>(A.rows(), A.cols(), A.batch_size());
}
}  // namespace detail

template <typename T>
inline Matrix<T, MatrixFormat::Dense> add(Queue& ctx,
                                          const MatrixView<T, MatrixFormat::Dense>& A,
                                          const MatrixView<T, MatrixFormat::Dense>& B) {
    auto C = detail::like(A);
    add_into<T>(ctx, A, B, C.view());
    return C;
}

template <typename T>
inline Matrix<T, MatrixFormat::Dense> subtract(Queue& ctx,
                                               const MatrixView<T, MatrixFormat::Dense>& A,
                                               const MatrixView<T, MatrixFormat::Dense>& B) {
    auto C = detail::like(A);
    subtract_into<T>(ctx, A, B, C.view());
    return C;
}

template <typename T>
inline Matrix<T, MatrixFormat::Dense> multiply(Queue& ctx,
                                               const MatrixView<T, MatrixFormat::Dense>& A,
                                               const MatrixView<T, MatrixFormat::Dense>& B) {
    auto C = detail::like(A);
    multiply_into<T>(ctx, A, B, C.view());
    return C;
}

template <typename T>
inline Matrix<T, MatrixFormat::Dense> divide(Queue& ctx,
                                             const MatrixView<T, MatrixFormat::Dense>& A,
                                             const MatrixView<T, MatrixFormat::Dense>& B) {
    auto C = detail::like(A);
    divide_into<T>(ctx, A, B, C.view());
    return C;
}

template <typename T>
inline Matrix<T, MatrixFormat::Dense> scaled(Queue& ctx,
                                             const MatrixView<T, MatrixFormat::Dense>& A,
                                             T alpha) {
    auto C = detail::like(A);
    axpby_into<T>(ctx, alpha, A, T(0), A, C.view());
    return C;
}

// ---- value-returning wrappers ---------------------------------------------

// Deliberately NOT GemmOptions: matmul allocates C, so a caller-supplied `beta`
// would read uninitialised memory. Omitting it makes naming it a compile error.
template <typename T>
struct MatmulOptions {
    T alpha = T(1);
    Transpose transA = Transpose::NoTrans;
    Transpose transB = Transpose::NoTrans;
    ComputePrecision precision = ComputePrecision::Default;
};

// C = alpha * op(A) * op(B), allocated here and fully written.
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
    gemm(ctx, A, B, C.view(),
         GemmOptions<T>{.alpha = opts.alpha,
                        .beta = T(0),  // C is fresh
                        .transA = opts.transA,
                        .transB = opts.transB,
                        .precision = opts.precision});
    return C;
}

inline constexpr Uplo kDefaultUplo = Uplo::Lower;

// Cholesky factor of A, as a new matrix. A is not modified.
template <typename T>
inline Matrix<T, MatrixFormat::Dense> cholesky(Queue& ctx,
                                               const MatrixView<T, MatrixFormat::Dense>& A,
                                               Uplo uplo = kDefaultUplo) {
    auto L = detail::like(A);
    MatrixView<T, MatrixFormat::Dense>::copy(ctx, L.view(), A);
    potrf(ctx, L.view(), {.uplo = uplo});
    return L;
}

// Eigenvalues of a symmetric/Hermitian A, ascending. A is not modified.
template <typename T>
inline UnifiedVector<typename base_type<T>::type> eigvalsh(Queue& ctx,
                                                           const MatrixView<T, MatrixFormat::Dense>& A,
                                                           Uplo uplo = kDefaultUplo) {
    UnifiedVector<typename base_type<T>::type> W(static_cast<size_t>(A.rows()) *
                                                 static_cast<size_t>(A.batch_size()));
    auto work = detail::like(A);
    MatrixView<T, MatrixFormat::Dense>::copy(ctx, work.view(), A);
    syev(ctx, work.view(), W.to_span(), {.jobz = JobType::NoEigenVectors, .uplo = uplo});
    return W;
}

template <typename T>
struct Eigh {
    UnifiedVector<typename base_type<T>::type> values;
    Matrix<T, MatrixFormat::Dense> vectors;
};

// Eigenvalues and eigenvectors of a symmetric/Hermitian A. A is not modified.
template <typename T>
inline Eigh<T> eigh(Queue& ctx,
                    const MatrixView<T, MatrixFormat::Dense>& A,
                    Uplo uplo = kDefaultUplo) {
    UnifiedVector<typename base_type<T>::type> W(static_cast<size_t>(A.rows()) *
                                                 static_cast<size_t>(A.batch_size()));
    auto V = detail::like(A);
    MatrixView<T, MatrixFormat::Dense>::copy(ctx, V.view(), A);
    syev(ctx, V.view(), W.to_span(), {.jobz = JobType::EigenVectors, .uplo = uplo});
    return Eigh<T>{std::move(W), std::move(V)};
}

// Solve A X = B for X by LU factorisation. Neither A nor B is modified. The
// pivots come from the arena: a local would be freed before the kernels run.
template <typename T>
inline Matrix<T, MatrixFormat::Dense> solve(Queue& ctx,
                                            const MatrixView<T, MatrixFormat::Dense>& A,
                                            const MatrixView<T, MatrixFormat::Dense>& B,
                                            Transpose trans = Transpose::NoTrans) {
    auto LU = detail::like(A);
    MatrixView<T, MatrixFormat::Dense>::copy(ctx, LU.view(), A);

    Matrix<T, MatrixFormat::Dense> X(B.rows(), B.cols(), B.batch_size());
    MatrixView<T, MatrixFormat::Dense>::copy(ctx, X.view(), B);

    const size_t n_pivots = static_cast<size_t>(A.rows()) * static_cast<size_t>(A.batch_size());
    auto pivot_bytes = ctx.workspace(n_pivots * sizeof(int64_t));
    Span<int64_t> pivots(reinterpret_cast<int64_t*>(pivot_bytes.data()), n_pivots);

    getrf(ctx, LU.view(), pivots);
    getrs(ctx, LU.view(), X.view(), pivots, {.trans = trans});
    return X;
}

// A with everything below (triu) or above (tril) the k-th diagonal zeroed.
template <typename T>
inline Matrix<T, MatrixFormat::Dense> triu(Queue& ctx,
                                           const MatrixView<T, MatrixFormat::Dense>& A,
                                           int64_t k = 0) {
    auto C = detail::like(A);
    triangular_mask_into<T>(ctx, A, C.view(), Uplo::Upper, k);
    return C;
}

template <typename T>
inline Matrix<T, MatrixFormat::Dense> tril(Queue& ctx,
                                           const MatrixView<T, MatrixFormat::Dense>& A,
                                           int64_t k = 0) {
    auto C = detail::like(A);
    triangular_mask_into<T>(ctx, A, C.view(), Uplo::Lower, k);
    return C;
}

// ---- forwarding aliases ----------------------------------------------------
// Every call below must stay EXPLICITLY qualified with ::batchlas::: linalg is
// nested inside batchlas, so `return inv(ctx, A);` here is infinite recursion.

// A^-1, as a new matrix. A is not modified. Square only (getrf's precondition).
template <typename T>
inline Matrix<T, MatrixFormat::Dense> inv(Queue& ctx,
                                          const MatrixView<T, MatrixFormat::Dense>& A) {
    return ::batchlas::inv(ctx, A);
}

// Plain transpose, NOT the conjugate transpose. Real only: transpose_impl does
// not conjugate, so the `requires` makes a complex call a compile error.
template <typename T>
    requires RealScalar<T>
inline Matrix<T, MatrixFormat::Dense> transpose(Queue& ctx,
                                                const MatrixView<T, MatrixFormat::Dense>& A) {
    return ::batchlas::transpose<T, MatrixFormat::Dense>(ctx, A);
}

// One norm per batch item. THIS ONE WAITS: ::batchlas::norm calls .wait()
// internally; its out-parameter form stays asynchronous.
template <typename T>
inline UnifiedVector<typename base_type<T>::type> norm(
        Queue& ctx,
        const MatrixView<T, MatrixFormat::Dense>& A,
        NormType norm_type = NormType::Frobenius) {
    return ::batchlas::norm<T, MatrixFormat::Dense>(ctx, A, norm_type);
}

// Condition number per batch item, ||A|| * ||A^-1|| (eigenvalue ratio for
// Spectral). WAITS, like norm. Real only.
template <typename T>
    requires RealScalar<T>
inline UnifiedVector<T> cond(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             NormType norm_type = NormType::Frobenius) {
    return ::batchlas::cond(ctx, A, norm_type);
}

// ---- composites ------------------------------------------------------------

template <typename T>
struct Svd {
    Matrix<T, MatrixFormat::Dense> U;                   // m x m (All) or m x k (Thin)
    UnifiedVector<typename base_type<T>::type> values;  // k = min(m, n) per batch item
    Matrix<T, MatrixFormat::Dense> Vh;                  // n x n (All) or k x n (Thin)
};

// Singular value decomposition. A is not modified -- gesvd overwrites its input,
// so it gets a copy. Complex input can throw at run time: no blocked route has a
// complex path, and the cta route rejects max(m, n) > 32.
template <typename T>
inline Svd<T> svd(Queue& ctx,
                  const MatrixView<T, MatrixFormat::Dense>& A,
                  SvdVectors vectors = SvdVectors::All) {
    if (vectors == SvdVectors::None) {
        throw std::invalid_argument(
            "linalg::svd: SvdVectors::None would leave U and Vh empty; use "
            "::batchlas::gesvd directly for a values-only decomposition");
    }
    const int64_t m = A.rows();
    const int64_t n = A.cols();
    const int64_t k = std::min(m, n);
    const int batch = A.batch_size();

    auto work = detail::like(A);
    MatrixView<T, MatrixFormat::Dense>::copy(ctx, work.view(), A);

    Matrix<T, MatrixFormat::Dense> U(static_cast<int>(m),
                                     static_cast<int>(svd_u_cols(vectors, m, k)), batch);
    Matrix<T, MatrixFormat::Dense> Vh(static_cast<int>(svd_vh_rows(vectors, n, k)),
                                      static_cast<int>(n), batch);
    UnifiedVector<typename base_type<T>::type> S(static_cast<size_t>(k) *
                                                 static_cast<size_t>(batch));

    ::batchlas::gesvd(ctx, work.view(), S.to_span(), U.view(), Vh.view(),
                      GesvdOptions{.jobu = vectors, .jobvh = vectors});

    // `work` is local scratch gesvd reads and overwrites, and ~Matrix frees its
    // USM without waiting: returning early frees it under enqueued kernels.
    ctx.wait();
    return Svd<T>{std::move(U), std::move(S), std::move(Vh)};
}

template <typename T>
struct Lu {
    Matrix<T, MatrixFormat::Dense> factors;  // L and U packed; L's unit diagonal is implicit
    UnifiedVector<int64_t> pivots;           // rows * batch_size, LAPACK ipiv convention
};

// LU factorisation with partial pivoting. A is not modified. Square only. The
// pivots are a UnifiedVector, not an arena lease as in `solve`: they outlive it.
template <typename T>
inline Lu<T> lu(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A) {
    auto LU = detail::like(A);
    MatrixView<T, MatrixFormat::Dense>::copy(ctx, LU.view(), A);
    UnifiedVector<int64_t> pivots(static_cast<size_t>(A.rows()) *
                                  static_cast<size_t>(A.batch_size()));
    ::batchlas::getrf(ctx, LU.view(), pivots.to_span());
    return Lu<T>{std::move(LU), std::move(pivots)};
}

// linalg::qr is deliberately absent: geqrf + triangular_mask_into + orgqr
// returned Q R != A once an earlier linalg::qr test had run in the same process,
// and passed alone; cause unknown (repro: tests/linalg_layer_tests.cc, 4x).


}  // namespace batchlas::linalg
