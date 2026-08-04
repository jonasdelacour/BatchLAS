#pragma once

#include <utility>

#include <blas/enums.hh>
#include <blas/matrix.hh>
#include <blas/options.hh>
#include <util/sycl-device-queue.hh>
#include <util/sycl-span.hh>
#include <util/sycl-vector.hh>

// batchlas::linalg -- the convenience layer.
//
// Everything below is a free function; there are no operator overloads. Two
// kinds of thing live here:
//
//   * elementwise operations, which the BLAS/LAPACK surface has no place for;
//   * value-returning wrappers, which allocate their own result so the caller
//     writes `auto C = linalg::matmul(ctx, A, B);` instead of sizing an output,
//     zeroing it, and threading it through as an out-parameter.
//
// The value-returning forms are for code where clarity matters more than
// controlling allocation -- setup, tests, exploration, the Python bindings. In
// an inner loop, keep using the out-parameter forms in `batchlas`, which let the
// caller own and reuse the output.
//
// Every entry point here takes its backend from the Queue and its workspace from
// the queue's arena, so none of them is templated on Backend and none takes a
// workspace.
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
}  // namespace sig

// C = op(A, B), elementwise, for matching shapes. Aliasing C with A or B is
// allowed: each work-item reads and writes one element.
//
// Declared here and instantiated in src/extra/elementwise.cc rather than being
// a header-only template, matching how the rest of the library handles kernels.
template <typename T, BinaryOp Op>
Event elementwise_into(Queue& ctx,
                       const MatrixView<T, MatrixFormat::Dense>& A,
                       const MatrixView<T, MatrixFormat::Dense>& B,
                       const MatrixView<T, MatrixFormat::Dense>& C);

// C = alpha*A + beta*B. Covers scaling (beta = 0) and accumulation.
template <typename T>
Event axpby_into(Queue& ctx,
                 T alpha,
                 const MatrixView<T, MatrixFormat::Dense>& A,
                 T beta,
                 const MatrixView<T, MatrixFormat::Dense>& B,
                 const MatrixView<T, MatrixFormat::Dense>& C);

// A <- alpha * A.
template <typename T>
Event scale(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, T alpha);

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

// C = alpha * op(A) * op(B). The result is allocated here and fully written, so
// it is never read before it is set and needs no zeroing.
template <typename T>
inline Matrix<T, MatrixFormat::Dense> matmul(Queue& ctx,
                                             const MatrixView<T, MatrixFormat::Dense>& A,
                                             const MatrixView<T, MatrixFormat::Dense>& B,
                                             const GemmOptions<T>& opts = {}) {
    const bool ta = opts.transA != Transpose::NoTrans;
    const bool tb = opts.transB != Transpose::NoTrans;
    const auto m = ta ? A.cols() : A.rows();
    const auto n = tb ? B.rows() : B.cols();
    Matrix<T, MatrixFormat::Dense> C(m, n, A.batch_size());
    auto o = opts;
    o.beta = T(0);  // C is fresh; anything else would read uninitialised memory
    gemm(ctx, A, B, C.view(), o);
    return C;
}

// Cholesky factor of A, as a new matrix. A is not modified.
inline constexpr Uplo kDefaultUplo = Uplo::Lower;

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
    // syev overwrites its input, so it gets a copy rather than the caller's A.
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

// Eigenvalues and eigenvectors of a symmetric/Hermitian A. A is not modified;
// the eigenvectors come back in their own matrix.
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

// Solve A X = B for X by LU factorisation. Neither A nor B is modified.
//
// The pivots come from the queue's arena rather than a local UnifiedVector: a
// local would be freed when this function returns, which is before the kernels
// using it have necessarily run.
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

}  // namespace batchlas::linalg
