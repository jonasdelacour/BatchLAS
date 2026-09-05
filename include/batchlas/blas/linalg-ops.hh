#pragma once

#include <algorithm>
#include <stdexcept>
#include <utility>

#include <batchlas/blas/enums.hh>
// extensions.hh (inv) and extra.hh (norm, transpose, cond) are needed here, not
// merely by whoever includes this header: the forwarding wrappers below name
// their targets with a QUALIFIED ::batchlas:: id, and qualified lookup happens
// at template-definition context, so the declarations have to be visible now.
// Neither costs measurable parse time -- both are declaration-only.
#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/extra.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/options.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>

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
// The rule that decides which namespace an operation belongs in:
//
//   value-returning, backend from the Queue, workspace from the arena => linalg::
//   out-parameter, workspace yours                                    => batchlas::
//
// So every entry point here takes its backend from the Queue and its workspace
// from the queue's arena; none of them is templated on Backend and none takes a
// workspace. Several are one-line forwards to a `batchlas::` entry point that
// already satisfies the rule -- they are here so that a caller reaching for the
// convenience layer finds the whole of it in one namespace, rather than having
// to know which of two namespaces a given convenience form happens to live in.
//
// Three exceptions, each documented again at the entry point:
//
//   * `norm` and `cond` WAIT. Their `batchlas::` implementations call .wait()
//     internally (src/extra/norm.cc, src/extra/cond.cc), so unlike everything
//     else here they do not merely enqueue. Fixing that belongs with those
//     implementations, not with a wrapper that would hide it.
//   * `svd` WAITS, for a different reason: it holds local scratch the caller
//     never sees, and a Matrix's destructor frees its USM without waiting, so
//     returning early would hand those pages back underneath kernels that have
//     only been enqueued. The wait is what keeps the scratch alive; it is not a
//     synchronisation convenience. `eigvalsh` (`work`) and `solve` (`LU`) have
//     the same shape and no wait -- they survive on a shorter kernel chain
//     rather than by construction, and want the same treatment or an
//     arena-owned scratch. Left as they were so this change stays scoped to
//     what it verifies.
//   * `transpose` is real-only. transpose_impl moves data without conjugating,
//     so the complex instantiations are deliberately switched off in
//     src/extra/transpose.cc rather than hand callers a non-conjugating
//     "transpose" under a name most read as the adjoint.
//   * `cond` is float/double only, for the same instantiation reason.
//
// The two scalar-type restrictions are spelled as `requires` clauses so that a
// rejected type is a compile error naming the constraint, rather than an
// undefined symbol at link time -- which is what the unconstrained wrapper
// would have produced, and the worst place to learn it.
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

// C = A with everything outside the requested triangle set to zero. `k` follows
// NumPy: k = 0 keeps the main diagonal, k > 0 moves the boundary toward the
// upper right, k < 0 toward the lower left. Aliasing C with A is allowed: each
// work-item reads and writes one element.
//
// This is a *mask* over an existing matrix, which is what MatrixView's
// fill_triangular does not do -- that one generates a triangular matrix from
// scalars. Masking is what lets an in-place LAPACK factorisation be split into
// its factors -- recovering R from a geqrf result before orgqr overwrites it is
// what this was written for. See the note further down about why the qr wrapper
// that would have used it is not here.
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

// matmul's own options -- deliberately NOT GemmOptions. matmul allocates C, so
// `beta` has no meaning here: any non-zero value would read uninitialised
// memory, and the wrapper used to overwrite the field silently, so
// `{.alpha = 2, .beta = 1}` compiled and quietly computed 2*A*B. Omitting the
// field makes naming it a compile error instead.
//
// This lives here, not in blas/options.hh, because it is a linalg-layer type:
// nothing in batchlas:: takes it.
template <typename T>
struct MatmulOptions {
    T alpha = T(1);
    Transpose transA = Transpose::NoTrans;
    Transpose transB = Transpose::NoTrans;
    ComputePrecision precision = ComputePrecision::Default;
};

// C = alpha * op(A) * op(B). The result is allocated here and fully written, so
// it is never read before it is set and needs no zeroing.
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
    // The designated initialisers are in GemmOptions' declaration order, which
    // C++20 requires; a field reorder there is a compile error here, not a
    // silent argument swap.
    gemm(ctx, A, B, C.view(),
         GemmOptions<T>{.alpha = opts.alpha,
                        .beta = T(0),  // C is fresh; anything else would read uninitialised memory
                        .transA = opts.transA,
                        .transB = opts.transB,
                        .precision = opts.precision});
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

// A with everything below (triu) or above (tril) the k-th diagonal zeroed. A is
// not modified.
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
//
// Every call below is EXPLICITLY qualified with ::batchlas::. batchlas::linalg
// is nested inside batchlas, so unqualified lookup finds the linalg:: name
// first and stops: `return inv(ctx, A);` written here is infinite recursion,
// and there is no diagnostic for it -- the call matches, so nothing is
// ill-formed. The wrappers above get away with unqualified `gemm`/`getrf`/
// `syev` only because no linalg:: name shadows those.

// A^-1, as a new matrix. A is not modified. Square only (getrf's precondition).
template <typename T>
inline Matrix<T, MatrixFormat::Dense> inv(Queue& ctx,
                                          const MatrixView<T, MatrixFormat::Dense>& A) {
    return ::batchlas::inv(ctx, A);
}

// Plain transpose, NOT the conjugate transpose. Real scalars only: the complex
// instantiations of ::batchlas::transpose are commented out in
// src/extra/transpose.cc because transpose_impl moves data without conjugating,
// so a complex call would either be an undefined symbol at link time or -- if
// those lines were uncommented -- silently compute a non-conjugating adjoint.
// The constraint turns the first failure into a compile error and refuses to
// make the second one possible.
template <typename T>
    requires RealScalar<T>
inline Matrix<T, MatrixFormat::Dense> transpose(Queue& ctx,
                                                const MatrixView<T, MatrixFormat::Dense>& A) {
    return ::batchlas::transpose<T, MatrixFormat::Dense>(ctx, A);
}

// One norm per batch item. THIS ONE WAITS: the value-returning
// ::batchlas::norm calls .wait() internally (src/extra/norm.cc), so unlike the
// rest of this header it has returned only once the result is readable. Use the
// out-parameter ::batchlas::norm(ctx, A, norm_type, norms) to stay asynchronous.
template <typename T>
inline UnifiedVector<typename base_type<T>::type> norm(
        Queue& ctx,
        const MatrixView<T, MatrixFormat::Dense>& A,
        NormType norm_type = NormType::Frobenius) {
    return ::batchlas::norm<T, MatrixFormat::Dense>(ctx, A, norm_type);
}

// Condition number per batch item, as ||A|| * ||A^-1|| (or the eigenvalue ratio
// for NormType::Spectral). WAITS, for the same reason norm does
// (src/extra/cond.cc).
//
// Real scalars only. ::batchlas::cond is instantiated for float and double on
// Dense alone (COND_INSTANTIATE in src/extra/cond.cc) because there is no
// complex implementation behind it; without the constraint a complex call would
// compile here and fail at link time with an undefined symbol, which is the
// worst place to learn it.
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
// so it gets a copy.
//
// `vectors` sizes U and Vh through the same helpers the shape check uses
// (svd_u_cols/svd_vh_rows in blas/enums.hh), so the two cannot disagree.
// SvdVectors::None is rejected here rather than silently returning empty
// matrices: a values-only spelling would need default-constructed views for U
// and Vh, and that path is not exercised by this layer.
//
// Complex input can throw std::runtime_error rather than fail to compile: the
// blocked provider's complex native path is not implemented
// (src/extensions/gesvd_blocked.cc) and the cta provider rejects
// max(m, n) > 32, so whether a given complex shape is served depends on the
// dispatch heuristic and on BATCHLAS_GESVD_PROVIDER.
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

    // `work` is local scratch that gesvd both reads and overwrites, and a
    // Matrix's destructor frees its USM without waiting -- see the longer note
    // in qr() for the failure this produces once the queue is under load.
    ctx.wait();
    return Svd<T>{std::move(U), std::move(S), std::move(Vh)};
}

template <typename T>
struct Lu {
    Matrix<T, MatrixFormat::Dense> factors;  // L and U packed; L's unit diagonal is implicit
    UnifiedVector<int64_t> pivots;           // rows * batch_size, LAPACK ipiv convention
};

// LU factorisation with partial pivoting. A is not modified. Square only, which
// is getrf's own precondition.
//
// The pivots are a UnifiedVector rather than an arena lease -- the deliberate
// opposite of `solve` above. solve's pivots die with the call, so a lease is
// right; these are returned to the caller, so they have to outlive it.
template <typename T>
inline Lu<T> lu(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A) {
    auto LU = detail::like(A);
    MatrixView<T, MatrixFormat::Dense>::copy(ctx, LU.view(), A);
    UnifiedVector<int64_t> pivots(static_cast<size_t>(A.rows()) *
                                  static_cast<size_t>(A.batch_size()));
    ::batchlas::getrf(ctx, LU.view(), pivots.to_span());
    return Lu<T>{std::move(LU), std::move(pivots)};
}

// linalg::qr is deliberately absent.
//
// It was implemented (geqrf, mask R out of the factored array with
// triangular_mask_into, then orgqr for Q) and it is WRONG under conditions the
// test suite reaches: with another linalg::qr-using test having run earlier in
// the same process, Q R != A by four orders of magnitude, at the same value
// every time, in 2-4 of 4 concurrent runs of the test binary -- and it passes
// every time the test runs alone or the suite runs serially.
//
// What it is NOT: the answer is deterministic rather than random, so it is not
// reading freed pages; zero-filling Q and R before use does not change it, so it
// is not reading them uninitialised; and draining the queue before the local
// scratch dies does not change it either. It looks like arena state left by an
// earlier call, but that was not established, and shipping a factorisation that
// returns confident wrong numbers is worse than not shipping one.
//
// triangular_mask_into / triu / tril below were built for it and are correct and
// tested on their own, so whoever picks this up has the piece that was missing.
// Reproducer: tests/linalg_layer_tests.cc, run the binary four times at once.


}  // namespace batchlas::linalg
