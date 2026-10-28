#pragma once

/// @file
/// @brief Matrix functions of symmetric/Hermitian matrices (syfn): \f$ F = V f(\Lambda) V^H \f$.
/// @ingroup api_eigen

#include <batchlas/export.hh>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/queue-dispatch.hh>

namespace batchlas {

/// @brief The scalar function syfn() applies to each eigenvalue \f$ \lambda \f$.
/// @ingroup api_eigen
enum class SyfnKind : int {
    InvSqrt,  ///< \f$ \lambda^{-1/2} \f$; domain \f$ \lambda > 0 \f$ (Loewdin \f$ S^{-1/2} \f$)
    Sqrt,     ///< \f$ \lambda^{1/2} \f$; domain \f$ \lambda \ge 0 \f$
    Inverse,  ///< \f$ \lambda^{-1} \f$; domain \f$ \lambda \ne 0 \f$
    Exp,      ///< \f$ e^{\lambda} \f$; every \f$ \lambda \f$
    Weights,  ///< caller-supplied \f$ f_j \f$ per eigenpair (e.g. Fermi occupations); every \f$ \lambda \f$
};

/// @brief Which function syfn() and syfn_from_eigen() apply: a built-in kind, or per-eigenpair weights.
///
/// With SyfnKind::Weights, @p weights holds one real value per eigenpair, packed like the
/// eigenvalues (`m * batch`, item-major): weight j of item b multiplies column j of item b's
/// eigenvectors. The span must stay valid until the call's event completes. Other kinds ignore it.
/// @ingroup api_eigen
template <typename Real>
struct SyfnFunction {
    SyfnKind kind = SyfnKind::InvSqrt;  ///< the function
    Span<Real> weights{};               ///< `m * batch` weights, read only for SyfnKind::Weights

    /// @brief \f$ \lambda^{-1/2} \f$.
    static SyfnFunction inv_sqrt() { return {SyfnKind::InvSqrt, {}}; }
    /// @brief \f$ \lambda^{1/2} \f$.
    static SyfnFunction sqrt() { return {SyfnKind::Sqrt, {}}; }
    /// @brief \f$ \lambda^{-1} \f$.
    static SyfnFunction inverse() { return {SyfnKind::Inverse, {}}; }
    /// @brief \f$ e^{\lambda} \f$.
    static SyfnFunction exp() { return {SyfnKind::Exp, {}}; }
    /// @brief Caller-supplied weights, `m * batch` values.
    static SyfnFunction with_weights(Span<Real> w) { return {SyfnKind::Weights, w}; }
};

// Signature aliases for explicit instantiation (BATCHLAS_INSTANTIATE in
// src/util/template-instantiations.hh); keep in sync with the declarations below.
namespace sig {
template <typename T>
using syfn = Event(Queue&,
                   const MatrixView<T, MatrixFormat::Dense>&,
                   const MatrixView<T, MatrixFormat::Dense>&,
                   SyfnFunction<typename base_type<T>::type>,
                   Uplo, Span<std::byte>, Span<int32_t>);

template <typename T>
using syfn_buffer_size = size_t(Queue&,
                                const MatrixView<T, MatrixFormat::Dense>&,
                                const MatrixView<T, MatrixFormat::Dense>&,
                                SyfnFunction<typename base_type<T>::type>,
                                Uplo);

template <typename T>
using syfn_from_eigen = Event(Queue&,
                              const MatrixView<T, MatrixFormat::Dense>&,
                              Span<typename base_type<T>::type>,
                              const MatrixView<T, MatrixFormat::Dense>&,
                              SyfnFunction<typename base_type<T>::type>,
                              Span<std::byte>, Span<int32_t>);

template <typename T>
using syfn_from_eigen_buffer_size = size_t(Queue&,
                                           const MatrixView<T, MatrixFormat::Dense>&,
                                           Span<typename base_type<T>::type>,
                                           const MatrixView<T, MatrixFormat::Dense>&,
                                           SyfnFunction<typename base_type<T>::type>);
}  // namespace sig

namespace detail {
template <typename T, typename Real>
inline void syfn_check_common(const char* op, const MatrixView<T, MatrixFormat::Dense>& X,
                              const MatrixView<T, MatrixFormat::Dense>& F, SyfnFunction<Real> fn, int64_t m) {
    const std::string o(op);
    if (X.is_heterogeneous() || F.is_heterogeneous()) {
        throw batchlas::invalid_argument(o + ": heterogeneous batches are not supported");
    }
    if (F.rows() != X.rows() || F.cols() != X.rows() || F.batch_size() != X.batch_size()) {
        throw batchlas::invalid_argument(o + ": F must be " + std::to_string(X.rows()) + "x" +
                                         std::to_string(X.rows()) + " with batch " + std::to_string(X.batch_size()) +
                                         ", got " + std::to_string(F.rows()) + "x" + std::to_string(F.cols()) +
                                         " (batch " + std::to_string(F.batch_size()) + ")");
    }
    if (int64_t(X.rows()) * int64_t(X.rows()) > int64_t(std::numeric_limits<int>::max())) {
        throw batchlas::invalid_argument(o + ": n * n exceeds the int range of the workspace views");
    }
    const int k = static_cast<int>(fn.kind);
    if (k < 0 || k > static_cast<int>(SyfnKind::Weights)) {
        throw batchlas::invalid_argument(o + ": invalid SyfnKind " + std::to_string(k));
    }
    const size_t need = size_t(m) * size_t(X.batch_size());
    if (fn.kind == SyfnKind::Weights && fn.weights.size() < need) {
        throw batchlas::invalid_argument(o + ": weights holds " + std::to_string(fn.weights.size()) +
                                         " values, need " + std::to_string(need));
    }
}
// Address span [first element, one past the last] of a nonempty view; {0, 0} when it addresses none.
template <typename T>
inline std::pair<std::uintptr_t, std::uintptr_t> syfn_storage_span(const MatrixView<T, MatrixFormat::Dense>& X) {
    if (X.rows() == 0 || X.cols() == 0 || X.batch_size() == 0) return {0, 0};
    const auto lo = reinterpret_cast<std::uintptr_t>(X.data_ptr());
    const size_t last = size_t(X.batch_size() - 1) * size_t(X.stride()) + size_t(X.cols() - 1) * size_t(X.ld()) +
                        size_t(X.rows());
    return {lo, lo + last * sizeof(T)};
}

template <typename T>
inline bool syfn_storage_overlaps(const MatrixView<T, MatrixFormat::Dense>& X,
                                  const MatrixView<T, MatrixFormat::Dense>& Y) {
    const auto [xl, xh] = syfn_storage_span(X);
    const auto [yl, yh] = syfn_storage_span(Y);
    return xl != xh && yl != yh && xl < yh && yl < xh;
}
}  // namespace detail

/// @brief Validates the arguments of syfn() and syfn_buffer_size().
///
/// Checks a square A, an n x n F with A's batch size, homogeneous batches, `n * n` in int range,
/// a valid @p uplo and kind, and `n * batch` weights for SyfnKind::Weights.
/// @throws batchlas::invalid_argument if any check fails
/// @ingroup api_eigen_lowlevel
template <typename T>
inline void syfn_validate_params(const MatrixView<T, MatrixFormat::Dense>& A, const MatrixView<T, MatrixFormat::Dense>& F,
                                 SyfnFunction<typename base_type<T>::type> fn, Uplo uplo) {
    if (A.rows() != A.cols()) {
        throw batchlas::invalid_argument("syfn: A must be square, got " + std::to_string(A.rows()) + "x" +
                                         std::to_string(A.cols()));
    }
    if (uplo != Uplo::Lower && uplo != Uplo::Upper) {
        throw batchlas::invalid_argument("syfn: invalid uplo " + std::to_string(static_cast<int>(uplo)));
    }
    detail::syfn_check_common("syfn", A, F, fn, A.rows());
}

/// @brief Validates the arguments of syfn_from_eigen() and syfn_from_eigen_buffer_size().
///
/// Checks `V.cols() <= V.rows()`, `V.cols() * batch` eigenvalues (and weights for
/// SyfnKind::Weights), an n x n F with V's batch size whose storage span (first to last addressed
/// element over the batch) does not overlap V's,
/// homogeneous batches, `n * n` in int range and a valid kind.
/// @throws batchlas::invalid_argument if any check fails
/// @ingroup api_eigen_lowlevel
template <typename T>
inline void syfn_from_eigen_validate_params(const MatrixView<T, MatrixFormat::Dense>& V,
                                            Span<typename base_type<T>::type> eigenvalues,
                                            const MatrixView<T, MatrixFormat::Dense>& F,
                                            SyfnFunction<typename base_type<T>::type> fn) {
    if (V.cols() > V.rows()) {
        throw batchlas::invalid_argument("syfn_from_eigen: V has more columns (" + std::to_string(V.cols()) +
                                         ") than rows (" + std::to_string(V.rows()) + ")");
    }
    detail::syfn_check_common("syfn_from_eigen", V, F, fn, V.cols());
    const size_t need = size_t(V.cols()) * size_t(V.batch_size());
    if (eigenvalues.size() < need) {
        throw batchlas::invalid_argument("syfn_from_eigen: eigenvalues holds " + std::to_string(eigenvalues.size()) +
                                         " values, need V.cols() * batch = " + std::to_string(need));
    }
    if (detail::syfn_storage_overlaps(F, V)) {
        throw batchlas::invalid_argument("syfn_from_eigen: F must not overlap V");
    }
}

/// @brief Batched matrix function \f$ F = V f(\Lambda) V^H \f$ of Hermitian matrices, from A.
///
/// For every batch item: the eigendecomposition \f$ A = V \Lambda V^H \f$ of the @p uplo triangle
/// of A (one syev() on a copy, so A is not modified and F may be A), then
/// syfn_from_eigen() with \f$ (V, \Lambda) \f$. F is written in full and is exactly Hermitian
/// (\f$ F_{ji} = \overline{F_{ij}} \f$ bit for bit, real diagonal).
///
/// Out-of-domain eigenvalues (\f$ \lambda \le 0 \f$ for InvSqrt, \f$ \lambda < 0 \f$ for Sqrt,
/// \f$ \lambda = 0 \f$ for Inverse) contribute nothing: their weight is 0, so F is f applied
/// on the complement of their eigenspace (for Inverse, the pseudo-inverse of an exactly singular
/// item). No threshold is applied: a positive eigenvalue at rounding level is inverted as is.
///
/// Asynchronous: F and @p info are readable after the returned event is waited on.
/// @tparam B  backend
/// @tparam T  float, double, std::complex<float> or std::complex<double>
/// @param ctx       queue the work is enqueued on
/// @param A         batch of n x n Hermitian matrices; only the @p uplo triangle is read
/// @param F         batch of n x n results, written in full; may be A
/// @param fn        the function; SyfnKind::Weights pairs weight j with the j-th smallest eigenvalue;
///                  the weights must be device-accessible (checked unless pointer checks are off)
/// @param uplo      triangle of A that holds the matrix
/// @param workspace at least syfn_buffer_size() bytes
/// @param info      per item (empty = not requested, at least `batch` otherwise): 0 success;
///                  1..n the number of out-of-domain eigenvalues (F defined as above);
///                  \f$ > n \f$ syev() did not converge, `info - n` is syev's status (F unspecified)
/// @return event of the last enqueued kernel
/// @throws batchlas::invalid_argument if syfn_validate_params() rejects the call, @p info is short or the
///         weights are host memory on a GPU queue
/// @throws batchlas::workspace_error if @p workspace is too short
/// @throws batchlas::NoRouteError if syev(), herk()/syrk() or gemm() has no route (vendor-free CPU)
/// @ingroup api_eigen
template <Backend B, typename T>
BATCHLAS_API Event syfn(Queue& ctx,
                        const MatrixView<T, MatrixFormat::Dense>& A,
                        const MatrixView<T, MatrixFormat::Dense>& F,
                        SyfnFunction<typename base_type<T>::type> fn,
                        Uplo uplo,
                        Span<std::byte> workspace,
                        Span<int32_t> info);

/// @brief Workspace, in bytes, that syfn() needs: \f$ 2 n^2 + 2 n \f$ scalars per item plus syev()'s.
/// @throws batchlas::invalid_argument if syfn_validate_params() rejects the operands
/// @throws batchlas::NoRouteError if syev() has no route (it sizes syev's workspace; vendor-free CPU)
/// @ingroup api_eigen_lowlevel
template <Backend B, typename T>
BATCHLAS_API size_t syfn_buffer_size(Queue& ctx,
                                     const MatrixView<T, MatrixFormat::Dense>& A,
                                     const MatrixView<T, MatrixFormat::Dense>& F,
                                     SyfnFunction<typename base_type<T>::type> fn,
                                     Uplo uplo);

/// @brief Batched \f$ F = V \operatorname{diag}(f(w)) V^H \f$ from given eigenpairs.
///
/// The per-iteration form: V is n x m (m <= n, e.g. the occupied orbitals), @p eigenvalues the
/// m values per item that pair with its columns. For InvSqrt, Sqrt and Exp the columns are scaled
/// by \f$ \sqrt{f(w_j)} \f$ and one herk() (real: syrk()) forms the product; for Inverse and
/// Weights, whose values may be negative, by \f$ f(w_j) \f$ and one gemm() with \f$ V^H \f$.
/// Both are public ops that select their own routes. V need not be orthonormal. Out-of-domain
/// values get weight 0, as in syfn(). F is written in full and is exactly Hermitian.
///
/// Asynchronous: F and @p info are readable after the returned event is waited on.
/// @tparam B  backend
/// @tparam T  float, double, std::complex<float> or std::complex<double>
/// @param ctx         queue the work is enqueued on
/// @param V           batch of n x m eigenvectors; read only
/// @param eigenvalues m real values per item, packed (`m * batch`); any order
/// @param F           batch of n x n results, written in full; its storage span must not overlap V's
/// @param fn          the function, or SyfnKind::Weights with `m * batch` device-accessible weights
/// @param workspace   at least syfn_from_eigen_buffer_size() bytes
/// @param info        per item (empty = not requested): 0, or the number of out-of-domain values
/// @return event of the last enqueued kernel
/// @throws batchlas::invalid_argument if syfn_from_eigen_validate_params() rejects the call, @p info is short or the
///         weights are host memory on a GPU queue
/// @throws batchlas::workspace_error if @p workspace is too short
/// @ingroup api_eigen
template <Backend B, typename T>
BATCHLAS_API Event syfn_from_eigen(Queue& ctx,
                                   const MatrixView<T, MatrixFormat::Dense>& V,
                                   Span<typename base_type<T>::type> eigenvalues,
                                   const MatrixView<T, MatrixFormat::Dense>& F,
                                   SyfnFunction<typename base_type<T>::type> fn,
                                   Span<std::byte> workspace,
                                   Span<int32_t> info);

/// @brief Workspace, in bytes, that syfn_from_eigen() needs: \f$ n m + m \f$ scalars per item.
/// @throws batchlas::invalid_argument if syfn_from_eigen_validate_params() rejects the operands
/// @ingroup api_eigen_lowlevel
template <Backend B, typename T>
BATCHLAS_API size_t syfn_from_eigen_buffer_size(Queue& ctx,
                                                const MatrixView<T, MatrixFormat::Dense>& V,
                                                Span<typename base_type<T>::type> eigenvalues,
                                                const MatrixView<T, MatrixFormat::Dense>& F,
                                                SyfnFunction<typename base_type<T>::type> fn);

BATCHLAS_DISPATCH_ON_QUEUE(syfn)
BATCHLAS_DISPATCH_ON_QUEUE(syfn_buffer_size)
BATCHLAS_DISPATCH_ON_QUEUE(syfn_from_eigen)
BATCHLAS_DISPATCH_ON_QUEUE(syfn_from_eigen_buffer_size)

}  // namespace batchlas
