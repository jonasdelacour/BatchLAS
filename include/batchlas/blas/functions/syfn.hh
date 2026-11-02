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
/// eigenvalues (\f$m \cdot \mathrm{batch}\f$, item-major): weight \f$j\f$ of item \f$b\f$ multiplies column \f$j\f$ of item \f$b\f$'s
/// eigenvectors. The span must stay valid until the call's event completes. Other kinds ignore it.
/// @ingroup api_eigen
template <typename Real>
struct SyfnFunction {
    SyfnKind kind = SyfnKind::InvSqrt;  ///< the function
    Span<Real> weights{};               ///< \f$m \cdot \mathrm{batch}\f$ weights, read only for SyfnKind::Weights

    /// @brief \f$ \lambda^{-1/2} \f$.
    static SyfnFunction inv_sqrt() { return {SyfnKind::InvSqrt, {}}; }
    /// @brief \f$ \lambda^{1/2} \f$.
    static SyfnFunction sqrt() { return {SyfnKind::Sqrt, {}}; }
    /// @brief \f$ \lambda^{-1} \f$.
    static SyfnFunction inverse() { return {SyfnKind::Inverse, {}}; }
    /// @brief \f$ e^{\lambda} \f$.
    static SyfnFunction exp() { return {SyfnKind::Exp, {}}; }
    /// @brief Caller-supplied weights, \f$m \cdot \mathrm{batch}\f$ values.
    /// @param[in] w  \f$m \cdot \mathrm{batch}\f$ real weights, item-major; must stay valid until the call's event completes
    /// @return a SyfnFunction of kind SyfnKind::Weights that reads @p w
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
/// @brief Checks shared by syfn_validate_params() and syfn_from_eigen_validate_params().
///
/// Checks homogeneous batches, an \f$n \times n\f$ F with X's batch size (\f$n\f$ = `X.rows()`),
/// \f$n^2\f$ in int range, a valid kind, and \f$m \cdot \mathrm{batch}\f$ weights for SyfnKind::Weights.
/// @param[in] op  op name that opens each error message
/// @param[in] X   input batch (A, or the eigenvectors V); only its shape is read
/// @param[in] F   output batch; only its shape is read
/// @param[in] fn  the function, and its weights for SyfnKind::Weights
/// @param[in] m   eigenvalues per item: \f$n\f$ for syfn(), `V.cols()` for syfn_from_eigen()
/// @throws batchlas::invalid_argument if any check fails
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
/// @brief Address span of the storage a view addresses over its whole batch.
/// @param[in] X  the view; nothing is dereferenced
/// @return `{first element, one past the last}` as addresses, or `{0, 0}` when X addresses no element
template <typename T>
inline std::pair<std::uintptr_t, std::uintptr_t> syfn_storage_span(const MatrixView<T, MatrixFormat::Dense>& X) {
    if (X.rows() == 0 || X.cols() == 0 || X.batch_size() == 0) return {0, 0};
    const auto lo = reinterpret_cast<std::uintptr_t>(X.data_ptr());
    const size_t last = size_t(X.batch_size() - 1) * size_t(X.stride()) + size_t(X.cols() - 1) * size_t(X.ld()) +
                        size_t(X.rows());
    return {lo, lo + last * sizeof(T)};
}

/// @brief Whether the storage spans of two views intersect (see syfn_storage_span()).
/// @param[in] X  first view; nothing is dereferenced
/// @param[in] Y  second view; nothing is dereferenced
/// @return true when both spans are nonempty and overlap
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
/// Checks a square A, an \f$n \times n\f$ F with A's batch size, homogeneous batches, `n * n` in int range,
/// a valid @p uplo and kind, and \f$n \cdot \mathrm{batch}\f$ weights for SyfnKind::Weights.
/// @param[in] A     batch of \f$n \times n\f$ input matrices
/// @param[in] F     batch of \f$n \times n\f$ output matrices (only the shape is checked)
/// @param[in] fn    the function, and its weights for SyfnKind::Weights
/// @param[in] uplo  triangle of A that holds the matrix
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
/// SyfnKind::Weights), an \f$n \times n\f$ F with V's batch size whose storage span (first to last addressed
/// element over the batch) does not overlap V's,
/// homogeneous batches, `n * n` in int range and a valid kind.
/// @param[in] V            batch of \f$n \times m\f$ eigenvectors
/// @param[in] eigenvalues  \f$m\f$ real eigenvalues per item, packed (\f$m \cdot \mathrm{batch}\f$)
/// @param[in] F            batch of \f$n \times n\f$ output matrices (only shape and overlap are checked)
/// @param[in] fn           the function, and its weights for SyfnKind::Weights
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
/// \f$ \lambda = 0 \f$ for Inverse) contribute nothing: their weight is 0, so F is \f$f\f$ applied
/// on the complement of their eigenspace (for Inverse, the pseudo-inverse of an exactly singular
/// item). No threshold is applied: a positive eigenvalue at rounding level is inverted as is.
///
/// Asynchronous: F and @p info are readable after the returned event is waited on.
/// @tparam B  backend
/// @tparam T  float, double, std::complex<float> or std::complex<double>
/// @param[in]  ctx       queue the work is enqueued on
/// @param[in]  A         batch of \f$n \times n\f$ Hermitian matrices; only the @p uplo triangle is read; A is not modified unless @p F is A
/// @param[out] F         batch of \f$n \times n\f$ results \f$f(A)\f$, written in full; may be A
/// @param[in]  fn        the function; SyfnKind::Weights pairs weight \f$j\f$ with the \f$j\f$-th smallest eigenvalue;
///                       the weights must be device-accessible (checked unless pointer checks are off)
/// @param[in]  uplo      triangle of A that holds the matrix
/// @param[out] workspace scratch of at least syfn_buffer_size() bytes; contents undefined on return
/// @param[out] info      per item (empty = not requested, at least `batch` otherwise): 0 success;
///                       \f$1, \ldots, n\f$ the number of out-of-domain eigenvalues (F defined as above);
///                       \f$ > n \f$ syev() did not converge, `info - n` is syev's status (F unspecified)
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
/// @param[in] ctx   queue the call will run on
/// @param[in] A     batch of \f$n \times n\f$ Hermitian matrices, as passed to syfn()
/// @param[in] F     batch of \f$n \times n\f$ results, as passed to syfn()
/// @param[in] fn    the function, as passed to syfn()
/// @param[in] uplo  triangle of A that holds the matrix
/// @return workspace size in bytes
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
/// The per-iteration form: V is \f$n \times m\f$ (\f$m \le n\f$, e.g. the occupied orbitals), @p eigenvalues the
/// \f$m\f$ values per item that pair with its columns. For InvSqrt, Sqrt and Exp the columns are scaled
/// by \f$ \sqrt{f(w_j)} \f$ and one herk() (real: syrk()) forms the product; for Inverse and
/// Weights, whose values may be negative, by \f$ f(w_j) \f$ and one gemm() with \f$ V^H \f$.
/// Both are public ops that select their own routes. V need not be orthonormal. Out-of-domain
/// values get weight 0, as in syfn(). F is written in full and is exactly Hermitian.
///
/// Asynchronous: F and @p info are readable after the returned event is waited on.
/// @tparam B  backend
/// @tparam T  float, double, std::complex<float> or std::complex<double>
/// @param[in]  ctx         queue the work is enqueued on
/// @param[in]  V           batch of \f$n \times m\f$ eigenvectors; read only
/// @param[in]  eigenvalues \f$m\f$ real values \f$w_j\f$ per item, packed (\f$m \cdot \mathrm{batch}\f$); any order
/// @param[out] F           batch of \f$n \times n\f$ results, written in full; its storage span must not overlap V's
/// @param[in]  fn          the function, or SyfnKind::Weights with \f$m \cdot \mathrm{batch}\f$ device-accessible weights
/// @param[out] workspace   scratch of at least syfn_from_eigen_buffer_size() bytes; contents undefined on return
/// @param[out] info        per item (empty = not requested, at least `batch` otherwise): 0, or the number of out-of-domain values
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
/// @param[in] ctx          queue the call will run on
/// @param[in] V            batch of \f$n \times m\f$ eigenvectors, as passed to syfn_from_eigen()
/// @param[in] eigenvalues  \f$m\f$ real values per item, packed (\f$m \cdot \mathrm{batch}\f$)
/// @param[in] F            batch of \f$n \times n\f$ results, as passed to syfn_from_eigen()
/// @param[in] fn           the function, as passed to syfn_from_eigen()
/// @return workspace size in bytes
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
