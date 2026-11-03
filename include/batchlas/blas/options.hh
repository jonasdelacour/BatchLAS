// SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <concepts>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <string>

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/queue-dispatch.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>

/// @file
/// @brief Option-struct spellings of the dense BLAS and LAPACK-style entry points.
/// @ingroup api_options

/**
 * @addtogroup api_options
 * @details
 * Each entry point here takes its non-matrix arguments as one designated-initialiser
 * struct, so a call names only what differs from the defaults:
 * @code
 * gemm(ctx, A, B, C, {.alpha = 2.0f, .transA = Transpose::Trans});
 * syev(ctx, A, W, {.jobz = JobType::NoEigenVectors});
 * @endcode
 * Every overload comes twice: `f(ctx, ...)` takes the backend from the Queue and
 * checks that pointer arguments are device-accessible; `f<Backend>(ctx, ...)` fixes
 * it at compile time. `T` is always deduced from the matrix arguments, never from
 * the option struct. Matrix arguments may be `Matrix` or `MatrixView`.
 *
 * The LAPACK-style calls (`potrf`, `getrf`, `getrs`, `getri`, `geqrf`, `orgqr`,
 * `syev`, `ormqr`, `gesvd`) come with and without a `Span<std::byte> ws` argument.
 * Without it, scratch is leased from the Queue's arena, sized by the matching
 * `*_buffer_size`; on an out-of-order queue releasing the lease drains the queue,
 * so pass your own span there to stay asynchronous. Every option overload, both
 * spellings, with and without `ws`, checks shapes (squareness, matching rows and
 * batch sizes, minimum span lengths) and throws `batchlas::invalid_argument` on a
 * violation; only the deducing spelling also checks pointers. The ordinary spelling
 * with non-const lvalue arguments can resolve to the queue-dispatch overload, which
 * lands on the `f<Backend>` form, so that form must check too.
 *
 * Every `uplo` defaults to `Uplo::Lower`. With an explicit workspace, write
 * `PotrfOptions{}`, never a bare `{}`.
 *
 * The `*Params` structs of `blas/extensions.hh` are ordinary trailing arguments,
 * not option structs; see @ref md_docs_2cpp-api. Design notes:
 * @ref design_api_conventions.
 */

namespace batchlas {

namespace detail {

/// @brief Named counterpart of require_pack_accessible: the error says "gemm: A".
///
/// A no-op when pointer checks are disabled.
/// @param[in] ctx    queue whose device must be able to access each argument
/// @param[in] fn     op name that opens the error message
/// @param[in] names  one name per element of @p args, in the same order
/// @param[in] args   the arguments whose pointers are checked
/// @throws batchlas::invalid_argument naming the first inaccessible argument
template <typename... Args>
inline void require_args_accessible(const Queue& ctx, const char* fn,
                                    const char* const* names, const Args&... args) {
    if (!pointer_checks_enabled()) return;
    int i = 0;
    (void)std::initializer_list<int>{
        (require_arg_accessible(ctx, args, std::string(fn) + ": " + names[i++]), 0)...};
}

}  // namespace detail

// Check the pointer arguments of an entry point, naming the offending one.
#define BATCHLAS_CHECK_ARGS(CTX, FN, ...)                                        \
    do {                                                                         \
        static const char* const _bl_names[] = {BATCHLAS_ARG_NAMES(__VA_ARGS__)}; \
        ::batchlas::detail::require_args_accessible((CTX), FN, _bl_names, __VA_ARGS__); \
    } while (0)

#define BATCHLAS_ARG_NAMES_1(a) #a
#define BATCHLAS_ARG_NAMES_2(a, b) #a, #b
#define BATCHLAS_ARG_NAMES_3(a, b, c) #a, #b, #c
#define BATCHLAS_ARG_NAMES_4(a, b, c, d) #a, #b, #c, #d
#define BATCHLAS_ARG_NAMES_5(a, b, c, d, e) #a, #b, #c, #d, #e
#define BATCHLAS_ARG_NAMES_6(a, b, c, d, e, f) #a, #b, #c, #d, #e, #f
#define BATCHLAS_ARG_NAMES_PICK(_1, _2, _3, _4, _5, _6, NAME, ...) NAME
#define BATCHLAS_ARG_NAMES(...)                                                  \
    BATCHLAS_ARG_NAMES_PICK(__VA_ARGS__, BATCHLAS_ARG_NAMES_6, BATCHLAS_ARG_NAMES_5, \
                            BATCHLAS_ARG_NAMES_4, BATCHLAS_ARG_NAMES_3,          \
                            BATCHLAS_ARG_NAMES_2, BATCHLAS_ARG_NAMES_1)(__VA_ARGS__)

namespace detail {

// Shape preconditions; invalid_argument (caller error), never runtime_error. They
// are repeated on BOTH the deducing and the <Backend> overloads on purpose: the
// variadic dispatch overload binds prvalues better and skips the deducing one.
// Positional primaries stay unchecked (library code under src/ops/ calls them per iteration).
// evidence: docs/design/api-conventions.md#api-conventions-shape-checks-live-on-both-option-overloads
/// @brief Throws unless a matrix is square.
/// @param[in] fn    op name that opens the error message
/// @param[in] name  argument name used in the message
/// @param[in] A     the matrix (view) whose shape is checked
/// @throws batchlas::invalid_argument if `A.rows() != A.cols()`
template <typename MV>
inline void require_square(const char* fn, const char* name, const MV& A) {
    if (A.rows() != A.cols())
        throw batchlas::invalid_argument(std::string(fn) + ": " + name +
            " must be square, got " + std::to_string(A.rows()) + "x" + std::to_string(A.cols()));
}

/// @brief Throws unless two matrices have the same number of rows.
/// @param[in] fn  op name that opens the error message
/// @param[in] an  name of @p A used in the message
/// @param[in] A   first matrix (view)
/// @param[in] bn  name of @p B used in the message
/// @param[in] B   second matrix (view)
/// @throws batchlas::invalid_argument if `A.rows() != B.rows()`
template <typename MA, typename MB>
inline void require_same_rows(const char* fn, const char* an, const MA& A,
                              const char* bn, const MB& B) {
    if (A.rows() != B.rows())
        throw batchlas::invalid_argument(std::string(fn) + ": " + an + ".rows() (" +
            std::to_string(A.rows()) + ") must equal " + bn + ".rows() (" +
            std::to_string(B.rows()) + ")");
}

/// @brief Throws unless two matrices have the same batch size.
/// @param[in] fn  op name that opens the error message
/// @param[in] an  name of @p A used in the message
/// @param[in] A   first matrix (view)
/// @param[in] bn  name of @p B used in the message
/// @param[in] B   second matrix (view)
/// @throws batchlas::invalid_argument if `A.batch_size() != B.batch_size()`
template <typename MA, typename MB>
inline void require_same_batch(const char* fn, const char* an, const MA& A,
                               const char* bn, const MB& B) {
    if (A.batch_size() != B.batch_size())
        throw batchlas::invalid_argument(std::string(fn) + ": " + an + " and " + bn +
            " must have the same batch size (" + std::to_string(A.batch_size()) + " vs " +
            std::to_string(B.batch_size()) + ")");
}

// `>=`, not `==`: a caller may slice one big pivot/tau arena across several calls.
/// @brief Throws unless a span holds at least @p need elements.
/// @param[in] fn    op name that opens the error message
/// @param[in] name  span name used in the message
/// @param[in] have  the span's size
/// @param[in] need  the minimum size
/// @throws batchlas::invalid_argument if `have < need`
inline void require_span_at_least(const char* fn, const char* name, size_t have, size_t need) {
    if (have < need)
        throw batchlas::invalid_argument(std::string(fn) + ": " + name + " holds " +
            std::to_string(have) + " elements, needs at least " + std::to_string(need));
}

// Empty `info` = "no status wanted". A too-short one would fail silently (stale
// zeros read as "all factorised"), hence the check.
/// @brief Throws when a nonempty `info` span is shorter than the batch.
/// @param[in] fn          op name that opens the error message
/// @param[in] have        the `info` span's size; 0 skips the check
/// @param[in] batch_size  the batch size it must cover
/// @throws batchlas::invalid_argument if `0 < have < batch_size`
inline void require_info_span(const char* fn, size_t have, size_t batch_size) {
    if (have != 0) require_span_at_least(fn, "info", have, batch_size);
}

}  // namespace detail

/// @addtogroup api_options
/// @{

// ---- dense BLAS ------------------------------------------------------------

/// @brief Options for gemm(): \f$C := \alpha\,\mathrm{op}(A)\,\mathrm{op}(B) + \beta C\f$.
template <typename T>
struct GemmOptions {
    T alpha = T(1);                                          ///< scale of the product
    T beta = T(0);                                           ///< scale of the incoming C; \f$\beta = 0\f$ means C is not read
    Transpose transA = Transpose::NoTrans;                   ///< \f$\mathrm{op}(A)\f$
    Transpose transB = Transpose::NoTrans;                   ///< \f$\mathrm{op}(B)\f$
    ComputePrecision precision = ComputePrecision::Default;  ///< internal compute type; Default = the input type
};

/// @brief Options for gemv(): \f$y := \alpha\,\mathrm{op}(A)\,x + \beta y\f$.
template <typename T>
struct GemvOptions {
    T alpha = T(1);                         ///< scale of the product
    T beta = T(0);                          ///< scale of the incoming y
    Transpose transA = Transpose::NoTrans;  ///< \f$\mathrm{op}(A)\f$
};

/// @brief Options for symm(): \f$C := \alpha A B + \beta C\f$ (Left) or \f$\alpha B A + \beta C\f$ (Right), A symmetric.
template <typename T>
struct SymmOptions {
    T alpha = T(1);           ///< scale of the product
    T beta = T(0);            ///< scale of the incoming C
    Side side = Side::Left;   ///< which side A multiplies from
    Uplo uplo = Uplo::Lower;  ///< triangle of A that is read
};

/// @brief Options for hemm(): as SymmOptions, with A Hermitian.
template <typename T>
struct HemmOptions {
    T alpha = T(1);           ///< scale of the product
    T beta = T(0);            ///< scale of the incoming C
    Side side = Side::Left;   ///< which side A multiplies from
    Uplo uplo = Uplo::Lower;  ///< triangle of A that is read
};

/// @brief Options for herk(): \f$C := \alpha\,\mathrm{op}(A)\,\mathrm{op}(A)^H + \beta C\f$.
///
/// `alpha` and `beta` are real, as the result must stay Hermitian.
template <typename T>
struct HerkOptions {
    float_t<T> alpha = float_t<T>(1);     ///< real scale of the product
    float_t<T> beta = float_t<T>(0);      ///< real scale of the incoming C
    Uplo uplo = Uplo::Lower;              ///< triangle of C that is written
    Transpose trans = Transpose::NoTrans; ///< NoTrans (\f$AA^H\f$) or ConjTrans (\f$A^HA\f$)
};

/// @brief Options for her2k(): \f$C := \alpha\,\mathrm{op}(A)\,\mathrm{op}(B)^H + \bar\alpha\,\mathrm{op}(B)\,\mathrm{op}(A)^H + \beta C\f$.
///
/// `alpha` is complex and `beta` real: the pair is Hermitian for any `alpha`.
template <typename T>
struct Her2kOptions {
    T alpha = T(1);                       ///< complex scale
    float_t<T> beta = float_t<T>(0);      ///< real scale of the incoming C
    Uplo uplo = Uplo::Lower;              ///< triangle of C that is written
    Transpose trans = Transpose::NoTrans; ///< NoTrans or ConjTrans
};

/// @brief Options for syrk(): \f$C := \alpha\,\mathrm{op}(A)\,\mathrm{op}(A)^T + \beta C\f$.
template <typename T>
struct SyrkOptions {
    T alpha = T(1);                       ///< scale of the product
    T beta = T(0);                        ///< scale of the incoming C
    Uplo uplo = Uplo::Lower;              ///< triangle of C that is written
    Transpose trans = Transpose::NoTrans; ///< NoTrans (\f$AA^T\f$) or Trans (\f$A^TA\f$)
};

/// @brief Options for syr2k(): \f$C := \alpha(\mathrm{op}(A)\,\mathrm{op}(B)^T + \mathrm{op}(B)\,\mathrm{op}(A)^T) + \beta C\f$.
template <typename T>
struct Syr2kOptions {
    T alpha = T(1);                       ///< scale of the product pair
    T beta = T(0);                        ///< scale of the incoming C
    Uplo uplo = Uplo::Lower;              ///< triangle of C that is written
    Transpose trans = Transpose::NoTrans; ///< NoTrans or Trans
};

/// @brief Options for trmm(): \f$C := \alpha\,\mathrm{op}(A)\,B\f$ (Left) or \f$\alpha B\,\mathrm{op}(A)\f$ (Right). No `beta`.
template <typename T>
struct TrmmOptions {
    T alpha = T(1);                       ///< scale of the product
    Side side = Side::Left;               ///< which side \f$\mathrm{op}(A)\f$ multiplies from
    Uplo uplo = Uplo::Lower;              ///< which triangle A is
    Transpose trans = Transpose::NoTrans; ///< \f$\mathrm{op}(A)\f$
    Diag diag = Diag::NonUnit;            ///< Unit: A's diagonal is taken as ones and not read
};

/// @brief Options for trsm(): solve \f$\mathrm{op}(A)X = \alpha B\f$ (Left) or \f$X\,\mathrm{op}(A) = \alpha B\f$ (Right).
template <typename T>
struct TrsmOptions {
    T alpha = T(1);                       ///< scale of the right-hand side
    Side side = Side::Left;               ///< which side \f$\mathrm{op}(A)\f$ is on
    Uplo uplo = Uplo::Lower;              ///< which triangle A is
    Transpose trans = Transpose::NoTrans; ///< \f$\mathrm{op}(A)\f$
    Diag diag = Diag::NonUnit;            ///< Unit: A's diagonal is taken as ones and not read
};

/// @}

// Library code under src/ops/ is templated on Backend and must call the `f<B>(ctx, ...)`
// spelling: `f(ctx, ...)` would silently use ctx.backend() instead of B.

namespace detail {
template <typename M>
struct dense_scalar {};
template <typename T>
struct dense_scalar<MatrixView<T, MatrixFormat::Dense>> {
    using type = T;
};
template <typename T>
struct dense_scalar<Matrix<T, MatrixFormat::Dense>> {
    using type = T;
};
template <typename M>
using dense_scalar_t = typename dense_scalar<std::remove_cvref_t<M>>::type;

template <typename M>
concept DenseMatrixLike = requires { typename dense_scalar<std::remove_cvref_t<M>>::type; };

}  // namespace detail

// Never collapse the two workspace spellings into `Span<std::byte> ws = {}` plus a
// null check: sizing passes hand out empty spans over live matrices.
// evidence: docs/design/api-conventions.md#api-conventions-two-workspace-spellings-never-a-defaulted-span

#define BATCHLAS_DENSE_VIEW(T) MatrixView<T, MatrixFormat::Dense>

/// @addtogroup api_options
/// @{

// ---- dense BLAS ------------------------------------------------------------

/// @brief gemm with a compile-time backend; same contract as the Queue-deducing
///        overload, without the pointer check.
/// @tparam Back  backend to run on
/// @param[in]     ctx   queue the kernels are enqueued on
/// @param[in]     A     batch; \f$\mathrm{op}(A_i)\f$ is \f$m \times k\f$
/// @param[in]     B     batch; \f$\mathrm{op}(B_i)\f$ is \f$k \times n\f$
/// @param[in,out] C     batch of \f$m \times n\f$ matrices; read when \f$\beta \ne 0\f$, overwritten
/// @param[in]     opts  alpha, beta, transA, transB, precision
/// @return        event of the last enqueued kernel
template <Backend Back, detail::DenseMatrixLike MA, detail::DenseMatrixLike MB,
          detail::DenseMatrixLike MC, typename T = detail::dense_scalar_t<MA>>
inline Event gemm(Queue& ctx, const MA& A, const MB& B, const MC& C, const GemmOptions<T>& opts) {
    using V = BATCHLAS_DENSE_VIEW(T);
    return gemm<Back, T>(ctx, V(A), V(B), V(C), opts.alpha, opts.beta, opts.transA, opts.transB,
                         opts.precision);
}

/// @brief Batched matrix product \f$C_i := \alpha\,\mathrm{op}(A_i)\,\mathrm{op}(B_i) + \beta C_i\f$.
///
/// A batch whose items carry differing active extents is handled natively.
/// @tparam MA,MB,MC  `Matrix` or `MatrixView`, dense; `T` is deduced from `A`
/// @param[in]     ctx   queue; supplies the backend
/// @param[in]     A     \f$\mathrm{op}(A)\f$ is \f$m \times k\f$ per item
/// @param[in]     B     \f$\mathrm{op}(B)\f$ is \f$k \times n\f$ per item
/// @param[in,out] C     \f$m \times n\f$ per item; read when \f$\beta \ne 0\f$, overwritten
/// @param[in]     opts  alpha, beta, transA, transB, precision
/// @return        event of the last enqueued kernel
/// @pre   all operands have the same batch size
/// @throws std::invalid_argument if a pointer is not reachable from the queue's device
template <detail::DenseMatrixLike MA, detail::DenseMatrixLike MB, detail::DenseMatrixLike MC,
          typename T = detail::dense_scalar_t<MA>>
inline Event gemm(Queue& ctx, const MA& A, const MB& B, const MC& C, const GemmOptions<T>& opts) {
    BATCHLAS_CHECK_ARGS(ctx, "gemm", A, B, C);
    return with_backend_for<T>(ctx, "gemm",
                               [&](auto Back) { return gemm<Back.value>(ctx, A, B, C, opts); });
}

/// @brief gemv with a compile-time backend; same contract as the Queue-deducing
///        overload, without the pointer check.
/// @tparam Back  backend to run on
/// @param[in]     ctx   queue the kernels are enqueued on
/// @param[in]     A     batch; \f$\mathrm{op}(A_i)\f$ is \f$m \times n\f$
/// @param[in]     x     batch of vectors of length \f$n\f$
/// @param[in,out] y     batch of vectors of length \f$m\f$; input scaled by \f$\beta\f$, overwritten
/// @param[in]     opts  alpha, beta, transA
/// @return        event of the last enqueued kernel
template <Backend Back, detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
inline Event gemv(Queue& ctx, const MA& A, const VectorView<T>& x, const VectorView<T>& y,
                  const GemvOptions<T>& opts) {
    using V = BATCHLAS_DENSE_VIEW(T);
    return gemv<Back, T>(ctx, V(A), x, y, opts.alpha, opts.beta, opts.transA);
}

/// @brief Batched matrix-vector product \f$y_i := \alpha\,\mathrm{op}(A_i)\,x_i + \beta y_i\f$.
/// @param[in]     ctx   queue; supplies the backend
/// @param[in]     A     \f$\mathrm{op}(A)\f$ is \f$m \times n\f$ per item
/// @param[in]     x     length \f$n\f$ per item
/// @param[in,out] y     length \f$m\f$ per item; input scaled by \f$\beta\f$, overwritten
/// @param[in]     opts  alpha, beta, transA
/// @return        event of the last enqueued kernel
/// @throws std::invalid_argument if a pointer is not reachable from the queue's device
template <detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
inline Event gemv(Queue& ctx, const MA& A, const VectorView<T>& x, const VectorView<T>& y,
                  const GemvOptions<T>& opts) {
    BATCHLAS_CHECK_ARGS(ctx, "gemv", A, x, y);
    return with_backend_for<T>(ctx, "gemv",
                               [&](auto Back) { return gemv<Back.value>(ctx, A, x, y, opts); });
}

/// @brief symm with a compile-time backend; same contract as the Queue-deducing
///        overload, without the pointer check.
/// @tparam Back  backend to run on
/// @param[in]     ctx   queue the kernels are enqueued on
/// @param[in]     A     symmetric, \f$m \times m\f$ (Left) or \f$n \times n\f$ (Right); only the `uplo` triangle is read
/// @param[in]     B     batch of \f$m \times n\f$ matrices
/// @param[in,out] C     batch of \f$m \times n\f$ matrices; input scaled by \f$\beta\f$, overwritten
/// @param[in]     opts  alpha, beta, side, uplo
/// @return        event of the last enqueued kernel
template <Backend Back, detail::DenseMatrixLike MA, detail::DenseMatrixLike MB,
          detail::DenseMatrixLike MC, typename T = detail::dense_scalar_t<MA>>
inline Event symm(Queue& ctx, const MA& A, const MB& B, const MC& C, const SymmOptions<T>& opts) {
    using V = BATCHLAS_DENSE_VIEW(T);
    return symm<Back, T>(ctx, V(A), V(B), V(C), opts.alpha, opts.beta, opts.side, opts.uplo);
}

/// @brief Batched symmetric product \f$C := \alpha AB + \beta C\f$ (Left) or \f$\alpha BA + \beta C\f$ (Right).
/// @param[in]     ctx   queue; supplies the backend
/// @param[in]     A     symmetric, \f$m \times m\f$ (Left) or \f$n \times n\f$ (Right); only the `uplo` triangle is read
/// @param[in]     B     \f$m \times n\f$ per item
/// @param[in,out] C     \f$m \times n\f$ per item; input scaled by \f$\beta\f$, overwritten
/// @param[in]     opts  alpha, beta, side, uplo
/// @return        event of the last enqueued kernel
/// @note  Complex `T` is complex-symmetric; use hemm() for Hermitian.
/// @throws std::invalid_argument if a pointer is not reachable from the queue's device
template <detail::DenseMatrixLike MA, detail::DenseMatrixLike MB, detail::DenseMatrixLike MC,
          typename T = detail::dense_scalar_t<MA>>
inline Event symm(Queue& ctx, const MA& A, const MB& B, const MC& C, const SymmOptions<T>& opts) {
    BATCHLAS_CHECK_ARGS(ctx, "symm", A, B, C);
    return with_backend_for<T>(ctx, "symm",
                               [&](auto Back) { return symm<Back.value>(ctx, A, B, C, opts); });
}

/// @brief hemm with a compile-time backend; same contract as the Queue-deducing
///        overload, without the pointer check.
/// @tparam Back  backend to run on
/// @param[in]     ctx   queue the kernels are enqueued on
/// @param[in]     A     Hermitian, \f$m \times m\f$ (Left) or \f$n \times n\f$ (Right); only the `uplo` triangle is read
/// @param[in]     B     batch of \f$m \times n\f$ matrices
/// @param[in,out] C     batch of \f$m \times n\f$ matrices; input scaled by \f$\beta\f$, overwritten
/// @param[in]     opts  alpha, beta, side, uplo
/// @return        event of the last enqueued kernel
template <Backend Back, detail::DenseMatrixLike MA, detail::DenseMatrixLike MB,
          detail::DenseMatrixLike MC, typename T = detail::dense_scalar_t<MA>>
    requires ComplexScalar<T>
inline Event hemm(Queue& ctx, const MA& A, const MB& B, const MC& C, const HemmOptions<T>& opts) {
    using V = BATCHLAS_DENSE_VIEW(T);
    return hemm<Back, T>(ctx, V(A), V(B), V(C), opts.alpha, opts.beta, opts.side, opts.uplo);
}

/// @brief Batched Hermitian product; as symm() with A Hermitian.
///
/// The unread triangle is taken as the conjugate transpose of the `uplo` one and
/// the diagonal's imaginary part as zero, whatever is stored there.
/// @param[in]     ctx   queue; supplies the backend
/// @param[in]     A     Hermitian, \f$m \times m\f$ (Left) or \f$n \times n\f$ (Right); only the `uplo` triangle is read
/// @param[in]     B     \f$m \times n\f$ per item
/// @param[in,out] C     \f$m \times n\f$ per item; input scaled by \f$\beta\f$, overwritten
/// @param[in]     opts  alpha, beta, side, uplo
/// @return        event of the last enqueued kernel
/// @note  Complex `T` only.
/// @throws std::invalid_argument if a pointer is not reachable from the queue's device
template <detail::DenseMatrixLike MA, detail::DenseMatrixLike MB, detail::DenseMatrixLike MC,
          typename T = detail::dense_scalar_t<MA>>
    requires ComplexScalar<T>
inline Event hemm(Queue& ctx, const MA& A, const MB& B, const MC& C, const HemmOptions<T>& opts) {
    BATCHLAS_CHECK_ARGS(ctx, "hemm", A, B, C);
    return with_backend_for<T>(ctx, "hemm",
                               [&](auto Back) { return hemm<Back.value>(ctx, A, B, C, opts); });
}

/// @brief herk with a compile-time backend; same contract as the Queue-deducing
///        overload, without the pointer check.
/// @tparam Back  backend to run on
/// @param[in]     ctx   queue the kernels are enqueued on
/// @param[in]     A     \f$n \times k\f$ (NoTrans) or \f$k \times n\f$ (ConjTrans) per item
/// @param[in,out] C     \f$n \times n\f$ per item; only the `uplo` triangle is read and written
/// @param[in]     opts  real alpha and beta, uplo, trans
/// @return        event of the last enqueued kernel
template <Backend Back, detail::DenseMatrixLike MA, detail::DenseMatrixLike MC,
          typename T = detail::dense_scalar_t<MA>>
    requires ComplexScalar<T>
inline Event herk(Queue& ctx, const MA& A, const MC& C, const HerkOptions<T>& opts) {
    using V = BATCHLAS_DENSE_VIEW(T);
    return herk<Back, T>(ctx, V(A), V(C), opts.alpha, opts.beta, opts.uplo, opts.trans);
}

/// @brief Batched Hermitian rank-k update \f$C := \alpha AA^H + \beta C\f$ (NoTrans) or \f$\alpha A^HA + \beta C\f$ (ConjTrans).
/// @param[in]     ctx   queue; supplies the backend
/// @param[in]     A     \f$n \times k\f$ (NoTrans) or \f$k \times n\f$ (ConjTrans) per item
/// @param[in,out] C     \f$n \times n\f$ per item; only the `uplo` triangle is written, the other is
///                      left exactly as it was (mirror it with `hermitize` if needed)
/// @param[in]     opts  real alpha and beta, uplo, trans
/// @return        event of the last enqueued kernel
/// @note  Complex `T` only.
/// @throws std::invalid_argument if a pointer is not reachable from the queue's device
template <detail::DenseMatrixLike MA, detail::DenseMatrixLike MC,
          typename T = detail::dense_scalar_t<MA>>
    requires ComplexScalar<T>
inline Event herk(Queue& ctx, const MA& A, const MC& C, const HerkOptions<T>& opts) {
    BATCHLAS_CHECK_ARGS(ctx, "herk", A, C);
    return with_backend_for<T>(ctx, "herk",
                               [&](auto Back) { return herk<Back.value>(ctx, A, C, opts); });
}

/// @brief her2k with a compile-time backend; same contract as the Queue-deducing
///        overload, without the pointer check.
/// @tparam Back  backend to run on
/// @param[in]     ctx   queue the kernels are enqueued on
/// @param[in]     A,B   \f$n \times k\f$ (NoTrans) or \f$k \times n\f$ (ConjTrans) per item
/// @param[in,out] C     \f$n \times n\f$ per item; only the `uplo` triangle is read and written
/// @param[in]     opts  complex alpha, real beta, uplo, trans
/// @return        event of the last enqueued kernel
template <Backend Back, detail::DenseMatrixLike MA, detail::DenseMatrixLike MB,
          detail::DenseMatrixLike MC, typename T = detail::dense_scalar_t<MA>>
    requires ComplexScalar<T>
inline Event her2k(Queue& ctx, const MA& A, const MB& B, const MC& C,
                   const Her2kOptions<T>& opts) {
    using V = BATCHLAS_DENSE_VIEW(T);
    return her2k<Back, T>(ctx, V(A), V(B), V(C), opts.alpha, opts.beta, opts.uplo, opts.trans);
}

/// @brief Batched Hermitian rank-2k update \f$C := \alpha\,\mathrm{op}(A)\,\mathrm{op}(B)^H + \bar\alpha\,\mathrm{op}(B)\,\mathrm{op}(A)^H + \beta C\f$.
/// @param[in]     ctx   queue; supplies the backend
/// @param[in]     A,B   \f$n \times k\f$ (NoTrans) or \f$k \times n\f$ (ConjTrans) per item
/// @param[in,out] C     \f$n \times n\f$ per item; only the `uplo` triangle is written, real diagonal
/// @param[in]     opts  complex alpha, real beta, uplo, trans
/// @return        event of the last enqueued kernel
/// @note  Complex `T` only.
/// @throws std::invalid_argument if a pointer is not reachable from the queue's device
template <detail::DenseMatrixLike MA, detail::DenseMatrixLike MB, detail::DenseMatrixLike MC,
          typename T = detail::dense_scalar_t<MA>>
    requires ComplexScalar<T>
inline Event her2k(Queue& ctx, const MA& A, const MB& B, const MC& C,
                   const Her2kOptions<T>& opts) {
    BATCHLAS_CHECK_ARGS(ctx, "her2k", A, B, C);
    return with_backend_for<T>(ctx, "her2k",
                               [&](auto Back) { return her2k<Back.value>(ctx, A, B, C, opts); });
}

/// @brief syrk with a compile-time backend; same contract as the Queue-deducing
///        overload, without the pointer check.
/// @tparam Back  backend to run on
/// @param[in]     ctx   queue the kernels are enqueued on
/// @param[in]     A     \f$n \times k\f$ (NoTrans) or \f$k \times n\f$ (Trans) per item
/// @param[in,out] C     \f$n \times n\f$ per item; only the `uplo` triangle is read and written
/// @param[in]     opts  alpha, beta, uplo, trans
/// @return        event of the last enqueued kernel
template <Backend Back, detail::DenseMatrixLike MA, detail::DenseMatrixLike MC,
          typename T = detail::dense_scalar_t<MA>>
inline Event syrk(Queue& ctx, const MA& A, const MC& C, const SyrkOptions<T>& opts) {
    using V = BATCHLAS_DENSE_VIEW(T);
    return syrk<Back, T>(ctx, V(A), V(C), opts.alpha, opts.beta, opts.uplo, opts.trans);
}

/// @brief Batched symmetric rank-k update \f$C := \alpha AA^T + \beta C\f$ (NoTrans) or \f$\alpha A^TA + \beta C\f$ (Trans).
/// @param[in]     ctx   queue; supplies the backend
/// @param[in]     A     \f$n \times k\f$ (NoTrans) or \f$k \times n\f$ (Trans) per item
/// @param[in,out] C     \f$n \times n\f$ per item; only the `uplo` triangle is written, the other is
///                      left exactly as it was (mirror it with `symmetrize` if needed)
/// @param[in]     opts  alpha, beta, uplo, trans
/// @return        event of the last enqueued kernel
/// @note  Complex `T` is complex-symmetric (NoTrans/Trans); use herk() for Hermitian.
/// @throws std::invalid_argument if a pointer is not reachable from the queue's device
template <detail::DenseMatrixLike MA, detail::DenseMatrixLike MC,
          typename T = detail::dense_scalar_t<MA>>
inline Event syrk(Queue& ctx, const MA& A, const MC& C, const SyrkOptions<T>& opts) {
    BATCHLAS_CHECK_ARGS(ctx, "syrk", A, C);
    return with_backend_for<T>(ctx, "syrk",
                               [&](auto Back) { return syrk<Back.value>(ctx, A, C, opts); });
}

/// @brief syr2k with a compile-time backend; same contract as the Queue-deducing
///        overload, without the pointer check.
/// @tparam Back  backend to run on
/// @param[in]     ctx   queue the kernels are enqueued on
/// @param[in]     A,B   \f$n \times k\f$ (NoTrans) or \f$k \times n\f$ (Trans) per item, \f$k > 0\f$
/// @param[in,out] C     \f$n \times n\f$ per item; only the `uplo` triangle is read and written
/// @param[in]     opts  alpha, beta, uplo, trans
/// @return        event of the last enqueued kernel
template <Backend Back, detail::DenseMatrixLike MA, detail::DenseMatrixLike MB,
          detail::DenseMatrixLike MC, typename T = detail::dense_scalar_t<MA>>
inline Event syr2k(Queue& ctx, const MA& A, const MB& B, const MC& C,
                   const Syr2kOptions<T>& opts) {
    using V = BATCHLAS_DENSE_VIEW(T);
    return syr2k<Back, T>(ctx, V(A), V(B), V(C), opts.alpha, opts.beta, opts.uplo, opts.trans);
}

/// @brief Batched symmetric rank-2k update \f$C := \alpha(\mathrm{op}(A)\,\mathrm{op}(B)^T + \mathrm{op}(B)\,\mathrm{op}(A)^T) + \beta C\f$.
/// @param[in]     ctx   queue; supplies the backend
/// @param[in]     A,B   \f$n \times k\f$ (NoTrans) or \f$k \times n\f$ (Trans) per item, \f$k > 0\f$
/// @param[in,out] C     \f$n \times n\f$ per item; only the `uplo` triangle is written
/// @param[in]     opts  alpha, beta, uplo, trans
/// @return        event of the last enqueued kernel
/// @note  Complex `T` is complex-symmetric (NoTrans/Trans); use her2k() for Hermitian.
/// @throws std::invalid_argument if a pointer is not reachable from the queue's device
template <detail::DenseMatrixLike MA, detail::DenseMatrixLike MB, detail::DenseMatrixLike MC,
          typename T = detail::dense_scalar_t<MA>>
inline Event syr2k(Queue& ctx, const MA& A, const MB& B, const MC& C,
                   const Syr2kOptions<T>& opts) {
    BATCHLAS_CHECK_ARGS(ctx, "syr2k", A, B, C);
    return with_backend_for<T>(ctx, "syr2k",
                               [&](auto Back) { return syr2k<Back.value>(ctx, A, B, C, opts); });
}

/// @brief trmm with a compile-time backend; same contract as the Queue-deducing
///        overload, without the pointer check.
/// @tparam Back  backend to run on
/// @param[in]  ctx   queue the kernels are enqueued on
/// @param[in]  A     triangular, \f$m \times m\f$ (Left) or \f$n \times n\f$ (Right); `uplo` and `diag` describe it
/// @param[in]  B     \f$m \times n\f$ per item; read only
/// @param[out] C     \f$m \times n\f$ per item; overwritten with the product
/// @param[in]  opts  alpha, side, uplo, trans, diag
/// @return     event of the last enqueued kernel
template <Backend Back, detail::DenseMatrixLike MA, detail::DenseMatrixLike MB,
          detail::DenseMatrixLike MC, typename T = detail::dense_scalar_t<MA>>
inline Event trmm(Queue& ctx, const MA& A, const MB& B, const MC& C, const TrmmOptions<T>& opts) {
    using V = BATCHLAS_DENSE_VIEW(T);
    return trmm<Back, T>(ctx, V(A), V(B), V(C), opts.alpha, opts.side, opts.uplo, opts.trans,
                         opts.diag);
}

/// @brief Batched triangular product, out of place: \f$C := \alpha\,\mathrm{op}(A)\,B\f$ (Left) or \f$\alpha B\,\mathrm{op}(A)\f$ (Right).
///
/// Unlike reference BLAS `?trmm`, the result goes to `C`; `B` is not modified.
/// @param[in]  ctx   queue; supplies the backend
/// @param[in]  A     triangular, \f$m \times m\f$ (Left) or \f$n \times n\f$ (Right); `uplo` and `diag` describe it
/// @param[in]  B     \f$m \times n\f$ per item; read only
/// @param[out] C     \f$m \times n\f$ per item; overwritten with the product
/// @param[in]  opts  alpha, side, uplo, trans, diag
/// @return     event of the last enqueued kernel
/// @throws std::invalid_argument if a pointer is not reachable from the queue's device
template <detail::DenseMatrixLike MA, detail::DenseMatrixLike MB, detail::DenseMatrixLike MC,
          typename T = detail::dense_scalar_t<MA>>
inline Event trmm(Queue& ctx, const MA& A, const MB& B, const MC& C, const TrmmOptions<T>& opts) {
    BATCHLAS_CHECK_ARGS(ctx, "trmm", A, B, C);
    return with_backend_for<T>(ctx, "trmm",
                               [&](auto Back) { return trmm<Back.value>(ctx, A, B, C, opts); });
}

/// @brief trsm with a compile-time backend; same contract as the Queue-deducing
///        overload, without the pointer check.
/// @tparam Back  backend to run on
/// @param[in]     ctx   queue the kernels are enqueued on
/// @param[in]     A     triangular, \f$m \times m\f$ (Left) or \f$n \times n\f$ (Right); `uplo` and `diag` describe it
/// @param[in,out] B     \f$m \times n\f$ per item; overwritten with \f$X\f$
/// @param[in]     opts  alpha, side, uplo, trans, diag
/// @return        event of the last enqueued kernel
template <Backend Back, detail::DenseMatrixLike MA, detail::DenseMatrixLike MB,
          typename T = detail::dense_scalar_t<MA>>
inline Event trsm(Queue& ctx, const MA& A, const MB& B, const TrsmOptions<T>& opts) {
    using V = BATCHLAS_DENSE_VIEW(T);
    return trsm<Back, T>(ctx, V(A), V(B), opts.alpha, opts.side, opts.uplo, opts.trans, opts.diag);
}

/// @brief Batched triangular solve \f$\mathrm{op}(A)X = \alpha B\f$ (Left) or \f$X\,\mathrm{op}(A) = \alpha B\f$ (Right), in place.
/// @param[in]     ctx   queue; supplies the backend
/// @param[in]     A     triangular, \f$m \times m\f$ (Left) or \f$n \times n\f$ (Right); `uplo` and `diag` describe it
/// @param[in,out] B     \f$m \times n\f$ per item; overwritten with \f$X\f$
/// @param[in]     opts  alpha, side, uplo, trans, diag
/// @return        event of the last enqueued kernel
/// @throws std::invalid_argument if a pointer is not reachable from the queue's device
template <detail::DenseMatrixLike MA, detail::DenseMatrixLike MB,
          typename T = detail::dense_scalar_t<MA>>
inline Event trsm(Queue& ctx, const MA& A, const MB& B, const TrsmOptions<T>& opts) {
    BATCHLAS_CHECK_ARGS(ctx, "trsm", A, B);
    return with_backend_for<T>(ctx, "trsm",
                               [&](auto Back) { return trsm<Back.value>(ctx, A, B, opts); });
}

// ---- dense LAPACK ----------------------------------------------------------

/// @brief Options for potrf(): Cholesky \f$A = LL^H\f$ (Lower) or \f$U^HU\f$ (Upper).
struct PotrfOptions {
    Uplo uplo = Uplo::Lower;  ///< triangle read and overwritten with the factor
    /// Per-item status, one int32 per batch item: 0 if factorised, \f$i > 0\f$ if the
    /// leading minor of order \f$i\f$ is not positive definite. Empty (the default) reports
    /// nothing; otherwise device-accessible, at least `batch_size` long, written in place.
    Span<int32_t> info = {};
};

/// @brief Options for getrs(): solve \f$\mathrm{op}(A)X = B\f$ with getrf's factors.
struct GetrsOptions {
    Transpose trans = Transpose::NoTrans;  ///< \f$\mathrm{op}(A)\f$
};

/// @brief Options for syev(): symmetric/Hermitian eigendecomposition.
struct SyevOptions {
    JobType jobz = JobType::EigenVectors;  ///< eigenvectors too, or eigenvalues only
    Uplo uplo = Uplo::Lower;               ///< triangle of A that is read
};

/// @brief potrf with a compile-time backend and a caller workspace.
/// @tparam B   backend to run on
/// @param[in]     ctx   queue the kernels are enqueued on
/// @param[in,out] A     \f$n \times n\f$ per item; the `uplo` triangle is overwritten with the factor
/// @param[in]     opts  uplo and an optional per-item `info` span whose device buffer receives
///                      the status (when non-empty)
/// @param[out]    ws    at least `potrf_buffer_size<B, T>(ctx, A, opts.uplo)` bytes; contents
///                      undefined on return
/// @return        event of the last enqueued kernel
/// @throws batchlas::invalid_argument as the Queue-deducing overload
template <Backend B, detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
inline Event potrf(Queue& ctx, const MA& A, const PotrfOptions& opts, Span<std::byte> ws) {
    using V = BATCHLAS_DENSE_VIEW(T);
    detail::require_square("potrf", "A", A);
    detail::require_info_span("potrf", opts.info.size(), static_cast<size_t>(A.batch_size()));
    return potrf<B, T>(ctx, V(A), opts.uplo, ws, opts.info);
}

/// @brief potrf with a compile-time backend; workspace leased from the arena.
/// @tparam B  backend to run on
/// @param[in]     ctx   queue the kernels are enqueued on; supplies the arena
/// @param[in,out] A     \f$n \times n\f$ per item; the `uplo` triangle is overwritten with the factor
/// @param[in]     opts  uplo and an optional per-item `info` span whose device buffer receives
///                      the status (when non-empty)
/// @return        event of the last enqueued kernel
/// @throws batchlas::invalid_argument if `A` is not square or `opts.info` is non-empty and short
template <Backend B, detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
inline Event potrf(Queue& ctx, const MA& A, const PotrfOptions& opts) {
    using V = BATCHLAS_DENSE_VIEW(T);
    detail::require_square("potrf", "A", A);
    detail::require_info_span("potrf", opts.info.size(), static_cast<size_t>(A.batch_size()));
    auto lease = ctx.workspace(potrf_buffer_size<B, T>(ctx, V(A), opts.uplo));
    return potrf<B, T>(ctx, V(A), opts.uplo, lease.span(), opts.info);
}

/// @brief Batched Cholesky factorisation with a caller-owned workspace.
///
/// Same contract as the arena overload; the call stays asynchronous on any queue.
/// @param[in]     ctx   queue; supplies the backend
/// @param[in,out] A     \f$n \times n\f$ per item; the `uplo` triangle is read and overwritten
///                      with the factor, the other triangle is not referenced
/// @param[in]     opts  uplo and an optional per-item `info` span whose device buffer receives
///                      the status (when non-empty)
/// @param[out]    ws    device-accessible scratch of at least `potrf_buffer_size` bytes;
///                      must outlive the kernels, contents undefined on return
/// @return        event of the last enqueued kernel
/// @warning   With an empty option struct, write `PotrfOptions{}`: a bare `{}` is
///            rejected as ambiguous (it would otherwise mean `Uplo::Upper`).
template <detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
inline Event potrf(Queue& ctx, const MA& A, const PotrfOptions& opts, Span<std::byte> ws) {
    BATCHLAS_CHECK_ARGS(ctx, "potrf", A, ws, opts.info);
    detail::require_square("potrf", "A", A);
    detail::require_info_span("potrf", opts.info.size(), static_cast<size_t>(A.batch_size()));
    return with_backend_for<T>(ctx, "potrf",
                               [&](auto Back) { return potrf<Back.value>(ctx, A, opts, ws); });
}

/// @brief Batched Cholesky factorisation \f$A_i = L_iL_i^H\f$ (Lower) or \f$U_i^HU_i\f$ (Upper), in place.
///
/// Workspace is leased from the queue's arena (blocking on an out-of-order queue).
/// @param[in]     ctx   queue; supplies the backend and the arena
/// @param[in,out] A     \f$n \times n\f$ per item; the `uplo` triangle is read and overwritten with the
///                      factor, the other triangle is not referenced
/// @param[in]     opts  uplo (default Lower) and an optional per-item `info` span whose device
///                      buffer receives the status (when non-empty)
/// @return        event of the last enqueued kernel
/// @note  A matrix that is not positive definite yields a non-zero `info` entry and
///        an undefined factor for that item; no exception.
/// @throws batchlas::invalid_argument if `A` is not square or `opts.info` is non-empty
///         and shorter than the batch
/// @throws std::invalid_argument if a pointer is not reachable from the queue's device
template <detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
inline Event potrf(Queue& ctx, const MA& A, const PotrfOptions& opts = {}) {
    BATCHLAS_CHECK_ARGS(ctx, "potrf", A, opts.info);
    detail::require_square("potrf", "A", A);
    detail::require_info_span("potrf", opts.info.size(), static_cast<size_t>(A.batch_size()));
    return with_backend_for<T>(ctx, "potrf",
                               [&](auto Back) { return potrf<Back.value>(ctx, A, opts); });
}

/// @}

namespace detail {
// Trap guard: `potrf(ctx, A, {}, ws)` matched Uplo{} == Upper exactly and silently
// factorised the wrong triangle. A third exact match for `{}` makes it ambiguous.
// evidence: docs/design/api-conventions.md#api-conventions-the-bare-braces-potrf-trap
enum class EmptyBracesAreAmbiguous {};
}  // namespace detail

/// @addtogroup api_options
/// @{

/// @brief Deleted: makes `potrf<B>(ctx, A, {}, ws)` ambiguous. Write `PotrfOptions{}`.
template <Backend B, detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
Event potrf(Queue&, const MA&, detail::EmptyBracesAreAmbiguous, Span<std::byte>) = delete;

/// @brief Deleted: makes `potrf(ctx, A, {}, ws)` ambiguous. Write `PotrfOptions{}`.
template <detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
Event potrf(Queue&, const MA&, detail::EmptyBracesAreAmbiguous, Span<std::byte>) = delete;

// getrf, getri, geqrf, orgqr: arena spelling only; a workspace parameter here
// would be ambiguous with the positional call.
/// @brief getrf with a compile-time backend; workspace leased from the arena.
/// @tparam B  backend to run on
/// @param[in]     ctx     queue the kernels are enqueued on; supplies the arena
/// @param[in,out] A       \f$m \times n\f$ per item; overwritten with \f$L\f$ and \f$U\f$
/// @param[out]    pivots  at least `min(m, n) * batch_size` entries; receives the pivots
/// @param[out]    info    optional per-item status: 0 = success, \f$i > 0\f$ if \f$U_{ii}\f$ is
///                        exactly zero (1-based; the factorisation completed but \f$U\f$ is
///                        singular); empty reports nothing
/// @return        event of the last enqueued kernel
/// @throws batchlas::invalid_argument as the Queue-deducing overload
template <Backend B, detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
inline Event getrf(Queue& ctx, const MA& A, Span<int64_t> pivots, Span<int32_t> info = {}) {
    using V = BATCHLAS_DENSE_VIEW(T);
    detail::require_span_at_least("getrf", "pivots", pivots.size(),
                                  static_cast<size_t>(std::min(A.rows(), A.cols())) * A.batch_size());
    detail::require_info_span("getrf", info.size(), static_cast<size_t>(A.batch_size()));
    auto lease = ctx.workspace(getrf_buffer_size<B, T>(ctx, V(A)));
    return getrf<B, T>(ctx, V(A), pivots, lease.span(), info);
}

/// @brief Batched LU factorisation with partial pivoting, \f$A_i = P_iL_iU_i\f$, in place.
///
/// Workspace is leased from the queue's arena. `L` (unit diagonal, implicit) and
/// `U` overwrite `A`.
/// @param[in]     ctx     queue; supplies the backend and the arena
/// @param[in,out] A       \f$m \times n\f$ per item (getrs() and getri() need \f$m = n\f$); overwritten
///                        with \f$L\f$ and \f$U\f$
/// @param[out]    pivots  at least `min(m, n) * batch_size` entries; pass it unchanged to getrs() or
///                        getri() (see the pivot format note on the positional getrf)
/// @param[out]    info    optional per-item status: 0 = success, \f$i > 0\f$ if \f$U_{ii}\f$ is
///                        exactly zero (1-based; the factorisation completed but \f$U\f$ is
///                        singular); empty reports nothing
/// @return        event of the last enqueued kernel
/// @throws batchlas::invalid_argument if a span is too short
/// @throws std::invalid_argument if a pointer is not reachable from the queue's device
template <detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
inline Event getrf(Queue& ctx, const MA& A, Span<int64_t> pivots, Span<int32_t> info = {}) {
    BATCHLAS_CHECK_ARGS(ctx, "getrf", A, pivots, info);
    detail::require_span_at_least("getrf", "pivots", pivots.size(),
                                  static_cast<size_t>(std::min(A.rows(), A.cols())) * A.batch_size());
    detail::require_info_span("getrf", info.size(), static_cast<size_t>(A.batch_size()));
    return with_backend_for<T>(ctx, "getrf",
                               [&](auto Back) { return getrf<Back.value>(ctx, A, pivots, info); });
}

/// @brief getrs with a compile-time backend and a caller workspace.
/// @tparam B  backend to run on
/// @param[in]     ctx     queue the kernels are enqueued on
/// @param[in]     A       \f$n \times n\f$ per item, already factorised by getrf()
/// @param[in,out] B_      \f$n \times n_{\mathrm{rhs}}\f$ per item; overwritten with \f$X\f$
/// @param[in]     pivots  getrf()'s pivots, at least `n * batch_size` entries
/// @param[in]     opts    trans
/// @param[out]    ws      at least `getrs_buffer_size` bytes; contents undefined on return
/// @return        event of the last enqueued kernel
/// @throws batchlas::invalid_argument as the Queue-deducing overload
template <Backend B, detail::DenseMatrixLike MA, detail::DenseMatrixLike MB,
          typename T = detail::dense_scalar_t<MA>>
inline Event getrs(Queue& ctx, const MA& A, const MB& B_, Span<int64_t> pivots,
                   const GetrsOptions& opts, Span<std::byte> ws) {
    using V = BATCHLAS_DENSE_VIEW(T);
    detail::require_square("getrs", "A", A);
    detail::require_same_rows("getrs", "A", A, "B", B_);
    detail::require_same_batch("getrs", "A", A, "B", B_);
    detail::require_span_at_least("getrs", "pivots", pivots.size(),
                                  static_cast<size_t>(A.rows()) * A.batch_size());
    return getrs<B, T>(ctx, V(A), V(B_), opts.trans, pivots, ws);
}

/// @brief getrs with a compile-time backend; workspace leased from the arena.
/// @tparam B  backend to run on
/// @param[in]     ctx     queue the kernels are enqueued on; supplies the arena
/// @param[in]     A       \f$n \times n\f$ per item, already factorised by getrf()
/// @param[in,out] B_      \f$n \times n_{\mathrm{rhs}}\f$ per item; overwritten with \f$X\f$
/// @param[in]     pivots  getrf()'s pivots, at least `n * batch_size` entries
/// @param[in]     opts    trans
/// @return        event of the last enqueued kernel
/// @throws batchlas::invalid_argument as the Queue-deducing overload
template <Backend B, detail::DenseMatrixLike MA, detail::DenseMatrixLike MB,
          typename T = detail::dense_scalar_t<MA>>
inline Event getrs(Queue& ctx, const MA& A, const MB& B_, Span<int64_t> pivots,
                   const GetrsOptions& opts) {
    using V = BATCHLAS_DENSE_VIEW(T);
    detail::require_square("getrs", "A", A);
    detail::require_same_rows("getrs", "A", A, "B", B_);
    detail::require_same_batch("getrs", "A", A, "B", B_);
    detail::require_span_at_least("getrs", "pivots", pivots.size(),
                                  static_cast<size_t>(A.rows()) * A.batch_size());
    auto lease = ctx.workspace(getrs_buffer_size<B, T>(ctx, V(A), V(B_), opts.trans));
    return getrs<B, T>(ctx, V(A), V(B_), opts.trans, pivots, lease.span());
}

/// @brief Batched LU solve with a caller-owned workspace; otherwise as the arena overload.
/// @param[in]     ctx     queue; supplies the backend
/// @param[in]     A       \f$n \times n\f$ per item, already factorised by getrf()
/// @param[in,out] B_      \f$n \times n_{\mathrm{rhs}}\f$ per item; overwritten with \f$X\f$
/// @param[in]     pivots  getrf()'s pivots, at least `n * batch_size` entries
/// @param[in]     opts    trans
/// @param[out]    ws      device-accessible scratch of at least `getrs_buffer_size` bytes;
///                        contents undefined on return
/// @return        event of the last enqueued kernel
template <detail::DenseMatrixLike MA, detail::DenseMatrixLike MB,
          typename T = detail::dense_scalar_t<MA>>
inline Event getrs(Queue& ctx, const MA& A, const MB& B_, Span<int64_t> pivots,
                   const GetrsOptions& opts, Span<std::byte> ws) {
    BATCHLAS_CHECK_ARGS(ctx, "getrs", A, B_, pivots, ws);
    detail::require_square("getrs", "A", A);
    detail::require_same_rows("getrs", "A", A, "B", B_);
    detail::require_same_batch("getrs", "A", A, "B", B_);
    detail::require_span_at_least("getrs", "pivots", pivots.size(),
                                  static_cast<size_t>(A.rows()) * A.batch_size());
    return with_backend_for<T>(
        ctx, "getrs", [&](auto Back) { return getrs<Back.value>(ctx, A, B_, pivots, opts, ws); });
}

/// @brief Batched LU solve \f$\mathrm{op}(A_i)X_i = B_i\f$ from getrf()'s factors, in place.
///
/// Workspace is leased from the queue's arena. No status is reported: a singular
/// factor produces non-finite or meaningless `X` without an exception.
/// @param[in]     ctx     queue; supplies the backend and the arena
/// @param[in]     A       \f$n \times n\f$ per item, already factorised by getrf()
/// @param[in,out] B_      \f$n \times n_{\mathrm{rhs}}\f$ per item; overwritten with \f$X\f$
/// @param[in]     pivots  getrf()'s pivots, at least `n * batch_size` entries
/// @param[in]     opts    trans
/// @return        event of the last enqueued kernel
/// @throws batchlas::invalid_argument if `A` is not square, `A` and `B_` differ in rows
///         or batch size, or `pivots` is too short
/// @throws std::invalid_argument if a pointer is not reachable from the queue's device
template <detail::DenseMatrixLike MA, detail::DenseMatrixLike MB,
          typename T = detail::dense_scalar_t<MA>>
inline Event getrs(Queue& ctx, const MA& A, const MB& B_, Span<int64_t> pivots,
                   const GetrsOptions& opts = {}) {
    BATCHLAS_CHECK_ARGS(ctx, "getrs", A, B_, pivots);
    detail::require_square("getrs", "A", A);
    detail::require_same_rows("getrs", "A", A, "B", B_);
    detail::require_same_batch("getrs", "A", A, "B", B_);
    detail::require_span_at_least("getrs", "pivots", pivots.size(),
                                  static_cast<size_t>(A.rows()) * A.batch_size());
    return with_backend_for<T>(ctx, "getrs",
                               [&](auto Back) { return getrs<Back.value>(ctx, A, B_, pivots, opts); });
}

/// @brief getri with a compile-time backend; workspace leased from the arena.
/// @tparam B  backend to run on
/// @param[in]  ctx     queue the kernels are enqueued on; supplies the arena
/// @param[in]  A       \f$n \times n\f$ per item, already factorised by getrf(); read only
/// @param[out] Ainv    \f$n \times n\f$ per item; overwritten with \f$A^{-1}\f$; must not alias `A`
/// @param[in]  pivots  getrf()'s pivots, at least `n * batch_size` entries
/// @param[out] info    optional per-item status: 0 = success, \f$i > 0\f$ if \f$U_{ii}\f$ is
///                     exactly zero; empty reports nothing
/// @return     event of the last enqueued kernel
/// @throws batchlas::invalid_argument as the Queue-deducing overload
template <Backend B, detail::DenseMatrixLike MA, detail::DenseMatrixLike MB,
          typename T = detail::dense_scalar_t<MA>>
inline Event getri(Queue& ctx, const MA& A, const MB& Ainv, Span<int64_t> pivots,
                   Span<int32_t> info = {}) {
    using V = BATCHLAS_DENSE_VIEW(T);
    detail::require_square("getri", "A", A);
    detail::require_square("getri", "Ainv", Ainv);
    detail::require_same_rows("getri", "A", A, "Ainv", Ainv);
    detail::require_same_batch("getri", "A", A, "Ainv", Ainv);
    detail::require_span_at_least("getri", "pivots", pivots.size(),
                                  static_cast<size_t>(A.rows()) * A.batch_size());
    detail::require_info_span("getri", info.size(), static_cast<size_t>(A.batch_size()));
    auto lease = ctx.workspace(getri_buffer_size<B, T>(ctx, V(A)));
    return getri<B, T>(ctx, V(A), V(Ainv), pivots, lease.span(), info);
}

/// @brief Batched inverse \f$A_{\mathrm{inv},i} := A_i^{-1}\f$ from getrf()'s factors.
///
/// Workspace is leased from the queue's arena. `A` is read only.
/// @param[in]  ctx     queue; supplies the backend and the arena
/// @param[in]  A       \f$n \times n\f$ per item, already factorised by getrf()
/// @param[out] Ainv    \f$n \times n\f$ per item; overwritten with the inverse; must not alias `A`
/// @param[in]  pivots  getrf()'s pivots, at least `n * batch_size` entries
/// @param[out] info    optional per-item status: 0 = success, \f$i > 0\f$ if \f$U_{ii}\f$ is
///                     exactly zero; empty reports nothing
/// @return     event of the last enqueued kernel
/// @throws batchlas::invalid_argument on a non-square or mismatched operand, `Ainv` aliasing
///         `A`, or a short span
/// @throws std::invalid_argument if a pointer is not reachable from the queue's device
template <detail::DenseMatrixLike MA, detail::DenseMatrixLike MB,
          typename T = detail::dense_scalar_t<MA>>
inline Event getri(Queue& ctx, const MA& A, const MB& Ainv, Span<int64_t> pivots,
                   Span<int32_t> info = {}) {
    BATCHLAS_CHECK_ARGS(ctx, "getri", A, Ainv, pivots, info);
    detail::require_square("getri", "A", A);
    detail::require_square("getri", "Ainv", Ainv);
    detail::require_same_rows("getri", "A", A, "Ainv", Ainv);
    detail::require_same_batch("getri", "A", A, "Ainv", Ainv);
    detail::require_span_at_least("getri", "pivots", pivots.size(),
                                  static_cast<size_t>(A.rows()) * A.batch_size());
    detail::require_info_span("getri", info.size(), static_cast<size_t>(A.batch_size()));
    return with_backend_for<T>(ctx, "getri",
                               [&](auto Back) { return getri<Back.value>(ctx, A, Ainv, pivots, info); });
}

/// @brief geqrf with a compile-time backend; workspace leased from the arena.
/// @tparam B  backend to run on
/// @param[in]     ctx  queue the kernels are enqueued on; supplies the arena
/// @param[in,out] A    \f$m \times n\f$ per item; overwritten with \f$R\f$ and the Householder vectors
/// @param[out]    tau  at least `min(m, n) * batch_size` entries; receives the reflector scalars
/// @return        event of the last enqueued kernel
/// @throws batchlas::invalid_argument if `tau` is too short
template <Backend B, detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
inline Event geqrf(Queue& ctx, const MA& A, Span<T> tau) {
    using V = BATCHLAS_DENSE_VIEW(T);
    detail::require_span_at_least("geqrf", "tau", tau.size(),
                                  static_cast<size_t>(std::min(A.rows(), A.cols())) * A.batch_size());
    auto lease = ctx.workspace(geqrf_buffer_size<B, T>(ctx, V(A), tau));
    return geqrf<B, T>(ctx, V(A), tau, lease.span());
}

/// @brief Batched Householder QR, \f$A_i = Q_iR_i\f$, in place.
///
/// `R` overwrites the upper triangle of `A`, the Householder vectors are stored
/// below it, and their scalars go to `tau`. Workspace is leased from the arena.
/// @param[in]     ctx  queue; supplies the backend and the arena
/// @param[in,out] A    \f$m \times n\f$ per item (rectangular allowed); overwritten with \f$R\f$
///                     and the Householder vectors
/// @param[out]    tau  at least `min(m, n) * batch_size` entries; item \f$i\f$'s scalars start at
///                     `i * min(m, n)`
/// @return        event of the last enqueued kernel
/// @throws batchlas::invalid_argument if `tau` is too short
/// @throws std::invalid_argument if a pointer is not reachable from the queue's device
template <detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
inline Event geqrf(Queue& ctx, const MA& A, Span<T> tau) {
    BATCHLAS_CHECK_ARGS(ctx, "geqrf", A, tau);
    // No squareness check (rectangular A is the point); the tau stride is fixed.
    detail::require_span_at_least("geqrf", "tau", tau.size(),
                                  static_cast<size_t>(std::min(A.rows(), A.cols())) * A.batch_size());
    return with_backend_for<T>(ctx, "geqrf",
                               [&](auto Back) { return geqrf<Back.value>(ctx, A, tau); });
}

/// @brief orgqr with a compile-time backend; workspace leased from the arena.
/// @tparam B  backend to run on
/// @param[in]     ctx  queue the kernels are enqueued on; supplies the arena
/// @param[in,out] A    \f$m \times n\f$ per item, \f$m \ge n\f$ (not checked), as left by geqrf();
///                     overwritten with \f$Q\f$
/// @param[in]     tau  geqrf()'s scalars, at least `min(m, n) * batch_size` entries
/// @return        event of the last enqueued kernel
/// @throws batchlas::invalid_argument if `tau` is too short
template <Backend B, detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
inline Event orgqr(Queue& ctx, const MA& A, Span<T> tau) {
    using V = BATCHLAS_DENSE_VIEW(T);
    detail::require_span_at_least("orgqr", "tau", tau.size(),
                                  static_cast<size_t>(std::min(A.rows(), A.cols())) * A.batch_size());
    auto lease = ctx.workspace(orgqr_buffer_size<B, T>(ctx, V(A), tau));
    return orgqr<B, T>(ctx, V(A), tau, lease.span());
}

/// @brief Form the explicit Q of geqrf() in place.
///
/// Overwrites `A` (holding geqrf()'s reflectors) with the first \f$n\f$ columns
/// of \f$Q\f$. Workspace is leased from the arena.
/// @param[in]     ctx  queue; supplies the backend and the arena
/// @param[in,out] A    \f$m \times n\f$ per item, \f$m \ge n\f$ (not checked), as left by geqrf();
///                     overwritten with \f$Q\f$
/// @param[in]     tau  geqrf()'s scalars, at least `min(m, n) * batch_size` entries
/// @return        event of the last enqueued kernel
/// @throws batchlas::invalid_argument if `tau` is too short
/// @throws std::invalid_argument if a pointer is not reachable from the queue's device
template <detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
inline Event orgqr(Queue& ctx, const MA& A, Span<T> tau) {
    BATCHLAS_CHECK_ARGS(ctx, "orgqr", A, tau);
    detail::require_span_at_least("orgqr", "tau", tau.size(),
                                  static_cast<size_t>(std::min(A.rows(), A.cols())) * A.batch_size());
    return with_backend_for<T>(ctx, "orgqr",
                               [&](auto Back) { return orgqr<Back.value>(ctx, A, tau); });
}

/// @brief syev with a compile-time backend and a caller workspace.
/// @tparam B  backend to run on
/// @param[in]     ctx   queue the kernels are enqueued on
/// @param[in,out] A     \f$n \times n\f$ per item; the `uplo` triangle is read, then overwritten with
///                      the eigenvectors (EigenVectors) or left unspecified
/// @param[out]    W     real, at least `n * batch_size` entries; eigenvalues ascending per item
/// @param[in]     opts  jobz, uplo
/// @param[out]    ws    at least `syev_buffer_size` bytes; contents undefined on return
/// @return        event of the last enqueued kernel
/// @throws batchlas::invalid_argument as the Queue-deducing overload
template <Backend B, detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
inline Event syev(Queue& ctx, const MA& A, Span<typename base_type<T>::type> W,
                  const SyevOptions& opts, Span<std::byte> ws) {
    using V = BATCHLAS_DENSE_VIEW(T);
    detail::require_square("syev", "A", A);
    detail::require_span_at_least("syev", "W", W.size(),
                                  static_cast<size_t>(A.rows()) * A.batch_size());
    return syev<B, T>(ctx, V(A), W, opts.jobz, opts.uplo, ws);
}

/// @brief syev with a compile-time backend; workspace leased from the arena.
/// @tparam B  backend to run on
/// @param[in]     ctx   queue the kernels are enqueued on; supplies the arena
/// @param[in,out] A     \f$n \times n\f$ per item; the `uplo` triangle is read, then overwritten with
///                      the eigenvectors (EigenVectors) or left unspecified
/// @param[out]    W     real, at least `n * batch_size` entries; eigenvalues ascending per item
/// @param[in]     opts  jobz, uplo
/// @return        event of the last enqueued kernel
/// @throws batchlas::invalid_argument as the Queue-deducing overload
template <Backend B, detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
inline Event syev(Queue& ctx, const MA& A, Span<typename base_type<T>::type> W,
                  const SyevOptions& opts) {
    using V = BATCHLAS_DENSE_VIEW(T);
    detail::require_square("syev", "A", A);
    detail::require_span_at_least("syev", "W", W.size(),
                                  static_cast<size_t>(A.rows()) * A.batch_size());
    auto lease = ctx.workspace(syev_buffer_size<B, T>(ctx, V(A), W, opts.jobz, opts.uplo));
    return syev<B, T>(ctx, V(A), W, opts.jobz, opts.uplo, lease.span());
}

/// @brief Batched symmetric/Hermitian eigensolver with a caller-owned workspace;
///        otherwise as the arena overload.
/// @param[in]     ctx   queue; supplies the backend
/// @param[in,out] A     \f$n \times n\f$ per item; the `uplo` triangle is read, then overwritten with
///                      the eigenvectors (EigenVectors) or left unspecified
/// @param[out]    W     real, at least `n * batch_size` entries; eigenvalues ascending per item
/// @param[in]     opts  jobz, uplo
/// @param[out]    ws    device-accessible scratch of at least `syev_buffer_size` bytes;
///                      contents undefined on return
/// @return        event of the last enqueued kernel
template <detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
inline Event syev(Queue& ctx, const MA& A, Span<typename base_type<T>::type> W,
                  const SyevOptions& opts, Span<std::byte> ws) {
    BATCHLAS_CHECK_ARGS(ctx, "syev", A, W, ws);
    detail::require_square("syev", "A", A);
    detail::require_span_at_least("syev", "W", W.size(),
                                  static_cast<size_t>(A.rows()) * A.batch_size());
    return with_backend_for<T>(ctx, "syev",
                               [&](auto Back) { return syev<Back.value>(ctx, A, W, opts, ws); });
}

/// @brief Batched symmetric/Hermitian eigendecomposition \f$A_i = V_i\Lambda_iV_i^H\f$.
///
/// Workspace is leased from the queue's arena. This spelling reports no per-item
/// convergence status; use the positional syev() with an `info` span for that.
/// @param[in]     ctx   queue; supplies the backend and the arena
/// @param[in,out] A     \f$n \times n\f$ per item; the `uplo` triangle is read. Overwritten with the
///                      eigenvectors (columns) when `opts.jobz == JobType::EigenVectors`,
///                      left unspecified otherwise
/// @param[out]    W     real, at least `n * batch_size` entries; eigenvalues ascending per item
/// @param[in]     opts  jobz, uplo
/// @return        event of the last enqueued kernel
/// @throws batchlas::invalid_argument if `A` is not square or `W` is too short
/// @throws std::invalid_argument if a pointer is not reachable from the queue's device
template <detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
inline Event syev(Queue& ctx, const MA& A, Span<typename base_type<T>::type> W,
                  const SyevOptions& opts = {}) {
    BATCHLAS_CHECK_ARGS(ctx, "syev", A, W);
    detail::require_square("syev", "A", A);
    detail::require_span_at_least("syev", "W", W.size(),
                                  static_cast<size_t>(A.rows()) * A.batch_size());
    return with_backend_for<T>(ctx, "syev",
                               [&](auto Back) { return syev<Back.value>(ctx, A, W, opts); });
}

// ---- ormqr and gesvd -------------------------------------------------------
// As with getrs, the option forms do not mirror the positional parameter order.

/// @brief Options for ormqr(): apply geqrf()'s Q, \f$C := \mathrm{op}(Q)C\f$ (Left) or \f$C\,\mathrm{op}(Q)\f$ (Right).
struct OrmqrOptions {
    Side side = Side::Left;                ///< which side Q multiplies from
    Transpose trans = Transpose::NoTrans;  ///< \f$\mathrm{op}(Q)\f$
    int32_t block_size_hint = 0;           ///< WY panel width; 0 lets the tuning table pick
};

/// @brief ormqr with a compile-time backend and a caller workspace.
/// @tparam B  backend to run on
/// @param[in]     ctx   queue the kernels are enqueued on
/// @param[in]     A     geqrf()'s output, \f$m \times n\f$ per item, holding \f$k = \min(m, n)\f$ reflectors
///                      below the diagonal; not modified
/// @param[in,out] C     \f$m\f$ rows (Left) or \f$m\f$ columns (Right) per item, \f$m\f$ = `A.rows()`;
///                      overwritten with the product
/// @param[in]     tau   geqrf()'s scalars, at least `min(A.rows(), A.cols()) * batch_size`
/// @param[in]     opts  side, trans, block_size_hint
/// @param[out]    ws    at least `ormqr_buffer_size` bytes, sized with the same
///                      `block_size_hint`; contents undefined on return
/// @return        event of the last enqueued kernel
/// @throws batchlas::invalid_argument as the Queue-deducing overload
template <Backend B, detail::DenseMatrixLike MA, detail::DenseMatrixLike MC,
          typename T = detail::dense_scalar_t<MA>>
inline Event ormqr(Queue& ctx, const MA& A, const MC& C, Span<T> tau,
                   const OrmqrOptions& opts, Span<std::byte> ws) {
    using V = BATCHLAS_DENSE_VIEW(T);
    detail::require_same_batch("ormqr", "A", A, "C", C);
    detail::require_span_at_least("ormqr", "tau", tau.size(),
                                  static_cast<size_t>(std::min(A.rows(), A.cols())) * A.batch_size());
    return ormqr<B, T>(ctx, V(A), V(C), opts.side, opts.trans, tau, ws, opts.block_size_hint);
}

/// @brief ormqr with a compile-time backend; workspace leased from the arena.
/// @tparam B  backend to run on
/// @param[in]     ctx   queue the kernels are enqueued on; supplies the arena
/// @param[in]     A     geqrf()'s output, \f$m \times n\f$ per item, holding \f$k = \min(m, n)\f$ reflectors
///                      below the diagonal; not modified
/// @param[in,out] C     \f$m\f$ rows (Left) or \f$m\f$ columns (Right) per item, \f$m\f$ = `A.rows()`;
///                      overwritten with the product
/// @param[in]     tau   geqrf()'s scalars, at least `min(A.rows(), A.cols()) * batch_size`
/// @param[in]     opts  side, trans, block_size_hint
/// @return        event of the last enqueued kernel
/// @throws batchlas::invalid_argument as the Queue-deducing overload
template <Backend B, detail::DenseMatrixLike MA, detail::DenseMatrixLike MC,
          typename T = detail::dense_scalar_t<MA>>
inline Event ormqr(Queue& ctx, const MA& A, const MC& C, Span<T> tau,
                   const OrmqrOptions& opts) {
    using V = BATCHLAS_DENSE_VIEW(T);
    detail::require_same_batch("ormqr", "A", A, "C", C);
    detail::require_span_at_least("ormqr", "tau", tau.size(),
                                  static_cast<size_t>(std::min(A.rows(), A.cols())) * A.batch_size());
    // Size with the same hint the call uses, or the buffer is under-sized.
    auto lease = ctx.workspace(ormqr_buffer_size<B, T>(ctx, V(A), V(C), opts.side, opts.trans,
                                                       tau, opts.block_size_hint));
    return ormqr<B, T>(ctx, V(A), V(C), opts.side, opts.trans, tau, lease.span(),
                       opts.block_size_hint);
}

/// @brief Apply \f$Q\f$ from geqrf() with a caller-owned workspace; otherwise as the arena overload.
/// @param[in]     ctx   queue; supplies the backend
/// @param[in]     A     geqrf()'s output, \f$m \times n\f$ per item, holding \f$k = \min(m, n)\f$ reflectors
///                      below the diagonal; not modified
/// @param[in,out] C     \f$m\f$ rows (Left) or \f$m\f$ columns (Right) per item, \f$m\f$ = `A.rows()`;
///                      overwritten with the product
/// @param[in]     tau   geqrf()'s scalars, at least `min(A.rows(), A.cols()) * batch_size`
/// @param[in]     opts  side, trans, block_size_hint
/// @param[out]    ws    device-accessible scratch of at least `ormqr_buffer_size` bytes, sized
///                      with the same `block_size_hint`; contents undefined on return
/// @return        event of the last enqueued kernel
template <detail::DenseMatrixLike MA, detail::DenseMatrixLike MC,
          typename T = detail::dense_scalar_t<MA>>
inline Event ormqr(Queue& ctx, const MA& A, const MC& C, Span<T> tau,
                   const OrmqrOptions& opts, Span<std::byte> ws) {
    BATCHLAS_CHECK_ARGS(ctx, "ormqr", A, C, tau, ws);
    detail::require_same_batch("ormqr", "A", A, "C", C);
    detail::require_span_at_least("ormqr", "tau", tau.size(),
                                  static_cast<size_t>(std::min(A.rows(), A.cols())) * A.batch_size());
    return with_backend_for<T>(ctx, "ormqr",
                               [&](auto Back) { return ormqr<Back.value>(ctx, A, C, tau, opts, ws); });
}

/// @brief Apply the Q of geqrf() to C: \f$C_i := \mathrm{op}(Q_i)C_i\f$ (Left) or \f$C_i\,\mathrm{op}(Q_i)\f$ (Right).
///
/// Workspace is leased from the queue's arena.
/// @param[in]     ctx   queue; supplies the backend and the arena
/// @param[in]     A     geqrf()'s output, \f$m \times n\f$ per item, holding \f$k = \min(m, n)\f$ reflectors
///                      below the diagonal; not modified
/// @param[in,out] C     \f$m\f$ rows (Left) or \f$m\f$ columns (Right) per item, \f$m\f$ = `A.rows()`;
///                      overwritten with the product
/// @param[in]     tau   geqrf()'s scalars, at least `min(A.rows(), A.cols()) * batch_size`
/// @param[in]     opts  side, trans, block_size_hint
/// @return        event of the last enqueued kernel
/// @throws batchlas::invalid_argument if `A` and `C` differ in batch size or `tau` is too short
/// @throws std::invalid_argument if a pointer is not reachable from the queue's device
template <detail::DenseMatrixLike MA, detail::DenseMatrixLike MC,
          typename T = detail::dense_scalar_t<MA>>
inline Event ormqr(Queue& ctx, const MA& A, const MC& C, Span<T> tau,
                   const OrmqrOptions& opts = {}) {
    BATCHLAS_CHECK_ARGS(ctx, "ormqr", A, C, tau);
    // tau is indexed at stride min(m, n): short = out of bounds (validate_ormqr_dims).
    detail::require_same_batch("ormqr", "A", A, "C", C);
    detail::require_span_at_least("ormqr", "tau", tau.size(),
                                  static_cast<size_t>(std::min(A.rows(), A.cols())) * A.batch_size());
    return with_backend_for<T>(ctx, "ormqr",
                               [&](auto Back) { return ormqr<Back.value>(ctx, A, C, tau, opts); });
}

/// @brief Options for gesvd(): \f$A = U\Sigma V^H\f$.
struct GesvdOptions {
    SvdVectors jobu = SvdVectors::All;   ///< which columns of U to compute
    SvdVectors jobvh = SvdVectors::All;  ///< which rows of \f$V^H\f$ to compute
    /// Engaged: A is Hermitian and the given triangle is read (square only); this
    /// selects a different entry point, hence optional rather than a sentinel.
    std::optional<Uplo> hermitian_uplo = std::nullopt;
};

/// @brief gesvd with a compile-time backend and a caller workspace.
/// @tparam B  backend to run on
/// @param[in]     ctx              queue the kernels are enqueued on
/// @param[in,out] A                \f$m \times n\f$ per item (square when `hermitian_uplo` is engaged);
///                                 overwritten (destroyed)
/// @param[out]    singular_values  real, at least `min(m, n) * batch_size`; descending per item
/// @param[out]    U                \f$m \times m\f$ (All) or \f$m \times k\f$ (Thin), \f$k = \min(m, n)\f$;
///                                 default-constructed for None
/// @param[out]    Vh               \f$V^H\f$: \f$n \times n\f$ (All) or \f$k \times n\f$ (Thin);
///                                 default-constructed for None
/// @param[in]     opts             jobu, jobvh, hermitian_uplo
/// @param[out]    ws               at least `gesvd_buffer_size` bytes for the same `hermitian_uplo`
///                                 branch; contents undefined on return
/// @return        event of the last enqueued kernel
/// @throws batchlas::invalid_argument as the Queue-deducing overload
template <Backend B, detail::DenseMatrixLike MA, detail::DenseMatrixLike MU,
          detail::DenseMatrixLike MV, typename T = detail::dense_scalar_t<MA>>
inline Event gesvd(Queue& ctx, const MA& A, Span<typename base_type<T>::type> singular_values,
                   const MU& U, const MV& Vh, const GesvdOptions& opts, Span<std::byte> ws) {
    using V = BATCHLAS_DENSE_VIEW(T);
    if (opts.hermitian_uplo) detail::require_square("gesvd", "A", A);
    detail::require_span_at_least("gesvd", "singular_values", singular_values.size(),
                                  static_cast<size_t>(std::min(A.rows(), A.cols())) * A.batch_size());
    return opts.hermitian_uplo
               ? gesvd<B, T>(ctx, V(A), singular_values, V(U), V(Vh), opts.jobu, opts.jobvh,
                             *opts.hermitian_uplo, ws)
               : gesvd<B, T>(ctx, V(A), singular_values, V(U), V(Vh), opts.jobu, opts.jobvh, ws);
}

/// @brief gesvd with a compile-time backend; workspace leased from the arena.
/// @tparam B  backend to run on
/// @param[in]     ctx              queue the kernels are enqueued on; supplies the arena
/// @param[in,out] A                \f$m \times n\f$ per item (square when `hermitian_uplo` is engaged);
///                                 overwritten (destroyed)
/// @param[out]    singular_values  real, at least `min(m, n) * batch_size`; descending per item
/// @param[out]    U                \f$m \times m\f$ (All) or \f$m \times k\f$ (Thin), \f$k = \min(m, n)\f$;
///                                 default-constructed for None
/// @param[out]    Vh               \f$V^H\f$: \f$n \times n\f$ (All) or \f$k \times n\f$ (Thin);
///                                 default-constructed for None
/// @param[in]     opts             jobu, jobvh, hermitian_uplo
/// @return        event of the last enqueued kernel
/// @throws batchlas::invalid_argument as the Queue-deducing overload
template <Backend B, detail::DenseMatrixLike MA, detail::DenseMatrixLike MU,
          detail::DenseMatrixLike MV, typename T = detail::dense_scalar_t<MA>>
inline Event gesvd(Queue& ctx, const MA& A, Span<typename base_type<T>::type> singular_values,
                   const MU& U, const MV& Vh, const GesvdOptions& opts) {
    using V = BATCHLAS_DENSE_VIEW(T);
    if (opts.hermitian_uplo) detail::require_square("gesvd", "A", A);
    detail::require_span_at_least("gesvd", "singular_values", singular_values.size(),
                                  static_cast<size_t>(std::min(A.rows(), A.cols())) * A.batch_size());
    // The query must take the same branch as the call (different scratch needs).
    const size_t bytes =
        opts.hermitian_uplo
            ? gesvd_buffer_size<B, T>(ctx, V(A), singular_values, V(U), V(Vh), opts.jobu,
                                      opts.jobvh, *opts.hermitian_uplo)
            : gesvd_buffer_size<B, T>(ctx, V(A), singular_values, V(U), V(Vh), opts.jobu,
                                      opts.jobvh);
    auto lease = ctx.workspace(bytes);
    return opts.hermitian_uplo
               ? gesvd<B, T>(ctx, V(A), singular_values, V(U), V(Vh), opts.jobu, opts.jobvh,
                             *opts.hermitian_uplo, lease.span())
               : gesvd<B, T>(ctx, V(A), singular_values, V(U), V(Vh), opts.jobu, opts.jobvh,
                             lease.span());
}

/// @brief Batched SVD with a caller-owned workspace; otherwise as the arena overload.
/// @param[in]     ctx              queue; supplies the backend
/// @param[in,out] A                \f$m \times n\f$ per item (square when `hermitian_uplo` is engaged);
///                                 overwritten (destroyed)
/// @param[out]    singular_values  real, at least `min(m, n) * batch_size`; descending per item
/// @param[out]    U                \f$m \times m\f$ (All) or \f$m \times k\f$ (Thin), \f$k = \min(m, n)\f$;
///                                 default-constructed for None
/// @param[out]    Vh               \f$V^H\f$: \f$n \times n\f$ (All) or \f$k \times n\f$ (Thin);
///                                 default-constructed for None
/// @param[in]     opts             jobu, jobvh, hermitian_uplo
/// @param[out]    ws               device-accessible scratch of at least `gesvd_buffer_size` bytes,
///                                 sized for the same `hermitian_uplo` branch; contents undefined
///                                 on return
/// @return        event of the last enqueued kernel
template <detail::DenseMatrixLike MA, detail::DenseMatrixLike MU, detail::DenseMatrixLike MV,
          typename T = detail::dense_scalar_t<MA>>
inline Event gesvd(Queue& ctx, const MA& A, Span<typename base_type<T>::type> singular_values,
                   const MU& U, const MV& Vh, const GesvdOptions& opts, Span<std::byte> ws) {
    BATCHLAS_CHECK_ARGS(ctx, "gesvd", A, singular_values, U, Vh, ws);
    if (opts.hermitian_uplo) detail::require_square("gesvd", "A", A);
    detail::require_span_at_least("gesvd", "singular_values", singular_values.size(),
                                  static_cast<size_t>(std::min(A.rows(), A.cols())) * A.batch_size());
    return with_backend_for<T>(
        ctx, "gesvd", [&](auto Back) { return gesvd<Back.value>(ctx, A, singular_values, U, Vh, opts, ws); });
}

/// @brief Batched singular value decomposition \f$A_i = U_i\Sigma_iV_i^H\f$.
///
/// Workspace is leased from the queue's arena. `A` is overwritten. This spelling
/// reports no per-item convergence status; use the positional gesvd() with an
/// `info` span for that.
/// @param[in]     ctx              queue; supplies the backend and the arena
/// @param[in,out] A                \f$m \times n\f$ per item (square when `hermitian_uplo` is engaged);
///                                 overwritten (destroyed)
/// @param[out]    singular_values  real, at least `min(m, n) * batch_size`; descending per item
/// @param[out]    U                \f$m \times m\f$ (All) or \f$m \times k\f$ (Thin), \f$k = \min(m, n)\f$;
///                                 default-constructed for None
/// @param[out]    Vh               \f$V^H\f$: \f$n \times n\f$ (All) or \f$k \times n\f$ (Thin);
///                                 default-constructed for None
/// @param[in]     opts             jobu, jobvh, hermitian_uplo
/// @return        event of the last enqueued kernel
/// @throws batchlas::invalid_argument if `singular_values` is too short, or `A` is
///         not square with `hermitian_uplo` engaged
/// @throws std::invalid_argument if a pointer is not reachable from the queue's device
template <detail::DenseMatrixLike MA, detail::DenseMatrixLike MU, detail::DenseMatrixLike MV,
          typename T = detail::dense_scalar_t<MA>>
inline Event gesvd(Queue& ctx, const MA& A, Span<typename base_type<T>::type> singular_values,
                   const MU& U, const MV& Vh, const GesvdOptions& opts = {}) {
    BATCHLAS_CHECK_ARGS(ctx, "gesvd", A, singular_values, U, Vh);
    // Only A is checked: a default-constructed U/Vh spells SvdVectors::None.
    if (opts.hermitian_uplo) detail::require_square("gesvd", "A", A);
    detail::require_span_at_least("gesvd", "singular_values", singular_values.size(),
                                  static_cast<size_t>(std::min(A.rows(), A.cols())) * A.batch_size());
    return with_backend_for<T>(
        ctx, "gesvd", [&](auto Back) { return gesvd<Back.value>(ctx, A, singular_values, U, Vh, opts); });
}

/// @}

#undef BATCHLAS_DENSE_VIEW

}  // namespace batchlas
