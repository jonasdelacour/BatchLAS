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
/// @ingroup options

/**
 * @addtogroup options
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
 * so pass your own span there to stay asynchronous. These overloads check shapes
 * (squareness, matching rows and batch sizes, minimum span lengths) and throw
 * `batchlas::invalid_argument` on a violation; the positional primaries do not.
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

// Named counterpart of require_pack_accessible: the error says "gemm: A".
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
// Positional primaries stay unchecked (src/extensions/ calls them per iteration).
// evidence: docs/design/api-conventions.md#api-conventions-shape-checks-live-on-both-option-overloads
template <typename MV>
inline void require_square(const char* fn, const char* name, const MV& A) {
    if (A.rows() != A.cols())
        throw batchlas::invalid_argument(std::string(fn) + ": " + name +
            " must be square, got " + std::to_string(A.rows()) + "x" + std::to_string(A.cols()));
}

template <typename MA, typename MB>
inline void require_same_rows(const char* fn, const char* an, const MA& A,
                              const char* bn, const MB& B) {
    if (A.rows() != B.rows())
        throw batchlas::invalid_argument(std::string(fn) + ": " + an + ".rows() (" +
            std::to_string(A.rows()) + ") must equal " + bn + ".rows() (" +
            std::to_string(B.rows()) + ")");
}

template <typename MA, typename MB>
inline void require_same_batch(const char* fn, const char* an, const MA& A,
                               const char* bn, const MB& B) {
    if (A.batch_size() != B.batch_size())
        throw batchlas::invalid_argument(std::string(fn) + ": " + an + " and " + bn +
            " must have the same batch size (" + std::to_string(A.batch_size()) + " vs " +
            std::to_string(B.batch_size()) + ")");
}

// `>=`, not `==`: a caller may slice one big pivot/tau arena across several calls.
inline void require_span_at_least(const char* fn, const char* name, size_t have, size_t need) {
    if (have < need)
        throw batchlas::invalid_argument(std::string(fn) + ": " + name + " holds " +
            std::to_string(have) + " elements, needs at least " + std::to_string(need));
}

// Empty `info` = "no status wanted". A too-short one would fail silently (stale
// zeros read as "all factorised"), hence the check.
inline void require_info_span(const char* fn, size_t have, size_t batch_size) {
    if (have != 0) require_span_at_least(fn, "info", have, batch_size);
}

}  // namespace detail

/// @addtogroup options
/// @{

// ---- dense BLAS ------------------------------------------------------------

/// @brief Options for gemm(): \f$C := \alpha\,\mathrm{op}(A)\,\mathrm{op}(B) + \beta C\f$.
template <typename T>
struct GemmOptions {
    T alpha = T(1);                                          ///< scale of the product
    T beta = T(0);                                           ///< scale of the incoming C; 0 means C is not read
    Transpose transA = Transpose::NoTrans;                   ///< op(A)
    Transpose transB = Transpose::NoTrans;                   ///< op(B)
    ComputePrecision precision = ComputePrecision::Default;  ///< internal compute type; Default = the input type
};

/// @brief Options for gemv(): \f$y := \alpha\,\mathrm{op}(A)\,x + \beta y\f$.
template <typename T>
struct GemvOptions {
    T alpha = T(1);                         ///< scale of the product
    T beta = T(0);                          ///< scale of the incoming y
    Transpose transA = Transpose::NoTrans;  ///< op(A)
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
    Side side = Side::Left;               ///< which side op(A) multiplies from
    Uplo uplo = Uplo::Lower;              ///< which triangle A is
    Transpose trans = Transpose::NoTrans; ///< op(A)
    Diag diag = Diag::NonUnit;            ///< Unit: A's diagonal is taken as ones and not read
};

/// @brief Options for trsm(): solve \f$\mathrm{op}(A)X = \alpha B\f$ (Left) or \f$X\,\mathrm{op}(A) = \alpha B\f$ (Right).
template <typename T>
struct TrsmOptions {
    T alpha = T(1);                       ///< scale of the right-hand side
    Side side = Side::Left;               ///< which side op(A) is on
    Uplo uplo = Uplo::Lower;              ///< which triangle A is
    Transpose trans = Transpose::NoTrans; ///< op(A)
    Diag diag = Diag::NonUnit;            ///< Unit: A's diagonal is taken as ones and not read
};

/// @}

// src/extensions/ is templated on Backend and must call the `f<B>(ctx, ...)`
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

/// @addtogroup options
/// @{

// ---- dense BLAS ------------------------------------------------------------

/// @brief gemm with a compile-time backend; same contract as the Queue-deducing
///        overload, without the pointer check.
/// @tparam Back  backend to run on
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
/// @param ctx   queue; supplies the backend
/// @param A     `op(A)` is m x k per item
/// @param B     `op(B)` is k x n per item
/// @param C     m x n per item; read when `beta != 0`, overwritten
/// @param opts  alpha, beta, transA, transB, precision
/// @return      event of the last enqueued kernel
/// @pre   all operands have the same batch size
/// @throws std::invalid_argument if a pointer is not reachable from the queue's device
template <detail::DenseMatrixLike MA, detail::DenseMatrixLike MB, detail::DenseMatrixLike MC,
          typename T = detail::dense_scalar_t<MA>>
inline Event gemm(Queue& ctx, const MA& A, const MB& B, const MC& C, const GemmOptions<T>& opts) {
    BATCHLAS_CHECK_ARGS(ctx, "gemm", A, B, C);
    return with_backend(ctx, [&](auto Back) { return gemm<Back.value>(ctx, A, B, C, opts); });
}

/// @brief gemv with a compile-time backend; same contract as the Queue-deducing
///        overload, without the pointer check.
/// @tparam Back  backend to run on
template <Backend Back, detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
inline Event gemv(Queue& ctx, const MA& A, const VectorView<T>& x, const VectorView<T>& y,
                  const GemvOptions<T>& opts) {
    using V = BATCHLAS_DENSE_VIEW(T);
    return gemv<Back, T>(ctx, V(A), x, y, opts.alpha, opts.beta, opts.transA);
}

/// @brief Batched matrix-vector product \f$y_i := \alpha\,\mathrm{op}(A_i)\,x_i + \beta y_i\f$.
/// @param ctx   queue; supplies the backend
/// @param A     `op(A)` is m x n per item
/// @param x     length n per item
/// @param y     length m per item; overwritten
/// @param opts  alpha, beta, transA
/// @return      event of the last enqueued kernel
/// @throws std::invalid_argument if a pointer is not reachable from the queue's device
template <detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
inline Event gemv(Queue& ctx, const MA& A, const VectorView<T>& x, const VectorView<T>& y,
                  const GemvOptions<T>& opts) {
    BATCHLAS_CHECK_ARGS(ctx, "gemv", A, x, y);
    return with_backend(ctx, [&](auto Back) { return gemv<Back.value>(ctx, A, x, y, opts); });
}

/// @brief symm with a compile-time backend; same contract as the Queue-deducing
///        overload, without the pointer check.
/// @tparam Back  backend to run on
template <Backend Back, detail::DenseMatrixLike MA, detail::DenseMatrixLike MB,
          detail::DenseMatrixLike MC, typename T = detail::dense_scalar_t<MA>>
inline Event symm(Queue& ctx, const MA& A, const MB& B, const MC& C, const SymmOptions<T>& opts) {
    using V = BATCHLAS_DENSE_VIEW(T);
    return symm<Back, T>(ctx, V(A), V(B), V(C), opts.alpha, opts.beta, opts.side, opts.uplo);
}

/// @brief Batched symmetric product \f$C := \alpha AB + \beta C\f$ (Left) or \f$\alpha BA + \beta C\f$ (Right).
/// @param A     symmetric, m x m (Left) or n x n (Right); only the `uplo` triangle is read
/// @param B     m x n per item
/// @param C     m x n per item; overwritten
/// @param opts  alpha, beta, side, uplo
/// @return      event of the last enqueued kernel
/// @note  Real `T` only; use hemm() for complex.
/// @throws std::invalid_argument if a pointer is not reachable from the queue's device
template <detail::DenseMatrixLike MA, detail::DenseMatrixLike MB, detail::DenseMatrixLike MC,
          typename T = detail::dense_scalar_t<MA>>
inline Event symm(Queue& ctx, const MA& A, const MB& B, const MC& C, const SymmOptions<T>& opts) {
    BATCHLAS_CHECK_ARGS(ctx, "symm", A, B, C);
    return with_backend(ctx, [&](auto Back) { return symm<Back.value>(ctx, A, B, C, opts); });
}

/// @brief hemm with a compile-time backend; same contract as the Queue-deducing
///        overload, without the pointer check.
/// @tparam Back  backend to run on
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
/// @param opts  alpha, beta, side, uplo
/// @return      event of the last enqueued kernel
/// @note  Complex `T` only.
/// @throws std::invalid_argument if a pointer is not reachable from the queue's device
template <detail::DenseMatrixLike MA, detail::DenseMatrixLike MB, detail::DenseMatrixLike MC,
          typename T = detail::dense_scalar_t<MA>>
    requires ComplexScalar<T>
inline Event hemm(Queue& ctx, const MA& A, const MB& B, const MC& C, const HemmOptions<T>& opts) {
    BATCHLAS_CHECK_ARGS(ctx, "hemm", A, B, C);
    return with_backend(ctx, [&](auto Back) { return hemm<Back.value>(ctx, A, B, C, opts); });
}

/// @brief herk with a compile-time backend; same contract as the Queue-deducing
///        overload, without the pointer check.
/// @tparam Back  backend to run on
template <Backend Back, detail::DenseMatrixLike MA, detail::DenseMatrixLike MC,
          typename T = detail::dense_scalar_t<MA>>
    requires ComplexScalar<T>
inline Event herk(Queue& ctx, const MA& A, const MC& C, const HerkOptions<T>& opts) {
    using V = BATCHLAS_DENSE_VIEW(T);
    return herk<Back, T>(ctx, V(A), V(C), opts.alpha, opts.beta, opts.uplo, opts.trans);
}

/// @brief Batched Hermitian rank-k update \f$C := \alpha AA^H + \beta C\f$ (NoTrans) or \f$\alpha A^HA + \beta C\f$ (ConjTrans).
/// @param A     n x k (NoTrans) or k x n (ConjTrans) per item
/// @param C     n x n per item; only the `uplo` triangle is written, the other is
///              left exactly as it was (mirror it with `hermitize` if needed)
/// @param opts  real alpha and beta, uplo, trans
/// @return      event of the last enqueued kernel
/// @note  Complex `T` only.
/// @throws std::invalid_argument if a pointer is not reachable from the queue's device
template <detail::DenseMatrixLike MA, detail::DenseMatrixLike MC,
          typename T = detail::dense_scalar_t<MA>>
    requires ComplexScalar<T>
inline Event herk(Queue& ctx, const MA& A, const MC& C, const HerkOptions<T>& opts) {
    BATCHLAS_CHECK_ARGS(ctx, "herk", A, C);
    return with_backend(ctx, [&](auto Back) { return herk<Back.value>(ctx, A, C, opts); });
}

/// @brief her2k with a compile-time backend; same contract as the Queue-deducing
///        overload, without the pointer check.
/// @tparam Back  backend to run on
template <Backend Back, detail::DenseMatrixLike MA, detail::DenseMatrixLike MB,
          detail::DenseMatrixLike MC, typename T = detail::dense_scalar_t<MA>>
    requires ComplexScalar<T>
inline Event her2k(Queue& ctx, const MA& A, const MB& B, const MC& C,
                   const Her2kOptions<T>& opts) {
    using V = BATCHLAS_DENSE_VIEW(T);
    return her2k<Back, T>(ctx, V(A), V(B), V(C), opts.alpha, opts.beta, opts.uplo, opts.trans);
}

/// @brief Batched Hermitian rank-2k update \f$C := \alpha\,\mathrm{op}(A)\,\mathrm{op}(B)^H + \bar\alpha\,\mathrm{op}(B)\,\mathrm{op}(A)^H + \beta C\f$.
/// @param A,B   n x k (NoTrans) or k x n (ConjTrans) per item
/// @param C     n x n per item; only the `uplo` triangle is written, real diagonal
/// @param opts  complex alpha, real beta, uplo, trans
/// @return      event of the last enqueued kernel
/// @note  Complex `T` only.
/// @throws std::invalid_argument if a pointer is not reachable from the queue's device
template <detail::DenseMatrixLike MA, detail::DenseMatrixLike MB, detail::DenseMatrixLike MC,
          typename T = detail::dense_scalar_t<MA>>
    requires ComplexScalar<T>
inline Event her2k(Queue& ctx, const MA& A, const MB& B, const MC& C,
                   const Her2kOptions<T>& opts) {
    BATCHLAS_CHECK_ARGS(ctx, "her2k", A, B, C);
    return with_backend(ctx, [&](auto Back) { return her2k<Back.value>(ctx, A, B, C, opts); });
}

/// @brief syrk with a compile-time backend; same contract as the Queue-deducing
///        overload, without the pointer check.
/// @tparam Back  backend to run on
template <Backend Back, detail::DenseMatrixLike MA, detail::DenseMatrixLike MC,
          typename T = detail::dense_scalar_t<MA>>
inline Event syrk(Queue& ctx, const MA& A, const MC& C, const SyrkOptions<T>& opts) {
    using V = BATCHLAS_DENSE_VIEW(T);
    return syrk<Back, T>(ctx, V(A), V(C), opts.alpha, opts.beta, opts.uplo, opts.trans);
}

/// @brief Batched symmetric rank-k update \f$C := \alpha AA^T + \beta C\f$ (NoTrans) or \f$\alpha A^TA + \beta C\f$ (Trans).
/// @param A     n x k (NoTrans) or k x n (Trans) per item
/// @param C     n x n per item; only the `uplo` triangle is written, the other is
///              left exactly as it was (mirror it with `symmetrize` if needed)
/// @param opts  alpha, beta, uplo, trans
/// @return      event of the last enqueued kernel
/// @note  Real `T` only; use herk() for complex.
/// @throws std::invalid_argument if a pointer is not reachable from the queue's device
template <detail::DenseMatrixLike MA, detail::DenseMatrixLike MC,
          typename T = detail::dense_scalar_t<MA>>
inline Event syrk(Queue& ctx, const MA& A, const MC& C, const SyrkOptions<T>& opts) {
    BATCHLAS_CHECK_ARGS(ctx, "syrk", A, C);
    return with_backend(ctx, [&](auto Back) { return syrk<Back.value>(ctx, A, C, opts); });
}

/// @brief syr2k with a compile-time backend; same contract as the Queue-deducing
///        overload, without the pointer check.
/// @tparam Back  backend to run on
template <Backend Back, detail::DenseMatrixLike MA, detail::DenseMatrixLike MB,
          detail::DenseMatrixLike MC, typename T = detail::dense_scalar_t<MA>>
inline Event syr2k(Queue& ctx, const MA& A, const MB& B, const MC& C,
                   const Syr2kOptions<T>& opts) {
    using V = BATCHLAS_DENSE_VIEW(T);
    return syr2k<Back, T>(ctx, V(A), V(B), V(C), opts.alpha, opts.beta, opts.uplo, opts.trans);
}

/// @brief Batched symmetric rank-2k update \f$C := \alpha(\mathrm{op}(A)\,\mathrm{op}(B)^T + \mathrm{op}(B)\,\mathrm{op}(A)^T) + \beta C\f$.
/// @param A,B   n x k (NoTrans) or k x n (Trans) per item, k > 0
/// @param C     n x n per item; only the `uplo` triangle is written
/// @param opts  alpha, beta, uplo, trans
/// @return      event of the last enqueued kernel
/// @note  Real `T` only; use her2k() for complex.
/// @throws std::invalid_argument if a pointer is not reachable from the queue's device
template <detail::DenseMatrixLike MA, detail::DenseMatrixLike MB, detail::DenseMatrixLike MC,
          typename T = detail::dense_scalar_t<MA>>
inline Event syr2k(Queue& ctx, const MA& A, const MB& B, const MC& C,
                   const Syr2kOptions<T>& opts) {
    BATCHLAS_CHECK_ARGS(ctx, "syr2k", A, B, C);
    return with_backend(ctx, [&](auto Back) { return syr2k<Back.value>(ctx, A, B, C, opts); });
}

/// @brief trmm with a compile-time backend; same contract as the Queue-deducing
///        overload, without the pointer check.
/// @tparam Back  backend to run on
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
/// @param A     triangular, m x m (Left) or n x n (Right); `uplo` and `diag` describe it
/// @param B     m x n per item; read only
/// @param C     m x n per item; overwritten
/// @param opts  alpha, side, uplo, trans, diag
/// @return      event of the last enqueued kernel
/// @throws std::invalid_argument if a pointer is not reachable from the queue's device
template <detail::DenseMatrixLike MA, detail::DenseMatrixLike MB, detail::DenseMatrixLike MC,
          typename T = detail::dense_scalar_t<MA>>
inline Event trmm(Queue& ctx, const MA& A, const MB& B, const MC& C, const TrmmOptions<T>& opts) {
    BATCHLAS_CHECK_ARGS(ctx, "trmm", A, B, C);
    return with_backend(ctx, [&](auto Back) { return trmm<Back.value>(ctx, A, B, C, opts); });
}

/// @brief trsm with a compile-time backend; same contract as the Queue-deducing
///        overload, without the pointer check.
/// @tparam Back  backend to run on
template <Backend Back, detail::DenseMatrixLike MA, detail::DenseMatrixLike MB,
          typename T = detail::dense_scalar_t<MA>>
inline Event trsm(Queue& ctx, const MA& A, const MB& B, const TrsmOptions<T>& opts) {
    using V = BATCHLAS_DENSE_VIEW(T);
    return trsm<Back, T>(ctx, V(A), V(B), opts.alpha, opts.side, opts.uplo, opts.trans, opts.diag);
}

/// @brief Batched triangular solve \f$\mathrm{op}(A)X = \alpha B\f$ (Left) or \f$X\,\mathrm{op}(A) = \alpha B\f$ (Right), in place.
/// @param A     triangular, m x m (Left) or n x n (Right); `uplo` and `diag` describe it
/// @param B     m x n per item; overwritten with X
/// @param opts  alpha, side, uplo, trans, diag
/// @return      event of the last enqueued kernel
/// @throws std::invalid_argument if a pointer is not reachable from the queue's device
template <detail::DenseMatrixLike MA, detail::DenseMatrixLike MB,
          typename T = detail::dense_scalar_t<MA>>
inline Event trsm(Queue& ctx, const MA& A, const MB& B, const TrsmOptions<T>& opts) {
    BATCHLAS_CHECK_ARGS(ctx, "trsm", A, B);
    return with_backend(ctx, [&](auto Back) { return trsm<Back.value>(ctx, A, B, opts); });
}

// ---- dense LAPACK ----------------------------------------------------------

/// @brief Options for potrf(): Cholesky \f$A = LL^H\f$ (Lower) or \f$U^HU\f$ (Upper).
struct PotrfOptions {
    Uplo uplo = Uplo::Lower;  ///< triangle read and overwritten with the factor
    /// Per-item status, one int32 per batch item: 0 = factorised, > 0 = the order of
    /// the leading minor that is not positive definite. Empty (the default) reports
    /// nothing; otherwise device-accessible, at least `batch_size` long, written in place.
    Span<int32_t> info = {};
};

/// @brief Options for getrs(): solve \f$\mathrm{op}(A)X = B\f$ with getrf's factors.
struct GetrsOptions {
    Transpose trans = Transpose::NoTrans;  ///< op(A)
};

/// @brief Options for syev(): symmetric/Hermitian eigendecomposition.
struct SyevOptions {
    JobType jobz = JobType::EigenVectors;  ///< eigenvectors too, or eigenvalues only
    Uplo uplo = Uplo::Lower;               ///< triangle of A that is read
};

/// @brief potrf with a compile-time backend and a caller workspace; no shape checks.
/// @tparam B   backend to run on
/// @param ws   at least `potrf_buffer_size<B, T>(ctx, A, opts.uplo)` bytes
template <Backend B, detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
inline Event potrf(Queue& ctx, const MA& A, const PotrfOptions& opts, Span<std::byte> ws) {
    using V = BATCHLAS_DENSE_VIEW(T);
    return potrf<B, T>(ctx, V(A), opts.uplo, ws, opts.info);
}

/// @brief potrf with a compile-time backend; workspace leased from the arena.
/// @tparam B  backend to run on
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
/// @param ws  device-accessible scratch of at least `potrf_buffer_size` bytes;
///            must outlive the kernels
/// @warning   With an empty option struct, write `PotrfOptions{}`: a bare `{}` is
///            rejected as ambiguous (it would otherwise mean `Uplo::Upper`).
template <detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
inline Event potrf(Queue& ctx, const MA& A, const PotrfOptions& opts, Span<std::byte> ws) {
    BATCHLAS_CHECK_ARGS(ctx, "potrf", A, ws, opts.info);
    detail::require_square("potrf", "A", A);
    detail::require_info_span("potrf", opts.info.size(), static_cast<size_t>(A.batch_size()));
    return with_backend(ctx, [&](auto Back) { return potrf<Back.value>(ctx, A, opts, ws); });
}

/// @brief Batched Cholesky factorisation \f$A_i = L_iL_i^H\f$ (Lower) or \f$U_i^HU_i\f$ (Upper), in place.
///
/// Workspace is leased from the queue's arena (blocking on an out-of-order queue).
/// @param ctx   queue; supplies the backend and the arena
/// @param A     n x n per item; the `uplo` triangle is read and overwritten with the
///              factor, the other triangle is not referenced
/// @param opts  uplo (default Lower) and an optional per-item `info` span
/// @return      event of the last enqueued kernel
/// @pre   A matrix that is not positive definite yields a non-zero `info` entry and
///        an undefined factor for that item; no exception.
/// @throws batchlas::invalid_argument if `A` is not square or `opts.info` is non-empty
///         and shorter than the batch
/// @throws std::invalid_argument if a pointer is not reachable from the queue's device
template <detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
inline Event potrf(Queue& ctx, const MA& A, const PotrfOptions& opts = {}) {
    BATCHLAS_CHECK_ARGS(ctx, "potrf", A, opts.info);
    detail::require_square("potrf", "A", A);
    detail::require_info_span("potrf", opts.info.size(), static_cast<size_t>(A.batch_size()));
    return with_backend(ctx, [&](auto Back) { return potrf<Back.value>(ctx, A, opts); });
}

/// @}

namespace detail {
// Trap guard: `potrf(ctx, A, {}, ws)` matched Uplo{} == Upper exactly and silently
// factorised the wrong triangle. A third exact match for `{}` makes it ambiguous.
// evidence: docs/design/api-conventions.md#api-conventions-the-bare-braces-potrf-trap
enum class EmptyBracesAreAmbiguous {};
}  // namespace detail

/// @addtogroup options
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
/// @throws batchlas::invalid_argument as the Queue-deducing overload
template <Backend B, detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
inline Event getrf(Queue& ctx, const MA& A, Span<int64_t> pivots, Span<int32_t> info = {}) {
    using V = BATCHLAS_DENSE_VIEW(T);
    detail::require_square("getrf", "A", A);
    detail::require_span_at_least("getrf", "pivots", pivots.size(),
                                  static_cast<size_t>(A.rows()) * A.batch_size());
    detail::require_info_span("getrf", info.size(), static_cast<size_t>(A.batch_size()));
    auto lease = ctx.workspace(getrf_buffer_size<B, T>(ctx, V(A)));
    return getrf<B, T>(ctx, V(A), pivots, lease.span(), info);
}

/// @brief Batched LU factorisation with partial pivoting, \f$A_i = P_iL_iU_i\f$, in place.
///
/// Workspace is leased from the queue's arena. `L` (unit diagonal, implicit) and
/// `U` overwrite `A`.
/// @param ctx     queue; supplies the backend and the arena
/// @param A       n x n per item
/// @param pivots  at least `n * batch_size` entries; pass it unchanged to getrs() or
///                getri() (see the pivot format note on the positional getrf)
/// @param info    optional per-item status: 0 = success, > 0 = index of the first
///                exactly-zero pivot; empty reports nothing
/// @return        event of the last enqueued kernel
/// @throws batchlas::invalid_argument if `A` is not square or a span is too short
/// @throws std::invalid_argument if a pointer is not reachable from the queue's device
template <detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
inline Event getrf(Queue& ctx, const MA& A, Span<int64_t> pivots, Span<int32_t> info = {}) {
    BATCHLAS_CHECK_ARGS(ctx, "getrf", A, pivots, info);
    detail::require_square("getrf", "A", A);
    detail::require_span_at_least("getrf", "pivots", pivots.size(),
                                  static_cast<size_t>(A.rows()) * A.batch_size());
    detail::require_info_span("getrf", info.size(), static_cast<size_t>(A.batch_size()));
    return with_backend(ctx, [&](auto Back) { return getrf<Back.value>(ctx, A, pivots, info); });
}

/// @brief getrs with a compile-time backend and a caller workspace; no shape checks.
/// @tparam B  backend to run on
template <Backend B, detail::DenseMatrixLike MA, detail::DenseMatrixLike MB,
          typename T = detail::dense_scalar_t<MA>>
inline Event getrs(Queue& ctx, const MA& A, const MB& B_, Span<int64_t> pivots,
                   const GetrsOptions& opts, Span<std::byte> ws) {
    using V = BATCHLAS_DENSE_VIEW(T);
    return getrs<B, T>(ctx, V(A), V(B_), opts.trans, pivots, ws);
}

/// @brief getrs with a compile-time backend; workspace leased from the arena.
/// @tparam B  backend to run on
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
/// @param ws  device-accessible scratch of at least `getrs_buffer_size` bytes
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
    return with_backend(
        ctx, [&](auto Back) { return getrs<Back.value>(ctx, A, B_, pivots, opts, ws); });
}

/// @brief Batched LU solve \f$\mathrm{op}(A_i)X_i = B_i\f$ from getrf()'s factors, in place.
///
/// Workspace is leased from the queue's arena. No status is reported: a singular
/// factor produces non-finite or meaningless `X` without an exception.
/// @param A       n x n per item, already factorised by getrf()
/// @param B_      n x nrhs per item; overwritten with X
/// @param pivots  getrf()'s pivots, at least `n * batch_size` entries
/// @param opts    trans
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
    return with_backend(ctx,
                        [&](auto Back) { return getrs<Back.value>(ctx, A, B_, pivots, opts); });
}

/// @brief getri with a compile-time backend; workspace leased from the arena.
/// @tparam B  backend to run on
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
/// @param A       n x n per item, already factorised by getrf()
/// @param Ainv    n x n per item; overwritten with the inverse
/// @param pivots  getrf()'s pivots, at least `n * batch_size` entries
/// @param info    optional per-item status (0 = success); empty reports nothing
/// @return        event of the last enqueued kernel
/// @throws batchlas::invalid_argument on a non-square or mismatched operand or a short span
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
    return with_backend(ctx,
                        [&](auto Back) { return getri<Back.value>(ctx, A, Ainv, pivots, info); });
}

/// @brief geqrf with a compile-time backend; workspace leased from the arena.
/// @tparam B  backend to run on
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
/// @param A    m x n per item (rectangular allowed)
/// @param tau  at least `min(m, n) * batch_size` entries; item i's scalars start at
///             `i * min(m, n)`
/// @return     event of the last enqueued kernel
/// @throws batchlas::invalid_argument if `tau` is too short
/// @throws std::invalid_argument if a pointer is not reachable from the queue's device
template <detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
inline Event geqrf(Queue& ctx, const MA& A, Span<T> tau) {
    BATCHLAS_CHECK_ARGS(ctx, "geqrf", A, tau);
    // No squareness check (rectangular A is the point); the tau stride is fixed.
    detail::require_span_at_least("geqrf", "tau", tau.size(),
                                  static_cast<size_t>(std::min(A.rows(), A.cols())) * A.batch_size());
    return with_backend(ctx, [&](auto Back) { return geqrf<Back.value>(ctx, A, tau); });
}

/// @brief orgqr with a compile-time backend; workspace leased from the arena.
/// @tparam B  backend to run on
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
/// Overwrites `A` (holding geqrf()'s reflectors) with the first min(m, n) columns
/// of Q. Workspace is leased from the arena.
/// @param A    m x n per item, as left by geqrf()
/// @param tau  geqrf()'s scalars, at least `min(m, n) * batch_size` entries
/// @return     event of the last enqueued kernel
/// @throws batchlas::invalid_argument if `tau` is too short
/// @throws std::invalid_argument if a pointer is not reachable from the queue's device
template <detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
inline Event orgqr(Queue& ctx, const MA& A, Span<T> tau) {
    BATCHLAS_CHECK_ARGS(ctx, "orgqr", A, tau);
    detail::require_span_at_least("orgqr", "tau", tau.size(),
                                  static_cast<size_t>(std::min(A.rows(), A.cols())) * A.batch_size());
    return with_backend(ctx, [&](auto Back) { return orgqr<Back.value>(ctx, A, tau); });
}

/// @brief syev with a compile-time backend and a caller workspace; no shape checks.
/// @tparam B  backend to run on
template <Backend B, detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
inline Event syev(Queue& ctx, const MA& A, Span<typename base_type<T>::type> W,
                  const SyevOptions& opts, Span<std::byte> ws) {
    using V = BATCHLAS_DENSE_VIEW(T);
    return syev<B, T>(ctx, V(A), W, opts.jobz, opts.uplo, ws);
}

/// @brief syev with a compile-time backend; workspace leased from the arena, no shape checks.
/// @tparam B  backend to run on
template <Backend B, detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
inline Event syev(Queue& ctx, const MA& A, Span<typename base_type<T>::type> W,
                  const SyevOptions& opts) {
    using V = BATCHLAS_DENSE_VIEW(T);
    auto lease = ctx.workspace(syev_buffer_size<B, T>(ctx, V(A), W, opts.jobz, opts.uplo));
    return syev<B, T>(ctx, V(A), W, opts.jobz, opts.uplo, lease.span());
}

/// @brief Batched symmetric/Hermitian eigensolver with a caller-owned workspace;
///        otherwise as the arena overload.
/// @param ws  device-accessible scratch of at least `syev_buffer_size` bytes
template <detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
inline Event syev(Queue& ctx, const MA& A, Span<typename base_type<T>::type> W,
                  const SyevOptions& opts, Span<std::byte> ws) {
    BATCHLAS_CHECK_ARGS(ctx, "syev", A, W, ws);
    detail::require_square("syev", "A", A);
    detail::require_span_at_least("syev", "W", W.size(),
                                  static_cast<size_t>(A.rows()) * A.batch_size());
    return with_backend(ctx, [&](auto Back) { return syev<Back.value>(ctx, A, W, opts, ws); });
}

/// @brief Batched symmetric/Hermitian eigendecomposition \f$A_i = V_i\Lambda_iV_i^H\f$.
///
/// Workspace is leased from the queue's arena. This spelling reports no per-item
/// convergence status; use the positional syev() with an `info` span for that.
/// @param A     n x n per item; the `uplo` triangle is read. Overwritten with the
///              eigenvectors (columns) when `opts.jobz == JobType::EigenVectors`,
///              left unspecified otherwise
/// @param W     real, at least `n * batch_size` entries; eigenvalues ascending per item
/// @param opts  jobz, uplo
/// @return      event of the last enqueued kernel
/// @throws batchlas::invalid_argument if `A` is not square or `W` is too short
/// @throws std::invalid_argument if a pointer is not reachable from the queue's device
template <detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
inline Event syev(Queue& ctx, const MA& A, Span<typename base_type<T>::type> W,
                  const SyevOptions& opts = {}) {
    BATCHLAS_CHECK_ARGS(ctx, "syev", A, W);
    detail::require_square("syev", "A", A);
    detail::require_span_at_least("syev", "W", W.size(),
                                  static_cast<size_t>(A.rows()) * A.batch_size());
    return with_backend(ctx, [&](auto Back) { return syev<Back.value>(ctx, A, W, opts); });
}

// ---- ormqr and gesvd -------------------------------------------------------
// As with getrs, the option forms do not mirror the positional parameter order.

/// @brief Options for ormqr(): apply geqrf()'s Q, \f$C := \mathrm{op}(Q)C\f$ (Left) or \f$C\,\mathrm{op}(Q)\f$ (Right).
struct OrmqrOptions {
    Side side = Side::Left;                ///< which side Q multiplies from
    Transpose trans = Transpose::NoTrans;  ///< op(Q)
    int32_t block_size_hint = 0;           ///< WY panel width; 0 lets the tuning table pick
};

/// @brief ormqr with a compile-time backend and a caller workspace; no shape checks.
/// @tparam B  backend to run on
template <Backend B, detail::DenseMatrixLike MA, detail::DenseMatrixLike MC,
          typename T = detail::dense_scalar_t<MA>>
inline Event ormqr(Queue& ctx, const MA& A, const MC& C, Span<T> tau,
                   const OrmqrOptions& opts, Span<std::byte> ws) {
    using V = BATCHLAS_DENSE_VIEW(T);
    return ormqr<B, T>(ctx, V(A), V(C), opts.side, opts.trans, tau, ws, opts.block_size_hint);
}

/// @brief ormqr with a compile-time backend; workspace leased from the arena, no shape checks.
/// @tparam B  backend to run on
template <Backend B, detail::DenseMatrixLike MA, detail::DenseMatrixLike MC,
          typename T = detail::dense_scalar_t<MA>>
inline Event ormqr(Queue& ctx, const MA& A, const MC& C, Span<T> tau,
                   const OrmqrOptions& opts) {
    using V = BATCHLAS_DENSE_VIEW(T);
    // Size with the same hint the call uses, or the buffer is under-sized.
    auto lease = ctx.workspace(ormqr_buffer_size<B, T>(ctx, V(A), V(C), opts.side, opts.trans,
                                                       tau, opts.block_size_hint));
    return ormqr<B, T>(ctx, V(A), V(C), opts.side, opts.trans, tau, lease.span(),
                       opts.block_size_hint);
}

/// @brief Apply Q from geqrf() with a caller-owned workspace; otherwise as the arena overload.
/// @param ws  device-accessible scratch of at least `ormqr_buffer_size` bytes, sized
///            with the same `block_size_hint`
template <detail::DenseMatrixLike MA, detail::DenseMatrixLike MC,
          typename T = detail::dense_scalar_t<MA>>
inline Event ormqr(Queue& ctx, const MA& A, const MC& C, Span<T> tau,
                   const OrmqrOptions& opts, Span<std::byte> ws) {
    BATCHLAS_CHECK_ARGS(ctx, "ormqr", A, C, tau, ws);
    detail::require_same_batch("ormqr", "A", A, "C", C);
    detail::require_span_at_least("ormqr", "tau", tau.size(),
                                  static_cast<size_t>(std::min(A.rows(), A.cols())) * A.batch_size());
    return with_backend(ctx, [&](auto Back) { return ormqr<Back.value>(ctx, A, C, tau, opts, ws); });
}

/// @brief Apply the Q of geqrf() to C: \f$C_i := \mathrm{op}(Q_i)C_i\f$ (Left) or \f$C_i\,\mathrm{op}(Q_i)\f$ (Right).
///
/// Workspace is leased from the queue's arena.
/// @param A     geqrf()'s output (reflectors below the diagonal), m x k per item
/// @param C     overwritten with the product
/// @param tau   geqrf()'s scalars, at least `min(A.rows(), A.cols()) * batch_size`
/// @param opts  side, trans, block_size_hint
/// @return      event of the last enqueued kernel
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
    return with_backend(ctx, [&](auto Back) { return ormqr<Back.value>(ctx, A, C, tau, opts); });
}

/// @brief Options for gesvd(): \f$A = U\Sigma V^H\f$.
struct GesvdOptions {
    SvdVectors jobu = SvdVectors::All;   ///< which columns of U to compute
    SvdVectors jobvh = SvdVectors::All;  ///< which rows of V^H to compute
    /// Engaged: A is Hermitian and the given triangle is read (square only); this
    /// selects a different entry point, hence optional rather than a sentinel.
    std::optional<Uplo> hermitian_uplo = std::nullopt;
};

/// @brief gesvd with a compile-time backend and a caller workspace; no shape checks.
/// @tparam B  backend to run on
template <Backend B, detail::DenseMatrixLike MA, detail::DenseMatrixLike MU,
          detail::DenseMatrixLike MV, typename T = detail::dense_scalar_t<MA>>
inline Event gesvd(Queue& ctx, const MA& A, Span<typename base_type<T>::type> singular_values,
                   const MU& U, const MV& Vh, const GesvdOptions& opts, Span<std::byte> ws) {
    using V = BATCHLAS_DENSE_VIEW(T);
    return opts.hermitian_uplo
               ? gesvd<B, T>(ctx, V(A), singular_values, V(U), V(Vh), opts.jobu, opts.jobvh,
                             *opts.hermitian_uplo, ws)
               : gesvd<B, T>(ctx, V(A), singular_values, V(U), V(Vh), opts.jobu, opts.jobvh, ws);
}

/// @brief gesvd with a compile-time backend; workspace leased from the arena, no shape checks.
/// @tparam B  backend to run on
template <Backend B, detail::DenseMatrixLike MA, detail::DenseMatrixLike MU,
          detail::DenseMatrixLike MV, typename T = detail::dense_scalar_t<MA>>
inline Event gesvd(Queue& ctx, const MA& A, Span<typename base_type<T>::type> singular_values,
                   const MU& U, const MV& Vh, const GesvdOptions& opts) {
    using V = BATCHLAS_DENSE_VIEW(T);
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
/// @param ws  device-accessible scratch of at least `gesvd_buffer_size` bytes, sized
///            for the same `hermitian_uplo` branch
template <detail::DenseMatrixLike MA, detail::DenseMatrixLike MU, detail::DenseMatrixLike MV,
          typename T = detail::dense_scalar_t<MA>>
inline Event gesvd(Queue& ctx, const MA& A, Span<typename base_type<T>::type> singular_values,
                   const MU& U, const MV& Vh, const GesvdOptions& opts, Span<std::byte> ws) {
    BATCHLAS_CHECK_ARGS(ctx, "gesvd", A, singular_values, U, Vh, ws);
    if (opts.hermitian_uplo) detail::require_square("gesvd", "A", A);
    detail::require_span_at_least("gesvd", "singular_values", singular_values.size(),
                                  static_cast<size_t>(std::min(A.rows(), A.cols())) * A.batch_size());
    return with_backend(
        ctx, [&](auto Back) { return gesvd<Back.value>(ctx, A, singular_values, U, Vh, opts, ws); });
}

/// @brief Batched singular value decomposition \f$A_i = U_i\Sigma_iV_i^H\f$.
///
/// Workspace is leased from the queue's arena. `A` is overwritten. This spelling
/// reports no per-item convergence status; use the positional gesvd() with an
/// `info` span for that.
/// @param A                m x n per item (square when `hermitian_uplo` is engaged)
/// @param singular_values  real, at least `min(m, n) * batch_size`; descending per item
/// @param U                m x m (All) or m x k (Thin); default-constructed for None
/// @param Vh               n x n (All) or k x n (Thin); default-constructed for None
/// @param opts             jobu, jobvh, hermitian_uplo
/// @return                 event of the last enqueued kernel
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
    return with_backend(
        ctx, [&](auto Back) { return gesvd<Back.value>(ctx, A, singular_values, U, Vh, opts); });
}

/// @}

#undef BATCHLAS_DENSE_VIEW

}  // namespace batchlas
