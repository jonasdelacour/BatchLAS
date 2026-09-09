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

// Option-struct spellings of the public entry points, e.g.
// `gemm(ctx, A, B, C, {.alpha = 2.0f, .transA = Transpose::Trans})`. The backend
// comes from the Queue; T is deduced from the matrix arguments, never from the
// option struct -- an option struct in a deduced position would make
// `{.alpha = 2.0f}` ill-formed. The *Params structs are a different thing; see
// docs/cpp-api.md.
namespace batchlas {

namespace detail {

// Named-argument counterpart to require_pack_accessible (blas/queue-dispatch.hh):
// these overloads know their parameter names, so the error says "gemm: A".
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

// Shape preconditions for the LAPACK-style entry points; they throw
// std::invalid_argument (caller error), never std::runtime_error, which is
// reserved for environment and backend failures.
//
// Repeating them on BOTH the deducing and the `template <Backend B, ...>`
// overloads is deliberate: BATCHLAS_DISPATCH_ON_QUEUE's variadic
// `NAME(Queue&, Args&&...)` binds a prvalue better than `const MA&`, so
// `getrf(ctx, A.view(), pivots)` reaches the <Backend> overload past every check
// on the deducing one. The positional primaries in blas/functions/*.hh that take
// an explicit workspace stay unchecked deliberately: src/extensions/ calls them
// per iteration and they must not pay a host-side branch.
template <typename MV>
inline void require_square(const char* fn, const char* name, const MV& A) {
    if (A.rows() != A.cols())
        throw std::invalid_argument(std::string(fn) + ": " + name +
            " must be square, got " + std::to_string(A.rows()) + "x" + std::to_string(A.cols()));
}

template <typename MA, typename MB>
inline void require_same_rows(const char* fn, const char* an, const MA& A,
                              const char* bn, const MB& B) {
    if (A.rows() != B.rows())
        throw std::invalid_argument(std::string(fn) + ": " + an + ".rows() (" +
            std::to_string(A.rows()) + ") must equal " + bn + ".rows() (" +
            std::to_string(B.rows()) + ")");
}

template <typename MA, typename MB>
inline void require_same_batch(const char* fn, const char* an, const MA& A,
                               const char* bn, const MB& B) {
    if (A.batch_size() != B.batch_size())
        throw std::invalid_argument(std::string(fn) + ": " + an + " and " + bn +
            " must have the same batch size (" + std::to_string(A.batch_size()) + " vs " +
            std::to_string(B.batch_size()) + ")");
}

// `>=`, not `==`: an oversized output buffer is legitimate -- a caller may slice
// one big pivot/tau arena across several calls.
inline void require_span_at_least(const char* fn, const char* name, size_t have, size_t need) {
    if (have < need)
        throw std::invalid_argument(std::string(fn) + ": " + name + " holds " +
            std::to_string(have) + " elements, needs at least " + std::to_string(need));
}

// An EMPTY `info` span spells "no status wanted", so only a non-empty one is
// measured. A too-short one fails silently: the backend falls back to its own
// scratch and the caller reads stale bytes, most often zeros ("all factorised").
inline void require_info_span(const char* fn, size_t have, size_t batch_size) {
    if (have != 0) require_span_at_least(fn, "info", have, batch_size);
}

}  // namespace detail

// ---- dense BLAS ------------------------------------------------------------

template <typename T>
struct GemmOptions {
    T alpha = T(1);
    T beta = T(0);
    Transpose transA = Transpose::NoTrans;
    Transpose transB = Transpose::NoTrans;
    ComputePrecision precision = ComputePrecision::Default;
};

template <typename T>
struct GemvOptions {
    T alpha = T(1);
    T beta = T(0);
    Transpose transA = Transpose::NoTrans;
};

template <typename T>
struct SymmOptions {
    T alpha = T(1);
    T beta = T(0);
    Side side = Side::Left;
    Uplo uplo = Uplo::Lower;
};

template <typename T>
struct HemmOptions {
    T alpha = T(1);
    T beta = T(0);
    Side side = Side::Left;
    Uplo uplo = Uplo::Lower;
};

// herk's alpha and beta are real even though its operands are complex: a complex
// alpha would make alpha * A * A^H non-Hermitian.
template <typename T>
struct HerkOptions {
    float_t<T> alpha = float_t<T>(1);
    float_t<T> beta = float_t<T>(0);
    Uplo uplo = Uplo::Lower;
    Transpose trans = Transpose::NoTrans;
};

// her2k pairs alpha * A * B^H with its own conjugate transpose, Hermitian for any
// alpha -- so alpha is complex here and only beta is real.
template <typename T>
struct Her2kOptions {
    T alpha = T(1);
    float_t<T> beta = float_t<T>(0);
    Uplo uplo = Uplo::Lower;
    Transpose trans = Transpose::NoTrans;
};

template <typename T>
struct SyrkOptions {
    T alpha = T(1);
    T beta = T(0);
    Uplo uplo = Uplo::Lower;
    Transpose trans = Transpose::NoTrans;
};

template <typename T>
struct Syr2kOptions {
    T alpha = T(1);
    T beta = T(0);
    Uplo uplo = Uplo::Lower;
    Transpose trans = Transpose::NoTrans;
};

template <typename T>
struct TrmmOptions {
    T alpha = T(1);
    Side side = Side::Left;
    Uplo uplo = Uplo::Lower;
    Transpose trans = Transpose::NoTrans;
    Diag diag = Diag::NonUnit;
};

template <typename T>
struct TrsmOptions {
    T alpha = T(1);
    Side side = Side::Left;
    Uplo uplo = Uplo::Lower;
    Transpose trans = Transpose::NoTrans;
    Diag diag = Diag::NonUnit;
};

// Each entry point has two spellings: `gemm(ctx, ...)` takes the backend from the
// queue, `gemm<B>(ctx, ...)` fixes it at compile time. src/extensions/ is
// templated on Backend and must use the second -- the runtime one would silently
// use ctx.backend() instead of the B the algorithm was instantiated for.

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

// Do not collapse the two workspace spellings into one function with
// `Span<std::byte> ws = {}` and a null check. A null span is not a synonym for
// "not passed": in a BumpAllocator sizing pass every pool allocation hands back
// an empty span while the input matrices stay real, so a null check would run the
// real factorisation over the caller's live data. Nothing crashes; the corrupted
// matrix surfaces later as an algorithm that stops converging.

#define BATCHLAS_DENSE_VIEW(T) MatrixView<T, MatrixFormat::Dense>

// ---- dense BLAS ------------------------------------------------------------

template <Backend Back, detail::DenseMatrixLike MA, detail::DenseMatrixLike MB,
          detail::DenseMatrixLike MC, typename T = detail::dense_scalar_t<MA>>
inline Event gemm(Queue& ctx, const MA& A, const MB& B, const MC& C, const GemmOptions<T>& opts) {
    using V = BATCHLAS_DENSE_VIEW(T);
    return gemm<Back, T>(ctx, V(A), V(B), V(C), opts.alpha, opts.beta, opts.transA, opts.transB,
                         opts.precision);
}

template <detail::DenseMatrixLike MA, detail::DenseMatrixLike MB, detail::DenseMatrixLike MC,
          typename T = detail::dense_scalar_t<MA>>
inline Event gemm(Queue& ctx, const MA& A, const MB& B, const MC& C, const GemmOptions<T>& opts) {
    BATCHLAS_CHECK_ARGS(ctx, "gemm", A, B, C);
    return with_backend(ctx, [&](auto Back) { return gemm<Back.value>(ctx, A, B, C, opts); });
}

template <Backend Back, detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
inline Event gemv(Queue& ctx, const MA& A, const VectorView<T>& x, const VectorView<T>& y,
                  const GemvOptions<T>& opts) {
    using V = BATCHLAS_DENSE_VIEW(T);
    return gemv<Back, T>(ctx, V(A), x, y, opts.alpha, opts.beta, opts.transA);
}

template <detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
inline Event gemv(Queue& ctx, const MA& A, const VectorView<T>& x, const VectorView<T>& y,
                  const GemvOptions<T>& opts) {
    BATCHLAS_CHECK_ARGS(ctx, "gemv", A, x, y);
    return with_backend(ctx, [&](auto Back) { return gemv<Back.value>(ctx, A, x, y, opts); });
}

template <Backend Back, detail::DenseMatrixLike MA, detail::DenseMatrixLike MB,
          detail::DenseMatrixLike MC, typename T = detail::dense_scalar_t<MA>>
inline Event symm(Queue& ctx, const MA& A, const MB& B, const MC& C, const SymmOptions<T>& opts) {
    using V = BATCHLAS_DENSE_VIEW(T);
    return symm<Back, T>(ctx, V(A), V(B), V(C), opts.alpha, opts.beta, opts.side, opts.uplo);
}

template <detail::DenseMatrixLike MA, detail::DenseMatrixLike MB, detail::DenseMatrixLike MC,
          typename T = detail::dense_scalar_t<MA>>
inline Event symm(Queue& ctx, const MA& A, const MB& B, const MC& C, const SymmOptions<T>& opts) {
    BATCHLAS_CHECK_ARGS(ctx, "symm", A, B, C);
    return with_backend(ctx, [&](auto Back) { return symm<Back.value>(ctx, A, B, C, opts); });
}

template <Backend Back, detail::DenseMatrixLike MA, detail::DenseMatrixLike MB,
          detail::DenseMatrixLike MC, typename T = detail::dense_scalar_t<MA>>
    requires ComplexScalar<T>
inline Event hemm(Queue& ctx, const MA& A, const MB& B, const MC& C, const HemmOptions<T>& opts) {
    using V = BATCHLAS_DENSE_VIEW(T);
    return hemm<Back, T>(ctx, V(A), V(B), V(C), opts.alpha, opts.beta, opts.side, opts.uplo);
}

template <detail::DenseMatrixLike MA, detail::DenseMatrixLike MB, detail::DenseMatrixLike MC,
          typename T = detail::dense_scalar_t<MA>>
    requires ComplexScalar<T>
inline Event hemm(Queue& ctx, const MA& A, const MB& B, const MC& C, const HemmOptions<T>& opts) {
    BATCHLAS_CHECK_ARGS(ctx, "hemm", A, B, C);
    return with_backend(ctx, [&](auto Back) { return hemm<Back.value>(ctx, A, B, C, opts); });
}

template <Backend Back, detail::DenseMatrixLike MA, detail::DenseMatrixLike MC,
          typename T = detail::dense_scalar_t<MA>>
    requires ComplexScalar<T>
inline Event herk(Queue& ctx, const MA& A, const MC& C, const HerkOptions<T>& opts) {
    using V = BATCHLAS_DENSE_VIEW(T);
    return herk<Back, T>(ctx, V(A), V(C), opts.alpha, opts.beta, opts.uplo, opts.trans);
}

template <detail::DenseMatrixLike MA, detail::DenseMatrixLike MC,
          typename T = detail::dense_scalar_t<MA>>
    requires ComplexScalar<T>
inline Event herk(Queue& ctx, const MA& A, const MC& C, const HerkOptions<T>& opts) {
    BATCHLAS_CHECK_ARGS(ctx, "herk", A, C);
    return with_backend(ctx, [&](auto Back) { return herk<Back.value>(ctx, A, C, opts); });
}

template <Backend Back, detail::DenseMatrixLike MA, detail::DenseMatrixLike MB,
          detail::DenseMatrixLike MC, typename T = detail::dense_scalar_t<MA>>
    requires ComplexScalar<T>
inline Event her2k(Queue& ctx, const MA& A, const MB& B, const MC& C,
                   const Her2kOptions<T>& opts) {
    using V = BATCHLAS_DENSE_VIEW(T);
    return her2k<Back, T>(ctx, V(A), V(B), V(C), opts.alpha, opts.beta, opts.uplo, opts.trans);
}

template <detail::DenseMatrixLike MA, detail::DenseMatrixLike MB, detail::DenseMatrixLike MC,
          typename T = detail::dense_scalar_t<MA>>
    requires ComplexScalar<T>
inline Event her2k(Queue& ctx, const MA& A, const MB& B, const MC& C,
                   const Her2kOptions<T>& opts) {
    BATCHLAS_CHECK_ARGS(ctx, "her2k", A, B, C);
    return with_backend(ctx, [&](auto Back) { return her2k<Back.value>(ctx, A, B, C, opts); });
}

template <Backend Back, detail::DenseMatrixLike MA, detail::DenseMatrixLike MC,
          typename T = detail::dense_scalar_t<MA>>
inline Event syrk(Queue& ctx, const MA& A, const MC& C, const SyrkOptions<T>& opts) {
    using V = BATCHLAS_DENSE_VIEW(T);
    return syrk<Back, T>(ctx, V(A), V(C), opts.alpha, opts.beta, opts.uplo, opts.trans);
}

template <detail::DenseMatrixLike MA, detail::DenseMatrixLike MC,
          typename T = detail::dense_scalar_t<MA>>
inline Event syrk(Queue& ctx, const MA& A, const MC& C, const SyrkOptions<T>& opts) {
    BATCHLAS_CHECK_ARGS(ctx, "syrk", A, C);
    return with_backend(ctx, [&](auto Back) { return syrk<Back.value>(ctx, A, C, opts); });
}

template <Backend Back, detail::DenseMatrixLike MA, detail::DenseMatrixLike MB,
          detail::DenseMatrixLike MC, typename T = detail::dense_scalar_t<MA>>
inline Event syr2k(Queue& ctx, const MA& A, const MB& B, const MC& C,
                   const Syr2kOptions<T>& opts) {
    using V = BATCHLAS_DENSE_VIEW(T);
    return syr2k<Back, T>(ctx, V(A), V(B), V(C), opts.alpha, opts.beta, opts.uplo, opts.trans);
}

template <detail::DenseMatrixLike MA, detail::DenseMatrixLike MB, detail::DenseMatrixLike MC,
          typename T = detail::dense_scalar_t<MA>>
inline Event syr2k(Queue& ctx, const MA& A, const MB& B, const MC& C,
                   const Syr2kOptions<T>& opts) {
    BATCHLAS_CHECK_ARGS(ctx, "syr2k", A, B, C);
    return with_backend(ctx, [&](auto Back) { return syr2k<Back.value>(ctx, A, B, C, opts); });
}

template <Backend Back, detail::DenseMatrixLike MA, detail::DenseMatrixLike MB,
          detail::DenseMatrixLike MC, typename T = detail::dense_scalar_t<MA>>
inline Event trmm(Queue& ctx, const MA& A, const MB& B, const MC& C, const TrmmOptions<T>& opts) {
    using V = BATCHLAS_DENSE_VIEW(T);
    return trmm<Back, T>(ctx, V(A), V(B), V(C), opts.alpha, opts.side, opts.uplo, opts.trans,
                         opts.diag);
}

template <detail::DenseMatrixLike MA, detail::DenseMatrixLike MB, detail::DenseMatrixLike MC,
          typename T = detail::dense_scalar_t<MA>>
inline Event trmm(Queue& ctx, const MA& A, const MB& B, const MC& C, const TrmmOptions<T>& opts) {
    BATCHLAS_CHECK_ARGS(ctx, "trmm", A, B, C);
    return with_backend(ctx, [&](auto Back) { return trmm<Back.value>(ctx, A, B, C, opts); });
}

template <Backend Back, detail::DenseMatrixLike MA, detail::DenseMatrixLike MB,
          typename T = detail::dense_scalar_t<MA>>
inline Event trsm(Queue& ctx, const MA& A, const MB& B, const TrsmOptions<T>& opts) {
    using V = BATCHLAS_DENSE_VIEW(T);
    return trsm<Back, T>(ctx, V(A), V(B), opts.alpha, opts.side, opts.uplo, opts.trans, opts.diag);
}

template <detail::DenseMatrixLike MA, detail::DenseMatrixLike MB,
          typename T = detail::dense_scalar_t<MA>>
inline Event trsm(Queue& ctx, const MA& A, const MB& B, const TrsmOptions<T>& opts) {
    BATCHLAS_CHECK_ARGS(ctx, "trsm", A, B);
    return with_backend(ctx, [&](auto Back) { return trsm<Back.value>(ctx, A, B, opts); });
}

// ---- dense LAPACK ----------------------------------------------------------
//
// Omitting the workspace leases scratch from the queue's arena, sized by the matching
// *_buffer_size. Releasing that lease on an OUT-OF-ORDER queue drains the queue,
// so every arena spelling below blocks until the device is idle; pass your own
// span to keep the call asynchronous. See batchlas/util/workspace.hh.

struct PotrfOptions {
    Uplo uplo = Uplo::Lower;
    // Per-item LAPACK status, one int32 per batch item: 0 = factorised, >0 = the
    // leading minor at which positive-definiteness failed. Empty (the default)
    // reports nothing. Must be device-accessible (USM); written in place.
    Span<int32_t> info = {};
};

struct GetrsOptions {
    Transpose trans = Transpose::NoTrans;
};

struct SyevOptions {
    JobType jobz = JobType::EigenVectors;
    Uplo uplo = Uplo::Lower;
};

template <Backend B, detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
inline Event potrf(Queue& ctx, const MA& A, const PotrfOptions& opts, Span<std::byte> ws) {
    using V = BATCHLAS_DENSE_VIEW(T);
    return potrf<B, T>(ctx, V(A), opts.uplo, ws, opts.info);
}

template <Backend B, detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
inline Event potrf(Queue& ctx, const MA& A, const PotrfOptions& opts) {
    using V = BATCHLAS_DENSE_VIEW(T);
    detail::require_square("potrf", "A", A);
    detail::require_info_span("potrf", opts.info.size(), static_cast<size_t>(A.batch_size()));
    auto lease = ctx.workspace(potrf_buffer_size<B, T>(ctx, V(A), opts.uplo));
    return potrf<B, T>(ctx, V(A), opts.uplo, lease.span(), opts.info);
}

template <detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
inline Event potrf(Queue& ctx, const MA& A, const PotrfOptions& opts, Span<std::byte> ws) {
    BATCHLAS_CHECK_ARGS(ctx, "potrf", A, ws, opts.info);
    detail::require_square("potrf", "A", A);
    detail::require_info_span("potrf", opts.info.size(), static_cast<size_t>(A.batch_size()));
    return with_backend(ctx, [&](auto Back) { return potrf<Back.value>(ctx, A, opts, ws); });
}

template <detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
inline Event potrf(Queue& ctx, const MA& A, const PotrfOptions& opts = {}) {
    BATCHLAS_CHECK_ARGS(ctx, "potrf", A, opts.info);
    detail::require_square("potrf", "A", A);
    detail::require_info_span("potrf", opts.info.size(), static_cast<size_t>(A.batch_size()));
    return with_backend(ctx, [&](auto Back) { return potrf<Back.value>(ctx, A, opts); });
}

namespace detail {
// `potrf`'s option and positional overloads have the same arity and both accept
// `{}`, which matches the enum exactly but the option struct only by a
// user-defined conversion -- so `potrf(ctx, A, {}, ws)` silently meant
// Uplo{} == Upper and factorised the opposite triangle. This third exact-match
// candidate makes the bare-`{}` call ambiguous instead; `PotrfOptions{}` and
// `Uplo::Lower` still resolve as before, and neither converts to this type.
enum class EmptyBracesAreAmbiguous {};
}  // namespace detail

template <Backend B, detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
Event potrf(Queue&, const MA&, detail::EmptyBracesAreAmbiguous, Span<std::byte>) = delete;

template <detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
Event potrf(Queue&, const MA&, detail::EmptyBracesAreAmbiguous, Span<std::byte>) = delete;

// getrf, getri, geqrf and orgqr carry no options, so their only new spelling is
// the arena-backed one. They deliberately take no workspace parameter -- such an
// overload would be ambiguous with the positional call. The trailing `info` is
// the per-item LAPACK status (one int32 per batch item, 0 = success); empty, the
// default, reports nothing.
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

template <detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
inline Event getrf(Queue& ctx, const MA& A, Span<int64_t> pivots, Span<int32_t> info = {}) {
    BATCHLAS_CHECK_ARGS(ctx, "getrf", A, pivots, info);
    detail::require_square("getrf", "A", A);
    detail::require_span_at_least("getrf", "pivots", pivots.size(),
                                  static_cast<size_t>(A.rows()) * A.batch_size());
    detail::require_info_span("getrf", info.size(), static_cast<size_t>(A.batch_size()));
    return with_backend(ctx, [&](auto Back) { return getrf<Back.value>(ctx, A, pivots, info); });
}

template <Backend B, detail::DenseMatrixLike MA, detail::DenseMatrixLike MB,
          typename T = detail::dense_scalar_t<MA>>
inline Event getrs(Queue& ctx, const MA& A, const MB& B_, Span<int64_t> pivots,
                   const GetrsOptions& opts, Span<std::byte> ws) {
    using V = BATCHLAS_DENSE_VIEW(T);
    return getrs<B, T>(ctx, V(A), V(B_), opts.trans, pivots, ws);
}

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

template <Backend B, detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
inline Event geqrf(Queue& ctx, const MA& A, Span<T> tau) {
    using V = BATCHLAS_DENSE_VIEW(T);
    detail::require_span_at_least("geqrf", "tau", tau.size(),
                                  static_cast<size_t>(std::min(A.rows(), A.cols())) * A.batch_size());
    auto lease = ctx.workspace(geqrf_buffer_size<B, T>(ctx, V(A), tau));
    return geqrf<B, T>(ctx, V(A), tau, lease.span());
}

template <detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
inline Event geqrf(Queue& ctx, const MA& A, Span<T> tau) {
    BATCHLAS_CHECK_ARGS(ctx, "geqrf", A, tau);
    // No squareness check: rectangular A is the point of geqrf. What is fixed is
    // the tau stride -- every backend indexes `tau.data() + i * min(m, n)`.
    detail::require_span_at_least("geqrf", "tau", tau.size(),
                                  static_cast<size_t>(std::min(A.rows(), A.cols())) * A.batch_size());
    return with_backend(ctx, [&](auto Back) { return geqrf<Back.value>(ctx, A, tau); });
}

template <Backend B, detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
inline Event orgqr(Queue& ctx, const MA& A, Span<T> tau) {
    using V = BATCHLAS_DENSE_VIEW(T);
    detail::require_span_at_least("orgqr", "tau", tau.size(),
                                  static_cast<size_t>(std::min(A.rows(), A.cols())) * A.batch_size());
    auto lease = ctx.workspace(orgqr_buffer_size<B, T>(ctx, V(A), tau));
    return orgqr<B, T>(ctx, V(A), tau, lease.span());
}

template <detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
inline Event orgqr(Queue& ctx, const MA& A, Span<T> tau) {
    BATCHLAS_CHECK_ARGS(ctx, "orgqr", A, tau);
    detail::require_span_at_least("orgqr", "tau", tau.size(),
                                  static_cast<size_t>(std::min(A.rows(), A.cols())) * A.batch_size());
    return with_backend(ctx, [&](auto Back) { return orgqr<Back.value>(ctx, A, tau); });
}

template <Backend B, detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
inline Event syev(Queue& ctx, const MA& A, Span<typename base_type<T>::type> W,
                  const SyevOptions& opts, Span<std::byte> ws) {
    using V = BATCHLAS_DENSE_VIEW(T);
    return syev<B, T>(ctx, V(A), W, opts.jobz, opts.uplo, ws);
}

template <Backend B, detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
inline Event syev(Queue& ctx, const MA& A, Span<typename base_type<T>::type> W,
                  const SyevOptions& opts) {
    using V = BATCHLAS_DENSE_VIEW(T);
    auto lease = ctx.workspace(syev_buffer_size<B, T>(ctx, V(A), W, opts.jobz, opts.uplo));
    return syev<B, T>(ctx, V(A), W, opts.jobz, opts.uplo, lease.span());
}

template <detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
inline Event syev(Queue& ctx, const MA& A, Span<typename base_type<T>::type> W,
                  const SyevOptions& opts, Span<std::byte> ws) {
    BATCHLAS_CHECK_ARGS(ctx, "syev", A, W, ws);
    detail::require_square("syev", "A", A);
    detail::require_span_at_least("syev", "W", W.size(),
                                  static_cast<size_t>(A.rows()) * A.batch_size());
    return with_backend(ctx, [&](auto Back) { return syev<Back.value>(ctx, A, W, opts, ws); });
}

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
//
// As with getrs, the option forms do not mirror the positional parameter order.

struct OrmqrOptions {
    Side side = Side::Left;
    Transpose trans = Transpose::NoTrans;
    // 0 lets the tuning table pick the WY panel width (ormqr_blocked.cc).
    int32_t block_size_hint = 0;
};

template <Backend B, detail::DenseMatrixLike MA, detail::DenseMatrixLike MC,
          typename T = detail::dense_scalar_t<MA>>
inline Event ormqr(Queue& ctx, const MA& A, const MC& C, Span<T> tau,
                   const OrmqrOptions& opts, Span<std::byte> ws) {
    using V = BATCHLAS_DENSE_VIEW(T);
    return ormqr<B, T>(ctx, V(A), V(C), opts.side, opts.trans, tau, ws, opts.block_size_hint);
}

template <Backend B, detail::DenseMatrixLike MA, detail::DenseMatrixLike MC,
          typename T = detail::dense_scalar_t<MA>>
inline Event ormqr(Queue& ctx, const MA& A, const MC& C, Span<T> tau,
                   const OrmqrOptions& opts) {
    using V = BATCHLAS_DENSE_VIEW(T);
    // Must size with the same hint the call uses: the hint picks the panel width
    // the workspace is sized for, so a mismatch under-sizes the buffer.
    auto lease = ctx.workspace(ormqr_buffer_size<B, T>(ctx, V(A), V(C), opts.side, opts.trans,
                                                       tau, opts.block_size_hint));
    return ormqr<B, T>(ctx, V(A), V(C), opts.side, opts.trans, tau, lease.span(),
                       opts.block_size_hint);
}

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

template <detail::DenseMatrixLike MA, detail::DenseMatrixLike MC,
          typename T = detail::dense_scalar_t<MA>>
inline Event ormqr(Queue& ctx, const MA& A, const MC& C, Span<T> tau,
                   const OrmqrOptions& opts = {}) {
    BATCHLAS_CHECK_ARGS(ctx, "ormqr", A, C, tau);
    // Every path indexes tau at a stride of min(A.rows(), A.cols()), so a short
    // tau is read out of bounds (mirrors validate_ormqr_dims, ormqr_blocked.cc).
    detail::require_same_batch("ormqr", "A", A, "C", C);
    detail::require_span_at_least("ormqr", "tau", tau.size(),
                                  static_cast<size_t>(std::min(A.rows(), A.cols())) * A.batch_size());
    return with_backend(ctx, [&](auto Back) { return ormqr<Back.value>(ctx, A, C, tau, opts); });
}

struct GesvdOptions {
    SvdVectors jobu = SvdVectors::All;
    SvdVectors jobvh = SvdVectors::All;
    // Engaged selects the Hermitian entry point (blas/functions/gesvd.hh) -- a
    // different overload, hence std::optional rather than a Uplo sentinel.
    std::optional<Uplo> hermitian_uplo = std::nullopt;
};

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

template <Backend B, detail::DenseMatrixLike MA, detail::DenseMatrixLike MU,
          detail::DenseMatrixLike MV, typename T = detail::dense_scalar_t<MA>>
inline Event gesvd(Queue& ctx, const MA& A, Span<typename base_type<T>::type> singular_values,
                   const MU& U, const MV& Vh, const GesvdOptions& opts) {
    using V = BATCHLAS_DENSE_VIEW(T);
    // The two branches pick providers independently and can need different
    // scratch, so the query must take the same branch as the call.
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

template <detail::DenseMatrixLike MA, detail::DenseMatrixLike MU, detail::DenseMatrixLike MV,
          typename T = detail::dense_scalar_t<MA>>
inline Event gesvd(Queue& ctx, const MA& A, Span<typename base_type<T>::type> singular_values,
                   const MU& U, const MV& Vh, const GesvdOptions& opts = {}) {
    BATCHLAS_CHECK_ARGS(ctx, "gesvd", A, singular_values, U, Vh);
    // Only A is shape-checked: a default-constructed U/Vh is how this API spells
    // SvdVectors::None, so any check against A's extents would reject it.
    if (opts.hermitian_uplo) detail::require_square("gesvd", "A", A);
    detail::require_span_at_least("gesvd", "singular_values", singular_values.size(),
                                  static_cast<size_t>(std::min(A.rows(), A.cols())) * A.batch_size());
    return with_backend(
        ctx, [&](auto Back) { return gesvd<Back.value>(ctx, A, singular_values, U, Vh, opts); });
}

#undef BATCHLAS_DENSE_VIEW

}  // namespace batchlas
