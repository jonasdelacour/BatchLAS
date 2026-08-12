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

// Option-struct spellings of the public entry points.
//
//     gemm(ctx, A, B, C, {.alpha = 2.0f, .transA = Transpose::Trans});
//     syev(ctx, A, W, {.jobz = JobType::EigenVectors});
//
// Structs here are named *Options and belong to this convenience layer: the
// backend comes from the Queue, T is deduced from the matrices, the struct
// carries every non-matrix argument, and the workspace may be omitted. The
// *Params structs in blas/extensions.hh and blas/functions/iluk.hh are a
// different thing -- ordinary arguments to entry points that have no
// convenience layer. The suffix does not tell you where the struct sits in the
// argument list; see docs/cpp-api.md, "*Options and *Params are two different
// things", for the table.
//
// rather than
//
//     gemm<Backend::CUDA, float>(ctx, A, B, C, 2.0f, 0.0f, Transpose::Trans,
//                                Transpose::NoTrans, ComputePrecision::Default);
//     { UnifiedVector<std::byte> ws(syev_buffer_size<Backend::CUDA>(ctx, A, W,
//           JobType::EigenVectors, Uplo::Lower));
//       syev<Backend::CUDA>(ctx, A, W, JobType::EigenVectors, Uplo::Lower, ws); }
//
// Three things make that shorter form work:
//   - the Backend comes from the Queue,
//   - the options carry their own defaults, so only what differs is written,
//   - the workspace defaults to a lease from the queue's arena, which collapses
//     the two-statement sizing dance into the call.
//
// T is deduced from the matrix arguments, never from the option struct. That is
// what makes `{.alpha = 2.0f}` work at the call: by the time the compiler looks
// at the option parameter its type is already fixed, so a braced initialiser has
// something concrete to initialise. An option struct in a deduced position would
// not compile.
namespace batchlas {

namespace detail {

// Named-argument counterpart to require_pack_accessible (blas/queue-dispatch.hh,
// where the shared machinery lives). The option-struct overloads know their
// parameter names, so they can say "gemm: A" where the variadic dispatch
// overload can only say "gemm: argument 2".
template <typename... Args>
inline void require_args_accessible(const Queue& ctx, const char* fn,
                                    const char* const* names, const Args&... args) {
    if (!pointer_checks_enabled()) return;
    int i = 0;
    (void)std::initializer_list<int>{
        (require_arg_accessible(ctx, args, std::string(fn) + ": " + names[i++]), 0)...};
}

}  // namespace detail

// Check the pointer arguments of an entry point. The stringised names make the
// error say which argument is wrong, which is the entire value of the check.
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

// Shape preconditions for the LAPACK-style entry points.
//
// The dense BLAS backends have always validated shapes (see the SYMM/HERK/TRMM
// checks in src/backends/cublas.cc); the LAPACK-style calls did not, and reached
// the vendor call exactly as written. That gap was not merely untidy, it was a
// memory-safety hole: netlib getrf reads `n = A.rows()` and then factorises an
// n x n block (src/backends/netlib_lapack.cc), so a 100x50 A -- a buffer of 5000
// elements -- was factorised as 100x100 and columns 50..99 were read and written
// past the end of the allocation. cuBLAS's batched getrf takes a single
// dimension and is square-only by construction, while rocSOLVER's does pass both
// rows and cols and genuinely handles rectangular input, so the three backends
// silently disagreed about what a rectangular A even meant. Pinning the contract
// host-side is what makes them agree.
//
// These throw std::invalid_argument, not std::runtime_error: everything they
// test is determined entirely by the caller's arguments. std::runtime_error is
// reserved for environment and backend failures.
//
// They are attached to the convenience overloads in THIS header -- both the
// backend-deducing ones and the `template <Backend B, ...>` arena/option ones.
// Attaching them to the deducing overloads alone was not enough, and the reason
// is worth keeping: BATCHLAS_DISPATCH_ON_QUEUE generates a variadic
// `NAME(Queue&, Args&&...)`, and for a call like `getrf(ctx, A.view(), pivots)`
// -- concrete argument types, no option struct, and `A.view()` a prvalue --
// `Args&&` binds the prvalue better than `const MA&` does, so the variadic WINS
// overload resolution and forwards straight to the `<Backend>` overload,
// stepping over every check on the deducing one. It was measured: a 8x4 A handed
// to `getrf(ctx, A.view(), pivots)` sailed through the squareness check. potrf
// hid the problem because its option struct is passed as a braced-init-list,
// which a parameter pack cannot deduce, so there the variadic drops out.
//
// What is still deliberately unchecked is the true inner loop: the positional
// primaries declared in blas/functions/*.hh that take an explicit workspace, and
// which src/extensions/ortho.cc, inv.cc and syevx_*.cc call per iteration. Those
// are shape-correct by construction and must not pay a host-side branch.
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

// `>=`, not `==`: an oversized output buffer is a legitimate thing to pass (a
// caller slicing one big pivot/tau arena across several calls), and rejecting it
// would break working code for no safety gain.
inline void require_span_at_least(const char* fn, const char* name, size_t have, size_t need) {
    if (have < need)
        throw std::invalid_argument(std::string(fn) + ": " + name + " holds " +
            std::to_string(have) + " elements, needs at least " + std::to_string(need));
}

// The per-item `info` output of the factorisations (potrf/getrf/getri). An
// EMPTY span is the API's spelling for "I do not want status", so it has no
// length to be wrong; only a non-empty one is measured.
//
// The check matters more here than for pivots, because the failure is silent
// rather than loud: a backend handed a too-short info span falls back to its
// own scratch allocation and writes nothing to the caller's buffer, so the
// caller reads whatever was already there -- most often zeros, i.e. "every item
// factorised" -- on precisely the batch it was trying to diagnose.
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

// hemm's A is Hermitian rather than symmetric; see blas/functions/hemm.hh.
template <typename T>
struct HemmOptions {
    T alpha = T(1);
    T beta = T(0);
    Side side = Side::Left;
    Uplo uplo = Uplo::Lower;
};

// herk's alpha and beta are real even though its operands are complex: a
// complex alpha would make alpha * A * A^H non-Hermitian. See
// blas/functions/herk.hh.
template <typename T>
struct HerkOptions {
    float_t<T> alpha = float_t<T>(1);
    float_t<T> beta = float_t<T>(0);
    Uplo uplo = Uplo::Lower;
    Transpose trans = Transpose::NoTrans;
};

// her2k pairs alpha * A * B^H with its own conjugate transpose, which is
// Hermitian for any alpha -- so alpha is complex here and only beta is real.
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

// Each entry point gets two option-struct spellings:
//
//   gemm(ctx, A, B, C, {...})       backend from the queue        (callers)
//   gemm<B>(ctx, A, B, C, {...})    backend fixed at compile time (library internals)
//
// The second exists because most of src/extensions/ is itself templated on
// Backend and must stay that way: propagating B is the whole point, and going
// through the queue there would re-dispatch at runtime on every inner call and,
// worse, silently use ctx.backend() instead of the B the algorithm was
// instantiated for. Those call sites still deserve to say `{.alpha = ...}`
// rather than counting positional arguments, so both spellings exist and the
// runtime one is written in terms of the compile-time one.
//
// The <B> forms are not ambiguous with the positional ones: no implicit
// conversion exists from Uplo/Transpose/T to an option struct, so an argument
// list either names an option struct or it does not.
//
// ---- how T is fixed ---------------------------------------------------------
//
// The matrix parameters are templates constrained to Matrix or MatrixView, and
// T is a *defaulted* template parameter computed from the first of them:
//
//     template <typename MA, ..., typename T = detail::dense_scalar_t<MA>>
//     Event gemm(Queue&, const MA& A, ..., const GemmOptions<T>& opts);
//
// Two things fall out of that, and both are the point.
//
// First, `{.alpha = 2.0f}` compiles. T is already fixed by the time the compiler
// considers the option parameter, so the braced initialiser has a concrete type
// to initialise. Deducing T *from* the option struct instead would make the
// option parameter a deduced context, and a braced initialiser deduces nothing.
//
// Second, `Matrix` and `MatrixView` are both accepted, and may be mixed, as they
// are on the positional entry points, which have a Matrix wrapper alongside the
// MatrixView primary. Everything is converted to MatrixView before the
// positional call, so mixing is fine.

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

// ---- why "workspace given" is an overload, not a null check -----------------
//
// Each workspace-taking entry point has two spellings: one that ends in a span,
// and one that does not and leases from the queue's arena instead. Write them as
// two overloads. Do not write a single function with `Span<std::byte> ws = {}`
// and an `if (ws.data() != nullptr)` inside.
//
// A null span is not a synonym for "I did not pass one". Library code that
// sub-allocates from a BumpAllocator runs the whole algorithm twice: once in
// sizing mode, where every pool allocation legitimately hands back an empty
// span, and once for real. The input matrices are real in *both* passes -- only
// the workspace-derived views are fabricated -- so a null check reads the sizing
// pass's empty span as "no workspace, allocate one and proceed" and executes the
// factorisation over the caller's live data during the measurement pass. Nothing
// crashes and no size comes out wrong; the corrupted matrix surfaces later as an
// algorithm that no longer converges.
//
// Two overloads remove the sentinel: an argument that is present is used exactly
// as given.

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
// These take workspace. Leaving it out leases from the queue's arena, sized by
// the matching *_buffer_size, which is what turns the usual three-step
// (size, allocate, call) into one call. Pass a span explicitly to keep control
// of it -- inside a larger algorithm that is already sub-allocating from its own
// pool, that is still the right thing to do.
//
// The lease is released when the call returns, so those bytes go to the next
// borrower rather than being freed. On an in-order queue the next borrower's
// work is ordered behind this call's, and the call returns as soon as the work
// is enqueued.
//
// On an OUT-OF-ORDER queue nothing orders the two, so releasing the lease drains
// the queue first -- these leases are innermost, which is the case that actually
// hands bytes back. Concretely: on an out-of-order Queue every overload below
// blocks until the device is idle before it returns, where the positional
// (caller-supplied span) spelling does not. That is the price of not managing
// the workspace yourself; pass your own span to keep the call asynchronous.
// See batchlas/util/workspace.hh.

struct PotrfOptions {
    Uplo uplo = Uplo::Lower;
    // Per-item LAPACK status, one int32 per batch item: 0 = factorised, >0 =
    // the leading minor at which the item stopped being positive definite.
    // Leave it empty (the default) and nothing is reported, which is what every
    // caller got before issue #73 -- a batch where item 37 was indefinite
    // returned an ordinary Event and the caller consumed the garbage. The span
    // must be device-accessible (USM); the vendor writes it in place.
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
// `potrf` is the one entry point whose option overload and positional overload
// have the same arity and both accept `{}`:
//
//     potrf(ctx, A, PotrfOptions{}, ws);   // uplo = Lower
//     potrf(ctx, A, Uplo::Lower,    ws);
//     potrf(ctx, A, {},             ws);   // <-- used to mean Uplo{} == Upper
//
// `{}` converts to an enum by an exact match but to a class type only by a
// user-defined conversion, so the positional overload won silently and the call
// factorised the opposite triangle -- no diagnostic, wrong numbers. (That is how
// it reached ortho's Cholesky path, where it surfaced only as LOBPCG failing to
// converge.)
//
// Giving the trap its own *enum* parameter type puts a third candidate at the
// same exact-match rank, so the bare-`{}` call is ill-formed (ambiguous) and
// names all three candidates. Both spellings above still resolve exactly as
// before: neither `PotrfOptions{}` nor `Uplo::Lower` converts to this type.
enum class EmptyBracesAreAmbiguous {};
}  // namespace detail

template <Backend B, detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
Event potrf(Queue&, const MA&, detail::EmptyBracesAreAmbiguous, Span<std::byte>) = delete;

template <detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
Event potrf(Queue&, const MA&, detail::EmptyBracesAreAmbiguous, Span<std::byte>) = delete;

// getrf, getri, geqrf and orgqr have no options to carry, so their only new
// spelling is the arena-backed one: drop the workspace and it is leased for you.
// They deliberately take no workspace parameter -- an overload with one would be
// indistinguishable from the positional call, and the two would be ambiguous.
// To manage the workspace yourself, use the positional spelling.
//
// getrf and getri do take one more thing: a trailing `info`, the per-item LAPACK
// status (one int32 per batch item, 0 = success). It is a plain trailing
// parameter rather than an option struct because these two have no struct to put
// it in, and a *Options type invented for one field would be a second way to
// spell the same call. Left empty -- the default -- nothing is reported and the
// behaviour is exactly what it was before issue #73. Passing it costs nothing:
// the backends already allocate that array for the vendor call, so a non-empty
// span replaces the scratch rather than adding to it, and the workspace size is
// unchanged either way.
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
    // No squareness check: rectangular A is the entire point of geqrf, and the
    // library's own panel factorisations (src/extensions/sytrd_sy2sb.cc,
    // band_reduction.cc) pass tall panels. What is fixed is the tau stride --
    // every backend indexes `tau.data() + i * min(m, n)`.
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
// These two were the last dense entry points without an option-struct spelling,
// and the reason given was that their positional forms do not end in the
// workspace -- ormqr's `workspace` is followed by a defaulted block-size hint --
// so the arena form could not simply drop the last argument.
//
// That reason does not apply: an option overload is a NEW overload and is under
// no obligation to mirror the positional parameter order. getrs has been doing
// exactly this since the option layer landed (its positional form takes
// `(A, B, transA, pivots, ws)` while its option form takes
// `(A, B, pivots, opts, ws)`), so nothing here moves an existing parameter and
// no existing call site changes. The hint that blocked the "just drop the last
// argument" reading becomes a field of OrmqrOptions instead.

struct OrmqrOptions {
    Side side = Side::Left;
    Transpose trans = Transpose::NoTrans;
    // 0 means "let the tuning table pick the WY panel width"; see the block-size
    // selection in src/extensions/ormqr_blocked.cc. It is an option field rather
    // than a trailing default argument because the arena spelling has no
    // trailing position left to put it in.
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
    // The sizing query must be given the same options the call gets: the hint
    // selects the panel width, and the panel width is what the workspace is
    // sized for, so sizing with the default hint and running with another would
    // under-size the buffer.
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
    // Both mirror validate_ormqr_dims (src/extensions/ormqr_blocked.cc): the
    // reflector count is min(A.rows(), A.cols()) per problem and every path
    // indexes tau at that stride, so a short tau is read out of bounds.
    detail::require_same_batch("ormqr", "A", A, "C", C);
    detail::require_span_at_least("ormqr", "tau", tau.size(),
                                  static_cast<size_t>(std::min(A.rows(), A.cols())) * A.batch_size());
    return with_backend(ctx, [&](auto Back) { return ormqr<Back.value>(ctx, A, C, tau, opts); });
}

struct GesvdOptions {
    SvdVectors jobu = SvdVectors::All;
    SvdVectors jobvh = SvdVectors::All;
    // Engaged selects the Hermitian entry point (blas/functions/gesvd.hh), which
    // is a different overload rather than a different argument value -- hence
    // std::optional and not a plain Uplo with a "not Hermitian" sentinel.
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
    // The Hermitian and general branches choose providers independently and can
    // therefore need different amounts of scratch, so the size query has to take
    // the same branch as the call below it.
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
    // Only A is shape-checked. U and Vh are deliberately left alone: a
    // default-constructed view is how this API spells "I asked for no vectors on
    // this side", so any check relating their extents to A's would reject the
    // documented SvdVectors::None call.
    if (opts.hermitian_uplo) detail::require_square("gesvd", "A", A);
    detail::require_span_at_least("gesvd", "singular_values", singular_values.size(),
                                  static_cast<size_t>(std::min(A.rows(), A.cols())) * A.batch_size());
    return with_backend(
        ctx, [&](auto Back) { return gesvd<Back.value>(ctx, A, singular_values, U, Vh, opts); });
}

#undef BATCHLAS_DENSE_VIEW

}  // namespace batchlas
