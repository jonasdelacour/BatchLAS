#pragma once

#include <cstddef>

#include <blas/enums.hh>
#include <blas/matrix.hh>
#include <blas/queue-dispatch.hh>
#include <util/sycl-device-queue.hh>
#include <util/sycl-span.hh>

// Option-struct spellings of the public entry points.
//
//     gemm(ctx, A, B, C, {.alpha = 2.0f, .transA = Transpose::Trans});
//     syev(ctx, A, W, {.jobz = JobType::EigenVectors});
//
// rather than
//
//     gemm<Backend::CUDA, float>(ctx, A, B, C, 2.0f, 0.0f, Transpose::Trans,
//                                Transpose::NoTrans, ComputePrecision::Default);
//     { UnifiedVector<std::byte> ws(syev_buffer_size<Backend::CUDA>(ctx, A, W,
//           JobType::EigenVectors, Uplo::Lower));
//       syev<Backend::CUDA>(ctx, A, W, JobType::EigenVectors, Uplo::Lower, ws); }
//
// Three things are going on, each from an earlier phase:
//   - the Backend comes from the Queue (P3),
//   - the options carry their own defaults, so only what differs is written,
//   - the workspace defaults to a lease from the queue's arena (P2), which is
//     what lets the two-statement sizing dance collapse into the call.
//
// T is deduced from the matrix arguments, never from the option struct. That is
// what makes `{.alpha = 2.0f}` work at the call: by the time the compiler looks
// at the option parameter its type is already fixed, so a braced initialiser has
// something concrete to initialise. An option struct in a deduced position would
// not compile.
namespace batchlas {

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
// Second, `Matrix` and `MatrixView` are both accepted, and may be mixed. The
// positional entry points have always had a Matrix wrapper alongside the
// MatrixView primary; without this the option spelling would have been the one
// place in the library that demanded an explicit `.view()`, which is exactly the
// kind of papercut that stops a new API from being adopted. Everything is
// converted to MatrixView before the positional call, so mixing is fine.

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
// and one that does not and leases from the queue's arena instead. It is
// tempting to write that as a single function with `Span<std::byte> ws = {}`
// and an `if (ws.data() != nullptr)` inside. That is wrong, and it silently
// corrupted results before it was caught.
//
// A null span is not a synonym for "I did not pass one". Library code that
// sub-allocates from a BumpAllocator runs the whole algorithm twice: once in
// sizing mode, where every pool allocation legitimately hands back an empty
// span, and once for real. The input matrices are real in *both* passes -- only
// the workspace-derived views are fabricated -- so treating the sizing pass's
// empty span as "no workspace, allocate one and proceed" makes the measurement
// pass actually execute the factorisation over the caller's live data.
//
// That failure is invisible where it happens: nothing crashes, no size is
// wrong, and the corrupted matrix only shows up much later as an algorithm that
// no longer converges. Splitting the two cases into two overloads removes the
// sentinel entirely -- an argument that is present is used exactly as given.

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
    return with_backend(ctx, [&](auto Back) { return symm<Back.value>(ctx, A, B, C, opts); });
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
    return with_backend(ctx, [&](auto Back) { return trmm<Back.value>(ctx, A, B, C, opts); });
}

template <Backend Back, detail::DenseMatrixLike MA, detail::DenseMatrixLike MB,
          typename T = detail::dense_scalar_t<MA>>
inline Event trsm(Queue& ctx, const MA& A, const MB& B, const TrsmOptions<T>& opts) {
    using V = BATCHLAS_DENSE_VIEW(T);
    return trsm<Back, T>(ctx, V(A), V(B), opts.side, opts.uplo, opts.trans, opts.diag, opts.alpha);
}

template <detail::DenseMatrixLike MA, detail::DenseMatrixLike MB,
          typename T = detail::dense_scalar_t<MA>>
inline Event trsm(Queue& ctx, const MA& A, const MB& B, const TrsmOptions<T>& opts) {
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
// See util/workspace.hh.

struct PotrfOptions {
    Uplo uplo = Uplo::Lower;
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
    return potrf<B, T>(ctx, V(A), opts.uplo, ws);
}

template <Backend B, detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
inline Event potrf(Queue& ctx, const MA& A, const PotrfOptions& opts) {
    using V = BATCHLAS_DENSE_VIEW(T);
    auto lease = ctx.workspace(potrf_buffer_size<B, T>(ctx, V(A), opts.uplo));
    return potrf<B, T>(ctx, V(A), opts.uplo, lease.span());
}

template <detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
inline Event potrf(Queue& ctx, const MA& A, const PotrfOptions& opts, Span<std::byte> ws) {
    return with_backend(ctx, [&](auto Back) { return potrf<Back.value>(ctx, A, opts, ws); });
}

template <detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
inline Event potrf(Queue& ctx, const MA& A, const PotrfOptions& opts = {}) {
    return with_backend(ctx, [&](auto Back) { return potrf<Back.value>(ctx, A, opts); });
}

// getrf, getri, geqrf and orgqr have no options to carry, so their only new
// spelling is the arena-backed one: drop the workspace and it is leased for you.
// They deliberately take no workspace parameter -- an overload with one would be
// indistinguishable from the positional call, and the two would be ambiguous.
// To manage the workspace yourself, use the positional spelling.
template <Backend B, detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
inline Event getrf(Queue& ctx, const MA& A, Span<int64_t> pivots) {
    using V = BATCHLAS_DENSE_VIEW(T);
    auto lease = ctx.workspace(getrf_buffer_size<B, T>(ctx, V(A)));
    return getrf<B, T>(ctx, V(A), pivots, lease.span());
}

template <detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
inline Event getrf(Queue& ctx, const MA& A, Span<int64_t> pivots) {
    return with_backend(ctx, [&](auto Back) { return getrf<Back.value>(ctx, A, pivots); });
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
    auto lease = ctx.workspace(getrs_buffer_size<B, T>(ctx, V(A), V(B_), opts.trans));
    return getrs<B, T>(ctx, V(A), V(B_), opts.trans, pivots, lease.span());
}

template <detail::DenseMatrixLike MA, detail::DenseMatrixLike MB,
          typename T = detail::dense_scalar_t<MA>>
inline Event getrs(Queue& ctx, const MA& A, const MB& B_, Span<int64_t> pivots,
                   const GetrsOptions& opts, Span<std::byte> ws) {
    return with_backend(
        ctx, [&](auto Back) { return getrs<Back.value>(ctx, A, B_, pivots, opts, ws); });
}

template <detail::DenseMatrixLike MA, detail::DenseMatrixLike MB,
          typename T = detail::dense_scalar_t<MA>>
inline Event getrs(Queue& ctx, const MA& A, const MB& B_, Span<int64_t> pivots,
                   const GetrsOptions& opts = {}) {
    return with_backend(ctx,
                        [&](auto Back) { return getrs<Back.value>(ctx, A, B_, pivots, opts); });
}

template <Backend B, detail::DenseMatrixLike MA, detail::DenseMatrixLike MB,
          typename T = detail::dense_scalar_t<MA>>
inline Event getri(Queue& ctx, const MA& A, const MB& Ainv, Span<int64_t> pivots) {
    using V = BATCHLAS_DENSE_VIEW(T);
    auto lease = ctx.workspace(getri_buffer_size<B, T>(ctx, V(A)));
    return getri<B, T>(ctx, V(A), V(Ainv), pivots, lease.span());
}

template <detail::DenseMatrixLike MA, detail::DenseMatrixLike MB,
          typename T = detail::dense_scalar_t<MA>>
inline Event getri(Queue& ctx, const MA& A, const MB& Ainv, Span<int64_t> pivots) {
    return with_backend(ctx, [&](auto Back) { return getri<Back.value>(ctx, A, Ainv, pivots); });
}

template <Backend B, detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
inline Event geqrf(Queue& ctx, const MA& A, Span<T> tau) {
    using V = BATCHLAS_DENSE_VIEW(T);
    auto lease = ctx.workspace(geqrf_buffer_size<B, T>(ctx, V(A), tau));
    return geqrf<B, T>(ctx, V(A), tau, lease.span());
}

template <detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
inline Event geqrf(Queue& ctx, const MA& A, Span<T> tau) {
    return with_backend(ctx, [&](auto Back) { return geqrf<Back.value>(ctx, A, tau); });
}

template <Backend B, detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
inline Event orgqr(Queue& ctx, const MA& A, Span<T> tau) {
    using V = BATCHLAS_DENSE_VIEW(T);
    auto lease = ctx.workspace(orgqr_buffer_size<B, T>(ctx, V(A), tau));
    return orgqr<B, T>(ctx, V(A), tau, lease.span());
}

template <detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
inline Event orgqr(Queue& ctx, const MA& A, Span<T> tau) {
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
    return with_backend(ctx, [&](auto Back) { return syev<Back.value>(ctx, A, W, opts, ws); });
}

template <detail::DenseMatrixLike MA, typename T = detail::dense_scalar_t<MA>>
inline Event syev(Queue& ctx, const MA& A, Span<typename base_type<T>::type> W,
                  const SyevOptions& opts = {}) {
    return with_backend(ctx, [&](auto Back) { return syev<Back.value>(ctx, A, W, opts); });
}

#undef BATCHLAS_DENSE_VIEW

}  // namespace batchlas
