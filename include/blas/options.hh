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

template <typename T>
inline Event gemm(Queue& ctx,
                  const MatrixView<T, MatrixFormat::Dense>& A,
                  const MatrixView<T, MatrixFormat::Dense>& B,
                  const MatrixView<T, MatrixFormat::Dense>& C,
                  const GemmOptions<T>& opts) {
    return with_backend(ctx, [&](auto Back) {
        return gemm<Back.value, T>(ctx, A, B, C, opts.alpha, opts.beta, opts.transA, opts.transB,
                                   opts.precision);
    });
}

template <typename T>
inline Event gemv(Queue& ctx,
                  const MatrixView<T, MatrixFormat::Dense>& A,
                  const VectorView<T>& x,
                  const VectorView<T>& y,
                  const GemvOptions<T>& opts) {
    return with_backend(ctx, [&](auto Back) {
        return gemv<Back.value, T>(ctx, A, x, y, opts.alpha, opts.beta, opts.transA);
    });
}

template <typename T>
inline Event symm(Queue& ctx,
                  const MatrixView<T, MatrixFormat::Dense>& A,
                  const MatrixView<T, MatrixFormat::Dense>& B,
                  const MatrixView<T, MatrixFormat::Dense>& C,
                  const SymmOptions<T>& opts) {
    return with_backend(ctx, [&](auto Back) {
        return symm<Back.value, T>(ctx, A, B, C, opts.alpha, opts.beta, opts.side, opts.uplo);
    });
}

template <typename T>
inline Event syrk(Queue& ctx,
                  const MatrixView<T, MatrixFormat::Dense>& A,
                  const MatrixView<T, MatrixFormat::Dense>& C,
                  const SyrkOptions<T>& opts) {
    return with_backend(ctx, [&](auto Back) {
        return syrk<Back.value, T>(ctx, A, C, opts.alpha, opts.beta, opts.uplo, opts.trans);
    });
}

template <typename T>
inline Event syr2k(Queue& ctx,
                   const MatrixView<T, MatrixFormat::Dense>& A,
                   const MatrixView<T, MatrixFormat::Dense>& B,
                   const MatrixView<T, MatrixFormat::Dense>& C,
                   const Syr2kOptions<T>& opts) {
    return with_backend(ctx, [&](auto Back) {
        return syr2k<Back.value, T>(ctx, A, B, C, opts.alpha, opts.beta, opts.uplo, opts.trans);
    });
}

template <typename T>
inline Event trmm(Queue& ctx,
                  const MatrixView<T, MatrixFormat::Dense>& A,
                  const MatrixView<T, MatrixFormat::Dense>& B,
                  const MatrixView<T, MatrixFormat::Dense>& C,
                  const TrmmOptions<T>& opts) {
    return with_backend(ctx, [&](auto Back) {
        return trmm<Back.value, T>(ctx, A, B, C, opts.alpha, opts.side, opts.uplo, opts.trans,
                                   opts.diag);
    });
}

template <typename T>
inline Event trsm(Queue& ctx,
                  const MatrixView<T, MatrixFormat::Dense>& A,
                  const MatrixView<T, MatrixFormat::Dense>& B,
                  const TrsmOptions<T>& opts) {
    return with_backend(ctx, [&](auto Back) {
        return trsm<Back.value, T>(ctx, A, B, opts.side, opts.uplo, opts.trans, opts.diag,
                                   opts.alpha);
    });
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
// work is ordered behind this call's; on an out-of-order queue it is not, so
// there the caller must either wait or pass its own span. See util/workspace.hh.

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

namespace detail {
// Run `call` with a workspace: the caller's if it gave one, otherwise a lease
// sized by `sizer`.
template <typename Sizer, typename Call>
inline Event with_workspace(Queue& ctx, Span<std::byte> ws, Sizer&& sizer, Call&& call) {
    if (ws.data() != nullptr) return call(ws);
    auto lease = ctx.workspace(sizer());
    return call(lease.span());
}
}  // namespace detail

template <typename T>
inline Event potrf(Queue& ctx,
                   const MatrixView<T, MatrixFormat::Dense>& A,
                   const PotrfOptions& opts = {},
                   Span<std::byte> ws = {}) {
    return with_backend(ctx, [&](auto Back) {
        constexpr Backend B = Back.value;
        return detail::with_workspace(
            ctx, ws,
            [&] { return potrf_buffer_size<B, T>(ctx, A, opts.uplo); },
            [&](Span<std::byte> w) { return potrf<B, T>(ctx, A, opts.uplo, w); });
    });
}

// getrf, getri, geqrf and orgqr have no options to carry, so their only new
// spelling is the arena-backed one: drop the workspace and it is leased for you.
// They deliberately take no workspace parameter -- an overload with one would be
// indistinguishable from the positional call, and the two would be ambiguous.
// To manage the workspace yourself, use the positional spelling.
template <typename T>
inline Event getrf(Queue& ctx,
                   const MatrixView<T, MatrixFormat::Dense>& A,
                   Span<int64_t> pivots) {
    return with_backend(ctx, [&](auto Back) {
        constexpr Backend B = Back.value;
        auto lease = ctx.workspace(getrf_buffer_size<B, T>(ctx, A));
        return getrf<B, T>(ctx, A, pivots, lease.span());
    });
}

template <typename T>
inline Event getrs(Queue& ctx,
                   const MatrixView<T, MatrixFormat::Dense>& A,
                   const MatrixView<T, MatrixFormat::Dense>& B_,
                   Span<int64_t> pivots,
                   const GetrsOptions& opts = {},
                   Span<std::byte> ws = {}) {
    return with_backend(ctx, [&](auto Back) {
        constexpr Backend B = Back.value;
        return detail::with_workspace(
            ctx, ws,
            [&] { return getrs_buffer_size<B, T>(ctx, A, B_, opts.trans); },
            [&](Span<std::byte> w) { return getrs<B, T>(ctx, A, B_, opts.trans, pivots, w); });
    });
}

template <typename T>
inline Event getri(Queue& ctx,
                   const MatrixView<T, MatrixFormat::Dense>& A,
                   const MatrixView<T, MatrixFormat::Dense>& Ainv,
                   Span<int64_t> pivots) {
    return with_backend(ctx, [&](auto Back) {
        constexpr Backend B = Back.value;
        auto lease = ctx.workspace(getri_buffer_size<B, T>(ctx, A));
        return getri<B, T>(ctx, A, Ainv, pivots, lease.span());
    });
}

template <typename T>
inline Event geqrf(Queue& ctx,
                   const MatrixView<T, MatrixFormat::Dense>& A,
                   Span<T> tau) {
    return with_backend(ctx, [&](auto Back) {
        constexpr Backend B = Back.value;
        auto lease = ctx.workspace(geqrf_buffer_size<B, T>(ctx, A, tau));
        return geqrf<B, T>(ctx, A, tau, lease.span());
    });
}

template <typename T>
inline Event orgqr(Queue& ctx,
                   const MatrixView<T, MatrixFormat::Dense>& A,
                   Span<T> tau) {
    return with_backend(ctx, [&](auto Back) {
        constexpr Backend B = Back.value;
        auto lease = ctx.workspace(orgqr_buffer_size<B, T>(ctx, A, tau));
        return orgqr<B, T>(ctx, A, tau, lease.span());
    });
}

template <typename T>
inline Event syev(Queue& ctx,
                  const MatrixView<T, MatrixFormat::Dense>& A,
                  Span<typename base_type<T>::type> W,
                  const SyevOptions& opts = {},
                  Span<std::byte> ws = {}) {
    return with_backend(ctx, [&](auto Back) {
        constexpr Backend B = Back.value;
        return detail::with_workspace(
            ctx, ws,
            [&] { return syev_buffer_size<B, T>(ctx, A, W, opts.jobz, opts.uplo); },
            [&](Span<std::byte> w) { return syev<B, T>(ctx, A, W, opts.jobz, opts.uplo, w); });
    });
}

}  // namespace batchlas
