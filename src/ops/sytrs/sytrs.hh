#pragma once

// sytrs (flat-kernel-selection.md §4.3): select::run takes the first entry of the nearest
// tuned/sytrs.<device>.txt row that can_run() admits. Shared by the public sytrs (sytrs.cc) and
// sysv's compose, which passes its sytrf status as the skip list. evidence: docs/perf/sytrf.md

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/sytrf.hh>
#include <batchlas/blas/functions/trsm.hh>

#include <src/ops/sytrf/sytrf_native.hh>
#include <src/ops/sytrs/choice.hh>
#include <src/select/select.hh>
#include <src/util/batch_mask.hh>
#include <src/select/vendor.hh>

#include <cstddef>
#include <cstdint>
#include <variant>

namespace batchlas::ops::sytrs {

using select::overloaded;

template <class T>
using MV = MatrixView<T, MatrixFormat::Dense>;

// LAPACKE only: cuSOLVER's Xsytrs wants int64 pivots and is per item (docs/perf/sytrf.md).
template <Backend B>
inline constexpr bool vendor_linked = B == Backend::NETLIB && select::kHasNetlib;

template <class T>
select::Key key_of(const MV<T>& A, const MV<T>& Bm) {
    return {{"n", A.rows()}, {"nrhs", Bm.cols()}, {"batch", A.batch_size()}};
}

// Correctness only (R3): each clause is the argument check at the top of that driver. cta, wg and
// LAPACKE's host loop honour a batch mask; blocked runs the public trsm on the masked part views.
template <Backend B, class T>
bool can_run(const SytrsChoice& c, const select::Device& d, const MV<T>& A, const MV<T>& Bm) {
    const std::int64_t n = A.rows(), nrhs = Bm.cols();
    const bool homogeneous = !A.is_heterogeneous() && !Bm.is_heterogeneous();
    const bool native = B != Backend::NETLIB && d.is_gpu && d.has_sg32 && homogeneous && n >= 1 && nrhs >= 1 &&
                        A.batch_size() >= 1 && d.max_wg >= 32;
    return std::visit(overloaded{
        [&](Cta) {
            return native && n <= sycl_sytrf::kSytrsCtaMaxN &&
                   static_cast<std::int64_t>(sycl_sytrf::sytrs_cta_slm_bytes<T>(static_cast<int>(n))) <= d.slm_budget;
        },
        // Its trsm selects for itself; the public trsm has a last resort.
        [&](Blocked) { return native && !detail::any_batch_mask(A, Bm); },  // mask: lift after trsm
        [&](Wg) { return native; },
        [&](Vendor) { return vendor_linked<B> && d.has_vendor && homogeneous; },
    }, c);
}

template <Backend B, class T>
Event launch(Queue& q, const SytrsChoice& c, const MV<T>& A, const MV<T>& Bm, Uplo uplo, Span<int64_t> pivots,
             Span<std::byte> ws, const int32_t* skip) {
    return std::visit(overloaded{
        [&](Cta) { return sycl_sytrf::sytrs_cta_dispatch<T>(q, A, Bm, uplo, pivots, skip); },
        [&](Blocked) {
            return sycl_sytrf::sytrs_blocked_dispatch<T>(
                q, A, Bm, uplo, pivots, ws, skip,
                [](Queue& c2, const MV<T>& ta, const MV<T>& tb, T talpha, Side tside, Uplo tuplo, Transpose ttrans,
                   Diag tdiag) { return trsm<B, T>(c2, ta, tb, talpha, tside, tuplo, ttrans, tdiag); });
        },
        [&](Wg) { return sycl_sytrf::sytrs_native_dispatch<T>(q, A, Bm, uplo, pivots, skip); },
        [&](Vendor) -> Event {
            if constexpr (vendor_linked<B>) return backend::sytrs_vendor<B, T>(q, A, Bm, uplo, pivots, skip);
            else select::no_vendor<B, T>(spec);
        },
    }, c);
}

// Exactly the chosen family's need (R5); nothing here dereferences A or B.
template <Backend B, class T>
std::size_t workspace(Queue& q, const SytrsChoice& c, const MV<T>& A, const MV<T>& Bm) {
    return std::visit(overloaded{
        [&](Blocked) { return sycl_sytrf::sytrs_blocked_buffer_size<T>(q, A, Bm); },
        [&](const auto&) { return std::size_t(0); },
    }, c);
}

// The routed solve; `skip` (one entry per item, or null): a nonzero entry keeps that item's B.
template <Backend B, class T>
Event run(Queue& ctx, const MV<T>& A, const MV<T>& Bm, Uplo uplo, Span<int64_t> pivots, Span<std::byte> ws,
          const int32_t* skip) {
    const coverage::Shape shape{.m = A.rows(), .n = Bm.cols(), .k = A.rows(), .batch = A.batch_size(), .uplo = uplo};
    const select::Key key = key_of<T>(A, Bm);
    return select::run<B, T>(
        spec, ctx, key, candidates<T>(), [&](const auto& c, const auto& d) { return can_run<B, T>(c, d, A, Bm); },
        shape, key, [&](const auto& c) { return launch<B, T>(ctx, c, A, Bm, uplo, pivots, ws, skip); });
}

template <Backend B, class T>
std::size_t buffer_size(Queue& ctx, const MV<T>& A, const MV<T>& Bm) {
    const auto c = select::pick<B, T>(spec, ctx, key_of<T>(A, Bm), candidates<T>(),
                                      [&](const auto& k, const auto& d) { return can_run<B, T>(k, d, A, Bm); });
    return workspace<B, T>(ctx, c, A, Bm);
}

}  // namespace batchlas::ops::sytrs
