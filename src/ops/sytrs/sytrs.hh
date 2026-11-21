// SPDX-License-Identifier: MIT
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

template <class Scalar>
using MV = MatrixView<Scalar, MatrixFormat::Dense>;

// LAPACKE only: cuSOLVER's Xsytrs wants int64 pivots and is per item (docs/perf/sytrf.md).
template <Backend B>
inline constexpr bool vendor_linked = B == Backend::NETLIB && select::kHasNetlib;

template <class Scalar>
select::Key key_of(const MV<Scalar>& A, const MV<Scalar>& Bm) {
    return {{"n", A.rows()}, {"nrhs", Bm.cols()}, {"batch", A.batch_size()}};
}

// Correctness only (R3): each clause is the argument check at the top of that driver. cta, wg and
// LAPACKE's host loop honour a batch mask; blocked runs the public trsm on the masked part views.
template <Backend B, class Scalar>
bool can_run(const SytrsChoice& c, const select::Device& d, const MV<Scalar>& A, const MV<Scalar>& Bm) {
    const std::int64_t n = A.rows(), nrhs = Bm.cols();
    const bool homogeneous = !A.is_heterogeneous() && !Bm.is_heterogeneous();
    const bool native = (B != Backend::NETLIB || select::kHostNative) && select::natives_run_on(d) && d.has_sg32 &&
                        homogeneous && n >= 1 && nrhs >= 1 && A.batch_size() >= 1 && d.max_wg >= 32;
    return std::visit(overloaded{
        [&](Cta) {
            return native && n <= ops::sytrf::kSytrsCtaMaxN &&
                   static_cast<std::int64_t>(ops::sytrf::sytrs_cta_slm_bytes<Scalar>(static_cast<int>(n))) <= d.slm_budget;
        },
        // Its trsm selects for itself; the public trsm has a last resort.
        [&](Blocked) { return native; },
        [&](Wg) { return native; },
        [&](Vendor) { return vendor_linked<B> && d.has_vendor && homogeneous; },
    }, c);
}

template <Backend B, class Scalar>
Event launch(Queue& q, const SytrsChoice& c, const MV<Scalar>& A, const MV<Scalar>& Bm, Uplo uplo, Span<int64_t> pivots,
             Span<std::byte> ws, const int32_t* skip) {
    return std::visit(overloaded{
        [&](Cta) { return ops::sytrf::sytrs_cta_dispatch<Scalar>(q, A, Bm, uplo, pivots, skip); },
        [&](Blocked) {
            return ops::sytrf::sytrs_blocked_dispatch<Scalar>(
                q, A, Bm, uplo, pivots, ws, skip,
                [](Queue& c2, const MV<Scalar>& ta, const MV<Scalar>& tb, Scalar talpha, Side tside, Uplo tuplo,
                   Transpose ttrans,
                   Diag tdiag) { return trsm<B, Scalar>(c2, ta, tb, talpha, tside, tuplo, ttrans, tdiag); });
        },
        [&](Wg) { return ops::sytrf::sytrs_native_dispatch<Scalar>(q, A, Bm, uplo, pivots, skip); },
        [&](Vendor) -> Event {
            if constexpr (vendor_linked<B>) return backend::sytrs_vendor<B, Scalar>(q, A, Bm, uplo, pivots, skip);
            else select::no_vendor<B, Scalar>(spec);
        },
    }, c);
}

// Exactly the chosen family's need (R5); nothing here dereferences A or B.
template <Backend B, class Scalar>
std::size_t workspace(Queue& q, const SytrsChoice& c, const MV<Scalar>& A, const MV<Scalar>& Bm) {
    return std::visit(overloaded{
        [&](Blocked) { return ops::sytrf::sytrs_blocked_buffer_size<Scalar>(q, A, Bm); },
        [&](const auto&) { return std::size_t(0); },
    }, c);
}

// The routed solve; `skip` (one entry per item, or null): a nonzero entry keeps that item's B.
template <Backend B, class Scalar>
Event run(Queue& ctx, const MV<Scalar>& A, const MV<Scalar>& Bm, Uplo uplo, Span<int64_t> pivots, Span<std::byte> ws,
          const int32_t* skip) {
    const coverage::Shape shape{.m = A.rows(), .n = Bm.cols(), .k = A.rows(), .batch = A.batch_size(), .uplo = uplo};
    const select::Key key = key_of<Scalar>(A, Bm);
    return select::run<B, Scalar>(
        spec, ctx, key, candidates<Scalar>(), [&](const auto& c,
                                                  const auto& d) { return can_run<B, Scalar>(c, d, A, Bm); },
        shape, key, [&](const auto& c) { return launch<B, Scalar>(ctx, c, A, Bm, uplo, pivots, ws, skip); });
}

template <Backend B, class Scalar>
std::size_t buffer_size(Queue& ctx, const MV<Scalar>& A, const MV<Scalar>& Bm) {
    const auto c = select::pick<B, Scalar>(spec, ctx, key_of<Scalar>(A, Bm), candidates<Scalar>(),
                                           [&](const auto& k,
                                               const auto& d) { return can_run<B, Scalar>(k, d, A, Bm); });
    return workspace<B, Scalar>(ctx, c, A, Bm);
}

}  // namespace batchlas::ops::sytrs
