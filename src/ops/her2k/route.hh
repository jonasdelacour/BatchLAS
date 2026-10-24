// SPDX-License-Identifier: MIT
#pragma once

// her2k's key and can_run, in a header so sytrd_blocked.cc asks the very selection her2k makes:
// a copy of the predicate drifts silently. evidence: docs/perf/sytrd.md#sytrd-her2k-scratch-fit-headroom

#include <batchlas/blas/matrix.hh>

#include <src/expansion_budget.hh>
#include <src/ops/her2k/choice.hh>
#include <src/select/select.hh>

#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <variant>

namespace batchlas::ops::her2k {

template <class T>
using MV = MatrixView<T, MatrixFormat::Dense>;

template <class T>
std::int64_t inner(const MV<T>& A, Transpose transA) {
    return transA == Transpose::NoTrans ? A.cols() : A.rows();
}

template <class T>
select::Key key_of(const MV<T>& A, const MV<T>& C, Transpose transA) {
    return {{"n", C.rows()}, {"k", inner<T>(A, transA)}, {"batch", C.batch_size()}};
}

// Correctness only (R3). Every family runs each item at the top-level (n, k), so none takes a
// heterogeneous operand. Fold is chunked over the batch (grid z, scratch budget), so it refuses
// only an item whose own n x n product does not fit; it runs over the public gemm, so ROCm too.
template <Backend B, class T>
bool can_run(const Her2kChoice& c, const select::Device& d, Queue& q, const MV<T>& A, const MV<T>& Bm,
             const MV<T>& C) {
    const bool homogeneous = !A.is_heterogeneous() && !Bm.is_heterogeneous() && !C.is_heterogeneous();
    return std::visit(select::overloaded{
        [&](Triangular) {
            const std::int64_t n = C.rows(), side = (n + 127) / 128;  // 128-wide tiles, T(T+1)/2 of them
            return select::native_gpu_backend<B> && std::is_same_v<T, std::complex<float>> && d.is_gpu &&
                   homogeneous && n >= 1 && C.batch_size() >= 1 &&
                   d.max_wg >= kTriangularWg && d.slm_budget >= triangular_slm_bytes<T>() &&
                   side * (side + 1) / 2 <= d.max_groups_yz;
        },
        [&](Fold) {
            return (select::native_gpu_backend<B> || B == Backend::ROCM) && d.is_gpu && homogeneous &&
                   d.max_wg >= kFoldWg && C.rows() >= 1 && C.batch_size() >= 1 &&
                   backend::detail::expansion_chunk<T>(q, C.rows(), 1, d.max_groups_yz) >= 1;
        },
        [&](Vendor) { return B != Backend::ROCM && d.has_vendor && homogeneous; },
    }, c);
}

/// Would her2k serve this call with a batched native route (not the per-item vendor loop)?
/// False when nothing can run it. sytrd's rank-2k trailing update asks before issuing her2k.
template <Backend B, class T>
bool picks_batched_route(Queue& q, const MV<T>& A, const MV<T>& Bm, const MV<T>& C, Transpose transA) {
    try {
        const Her2kChoice c = select::pick<B, T>(spec, q, key_of<T>(A, C, transA), candidates<T>(),
            [&](const Her2kChoice& k, const select::Device& d) { return can_run<B, T>(k, d, q, A, Bm, C); });
        return select::family_of(c) != "vendor";
    } catch (const std::runtime_error&) {
        return false;
    }
}

}  // namespace batchlas::ops::her2k
