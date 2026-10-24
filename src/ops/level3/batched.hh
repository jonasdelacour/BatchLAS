// SPDX-License-Identifier: MIT
#pragma once

// Shared pieces of the level-3 native families: batch chunking (a launch holds at most
// Device::max_groups_yz items, and an expansion at most what its scratch budget admits), the
// mirrored expansion feeding the public gemm (symm, hemm), and the gemm-plus-fold of a rank-k
// product into one triangle (herk, her2k).
// evidence: docs/perf/level3.md#level-3-scratch-expansions-and-their-ceilings

#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/mempool.hh>

#include <src/expansion_budget.hh>
#include <src/kernels/common/triangular_expand.hh>
#include <src/math-helpers.hh>
#include <src/queue.hh>
#include <src/select/select.hh>
#include <src/util/kernel-trace.hh>

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace batchlas::ops::level3 {

template <typename T>
using MV = MatrixView<T, MatrixFormat::Dense>;

/// Items [first, first + count) of a homogeneous view: same extents, ld and stride.
template <typename T>
MV<T> batch_range(const MV<T>& v, int first, int count) {
    return MV<T>(v.data_ptr() + static_cast<std::ptrdiff_t>(first) * v.stride(), v.rows(), v.cols(), v.ld(),
                 v.stride(), count);
}

/// The grid-z item ceiling of one launch on this device, as an int.
template <Backend B>
int grid_chunk(const Queue& q) {
    const std::int64_t z = select::device_of<B>(q).max_groups_yz;
    return static_cast<int>(std::clamp<std::int64_t>(z, 1, std::numeric_limits<int>::max()));
}

/// f(first, count) per chunk of at most `chunk` items: batch / chunk launches, not batch. An
/// out-of-order queue orders nothing between chunks, so there each one is waited out.
template <typename F>
Event for_each_chunk(Queue& q, int batch, int chunk, F&& f) {
    if (batch <= chunk) return f(0, batch);
    Event last;
    for (int first = 0; first < batch; first += chunk) {
        last = f(first, std::min(chunk, batch - first));
        if (!q.in_order()) last.wait();
    }
    return last;
}

/// Fold a dense rank-k product P into the referenced triangle of a Hermitian C:
/// C = P + beta C, or for TwoSided C = P + P^H + beta C. alpha A B^H and conj(alpha) B A^H are
/// conjugate transposes of each other, so one gemm and a mirrored read give her2k both terms.
/// The other triangle is neither read nor written; the diagonal is real on exit, and its
/// imaginary part is not an input (beta is real and C = C^H).
template <typename T, bool TwoSided>
Event fold_hermitian(Queue& q, const MV<T>& C, const MV<T>& P, float_t<T> beta, Uplo uplo) {
    using real_t = float_t<T>;
    const int n = C.rows();
    const int batch = C.batch_size();
    const bool lower = uplo == Uplo::Lower;
    T* dst = C.data_ptr();
    const T* src = P.data_ptr();
    const int ldc = C.ld();
    const int ldp = P.ld();
    const std::size_t stride_c = static_cast<std::size_t>(C.stride());
    const std::size_t stride_p = static_cast<std::size_t>(P.stride());

    const auto shape = backend::detail::expand_group_shape(n);
    const sycl::range<3> global(static_cast<std::size_t>(batch),
                                static_cast<std::size_t>(internal::ceil_div(n, shape.cols) * shape.cols),
                                static_cast<std::size_t>(internal::ceil_div(n, shape.rows) * shape.rows));
    const sycl::range<3> local(1, static_cast<std::size_t>(shape.cols), static_cast<std::size_t>(shape.rows));

    q->parallel_for(sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> item) {
        const int i = static_cast<int>(item.get_global_id(2));
        const int j = static_cast<int>(item.get_global_id(1));
        if (i >= n || j >= n || (lower ? (i < j) : (i > j))) return;
        const int b = static_cast<int>(item.get_group(0));
        const std::size_t p_base = static_cast<std::size_t>(b) * stride_p;
        T value = src[p_base + static_cast<std::size_t>(j) * ldp + i];
        if constexpr (TwoSided) {
            const T mirrored = src[p_base + static_cast<std::size_t>(i) * ldp + j];
            value = T(value.real() + mirrored.real(), value.imag() - mirrored.imag());
        }
        T* c = dst + static_cast<std::size_t>(b) * stride_c + static_cast<std::size_t>(j) * ldc + i;
        if (beta != real_t(0)) {
            const T prev = *c;
            value = T(value.real() + beta * prev.real(), i == j ? value.imag() : value.imag() + beta * prev.imag());
        }
        *c = i == j ? T(value.real(), real_t(0)) : value;
    });
    return q.get_event();
}

/// The dense symmetric (Conjugate = false) or Hermitian (true) copy of A's `uplo` triangle in a
/// workspace lease, times B through the public gemm: C = alpha A B + beta C (Left) or
/// alpha B A + beta C (Right). Chunked so each chunk's scratch fits; @pre expansion_chunk >= 1.
// The scratch is a lease on the queue's arena, not a Matrix: a managed allocation migrates on first
// touch (an order of magnitude over the gemm at n=512 batch=512) and would be freed while enqueued.
template <Backend B, typename T, bool Conjugate>
Event expand_gemm(Queue& q, const MV<T>& A, const MV<T>& Bm, const MV<T>& C, T alpha, T beta, Side side,
                  Uplo uplo) {
    const int k = A.rows(), batch = A.batch_size();
    const int ld = backend::detail::expanded_ld<T>(k);
    const int chunk = backend::detail::expansion_chunk<T>(q, k, batch, grid_chunk<B>(q));
    auto ws = q.workspace(backend::detail::expanded_workspace_bytes<T>(q, k, chunk));
    BumpAllocator pool(ws.span());
    auto storage = pool.allocate<T>(q, std::size_t(ld) * std::size_t(k) * std::size_t(chunk));
    return for_each_chunk(q, batch, chunk, [&](int first, int count) {
        const MV<T> full(storage.data(), k, k, ld, ld * k, count);
        Event expansion;
        {
            BATCHLAS_KERNEL_TRACE_SCOPE(Conjugate ? "hemm.expand" : "symm.expand");
            expansion = backend::detail::expand_mirrored<T, Conjugate>(q, full, batch_range(A, first, count), uplo);
        }
        // An out-of-order queue orders nothing between the expansion and a vendor gemm's stream.
        if (!q.in_order()) expansion.wait();
        const MV<T> Bc = batch_range(Bm, first, count), Cc = batch_range(C, first, count);
        if (side == Side::Left)
            return gemm<B, T>(q, full, Bc, Cc, alpha, beta, Transpose::NoTrans, Transpose::NoTrans,
                              ComputePrecision::Default);
        return gemm<B, T>(q, Bc, full, Cc, alpha, beta, Transpose::NoTrans, Transpose::NoTrans,
                          ComputePrecision::Default);
    });
}

/// herk (TwoSided = false, Bop = A) and her2k (true): one gemm of op(A) op(Bop)^H over the whole
/// n x n product into scratch, then fold_hermitian. The gemm cannot target C: it would write both
/// triangles. Chunked as expand_gemm; @pre expansion_chunk >= 1.
template <Backend B, typename T, bool TwoSided>
Event gemm_fold(Queue& q, const MV<T>& A, const MV<T>& Bop, const MV<T>& C, T alpha, float_t<T> beta, Uplo uplo,
                Transpose transA) {
    const int n = C.rows(), batch = C.batch_size();
    const int ld = backend::detail::expanded_ld<T>(n);
    const int chunk = backend::detail::expansion_chunk<T>(q, n, batch, grid_chunk<B>(q));
    auto ws = q.workspace(backend::detail::expanded_workspace_bytes<T>(q, n, chunk));
    BumpAllocator pool(ws.span());
    auto storage = pool.allocate<T>(q, std::size_t(ld) * std::size_t(n) * std::size_t(chunk));
    const Transpose transB = transA == Transpose::NoTrans ? Transpose::ConjTrans : Transpose::NoTrans;
    return for_each_chunk(q, batch, chunk, [&](int first, int count) {
        const MV<T> product(storage.data(), n, n, ld, ld * n, count);
        Event g = gemm<B, T>(q, batch_range(A, first, count), batch_range(Bop, first, count), product, alpha, T(0),
                             transA, transB, ComputePrecision::Default);
        if (!q.in_order()) g.wait();  // the vendor gemm runs on the native stream
        BATCHLAS_KERNEL_TRACE_SCOPE(TwoSided ? "her2k.fold" : "herk.fold");
        return fold_hermitian<T, TwoSided>(q, batch_range(C, first, count), product, beta, uplo);
    });
}

}  // namespace batchlas::ops::level3
