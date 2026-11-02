// SPDX-License-Identifier: MIT
#pragma once
// Device LA v2, the batched 2-D executor and its host launcher: one item per matrix element,
// 2-D indices (no % or /), the batch dimension grid-strided past the dimension-0 group limit.
// for_each_owned_2d (level1.hh) on a Grid2D visits the item's one element.

#include <cstddef>
#include <type_traits>

#include <sycl/sycl.hpp>

#include <batchlas/blas/device/v2/core.hh>

namespace batchlas::device::v2 {

// Local range (LB, GC, GR), dimension 2 fastest, so consecutive items walk a column and both
// sides of a copy coalesce. Item (row(), col()) of batch items batch() + k * global_range(0).
// A template value of 0 is read from the launch.
template <int GR = 0, int GC = 0, int LB = 0> struct Grid2D {
    static constexpr int static_group_rows = GR, static_group_cols = GC, static_group_batch = LB;
    sycl::nd_item<3> it;

    BATCHLAS_DLA_INLINE int group_rows() const {
        if constexpr (GR > 0) return GR;
        else return static_cast<int>(it.get_local_range(2));
    }
    BATCHLAS_DLA_INLINE int group_cols() const {
        if constexpr (GC > 0) return GC;
        else return static_cast<int>(it.get_local_range(1));
    }
    BATCHLAS_DLA_INLINE int lrow() const { return static_cast<int>(it.get_local_id(2)); }
    BATCHLAS_DLA_INLINE int lcol() const { return static_cast<int>(it.get_local_id(1)); }
    BATCHLAS_DLA_INLINE int row() const { return static_cast<int>(it.get_global_id(2)); }
    BATCHLAS_DLA_INLINE int col() const { return static_cast<int>(it.get_global_id(1)); }
    // The first batch item of this item (the only one when the launch covers the batch).
    BATCHLAS_DLA_INLINE int batch() const {
        if constexpr (LB == 1) return static_cast<int>(it.get_group(0));
        else return static_cast<int>(it.get_global_id(0));
    }
    template <class F> BATCHLAS_DLA_INLINE void for_each_batch(int nbatch, F&& f) const {
        const int step = static_cast<int>(it.get_global_range(0));
        for (int b = batch(); b < nbatch; b += step) f(b);
    }
    BATCHLAS_DLA_INLINE void barrier(Mem = Mem::Local) const { it.barrier(sycl::access::fence_space::local_space); }
};

template <class Ex> inline constexpr bool is_grid2d_v = false;
template <int GR, int GC, int LB> inline constexpr bool is_grid2d_v<Grid2D<GR, GC, LB>> = true;

// Host side. Rows: the smallest power of two >= min(m, 32); columns: the smallest power of two
// >= min(n, items / rows); batch items fill the rest, so tiny matrices still fill a group.
struct Shape2D {
    int gr, gc, lb;
};
inline Shape2D batched2d_shape(int m, int n, int items = 256) {
    int gr = 1;
    while (gr < 32 && gr < m) gr *= 2;
    int gc = 1;
    while (gc < items / gr && gc < n) gc *= 2;
    return {gr, gc, items / (gr * gc)};
}
// Dimension 0 holds at most max_groups0 groups (CUDA gridDim.z); Grid2D::for_each_batch walks
// the rest. m, n, batch >= 1.
inline sycl::nd_range<3> batched2d_range(int batch, int m, int n, Shape2D s, int max_groups0 = 65535) {
    const auto up = [](int x, int q) { return static_cast<std::size_t>((x + q - 1) / q) * static_cast<std::size_t>(q); };
    const int groups0 = (batch + s.lb - 1) / s.lb;
    const int g0 = groups0 < max_groups0 ? groups0 : max_groups0;
    return {sycl::range<3>(static_cast<std::size_t>(g0) * s.lb, up(n, s.gc), up(m, s.gr)),
            sycl::range<3>(static_cast<std::size_t>(s.lb), static_cast<std::size_t>(s.gc), static_cast<std::size_t>(s.gr))};
}

// Launch f(Grid2D<>, b) for every batch item b < batch over an m x n grid (f guards its own
// extent, or calls for_each_owned_2d). Name = void: an unnamed kernel. f is a functor or lambda
// captured by value; the kernel body calls it through a named member (SemaSYCL inlining quirk).
template <class Name = void, class F>
sycl::event batched2d(sycl::queue& q, int batch, int m, int n, F f, Shape2D s, int max_groups0 = 65535) {
    const sycl::nd_range<3> r = batched2d_range(batch, m, n, s, max_groups0);
    struct Body {
        F f;
        int batch;
        void operator()(sycl::nd_item<3> it) const {
            const Grid2D<> g{it};
            g.for_each_batch(batch, [&](int b) { f(g, b); });
        }
    };
    if constexpr (std::is_void_v<Name>) return q.parallel_for(r, Body{f, batch});
    else return q.parallel_for<Name>(r, Body{f, batch});
}
template <class Name = void, class F> sycl::event batched2d(sycl::queue& q, int batch, int m, int n, F f) {
    return batched2d<Name>(q, batch, m, n, f, batched2d_shape(m, n));
}

}  // namespace batchlas::device::v2
