// SPDX-License-Identifier: MIT
#pragma once
/// @file
/// @brief The batched 2-D executor Grid2D and its host launcher batched2d: one work-item per
/// matrix element.
/// @ingroup api_device_exec
///
/// Indices are 2-D (no `%` or `/`), and the batch dimension is grid-strided past the dimension-0
/// group limit. for_each_owned_2d (level1.hh) on a Grid2D visits the item's one element.

#include <cstddef>
#include <type_traits>

#include <sycl/sycl.hpp>

#include <batchlas/device/core.hh>

namespace batchlas::device {

/// @brief One work-item per element of a batch of matrices, in a 3-D launch.
/// @ingroup api_device_exec
///
/// The local range is (LB, GC, GR) with dimension 2 fastest, so consecutive items walk a column and
/// both sides of a copy coalesce. The item is element (row(), col()) of batch items batch() + k *
/// global_range(0), k = 0, 1, ... (for_each_batch).
/// @tparam GR the group's rows, or 0 to read them from the launch
/// @tparam GC the group's columns, or 0 to read them from the launch
/// @tparam LB the group's batch items, or 0 to read them from the launch
template <int GR = 0, int GC = 0, int LB = 0> struct Grid2D {
    ///@{
    /// @brief The GR, GC and LB template arguments (0: read from the launch).
    static constexpr int static_group_rows = GR, static_group_cols = GC, static_group_batch = LB;
    ///@}
    /// @brief The kernel's item.
    sycl::nd_item<3> it;

    /// @brief The rows of a work-group: GR, or the local range of dimension 2.
    BATCHLAS_DEVICE_INLINE int group_rows() const {
        if constexpr (GR > 0) return GR;
        else return static_cast<int>(it.get_local_range(2));
    }
    /// @brief The columns of a work-group: GC, or the local range of dimension 1.
    BATCHLAS_DEVICE_INLINE int group_cols() const {
        if constexpr (GC > 0) return GC;
        else return static_cast<int>(it.get_local_range(1));
    }
    /// @brief This item's row within its work-group.
    BATCHLAS_DEVICE_INLINE int lrow() const { return static_cast<int>(it.get_local_id(2)); }
    /// @brief This item's column within its work-group.
    BATCHLAS_DEVICE_INLINE int lcol() const { return static_cast<int>(it.get_local_id(1)); }
    /// @brief This item's matrix row.
    BATCHLAS_DEVICE_INLINE int row() const { return static_cast<int>(it.get_global_id(2)); }
    /// @brief This item's matrix column.
    BATCHLAS_DEVICE_INLINE int col() const { return static_cast<int>(it.get_global_id(1)); }
    /// @brief The first batch item of this item: the only one when the launch covers the batch.
    BATCHLAS_DEVICE_INLINE int batch() const {
        if constexpr (LB == 1) return static_cast<int>(it.get_group(0));
        else return static_cast<int>(it.get_global_id(0));
    }
    /// @brief Calls f(b) for each batch item b < nbatch of this item, grid-strided.
    /// @tparam F a callable taking the batch index
    /// @param nbatch the number of matrices
    /// @param f the callable
    template <class F> BATCHLAS_DEVICE_INLINE void for_each_batch(int nbatch, F&& f) const {
        const int step = static_cast<int>(it.get_global_range(0));
        for (int b = batch(); b < nbatch; b += step) f(b);
    }
    /// @brief The work-group barrier, fencing local memory.
    BATCHLAS_DEVICE_INLINE void barrier(Mem = Mem::Local) const { it.barrier(sycl::access::fence_space::local_space); }
};

/// @brief True when Ex is a Grid2D.
/// @ingroup api_device_exec
/// @tparam Ex the executor type
template <class Ex> inline constexpr bool is_grid2d_v = false;
/// @cond
template <int GR, int GC, int LB> inline constexpr bool is_grid2d_v<Grid2D<GR, GC, LB>> = true;
/// @endcond

/// @brief Host side: the work-group shape of a batched 2-D launch.
/// @ingroup api_device_exec
struct Shape2D {
    ///@{
    /// @brief Rows, columns and batch items of one work-group.
    int gr, gc, lb;
    ///@}
};
/// @brief The most batch items in one work-group (dimension 0), 64.
/// @ingroup api_device_exec
inline constexpr int kMaxGroupBatch2D = 64;
/// @brief Host side: the work-group shape for m x n matrices in groups of about `items` work-items.
/// @ingroup api_device_exec
///
/// Rows are the smallest power of two >= min(m, 32); columns the smallest power of two >= min(n,
/// items / rows); batch items fill the rest, so tiny matrices still fill a group. Batch items sit
/// in dimension 0 (CUDA blockDim.z), at most kMaxGroupBatch2D.
/// @param m the matrix rows
/// @param n the matrix columns
/// @param items the target work-group size
/// @return the Shape2D
inline Shape2D batched2d_shape(int m, int n, int items = 256) {
    int gr = 1;
    while (gr < 32 && gr < m) gr *= 2;
    int gc = 1;
    while (gc < items / gr && gc < n) gc *= 2;
    const int lb = items / (gr * gc);
    return {gr, gc, lb < kMaxGroupBatch2D ? lb : kMaxGroupBatch2D};
}
/// @brief Host side: the nd-range of a batched 2-D launch.
/// @ingroup api_device_exec
///
/// Dimension 0 holds at most max_groups0 groups (CUDA gridDim.z); Grid2D::for_each_batch walks the
/// rest of the batch.
/// @param batch the number of matrices, >= 1
/// @param m the matrix rows, >= 1
/// @param n the matrix columns, >= 1
/// @param s the work-group shape
/// @param max_groups0 the most groups in dimension 0
/// @return the 3-D nd_range
inline sycl::nd_range<3> batched2d_range(int batch, int m, int n, Shape2D s, int max_groups0 = 65535) {
    const auto up = [](int x, int q) { return static_cast<std::size_t>((x + q - 1) / q) * static_cast<std::size_t>(q); };
    const int groups0 = (batch + s.lb - 1) / s.lb;
    const int g0 = groups0 < max_groups0 ? groups0 : max_groups0;
    return {sycl::range<3>(static_cast<std::size_t>(g0) * s.lb, up(n, s.gc), up(m, s.gr)),
            sycl::range<3>(static_cast<std::size_t>(s.lb), static_cast<std::size_t>(s.gc), static_cast<std::size_t>(s.gr))};
}

/// @brief Host side: launches f(Grid2D<>, b) for every batch item b < batch over an m x n grid.
/// @ingroup api_device_exec
///
/// f guards its own extent, or calls for_each_owned_2d. f is a functor or lambda captured by
/// value.
/// @tparam Name the kernel name, or void for an unnamed kernel
/// @tparam F the callable, `f(const Grid2D<>&, int b)`
/// @param q the queue
/// @param batch the number of matrices, >= 1
/// @param m the matrix rows, >= 1
/// @param n the matrix columns, >= 1
/// @param f the callable
/// @param s the work-group shape
/// @param max_groups0 the most groups in dimension 0
/// @return the kernel's event
// The kernel body calls f through a named member: a SemaSYCL inlining quirk.
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
/// @brief Host side: batched2d with the default shape, `batched2d_shape(m, n)`.
/// @ingroup api_device_exec
/// @tparam Name the kernel name, or void for an unnamed kernel
/// @tparam F the callable, `f(const Grid2D<>&, int b)`
/// @param q the queue
/// @param batch the number of matrices, >= 1
/// @param m the matrix rows, >= 1
/// @param n the matrix columns, >= 1
/// @param f the callable
/// @return the kernel's event
template <class Name = void, class F> sycl::event batched2d(sycl::queue& q, int batch, int m, int n, F f) {
    return batched2d<Name>(q, batch, m, n, f, batched2d_shape(m, n));
}

}  // namespace batchlas::device
