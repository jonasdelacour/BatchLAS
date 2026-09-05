#pragma once

// LASWP -- apply a LAPACK interchange list to the rows of a column block, and
// the deferred left-hand interchange of a blocked LU as one SLM-staged gather.
// Templated on a caller TAG, not shared as a linked symbol: the three LU callers
// sit in different device-link clusters, so a shared kernel would fail at ptxas.
// evidence: docs/perf/lu.md#the-laswp-gather

#include "../sycl/device_scalar.hh"
// ../queue.hh: the public header only forward-declares QueueImpl; ctx->submit needs it defined.
#include "../queue.hh"

#include <batchlas/util/sycl-device-queue.hh>

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

namespace batchlas::lu_native {

template <typename Tag, typename T> class LuLaswpKernel;

// Apply pivots k0 .. k1-1 (0-based positions in the list) to the column block at
// `base`: forward gives P B (?LASWP incx = +1), !forward gives P^T B (incx = -1).
// Values are GLOBAL 1-BASED rows (getrf_native.hh's pivot contract), so `base`
// must point at row 0, offset only in COLUMN; `piv_stride` is the order of A.
template <typename Tag, typename T>
Event lu_laswp_launch(Queue& ctx,
                      T* base, int ld, int stride, int ncols, int batch,
                      const int* piv, int piv_stride,
                      int k0, int k1, bool forward) {
    if (ncols <= 0 || batch <= 0 || k1 <= k0) return ctx.get_event();

    // Re-type std::complex at the pointer boundary only; device_scalar.hh's rule.
    using DM = sycl_device::DevMap<T>;
    using D = typename DM::type;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");

    D* const bp = reinterpret_cast<D*>(base);

    ctx->submit([&](sycl::handler& h) {
        h.parallel_for<LuLaswpKernel<Tag, T>>(
            sycl::range<2>(static_cast<std::size_t>(batch),
                           static_cast<std::size_t>(ncols)),
            [=](sycl::item<2> it) {
                const int b = static_cast<int>(it.get_id(0));
                const int c = static_cast<int>(it.get_id(1));

                D* const col = bp + static_cast<std::ptrdiff_t>(b) * stride +
                               static_cast<std::ptrdiff_t>(c) * ld;
                const int* const ip = piv + static_cast<std::ptrdiff_t>(b) * piv_stride;

                if (forward) {
                    for (int k = k0; k < k1; ++k) {
                        const int p = ip[k] - 1;      // 1-BASED on the wire
                        if (p != k) {
                            const D t = col[k];
                            col[k] = col[p];
                            col[p] = t;
                        }
                    }
                } else {
                    // REVERSE ORDER: the same list applied forwards computes P,
                    // not P^T -- a silently wrong answer in a transposed getrs.
                    for (int k = k1 - 1; k >= k0; --k) {
                        const int p = ip[k] - 1;
                        if (p != k) {
                            const D t = col[k];
                            col[k] = col[p];
                            col[p] = t;
                        }
                    }
                }
            });
    });
    return ctx.get_event();
}

// The deferred left-hand interchange as one SLM-staged gather: one launch replaces
// the blocked driver's P-1 left-side launches, and composes to the same permutation.
// evidence: docs/perf/lu.md#getrf-deferred-left-gather

template <typename Tag, typename T> class LuLaswpGatherKernel;

// The 48 KB launch hole: static shared-memory sizes in this band fail to launch
// on this box, so pad past it. evidence: docs/perf/lu.md#the-48-kb-launch-hole
constexpr std::size_t kLuLaswpHoleLo = 47104;
constexpr std::size_t kLuLaswpHoleHi = 49664;
constexpr std::size_t kLuLaswpHolePadTo = 49920;

constexpr std::size_t lu_laswp_hole_padded(std::size_t bytes) {
    return (bytes > kLuLaswpHoleLo && bytes <= kLuLaswpHoleHi) ? kLuLaswpHolePadTo : bytes;
}

// The tile's share of local memory: bigger buys nothing and costs occupancy.
constexpr std::size_t kLuLaswpTileCap = 24576;

// Apply, for every column block r, the transposition suffix [j0_{r+1}, n) to that
// block's own columns. `base` points at row 0 column 0 of item 0; `piv` is the
// FULL interchange list, global 1-based. Returns false having enqueued NOTHING
// when the staging tile does not fit -- the caller must then use the walk.
template <typename Tag, typename T>
bool lu_laswp_deferred_left_launch(Queue& ctx,
                                   T* base, int ld, int stride, int batch,
                                   const int* piv, int piv_stride,
                                   int n, int nb,
                                   std::size_t slm_budget, int max_wg) {
    if (batch <= 0 || n <= 0 || nb <= 0) return true;

    const int P = (n + nb - 1) / nb;
    const int nblk = P - 1;                 // blocks that receive anything
    if (nblk <= 0) return true;             // a single panel defers nothing

    using DM = sycl_device::DevMap<T>;
    using D = typename DM::type;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");

    // r = 0 carries the longest suffix and sizes the allocation for every group.
    const int rmax = n - nb;
    const int ldt = rmax | 1;               // ODD: an even ld banks a permuted column.
    const std::size_t int_bytes =
        2u * static_cast<std::size_t>(rmax) * sizeof(int);
    if (slm_budget <= int_bytes) return false;

    const std::size_t col_bytes = static_cast<std::size_t>(ldt) * sizeof(D);
    std::size_t data_budget = slm_budget - int_bytes;
    if (data_budget > kLuLaswpTileCap) data_budget = kLuLaswpTileCap;
    std::size_t cs = data_budget / col_bytes;
    if (cs == 0) {
        // The CAP, not the device, refused: retry against the whole budget.
        cs = (slm_budget - int_bytes) / col_bytes;
        if (cs == 0) return false;
    }
    if (cs > static_cast<std::size_t>(nb)) cs = static_cast<std::size_t>(nb);
    const int Cs = static_cast<int>(cs);

    std::size_t tile_elems = static_cast<std::size_t>(Cs) * static_cast<std::size_t>(ldt);
    const std::size_t raw = int_bytes + tile_elems * sizeof(D);
    const std::size_t padded = lu_laswp_hole_padded(raw);
    if (padded > raw) {
        tile_elems = (padded - int_bytes + sizeof(D) - 1) / sizeof(D);
    }

    int wg = (max_wg < 256) ? max_wg : 256;
    if (wg < 32) wg = 32;

    D* const bp = reinterpret_cast<D*>(base);
    const int nb_k = nb;

    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<int, 1> ints(
            sycl::range<1>(2u * static_cast<std::size_t>(rmax)), h);
        sycl::local_accessor<D, 1> tile(sycl::range<1>(tile_elems), h);

        h.parallel_for<LuLaswpGatherKernel<Tag, T>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(nblk) *
                                             static_cast<std::size_t>(batch) *
                                             static_cast<std::size_t>(wg)),
                              sycl::range<1>(static_cast<std::size_t>(wg))),
            [=](sycl::nd_item<1> it) {
                const auto grp = it.get_group();
                const int gid = static_cast<int>(it.get_group(0));
                const int lid = static_cast<int>(it.get_local_id(0));
                const int r = gid / batch;
                const int b = gid - r * batch;

                const int c0 = r * nb_k;                  // first column
                const int ib = (nb_k < n - c0) ? nb_k : (n - c0);   // the driver's std::min
                const int k0 = c0 + ib;                   // = j0_{r+1}
                const int R = n - k0;                     // rows AND transpositions
                if (R <= 0 || ib <= 0) return;

                int* const idxs = &ints[0];
                int* const ips = &ints[static_cast<std::size_t>(rmax)];

                D* const Ab = bp + static_cast<std::ptrdiff_t>(b) * stride;
                const int* const ip = piv + static_cast<std::ptrdiff_t>(b) * piv_stride;

                for (int i = lid; i < R; i += wg) {
                    int p = ip[k0 + i] - 1 - k0;
                    // Contract is p in [i, R); clamped anyway because a bad
                    // value corrupts the index array for the WHOLE block.
                    if (p < 0 || p >= R) p = i;
                    ips[i] = p;
                    idxs[i] = i;
                }
                sycl::group_barrier(grp);

                // The only serial phase, and it is on the INT array.
                if (lid == 0) {
                    for (int i = 0; i < R; ++i) {
                        const int p = ips[i];
                        if (p != i) {
                            const int t = idxs[i];
                            idxs[i] = idxs[p];
                            idxs[p] = t;
                        }
                    }
                }
                sycl::group_barrier(grp);

                for (int cb = 0; cb < ib; cb += Cs) {
                    const int cw = ((ib - cb) < Cs) ? (ib - cb) : Cs;

                    // Flat over (column, row), ROW fastest -- the contiguous direction.
                    int col = lid / R;
                    int row = lid - col * R;
                    while (col < cw) {
                        tile[static_cast<std::size_t>(col) * ldt + row] =
                            Ab[static_cast<std::ptrdiff_t>(c0 + cb + col) * ld + k0 + row];
                        row += wg;
                        while (row >= R) { row -= R; ++col; }
                    }
                    sycl::group_barrier(grp);

                    col = lid / R;
                    row = lid - col * R;
                    while (col < cw) {
                        Ab[static_cast<std::ptrdiff_t>(c0 + cb + col) * ld + k0 + row] =
                            tile[static_cast<std::size_t>(col) * ldt + idxs[row]];
                        row += wg;
                        while (row >= R) { row -= R; ++col; }
                    }
                    sycl::group_barrier(grp);
                }
            });
    });
    return true;
}

}  // namespace batchlas::lu_native
