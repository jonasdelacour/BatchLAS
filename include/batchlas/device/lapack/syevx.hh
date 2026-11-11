// SPDX-License-Identifier: MIT
#pragma once
// xSYEVX's selection step on one matrix with a known eigendecomposition (the Direct route).

#include <cstdint>

#include <batchlas/device/exec.hh>
#include <batchlas/device/lapack_aux.hh>
#include <batchlas/device/level1.hh>
#include <batchlas/device/view.hh>

namespace batchlas::device {

// lam: n ascending eigenvalues, q: their vectors. Block [first, first + count), or for a value
// range (vl, vu] the leader's Sturm counts (stebz's `<=`) published in block[0..1] (int64 SLM).
// The lowest `capacity` are kept (reversed when reverse) into w and v; v's columns past them
// are exactly zero. q, v are touched only when want_v. Returns the true count.
// evidence: docs/design/syevx.md#syevx-range-direct-and-directsubset-may-disagree-on-m-by-one
template <class Ex, class L, class R, class Blk, class W, class Q, class V>
BATCHLAS_DLA_INLINE std::int64_t syevx_select(const Ex& ex, const L& lam, bool value_range, R vl, R vu,
                                              std::int64_t first, std::int64_t count, std::int64_t capacity,
                                              bool reverse, const Blk& block, const W& w, bool want_v, const Q& q,
                                              const V& v) {
    if (value_range) {
        if (ex.leader()) {
            const std::int64_t c_lo = upper_bound(Thread{}, lam, vl);
            const std::int64_t c_hi = upper_bound(Thread{}, lam, vu);
            block[0] = c_lo;
            block[1] = (c_hi > c_lo) ? (c_hi - c_lo) : std::int64_t(0);  // a loop bound downstream
        }
        sync(ex, Mem::Global);  // as sycl::group_barrier: every space
        first = block[0];
        count = block[1];
    }
    const int write = static_cast<int>((count < capacity) ? count : capacity);
    const int f0 = static_cast<int>(first);
    copy(ex, flip(lam.tail(f0).head(write), reverse), w.head(write));
    if (want_v) {
        const int n = q.rows();
        lacpy<Part::Full, Transpose::NoTrans, alg::Flat>(ex, flip<axis::Cols>(q.sub(0, f0, n, write), reverse),
                                                         v.sub(0, 0, n, write));
        // evidence: docs/design/syevx.md#syevx-range-the-m-output-and-what-neigs-means
        laset<Part::Full, alg::Flat>(ex, Zero{}, Zero{}, v.sub(0, write, n, v.cols() - write));
    }
    return count;
}

}  // namespace batchlas::device
