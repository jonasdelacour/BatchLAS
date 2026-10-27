// SPDX-License-Identifier: MIT
#pragma once

// syfn's element-wise kernels, launched by syfn.cc's composition (kernels/syfn_kernels.cc).

#include <batchlas/blas/functions/syfn.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>

#include <cstdint>

namespace batchlas {
namespace ops::syfn {

template <class T>
using MV = MatrixView<T, MatrixFormat::Dense>;

template <class T>
struct Core {
    Span<T> y;           // n x m per item, ld n: the scaled columns
    Span<float_t<T>> g;  // m per item: the column scales
};

// info[b]: the out-of-domain count of W's item b, or n + syev_info[b] where that is nonzero.
template <Backend B, class T>
void launch_info(Queue& q, const MV<T>& V, Span<float_t<T>> W, SyfnKind kind, Span<int32_t> info,
                 Span<int32_t> syev_info);

// F := 0 (the m == 0 case).
template <Backend B, class T>
Event launch_zero(Queue& q, const MV<T>& V, const MV<T>& F);

// ws.g := the column scales, ws.y := V diag(ws.g).
template <Backend B, class T>
void launch_scale(Queue& q, const MV<T>& V, Span<float_t<T>> W, SyfnFunction<float_t<T>> fn, const Core<T>& ws);

}  // namespace ops::syfn
}  // namespace batchlas
