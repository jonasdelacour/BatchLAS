// SPDX-License-Identifier: MIT
#pragma once

// syfn's element-wise kernels, launched by syfn.cc's composition (kernels/syfn_kernels.cc).

#include <batchlas/blas/functions/syfn.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>

#include <cstdint>

namespace batchlas::ops::syfn {

template <typename Scalar>
using MV = MatrixView<Scalar, MatrixFormat::Dense>;

template <typename Scalar>
struct Core {
    Span<Scalar> y;           // n x m per item, ld n: the scaled columns
    Span<float_t<Scalar>> g;  // m per item: the column scales
};

// Each launch skips item b where a non-null mask has mask[b] == 0.
// info[b]: the out-of-domain count of W's item b, or n + syev_info[b] where that is nonzero.
template <Backend B, typename Scalar>
void launch_info(Queue& q, const MV<Scalar>& V, Span<float_t<Scalar>> W, SyfnKind kind, Span<int32_t> info,
                 Span<int32_t> syev_info, const std::uint8_t* mask);

// F := 0 (the m == 0 case).
template <Backend B, typename Scalar>
Event launch_zero(Queue& q, const MV<Scalar>& V, const MV<Scalar>& F, const std::uint8_t* mask);

// ws.g := the column scales, ws.y := V diag(ws.g).
template <Backend B, typename Scalar>
void launch_scale(Queue& q, const MV<Scalar>& V, Span<float_t<Scalar>> W, SyfnFunction<float_t<Scalar>> fn,
                  const Core<Scalar>& ws, const std::uint8_t* mask);

}  // namespace batchlas::ops::syfn
