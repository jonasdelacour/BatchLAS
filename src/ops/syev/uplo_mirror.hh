// SPDX-License-Identifier: MIT
#pragma once
// A := its upper triangle mirrored into the lower, in place (masked-out items untouched), so Lower-only
// syev drivers serve Upper. The complex diagonal's imaginary part is NOT zeroed. Declaration only: an
// inline kernel here is an ODR error once two TUs call it; instantiations are in uplo_mirror.cc.
// evidence: docs/perf/syev.md#syev-the-upper-to-lower-mirror-for-lower-only-providers

#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>

namespace batchlas {

template <Backend B, typename Scalar>
Event mirror_upper_to_lower(Queue& ctx, const MatrixView<Scalar, MatrixFormat::Dense>& a);

}  // namespace batchlas
