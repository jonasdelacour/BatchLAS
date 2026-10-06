// Mirror the upper triangle of a Hermitian/symmetric matrix into its lower triangle, so the
// Lower-only syev providers can serve Uplo::Upper: O(n^2) in front of an O(n^3) solve.
// evidence: docs/perf/syev.md#syev-the-upper-to-lower-mirror-for-lower-only-providers
//
// In place is safe because syev documents A as overwritten. The diagonal is left alone, and a
// complex diagonal's imaginary part is NOT zeroed -- the Lower path assumes the same of its input.
// DECLARATION ONLY: defining the kernel inline here gives an ODR "same mangled name" error once
// two TUs call it; the explicit instantiations live in uplo_mirror.cc.
#pragma once

#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>

namespace batchlas {

template <Backend B, typename T>
Event mirror_upper_to_lower(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& a);

} // namespace batchlas
