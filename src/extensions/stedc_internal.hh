#pragma once

#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-span.hh>

namespace batchlas {

// stedc's body WITHOUT the `info` clear (cf. steqr_dispatch). A caller that runs stedc twice
// over the same items (gesvd) must use this, or the second call erases the first's failures.
// evidence: docs/perf/stedc.md#stedc-convergence-reporting-through-info
template <Backend B, typename T>
Event stedc_dispatch(Queue& ctx,
                     const VectorView<T>& d,
                     const VectorView<T>& e,
                     const VectorView<T>& eigenvalues,
                     const Span<std::byte>& ws,
                     JobType jobz,
                     StedcParams<T> params,
                     const MatrixView<T, MatrixFormat::Dense>& eigvects,
                     Span<int32_t> info = Span<int32_t>());

} // namespace batchlas
