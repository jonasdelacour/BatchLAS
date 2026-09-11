#pragma once

#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-span.hh>

namespace batchlas {

// stedc's body WITHOUT the `info` clear -- the exact counterpart of
// steqr_dispatch in steqr_internal.hh, and for the same reason.
//
// The public `stedc` is argument validation, one `detail::info_clear`, and then
// this. Everything below it only ever RAISES a status (src/extensions/info_span.hh),
// which is what lets the recursive driver's two half-solves and the level driver's
// L merges all accumulate into the same slots.
//
// A caller that runs stedc MORE THAN ONCE over the same batch items within one
// operation must therefore call this, not `stedc`, or the second call erases the
// first one's failures. gesvd's solve_tridiagonal is exactly that case: it solves
// the right tridiagonal and then, when U is wanted and the matrix is tall or
// rank-deficient, the left one, both over the same items.
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
