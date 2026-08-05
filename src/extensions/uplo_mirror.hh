// Mirror the upper triangle of a Hermitian/symmetric matrix into its lower triangle.
//
// WHY THIS EXISTS. syev_blocked and syev_two_stage (and everything under them: sytrd_blocked,
// sytrd_sy2sb, sytrd_sb2st) implement Uplo::Lower only -- sytrd_blocked threw outright on
// Upper. So every Uplo::Upper call fell back to the vendor no matter how much faster our own
// providers were at that shape. That is a routing loss caused by a missing O(n^2) step in
// front of an O(n^3) solve.
//
// For a Hermitian matrix the two triangles carry the same operator: A[j][i] == conj(A[i][j]).
// Writing the upper triangle into the lower one therefore yields a matrix whose LOWER
// triangle describes exactly the input operator, and the existing Lower path then produces
// identical eigenvalues and eigenvectors. Cost is O(n^2 * batch) against the solve's
// O(n^3 * batch), i.e. below noise at every size where routing matters.
//
// In-place is safe here: `syev` documents A as overwritten (include/blas/functions/syev.hh),
// and the Lower path destroys A during the reduction regardless. The diagonal is left alone;
// for complex input its imaginary part is not forced to zero, matching what the Lower path
// already assumes of a Hermitian input.
//
// DECLARATION ONLY. The definition lives in uplo_mirror.cc with explicit instantiations,
// because a SYCL kernel name class must have exactly one definition across the program --
// defining it inline in a header and calling it from both syev_blocked.cc and
// syev_two_stage.cc produced "definition with same mangled name" ODR errors.
#pragma once

#include <blas/matrix.hh>
#include <util/sycl-device-queue.hh>

namespace batchlas {

template <Backend B, typename T>
Event mirror_upper_to_lower(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& a);

} // namespace batchlas
