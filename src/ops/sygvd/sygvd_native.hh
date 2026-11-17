// SPDX-License-Identifier: MIT
#pragma once  // sygvd compose's element-wise launches. evidence: docs/perf/sygvd.md#sygvd-api

#include <src/util/internal-api.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>

#include <cstddef>
#include <cstdint>

namespace batchlas::ops::sygvd {

// Both skip item b where a non-null mask has mask[b] == 0 (potrf_info[b] is then never written).
// Where potrf_info[b] != 0, B's uplo triangle := the identity's (a partial factor may hold NaN).
template <typename Scalar>
BATCHLAS_INTERNAL_API Event sygvd_identity_failed(Queue& ctx, const MatrixView<Scalar, MatrixFormat::Dense>& B,
                                                  Uplo uplo, Span<const int32_t> potrf_info, const std::uint8_t* mask);

// info[b] := n + potrf_info[b] where potrf_info[b] != 0 (LAPACK's ?sygvd convention), else unchanged.
BATCHLAS_INTERNAL_API Event sygvd_merge_info(Queue& ctx, int n, Span<const int32_t> potrf_info, Span<int32_t> info,
                                             const std::uint8_t* mask);

}  // namespace batchlas::ops::sygvd

namespace batchlas::ops::sygvd {

template <class Scalar>
using MV = MatrixView<Scalar, MatrixFormat::Dense>;
template <class Scalar>
using Real = typename base_type<Scalar>::type;

// Everything one call reads besides the queue and the workspace.
template <class Scalar>
struct Args {
    MV<Scalar> A, B;
    Span<Real<Scalar>> w;
    int itype;
    JobType jobz;
    Uplo uplo;
    bool factored;  ///< sygvd_factored: B holds potrf's factor
};

// Compose (kernels/sygvd_compose.cc): workspace bytes, and the child sequence on an in-order queue.
template <Backend B, class Scalar>
std::size_t compose_workspace(Queue& q, const Args<Scalar>& a);
template <Backend B, class Scalar>
Event compose(Queue& q, const Args<Scalar>& a, Span<std::byte> ws, Span<int32_t> info);

}  // namespace batchlas::ops::sygvd
