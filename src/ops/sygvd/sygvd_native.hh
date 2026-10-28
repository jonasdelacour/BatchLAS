#pragma once  // sygvd compose's element-wise launches. evidence: docs/perf/sygvd.md#sygvd-api

#include <src/util/internal-api.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>

#include <cstddef>
#include <cstdint>

namespace batchlas::sycl_sygvd {

// Where potrf_info[b] != 0, B's uplo triangle := the identity's (a partial factor may hold NaN).
template <typename T>
BATCHLAS_INTERNAL_API Event sygvd_identity_failed(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& B, Uplo uplo,
                                                  Span<const int32_t> potrf_info);

// info[b] := n + potrf_info[b] where potrf_info[b] != 0 (LAPACK's ?sygvd convention), else unchanged.
BATCHLAS_INTERNAL_API Event sygvd_merge_info(Queue& ctx, int n, Span<const int32_t> potrf_info, Span<int32_t> info);

}  // namespace batchlas::sycl_sygvd

namespace batchlas::ops::sygvd {

template <class T>
using MV = MatrixView<T, MatrixFormat::Dense>;
template <class T>
using Real = typename base_type<T>::type;

// Everything one call reads besides the queue and the workspace.
template <class T>
struct Args {
    MV<T> A, B;
    Span<Real<T>> w;
    int itype;
    JobType jobz;
    Uplo uplo;
    bool factored;  ///< sygvd_factored: B holds potrf's factor
};

// Compose (kernels/sygvd_compose.cc): workspace bytes, and the child sequence on an in-order queue.
template <Backend B, class T>
std::size_t compose_workspace(Queue& q, const Args<T>& a);
template <Backend B, class T>
Event compose(Queue& q, const Args<T>& a, Span<std::byte> ws, Span<int32_t> info);

}  // namespace batchlas::ops::sygvd
