#pragma once

// GESV: solve A X = B by LU with partial pivoting, as one op.
//
// NO VENDOR ARM ANYWHERE -- no vendor ships a batched gesv -- so there is no
// `gesv_vendor` here and the composition `getrf; getrs` is the fallback every other
// op gets from the vendor. SEMANTICS ARE EXACTLY THAT COMPOSITION, as a contract:
// A comes back holding L and U, `pivots` the same 1-based interchange list getrf
// writes, `info` getrf's per-item status. evidence: docs/perf/lu.md#the-fused-gesv-tier

#include <batchlas/export.hh>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/queue-dispatch.hh>

namespace batchlas {

// Signature aliases for explicit instantiation; see BATCHLAS_INSTANTIATE in
// src/util/template-instantiations.hh. Keep in sync with the declarations below.
namespace sig {
template <typename T>
using gesv = Event(Queue&,
                   const MatrixView<T, MatrixFormat::Dense>&,
                   const MatrixView<T, MatrixFormat::Dense>&,
                   Span<int64_t>, Span<std::byte>, Span<int32_t>);

template <typename T>
using gesv_buffer_size = size_t(Queue&,
                                const MatrixView<T, MatrixFormat::Dense>&,
                                const MatrixView<T, MatrixFormat::Dense>&);
}  // namespace sig

// IT CHECKS MORE THAN getrs_validate_params DOES, structurally: getrs lets a
// non-conforming pair through to the vendor, which is a routing decision and not an
// acceptance. gesv has no vendor, so those shapes would name the wrong cause.
template <typename T>
inline void gesv_validate_params(const MatrixView<T, MatrixFormat::Dense>& A,
                                 const MatrixView<T, MatrixFormat::Dense>& B) {
    if (A.rows() < 0 || A.cols() < 0 || B.rows() < 0 || B.cols() < 0) {
        throw batchlas::invalid_argument(
            "GESV: Matrix dimensions cannot be negative (A: rows=" +
            std::to_string(A.rows()) + ", cols=" + std::to_string(A.cols()) +
            "; B: rows=" + std::to_string(B.rows()) +
            ", cols=" + std::to_string(B.cols()) + ")");
    }
    if (A.rows() != A.cols()) {
        throw batchlas::invalid_argument(
            "GESV: A must be square, got " + std::to_string(A.rows()) + "x" +
            std::to_string(A.cols()));
    }
    if (A.rows() != B.rows()) {
        throw batchlas::invalid_argument(
            "GESV: B must have A.rows() rows, got " + std::to_string(B.rows()) +
            " against " + std::to_string(A.rows()));
    }
    if (A.batch_size() != B.batch_size()) {
        throw batchlas::invalid_argument(
            "GESV: A and B must share a batch size, got " +
            std::to_string(A.batch_size()) + " and " + std::to_string(B.batch_size()));
    }
}

// A -> its LU factors, B -> X, `pivots` -> n * batch 1-based entries. An EMPTY
// `info` span means "not requested", as potrf's does.
template <Backend Back, typename T>
BATCHLAS_API Event gesv(Queue& ctx,
                        const MatrixView<T, MatrixFormat::Dense>& A,
                        const MatrixView<T, MatrixFormat::Dense>& B,
                        Span<int64_t> pivots,
                        Span<std::byte> work_space,
                        Span<int32_t> info);

template <Backend Back, typename T>
BATCHLAS_API size_t gesv_buffer_size(Queue& ctx,
                                     const MatrixView<T, MatrixFormat::Dense>& A,
                                     const MatrixView<T, MatrixFormat::Dense>& B);

// `info` cannot be defaulted: the sig:: aliases are function TYPES.
template <Backend Back, typename T>
inline Event gesv(Queue& ctx,
                  const MatrixView<T, MatrixFormat::Dense>& A,
                  const MatrixView<T, MatrixFormat::Dense>& B,
                  Span<int64_t> pivots,
                  Span<std::byte> work_space) {
    return gesv<Back, T>(ctx, A, B, pivots, work_space, Span<int32_t>{});
}

BATCHLAS_ACCEPT_OWNING(gesv)
BATCHLAS_ACCEPT_OWNING(gesv_buffer_size)

BATCHLAS_DISPATCH_ON_QUEUE(gesv)
BATCHLAS_DISPATCH_ON_QUEUE(gesv_buffer_size)

}  // namespace batchlas
