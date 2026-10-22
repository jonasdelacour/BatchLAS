#pragma once

// potri: info scan and triangle store. evidence: docs/perf/potri.md#potrs-and-potri

#include <src/util/internal-api.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>

#include <cstdint>

namespace batchlas::sycl_potri {

// info[b] := 1-based first exactly-zero diagonal index, else 0.
template <typename T>
BATCHLAS_INTERNAL_API Event potri_diag_info(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                                            Span<int32_t> info);

// A's uplo triangle := X's; with full the other one := its conjugate transpose.
template <typename T>
BATCHLAS_INTERNAL_API Event potri_store(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& X,
                                        const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo, bool full);

}  // namespace batchlas::sycl_potri
