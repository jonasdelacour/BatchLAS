#pragma once

// Native batched TRSM declarations: V1 (CTA solver, one work-group per matrix)
// and V2 (blocked driver that calls V1 on each diagonal block). See docs/perf/trsm.md.

#include "../util/internal-api.hh"
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>

#include <batchlas/blas/dispatch/route.hh>

#include <complex>
#include <cstdlib>
#include <type_traits>
#include <functional>
#include <batchlas/util/sycl-device-queue.hh>

namespace batchlas::sycl_trsm {

// Zero means "no native TRSM kernel in this build": RouteTable<Op::trsm,T> reads
// this through TrsmShape::cta_max_n and reports both native routes unsupported.
template <typename T>
int trsm_cta_max_n();

template <typename T>
bool trsm_blocked_available();

template <typename T>
BATCHLAS_INTERNAL_API Event trsm_native_v1_dispatch(Queue& ctx,
                                                    const MatrixView<T, MatrixFormat::Dense>& A,
                                                    const MatrixView<T, MatrixFormat::Dense>& B,
                                                    T alpha,
                                                    Side side,
                                                    Uplo uplo,
                                                    Transpose transA,
                                                    Diag diag);

// The Side::Left sub-group kernel (trsm_sg_left.cc): one lane per triangular row,
// 32/N matrices per sub-group, orders 1..32. Throws above 32.
template <typename T>
BATCHLAS_INTERNAL_API Event trsm_native_sg_left_dispatch(Queue& ctx,
                                                         const MatrixView<T, MatrixFormat::Dense>& A,
                                                         const MatrixView<T, MatrixFormat::Dense>& B,
                                                         T alpha,
                                                         Uplo uplo,
                                                         Transpose transA,
                                                         Diag diag);

// Which kernel trsm_native_v1_dispatch runs for a Side::Left solve of order n with
// q right-hand sides. Only sm_120 was measured; every other device keeps V1.
// evidence: docs/perf/blackwell.md#trsm-sub-group-left-kernel
template <typename T>
inline bool trsm_left_use_sg(int cuda_cc, int n, int q) {
    if (const char* e = std::getenv("BATCHLAS_DEV_TRSM_SG")) return e[0] == '1';
    if (!dispatch::is_sm120_family(cuda_cc) || n > 32) return false;
    if constexpr (std::is_same_v<T, float>) return q < 32;
    if constexpr (std::is_same_v<T, std::complex<float>>) return q < 64;
    return false;   // double and complex<double> were not measured
}

// Trailing-update GEMM. An EMPTY function means sycl_gemm::gemm_custom, keeping
// this layer dispatch-free; inject the routed gemm where dispatch is available,
// since the native kernel collapses on the strided sub-views a panel passes.
// evidence: docs/perf/trsm.md#the-final-grid-after-the-routed-trailing-gemm
template <typename T>
using TrsmTrailingGemm = std::function<Event(
    Queue&,
    const MatrixView<T, MatrixFormat::Dense>&,
    const MatrixView<T, MatrixFormat::Dense>&,
    const MatrixView<T, MatrixFormat::Dense>&,
    T, T, Transpose, Transpose, ComputePrecision)>;

template <typename T>
BATCHLAS_INTERNAL_API Event trsm_native_blocked(Queue& ctx,
                                                const MatrixView<T, MatrixFormat::Dense>& A,
                                                const MatrixView<T, MatrixFormat::Dense>& B,
                                                T alpha,
                                                Side side,
                                                Uplo uplo,
                                                Transpose transA,
                                                Diag diag,
                                                TrsmTrailingGemm<T> trailing_gemm = {});

} // namespace batchlas::sycl_trsm
