#pragma once

// Native batched TRSM declarations: V1 (CTA solver, one work-group per matrix)
// and V2 (blocked driver that calls V1 on each diagonal block). See docs/perf/trsm.md.

#include "../util/internal-api.hh"
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>

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

// V1's work-group width: no rung may leave over half its lanes without a rhs column.
// evidence: docs/perf/blackwell.md#trsm-v1-ladder-cap
inline constexpr int kTrsmV1MaxWg = 256;
constexpr int trsm_v1_ladder_wg(int max_wg, int cu, int q, int bs) {
    int wg = 32;
    for (int cand : {kTrsmV1MaxWg, 128, 64, 32}) {
        if (cand > max_wg) continue;
        if (cand > 32 && cand / 2 >= q) continue;
        wg = cand;
        const long long groups_c = (q + cand - 1) / cand;
        if (static_cast<long long>(bs) * groups_c >= 4LL * cu) break;
    }
    return wg;
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
