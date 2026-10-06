#pragma once

#include "register_tiled_common.hh"

#include <stdexcept>

namespace batchlas::sycl_gemm {

// One row of the register-tiled GEMM dispatch table: launch_register_tiled<>'s
// template parameters as a structural NTTP, written at gemm_custom's case label.
// TRAP: TC defaults to 4 here but to ThreadTileRows there, so every row states
// TR and TC. evidence: docs/perf/gemm.md#gemm-the-register-tiled-launcher-table
struct RegTile {
    int M;              // TileM
    int N;              // TileN
    int K;              // TileK
    int TR = 4;         // ThreadTileRows
    int TC = 4;         // ThreadTileCols
    int VA = 4;         // VecA
    int VB = 4;         // VecB
    int Unroll = 1;     // UnrollK
    int Stages = 1;     // software-pipeline depth (1 or 2)
    Transpose OpA = Transpose::NoTrans;
    Transpose OpB = Transpose::NoTrans;
    // try_aligned: unpredicated when eligible, else predicated. require_aligned:
    // throw when ineligible (...S2U1Aligned). NN rows only (static_asserted).
    bool try_aligned = false;
    bool require_aligned = false;
};

// The single register-tiled launcher. `trace` names the predicated
// instantiation's trace scope, `trace_aligned` (if given) the unpredicated one.
template <typename T, RegTile P>
Event launch_reg(Queue& ctx,
                 const MatrixView<T, MatrixFormat::Dense>& A,
                 const MatrixView<T, MatrixFormat::Dense>& B,
                 const MatrixView<T, MatrixFormat::Dense>& C,
                 T alpha,
                 T beta,
                 const char* trace,
                 const char* trace_aligned = nullptr) {
    if constexpr (P.try_aligned || P.require_aligned) {
        const bool eligible = can_use_aligned_nn_fast_path<T, P.M, P.N, P.K, P.VA, P.VB>(A, B, C);
        if constexpr (P.require_aligned) {
            if (!eligible) {
                throw batchlas::unsupported(
                    "Requested aligned GEMM kernel for a non-eligible matrix layout");
            }
        }
        if (eligible) {
            return launch_register_tiled<T, P.M, P.N, P.K, P.TR, P.TC, P.VA, P.VB, P.Unroll,
                                         P.Stages, true, P.OpA, P.OpB>(
                ctx, A, B, C, alpha, beta, trace_aligned != nullptr ? trace_aligned : trace);
        }
    }
    return launch_register_tiled<T, P.M, P.N, P.K, P.TR, P.TC, P.VA, P.VB, P.Unroll, P.Stages,
                                 false, P.OpA, P.OpB>(ctx, A, B, C, alpha, beta, trace);
}

} // namespace batchlas::sycl_gemm
