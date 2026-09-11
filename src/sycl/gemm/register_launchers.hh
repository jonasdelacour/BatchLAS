#pragma once

#include "register_tiled_common.hh"

#include <stdexcept>

namespace batchlas::sycl_gemm {

// One row of the register-tiled GEMM dispatch table.
//
// These are exactly the template parameters of launch_register_tiled<>.
// Gathering them into a structural (C++20 NTTP-usable) type lets a single
// launcher stand in for what used to be one hand-written forwarder per tile
// shape: the shape is now written at the `case` label in gemm_custom's switch
// that is its only caller, so the tuning grid reads as a table instead of as
// thirty-odd near-identical function bodies scattered across a header.
//
// The defaults match launch_register_tiled<>'s own defaults with one
// exception: ThreadTileCols there defaults to ThreadTileRows, whereas here TR
// and TC default independently to 4. Every row therefore states TR and TC
// explicitly.
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
    // Aligned-fast-path policy. `try_aligned` takes the unpredicated
    // instantiation whenever the layout is eligible and falls back to the
    // predicated one otherwise; `require_aligned` additionally refuses an
    // ineligible layout, which is what the by-name ...S2U1Aligned variant asks
    // for. Only NN rows may set either -- launch_register_tiled static_asserts
    // that.
    bool try_aligned = false;
    bool require_aligned = false;
};

// The single register-tiled launcher.
//
// `trace` names the trace scope for the predicated instantiation and
// `trace_aligned`, when supplied, names it for the unpredicated one. Passing
// the name in is what lets launch_register_tiled take a plain string: it used
// to take a `const char*(*)(KernelVariant)` and recover the variant from its
// own tile parameters through a constexpr inverse lookup, which existed for no
// other purpose than to name this scope. The caller already knows the variant.
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
